# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract measured-window DSpark acceptance from pinned vLLM counters."""

import argparse
import json
import math
import re
from pathlib import Path

NAMES = [
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
]
SAMPLE = re.compile(r"^(vllm:spec_decode_[a-z_]+)(\{[^}]*\})?\s+([^ ]+)")
POSITION = re.compile(r'position="(\d+)"')


def counters(snapshot):
    values = {}
    for endpoint, text in snapshot.items():
        for line in text.splitlines():
            match = SAMPLE.match(line)
            if match and match[1] in NAMES:
                key = (endpoint, match[1], match[2] or "")
                if key in values:
                    raise ValueError("duplicate counter series")
                value = float(match[3])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("counter must be finite and nonnegative")
                values[key] = value
    return values


def measured_acceptance(before, after, k):
    initial, final = counters(before), counters(after)
    if not final:
        raise ValueError("no speculation counters available")
    if not initial.keys() <= final.keys():
        raise ValueError("counter series disappeared during measurement")
    totals, positions = dict.fromkeys(NAMES, 0.0), [0.0] * k
    for key, value in final.items():
        delta = value - initial.get(key, 0)
        if delta < 0:
            raise ValueError("counter reset during measurement")
        name, labels = key[1:]
        totals[name] += delta
        if name == NAMES[3]:
            position = POSITION.search(labels)
            if position is None or not 0 <= int(position[1]) < k:
                raise ValueError("invalid accepted-token position")
            positions[int(position[1])] += delta
    rounds, proposed, accepted = (totals[name] for name in NAMES[:3])
    if rounds <= 0 or not 0 <= accepted <= proposed <= rounds * k:
        raise ValueError("invalid or empty measured speculative window")
    has_positions = any(key[1] == NAMES[3] for key in final)
    if has_positions and (sum(positions) != accepted or positions != sorted(positions, reverse=True)):
        raise ValueError("accepted-position counts inconsistent with total or prefix acceptance")
    conditional = None
    if has_positions:
        denominators = [rounds, *positions[:-1]]
        conditional = [n / d if d else 0.0 for n, d in zip(positions, denominators, strict=True)]
    return {
        "rounds": rounds,
        "proposed_draft_tokens": proposed,
        "accepted_draft_tokens": accepted,
        "accepted_tokens": accepted / rounds,
        "committed_tokens_per_round": 1 + accepted / rounds,
        "acceptance_rates": conditional,
        "accepted_counts_per_position": positions if has_positions else None,
        "sampling_model": "conditional_prefix_rates" if has_positions else "adjacent_integer_mean_only",
        "observed_proposed_width": proposed / rounds,
        "counter_source": "vLLM ac9126e58aa7 v1/spec_decode/metrics.py",
        "scope": "counter deltas after drained warmup; sampled engine rounds, not per-request acceptance trace",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--endpoint",
        action="append",
        help="Restrict counters to decode endpoints for P/D",
    )
    args = parser.parse_args()
    before = json.loads((args.run / "metrics-before.json").read_text())
    after = json.loads((args.run / "metrics-after.json").read_text())
    if args.endpoint:
        before = {endpoint: before[endpoint] for endpoint in args.endpoint}
        after = {endpoint: after[endpoint] for endpoint in args.endpoint}
    result = measured_acceptance(before, after, args.k)
    result["endpoints"] = list(after)
    (args.run / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
