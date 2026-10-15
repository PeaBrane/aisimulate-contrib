# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit producer capture for closed-loop Replay without using measured arrivals.

Original adapter for pinned AIPerf raw-record and trajectory contracts. See
COLLECTION.md for immutable API sources, limits and attribution.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

PRODUCER = "754356e9a39acc6cc6afb242d123bb57c3fb6f75"


def jsonl(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield number, json.loads(line)
            except ValueError as error:
                raise ValueError(f"invalid/truncated JSONL: {path}:{number}") from error


def token_ids(values):
    if not isinstance(values, list) or any(type(value) is not int or not 0 <= value < 2**32 for value in values):
        raise ValueError("native token IDs must be uint32 arrays")
    return values


def extract_request(row):
    metadata = row["metadata"]
    payload = row.get("payload")
    prompt, output, usage, finish = None, [], None, []
    for response in row.get("responses") or []:
        for packet in response.get("packets", []):
            if packet["name"] != "data" or packet.get("value") in (None, "", "[DONE]"):
                continue
            data = json.loads(packet["value"])
            if data.get("prompt_token_ids") is not None:
                ids = token_ids(data["prompt_token_ids"])
                if prompt is not None and ids != prompt:
                    raise ValueError("native prompt token IDs changed within one response")
                prompt = ids
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise ValueError("capture only supports one completion per request")
                if choice.get("token_ids") is not None:
                    output.extend(token_ids(choice["token_ids"]))
                if choice.get("finish_reason") is not None:
                    finish.append(choice["finish_reason"])
    failure = bool(
        row.get("error")
        or metadata.get("was_cancelled")
        or metadata.get("context_overflow_skip")
        or (row.get("status") or 0) >= 400
    )
    exact = (
        prompt is not None
        and usage is not None
        and usage.get("prompt_tokens") == len(prompt)
        and usage.get("completion_tokens") == len(output)
    )
    requested = (payload.get("max_completion_tokens") or payload.get("max_tokens")) if payload else None
    return {
        "request_id": metadata["x_request_id"],
        "phase": metadata.get("benchmark_phase"),
        "conversation_id": metadata["conversation_id"],
        "session_id": metadata["x_correlation_id"],
        "turn_index": metadata["turn_index"],
        "parent_session_id": metadata.get("parent_correlation_id"),
        "root_session_id": metadata.get("root_correlation_id"),
        "source": {key: value for key, value in metadata.items() if key.startswith("source_")},
        "payload_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "input_token_ids": prompt,
        "output_token_ids": output,
        "requested_output_length": requested,
        "ignore_eos": payload.get("ignore_eos") if payload else None,
        "finish_reasons": finish,
        "usage": usage,
        "native_tokens_complete": exact,
        "output_length_matches_request": exact and requested == len(output),
        "cache_bust_marker": row.get("cache_bust_marker"),
        "cache_bust_target": row.get("cache_bust_target"),
        "failed_or_cancelled": failure,
        "context_overflow_skip": metadata.get("context_overflow_skip", False),
        "error": row.get("error"),
    }


def audit(capture, raw_paths, destination):
    capture, destination = Path(capture), Path(destination)
    events = [row for path in sorted(capture.glob("events-*.jsonl")) for _, row in jsonl(path)]
    captured_templates = {}
    for event in events:
        if event["producer_revision"] != PRODUCER:
            raise ValueError("capture producer pin mismatch")
        all_templates = {}
        for relative in event["templates"].values():
            path = (capture / relative).resolve()
            if not path.is_relative_to(capture.resolve()):
                raise ValueError("template path escapes capture")
            templates = json.loads(path.read_text())
            all_templates.update(templates)
            captured_templates.update(templates)
        for trajectory in event.get("trajectories") or []:
            snapshot = trajectory.get("snapshot") or {}
            for state in snapshot.get("states", []):
                if state["conversation_id"] not in all_templates:
                    raise ValueError("frontier references missing conversation template")
                turns = all_templates[state["conversation_id"]]["turns"]
                if not 0 <= state["next_turn_index"] <= len(turns):
                    raise ValueError("frontier turn lies outside template")
    boundaries = [row for row in events if row["event"] != "recycle"]
    stages = {(row["phase"], row["event"]): row for row in boundaries}
    if len(stages) != len(boundaries):
        raise ValueError("duplicate phase boundaries; capture directory must belong to one run")
    warmup = stages.get(("warmup", "finalize"))
    profile = stages.get(("profiling", "setup"))
    if warmup is None or profile is None or warmup["trajectories"] != profile["trajectories"]:
        raise ValueError("missing or changed warmup-to-profile frontier")
    destination.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    seen = set()
    with (destination / "requests.jsonl").open("w") as output:
        for path in raw_paths:
            for line, row in jsonl(path):
                request = extract_request(row)
                template = captured_templates.get(request["conversation_id"])
                if template is None or not 0 <= request["turn_index"] < len(template["turns"]):
                    raise ValueError("raw request has no captured template turn")
                if request["request_id"] in seen:
                    raise ValueError("duplicate raw request ID; use shards OR merged export, never both")
                seen.add(request["request_id"])
                request["raw_source"] = {"path": str(path), "line": line}
                counts[str(request["phase"])] += 1
                counts["requests"] += 1
                for field in (
                    "native_tokens_complete",
                    "output_length_matches_request",
                    "failed_or_cancelled",
                    "context_overflow_skip",
                ):
                    counts[field] += bool(request[field])
                output.write(json.dumps(request, separators=(",", ":")) + "\n")
    if not seen:
        raise ValueError("capture has no raw requests")
    result = {
        "schema": "dspark-agentx-replay-capture-v1",
        "counts": dict(counts),
        "event_count": len(events),
        "warmup_handoff": "drained-wire-live-DAG",
        "measured_arrivals_used_for_replay": False,
        "scoring_eligible": False,
        "remaining_gates": [
            "Reconcile raw request count with producer completion barriers and raw writer dropped_record_count.",
            "Import the external sampled frontier and lower captured requests "
            "into the native exact-token AgentX graph API.",
            "Lower captured template dependencies/tool delays; validate idle caps, markers and recycle choices.",
            "Match actual stop/cancellation cohorts and cache/server cleanup across warmup.",
        ],
    }
    (destination / "capture-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.capture, args.raw, args.output), indent=2))


if __name__ == "__main__":
    main()
