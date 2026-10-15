# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze a synthetic control from a separate successful 128-request real pilot."""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path


def freeze(pilot):
    summary = json.loads((pilot / "summary.json").read_text())
    identity = json.loads((pilot / "identity.json").read_text())
    acceptance_path = pilot / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text())
    cell = identity["cell"]
    if identity.get("phase") != "pilot" or cell["acceptance"] != "real":
        raise ValueError("control requires a separate real-acceptance pilot")
    metrics = summary["metrics"]
    if (
        summary["status"] != "success"
        or summary["expected_requests"] != 128
        or metrics["completed_requests"] != 128
        or metrics["failed_requests"] != 0
    ):
        raise ValueError("pilot must complete all 128 requests successfully")
    accepted = acceptance["accepted_tokens"]
    k = cell["num_speculative_tokens"]
    if not math.isfinite(accepted) or not 0 <= accepted <= k:
        raise ValueError("accepted draft mean must be finite and within [0, K]")
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if identity_sha != summary["identity_sha256"]:
        raise ValueError("pilot identity does not match its measurement summary")
    return {
        "schema": "dspark-synthetic-control-v1",
        "model_key": cell["model_key"],
        "model_revision": identity["model"]["revision"],
        "k": k,
        "accepted_tokens": accepted,
        "synthetic_acceptance_length": 1 + accepted,
        "frozen_before_measurement": True,
        "frozen_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "source": str(pilot.resolve()),
        "source_identity_sha256": identity_sha,
        "source_acceptance_sha256": hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
        "sampling_model": "adjacent_integer_mean_control",
        "observed_acceptance_rates": acceptance.get("acceptance_rates"),
        "scope": "Frozen per model/K and reused across deployments; no latency fitting.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    control = freeze(args.pilot)
    with args.output.open("x") as output:
        output.write(json.dumps(control, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
