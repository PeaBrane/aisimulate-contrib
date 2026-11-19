#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconstruct the synthetic fixture's deterministic recipe and verify its summary.

No production trace or external data is used. Default mode verifies the checked-in
bytes and summary; --write regenerates them in this script's own directory.
"""

import argparse
import hashlib
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path


def artifacts():
    rows = []
    next_hash = 0
    for index in range(200):
        isl, osl = (512, 16) if index % 2 == 0 else (2048, 1024)
        count = isl // 512
        rows.append(
            {
                "timestamp": index * 100,
                "input_length": isl,
                "output_length": osl,
                "hash_ids": list(range(next_hash, next_hash + count)),
            }
        )
        next_hash += count
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    hashes = [h for row in rows for h in row["hash_ids"]]
    summary = {
        "provenance": "synthetic deterministic alternating lengths; recipe reconstructed from the committed fixture",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "records": len(rows),
        "block_size_tokens": 512,
        "timestamp_unit": "ms",
        "timestamp_range": [rows[0]["timestamp"], rows[-1]["timestamp"]],
        "timestamp_step": 100,
        "input_length_counts": dict(sorted(Counter(row["input_length"] for row in rows).items())),
        "output_length_counts": dict(sorted(Counter(row["output_length"] for row in rows).items())),
        "hash_id_range": [min(hashes), max(hashes)],
        "unique_hash_ids": len(set(hashes)),
        "anomalies": {
            "non_increasing_timestamps": sum(a["timestamp"] >= b["timestamp"] for a, b in pairwise(rows)),
            "invalid_lengths": sum(row["input_length"] <= 0 or row["output_length"] <= 0 for row in rows),
            "block_count_mismatches": sum(len(row["hash_ids"]) != row["input_length"] // 512 for row in rows),
            "repeated_hash_ids": len(hashes) - len(set(hashes)),
            "hash_ids_outside_expected_range": sum(not 0 <= h < 500 for h in hashes),
        },
    }
    return raw, (json.dumps(summary, indent=2) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    raw, summary = artifacts()
    for name, data in (("mixed-lengths.jsonl", raw), ("mixed-lengths.summary.json", summary)):
        path = root / name
        if args.write:
            path.write_bytes(data)
        elif path.read_bytes() != data:
            parser.exit(1, f"{name} does not match the deterministic recipe\n")
    print("synthetic trace bytes and anomaly summary verified")


if __name__ == "__main__":
    main()
