#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combine dependency inventories into the CSV evidence used by OSRB."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

FIELDS = ["dependency_type", "name", "version", "spdx_license"]
DIFF_FIELDS = ["change", *FIELDS, "prior_version", "prior_spdx_license"]


def identity(row: dict[str, str]) -> tuple[str, str, str]:
    return row["dependency_type"], row["name"], row["version"]


def index_inventory(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    unique = {}
    for row in rows:
        if any(not isinstance(row.get(field), str) or not row[field].strip() for field in FIELDS):
            raise ValueError("invalid license inventory row: missing or empty required fields")
        key = identity(row)
        if key in unique and unique[key] != row:
            raise ValueError("conflicting license metadata for one dependency version")
        unique[key] = row
    return unique


def inventory(directory: Path, *, include_bundled: bool = False) -> list[dict[str, str]]:
    rows = []
    for filename in sorted((directory / "python").glob("*.csv")):
        with filename.open(newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append(
                    {
                        "dependency_type": "python",
                        "name": row["Name"],
                        "version": row["Version"],
                        "spdx_license": row["License"],
                    }
                )
    if not rows:
        raise ValueError("missing Python dependency inventories")
    if include_bundled:
        bundled = sorted((directory / "bundled").rglob("bundled-deps.csv"))
        expected = [
            directory / "bundled" / f"bundled-configs-nightly-{arch}" / "bundled-deps.csv"
            for arch in ("amd64", "arm64")
        ]
        if bundled != expected:
            raise SystemExit("missing or duplicate architecture-specific bundled configuration inventories")
        for filename in bundled:
            with filename.open(newline="") as stream:
                entries = list(csv.DictReader(stream))
            if not entries or any(row["dependency_type"] != "bundled_model_config" for row in entries):
                raise SystemExit("invalid bundled configuration inventory")
            rows.extend(entries)
    with (directory / "cargo-metadata.json").open() as stream:
        metadata = json.load(stream)
    workspace = set(metadata.get("workspace_members") or [])
    for package in metadata["packages"]:
        if package.get("id") not in workspace:
            rows.append(
                {
                    "dependency_type": "crate",
                    "name": package["name"],
                    "version": package["version"],
                    "spdx_license": package.get("license") or "UNKNOWN",
                }
            )
    unique = index_inventory(rows)
    return [unique[key] for key in sorted(unique)]


def previous_nightly() -> list[dict[str, str]]:
    """Keep nightlies compared with scheduled main runs, never manual releases."""
    token = os.environ["GH_API_TOKEN"]
    repository = os.environ["GITHUB_REPOSITORY"]
    current = os.environ["GITHUB_RUN_ID"]

    def get(url: str):
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        request.add_unredirected_header("Authorization", f"Bearer {token}")
        return urllib.request.urlopen(request, timeout=30)

    run_page = 1
    while True:
        with get(
            f"https://api.github.com/repos/{repository}/actions/workflows/"
            f"nightly-ci.yml/runs?branch=main&event=schedule&status=success&per_page=10&page={run_page}"
        ) as response:
            runs = json.load(response).get("workflow_runs", [])
        for run in runs:
            if str(run["id"]) == current:
                continue
            artifact_page = 1
            while True:
                with get(f"{run['artifacts_url']}?per_page=100&page={artifact_page}") as response:
                    artifacts = json.load(response).get("artifacts", [])
                for artifact in artifacts:
                    if artifact["name"] != "license-artifacts" or artifact.get("expired"):
                        continue
                    with get(artifact["archive_download_url"]) as response:
                        data = response.read()
                    with zipfile.ZipFile(io.BytesIO(data)) as archive:
                        for name in archive.namelist():
                            if Path(name).name == "deps.csv":
                                with io.TextIOWrapper(archive.open(name)) as stream:
                                    reader = csv.DictReader(stream)
                                    if sorted(reader.fieldnames or []) != sorted(FIELDS):
                                        raise ValueError("invalid prior license inventory headers")
                                    rows = list(reader)
                                if not rows or any(
                                    None in row or any(not (row[field] or "").strip() for field in FIELDS)
                                    for row in rows
                                ):
                                    raise ValueError("invalid prior license inventory rows")
                                return rows
                    raise RuntimeError("prior license-artifacts is missing deps.csv")
                if len(artifacts) < 100:
                    break
                artifact_page += 1
        if len(runs) < 10:
            return []
        run_page += 1


def difference(rows: list[dict[str, str]], prior: list[dict[str, str]]) -> list[dict[str, str]]:
    current = index_inventory(rows)
    baseline = index_inventory(prior)
    result = []
    for key in sorted(current.keys() | baseline.keys()):
        new, old = current.get(key), baseline.get(key)
        if new and not old:
            result.append({"change": "added", **new, "prior_version": "", "prior_spdx_license": ""})
        elif old and not new:
            result.append(
                {
                    "change": "removed",
                    "dependency_type": old["dependency_type"],
                    "name": old["name"],
                    "version": "",
                    "spdx_license": "",
                    "prior_version": old["version"],
                    "prior_spdx_license": old["spdx_license"],
                }
            )
        elif new and old and new["spdx_license"] != old["spdx_license"]:
            result.append(
                {"change": "changed", **new, "prior_version": old["version"], "prior_spdx_license": old["spdx_license"]}
            )
    return result


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing python/*.csv and cargo-metadata.json")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--base-dir", type=Path, help="Inventories collected from the exact comparison commit")
    baseline.add_argument("--previous-nightly", action="store_true")
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--source-ref", default="")
    args = parser.parse_args(argv)
    rows = inventory(args.directory, include_bundled=args.previous_nightly)
    prior = []
    baseline_status = "not_requested"
    if args.base_dir:
        # Exact-commit CI must fail if its baseline inputs are missing; it must
        # never present an all-added report as a successful comparison.
        prior = inventory(args.base_dir)
        baseline_status = "exact_commit"
    elif args.previous_nightly:
        baseline_status = "previous_scheduled_nightly"
        prior = previous_nightly()
        if not prior:
            print("No prior scheduled license evidence; using an empty baseline")
            baseline_status = "unavailable"
    changes = difference(rows, prior)
    write_csv(args.directory / "deps.csv", FIELDS, rows)
    write_csv(args.directory / "deps-diff.csv", DIFF_FIELDS, changes)
    (args.directory / "evidence.json").write_text(
        json.dumps(
            {
                "source_sha": args.source_sha,
                "base_sha": args.base_sha,
                "source_ref": args.source_ref,
                "baseline_status": baseline_status,
                "dependency_count": len(rows),
                "change_count": len(changes),
                "python_resolution": "runtime requirements resolved at inventory collection time",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"deps.csv: {len(rows)} rows; deps-diff.csv: {len(changes)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
