#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply a dev-version suffix to every AISim version site, in place.

Invoked by nightly CI before `scripts/release/build_release_artifacts.py`. Takes one
argument -- a PEP 440 suffix like '.dev202609170000001234' -- and rewrites:
  - [project].version in python/aisim/pyproject.toml (PEP 440 form)
  - [workspace.package].version in Cargo.toml, inherited by the core crate,
    and explicit core versions in historical manifests (SemVer form: dash instead of
    dot, so '0.15.0-dev.202609170000001234' -- cargo rejects the PEP 440 spelling)

The suffix is the UTC creation date followed by a ten-digit workflow run
number; legacy date-only suffixes remain accepted. The crate form keeps the
suffix as a dotted numeric identifier so SemVer pre-release
ordering compares it numerically (matching the crate's published lineage on
crates.io). Idempotent: re-running with the same suffix is a no-op, and an
empty suffix changes nothing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PYPROJECT = Path("python/aisim/pyproject.toml")
# GitLab copies this file alone and runs it with system Python (including <3.11).
VERSION_LINE_RE = re.compile(r"""^[ \t]*version\s*=\s*(["'])([^"']+)\1\s*(?:#.*)?$""")
INHERITED_VERSION_RE = re.compile(r"version\s*(?:\.\s*workspace\s*=\s*true|=\s*\{\s*workspace\s*=\s*true\s*\})")
SUFFIX_RE = re.compile(r"^\.dev[0-9]{8}(?:[0-9]{10})?$")


def semver(suffix: str) -> str:
    return "-dev." + suffix[len(".dev") :]


def rewrite(path: Path, table: str, tail: str) -> str | None:
    lines = path.read_text().splitlines(keepends=True)
    in_table = False
    for index, line in enumerate(lines):
        entry = line.partition("#")[0].strip()
        if entry.startswith("["):
            in_table = entry == f"[{table}]"
        elif in_table:
            if table == "package" and INHERITED_VERSION_RE.fullmatch(entry):
                return None  # Cargo inherits the stamped workspace version.
            match = VERSION_LINE_RE.fullmatch(line.rstrip("\r\n"))
            if match:
                version = match[2]
                if not version.endswith(tail):
                    end = match.end(2)
                    lines[index] = line[:end] + tail + line[end:]
                    path.write_text("".join(lines))
                    version += tail
                return version
    raise SystemExit(f"no supported version declaration in [{table}] in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suffix", help="e.g. .dev20260827 (empty = no-op)")
    parser.add_argument("root", nargs="?", default=".", help="repo root")
    args = parser.parse_args()

    if not args.suffix:
        print("apply_dev_version: empty suffix, no-op", file=sys.stderr)
        return 0
    if not SUFFIX_RE.fullmatch(args.suffix):
        raise SystemExit(f"suffix must be .devYYYYMMDD with an optional ten-digit run number, got {args.suffix!r}")

    root = Path(args.root).resolve()
    project = root / PYPROJECT
    if not project.is_file():
        project = root / "python/aisimulate/pyproject.toml"
    crate_suffix = semver(args.suffix)
    stamped = [
        rewrite(project, "project", args.suffix),
        rewrite(root / "Cargo.toml", "workspace.package", crate_suffix),
        rewrite(root / "crates/core/Cargo.toml", "package", crate_suffix),
    ]
    print(f"apply_dev_version: {', '.join(v for v in stamped if v is not None)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
