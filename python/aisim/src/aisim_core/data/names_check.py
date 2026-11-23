# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a dataset directory for non-canonical names: the pre-PR check for contributors and their agents.

Applies to every category (op-latency, FPM, e2e, eval): the envelope identity of each configuration manifest
(hardware, framework, precision, deployment) is checked, and e2e configurations are checked down to their
table values. Nothing is rewritten; each finding names the canonical spelling to use. Offline only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aisim_core.data import Manifest
from aisim_core.data.e2e import load_e2e, name_findings_for_dataset
from aisim_core.data.loaders import LoaderError
from aisim_core.data.names import NameFinding, name_findings

MANIFEST_NAMES = ("manifest.json", "aisim-manifest.json")
MANIFEST_SCHEMA = "aisim-artifact-manifest/v1"


@dataclass(frozen=True)
class PathFinding:
    path: str  # manifest path relative to the checked root
    finding: NameFinding

    def to_dict(self) -> dict[str, Any]:
        f = self.finding
        return {"path": self.path, "kind": f.kind, "location": f.location, "value": f.value, "canonical": f.canonical}


def discover_manifests(root: Path) -> tuple[list[Path], list[str]]:
    """``(manifests, errors)`` under ``root``; a file is checked on its own.

    A file with a manifest name that cannot be read or parsed is an error, never silently dropped: a truncated
    manifest in a directory must fail the check as it does when the file is checked directly. A readable JSON
    document with another schema is simply not ours.
    """
    root = Path(root)
    if root.is_file():
        return [root], []
    found: list[Path] = []
    errors: list[str] = []
    for name in MANIFEST_NAMES:
        for path in sorted(root.rglob(name)):
            try:
                head = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(f"{path.relative_to(root).as_posix()}: {type(exc).__name__}: {exc}")
                continue
            if isinstance(head, dict) and head.get("schema_version") == MANIFEST_SCHEMA:
                found.append(path)
    return found, errors


def iter_manifests(root: Path) -> list[Path]:
    """Configuration manifests under ``root`` (see :func:`discover_manifests` for the unreadable ones)."""
    return discover_manifests(root)[0]


def _identity_findings(manifest: Manifest) -> list[NameFinding]:
    identity = manifest.identity
    out = name_findings("hardware", "identity.hardware", identity.hardware)
    out += name_findings("framework", "identity.framework", identity.framework)
    out += name_findings("deployment", "identity.deployment", identity.deployment)
    for key, value in (identity.precision or {}).items():
        out += name_findings("precision", f"identity.precision.{key}", value)
    return out


def _dataset_root(path: Path, manifest: Manifest) -> Path:
    """The directory the manifest's artifact paths are relative to, found by walking up from the manifest.

    Artifact paths are dataset-root relative, so a manifest checked on its own or through a category directory
    still resolves its tables.
    """
    for candidate in (path.parent, *path.parent.parents):
        if all(candidate.joinpath(*a.path.split("/")).is_file() for a in manifest.artifacts):
            return candidate
    raise LoaderError(f"cannot find the dataset root that holds the artifacts of {path}")


def check_names(root: Path | str) -> tuple[list[PathFinding], list[str]]:
    """``(findings, errors)`` for every configuration manifest under ``root``.

    ``errors`` lists manifests that could not be read or verified; they are not name findings.
    """
    root = Path(root)
    base = root if root.is_dir() else root.parent
    findings: list[PathFinding] = []
    errors: list[str] = []
    manifests, discovery_errors = discover_manifests(root)
    errors.extend(discovery_errors)
    for path in manifests:
        relative = path.relative_to(base).as_posix()
        try:
            manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
            found = _identity_findings(manifest)
            if manifest.category == "e2e":
                dataset_root = _dataset_root(path, manifest)
                found += name_findings_for_dataset(load_e2e(dataset_root, path.relative_to(dataset_root).as_posix()))
        except (LoaderError, ValueError, OSError) as exc:
            errors.append(f"{relative}: {type(exc).__name__}: {exc}")
            continue
        findings.extend(PathFinding(relative, f) for f in found)
    return findings, errors
