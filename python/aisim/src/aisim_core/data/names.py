# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical names for the dataset: one spelling per concept, aliases for converters and agents.

Stored data uses only canonical names (``b300_sxm``, ``sglang``, ``ttft``, ``median``, ``profile``).
The registry (``names_registry.json``) also lists the spellings sources use (``B300``, ``dynamo-sglang``,
``p50``, ``preparation``) so converters, collectors and contributors' agents can translate them. Dataset
readers never translate aliases: a stored name that is not canonical is reported by :func:`name_findings`
with the canonical spelling to use, and consumers (dashboard, gym) ignore it until it is registered.

A *closed* kind (hardware, framework, metric statistic, ...) lists every accepted canonical name; a name
outside it is a finding. An *open* kind (model, metric) accepts unknown names unchanged and only flags a
known alias. Offline only; no network access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).with_name("names_registry.json")
REGISTRY_SCHEMA = "aisim-names/v1"


class UnknownNameError(ValueError):
    """A closed kind has no canonical name or alias for the given spelling."""


@dataclass(frozen=True)
class NameFinding:
    """One stored name that is not canonical, with the spelling to use instead."""

    kind: str
    location: str  # where the name was found, e.g. "manifest.hardware" or "metrics[3].statistic"
    value: str
    canonical: str | None  # None: no known canonical name (closed kind) or nothing to suggest

    def __str__(self) -> str:
        hint = f"; use {self.canonical!r}" if self.canonical else "; not in the names registry"
        return f"non-canonical {self.kind} {self.value!r} at {self.location}{hint}"


def _key(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


@lru_cache(maxsize=1)
def registry() -> dict[str, Any]:
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError(f"{REGISTRY_PATH.name}: unsupported schema {data.get('schema_version')!r}")
    return data


@cache
def _alias_index(kind: str) -> dict[str, str]:
    """Normalized spelling -> canonical name (the canonical name maps to itself)."""
    spec = registry()["kinds"][kind]
    index: dict[str, str] = {}
    for canonical, entry in spec["entries"].items():
        index[_key(canonical)] = canonical
        for alias in entry.get("aliases", []):
            index[_key(alias)] = canonical
    return index


def kinds() -> tuple[str, ...]:
    return tuple(registry()["kinds"])


def canonical_names(kind: str) -> tuple[str, ...]:
    return tuple(registry()["kinds"][kind]["entries"])


def aliases(kind: str, canonical: str) -> tuple[str, ...]:
    return tuple(registry()["kinds"][kind]["entries"][canonical].get("aliases", []))


def is_canonical(kind: str, name: str) -> bool:
    """True for an exact canonical name of ``kind``; open kinds also accept any unregistered name."""
    spec = registry()["kinds"][kind]
    if name in spec["entries"]:
        return True
    return not spec["closed"] and _alias_index(kind).get(_key(name)) is None


def canonical_name(kind: str, name: str) -> str | None:
    """Canonical spelling of ``name`` (itself when canonical or an unknown name of an open kind), else None."""
    spec = registry()["kinds"][kind]
    found = _alias_index(kind).get(_key(name))
    if found is not None:
        return found
    return None if spec["closed"] else name


def normalize(kind: str, name: str) -> str:
    """Converter helper: the canonical name for ``name``; closed kinds raise :class:`UnknownNameError`."""
    found = canonical_name(kind, name)
    if found is None:
        raise UnknownNameError(f"unknown {kind} {name!r}; add it to {REGISTRY_PATH.name} or fix the spelling")
    return found


def canonical_unit(metric: str) -> str | None:
    """The one canonical unit of a registered metric name, else None."""
    entry = registry()["kinds"]["metric"]["entries"].get(metric)
    return None if entry is None else entry["unit"]


def split_framework(name: str) -> dict[str, Any]:
    """Split a source framework spelling into canonical ``framework`` and the facts fused into it.

    ``dynamo-sglang`` -> ``{"framework": "sglang", "orchestrator": "dynamo"}``; a plain engine name maps to
    ``{"framework": <canonical>}``. Raises :class:`UnknownNameError` for an unknown engine.
    """
    composite = registry()["kinds"]["framework_composite"]["entries"].get(_key(name).replace("_", "-"))
    if composite is None:
        composite = registry()["kinds"]["framework_composite"]["entries"].get(name)
    if composite is not None:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in composite.items()}
    return {"framework": normalize("framework", name)}


def name_findings(kind: str, location: str, value: str | None) -> list[NameFinding]:
    """Findings for one stored name: empty when canonical (or null, or an unknown name of an open kind)."""
    if value is None or is_canonical(kind, value):
        return []
    return [NameFinding(kind, location, value, canonical_name(kind, value))]
