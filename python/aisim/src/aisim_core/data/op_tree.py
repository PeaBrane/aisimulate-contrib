# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the whole op-latency tree (Hugging Face layout) into one ``systems/`` root.

:class:`~aisim_core.data.loaders.OpLatencyLoader` materializes one catalog entry (one
``system/family/backend/version``). A predictor needs many entries at once, so this module walks the
category index, verifies every selected entry through that loader and writes a single root that
``systems_paths`` / ``AICONFIGURATOR_SYSTEMS_PATH`` can point at, next to the packaged path (which it never
modifies). The two lookup YAML files (``perf_data_reuse_manifest.yaml``, ``query_versions.yaml``) are rebuilt
byte-for-byte from the source text embedded in their published Parquet tables. Packaged system specs and other
non-op files are copied from the packaged systems directory because the op-latency category does not publish them.

Offline: reads only a local dataset-layout directory. :func:`aisim_core.data.ensure.ensure_data` uses it for the
op-latency pin, so every operation family of one system/backend/version arrives together with its system spec.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from aisim_core.data import CategoryIndex, DatasetIndex, Entry, Manifest, _file, _verify
from aisim_core.data.fetch import DATASET_INDEX_PATH
from aisim_core.data.loaders import LoaderError, OpLatencyLoader
from aisim_core.data.op_schema import DATA_PREFIX, QUERY_VERSIONS_NAME, REUSE_MANIFEST_NAME
from aisim_core.data.systems import PACKAGED_SYSTEMS_DIR

_DATA_DIR = "data"
_SOURCE_YAML_KEY = b"aisim.source_yaml"


@dataclass(frozen=True)
class OpTreeLoad:
    """What :func:`load_op_latency_tree` wrote."""

    root: Path
    entries: int
    files: int
    bytes: int
    lookup_files: tuple[str, ...]
    runtime_paths: frozenset[str]  # every runtime input the manifests declare, relative to ``root``
    seconds: float


def _op_latency_index(tree: Path) -> CategoryIndex:
    tree = Path(tree)
    dataset = DatasetIndex.model_validate_json(_file(tree, DATASET_INDEX_PATH).read_bytes())
    reference = dataset.categories.get("op-latency")
    if reference is None:
        raise LoaderError(f"{tree} has no op-latency category")
    return CategoryIndex.model_validate_json(_verify(tree, reference.index).read_bytes())


def op_latency_entries(
    tree: Path,
    *,
    hardware: Iterable[str] | None = None,
    framework: Iterable[str] | None = None,
) -> list[Entry]:
    """Catalog entries of the ``op-latency`` category under ``tree`` (verified index files)."""
    index = _op_latency_index(tree)
    hardware, framework = set(hardware or ()), set(framework or ())
    return [
        e
        for e in index.entries
        if (not hardware or e.identity.hardware in hardware) and (not framework or e.identity.framework in framework)
    ]


def read_entry_manifest(
    tree: Path, entry: Entry, config: Mapping[str, Any] | None = None, loader: Any | None = None
) -> Manifest:
    """The verified manifest of a catalog entry: the index's pinned digest first, then the contents, then identity.

    A replacement manifest can keep an identity and declare other table bytes; only the catalog's digest for it
    says which manifest the dataset published. ``loader`` is the category loader (default: op-latency).
    """
    _verify(Path(tree), entry.manifest)
    manifest = (loader or OpLatencyLoader()).read_manifest(Path(tree), entry.manifest.path, config)
    if (manifest.configuration_id, manifest.identity, manifest.purpose) != (
        entry.configuration_id,
        entry.identity,
        entry.purpose,
    ):
        # An entry that lists one hardware but points at another's manifest would materialize the wrong files.
        raise LoaderError(f"manifest {entry.manifest.path} does not match its catalog entry")
    return manifest


def composed_root_matches(
    systems: Path,
    tree: Path,
    manifests: Iterable[Manifest],
    packaged_systems: Path | None = PACKAGED_SYSTEMS_DIR,
) -> bool:
    """True when a composed root still holds exactly what :func:`load_op_latency_tree` writes.

    That is every runtime artifact at its declared digest, the lookup YAML rebuilt from the pinned tables, and the
    packaged specs and support files the consumer reads beside them. The source digest cannot see an edit to
    these copies, so a reused root is checked file by file and rebuilt by the caller when this is false.
    """
    systems, tree = Path(systems), Path(tree)
    try:
        for manifest in manifests:
            for artifact in manifest.artifacts:
                if artifact.runtime_path is not None:
                    _verify(systems, artifact.model_copy(update={"path": artifact.runtime_path}))
        lookups = set()
        for name in (REUSE_MANIFEST_NAME, QUERY_VERSIONS_NAME):
            lookups.add(f"{name}.yaml")
            expected = lookup_yaml(tree / DATA_PREFIX / f"{name}.parquet")
            if (systems / f"{name}.yaml").read_bytes() != expected:
                return False
        if packaged_systems is not None:
            for source in sorted(Path(packaged_systems).iterdir()):
                if source.name == _DATA_DIR or source.name in lookups:
                    continue
                if not _same_tree(source, systems / source.name):
                    return False
    except (OSError, ValueError, LoaderError):
        return False
    return True


def _same_tree(source: Path, copy: Path) -> bool:
    """True when ``copy`` holds the same files with the same bytes as ``source`` (a file or a directory)."""
    if source.is_dir():
        if not copy.is_dir():
            return False
        names = {p.name for p in source.iterdir()}
        return all(_same_tree(source / n, copy / n) for n in names)
    return copy.is_file() and hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(copy.read_bytes()).digest()


def verify_op_latency_support(tree: Path) -> None:
    """Verify every file the category index pins as support (the lookup tables) against its digest.

    Reusing a composed root is only safe after this passes, exactly as a cold build requires.
    """
    tree = Path(tree)
    for artifact in _op_latency_index(tree).support:
        _verify(tree, artifact)


def load_op_latency_tree(
    tree: Path,
    destination: Path,
    *,
    hardware: Iterable[str] | None = None,
    framework: Iterable[str] | None = None,
    entries: Sequence[Entry] | None = None,
    packaged_systems: Path | None = PACKAGED_SYSTEMS_DIR,
) -> OpTreeLoad:
    """Verify the selected entries of ``tree`` and write one ``systems/`` root at ``destination``.

    ``destination`` must not exist; a failed call removes it. Every artifact is hash-checked in the tree and again
    after the copy. ``packaged_systems=None`` skips copying specs and other non-op files (the root then holds only
    what the tree publishes). ``entries`` names catalog entries already chosen by the caller (for example by a
    version-aware :class:`~aisim_core.data.fetch.Selection`); ``hardware`` and ``framework`` are ignored then.
    """
    started = time.perf_counter()
    tree, destination = Path(tree), Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; choose a new directory")
    if entries is None:
        entries = op_latency_entries(tree, hardware=hardware, framework=framework)
    if not entries:
        raise LoaderError("no op-latency entries selected")
    destination.mkdir(parents=True)
    files = size = 0
    runtime: set[str] = set()
    try:
        declared: dict[str, tuple[str, int]] = {}
        for entry in sorted(entries, key=lambda e: e.manifest.path):
            manifest = read_entry_manifest(tree, entry)
            if manifest.identity.model is not None:
                raise LoaderError("operation table storage must remain model-independent")
            for artifact in manifest.artifacts:
                if artifact.runtime_path is None:
                    continue
                content = (artifact.sha256, artifact.bytes)
                if declared.setdefault(artifact.runtime_path, content) != content:
                    # Two manifests meeting at one destination must agree; neither may silently replace the other.
                    raise LoaderError(f"manifests declare different content for {artifact.runtime_path}")
                if artifact.runtime_path in runtime:
                    continue
                target = destination / artifact.runtime_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_verify(tree, artifact), target)
                _verify(destination, artifact.model_copy(update={"path": artifact.runtime_path}))
                files, size = files + 1, size + artifact.bytes
                runtime.add(artifact.runtime_path)
        lookups = _write_lookup_yaml(tree, destination)
        if packaged_systems is not None:
            for source in sorted(Path(packaged_systems).iterdir()):
                if source.name == _DATA_DIR or (destination / source.name).exists():
                    continue
                if source.is_dir():
                    shutil.copytree(source, destination / source.name)
                else:
                    shutil.copyfile(source, destination / source.name)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return OpTreeLoad(
        destination, len(entries), files, size, lookups, frozenset(runtime), time.perf_counter() - started
    )


def op_latency_inputs_digest(
    tree: Path, manifests: Iterable[Manifest], packaged_systems: Path | None = PACKAGED_SYSTEMS_DIR
) -> str:
    """SHA-256 of every input :func:`load_op_latency_tree` turns into a root.

    A composed root can be reused only while this digest is unchanged: it covers each manifest's identity and
    runtime-artifact digests, the lookup tables the YAML files are rebuilt from, and the packaged spec files
    copied beside the tables. Configuration IDs alone name an identity, not its content, so a republished
    dataset or a newer package would otherwise keep serving a stale root.
    """
    digest = hashlib.sha256()
    for manifest in sorted(manifests, key=lambda m: m.configuration_id):
        digest.update(f"manifest\0{manifest.configuration_id}\n".encode())
        for artifact in sorted(
            (a for a in manifest.artifacts if a.runtime_path is not None), key=lambda a: a.runtime_path
        ):
            digest.update(f"{artifact.runtime_path}\0{artifact.sha256}\0{artifact.bytes}\n".encode())
    for name in (REUSE_MANIFEST_NAME, QUERY_VERSIONS_NAME):
        path = Path(tree) / DATA_PREFIX / f"{name}.parquet"
        if path.is_file() and not path.is_symlink():
            digest.update(f"lookup\0{name}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode())
    if packaged_systems is not None:
        packaged = Path(packaged_systems)
        for source in sorted(p for p in packaged.rglob("*") if p.is_file()):
            relative = source.relative_to(packaged)
            if relative.parts[0] != _DATA_DIR:
                content = hashlib.sha256(source.read_bytes()).hexdigest()
                digest.update(f"packaged\0{relative.as_posix()}\0{content}\n".encode())
    return digest.hexdigest()


def lookup_yaml(path: Path) -> bytes:
    """The source YAML a lookup Parquet table embeds, after checking its embedded checksum."""
    metadata = pq.read_metadata(path).metadata or {}
    text = metadata.get(_SOURCE_YAML_KEY)
    expected = metadata.get(b"aisim.source_sha256")
    if text is None or expected is None or hashlib.sha256(text).hexdigest() != expected.decode():
        raise LoaderError(f"{Path(path).name} does not embed its source YAML intact")
    return text


def _write_lookup_yaml(tree: Path, destination: Path) -> tuple[str, ...]:
    """Rebuild the lookup YAML files from the source text embedded in their pinned Parquet tables.

    The tables are bound to the category index (``support``) and verified against its digests before their
    metadata is read. A dataset that does not publish one fails the load: the packaged copy is never a substitute
    for a pinned lookup. A tree whose index predates ``support`` can still be read from a local directory, with
    only the checksum embedded in the file to guard it.
    """
    support = {a.path: a for a in _op_latency_index(tree).support}
    written = []
    for name in (REUSE_MANIFEST_NAME, QUERY_VERSIONS_NAME):
        relative = f"{DATA_PREFIX}/{name}.parquet"
        path = Path(tree) / relative
        if relative in support:
            _verify(Path(tree), support[relative])
        elif not path.is_file() or path.is_symlink():
            raise LoaderError(f"the dataset does not publish {relative}; it is required and has no packaged substitute")
        (destination / f"{name}.yaml").write_bytes(lookup_yaml(path))
        written.append(f"{name}.yaml")
    return tuple(written)
