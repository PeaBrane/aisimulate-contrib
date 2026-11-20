# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The op-latency pin composes every operation family with the system specs into one verified root."""

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisim_core.data.ensure import DataConflictError, DataPin, MissingDataError, ensure_data
from aisim_core.data.systems import PACKAGED_SYSTEMS_DIR
from aisim_core.sdk import ForwardPassPerfModelConfig
from aisim_core.sdk import rust_engine_step as res
from aisim_core.sdk.errors import PerfDataNotAvailableError

from .test_op_publish import _build, _table, tree  # noqa: F401  (fixture reuse)

pytestmark = pytest.mark.unit

CONFIG = {"system": "h200_sxm", "backend": "vllm", "backend_version": "1.0", "model": "any/model"}


@pytest.fixture
def two_families(tree, tmp_path):  # noqa: F811
    """gemm and attention at one hardware/backend/version, as the publisher emits them."""
    v1 = tree / "data/h200_sxm/gemm/vllm/1.0"
    attention = tree / "data/h200_sxm/attention/vllm/1.0"
    attention.mkdir(parents=True)
    pq.write_table(_table(3), attention / "attention_perf.parquet")
    shutil.copyfile(v1 / "collection_meta.yaml", attention / "collection_meta.yaml")
    out = tmp_path / "published"
    _build(tree, out)
    return out


def _pin(root, tmp_path, **kw):
    return DataPin(local_root=root, categories=("op-latency",), cache_dir=tmp_path / "cache", **kw)


def test_all_families_and_the_system_spec_arrive_in_one_root(two_families, tmp_path):
    out = ensure_data(dict(CONFIG), _pin(two_families, tmp_path))
    root = out.config["systems_paths"][0]
    families = {p.parent.parent.parent.name for p in (tmp_path / "cache").rglob("*_perf.parquet")}
    assert {"gemm", "attention"} <= families
    assert (Path(root) / "h200_sxm.yaml").is_file(), "the native database needs the system spec beside its tables"
    assert (Path(root) / "data/h200_sxm/attention/vllm/1.0").is_dir()
    assert len(out.datasets["op-latency"]["configuration_ids"]) == 2


@pytest.fixture
def native_dataset(tmp_path, monkeypatch):
    """Small synthetic tables, published through the real producer; no model downloads."""
    from aisim_core.sdk import utils

    def no_download(*args, **kwargs):
        pytest.fail("the synthetic local model must not download metadata")

    monkeypatch.setattr(utils, "_download_hf_json", no_download)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "vocab_size": 16,
                "torch_dtype": "bfloat16",
            }
        )
    )
    systems = tmp_path / "producer-systems"
    gemm = systems / "data/h200_sxm/gemm/vllm/1.0"
    attention = systems / "data/h200_sxm/attention/vllm/1.0"
    gemm.mkdir(parents=True)
    attention.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"gemm_dtype": "bfloat16", "m": 4, "n": n, "k": k, "latency": 1.0}
                for n in (8, 16, 24, 32)
                for k in (8, 16)
            ]
        ),
        gemm / "gemm_perf.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "batch_size": 1,
                    "isl": 4,
                    "num_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "attn_dtype": "bfloat16",
                    "kv_cache_dtype": "bfloat16",
                    "latency": 7.0,
                }
            ]
        ),
        attention / "context_attention_perf.parquet",
    )
    shutil.copyfile(PACKAGED_SYSTEMS_DIR / "h200_sxm.yaml", systems / "h200_sxm.yaml")
    (systems / "perf_data_reuse_manifest.yaml").write_text("groups: []\n")
    (systems / "query_versions.yaml").write_text(
        'schema_version: 1\ndefaults: {vllm: {current: "1.0", previous: null}}\noverrides: {}\n'
    )
    published = tmp_path / "native-published"
    _build(systems, published)
    config = ForwardPassPerfModelConfig(
        model=str(model),
        system="h200_sxm",
        backend="vllm",
        backend_version="1.0",
        worker_type="prefill",
        estimation_mode="op_level",
        enable_shared_layer=False,
        systems_paths=(str(systems),),
        estimator_config={"correction": {"enabled": False}},
    )
    return published, systems, config


@pytest.mark.parametrize("as_mapping", [False, True])
def test_the_best_available_consumer_receives_the_complete_root(native_dataset, tmp_path, as_mapping):
    published, systems, config = native_dataset
    # Remove the producer tables: prediction must consume both published families.
    shutil.rmtree(systems / "data")
    payload = config.to_dict() if as_mapping else config
    if as_mapping:
        payload.pop("systems_paths")
    model = res.RustForwardPassPerfModel.best_available(payload, dataset=_pin(published, tmp_path))
    provenance = model.diagnostics()["provenance"]
    selected = Path(provenance["selected_systems_root"])
    assert selected != systems
    assert (selected / "h200_sxm.yaml").is_file()
    assert (selected / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").is_file()
    assert (selected / "data/h200_sxm/attention/vllm/1.0/context_attention_perf.parquet").is_file()
    assert model.dataset_revisions == {"op-latency": f"local:{published}"}
    # The single synthetic layer alone has a 7 ms attention measurement;
    # GEMMs and other operations add positive time. This is no accuracy claim.
    assert model.predict_prefill_latency(1, 4) >= 7.0


@pytest.mark.parametrize("as_mapping", [False, True])
@pytest.mark.parametrize("family", ["gemm", "attention"])
def test_incomplete_pin_cannot_fall_back_to_other_operation_tables(native_dataset, tmp_path, as_mapping, family):
    from aisim_core.data.ensure import _matching_entries
    from aisim_core.data.loaders import OpLatencyLoader

    published, systems, config = native_dataset
    payload = config.to_dict() if as_mapping else config
    # The supplied fallback root is valid and complete, so it would mask a
    # missing pinned family if best_available tried it after the ensured root.
    assert res.RustForwardPassPerfModel.best_available(payload).predict_prefill_latency(1, 4) >= 7.0
    entry = next(
        e
        for e in _matching_entries(published, OpLatencyLoader().select(config.to_dict()))
        if e.identity.operation_family == family
    )
    missing_table = "context_attention_perf" if family == "gemm" else "gemm_perf"
    with pytest.raises(PerfDataNotAvailableError, match=missing_table):
        res.RustForwardPassPerfModel.best_available(
            payload, dataset=_pin(published, tmp_path, configuration_id=entry.configuration_id)
        )


def test_second_call_reuses_the_materialized_root(two_families, tmp_path):
    first = ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0]
    again = ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0]
    assert first == again


def test_two_entries_for_one_family_are_a_real_conflict(two_families, tmp_path, monkeypatch):
    from aisim_core.data import ensure as mod

    real = mod._matching_entries
    monkeypatch.setattr(mod, "_matching_entries", lambda root, sel: [*real(root, sel), real(root, sel)[0]])
    with pytest.raises(DataConflictError, match="one operation family"):
        ensure_data(dict(CONFIG), _pin(two_families, tmp_path))


def test_configuration_id_still_selects_one_entry(two_families, tmp_path):
    from aisim_core.data.ensure import _matching_entries
    from aisim_core.data.loaders import OpLatencyLoader

    entries = _matching_entries(two_families, OpLatencyLoader().select(CONFIG))
    out = ensure_data(dict(CONFIG), _pin(two_families, tmp_path, configuration_id=entries[0].configuration_id))
    assert out.datasets["op-latency"]["configuration_ids"] == [entries[0].configuration_id]


def test_unpublished_version_has_no_fallback(two_families, tmp_path):
    with pytest.raises(MissingDataError):
        ensure_data({**CONFIG, "backend_version": "9.9"}, _pin(two_families, tmp_path))


def test_a_dataset_republished_in_place_does_not_serve_the_old_root(tree, tmp_path):  # noqa: F811
    published = tmp_path / "published"
    _build(tree, published)
    first = ensure_data(dict(CONFIG), _pin(published, tmp_path)).config["systems_paths"][0]
    gemm = Path(first) / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet"
    assert pq.read_table(gemm).num_rows == 3
    # Corrected measurements: same identity (same configuration IDs), different table content, same directory.
    pq.write_table(_table(5), tree / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet")
    shutil.rmtree(published)
    _build(tree, published)
    second = ensure_data(dict(CONFIG), _pin(published, tmp_path)).config["systems_paths"][0]
    assert second != first
    assert pq.read_table(Path(second) / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").num_rows == 5
    again = ensure_data(dict(CONFIG), _pin(published, tmp_path)).config["systems_paths"][0]
    assert again == second  # unchanged content still reuses its root


def test_a_changed_packaged_system_spec_does_not_serve_the_old_root(two_families, tmp_path, monkeypatch):
    from aisim_core.data import ensure as mod

    specs = tmp_path / "packaged"
    specs.mkdir()
    (specs / "h200_sxm.yaml").write_text("name: h200_sxm\n")
    monkeypatch.setattr(mod, "PACKAGED_SYSTEMS_DIR", specs)
    first = ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0]
    assert (Path(first) / "h200_sxm.yaml").read_text() == "name: h200_sxm\n"
    (specs / "h200_sxm.yaml").write_text("name: h200_sxm_v2\n")
    second = ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0]
    assert second != first and (Path(second) / "h200_sxm.yaml").read_text() == "name: h200_sxm_v2\n"


def test_declared_donor_versions_arrive_in_the_root(tree, tmp_path):  # noqa: F811
    # gemm 2.0 declares reuse of its gemm_perf from 1.0 (published) and of ghost_perf from 0.9 (not published).
    published = tmp_path / "published"
    _build(tree, published)
    out = ensure_data({**CONFIG, "backend_version": "2.0"}, _pin(published, tmp_path))
    root = Path(out.config["systems_paths"][0])
    assert (root / "data/h200_sxm/gemm/vllm/2.0/reuse.yaml").is_file()
    assert (root / "data/h200_sxm/gemm/vllm/2.0/other_perf.parquet").is_file()
    assert (root / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").is_file(), "the declared donor must be present"
    assert not (root / "data/h200_sxm/gemm/vllm/0.9").exists()  # an unpublished donor is skipped, as natively
    assert len(out.datasets["op-latency"]["configuration_ids"]) == 2


def test_a_version_without_reuse_declares_no_donors(tree, tmp_path):  # noqa: F811
    published = tmp_path / "published"
    _build(tree, published)
    out = ensure_data(dict(CONFIG), _pin(published, tmp_path))
    assert len(out.datasets["op-latency"]["configuration_ids"]) == 1


def test_the_hf_pin_fetches_the_declared_donor_selection(tree, tmp_path, monkeypatch):  # noqa: F811
    from types import SimpleNamespace

    from aisim_core.data import ensure as mod

    published = tmp_path / "published"
    _build(tree, published)
    requested = []

    def fake_fetch(selection, pin, transport, out=None):
        requested.append((selection.framework_version, selection.operation_family, selection.configuration_ids))
        return SimpleNamespace(root=published, downloaded_bytes=0, plan=SimpleNamespace(revision="a" * 40))

    monkeypatch.setattr(mod, "_fetch_pinned", fake_fetch)
    pin = DataPin(repo="org/data", revision="a" * 40, categories=("op-latency",), cache_dir=tmp_path / "cache")
    out = ensure_data({**CONFIG, "backend_version": "2.0"}, pin)
    assert requested == [("2.0", None, ()), ("1.0", "gemm", ())]  # the declared 1.0 donor, then nothing for 0.9
    assert (Path(out.config["systems_paths"][0]) / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").is_file()


@pytest.mark.parametrize(
    "document", [None, {}, {"reuse": {}}, {"reuse": [{"table": "t"}]}, {"reuse": [{"from_version": 1.0}]}]
)
def test_malformed_reuse_declaration_is_refused(document):
    from aisim_core.data.ensure import _donor_versions
    from aisim_core.data.loaders import LoaderError

    with pytest.raises(LoaderError, match="from_version"):
        _donor_versions(document)
    assert _donor_versions({"reuse": [{"from_version": "1.0"}, {"from_version": "0.9"}, {"from_version": "1.0"}]}) == [
        "0.9",
        "1.0",
    ]


# ---------------------------------------------------------------- pinned lookups and manifests


def _tree_files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_a_cold_fetch_brings_the_pinned_lookup_tables(tree, tmp_path):  # noqa: F811
    from aisim_core.data.systems import PACKAGED_SYSTEMS_DIR

    from .test_fetch import FakeTransport

    published = tmp_path / "published"
    _build(tree, published)
    pin = DataPin(repo="org/dataset", revision="a" * 40, categories=("op-latency",), cache_dir=tmp_path / "cache")
    out = ensure_data(dict(CONFIG), pin, transport=FakeTransport(_tree_files(published)))
    root = Path(out.config["systems_paths"][0])
    pinned = (tree / "query_versions.yaml").read_text()
    assert (root / "query_versions.yaml").read_text() == pinned
    assert pinned != (PACKAGED_SYSTEMS_DIR / "query_versions.yaml").read_text()
    assert (root / "perf_data_reuse_manifest.yaml").read_text() == (tree / "perf_data_reuse_manifest.yaml").read_text()


def test_a_missing_pinned_lookup_fails_instead_of_using_the_packaged_one(tree, tmp_path):  # noqa: F811
    published = tmp_path / "published"
    _build(tree, published)
    next(published.rglob("query_versions.parquet")).unlink()
    with pytest.raises(MissingDataError, match="query_versions"):
        ensure_data(dict(CONFIG), _pin(published, tmp_path))


def test_a_lookup_rewritten_with_matching_embedded_fields_is_refused(tree, tmp_path):  # noqa: F811
    import hashlib

    published = tmp_path / "published"
    _build(tree, published)
    path = next(published.rglob("query_versions.parquet"))
    table = pq.read_table(path)
    forged = b"schema_version: 1\ndefaults: {vllm: {current: '9.9', previous: null}}\noverrides: {}\n"
    metadata = {
        **(table.schema.metadata or {}),
        b"aisim.source_yaml": forged,
        b"aisim.source_sha256": hashlib.sha256(forged).hexdigest().encode(),
    }
    pq.write_table(table.replace_schema_metadata(metadata), path)  # both embedded fields changed together
    with pytest.raises(MissingDataError, match="integrity mismatch"):
        ensure_data(dict(CONFIG), _pin(published, tmp_path))


def test_a_replacement_manifest_is_refused_and_leaves_no_destination(tree, tmp_path):  # noqa: F811
    import json

    from aisim_core.data.op_tree import load_op_latency_tree

    published = tmp_path / "published"
    _build(tree, published)
    manifest = next(published.rglob("gemm/vllm/1.0/manifest.json"))
    manifest.write_text(json.dumps(json.loads(manifest.read_text()), indent=4))  # same content, not the pinned bytes
    destination = tmp_path / "dest"
    with pytest.raises(ValueError, match="integrity mismatch"):
        load_op_latency_tree(published, destination)
    assert not destination.exists()


def test_manifests_declaring_different_bytes_for_one_runtime_path_conflict(tree, tmp_path, monkeypatch):  # noqa: F811
    from aisim_core.data import op_tree
    from aisim_core.data.loaders import LoaderError

    published = tmp_path / "published"
    _build(tree, published)
    real = op_tree.read_entry_manifest
    seen = []

    def colliding(tree_root, entry, config=None):
        manifest = real(tree_root, entry, config)
        runtime = [a for a in manifest.artifacts if a.runtime_path is not None]
        if not seen:
            seen.append(runtime[0].runtime_path)
            return manifest
        moved = [
            a.model_copy(update={"runtime_path": seen[0], "sha256": "0" * 64}) if a is runtime[0] else a
            for a in manifest.artifacts
        ]
        return manifest.model_copy(update={"artifacts": moved})

    monkeypatch.setattr(op_tree, "read_entry_manifest", colliding)
    with pytest.raises(LoaderError, match="different content"):
        op_tree.load_op_latency_tree(published, tmp_path / "dest")
    assert not (tmp_path / "dest").exists()


def test_concurrent_materializations_of_one_pin_agree_and_leave_no_staging(two_families, tmp_path):
    import threading

    results = []

    def ensure():
        results.append(ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0])

    threads = [threading.Thread(target=ensure) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert len(results) == 4 and len(set(results)) == 1 and (Path(results[0]) / "h200_sxm.yaml").is_file()
    assert not list((tmp_path / "cache").rglob("*.tmp"))


# ---------------------------------------------------------------- backend version defaults and aliases


@pytest.mark.parametrize(
    "requested,literal",
    [(None, "2.0"), ("current", "2.0"), ("previous", "1.0"), ("1.0", "1.0"), ("2.0", "2.0")],
)
def test_omitted_versions_and_slot_aliases_follow_the_pinned_policy(tree, tmp_path, requested, literal):  # noqa: F811
    published = tmp_path / "published"
    _build(tree, published)  # QUERY_VERSIONS: h200_sxm vllm current 2.0, previous 1.0
    config = {k: v for k, v in CONFIG.items() if k != "backend_version"}
    if requested is not None:
        config["backend_version"] = requested
    out = ensure_data(config, _pin(published, tmp_path))
    assert out.config["backend_version"] == literal
    root = Path(out.config["systems_paths"][0])
    assert (root / f"data/h200_sxm/gemm/vllm/{literal}").is_dir()


def test_an_alias_the_pinned_policy_does_not_define_is_an_error(tree, tmp_path):  # noqa: F811
    published = tmp_path / "published"
    _build(tree, published)
    with pytest.raises(MissingDataError, match="no 'next' version"):  # a frozen-baseline system has no next slot
        ensure_data({**CONFIG, "backend_version": "next"}, _pin(published, tmp_path))


def test_the_hf_pin_resolves_an_omitted_version_from_the_fetched_policy(tree, tmp_path):  # noqa: F811
    from .test_fetch import FakeTransport

    published = tmp_path / "published"
    _build(tree, published)
    pin = DataPin(repo="org/dataset", revision="a" * 40, categories=("op-latency",), cache_dir=tmp_path / "cache")
    config = {k: v for k, v in CONFIG.items() if k != "backend_version"}
    out = ensure_data(config, pin, transport=FakeTransport(_tree_files(published)))
    assert out.config["backend_version"] == "2.0"
    assert (Path(out.config["systems_paths"][0]) / "data/h200_sxm/gemm/vllm/2.0").is_dir()


# ---------------------------------------------------------------- collective dependencies


@pytest.fixture
def with_nccl(tree):  # noqa: F811
    """The serving tables plus the NCCL table the h200_sxm spec declares (misc.nccl_version 2.26.2)."""
    nccl = tree / "data/h200_sxm/comm/nccl/2.26.2"
    other = tree / "data/h200_sxm/comm/nccl/2.99"
    for directory in (nccl, other):
        directory.mkdir(parents=True)
        pq.write_table(_table(2), directory / "nccl_perf.parquet")
        shutil.copyfile(tree / "data/h200_sxm/gemm/vllm/1.0/collection_meta.yaml", directory / "collection_meta.yaml")
    return tree


def test_the_spec_declared_collective_tables_arrive_beside_the_serving_tables(with_nccl, tmp_path):
    published = tmp_path / "published"
    _build(with_nccl, published)
    out = ensure_data(dict(CONFIG), _pin(published, tmp_path))
    root = Path(out.config["systems_paths"][0])
    assert (root / "data/h200_sxm/comm/nccl/2.26.2/nccl_perf.parquet").is_file()
    assert not (root / "data/h200_sxm/comm/nccl/2.99").exists()  # only the version the system spec declares
    assert len(out.datasets["op-latency"]["configuration_ids"]) == 2


def test_the_hf_pin_fetches_the_collective_entries(with_nccl, tmp_path):
    from .test_fetch import FakeTransport

    published = tmp_path / "published"
    _build(with_nccl, published)
    transport = FakeTransport(_tree_files(published))
    pin = DataPin(repo="org/dataset", revision="a" * 40, categories=("op-latency",), cache_dir=tmp_path / "cache")
    out = ensure_data(dict(CONFIG), pin, transport=transport)
    assert any(p.endswith("comm/nccl/2.26.2/nccl_perf.parquet") for p in transport.opened)
    assert (Path(out.config["systems_paths"][0]) / "data/h200_sxm/comm/nccl/2.26.2/nccl_perf.parquet").is_file()
    assert not any("nccl/2.99" in p and p.endswith(".parquet") for p in transport.opened)


def test_a_dataset_without_the_declared_collective_library_is_not_an_error_here(two_families, tmp_path):
    out = ensure_data(dict(CONFIG), _pin(two_families, tmp_path))
    assert len(out.datasets["op-latency"]["configuration_ids"]) == 2


def test_an_entry_whose_manifest_belongs_to_other_hardware_is_refused(tree, tmp_path):  # noqa: F811
    import json

    from aisim_core.data.loaders import LoaderError
    from aisim_core.data.op_tree import load_op_latency_tree, op_latency_entries, read_entry_manifest

    published = tmp_path / "published"
    _build(tree, published)
    (entry,) = [e for e in op_latency_entries(published) if e.identity.framework_version == "1.0"]
    liar = entry.model_copy(update={"identity": entry.identity.model_copy(update={"hardware": "b200_sxm"})})
    with pytest.raises(LoaderError, match="does not match its catalog entry"):
        read_entry_manifest(published, liar)
    with pytest.raises(LoaderError, match="does not match its catalog entry"):
        load_op_latency_tree(published, tmp_path / "dest", entries=[liar])
    assert not (tmp_path / "dest").exists() and json.loads((published / entry.manifest.path).read_text())


def test_a_reused_root_still_verifies_the_indexed_lookup_digests(tree, tmp_path):  # noqa: F811
    import json

    published = tmp_path / "published"
    _build(tree, published)
    ensure_data(dict(CONFIG), _pin(published, tmp_path))  # warm: the composed root now exists in the cache
    index_path = next(published.rglob("op-latency.json"))
    index = json.loads(index_path.read_text())
    index["support"][0]["sha256"] = "0" * 64  # the pinned digest changes; the lookup file itself does not
    index_path.write_text(json.dumps(index))
    dataset_path = published / "catalog/dataset.json"
    dataset = json.loads(dataset_path.read_text())
    reference = dataset["categories"]["op-latency"]["index"]
    reference["sha256"] = __import__("hashlib").sha256(index_path.read_bytes()).hexdigest()
    reference["bytes"] = index_path.stat().st_size
    dataset_path.write_text(json.dumps(dataset))
    with pytest.raises(MissingDataError, match="integrity mismatch"):
        ensure_data(dict(CONFIG), _pin(published, tmp_path))


# ---------------------------------------------------------------- native shared-layer sources


@pytest.fixture
def shared_layer_tree(tree):  # noqa: F811
    """vllm 1.0 plus what the native resolver may borrow: vllm 0.9 (earlier), trtllm 1.0 (the manifest allows it)."""
    meta = tree / "data/h200_sxm/gemm/vllm/1.0/collection_meta.yaml"
    for relative in (
        "data/h200_sxm/gemm/vllm/0.9",
        "data/h200_sxm/attention/vllm/0.9",
        "data/h200_sxm/gemm/trtllm/1.0",
        "data/h200_sxm/gemm/sglang/1.0",  # sglang is not in the reuse manifest group: never a source for vllm
    ):
        directory = tree / relative
        directory.mkdir(parents=True)
        pq.write_table(_table(2), directory / "gemm_perf.parquet")
        shutil.copyfile(meta, directory / "collection_meta.yaml")
    return tree


def _systems_files(root):
    return {p.relative_to(root).as_posix() for p in (root / "data").rglob("*.parquet")}


def test_earlier_versions_and_allowed_cross_backend_sources_join_the_root(shared_layer_tree, tmp_path):
    published = tmp_path / "published"
    _build(shared_layer_tree, published)
    out = ensure_data(dict(CONFIG), _pin(published, tmp_path))
    files = _systems_files(Path(out.config["systems_paths"][0]))
    assert "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet" in files  # requested
    assert "data/h200_sxm/gemm/vllm/0.9/gemm_perf.parquet" in files  # earlier version of the same backend
    assert "data/h200_sxm/attention/vllm/0.9/gemm_perf.parquet" in files  # a family the requested version lacks
    assert "data/h200_sxm/gemm/trtllm/1.0/gemm_perf.parquet" in files  # allowed by perf_data_reuse_manifest
    assert not any("/sglang/" in f for f in files)  # not allowed by the manifest
    assert not any("/vllm/2.0/" in f for f in files)  # newer versions are only reachable through reuse.yaml


def test_explicitly_disabling_shared_layer_keeps_the_root_to_the_requested_version(shared_layer_tree, tmp_path):
    published = tmp_path / "published"
    _build(shared_layer_tree, published)
    out = ensure_data({**CONFIG, "enable_shared_layer": False}, _pin(published, tmp_path))
    files = _systems_files(Path(out.config["systems_paths"][0]))
    assert not any("/0.9/" in f or "/trtllm/" in f for f in files)


def test_the_hf_pin_fetches_the_shared_layer_sources(shared_layer_tree, tmp_path):
    from .test_fetch import FakeTransport

    published = tmp_path / "published"
    _build(shared_layer_tree, published)
    transport = FakeTransport(_tree_files(published))
    pin = DataPin(repo="org/dataset", revision="a" * 40, categories=("op-latency",), cache_dir=tmp_path / "cache")
    out = ensure_data(dict(CONFIG), pin, transport=transport)
    assert any(p.endswith("gemm/vllm/0.9/gemm_perf.parquet") for p in transport.opened)
    assert any(p.endswith("gemm/trtllm/1.0/gemm_perf.parquet") for p in transport.opened)
    assert not any(p.endswith("gemm/sglang/1.0/gemm_perf.parquet") for p in transport.opened)
    assert "data/h200_sxm/gemm/vllm/0.9/gemm_perf.parquet" in _systems_files(Path(out.config["systems_paths"][0]))


def test_partial_shape_coverage_predicts_like_the_complete_source_through_best_available(tmp_path, monkeypatch):
    """vllm 1.0 holds one GEMM shape, 0.9 the rest: the pinned root must predict like the producer's complete tree."""
    from aisim_core.sdk import utils

    monkeypatch.setattr(utils, "_download_hf_json", lambda *a, **k: pytest.fail("no model download"))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "vocab_size": 16,
                "torch_dtype": "bfloat16",
            }
        )
    )

    def producer(name, shapes_by_version):
        systems = tmp_path / name
        for version, ns in shapes_by_version.items():
            directory = systems / f"data/h200_sxm/gemm/vllm/{version}"
            directory.mkdir(parents=True)
            rows = [
                {"gemm_dtype": "bfloat16", "m": 4, "n": n, "k": k, "latency": float(n)} for n in ns for k in (8, 16)
            ]
            pq.write_table(pa.Table.from_pylist(rows), directory / "gemm_perf.parquet")
        attention = systems / "data/h200_sxm/attention/vllm/1.0"
        attention.mkdir(parents=True)
        row = dict(
            batch_size=1, isl=4, num_heads=2, num_key_value_heads=2, head_dim=4,
            attn_dtype="bfloat16", kv_cache_dtype="bfloat16", latency=7.0,
        )  # fmt: skip
        pq.write_table(pa.Table.from_pylist([row]), attention / "context_attention_perf.parquet")
        shutil.copyfile(PACKAGED_SYSTEMS_DIR / "h200_sxm.yaml", systems / "h200_sxm.yaml")
        (systems / "perf_data_reuse_manifest.yaml").write_text("groups: []\n")
        (systems / "query_versions.yaml").write_text(
            'schema_version: 1\ndefaults: {vllm: {current: "1.0", previous: null}}\noverrides: {}\n'
        )
        return systems

    complete = producer("complete", {"1.0": (8,), "0.9": (16, 24, 32)})
    only_current = producer("only-current", {"1.0": (8,)})
    published = tmp_path / "published"
    _build(complete, published)

    def config(**extra):
        return ForwardPassPerfModelConfig(
            model=str(model), system="h200_sxm", backend="vllm", backend_version="1.0", worker_type="prefill",
            estimation_mode="op_level", estimator_config={"correction": {"enabled": False}}, **extra,
        )  # fmt: skip

    reference = res.RustForwardPassPerfModel.best_available(config(systems_paths=(str(complete),)))
    degraded = res.RustForwardPassPerfModel.best_available(config(systems_paths=(str(only_current),)))
    pinned = res.RustForwardPassPerfModel.best_available(config(), dataset=_pin(published, tmp_path))
    expected = reference.predict_prefill_latency(1, 4)
    assert pinned.predict_prefill_latency(1, 4) == expected
    assert degraded.predict_prefill_latency(1, 4) != expected, "the fixture must make the older rows matter"


def test_a_composed_root_edited_in_place_is_rebuilt_from_the_dataset(two_families, tmp_path):
    first = Path(ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0])
    table = first / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet"
    original = table.read_bytes()
    table.chmod(0o644)
    table.write_bytes(b"edited in place")
    second = Path(ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0])
    assert (second / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").read_bytes() == original


def test_an_exported_root_never_aliases_the_cached_tables(native_dataset, tmp_path):
    from aisim_core.data.materialize import materialize_systems_root

    published, systems, config = native_dataset
    shutil.rmtree(systems / "data")
    payload = config.to_dict()
    payload.pop("systems_paths")
    pin = _pin(published, tmp_path)
    before = res.RustForwardPassPerfModel.best_available(payload, dataset=pin).predict_prefill_latency(1, 4)
    exported = materialize_systems_root(payload, pin, tmp_path / "export").root
    table = exported / "data/h200_sxm/attention/vllm/1.0/context_attention_perf.parquet"
    assert table.stat().st_nlink == 1  # a copy, not a hard link into the cache
    pq.write_table(pa.Table.from_pylist([{**pq.read_table(table).to_pylist()[0], "latency": 700.0}]), table)
    after = res.RustForwardPassPerfModel.best_available(dict(payload), dataset=pin).predict_prefill_latency(1, 4)
    assert after == before  # the pinned prediction still uses the dataset's 7 ms, not the edited export


def test_ensuring_the_real_configuration_connected_makes_the_offline_run_complete(tree, tmp_path):  # noqa: F811
    """The documented air-gapped recipe: one connected ensure_data prepares donors too, so the offline run works."""
    from .test_fetch import FakeTransport

    published = tmp_path / "published"
    _build(tree, published)
    cache = tmp_path / "cache"
    connected = DataPin(repo="org/dataset", revision="a" * 40, categories=("op-latency",), cache_dir=cache)
    config = {**CONFIG, "backend_version": "2.0"}  # 2.0 declares 1.0 as a donor
    ensure_data(config, connected, transport=FakeTransport(_tree_files(published)))
    offline = DataPin(repo="org/dataset", revision="a" * 40, categories=("op-latency",), cache_dir=cache, offline=True)
    out = ensure_data(config, offline)  # no transport: everything must come from the cache
    assert (Path(out.config["systems_paths"][0]) / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").is_file()


@pytest.mark.parametrize("victim", ["query_versions.yaml", "perf_data_reuse_manifest.yaml", "h200_sxm.yaml"])
def test_copied_specs_and_generated_lookups_are_verified_before_a_root_is_reused(two_families, tmp_path, victim):
    first = Path(ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0])
    original = (first / victim).read_bytes()
    (first / victim).write_bytes(original + b"\n# edited in the cache\n")
    second = Path(ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0])
    assert (second / victim).read_bytes() == original  # rebuilt, not served damaged


def test_an_edited_cached_system_spec_does_not_reach_the_native_predictor(native_dataset, tmp_path):
    published, systems, config = native_dataset
    shutil.rmtree(systems / "data")
    payload = config.to_dict()
    payload.pop("systems_paths")
    pin = _pin(published, tmp_path)
    baseline = res.RustForwardPassPerfModel.best_available(dict(payload), dataset=pin).predict_prefill_latency(1, 4)
    cached = Path(ensure_data(dict(payload), pin).config["systems_paths"][0]) / "h200_sxm.yaml"
    text = cached.read_text()
    assert "mem_empirical_constant_latency" in text
    cached.write_text(
        text.replace("mem_empirical_constant_latency: 0.000003", "mem_empirical_constant_latency: 1.000003")
    )
    assert cached.read_text() != text
    again = res.RustForwardPassPerfModel.best_available(dict(payload), dataset=pin).predict_prefill_latency(1, 4)
    assert again == baseline  # the damaged copy was rebuilt from the packaged spec


def test_a_local_policy_file_that_the_index_does_not_list_cannot_choose_the_version(tree, tmp_path):  # noqa: F811
    import hashlib
    import json

    published = tmp_path / "published"
    _build(tree, published)
    index_path = next(published.rglob("op-latency.json"))
    index = json.loads(index_path.read_text())
    index.pop("support")  # an index that does not pin the lookups: the parquet beside it is a stray file
    index_path.write_text(json.dumps(index))
    dataset_path = published / "catalog/dataset.json"
    dataset = json.loads(dataset_path.read_text())
    reference = dataset["categories"]["op-latency"]["index"]
    reference.update(sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(), bytes=index_path.stat().st_size)
    dataset_path.write_text(json.dumps(dataset))
    config = {k: v for k, v in CONFIG.items() if k != "backend_version"}
    with pytest.raises(MissingDataError, match="does not index"):
        ensure_data(config, _pin(published, tmp_path))
    # An explicit version never consults the policy.
    assert ensure_data(dict(CONFIG), _pin(published, tmp_path)).config["backend_version"] == "1.0"


def test_a_stale_repair_cannot_delete_a_root_another_caller_just_rebuilt(two_families, tmp_path, monkeypatch):
    import threading

    from aisim_core.data import ensure as mod

    first = Path(ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0])
    table = first / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet"
    original = table.read_bytes()
    table.write_bytes(b"damaged")
    paused, resume, calls = threading.Event(), threading.Event(), []
    real = mod.composed_root_matches

    def pausing(*args, **kwargs):
        matches = real(*args, **kwargs)
        if not matches and not calls:  # the first caller has just decided the root is damaged
            calls.append(True)
            paused.set()
            assert resume.wait(20)
        return matches

    monkeypatch.setattr(mod, "composed_root_matches", pausing)
    results: dict[str, Path] = {}

    def ensure(name):
        results[name] = Path(ensure_data(dict(CONFIG), _pin(two_families, tmp_path)).config["systems_paths"][0])

    a = threading.Thread(target=ensure, args=("a",))
    a.start()
    assert paused.wait(20)
    b = threading.Thread(target=ensure, args=("b",))
    b.start()
    b.join(1.0)
    assert b.is_alive(), "the second caller must wait for the first repair instead of racing it"
    resume.set()
    a.join(20)
    b.join(20)
    assert results["a"] == results["b"]
    assert (results["a"] / "data/h200_sxm/gemm/vllm/1.0/gemm_perf.parquet").read_bytes() == original
