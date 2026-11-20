# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-tree op-latency loading."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisim_core.data.loaders import LoaderError
from aisim_core.data.op_tree import load_op_latency_tree, op_latency_entries

from .test_op_publish import _build, tree  # noqa: F401  (fixture reuse)

pytestmark = pytest.mark.unit


@pytest.fixture
def published(tree, tmp_path):  # noqa: F811
    out = tmp_path / "published"
    _build(tree, out)
    return tree, out


def test_load_reproduces_packaged_rows_and_lookup_files(published, tmp_path):
    systems, out = published
    loaded = load_op_latency_tree(out, tmp_path / "root", packaged_systems=None)
    assert loaded.entries == 2 and set(loaded.lookup_files) == {
        "perf_data_reuse_manifest.yaml",
        "query_versions.yaml",
    }
    for name in loaded.lookup_files:
        assert (loaded.root / name).read_bytes() == (systems / name).read_bytes()


def test_load_copies_packaged_specs_but_never_packaged_tables(published, tmp_path):
    systems, out = published
    (systems / "h200_sxm.yaml").write_text("gpu: {}\nnode: {}\n")
    loaded = load_op_latency_tree(out, tmp_path / "root", packaged_systems=systems)
    assert (loaded.root / "h200_sxm.yaml").is_file()
    assert (loaded.root / "perf_data_reuse_manifest.yaml").read_bytes() == (
        systems / "perf_data_reuse_manifest.yaml"
    ).read_bytes()


def test_selection_filters(published):
    _, out = published
    assert len(op_latency_entries(out)) == 2
    assert op_latency_entries(out, hardware=["l4"]) == []
    assert len(op_latency_entries(out, framework=["vllm"], hardware=["h200_sxm"])) == 2


def test_tampered_tree_is_rejected_and_leaves_no_root(published, tmp_path):
    _, out = published
    table = next((out / "data/op-latency/h200_sxm/gemm/vllm/1.0").glob("*.parquet"))
    pq.write_table(pa.table({"b": [1]}), table)
    with pytest.raises(ValueError, match="mismatch"):
        load_op_latency_tree(out, tmp_path / "root", packaged_systems=None)
    assert not (tmp_path / "root").exists()


def test_existing_destination_and_empty_selection_are_refused(published, tmp_path):
    _, out = published
    (tmp_path / "root").mkdir()
    with pytest.raises(FileExistsError):
        load_op_latency_tree(out, tmp_path / "root")
    with pytest.raises(LoaderError, match="no op-latency entries"):
        load_op_latency_tree(out, tmp_path / "other", hardware=["l4"])
    assert not (tmp_path / "other").exists()
