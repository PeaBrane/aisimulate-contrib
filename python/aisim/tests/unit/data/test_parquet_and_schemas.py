# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jsonschema import Draft202012Validator

from aisim_core.data.parquet import (
    SCHEMA_METADATA_KEY,
    ParquetSchemaError,
    read_parquet,
    read_parquet_schema_id,
    write_parquet,
)
from aisim_core.data.schemas import (
    PACKAGED_SNAPSHOT_DIR,
    SCHEMA_MODELS,
    check_snapshots,
    render_all,
    snapshot_filename,
    write_snapshots,
)

pytestmark = pytest.mark.unit

SCHEMA = "aisim-e2e-test/v1"


def _table():
    return pa.table({"a": [1, 2, None], "b": ["x", None, "z"], "c": [1.5, None, 2.5]})


def test_round_trip_zstd_with_schema_id(tmp_path):
    path = write_parquet(_table(), tmp_path / "sub" / "t.parquet", SCHEMA, extra_metadata={"k": "v"})
    assert read_parquet_schema_id(path) == SCHEMA
    assert read_parquet(path, SCHEMA).equals(_table().replace_schema_metadata(None))
    meta = pq.read_metadata(path)
    assert meta.row_group(0).column(0).compression == "ZSTD"
    assert meta.metadata[b"k"] == b"v"
    assert not list(path.parent.glob(".*.tmp"))


def test_nulls_stay_null(tmp_path):
    out = read_parquet(write_parquet(_table(), tmp_path / "t.parquet", SCHEMA), SCHEMA)
    assert out.column("a").null_count == 1 and out.column("c").null_count == 1


def test_schema_mismatch_and_missing_rejected(tmp_path):
    path = write_parquet(_table(), tmp_path / "t.parquet", SCHEMA)
    with pytest.raises(ParquetSchemaError):
        read_parquet(path, "other/v1")
    assert read_parquet(path, ("other/v1", SCHEMA)).num_rows == 3
    bare = tmp_path / "bare.parquet"
    pq.write_table(_table(), bare)
    assert read_parquet_schema_id(bare) is None
    with pytest.raises(ParquetSchemaError):
        read_parquet(bare, SCHEMA)


def test_cannot_override_schema_key(tmp_path):
    with pytest.raises(ValueError):
        write_parquet(_table(), tmp_path / "t.parquet", SCHEMA, extra_metadata={SCHEMA_METADATA_KEY.decode(): "x"})
    with pytest.raises(ValueError):
        write_parquet(_table(), tmp_path / "t.parquet", "")


def test_failed_write_leaves_no_file(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pq, "write_table", boom)
    with pytest.raises(RuntimeError):
        write_parquet(_table(), tmp_path / "t.parquet", SCHEMA)
    assert list(tmp_path.iterdir()) == []


def test_writes_are_deterministic(tmp_path):
    a = write_parquet(_table(), tmp_path / "a.parquet", SCHEMA).read_bytes()
    b = write_parquet(_table(), tmp_path / "b.parquet", SCHEMA).read_bytes()
    assert a == b


def test_packaged_snapshots_have_no_drift():
    assert check_snapshots(PACKAGED_SNAPSHOT_DIR) == []


def test_snapshots_are_valid_json_schema():
    for text in render_all().values():
        Draft202012Validator.check_schema(json.loads(text))
    assert set(render_all()) == {snapshot_filename(s) for s in SCHEMA_MODELS}


def test_drift_is_detected(tmp_path):
    write_snapshots(tmp_path)
    assert check_snapshots(tmp_path) == []
    victim = next(tmp_path.glob("*.json"))
    victim.write_text(victim.read_text() + " ")
    (tmp_path / "stray.v9.json").write_text("{}")
    problems = check_snapshots(tmp_path)
    assert any("differs" in p for p in problems) and any("unexpected" in p for p in problems)
    victim.unlink()
    assert any("missing" in p for p in check_snapshots(tmp_path))
    # write_snapshots repairs drift and removes stale files.
    write_snapshots(tmp_path)
    assert check_snapshots(tmp_path) == []


def test_check_on_missing_dir(tmp_path):
    assert len(check_snapshots(tmp_path / "nope")) == len(SCHEMA_MODELS)


def test_gpu_time_schema_registered():
    from aisim_core.data.schemas import SCHEMA_MODELS

    assert "aisim-gpu-time/v1" in SCHEMA_MODELS


def test_read_validates_and_reads_the_same_file(tmp_path, monkeypatch):
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    from aisim_core.data import parquet as parquet_module
    from aisim_core.data.parquet import ParquetSchemaError, read_parquet, write_parquet

    target = tmp_path / "t.parquet"
    write_parquet(pa.table({"x": [1, 2]}), target, "a/v1")
    replacement = write_parquet(pa.table({"x": [9]}), tmp_path / "other.parquet", "b/v1")
    real = pq.ParquetFile
    swapped = []

    class Swapping(real):
        """Replaces the path as soon as the file has been opened, as a concurrent writer could."""

        def __init__(self, source, *args, **kwargs):
            super().__init__(source, *args, **kwargs)
            if not swapped:
                swapped.append(os.replace(replacement, target))

    monkeypatch.setattr(parquet_module.pq, "ParquetFile", Swapping)
    table = read_parquet(target, "a/v1")  # the validated file is the one read, never the replacement
    assert table.column("x").to_pylist() == [1, 2]
    with pytest.raises(ParquetSchemaError):  # the replacement itself is still checked on the next read
        read_parquet(target, "a/v1")


@pytest.mark.parametrize(
    "status,value,valid",
    [("valid", None, False), ("missing", 1.0, False), ("valid", 1.0, True), ("missing", None, True)],
)
def test_published_metric_schema_enforces_the_status_value_rule(status, value, valid):
    from jsonschema import Draft202012Validator

    from aisim_core.data.schemas import render_snapshot

    schema = Draft202012Validator(json.loads(render_snapshot("aisim-e2e-metrics/v1")))
    row = {"observation_id": "o", "metric": "ttft", "statistic": "mean", "unit": "ms", "status": status, "value": value}
    assert schema.is_valid(row) is valid
