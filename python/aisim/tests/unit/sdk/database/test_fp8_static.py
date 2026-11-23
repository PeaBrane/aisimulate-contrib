# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from aisim.sdk import common
from aisim.sdk.perf_database import LoadedOpData

pytestmark = pytest.mark.unit


def _gemm_loaded_data(*quant_modes: common.GEMMQuantMode) -> LoadedOpData:
    return LoadedOpData(
        {
            quant_mode: {
                32: {
                    256: {
                        512: {"latency": 1.0, "energy": 10.0},
                    }
                }
            }
            for quant_mode in quant_modes
        },
        common.PerfDataFilename.gemm,
        "dummy_path",
    )


def _overhead_loaded_data(
    filename: common.PerfDataFilename,
    quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.fp8,
) -> LoadedOpData:
    return LoadedOpData(
        {
            quant_mode: {
                32: {
                    512: {"latency": 1.0, "energy": 10.0},
                }
            }
        },
        filename,
        "dummy_path",
    )


@pytest.mark.parametrize("backend", [common.BackendName.trtllm.value, common.BackendName.vllm.value])
def test_supported_quant_modes_include_fp8_static_only_when_base_and_overhead_data_exist(
    mutable_comprehensive_perf_db, backend
):
    db = mutable_comprehensive_perf_db
    db.backend = backend
    db._gemm_data = _gemm_loaded_data(common.GEMMQuantMode.fp8)
    db._update_support_matrix()

    modes = db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8.name in modes
    assert common.GEMMQuantMode.fp8_static.name not in modes

    db._gemm_data = _gemm_loaded_data(common.GEMMQuantMode.fp8_static)
    db._update_support_matrix()
    modes = db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8_static.name not in modes

    db._gemm_data = _gemm_loaded_data(common.GEMMQuantMode.fp8)
    db._compute_scale_data = _overhead_loaded_data(common.PerfDataFilename.compute_scale)
    db._scale_matrix_data = _overhead_loaded_data(common.PerfDataFilename.scale_matrix)
    db._update_support_matrix()

    modes = db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8.name in modes
    assert common.GEMMQuantMode.fp8_static.name in modes
    assert modes.count(common.GEMMQuantMode.fp8_static.name) == 1


def test_sglang_supported_quant_modes_include_fp8_static_only_when_base_and_overhead_data_exist(
    mutable_comprehensive_perf_db,
):
    db = mutable_comprehensive_perf_db
    db.backend = common.BackendName.sglang.value
    db._gemm_data = LoadedOpData(
        {
            common.GEMMQuantMode.fp8: {
                32: {
                    256: {
                        512: {"latency": 1.0, "energy": 10.0},
                    }
                }
            }
        },
        common.PerfDataFilename.gemm,
        "dummy_path",
    )
    db._update_support_matrix()

    modes = db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8.name in modes
    assert common.GEMMQuantMode.fp8_static.name not in modes

    db._gemm_data = _gemm_loaded_data(common.GEMMQuantMode.fp8_static)
    db._update_support_matrix()
    modes = db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8_static.name not in modes

    db._gemm_data = _gemm_loaded_data(common.GEMMQuantMode.fp8)
    db._compute_scale_data = _overhead_loaded_data(common.PerfDataFilename.compute_scale)
    db._scale_matrix_data = _overhead_loaded_data(common.PerfDataFilename.scale_matrix)
    db._update_support_matrix()

    modes = db.supported_quant_mode["gemm"]
    assert common.GEMMQuantMode.fp8.name in modes
    assert common.GEMMQuantMode.fp8_static.name in modes
    assert modes.count(common.GEMMQuantMode.fp8_static.name) == 1


@pytest.mark.parametrize(
    "backend",
    [common.BackendName.trtllm.value, common.BackendName.vllm.value, common.BackendName.sglang.value],
)
def test_supported_quant_modes_admit_fp8_static_from_its_own_rows_with_scale_matrix(
    mutable_comprehensive_perf_db, backend
):
    db = mutable_comprehensive_perf_db
    db.backend = backend
    fp8, fp8_static = common.GEMMQuantMode.fp8, common.GEMMQuantMode.fp8_static
    scale_matrix = _overhead_loaded_data(common.PerfDataFilename.scale_matrix)
    compute_scale = _overhead_loaded_data(common.PerfDataFilename.compute_scale)
    scale_matrix_sq = _overhead_loaded_data(common.PerfDataFilename.scale_matrix, common.GEMMQuantMode.sq)
    cases = [
        # (gemm keys, compute_scale, scale_matrix, fp8_static listed)
        ((fp8_static,), None, None, False),
        ((fp8_static,), None, scale_matrix, True),
        ((fp8_static,), compute_scale, None, False),
        ((fp8, fp8_static), None, scale_matrix, True),
        # The scale_matrix table has no fp8 rows, only sq rows: not listed.
        ((fp8_static,), None, scale_matrix_sq, False),
    ]
    for gemm_keys, cs, sm, listed in cases:
        db._gemm_data = _gemm_loaded_data(*gemm_keys)
        db._compute_scale_data = cs
        db._scale_matrix_data = sm
        db._update_support_matrix()
        modes = db.supported_quant_mode["gemm"]
        assert (fp8_static.name in modes) is listed, (gemm_keys, cs, sm)
        assert modes.count(fp8_static.name) <= 1
        if fp8 in gemm_keys:
            assert fp8.name in modes


@pytest.mark.parametrize("strict_provenance", [False, True])
def test_fp8_static_parquet_rows_reach_the_listing_and_validate(tmp_path, monkeypatch, strict_provenance):
    """Final consumer of a table with fp8_static rows: a parquet file goes
    through the engine table view to the GEMM listing and to Task.validate().
    Synthetic systems root: the public b200_sxm spec with its own data folder
    that holds only bfloat16 and fp8_static GEMM rows plus fp8 scale_matrix
    rows (invented latencies). There is no fp8 GEMM row and no compute_scale
    table, so only the direct rule can list fp8_static."""
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    from aisim.sdk.task_v2 import Task
    from aisim_core.sdk import perf_database as pdb

    shutil.copy(pdb.get_systems_paths()[0] + "/b200_sxm.yaml", tmp_path / "b200_sxm.yaml")
    data = tmp_path / "data" / "b200_sxm"
    gemm_dir = data / "gemm" / "vllm" / "0.30.0"
    quantize_dir = data / "quantize" / "vllm" / "0.30.0"
    gemm_dir.mkdir(parents=True)
    quantize_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "gemm_dtype": ["bfloat16", "bfloat16", "fp8_static", "fp8_static"],
                "m": [128, 256, 128, 256],
                "n": [1024] * 4,
                "k": [1024] * 4,
                "latency": [1.0, 2.0, 0.5, 1.0],
            }
        ),
        gemm_dir / "gemm_perf.parquet",
    )
    pq.write_table(
        pa.table({"quant_dtype": ["fp8", "fp8"], "m": [128, 256], "k": [1024, 1024], "latency": [0.1, 0.2]}),
        quantize_dir / "scale_matrix_perf.parquet",
    )
    # A legacy-schema sidecar in each folder, so that the check of strict
    # provenance accepts the synthetic tables. The data is complete.
    for folder, stem in ((gemm_dir, "gemm_perf"), (quantize_dir, "scale_matrix_perf")):
        meta = {"schema_version": 1, "tables": {stem: {"status": "complete"}}}
        (folder / "collection_meta.yaml").write_text(yaml.safe_dump(meta))
    if strict_provenance:
        monkeypatch.setenv("AIC_STRICT_PROVENANCE", "1")
    else:
        monkeypatch.delenv("AIC_STRICT_PROVENANCE", raising=False)

    previous = pdb.get_systems_paths()
    pdb.set_systems_paths([str(tmp_path)])
    try:
        db = pdb.get_database("b200_sxm", "vllm", "0.30.0", database_mode="SILICON")
        assert db.supported_quant_mode["gemm"] == ["bfloat16", "fp8_static"]
        assert common.GEMMQuantMode.fp8_static in db._gemm_data  # the rows passed the engine table view

        def task(quant, mode, policy=None):
            return Task(
                serving_mode="agg",
                model_path="meta-llama/Meta-Llama-3.1-8B",
                system_name="b200_sxm",
                backend_name="vllm",
                backend_version="0.30.0",
                gemm_quant_mode=quant,
                database_mode=mode,
                transfer_policy=policy,
            )

        task(common.GEMMQuantMode.fp8_static, "SILICON").validate()
        # fp8_static rows are no transfer donor: fp8 needs XPROFILE from bfloat16.
        with pytest.raises(ValueError, match="Unsupported gemm quant mode 'fp8'"):
            task(common.GEMMQuantMode.fp8, "HYBRID", "balanced").validate()
        task(common.GEMMQuantMode.fp8, "HYBRID").validate()
    finally:
        pdb.set_systems_paths(previous)


# ---------------------------------------------------------------------------
# Retired with #1357 PR-5 (single oracle = the compiled engine):
# - query_gemm fp8_static -> dynamic-fp8 table reuse, the structured miss, and
#   the nearest-site transfer (quant normalization + interpolation internals);
# - query_compute_scale / query_scale_matrix (now tombstoned facades);
# - the GEMM.query fp8_static overhead-subtraction closure and its SOL floor —
#   fp8_static is one of the two approved semantic moves: the engine now
#   prices it through the op model rather than the Python subtraction chain.
# Value behaviour is anchored by the frozen parity goldens
# and the frozen parity goldens. The support-matrix admission tests above stay:
# they pin what LOADED data admits fp8_static, which is Python-owned.
# ---------------------------------------------------------------------------
