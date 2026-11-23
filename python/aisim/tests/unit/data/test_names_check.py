# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``aisim data check-names``: non-canonical names are reported for every category, with the name to use."""

import argparse
import json

import pytest

from aisim.data_cli import run_check_names
from aisim_core.data import Identity, Manifest, configuration_id
from aisim_core.data.e2e import E2eConfigurationData, E2eManifestBody, ObservationRow, make_metric, write_configuration
from aisim_core.data.names_check import check_names

pytestmark = pytest.mark.unit


def _e2e(root, prefix="", **body_over):
    fields = dict(
        origin="self-collected",
        model_id="Qwen/Qwen3.5-397B-A17B",
        deployment="aggregated",
        hardware="b300_sxm",
        framework="vllm",
        framework_version="0.31.0",
        parallelism={"aggregated": {"tp": 8}},
        parallelism_complete=True,
        precision={"weights": "bfloat16"},
        source_kind="developer",
    )
    fields.update(body_over)
    body = E2eManifestBody(**fields)
    obs = ObservationRow(
        observation_id="o1",
        configuration_id=body.configuration_id(),
        record_sha256="0" * 64,
        source_kind="developer",
        outcome="complete",
    )
    metric = make_metric("o1", "ttft", "median", 1.0, "ms")
    return write_configuration(root, E2eConfigurationData(body, [obs], [metric]), prefix=prefix)


def _fpm_manifest(root, hardware, precision):
    identity = Identity(
        hardware=hardware,
        framework="sglang",
        framework_version="0.5.20",
        operation_family=None,
        model="m/x",
        effective_settings={},
        unresolved_identity=None,
        precision={"weights": precision},
    )
    manifest = Manifest(
        schema_version="aisim-artifact-manifest/v1",
        category="fpm",
        purpose="execution-profile",
        configuration_id=configuration_id("fpm", "execution-profile", identity),
        identity=identity,
        requested_settings=None,
        artifacts=[
            {
                "artifact_id": "a",
                "path": "a.json",
                "role": "profile",
                "format": "json",
                "schema": "s/v1",
                "sha256": "0" * 64,
                "bytes": 1,
            }
        ],
        provenance={},
        evidence_gaps=[],
    )
    (root / "aisim-manifest.json").write_text(manifest.model_dump_json(by_alias=True))


def test_canonical_e2e_and_fpm_are_clean(tmp_path):
    _e2e(tmp_path)
    fpm = tmp_path / "fpm"
    fpm.mkdir()
    _fpm_manifest(fpm, "b300_sxm", "nvfp4")
    assert check_names(tmp_path) == ([], [])


def test_non_canonical_names_are_reported_with_the_canonical_spelling(tmp_path):
    _e2e(tmp_path, hardware="B300", precision={"weights": "bf16"})
    fpm = tmp_path / "fpm"
    fpm.mkdir()
    _fpm_manifest(fpm, "h200", "fp16")
    findings, errors = check_names(tmp_path)
    assert errors == []
    by = {(f.finding.kind, f.finding.value): f.finding.canonical for f in findings}
    assert by[("hardware", "B300")] == "b300_sxm"
    assert by[("precision", "bf16")] == "bfloat16"
    assert by[("hardware", "h200")] == "h200_sxm"
    assert by[("precision", "fp16")] == "float16"


def test_cli_exit_status_and_json(tmp_path, capsys):
    _e2e(tmp_path, hardware="B300")
    assert run_check_names(argparse.Namespace(path=str(tmp_path))) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["findings"][0]["canonical"] == "b300_sxm"
    clean = tmp_path / "clean"
    clean.mkdir()
    _e2e(clean)
    assert run_check_names(argparse.Namespace(path=str(clean))) == 0


@pytest.mark.parametrize("scope", ["root", "category-directory", "manifest-file"])
def test_e2e_artifacts_resolve_from_the_dataset_root_whatever_is_checked(tmp_path, scope):
    manifest = _e2e(tmp_path, prefix="data", hardware="B300")  # artifact paths are dataset-root relative
    target = {
        "root": tmp_path,
        "category-directory": tmp_path / "data" / "e2e",
        "manifest-file": tmp_path / manifest,
    }[scope]
    findings, errors = check_names(target)
    assert errors == []
    assert {(f.finding.kind, f.finding.value) for f in findings} == {("hardware", "B300")}


def test_a_truncated_manifest_in_a_directory_is_an_error_like_the_file_checked_directly(tmp_path, capsys):
    manifest = _e2e(tmp_path)
    path = tmp_path / manifest
    path.write_text(path.read_text()[:40])  # truncated
    findings, errors = check_names(tmp_path)
    assert findings == [] and len(errors) == 1 and errors[0].startswith(manifest)
    assert check_names(path)[1], "checking the file directly reports an error too"
    assert run_check_names(argparse.Namespace(path=str(tmp_path))) == 1  # the documented directory precheck fails
    assert json.loads(capsys.readouterr().out)["ok"] is False
