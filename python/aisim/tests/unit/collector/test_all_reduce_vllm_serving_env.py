# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vLLM all-reduce benchmark exports the serving environment on GB200 and GB300.

A collection job passes no NCCL variables, and on these systems the default environment
measures an all-reduce about twice as slow as the one serving runs with NCCL symmetric
memory. Static (ast) checks: the collector needs a framework plus GPUs to import.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.glm53flash_attention_launch import VLLM_ENV, VLLM_NVL72_ENV

pytestmark = pytest.mark.unit

_COLLECTOR = Path(__file__).resolve().parents[3] / "collector" / "network" / "collect_all_reduce.py"
_NAMES = {"VLLM_NVL72_SERVING_ENV", "NVL72_DEVICES", "vllm_serving_env", "export_vllm_serving_env"}


def _load(torch) -> dict:
    tree = ast.parse(_COLLECTOR.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in _NAMES)
        or (isinstance(node, ast.Assign) and {t.id for t in node.targets if isinstance(t, ast.Name)} & _NAMES)
    ]
    assert len(nodes) == len(_NAMES)
    namespace = {"torch": torch, "os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_COLLECTOR), "exec"), namespace)
    return namespace


def _torch(device_name: str | None):
    cuda = SimpleNamespace(is_available=lambda: device_name is not None, get_device_name=lambda index: device_name)
    return SimpleNamespace(cuda=cuda)


@pytest.mark.parametrize("device", ["NVIDIA GB200", "NVIDIA GB300"])
def test_nvl72_systems_get_the_attention_launcher_environment_with_multi_node_nvlink_on(device):
    assert VLLM_NVL72_ENV == {"NCCL_MNNVL_ENABLE": "0"}
    assert _load(_torch(device))["vllm_serving_env"](device) == {**VLLM_ENV, "NCCL_MNNVL_ENABLE": "1"}


@pytest.mark.parametrize("device", ["NVIDIA B300 SXM6 AC", "NVIDIA H200", "NVIDIA L40S", "NVIDIA B200"])
def test_other_systems_keep_the_environment_of_the_job(device, monkeypatch):
    monkeypatch.delenv("VLLM_USE_NCCL_SYMM_MEM", raising=False)
    namespace = _load(_torch(device))
    assert namespace["vllm_serving_env"](device) == {}
    assert namespace["export_vllm_serving_env"]() == {}
    assert "VLLM_USE_NCCL_SYMM_MEM" not in os.environ


def test_the_export_overrides_what_the_job_set(monkeypatch):
    monkeypatch.setenv("VLLM_USE_NCCL_SYMM_MEM", "0")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")
    for name in ("NCCL_CUMEM_ENABLE", "NCCL_P2P_LEVEL", "NCCL_MNNVL_ENABLE"):
        monkeypatch.setenv(name, "unset-by-the-test")
    exported = _load(_torch("NVIDIA GB200"))["export_vllm_serving_env"]()
    assert exported and all(os.environ[name] == value for name, value in exported.items())
    assert os.environ["VLLM_USE_NCCL_SYMM_MEM"] == "1" and os.environ["NCCL_NVLS_ENABLE"] == "1"


def test_nothing_is_exported_without_a_cuda_device():
    assert _load(_torch(None))["export_vllm_serving_env"]() == {}


def test_the_environment_is_exported_before_vllm_builds_its_groups():
    tree = ast.parse(_COLLECTOR.read_text(encoding="utf-8"))
    setup = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "setup_vllm_distributed")
    source = ast.get_source_segment(_COLLECTOR.read_text(encoding="utf-8"), setup)
    export = source.index("export_vllm_serving_env()")
    assert export < source.index("from vllm.distributed")
    assert export < source.index("init_distributed_environment")
    assert export < source.index("init_mp(")
