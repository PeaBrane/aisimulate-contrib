# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The all-reduce collector sets SGLang's all-reduce switches as a default server does.

sglang v0.5.23 ``distributed/bootstrap.py::_set_all_reduce_flags`` sets four module switches
before the parallel groups are built. The collector imported ``set_mscclpp_all_reduce``, which
the release no longer has (it is ``set_mscclpp``), so every SGLang all-reduce run died at import.

From sglang v0.5.21 ``initialize_model_parallel()`` has no width parameters: it builds the groups
from the published topology (``get_parallel().tp_size``) and the launcher's rank, the way SGLang's
own ``benchmark/kernels/all_reduce/benchmark_all_reduce.py`` sets them up."""

import ast
import inspect
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[3] / "collector" / "network" / "collect_all_reduce.py"
_SWITCHES = ("set_custom_all_reduce", "set_mscclpp", "set_torch_symm_mem_all_reduce", "set_flashinfer_allreduce_only")


def _functions(*names):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    ns = {"sys": sys, "inspect": inspect}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def _fake_sglang(monkeypatch, state):
    """parallel_state with exactly the names sglang v0.5.23 exports for this collector."""
    for name in ("sglang", "sglang.srt", "sglang.srt.distributed"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    comm = types.ModuleType("sglang.srt.distributed.communication_op")
    comm.tensor_model_parallel_all_reduce = lambda x: x
    par = types.ModuleType("sglang.srt.distributed.parallel_state")
    plain = ("destroy_model_parallel", "graph_capture", "init_distributed_environment", "initialize_model_parallel")
    for name in plain:
        setattr(par, name, lambda *a, **k: None)
    for name in _SWITCHES:
        setattr(par, name, lambda enable, name=name: state.__setitem__(name, enable))
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed.communication_op", comm)
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed.parallel_state", par)


def test_the_collector_imports_the_switches_the_pinned_release_has(monkeypatch):
    state = {}
    _fake_sglang(monkeypatch, state)
    ns = _functions("import_sglang", "set_sglang_serving_all_reduce_flags")
    mods = ns["import_sglang"]()
    ns["set_sglang_serving_all_reduce_flags"](mods)
    assert state == {
        "set_custom_all_reduce": True,
        "set_mscclpp": False,
        "set_torch_symm_mem_all_reduce": False,
        "set_flashinfer_allreduce_only": False,
    }


def test_the_switches_are_set_before_the_groups_are_built():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    setup = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "setup_sglang_distributed")
    text = ast.unparse(setup)
    assert text.index("set_sglang_serving_all_reduce_flags(") < text.index("sglang_mods['initialize_model_parallel'](")


def _published_groups(*, old_signature):
    """What init_model_parallel is called with: 0.5.20 takes the widths, 0.5.21+ takes none."""
    calls = []
    if old_signature:

        def initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
            calls.append((tensor_model_parallel_size, pipeline_model_parallel_size))

    else:

        def initialize_model_parallel(backend=None, duplicate_tp_group=False, enable_symm_mem=False):
            calls.append(())

    return initialize_model_parallel, calls


def _fake_publish(monkeypatch, published):
    server_args = types.ModuleType("sglang.srt.server_args")

    class ServerArgs:
        resolve_once = True

        def __init__(self, **fields):
            self.fields = fields

    server_args.ServerArgs = ServerArgs
    server_args.set_global_server_args_for_scheduler = lambda args: published.append(("legacy", args.fields, None))
    context = types.ModuleType("sglang.srt.runtime_context")
    context.SpawnRanks = lambda world_rank: {"world_rank": world_rank}
    context.publish = lambda args, *, role, ranks: published.append((role, args.fields, ranks))
    for name in ("sglang", "sglang.srt"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", context)


def test_the_topology_is_published_where_the_groups_are_built_from_it(monkeypatch):
    published = []
    _fake_publish(monkeypatch, published)
    ns = _functions("sglang_groups_take_widths", "publish_sglang_server_args")
    init, _ = _published_groups(old_signature=False)
    assert ns["sglang_groups_take_widths"](init) is False
    ns["publish_sglang_server_args"](4, 3, False)
    assert published == [
        ("scheduler", {"model_path": "dummy", "tp_size": 4, "enable_symm_mem": False}, {"world_rank": 3})
    ]


def test_a_release_that_takes_the_widths_keeps_its_publish(monkeypatch):
    published = []
    _fake_publish(monkeypatch, published)
    ns = _functions("sglang_groups_take_widths", "publish_sglang_server_args")
    init, _ = _published_groups(old_signature=True)
    assert ns["sglang_groups_take_widths"](init) is True
    ns["publish_sglang_server_args"](4, 3, True)
    assert published == [("legacy", {"model_path": "dummy", "enable_symm_mem": False}, None)]


@pytest.mark.parametrize(("old_signature", "expected"), [(False, [()]), (True, [(8, 1)])])
def test_the_groups_are_built_the_way_the_release_takes_them(monkeypatch, old_signature, expected):
    init, calls = _published_groups(old_signature=old_signature)
    ns = _functions("setup_sglang_distributed", "set_sglang_serving_all_reduce_flags", "sglang_groups_take_widths")
    mods = {name: (lambda *a, **k: None) for name in _SWITCHES}
    mods["initialize_model_parallel"] = init
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(set_device=lambda rank: None),
        distributed=types.SimpleNamespace(is_initialized=lambda: True),
    )
    ns.update(import_sglang=lambda: mods, torch=torch, os=__import__("os"))
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert ns["setup_sglang_distributed"](8, 0, False) == (mods, 0)
    assert calls == expected
