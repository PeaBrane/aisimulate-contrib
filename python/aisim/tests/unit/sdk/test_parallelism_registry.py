# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path

import pytest

from aisim_core import systems as _systems_pkg
from aisim_core.data.registry import (
    DIMENSIONS,
    RULE_SET_VERSION,
    RULES,
    Context,
    evaluate,
    normalize,
    world_size,
)

pytestmark = pytest.mark.unit


def ctx(par, **kw):
    kw.setdefault("parallelism_complete", True)
    return Context(parallelism=par, **kw)


# rule id -> (valid context, invalid context)
FIXTURES = {
    "U001-positive-degrees": (ctx({"tp": 2}), ctx({"tp": 0})),
    "U002-registered-dimensions": (ctx({"tp": 2, "x-foo": 3}), ctx({"tp": 2, "foo": 3})),
    "U003-world-size-consistent": (
        ctx({"tp": 2, "attention_dp": 2}, declared_world_size=4),
        ctx({"tp": 2, "attention_dp": 2}, declared_world_size=8),
    ),
    "U004-node-layout": (
        ctx({"tp": 8, "pp": 2}, gpus_per_node=8),
        ctx({"tp": 6, "pp": 2}, gpus_per_node=8),
    ),
    "U005-expert-divisibility": (
        ctx({"ep": 4}, num_experts=256),
        ctx({"ep": 3}, num_experts=256),
    ),
    "U006-head-divisibility": (ctx({"tp": 4}, num_attention_heads=32), ctx({"tp": 5}, num_attention_heads=32)),
    "U007-boolean-numeric-agreement": (
        ctx({"tp": 1, "attention_dp": 4}, attention_dp_enabled=True),
        ctx({"tp": 1, "attention_dp": 4}, attention_dp_enabled=False),
    ),
    "M001-minimax-m3-moe-world": (
        ctx({"tp": 2, "attention_dp": 2, "moe_tp": 1, "ep": 4}, model="minimax_m3", num_experts=256),
        ctx({"tp": 2, "attention_dp": 2, "moe_tp": 1, "ep": 2}, model="minimax_m3"),
    ),
    "F-trtllm-moe-world": (
        ctx({"tp": 2, "attention_dp": 4, "moe_tp": 2, "ep": 4}, framework="trtllm"),
        ctx({"tp": 2, "attention_dp": 4, "moe_tp": 2, "ep": 2}, framework="trtllm"),
    ),
    "F-trtllm-cluster-ep": (
        ctx({"tp": 4, "moe_cluster": 2, "moe_tp": 2}, framework="trtllm"),
        ctx({"tp": 4, "moe_cluster": 2, "ep": 2}, framework="trtllm"),
    ),
    "F-trtllm-ep-cp": (
        ctx({"tp": 2, "cp": 2, "ep": 2}, framework="trtllm", cp_type="HELIX"),
        ctx({"tp": 2, "cp": 2, "ep": 2}, framework="trtllm", cp_type="STAR"),
    ),
    "F-vllm-ep": (
        ctx({"tp": 1, "attention_dp": 4, "ep": 4}, framework="vllm"),
        ctx({"tp": 1, "attention_dp": 4, "ep": 2}, framework="vllm"),
    ),
    "F-vllm-dcp": (ctx({"tp": 8, "dcp": 8}, framework="vllm"), ctx({"tp": 8, "dcp": 3}, framework="vllm")),
    "F-vllm-heads": (
        ctx({"tp": 4}, framework="vllm", num_attention_heads=32),
        ctx({"tp": 5}, framework="vllm", num_attention_heads=32),
    ),
    "F-sglang-moe-dp-pp": (
        ctx({"tp": 4, "moe_dp": 2}, framework="sglang"),
        ctx({"tp": 4, "moe_dp": 2, "pp": 2}, framework="sglang"),
    ),
}


def test_every_rule_has_fixtures():
    assert {r.rule_id for r in RULES} == set(FIXTURES)


@pytest.mark.parametrize("rule_id", sorted(FIXTURES))
def test_rule_valid_and_invalid(rule_id):
    good, bad = FIXTURES[rule_id]
    assert not [f for f in evaluate(good).findings if f.rule_id == rule_id]
    findings = [f for f in evaluate(bad).findings if f.rule_id == rule_id]
    assert findings
    rule = next(r for r in RULES if r.rule_id == rule_id)
    expected = rule.severity if rule.verification.verified else "warn"
    assert findings[0].severity == expected


def test_unverified_rules_never_error():
    assert all(r.severity == "warn" for r in RULES if not r.verification.verified)


def test_unknown_input_is_gap_not_finding():
    report = evaluate(Context(parallelism={"tp": 2}, framework="vllm"))
    assert not report.errors
    assert any(g.rule_id == "F-vllm-ep" for g in report.gaps)


def test_gating_and_versioning():
    bad_universal = evaluate(ctx({"tp": 0}))
    assert bad_universal.blocks_import()
    framework_only = evaluate(ctx({"tp": 1, "attention_dp": 4, "ep": 2}, framework="vllm"))
    assert framework_only.blocks_qualification() and not framework_only.blocks_import()
    assert framework_only.rule_set_version == RULE_SET_VERSION
    with pytest.raises(ValueError):
        evaluate(ctx({"tp": 1}), "parallelism-rules/v0")


def test_normalize_and_world_size():
    assert normalize({"tp": 4, "ep": 1, "attention_dp": 1, "pp": 2}) == {"pp": 2, "tp": 4}
    assert world_size({"tp": 2, "attention_dp": 2, "pp": 2, "ep": 4}) == 8
    assert {d.name for d in DIMENSIONS.values() if d.world_role == "multiplies"} == {"tp", "pp", "attention_dp", "cp"}


_LABEL = re.compile(r"-(tp|dp|ep|dcp|pp)(\d+)(?=-|$)")
_FRAMEWORKS = {"vllm", "sglang"}


def _published_profiles():
    root = Path(next(iter(_systems_pkg.__path__)))
    files = sorted(root.glob("profiles/*/hf_dataset.json")) + [root / "dsv41_fpm_hf.json"]
    for f in files:
        if not f.exists():
            continue
        for key in json.loads(f.read_text())["profiles"]:
            yield f.name if f.parent == root else f.parent.name, key


def test_published_fpm_profiles_pass_or_are_flagged():
    """Published FPM labels (tp/dp/ep/dcp) are checked; anything flagged must be an explicit warn/error."""
    seen = errors = 0
    for source, key in _published_profiles():
        degrees = {"dp": "attention_dp"}
        par = {degrees.get(k, k): int(v) for k, v in _LABEL.findall(key)}
        framework = next((f for f in _FRAMEWORKS if f"-{f}-" in key), None)
        report = evaluate(Context(parallelism=par, framework=framework, parallelism_complete=True))
        seen += 1
        errors += len(report.errors)
        assert not report.errors, (source, key, report.errors)
    assert seen >= 40


@pytest.mark.parametrize(
    "par,cp_type,problem",
    [
        ({"tp": 4, "cp": 2, "moe_tp": 4}, "ULYSSES", False),  # ULYSSES leaves cp out of the MoE world size
        ({"tp": 4, "cp": 2, "moe_tp": 8}, "ULYSSES", True),
        ({"tp": 4, "cp": 2, "moe_tp": 8}, "STAR", False),  # other cp types multiply it
        ({"tp": 4, "cp": 2, "moe_tp": 4}, "STAR", True),
        ({"tp": 4, "moe_tp": 4}, None, False),  # without cp the type does not matter
    ],
)
def test_trtllm_moe_world_uses_the_cp_type(par, cp_type, problem):
    report = evaluate(ctx(par, framework="trtllm", cp_type=cp_type))
    findings = [f for f in report.findings if f.rule_id == "F-trtllm-moe-world"]
    assert bool(findings) is problem


def test_trtllm_moe_world_reports_a_gap_for_an_unknown_cp_type():
    report = evaluate(ctx({"tp": 4, "cp": 2, "moe_tp": 4}, framework="trtllm"))
    assert not [f for f in report.findings if f.rule_id == "F-trtllm-moe-world"]
    gap = next(g for g in report.gaps if g.rule_id == "F-trtllm-moe-world")
    assert "cp_type" in gap.missing
