# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR validation with a fake transport; no network, GitHub or credentials."""

import hashlib
import json

import pytest

from aisim_core.data import Identity, configuration_id
from aisim_core.data.eval import EvalRecord, GpuTime, PredictionIdentity, compute_results, write_truth_table
from aisim_core.data.fetch import Cache
from aisim_core.data.validate import (
    PrHead,
    PrValidationError,
    PrValidationReport,
    StaleHeadError,
    ValidationRequest,
    check_current,
    evaluator_revision,
    main,
    parse_request,
    refuse_credentials,
    resolve_pr_head,
    run_validation,
)

pytestmark = pytest.mark.unit

REPO = "org/dataset"
DSHA = "d" * 40
HEAD = "1" * 40
BASE = "2" * 40
TRUSTED = "3" * 40
SPEC = "a" * 64
PIN = "b" * 64
RECORD_PATH = "eval/record.json"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _art(artifact_id, path, data, fmt, schema, **extra):
    return {
        "artifact_id": artifact_id,
        "path": path,
        "role": "x",
        "format": fmt,
        "schema": schema,
        "sha256": _sha(data),
        "bytes": len(data),
        **extra,
    }


def _predict(inputs):
    return inputs["tokens"] * 1.1


def _identity():
    return Identity(
        hardware="b200_sxm",
        framework="vllm",
        framework_version="0.11.0",
        operation_family=None,
        model="Qwen/Qwen3-32B",
        effective_settings={},
        unresolved_identity=None,
    )


def build_dataset(
    tmp_path,
    claim_scale=1.0,
    *,
    list_truth=True,
    drop_exclusions=False,
    record_hardware=None,
    identity_update=None,
    record_identity=None,
):
    rows = [(f"s{i}", {"tokens": 10 * i}, 10.0 * i) for i in range(1, 5)] + [("bad", {"tokens": 1}, None)]
    art = write_truth_table(tmp_path / "t.parquet", rows)
    truth_bytes = (tmp_path / "t.parquet").read_bytes()
    art = art.model_copy(update={"path": "eval/truth.parquet"})
    cases = [(c, t, _predict(i) * claim_scale) for c, i, t in rows]
    results = compute_results(cases)
    if drop_exclusions:  # a producer that omits an exclusion but keeps every aggregate
        results = results.model_copy(update={"exclusions": []})
    pred = PredictionIdentity.build("fpm_interpolation", {"p": PIN}, SPEC)
    ident = _identity().model_copy(update=identity_update or {})
    cfg = configuration_id("eval", "evaluation-record", ident)
    record = EvalRecord.model_validate(
        {
            "schema_version": "aisim-eval/v1",
            "prediction": pred.model_dump(),
            "truth": {
                "layer": "fpm",
                "configuration_ids": [cfg],
                "identity": ident.model_copy(
                    update={**({"hardware": record_hardware} if record_hardware else {}), **(record_identity or {})}
                ).model_dump(),
                "artifacts": [art.model_dump(by_alias=True)],
                "truth_table_artifact_id": "truth-table",
            },
            "results": results.model_dump(),
            "assumptions": [],
            "gpu_time": GpuTime(authority="unknown").model_dump(),
            "simulation_cost": {"wall_seconds": 1.0, "gpu_seconds": 0.0},
            "producer": {"kind": "human", "name": "t", "evaluator_revision": "r"},
            "tier": "submitted",
            "created_at": "2026-11-07T00:00:00Z",
        }
    )
    rec_name = f"eval/{record.record_id}.json"
    rec_bytes = json.dumps(record.model_dump(by_alias=True, mode="json"), indent=2, sort_keys=True).encode() + b"\n"
    manifest = {
        "schema_version": "aisim-artifact-manifest/v1",
        "category": "eval",
        "purpose": "evaluation-record",
        "configuration_id": cfg,
        "identity": ident.model_dump(),
        "requested_settings": None,
        "artifacts": [
            _art("r", rec_name, rec_bytes, "json", "aisim-eval/v1"),
            *(
                [
                    _art(
                        "truth-table",
                        "eval/truth.parquet",
                        truth_bytes,
                        "parquet",
                        "aisim-eval-truth/v1",
                        rows=len(rows),
                    )
                ]
                if list_truth
                else []
            ),
        ],
        "provenance": {},
        "evidence_gaps": [],
    }
    files = {rec_name: rec_bytes, "eval/truth.parquet": truth_bytes}
    files["eval/manifest.json"] = json.dumps(manifest).encode()
    index = {
        "schema_version": "aisim-category-index/v1",
        "category": "eval",
        "entries": [
            {
                "configuration_id": cfg,
                "purpose": "evaluation-record",
                "identity": ident.model_dump(),
                "manifest": _art(
                    "m", "eval/manifest.json", files["eval/manifest.json"], "json", "aisim-artifact-manifest/v1"
                ),
            }
        ],
    }
    files["catalog/eval.json"] = json.dumps(index).encode()
    dataset = {
        "schema_version": "aisim-dataset-index/v1",
        "dataset_id": REPO,
        "categories": {
            "eval": {
                "root": "eval",
                "layout": "category",
                "index": _art("i", "catalog/eval.json", files["catalog/eval.json"], "json", "aisim-category-index/v1"),
            }
        },
    }
    files["catalog/dataset.json"] = json.dumps(dataset).encode()
    return files, rec_name, cfg, record


class FakeTransport:
    def __init__(self, files, fail=None):
        self.files, self.fail, self.opened = files, fail, []

    def resolve_revision(self, repo, revision):
        return DSHA

    def open_file(self, repo, sha, path):
        if self.fail:
            raise self.fail
        self.opened.append(path)
        yield self.files[path]


def request(rec_name, cfg, **kw):
    base = {
        "schema_version": "aisim-pr-validation/v1",
        "dataset_repo": REPO,
        "dataset_revision": DSHA,
        "record_path": rec_name,
        "configuration_ids": [cfg],
        "hardware": "b200_sxm",
        "model": "Qwen/Qwen3-32B",
        "required_cases": ["s1", "s2"],
    }
    return ValidationRequest.model_validate({**base, **kw})


def pr(head=HEAD):
    return PrHead(REPO, 7, head, BASE, "open")


def factory(record):
    return _predict, record.prediction.fingerprint


def run(tmp_path, files, req, factory=factory, transport=None):
    return run_validation(
        req,
        pr(),
        trusted_sha=TRUSTED,
        transport=transport or FakeTransport(files),
        predictor_factory=factory,
        cache=Cache(tmp_path / "cache"),
    )


def test_pass_binds_all_identities(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    req = request(rec, cfg)
    r = run(tmp_path, files, req)
    assert r.verdict == "pass", r.problems
    assert (r.code_sha, r.dataset_sha, r.request_digest) == (HEAD, DSHA, req.digest())
    assert r.evaluator_revision == evaluator_revision(TRUSTED)
    assert [o.status for o in r.required_cases] == ["pass", "pass"]
    assert r.accuracy is not None and r.accuracy.n_scored == 4
    assert PrValidationReport.model_validate_json(r.model_dump_json()) == r


def test_wrong_claims_fail(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path, claim_scale=1.5)
    r = run(tmp_path, files, request(rec, cfg))
    assert r.verdict == "fail" and any("mape_pct" in p for p in r.problems)
    assert all(o.status == "fail" for o in r.required_cases)


def test_required_case_unpredicted_or_missing_or_excluded(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    r = run(tmp_path, files, request(rec, cfg, required_cases=["bad", "nope"]))
    reasons = {o.case_id: o.reason for o in r.required_cases}
    assert r.verdict == "fail"
    assert reasons["bad"] == "not scored: missing truth" and reasons["nope"] == "not in truth table"
    r = run(tmp_path, files, request(rec, cfg), factory=lambda rec_: (lambda i: None, rec_.prediction.fingerprint))
    assert r.verdict == "fail"


def test_required_case_gate_uses_the_verifiers_exclusions_not_the_producers(tmp_path):
    files, rec, cfg, record = build_dataset(tmp_path, drop_exclusions=True)
    assert record.results.exclusions == []  # the producer claims every case scored
    r = run(tmp_path, files, request(rec, cfg, required_cases=["s1", "bad"]))
    assert r.verdict == "fail"
    assert {o.case_id: (o.status, o.reason) for o in r.required_cases} == {
        "s1": ("pass", None),
        "bad": ("fail", "not scored: missing truth"),
    }


@pytest.mark.parametrize("warm_cache", [False, True])
def test_truth_table_outside_the_selected_evidence_fails_regardless_of_cache(tmp_path, warm_cache):
    files, rec, cfg, _ = build_dataset(tmp_path, list_truth=False)
    if warm_cache:  # the same file at the same commit left by an earlier run
        cached = Cache(tmp_path / "cache").path_for(REPO, DSHA, "eval/truth.parquet")
        cached.parent.mkdir(parents=True)
        cached.write_bytes(files["eval/truth.parquet"])
    transport = FakeTransport(files)
    r = run(tmp_path, files, request(rec, cfg), transport=transport)
    assert r.verdict == "fail" and r.accuracy is None
    assert "truth-table" in r.problems[0] and "eval/truth.parquet" in r.problems[0]
    assert "eval/truth.parquet" not in transport.opened


def test_fingerprint_mismatch_fails(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    r = run(tmp_path, files, request(rec, cfg), factory=lambda rec_: (_predict, "f" * 64))
    assert r.verdict == "fail" and "fingerprint" in r.problems[0]


def test_record_not_selected_fails(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    assert run(tmp_path, files, request("eval/other.json", cfg)).verdict == "fail"


def test_outage_is_error_not_substitution(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    r = run(tmp_path, files, request(rec, cfg), transport=FakeTransport(files, fail=ConnectionError("hf down")))
    assert r.verdict == "error" and "hf down" in r.problems[0]


def test_budget_exceeded_is_error(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    r = run(tmp_path, files, request(rec, cfg, max_bytes=10))
    assert r.verdict == "error" and "BudgetExceededError" in r.problems[0]


def test_changed_head_request_or_evaluator_invalidates(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    req = request(rec, cfg)
    r = run(tmp_path, files, req)
    rev = evaluator_revision(TRUSTED)
    assert check_current(r, head_sha=HEAD, request_digest=req.digest(), evaluator_rev=rev, dataset_sha=DSHA) == []
    assert check_current(r, head_sha="9" * 40, request_digest=req.digest(), evaluator_rev=rev)
    edited = request(rec, cfg, required_cases=["s1"])
    assert edited.digest() != req.digest()
    assert check_current(r, head_sha=HEAD, request_digest=edited.digest(), evaluator_rev=rev)
    assert check_current(r, head_sha=HEAD, request_digest=req.digest(), evaluator_rev=evaluator_revision("4" * 40))
    assert check_current(r, head_sha=HEAD, request_digest=req.digest(), evaluator_rev=rev, dataset_sha="e" * 40)


def test_resolve_pr_head_rules():
    assert resolve_pr_head(pr(), HEAD).head_sha == HEAD
    with pytest.raises(StaleHeadError):
        resolve_pr_head(pr("9" * 40), HEAD)
    with pytest.raises(PrValidationError):
        resolve_pr_head(PrHead(REPO, 1, "abc", BASE, "open"), None)
    with pytest.raises(PrValidationError):
        resolve_pr_head(PrHead(REPO, 1, HEAD, BASE, "closed"), None)
    payload = {"number": 3, "state": "open", "merged": False, "head": {"sha": HEAD}, "base": {"sha": BASE}}
    assert PrHead.from_github(payload, REPO).number == 3
    with pytest.raises(PrValidationError):
        PrHead.from_github({}, REPO)


def test_request_parsing_is_strict():
    ok = {
        "schema_version": "aisim-pr-validation/v1",
        "dataset_repo": REPO,
        "dataset_revision": DSHA,
        "record_path": "a",
        "configuration_ids": ["c"],
        "required_cases": ["x"],
    }
    assert parse_request(json.dumps(ok)).max_bytes > 0
    for bad in (
        {**ok, "dataset_revision": "main"},
        {**ok, "extra": 1},
        {**ok, "max_bytes": 1 << 40},
        {**ok, "required_cases": []},
    ):
        with pytest.raises(PrValidationError):
            parse_request(json.dumps(bad))
    with pytest.raises(PrValidationError):
        parse_request("not json")


def test_refuse_credentials():
    refuse_credentials({"PATH": "/bin"})
    with pytest.raises(PrValidationError):
        refuse_credentials({"HF_TOKEN": "x"})


def test_cli_run_and_recheck(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    files, rec, cfg, _ = build_dataset(tmp_path)
    req = request(rec, cfg)
    (tmp_path / "req.json").write_text(req.model_dump_json())
    payload = {"number": 7, "state": "open", "head": {"sha": HEAD}, "base": {"sha": BASE}}
    (tmp_path / "pr.json").write_text(json.dumps(payload))
    common = [
        "--pr-json",
        str(tmp_path / "pr.json"),
        "--repo",
        REPO,
        "--request-file",
        str(tmp_path / "req.json"),
        "--trusted-sha",
        TRUSTED,
    ]
    out = tmp_path / "report.json"
    run_args = [
        "run",
        *common,
        "--expected-head",
        HEAD,
        "--predictor-factory",
        "aisim_core.data.validate:nope",
        "--cache-dir",
        str(tmp_path / "c"),
        "--output",
        str(out),
        "--summary",
        str(tmp_path / "sum.md"),
    ]
    assert main(run_args, transport=FakeTransport(files)) == 2  # unknown factory is bad input
    i = run_args.index("--predictor-factory") + 1
    bad_factory = [*run_args[:i], "os:path", *run_args[i + 1 :]]
    assert main(bad_factory) == 2  # outside the trusted prefixes

    monkeypatch.setattr("aisim_core.data.validate._load_factory", lambda spec: factory)
    assert main(run_args, transport=FakeTransport(files)) == 0
    assert "PR validation: PASS" in (tmp_path / "sum.md").read_text()
    report = PrValidationReport.model_validate_json(out.read_text())
    assert report.verdict == "pass"

    recheck = ["recheck", *common, "--report", str(out)]
    assert main(recheck) == 0
    failed_out = tmp_path / "failed-report.json"
    failed_out.write_text(report.model_copy(update={"verdict": "fail", "problems": ["x"]}).model_dump_json())
    assert main(["recheck", *common, "--report", str(failed_out)]) == 1  # current binding, but not a pass
    assert "NOT PASSING" in capsys.readouterr().err
    payload["head"]["sha"] = "9" * 40  # the PR head moved
    (tmp_path / "pr.json").write_text(json.dumps(payload))
    assert main(recheck) == 1
    assert "PR head changed" in capsys.readouterr().err
    # a dispatch for the old head is refused outright
    assert main(run_args, transport=FakeTransport(files)) == 2
    monkeypatch.setenv("HF_TOKEN", "x")
    assert main(recheck) == 2


def test_cli_fetches_anonymously_even_with_a_saved_login(tmp_path, monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HOME_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    files, rec, cfg, _ = build_dataset(tmp_path)
    (tmp_path / "req.json").write_text(request(rec, cfg).model_dump_json())
    (tmp_path / "pr.json").write_text(
        json.dumps({"number": 7, "state": "open", "head": {"sha": HEAD}, "base": {"sha": BASE}})
    )
    tokens = []

    class RecordingTransport(FakeTransport):
        def __init__(self, token=None):
            super().__init__(files)
            tokens.append(token)

    monkeypatch.setattr("aisim_core.data.validate.HfTransport", RecordingTransport)
    monkeypatch.setattr("aisim_core.data.validate._load_factory", lambda spec: factory)
    args = [
        "run",
        "--pr-json",
        str(tmp_path / "pr.json"),
        "--repo",
        REPO,
        "--request-file",
        str(tmp_path / "req.json"),
        "--trusted-sha",
        TRUSTED,
        "--predictor-factory",
        "aisim_core.data.validate:x",
        "--cache-dir",
        str(tmp_path / "c"),
        "--output",
        str(tmp_path / "report.json"),
    ]
    assert main(args) == 0
    assert tokens == [False]  # False: no token at all; None would let huggingface_hub use a saved login


def test_default_predictor_fails_closed(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    from aisim_core.data.validate import no_predictor

    with pytest.raises(PrValidationError):
        no_predictor(build_dataset(tmp_path)[3])


def test_required_cases_use_the_rows_that_were_verified_not_a_replaced_file(tmp_path, monkeypatch):
    import pyarrow as pa

    from aisim_core.data import validate as validate_module
    from aisim_core.data.eval import TRUTH_SCHEMA_ID
    from aisim_core.data.parquet import write_parquet

    files, rec, cfg, _ = build_dataset(tmp_path)
    cache = Cache(tmp_path / "cache")
    real_verify = validate_module.verify_record

    def verify_then_replace(*args, **kwargs):
        report = real_verify(*args, **kwargs)
        # After verification the cached table is swapped for one that calls an unverified case "zz" by s1's inputs.
        table = pa.table({"case_id": ["zz"], "inputs_json": ['{"tokens":10}'], "truth_value": [10.0]})
        write_parquet(table, cache.path_for(REPO, DSHA, "eval/truth.parquet"), TRUTH_SCHEMA_ID)
        return report

    monkeypatch.setattr(validate_module, "verify_record", verify_then_replace)
    r = run_validation(
        request(rec, cfg, required_cases=["zz"]),
        pr(),
        trusted_sha=TRUSTED,
        transport=FakeTransport(files),
        predictor_factory=factory,
        cache=cache,
    )
    assert r.verdict == "fail"
    assert [(o.case_id, o.status, o.reason) for o in r.required_cases] == [("zz", "fail", "not in truth table")]


def test_a_record_for_other_hardware_cannot_pass_under_a_selected_envelope(tmp_path):
    # The catalog entry and envelope stay b200_sxm and the request filters b200_sxm; the record inside says h100_sxm.
    files, rec, cfg, _ = build_dataset(tmp_path, record_hardware="h100_sxm")
    received = []

    def spying_factory(record):
        received.append(record.truth.identity.hardware)
        return factory(record)

    r = run(tmp_path, files, request(rec, cfg), factory=spying_factory)
    assert r.verdict == "fail" and "differs" in r.problems[0]
    assert received == []  # the predictor is never built for the wrong cohort


def test_the_request_filters_bind_the_record_identity(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    assert run(tmp_path, files, request(rec, cfg)).verdict == "pass"
    for field, value in (("hardware", "h100_sxm"), ("model", "Qwen/Other")):
        r = run(tmp_path, files, request(rec, cfg, **{field: value}))
        assert r.verdict != "pass", field  # no catalog entry matches the filter


@pytest.mark.parametrize(
    "envelope,record",
    [
        ({"parallelism": {"aggregated": {"tp": 4}}}, {"parallelism": {"aggregated": {"tp": 8}}}),
        ({"precision": {"weights": "bfloat16"}}, {"precision": {"weights": "fp8"}}),
        ({"deployment": "aggregated"}, {"deployment": "disaggregated"}),
        ({"effective_settings": {"max_num_seqs": 64}}, {"effective_settings": {"max_num_seqs": 128}}),
    ],
)
def test_the_whole_effective_identity_is_bound_to_the_selected_manifest(tmp_path, envelope, record):
    files, rec, cfg, _ = build_dataset(tmp_path, identity_update=envelope, record_identity=record)
    received = []

    def spying_factory(loaded):
        received.append(loaded.truth.identity)
        return factory(loaded)

    r = run(tmp_path, files, request(rec, cfg), factory=spying_factory)
    assert r.verdict == "fail" and "differs from its selected manifest" in r.problems[0]
    assert received == []
    # The matching identity still passes.
    (tmp_path / "ok").mkdir()
    matching, rec, cfg, _ = build_dataset(tmp_path / "ok", identity_update=envelope)
    assert run(tmp_path / "ok", matching, request(rec, cfg)).verdict == "pass"


def test_a_report_for_another_pull_request_is_not_current(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)
    req = request(rec, cfg)
    r = run(tmp_path, files, req)  # validated as PR 7 at base BASE
    kwargs = dict(head_sha=HEAD, request_digest=req.digest(), evaluator_rev=evaluator_revision(TRUSTED))
    assert check_current(r, **kwargs, pr_number=7, base_sha=BASE) == []  # the unchanged report still applies
    assert any("PR #7, not #8" in reason for reason in check_current(r, **kwargs, pr_number=8, base_sha=BASE))
    assert any("base changed" in reason for reason in check_current(r, **kwargs, pr_number=7, base_sha="9" * 40))


def test_a_predictor_that_normalizes_its_input_in_place_does_not_change_the_recorded_key(tmp_path):
    files, rec, cfg, _ = build_dataset(tmp_path)

    def normalizing_factory(record):
        def predict(inputs):
            value = _predict(inputs)
            inputs["normalized"] = True  # mutates the dict the verifier handed over
            return value

        return predict, record.prediction.fingerprint

    r = run(tmp_path, files, request(rec, cfg), factory=normalizing_factory)
    assert r.verdict == "pass", r.problems
    assert [o.status for o in r.required_cases] == ["pass", "pass"]
