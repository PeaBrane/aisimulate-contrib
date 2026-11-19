<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Catalog skill repair evidence

This directory contains local source tests, synthetic consumer fixtures, and a
separately labeled **native AISim 0.14.0 CPU compatibility run** for PR #937.
Native simulation outputs are not GPU measurements or predictive-accuracy proof.
No catalog agent evaluation or hosted Full CI result is claimed here. Historical
observed-result claims were removed from the skill eval ground truths; workload
inputs, user requirements and illustrative arithmetic remain.

## Producer and fixtures

`producer-contract.json` records the metric whitelist inspected in
[`release/0.14.0` at 77d82652b2f83fd67afefc3141d69a5f06e630be](https://github.com/ai-dynamo/aisim/blob/77d82652b2f83fd67afefc3141d69a5f06e630be/python/aisim/src/aisim/sweeper/score.py).
The pinned file SHA-256 is
`39d74e39651fc613fc5d1e3ac2c06cc5b034bf0f57bfde0b4b8f7daeacbf775b`.
It was verified against the complete source archive used to build the native
wheel; all 26 `_METRIC_KEYS` entries match `producer-contract.json`.
That tuple belongs to the Python candidate producer. The native engine's saved
ledger includes additional raw report metrics, including percentiles and
`num_requests`; the tuple is a shared compatibility floor, not an exhaustive
native metric whitelist. Both summarizers use the shared subset and are tested
against both producer forms. See `native-0.14.0/provenance.json` for the archive,
wheel, toolchain, source pin and scope.

`consumer-fixture.json` is hand-authored synthetic data with explicit expected
fields. It contains a prediction-report shape and a corresponding recommendation
ledger. Its values do not come from a model or hardware benchmark. Tests run
both summarizer copies in text and JSON modes, feed the prediction report
through the checked-out `make_candidate` producer, test export and malformed
artifacts, and exercise ladder classification and headroom JSON parsing.
The synthetic configuration labels are not runtime/backend validation evidence.

The existing `mixed-lengths.jsonl` is a synthetic alternating-length trace.
Its deterministic recipe was reconstructed from the committed bytes; it is not
a production trace. Regeneration and anomaly verification are available with:

```bash
python3 skills/aisim-deployment-search/evals/fixtures/generate_mixed_lengths.py
```

The adjacent `mixed-lengths.summary.json` records the SHA-256, 200 records,
timestamp range/step, length distributions, hash-ID range, uniqueness and
anomaly counts. `--write` regenerates the fixture and summary together.

## Reproduce local checks

Use a Python environment with pytest, pytest-asyncio, pydantic, PyYAML and NumPy.
Set `PYTHON` to its interpreter and `SCRATCH` to a fresh directory outside the
checkout. These tests do not import or build the native runtime.

```bash
export PYTHONPATH=python/aisim/src
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m pytest tests/sweeper/test_catalog_size_ladder.py \
  tests/sweeper/test_catalog_recommendation_summary.py tests/sweeper/test_score.py \
  -q -p no:cacheprovider
"$PYTHON" benchmarks/evidence/skills-catalog/collect_evidence.py \
  --scratch "$SCRATCH/historical-controls" > "$SCRATCH/behavior-evidence.json"
```

`behavior-evidence.json` saves both SLA checkers' results and before/after test
outputs for bdb0a4a0 (arrival windows), 92c2564e (probe timeout), 16da056e
(effective override validation), and f49e7925 (first-budget validation).
Historical failures are intentional negative controls: the same current tests
fail on the pre-fix script and pass on the repaired script. Reruns require a
fresh scratch directory. Each archived script is read with `git show`; no
published history is rewritten.

The fallback and AISim's `aggregate_sla_violations` agree on expected verdicts
at inclusive bounds, just above bounds, and with absent/unqualified numeric
samples. Diagnostic wording differs for missing or zero samples. The fallback
also rejects numeric strings and booleans; these are outside the numeric
prediction-report contract. Agreement validates SLA classification only, not
the accuracy of predictions. Additional tests cover non-finite values.

`local-validation.json` captures command output, environment and the source
commit tested. It distinguishes local checks and fake-runner controls from
hosted CI. The first development test run had 15 test-setup failures because
its new oracle case combined mutually exclusive E2E and TTFT/TPOT bounds. The
case now constructs valid SLA modes and passes; production validation was not
weakened. Initial lint findings in new files were corrected with `pairwise`
and formatting. The system Ruff 0.15.4 also flagged unchanged files; the
repository-pinned Ruff 0.14.1 passes the full paths. A missing documentation
parser dependency in the isolated environment was installed and its check
rerun successfully. Full development logs are retained in the repair run's scratch
directory.

## Native CPU compatibility evidence

`native-0.14.0/` preserves unmodified producer JSON, selected prediction YAMLs,
input YAMLs, byte-preserving gzip stdout/stderr archives, wheel SHA-256, dependency versions, command arguments,
and exit codes. `sha256.json` identifies each saved artifact. The source archive
was fetched with:

```bash
gh api repos/ai-dynamo/aisim/tarball/77d82652b2f83fd67afefc3141d69a5f06e630be > release-source.tar.gz
```

The wheel was built from that archive's `python/aisim` directory with Rust
1.96.0, Python 3.12.3, four Cargo build jobs, and:

```bash
maturin build --release --locked --auditwheel skip --out "$RUN/dist" --interpreter "$RUN/venv/bin/python"
```

The build took 253.26 seconds. `build.json`, `wheel-sha256.txt`, and
`requirements-freeze.txt` identify the locally installed wheel and environment.
The exact executed paths and commands are in `commands.json`. For a fresh run,
install that locally built wheel into a separate environment, put its `bin` on
`PATH`, and use a new output directory for every command. Run from the checkout:

```bash
aisim --version  # expected version 0.14.0
aisim recommend --stack engine --config benchmarks/evidence/skills-catalog/native-0.14.0/recommendation-input.yaml --output-dir "$RUN/recommendation"
python skills/aisim-deployment-search/scripts/summarize_recommendation.py "$RUN/recommendation" --json
python skills/aisim-gpu-selection-and-sizing/scripts/summarize_recommendation.py "$RUN/recommendation"
aisim predict --stack engine --config "$RUN/recommendation/recommendations/0001.yaml" --capture-per-request --format json --output-dir "$RUN/prediction"
```

Both documented deployment templates ran successfully with the optional
8 GiB/four-CPU resource cap. Template A produced 12 evaluated feasible candidates.
The aggregate/disaggregate template produced 16 evaluated feasible candidates.
Their complete ledgers are saved under `recommendation/` and `mode-comparison/`.
The shared mode-comparison input omits `engine.kv_transfer`; all recorded
`kv_transfer_bandwidth` values are null. Its disaggregated branch therefore
assumes **zero KV-transfer delay** and can be optimistic. This is compatibility
smoke evidence, not a deployment trade-off recommendation. A deployment decision
needs a separate disaggregated-only run with a representative
`engine.kv_transfer.bandwidth_gb_per_second`; the shared aggregated/disaggregated
input cannot carry that block because aggregated mode rejects it.

Only the first selected YAML is retained for each search; other concrete configs
remain in the ledgers and can be exported with the summarizer.
Both summarizers succeeded in text and JSON modes. `classify.json` records the
ladder classifier consuming the native recommendation ledger. `ladder.json` also
records a complete native size-ladder invocation for a deliberately small smoke
workload, including its coverage probe, selected config, and candidate ledger.
Its one-GPU result is only for that saved input, not a claim about an eval case.

For headroom, the selected Template A config was reused with `constant_rate`
loads of 1 and 2 requests/s, a 300-second arrival window, TTFT <=500 ms, TPOT
<=50 ms, minimum goodput share 0.9 and minimum compliant share 0.95. These are
explicit smoke-test bounds, not a user workload or a hardware-sizing conclusion.
`headroom.json` and `headroom/x{1,2}/prediction.json` preserve the native metrics.
Both loads passed, with 300/600 requests. The generated config contains absolute
`systems_paths` for its original installation: regenerate the selected config
with the new wheel before rerunning elsewhere, then apply those load/SLA values.
The first harness attempt used the unsupported load name `request_rate`; its
exit-2 validation error is retained separately. The corrected supported
`constant_rate` input passed without a production-code change.

Raw logs retain the runtime's low-fidelity fallback warnings. These runs establish
producer/consumer compatibility only; they do not establish measurement accuracy,
full search coverage, catalog grading, or release availability. Replay tests read
these native bytes without building or running a simulator in ordinary unit CI.

## Remaining external evidence

The public `aisim==0.14.0` PyPI endpoint returned HTTP 404 during this validation.
The source-built wheel above permits native CPU smoke tests; it does not claim
that the public release or the catalog agent eval/signing workflow is available.
No `eval-result.json`, catalog signature, or hosted Full CI result is invented.

The PR description was corrected to distinguish source tests, synthetic controls,
and the native smoke run. Final-head Full CI, catalog eval/signing, and fresh
required bot reviews/reapproval remain separate merge-readiness conditions.
