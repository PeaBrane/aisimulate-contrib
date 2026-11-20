<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Historical Qwen hybrid-cache diagnostic

[historical-accuracy.json](historical-accuracy.json) restores the measurement
receipt originally committed in this PR at
`f2a460101cfb09bdb52c00ddf2680bc16b0dba84`, formerly under
`docs/replay/engine/mamba-cache-accuracy.json`. The prose typo `all95` is
corrected and the scalar-test citation now identifies its archived source.
The measured values are unchanged. These are historical author-reported
measurements, not a rerun on the repaired head. The input captures are not
distributed in this repository and were unavailable to the repair agent.

The capture `qwen-tp4-c1-source24-5563558` used Qwen3.5-397B-A17B FP8,
4 B200 GPUs, SGLang TP4 / attention DP 1, one lane, STP, page size 64 and
chunked prefill size 16,384. Its 95 requests came from 24 selected short trace
windows, with outputs capped at 128 and inter-request gaps capped at one
second. It is not an unmodified published benchmark or a submission result.

| Historical metric | Before | After |
| --- | ---: | ---: |
| Exact cached-prefix agreement, 95 requests | 77/95 | 95/95 |
| Cached-prefix MAE | 663.58 tokens | 0 tokens |
| Conditional first-prefill step MAPE | 7.3890% | 0.02392% |
| Client TTFT MAPE, 94 profiling requests | 50.8630% | 47.1290% |
| Client TPOT MAPE, 94 profiling requests | 2.6444% | 2.6444% |

The fixed FPM was trained on this same capture: **zero independent test rows**.
The first-step score evaluates that unchanged forest at the admission geometry;
it does not establish prediction accuracy on held-out data or aligned later
batches. Native generated output IDs and complete live checkpoint inventory
were not captured. The remaining client TTFT error is unresolved. The receipt
does not qualify memory sizing, state eviction, downstream deployment recipes,
or current-head accuracy. Current repair tests instead assert synthetic cache
and scheduler transitions and unsupported-scope rejection.

## Reproduce the historical comparison

Use the exact six inputs named by `inputs_sha256` in the receipt: execution
descriptor, observed agentic trace, frozen forest, canonical native FPM rows,
prompt map and client metrics. Verify their SHA-256 values first. The trace
header must bind the native prompt map, and the execution descriptor must
reference the captured trace and forest. No model weights or GPU rerun are
needed. Missing inputs mean the measurement cannot be independently reproduced.

Build separate checkouts of baseline
`4ff7bd2ab8748c4c843db8a1755eb0bc59c8a0c6` and behavior candidate
`93a9c551555331b0b9825b9f6c332bcc1819abfe`, with the same toolchain and flags:

```bash
cargo +1.99.0 build --release --locked -p aisim-core --features extension-module
```

<!-- coderabbit-ai-generated:v1 -->

<!-- coderabbit-ai-generated-section:v1:8d8b27909fa07d7dfeed116170b4778683527f0feae695c394a1a83de8ddf971 -->

Set `MAMBA_BASE_CHECKOUT` and `MAMBA_FIXED_CHECKOUT` to the absolute paths of
the baseline and candidate checkouts, respectively. Point each runtime variable
to its checkout's built library and verify both files exist:

```bash
export MAMBA_BASE_RUNTIME="$MAMBA_BASE_CHECKOUT/target/release/libaisimulate_core.so"
export MAMBA_FIXED_RUNTIME="$MAMBA_FIXED_CHECKOUT/target/release/libaisimulate_core.so"
test -f "$MAMBA_BASE_RUNTIME" && test -f "$MAMBA_FIXED_RUNTIME"
```

Proceed only if the check succeeds. Set `MAMBA_EXECUTION` to the captured
execution descriptor, `MAMBA_NATIVE` to its native postprocess directory and
`MAMBA_RESULTS` to a fresh scratch output directory. Extract the exact
historical scorer from this repository's immutable commit, keeping it outside
the checkout:


<!-- /coderabbit-ai-generated:v1 -->
```bash
git show f2a460101cfb09bdb52c00ddf2680bc16b0dba84:scripts/prediction_regression/compare_mamba_cache_accuracy.py > "$MAMBA_RESULTS/scorer.py"
python3 "$MAMBA_RESULTS/scorer.py" run \
  --runtime "$MAMBA_BASE_RUNTIME" --revision 4ff7bd2ab8748c4c843db8a1755eb0bc59c8a0c6 \
  --execution "$MAMBA_EXECUTION" --output "$MAMBA_RESULTS/before"
python3 "$MAMBA_RESULTS/scorer.py" run \
  --runtime "$MAMBA_FIXED_RUNTIME" --revision 93a9c551555331b0b9825b9f6c332bcc1819abfe \
  --execution "$MAMBA_EXECUTION" --output "$MAMBA_RESULTS/after" \
  --mamba-config '{"prefill_checkpoint_interval":64,"decode_checkpoint_interval":256,"max_states":994}'
python3 "$MAMBA_RESULTS/scorer.py" compare \
  --before "$MAMBA_RESULTS/before" --after "$MAMBA_RESULTS/after" \
  --native-canonical "$MAMBA_NATIVE/canonical/canonical-fpm.jsonl" \
  --native-prompts "$MAMBA_NATIVE/prompt-map/native-request-token-map.jsonl" \
  --native-client-metrics "$MAMBA_NATIVE/request-metrics.json" \
  --output "$MAMBA_RESULTS/accuracy.json"
```

This frozen scorer is retained in Git history for reproducing this capture,
not restored as a supported general-purpose tool. It does not validate which
side enables Mamba and assumes the captured forest's `-1` leaf sentinel. Check
that `before/execution.json` omits `mamba_cache` and `after/execution.json`
enables exactly the receipt's 64/256/994 configuration before interpreting
its output. Do not use it with other leaf sentinels. Compare the executed
payload, runtime, runner and report hashes against the receipt. The receipt's
forest-interpreter control covers all 210 captured prefill rows only.

Repeat each run into fresh directories and use the scorer's `parity`
subcommand with `--first`, `--second` and `--output`. It excludes only
`wall_time_ms`, `processed_tokens_per_s` and `processed_output_tokens_per_s`.
Also compare the baseline with the candidate run without `--mamba-config`.
These historical parity controls are separate from numerical accuracy and
from host performance measurements.

## Changes after the measured candidate

The receipt measures `93a9c551555331b0b9825b9f6c332bcc1819abfe`, not the
current PR head. The following later commits must be considered before using
its numbers:

| Commit | Effect on matching, publication or capacity |
| --- | --- |
| `b7ec2b04809395faebabd657d0af6e1982b6a53b` | Narrows the feature to native Replay by removing the Python YAML/compiler/runner integration, scorer and associated tests. The native checkpoint selection and publication implementation is retained; this commit supplies no new measurement of it. |
| `cab61df90b78c9855230718371d22e9a46ffd642` | Rounds a shared branch down to the configured prefill checkpoint interval. Counts private state copies retained by waiting requests once across retries, transfers them into active allocation, and releases them on cancellation. Preserves pending checkpoint endpoints until materialized and defers host-loop publication to its commit event. Capacity includes three slots per active request, waiting copies and resident checkpoints, with explicit errors on unsupported state/KV pressure. These changes can alter reused prefixes, prefill geometry, admission and TTFT. |
| `6eb3b8665adda928778815c720085eb6d85f34e5` | Skips speculative key/value vector reservations on checkpoint leaves, whose endpoints must remain fixed and whose extensions create child nodes. This avoids unused host allocation; it is intended to preserve checkpoint selection and simulated timing, but no capture rerun verifies that expectation. |
| `958e3bbc13fcec7f5e12408594f121e11b3291e6` | Rejects Mamba attention DP greater than one through `EngineFactory::build_rank`, matching the existing full-factory rejection. Unsupported multi-DP executions now fail at construction; the historical DP1 capture is within the retained scope. |

**Whether the historical metrics still hold after these commits is unknown.**
Synthetic Rust tests cover checkpoint geometry, deferred publication,
retry/cancellation accounting, capacity boundaries and unsupported configuration
rejection. They do not replace a baseline/current-candidate comparison or
held-out measured accuracy. The six hash-matching inputs have no supplied
retrievable location; the capture owner must provide them before that comparison
and its parity controls can be rerun. An independent capture or output oracle
is also needed to qualify predictive accuracy beyond the training capture.

## Local review verification (2026-11-18)

The corrective change adds adapter test cases and evidence clarification.
Rust sources match starting head
`958e3bbc13fcec7f5e12408594f121e11b3291e6`. Local checks use Python 3.12.3,
Ruff 0.14.1, and the optimized Rust release profile. These checks do not rerun
the historical capture.

The local native command, using rustc/cargo 1.91.1, completed successfully:

```text
cargo test --release --locked -p aisim-core --lib mamba -- --nocapture
test result: ok. 47 passed; 0 failed; 0 ignored; 0 measured; 3237 filtered out
```

This includes the PR's added cases in `engine/cache/radix_cache.rs`,
`engine/config.rs`, `engine/kv_manager/sglang_backend.rs`, `engine/runtime.rs`,
`replay/engine.rs`, and `engine/scheduler/sglang/tests.rs::mamba_tests` (paths
relative to `crates/core/src/`). Negative/boundary cases pass for invalid
intervals/chunk alignment, unknown/missing fields, unsupported backend/worker
role/speculation/offload, DP2 through both factory entry points, and Replay
DP/CP/PP/Belady rejection. The waiting-copy capacity case rejects six required
states against `max_states=5` without publishing another checkpoint. Both
host-loop modes return a state-pressure error instead of an idle pass; KV
eviction pressure also errors explicitly. Retry/cancellation accounting,
deferred publication, short final chunks and retained checkpoint endpoints pass.

The following local CI-contract command passed **803 tests**, including
`test_simulation_dynamo_sglang_adapter_against_current_api[omitted]`,
`[explicit-none]` and `[custom]`. The cases check insertion of `mamba_cache: None`
only when absent, preservation of explicit values, absence of duplicate fields
and the adapter's inserted-field manifest against the current Rust API.

```bash
python -m pytest -c /dev/null \
  tests/test_ci_workflow_contracts.py tests/test_full_ci_fail_fast.py \
  tests/test_artifactory_wheel_handoff.py tests/test_ci_qualification.py \
  tests/test_release_fpe.py -q -p no:cacheprovider \
  --override-ini=addopts= --override-ini=filterwarnings=error
```

The local ownership suite passed **107 tests**; strict whole-tree coverage and
generated `CODEOWNERS`/`CONTRIBUTORS.md` comparisons passed. Repository Ruff,
Rust formatting, Python compilation, copyright, packaged legal files and
documentation-destination checks passed. The Python environment uses
`scripts/ci/requirements.txt`, with its virtualenv `bin` on `PATH` and a
process-local `commit.gpgsign=false` override for synthetic Git fixture commits.

Hosted evidence inspected for `958e3bbc`:
[Fast CI run 44385756262](https://github.com/ai-dynamo/aisim/actions/runs/44385756262)
passed, including Repository Policy, Python Static Checks and Rust Format.
The same head's codeowners, DCO and Pages build checks passed. The workflow API
returned no Full CI run for that SHA. Hosted checks and fresh bot reviews for
the corrective commit remain pending controller publication/admission.
