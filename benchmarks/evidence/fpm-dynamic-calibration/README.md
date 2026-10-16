<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamic calibration with the PR 460 forest

The production dynamic-local calibrator lowers aggregate B+C MAPE from **3.543082% to 3.277720%** (7.49% relative) on 36 paired captures. It admits valid feedback outside the former fixed grid and uses recent global feedback for queries beyond the currently observed range. The default learned model selects this policy; `correction_mode="fixed"` preserves the previous policy. Native AIC keeps its fixed default.

## Evaluation contract

- Frozen PR 460 recipe at `089b1d54936dfe5e400f60e7af0427a917c8c042`: 17-feature `fpm5`, 150 trees, maximum 15 leaves, learning rate 0.2, minimum leaf size 5, minimum store size 20, `random_state=0`, no early stopping, log-ms targets. All methods use identical saved forests.
- 36 captures: prefill and decode from 17 runs across AgentX, LongBench, and ShareGPT on vLLM/SGLang, plus two aggregated gym captures. Each chronological prefix contains `floor(N/2)` observations; the suffix is evaluated. There are 6,230,619 eligible observations and 3,115,316 suffix predictions.
- Predict before admitting the target. One clock per capture admits the first observation reaching each new five-second tick; timestamp gaps do not generate catch-up labels. Model state is separate per worker/workload scope. A missing prefix scope fits only after 20 emitted observations; correction buffers begin empty.
- MAPE first averages captures within workload/backend/dataset groups, then equally weights workloads, backends, and dataset families. It is not event-pooled. Both policies have identical 100% coverage in this run.
- Production starts with empty correction bounds because v1 forest artifacts contain no training-coordinate bounds. A separate research-reference arm initializes bounds from the prefix. Both expand only after feedback and begin with empty correction buffers. No model artifact schema change is needed.

## Accuracy

| Method | Hierarchical MAPE |
|---|---:|
| Uncorrected forest | 4.913437% |
| Fixed calibration | 3.543082% |
| Production dynamic local | 3.277720% |
| Independent Python, empty bounds | 3.277720% |
| Research reference, prefix bounds | 3.268278% |

Native predictions match the independent Python empty-bound reference on **all 3,115,316 ready events**, within `1e-9 ms + 1e-9 * abs(reference)`. Independent audits passed for all 36 cases, checking event membership, admission clocks, readiness, model/input/source hashes, and recomputed errors.

| Dataset family | Fixed MAPE | Dynamic MAPE |
|---|---:|---:|
| agentx | 4.529779% | 3.233803% |
| gym | 4.148448% | 4.423645% |
| longbench | 1.573282% | 1.605456% |
| sharegpt | 3.920821% | 3.847975% |

| Query cohort | Fixed MAPE | Dynamic MAPE |
|---|---:|---:|
| Inside fixed grid | 3.475539% | 3.587687% |
| Outside fixed grid | 7.128468% | 5.202244% |

254,953 suffix events (8.18%) are outside the fixed grid. Cohort scores independently apply the same hierarchical weighting to their available groups, so they do not combine into the overall score by event-count weighting. Fixed limits are inclusive: prefill tokens ≤8,192; decode requests ≤512 and decode KV tokens ≤2,000,000; mixed workloads use the prefill-token and KV limits.

**Tradeoffs:** 24 of 36 capture-level MAPEs improve and 12 worsen. The largest regression is a ShareGPT/SGLang decode capture: 1.563055% → 3.090794% (+1.527739 percentage points). Four AgentX/SGLang decode captures improve by 1.72–7.03 percentage points. In-grid aggregate accuracy is slightly worse, and the gym and LongBench family averages regress. This supports the boundary fix and an aggregate improvement, not a universal accuracy win. The prefix-initialized research reference is 0.009442 percentage points better than the production version.

## Host cost

Seven trials alternate fixed/dynamic order in one process after the accuracy campaign. Values below are the median of per-trial mean call times, in microseconds, on an Intel Core Ultra 9 285K. Each native call includes Rust JSON parsing, feature extraction, forest evaluation, and calibration; Python serialization and process startup are excluded. Construction and first-use calls are timed separately. These are focused API timings, not deployment latency or backlog guarantees.

| Capture/store | Fixed query | Dynamic query | Fixed update | Dynamic update |
|---|---:|---:|---:|---:|
| Kimi, pure_decode | 2.874 | 2.678 | 2.053 | 3.695 |
| Kimi, pure_prefill | 3.460 | 3.182 | 2.213 | 3.852 |
| Kimi, contains_locally_mixed | 3.080 | 3.038 | 2.875 | 3.444 |
| AgentX/SGLang, pure_prefill | 3.214 | 3.244 | 2.639 | 4.699 |
| AgentX/SGLang, pure_decode | 2.443 | 2.320 | 2.133 | 3.384 |

Dynamic update medians range from 3.38 to 4.70 μs. Its extra update cost is 0.57–2.06 μs in these workloads. The calibrator rebuilds cached factors only on admission; prediction does not scan observations, allocate a bucket key, or mutate bounds. Query timing changes are small and workload-dependent.

## Evidence and limits

- Input-manifest SHA-256: `2ebf46b847b50708a1f44587639724732b3afdb8041b2d3025476c9e36fcb749`.
- Evaluated native-library SHA-256: `8836c30c005ddb971705cff9e8b5728a60b1c10cfdba050b906766f41f16144e`.
- Dynamic-calibration source SHA-256: `94e0a47e326f8a6a7c151b18c99467dc47e25ba84a0f236d90b5683c73bbd30f`.
- The two gym captures use `nvidia/aisimulate-fpm-dataset` revision `68fa3add95b32a0399d781b043cb0f1008c8040d`. The remaining 34 captures are access-controlled archived inputs; raw traces are not included here. The full table requires that prepared input/seed bundle, so it is not a public one-command benchmark.
- These captures were previously examined during calibration research. This is a paired regression comparison, not a fresh parameter-selection holdout. No new tuning sweep was performed. C-only, A+C, and other train/test ratios were not requalified here.
- Fixed calibration retains existing `HashMap` tie behavior for equally full buckets. Baseline eviction choices can vary between fresh runs; the per-case result hashes in the evidence file identify this run. Dynamic retention is deterministic.
- [Compact per-case evidence](results.json) records paired errors, cohort counts, artifact identities, and timing summaries without raw traces.
- Validation: 202 native FPM tests, 91 focused Python SDK tests, six experiment-harness tests, and 22 portable-comparison tests passed. The portable CLI also passed native-library smoke tests with valid and mixed-null rank labels. Both required frozen-golden suites passed against a fresh local native build and current profiles: 305 engine-step and 85 compile-engine cases, with zero skips or failures and no golden changes. Formatting and targeted Python lint passed. Strict Clippy is blocked by 122 diagnostics on unchanged code; none point to added or modified lines.

For a portable check on your own frozen forest and held-out chronological observations, use `python -m scripts.fpm_accuracy.compare_calibration --help`. This utility compares fixed and dynamic calibration without retraining; it is separate from reproducing the archived 36-case campaign.
