# Fixed-length DSpark paired results — 2026-10-15

Five configurations, 13 production runs and 1,664 measured requests are paired with native Engine Replay. All requests completed successfully. Eleven new P/D runs add 1,408 requests to the two previous pairs. Every scored point uses DeepSeek-V4-Pro-0813, fixed K=6, ISL/OSL=1024/1024, vLLM, B200, adaptive verification off, prefix caching off and external offload off.

## MAPE by configuration

Each value is the unweighted mean of per-run APE. Repetitions are separate observations; these are not request-level errors or errors of averaged latencies.

| Deployment | Acceptance | Concurrency | Runs / requests | TTFT MAPE | TPOT MAPE | Throughput MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Aggregated, 8 B200 | synthetic | 1 | 1 / 128 | 60.39% | 10.77% | 6.47% |
| P/D, 8P+8D B200 | synthetic | 1 | 3 / 384 | 57.60% | 10.05% | 4.11% |
| P/D, 8P+8D B200 | synthetic | 4 | 3 / 384 | 59.91% | 42.53% | 25.22% |
| P/D, 8P+8D B200 | real | 1 | 3 / 384 | 57.47% | 8.36% | 0.37% |
| P/D, 8P+8D B200 | real | 4 | 3 / 384 | 59.76% | 44.84% | 22.58% |

Across all 13 runs: TTFT MAPE **58.81%**, TPOT **25.24%**, throughput **12.56%**. Signed biases are **-58.81%**, **+25.24%**, and **-12.39%**, respectively. This combined average is dominated by 12 P/D runs and does not represent model-wide accuracy.

## Measured versus predicted means

These means summarize repetitions; MAPE above is calculated before averaging repetitions. Sample standard deviations for each configuration and metric are retained in `report.json` → `variability`.

| Deployment / acceptance / concurrency | TTFT measured / Replay (ms) | TPOT measured / Replay (ms) | Throughput measured / Replay (tokens/s) |
| --- | ---: | ---: | ---: |
| agg / synthetic / C1 | 193.509 / 76.642 | 3.300 / 3.656 | 286.840 / 268.293 |
| disagg / synthetic / C1 | 316.326 / 134.125 | 3.322 / 3.656 | 275.633 / 264.312 |
| disagg / synthetic / C4 | 388.341 / 155.690 | 4.193 / 5.977 | 871.790 / 651.901 |
| disagg / real / C1 | 315.329 / 134.125 | 2.019 / 2.188 | 430.154 / 431.739 |
| disagg / real / C4 | 398.855 / 160.516 | 2.419 / 3.504 | 1408.340 / 1090.310 |

## Interpretation and limits

- TTFT is underestimated at every scored point. TPOT is overestimated at every point, with the largest gap at C4: 42.53% for synthetic and 44.84% for real acceptance. These errors remain in the report; no numerical accuracy threshold is imposed.
- P/D C1 real throughput agreement (0.37% MAPE) reflects offsetting TTFT and TPOT errors. It does not establish accurate component timing.
- P/D uses P TP1/attention-DP8/EP8 and D TP8/EP1. Transfer latency is unmodeled; historical vLLM0.24 op tables, attention proxies, uneven prefill rank work and frontend/runtime overhead remain unresolved contributors. No individual cause has been quantitatively isolated.
- Real acceptance uses D-only measured-window conditional rates; synthetic uses the pre-frozen 2.77 accepted drafts plus one bonus, without fitting to latency. Native P/D padded startup contributes a zero-accepted round to measured counters, while Replay handles startup separately. The real rates therefore retain a startup-conditioning approximation; the scores include acceptance assumptions as well as cost-model error.
- The nine native peak-observer receipts in each new job remained startup-only after forced cleanup. They are preserved as incomplete. Qualification instead uses exact-zero drained physical-pool gauges tied to current completion counters and a source-audited frontend update order. Each P pool admits 16 measured requests; retaining every block bounds usage at 6,993 < 7,640 blocks. C4 uses its own Replay and D capacity checks.
- Absolute and per-GPU throughput are retained in `paired.csv`. Aggregated and P/D runs have different GPU counts; their comparison is not an equal-resource speedup.
- The full matrix has 196 configurations and 588 planned runs. Five configurations and 13 runs are scored; 575 runs remain without measured pairs. K3 acceptance pilot, all smokes, failed V4.1/Kimi op collections, AgentX sessions and FPM are outside this MAPE.

## Reproduction and evidence

- [Full paired rows](paired.csv), [machine-readable report](report.json), and [measured-versus-Replay plot](measured-vs-replay.png).
- `reproduction-command.json` contains the exact argument vector for the existing `compare.py`; run from the repository root after restoring archived raw records.
- [New-window qualification index](../qualification-20261015/index.json) and [physical-pool source proof](../pd-drained-pool-boundary-source-audit.json).
- [New predictions](../pro-pd-followup-11/README.md) retain per-window configurations, matching receipts, acceptance evidence and full native reports. Predictor revision is `19d7aa53c474b56d2e100889fac5a86364313a0d`; native SHA256 is `3af8fb12cff28b87249648ab15e26ed18551bd1e17f334318487ea448ff0be40`. The old P/D pair reproduces bit-for-bit on this build.
- [Collection ledger](../collection-status.json) retains failures, exclusions and archive paths. No new GPU runs were needed for this pairing.
- [Independent review](../pro-pd-followup-independent-review.json) recomputes all 39 scored metric rows, per-cell variability, and the new Replay metrics from archived per-request reports.
