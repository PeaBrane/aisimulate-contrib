# Fixed-length DSpark serving validation

This campaign implements the approved AgentX-model matrix and MTP-aligned
V4-Pro P/D topology. **No qualified serving measurement/prediction pairs are
available yet. MAPE and signed bias are unavailable, not zero.** Functional
tests and op-table query coverage are separate from serving accuracy.

## Matrix and references

All fresh runs use vLLM, fixed draft length, adaptive verification disabled,
external KV offload disabled, and the checkpoint/draft pins in
[`checkpoint-references.json`](checkpoint-references.json). FPM is deferred.

| Model | Deployment | GPUs and topology | K | Reference and matching boundary |
| --- | --- | --- | --- | --- |
| DeepSeek-V4-Pro-0813 | Aggregated | 8 B200, TP8/EP1 | 3, 6 | AgentX [442741 / 35434571418 attempt 2](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35434571418/attempts/2), K6, AL3.77; same GPU/framework/topology. Fresh runtime is separately pinned. |
| DeepSeek-V4.1-Flash | Aggregated | 4 B200, TP4/EP1 | 3, 5 | AgentX [444090 / 37070984585 attempt 1](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37070984585/attempts/1), K5, AL3.51. Fresh collection disables the reference's Engram CPU weight offload. |
| Kimi-K3 + Inferact/Kimi-K3-DSpark | Aggregated | 8 B300, TP8/EP1 | 2, 3 | AgentX [439539 / 31509160691 attempt 2](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31509160691/attempts/2), K2, AL2.51; same GPU/framework/topology, replacement runtime. |
| DeepSeek-V4-Pro-0813 | P/D | 8P + 8D B200; P TP1/attention-DP8/EP8, D TP8/EP1 | 3, 6 | MTP [438808/438809/438807 / 30937733020 attempt 1](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/30937733020/attempts/1), 8192/1024, concurrency1/8/16. Topology/workload anchors only; older Pro checkpoint and MTP latency are not DSpark truth. |
| DeepSeek-V4.1-Flash | P/D | 4P + 4D B200; TP4/EP1 per worker | 3, 5 | New configuration. AgentX [444313 / 37260940238 attempt 1](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/37260940238/attempts/1) is SGLang/GB300/1P2D and reference-only. |
| Kimi-K3 + Inferact/Kimi-K3-DSpark | P/D | 8P + 8D B300; TP8/EP1 per worker | 2, 3 | New configuration. AgentX [442093 / 34873998796 attempt 1](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34873998796/attempts/1) uses GB300/K4/offload and is reference-only. |

The generated [`matrix.json`](matrix.json) has 196 configurations, 588 runs and
153,600 measured requests, excluding warmup and pilots. The initial gate has
48 configurations: 1024/1024, concurrency 1/4, both K values and acceptance modes.
Expansion adds 8192/1024, concurrency 16/64, and the Pro P/D concurrency 8 anchors.
Three repetitions measure `max(128, 10 × concurrency)` requests each.

## Execution and evidence

- [`collection-status.json`](collection-status.json) records actual jobs, failures,
  cancellation/retry reasons and result paths. It is the resumable execution
  ledger; a queued job or imported image is not a measurement.
- [`COLLECTION.md`](COLLECTION.md) describes exact token-ID workloads, drained
  warmup, closed-loop concurrency, acceptance counters, runtime qualification,
  Slurm entrypoints and NIXL source contracts. Both P/D roles share an allocation.
  At most one GPU collection job runs per platform.
- [`model-support.json`](model-support.json) records functional model coverage
  and remaining cache/operator approximations.
- [`op-coverage.json`](op-coverage.json) records canonical `op_level` + `SILICON`
  + `fallback_policy: deny` construction/query checks. These CPU queries are
  predictions, not collected kernel or serving measurements.
- [`missing-op-plan.json`](missing-op-plan.json) identifies blocked V4.1 and
  Kimi table contracts and the minimum source-qualified collection work.
- [`COMPARISON.md`](COMPARISON.md) defines immutable match receipts, exact-token
  Replay, metric definitions, acceptance evidence and scoring commands.
- [`comparison/report.json`](comparison/report.json) and
  [`comparison/paired.csv`](comparison/paired.csv) retain every planned point,
  including missing/failed/unqualified cells. The plot shows only qualified pairs.

The initial 256 GB host-memory smoke attempts did not produce requests. Their
slow weight-loading observations and subsequent resource changes remain in the
ledger. Fresh collection uses eager execution. Model/GPU identities are retained;
resource retries are not treated as completed coverage.

## Matching and progression

Serving identity is frozen before each measurement. Exact token IDs, runtime
and checkpoint identities, role/rank topology, graph/cache policy, counters,
raw records and content hashes accompany every window. Real acceptance is
extracted after that window; synthetic controls are frozen beforehand from a
qualified producer or separate 128-request pilot. Neither is fitted to latency.

The scorer reports request TTFT/TPOT, absolute and per-GPU output throughput,
completions, failures, per-point APE, MAPE, signed bias and repetition variability.
It separates model/deployment/K/acceptance strata. Kernel timing, streaming chunk
gaps and simulator host cost are different quantities. Different GPU counts are
not equal-resource speedups.

Each model/deployment advances to AgentX only after fixed-length functional
checks and paired results exist; there is no numerical accuracy threshold.
AgentX then enables matched prefix caching and the pinned corpus/producer,
using `live_trajectories` warmup. It drains HTTP requests but retains live DAG
frontiers and cache state. Historical live streams were not unfinished HTTP
requests; see the corrected [earlier audit](../dspark-fixed-20261014/README.md).
Planned AgentX pilots are 10 minutes at session concurrency 1/4, followed by
one-hour profiles at 1/4/16 with three repetitions after lifecycle/cache checks.

Full fixed-length collection, P/D qualification and AgentX profiling remain
unfinished until their actual records and paired reports are present.
