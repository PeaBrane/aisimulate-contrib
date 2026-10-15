# Fixed-length DSpark serving validation

This campaign implements the approved AgentX-model matrix and MTP-aligned
V4-Pro P/D topology. **Two qualified serving pairs are available: two configurations,
one repetition each and 256 measured requests.** Their TTFT/TPOT/throughput MAPE is
58.90%/9.58%/3.35%. The remaining 586 planned runs have no scored error.
These two points use different deployments and acceptance modes; see the
stratified report before interpreting the combined average. Functional tests and op-table query
coverage are separate from serving accuracy.

The [first production pair](pro-initial-c1-r1/README.md) uses Pro0813 on 8 B200,
TP8/EP1, K6, synthetic AL3.77, 1024/1024 and concurrency1. All 128 requests
completed, with zero failures. Measurements, frozen matching evidence, original
and qualified summaries, source patches and raw records are retained.

The [first production P/D pair](pro-pd-initial-c1-r1/README.md) uses the same
Pro0813 checkpoint on 8P+8D B200, P TP1/attention-DP8/EP8 and D TP8/EP1,
K6 with real acceptance, 1024/1024 and concurrency1. All 128 requests completed
after warming every P engine. TTFT/TPOT/throughput APE is 57.41%/8.39%/0.24%.
The small throughput error reflects cancellation between underestimated TTFT
and overestimated TPOT; it does not establish accurate component timing.
Transfer latency remains explicitly unmodeled. Different GPU counts and
acceptance modes prevent an equal-resource aggregated-versus-P/D speedup claim.

Two aggregated single-request Pro0813 diagnostics are available:
[`eager with real acceptance`](smoke-pro-eager/README.md) and
[`graph with synthetic acceptance`](smoke-pro-graph/README.md). The graph smoke
has measured/predicted TTFT 195.70/76.64 ms and TPOT 3.34/3.62 ms. Both are
excluded from production MAPE. Their acceptance settings differ, so they do not
establish an eager-versus-graph speedup.

The first Pro P/D diagnostic (job2257933) also completed: 8P+8D B200, K6,
real acceptance, 1024/1024, one drained warmup and one measured request.
Measured TTFT is 631.20 ms, TPOT 1.87 ms and throughput 402.25 tokens/s.
Both requests returned exactly 1024 client-visible output tokens. This is a
functional smoke, excluded from production MAPE; the measured request used a
different P engine from warmup, so cold-peer setup may affect its latency.
The [paired diagnostic](smoke-pro-pd/README.md) predicts TTFT 134.13 ms and
TPOT 1.98 ms, with transfer latency explicitly unmodeled. All 16 rank cache
receipts, requests, metrics and logs are preserved in
`pd-smoke-2257933.tar.gz` (SHA256
`7e055103aa13718e4fb837e64c5eee2d654a37a7d39ba4ac405acf6080a7c028`).
Extract it at this directory to restore `raw/pd-2257933/`.

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
- [`pd-installed-source-audit.json`](pd-installed-source-audit.json) compares
  actual-image source hashes with the nominal runtime revision. The installed
  draft graph moves context preparation into capture; source-level ownership
  and transport compatibility remain separate from measured P/D qualification.
- [`verification.json`](verification.json) records schema31 functional checks:
  2,000 Rust library tests plus integration suites, 13 public API tests, 145
  affected Python tests, and 107 ownership checks. These are not serving samples.
  The production checkpoint adds 170 passing campaign/import tests and six cache
  subtests, including the Kimi legacy-import regression found by Full CI.
  The AgentX/P-D checkpoint adds 198 loadgen checks, 491 Runner API checks,
  44 campaign checks and 15 cache checks with six subtests.
  The P/D production checkpoint passes 2,017 Rust library tests (one ignored),
  134 integration tests, 13 public API tests and three doctests; 505 Runner/import
  tests, 82 campaign/scoring tests with 25 subtests, 31 compatibility/CLI checks
  and 94 model contract checks. Unknown AgentX output IDs retain cache occupancy
  without inventing real-token prefix hits; legacy synthetic output hashes stay
  unchanged.
- [`op-coverage-schema31.json`](op-coverage-schema31.json) records current canonical `op_level` + `SILICON`
  + `fallback_policy: deny` construction/query checks. These CPU queries are
  predictions, not collected kernel or serving measurements.
  The earlier `op-coverage.json` remains historical evidence. Current V4.1/Kimi
  construction rejects missing native-stage/MLA tables instead of reusing
  incompatible measurements.
- [`missing-op-plan.json`](missing-op-plan.json) identifies blocked V4.1 and
  Kimi table contracts and the minimum source-qualified collection work.
- [`COMPARISON.md`](COMPARISON.md) defines immutable match receipts, exact-token
  Replay, metric definitions, acceptance evidence and scoring commands.
- [`comparison/report.json`](comparison/report.json) and
  [`comparison/paired.csv`](comparison/paired.csv) retain every planned point,
  including missing/failed/unqualified cells. The plot shows only qualified pairs.

The initial 256 GB host-memory smoke attempts did not produce requests. Their
slow weight-loading observations and subsequent resource changes remain in the
ledger. Pro K=6 eager and graph smokes each completed one 1024/1024 request; both are
diagnostic evidence, outside production coverage. The graph smoke exported all
eight ranks' cache layouts. Production uses the pinned
runtime's default graph policy after a separate graph smoke, with a fresh cache
receipt and identity. Model/GPU identities are retained; resource retries are
not treated as completed coverage.

Kimi job790797 completed eager weight loading but failed before requests: the
automatic MLA prefill backend could not quantize prefill queries for FP8 KV.
The query-quantization flag was present; the backend also needs an explicit
compatible selection. The failed launch and logs are retained in
`kimi-prefill-failed-790797.tar.gz` (SHA256
`95e17271eb1d6f60eadb81e359e2e8fd55b635f001665da4a23eed8d5e95bf6f`).
It contributes zero measured requests and no accuracy result. The next serving
and collector configurations explicitly pin `TRTLLM_RAGGED` prefill and
`FLASHINFER_MLA` target/draft attention, with source/API checks before GPU retry.

`reconcile_pro_cache.py` checks the Pro TP8/PP1 cache receipt against the pinned
runtime's shared-pool contract. Tensor descriptors overlay one allocation;
their sizes are not summed. Replay preserves the physical pool block count,
reports byte rounding, and uses an approximate steady-state reservation for
sliding groups. A separate conservative capacity check retains the entire
input/output history in every group, two unsettled K+1 verify rounds, K draft
lookahead slots, alignment slack, and the shared null block. This bound follows
vLLM `ac9126e58aa7bbab1856ba6593ba4d5003fea516`:
`vllm/v1/kv_cache_interface.py:1416` (aliasing),
`vllm/config/vllm.py:592` and `:630` (PP1 in-flight batches/lookahead), and
`vllm/v1/core/sched/scheduler.py:637` (unclipped final speculative rounds).
The mapping qualifies only the pinned prefix-disabled aggregated workload;
it does not reproduce native sliding-block lifetimes or P/D transfers.

The separate Pro P/D adapter preserves eight independent prefill pools and one
TP8 decode pool. It reserves rolling groups conservatively and records the
target-plus-draft transfer payload bound separately from physical network bytes.
Source caches may remain allocated while NIXL completion notifications arrive;
concurrency alone does not prove their native occupancy bound. P/D qualification
therefore also needs source-residency evidence. Transfer bandwidth is unmodeled
unless an explicit evidence-backed value is supplied.

Canonical AgentX graph input now accepts exact `input_token_ids` as an alternative
to synthetic `hash_ids`. Exact prompts retain native prefix identity through
snapshot preparation, warmup and recycling. A graph cannot mix the two prompt
representations because synthetic tokens could otherwise collide with real IDs.
Producer capture retains selected trajectories and warmup boundaries; importing
an external producer frontier remains separate work before AgentX scoring.

Host regression comparisons use the frozen performance-gate trace and verify
its hash and recursively counted requests against its manifest before timing.
The current fixture has 255 requests and 286,917 output tokens. The earlier
129-request fixture's totals cannot validate this workload. The interrupted
attempt in `host/failed-attempt.log` retains that mismatch; it is not a completed
host comparison. Host wall time is separate from serving accuracy.

The completed five-repetition incremental comparison is in
[`host-regression.json`](host-regression.json), with every original CLI artifact
in `host-matched-raw.tar.gz` and the interrupted attempt in `host-raw.tar.gz`.
Against `e4cdd954`, median host time changed by -0.77% for the non-speculative
control, +1.40% for AgentX, and +0.48% for AgentX with DSpark. Both builds used
release binaries, identical dependencies, one CPU thread, and the same workload;
one warmup per case/build was excluded. This measures the incremental schema30
implementation, not the total DSpark feature overhead or serving accuracy.

The later output-identity check compares clean `b1c56e9c` and `5bf8f6e8` in
[`host-output-identity.json`](host-output-identity.json), with full samples and
raw CLI artifacts in its two hashed archives. Five alternating repetitions
preserve request/token totals, prefix metrics and lifecycle digests. Full CLI
median changes are +3.31% control, +0.37% AgentX and +1.21% AgentX DSpark;
native Replay changes are -2.72%, +0.46% and -0.32%. Sample ranges overlap.
An original exact-token CPU stress fixture adds 256 requests and 32,768 output
tokens per run, exercising unknown output identities. Its native changes are
+1.49% without speculation and -1.89% with DSpark, also within overlapping
ranges. This fixture is not a real AgentX measurement. No substantial native
regression is established; CLI startup/preparation variability is retained.

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

Phase B preparation is in [`agentx-manifest.json`](agentx-manifest.json) and the
[AgentX entrypoint instructions](COLLECTION.md#phase-b-agentx-entrypoint).
The pinned producer's exact pilot/profile commands passed CPU parsing. No
AgentX serving run has been submitted by these tools. The complete pinned
corpus passed content verification; [`agentx-context-audit.json`](agentx-context-audit.json)
records 98,827 requests and a largest source input/output sum of 996,579 tokens.
The current 16K fixed-run identity cannot serve 91,944 of those requests.
An explicit 1M context/prefix-disabled fixed anchor and native formatter
preflight are required before execution. Pro preflight is available in
[`agentx-context-preflight-pro.json`](agentx-context-preflight-pro.json); the
live server validates sampled requests, and all failures remain evidence. Final snapshot/frontier and
metric parity remain separate gates before any AgentX accuracy claim.
