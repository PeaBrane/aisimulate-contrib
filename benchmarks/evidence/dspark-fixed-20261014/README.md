# Fixed-length DSpark: implementation and evidence

Base revision: `cb71f236a7a4d658f6be90ada3c37241d209fadb`.
Phase 1 implements op-level costs and Replay. Phase 2 is a placeholder;
explicit DSpark FPM requests are rejected. No FPM accuracy is claimed.

## Implemented contract

- Canonical Rust/Python `best_available` accepts typed `Dspark` identity.
  Resolved draft configuration and revision survive serialization and caches.
- Fixed K drafts, K+1 verification queries, variable accepted drafts plus one
  bonus. Adaptive verification and simultaneous `nextn` are rejected.
- V4 context projection/KV preparation, parallel draft backbone, logits,
  Markov sampling and target verification use the existing op graph.
  A typed sequence cap carries the draft window through compiled execution.
- Draft weights/KV participate in automatic capacity. Shared target weights
  are counted once. Accepted tokens determine progress, not current-round cost.
- Aggregated and P/D vLLM/SGLang Replay accepts DSpark, including AgentX.
  Recommendations preserve the configuration in saved prediction YAML.
- Automatic P/D transfer includes prepared draft KV once, including when the
  scalar target cache reports a destination prefix hit. Explicit byte overrides
  remain caller-owned totals.

See the existing [speculation guide](../../../docs/replay/engine/speculation.md),
[API contract](../../../docs/perf-model/configuration.md), and
[transfer contract](../../../docs/replay/engine/kv-transfer.md).

## Measurement eligibility

| Priority | Combination | Evidence | Status |
| --- | --- | --- | --- |
| P0 | V4-Pro-0813 / vLLM / 8 B200, TP8 EP1 | AgentX IDs 442741, 442744, 442734, 442737, 442740, 442739; concurrency 1/4/6/10/14/16 | Reference only. K=6 and synthetic AL=3.77 observed; exact adaptive-off execution and workload identity remain unresolved. |
| P1 | V4-Pro-0813 / SGLang / 8 B200, TP8 EP1 | AgentX IDs 442271, 442275, 442272, 442273, 442276; concurrency 1–5 | Reference only. Draft width 6 observed; acceptance, ragged verification and workload reconciliation remain unresolved. |
| Diagnostic | V4-Flash-0731 / vLLM / 4 B200, TP4 EP4 | Playground K=3/5/7 static waves | Available, but `round_ms_*` measures SSE chunk gaps. No independent draft/verify kernel timing; not scored as per-token or operation latency. |
| Existing P/D | V4-Pro-0813 / Dynamo+vLLM / GB200 | Eight offload-disabled AgentX candidates, including 444916 | Preferred before recollection. Exact attempt-3 artifacts recovered for 444916; runtime defaults and workload/transfer contracts still prevent matched scoring. See below. |
| P/D | V4-Flash-0731 / vLLM / 8 B200, 1P1D TP4 EP4 | Controlled collection manifest and driver | Runtime/checkpoint inspected. GPU collection blocked on available 8-GPU capacity. |

`references.json` records API metadata, artifact IDs, archive/log hashes and
observed launch fields. The source runs are
[vLLM attempt 2](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35434571418/attempts/2)
and [SGLang attempt 1](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34603861241/attempts/1).
Sibling results include offload variants; only the eleven offload-disabled
IDs above belong to the initial matrix.

Eligible paired measured points: **0**. TTFT/TPOT/throughput error, MAPE and
signed bias are **unavailable**, not zero. No numerical pass is declared.
Do not infer fixed DSpark from the public `draft_model` or `mtp` labels.
Recollect a combination if fixed verification or exact corpus identity cannot
be established. Kimi-K3 remains deferred.

## Existing AgentX P/D follow-up

Existing measurements take priority over the controlled 8-B200 collection.
`agentx-pd-audit.json` freezes eight offload-disabled V4-Pro/GB200/Dynamo+vLLM
candidates from the [same workflow](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36425419107/attempts/3).
The launch evidence covers both serving roles, including each node's command.
Only 444916 has been reconciled to the exact attempt's raw request artifacts;
the other seven remain candidates, not matched validation points.

| Benchmark IDs | Session concurrency | Prefill + decode GPUs | Observed K / synthetic AL |
| --- | --- | --- | --- |
| 444916, 444907, 444914, 444909 | 32, 64, 128, 256 | 8 + 8 | 5 / 3.61 |
| 444912 | 384 | 8 + 8 | 6 / 3.77 |
| 444911 | 360 | 8 + 12 | 6 / 3.77 |
| 444915 | 384 | 12 + 12 | 5 / 3.61 |
| 444908 | 512 | 12 + 16 | 6 / 3.77 |

These are total role GPU counts with attention DP and EP. For 444916 the actual
runtime is **TP1 / attention-DP8 / EP8** on each role, across two 4-GPU nodes per
role. Dashboard `tp=8` with `dp_attention=true` must not become attention TP8.
Current canonical Replay accepts this DSpark topology: a fixed-timing P/D smoke
completed one request and all 19 output tokens. This is functional evidence only.

### Matched artifact identity for 444916

- Workflow revision: `f4ad60fba73439cf4833a674e6a84feb2014bfaf`; attempt 3,
  Slurm job **26842**, September 30, 2026. The API's publication date is not the
  execution date.
- Artifact IDs: metrics **11092304949**, AgentX **11092647256**, server logs
  **11092552440**. The script verifies all three SHA-256 digests before parsing.
- Both role fingerprints report vLLM `0.17.2rc1.dev6312+g20ebb2850` and Dynamo
  `1.3.0.dev20260720`. Image tag:
  `vllm/vllm-openai:nightly-dev-arm64-cu13.0.1-20ebb28`; an immutable image digest
  is not recovered.
- Both fingerprints confirm target/bundled-draft revision
  `72e1d3230f6c080a530b0a1d46f8eb4602340597`. Its
  [model config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/72e1d3230f6c080a530b0a1d46f8eb4602340597/config.json)
  supplies DSpark block size 5, target layers 58/59/60 and Markov rank 512.
  No separate draft checkpoint is launched.
- Both roles use K=5, probabilistic draft sampling and synthetic AL=3.61. The
  launch/config dumps omit `enable_adaptive_verification`. The exact public
  vLLM revision returns 404 for source and 422 for commit lookup. Defaults from
  another revision cannot qualify adaptive-off or the AL bonus convention.
- Decode reserves 35 GiB KV per rank and reports 2,964,374 token-equivalent
  capacity per rank. This grouped-cache capacity cannot simply be copied into
  the simulator's scalar block model without accounting for draft/cache layout.
- Dataset: `semianalysisai/cc-traces-weka-062126`, revision
  `23f152f6f0f9399a85901b89a6458def0ef16729`, all 393 eligible root traces.
  Snapshot ratios 0.25–0.75, seed 42, 32 closed-loop lanes, 3600-second profile,
  300-second idle cap, and 10 warmup requests per lane are recorded.

The public server-log API for 444916 returned **attempt 1 / job 26725** on
September 28 (nodes cn09–cn12). The selected attempt-3 artifacts use nodes
cn01–cn04 and report different completion counts. Their settings cannot be
joined by benchmark ID alone. This also explains why the earlier frozen audit's
mean TTFT of 2.54149 seconds is not used for attempt 3.

### Reconciled measured metrics, without predictions

| Attempt-3 measurement | Value | Definition/source |
| --- | --- | --- |
| Mean TTFT | 2426.745607 ms | AIPerf `time_to_first_token` |
| Mean request TPOT/ITL | 6.401720 ms | AIPerf `inter_token_latency`; request decode span divided by output tokens minus one |
| Output throughput, AIPerf | 1194.468951 tokens/s | AIPerf aggregate; retained as its own metric |
| Output throughput, published converter | 1198.24914 tokens/s | 4,347,870 recorded output tokens over 3628.51918 s |
| Profiling requests | 4218 successful, 15 errors | Raw per-request records, excluding warmup |
| Warmup requests | 354 | Recorded separately; excluded from scored requests |

The published converter's `records_error_dropped=369` includes the 354 warmup
records. It does **not** mean 369 profiling errors. AIPerf records those warmup
responses as usage-only/empty-content errors; the producer's warmup lifecycle
still completes. Successful request counts are not completed session counts.
Branch counts and actual response lengths are retained in the audit; no session
completion score is inferred from them. The metrics are observations, not
simulator comparisons, and the two throughput definitions are not interchanged.

### Why this point is not scored yet

1. **Warmup needs matched qualification.** The pinned producer advances live
   trajectories with zero idle delay and one-token outputs. After all issued
   wire requests return, it persists 63 unfinished streams across 32 lanes.
   These are live conversations, not 63 outstanding requests. The default
   AISimulate `primer` policy repeats saved prefixes. The explicit
   `live_trajectories` policy advances the DAG, retains pending joins and stream
   positions, and restores the timestamped global spread. Its additional wait
   for native server cleanup remains a validation difference. Matching seed
   and request counts alone does not establish identical snapshots, cache
   contents, or request payloads. Recorded arrival replay would also replace
   the closed-loop concurrency controller.

   Source anchors: attempt-3 `benchmark.log` lines 205–210 records quota stop,
   wire return, then 63-stream handoff. Pinned `agentic_replay.py` implements
   drained handoff in `finalize_phase` (952–972), timestamped offsets in
   `_handoff_replay_offset_ms` (1196–1227), and global normalization in
   `_profiling_phase_t0_offset_ms` (1387–1400). The timestamped implementation,
   rather than the file's older generic full-delay comment, defines this path.
2. **Fixed verification still needs proof.** Numeric K and AL are recovered,
   but exact runtime defaults and synthetic acceptance semantics remain unknown.
   AL minus one would be 2.61 accepted drafts if the pinned runtime uses the
   inspected v0.29 convention; that remains a conditional conversion.
3. **Transfer/routing needs a pinned contract.** Both roles use NIXL through
   MultiConnector, plus MooncakeStoreConnector with `enable_offload=false`.
   Decode logs report capacity-only Mooncake operation. Prefill initializes
   store threads; the flag alone does not establish every store/prefix path.
   `external_kv_transfer` counters include P/D traffic and cannot prove CPU
   offload. The frontend uses random routing with 900-second session affinity;
   actual rank selection and draft KV handoff/initialization remain unqualified.
4. **Metric and corpus replay identity is incomplete.** The artifacts expose
   request lengths, IDs and timing, not complete token/prefix identity and
   output-stop semantics. Actual outputs differ from requested output lengths.
   No kernel-level draft/verify timing is available in this evidence.

The source-pinned AIPerf submodule is
[`754356e9a39acc6cc6afb242d123bb57c3fb6f75`](https://github.com/SemiAnalysisAI/aiperf/tree/754356e9a39acc6cc6afb242d123bb57c3fb6f75).
Its `src/aiperf/timing/strategies/agentic_replay.py` describes live-trajectory
warmup; `src/aiperf/metrics/types/inter_token_latency_metric.py` defines request
ITL. This is source corroboration, not proof of an unrecorded runtime default.

No matched prediction is produced: eligible paired points **0**, MAPE and signed
bias **unavailable**. The prior audit's blanket DSpark/AgentX unsupported flag
is obsolete; the blockers above are narrower evidence and behavior mismatches.
B200 and GB200 SGLang P/D points with HiCache remain outside the offload-free
scope. No new GPU collection or extension into offload, routing, or FPM was run.

```sh
# gh authentication is needed for GitHub artifact downloads; raw archives stay ignored.
python/aisimulate/.venv/bin/python \
  benchmarks/evidence/dspark-fixed-20261014/audit_agentx_pd.py --download
# Recheck the frozen local artifacts without fetching new API responses.
python/aisimulate/.venv/bin/python \
  benchmarks/evidence/dspark-fixed-20261014/audit_agentx_pd.py
```

The first command fetches about 450 MB of compressed artifacts. The server-log
archive expands to much more; parsing streams past large telemetry members and
only retains small role configurations, fingerprints and sweep logs. API sibling
responses can change: use the saved hashes and local files to reproduce this
audit. Original raw sources are not committed.

## Controlled collection

`runtime-receipt.json` records the successfully inspected vLLM 0.29.0 image,
immutable image/tag identity and available Flash checkpoint. The inspected
image supports `enable_adaptive_verification=false`. The Pro checkpoint was
not found at its expected shared-cache path.
The pinned v0.29.0 configuration source maps synthetic length 3.77 to 2.77
accepted drafts using rates `[1, 1, 0.77, 0, 0, 0]`, matching Replay's
adjacent-integer assumption. This does not qualify the older AgentX producer.

`collect_pd.py` launches one TP4/EP4 producer and consumer, K=6, synthetic
committed length 3.77, FP8 KV, eager execution, prefix caching, 8 GiB KV per GPU,
four sequences per worker, and Mooncake TCP. It records the exact commands,
model-config hash, logs, all input/output token IDs and request metrics.
The controlled corpus has three turns, a 50 ms tool delay, 128 initial tokens,
32 additional input tokens per later turn and 64 output tokens per turn.
One session warms the servers before measured session concurrency 1 and 4.

TTFT starts before the producer HTTP request. TPOT is `(last token arrival -
first token arrival)/(output token count - 1)`, never a mean streaming chunk
gap. This small controlled corpus is separate from AgentX accuracy evidence.
The driver and connector handshake have not yet executed on GPUs.

Target: `nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster`, namespace
`spec-decoding-eval`. Resources: 8 B200, 32 CPU, 512 GiB RAM, read-only
`shared-model-cache` and the task-owned `aisim-dspark-results-20261014` PVC.
KAI queue `dynamo`; no dedicated-node tolerations or unrelated workloads changed.
After 18 minutes without a GPU assignment, the task's inspection/collection Jobs
and ConfigMap were deleted. The results PVC was retained. `collection-status.json`
records the scheduler's resource-shortage events and resume action. No GPU
measurement was produced, and no task GPU allocation remains.

```sh
PY=python/aisimulate/.venv/bin/python
EVIDENCE=benchmarks/evidence/dspark-fixed-20261014
$PY "$EVIDENCE/collect_reference.py"
kubectl --context nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster \
  -n spec-decoding-eval create configmap aisim-dspark-scripts-20261014 \
  --from-file=collect_pd.py="$EVIDENCE/collect_pd.py" --dry-run=client -o yaml | \
kubectl --context nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster \
  -n spec-decoding-eval apply -f -
kubectl --context nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster \
  -n spec-decoding-eval apply -f "$EVIDENCE/collect-pd.yaml"
```

Before scoring the collected run, replay the recorded token IDs, warmup and
closed-loop session load. Resolve actual cache capacity and transfer behavior
from server logs. Pin the serving revision separately from the op database
version. The existing B200 vLLM op data is 0.24.0; it is not vLLM 0.29.0 data.

## Verification

- Rust library: 1920 passed, 1 ignored.
- Python DSpark/ngram/speculation suite: 313 passed.
- Final DSpark feature suite: 51 passed, including Pro TP8/EP1 vLLM/SGLang
  construction and memory, and automatic draft-resolution round trips.
- Final DSpark/ngram and affected Sweeper regression subset: 163 passed.
- Earlier config/runner/Sweeper regression suite: 780 passed.
- Final release parity: 386 passed in 197.59 seconds.
- K=3/5/6/7, zero/full/fractional acceptance, repeatable seeds, short outputs,
  context limits, KV exhaustion, cancellation and rejection rollback covered.
  The state-pressure test also exercises DSpark K=6/7 preemption and completion.
- Aggregated/P/D AgentX with snapshot/warmup, saved recommendation round trips,
  real op-cost P/D with automatic transfer, draft memory, and window caps covered.
- Actual op-level P/D smoke completed one request with all 19 output tokens.
  This is functional evidence, not a silicon accuracy result.
- Ruff, Rust formatting, packaged legal-file checks and explicit ownership
  coverage passed. CODEOWNERS tests: 107 passed.

```sh
cargo test -p aisimulate-core --lib --no-default-features
python/aisimulate/.venv/bin/pytest -p no:timeout -q \
  tests/test_dspark_cli.py tests/test_ngram_cli.py \
  python/aisimulate/tests/unit/sdk/speculation
python/aisimulate/.venv/bin/pytest -p no:timeout -q \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py
```

## PR integration checks

The draft PR rebases only the DSpark implementation/evidence commits onto
`9666978555325f9caa4b816e8509527313f8e330`. It excludes the unrelated commits
from the original development base. Conflict resolution preserves the newer
KV memory module, uses engine-spec schema 29, and adds draft KV to the existing
once-per-handoff `state_bytes` field.

- Rebased Rust library: 1975 passed, 1 ignored, including a regression for
  window-capped draft transfer alongside other request state.
- Ruff, Rust formatting, packaged legal-file equality and CODEOWNERS regeneration
  passed.
- The host measurements and earlier parity counts below describe the recorded
  pre-rebase implementation; they are not new measurements of current main.

## Host cost

`compare_host.py` measures complete release CLI wall time, with one discarded
warmup and alternating base/head repetitions. It fixes thread counts,
checks completed request/token counts, records native binary/config/lockfile
identity, and records peak RSS on macOS. The AgentX fixture is the repository's
129-request, 114540-output-token play. A 128-request non-speculative control
checks ordinary execution. Candidate DSpark is a separate feature-cost cell;
its reduced forward count is not a baseline implementation speedup.

```sh
git worktree add --detach /tmp/aisim-dspark-base-cb71f236 cb71f236
uv sync --project /tmp/aisim-dspark-base-cb71f236/python/aisimulate \
  --extra dev --python "$PWD/python/aisimulate/.venv/bin/python"
# Align the optional HTTP packages present in the measured candidate environment.
uv pip install --python /tmp/aisim-dspark-base-cb71f236/python/aisimulate/.venv/bin/python \
  certifi==2026.7.22 charset-normalizer==3.5.2 idna==3.20 requests==2.34.2 urllib3==2.8.0
# Build the candidate from its Python package directory, which owns the module name.
(cd python/aisimulate && VIRTUAL_ENV="$PWD/.venv" uv tool run maturin develop --release)
python/aisimulate/.venv/bin/python \
  benchmarks/evidence/dspark-fixed-20261014/compare_host.py \
  --base /tmp/aisim-dspark-base-cb71f236 --head "$PWD" \
  --output /tmp/aisim-dspark-host --repeats 5
```

The initial three-repetition run (`host-initial.json`) showed noisy control
timing, including a 4.19 s baseline sample. A second five-repetition run
(`host-stable.json`) used identical production code, binaries and workloads:

| Full CLI case | Baseline median (range), s | Candidate median (range), s | Median RSS, MiB (base / head) |
| --- | --- | --- | --- |
| Non-speculative control | 2.470 (2.408–2.684) | 2.432 (2.339–2.573) | 197.27 / 197.58 |
| Non-speculative AgentX | 3.848 (3.773–4.246) | 3.957 (3.670–4.182) | 196.80 / 197.00 |
| DSpark AgentX | Unsupported | 4.011 (3.667–4.188) | — / 197.70 |

Baseline is `cb71f236`; candidate is signed commit `9d841505`. Loaded native
hashes match their release artifacts; common dependency versions and lockfile
hashes match. A later audit found five extra HTTP packages in the candidate
environment, so these two runs are preliminary. The baseline environment was
then aligned for the final comparison. Candidate changes during the second run
were ownership/evidence only. The preliminary samples vary: initial
control median changed +7.8%, then -1.6%; AgentX changed -2.2%, then +2.8%.
These short CLI samples do not establish performance at larger resident state.
`host-semantic-comparison.json` confirms equal non-speculative completion/token
counts and TTFT/TPOT/E2E predictions; AgentX lifecycle digests also match.

Final dependency-matched results (`host-matched.json`, five repetitions):

| Full CLI case | Baseline median (range), s | Candidate median (range), s | Median RSS, MiB (base / head) |
| --- | --- | --- | --- |
| Non-speculative control | 2.523 (2.432–2.659) | 2.529 (2.408–2.728) | 197.25 / 198.00 |
| Non-speculative AgentX | 3.736 (3.681–3.942) | 3.896 (3.729–3.972) | 199.39 / 198.12 |
| DSpark AgentX | Unsupported | 3.470 (3.418–3.583) | — / 199.27 |

The final medians change +0.24% for the control and +4.29% for AgentX
(+0.160 s). The ranges overlap, but these samples do not rule out a small
host-cost regression. No numerical performance threshold was specified.
The harness now rejects mismatched installed dependency sets before timing.

Implementation cost: resolution and draft-capacity inversion occur during model
construction. The scheduler reuses the existing sampler and reservation paths;
there is no new per-round checkpoint read or draft-model construction.
Ordinary mixed-step composition adds draft-name checks within its existing
operation traversal; it adds no resident-state scan or new retained cache.

## Remaining causes and coverage gaps

1. V4 draft SWA uses a window-capped HCA proxy. Wide-query target verification
   uses token-basis attention; matched query-width measurements are absent.
2. V4 context preparation follows pinned vLLM source. SGLang uses that cost
   approximation until its exact implementation is measured. Norm/KV-store
   work uses bandwidth estimates.
3. Host launch/sampling overhead and eager/graph behavior are uncalibrated.
4. V4 grouped-cache identity is approximated by scalar capacity/prefix reuse.
   Automatic transfer assumes the connector carries prepared draft KV; exact
   payload and initialization behavior need confirmation on the pinned runtime.
5. AgentX acceptance and fixed-verification identity are not yet qualified.
   The observed vLLM 3.77 convention is kept separate from cost-model error.
6. Missing GPU capacity prevented controlled P/D collection. Existing AgentX P/D
   artifacts are available and take priority; their runtime and live-trajectory
   warmup contracts currently prevent a matched replay. Missing evidence must
   not be reported as successful validation.

Phase 2 FPM design, collection planning and accuracy tests are deferred.

## Playground diagnostic reproduction

`diagnose_playground.py` retains separate SSE-gap and counter-derived round
observations, plus their source hashes. It queries the canonical op model at
the approximate start/end KV lengths of each wave, reporting predicted draft
and verify components. These components were not measured independently.
The difference percentages are diagnostic; eligible accuracy points remain zero.

`playground-diagnostics.json` contains 18 batch/depth points. At batch 1, the
predicted round ranges are 6.40–6.45 ms (K=3), 6.95–6.99 ms (K=5), and
7.43–7.58 ms (K=7), versus SSE-gap medians 9.55/10.17/10.64 ms. The gaps are
about -28.8% to -33.1%. Unmodeled host overhead, attention proxies and runtime
identity differences remain possible causes; this evidence cannot isolate them.

```sh
python/aisimulate/.venv/bin/python \
  benchmarks/evidence/dspark-fixed-20261014/diagnose_playground.py \
  --playground /path/to/AISimPlayground/playground/dsv4flash-b200-specdec-validation \
  --output benchmarks/evidence/dspark-fixed-20261014/playground-diagnostics.json
```

## Live warmup functional qualification

`traffic.load.agentic_warmup_policy: live_trajectories` adds the pinned
producer's one-token, zero-idle, ten-admissions-per-lane live DAG preparation.
The default remains `primer`. The live policy requires `agentic_profile` for
lane recycling and preserves its corpus cursor into profiling.

- Six native regression cases cover quota admission with concurrent requests,
  mandatory primer spread, cancellation, deterministic fork/join and recycling,
  timestamped stream spread, aggregate vLLM/SGLang, and P/D vLLM/SGLang.
- Profile request counters and play outcomes exclude preparation. Prepared
  cache identities remain available, with actual admission reuse checked in
  native runtime fixtures. Original profile output lengths return after warmup.
- The functional server-cleanup fixtures deliberately hold cleanup after wire
  completion: one case adds 20 ms (terminal 10 ms, cleanup 30 ms), another adds
  4 ms (terminal 1 ms, cleanup 5 ms). Profiling waits for cleanup. These are
  synthetic contract checks, not measured estimates of AgentX cleanup overhead.
- The 444916 audit still has zero paired prediction/measurement points. Matching
  this phase policy alone does not resolve snapshot, input identity, routing,
  runtime acceptance, or transfer-contract qualification.

```sh
CARGO_TARGET_DIR=/tmp/aisim-dspark-warmup-tests CARGO_BUILD_JOBS=4 \
  cargo test -p aisimulate-core --lib live_warmup --no-default-features
python/aisimulate/.venv/bin/pytest -p no:timeout \
  tests/test_cli_config.py -k agentic -q
python/aisimulate/.venv/bin/pytest -p no:timeout \
  tests/test_runner.py -k agentic -q
```

The focused Python runs passed 57 configuration tests and 276 runner tests.
These tests use local fixtures and do not constitute DSpark timing accuracy
measurements or an AgentX paired-record count.
