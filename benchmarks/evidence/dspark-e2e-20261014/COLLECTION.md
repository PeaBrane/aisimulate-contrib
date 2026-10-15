# Collection protocol and provenance

## Op-table coverage preflight

Run `python/aisimulate/.venv/bin/python benchmarks/evidence/dspark-e2e-20261014/audit_op_coverage.py`
from the repository root. The script verifies immutable checkpoint config hashes
and calls the canonical performance API with `op_level`, `SILICON`, and
`fallback_policy=deny`. It records one-sequence prefill at 1024 tokens and decode
with 1024 past-KV tokens in `op-coverage.json`, including native source diagnostics,
admitted reuse sources, and relevant packaged-table identities.

The current snapshot constructs Pro0813 TP8/EP1 K6 and answers both queries.
V4.1 TP4/EP1 K5 and Kimi TP8/EP1 K2 fail construction on missing vLLM
`dsv41_module` and `generation_mla` tables respectively. Existing cross-backend
sources are not admitted by the canonical kernel whitelist. This is two answered
queries out of six planned queries, with **zero paired measurements and no MAPE**.
Pro still includes empirical primitives and declared attention approximations;
successful lookup does not establish a match to the campaign runtime.

`missing-op-plan.json` records the smallest follow-up collection scope and its
producer/consumer prerequisites. It is a design receipt, not executable collector
cases or a submitted GPU job. Repeat this preflight after model/native changes;
the snapshot includes the native-library and working-tree identities.

For the asymmetric Pro prefill worker, run the same script with
`--pro-pd-prefill --output benchmarks/evidence/dspark-e2e-20261014/op-coverage-pro-pd-prefill.json`.
The TP1/attention-DP8/EP8, K6 probe includes initial draft proposal and answers all
four SILICON/deny queries: local batch 1/4 at ISL1024/8192. Predicted milliseconds
are respectively 120.141/894.171 and 409.579/3433.524. Each diagnostic contains
24 silicon-source operations and nine built-in empirical primitives; there are
no missing-query errors. These are CPU coverage queries, not serving records.
Batch is per attention-DP rank: MoE multiplies local tokens by eight, assuming
equal rank workloads. Global concurrency 1/4 is therefore not equivalent to
these local batch sizes. The batch4/ISL8192 query also exceeds the campaign's
8192-token scheduling budget and does not describe one actual Replay step.
The existing attention proxy and runtime/table differences remain disclosed.

The approved matrix has 196 configurations, 588 repetitions, and 153,600 measured
requests. Its first gate is 48 configurations. Generate the matrix with:

```bash
python campaign.py --output matrix.json
python -m unittest discover -s . -p test_collection.py -v
```

`collection-status.json` is the execution ledger. Submission and successful image
preparation are not silicon measurements. Each production cell remains pending
until runtime, request, acceptance, and comparison qualification succeeds.

## Pinned runtime and models

The candidate runtime is vLLM `ac9126e58aa7bbab1856ba6593ba4d5003fea516`, image digest
`sha256:3d17635c5aa0340e2f2502c067ce4b587d8e7dc678195e90b71142490dc6d71c`.
It matches the V4.1-Flash reference runtime and replaces the historical Pro/Kimi
runtimes for fresh collection. The historical Kimi image tag returned HTTP 404.
Independent Enroot imports have distinct squashfs hashes; both use this OCI digest.

Exact shared target/draft revisions are recorded in `checkpoint-references.json`.
`inspect_runtime.py` verifies the shared config against the immutable Hugging Face
config hash and checks every index-referenced weight file exists. It does not hash
all multi-terabyte weight contents, so the receipt preserves that limitation.
Shared model paths are mounted read-only, with no copied or downloaded checkpoint.

The V4.1 AgentX recipe has `engram-config.cpu_offload=true` despite its published
KV-offload label being `off`. The fresh smoke explicitly disables Engram CPU
residency. Eager execution is limited to diagnostic smokes. Production selects
`--graph-policy runtime-default`, leaving graph capture settings to the pinned
runtime and retaining its effective settings in logs. A separate graph smoke
exports cache capacity after graph reservation. This is a recorded configuration
change, not a claim of historical-run equivalence; retain the selected production
policy into subsequent fixed-length and AgentX runs.

Each production job can select `DSPARK_CONCURRENCY` and `DSPARK_REPETITION`
(or launcher `--concurrency` / `--repetition`) to collect one window. Completed
windows retain separate summaries and raw records. Resume the campaign by
submitting only missing windows into new result directories; never overwrite a
completed window. Budget jobs from graph-smoke throughput before grouping windows.

## Exact-token client

The InferenceX fixed-length benchmark was inspected at
`38799fc52bb3f133ba4badf86ec777d2841b4801`, path
`inferencex-e2e/infx/bench_serving/benchmark_serving.py`. It does not retain the exact
input/output IDs needed by this request audit. `collect_fixed.py` is an original
client using the public vLLM `/tokenize` and `/v1/completions` endpoints.
It generates deterministic natural-language token pools, slices exact input
lengths, sends token-ID prompts, forces output length with `ignore_eos`, and
requires returned token IDs and server usage to agree.

Warmup requests complete before the measured counter snapshot. A fixed thread
pool replenishes completed requests, preserving closed-loop concurrency. Both
failed and successful request records are retained. TTFT is first returned-token
time; TPOT is last-minus-first returned-token time divided by output tokens minus
one. Stream-event records describe token bursts and are never treated as per-token
latency samples. The client boundary includes HTTP/frontend work.

Each run preserves `identity.json`, `workload.jsonl`, `warmup.jsonl`,
`request_records.jsonl`, metrics before/after, and `summary.json` with
`metrics_contract=dspark-e2e-client-v1`. Repetitions are 1, 2, 3. Serving/workload identity is frozen before each measurement window. Real-acceptance
evidence and the canonical Replay configuration are finalized after the window,
before prediction, in a separate hash-linked `replay-match.json` receipt.
Smoke results are always ineligible for production scoring.

`acceptance.py --run RUN --k K` computes measured-window counter differences.
vLLM per-position accepted counts are cumulative-prefix counts; the adapter
converts them to conditional probabilities, rejects counter resets, and preserves
mean-only sampling as a separate approximation when position counters are absent.
AL includes the bonus token; accepted draft mean is AL minus one. Counter evidence
is an aggregate over engine rounds, not an exact per-request acceptance trace.

The pinned producer `vllm/config/speculative.py::_acceptance_length_to_rates`
converts an AL of L to accepted-draft mean a=L-1: floor(a) unconditional
position rates are 1, the next is a-floor(a), and all remaining rates are 0.
This commits adjacent integer token counts, matching Replay `accepted_tokens`.
Synthetic controls preserve `accepted_tokens`, `frozen_before_measurement: true`,
and the source reference/pilot separately from observed window counters.

## NIXL contract audit

The following source references use vLLM
`ac9126e58aa7bbab1856ba6593ba4d5003fea516`; no upstream implementation is copied.

- `vllm/v1/worker/gpu/model_runner.py`: `get_kv_cache_spec` includes draft attention;
  all cache tensors are registered. `sample_tokens` calls the draft proposer before
  connector post-forward processing.
- `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` inherits DFlash context-KV
  precomputation. `dflash/speculator.py` stores prefill context into every draft KV
  group before drafting. Auxiliary hidden states are consumed locally; registered
  draft KV is the transferred representation.
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`: transfer groups
  support target and draft cache regions; replicated MLA regions differ from
  head-sharded full-attention regions.
- `nixl/tp_mapping.py:91-104` supports Pro's TP1-to-TP8 MLA handoff: each D rank
  reads the complete replicated MLA region from rank zero of the selected P
  engine. `nixl/base_scheduler.py:76-80` adds the DP index to the base side-channel
  port, so the eight P engines have distinct listeners. The launcher uses base
  ports 5557 (P) and 5657 (D), with both roles in the same allocation.
- `nixl/base_scheduler.py` uses prefill backoff
  `max(has_mamba ? 1 : 0, num_prefill_lookahead_tokens - 1)`. DSpark's lookahead is
  one (`vllm/config/vllm.py`). Kimi KDA returns `MambaSpec`, so Kimi trims one prompt
  token on P and recomputes it on D. V4/V4.1 DSpark has no such lookahead backoff.
- `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py` sends P a one-output
  request, forwards returned transfer parameters to D with the original prompt
  and output length, and streams only D's output. The P token is not exposed to the
  client. Actual D suffix execution and transfer sizes still require silicon audit.
- `docs/features/nixl_connector_compatibility.md` requires compatible speculation
  identity. Hybrid SSM speculation is marked uncertain; Kimi homogeneous TP avoids
  the documented heterogeneous-TP restriction but still needs a runtime smoke.

Source inspection does not establish P/D performance or correctness. No connector
payload, timing, or first-token behavior is declared measured before a GPU run.

## Execution entrypoints

- `prepare-runtime.sh`: CPU-only image import and unmodified pinned upstream NIXL
  proxy preparation. `pd_driver.py` verifies the proxy content hash before use.
- `inspect-runtime.sh`: CPU-only package/source/checkpoint receipt.
- `agg-smoke.sh`: one-model allocation. Set `DSPARK_MODEL_KEY` and `DSPARK_K`;
  `DSPARK_RUN_MODE` defaults to `smoke`, or accepts `pilot` (128 real requests) and
  `initial` (concurrency 1/4, three repetitions). `DSPARK_SYNTHETIC_AL`, when set,
  is the pre-frozen control including bonus. `server-ready.json` links the identity,
  worker logs, endpoint and metrics. Effective cache capacity must still be parsed
  and checked before pairing; a ready server is not a qualified prediction.
- `pd-smoke.sh`: one combined allocation, Flash one node/eight GPUs or Pro/Kimi
  two nodes/eight GPUs per node. Each role uses `serve_role.py`; Pro prefill is
  TP1/attention-DP8/EP8, all other worker topologies follow the matrix. The client
  uses the original pinned NIXL proxy, with tokenization against P outside timing.
  It supports smoke, pilot and selected initial windows with matching controls
  on both roles. Pro P/D GPU smoke is running after recorded startup failures;
  source compatibility and cache admission do not yet prove a completed handoff.

Aggregated and P/D smoke/pilot/initial collection are implemented. Expanded
sweeps and AgentX collection remain gated by matched initial results; no script
silently launches those stages. Capture the exact Slurm submission arguments in
execution receipts, including CPUs, memory, nodes, GPU count, and dependencies.

## Host allocation correction

The first Flash and Kimi smokes used 256 GB host RAM. Flash repeatedly reached
the cgroup memory limit during MoE weight staging, while Kimi shard loading
slowed to roughly 90 seconds per shard. No OOM kill occurred; memory pressure
and inherited OMP=8 are diagnostic observations, not an isolated causal proof.
They were cancelled before any measured
request. Replacement allocations use 768 GB for Flash, 1200 GB for Pro, and
2048 GB for Kimi per role, with explicit OMP/MKL thread count 1. P/D reserves
both role memory budgets together. This changes startup resources, not GPU
topology or checkpoint identity; startup logs remain in the remote result paths.

## Freeze additional-K controls

Run `serve_smoke.py --mode pilot` with real acceptance to complete an independent
128-request pilot. Then run `freeze_acceptance.py --pilot PILOT_RUN --output CONTROL.json`.
The tool verifies the pilot identity hash and complete request count, records the
acceptance evidence hash, and refuses to overwrite an existing control. It uses
mean acceptance for the adjacent-integer synthetic control; observed conditional
rates remain evidence and are not silently substituted for that distribution.

Pass `--synthetic-control CONTROL.json` to the serving runner, or set
`DSPARK_SYNTHETIC_CONTROL=/workspace/.../CONTROL.json` for `agg-smoke.sh`.
The runner validates model revision and K, and records the entire frozen control
in each serving identity. Reference-K `--synthetic-al` accepts only the verified
historical value. No pilot or control is selected by latency error.

## Exact runtime cache receipts

Subsequent aggregated runs load the read-only `runtime_receipt.DsparkReceiptWorker`
extension and enable the pinned runtime's development RPC route. The API binds
only to `127.0.0.1`. Before warmup, the launcher exports every rank's initialized
cache groups, per-group spec/page bytes, padded tensor strides, pool block count,
and available cache bytes to `effective-cache.json`. Its hash and the extension
source hash are frozen into the serving identity before requests begin. The
extension never replaces a forward method or runs inside the measured window.
Initial smokes launched before this addition retain their original logs and are
unqualified.

P/D now binds each role's API and inspection RPC to its explicit private
allocation hostname. Before warmup, each worker atomically writes a unique
`cache-receipts/ROLE/dp-D-tp-T.json`; the driver requires all eight workers per
Pro role with the expected DP/TP/EP configuration. This is necessary because
`vllm/v1/engine/core_client.py:1669-1671` broadcasts the utility call to all DP
engines but returns only the first engine's response. Missing or duplicate
worker identities fail before collection. The combined `effective-cache.json`
and its hash retain target/draft transfer groups and physical page sizes.
No serving forward path is changed.

`reconcile_pro_cache.py --deployment disaggregated` validates these role receipts
separately. P uses TP1/attention-DP8/EP8: each of its eight independent cache pools
must pass the full-concurrency retention bound, without assuming evenly routed
requests. The adapter statically reserves a full-retention upper bound for all
sliding groups, including draft windows and compressor state, in each pool.
The shared Replay P template uses the smallest remaining pool capacity, applied
independently to every DP rank. D uses TP8/EP1: all eight physical replicas
must agree, and capacity is counted once per request set. P retains its prompt and
hidden one-token response; D retains the requested output window. Both bounds
include unsettled verification rounds and draft lookahead. Missing workers,
missing draft layers, partial transfer groups, and unsupported geometry fail.
For role token bound `T = ISL + role_OSL + 2*(K+1) + K`, the reserved blocks are
`C * sum(ceil(T / group_block_size) + 1)` over sliding groups; `role_OSL` is one
on P and the full requested OSL on D. Original physical capacities and all
per-pool bounds remain in the report. A failed nonbinding-capacity proof returns
no usable role cache configuration. This conservative mapping is restricted to
the fixed-length, closed-loop, prefix-disabled workload. It omits dynamic
`state_cache` because the Pro rolling windows must not be treated as a recurrent
state boundary under B0/R1; the existing Kimi recurrent-state guard is unchanged.

Replay preserves these independent pools: `compiler.py` passes attention-DP into
the role configuration; `runner.py` validates its timing identity;
`engine/runtime.rs::EngineFactory::build` and
`engine/generalized/engine.rs::new_with_rank_factory` create a separate scheduler
and allocator for each DP rank. Disaggregated placement addresses those ranks
individually. This mapping establishes cache capacity only. It does not qualify
runtime routing latency or NIXL transport. Synthetic contract tests cover the
adapter, and the complete 16-worker receipt from smoke `2257933` passes its
physical checks. That smoke has 7640 blocks per P pool and 31426 per D replica.
`pd-cache-smoke-2257933.json` records the mapping; the source-residency receipt
accounts for both issued requests (one warmup, one measured), with one on each
of P engines 0 and 1. The bounded cache proof applies to that smoke only.

The adapter also emits `replay_kv_transfer`: explicit numeric `bytes_per_token`,
`full_prompt`, source backoff zero and decode recomputation one. Its per-reader
registered-payload bound includes every target, draft and compressor group once. For
each group, it bounds block count by `ceil(T_prefill/block_size)` and caps sliding
groups at `ceil(window/block_size)+1`, matching the pinned NIXL clipping contract.
The report retains logical content from summed layer-page bytes separately.
Packed MLA NIXL registration transfers a full shared storage row for every block,
including padding: the transfer bound therefore multiplies the summed block
counts by the physical block stride. Dividing by ISL and rounding up makes the
existing scalar transfer field usable for this fixed input length. This is a
per-reader registration bound, not measured payload or aggregate wire traffic.
Numeric override disables automatic draft-byte addition, preventing
double counting. No bandwidth is inferred: null leaves transfer delay unmodeled.
An optional `--bandwidth-receipt` JSON must supply `kind` (`measured` or `assumed`),
positive finite `gb_per_second`, and a nonempty `source`. Its provenance is saved;
it does not automatically qualify transport timing or the overall comparison.

Native source retention needs a separate bound. Global concurrency and
`max_num_seqs` exclude completed requests still holding source blocks until
connector cleanup. The adapter preserves the original C-request bound as
`capacity_proven_nonbinding`, reports the generated mapping under
`replay_capacity_proven_nonbinding`, and defaults
`native_capacity_proven_nonbinding` to false. Diagnostic predictions remain
possible without claiming a native source capacity proof.
An optional `--source-residency-receipt` supplies `kind` (`measured` or
`conservative`), `source`, `includes_active_and_finished_holds: true`, and exactly
one of `max_resident_requests_per_pool` or `max_resident_requests_by_dp_rank`
(all eight string keys `"0"` through `"7"`, nonnegative integer bounds).
Bounds must cover all allocated P requests throughout the qualified window,
including warmup and earlier windows since startup. Audited cumulative issued
request counts per source engine provide a deterministic conservative bound,
even if no completed source copy were released. Do not divide the global count
by eight without per-engine routing evidence. Each pool's static reservation
uses the larger of its source-resident bound and Replay concurrency. A native
capacity proof is reported only if these bounds fit the actual role pools.

Automatic simulator transfer bytes remain a rank-local logical payload. The
Pro source sends replicated MLA to eight D ranks; these parallel reads do not
imply eight times the latency. A measured effective per-handoff bandwidth may
represent that fanout, while a shared-source NIC limit would require aggregate
wire accounting. The first P/D smoke now preserves NIXL histogram deltas in
`pd-transfer-smoke-2257933.json`: one request produced eight rank observations,
298,681,344 transferred bytes and 208 descriptors. Summed rank transfer time
is not the request critical path; the measured request also used a cold prefill
peer after warmup touched only engine 0. The receipt labels these limits;
it never treats logical bytes as measured total wire traffic or inserts an
unmeasured bandwidth. The original proxy streams D's token IDs unchanged and
does not prepend P's discarded sampled token.

New aggregated runs snapshot launcher/client sources and the matrix into their
result directory, hash each file, and execute the client and read-only cache
extension from that snapshot. Older smokes retain their recorded source hashes
and are explicitly distinguished from this stronger source receipt.

Kimi FP8 KV requires `attention_config.use_prefill_query_quantization=true` in
the pinned runtime. The first eager startup reached native warmup and rejected
the missing flag; its failure is retained. Corrected launches explicitly enable
it, preserving BF16 projection weights and the runtime's FP8 prefill query path.

## Phase B AgentX entrypoint

`agentx_phaseb.py` prepares or runs the pinned AgentX client against an existing
server; it never allocates GPUs. `agentx_server.py` records launch commands and
all native cache workers. `agentx_context_audit.py` verifies the complete corpus
and checks every source request, including subagents. The full corpus stays
outside Git. No Phase B measurement has been completed.

The producer is [SemiAnalysisAI/aiperf at 754356e9](https://github.com/SemiAnalysisAI/aiperf/tree/754356e9a39acc6cc6afb242d123bb57c3fb6f75).
The original CLI/dataset adapter follows its `src/aiperf/config/flags/cli_config.py`,
`common/scenario/inferencex_agentx_mvp.py`, `dataset/loader/base_hf_dataset.py`,
`dataset/loader/semianalysis_cc_traces_weka.py`, `dataset/loader/weka_trace.py`,
`dataset/loader/weka_trace_models.py`, and `plugin/plugins.yaml` contracts
(Apache-2.0; NVIDIA Corporation & Affiliates). It does not replace producer
scheduling or reconstruct requests independently. The external loader plugin
only fixes the HF revision and verifies the file's bytes before delegation.

The Apache-2.0 [public corpus](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/23f152f6f0f9399a85901b89a6458def0ef16729)
is pinned by revision, LFS SHA256 and size in `agentx-context-audit.json`.
The complete 1,847,151,435-byte local file passed its SHA256 check. Its 393 traces
contain 98,827 model requests; 91,944 exceed the fixed campaign's 16,384 context
limit. The largest source input plus requested output is 996,579 tokens. All
three pinned model configs declare 1,048,576 positions. This is the proposed
explicit Phase B context limit; source lengths alone do not prove the final
chat-template/tokenizer payload bound.

Before executing AgentX, collect and score a separate fixed-length anchor with
`--context-length 1048576` with prefix caching still disabled. Preserve the other serving settings.
This anchor has a different serving identity and must not replace a 16K point
in the initial campaign. The launcher defaults remain 16K/prefix-disabled.
A native-formatter witness plus complete source bounds supports pilot preflight.
The live server validates every sampled request with no filtering or truncation.
All context failures must be retained and reviewed before scoring. Unsampled
corpus payloads remain unverified; an exhaustive tokenizer pass is optional.

Prepare the producer environment on the authorized CPU/client host:

```bash
git clone https://github.com/SemiAnalysisAI/aiperf.git /work/aiperf-agentx
git -C /work/aiperf-agentx checkout --detach 754356e9a39acc6cc6afb242d123bb57c3fb6f75
uv venv /work/agentx-venv --python 3.12
uv pip install --python /work/agentx-venv/bin/python -e /work/aiperf-agentx
cp -R ./agentx_corpus_pin /work/dspark-agentx-plugin
mv /work/dspark-agentx-plugin/pyproject.toml.in /work/dspark-agentx-plugin/pyproject.toml
uv pip install --python /work/agentx-venv/bin/python -e /work/dspark-agentx-plugin
/work/agentx-venv/bin/aiperf plugins public_dataset_loader semianalysis_cc_traces_weka_062126
```

Inside an already allocated, pinned-image worker, the following starts an
aggregated Pro service; it does not submit a job. Run only after the context
identity change and matching anchor are selected for that deployment.

```bash
python serve_role.py --model-key v4-pro --role agg --k 6 --port 8010 \
  --graph-policy runtime-default --prefix-cache --context-length 1048576 \
  --output /work/agentx-server
python agentx_server.py --model-key v4-pro --deployment agg --k 6 \
  --role agg http://127.0.0.1:8010 /work/agentx-server/agg-identity.json \
  --output /work/agentx-server-receipt
```

For P/D, use the same role launcher with `--role prefill` / `--role decode`,
private `VLLM_NIXL_SIDE_CHANNEL_HOST`, separate side-channel ports, and the same
context/prefix settings. Start the pinned unmodified proxy with the command
already used by `pd_driver.py`, then give `agentx_server.py` both role endpoints
and launch receipts. All worker receipt directories must be visible to every
role. Preserve allocation/image, proxy and native-cache receipts alongside the
client output; declared image metadata alone is not proof of the running OCI.

A C1 pilot can advance once that model/deployment/K's C1 fixed anchor qualifies;
C4 similarly requires its C4 anchor. No other model, deployment or numerical
error threshold blocks this progress. Example preparation (add `--execute --context-audit agentx-context-preflight-pro.json`
after the matching fixed anchor qualifies):

```bash
python agentx_phaseb.py --model-key v4-pro --deployment agg --k 6 \
  --phase pilot --concurrency 1 --repetition 1 \
  --fixed-pair /work/anchor-c1/summary.json /work/prediction-c1.json \
  --server-identity /work/agentx-server-receipt/identity.json \
  --producer-checkout /work/aiperf-agentx --aiperf /work/agentx-venv/bin/aiperf \
  --endpoint http://127.0.0.1:8010 --metrics-url http://127.0.0.1:8010/metrics \
  --output /work/agentx-pro-agg-c1-pilot
```

The exact command is parsed in the executable's own Python environment before
execution. The manifest verifies the imported producer checkout and loader
content, and records installed dependency versions. Corpus mmap reuse is
disabled so the producer cannot bypass content verification with an older
cached dataset. HF's immutable download cache remains available.

Requested pilots last 600 seconds at C1/C4. The pinned scenario requires at
least 900 seconds, so pilots explicitly use `--unsafe-override` and are
**diagnostics, not valid scenario submissions**. Full profiles use strict
scenario validation: 3,600 seconds at C1/C4/C16, three repetitions each.
They require completed C1 and C4 pilots with hashed lifecycle evidence for
`dependencies`, `fork_join`, `warmup_frontier`, `prefix_reuse`, and
`stop_and_cancel`. Review receipts contain `run_receipt`, its SHA256, those
boolean `checks`, and an `evidence` list of `{path, sha256}` artifacts. These
receipts record evidence review; the tool never manufactures passing reviews.

Both phases use seed 42, snapshot ratios 0.25–0.75, all 393 traces, prefix cache,
10 warmup requests per lane after mandatory primers, zero idle warmup, and the
producer's drained-wire/live-DAG transition. No `--burst-phase-starts` is set.
The correct auto-warmup timeout flag is `--agentic-warmup-grace-period 1800`;
`--warmup-grace-period` alone does not configure that generated phase at this
pin. Tree/global idle caps are 300/10 seconds; response grace is 30 seconds,
cancel drain 10 seconds, and failed-request threshold 0.1. Scenario output
policy ignores EOS, uses recorded output length (minimum one), and preserves
parent/subagent dependencies. No input/output cap or corpus filter is supplied.

Execution also requires `--context-audit PATH`. An optional exhaustive receipt identifies the
exact corpus and tokenizer revision, `all_generated_payloads_verified: true`,
`truncation_or_filtering: false`, output policy
`recorded_out_minimum_one_ignore_eos`, `max_prompt_plus_requested_output`, and
hashed `evidence`. The checked maximum must fit the server and each anchor's
identical context limit. Fixed anchors keep prefix cache disabled; AgentX enables it as an explicit workload-policy change. Do not convert a source-only
length audit into this receipt.

Raw producer exports, logs, dependency versions and corpus receipt are hashed.
A completed client run remains **unscored** until Replay matches the actual
sampled snapshot/frontier, synthesized prompts and prefix identities, warmup
cleanup, output stops, acceptance, routing and metric cohorts. Equal random
seeds do not establish identical snapshots across different implementations.
The Weka producer intentionally skips `inputs.json`; retain raw exports and do
not treat that missing optional file as lost requests or invent its contents.

For slow network checkpoint reads, `DSPARK_LOAD_STRATEGY=eager` explicitly
selects the pinned runtime's full-shard reader. This is a weight-loading choice,
independent of graph execution. `DSPARK_STARTUP_TIMEOUT` sets both the supervisor
and vLLM engine-ready deadline; the Slurm allocation must leave room afterward
for measurement and cleanup. Kimi retries use 6000 seconds within a two-hour
allocation. Every run records actual command/environment as well as requested
settings. Attempt790786 exposed a missing pass-through; it was cancelled before
requests, and the main-entrypoint regression now covers both parameters.

The actual CPU tokenizer probe is retained in `agentx-tokenizer-probe.json`.
All three pinned tokenizers load. Pro0813 has no HF chat template; V4.1 and
Kimi execute their respective single-user template probes. These probes do not
prove equivalence to the serving image's native formatters or a full-corpus
payload bound. Reproduce with `agentx_corpus_pin/probe_tokenizers.py --model
HF_ID REVISION --output PATH` in the pinned producer environment (repeat
`--model` for each target). Adapter tests run there with
`python agentx_corpus_pin/test_loader.py`; the campaign's CPU gate tests use
`python -m unittest test_agentx_phaseb`.

The supported pilot policy is `native_runtime_per_request`, demonstrated by
`agentx-context-preflight-pro.json`. It binds the full source audit, a hashed
actual-image formatter witness, tokenizer revision and explicit 1M context.
It preserves every context failure and labels unsampled corpus payloads
unverified. This authorizes preflight only: review actual native usage, request
lengths, failures and cancellation cohorts before qualifying the measured window.
No multi-hour full-corpus tokenizer sweep is required to start the authorized pilot.

`agentx_corpus_pin/probe_native_context.py` reproduces the Pro witness with the
pinned producer and an unmodified native encoder exported from image job 2257688:

```bash
/work/agentx-venv/bin/python agentx_corpus_pin/probe_native_context.py \
  --manifest agentx-manifest.json --corpus /work/cache/traces.jsonl \
  --native-encoder /work/native-source/tokenizers/deepseek_v4_encoding.py \
  --output /work/native-context-probe.json
```

It reconstructs the complete trace containing the source maximum (37
conversations, 1,569 turns), then encodes that request with the actual image's
Pro formatter: 987,290 prompt tokens plus 9,379 requested output tokens gives
996,669. The session marker is excluded from this CPU witness; actual sampled
requests, including markers, are checked by the native server. The script
imports AIPerf's original reconstruction implementation and the external
encoder; it does not vendor either source or reimplement token generation.

P/D launches freeze Python files and JSON configuration dependencies together
under each allocation's `source/` directory. Run an image-backed CPU preflight
from that exact bundle before a new GPU submission: resolve the shared pinned
checkpoint and construct both role commands with their actual private host
environment. Preserve the bundle hashes and commands in
`bundle-cpu-preflight.json`. The supervisor exits when either role fails; it
does not leave the client waiting for the full startup deadline. Attempt2257706
failed before weight loading because its snapshot omitted checkpoint references;
the corrected2257733 bundle passed independent CPU check2257744.

[`acceptance-publication-audit.json`](acceptance-publication-audit.json) records
the installed source chain for counter updates. Prometheus speculation counters
update on every engine output, independently of periodic console logging. For
the qualified C1 window, completion is queued and counters are recorded without
an intervening asynchronous yield. Final-round proposals can exceed client
output after clipping; these counters describe sampled acceptance, not committed
client output counts.

`serve_role.py --preflight-only` performs checkpoint inspection and command
construction, writes `<role>-preflight.json`, then returns without running
`nvidia-smi` or starting a server. Execute it in the pinned image from the
immutable bundle, mounted with the shared model path, for both roles. For example:

```bash
VLLM_NIXL_SIDE_CHANNEL_HOST=private-prefill-node \
  VLLM_NIXL_SIDE_CHANNEL_PORT=5557 python serve_role.py \
  --model-key v4-pro --role prefill --k 6 --port 8010 \
  --graph-policy runtime-default --gpu-memory-utilization 0.92 \
  --preflight-only --output /work/preflight
```

Repeat for decode with port8020, side-channel port5657 and memory utilization0.90.
The allocation resolves private role hostnames. `DSPARK_BUNDLE_DIR` selects the
CPU-checked immutable bundle for `pd-smoke.sh`; files are copied into the result
so later uploads cannot change a running job.

P/D supports `DSPARK_RUN_MODE=smoke|pilot|initial`, optional
`DSPARK_CONCURRENCY=1|4` and `DSPARK_REPETITION=1|2|3`. Initial defaults cover
C1/C4 and all three repetitions. Each completed window has separate raw records
and acceptance evidence. `DSPARK_SYNTHETIC_AL` accepts only the verified
reference K/AL; other K values require `DSPARK_SYNTHETIC_CONTROL` from the separate
128-request real pilot. Both roles receive and record the same control. Real
pilots reject synthetic settings. These selection flags do not expand the
initial48-configuration boundary.

Pro P/D attempt2257733 failed before requests: default graph capture reserved
10.70GiB per prefill worker, leaving6.65GiB for KV at memory utilization0.90,
below the8.99GiB required for16K context. The next configuration explicitly sets
`DSPARK_PREFILL_GPU_MEMORY_UTILIZATION=0.92` and keeps decode at0.90. Model,
topology, graph policy and context remain the same; the cache receipt is
re-exported and the changed serving identity is retained.

## AgentX capture adapter (CPU qualified; native graph import pending)

The installed `aisim_agentx_pin` plugin also selects a read-only timing-strategy
subclass. `agentx_phaseb.py` verifies its source hash and sets
`DSPARK_AGENTX_CAPTURE_DIR`. The subclass delegates setup, finalization and trace
recycling to the pinned producer, then exports the realized trajectories,
snapshot `t_star_ms`, live stream/frontier states, root/parent identities,
join-gate memberships, completed-prefix replay boundaries and session cache
markers. Selected-tree metadata includes the original turn dependencies, branch
structure, tool delays, source coordinates and output-length settings. Templates
are written once per sampled source tree; ordinary request/token dispatch does
not perform extra capture writes. Capture is inactive without this environment
variable. The hook adds boundary/recycle serialization work; equivalence of host
cost has not been measured.

Warmup finalization exports the producer's **drained-wire/live-DAG** handoff.
Profiling setup must contain the identical frontier. Live streams are not
in-flight wire requests. Recycling records actual source selections and session
IDs; the adapter does not regenerate producer RNG decisions. The native Replay
clock must drive future completions and dependencies; measured client dispatch
or response timestamps are never scheduling inputs.

The existing `--export-level raw` writer preserves full request payloads, raw
responses, phase/session/turn identities, inserted cache markers, cancellations
and context-overflow records. In the pinned producer, record observers execute
before the records manager skips context overflows from public metrics. Audit
these raw records even when the public report shows no failures. Reconcile
raw counts with completion barriers and the writer's `dropped_record_count`;
missing or truncated records cannot qualify a window. Use either the per-process
raw shards or the merged export, never both.

```bash
python agentx_capture.py --capture /work/agentx-run/capture \
  --raw /work/agentx-run/aiperf/raw_records/raw_records_*.jsonl \
  --output /work/agentx-run/replay-capture
/work/agentx-venv/bin/python -m unittest discover \
  -s agentx_corpus_pin -p test_capture.py
python -m unittest test_agentx_capture
```

`requests.jsonl` retains phase/source/session mapping, canonical payload digest,
stop settings, actual finish reason, errors and any native token IDs. The original
payload remains at its raw-file/line reference. It does not infer IDs from text,
lengths or hashes, and does not replace missing requests. Default Phase B commands
preserve normal response payloads: exact input reconstruction still requires the
pinned native formatter applied offline to each actual payload, with cache-prefix
deduplication and usage checks. The current adapter intentionally leaves missing
native token IDs explicit.

`--diagnostic-token-echo` is an optional **unscored diagnostic**. It requests
`return_token_ids:true`; pinned vLLM returns prompt IDs in the first chunk and
output delta IDs in subsequent chunks. There is no independent output-only
switch in this chat contract. Returning up to one million input IDs adds material
serialization/network/export cost, so these metrics cannot silently replace a
normal AgentX measurement. A paired overhead study is required before claiming
metric equivalence. The adapter validates uint32 IDs and native usage, preserves
cancelled/short output, and reports output-stop mismatches.

`agentx-capture-preflight.json` records the actual pinned-environment command
parse, external plugin resolution and CPU tests. The original fixtures cover a
live fork/join handoff, dependency/marker preservation, missing and invalid IDs,
context overflow, duplicate raw records and truncated JSONL. No GPU profile or
paired AgentX result is claimed. Remaining native consumer work is explicit:
AgenticMooncake now accepts exact input/output token arrays as well as hash-based
prefixes, but not an externally supplied sampled frontier; captured dependencies/idle caps,
recycling and stop/cancellation cohorts still need canonical graph lowering and
integration tests. `capture-audit.json` therefore always remains unscored.

Capture API anchors are
[SemiAnalysisAI/aiperf@754356e9](https://github.com/SemiAnalysisAI/aiperf/tree/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf):
`timing/strategies/agentic_replay.py`, `timing/trajectory_source.py`,
`common/models/{dataset_models,record_models}.py`,
`post_processors/raw_record_writer_processor.py`,
`records/record_processor_service.py`, `plugin/plugins.yaml` and
`config/flags/cli_config.py` (Apache-2.0, NVIDIA Corporation & Affiliates).
Token echo is grounded in
[vLLM@ac9126 chat completion](https://github.com/vllm-project/vllm/tree/ac9126e58aa7bbab1856ba6593ba4d5003fea516/vllm/entrypoints/openai/chat_completion)
`protocol.py` and `serving.py` (Apache-2.0, vLLM contributors). All capture and
fixture code here is original; upstream scheduling and serving code is unmodified.
