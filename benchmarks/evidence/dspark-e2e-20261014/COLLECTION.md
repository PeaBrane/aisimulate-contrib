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
  This launcher remains GPU-untested until its smoke is recorded.

Aggregated smoke/pilot/initial collection and P/D smoke are implemented. Expanded
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
unqualified. P/D still requires an equivalent per-role receipt before scoring.

New aggregated runs snapshot launcher/client sources and the matrix into their
result directory, hash each file, and execute the client and read-only cache
extension from that snapshot. Older smokes retain their recorded source hashes
and are explicitly distinguished from this stronger source receipt.

Kimi FP8 KV requires `attention_config.use_prefill_query_quantization=true` in
the pinned runtime. The first eager startup reached native warmup and rejected
the missing flag; its failure is retained. Corrected launches explicitly enable
it, preserving BF16 projection weights and the runtime's FP8 prefill query path.
