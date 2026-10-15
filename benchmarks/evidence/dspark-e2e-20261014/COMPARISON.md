# Paired serving comparison

`compare.py` joins one measurement and one native Engine Replay prediction by
`cell_id` and repetition (1–3). The matrix includes unrun and failed cells.
Duplicate attempts must be selected explicitly before scoring; the tool does
not select the fastest measurement or the smallest error.

```bash
python/aisimulate/.venv/bin/python benchmarks/evidence/dspark-e2e-20261014/compare.py \
  --matrix benchmarks/evidence/dspark-e2e-20261014/matrix.json \
  --measurements results/cell/repeat-1/summary.json \
  --predictions predictions/cell/repeat-1/summary.json \
  --output-dir benchmarks/evidence/dspark-e2e-20261014/comparison
```

Omit either input list to report missing evidence. Outputs are `paired.csv`,
`report.json`, and `measured-vs-replay.png`; source file hashes are retained.
An empty campaign has unavailable errors, not zero MAPE.
The CSV uses LF line endings so generated evidence also passes repository
whitespace checks.

## Produce a paired prediction

```bash
python/aisimulate/.venv/bin/python benchmarks/evidence/dspark-e2e-20261014/predict.py \
  --measurement results/cell/repeat-1/summary.json \
  --config replay-config.yaml --match-receipt replay-match.json \
  --output-dir predictions/cell/repeat-1
```

The YAML uses the public prediction API: `estimation_mode: op_level`,
`fallback_policy: deny`, fixed DSpark, measured per-role parallelism and cache
capacities, prefix caching disabled, and no external offload. Configure
`decoder_replay: true` for the pinned V4.1 vLLM runtime. Kimi NIXL P/D uses
`engine.kv_transfer.prefill_backoff_tokens: 1`; the V4 families use zero.
Set `decode_recompute_tokens: 1` for V4/Kimi and `128` for V4.1's bounded
decoder replay; this destination suffix is independent of source backoff.
Serving identity includes model and draft revisions, precision, attention
backend, graph policy, runtime/image and effective cache settings. Qualification
must reconcile those settings with the resolved canonical model and explicitly
retain differences in operator-table coverage and cache representation.

Qualification establishes a reproducible, matched workload and serving setup;
it does not require an exact cost model. Disclosed operator-table version reuse,
attention approximations, and unmodeled host overhead remain part of the measured
prediction error once the model, workload, topology and effective cache settings
are reconciled. They must not exclude a complete pair merely because its error
is large. A missing timing query that prevents Replay from completing, an unknown
serving identity, or an unverified cache mapping remains an explicit gap.

The predictor passes the measured input and output token IDs to native Engine
Replay with the measured closed-loop request concurrency. It saves the resolved
Replay specification, raw report, performance/memory diagnostics, source hashes
and host wall time. Host wall time is never used as serving latency.
The predictor saves a binary-capable patch for tracked runtime sources and its
own entrypoint against the recorded Git revision. New runtime source files must
be committed first. A source or native-library change during prediction rejects
the attempt, so concurrent builds cannot silently change the reported identity.

Freeze a `replay-match.json` receipt before running the predictor:

```json
{
  "measurement_identity_sha256": "<digest from measured summary>",
  "replay_config_sha256": "<SHA256 of canonical YAML bytes>",
  "target_config_path": "target/config.json",
  "target_config_sha256": "<pinned target config SHA256>",
  "draft_config_path": "draft/config.json",
  "draft_config_sha256": "<pinned draft config SHA256>",
  "acceptance_path": "acceptance.json",
  "acceptance_sha256": "<SHA256 of acceptance evidence bytes>",
  "cache_reconciliation_path": "cache-reconciliation.json",
  "cache_reconciliation_sha256": "<SHA256 of cache mapping bytes>",
  "runtime_cache_receipt_path": "effective-cache.json",
  "qualification": {"eligible": false, "reasons": ["unresolved identity differences"]}
}
```

Acceptance paths are relative to the receipt. For real acceptance, use measured
counter deltas from `acceptance.py`, preferring conditional rates. The predictor
rejects a mean-only configuration when per-position rates are available. For
synthetic acceptance, use a separate frozen control containing `accepted_tokens`
(excluding the bonus), `frozen_before_measurement: true`, and its producer or
128-request pilot source. Retain the measured acceptance separately. A fresh
runtime's synthetic sampler must be shown to match adjacent-integer sampling
before using the mean control. Neither receipt nor acceptance can be adjusted
to reduce latency error.

Config paths are also relative to the receipt. Their hashes and revisions must
match `checkpoint-references.json`. The predictor checks the canonical target
loader against the pinned source config and requires the exact pinned
`draft_config` in Replay YAML. A public model name alone is insufficient.
The predictor also checks context and scheduler limits against frozen launch
arguments. Qualified Pro aggregated runs require the cache adapter's matching
workload, capacity, runtime receipt hash, and nonbinding-capacity proof. Other
families and P/D need their own checked adapters before scoring. An approximate
cache mapping cannot qualify a workload whose capacity may be binding.

If runtime qualification finishes after collection, preserve the original
`summary.json` and frozen `identity.json`. Write a separate
`qualified-summary.json` beside them, retaining all original metrics and
identity fields, and attach the original summary hash and qualification receipt.
Pass that reviewed summary explicitly to prediction and scoring. Qualification
must explain resolved runtime/cache checks and remaining model approximations;
it must not alter measurements, workloads or acceptance to improve error.

The serving identity is frozen before collection. Real acceptance evidence and
the corresponding Replay receipt are necessarily created after that measurement
window, but before prediction and scoring. They do not rewrite serving identity.
An eligible receipt records the completed reconciliation, not merely a matching
hash. A failed or unqualified prediction remains visible with no scored error.

## Run summary contract

Both summaries contain `cell_id`, `repetition`, `run_id`, `identity_sha256`,
`total_gpus`, `expected_requests`, `status`, and
`qualification: {eligible: true|false, reasons: [...]}`. Predictions additionally
identify `predictor: aisimulate.engine_replay`. The identity digest covers the
frozen model/draft, serving configuration, exact workload and acceptance inputs.
Runtime, predictor revision/native binary and performance-data identities are
retained separately. A shared digest alone does not establish qualification;
the source receipts must establish the settings it describes.

`metrics_contract` must be `dspark-e2e-client-v1`. Required metrics:

- `mean_ttft_ms`: arithmetic mean of request first-token latency, from client
  dispatch to the first nonempty token event.
- `mean_tpot_ms`: arithmetic mean of each request's
  `(last_token_time - first_token_time) / (output_tokens - 1)` in milliseconds.
  Multi-token chunks contribute their full token count to the denominator.
  Streaming chunk gaps are retained for diagnostics and are not TPOT samples.
- `output_throughput_tok_s`: successful client-visible output tokens divided
  by the full drained measurement window, including startup/tail within that
  window. Warmup is outside the window.
- `completed_requests` and `failed_requests`: measured-window counts. A scored
  fixed-length run must complete the planned count without failures.

Absolute and per-GPU output throughput are both reported. P/D uses the sum of
prefill and decode GPUs; comparisons with different GPU counts are not
equal-resource speedups. These are serving E2E metrics. Kernel timing, FPM
accuracy and simulator host overhead are separate evidence.

## Error and coverage

For measured value `y > 0` and prediction `p > 0`,
`APE = 100 × abs(p - y) / y` and signed error is `100 × (p - y) / y`.
MAPE and signed bias are unweighted arithmetic means over scored run points,
with a separate denominator for each metric. They are not error of averages
or request-weighted estimates. Per-configuration variability uses the sample
standard deviation across paired repetitions; one repetition yields no
standard deviation.

The report distinguishes planned configurations, runs, measured/predicted
requests and paired requests. Fixed-length runs have no session count.
Identity mismatch, unsupported predictor, unqualified inputs, incomplete runs
and invalid metrics remain visible with null errors. No numerical pass/fail
threshold is used. Real-acceptance mean-only approximations must be identified
in the run identity and compared separately from conditional-rate runs.
