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

## Controlled collection

`runtime-receipt.json` records the successfully inspected vLLM 0.29.0 image,
immutable image/tag identity and available Flash checkpoint. The inspected
image supports `enable_adaptive_verification=false`. The Pro checkpoint was
not found at its expected shared-cache path.

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
- Recommendation and affected Sweeper regression subset: 117 passed.
- Earlier config/runner/Sweeper regression suite: 780 passed.
- Final release parity: 386 passed in 197.59 seconds.
- K=3/5/6/7, zero/full/fractional acceptance, repeatable seeds, short outputs,
  context limits, KV exhaustion, cancellation and rejection rollback covered.
- Aggregated/P/D AgentX with snapshot/warmup, saved recommendation round trips,
  real op-cost P/D with automatic transfer, draft memory, and window caps covered.
- Actual op-level P/D smoke completed one request with all 19 output tokens.
  This is functional evidence, not a silicon accuracy result.

```sh
cargo test -p aisimulate-core --lib --no-default-features
python/aisimulate/.venv/bin/pytest -p no:timeout -q \
  tests/test_dspark_cli.py tests/test_ngram_cli.py \
  python/aisimulate/tests/unit/sdk/speculation
python/aisimulate/.venv/bin/pytest -p no:timeout -q \
  crates/core/parity_tests/perfmodel/test_compile_engine_parity.py \
  crates/core/parity_tests/perfmodel/test_engine_step_parity.py
```

## Host cost

`compare_host.py` measures complete release CLI wall time, with one discarded
warmup and three alternating base/head repetitions. It fixes thread counts,
checks completed request/token counts, records native binary/config/lockfile
identity, and records peak RSS on macOS. The AgentX fixture is the repository's
129-request, 114540-output-token play. A 128-request non-speculative control
checks ordinary execution. Candidate DSpark is a separate feature-cost cell;
its reduced forward count is not a baseline implementation speedup.

```sh
python/aisimulate/.venv/bin/python \
  benchmarks/evidence/dspark-fixed-20261014/compare_host.py \
  --base /tmp/aisim-dspark-base-cb71f236 --head "$PWD" \
  --output /tmp/aisim-dspark-host
```

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
6. Missing GPU capacity prevents the controlled P/D measurement and paired
   replay. Missing evidence must not be reported as successful validation.

Phase 2 FPM design, collection planning and accuracy tests are deferred.
