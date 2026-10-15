# Pro0813 graph smoke diagnostic

Job 2257371 completed one measured 1024/1024 request after drained warmup:
8 B200, TP8/EP1/PP1, vLLM ac9126 image, fixed DSpark K6, synthetic AL 3.77,
prefix caching and external offload disabled. This is a smoke, not one of the
128-request production windows; campaign MAPE remains unavailable.

| Metric | Measured | Replay | Signed diagnostic difference |
| --- | ---: | ---: | ---: |
| TTFT | 195.70 ms | 76.64 ms | -60.84% |
| Request TPOT | 3.34 ms | 3.62 ms | +8.42% |
| Output throughput | 283.58 tokens/s | 270.96 tokens/s | -4.45% |

Replay uses the exact token IDs and frozen accepted-draft mean 2.77. The observed
273 rounds accepted 750 drafts (mean 2.7473), with conditional rates
`[1, 1, 0.7473, 0, 0, 0]`. The control was not changed to match either the measured
acceptance realization or latency. This differs from the real-acceptance eager
smoke, so the two serving timings do not establish a graph speedup.

All eight rank receipts agree on 31,426 shared physical blocks, each 1,435,968
bytes. Eight tensor descriptors alias one 45,126,730,368-byte allocation per rank;
they are not eight allocations. The five cache groups include 61 target and 3
draft layers. `cache-reconciliation.json` records manual Replay sizing:
31 approximate steady-state blocks per request, unchanged pool block count,
and 192 bytes per block of integer representation rounding. The independent
full-retention bound includes prefill and two unsettled speculative batches:
857 blocks at C1, well below 31,426. Native sliding-window lifetimes remain
approximate. This reconciliation does not qualify AgentX prefix behavior or P/D.

`replay-match.json` freezes inputs and disclosed modeling approximations.
`prediction/` retains full metrics, model diagnostics, a source patch and hashes.
The source patch is losslessly gzip-compressed; decompress it before applying
it to the recorded revision.
Remaining errors include historical op-table reuse, target query-width and
draft-SWA proxies, and unmodeled host/frontend work; their individual contributions
have not been measured. Large error does not by itself exclude a production pair.

Raw logs, runtime/cache identities, source snapshot, token records and counters:
`../pro-graph-smoke-2257371.tar.gz`, SHA256
`e3a414e9e76231eecadc4d985ba89fe6708a074ce7dc6773a51fcac74f69707c`.
The provisional prediction before the cache-bound audit is retained separately in
`../pro-graph-provisional-prediction.tar.gz`, SHA256
`022a045733da72af8c143ad9c3ad1f63192e16e8337c42ef08bb04ad7f5eba10`.
Only `prediction/` supplies the table above. Embedded model configuration is
attributed in `../configs/pro0813/README.md`.

From the repository root, with the recorded predictor revision/source patch and
a matching native build:

```bash
tar -xzf benchmarks/evidence/dspark-e2e-20261014/pro-graph-smoke-2257371.tar.gz \
  -C benchmarks/evidence/dspark-e2e-20261014
python/aisimulate/.venv/bin/python benchmarks/evidence/dspark-e2e-20261014/reconcile_pro_cache.py \
  --receipt benchmarks/evidence/dspark-e2e-20261014/raw/pro-graph-smoke-2257371/effective-cache.json \
  --k 6 --concurrency 1 --input-length 1024 --output-length 1024 \
  --output /tmp/pro-graph-cache-check.json
python/aisimulate/.venv/bin/python benchmarks/evidence/dspark-e2e-20261014/predict.py \
  --measurement benchmarks/evidence/dspark-e2e-20261014/raw/pro-graph-smoke-2257371/smoke-c1-r1/summary.json \
  --config benchmarks/evidence/dspark-e2e-20261014/smoke-pro-graph/replay-config.yaml \
  --match-receipt benchmarks/evidence/dspark-e2e-20261014/smoke-pro-graph/replay-match.json \
  --output-dir /tmp/dspark-pro-graph-replay
```
