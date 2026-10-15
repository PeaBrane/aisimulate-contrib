# Pro-0813 eager smoke diagnostic

This is one completed measured request after one drained warmup request:
DeepSeek-V4-Pro-0813, vLLM, 8 B200, TP8/EP1, fixed DSpark K6, real acceptance,
1024 input and exactly 1024 output tokens, concurrency 1. Slurm job 2257185
completed successfully. Prefix caching and external KV offload were disabled.

**This smoke is unqualified for production accuracy scoring.** It contributes
zero runs to the 196-configuration matrix and has no campaign MAPE.

| Metric | Measured | Original Replay | Corrected Replay |
| --- | ---: | ---: | ---: |
| TTFT | 325.57 ms | 72.69 ms | 76.64 ms |
| Request TPOT | 29.22 ms | 2.26 ms | 2.43 ms |
| Output throughput | 33.88 tokens/s | 430.26 tokens/s | 398.83 tokens/s |

The corrected prediction uses three draft layers, the fused context WKV
projection and its resident allocation, and the missing attention output
reductions. These changes follow the pinned serving source; neither acceptance
nor measured latency was used to fit costs. Input, output, acceptance and Replay
configuration hashes are unchanged. See the [source audit](../pro0813-attention-audit.md).

Replay uses the exact measured input/output token IDs and conditional acceptance
rates from counters around the measurement window: 184 rounds, 1,104 proposed
drafts and 843 accepted drafts. Accepted drafts average 4.5815 per round;
the final output round is clipped. Acceptance was not adjusted to match latency.

The difference remains unresolved. The measured runtime was eager; the op model
does not represent its CPU/launch overhead. The tables use vLLM 0.24.0 rather
than serving revision `ac9126e58aa7bbab1856ba6593ba4d5003fea516`. Wide-query target
attention and draft SWA retain documented token-basis/HCA approximations. The
smoke predates exact cache RPC receipts, so Replay uses a conservative 48 GiB
pool from the rounded 48.08 GiB log and the existing pooled-state approximation.
These are candidate causes and known mismatches, not a measured decomposition
of the latency gap. Graph-enabled production collection is a separate identity.

`diagnostic.json` and `prediction/` retain the original values, exclusions and
source hashes. `diagnostic-pro-fix.json` and `prediction-pro-fix/` retain the
corrected result, including raw Replay metrics and diagnostics. The large
remaining difference is unresolved; neither version is a production pair.
`prediction-install-overlap/` is a superseded attempt repeated after native
installation finished; it is excluded from both comparisons.

The original serving logs, identities, exact request records, counters and
warmup records are archived in `../pro-eager-smoke-2257185.tar.gz`.
SHA256: `d055ebba1ab4240356582b622dd088b3310f856c6c2c8579a5215540853122cc`.
The embedded draft configuration is attributed in `../configs/pro0813/README.md`.

To reproduce from the repository root:

```bash
tar -xzf benchmarks/evidence/dspark-e2e-20261014/pro-eager-smoke-2257185.tar.gz \
  -C benchmarks/evidence/dspark-e2e-20261014
python/aisimulate/.venv/bin/python benchmarks/evidence/dspark-e2e-20261014/predict.py \
  --measurement benchmarks/evidence/dspark-e2e-20261014/raw/pro-eager-smoke-2257185/smoke-c1-r1/summary.json \
  --config benchmarks/evidence/dspark-e2e-20261014/smoke-pro-eager/replay-config.yaml \
  --match-receipt benchmarks/evidence/dspark-e2e-20261014/smoke-pro-eager/replay-match.json \
  --output-dir /tmp/dspark-pro-eager-replay
```
