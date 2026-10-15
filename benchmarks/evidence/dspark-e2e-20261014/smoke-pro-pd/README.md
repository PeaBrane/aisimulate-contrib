# Pro0813 P/D diagnostic pair

Job 2257933 completed one warmup and one measured request with real acceptance,
K6 and exact 1024/1024 lengths. P uses eight B200 GPUs with TP1/attention-DP8/EP8;
D uses eight B200 GPUs with TP8/EP1. Both use the pinned vLLM runtime and default
graph policy, prefix caching off, external offload off, and context length 16,384.
This single-request smoke is excluded from production MAPE.

| Metric | Measured | Replay |
|---|---:|---:|
| TTFT | 631.197 ms | 134.125 ms |
| Request TPOT | 1.870566 ms | 1.977801 ms |
| Output throughput | 402.249 tokens/s | 474.642 tokens/s |
| Output throughput per GPU | 25.141 tokens/s | 29.665 tokens/s |
| Completed requests / output tokens | 1 / 1024 | 1 / 1024 |

The exact input/output IDs, server usage and Replay completion agree. P produced
one hidden output token; the proxy streamed only D's 1024-token response. Replay
admits the request once to each role, with no readmission or prefix reuse.
Conditional acceptance rates come from this window's counters (149 rounds,
894 proposed drafts, 876 accepted); they are not fitted to latency.

All 16 rank receipts pass the shared-pool geometry checks. P has 7,640 physical
blocks per independent DP pool, D 31,426 blocks per TP replica. The conservative
native source-hold bound counts every issued request since startup, including
warmup, even if none had been released. It qualifies only this two-request run.

NIXL reports eight rank reads totaling 298,681,344 bytes for the measured request.
The registered-region bound includes padded storage rows and draft state once;
logical page content alone undercounts those reads. The 45.25 ms summed rank
transfer time is not request critical-path latency. This diagnostic leaves
transfer bandwidth unmodeled, so Replay's zero transfer delay is an approximation.

Serving counters show 293.31 ms of P inference, 310.16 ms of D queue time and 14.70 ms
of D first-forward time. Replay predicts 120.14 ms of P work and 13.98 ms before the
first D output. These counters have different boundaries from isolated kernels.
Warmup used P engine 0, whereas measurement used engine 1; cold-peer setup is a
possible contributor, not a proven explanation of the entire TTFT gap. The
current MoE model also assumes balanced attention-DP work, unlike the actual
uneven eager prefill batches. Runtime/table reuse and V4 attention proxies remain
explicit cost-model approximations.

## Reproduce

Extract `../pd-smoke-2257933.tar.gz` at the campaign root, then run from the
repository root with the recorded native build:

```bash
python/aisimulate/.venv/bin/python benchmarks/evidence/dspark-e2e-20261014/predict.py \
  --measurement benchmarks/evidence/dspark-e2e-20261014/raw/pd-2257933/smoke-c1-r1/summary.json \
  --config benchmarks/evidence/dspark-e2e-20261014/smoke-pro-pd/replay-config.yaml \
  --match-receipt benchmarks/evidence/dspark-e2e-20261014/smoke-pro-pd/replay-match.json \
  --output-dir /tmp/dspark-pro-pd-prediction
```

Prediction revision: `9e0338ed5ac588bd07aa547dc5a95f78992398ef`, clean runtime
sources. Native SHA256:
`b0a7066a8ca5634b859a61ef615e90246a4aabfdf1d438e4b9dd5e5b93e6fdbe`.
`diagnostic-comparison.json` retains metric errors and source hashes;
`prediction/` retains the full specification, report and provenance.
