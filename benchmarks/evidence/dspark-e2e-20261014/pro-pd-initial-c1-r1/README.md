# First Pro0813 P/D production pair

Job 2258192 completed 128 measured requests after eight drained warmup requests:
Pro0813, 8P+8D B200, P TP1/attention-DP8/EP8, D TP8/EP1, K6, real acceptance,
1024/1024, concurrency1, repetition1. Adaptive verification, prefix caching and
external offload are off. Each P engine handled one warmup and 16 measured
requests. There were zero failures and exactly 131,072 measured output tokens.

| Metric | Measured | Replay | Signed error |
| --- | ---: | ---: | ---: |
| Mean request TTFT | 314.9310 ms | 134.1251 ms | -57.4113% |
| Mean request TPOT | 2.0422 ms | 2.2135 ms | +8.3854% |
| Output throughput | 425.8957 tokens/s | 426.9294 tokens/s | +0.2427% |
| Output throughput per GPU | 26.6185 tokens/s | 26.6831 tokens/s | +0.2427% |

This is one P/D configuration and one repetition. Repetition variability is
unavailable. Throughput error is small because TTFT and TPOT errors offset;
component accuracy is substantially worse. Simulator host time was 2.70 seconds
and is not serving latency. The earlier aggregated pair has different acceptance
and half the GPUs; these points do not measure an equal-resource speedup.

## Matching and limitations

`request-audit.json` independently checks all token IDs, usage counts, stream
token counts, timing definitions, closed-loop admissions, frozen workload hash,
client source hash and per-engine completion counters. `qualification.json`
records the matching decision. The original summary is unchanged; a separate
qualified summary carries the reviewed eligibility receipt.

Serving identity and geometry match the installed-source audit at
`../pd-installed-source-audit.json`. Each of eight independent P pools has
7,640 blocks; the D pool has 31,426 blocks replicated across eight TP ranks.
Tensor views alias those physical pools. The conservative source bound retains
all 17 requests ever issued to each P engine, even if none were released;
capacity remains nonbinding. Replay reserves those rolling groups and retains
313 full-context P blocks per pool and 30,580 D blocks. This is a deliberately
conservative, prefix-disabled mapping, not native grouped-cache lifetime parity.

The measurement window contains 21,136 draft rounds, 126,816 proposed drafts
and 110,498 accepted drafts. Conditional per-position acceptance rates are
derived from counter deltas after warmup. Replay uses those rates with seed42;
it does not reproduce each request's RNG sequence. No acceptance value was
fitted to latency.

NIXL reports 1,024 rank observations over 128 requests and 38,231,212,032 bytes.
The 6.859 seconds of summed rank transfer time is not request critical-path
latency. Registered-row payload includes padded storage and draft state once.
Transfer bandwidth remains null, explicitly leaving transfer delay unmodeled.
Prefill-produced tokens are transferred as state and counted once at decode.

Replay reuses historical vLLM0.24 tables and declared V4 attention proxies
against the pinned vLLM0.30.1 development runtime. The current prefill MoE model
assumes balanced attention-DP work, whereas low-concurrency eager prefill has
unequal rank inputs. Frontend, proxy, graph launch and transfer gaps remain
possible error sources; their individual contributions have not been measured.

## Reproduction

Extract `../pd-production-2258192.tar.gz` at the campaign root. SHA256:
`efaba3d5faf6560cf918acfdf769b0a1eeb856fa078e92e2019b33181708a2a8`.
Its adjacent archive index retains every member hash. Cache mapping is
`../pd-cache-production-2258192.json`; its adapter source is pinned by hash
and available at predictor revision below. Model configuration attribution
is in `../configs/pro0813/README.md`.

The clean predictor revision is `b1c56e9cb6c381fbc94c47e3317f70c34d9cf664`.
Native SHA256 is
`6c8ccf1ee779ec4ce507c8d24c41e422984fd46dd72a7ee5034353a5228082a7`.
Build that revision in an isolated checkout to reproduce the numerical result.
The resolved specification, full report and empty source patch are archived in
`prediction-records.tar.gz`, with hashes in `prediction-records.json`.
The summary remains directly readable in `prediction/summary.json`.

From the repository root, using the recorded predictor build:

```bash
python/aisimulate/.venv/bin/python benchmarks/evidence/dspark-e2e-20261014/predict.py \
  --measurement benchmarks/evidence/dspark-e2e-20261014/raw/pd-production-2258192/initial-c1-r1/qualified-summary.json \
  --config benchmarks/evidence/dspark-e2e-20261014/pro-pd-initial-c1-r1/replay-config.yaml \
  --match-receipt benchmarks/evidence/dspark-e2e-20261014/pro-pd-initial-c1-r1/replay-match.json \
  --output-dir /tmp/dspark-pro-pd-production-replay
```
