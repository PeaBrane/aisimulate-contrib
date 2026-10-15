# First Pro0813 fixed-length production pair

Job 2257500 completed 128 measured requests after eight drained warmup requests:
Pro0813, 8 B200, TP8/EP1/PP1, K6, synthetic AL3.77, 1024/1024, concurrency1,
repetition1. Adaptive verification, prefix caching and external offload are off.
There were zero failures and exactly 131,072 measured output tokens.

| Metric | Measured | Replay | Signed error |
| --- | ---: | ---: | ---: |
| Mean request TTFT | 193.5092 ms | 76.6420 ms | -60.3936% |
| Mean request TPOT | 3.3004 ms | 3.6560 ms | +10.7740% |
| Output throughput | 286.8397 tokens/s | 268.2929 tokens/s | -6.4659% |
| Output throughput per GPU | 35.8550 tokens/s | 33.5366 tokens/s | -6.4659% |

Coverage is one of 196 configurations and one of 588 runs. Each metric's MAPE
equals this run's APE because there is only one pair. Repetition variability is
unavailable. This is fixed-length serving evidence, not AgentX, P/D, FPM or
kernel accuracy. Simulator host time was 1.59 seconds and is not serving latency.

## Matching and limitations

`qualification.json` records the source, launch, workload and cache checks.
The installed image's graph patches preserve Pro geometry and execute context
preparation once; see `../pd-installed-source-audit.json`. All eight ranks report
31,426 shared pool blocks. The conservative bound is 857 blocks at concurrency1;
cache capacity is nonbinding. Replay retains the measured pool block count and
discloses its approximate sliding-state reservation and byte rounding.

Serving uses the immutable ac9126 image, BF16 activations, DeepSeek FP8 target
quantization, FP4 experts, FP8 KV and FLASHMLA_SPARSE_DSV4. Its effective graph
mode is FULL_AND_PIECEWISE with compilation mode NONE. Replay reuses historical
vLLM0.24 operation tables, token-basis target verification and an HCA draft-SWA
proxy. Frontend and graph launch gaps are unmodeled. These are possible error
sources; their individual contributions have not been measured.

The frozen accepted-draft mean is 2.77. Measured counters contain 34,944 rounds,
209,664 proposed drafts and 96,000 accepted drafts, mean2.7472527. Serving uses
seed42 for every request, repeating the synthetic acceptance trajectory; Replay
uses its seed42 adjacent-integer sampler. Exact RNG sequence parity is not
claimed. The observed mean was not used to retune the synthetic control or fit
latency. The original measured summary remains unchanged; the separate
`qualified-summary.json` adds the reviewed qualification receipt only.

## Reproduction

Raw logs, all token records, runtime receipts, original/qualified summaries,
counter snapshots and the collection source snapshot are in
`../pro-production-2257500.tar.gz`, SHA256
`65df7e74ee53408dc2355b885d19a6fefff4d80928aae3b45ec0a6e226e77818`.
The exact cache adapter source is in `../cache-adapter-v1-source.tar.gz`, SHA256
`94b3ce79494c368a816af295f523b342e2e7ade665affadcca92aab8a1d048e4`.
Embedded checkpoint configuration is attributed in `../configs/pro0813/README.md`.

`prediction/summary.json` pins the predictor revision and native binary.
Apply its `predictor-source.patch` to that revision and build the native library
before reproducing the numerical result. The patch, full resolved spec and report
are stored losslessly in `prediction-records.tar.gz`; `prediction-records.json`
records archive and member hashes. Extract it in this directory to inspect
the original files. From the repository root:

```bash
tar -xzf benchmarks/evidence/dspark-e2e-20261014/pro-production-2257500.tar.gz \
  -C benchmarks/evidence/dspark-e2e-20261014
python/aisimulate/.venv/bin/python benchmarks/evidence/dspark-e2e-20261014/predict.py \
  --measurement benchmarks/evidence/dspark-e2e-20261014/raw/pro-production-2257500/initial-c1-r1/qualified-summary.json \
  --config benchmarks/evidence/dspark-e2e-20261014/pro-initial-c1-r1/replay-config.yaml \
  --match-receipt benchmarks/evidence/dspark-e2e-20261014/pro-initial-c1-r1/replay-match.json \
  --output-dir /tmp/dspark-pro-production-replay
python/aisimulate/.venv/bin/python benchmarks/evidence/dspark-e2e-20261014/compare.py \
  --matrix benchmarks/evidence/dspark-e2e-20261014/matrix.json \
  --measurements benchmarks/evidence/dspark-e2e-20261014/raw/pro-production-2257500/initial-c1-r1/qualified-summary.json \
  --predictions /tmp/dspark-pro-production-replay/summary.json \
  --output-dir /tmp/dspark-pro-production-comparison
```
