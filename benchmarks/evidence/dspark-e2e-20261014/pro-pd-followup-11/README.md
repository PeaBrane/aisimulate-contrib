# Pro0813 P/D follow-up predictions

These are canonical Engine Replay predictions for five real-acceptance windows
from job 2258652 and six synthetic-control windows from job 2258677. Each window
contains 128 requests with 1024 input and 1024 output tokens, fixed K6, and C1
or C4 closed-loop admission. Serving uses 16 B200 GPUs: P TP1/attention-DP8/EP8,
D TP8/EP1. Prefix caching, external offload and adaptive verification are disabled.

Real windows use their measured conditional acceptance rates. Synthetic windows
use the setting frozen before measurement: 2.77 accepted drafts plus one
committed replacement/bonus token per round. Their observed counters remain
separate diagnostics. Seed 42 does not reproduce individual serving RNG draws.

`predictor-environment.json` and `compatibility-summary.json` record the clean
predictor/native identities and an exact numerical reproduction of the previous
128-request P/D pair. The current source is 19d7aa53; its production Rust/Python
matches the documented 5bf8f6e8 native build. The difference adds only a V4.1 test.
No silicon latency is fitted or used as a timing override.

Each window retains a Replay configuration, match receipt and prediction summary.
The match receipt binds measured identity, exact checkpoint configuration,
acceptance and reviewed per-role cache reconciliation by SHA256. The canonical
runner receives the exact input/output token IDs with `replay_concurrency` C1
or C4; it does not replay measured arrival timestamps.

Predictor limitations remain: historical vLLM 0.24 op tables against the pinned
0.30.1 development runtime, declared attention proxies, equal-work prefill MoE
approximation, and unmodeled frontend/proxy/graph/transfer latency. The grouped
cache is conservatively represented only where the separate qualification
receipt proves capacity nonbinding. These are fixed-length workload pairs,
not AgentX multi-turn validation.

## Completion and acceptance limitation

All 11 Replay runs completed 128 requests each: **1,408 requests total**, with
exactly 1024 generated tokens per request and no Replay failures. `pair-index.json`
binds each qualified measurement summary to its prediction. Errors and MAPE are
reported by the campaign comparison; this directory does not redefine scoring.

The measured acceptance counters include P/D startup rounds with invalid padded
drafts. For example, each synthetic window records 35,072 rounds but only 34,944
first-position acceptances: the difference is exactly 128 startup rounds. Real
window conditional rates include this startup contribution too, while Replay
models forced startup separately. The already frozen real-window rate contract
is preserved. Its resulting small acceptance approximation remains mixed with
cost-model error; the paired metrics do not isolate pure operation-model error.
Synthetic predictions continue to use the pre-measurement 2.77 setting.

## Reproduction

Use the recorded predictor revision/native build. Extract the raw measurement
archives at the campaign root, then follow each entry in `pair-index.json`:

```bash
python/aisimulate/.venv/bin/python benchmarks/evidence/dspark-e2e-20261014/predict.py \
  --measurement benchmarks/evidence/dspark-e2e-20261014/raw/pd-production-2258652/initial-c1-r2/qualified-summary.json \
  --config benchmarks/evidence/dspark-e2e-20261014/pro-pd-followup-11/2258652-initial-c1-r2/replay-config.yaml \
  --match-receipt benchmarks/evidence/dspark-e2e-20261014/pro-pd-followup-11/2258652-initial-c1-r2/replay-match.json \
  --output-dir /tmp/dspark-followup-real-c1-r2
```

Change only the indexed window paths for the other runs. The archived Replay
specifications retain exact request IDs and input/output token IDs; generated
arrival times are all zero with explicit closed-loop concurrency. Summaries
remain directly readable. `prediction-records.json` records SHA256 and byte size
for the lossless `prediction-records.tar.gz` archive and each member. Extract it
inside this directory to restore the full Replay reports/specifications.
