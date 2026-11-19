<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISIM-2001 Replay host cost

November 18, 2026. The baseline is upstream `78370acd7c3f6b9547eea59b36dfabe0b64cf9eb`.
The candidate is PR #934 at `25a347959cc466062bd438ffa083df4e2e6b2367`,
which includes `e4b5c1cac` and the legacy aggregate chunked-prefill fix.
Both native libraries were built from their pinned source with
`cargo build --release --offline -p aisim-core --lib --features embed-python`
using Rust 1.99.1 and Python 3.12.15. The paired runner used logical CPU 7
on the same Intel Core Ultra 9 285H WSL host.

The timer surrounds the complete native `run_replay_json` call, including model
construction and final power aggregation. Each case uses identical serialized
input for both binaries, two warm-up calls per binary, and 20 pairs that alternate
which binary runs first. Timed calls disable detail capture. A separate candidate
call with diagnostics checks that the workload exercises mixed ranks.

## Reproduce

From the candidate source root, use the Python environment with the project dev
dependencies. Put the built baseline and candidate libraries at
`$AISIM_BENCHMARK_DIR/base-runtime.so` and
`$AISIM_BENCHMARK_DIR/change-runtime.so`, respectively. Set `PYTHON` to that
environment's interpreter and `PYTHON_LIBDIR` to the directory holding its
`libpython3.12.so` (needed by the `embed-python` builds). The run used Python
3.12.15 from uv:

```bash
export AISIM_BENCHMARK_DIR=/path/to/benchmark-dir
overlay=$(mktemp -d /tmp/aisim-934-benchmark-overlay.XXXXXX)
mkdir -p "$overlay/aisim"
cp -as "$PWD/python/aisim/src/aisim/." "$overlay/aisim/"
ln -s "$AISIM_BENCHMARK_DIR/change-runtime.so" "$overlay/aisim/_runtime.so"
export PYTHONPATH="$overlay:$PWD/python/aisim/src:$PWD/python/aisim"
export LD_LIBRARY_PATH="$PYTHON_LIBDIR"
taskset -c 7 "$PYTHON" \
  benchmarks/evidence/aisim-2001-replay-power/benchmark.py --pairs 20 \
  > "$AISIM_BENCHMARK_DIR/paired.jsonl" 2> "$AISIM_BENCHMARK_DIR/replay.stderr"
```

The [runner](benchmark.py) generates all input rows and Replay arguments. All
three cases use one aggregate H200/vLLM worker, 512 input tokens, 32 output
tokens, a 512-token scheduler budget, up to 32 sequences, a 1,024-token context
limit and Poisson arrivals at 200 requests/s with seed 7. The powered cases run
1,024 requests with attention DP=2. SOL uses Qwen3-0.6B and direct uses the
verified unregistered profile from `tests/test_fpm_profile_workflow.py`. The
schema-v6 SOL control runs 128 requests with DP=1 and has latency-only rows.
The shared synthetic grid spans batches 1–32 with separate prefill and decode
coordinates and deterministic latency and power formulas in `input_grid`.
The runner prints the native-call input and Parquet hashes. Absolute temporary
paths are part of the serialized input, so its hash can differ between runs.

## Results

Baseline and candidate matched completed requests, simulated duration, mean TTFT,
and mean TPOT exactly in every pair. The powered cases each exercised 1,084
mixed rank calls and the candidate published watts at 100% coverage. The v6
control kept both power fields null.

| Case | Baseline median | Candidate median | Median paired change | Paired middle 50% | Prior worktree median paired change |
| --- | ---: | ---: | ---: | ---: | ---: |
| v8 SOL, DP=2 | 34.05 ms | 36.37 ms | +7.0% | +3.5% to +13.1% | +4.6% |
| v8 direct, DP=2 | 33.09 ms | 34.23 ms | +2.8% | +1.3% to +6.3% | +5.4% |
| v6 SOL, DP=1 | 23.37 ms | 23.22 ms | +0.1% | -3.6% to +1.5% | -3.4% |

The current median host-time increase is 2.32 ms for SOL and 1.14 ms for
direct per 1,024-request run. Relative to the prior worktree run, SOL moved
from +4.6% to +7.0%, direct from +5.4% to +2.8%, and the v6 control from
-3.4% to +0.1%. Both new powered medians fall within the prior paired middle
50% ranges. These runs measure host overhead on synthetic data. Mixed-step
watts remain unqualified pending measured validation.

Release binary SHA-256 values

- Baseline `91d4a3e597510595dc441964e06acc9caac91e833148a32eea4d0aa9e00d3e0b`
- Candidate `7ffb72f8a9a8620d5dabe2e1fbd76c7a2f38be4973c074b537008afbfa174ba5`

<details>
<summary>Input hashes for this run</summary>

| Case | Serialized Replay input SHA-256 | Parquet SHA-256 |
| --- | --- | --- |
| v8 SOL, DP=2 | `d1cd7c93d7517a314edb7623ed0e46c446dc6036e7af47e45e03380f245c02f5` | `323ed1f814b20d9f8506f3bbee0f901fff61e6b20375c231a8bd68239b775584` |
| v8 direct, DP=2 | `1f76442214df63b083e496ddda91007e1c58a4ac91f4ad34640f232b30424f54` | `d0bde57b9e344efcfcb7747a804ed922521f635fb1bbdc1038d01567ba5bda57` |
| v6 SOL, DP=1 | `5c374b38566b50e427318adfcf390ffd6c1b36a6b98eea15d59e32579272a469` | `fee4c815a06810e73ba64d152eb5f093773ced795bb4043021f8397d7598898b` |

</details>

<details>
<summary>Paired samples in ms (baseline / candidate)</summary>

| Pair | v8 SOL, DP=2 | v8 direct, DP=2 | v6 SOL, DP=1 |
| ---: | ---: | ---: | ---: |
| 1 | 40.14 / 40.49 | 33.01 / 33.92 | 24.23 / 23.02 |
| 2 | 33.50 / 38.72 | 33.22 / 35.28 | 23.01 / 22.91 |
| 3 | 33.68 / 39.49 | 33.47 / 34.28 | 23.60 / 23.80 |
| 4 | 35.59 / 40.14 | 32.97 / 33.89 | 22.73 / 22.87 |
| 5 | 33.77 / 35.24 | 34.48 / 34.02 | 22.80 / 22.65 |
| 6 | 34.26 / 35.52 | 36.85 / 37.11 | 24.05 / 22.55 |
| 7 | 35.32 / 35.76 | 34.17 / 34.65 | 23.36 / 23.21 |
| 8 | 33.47 / 36.04 | 34.57 / 36.22 | 24.10 / 22.69 |
| 9 | 32.94 / 38.26 | 34.36 / 33.96 | 24.45 / 23.24 |
| 10 | 32.97 / 36.26 | 32.53 / 34.69 | 24.19 / 23.30 |
| 11 | 34.69 / 36.33 | 31.78 / 34.05 | 23.83 / 22.97 |
| 12 | 33.45 / 36.41 | 33.08 / 35.46 | 23.37 / 24.16 |
| 13 | 52.65 / 36.90 | 31.66 / 34.30 | 22.55 / 22.82 |
| 14 | 34.81 / 35.81 | 32.85 / 35.68 | 23.00 / 23.61 |
| 15 | 33.83 / 36.00 | 32.60 / 34.17 | 21.94 / 23.22 |
| 16 | 33.77 / 35.75 | 48.13 / 35.85 | 23.53 / 22.70 |
| 17 | 33.14 / 37.86 | 32.12 / 33.02 | 23.91 / 24.10 |
| 18 | 38.28 / 36.31 | 33.26 / 33.63 | 23.07 / 23.32 |
| 19 | 35.96 / 41.01 | 33.10 / 33.65 | 22.77 / 23.70 |
| 20 | 35.47 / 38.87 | 32.23 / 33.32 | 23.19 / 24.39 |

</details>
