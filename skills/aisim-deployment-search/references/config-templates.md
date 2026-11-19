<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recommendation config templates

Schema reference: `docs/reference/configuration.md` and
`docs/sweeper/search-space.md` in ai-dynamo/aisim (0.14.0). Both templates
below were run with aisim 0.14.0 on CPU.

## A. Best throughput per GPU on a fixed GPU type and budget (8 x B300)

```yaml
traffic:
  source: {type: synthetic, input_tokens: 1024, output_tokens: 128}
  load: {type: concurrency, concurrency: 10}
  stop: {requests: 100}

engine:
  mode: aggregated
  model: Qwen/Qwen3.5-35B-A3B
  hardware: b300_sxm
  backend: {choices: [vllm, sglang, trtllm]}
  context_length: 2048
  workers:
    aggregated:
      parallelism: {preset: default}

optimization:
  target: throughput_per_gpu
  constraints:
    max_candidate_gpus: 8

optimizer:
  max_trials: 12
```

## B. Aggregated against disaggregated in one search (16 x GB300)

`engine.mode` accepts a `choices` domain. Give all three worker roles; the
search uses `aggregated` for one branch and `prefill` plus `decode` for the
other, each in its own optimizer study.

```yaml
traffic:
  source: {type: synthetic, input_tokens: 4096, output_tokens: 512}
  load: {type: concurrency, concurrency: 32}
  stop: {requests: 320}

engine:
  mode: {choices: [aggregated, disaggregated]}
  model: Qwen/Qwen3.5-35B-A3B
  hardware: gb300
  backend: vllm
  context_length: 8192
  workers:
    aggregated:
      parallelism: {preset: default}
    prefill:
      parallelism: {preset: default}
    decode:
      parallelism: {preset: default}

optimization:
  target: throughput_per_gpu
  constraints:
    max_candidate_gpus: 16

optimizer:
  max_trials: 16
```

This shared input omits `engine.kv_transfer`, so disaggregated candidates assume
zero KV-transfer delay (`kv_transfer_bandwidth: null` in the result). Their
latency and throughput can be optimistic relative to a real deployment. Report
that assumption with every `--best-per-mode` comparison. Before using the result
for a deployment decision, rerun disaggregated mode separately with a
representative `engine.kv_transfer.bandwidth_gb_per_second`. Keep that block out
of the shared input: aggregated mode rejects KV-transfer settings.

Read it with `summarize_recommendation.py --best-per-mode`. The best
disaggregated candidate can rank below the overall top three. GB200 and
GB300 are modeled as 4-GPU nodes in one NVLink domain; candidates up to 16
GPUs (multi-node, both modes) evaluated without errors in testing.

For NVFP4 checkpoints (for example `nvidia/Qwen3-235B-A22B-NVFP4`), keep the
GPU type Blackwell. AISim reads the quantization from the checkpoint. On
Hopper every candidate fails with "quant mode 'nvfp4' needs
'fp4_tc_flops'", and the run exits 1.

## C. Goals with an SLA

```yaml
evaluation:
  sla:
    ttft_ms: 500
    itl_ms: 50          # checked against mean TPOT

optimization:
  target: goodput_per_gpu
  strict_sla: true      # reject candidates whose means miss a bound
  constraints:
    max_candidate_gpus: 8
```

`goodput` and `goodput_per_gpu` need at least one SLA bound.

## Optional: cap local resources

```yaml
execution:
  resources: {memory_limit_gb: 4, cpu_limit: 4}
```

## Overrides without editing the file

```bash
aisim recommend --config recommendation.yaml \
  --set traffic.load.concurrency=16 \
  --set optimization.constraints.max_candidate_gpus=4 \
  --output-dir ./aisim-recommendation-c16
```

Unknown paths are rejected. The last assignment to a path wins.
