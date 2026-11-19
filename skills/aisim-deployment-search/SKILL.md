---
name: aisim-deployment-search
description: Ranks serving configs for one model on a GPU type and budget already chosen, simulated on CPU with AISim. For how many GPUs a load needs, use aisim-gpu-selection-and-sizing.
license: Apache-2.0
compatibility: "Python 3.11-3.13; aisim 0.14.x (targets 0.14.0; full evals pending); CPU only for the search; optional GPU spot-check needs a Dynamo host or Kubernetes cluster"
allowed-tools: Bash Read Write
permissions: [env, file_read, file_write, network, shell, kubectl_exec]
metadata:
  author: NVIDIA AISim team
  version: "0.1.0"
  tags:
    - aisim
    - dynamo
    - inference
    - serving-configuration
    - simulation
---

# AISim Deployment Search

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Purpose

The GPU type and budget are given, and the question is how to run the
model on them: "maximize throughput on these eight B200s", "aggregated or
disaggregated on this node?". AISim simulates candidate
deployments on CPU (backend, aggregated or prefill/decode disaggregated,
tensor/data/expert parallelism, replicas, scheduler limits), ranks them for
the user's goal, and writes each pick as a concrete config. GPUs are used
only at the end, with approval, to spot-check a short list.

AISim narrows the search. It does not replace measurement on the target
hardware. Label predicted numbers as simulated, also after a spot-check;
the spot-check adds measured numbers next to them, it does not turn the
predictions into measurements.

## When to use

- Fixed GPU type and budget; the question is the best configuration on them
  (most throughput per GPU, lowest TTFT, best goodput under an SLA, or the
  throughput/latency trade-off).
- Aggregated against disaggregated, or vLLM against SGLang, on given GPUs.
- Deploying and benchmarking the top picks of an AISim search.

Do not use this skill when:

- the decision is the GPU count or GPU type: "how many do we need?", "how
  many of my eight must I allocate?", "can these eight handle 100 requests
  per second?". That is sizing: use `aisim-gpu-selection-and-sizing` if it
  is installed, otherwise say this skill does not size hardware;
- the user wants a published, already-measured recipe, a broken
  deployment fixed, or an endpoint that is already running benchmarked.
  Those need recipe, troubleshooting or benchmarking tools, not
  simulation.

## Former names

AISim was previously called AIConfigurator and then AISimulate. You will still
see those names in places: the `aiconfigurator` and `aisimulate` packages, an
`aiconfigurator` command that AISim 0.14 still installs, `AICONFIGURATOR_*`
variables in generated scripts, and the AIConfigurator paper. Treat them as
legacy names for AISim. Use the `aisim` package and the `aisim` CLI only. Do
not install `aiconfigurator` or `aisimulate`, and do not use the `aiconfigurator`
command (its `support` subcommand answers a different question and often says
NO for combinations AISim handles).

## Prerequisites

- `aisim` 0.14.x in a Python 3.11 to 3.13 environment; check with
  `aisim --version`. If it is missing, ask before installing
  (`python3 -m pip install "aisim==0.14.0"`), because that downloads
  packages. Remove an old `aisimulate` install first (see
  [references/install-and-support.md](references/install-and-support.md)).
- Model metadata: AISim ships configs for about 95 models. For any other model it
  fetches `config.json` from huggingface.co over HTTPS. `HF_HUB_OFFLINE` does
  not stop that fetch, so a model outside the bundled set needs network
  access (and `HF_TOKEN` for gated models).
- About 1 GB of free RAM per parallel candidate. On a small machine or
  sandbox, cap AISim with
  `execution: {resources: {memory_limit_gb: 4, cpu_limit: 4}}`.
  A cap above free RAM fails at once with exit 3.
- Files are written under the output directories you pass, plus the
  Python package cache during installation. The optional spot-check also
  writes on the GPU host or cluster.
- For step 6 only: a GPU host or Kubernetes cluster with NVIDIA Dynamo, the
  model weights, and explicit approval from the user.
- `SKILL_DIR` below means the directory that contains this `SKILL.md`.

## Instructions

### 1. Pin the question

Confirm back to the user, one line each:

- Model: Hugging Face ID, for example `Qwen/Qwen3.5-35B-A3B`.
- GPU type: an AISim system ID. Blackwell first: `b200_sxm`, `b300_sxm`,
  `gb200`, `gb300` (all have vLLM, SGLang and TensorRT-LLM data in 0.14.0).
  Also available: `h200_sxm`, `h100_sxm`, `h100_pcie`, `a100_sxm`,
  `a100_pcie`, `a30`, `l40s`, `l4`, `b60`, `rtx_pro_6000_server`; coverage
  per backend varies. GB200 and GB300 systems are modeled as 4 GPUs per
  node in an NVLink domain, so candidates above 4 GPUs span nodes.
- GPU budget: the most GPUs a candidate may use.
- Workload: input and output tokens, and fixed concurrency or a request
  rate. Also the longest request the user needs, for `context_length`.
- Goal: `throughput_per_gpu`, `throughput`, `throughput_per_user`,
  `goodput`, `goodput_per_gpu` (needs an SLA), `ttft`, `e2e_latency` or
  `pareto`, plus any SLA (`ttft_ms`, `itl_ms`, or `e2e_ms`). `itl_ms` is
  checked against mean TPOT.

Ask once for anything missing. Do not invent an SLA.

### 2. Write recommendation.yaml

Use a template from
[references/config-templates.md](references/config-templates.md) and fill in
only the pinned values. Always set `engine.context_length`; leaving it unset
uses the model maximum (262144 tokens for Qwen3.5) and silently drops
parallel shapes whose KV cache cannot hold one such sequence. Keep
`parallelism: {preset: default}` unless the user names shapes. Start with
`optimizer.max_trials` between 8 and 32 and tell the user it is a sample.

### 3. Run the search on CPU

```bash
aisim recommend --stack engine \
  --config recommendation.yaml \
  --output-dir ./aisim-recommendation
```

Use a new output directory for every run. An existing non-empty directory
fails with exit 2 unless `--overwrite` is passed, and a failed run can leave
files behind. Exit codes:

- `0`: at least one candidate selected.
- `1`: no feasible candidate. Explain why from the not-feasible rows below.
- `2`: invalid configuration, or no shape is viable for the model, GPU type
  and backend (for example "no deployment_mode has a viable parallel
  config"). Report the message; do not retry with invented data.
- `3`: local resource limit. If `recommendations/` was still written, the
  result covers only the candidates that ran; say so. Then read the reason
  in `resource-runtime.json` and the error message, and act on the cause:

  - the run's processes exceeded the memory cap: a lower cap fails again.
    Raise the cap (it must stay below free RAM) or shrink the search, for
    example fewer GPUs or backends, then rerun.
  - free host memory fell below the reserved headroom: free host memory
    until the reserve is available, then rerun. Lowering the cap does not
    restore free-memory headroom.
  - the cap was above free RAM at start: lower the cap below free RAM or
    free enough host memory for that cap, then rerun.

  Rerun into a new directory.

The run log can include "Loading low-fidelity fallback rows" or "Sibling
version ... excluded from implicit nearest-earlier fallback". They do not
stop the run, but they mean some operations are priced from substituted
data. Tell the user when they appear.

### 4. Read the result

`recommendation.csv` stores each candidate's config and metrics as JSON
blobs and is hard to read. Use the bundled summarizer:

```bash
python3 "$SKILL_DIR/scripts/summarize_recommendation.py" ./aisim-recommendation --top 5
```

It prints, in rank order: the YAML file, backend and version, parallel
shape, GPUs used, score, output tokens per second (total and per GPU),
mean TTFT, TPOT and E2E, and completed and SLA-compliant requests. It then
lists the not-feasible candidates and why, and states how many rows it
left out. Add `--json` for machine-readable output. Report the top three
and say how many candidates were evaluated and rejected.

When the user compares modes (aggregated against disaggregated), add
`--best-per-mode`. The best candidate of a mode can rank below the top
three; the script prints it anyway. If it has no YAML file, export one
with `--export <candidate_id> --to best-<mode>.yaml`.
When comparing modes, disclose that an omitted `engine.kv_transfer` (a null
`kv_transfer_bandwidth` in the result) assumes zero KV-transfer delay, so the
disaggregated result can be optimistic. For a deployment decision, rerun the
disaggregated mode separately with a representative
`engine.kv_transfer.bandwidth_gb_per_second`; aggregated mode rejects that block.


At fixed concurrency, `throughput_per_gpu` favors the smallest pool that
holds the load. If the user cares about latency too, say so and offer a
`pareto` run.

### 5. Predict the short list in detail

```bash
aisim predict --stack engine \
  --config ./aisim-recommendation/recommendations/0001.yaml \
  --capture-per-request \
  --output-dir ./aisim-pick-0001
```

The summary prints a `Timing (<role>): estimator=...; source=...;
data_version=...` line per worker role; report it with the numbers. Use the
same `--stack` as the recommendation. The numbered YAML files contain
absolute paths into the local AISim install (`systems_paths`), so re-run
them on the same installation rather than copying them elsewhere.

### 6. Spot-check on GPUs (only with approval)

Stop and ask before this step: which picks, how many GPUs, for how long.
Then follow [references/gpu-spot-check.md](references/gpu-spot-check.md):
render Dynamo launch scripts for a pick with `aisim.generator` (CPU, no
approval needed), launch on the user's host or cluster, and benchmark with
the same arrival process the simulation used:

- concurrency workloads use the generated `bench_run.sh`, which is closed
  loop;
- request-rate workloads use the AIPerf request-rate command in the
  reference.

Never apply a numbered `recommendations/*.yaml` file as a Kubernetes
manifest.

### 7. Compare and conclude

Put predicted next to measured for each pick: output tokens per second per
GPU, TTFT, per-request TPOT (AIPerf's per-request average ITL is AISim's
TPOT), and completed requests. Say that the prediction used
`--stack engine` while the hardware ran a full Dynamo deployment, so
frontend and routing effects are only in the measurement. Name the pick
that won on hardware. If the measured order differs from the predicted
order, say so and keep the measured result.

## Examples

**User:** "Maximize throughput per GPU for Qwen3.5-35B-A3B on these eight
B200s, 1k in / 128 out at concurrency 10."

**Agent:** writes `recommendation.yaml` with `hardware: b200_sxm`,
`backend: {choices: [vllm, sglang]}`, `context_length: 2048`,
`max_candidate_gpus: 8` and target `throughput_per_gpu`. It runs
`aisim recommend` and summarizes the top three with the script. It notes
when the per-GPU winner uses fewer than the eight GPUs, and asks before
any spot-check.

**User:** "On sixteen GB300s, is prefill/decode disaggregation worth it for
Qwen3.5-35B-A3B at 4k in / 512 out, concurrency 32, for throughput per
GPU?"

**Agent:** uses `mode: {choices: [aggregated, disaggregated]}` with
aggregated, prefill and decode worker roles in one run. It runs the
summarizer with `--best-per-mode` and reports the best of each mode side by
side, even when one ranks below the top three.

## Limits to state to the user

- Rankings come from simulation. Accuracy varies by model, GPU, backend and
  version; see the end-to-end accuracy page in the references.
- A small `max_trials` is a sample, not an exhaustive search.
- Missing data for a model, GPU and backend means AISim cannot price that
  candidate. It is not evidence that the candidate is bad.
