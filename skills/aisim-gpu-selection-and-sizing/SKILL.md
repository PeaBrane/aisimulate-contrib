---
name: aisim-gpu-selection-and-sizing
description: Sizes how many GPUs of which type serve a model's request rate within a latency SLA (AISim on CPU). To tune GPUs already chosen, use aisim-deployment-search.
license: Apache-2.0
compatibility: "Python 3.11-3.13; aisim 0.14.x (targets 0.14.0; full evals pending); CPU only; PyYAML (installed with aisim)"
allowed-tools: Bash Read Write
permissions: [env, file_read, file_write, network, shell]
metadata:
  author: NVIDIA AISim team
  version: "0.1.0"
  tags:
    - aisim
    - capacity-planning
    - gpu-sizing
    - inference
    - simulation
---

# AISim GPU Selection and Sizing

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Purpose

Answer "how much hardware does this workload need?" for a model, an
expected request rate and a latency SLA. For each candidate GPU type, the
result is the smallest qualifying configuration found in the tested search
space: GPU count, serving configuration, latency at the planned load, the
highest tested load multiple that still qualifies, and cost if the user
supplies prices. Everything runs on CPU, in simulation.

The decision variable is the GPU count. Owning GPUs does not change that:
"I own eight B200s; how many must I allocate to this service?" and "can
these eight B200s handle 100 requests per second?" are sizing questions.
"How do I get the most out of these eight B200s?" is not; that is
`aisim-deployment-search`.

## Scope of this version

- Load must be a request rate: `traffic.load.type` `poisson` or
  `constant_rate`, with input and output token lengths.
- If the user gives the load as a fixed concurrency (closed loop) or as a
  recorded trace, say this version sizes request-rate workloads only. Ask
  for the expected request rate, or decline. The bundled scripts reject
  other load types.
- Use one backend per run, such as `vllm` or `sglang`.
- The GPU count is for a static deployment at one load, not autoscaling.

## When not to use

- Best configuration, throughput or latency on a fixed GPU budget:
  `aisim-deployment-search`.
- Published benchmark results for a GPU, or sizing training jobs.

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

- `aisim` 0.14.x in a Python 3.11 to 3.13 environment. Check with
  `aisim --version`. If it is not installed, ask before installing
  (`python3 -m pip install "aisim==0.14.0"`), because that downloads
  packages.
- Network: none for the roughly 95 models whose configs ship with AISim.
  For other models, AISim fetches `config.json` from huggingface.co over
  HTTPS, and `HF_HUB_OFFLINE` does not stop that fetch. Gated models also
  need `HF_TOKEN`.
- Files are written under the output directories you pass, plus the
  Python package cache during installation.
- Memory: AISim exits 3 at once if its memory cap is above free RAM. On a
  small machine add `--set execution.resources.memory_limit_gb=<GB>`.
- `SKILL_DIR` below means the directory that contains this `SKILL.md`.
  Run the scripts as `python3 "$SKILL_DIR/scripts/<name>.py"`.

## Instructions

### 1. Pin the sizing question

Confirm each of these with the user and write them down in the answer:

- Model (Hugging Face ID) and the GPU types to evaluate. Blackwell first:
  `b200_sxm`, `b300_sxm`, `gb200`, `gb300`. Hopper (`h200_sxm`, `h100_sxm`)
  is available for comparison. GB200 and GB300 are modeled as 4-GPU nodes
  in one NVLink domain, so counts above 4 span nodes. NVFP4 checkpoints need
  a Blackwell GPU type; on Hopper every candidate fails.
- Request rate, arrival pattern (`poisson` unless they say evenly spaced)
  and input/output token lengths. Mixed-length workloads are fine.
- SLA: `ttft_ms` and/or `itl_ms` (AISim checks `itl_ms` against mean TPOT,
  time per output token per request), or `e2e_ms` alone. These are bounds
  on aggregate means, not percentiles.
- The capacity rule, stated as two numbers the user agrees to:
  - **required SLA-compliant throughput** (`min_goodput_rps`): requests per
    second that must finish within the per-request SLA. Do not pick it
    silently. Offer the planned rate as the default and explain why the
    measured value runs slightly below it on a finite run.
  - **required compliance share**: the share of all requests that must meet
    the per-request SLA, for example 99%. It is checked in step 5.
- Optional price per GPU-hour for each GPU type. Never invent prices.

### 2. Write sizing.yaml

```yaml
traffic:
  source: {type: synthetic, input_tokens: 2048, output_tokens: 256}
  load: {type: poisson, requests_per_second: 20}
  stop: {requests: 6000}          # at least 300 s of arrivals: rate x 300

engine:
  mode: {choices: [aggregated, disaggregated]}
  model: Qwen/Qwen3.5-35B-A3B
  hardware: b200_sxm              # replaced per GPU type by the ladder
  backend: vllm
  context_length: 4096            # at least input + output tokens
  workers:
    aggregated: {parallelism: {preset: default}}
    prefill: {parallelism: {preset: default}}
    decode: {parallelism: {preset: default}}

evaluation:
  sla: {ttft_ms: 500, itl_ms: 50}

optimization:
  target: min_gpus
  constraints:
    max_candidate_gpus: 16        # replaced per step by the ladder
    min_goodput_rps: 18           # the agreed SLA-compliant throughput

optimizer:
  max_trials: 16
```

Use an observation window of at least 300 seconds. A shorter run can miss
queue buildup. Check the per-request compliance share and SLA-compliant
throughput as well as aggregate latency means before presenting a GPU count
as sufficient.

### 3. Run the sizing ladder for each GPU type

```bash
python3 "$SKILL_DIR/scripts/size_ladder.py" sizing.yaml \
  --hardware b200_sxm --out-root ./sizing-b200_sxm
```

Use `--hardware` for the GPU type and `--budgets` for the maximum GPU counts
at each step. Do not override `engine.hardware` or
`optimization.constraints.max_candidate_gpus` with `--set`, including through
parent mappings such as `engine` or `optimization.constraints`. The script
rejects those overrides with exit 2 before probing or searching, so its
coverage probe, search steps and report refer to the same hardware and budgets.
Other overrides, such as resource limits and `min_candidate_gpus`, remain available.

The script works around an issue in AISim 0.14.0: one large-budget
`min_gpus` search can miss smaller qualifying configurations. It is fixed on
`main` after 0.14.0 (https://github.com/ai-dynamo/aisim/pull/913). Remove the
script once this skill pins a release that includes the fix. It:

1. validates the config and refuses non-request-rate loads or a missing
   `min_goodput_rps`;
2. probes coverage with a one-request prediction per backend, and stops
   with exit 4 if AISim has no timing data or does not model the
   architecture;
3. runs `aisim recommend` at budgets 1, 2, 4, 8, 16, each in a fresh
   directory. It raises the budget only on genuine infeasibility: every
   evaluated candidate missed the SLA or load constraints, or no shape
   fits the budget. It stops on configuration, data, resource or
   evaluation errors;
4. runs an ascending refinement: each budget from the last failing one +
   1 up to the qualifying GPU count - 1.

It prints a JSON report with every step and an `answer`. Exit codes:

| Exit | Meaning | What to tell the user |
|---|---|---|
| 0 | `answer` holds the smallest qualifying configuration found in the tested search space | GPU count and configuration, with that wording |
| 1 | nothing qualified up to 16 GPUs in the sampled search | not "impossible"; give the closest misses from the not-feasible rows |
| 2 | input or configuration error | fix the input; no sizing conclusion |
| 3 | local resource limit, timeout or evaluation error | no sizing conclusion; retry with a lower memory cap or longer timeout |
| 4 | no AISim coverage for model, GPU type and backend | AISim cannot size this combination |

A step with status `qualified_partial` (AISim exit 3 with a selected
result) means some candidates were resource-limited; say the result covers
only the evaluated candidates.

Read any step's candidates with:

```bash
python3 "$SKILL_DIR/scripts/summarize_recommendation.py" <output_dir> --top 3
```

It shows mean TTFT, TPOT and E2E, completed and SLA-compliant request
counts, goodput and output tokens per second per GPU. It also states how
many rows it left out.

### 4. Keep the fallback disclosures

If the AISim log shows "Loading low-fidelity fallback rows" or "No exact
CLI args template", tell the user the prediction uses substituted
performance data or templates for some operations. `size_ladder.py` and
`headroom_scan.py` save each AISim run's log next to its output directory
(`log_file`) and copy any such lines into `fallback_warnings` in their JSON
report, including for runs that succeeded. Check `fallback_warnings` on the
step or row you report.

### 5. Confirm the compliance share and measure headroom

```bash
python3 "$SKILL_DIR/scripts/headroom_scan.py" \
  ./sizing-b200_sxm/<phase>-b<budget>/recommendations/0001.yaml \
  --min-compliant-share 0.99 --min-goodput-share 0.90 \
  --steps 1.0,1.25,1.5,2.0 --workdir ./headroom-b200_sxm
```

Use the `config_file` from the ladder's `answer`. Set `--min-goodput-share`
to the agreed `min_goodput_rps` divided by the planned request rate: the
template uses 18 / 20 = 0.90. This keeps the agreed floor at 1.0x and scales
it with the offered rate at higher multiples. For each tested load, the
scan reports `pass`, `fail` or `unknown`:

- the SLA is checked with AISim's own aggregate check (all configured
  bounds, fail-closed on missing data);
- the compliance share and the goodput share must both meet the agreed
  numbers;
- it also reports completed and compliant counts, the arrival window and
  the run duration.

Set `traffic.load`, `traffic.stop` and `evaluation.sla` in the prediction
YAML before scanning. `--set` overrides of these sections, their children or
parent mappings are rejected with exit 2 before any prediction; otherwise a
late override could replace a scanned rate or make the verdict use a different
SLA. Overrides for other settings, such as
`--set execution.resources.memory_limit_gb=4`, remain available.

It exits 3 if any load could not be evaluated.

- If the 1.0x row fails, the configuration does not meet the user's
  compliance rule at the planned load. To retry above its GPU count, set
  `optimization.constraints.min_candidate_gpus` to that count + 1, start
  `--budgets` at that minimum, and use `--no-refine` so refinement cannot
  revisit smaller budgets. For a rejected two-GPU answer:

  ```bash
  python3 "$SKILL_DIR/scripts/size_ladder.py" sizing.yaml \
    --hardware b200_sxm --out-root ./sizing-b200_sxm-retry \
    --budgets 3,4,8,16 --no-refine \
    --set optimization.constraints.min_candidate_gpus=3
  ```

  `--budgets` alone sets upper bounds and can select the rejected answer
  again. Check that the new `answer.used_gpus` is above the rejected count
  before running its headroom scan. If it is not, stop and report that the
  retry made no progress. Only report a qualifying replacement after its
  1.0x row passes the same compliance and throughput rules.
- Report `highest_tested_passing_multiplier` as "qualifies at the tested
  loads up to N x the planned rate". It says nothing about loads between
  or beyond the tested points.

### 6. Cost (only with user-supplied prices)

Per GPU type: cost per hour = GPUs x price per GPU-hour. Cost per million
SLA-compliant output tokens = cost per hour / (SLA-compliant output tokens
per second x 3600 / 1,000,000). SLA-compliant output tokens per second is
`sla_compliant_output_tok_s` in the summarizer's `--json` output. Label
it as SLA-compliant output tokens, not total generated tokens.

### 7. Report

One table per GPU type, with these columns:

- smallest qualifying GPU count found, and the configuration;
- mean TTFT, TPOT and E2E at the planned rate;
- completed and SLA-compliant requests;
- highest tested passing multiple;
- cost, if priced.

Then state:

- the results are simulated;
- the count is the smallest qualifying configuration found in the tested
  search space;
- the SLA is checked on means;
- the final choice should be measured on hardware with the same request
  rate and arrival pattern.

## Examples

**User:** "We expect 100 requests per second of 2k in / 256 out on
Qwen3.5-35B-A3B. TTFT must stay under 500 ms and TPOT under 50 ms, 99% of
requests within SLA. How many B200s or GB300s do we need?"

**Agent:**

1. Confirms the inputs and sets `min_goodput_rps: 90` after the user
   agrees.
2. Runs `size_ladder.py` for `b200_sxm` and `gb300`.
3. Runs `headroom_scan.py` with `--min-compliant-share 0.99`.
4. Reports per type: the GPU count found, whether 1.0x passed the 99% rule,
   the highest tested passing multiple, and that all of it is simulated.

Read each result from the saved artifacts. If a type fails the 99% rule at
1.0x, report that failure and do not present its ladder count as sufficient.

**User:** "I own eight B200s. How many do I need to set aside for 100
requests per second of the same workload?"

**Agent:** treats it as sizing with `--budgets 1,2,4,8` and reports the
GPU count found, or that nothing qualified within eight.

**User:** "We run at concurrency 64. How many GPUs?"

**Agent:** asks for the expected request rate, because this version sizes
request-rate workloads only.

## Limits to state to the user

- Simulated, not measured. Accuracy varies by model, GPU type and backend;
  see https://ai-dynamo.org/aisim/e2e-accuracy/.
- Each budget step is a sampled search (`optimizer.max_trials`), so a
  smaller configuration can exist.
- SLA bounds are checked on aggregate means; tails (p90, p99) can exceed
  them. The 0.14.0 recommendation ledger provides mean TTFT, TPOT and E2E;
  the summarizer does not report tail latency from this ledger.
