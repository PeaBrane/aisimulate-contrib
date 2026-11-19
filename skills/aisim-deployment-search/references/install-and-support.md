<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Install and coverage

## Fresh install

```bash
python3 -m venv .venv && . .venv/bin/activate
python3 -m pip install "aisim==0.14.0"
aisim --version          # expect: 0.14.0
```

AISim supports Python 3.11 to 3.13. If no wheel exists for the platform,
pip builds from source, which needs a Rust toolchain and takes about four
minutes.

## Migrating from `aisimulate` or `aiconfigurator`

Both are former names of AISim. If `aiconfigurator` or `aiconfigurator-core` is
installed, uninstall it too (`python3 -m pip uninstall -y aiconfigurator aiconfigurator-core`).

`aisim` and `aisimulate` install the same Python packages. Remove the old
distribution first, then force-reinstall so shared files are restored:

```bash
python3 -m pip uninstall -y aisimulate
python3 -m pip install aisim==0.14.0
python3 -m pip install --force-reinstall --no-deps aisim==0.14.0
python3 -m pip check
```

## Dynamo stack

The default `--stack engine` needs nothing else. `--stack dynamo` (Dynamo
router and Planner behavior) needs a Dynamo release whose declared
dependency is `aisim`, in the same environment. Use the same stack for
`recommend` and the follow-up `predict`.

## Is my model and GPU covered?

The quickest check is a one-request prediction. It takes a few seconds:

```yaml
# probe.yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3.5-35B-A3B
  hardware: b200_sxm
  backend: vllm
  workers:
    aggregated: {}
```

```bash
aisim predict --config probe.yaml --output-dir ./aisim-probe
```

What the failures look like in 0.14.0. Missing coverage and a shape that
does not fit are different conclusions; do not report one as the other.

| Situation | Exit | Message starts with |
|---|---|---|
| No timing data for the GPU and backend (for example `l4` with vLLM), or a misspelled GPU ID | 1 | `aisim predict failed: ValueError: no timing data for <model> with <backend> ... on <gpu>` |
| Model architecture not modeled (for example `GraniteMoeHybridForCausalLM`) | 1 | `... architecture <Arch> (<model>) is not supported by the op-level model; supported architectures: ...` |
| NVFP4 checkpoint on a GPU without FP4 tensor cores (Hopper and older) | search: 1, every candidate `failed` | `... quant mode 'nvfp4' needs 'fp4_tc_flops', which this system's YAML does not ...` (a one-GPU probe may report the fit error below first) |
| Model does not fit the single-GPU probe (a fit problem, not missing coverage) | 1 | `... no KV budget: non-KV memory ... meets/exceeds the KV-cache memory limit` (probe again with more GPUs) |
| In a search: no shape is viable for the model, GPU and budget (missing coverage or too small a budget; the message does not say which) | 2 | `no deployment_mode has a viable parallel config` |

The published pages give the wider picture:

- Forward Pass Engine support matrix: https://ai-dynamo.org/aisim/fpe-support-matrix/
- Feature support matrix: https://ai-dynamo.org/aisim/feature-support-matrix/
- End-to-end accuracy against measured hardware: https://ai-dynamo.org/aisim/e2e-accuracy/
