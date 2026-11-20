<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Plan diff: base 51607c9c against commit 2df535ed

The branch changes how `_model_native_gemm_quant` reads the checkpoint quantization. This diff shows that the change leaves every SGLang MLA and DSA module plan unchanged.

Result: no difference. For each SM, op and plan, the `plans` block of `--plan-only` and the `--plan-table` CSV are identical on both sides.

## Method

The base side is a `git worktree` of `51607c9c`. The head side is commit `2df535ed`. Its executable collector code is the same as in tree A and in the final branch. Each side runs from an empty directory outside the repository, with the repository virtual environment and `PYTHONPATH=<tree>/python/aisim`. No run needs a GPU.

```bash
PYTHONPATH=<tree>/python/aisim python <tree>/python/aisim/collector/collect.py \
  --backend sglang --sm <90|100> --ops <op> --plan-only --plan-table <dir> <plan>
```

- `<op>` is one of `dsa_context_module`, `dsa_generation_module`, `mla_context_module` and `mla_generation_module`, one op per call.
- `<plan>` is `--model-cases-full` for the default plan and `--model-path nvidia/GLM-5.3-NVFP4` for the GLM-5.3 plan.

`--plan-only` refuses a call without a model selector, so the default plan is the full model plan (`--model-cases-full`). One call with all four ops fails at both commits, because the MLA module ops are not in the SGLang case plan. So each op has its own call. The comparison takes the `plans` block of the JSON output without the `table` path, and compares the CSV files byte for byte.

## MLA module ops

`collector/sglang/registry.py` registers no `mla_context_module` and no `mla_generation_module` entry. At both commits, for both SMs and both plans, `collect.py` exits with code 3 and the same message:

```text
collect.py: error: Requested ops are not present in the collector v2 case plan: mla_context_module
collect.py: error: Requested ops are not present in the collector v2 case plan: mla_generation_module
```

## DSA module ops

The skip entries are the `collector/skips.py` entries that `--plan-only` reports under `removed`. The entry names in the table are these:

| Region | `collector/skips.py` entry |
|---|---|
| `own_gemm=False` | `{'backend': 'sglang', 'op': [DSA module ops], 'own_gemm': False}` |
| `seq_len=1` | `{'backend': 'sglang', 'op': ['dsa_context_module', 'dsa_context_module_skip_indexer'], 'seq_len': 1}` |
| `dtype=nvfp4` | `{'dtype': ['nvfp4', 'w4a8_mxfp4_mxfp8'], 'sm': [80, 86, 89, 90]}` |
| `gemm_type=nvfp4` | `{'backend': 'sglang', 'op': [DSA module ops], 'gemm_type': 'nvfp4'}` |
| `checkpoint_quant=nvfp4` | `{'backend': 'sglang', 'op': [DSA module ops], 'checkpoint_quant': 'nvfp4', 'sm': [80, 86, 89, 90]}` |

`[DSA module ops]` is `dsa_context_module`, `dsa_generation_module` and their two `_skip_indexer` ops.

The counts are the same at base and head, so each cell gives one value for both sides.

| SM | Plan | Op | Cases | Rows | Skip counts (base = head) | `plan_hash` (base = head) | `diff` of the tables |
|---|---|---|---:|---:|---|---|---|
| 90 | default | `dsa_context_module` | 14590 | 270 | `own_gemm=False` 5480, `seq_len=1` 2150, `dtype=nvfp4` 11110 | `sha256:2bad01665591f88975f50fed666e6bf6523c3f356405b331b44ffbee11f3364f` | identical (14590 cases) |
| 90 | default | `dsa_generation_module` | 5700 | 30 | `own_gemm=False` 1840, `dtype=nvfp4` 3770 | `sha256:f264767f9abdfcb76e51bee4a2201270ef4b69592578efb3a19d314aa0236365` | identical (5700 cases) |
| 90 | GLM-5.3 | `dsa_context_module` | 0 | 0 | `checkpoint_quant=nvfp4` 5630, `own_gemm=False` 5630, `dtype=nvfp4` 5630 | `sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | identical (header only) |
| 90 | GLM-5.3 | `dsa_generation_module` | 0 | 0 | `checkpoint_quant=nvfp4` 1930, `own_gemm=False` 1930, `dtype=nvfp4` 1930 | `sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | identical (header only) |
| 100 | default | `dsa_context_module` | 14590 | 270 | `own_gemm=False` 11110, `seq_len=1` 2150, `gemm_type=nvfp4` 5480 | `sha256:4761c4b2433d32814a95d47be17591bff2b6e699c71ffeae4c54c757bd4b4e49` | identical (14590 cases) |
| 100 | default | `dsa_generation_module` | 5700 | 30 | `own_gemm=False` 3770, `gemm_type=nvfp4` 1840 | `sha256:b0a3bff85967a6b9ec18e879223281f789e8ba5180e1943ade8cf72d929ce9ef` | identical (5700 cases) |
| 100 | GLM-5.3 | `dsa_context_module` | 4900 | 90 | `seq_len=1` 730, `own_gemm=False` 11260 | `sha256:c1a33066614b64df21530d9f615ec18e2cb57fa1533ab1581e8f52788caa6d4d` | identical (4900 cases) |
| 100 | GLM-5.3 | `dsa_generation_module` | 1930 | 10 | `own_gemm=False` 3860 | `sha256:152083a152b91938d8e20174de7a68786c15a1264901bbe0171c17c72e6c0441` | identical (1930 cases) |

On SM90 the GLM-5.3 plan holds no case. The checkpoint is NVFP4, and the collector collects no Marlin-backed NVFP4. The empty plan has the hash of empty input.

The case counts per dtype (`kv_cache_dtype/compute_dtype/gemm_type`) are also the same on both sides:

| SM | Plan | Op | Cases per dtype |
|---|---|---|---|
| 90 and 100 | default | `dsa_context_module` | `bfloat16/bfloat16/fp8_block` 4845, `fp8/bfloat16/fp8_block` 4845, `bfloat16/bfloat16/bfloat16` 2450, `fp8/bfloat16/bfloat16` 2450 |
| 90 and 100 | default | `dsa_generation_module` | `bfloat16/bfloat16/fp8_block` 1885, `fp8/bfloat16/fp8_block` 1885, `bfloat16/bfloat16/bfloat16` 965, `fp8/bfloat16/bfloat16` 965 |
| 100 | GLM-5.3 | `dsa_context_module` | `bfloat16/bfloat16/bfloat16` 2450, `fp8/bfloat16/bfloat16` 2450 |
| 100 | GLM-5.3 | `dsa_generation_module` | `bfloat16/bfloat16/bfloat16` 965, `fp8/bfloat16/bfloat16` 965 |
