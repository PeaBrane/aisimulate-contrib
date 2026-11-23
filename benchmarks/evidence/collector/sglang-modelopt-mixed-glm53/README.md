<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang mixed-precision load of GLM-5.3-NVFP4

## Purpose

`nvidia/GLM-5.3-NVFP4` is a ModelOpt `MIXED_PRECISION` export. Its `quantization_config` names one lane per module:

- FP8 for the dense MLP in layers 0 to 2.
- NVFP4 for the routed experts.
- No lane for attention, so attention stays BF16.

The export also declares an FP8 KV cache algorithm.

The SGLang MLA and DSA module collector reads this checkpoint as `"nvfp4"` (`_model_native_gemm_quant` in `python/aisim/collector/sglang/collect_mla_module.py`). `load_model_runner` then asks SGLang for `modelopt_fp4`, and SGLang replaces that request with `modelopt_mixed`. The comment in `_model_native_gemm_quant` holds the reasons. This bundle gives the SGLang source lines of those reasons at each release. It also holds a load probe and a narrow collector run on a Blackwell GPU.

The bundle has three files and the run results:

- `probe_mixed_load.py` loads the model the way the collector does and reports what SGLang built.
- `plan-diff.md` shows that the reader change leaves the SGLang MLA and DSA module plans unchanged.
- This README gives the source table, the commands and the run records.

## What the two runs prove

The runs use SGLang 0.5.21 and SGLang 0.5.23 on one SM100 GPU. Each SGLang version gets two probe runs and one collector run. If all runs pass, they show these facts for each version:

1. The collector reads this checkpoint as `"nvfp4"` and asks SGLang for `modelopt_fp4` (`server_args.quantization`). SGLang resolves that request to `modelopt_mixed` (`model_runner.model_config.quantization`).
2. Each linear module inside `model.layers[i].self_attn` uses `UnquantizedLinearMethod` and has a BF16 weight. This covers the DSA projections and the DSA indexer projections, and the probe requires at least one indexer projection. So the `bfloat16` label of the DSA module rows describes the loaded projections.
3. The model has two layers, and each layer has a dense MLP (`is_layer_sparse` is `False` and the `mlp` is `DeepseekV2MLP`).
4. The KV pool dtype follows the `kv_cache_dtype` that the collector passes, and not the FP8 KV algorithm of the export. The `bfloat16` probe run proves this, because a BF16 pool is not what the export asks for.
5. The collector runs the two `[0, 256, 64, ...]` rows of the `dsa_context_module` plan on this load path and writes rows with both KV dtypes.

## What the two runs do not prove

- The loader builds two layers (`SGLANG_TEST_NUM_LAYERS`, default 2). The model has `first_k_dense_replace` 3, so both layers have a dense MLP. No NVFP4 expert module is built, and the runs say nothing about the NVFP4 MoE path. The DSA module rows do not time the MoE.
- The loader uses dummy weights (`SGLANG_LOAD_FORMAT`, default `dummy`). The runs do not load real checkpoint tensors or scales, and they say nothing about accuracy.
- The runs use SM100 only. They say nothing about SM103, SM110 or SM120. On SM90 the plan holds no case for this checkpoint (`plan-diff.md`), because the collector collects no Marlin-backed NVFP4.
- The collector run covers points of one batch size: 24 points at 0.5.21 and 42 points in tree B. It is not a full collection, and it does not compare the timings with serving.
- The probe sizes the KV pool from all 12 points of its plan row. `run_mla_module` first refuses the points over the prefill chunk, so its pool can be smaller. The pool size changes no quantization method and no KV dtype.
- The runs exercise no other checkpoint. The one reader change in the packaged models (`nvidia/Kimi-K2.5-NVFP4`) is not a checkpoint of this collector.
- The SGLang 0.5.23 runs use tree B, a throwaway merge of the branch of the pull request that added this directory onto pull request #777. They do not test a merged state of the repository.

## SGLang source table

Each row is one premise of the comment in `_model_native_gemm_quant`. Paths are under `python/sglang/srt/` in the release tarballs `sglang-0.5.21` and `sglang-0.5.23`. The line numbers come from `awk NR` on those files.

| # | Premise | File and symbol | v0.5.21 | v0.5.23 |
|---|---|---|---|---|
| 1 | SGLang reads one quantization section per checkpoint, in this order: `quantization_config`, `compression_config`, the `hf_quant_config.json` sidecar. Only without the first two does it read the sidecar. | `configs/model_config.py`, `ModelConfig._parse_quant_hf_config` | 1539-1558 (order), 1627-1634 (sidecar read) | 1727-1746, 1815-1822 |
| 2 | Without a `quant_method`, or with the value `modelopt`, a `quantization_config` is ModelOpt. | `configs/model_config.py`, `ModelConfig._parse_quant_hf_config` | 1543-1553 | 1731-1741 |
| 3 | SGLang parses the sidecar as ModelOpt without that gate. | `configs/model_config.py`, `ModelConfig._parse_quant_hf_config` | 1631-1633 | 1819-1821 |
| 4 | A `quant_algo` of exactly `MIXED_PRECISION` names one lane per module in `quantized_layers`. An `NVFP4` or `W4A16_NVFP4` lane gives `modelopt_mixed`, and otherwise the result is `w4afp8`. The parser has no architecture exception. | `configs/model_config.py`, `ModelConfig._parse_modelopt_quant_config` | 1650-1677 | 1838-1865 |
| 4a | A pure ModelOpt `FP4` or `NVFP4` algorithm gives `modelopt_fp4`, and `FP8` gives `modelopt_fp8`. | `configs/model_config.py`, `ModelConfig._parse_modelopt_quant_config` | 1678-1681 | 1866-1869 |
| 5 | `_verify_quantization` calls `_parse_quant_hf_config`. | `configs/model_config.py`, `ModelConfig._verify_quantization` | 1812, 1871 | 2000, 2062 |
| 6 | If the parsed method is `modelopt_mixed`, SGLang replaces an explicit `modelopt_fp4` with `modelopt_mixed`. | `configs/model_config.py`, override loop in `ModelConfig._verify_quantization`, and `layers/quantization/modelopt_quant.py`, `ModelOptMixedPrecisionConfig.override_quantization_method` | model_config 1902-1912, modelopt_quant 820-826 | model_config 2093-2103, modelopt_quant 829-835 |
| 6a | The ModelOpt override helper returns nothing for a parsed `modelopt_mixed` or `w4afp8` with a `modelopt_fp4` request, so the mixed override is the one that applies. | `layers/quantization/base_config.py`, `_modelopt_override_quantization_method` | 203-235 | 222-254 |
| 7 | If the parsed method is `w4afp8`, `_verify_quantization` rejects `modelopt_fp4`. | `configs/model_config.py`, `REQUANTIZATION_METHODS`, the compatibility map and `ModelConfig._verify_quantization` | 413, 1853-1858, 1917-1957 (raise at 1951-1957) | 545, 2042-2047, 2108-2148 (raise at 2142-2148) |
| 8 | `W4AFp8Config` gives a linear layer that is not ignored `Fp8LinearMethod` with `weight_block_size` `[128, 128]`. | `layers/quantization/w4afp8.py`, `W4AFp8Config` and `W4AFp8Config.get_quant_method` | 35, 58, 93-102 | 35, 58, 93-102 |
| 8a | If `weight_block_size` is not `None`, `Fp8LinearMethod` is block-quantized. | `layers/quantization/fp8.py`, `Fp8LinearMethod.__init__` | 497, 510-511 | 508, 521-522 |
| 9 | A linear module without a lane gets `UnquantizedLinearMethod`. | `layers/quantization/modelopt_quant.py`, `ModelOptMixedPrecisionConfig._resolve_quant_algo` and `get_quant_method` | 969-999, 1027-1054 (fallback at 1054) | 978-1008, 1036-1063 (fallback at 1063) |
| 10 | `kv_cache_quant_algo` attaches `ModelOptFp8KVCacheMethod` to the attention layer only. | `layers/quantization/modelopt_quant.py`, `ModelOptMixedPrecisionConfig.get_quant_method` | 1067-1068 | 1076-1077 |
| 11 | The KV dtype follows an explicit `kv_cache_dtype`. SGLang reads `kv_cache_quant_algo` only for `"auto"`. | `mem_cache/kv_cache_dtype.py`, `configure_kv_cache_dtype` | 23-79 (`auto` branch 38-48) | 34-90 (`auto` branch 49-59) |
| 12 | SGLang reroutes NemotronH later, after the parse. | `arg_groups/model_overrides/nemotron_h.py`, `_nemotron_h_overrides` | 22-55 | 22-55 |

The repository premises of the comment are these:

- `load_model_runner` maps `"nvfp4"` to `modelopt_fp4`.
- `load_model_runner` passes `kv_cache_dtype` through `SGLANG_KV_DTYPE`. That table holds only `bfloat16` and `fp8_e4m3`, never `auto`.
- All 31 packaged `*_hf_quant_config.json` files have no `quant_method` (30 files) or `modelopt` (1 file) in their `quantization` section.

## Reader result before and after

The reader snapshots run `_model_native_gemm_quant` over all 97 packaged checkpoints, before the reader change (`51607c9c`) and after it.

| Result | Before | After |
|---|---:|---:|
| `None` | 44 | 43 |
| `"fp8_block"` | 28 | 28 |
| `"nvfp4"` | 22 | 23 |
| `ValueError` (a `MIXED_PRECISION` export with attention lanes) | 3 | 3 |

One checkpoint changes. `nvidia/Kimi-K2.5-NVFP4` goes from `None` to `"nvfp4"`. Its `config.json` holds a ModelOpt NVFP4 `quantization_config` and it has no sidecar. SGLang parses that section as `modelopt_fp4` (premise 4a). The three errors are `nvidia/MiniMax-M3-NVFP4`, `nvidia/Qwen3.6-27B-NVFP4` and `nvidia/Qwen3.6-35B-A3B-NVFP4`, with the same error before and after.

The SGLang MLA and DSA module collector declares 18 checkpoints (`collector.case_generator.get_mla_module_model_specs(backend="sglang", apply_model_filter=False)`). None of them changes, and `nvidia/Kimi-K2.5-NVFP4` is not one of them. `nvidia/GLM-5.3-NVFP4` reads as `"nvfp4"` before and after.

## Plan diff

`plan-diff.md` compares the SGLang plans of `dsa_context_module`, `dsa_generation_module`, `mla_context_module` and `mla_generation_module` at `51607c9c` and at `2df535ed`. It covers SM90 and SM100, the default plan and the `nvidia/GLM-5.3-NVFP4` plan. No case count, skip count, plan hash or plan table differs. The SGLang registry has no MLA module ops, so `collect.py` refuses those two ops at both commits.

## Commands for the GPU runs

### Images and trees

| SGLang | Image | Tree |
|---|---|---|
| 0.5.21 | `lmsysorg/sglang:v0.5.21@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469` | Tree A: the branch of the pull request that added this directory |
| 0.5.23 | `lmsysorg/sglang:v0.5.23@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f` | Tree B: the branch of the pull request that added this directory, merged onto the head of pull request #777 |

The 0.5.21 image is the pin of `python/aisim/collector/framework_manifest.yaml` on the branch of the pull request that added this directory:

```yaml
    default:
      version: "0.5.21"
      images:
        default: "lmsysorg/sglang:v0.5.21@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469"
```

Only the open pull request #777 (branch `feat/ops-upgrade-202611`, not merged) names the 0.5.23 image in its manifest. `collect.py` refuses any SGLang version other than its pin, so the 0.5.23 runs need tree B. Make tree B in a throwaway directory. Check out the head of pull request #777 there, and merge the branch of the pull request that added this directory into it. Record the two commits in the run record. Do not push tree B.

### Steps per SGLang version

Run these steps inside the image of that version, on one SM100 GPU. Replace `<tree>` with the absolute path of tree A for 0.5.21 and of tree B for 0.5.23. Replace `<ver>` with `0.5.21` or `0.5.23`.

1. Install the two packages that the collector needs to write its tables.

   ```bash
   pip install pyarrow pandas
   ```

2. Run the probe twice, once for each KV dtype, each in its own process. Run it from an empty directory outside the repository.

   ```bash
   export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=<tree>/python/aisim
   PROBE=<tree>/benchmarks/evidence/collector/sglang-modelopt-mixed-glm53/probe_mixed_load.py
   python3 $PROBE --model nvidia/GLM-5.3-NVFP4 --kv-cache-dtype fp8 --output probe-<ver>-fp8.json
   python3 $PROBE --model nvidia/GLM-5.3-NVFP4 --kv-cache-dtype bfloat16 --output probe-<ver>-bfloat16.json
   ```

3. Run the narrow collection from another empty directory outside the repository. Use a local file system for this directory. The collector checkpoint needs an atomic rename that does not replace a file, and Lustre does not have it.

   ```bash
   export HF_HUB_OFFLINE=1 PYTHONPATH=<tree>/python/aisim
   python3 <tree>/python/aisim/collector/collect.py --backend sglang --sm 100 \
     --model-path nvidia/GLM-5.3-NVFP4 --ops dsa_context_module --case-filter "[0, 256, 64, " --keep-csv
   cp dsa_context_module_perf.txt rows-<ver>.csv
   ```

4. Copy `probe-<ver>-fp8.json`, `probe-<ver>-bfloat16.json` and `rows-<ver>.csv` into this directory, and fill the run record.

### What the probe does

The probe takes the `dsa_context_module` plan row that starts with `[0, 256, 64, '<kv-cache-dtype>', ` for the model on this SM. It calls `collector.sglang.collect_mla_module.load_model_runner` with the arguments that `run_mla_module` passes for that row. It writes one JSON object with these keys, among others:

| Key | Content |
|---|---|
| `sglang_version` | The installed `sglang` package version |
| `gpu_name`, `capability` | The device name and the compute capability |
| `checkpoint_quant` | The value of `_model_native_gemm_quant` for the model |
| `requested_quantization` | `server_args.quantization`, the request of the collector |
| `resolved_quantization` | `model_runner.model_config.quantization` |
| `num_layers` | The number of layers in `model_runner.model.model.layers` |
| `load_format` | `server_args.load_format` |
| `layers` | Per layer: class, `is_layer_sparse`, `mlp` class, indexer class, and the counts of attention and indexer projections |
| `linear_modules` | Per `LinearBase` module: qualified name, quantization method class, weight dtype, and whether it is an attention projection and an indexer projection |
| `attention_projection_count`, `indexer_projection_count` | The totals over all layers |
| `kv_cache_dtype` | The dtype of the KV pool (`token_to_kv_pool.dtype`), with the storage dtype under `kv_pool` |
| `failures` | The failed conditions, empty on success |

The probe writes the JSON and then exits with code 1 in each of these cases:

- `checkpoint_quant` is not `"nvfp4"`.
- `server_args.quantization` is missing, or it is not `modelopt_fp4`.
- `resolved_quantization` is not `modelopt_mixed`.
- The model has a number of layers other than 2.
- A layer has `is_layer_sparse` other than `False`, or an `mlp` other than `DeepseekV2MLP`.
- It finds no attention projection, or no indexer projection.
- An attention projection has a method other than `UnquantizedLinearMethod`, or a weight dtype other than `bfloat16`.
- A `LinearBase` module has no `quant_method` attribute.
- The KV pool dtype does not correspond to `--kv-cache-dtype` through the collector's `SGLANG_KV_DTYPE` (`fp8` gives `float8_e4m3fn`, `bfloat16` gives `bfloat16`).
- The collector's KV dtype has no known torch dtype name. The probe maps the `SGLANG_KV_DTYPE` value for `--kv-cache-dtype` to a torch dtype name, and the map has no entry for it.

The docstring of the probe gives the module tree that it reads, and the SGLang source that makes that tree. It also gives the source that proves an indexer in both loaded layers. The load format is recorded, not asserted.

The probe does not set the environment that `collect.py` sets for SGLang before it starts workers: `FLASHINFER_LOG_LEVEL=ERROR`, `SGLANG_OPT_DEEPGEMM_HC_PRENORM=1`, `SGLANG_OPT_USE_TILELANG_MHC_PRE=1`, `COLLECTOR_MEASURE_POWER` and `PYTHONWARNINGS`. It also imports `collector.sglang.collect_mla_module`, where the collector worker imports the top-level module `collect_mla_module`. These differences change no quantization method and no KV dtype.

### What the narrow collection selects

A CPU plan of commit `2df535ed` with `COLLECTOR_MODEL_PATH=nvidia/GLM-5.3-NVFP4` on SM100 gives 90 `dsa_context_module` rows with 4900 points. The filter `"[0, 256, 64, "` keeps 2 rows with 24 points:

| Row | `kv_cache_dtype` | Compute dtype | `gemm_type` | Pinned DSA backend | Points |
|---|---|---|---|---|---|
| 1 | `bfloat16` | `bfloat16` | `bfloat16` | none (runtime default) | 12 |
| 2 | `fp8` | `bfloat16` | `bfloat16` | `trtllm` | 12 |

Both rows have 64 heads and target TP 1. Their points are batch size 256, sequence lengths 16, 32 and 64, and prefix lengths 0, 512, 1024 and 2048. So a complete run writes 24 rows, 12 with KV dtype `fp8` and 12 with `bfloat16`. When the device refuses a point, the point becomes a `FIXME(kernel-limit)` failure record instead of a row. `--plan-only` does not apply `--case-filter`, so these numbers come from the plan rows that the getter returns.

## Run records

Pull request #983 added this directory. Tree A is commit `5df15da0` of the branch of that pull request. Later commits of that branch change no executable line under `python/aisim/collector/`. They change only comments and docstrings there. The plan hashes of `plan-diff.md` hold at the final content.

Each SGLang version has one record. The six result files are in this directory.

### SGLang 0.5.21

| Field | Value |
|---|---|
| Date | 2026-11-20 (UTC) |
| GPU and node type | One NVIDIA B200, compute capability 10.0 (SM100), 183359 MiB, driver 580.82.07, on a Slurm node with 8 GPUs |
| Image digest seen in the container | The job imported the image by its digest (`sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`). The container reports `sglang` 0.5.21 and `torch` 2.13.0+cu130. |
| Tree A commit | `5df15da09bb69a6396e830d182907f1931f8f84f` |
| `probe-0.5.21-fp8.json` exit code | 0 |
| `probe-0.5.21-fp8.json`: `resolved_quantization`, `kv_cache_dtype`, attention projections | `modelopt_mixed`, `float8_e4m3fn` (stored as `uint8`), 12 projections (4 of the indexer), all `UnquantizedLinearMethod` with `bfloat16` weights |
| `probe-0.5.21-bfloat16.json` exit code | 0 |
| `probe-0.5.21-bfloat16.json`: `resolved_quantization`, `kv_cache_dtype`, attention projections | `modelopt_mixed`, `bfloat16`, 12 projections (4 of the indexer), all `UnquantizedLinearMethod` with `bfloat16` weights |
| Method classes of the dense MLP linear modules | `ModelOptFp8LinearMethod` with `float8_e4m3fn` weights for `mlp.gate_up_proj` and `mlp.down_proj` of both layers |
| Collector exit code | 0 |
| `rows-0.5.21.csv`: rows per `kv_cache_dtype` | 24 rows: 12 `fp8` and 12 `bfloat16`. Each row has model `nvidia/GLM-5.3-NVFP4`, `gemm_type` `bfloat16` and version `0.5.21`. |
| Failure records (count and classification) | 0 (`collection_summary_sglang.json`: `total_errors` 0) |
| Notes | A first job ran the collector from a directory on a Lustre file system. The probes passed, and the collector stopped before the first case with `RuntimeError: Atomic no-replace rename is unsupported by this filesystem`. This record is the second job, which ran the collector from a node-local directory. No code changed between the two jobs. |

### SGLang 0.5.23

| Field | Value |
|---|---|
| Date | 2026-11-20 (UTC) |
| GPU and node type | One NVIDIA B200, compute capability 10.0 (SM100), 183359 MiB, driver 580.82.07, on a Slurm node with 8 GPUs |
| Image digest seen in the container | The job imported the image by its digest (`sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f`). The container reports `sglang` 0.5.23 and `torch` 2.14.1+cu130. |
| Tree B: head commit of pull request #777 | `6f3ac1b073a011c38eed05dd7572184e4157055c` |
| Tree B: merged commit of the branch of the pull request that added this directory | `5df15da09bb69a6396e830d182907f1931f8f84f` (the merge had no conflict) |
| `probe-0.5.23-fp8.json` exit code | 0 |
| `probe-0.5.23-fp8.json`: `resolved_quantization`, `kv_cache_dtype`, attention projections | `modelopt_mixed`, `float8_e4m3fn` (stored as `uint8`), 12 projections (4 of the indexer), all `UnquantizedLinearMethod` with `bfloat16` weights |
| `probe-0.5.23-bfloat16.json` exit code | 0 |
| `probe-0.5.23-bfloat16.json`: `resolved_quantization`, `kv_cache_dtype`, attention projections | `modelopt_mixed`, `bfloat16`, 12 projections (4 of the indexer), all `UnquantizedLinearMethod` with `bfloat16` weights |
| Method classes of the dense MLP linear modules | `ModelOptFp8LinearMethod` with `float8_e4m3fn` weights for `mlp.gate_up_proj` and `mlp.down_proj` of both layers |
| Collector exit code | 0 |
| `rows-0.5.23.csv`: rows per `kv_cache_dtype` | 42 rows: 21 `fp8` and 21 `bfloat16`. Each row has model `nvidia/GLM-5.3-NVFP4`, `gemm_type` `bfloat16` and version `0.5.23`. |
| Failure records (count and classification) | 0 (`collection_summary_sglang.json`: `total_errors` 0) |
| Notes | A first job ran the collector from a directory on a Lustre file system. The probes passed, and the collector stopped before the first case with `RuntimeError: Atomic no-replace rename is unsupported by this filesystem`. This record is the second job, which ran the collector from a node-local directory. No code changed between the two jobs. Pull request #777 adds the prefix lengths 1, 1536 and 3072 to the DSA plan. So each of the two rows has 21 points in tree B, and the run writes 42 rows. |
