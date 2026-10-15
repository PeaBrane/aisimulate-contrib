// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Modified CPU contract adaptation of vLLM; copyright contributors to the vLLM project.
// Apache-2.0, vllm-project/vllm@ac9126e58aa7bbab1856ba6593ba4d5003fea516.
// Original paths: vllm/models/kimi_k3/nvidia/{mla.py,model.py,dspark_mla.py}.
// https://github.com/vllm-project/vllm/tree/ac9126e58aa7bbab1856ba6593ba4d5003fea516

//! Full native MLA modules. No legacy, sibling-runtime or causal-width reuse.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::axis_curve::LeafAxisCurve;
use super::parquet_loader::PerfReader;
use super::perf_interp::LeafValue;
use crate::common::error::AicError;

pub const BASENAME: &str = "mla_native_module_perf.parquet";
pub const RUNTIME_REVISION: &str = "ac9126e58aa7bbab1856ba6593ba4d5003fea516";
const TARGET_CONFIG_SHA256: &str =
    "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213";
const DRAFT_CONFIG_SHA256: &str =
    "5a3c2f4f91c965ed93b14de5f12a4e9c17fd98d8c99916ed2deb26ce8702f970";

/// Pinned Kimi module includes input projections, norms, attention, gate and
/// output projection. Target includes its output collective; draft does not.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeMlaScope {
    pub draft: bool,
    /// Zero means context phase; positive widths are exact decode buckets.
    pub query_tokens: u32,
    pub tensor_parallel_size: u32,
    /// Difference from the engine's one-query inclusive end. Steady draft
    /// uses a full-acceptance upper bound; startup already carries its end.
    pub kv_tokens_offset: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeMlaKey {
    pub draft: bool,
    pub causal: bool,
    pub is_context: bool,
    pub query_tokens: u32,
    pub sequence_count: u32,
    pub num_heads: u32,
    pub tensor_parallel_size: u32,
    pub kv_cache_dtype: String,
    pub gemm_quant_mode: String,
}

pub struct NativeMlaTable {
    path: Option<PathBuf>,
    curves: OnceLock<Result<BTreeMap<NativeMlaKey, LeafAxisCurve>, String>>,
}

impl NativeMlaTable {
    pub fn with_sources(root: &Path, resolver: &SourceResolver) -> Result<Self, AicError> {
        let primary = resolver
            .prioritized_sources_for(BASENAME, root)?
            .into_iter()
            .find(|s| s.channel == "primary");
        if primary
            .as_ref()
            .is_some_and(|s| s.source.kernel_sources().is_some())
        {
            return Err(invalid("native MLA requires an unfiltered primary table"));
        }
        Ok(Self {
            path: primary.map(|s| s.source.0),
            curves: OnceLock::new(),
        })
    }

    pub fn query(
        &self,
        key: &NativeMlaKey,
        total_kv: u32,
        sol: &dyn Fn(f64) -> f64,
    ) -> Result<LeafValue, AicError> {
        let curves = self
            .curves
            .get_or_init(|| self.load().map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| invalid(e))?;
        let curve = curves
            .get(key)
            .ok_or_else(|| missing(format!("native MLA data missing for {key:?}")))?;
        let first = curve.iter().next().map(|(kv, _)| kv);
        let last = curve.iter().next_back().map(|(kv, _)| kv);
        if first.is_none_or(|first| total_kv < first) || last.is_none_or(|last| total_kv > last) {
            return Err(missing(format!(
                "native MLA total_kv_tokens={total_kv} outside measured range {first:?}..={last:?} for {key:?}"
            )));
        }
        curve.query(f64::from(total_kv), sol)
    }

    fn load(&self) -> Result<BTreeMap<NativeMlaKey, LeafAxisCurve>, AicError> {
        let Some(path) = &self.path else {
            return Ok(BTreeMap::new());
        };
        if !path.try_exists().map_err(invalid)? {
            return Ok(BTreeMap::new());
        }
        let reader = PerfReader::open(path)?;
        let identity = reader.col("identity")?;
        let kv = reader.col("total_kv_tokens")?;
        let latency = reader.col("latency")?;
        let revision = reader.col("runtime_revision")?;
        let scope = reader.col("measurement_scope")?;
        let source = reader.col("kernel_source")?;
        let target_config = reader.col("target_config_sha256")?;
        let draft_config = reader.col("draft_config_sha256")?;
        let image = reader.col("image_digest")?;
        let samples = reader.col("sample_count")?;
        let ranks = reader.col("rank_count")?;
        let graph = reader.col("use_cuda_graph")?;
        let prefill_quant = reader.col("prefill_query_quantization")?;
        let prefill_backend = reader.col("prefill_backend")?;
        let mut image_identity: Option<String> = None;
        let mut raw: BTreeMap<NativeMlaKey, BTreeMap<u32, LeafValue>> = BTreeMap::new();
        for row in reader.rows()? {
            let row = row?;
            let digest = row.str(image)?;
            if row.str_owned(revision)? != RUNTIME_REVISION
                || row.str_owned(scope)? != "kimi_native_mla_inclusive"
                || row.str_owned(source)?.is_empty()
                || row.str_owned(target_config)? != TARGET_CONFIG_SHA256
                || row.str_owned(draft_config)? != DRAFT_CONFIG_SHA256
                || !digest.starts_with("sha256:")
                || digest.len() != 71
                || !digest[7..]
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                || row.u32(samples)? == 0
                || row.u32(ranks)? != 8
                || row.bool_strict(graph)?
                || !row.bool_strict(prefill_quant)?
                || row.str(prefill_backend)? != "TRTLLM_RAGGED"
            {
                return Err(invalid(
                    "native MLA runtime, scope or kernel identity mismatch",
                ));
            }
            if image_identity
                .as_ref()
                .is_some_and(|previous| previous != digest)
            {
                return Err(invalid("mixed native MLA runtime images"));
            }
            image_identity = Some(digest.to_owned());
            let key: NativeMlaKey = serde_json::from_str(&row.str_owned(identity)?)
                .map_err(|e| invalid(e.to_string()))?;
            let kernel: serde_json::Value =
                serde_json::from_str(row.str(source)?).map_err(|e| invalid(e.to_string()))?;
            let expected_attention = if key.is_context {
                "FLASHINFER_MLA/vllm.v1.attention.backends.mla.prefill.trtllm_ragged.TrtllmRaggedPrefillBackend"
            } else {
                "FLASHINFER_MLA"
            };
            if kernel.get("attention").and_then(serde_json::Value::as_str)
                != Some(expected_attention)
                || kernel
                    .get("gemm_reduce_scatter")
                    .and_then(serde_json::Value::as_bool)
                    .is_none()
            {
                return Err(invalid("native MLA attention backend identity mismatch"));
            }
            let total = row.u32(kv)?;
            let value = row.f64(latency)?;
            if key.causal == key.draft
                || key.query_tokens == 0
                || key.sequence_count == 0
                || key.tensor_parallel_size == 0
                || key.num_heads == 0
                || key.tensor_parallel_size != 8
                || key.num_heads != if key.draft { 8 } else { 12 }
                || key.kv_cache_dtype != "fp8"
                || key.gemm_quant_mode != "bfloat16"
                || (key.draft && key.is_context)
                || total < key.query_tokens
                || !value.is_finite()
                || value <= 0.0
            {
                return Err(invalid("invalid native MLA physical identity or latency"));
            }
            if raw
                .entry(key)
                .or_default()
                .insert(total, LeafValue::with_power(value, 0.0))
                .is_some()
            {
                return Err(invalid("duplicate native MLA physical timing key"));
            }
        }
        Ok(raw
            .into_iter()
            .map(|(key, points)| (key, LeafAxisCurve::from_map("total_kv_tokens", points)))
            .collect())
    }
}

fn invalid(message: impl std::fmt::Display) -> AicError {
    AicError::InvalidPerfData(message.to_string())
}

fn missing(message: impl std::fmt::Display) -> AicError {
    AicError::PerfDatabase(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

    #[test]
    fn native_mla_loader_requires_pinned_precision_configs_and_all_ranks() {
        let identity = r#"{"draft":true,"causal":false,"is_context":false,"query_tokens":2,"sequence_count":1,"num_heads":8,"tensor_parallel_size":8,"kv_cache_dtype":"fp8","gemm_quant_mode":"bfloat16"}"#;
        let valid_kernel = r#"{"attention":"FLASHINFER_MLA","gemm_reduce_scatter":false}"#;
        let other_kernel = r#"{"attention":"FLASH_ATTN","gemm_reduce_scatter":false}"#;
        for (prefill_quant, prefill_backend, target_hash, ranks, kernel, valid) in [
            (
                true,
                "TRTLLM_RAGGED",
                TARGET_CONFIG_SHA256,
                8,
                valid_kernel,
                true,
            ),
            (
                true,
                "TRTLLM_RAGGED",
                TARGET_CONFIG_SHA256,
                8,
                other_kernel,
                false,
            ),
            (
                false,
                "TRTLLM_RAGGED",
                TARGET_CONFIG_SHA256,
                8,
                valid_kernel,
                false,
            ),
            (
                true,
                "FLASH_ATTN",
                TARGET_CONFIG_SHA256,
                8,
                valid_kernel,
                false,
            ),
            (
                true,
                "FLASHINFER",
                TARGET_CONFIG_SHA256,
                8,
                valid_kernel,
                false,
            ),
            (
                true,
                "TRTLLM_RAGGED",
                "other-checkpoint",
                8,
                valid_kernel,
                false,
            ),
            (
                true,
                "TRTLLM_RAGGED",
                TARGET_CONFIG_SHA256,
                7,
                valid_kernel,
                false,
            ),
        ] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join(BASENAME);
            write_parquet(
                &path,
                &[
                    Col::Str("identity", vec![identity]),
                    Col::I64("total_kv_tokens", vec![1026]),
                    Col::F64("latency", vec![1.0]),
                    Col::Str("runtime_revision", vec![RUNTIME_REVISION]),
                    Col::Str("measurement_scope", vec!["kimi_native_mla_inclusive"]),
                    Col::Str("kernel_source", vec![kernel]),
                    Col::Str("target_config_sha256", vec![target_hash]),
                    Col::Str("draft_config_sha256", vec![DRAFT_CONFIG_SHA256]),
                    Col::Str(
                        "image_digest",
                        vec![
                            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        ],
                    ),
                    Col::I64("sample_count", vec![5]),
                    Col::I64("rank_count", vec![ranks]),
                    Col::Bool("use_cuda_graph", vec![false]),
                    Col::Bool("prefill_query_quantization", vec![prefill_quant]),
                    Col::Str("prefill_backend", vec![prefill_backend]),
                ],
            );
            let table = NativeMlaTable {
                path: Some(path),
                curves: OnceLock::new(),
            };
            let key: NativeMlaKey = serde_json::from_str(identity).unwrap();
            let result = table.query(&key, 1026, &|x| x);
            assert_eq!(result.is_ok(), valid);
            if let Err(error) = result {
                assert!(matches!(error, AicError::InvalidPerfData(_)), "{error}");
                assert!(!error.is_missing_perf_data());
            }
        }
    }

    #[test]
    fn native_width_and_causal_keys_do_not_reuse_adjacent_buckets() {
        let key = NativeMlaKey {
            draft: true,
            causal: false,
            is_context: false,
            query_tokens: 2,
            sequence_count: 1,
            num_heads: 8,
            tensor_parallel_size: 8,
            kv_cache_dtype: "fp8".into(),
            gemm_quant_mode: "bfloat16".into(),
        };
        let table = NativeMlaTable {
            path: None,
            curves: OnceLock::from(Ok(BTreeMap::from([(
                key.clone(),
                LeafAxisCurve::from_map(
                    "total_kv_tokens",
                    BTreeMap::from([
                        (1026, LeafValue::with_power(1.0, 0.0)),
                        (1032, LeafValue::with_power(2.0, 0.0)),
                    ]),
                ),
            )]))),
        };
        assert_eq!(table.query(&key, 1026, &|x| x).unwrap().latency, 1.0);
        assert_eq!(table.query(&key, 1029, &|x| x).unwrap().latency, 1.5);
        assert!(
            table
                .query(&key, 1025, &|x| x)
                .unwrap_err()
                .is_missing_perf_data()
        );
        assert!(
            table
                .query(&key, 1033, &|x| x)
                .unwrap_err()
                .is_missing_perf_data()
        );
        for other in [
            NativeMlaKey {
                query_tokens: 3,
                ..key.clone()
            },
            NativeMlaKey {
                causal: true,
                ..key.clone()
            },
            NativeMlaKey {
                sequence_count: 2,
                ..key.clone()
            },
        ] {
            assert!(table.query(&other, 1026, &|x| x).is_err());
        }
    }
}
