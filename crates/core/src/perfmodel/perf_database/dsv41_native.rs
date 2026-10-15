// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Modified CPU contract adaptation of vLLM; copyright contributors to the vLLM project.
// Apache-2.0, vllm-project/vllm@ac9126e58aa7bbab1856ba6593ba4d5003fea516.
// Original paths: vllm/models/deepseek_v41/nvidia/{model.py,dspark.py} and decoder_replay_layers.py.
// https://github.com/vllm-project/vllm/tree/ac9126e58aa7bbab1856ba6593ba4d5003fea516

//! Inclusive native V4.1 layer measurements. These are not interchangeable
//! with the local-compute component table. Every physical query is exact;
//! neither query width nor causality nor absolute positions interpolate.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::SourceResolver;
use super::parquet_loader::PerfReader;
use crate::common::error::AicError;

pub const BASENAME: &str = "dsv41_native_stage_perf.parquet";
pub const RUNTIME_REVISION: &str = "ac9126e58aa7bbab1856ba6593ba4d5003fea516";

/// Native serving ownership is explicit; a stage's analytical children are
/// retained for weight inventory and SOL, never added to a measured stage.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeDsv41Scope {
    pub site: String,
    pub layer_id: u32,
    pub draft: bool,
    pub tensor_parallel_size: u32,
    pub expert_parallel_size: u32,
    pub config_sha256: String,
    pub kv_cache_layout: String,
    pub decoder_replay: bool,
    /// Zero for prefill, otherwise the exact per-request block width.
    pub query_tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeDsv41Key {
    pub scope: NativeDsv41Scope,
    pub sequence_count: u32,
    pub query_width: u32,
    pub past_kv_tokens: u32,
    pub is_context: bool,
    pub causal: bool,
}

impl NativeDsv41Key {
    pub fn validate(&self) -> Result<(), AicError> {
        let scope = &self.scope;
        if !matches!(
            scope.site.as_str(),
            "layer" | "finalize" | "context_preparation" | "input_preparation"
        ) {
            return Err(invalid("unknown V4.1 native ownership site"));
        }
        if self.sequence_count == 0
            || self.query_width == 0
            || scope.tensor_parallel_size == 0
            || scope.expert_parallel_size == 0
            || self.causal == scope.draft
            || scope.draft != (scope.layer_id >= 40)
            || scope.layer_id >= 43
            || self.is_context != (scope.query_tokens == 0)
            || (!self.is_context && self.query_width != scope.query_tokens)
            || scope.config_sha256.len() != 64
            || !scope
                .config_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || scope.kv_cache_layout != "vllm_mxfp8"
            || self.past_kv_tokens.checked_add(self.query_width).is_none()
        {
            return Err(invalid("invalid V4.1 native stage physical identity"));
        }
        Ok(())
    }
}

pub struct NativeDsv41Table {
    path: Option<PathBuf>,
    rows: OnceLock<Result<BTreeMap<NativeDsv41Key, f64>, String>>,
}

impl NativeDsv41Table {
    pub fn with_sources(root: &Path, resolver: &SourceResolver) -> Result<Self, AicError> {
        let primary = resolver
            .prioritized_sources_for(BASENAME, root)?
            .into_iter()
            .find(|s| s.channel == "primary");
        if primary
            .as_ref()
            .is_some_and(|s| s.source.kernel_sources().is_some())
        {
            return Err(invalid(
                "V4.1 native stages require an unfiltered primary table",
            ));
        }
        Ok(Self {
            path: primary.map(|s| s.source.0),
            rows: OnceLock::new(),
        })
    }

    pub fn query(&self, key: &NativeDsv41Key) -> Result<f64, AicError> {
        key.validate()?;
        let rows = self
            .rows
            .get_or_init(|| self.load().map_err(|e| e.to_string()))
            .as_ref()
            .map_err(invalid)?;
        rows.get(key).copied().ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "V4.1 native stage missing exact SILICON query: {key:?}"
            ))
        })
    }

    fn load(&self) -> Result<BTreeMap<NativeDsv41Key, f64>, AicError> {
        let Some(path) = &self.path else {
            return Ok(BTreeMap::new());
        };
        if !path.try_exists().map_err(invalid)? {
            return Ok(BTreeMap::new());
        }
        let reader = PerfReader::open(path)?;
        let identity = reader.col("identity")?;
        let latency = reader.col("latency")?;
        let revision = reader.col("runtime_revision")?;
        let measurement_scope = reader.col("measurement_scope")?;
        let kernel_source = reader.col("kernel_source")?;
        let image = reader.col("runtime_image_digest")?;
        let samples = reader.col("sample_count")?;
        let ranks = reader.col("rank_count")?;
        let graph = reader.col("used_cuda_graph")?;
        let mut rows = BTreeMap::new();
        let mut image_identity = None;
        for row in reader.rows()? {
            let row = row?;
            let key: NativeDsv41Key = serde_json::from_str(row.str(identity)?).map_err(invalid)?;
            key.validate()?;
            let value = row.f64(latency)?;
            let digest = row.str(image)?;
            if row.str(revision)? != RUNTIME_REVISION
                || row.str(measurement_scope)? != "dsv41_native_stage_inclusive"
                || row.str(kernel_source)?.is_empty()
                || !digest.starts_with("sha256:")
                || digest.len() != 71
                || !digest[7..]
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                || row.u32(samples)? == 0
                || row.u32(ranks)? != key.scope.tensor_parallel_size
                || row.bool_strict(graph)?
                || !value.is_finite()
                || value <= 0.0
            {
                return Err(invalid(
                    "V4.1 native layer runtime, rank, graph or measurement scope mismatch",
                ));
            }
            if let Some(previous) = &image_identity {
                if previous != digest {
                    return Err(invalid("mixed native runtime images"));
                }
            } else {
                image_identity = Some(digest.to_owned());
            }
            if rows.insert(key, value).is_some() {
                return Err(invalid("duplicate V4.1 native layer physical query"));
            }
        }
        Ok(rows)
    }
}

fn invalid(message: impl std::fmt::Display) -> AicError {
    AicError::InvalidPerfData(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::energy_test_fixtures::{Col, write_parquet};

    fn key() -> NativeDsv41Key {
        NativeDsv41Key {
            scope: NativeDsv41Scope {
                site: "layer".into(),
                layer_id: 0,
                draft: false,
                tensor_parallel_size: 4,
                expert_parallel_size: 1,
                config_sha256: "a".repeat(64),
                kv_cache_layout: "vllm_mxfp8".into(),
                decoder_replay: true,
                query_tokens: 6,
            },
            sequence_count: 1,
            query_width: 6,
            past_kv_tokens: 1024,
            is_context: false,
            causal: true,
        }
    }

    fn table(
        revision: &'static str,
        scope: &'static str,
        rank_count: i64,
        duplicate: bool,
    ) -> (tempfile::TempDir, NativeDsv41Table) {
        let root = tempfile::tempdir().unwrap();
        let identity = r#"{"scope":{"site":"layer","layer_id":0,"draft":false,"tensor_parallel_size":4,"expert_parallel_size":1,"config_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","kv_cache_layout":"vllm_mxfp8","decoder_replay":true,"query_tokens":6},"sequence_count":1,"query_width":6,"past_kv_tokens":1024,"is_context":false,"causal":true}"#;
        let digest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let n = if duplicate { 2 } else { 1 };
        write_parquet(
            &root.path().join(BASENAME),
            &[
                Col::Str("identity", vec![identity; n]),
                Col::F64("latency", vec![2.5; n]),
                Col::Str("runtime_revision", vec![revision; n]),
                Col::Str("measurement_scope", vec![scope; n]),
                Col::Str("kernel_source", vec!["observed.layer.forward"; n]),
                Col::Str("runtime_image_digest", vec![digest; n]),
                Col::I64("sample_count", vec![5; n]),
                Col::I64("rank_count", vec![rank_count; n]),
                Col::Bool("used_cuda_graph", vec![false; n]),
            ],
        );
        let table = NativeDsv41Table {
            path: Some(root.path().join(BASENAME)),
            rows: OnceLock::new(),
        };
        (root, table)
    }

    #[test]
    fn exact_layer_query_never_reuses_other_width_position_or_sequence_count() {
        let (_root, table) = table(RUNTIME_REVISION, "dsv41_native_stage_inclusive", 4, false);
        assert_eq!(table.query(&key()).unwrap(), 2.5);
        for (sequences, width, past) in [(2, 6, 1024), (1, 5, 1024), (1, 6, 1025)] {
            let mut other = key();
            other.sequence_count = sequences;
            other.query_width = width;
            other.scope.query_tokens = width;
            other.past_kv_tokens = past;
            assert!(table.query(&other).is_err());
        }
        let mut other = key();
        other.scope.layer_id = 1;
        assert!(table.query(&other).is_err());
        other = key();
        other.causal = false;
        assert!(table.query(&other).is_err());
    }

    #[test]
    fn bracketing_native_rows_do_not_establish_a_safe_kv_interpolation_interval() {
        // These are deliberately synthetic timings, not silicon evidence. Even
        // two neighboring measurements do not establish the omitted compressor,
        // indexer and attention-planner execution identity at an interior point.
        let mut left = key();
        left.past_kv_tokens = 1022;
        let mut right = left.clone();
        right.past_kv_tokens = 1026;
        let table = NativeDsv41Table {
            path: None,
            rows: OnceLock::from(Ok(BTreeMap::from([
                (left.clone(), 2.0),
                (right.clone(), 6.0),
            ]))),
        };
        assert_eq!(table.query(&left).unwrap(), 2.0);
        assert_eq!(table.query(&right).unwrap(), 6.0);
        // Include both parities, a possible ring wrap, and either exterior.
        for past in [1021, 1023, 1024, 1025, 1027] {
            let mut missing = left.clone();
            missing.past_kv_tokens = past;
            assert!(matches!(
                table.query(&missing),
                Err(AicError::PerfDatabase(_))
            ));
        }
        for axis in ["config", "layout", "owner", "replay", "sequence", "query"] {
            let mut other = left.clone();
            match axis {
                "config" => other.scope.config_sha256 = "b".repeat(64),
                "layout" => other.scope.kv_cache_layout = "sglang_584".into(),
                "owner" => {
                    other.scope.draft = true;
                    other.scope.layer_id = 40;
                    other.causal = false;
                }
                "replay" => other.scope.decoder_replay = false,
                "sequence" => other.sequence_count = 4,
                "query" => {
                    other.query_width = 5;
                    other.scope.query_tokens = 5;
                }
                _ => unreachable!(),
            }
            assert!(table.query(&other).is_err(), "must retain {axis} identity");
        }
    }

    #[test]
    fn native_layer_rejects_wrong_runtime_local_compute_partial_ranks_and_duplicates() {
        for (revision, scope, ranks, duplicate) in [
            ("other", "dsv41_native_stage_inclusive", 4, false),
            (RUNTIME_REVISION, "local_compute", 4, false),
            (RUNTIME_REVISION, "dsv41_native_stage_inclusive", 3, false),
            (RUNTIME_REVISION, "dsv41_native_stage_inclusive", 4, true),
        ] {
            let (_root, table) = table(revision, scope, ranks, duplicate);
            assert!(matches!(
                table.query(&key()),
                Err(AicError::InvalidPerfData(_))
            ));
        }
    }
}
