// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GEMM operator.
//!
//! Mirrors `aisim.sdk.operations.gemm.GEMM`. Config-time fields
//! (`n`, `k`, quant mode, scale_factor, scale_num_tokens) are set once when
//! the model graph is built; the query takes only `x` (the M dimension /
//! number of tokens) and routes through the GEMM perf table.
//!
//! For `fp8_static` quant mode there are two paths. When the loaded GEMM
//! table has `fp8_static` rows (direct path, SILICON / HYBRID / EMPIRICAL
//! only), SILICON looks up those rows and HYBRID / EMPIRICAL can use them
//! to calibrate utilization; only a low-precision input subtracts
//! `scale_matrix`. Without direct rows, these modes use the dynamic-FP8
//! lookup or estimate minus `compute_scale` overhead (and minus
//! `scale_matrix` when the input is also low-precision). SOL / SOL_FULL
//! always use the derived path with analytic GEMM and overhead values,
//! without reading any of these tables.
//! Post-subtraction latency is floored at the GEMM's own SOL roofline —
//! mirroring Python's `max(latency_floor, latency)` where
//! `latency_floor = query_gemm(..., DatabaseMode.SOL)` — not at 0.

use crate::common::enums::{DatabaseMode, GemmQuantMode, TransferKind};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, SolComponents, Source, subtract_sol};
use crate::operators::moe::policy_fingerprint;
use crate::operators::util_empirical::{self, UtilGrid, ZeroAwareDeltaLookup};
use crate::perf_database::PerfDatabase;
use crate::perf_database::gemm::{
    gemm_quant_by_name, gemm_sol_latency_ms_with_flops, gemm_sol_with_flops,
    normalize_fp8_static_quant, quant_tc_flops,
};
use serde::{Deserialize, Serialize};

/// GEMM operation: a dense matrix multiply of shape `M=x, N=n, K=k`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GemmOp {
    pub name: String,
    pub scale_factor: f64,
    pub n: u32,
    pub k: u32,
    pub quant_mode: GemmQuantMode,
    pub scale_num_tokens: u32,
    pub low_precision_input: bool,
    /// Context-parallel sequence-shard factor (Python's `_seq_split`, = `cp_size`).
    /// The per-rank token count is `ceil(m / seq_split)`. Defaults to 1 (no CP).
    #[serde(default = "default_seq_split")]
    pub seq_split: u32,
    /// Opt-in: a shape outside the collected grid degrades to SOL instead of
    /// erroring (SILICON only; quant-mode misses stay strict, HYBRID keeps
    /// its empirical fallback). Mirrors Python `below_grid_sol`.
    #[serde(default)]
    pub below_grid_sol: bool,
}

pub(crate) fn default_seq_split() -> u32 {
    1
}

/// Deserialize a parallel-split width that must be at least one. The
/// tail-appended `dcp_size` fields default to 1 when absent, but an explicit
/// zero in a serialized spec must fail at deserialization instead of being
/// normalized away by the geometry helpers (`dcp_geometry`'s `.max(1)`).
pub(crate) fn deserialize_positive_split<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = <u32 as serde::Deserialize>::deserialize(deserializer)?;
    if value == 0 {
        return Err(serde::de::Error::custom(
            "parallel split size must be a positive integer, got 0",
        ));
    }
    Ok(value)
}

impl GemmOp {
    /// Convenience constructor for the most common case (no token scaling,
    /// standard input precision).
    pub fn new(name: impl Into<String>, n: u32, k: u32, quant_mode: GemmQuantMode) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            n,
            k,
            quant_mode,
            scale_num_tokens: 1,
            low_precision_input: false,
            seq_split: 1,
            below_grid_sol: false,
        }
    }

    /// Query GEMM latency for the given `x` (M / number of tokens) and an
    /// optional quant-mode override.
    ///
    /// Mirrors Python's `GEMM.query`:
    /// 1. `m = x // scale_num_tokens`
    /// 2. Query GEMM table at `(m, n, k, quant_mode)`.
    /// 3. For `fp8_static`: on the derived path (no `fp8_static` rows in the
    ///    table, or SOL / SOL_FULL mode), subtract `compute_scale(m, k)` (and
    ///    `scale_matrix(m, k)` when low-precision-input). On the direct path
    ///    (the table has `fp8_static` rows), subtract only `scale_matrix(m, k)`
    ///    when low-precision-input. A subtraction floors the latency at the
    ///    GEMM roofline and tags the result `Estimated`.
    /// 4. Clamp to `>= 0`, scale by `scale_factor`.
    pub fn query(
        &self,
        db: &PerfDatabase,
        x: u32,
        quant_override: Option<GemmQuantMode>,
    ) -> Result<PerformanceResult, AicError> {
        let m = x / self.scale_num_tokens.max(1);
        // CP: per-rank token count = ceil(m / seq_split) (busiest rank).
        let m = m.div_ceil(self.seq_split.max(1));
        let quant = quant_override.unwrap_or(self.quant_mode);

        // Opt-in below-grid degrade (Python `_query_gemm_table`): a
        // SILICON shape miss falls to SOL when the quant's table exists;
        // quant-mode misses stay strict and HYBRID keeps its empirical
        // fallback. SOL carries no energy (Python passes energy=0.0).
        let base = match query_gemm_table(db, quant, m, self.n, self.k) {
            Err(err)
                if self.below_grid_sol
                    && db.database_mode == DatabaseMode::Silicon
                    && err.is_missing_perf_data()
                    && db.gemm.has_quant(quant)? =>
            {
                let tc_flops = quant_tc_flops(&db.system_spec, quant.mapping())?;
                PerformanceResult::sol(gemm_sol_with_flops(
                    &db.system_spec,
                    quant,
                    tc_flops,
                    m as f64,
                    self.n as f64,
                    self.k as f64,
                ))
            }
            other => other?,
        };
        let mut latency = base.latency_ms;
        let mut energy = base.energy_wms;
        let mut source = base.source;
        let mut sol = base.sol;
        let mut latency_floor = 0.0_f64;

        if quant == GemmQuantMode::Fp8Static {
            // Direct path: the merged GEMM table has `fp8_static` rows, so
            // `base` already is the fp8_static price and no compute_scale
            // overhead is subtracted. SOL / SOL_FULL read no table and keep
            // the derived path.
            let direct = !matches!(db.database_mode, DatabaseMode::Sol | DatabaseMode::SolFull)
                && db.gemm.has_direct_fp8_static()?;
            let mut subtracted = false;
            // The component sources are irrelevant: a subtracted result is
            // tagged Estimated below regardless of mode. Energy is
            // subtracted alongside latency (Python `GEMM.query`).
            if !direct {
                let cs = query_compute_scale_table(db, quant, m, self.k)?;
                latency -= cs.latency_ms;
                energy -= cs.energy_wms;
                sol = subtract_sol(sol, cs.sol);
                subtracted = true;
            }

            if self.low_precision_input {
                let sm = query_scale_matrix_table(db, quant, m, self.k)?;
                latency -= sm.latency_ms;
                energy -= sm.energy_wms;
                sol = subtract_sol(sol, sm.sol);
                subtracted = true;
            }
            if subtracted {
                // Python (`operations/gemm.py`): the subtraction leaves a
                // path that still contains the GEMM; independently
                // interpolated component tables can cross, but that path
                // cannot be faster than the GEMM's own roofline. Floor at the
                // SOL (NOT at 0), and tag the result "estimated" (a
                // subtraction of overhead tables is modeled, not measured
                // directly). Without a subtraction, the direct path keeps the
                // row's source. Energy has no analogous SOL model, so it keeps
                // the plain non-negative clamp.
                let tc_flops = quant_tc_flops(&db.system_spec, quant.mapping())?;
                let floor_components = gemm_sol_with_flops(
                    &db.system_spec,
                    quant,
                    tc_flops,
                    m as f64,
                    self.n as f64,
                    self.k as f64,
                );
                latency_floor = floor_components.time_ms();
                // When the floor fires, the result IS the GEMM's own roofline —
                // its decomposition replaces the subtracted one. (Under SOL mode
                // the floor always fires: the subtrahends are non-negative SOL
                // values taken off exactly this roofline.)
                if sol.is_some() && latency < latency_floor {
                    sol = Some(floor_components);
                }
                source = Source::Estimated;
            }
        }

        let mut result = PerformanceResult::with_energy(latency.max(latency_floor), energy, source);
        if let Some(components) = sol {
            result = result.with_sol(components);
        }
        Ok(result.clamp_non_negative().scaled(self.scale_factor))
    }

    /// Per-tensor weight count in bytes, matching Python's
    /// `GEMM.get_weights()`: `n * k * quant_mode.value.memory *
    /// scale_factor`. The `scale_factor` multiplier mirrors the way model
    /// builders count weights once at construction and let the op replicate
    /// per layer (e.g. `scale_factor = num_hidden_layers` for body GEMMs).
    pub fn weights_bytes(&self) -> f64 {
        (self.n as f64) * (self.k as f64) * self.quant_mode.mapping().memory * self.scale_factor
    }
}

// ---------------------------------------------------------------------------
// Database-mode dispatch, mirroring the Python `_query_*_table` classmethods
// (`operations/gemm.py`): SILICON queries the table; HYBRID converts a typed
// silicon miss into the util-space empirical estimate; EMPIRICAL always
// estimates; SOL (and the retired SOL_FULL alias) returns the pure
// speed-of-light roofline with `Source::Sol` and zero energy.
// ---------------------------------------------------------------------------

/// GEMM latency + energy for `(m, n, k, quant)` under the database's query
/// mode. Silicon-table hits carry the table's energy (Python `_interp_pr`);
/// empirical fallbacks are energy-0.0 (Python passes `energy=0.0`).
fn query_gemm_table(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    n: u32,
    k: u32,
) -> Result<PerformanceResult, AicError> {
    let silicon = |v: crate::perf_database::perf_interp::LeafValue| {
        PerformanceResult::with_energy(v.latency, v.energy, Source::Silicon)
    };
    match db.database_mode {
        // Python `_query_gemm_table`: `get_sol(m, n, k, quant_mode)[0]` at the
        // RAW quant mode (not the fp8_static-normalized table quant).
        DatabaseMode::Sol | DatabaseMode::SolFull => {
            let tc_flops = quant_tc_flops(&db.system_spec, quant.mapping())?;
            Ok(PerformanceResult::sol(gemm_sol_with_flops(
                &db.system_spec,
                quant,
                tc_flops,
                m as f64,
                n as f64,
                k as f64,
            )))
        }
        DatabaseMode::Empirical => Ok(PerformanceResult::new(
            gemm_empirical(db, quant, m, n, k)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match db.gemm.query(quant, m, n, k) {
            Ok(value) => Ok(silicon(value)),
            Err(err) if err.is_missing_perf_data() => Ok(PerformanceResult::new(
                gemm_empirical(db, quant, m, n, k)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok(silicon(db.gemm.query(quant, m, n, k)?)),
    }
}

/// Per-quant achieved-util LEVEL `e(q)` for GEMM, keyed by the
/// `(memory, compute)` profile. Mirrors `_GEMM_QUANT_UTIL_LEVEL`
/// (`operations/gemm.py`); consumed ONLY by the cross-profile relation of
/// the quant-transfer ladder, and only as the ratio `e(query)/e(ref)`.
/// Derivation method + LOO evidence documented on the Python table.
pub(crate) const GEMM_QUANT_UTIL_LEVEL: &[(f64, f64, f64)] = &[
    (2.0, 1.0, 0.70),    // w16a16 / bfloat16              [data 0.55-0.79]
    (1.0, 1.0, 0.55),    // w8a16 / int8_wo                [inferred]
    (0.5625, 1.0, 0.45), // w4a16+scales / nvfp4_wo (Marlin FP4) [copies inferred (0.5,1)]
    (0.5, 1.0, 0.45),    // w4a16 / int4_wo                [inferred]
    (1.0, 2.0, 0.45),    // w8a8 / fp8(_block/_ootb), sq   [data 0.28-0.55]
    (0.5, 2.0, 0.35),    // w4a8                           [inferred]
    (1.0, 4.0, 0.30),    // w8a4                           [inferred]
    (0.5, 4.0, 0.30),    // w4a4                           [inferred ≈ nvfp4]
    (0.5625, 4.0, 0.30), // w4a4 / nvfp4                   [data 0.21-0.36]
];
/// Unlisted profile: mid-range relative level (Python `_GEMM_QUANT_UTIL_DEFAULT`).
const GEMM_QUANT_UTIL_DEFAULT: f64 = 0.45;

/// Achieved-util level `e(q)` for a GEMM quant, by `(memory, compute)`
/// profile (mirrors `_gemm_quant_util_level`, `operations/gemm.py`).
fn gemm_quant_util_level(quant: GemmQuantMode) -> f64 {
    let mapping = quant.mapping();
    GEMM_QUANT_UTIL_LEVEL
        .iter()
        .find(|(memory, compute, _)| *memory == mapping.memory && *compute == mapping.compute)
        .map(|(_, _, level)| *level)
        .unwrap_or(GEMM_QUANT_UTIL_DEFAULT)
}

/// Collected quants with a DIFFERENT `(memory, compute)` profile than the
/// query, ordered lexicographically by `(|Δcompute|, |Δmemory|)` — GEMM uses
/// the compute-first metric (Python `xprofile_quant_order(...,
/// prefer_same_compute=True)`): the COMPUTE factor (activation precision)
/// determines the dense kernel's compute family (a weight-only quant runs
/// bf16 MMA after fused dequant), so a same-compute reference must always
/// beat a cross-compute one — under plain L1 a weight-only quant such as
/// (0.5625, 1) TIES between bfloat16 and fp8 and the reference would be a
/// file-order lottery. Stable sort keeps file order on full ties.
fn xprofile_gemm_quants(
    query: GemmQuantMode,
    table_quants: &[GemmQuantMode],
) -> Vec<GemmQuantMode> {
    let qp = query.mapping();
    let mut refs: Vec<GemmQuantMode> = table_quants
        .iter()
        .copied()
        .filter(|q| {
            let mapping = q.mapping();
            *q != query && !(mapping.memory == qp.memory && mapping.compute == qp.compute)
        })
        .collect();
    let dist = |q: GemmQuantMode| {
        let mapping = q.mapping();
        (
            (mapping.compute - qp.compute).abs(),
            (mapping.memory - qp.memory).abs(),
        )
    };
    refs.sort_by(|a, b| {
        dist(*a)
            .partial_cmp(&dist(*b))
            .expect("finite profile distances")
    });
    refs
}

/// Distinct quant names of the loaded GEMM table in first-seen (file row)
/// order, parsed to the enum. A typed table miss yields an empty list —
/// mirroring Python, where the ladder's table accesses happen inside
/// `grid_from_reference`'s typed-miss catch. `fp8_static` rows are left out:
/// they are never an xquant or xprofile reference.
fn gemm_table_quants(db: &PerfDatabase) -> Result<Vec<GemmQuantMode>, AicError> {
    match db.gemm.available_quants() {
        Ok(names) => Ok(names
            .iter()
            .filter_map(|n| gemm_quant_by_name(n))
            .filter(|q| *q != GemmQuantMode::Fp8Static)
            .collect()),
        Err(err) if err.is_missing_perf_data() => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}

/// A borrowed util grid over `source_quant`'s whole `(m, n, k)` table, built
/// with `sol_quant`'s SOL (ReferenceCandidate contract: QUERY quant for the
/// same-profile relation — numerically identical SOL — REFERENCE quant for
/// cross-profile). `None` on a typed data miss; emptiness is the caller's
/// check.
fn gemm_reference_grid(
    db: &PerfDatabase,
    source_quant: GemmQuantMode,
    sol_quant: GemmQuantMode,
    provenance: &'static str,
    key: &str,
) -> Result<Option<std::sync::Arc<UtilGrid>>, AicError> {
    let spec = &db.system_spec;
    db.util_grids.get_or_try_build(key, || {
        let tc_flops = quant_tc_flops(spec, sol_quant.mapping())?;
        let sol =
            |c: &[f64]| gemm_sol_latency_ms_with_flops(spec, sol_quant, tc_flops, c[0], c[1], c[2]);
        match db.gemm.gemm_points(source_quant) {
            Ok(points) => {
                let mut grid = UtilGrid::new(util_empirical::build_samples(points, sol));
                grid.reference_provenance = Some(provenance);
                Ok(Some(grid))
            }
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })
}

/// `SOL(query)/util` over the quant's own collected `(m, n, k)` grid, with
/// the quant-transfer ladder behind an own-data miss. Mirrors Python
/// `_query_gemm_table::get_empirical` + `util_empirical.quant_transfer_grid`
/// (grid depth 3; the own-grid SOL uses the ORIGINAL quant, numerically
/// identical to the table quant since the profiles match). The table quant
/// of a fp8_static query is `fp8_static` itself on a table with fp8_static
/// rows (direct path), and `fp8` on a table without them (derived path,
/// as before):
///
/// 1. own-quant grid — inherently cross-shape (depth-3 over every collected
///    `(m, n, k)`), so GEMM's **xshape relation class is structurally
///    empty** and no xshape step exists;
/// 2. **xquant** — FIRST same-profile sibling in file order (Python pools
///    one whole-table candidate per sibling with constant features; the
///    stable nearest-candidate argmin selects the first). Not retried on an
///    empty grid — parity with the single pooled selection;
/// 3. **xprofile** — compute-first nearest-profile walk, first quant with a
///    non-empty grid wins, util rescaled by `e(query)/e(ref)`.
fn gemm_empirical(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    n: u32,
    k: u32,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let tc_flops = quant_tc_flops(spec, quant.mapping())?;
    let sol = |c: &[f64]| gemm_sol_latency_ms_with_flops(spec, quant, tc_flops, c[0], c[1], c[2]);
    // The table quant: `fp8_static` on the direct path, `fp8` for a
    // fp8_static query without such rows, the quant itself otherwise. It
    // names the grid key, so the two paths never share a util grid.
    let tqm = db.gemm.gemm_lookup_quant(quant)?;
    let key = format!("gemm:{}", tqm.name());
    let mut grid = db.util_grids.get_or_try_build(&key, || {
        match db.gemm.gemm_points(quant) {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(
                points, sol,
            )))),
            // Typed coverage miss -> no grid (the ladder below, then
            // estimate(), takes over); schema/load errors propagate.
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;

    let mut util_scale = 1.0;
    if grid.as_deref().is_none_or(UtilGrid::is_empty) {
        let policy = db.transfer_policy;
        let table_quants = gemm_table_quants(db)?;
        let fingerprint = policy_fingerprint(policy);

        if policy.contains(TransferKind::XQuant) {
            let qp = tqm.mapping();
            if let Some(ref_q) = table_quants.iter().copied().find(|q| {
                let mapping = q.mapping();
                *q != tqm && mapping.memory == qp.memory && mapping.compute == qp.compute
            }) {
                let key = format!(
                    "gemm_xquant:{}:policy={}:ref={}",
                    tqm.name(),
                    fingerprint,
                    ref_q.name()
                );
                if let Some(reference) = gemm_reference_grid(db, ref_q, tqm, "xquant", &key)? {
                    grid = Some(reference);
                }
            }
        }

        if grid.as_deref().is_none_or(UtilGrid::is_empty) && policy.contains(TransferKind::XProfile)
        {
            for ref_q in xprofile_gemm_quants(tqm, &table_quants) {
                let key = format!(
                    "gemm_xprofile:{}:policy={}:ref={}",
                    tqm.name(),
                    fingerprint,
                    ref_q.name()
                );
                if let Some(reference) = gemm_reference_grid(db, ref_q, ref_q, "xprofile", &key)? {
                    if !reference.is_empty() {
                        grid = Some(reference);
                        util_scale = gemm_quant_util_level(tqm) / gemm_quant_util_level(ref_q);
                        break;
                    }
                }
            }
        }
    }

    let query = [m as f64, n as f64, k as f64];
    let (latency, _) = util_empirical::estimate(sol(&query), &query, grid.as_deref(), util_scale)?;
    // Relation fired = the reference grid's transfer kind (xquant/xprofile),
    // or own-quant "empirical" when no borrow happened.
    db.note_provenance(
        grid.as_deref()
            .and_then(|g| g.reference_provenance)
            .and_then(util_empirical::ProvenanceTier::from_tag)
            .unwrap_or(util_empirical::ProvenanceTier::Empirical),
    );
    Ok(latency)
}

/// compute_scale delta (latency + energy) for `(m, k)` under the database's
/// query mode. Empirical estimates are energy-0.0.
fn query_compute_scale_table(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    k: u32,
) -> Result<PerformanceResult, AicError> {
    let silicon = |v: crate::perf_database::perf_interp::LeafValue| {
        PerformanceResult::with_energy(v.latency, v.energy, Source::Silicon)
    };
    match db.database_mode {
        // Python `_query_compute_scale_table::get_sol`:
        // `sol_mem = 2 m k / bw * 1000` (read + write of the activation);
        // triple is `(sol_mem, 0, sol_mem)` — pure memory bound.
        DatabaseMode::Sol | DatabaseMode::SolFull => {
            Ok(PerformanceResult::sol(SolComponents::new(
                0.0,
                2.0 * m as f64 * k as f64 / db.system_spec.gpu.mem_bw * 1000.0,
            )))
        }
        DatabaseMode::Empirical => Ok(PerformanceResult::new(
            compute_scale_empirical(db, quant, m, k)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match db.gemm.query_compute_scale(quant, m, k) {
            Ok(value) => Ok(silicon(value)),
            Err(err) if err.is_missing_perf_data() => Ok(PerformanceResult::new(
                compute_scale_empirical(db, quant, m, k)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok(silicon(db.gemm.query_compute_scale(quant, m, k)?)),
    }
}

/// Zero-aware nearest-point delta estimate over the compute_scale grid.
/// Mirrors Python `_query_compute_scale_table::get_empirical`: a typed miss
/// on the slice itself becomes the terminal `EmpiricalNotImplemented`.
fn compute_scale_empirical(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    k: u32,
) -> Result<f64, AicError> {
    let key = format!("compute_scale:{}", normalize_fp8_static_quant(quant).name());
    let lookup =
        db.delta_lookups
            .get_or_try_build(&key, || match db.gemm.compute_scale_points(quant) {
                Ok(points) => Ok(ZeroAwareDeltaLookup::new(points)),
                Err(err) if err.is_missing_perf_data() => Err(AicError::EmpiricalNotImplemented(
                    format!("No empirical compute_scale data is available for m={m}, k={k}."),
                )),
                Err(err) => Err(err),
            })?;
    // sol_mem = 2 m k / bw * 1000 (read + write of the activation).
    let spec = &db.system_spec;
    let latency = lookup.estimate(&[m as f64, k as f64], |c| {
        2.0 * c[0] * c[1] / spec.gpu.mem_bw * 1000.0
    })?;
    // The delta lookup fired (Python `_ZeroAwareDeltaLookup.estimate` notes
    // "empirical"; zero deltas count — they are measured values).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

/// scale_matrix latency + energy for `(m, k)` under the database's query
/// mode. Empirical estimates are energy-0.0.
fn query_scale_matrix_table(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    k: u32,
) -> Result<PerformanceResult, AicError> {
    let silicon = |v: crate::perf_database::perf_interp::LeafValue| {
        PerformanceResult::with_energy(v.latency, v.energy, Source::Silicon)
    };
    match db.database_mode {
        // Python `_query_scale_matrix_table::get_sol`:
        // `sol_mem = 3 m k / bw * 1000`; triple `(sol_mem, 0, sol_mem)`.
        DatabaseMode::Sol | DatabaseMode::SolFull => {
            Ok(PerformanceResult::sol(SolComponents::new(
                0.0,
                3.0 * m as f64 * k as f64 / db.system_spec.gpu.mem_bw * 1000.0,
            )))
        }
        DatabaseMode::Empirical => Ok(PerformanceResult::new(
            scale_matrix_empirical(db, quant, m, k)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match db.gemm.query_scale_matrix(quant, m, k) {
            Ok(value) => Ok(silicon(value)),
            Err(err) if err.is_missing_perf_data() => Ok(PerformanceResult::new(
                scale_matrix_empirical(db, quant, m, k)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok(silicon(db.gemm.query_scale_matrix(quant, m, k)?)),
    }
}

/// `SOL(query)/util` over the scale_matrix `(m, k)` grid (a real memory
/// kernel, unlike the compute_scale delta). Mirrors Python
/// `_query_scale_matrix_table::get_empirical` (grid depth 2,
/// `sol_mem = 3 m k / bw * 1000`).
fn scale_matrix_empirical(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    k: u32,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let sol = |c: &[f64]| 3.0 * c[0] * c[1] / spec.gpu.mem_bw * 1000.0;
    let key = format!("scale_matrix:{}", normalize_fp8_static_quant(quant).name());
    let grid =
        db.util_grids
            .get_or_try_build(&key, || match db.gemm.scale_matrix_points(quant) {
                Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(
                    points, sol,
                )))),
                Err(err) if err.is_missing_perf_data() => Ok(None),
                Err(err) => Err(err),
            })?;
    let query = [m as f64, k as f64];
    let (latency, _) = util_empirical::estimate(sol(&query), &query, grid.as_deref(), 1.0)?;
    // Own-shape util fired (Python estimate()'s default provenance).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::TransferPolicy;
    use std::path::PathBuf;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_db() -> PerfDatabase {
        let systems_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("python/aisim/src/aisim_core/systems");
        PerfDatabase::load(&systems_root, "b200_sxm", "vllm", "0.24.0").expect("db must load")
    }

    // Scale semantics are pinned RELATIVE to the op's own base query — no
    // recorded-value constants, so data refreshes cannot break them. (The
    // absolute exact-hit pin lived at the table layer and was deleted as a
    // duplicate of the table-level v2 exact case.)
    #[test]
    fn gemm_op_scale_factor_multiplies_latency() {
        let db = b200_vllm_db();
        let base = GemmOp::new("base", 65536, 16384, GemmQuantMode::Bfloat16)
            .query(&db, 32768, None)
            .expect("query must succeed");
        assert_eq!(base.source, Source::Silicon);
        let op = GemmOp {
            name: "scaled".to_string(),
            scale_factor: 0.5,
            n: 65536,
            k: 16384,
            quant_mode: GemmQuantMode::Bfloat16,
            scale_num_tokens: 1,
            low_precision_input: false,
            seq_split: 1,
            below_grid_sol: false,
        };
        let result = op.query(&db, 32768, None).expect("query must succeed");
        assert!(
            (result.latency_ms - base.latency_ms * 0.5).abs() < 1e-12,
            "scale_factor must halve the base latency: got {} vs base {}",
            result.latency_ms,
            base.latency_ms
        );
    }

    #[test]
    fn gemm_op_scale_num_tokens_divides_x() {
        let db = b200_vllm_db();
        // scale_num_tokens=2 means x=65536 must query at m=32768: identical
        // to the direct m=32768 query, pinned relatively.
        let base = GemmOp::new("base", 65536, 16384, GemmQuantMode::Bfloat16)
            .query(&db, 32768, None)
            .expect("query must succeed");
        let op = GemmOp {
            name: "halved".to_string(),
            scale_factor: 1.0,
            n: 65536,
            k: 16384,
            quant_mode: GemmQuantMode::Bfloat16,
            scale_num_tokens: 2,
            low_precision_input: false,
            seq_split: 1,
            below_grid_sol: false,
        };
        let result = op.query(&db, 65536, None).expect("query must succeed");
        assert!(
            (result.latency_ms - base.latency_ms).abs() < 1e-12,
            "scale_num_tokens must divide x: got {} vs base {}",
            result.latency_ms,
            base.latency_ms
        );
    }

    /// Oracle from the Python reference on the same data:
    /// `float(db.query_gemm(8, 1, 2048, GEMMQuantMode.bfloat16,
    /// database_mode=DatabaseMode.SOL))` = 4.78961038961039e-06.
    #[test]
    fn gemm_op_below_grid_sol_flag_degrades_shape_miss_to_sol() {
        let db = b200_vllm_db();
        // n=1 is ~5 octaves below the smallest collected n: a strict miss.
        let strict = GemmOp::new("gate", 1, 2048, GemmQuantMode::Bfloat16);
        assert!(strict.query(&db, 8, None).is_err());

        let op = GemmOp {
            below_grid_sol: true,
            ..strict
        };
        let result = op
            .query(&db, 8, None)
            .expect("below-grid opt-in must degrade to SOL");
        assert_eq!(result.source, Source::Sol);
        assert!(
            (result.latency_ms - 4.78961038961039e-06).abs() < 1e-15,
            "expected the Python SOL oracle, got {}",
            result.latency_ms
        );
    }

    #[test]
    fn gemm_op_quant_override_routes_to_different_quant() {
        let db = b200_vllm_db();
        let op = GemmOp::new("default-bf16", 65536, 16384, GemmQuantMode::Bfloat16);
        let default = op.query(&db, 32768, None).expect("default query");
        // The override must route to the nvfp4 table: identical to querying a
        // native nvfp4 op, and different from the bf16 default. Relative pins
        // only — no recorded constants to re-mint on data refreshes.
        let overridden = op
            .query(&db, 32768, Some(GemmQuantMode::Nvfp4))
            .expect("override query must succeed");
        let native = GemmOp::new("native-nvfp4", 65536, 16384, GemmQuantMode::Nvfp4)
            .query(&db, 32768, None)
            .expect("native query");
        assert!(
            (overridden.latency_ms - native.latency_ms).abs() < 1e-12,
            "override must equal the native nvfp4 lookup: {} vs {}",
            overridden.latency_ms,
            native.latency_ms
        );
        assert!(
            (overridden.latency_ms - default.latency_ms).abs() > 1e-9,
            "override must change the lookup away from bf16"
        );
    }

    /// Structural wiring for the GEMM util-empirical estimator (off-grid m,
    /// off-site, exact site, small-shape corner). Math pinned on synthetic
    /// grids in `util_empirical`/`perf_interp`; values in the goldens.
    #[test]
    fn gemm_empirical_regime_routing() {
        let mut db = b200_vllm_db();
        db.database_mode = crate::common::enums::DatabaseMode::Empirical;
        let cases = [
            (3000u32, 65536u32, 16384u32, GemmQuantMode::Bfloat16),
            (777, 4000, 5000, GemmQuantMode::Bfloat16),
            (32768, 65536, 16384, GemmQuantMode::Nvfp4),
            (1, 129, 130, GemmQuantMode::Fp8),
        ];
        for (m, n, k, quant) in cases {
            let result = query_gemm_table(&db, quant, m, n, k).expect("empirical query");
            assert!(result.latency_ms.is_finite() && result.latency_ms > 0.0);
            assert_eq!(
                result.source,
                Source::Empirical,
                "({m}, {n}, {k}, {quant:?})"
            );
            assert_eq!(
                result.energy_wms, 0.0,
                "empirical fallback carries no energy"
            );
        }
    }

    #[test]
    fn w4a16_nvfp4_uses_xprofile_only_in_hybrid() {
        let systems_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("python/aisim/src/aisim_core/systems");
        let mut db = PerfDatabase::load(&systems_root, "h200_sxm", "trtllm", "1.3.0rc20")
            .expect("database must load");

        let silicon = query_gemm_table(&db, GemmQuantMode::W4a16Nvfp4, 8192, 1536, 32);
        assert!(matches!(silicon, Err(AicError::PerfDatabase(_))));

        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let hybrid = query_gemm_table(&db, GemmQuantMode::W4a16Nvfp4, 8192, 1536, 32)
            .expect("XPROFILE transfer should make W4A16-NVFP4 HYBRID-estimable");
        assert_eq!(hybrid.source, Source::Empirical);
        assert!(hybrid.latency_ms.is_finite() && hybrid.latency_ms > 0.0);
    }

    /// HYBRID on a quant with NO collected table under a policy WITHOUT
    /// XProfile must surface the terminal EmpiricalNotImplemented miss, never
    /// a fabricated value (mirrors the Python contract: cross-profile
    /// borrowing is policy-gated, never implicit). int4_wo (0.5, 1) has no
    /// same-profile sibling anywhere, so "balanced" finds nothing.
    #[test]
    fn gemm_hybrid_missing_quant_raises_when_policy_forbids_transfer() {
        let balanced = crate::common::enums::TransferPolicy {
            xshape: true,
            xquant: true,
            xprofile: false,
            xop: false,
        };
        let db = b200_vllm_db().with_mode(crate::common::enums::DatabaseMode::Hybrid, balanced);
        let result = query_gemm_table(&db, GemmQuantMode::Int4Wo, 64, 64, 64);
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "got {result:?}"
        );
    }

    /// Under the default (all-on) policy an uncollected quant resolves via
    /// the xprofile borrow, and the donor selection is COMPUTE-FIRST, not
    /// file-order-first. Synthetic discrimination vehicle (no production
    /// data): fp8 rows come FIRST in file order with latencies 1000x the
    /// bfloat16 rows. int4_wo (profile (0.5, 1), bf16 compute pipeline) must
    /// borrow bfloat16 (delta-compute = 0) — a regression to file-order
    /// selection would land three orders of magnitude higher.
    #[test]
    fn gemm_hybrid_missing_quant_borrows_xprofile_under_default_policy() {
        use crate::perf_database::energy_test_fixtures::{
            Col, write_energy_systems_root, write_parquet,
        };
        let tmp = tempfile::tempdir().expect("tmpdir");
        let data = write_energy_systems_root(tmp.path());
        write_parquet(
            &data.join("gemm_perf.parquet"),
            &[
                Col::Str("gemm_dtype", vec!["fp8", "fp8", "bfloat16", "bfloat16"]),
                Col::I64("m", vec![64, 128, 64, 128]),
                Col::I64("n", vec![64, 64, 64, 64]),
                Col::I64("k", vec![64, 64, 64, 64]),
                Col::F64("latency", vec![1000.0, 2000.0, 1.0, 2.0]),
            ],
        );
        let mut db = PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0").expect("db");
        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let result =
            query_gemm_table(&db, GemmQuantMode::Int4Wo, 64, 64, 64).expect("xprofile borrow");
        assert_eq!(result.source, Source::Empirical);
        assert_eq!(
            db.worst_provenance(),
            util_empirical::ProvenanceTier::XProfile
        );
        assert!(
            result.latency_ms < 100.0,
            "compute-first must borrow the bfloat16 donor (~O(1) ms), got {} \
             (file-order fp8 donor would be ~O(1000))",
            result.latency_ms
        );
    }

    fn h200_vllm_db() -> PerfDatabase {
        let systems_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("python/aisim/src/aisim_core/systems");
        PerfDatabase::load(&systems_root, "h200_sxm", "vllm", "0.24.0").expect("db must load")
    }

    /// Quant-transfer ladder pins on h200/vllm/0.24.0 (collected quants in
    /// file order: [bfloat16, fp8, fp8_block]). Values are rust-engine
    /// regression pins re-minted at the 0.24 re-anchor (the python query
    /// layer retired with #1357; the python-era oracles live in git history).
    ///
    /// - sq (1, 2): xquant borrow from fp8 (first same-profile in file
    ///   order; identical SOL, no rescale).
    /// - int4_wo (0.5, 1): xprofile borrow from bfloat16 (Δcompute=0). NOTE:
    ///   since the 0.24 refresh bfloat16 is ALSO file-order-first here, so
    ///   this case no longer distinguishes compute-first selection from
    ///   file-order selection — that regression guard lives in
    ///   `gemm_hybrid_missing_quant_borrows_xprofile_under_default_policy`
    ///   on b200, where the file order starts with nvfp4.
    /// - nvfp4 (0.5625, 4): NO LONGER a ladder vehicle on h200 — the strict
    ///   per-dtype resolution (#1398) rejects fp4 at query entry because
    ///   h200 has no fp4 tensor cores (its old xprofile estimate was
    ///   anchored on a fictional bf16*4 SOL). Weight-only fp4 modes (the
    ///   #1392 nvfp4_wo plan) declare the bf16 compute pipeline and keep
    ///   using the ladder like int4_wo does.
    /// Regenerate if the shipped GEMM tables or the util math change.
    #[test]
    fn gemm_quant_transfer_ladder_matches_python_oracles() {
        let mut db = h200_vllm_db();
        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        // Tier-only pins: WHICH ladder rung admits each quant is the
        // regression surface; the borrowed values are data-refresh churn and
        // are deliberately not pinned (the one value-discriminating xprofile
        // pin lives in the b200 test above).
        let cases = [
            (
                GemmQuantMode::Sq,
                512u32,
                util_empirical::ProvenanceTier::XQuant,
            ),
            (
                GemmQuantMode::Sq,
                8192,
                util_empirical::ProvenanceTier::XQuant,
            ),
            (
                GemmQuantMode::Int4Wo,
                512,
                util_empirical::ProvenanceTier::XProfile,
            ),
            (
                GemmQuantMode::Int4Wo,
                8192,
                util_empirical::ProvenanceTier::XProfile,
            ),
        ];
        for (quant, m, tier) in cases {
            db.reset_provenance();
            let result = query_gemm_table(&db, quant, m, 4096, 4096).expect("ladder query");
            let latency = result.latency_ms;
            assert!(
                latency.is_finite() && latency > 0.0,
                "({quant:?}, m={m}): ladder estimate must be positive, got {latency}"
            );
            assert_eq!(
                result.source,
                Source::Empirical,
                "({quant:?}, m={m}): wrong source"
            );
            assert_eq!(
                db.worst_provenance(),
                tier,
                "({quant:?}, m={m}): wrong tier"
            );
        }
        // fp4 on h200: strict resolution fires before the ladder.
        assert!(matches!(
            query_gemm_table(&db, GemmQuantMode::Nvfp4, 512, 4096, 4096),
            Err(AicError::MissingSystemFlops(_))
        ));
    }

    #[test]
    fn gemm_op_weights_bytes_matches_python_formula() {
        let op = GemmOp::new("w", 1024, 4096, GemmQuantMode::Bfloat16);
        // bfloat16 memory factor is 2.0; weights = 1024 * 4096 * 2.0.
        assert_eq!(op.weights_bytes(), 1024.0 * 4096.0 * 2.0);

        let fp8_op = GemmOp::new("w-fp8", 1024, 4096, GemmQuantMode::Fp8);
        // fp8 memory factor is 1.0.
        assert_eq!(fp8_op.weights_bytes(), 1024.0 * 4096.0 * 1.0);
    }

    /// ENERGY oracle for the fp8_static composition on synthetic
    /// power-carrying fixtures through a full `PerfDatabase` (systems root =
    /// `energy_test_fixtures::write_energy_systems_root`). Python twin
    /// (pandas fixtures with the same rows; `GEMM` op from
    /// `operations/gemm.py`):
    ///
    /// ```text
    /// gemm_perf:         fp8 m=128/256 @ (n=1024, k=1024): lat 1.0/3.0, power 100/200
    /// computescale_perf: fp8 m=128/256 @ k=1024:           lat 0.25/0.75, power 40/80
    /// scale_matrix_perf: fp8 m=128/256 @ k=1024:           lat 0.125/0.375, power 20/60
    /// GEMM("g", 2.0, 1024, 1024, fp8_static, low_precision_input=True).query(db, x=192)
    /// # -> latency=2.5, energy=520.0     (= (2.0-0.5-0.25, 300-30-10) * 2)
    /// GEMM("g1", 1.0, 1024, 1024, fp8_static).query(db, x=192)
    /// # -> latency=1.5, energy=270.0     (= (2.0-0.5, 300-30) * 1)
    /// ```
    ///
    /// Exercises: energy subtraction of the compute_scale and scale_matrix
    /// pieces, the (non-binding here) SOL latency floor + non-negative
    /// energy clamp, and `_scale_factor` scaling of BOTH latency and energy.
    #[test]
    fn gemm_fp8_static_energy_composition_matches_python_oracle() {
        use crate::perf_database::energy_test_fixtures::{
            Col, write_energy_systems_root, write_parquet,
        };
        let tmp = tempfile::tempdir().expect("tmpdir");
        let data = write_energy_systems_root(tmp.path());
        write_parquet(
            &data.join("gemm_perf.parquet"),
            &[
                Col::Str("gemm_dtype", vec!["fp8", "fp8"]),
                Col::I64("m", vec![128, 256]),
                Col::I64("n", vec![1024, 1024]),
                Col::I64("k", vec![1024, 1024]),
                Col::F64("latency", vec![1.0, 3.0]),
                Col::F64("power", vec![100.0, 200.0]),
            ],
        );
        write_parquet(
            &data.join("computescale_perf.parquet"),
            &[
                Col::Str("quant_dtype", vec!["fp8", "fp8"]),
                Col::I64("m", vec![128, 256]),
                Col::I64("k", vec![1024, 1024]),
                Col::F64("latency", vec![0.25, 0.75]),
                Col::F64("power", vec![40.0, 80.0]),
            ],
        );
        write_parquet(
            &data.join("scale_matrix_perf.parquet"),
            &[
                Col::Str("quant_dtype", vec!["fp8", "fp8"]),
                Col::I64("m", vec![128, 256]),
                Col::I64("k", vec![1024, 1024]),
                Col::F64("latency", vec![0.125, 0.375]),
                Col::F64("power", vec![20.0, 60.0]),
            ],
        );
        let db = PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0").expect("db must load");
        assert!(!db.gemm.has_direct_fp8_static().unwrap());

        let op = GemmOp {
            name: "g".to_string(),
            scale_factor: 2.0,
            n: 1024,
            k: 1024,
            quant_mode: GemmQuantMode::Fp8Static,
            scale_num_tokens: 1,
            low_precision_input: true,
            seq_split: 1,
            below_grid_sol: false,
        };
        let r = op.query(&db, 192, None).expect("fp8_static query");
        assert!(
            (r.latency_ms - 2.5).abs() < 1e-9,
            "latency {}",
            r.latency_ms
        );
        assert!(
            (r.energy_wms - 520.0).abs() < 1e-9 * 520.0,
            "energy {}",
            r.energy_wms
        );
        assert_eq!(r.source, Source::Estimated);

        let op1 = GemmOp {
            name: "g1".to_string(),
            scale_factor: 1.0,
            n: 1024,
            k: 1024,
            quant_mode: GemmQuantMode::Fp8Static,
            scale_num_tokens: 1,
            low_precision_input: false,
            seq_split: 1,
            below_grid_sol: false,
        };
        let r1 = op1.query(&db, 192, None).expect("fp8_static query");
        assert!(
            (r1.latency_ms - 1.5).abs() < 1e-9,
            "latency {}",
            r1.latency_ms
        );
        assert!(
            (r1.energy_wms - 270.0).abs() < 1e-9 * 270.0,
            "energy {}",
            r1.energy_wms
        );
    }

    /// SOL mode (and the retired SOL_FULL alias) returns the pure roofline
    /// tagged `Source::Sol` with zero energy — Python `_query_gemm_table` /
    /// `_query_compute_scale_table` / `_query_scale_matrix_table` SOL branches.
    #[test]
    fn gemm_sol_mode_returns_roofline_with_sol_source() {
        let mut db = b200_vllm_db();
        db.database_mode = DatabaseMode::Sol;
        let quant = GemmQuantMode::Bfloat16;
        let op = GemmOp::new("gemm", 4096, 4096, quant);
        let result = op.query(&db, 512, None).expect("sol query");
        let tc_flops = quant_tc_flops(&db.system_spec, quant.mapping()).expect("flops");
        let expected =
            gemm_sol_latency_ms_with_flops(&db.system_spec, quant, tc_flops, 512.0, 4096.0, 4096.0);
        assert_eq!(result.latency_ms, expected);
        assert_eq!(result.source, Source::Sol);
        assert_eq!(result.energy_wms, 0.0);

        db.database_mode = DatabaseMode::SolFull;
        let alias = op.query(&db, 512, None).expect("sol_full query");
        assert_eq!(alias.latency_ms, expected);
        assert_eq!(alias.source, Source::Sol);

        // compute_scale / scale_matrix SOL arms: pure mem-bandwidth bounds
        // (`2 m k / bw` and `3 m k / bw`).
        let mem_bw = db.system_spec.gpu.mem_bw;
        let cs = query_compute_scale_table(&db, quant, 512, 4096).expect("cs");
        assert_eq!(cs.latency_ms, 2.0 * 512.0 * 4096.0 / mem_bw * 1000.0);
        assert_eq!(cs.source, Source::Sol);
        let sm = query_scale_matrix_table(&db, quant, 512, 4096).expect("sm");
        assert_eq!(sm.latency_ms, 3.0 * 512.0 * 4096.0 / mem_bw * 1000.0);
        assert_eq!(sm.source, Source::Sol);
    }

    /// Synthetic GEMM rows "S", all at `n = k = 1024`, `(quant, m, latency
    /// ms, power W)`. The m-curve lerps latency and power, and energy =
    /// power x latency. At m=192: fp8 gives latency 2.0, power 150, energy
    /// 300; fp8_static gives latency 1.0, power 140, energy 140.
    const ROWS_S_GEMM: [(&str, i64, f64, f64); 4] = [
        ("fp8", 128, 1.0, 100.0),
        ("fp8", 256, 3.0, 200.0),
        ("fp8_static", 128, 0.5, 120.0),
        ("fp8_static", 256, 1.5, 160.0),
    ];

    /// Write the rows of S whose quant is in `quants` as `gemm_perf.parquet`.
    fn write_rows_s_gemm(data: &std::path::Path, quants: &[&str]) {
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
        let rows: Vec<_> = ROWS_S_GEMM
            .iter()
            .filter(|row| quants.contains(&row.0))
            .collect();
        write_parquet(
            &data.join("gemm_perf.parquet"),
            &[
                Col::Str("gemm_dtype", rows.iter().map(|r| r.0).collect()),
                Col::I64("m", rows.iter().map(|r| r.1).collect()),
                Col::I64("n", vec![1024; rows.len()]),
                Col::I64("k", vec![1024; rows.len()]),
                Col::F64("latency", rows.iter().map(|r| r.2).collect()),
                Col::F64("power", rows.iter().map(|r| r.3).collect()),
            ],
        );
    }

    /// Write an fp8 `scale_matrix_perf.parquet` at k = 1024 from `(m,
    /// latency ms, power W)` rows.
    fn write_scale_matrix(data: &std::path::Path, rows: [(i64, f64, f64); 2]) {
        use crate::perf_database::energy_test_fixtures::{Col, write_parquet};
        write_parquet(
            &data.join("scale_matrix_perf.parquet"),
            &[
                Col::Str("quant_dtype", vec!["fp8"; 2]),
                Col::I64("m", rows.iter().map(|r| r.0).collect()),
                Col::I64("k", vec![1024; 2]),
                Col::F64("latency", rows.iter().map(|r| r.1).collect()),
                Col::F64("power", rows.iter().map(|r| r.2).collect()),
            ],
        );
    }

    /// Fixture D: all rows of S, and the scale_matrix rows of S (fp8 m=128/256
    /// @ k=1024: lat 0.125/0.375, power 20/60; at m=192 the lerp gives
    /// latency 0.25, power 40, energy 10). No `computescale_perf.parquet`: a
    /// path that still asks for compute_scale fails.
    fn fixture_d_db(tmp: &tempfile::TempDir) -> PerfDatabase {
        use crate::perf_database::energy_test_fixtures::write_energy_systems_root;
        let data = write_energy_systems_root(tmp.path());
        write_rows_s_gemm(&data, &["fp8", "fp8_static"]);
        write_scale_matrix(&data, [(128, 0.125, 20.0), (256, 0.375, 60.0)]);
        PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0").expect("db must load")
    }

    /// The fp8 / fp8_static GEMM roofline at (m, n, k) = (192, 1024, 1024)
    /// on the synthetic system of `write_energy_systems_root` (mem_bw
    /// 7.7e12 B/s, fp8_tc_flops 4.5e15), computed by hand:
    ///
    /// - memory: 1 B/element x (m*n + m*k + n*k)
    ///   = 196608 + 196608 + 1048576 = 1441792 B;
    ///   1441792 / 7.7e12 x 1000 = 1.8724571428571426e-4 ms.
    /// - math: 2 x 192 x 1024 x 1024 / 4.5e15 x 1000 = 8.947848533e-5 ms.
    ///
    /// The roofline is the larger term: memory bound, 1.8724571428571426e-4 ms.
    const HAND_ROOFLINE_192_1024_1024_MS: f64 = 1.8724571428571426e-4;

    fn assert_hand_roofline(got: f64, what: &str) {
        let want = HAND_ROOFLINE_192_1024_1024_MS;
        assert!(
            (got - want).abs() <= 1e-9 * want,
            "{what} {got} != hand roofline {want}"
        );
    }

    fn fp8_static_op(name: &str, scale_factor: f64, low_precision_input: bool) -> GemmOp {
        GemmOp {
            name: name.to_string(),
            scale_factor,
            low_precision_input,
            ..GemmOp::new(name, 1024, 1024, GemmQuantMode::Fp8Static)
        }
    }

    /// Direct path, hand oracle on fixture D at x=192. The roofline floor of
    /// the energy test spec at this shape is below 1e-3 ms, so it does not
    /// fire.
    #[test]
    fn gemm_fp8_static_direct_rows_match_hand_oracle() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db = fixture_d_db(&tmp);
        assert!(db.gemm.has_direct_fp8_static().unwrap());

        // "g": scale_factor 2.0, low_precision_input true. The fp8_static
        // row (1.0, 140) minus scale_matrix (0.25, 10), times 2:
        // latency (1.0 - 0.25) * 2 = 1.5; energy (140 - 10) * 2 = 260;
        // Estimated after a subtraction.
        let r = fp8_static_op("g", 2.0, true)
            .query(&db, 192, None)
            .expect("fp8_static query");
        assert!(
            (r.latency_ms - 1.5).abs() < 1e-9,
            "latency {}",
            r.latency_ms
        );
        assert!(
            (r.energy_wms - 260.0).abs() < 1e-9 * 260.0,
            "energy {}",
            r.energy_wms
        );
        assert_eq!(r.source, Source::Estimated);

        // "g1": scale_factor 1.0, low_precision_input false. The row as is:
        // no subtraction, no floor, the row's source.
        let r1 = fp8_static_op("g1", 1.0, false)
            .query(&db, 192, None)
            .expect("fp8_static query");
        let row = db
            .gemm
            .query(GemmQuantMode::Fp8Static, 192, 1024, 1024)
            .unwrap();
        assert_eq!(r1.latency_ms.to_bits(), row.latency.to_bits()); // 1.0
        assert_eq!(r1.energy_wms.to_bits(), row.energy.to_bits()); // 140.0
        assert!(
            (r1.latency_ms - 1.0).abs() < 1e-9,
            "latency {}",
            r1.latency_ms
        );
        assert!((r1.energy_wms - 140.0).abs() < 1e-9 * 140.0);
        assert_eq!(r1.source, Source::Silicon);

        // fp8 on the same table is unchanged: the fp8 rows of S at m=192
        // give latency 2.0, energy 300, Silicon.
        let f = GemmOp::new("f", 1024, 1024, GemmQuantMode::Fp8)
            .query(&db, 192, None)
            .expect("fp8 query");
        assert!(
            (f.latency_ms - 2.0).abs() < 1e-9,
            "latency {}",
            f.latency_ms
        );
        assert!((f.energy_wms - 300.0).abs() < 1e-9 * 300.0);
        assert_eq!(f.source, Source::Silicon);
    }

    /// Direct path, low-precision input, and a scale_matrix price larger
    /// than the GEMM row: the latency stops at the GEMM roofline.
    #[test]
    fn gemm_fp8_static_direct_low_precision_input_keeps_the_roofline_floor() {
        use crate::perf_database::energy_test_fixtures::write_energy_systems_root;
        let tmp = tempfile::tempdir().expect("tmpdir");
        let data = write_energy_systems_root(tmp.path());
        // gemm_perf: fp8_static rows of S only (m=192: latency 1.0, energy 140).
        write_rows_s_gemm(&data, &["fp8_static"]);
        // scale_matrix: fp8 m=128/256 @ k=1024, lat 2.0/2.0, power 10/10
        // (m=192: latency 2.0, energy 2.0 * 10 = 20).
        write_scale_matrix(&data, [(128, 2.0, 10.0), (256, 2.0, 10.0)]);
        let db = PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0").expect("db must load");

        // At x=192: 1.0 - 2.0 < floor, so the result is the roofline.
        let r = fp8_static_op("g", 1.0, true)
            .query(&db, 192, None)
            .expect("fp8_static query");
        let tc = quant_tc_flops(&db.system_spec, GemmQuantMode::Fp8Static.mapping()).unwrap();
        let floor = gemm_sol_with_flops(
            &db.system_spec,
            GemmQuantMode::Fp8Static,
            tc,
            192.0,
            1024.0,
            1024.0,
        )
        .time_ms();
        assert!(floor > 0.0);
        assert_hand_roofline(floor, "floor");
        assert_eq!(r.latency_ms, floor);
        assert_hand_roofline(r.latency_ms, "floored latency");
        assert_eq!(r.source, Source::Estimated);
        // Energy keeps the plain subtraction: 140 - 2.0 * 10 = 120.
        assert!(
            (r.energy_wms - 120.0).abs() < 1e-9 * 120.0,
            "energy {}",
            r.energy_wms
        );
    }

    /// Direct path with no scale_matrix table: a low-precision input is a
    /// loud missing-data error, not a silent skip of the subtraction.
    #[test]
    fn gemm_fp8_static_direct_low_precision_input_without_scale_matrix_errors() {
        use crate::perf_database::energy_test_fixtures::write_energy_systems_root;
        let tmp = tempfile::tempdir().expect("tmpdir");
        let data = write_energy_systems_root(tmp.path());
        // gemm_perf: fp8_static rows of S only; no computescale, no
        // scale_matrix file.
        write_rows_s_gemm(&data, &["fp8_static"]);
        let db = PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0").expect("db must load");

        // low_precision_input false: the row of S at m=192 (latency 1.0).
        let ok = fp8_static_op("g1", 1.0, false)
            .query(&db, 192, None)
            .expect("no scale table is needed");
        assert!(
            (ok.latency_ms - 1.0).abs() < 1e-9,
            "latency {}",
            ok.latency_ms
        );
        assert_eq!(ok.source, Source::Silicon);

        // low_precision_input true: the scale_matrix table is missing.
        match fp8_static_op("g", 1.0, true).query(&db, 192, None) {
            Err(e) => assert!(e.is_missing_perf_data(), "error {e:?}"),
            Ok(r) => panic!("expected a missing-data error, got {r:?}"),
        }
    }

    /// EMPIRICAL keeps a separate util grid per mode on the direct path;
    /// SOL reads no table and stays as today (derived path, Estimated).
    #[test]
    fn gemm_fp8_static_direct_rows_in_empirical_and_sol_modes() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db = fixture_d_db(&tmp);

        // EMPIRICAL: price fp8 first, then fp8_static, both at x=128 (an
        // exact sample, so SOL / (SOL / lat) returns the sample latency:
        // fp8 1.0, fp8_static 0.5).
        let emp = db
            .silicon_view()
            .with_mode(DatabaseMode::Empirical, TransferPolicy::ALL);
        db.reset_provenance();
        let f = GemmOp::new("f", 1024, 1024, GemmQuantMode::Fp8)
            .query(&emp, 128, None)
            .expect("fp8 empirical query");
        assert!(
            (f.latency_ms - 1.0).abs() < 1e-9,
            "latency {}",
            f.latency_ms
        );
        let s = GemmOp::new("s", 1024, 1024, GemmQuantMode::Fp8Static)
            .query(&emp, 128, None)
            .expect("fp8_static empirical query");
        // Own grid, not the fp8 grid (a shared cache key would give 1.0).
        assert!(
            (s.latency_ms - 0.5).abs() < 1e-9,
            "latency {}",
            s.latency_ms
        );
        assert_eq!(s.source, Source::Empirical);
        assert_eq!(
            emp.worst_provenance(),
            util_empirical::ProvenanceTier::Empirical
        );

        // SOL: no table is read; the result is the GEMM roofline with the
        // tag of today.
        let sol = db
            .silicon_view()
            .with_mode(DatabaseMode::Sol, TransferPolicy::ALL);
        let r = GemmOp::new("s", 1024, 1024, GemmQuantMode::Fp8Static)
            .query(&sol, 192, None)
            .expect("fp8_static sol query");
        let tc = quant_tc_flops(&db.system_spec, GemmQuantMode::Fp8Static.mapping()).unwrap();
        let roofline = gemm_sol_with_flops(
            &db.system_spec,
            GemmQuantMode::Fp8Static,
            tc,
            192.0,
            1024.0,
            1024.0,
        )
        .time_ms();
        assert_hand_roofline(roofline, "roofline");
        assert_eq!(r.latency_ms, roofline);
        assert_hand_roofline(r.latency_ms, "SOL latency");
        assert_eq!(r.source, Source::Estimated);
    }

    /// Coverage rule of the direct rows. When the table has `fp8_static` rows,
    /// those rows alone give the coverage of the mode, also for a shape that
    /// `fp8` rows cover. A far shape is a coverage error in SILICON and an
    /// empirical estimate in HYBRID, while `fp8` keeps its own row there.
    ///
    /// Synthetic table: `fp8` rows at the far site (m=128, n=k=8192), with
    /// the overhead tables, and `fp8_static` rows on a small grid
    /// (n=k=1024, m=128 and 256).
    #[test]
    fn gemm_fp8_static_rows_alone_give_the_coverage_of_the_mode() {
        use crate::perf_database::energy_test_fixtures::{
            Col, write_energy_systems_root, write_parquet,
        };
        let tmp = tempfile::tempdir().expect("tmpdir");
        let data = write_energy_systems_root(tmp.path());
        write_parquet(
            &data.join("gemm_perf.parquet"),
            &[
                Col::Str("gemm_dtype", vec!["fp8", "fp8", "fp8_static", "fp8_static"]),
                Col::I64("m", vec![128, 256, 128, 256]),
                Col::I64("n", vec![8192, 8192, 1024, 1024]),
                Col::I64("k", vec![8192, 8192, 1024, 1024]),
                Col::F64("latency", vec![1.0, 3.0, 0.5, 1.5]),
            ],
        );
        for (file, lat) in [
            ("computescale_perf.parquet", [0.25, 0.75]),
            ("scale_matrix_perf.parquet", [0.125, 0.375]),
        ] {
            write_parquet(
                &data.join(file),
                &[
                    Col::Str("quant_dtype", vec!["fp8", "fp8"]),
                    Col::I64("m", vec![128, 256]),
                    Col::I64("k", vec![8192, 8192]),
                    Col::F64("latency", lat.to_vec()),
                ],
            );
        }
        let load = |mode| {
            PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0")
                .expect("db must load")
                .with_mode(mode, TransferPolicy::ALL)
        };

        // SILICON: the static rows sit at n=k=1024. The far site n=k=8192 is
        // 3 octaves away on each axis, so the distance is 3 * sqrt(2) = 4.24,
        // above the gate of 2.0. The fp8 rows at the far site do not help.
        let silicon = load(DatabaseMode::Silicon);
        match GemmOp::new("s", 8192, 8192, GemmQuantMode::Fp8Static).query(&silicon, 128, None) {
            Err(AicError::InterpolationCoverage(_)) => {}
            other => panic!("expected a coverage error, got {other:?}"),
        }

        // fp8 at the same shape reads its own row: latency 1.0, Silicon.
        let f = GemmOp::new("f", 8192, 8192, GemmQuantMode::Fp8)
            .query(&silicon, 128, None)
            .expect("fp8 query");
        assert!(
            (f.latency_ms - 1.0).abs() < 1e-9,
            "latency {}",
            f.latency_ms
        );
        assert_eq!(f.source, Source::Silicon);

        // HYBRID: the silicon miss is estimated from the own fp8_static util
        // grid. The result is tagged Empirical. Its value comes from an
        // interpolated utilization, so the tag and a positive latency are
        // pinned, not the value. An estimate from the fp8 grid gives the fp8
        // row at this shape, 1.0, so the result must differ from it.
        let hybrid = load(DatabaseMode::Hybrid);
        let h = GemmOp::new("s", 8192, 8192, GemmQuantMode::Fp8Static)
            .query(&hybrid, 128, None)
            .expect("hybrid fp8_static query");
        assert_eq!(h.source, Source::Empirical);
        assert!(h.latency_ms > 0.0, "latency {}", h.latency_ms);
        assert!(
            (h.latency_ms - 1.0).abs() > 1e-3,
            "latency {}",
            h.latency_ms
        );
    }

    /// fp8_static rows are never a transfer-ladder reference. Synthetic
    /// discrimination vehicle: fp8_static rows come FIRST in file order with
    /// latencies 1000x the bfloat16 rows. fp8, sq and fp8_block have no rows
    /// and share the fp8_static profile (1, 2); they must borrow bfloat16
    /// through xprofile (a fp8_static borrow would land near 1000).
    #[test]
    fn gemm_transfer_ladder_never_borrows_fp8_static_rows() {
        use crate::perf_database::energy_test_fixtures::{
            Col, write_energy_systems_root, write_parquet,
        };
        let tmp = tempfile::tempdir().expect("tmpdir");
        let data = write_energy_systems_root(tmp.path());
        write_parquet(
            &data.join("gemm_perf.parquet"),
            &[
                Col::Str(
                    "gemm_dtype",
                    vec!["fp8_static", "fp8_static", "bfloat16", "bfloat16"],
                ),
                Col::I64("m", vec![64, 128, 64, 128]),
                Col::I64("n", vec![64, 64, 64, 64]),
                Col::I64("k", vec![64, 64, 64, 64]),
                Col::F64("latency", vec![1000.0, 2000.0, 1.0, 2.0]),
            ],
        );
        let db = PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0")
            .expect("db")
            .with_mode(DatabaseMode::Hybrid, TransferPolicy::ALL);
        for quant in [
            GemmQuantMode::Fp8,
            GemmQuantMode::Sq,
            GemmQuantMode::Fp8Block,
        ] {
            db.reset_provenance();
            let r = query_gemm_table(&db, quant, 64, 64, 64).expect("ladder query");
            assert_eq!(r.source, Source::Empirical, "{quant:?}");
            assert_eq!(
                db.worst_provenance(),
                util_empirical::ProvenanceTier::XProfile,
                "{quant:?}"
            );
            assert!(
                r.latency_ms < 100.0,
                "{quant:?}: must borrow the bfloat16 donor (~O(1) ms), got {} \
                 (a fp8_static donor would be ~O(1000))",
                r.latency_ms
            );
        }
    }

    /// HYBRID on a table with direct rows (fixture D). A silicon hit reads
    /// the fp8_static rows; a silicon miss is estimated from the own
    /// fp8_static util grid, never from the fp8 grid.
    #[test]
    fn gemm_fp8_static_direct_rows_in_hybrid_mode() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db = fixture_d_db(&tmp);
        let hyb = db
            .silicon_view()
            .with_mode(DatabaseMode::Hybrid, TransferPolicy::ALL);

        // Hit at (192, 1024, 1024): the fp8_static rows of S lerp to latency
        // 1.0 and energy 140 (see ROWS_S_GEMM); no subtraction, row source.
        hyb.reset_provenance();
        let hit = fp8_static_op("g1", 1.0, false)
            .query(&hyb, 192, None)
            .expect("hybrid hit");
        assert!(
            (hit.latency_ms - 1.0).abs() < 1e-9,
            "latency {}",
            hit.latency_ms
        );
        assert!((hit.energy_wms - 140.0).abs() < 1e-9 * 140.0);
        assert_eq!(hit.source, Source::Silicon);
        assert_eq!(
            hyb.worst_provenance(),
            util_empirical::ProvenanceTier::Silicon
        );

        // Miss at (192, 32, 32): no row has n = k = 32. Every fp8_static row
        // of S is half the fp8 row at the same (m, n, k) (0.5 vs 1.0, 1.5 vs
        // 3.0), and the two modes share one roofline, so the fp8_static util
        // (roofline / latency) is twice the fp8 util at every sample. The own
        // grid estimate (roofline / util) is therefore half the fp8 estimate;
        // a borrowed or shared fp8 grid would give the fp8 estimate itself.
        hyb.reset_provenance();
        let miss = GemmOp::new("s", 32, 32, GemmQuantMode::Fp8Static)
            .query(&hyb, 192, None)
            .expect("hybrid miss");
        assert_eq!(miss.source, Source::Empirical);
        assert_eq!(
            hyb.worst_provenance(),
            util_empirical::ProvenanceTier::Empirical
        );
        let fp8 = GemmOp::new("f", 32, 32, GemmQuantMode::Fp8)
            .query(&hyb, 192, None)
            .expect("fp8 hybrid miss");
        assert_eq!(fp8.source, Source::Empirical);
        assert!(
            (2.0 * miss.latency_ms - fp8.latency_ms).abs() <= 1e-12 * fp8.latency_ms,
            "fp8_static {} is not half of fp8 {}",
            miss.latency_ms,
            fp8.latency_ms
        );
    }

    /// SOL_FULL on a table with direct rows reads no table: the result is the
    /// GEMM roofline with the decomposition and the tag of today.
    #[test]
    fn gemm_fp8_static_direct_rows_in_sol_full_mode() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        let db = fixture_d_db(&tmp);
        assert!(db.gemm.has_direct_fp8_static().unwrap());
        let sol_full = db
            .silicon_view()
            .with_mode(DatabaseMode::SolFull, TransferPolicy::ALL);
        for low_precision_input in [false, true] {
            let r = fp8_static_op("s", 1.0, low_precision_input)
                .query(&sol_full, 192, None)
                .expect("fp8_static sol_full query");
            assert_hand_roofline(r.latency_ms, "SOL_FULL latency");
            let sol = r.sol.expect("SOL_FULL keeps the decomposition");
            assert_hand_roofline(sol.time_ms(), "SOL_FULL decomposition");
            assert_eq!(r.source, Source::Estimated);
        }
    }

    /// Invariant: adding fp8_static rows to a table changes no answer of
    /// another mode. Table X has bfloat16 rows and real same-profile siblings
    /// of fp8_static (fp8_block, and fp8 in one variant); table Y is X with
    /// fp8_static rows FIRST in file order at 1000x latency, so a leak into a
    /// ladder reference or a shared key would move an answer by far.
    #[test]
    fn gemm_fp8_static_rows_change_no_answer_of_another_mode() {
        use crate::perf_database::energy_test_fixtures::{
            Col, write_energy_systems_root, write_parquet,
        };
        const BALANCED: TransferPolicy = TransferPolicy {
            xshape: true,
            xquant: true,
            xprofile: false,
            xop: false,
        };
        type Row = (&'static str, i64, f64);
        let siblings: [Row; 6] = [
            ("bfloat16", 64, 1.0),
            ("bfloat16", 128, 2.0),
            ("fp8_block", 64, 3.0),
            ("fp8_block", 128, 5.0),
            ("fp8", 64, 7.0),
            ("fp8", 128, 9.0),
        ];
        let statics: [Row; 2] = [("fp8_static", 64, 1000.0), ("fp8_static", 128, 2000.0)];
        let write = |dir: &std::path::Path, rows: &[Row]| {
            let data = write_energy_systems_root(dir);
            write_parquet(
                &data.join("gemm_perf.parquet"),
                &[
                    Col::Str("gemm_dtype", rows.iter().map(|r| r.0).collect()),
                    Col::I64("m", rows.iter().map(|r| r.1).collect()),
                    Col::I64("n", vec![64; rows.len()]),
                    Col::I64("k", vec![64; rows.len()]),
                    Col::F64("latency", rows.iter().map(|r| r.2).collect()),
                ],
            );
            PerfDatabase::load(dir, "testsys", "vllm", "1.0").expect("db must load")
        };
        // One answer as text: bit patterns of the numbers, tag, provenance,
        // or the error text without the folder and the list of quants.
        let show = |r: Result<PerformanceResult, AicError>,
                    tier: util_empirical::ProvenanceTier,
                    root: &std::path::Path| {
            match r {
                Ok(v) => format!(
                    "{:x} {:x} {:?} {:?} {tier:?}",
                    v.latency_ms.to_bits(),
                    v.energy_wms.to_bits(),
                    v.source,
                    v.sol.map(|s| (s.math_ms.to_bits(), s.mem_ms.to_bits())),
                ),
                Err(e) => {
                    let text = format!("{e:?}").replace(&root.display().to_string(), "<root>");
                    let text = text.split("; available:").next().unwrap_or("").to_string();
                    format!("Err {text}")
                }
            }
        };
        let mut compared = 0;
        for with_fp8 in [true, false] {
            let x_rows: Vec<Row> = siblings
                .iter()
                .copied()
                .filter(|r| with_fp8 || r.0 != "fp8")
                .collect();
            let y_rows: Vec<Row> = statics
                .iter()
                .copied()
                .chain(x_rows.iter().copied())
                .collect();
            let (tmp_x, tmp_y) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
            let x = write(tmp_x.path(), &x_rows);
            let y = write(tmp_y.path(), &y_rows);
            assert!(!x.gemm.has_direct_fp8_static().unwrap());
            assert!(y.gemm.has_direct_fp8_static().unwrap());
            for policy in [TransferPolicy::ALL, BALANCED, TransferPolicy::OFF] {
                for mode in [
                    DatabaseMode::Silicon,
                    DatabaseMode::Hybrid,
                    DatabaseMode::Empirical,
                    DatabaseMode::Sol,
                    DatabaseMode::SolFull,
                ] {
                    let vx = x.silicon_view().with_mode(mode, policy);
                    let vy = y.silicon_view().with_mode(mode, policy);
                    for quant in [
                        GemmQuantMode::Fp8,
                        GemmQuantMode::Sq,
                        GemmQuantMode::Fp8Block,
                        GemmQuantMode::Fp8Ootb,
                        GemmQuantMode::Bfloat16,
                        GemmQuantMode::Int4Wo,
                    ] {
                        for (m, n, k) in [(64u32, 64u32, 64u32), (100, 64, 64), (64, 128, 128)] {
                            let op = GemmOp::new("g", n, k, quant);
                            vx.reset_provenance();
                            let rx =
                                show(op.query(&vx, m, None), vx.worst_provenance(), tmp_x.path());
                            vy.reset_provenance();
                            let ry =
                                show(op.query(&vy, m, None), vy.worst_provenance(), tmp_y.path());
                            assert_eq!(
                                rx, ry,
                                "with_fp8={with_fp8} {policy:?} {mode:?} {quant:?} ({m}, {n}, {k})"
                            );
                            compared += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(compared, 2 * 3 * 5 * 6 * 3);
    }
}
