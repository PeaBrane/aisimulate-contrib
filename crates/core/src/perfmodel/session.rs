// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-phase op-list execution primitives.
//!
//! Mirrors `aisim.sdk.backends.base_backend`: iterates a context /
//! generation op list to compute per-phase latency, and composes the mix-step
//! latency exactly the way Python's `_get_mix_step_latency` does — one combined
//! non-attention pass plus per-phase attention. The compiled
//! [`crate::perfmodel::engine::Engine`] drives these free functions over the precompiled
//! op lists from an [`crate::perfmodel::engine::spec::EngineSpec`].

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::common::error::AicError;
use crate::memo::Memo;
use crate::operators::base::PerformanceResult;
use crate::operators::op::MemoField;
use crate::operators::{Dsv41StageOp, Op, RuntimeContext};
use crate::perf_database::PerfDatabase;

/// The engine op list an [`OpMemo`] entry belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpList {
    Context = 0,
    Generation = 1,
}

/// Results of ops whose result depends on one runtime field
/// ([`Op::memo_field`]), kept across walks of one engine's op lists. Such a
/// result is a pure function of the op and that field, so a walk at a new KV
/// shape re-prices only the ops that read it. Debug builds re-price every hit.
pub(crate) struct OpMemo {
    /// Each op's field by list and index, computed once from the lists.
    fields: [Box<[Option<MemoField>]>; 2],
    /// Keyed by list, index and field value ([`Self::key`]).
    results: Memo<u64, PerformanceResult>,
    /// A generation stage's children that read only the token count, in
    /// child order with `None` for the others; keyed like `results`.
    stage_children: Memo<u64, Arc<[Option<PerformanceResult>]>>,
}

impl OpMemo {
    /// A memo for walks of `context_ops` and `generation_ops`; it starts
    /// over once it holds `capacity` results.
    pub(crate) fn new(capacity: usize, context_ops: &[Op], generation_ops: &[Op]) -> Self {
        let fields = |ops: &[Op]| ops.iter().map(Op::memo_field).collect();
        Self {
            fields: [fields(context_ops), fields(generation_ops)],
            results: Memo::flat(capacity).verifying_hits(),
            // One entry per generation stage and token count, about 1.5 KB each.
            stage_children: Memo::flat(4096).verifying_hits(),
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.results.len() + self.stage_children.len()
    }

    /// The entry key of the op at `index` of `list` with its field at `value`.
    fn key(list: OpList, index: usize, value: u32) -> u64 {
        debug_assert!(index < 1 << 31, "op index {index}");
        (list as u64) << 63 | (index as u64) << 32 | u64::from(value)
    }

    /// The result at `ctx` of the op at `index` of `list`: `query`'s, or the
    /// one an earlier walk stored for the same value of the field it reads.
    pub(crate) fn query(
        &self,
        db: &PerfDatabase,
        list: OpList,
        index: usize,
        ctx: &RuntimeContext,
        query: impl FnOnce() -> Result<PerformanceResult, AicError>,
    ) -> Result<PerformanceResult, AicError> {
        let Some(field) = self.fields[list as usize].get(index).copied().flatten() else {
            return query();
        };
        self.results
            .get_or_price(db, Self::key(list, index, field.of(ctx)), query)
    }

    /// [`Dsv41StageOp::query`] of the generation stage at `index` of `list`,
    /// which queries every child at `ctx` itself: the children that read
    /// only the token count come from the walk that first reached this count.
    fn query_stage(
        &self,
        db: &PerfDatabase,
        list: OpList,
        index: usize,
        stage: &Dsv41StageOp,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        let key = Self::key(list, index, ctx.num_tokens);
        // A child that fails is left out and queried again below, so the
        // stage still reports its first failing child.
        let shared = self.stage_children.get_or_price(db, key, || {
            let children = stage.children.iter().map(|op| match op.memo_field() {
                Some(MemoField::NumTokens) => op.query(db, ctx).ok(),
                _ => None,
            });
            Ok::<_, AicError>(children.collect::<Arc<[_]>>())
        })?;
        stage.query_with(ctx, |i, op, child_ctx| match &shared[i] {
            Some(result) => Ok(result.clone()),
            None => op.query(db, child_ctx),
        })
    }
}

/// [`OpMemo::query`] through `op_memo` when there is one, and
/// [`OpMemo::query_stage`] for a DeepSeek-V4.1 generation stage.
fn shared_query(
    op_memo: Option<&OpMemo>,
    db: &PerfDatabase,
    list: OpList,
    index: usize,
    op: &Op,
    ctx: &RuntimeContext,
) -> Result<PerformanceResult, AicError> {
    match (op_memo, op) {
        (Some(op_memo), Op::Dsv41Stage(stage)) if !stage.is_context => {
            op_memo.query_stage(db, list, index, stage, ctx)
        }
        (Some(op_memo), _) => op_memo.query(db, list, index, ctx, || op.query(db, ctx)),
        (None, _) => op.query(db, ctx),
    }
}

/// Build-time facts about one immutable op list. Passes use them to skip
/// repeated work; every op still reports its own name and value.
#[derive(Debug)]
pub(crate) struct OpListPlan {
    /// Ops sharing a slot price identically, so a pass queries the first one
    /// it visits and reuses that result. `None` always queries.
    query_slot: Vec<Option<usize>>,
    query_slots: usize,
    /// First-encounter slot of each op's name, for folding without name scans.
    name_slot: Vec<usize>,
    /// Distinct op names indexed by name slot.
    names: Arc<[String]>,
}

impl OpListPlan {
    pub(crate) fn new(ops: &[Op]) -> Self {
        let mut slots = HashMap::<&str, usize>::new();
        let mut names = Vec::new();
        let name_slot = ops
            .iter()
            .map(|op| {
                *slots.entry(op.name()).or_insert_with(|| {
                    names.push(op.name().to_owned());
                    names.len() - 1
                })
            })
            .collect();
        let mut identities = HashMap::<String, usize>::new();
        let query_slot = ops
            .iter()
            .map(|op| {
                pricing_identity(op).map(|key| {
                    let next = identities.len();
                    *identities.entry(key).or_insert(next)
                })
            })
            .collect();
        Self {
            query_slot,
            query_slots: identities.len(),
            name_slot,
            names: names.into(),
        }
    }

    /// Name slot of the op at `index`; slots number names in first-encounter order.
    pub(crate) fn name_slot(&self, index: usize) -> usize {
        self.name_slot[index]
    }

    /// Number of distinct op names.
    pub(crate) fn names(&self) -> usize {
        self.names.len()
    }

    /// Distinct op names indexed by name slot, shared by every pass.
    pub(crate) fn slot_names(&self) -> &Arc<[String]> {
        &self.names
    }

    pub(crate) fn query_slot(&self, index: usize) -> Option<usize> {
        self.query_slot[index]
    }

    pub(crate) fn query_slots(&self) -> usize {
        self.query_slots
    }
}

/// Pricing key of a GLM-5.3-Flash boundary or a DeepSeek-V4.1 stage without
/// its per-layer display names. These names, including those of GLM generic
/// children, never select data, so identical layers price identically; debug
/// builds re-price every shared query to check this ([`QueryMemo::query`]).
/// A DeepSeek-V4.1 stage's children carry no layer in their names and keep
/// them. Other families return `None` and are always queried.
fn pricing_identity(op: &Op) -> Option<String> {
    if !matches!(
        op,
        Op::Glm53Attention(_)
            | Op::Glm53Mhc(_)
            | Op::Glm53Router(_)
            | Op::Glm53Ffn(_)
            | Op::Glm53Primitive(_)
            | Op::Dsv41Stage(_)
    ) {
        return None;
    }
    // Keep the name-selected session roles (filters and token-count policy).
    let roles = (
        op.is_context_attention(),
        op.is_generation_attention(),
        op.is_logits_gemm(),
        op.name().starts_with("draft_"),
    );
    let clear = |ops: &mut [Op]| ops.iter_mut().for_each(|op| op.set_name(String::new()));
    let mut op = op.clone();
    op.set_name(String::new());
    match &mut op {
        Op::Glm53Attention(o) => clear(&mut o.measured),
        Op::Glm53Ffn(o) => {
            clear(&mut o.children);
            clear(&mut o.measured);
        }
        Op::Glm53Primitive(o) => {
            clear(&mut o.children);
            clear(&mut o.measured);
        }
        _ => {}
    }
    let body = serde_json::to_string(&op).ok()?;
    Some(format!("{roles:?}{body}"))
}

/// Results of one pass, keyed by [`OpListPlan`] query slot. Every op in a
/// pass shares the pass's runtime shape, so a slot's first result is exact
/// for the rest of the slot, and [`Self::query`] lends it to every op of the
/// slot.
pub(crate) struct QueryMemo<'p> {
    plan: Option<&'p OpListPlan>,
    results: Vec<Option<PerformanceResult>>,
}

impl<'p> QueryMemo<'p> {
    pub(crate) fn new(plan: Option<&'p OpListPlan>, ops: &[Op]) -> Self {
        debug_assert!(plan.is_none_or(|plan| plan.query_slot.len() == ops.len()));
        Self {
            plan,
            results: vec![None; plan.map_or(0, OpListPlan::query_slots)],
        }
    }

    pub(crate) fn query(
        &mut self,
        index: usize,
        query: impl FnOnce() -> Result<PerformanceResult, AicError>,
    ) -> Result<Cow<'_, PerformanceResult>, AicError> {
        let Some(slot) = self.plan.and_then(|plan| plan.query_slot(index)) else {
            return query().map(Cow::Owned);
        };
        let result = match &mut self.results[slot] {
            Some(result) => {
                #[cfg(debug_assertions)]
                {
                    let repriced = query();
                    assert!(
                        repriced.as_ref().is_ok_and(|repriced| repriced == &*result),
                        "op {index} shares query slot {slot} but prices as {repriced:?}, not {result:?}"
                    );
                }
                &*result
            }
            empty => &*empty.insert(query()?),
        };
        Ok(Cow::Borrowed(result))
    }
}

/// Context-op selection for [`run_context_ops`]. Mirrors the name-based
/// filtering Python's `_get_mix_step_latency` applies to `run_static`'s
/// per-op breakdown: pass 1 keeps every op EXCEPT `"context_attention"`,
/// pass 2 keeps ONLY `"context_attention"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContextOpFilter {
    All,
    SkipContextAttention,
    OnlyContextAttention,
}

/// Python `_run_context_phase` (`base_backend.py:144`) — one pass over the
/// context op list. A free function so the compiled
/// [`crate::perfmodel::engine::Engine`] iterates one canonical body over its op list.
///
/// `effective_isl = isl - prefix` is the caller's responsibility (Python
/// computes it in `_run_context_phase` and `_run_static_breakdown`); this
/// fn takes the already-effective ISL and does NOT validate it (the Engine
/// caller performs Python's `effective_isl > 0` check before calling).
///
/// `seq_imbalance_correction_scale` mirrors Python's per-op kwarg
/// (`base_backend.py:331`) — the context-attention ops multiply their result
/// by it (`operations/attention.py:550-552`); every other op ignores it.
pub(crate) fn run_context_ops(
    ops: &[Op],
    db: &PerfDatabase,
    batch_size: u32,
    effective_isl: u32,
    prefix: u32,
    seq_imbalance_correction_scale: f64,
    filter: ContextOpFilter,
) -> Result<f64, AicError> {
    run_context_ops_with(
        ops,
        db,
        batch_size,
        effective_isl,
        prefix,
        seq_imbalance_correction_scale,
        filter,
        |_, _| {},
    )
}

/// [`run_context_ops`] with a per-op sink: `on_op` observes every queried
/// op's full [`PerformanceResult`] (the per-op FFI and the mixed-step
/// breakdown collect through it). The no-op-sink wrapper above monomorphizes
/// to the pre-sink code, so the scalar hot path pays nothing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_context_ops_with<'a>(
    ops: &'a [Op],
    db: &PerfDatabase,
    batch_size: u32,
    effective_isl: u32,
    prefix: u32,
    seq_imbalance_correction_scale: f64,
    filter: ContextOpFilter,
    mut on_op: impl FnMut(&'a Op, PerformanceResult),
) -> Result<f64, AicError> {
    run_context_ops_planned(
        ops,
        None,
        None,
        db,
        batch_size,
        effective_isl,
        prefix,
        seq_imbalance_correction_scale,
        filter,
        |_, op, result| on_op(op, result.into_owned()),
    )
}

/// [`run_context_ops_with`] over an op list with its [`OpListPlan`], reusing
/// `op_memo`'s results when given one. The sink also receives each op's index
/// in `ops`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_context_ops_planned<'a>(
    ops: &'a [Op],
    plan: Option<&OpListPlan>,
    op_memo: Option<&OpMemo>,
    db: &PerfDatabase,
    batch_size: u32,
    effective_isl: u32,
    prefix: u32,
    seq_imbalance_correction_scale: f64,
    filter: ContextOpFilter,
    mut on_op: impl FnMut(usize, &'a Op, Cow<'_, PerformanceResult>),
) -> Result<f64, AicError> {
    let mut memo = QueryMemo::new(plan, ops);
    let mut total = 0.0_f64;
    for (index, op) in ops.iter().enumerate() {
        match filter {
            ContextOpFilter::All => {}
            ContextOpFilter::SkipContextAttention if op.is_context_attention() => continue,
            ContextOpFilter::OnlyContextAttention if !op.is_context_attention() => continue,
            _ => {}
        }
        let result = memo.query(index, || {
            let ctx = context_op_runtime(
                op,
                batch_size,
                effective_isl,
                prefix,
                seq_imbalance_correction_scale,
                None,
            );
            shared_query(op_memo, db, OpList::Context, index, op, &ctx)
        })?;
        total += result.latency_ms;
        on_op(index, op, result);
    }
    Ok(total)
}

/// One context op at the context-phase shape — the single query body every
/// context walk shares. Default token count is `x = batch * effective_isl`,
/// except the logits GEMM's `x = batch` (mirroring `base_backend.py:337`);
/// `x_override` replaces it verbatim for Python orchestrations with their own
/// x policy (AFD's `_sum_latency` uses a uniform `batch * s` with NO logits
/// exception). The index-addressed `evaluate` FFI calls this directly.
pub(crate) fn query_context_op(
    op: &Op,
    db: &PerfDatabase,
    batch_size: u32,
    effective_isl: u32,
    prefix: u32,
    seq_imbalance_correction_scale: f64,
    x_override: Option<u32>,
) -> Result<PerformanceResult, AicError> {
    op.query(
        db,
        &context_op_runtime(
            op,
            batch_size,
            effective_isl,
            prefix,
            seq_imbalance_correction_scale,
            x_override,
        ),
    )
}

/// The [`RuntimeContext`] [`query_context_op`] prices `op` at.
pub(crate) fn context_op_runtime(
    op: &Op,
    batch_size: u32,
    effective_isl: u32,
    prefix: u32,
    seq_imbalance_correction_scale: f64,
    x_override: Option<u32>,
) -> RuntimeContext {
    let x = x_override.unwrap_or(if op.is_logits_gemm() {
        batch_size
    } else {
        batch_size * effective_isl
    });
    RuntimeContext {
        batch_size,
        beam_width: 1,
        s: effective_isl,
        prefix,
        num_tokens: x,
        seq_imbalance_correction_scale,
        gen_seq_imbalance_correction_scale: 1.0,
        num_image_tokens: 0,
    }
}

/// Python `_run_generation_phase` inner loop body for a single decode step
/// (`base_backend.py:185`). A free function shared by the compiled
/// [`crate::perfmodel::engine::Engine`]. Computes one step's latency at decode position
/// `kv_seq_tokens` (= Python `s = isl + i + 1`); the stride quadrature
/// (`for i in range(0, osl-1, stride)` × `repeat_count`) and the `(nextn + 1)`
/// decode-batch multiplier are applied by the caller (the Engine).
///
/// `gen_seq_imbalance_correction_scale` mirrors Python's per-op kwarg
/// (`base_backend.py:372`) — generation-attention ops multiply their result
/// by it (`operations/attention.py:899-906`). `only_generation_attention`
/// mirrors the mixed-step pass-3 name filter (only
/// `latency_dict["generation_attention"]` is read).
pub(crate) fn run_generation_ops_step(
    ops: &[Op],
    db: &PerfDatabase,
    batch_size: u32,
    kv_seq_tokens: u32,
    gen_seq_imbalance_correction_scale: f64,
    only_generation_attention: bool,
) -> Result<f64, AicError> {
    run_generation_ops_step_beamed(
        ops,
        db,
        batch_size,
        1,
        kv_seq_tokens,
        gen_seq_imbalance_correction_scale,
        only_generation_attention,
    )
}

/// [`run_generation_ops_step`] with an explicit beam width. Python's
/// `_run_generation_phase` queries with `x = batch_size * beam_width` while
/// `batch_size` itself stays unscaled (`base_backend.py:368-372`) — token-major
/// ops (GEMM, elementwise, comm) see the beam-scaled token count, attention
/// ops key on the raw decode batch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_generation_ops_step_beamed(
    ops: &[Op],
    db: &PerfDatabase,
    batch_size: u32,
    beam_width: u32,
    kv_seq_tokens: u32,
    gen_seq_imbalance_correction_scale: f64,
    only_generation_attention: bool,
) -> Result<f64, AicError> {
    run_generation_ops_step_beamed_with(
        ops,
        db,
        batch_size,
        beam_width,
        kv_seq_tokens,
        gen_seq_imbalance_correction_scale,
        only_generation_attention,
        |_, _| {},
    )
}

/// [`run_generation_ops_step_beamed`] with a per-op sink (see
/// [`run_context_ops_with`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_generation_ops_step_beamed_with<'a>(
    ops: &'a [Op],
    db: &PerfDatabase,
    batch_size: u32,
    beam_width: u32,
    kv_seq_tokens: u32,
    gen_seq_imbalance_correction_scale: f64,
    only_generation_attention: bool,
    mut on_op: impl FnMut(&'a Op, PerformanceResult),
) -> Result<f64, AicError> {
    run_generation_ops_step_planned(
        ops,
        None,
        None,
        db,
        batch_size,
        beam_width,
        kv_seq_tokens,
        gen_seq_imbalance_correction_scale,
        only_generation_attention,
        |_, op, result| on_op(op, result.into_owned()),
    )
}

/// [`run_generation_ops_step_beamed_with`] over an op list with its
/// [`OpListPlan`], reusing `op_memo`'s results when given one. The sink also
/// receives each op's index in `ops`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_generation_ops_step_planned<'a>(
    ops: &'a [Op],
    plan: Option<&OpListPlan>,
    op_memo: Option<&OpMemo>,
    db: &PerfDatabase,
    batch_size: u32,
    beam_width: u32,
    kv_seq_tokens: u32,
    gen_seq_imbalance_correction_scale: f64,
    only_generation_attention: bool,
    mut on_op: impl FnMut(usize, &'a Op, Cow<'_, PerformanceResult>),
) -> Result<f64, AicError> {
    let mut memo = QueryMemo::new(plan, ops);
    let ctx = generation_op_runtime(
        batch_size,
        beam_width,
        kv_seq_tokens,
        gen_seq_imbalance_correction_scale,
        0,
        None,
    );
    let mut total = 0.0_f64;
    for (index, op) in ops.iter().enumerate() {
        if only_generation_attention && !op.is_generation_attention() {
            continue;
        }
        let result = memo.query(index, || {
            shared_query(op_memo, db, OpList::Generation, index, op, &ctx)
        })?;
        total += result.latency_ms;
        on_op(index, op, result);
    }
    Ok(total)
}

/// One generation op at the decode-step shape — the single query body every
/// generation walk shares (`x = batch * beam`, attention keyed on the raw
/// decode batch, mirroring `base_backend.py:368-372`; the base decode walk
/// carries no prefix). `prefix` / `x_override` exist for Python
/// orchestrations with their own conventions (AFD's `_sum_latency` threads
/// the runtime prefix into decode queries and uses `x = batch`). The
/// index-addressed `evaluate` FFI calls this directly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn query_generation_op(
    op: &Op,
    db: &PerfDatabase,
    batch_size: u32,
    beam_width: u32,
    kv_seq_tokens: u32,
    gen_seq_imbalance_correction_scale: f64,
    prefix: u32,
    x_override: Option<u32>,
) -> Result<PerformanceResult, AicError> {
    op.query(
        db,
        &generation_op_runtime(
            batch_size,
            beam_width,
            kv_seq_tokens,
            gen_seq_imbalance_correction_scale,
            prefix,
            x_override,
        ),
    )
}

/// The [`RuntimeContext`] [`query_generation_op`] prices an op at.
pub(crate) fn generation_op_runtime(
    batch_size: u32,
    beam_width: u32,
    kv_seq_tokens: u32,
    gen_seq_imbalance_correction_scale: f64,
    prefix: u32,
    x_override: Option<u32>,
) -> RuntimeContext {
    RuntimeContext {
        batch_size,
        beam_width: beam_width.max(1),
        s: kv_seq_tokens,
        prefix,
        num_tokens: x_override.unwrap_or(batch_size.saturating_mul(beam_width.max(1))),
        seq_imbalance_correction_scale: 1.0,
        gen_seq_imbalance_correction_scale,
        num_image_tokens: 0,
    }
}

/// FPM-telemetry mix-step composition (the Dynamo mocker contract). Consumed
/// ONLY by `Engine::rank_latency_ms` (observed `ForwardPassMetrics`
/// dispatch); the live SDK engine-step path (`Engine::mixed_step_latency`)
/// mirrors Python's `_get_mix_step_latency` three-pass composition directly
/// via [`run_context_ops`] / [`run_generation_ops_step`] filters and does NOT
/// route through here.
///
/// Algorithm (mirrors Python):
/// 1. Combined non-attention pass: iterate `context_ops`, **skip**
///    context-attention ops, with `batch=1`, `isl=ctx_tokens+gen_tokens`,
///    `prefix=combined_prefix`.
/// 2. Context attention contribution: re-run only context-attention ops with
///    the prefill batch shape (`batch = prefill_batch` requests of
///    `new_tokens_per_prefill_req` tokens).
/// 3. Decode attention contribution: iterate `generation_ops`, only
///    generation-attention ops, with the decode batch shape.
///
/// `on_op` observes each queried op: whether it runs in a context pass
/// (passes 1 and 2) or the decode pass, its index in its op list, the op,
/// its runtime context, its RAW result and the divisor the total applies to
/// it (pass 2's `scale_factor`, otherwise 1). `op_memo`, when given, answers
/// single-field ops ([`OpMemo`]). With the op lists' [`OpListPlan`]s, each
/// pass queries every query slot once ([`QueryMemo`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_mix_step_ops<'a>(
    context_ops: &'a [Op],
    generation_ops: &'a [Op],
    context_plan: Option<&OpListPlan>,
    generation_plan: Option<&OpListPlan>,
    op_memo: Option<&OpMemo>,
    db: &PerfDatabase,
    ctx_tokens: u32,
    gen_tokens: u32,
    new_tokens_per_prefill_req: u32,
    prefill_batch: u32,
    prefix_per_req: u32,
    combined_prefix: u32,
    kv_per_decode_req: u32,
    decode_batch: u32,
    mut on_op: impl FnMut(bool, usize, &'a Op, &RuntimeContext, Cow<'_, PerformanceResult>, f64),
) -> Result<f64, AicError> {
    // ---- Pass 1: combined non-attention work (batch=1, isl=ctx+gen) ----
    // Python: `run_static` is called with `isl = num_tokens_combined`
    // and `prefix = prefix * floor(ctx_tokens / isl)`, which makes
    // `effective_isl = isl - prefix = ctx_new + gen` (same as
    // `(ctx_tokens + gen_tokens)` here, since `ctx_tokens` is already
    // the NEW prefill count). The op queries get `s = effective_isl`
    // and **`prefix = combined_prefix`** — not zero.
    //
    // The non-zero prefix matters for attention ops that travel
    // through pass 1 because they live inside a composite Op (e.g.
    // DSv3's `FallbackOp("context_mla_block")` wraps `MlaModule`).
    // The pass-1 filter is `op.is_context_attention()`, which only
    // matches the name `"context_attention"`; an MLA module under a
    // different name is included in pass 1 and needs the prefix to
    // produce the same `(full_s, prefix_correction)` Python applies.
    // For non-attention ops (GEMM, MoE, etc.) the prefix field is
    // ignored, so threading it through is harmless.
    let effective_isl_combined = (ctx_tokens + gen_tokens).max(1);
    let mut total = 0.0_f64;
    let mut memo = QueryMemo::new(context_plan, context_ops);
    for (index, op) in context_ops.iter().enumerate() {
        if op.is_context_attention() {
            continue;
        }
        let x = if op.is_logits_gemm() {
            1
        } else {
            effective_isl_combined
        };
        let ctx = RuntimeContext {
            batch_size: 1,
            beam_width: 1,
            s: effective_isl_combined,
            prefix: combined_prefix,
            num_tokens: x,
            seq_imbalance_correction_scale: 1.0,
            gen_seq_imbalance_correction_scale: 1.0,
            num_image_tokens: 0,
        };
        let result = memo.query(index, || {
            shared_query(op_memo, db, OpList::Context, index, op, &ctx)
        })?;
        total += result.latency_ms;
        on_op(true, index, op, &ctx, result, 1.0);
    }

    // ---- Pass 2: context attention with prefill batch ----
    // Python's pass 2 simulates one prefill request's full attention:
    //   batch_size = ceil(ctx_tokens / isl)
    //   query context_attention with isl = full per-req new tokens,
    //   prefix = prefix_per_req, then divide by scale = ceil(isl/ctx_tokens).
    //
    // The FPM only carries this chunk's metrics. For the common
    // un-chunked path (chunk == one request's new tokens) we treat
    // `new_tokens_per_prefill_req` as the per-request ISL and the
    // scale is 1; chunked-prefill callers should encode their full ISL
    // through the packing if they need exact parity with Python's
    // chunked path. The batch is the pass's own request count: with the
    // integer mean as the ISL, `ceil(ctx_tokens / isl)` would add a request
    // whenever the count does not divide `ctx_tokens`.
    let isl_eff_pass2 = new_tokens_per_prefill_req.max(1);
    let ctx_attn_batch = prefill_batch.max(1);
    let scale_factor = ((isl_eff_pass2 + ctx_tokens - 1) / ctx_tokens.max(1)).max(1) as f64;
    let mut memo = QueryMemo::new(context_plan, context_ops);
    for (index, op) in context_ops.iter().enumerate() {
        if !op.is_context_attention() {
            continue;
        }
        let ctx = RuntimeContext {
            batch_size: ctx_attn_batch,
            beam_width: 1,
            s: isl_eff_pass2,
            prefix: prefix_per_req,
            num_tokens: ctx_tokens,
            seq_imbalance_correction_scale: 1.0,
            gen_seq_imbalance_correction_scale: 1.0,
            num_image_tokens: 0,
        };
        let result = memo.query(index, || {
            shared_query(op_memo, db, OpList::Context, index, op, &ctx)
        })?;
        total += result.latency_ms / scale_factor;
        on_op(true, index, op, &ctx, result, scale_factor);
    }

    // ---- Pass 3: generation attention with decode batch ----
    // Mirror Python `_get_mix_step_latency`, which only adds the overlapped
    // decode-attention term `if gen_tokens > 0`. When this mixed step carries no
    // decode requests (e.g. the prefill-only first step that drives `ttft` at low
    // batch), Python contributes zero — so we must skip the pass entirely rather
    // than query at the `decode_batch.max(1)` floor, which would add a spurious
    // batch-1 generation_attention and inflate the step latency.
    if decode_batch > 0 {
        let mut memo = QueryMemo::new(generation_plan, generation_ops);
        for (index, op) in generation_ops.iter().enumerate() {
            if !op.is_generation_attention() {
                continue;
            }
            let ctx = RuntimeContext {
                batch_size: decode_batch,
                beam_width: 1,
                s: kv_per_decode_req.max(1),
                prefix: 0,
                num_tokens: decode_batch,
                seq_imbalance_correction_scale: 1.0,
                gen_seq_imbalance_correction_scale: 1.0,
                num_image_tokens: 0,
            };
            let result = memo.query(index, || {
                shared_query(op_memo, db, OpList::Generation, index, op, &ctx)
            })?;
            total += result.latency_ms;
            on_op(false, index, op, &ctx, result, 1.0);
        }
    }

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::GemmQuantMode;
    use crate::operators::GemmOp;
    use crate::perf_database::energy_test_fixtures::{
        Col, write_energy_systems_root, write_parquet,
    };

    #[test]
    fn dsv41_stage_memo_preserves_first_child_error_and_omits_failed_children() {
        // Valid SILICON queries: BF16 has a measurement, FP8 and NVFP4 do not.
        // Missing measurements must stay errors, including on memo hits.
        let tmp = tempfile::tempdir().unwrap();
        let data = write_energy_systems_root(tmp.path());
        write_parquet(
            &data.join("gemm_perf.parquet"),
            &[
                Col::Str("gemm_dtype", vec!["bfloat16"]),
                Col::I64("m", vec![128]),
                Col::I64("n", vec![1024]),
                Col::I64("k", vec![1024]),
                Col::F64("latency", vec![1.0]),
                Col::F64("power", vec![100.0]),
            ],
        );
        let db = PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0").unwrap();
        let child = |name, quant| Op::Gemm(GemmOp::new(name, 1024, 1024, quant));
        let ctx = RuntimeContext {
            batch_size: 128,
            num_tokens: 128,
            s: 512,
            ..RuntimeContext::default()
        };
        for two_failures in [false, true] {
            let stage = Dsv41StageOp {
                name: "generation_stage".into(),
                is_context: false,
                decoder_replay: true,
                bounded: true,
                window_size: 128,
                children: vec![
                    child("before", GemmQuantMode::Bfloat16),
                    child("first_missing", GemmQuantMode::Fp8),
                    child("after", GemmQuantMode::Bfloat16),
                    child(
                        "last",
                        if two_failures {
                            GemmQuantMode::Nvfp4
                        } else {
                            GemmQuantMode::Bfloat16
                        },
                    ),
                ],
            };
            assert!(
                stage
                    .children
                    .iter()
                    .all(|op| op.memo_field() == Some(MemoField::NumTokens))
            );
            let first_error = stage.children[1].query(&db, &ctx).unwrap_err();
            assert!(first_error.is_missing_perf_data());
            let expected = format!("{first_error:?}");
            assert_eq!(
                format!("{:?}", stage.query(&db, &ctx).unwrap_err()),
                expected
            );
            if two_failures {
                let later_error = stage.children[3].query(&db, &ctx).unwrap_err();
                assert!(later_error.is_missing_perf_data());
                assert_ne!(format!("{later_error:?}"), expected);
            }

            let op = Op::Dsv41Stage(stage.clone());
            let memo = OpMemo::new(16, &[], std::slice::from_ref(&op));
            assert_eq!(memo.stage_children.len(), 0);
            // A cold query, a hit at another KV length, and a repeated hit
            // all report the first failing child, never the later failure.
            for s in [512, 4096, 4096] {
                let ctx = RuntimeContext { s, ..ctx };
                assert_eq!(
                    format!("{:?}", stage.query(&db, &ctx).unwrap_err()),
                    expected
                );
                let error =
                    shared_query(Some(&memo), &db, OpList::Generation, 0, &op, &ctx).unwrap_err();
                assert_eq!(format!("{error:?}"), expected);
                assert_eq!(memo.stage_children.len(), 1);
                let key = OpMemo::key(OpList::Generation, 0, ctx.num_tokens);
                let (stored, _) = memo.stage_children.peek(&key).unwrap();
                assert_eq!(stored.len(), stage.children.len());
                // Inspect every child, including successful children after
                // the failure: only successful queries have a stored result.
                for (op, cached) in stage.children.iter().zip(stored.iter()) {
                    match op.query(&db, &ctx) {
                        Ok(result) => assert_eq!(cached.as_ref(), Some(&result)),
                        Err(_) => assert!(cached.is_none()),
                    }
                }
            }
        }
    }
}
