// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project.
// Tests of a modified Rust simulation of the vLLM v0.31.0
// (db9527a46873454610df6dbedf79a36d6bf1a7f6), v0.32.0
// (5df63a7203cdb9328d824a27b682e77c7b9928be) and v0.33.0
// (4ba787ec9959e566d3b587a796dff07673324da5) grouped KV rules.
// Immutable sources and license: THIRD_PARTY_NOTICES.md.

use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::engine::common::perf_model::PerfModel;
use crate::engine::common::protocols::{DirectRequest, MockEngineArgs};
use crate::engine::kv_manager::BlockRequestLease;
use crate::engine::scheduler::SchedulerCommand;
use crate::engine::trace::TraceCollector;
use crate::perfmodel::{FpmCacheGroup, FpmCacheKind};

use super::core::VllmCore;
use super::request::RequestKvState;

/// Decode queries `(active_tokens, mean_context, token_bound)` and prefill
/// queries `(batch, new_tokens, prefix_tokens)`.
#[derive(Default)]
struct CapturedDecodeQueries(
    Mutex<Vec<(usize, usize, usize)>>,
    Mutex<Vec<(usize, usize, usize)>>,
);

impl crate::engine::TimingModel for CapturedDecodeQueries {
    fn predict_prefill_ms(&self, _: usize, _: usize, _: usize) -> anyhow::Result<f64> {
        Ok(0.0)
    }

    fn predict_prefill_totals_ms(
        &self,
        batch_size: usize,
        new_tokens: usize,
        prefix_tokens: usize,
    ) -> anyhow::Result<f64> {
        self.1
            .lock()
            .unwrap()
            .push((batch_size, new_tokens, prefix_tokens));
        Ok(0.0)
    }

    fn predict_decode_ms(
        &self,
        _: usize,
        active_tokens: usize,
        mean_context: usize,
        token_bound: usize,
    ) -> anyhow::Result<f64> {
        self.0
            .lock()
            .unwrap()
            .push((active_tokens, mean_context, token_bound));
        Ok(0.0)
    }
}

fn group(name: &str, block: u32, bytes: u64, window: Option<u32>) -> FpmCacheGroup {
    FpmCacheGroup {
        name: name.into(),
        kind: FpmCacheKind::Attention,
        num_layers: 1,
        block_size_tokens: block,
        page_size_bytes: bytes,
        sliding_window: window,
        blocks_per_request: None,
        prefix_cacheable: None,
        transfer_bytes_per_block: None,
        prefill_checkpoint_blocks: None,
        prefill_checkpoint_alignment: None,
    }
}

fn state(name: &str, bytes: u64, blocks: u32) -> FpmCacheGroup {
    FpmCacheGroup {
        name: name.into(),
        kind: FpmCacheKind::State,
        num_layers: 1,
        block_size_tokens: 4,
        page_size_bytes: bytes,
        sliding_window: None,
        blocks_per_request: Some(blocks),
        prefix_cacheable: None,
        transfer_bytes_per_block: None,
        prefill_checkpoint_blocks: None,
        prefill_checkpoint_alignment: None,
    }
}

fn core(groups: Vec<FpmCacheGroup>, capacity: u64, chunk: usize) -> VllmCore {
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            // Deliberately smaller than the prompts: this linear capacity
            // must not decide admission or accounting for grouped caches.
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(chunk))
            .max_num_seqs(Some(3))
            .enable_prefix_caching(false)
            .kv_cache_groups(groups)
            .kv_cache_capacity_bytes(Some(capacity))
            .speedup_ratio(0.0)
            .build()
            .unwrap(),
    )
}

fn submit(core: &mut VllmCore, id: u128, prompt: usize, output: usize) -> Uuid {
    submit_tokens(core, id, (0..prompt as u32).collect(), output)
}

fn submit_tokens(core: &mut VllmCore, id: u128, tokens: Vec<u32>, output: usize) -> Uuid {
    core.receive(DirectRequest {
        uuid: Some(Uuid::from_u128(id)),
        tokens,
        max_output_tokens: output,
        ..Default::default()
    })
}

const GLM_PAGE: u64 = 13_045_120;
const GLM_BLOCK: u32 = 2176;

fn glm_state(name: &str, block: u32, cacheable: bool, checkpoint: bool) -> FpmCacheGroup {
    FpmCacheGroup {
        name: name.into(),
        kind: FpmCacheKind::State,
        num_layers: 1,
        block_size_tokens: block,
        page_size_bytes: GLM_PAGE,
        sliding_window: None,
        blocks_per_request: Some(1),
        prefix_cacheable: Some(cacheable),
        transfer_bytes_per_block: None,
        prefill_checkpoint_blocks: cacheable.then_some(u32::from(checkpoint)),
        prefill_checkpoint_alignment: (cacheable && checkpoint).then_some(16),
    }
}

/// Six GLM-5.3-Flash TP 4 cache groups as a hand-made profile of vLLM 0.31.0,
/// 0.32.0 or 0.33.0, at one page size. Their kinds and blocks are those that
/// the vLLM 0.33.0 runtime probe records (fixture
/// `glm53flash_vllm_0330_probe`): MLA attention and four KDA `MambaSpec` states
/// with 2,176-token blocks, and a 4-token `CircularBufferSpec` ring. The layer
/// counts and the page size are hand-made. With `checkpoint`, G2 to G5 carry
/// one FlashKDA checkpoint block at alignment 16.
/// The page is the sum of the pages of the attention layers, 13,045,120 bytes,
/// the block row of vLLM 0.31.0. vLLM 0.32.0 rounds the row up to the least
/// common multiple of the block-stride alignments of all layers (v0.32.0
/// `vllm/v1/core/kv_cache_utils.py:1533-1551`): 13,516,800 bytes for this
/// model at TP 4 on GB300, 25 times the least common multiple of the
/// 16,384-byte alignment of the MLA layers and the 4,224-byte alignment of the
/// indexer layers. The fixture keeps the page of the attention layers. The
/// tests count blocks and tokens, which do not depend on the page, and their
/// pool byte values are multiples of this page, not the pool of a vLLM 0.32.0
/// deployment.
fn glm_groups(checkpoint: bool) -> Vec<FpmCacheGroup> {
    let mut mla = group("mla", GLM_BLOCK, GLM_PAGE, None);
    mla.prefix_cacheable = Some(true);
    let mut groups = vec![mla, glm_state("kpool_tail", 4, false, false)];
    for i in 0..4 {
        groups.push(glm_state(&format!("kda_{i}"), GLM_BLOCK, true, checkpoint));
    }
    groups
}

/// Five GLM cores: vLLM 0.31.0, and vLLM 0.32.0 and 0.33.0 each without and
/// with checkpoint blocks, which GLM-5.3-Flash takes only under the FlashKDA
/// prefill backend.
const GLM_CORES: [(&str, bool); 5] = [
    ("0.31.0", false),
    ("0.32.0", false),
    ("0.32.0", true),
    ("0.33.0", false),
    ("0.33.0", true),
];

fn prefix_core(
    groups: Vec<FpmCacheGroup>,
    capacity: u64,
    chunk: usize,
    unit: usize,
    version: &str,
) -> VllmCore {
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(chunk))
            .max_num_seqs(Some(3))
            .max_model_len(Some(131_072))
            .enable_prefix_caching(true)
            .prefix_match_unit(Some(unit))
            .vllm_version(Some(version.to_string()))
            .kv_cache_groups(groups)
            .kv_cache_capacity_bytes(Some(capacity))
            .speedup_ratio(0.0)
            .build()
            .unwrap(),
    )
}

/// Step until the core drains. Returns `(uuid, reused_input_tokens)` of every
/// admission in order.
fn drain(
    core: &mut VllmCore,
    trace: &mut TraceCollector,
    from_step: &mut usize,
) -> Vec<(Uuid, usize)> {
    let mut admissions = Vec::new();
    for _ in 0..200 {
        let pass = core.execute_pass(trace, *from_step as f64);
        *from_step += 1;
        admissions.extend(
            pass.admissions
                .iter()
                .map(|admission| (admission.uuid, admission.reused_input_tokens)),
        );
        for output in pass.output_signals {
            assert!(!output.rejected, "{} rejected", output.uuid);
        }
        if core.is_drained() {
            return admissions;
        }
    }
    panic!("the core did not drain");
}

fn block_pool(core: &VllmCore) -> &crate::engine::kv_manager::GroupedBlockPool {
    core.kv_manager.grouped().unwrap().block_pool().unwrap()
}

#[test]
fn grouped_resumed_request_takes_a_checkpoint_block_only_at_its_prefill_end() {
    // vLLM 0.32.0 ends the prefill of a request at `max(prompt, num_tokens -
    // 1)`, which extends it to a resumed request that replays its output
    // tokens (v0.32 kv_cache_manager.py:518-520, sched:463-465,
    // sched:3075). Mamba reserves a checkpoint block only on the chunk that
    // reaches that end (v0.32 single:1643-1674, :1750-1758). A prompt of 33
    // tokens grows to 113 known tokens and is preempted. Its re-admission
    // hits the 32 tokens that its prefill published, and the first chunk of
    // 32 tokens ends at 64 < 112, so it takes no checkpoint block.
    let mut full = group("full", 16, GLM_PAGE, None);
    full.prefix_cacheable = Some(true);
    let groups = vec![full, glm_state("s", 16, true, true)];
    let mut core = prefix_core(groups, 8_200 * GLM_PAGE, 32, 16, "0.32.0");
    let id = submit(&mut core, 1700, 33, 200);
    let mut trace = TraceCollector::default();
    let mut step = 0.0;
    for _ in 0..200 {
        if core.state.requests[&id].sequence.len() >= 113 {
            break;
        }
        core.execute_pass(&mut trace, step);
        step += 1.0;
    }
    assert_eq!(core.state.requests[&id].sequence.len(), 113);
    assert!(core.policy_preempt(step).is_some());
    assert_eq!(block_pool(&core).active_blocks(), 0);
    let pass = core.execute_pass(&mut trace, step);
    let admissions: Vec<_> = pass
        .admissions
        .iter()
        .map(|admission| (admission.uuid, admission.reused_input_tokens))
        .collect();
    assert_eq!(admissions, [(id, 32)]);
    assert_eq!(core.state.requests[&id].num_computed_tokens, 64);
    // Four full-attention blocks, the running state block and the hit state
    // block, which stays until the request's next allocation, as in vLLM
    // (v0.32 kv_cache_manager.py:555-559).
    assert_eq!(block_pool(&core).active_blocks(), 4 + 1 + 1);
}

#[test]
fn grouped_window_replays_long_context_with_bounded_storage_and_logical_fpm_lengths() {
    let mut core = core(vec![group("window", 4, 8, Some(4))], 24, 8);
    let captured = Arc::new(CapturedDecodeQueries::default());
    core.args.perf_model = Arc::new(PerfModel::External {
        timing: captured.clone(),
    });
    let id = submit(&mut core, 101, 32, 5);
    let mut trace = TraceCollector::default();
    let mut outputs = 0;
    let mut prefixes = Vec::new();
    let mut decode_contexts = Vec::new();
    let mut bytes = Vec::new();
    for step in 0..20 {
        let pass = core.execute_pass(&mut trace, step as f64);
        let fpm = pass.fpm.unwrap();
        if fpm.num_prefill_requests > 0 {
            prefixes.push(fpm.sum_prefill_kv_tokens);
            assert_eq!(fpm.sum_prefill_tokens, 8);
        }
        if fpm.num_decode_requests > 0 {
            decode_contexts.push(fpm.sum_decode_kv_tokens);
        }
        bytes.push(pass.mocker_metrics.kv_cache_used_bytes.unwrap());
        assert_eq!(pass.mocker_metrics.total_blocks, 0);
        assert_eq!(pass.mocker_metrics.kv_cache_capacity_bytes, Some(24));
        assert!(pass.mocker_metrics.physical_gpu_cache_usage_perc <= 1.0);
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
        for output in pass.output_signals {
            assert_eq!(output.uuid, id);
            assert!(!output.rejected);
            outputs += usize::from(output.token_id.is_some());
        }
        if core.is_drained() {
            break;
        }
    }
    assert!(core.is_drained());
    assert_eq!(outputs, 5);
    assert_eq!(prefixes, vec![0, 8, 16, 24]);
    assert_eq!(decode_contexts, vec![33, 34, 35, 36]);
    assert_eq!(
        *captured.0.lock().unwrap(),
        // Prefill produces output token one. The four decode forwards include
        // their current input token and retain the full logical context.
        vec![(33, 33, 33), (34, 34, 34), (35, 35, 35), (36, 36, 36)]
    );
    assert_eq!(&bytes[..4], &[16, 24, 24, 24]);
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_full_window_and_convolution_charge_distinct_padded_pages() {
    let mut convolution = group("convolution", 2, 2, Some(2));
    convolution.kind = FpmCacheKind::Convolution;
    let mut core = core(
        vec![
            group("full", 4, 8, None),
            group("window", 4, 4, Some(4)),
            convolution,
        ],
        100,
        4,
    );
    let id = submit(&mut core, 102, 16, 5);
    let mut trace = TraceCollector::default();
    let mut bytes = Vec::new();
    for step in 0..5 {
        core.execute_pass(&mut trace, step as f64);
        bytes.push(core.mocker_metrics().kv_cache_used_bytes.unwrap());
    }
    assert_eq!(bytes, vec![16, 30, 38, 46, 52]);
    assert_eq!(core.state.requests[&id].num_computed_tokens, 17);
    assert_eq!(core.state.requests[&id].sequence.num_allocated_tokens(), 17);
    assert_eq!(core.state.requests[&id].sequence.len(), 18);
    core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
        .unwrap();
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_contention_preempts_and_retries_without_leaking_capacity() {
    let mut core = core(
        vec![group("full", 4, 4, None), group("window", 4, 4, Some(4))],
        24,
        8,
    );
    let first = submit(&mut core, 103, 4, 6);
    let second = submit(&mut core, 104, 4, 6);
    let mut trace = TraceCollector::default();
    let initial = core.execute_pass(&mut trace, 0.0);
    assert_eq!(initial.admissions.len(), 2);
    assert_eq!(initial.mocker_metrics.kv_cache_used_bytes, Some(16));
    let pressure = core.execute_pass(&mut trace, 1.0);
    assert_eq!(pressure.mocker_metrics.vllm_preemptions_total, 1);
    assert_eq!(pressure.mocker_metrics.kv_cache_used_bytes, Some(16));
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    assert_eq!(
        core.state.requests[&second].sequence.num_allocated_tokens(),
        0
    );
    assert_eq!(core.state.requests[&first].num_computed_tokens, 5);
    let mut completed = Vec::new();
    for step in 2..40 {
        let pass = core.execute_pass(&mut trace, step as f64);
        assert!(pass.mocker_metrics.kv_cache_used_bytes.unwrap() <= 24);
        for output in pass.output_signals {
            assert!(!output.rejected);
            if output.completed {
                completed.push(output.uuid);
            }
        }
        if core.is_drained() {
            break;
        }
    }
    assert_eq!(completed, vec![first, second]);
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_waiting_request_cannot_preempt_running_work() {
    let mut core = core(vec![group("window", 4, 4, Some(4))], 12, 8);
    let first = submit(&mut core, 105, 8, 2);
    let second = submit(&mut core, 106, 8, 2);
    let mut trace = TraceCollector::default();
    let initial = core.execute_pass(&mut trace, 0.0);
    assert_eq!(initial.admissions[0].uuid, first);
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    let next = core.execute_pass(&mut trace, 1.0);
    assert!(next.admissions.is_empty());
    assert_eq!(next.mocker_metrics.vllm_preemptions_total, 0);
    assert_eq!(next.mocker_metrics.kv_cache_used_bytes, Some(0));
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    let admitted = core.execute_pass(&mut trace, 2.0);
    assert_eq!(admitted.admissions[0].uuid, second);
}

#[test]
fn grouped_oversized_chunk_fails_without_partial_allocation_or_endless_preemption() {
    let mut core = core(vec![group("window", 4, 4, Some(4))], 8, 8);
    let id = submit(&mut core, 107, 16, 2);
    let mut trace = TraceCollector::default();
    core.execute_pass(&mut trace, 0.0);
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(8));
    let error = core.try_execute_pass(&mut trace, 1.0).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("exceeding the entire 8-byte cache pool")
    );
    assert!(error.to_string().contains("reduce max_num_batched_tokens"));
    // Without a state group the message has no sentence about fixed bytes.
    assert!(!error.to_string().contains("fixed state blocks"));
    assert_eq!(core.state.requests[&id].num_computed_tokens, 8);
    assert_eq!(core.state.requests[&id].sequence.num_allocated_tokens(), 8);
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(8));
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 0);
    core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
        .unwrap();
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_state_blocks_exceeding_the_pool_reject_at_admission() {
    // The 16-byte state block alone is larger than the 12-byte pool, so even
    // the one-token forward cannot fit and the request never reaches a forward.
    let mut core = core(vec![group("full", 4, 4, None), state("kda", 16, 1)], 12, 8);
    let id = submit(&mut core, 115, 4, 2);
    let mut trace = TraceCollector::default();
    let pass = core.execute_pass(&mut trace, 0.0);
    assert!(pass.admissions.is_empty());
    assert!(
        pass.output_signals
            .iter()
            .any(|output| output.uuid == id && output.rejected)
    );
    assert_eq!(pass.mocker_metrics.kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_oversized_chunk_names_the_fixed_state_bytes() {
    let mut core = core(
        vec![group("window", 4, 4, Some(4)), state("kda", 2, 1)],
        9,
        8,
    );
    submit(&mut core, 116, 16, 2);
    let mut trace = TraceCollector::default();
    // The one-token forward needs one 4-byte window page and the 2-byte state
    // block, 6 bytes, so admission passes. The first chunk of 8 tokens needs
    // two window pages (8 bytes) and the state block, 10 bytes, above the pool.
    let error = core
        .try_execute_pass(&mut trace, 0.0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("needs 10 bytes for 8 scheduled tokens after 0 computed tokens"));
    assert!(error.contains("exceeding the entire 9-byte cache pool"));
    assert!(error.contains("the fixed state blocks of one request take 2 bytes"));
    assert!(error.contains("reduce max_num_batched_tokens"));
    // The sentence comes before the remedy.
    assert!(error.contains("2 bytes of this pool; reduce max_num_batched_tokens"));
}

#[test]
fn grouped_handoff_commands_fail_before_acquiring_state() {
    let mut core = core(vec![group("window", 4, 4, Some(4))], 24, 8);
    let handoff_id = crate::engine::HandoffId::new(Uuid::from_u128(108));
    let error = core
        .apply_command(SchedulerCommand::SubmitHandoffPrefill {
            handoff_id,
            request: DirectRequest {
                tokens: vec![0, 1, 2, 3],
                max_output_tokens: 2,
                ..Default::default()
            },
        })
        .unwrap_err();
    assert!(error.to_string().contains("not KV handoff commands"));
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
}

#[test]
fn grouped_state_blocks_stay_constant_through_prefill_and_decode() {
    let mut core = core(vec![group("full", 4, 8, None), state("kda", 10, 2)], 200, 4);
    let id = submit(&mut core, 109, 16, 5);
    let mut trace = TraceCollector::default();
    let mut bytes = Vec::new();
    for step in 0..5 {
        core.execute_pass(&mut trace, step as f64);
        bytes.push(core.mocker_metrics().kv_cache_used_bytes.unwrap());
    }
    // Every pass adds one 8-byte page; the two 10-byte state blocks stay.
    assert_eq!(bytes, vec![28, 36, 44, 52, 60]);
    assert!(core.kv_manager.grouped().unwrap().block_pool().is_none());
    core.apply_command(SchedulerCommand::CancelRequest { request_id: id })
        .unwrap();
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_state_bytes_stay_constant_while_window_pages_expire() {
    let run = |groups: Vec<FpmCacheGroup>, capacity: u64| {
        let mut core = core(groups, capacity, 8);
        let id = submit(&mut core, 110, 32, 5);
        let mut trace = TraceCollector::default();
        let mut outputs = 0;
        let mut bytes = Vec::new();
        for step in 0..20 {
            let pass = core.execute_pass(&mut trace, step as f64);
            bytes.push(pass.mocker_metrics.kv_cache_used_bytes.unwrap());
            assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
            for output in pass.output_signals {
                assert_eq!(output.uuid, id);
                assert!(!output.rejected);
                outputs += usize::from(output.token_id.is_some());
            }
            if core.is_drained() {
                break;
            }
        }
        assert!(core.is_drained());
        assert_eq!(outputs, 5);
        assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
        assert_eq!(core.kv_manager.num_active_blocks(), 0);
        bytes
    };
    let without = run(vec![group("window", 4, 8, Some(4))], 24);
    let with = run(
        vec![group("window", 4, 8, Some(4)), state("kda", 10, 2)],
        44,
    );
    // Window pages expire as the decode moves on (24 to 16 bytes), while the
    // two state blocks add a constant 20 bytes. The last pass completes the
    // request, which frees every group before the pass reports its bytes.
    assert_eq!(without, vec![16, 24, 24, 24, 16, 16, 16, 0]);
    assert_eq!(with, vec![36, 44, 44, 44, 36, 36, 36, 0]);
}

#[test]
fn grouped_state_preemption_frees_and_rereserves_state_blocks() {
    let mut core = core(
        vec![
            group("full", 4, 4, None),
            group("window", 4, 4, Some(4)),
            state("kda", 4, 1),
        ],
        36,
        8,
    );
    let first = submit(&mut core, 111, 4, 6);
    let second = submit(&mut core, 112, 4, 6);
    let mut trace = TraceCollector::default();
    let initial = core.execute_pass(&mut trace, 0.0);
    // Each prefill holds a full page, a window page and its state block.
    assert_eq!(initial.admissions.len(), 2);
    assert_eq!(initial.mocker_metrics.kv_cache_used_bytes, Some(24));
    let pressure = core.execute_pass(&mut trace, 1.0);
    // The first request grows to 20 bytes. The preempted second request
    // returns all 12 of its bytes, its state block included.
    assert_eq!(pressure.mocker_metrics.vllm_preemptions_total, 1);
    assert_eq!(pressure.mocker_metrics.kv_cache_used_bytes, Some(20));
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    assert_eq!(
        core.state.requests[&second].sequence.num_allocated_tokens(),
        0
    );
    let mut completed = Vec::new();
    let mut resumed = Vec::new();
    for step in 2..40 {
        let pass = core.execute_pass(&mut trace, step as f64);
        let bytes = pass.mocker_metrics.kv_cache_used_bytes.unwrap();
        assert!(bytes <= 36);
        if pass
            .admissions
            .iter()
            .any(|admission| admission.uuid == second)
        {
            resumed.push(bytes);
        }
        for output in pass.output_signals {
            assert!(!output.rejected);
            if output.completed {
                completed.push(output.uuid);
            }
        }
        if core.is_drained() {
            break;
        }
    }
    // Each resume reserves the state block once more: two full pages, two
    // window pages and one state block, 20 bytes. The first resume fills the
    // pool next to the first request's 16 bytes. The first request's next
    // pages then preempt the second request again, and it resumes alone.
    assert_eq!(resumed, vec![36, 20]);
    assert_eq!(core.mocker_metrics().vllm_preemptions_total, 2);
    assert_eq!(completed, vec![first, second]);
    assert!(core.is_drained());
    assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_state_request_waits_when_only_attention_pages_fit() {
    let full = group("full", 4, 4, None);
    let mut core = core(vec![full.clone(), state("kda", 8, 1)], 28, 8);
    let first = submit(&mut core, 113, 8, 2);
    let second = submit(&mut core, 114, 8, 2);
    let mut trace = TraceCollector::default();
    let initial = core.execute_pass(&mut trace, 0.0);
    assert_eq!(initial.admissions.len(), 1);
    assert_eq!(initial.admissions[0].uuid, first);
    assert_eq!(initial.mocker_metrics.kv_cache_used_bytes, Some(16));
    // The next pass first grows the running request to 9 tokens: three pages
    // and its state block hold 12 + 8 = 20 bytes, so 8 of 28 are free. The
    // waiting request's chunk, the 7 tokens left in the 8-token batch, would
    // fit its two pages (8 bytes) there, but not its 8-byte state block too.
    let pool = core.kv_manager.grouped().unwrap();
    let free = pool.capacity_bytes() - pool.required_bytes(8, 9).unwrap();
    assert_eq!(free, 8);
    assert!(full.bytes_for_forward(0, 7).unwrap() <= free);
    assert!(pool.required_bytes(0, 7).unwrap() > free);
    let next = core.execute_pass(&mut trace, 1.0);
    assert!(next.admissions.is_empty());
    assert_eq!(next.mocker_metrics.vllm_preemptions_total, 0);
    assert_eq!(core.state.requests[&second].num_computed_tokens, 0);
    assert!(
        next.output_signals
            .iter()
            .any(|output| output.uuid == first && output.completed)
    );
    let admitted = core.execute_pass(&mut trace, 2.0);
    assert_eq!(admitted.admissions[0].uuid, second);
    for pass in [&initial, &next, &admitted] {
        assert!(pass.output_signals.iter().all(|output| !output.rejected));
    }
}

#[test]
fn grouped_prefix_caching_off_keeps_byte_mode_accounting() {
    let mut core = core(glm_groups(false), 64 * GLM_PAGE, 8192);
    assert!(core.kv_manager.grouped().unwrap().block_pool().is_none());
    let mut trace = TraceCollector::default();
    let mut step = 0;
    let first = submit(&mut core, 301, 9000, 4);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(first, 0)]);
    // A resend of the same prompt replays cold: nothing is cached in byte mode.
    let second = submit(&mut core, 302, 9000, 4);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(second, 0)]);
    let metrics = core.mocker_metrics();
    assert_eq!(metrics.kv_cache_used_bytes, Some(0));
    assert_eq!(metrics.kv_cache_cached_bytes, None);
    assert_eq!(core.kv_manager.grouped().unwrap().cached_bytes(), 0);
    assert_eq!(core.kv_manager.num_active_blocks(), 0);
}

#[test]
fn grouped_glm_shaped_multi_turn_replay_reports_hit_tokens_and_block_counts() {
    for (version, checkpoint) in GLM_CORES {
        let mut core = prefix_core(glm_groups(checkpoint), 64 * GLM_PAGE, 8192, 2176, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        let turn_one: Vec<u32> = (0..9000).collect();
        let first = submit_tokens(&mut core, 401, turn_one.clone(), 4);
        assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(first, 0)]);
        assert_eq!(block_pool(&core).active_blocks(), 0);

        // Turn two: the first 9,000 tokens again plus 4,000 new ones. The hit
        // is the state at request one's replay boundary (9000 - 1) / 2176 *
        // 2176 = 8704, where G0 holds four cached blocks too.
        let mut turn_two = turn_one.clone();
        turn_two.extend(100_000..104_000);
        let second = submit_tokens(&mut core, 402, turn_two.clone(), 4);
        block_pool(&core).reset_peak_active_blocks();
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.admissions.len(), 1, "{version} {checkpoint}");
        assert_eq!(pass.admissions[0].uuid, second);
        assert_eq!(
            pass.admissions[0].reused_input_tokens, 8704,
            "{version} {checkpoint}"
        );
        // During the first step: G0 holds the four pinned blocks and the fresh
        // ones of the chunk, the ring one, each KDA group the cached and the
        // running block, plus the checkpoint block on the cores with checkpoint
        // blocks. The cached state blocks stay until the request's next
        // allocation, as in vLLM (v0.31 kv_cache_manager.py:552-556, v0.32
        // :555-559, v0.33 :564-568).
        let (peak, after) = if checkpoint { (19, 19) } else { (14, 14) };
        assert_eq!(
            block_pool(&core).peak_active_blocks(),
            peak,
            "{version} {checkpoint}"
        );
        assert_eq!(
            block_pool(&core).active_blocks(),
            after,
            "{version} {checkpoint}"
        );
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 0);
        assert_eq!(
            pass.mocker_metrics.kv_cache_used_bytes,
            Some(after as u64 * GLM_PAGE)
        );
        assert_eq!(pass.mocker_metrics.kv_cache_cached_bytes, Some(0));
        // The next pass releases the four cached state blocks, and on the
        // third core the checkpoint block, before it takes the blocks of the
        // next chunk.
        core.execute_pass(&mut trace, step as f64);
        step += 1;
        let (active, cached) = if checkpoint { (11, 8) } else { (15, 4) };
        assert_eq!(
            block_pool(&core).active_blocks(),
            active,
            "{version} {checkpoint}"
        );
        assert_eq!(
            block_pool(&core).inactive_cached_blocks(),
            cached,
            "{version} {checkpoint}"
        );
        assert!(drain(&mut core, &mut trace, &mut step).is_empty());

        // A request that shares only the first 2,176 tokens finds a G0 block
        // but no state key at 2,176 (retention 0), so it hits nothing.
        let mut other: Vec<u32> = (0..2176).collect();
        other.extend(200_000..202_824);
        let third = submit_tokens(&mut core, 403, other, 4);
        assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(third, 0)]);

        // The whole second turn plus 100 tokens hits request two's replay
        // boundary (13000 - 1) / 2176 * 2176 = 10880.
        let mut turn_three = turn_two;
        turn_three.extend(300_000..300_100);
        let fourth = submit_tokens(&mut core, 404, turn_three, 4);
        assert_eq!(
            drain(&mut core, &mut trace, &mut step),
            vec![(fourth, 10880)]
        );
        assert_eq!(block_pool(&core).active_blocks(), 0);
        assert_eq!(core.mocker_metrics().kv_cache_used_bytes, Some(0));
    }
}

#[test]
fn grouped_prefix_hit_lowers_prefill_tokens_and_raises_prefix_tokens() {
    for (version, checkpoint) in GLM_CORES {
        let mut core = prefix_core(glm_groups(checkpoint), 64 * GLM_PAGE, 8192, 2176, version);
        let captured = Arc::new(CapturedDecodeQueries::default());
        core.args.perf_model = Arc::new(PerfModel::External {
            timing: captured.clone(),
        });
        let mut trace = TraceCollector::default();
        let mut step = 0;
        let turn_one: Vec<u32> = (0..9000).collect();
        submit_tokens(&mut core, 501, turn_one.clone(), 1);
        drain(&mut core, &mut trace, &mut step);
        // The first prefill stops at the block boundary 6528 inside the 8192
        // budget, then at the last cacheable stop 8704, then at the prompt.
        // In checkpoint mode the final chunk runs from 6528 to the prompt.
        let first_turn: Vec<(usize, usize, usize)> = if checkpoint {
            vec![(1, 6528, 0), (1, 2472, 6528)]
        } else {
            vec![(1, 6528, 0), (1, 2176, 6528), (1, 296, 8704)]
        };
        assert_eq!(
            *captured.1.lock().unwrap(),
            first_turn,
            "{version} {checkpoint}"
        );
        captured.1.lock().unwrap().clear();

        let mut turn_two = turn_one;
        turn_two.extend(100_000..104_000);
        submit_tokens(&mut core, 502, turn_two, 1);
        drain(&mut core, &mut trace, &mut step);
        // From the hit at 8704 the split ends the first chunk at the last
        // cacheable stop 10880 and the second at the prompt. The checkpoint
        // branch runs to the prompt end in one chunk.
        let second_turn: Vec<(usize, usize, usize)> = if checkpoint {
            vec![(1, 4296, 8704)]
        } else {
            vec![(1, 2176, 8704), (1, 2120, 10880)]
        };
        assert_eq!(
            *captured.1.lock().unwrap(),
            second_turn,
            "{version} {checkpoint}"
        );
    }
}

#[test]
fn grouped_prefix_admission_tests_the_whole_prompt() {
    for (version, checkpoint) in GLM_CORES {
        // A prompt of 40,000 tokens needs 19 G0 blocks, 1 ring block and one
        // running block per KDA group, plus one checkpoint block each on the
        // cores with checkpoint blocks: 24 or 28, above the 20-block pool.
        let mut core = prefix_core(glm_groups(checkpoint), 20 * GLM_PAGE, 8192, 2176, version);
        let id = submit(&mut core, 601, 40_000, 2);
        let request = &core.state.requests[&id].sequence;
        let cost = core
            .kv_manager
            .get_native_prefill_cost(&request.sequence, &request.lease);
        assert_eq!(
            cost.new_blocks,
            if checkpoint { 28 } else { 24 },
            "{version}"
        );
        assert_eq!(cost.cached_tokens, 0);
        let mut trace = TraceCollector::default();
        let pass = core.execute_pass(&mut trace, 0.0);
        assert!(pass.admissions.is_empty(), "{version} {checkpoint}");
        assert!(
            pass.output_signals
                .iter()
                .any(|output| output.uuid == id && output.rejected)
        );
        assert_eq!(block_pool(&core).active_blocks(), 0);
        assert!(core.is_drained());

        // Under contention the whole prompt decides, as in vLLM's default
        // admission (mgr:519-536). A running request of 13,000 tokens holds
        // 3 G0 pages, the ring and four state blocks after its first chunk,
        // which the split ends at 6,528, so 22 of 30 blocks are free. The
        // 40,000-token prompt needs 24 and waits without preempting anything.
        let mut core = prefix_core(glm_groups(checkpoint), 30 * GLM_PAGE, 8192, 2176, version);
        let running = submit(&mut core, 603, 13_000, 2);
        let waiting = submit_tokens(&mut core, 604, (100_000..140_000).collect(), 2);
        let pass = core.execute_pass(&mut trace, 0.0);
        assert_eq!(pass.admissions.len(), 1, "{version} {checkpoint}");
        assert_eq!(pass.admissions[0].uuid, running);
        assert_eq!(block_pool(&core).active_blocks(), 8);
        // The second pass has budget left after the running request's
        // chunk, and the waiting request still waits for the whole prompt.
        let pass = core.execute_pass(&mut trace, 1.0);
        assert!(pass.admissions.is_empty(), "{version} {checkpoint}");
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
        assert_eq!(core.state.requests[&waiting].num_computed_tokens, 0);
    }
    // Prefix caching off keeps the first-chunk test of byte mode. The first
    // pass spends its budget on the running request. In the second pass the
    // running request takes its last 4,808 tokens (6 G0 pages, the ring and
    // the states: 11 pages) and the waiting request's first chunk of the
    // 3,384 tokens left (2 pages, the ring and the states: 7 pages) fits.
    let mut off = core(glm_groups(false), 30 * GLM_PAGE, 8192);
    let running = submit(&mut off, 605, 13_000, 2);
    let waiting = submit_tokens(&mut off, 606, (100_000..140_000).collect(), 2);
    let mut trace = TraceCollector::default();
    let pass = off.execute_pass(&mut trace, 0.0);
    assert_eq!(pass.admissions.len(), 1);
    assert_eq!(pass.admissions[0].uuid, running);
    let pass = off.execute_pass(&mut trace, 1.0);
    assert_eq!(pass.admissions.len(), 1);
    assert_eq!(pass.admissions[0].uuid, waiting);
    assert_eq!(pass.mocker_metrics.kv_cache_used_bytes, Some(18 * GLM_PAGE));
    // Byte mode also refuses a final forward that can never fit, as
    // grouped_state_blocks_exceeding_the_pool_reject_at_admission shows: the
    // 40,000-token prompt's last forward needs 24 pages of a 20-page pool.
    let mut off = core(glm_groups(false), 20 * GLM_PAGE, 8192);
    let id = submit(&mut off, 602, 40_000, 2);
    let pass = off.execute_pass(&mut trace, 0.0);
    assert!(pass.admissions.is_empty());
    assert!(
        pass.output_signals
            .iter()
            .any(|output| output.uuid == id && output.rejected)
    );
}

#[test]
fn grouped_prefix_waiting_request_keeps_fifo() {
    // Each 20,000-token prompt needs 10 G0 blocks, the ring block and four
    // running blocks: 15 of 20. The second request waits for the first one
    // to finish and preempts nothing.
    let mut core = prefix_core(glm_groups(false), 20 * GLM_PAGE, 8192, 2176, "0.31.0");
    let first = submit_tokens(&mut core, 701, (0..20_000).collect(), 2);
    let second = submit_tokens(&mut core, 702, (100_000..120_000).collect(), 2);
    let mut trace = TraceCollector::default();
    let mut step = 0;
    let initial = core.execute_pass(&mut trace, step as f64);
    step += 1;
    assert_eq!(initial.admissions.len(), 1);
    assert_eq!(initial.admissions[0].uuid, first);
    let mut saw_first_complete = false;
    let mut admitted_second_at = None;
    for _ in 0..40 {
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0);
        if pass
            .admissions
            .iter()
            .any(|admission| admission.uuid == second)
        {
            admitted_second_at = Some(step);
            assert!(
                saw_first_complete,
                "the second request waited for the first"
            );
        }
        for output in pass.output_signals {
            assert!(!output.rejected);
            saw_first_complete |= output.uuid == first && output.completed;
        }
        if core.is_drained() {
            break;
        }
    }
    assert!(admitted_second_at.is_some());
    assert!(core.is_drained());
    assert_eq!(block_pool(&core).active_blocks(), 0);
}

#[test]
fn grouped_prefix_waiting_hit_waits_when_its_first_chunk_does_not_fit() {
    // One sliding-window group of 4-token blocks with an 8-token window, a
    // 9-block pool and a step budget of 8 tokens.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut window = group("window", 4, 8, Some(8));
        window.prefix_cacheable = Some(true);
        let mut core = prefix_core(vec![window], 9 * 8, 8, 4, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        // A finished request of 100 tokens keeps the two blocks before its
        // replay boundary 96 cached.
        submit_tokens(&mut core, 1101, (0..100).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 2, "{version}");
        submit_tokens(&mut core, 1102, (1000..1008).collect(), 30);
        submit_tokens(&mut core, 1103, (2000..2008).collect(), 30);
        for _ in 0..3 {
            core.execute_pass(&mut trace, step as f64);
            step += 1;
        }
        assert_eq!(block_pool(&core).active_blocks(), 5, "{version}");
        // A request of 104 tokens hits those blocks at 96. The whole-prompt
        // test counts only the two inactive hit blocks, because the window
        // caps it (mgr:519-536). After their decode steps the running
        // requests hold 6 of the 9 blocks, which leaves 3, but the first
        // chunk of 6 tokens needs the two hit blocks and two fresh blocks
        // (mgr:558-576). vLLM then stops its waiting loop (v0.31
        // sched:1223-1230, v0.32 sched:1275-1282), so the request waits and
        // preempts nothing.
        let waiting = submit_tokens(&mut core, 1104, (0..104).collect(), 1);
        let request = &core.state.requests[&waiting].sequence;
        let cost = core
            .kv_manager
            .get_native_prefill_cost(&request.sequence, &request.lease);
        assert_eq!((cost.new_blocks, cost.cached_tokens), (2, 96), "{version}");
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0, "{version}");
        assert!(pass.admissions.is_empty(), "{version}");
        assert_eq!(
            core.state.requests[&waiting].num_computed_tokens, 0,
            "{version}"
        );
        assert_eq!(core.state.running.len(), 2, "{version}");
        // The waiting request starts with its hit once the decode steps of the
        // running requests leave room.
        let admissions = drain(&mut core, &mut trace, &mut step);
        assert_eq!(admissions.first(), Some(&(waiting, 96)), "{version}");
    }
}

#[test]
fn grouped_prefix_waiting_hit_keeps_its_lru_place_when_its_first_chunk_does_not_fit() {
    // vLLM tests the blocks of the first chunk before it touches the hit
    // (mgr:558-589). So a waiting request that fails that test leaves its hit
    // blocks where they are in the LRU, ahead of a block cached after them.
    // Pinning and then releasing them would move them behind that block.
    let cached = |core: &VllmCore, tokens: &[u32], end: usize| {
        let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
        lease.configure_prefix_hashes(tokens, 4);
        block_pool(core)
            .key_hit(0, lease.prefix_hash_at(end).unwrap())
            .is_some()
    };
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut window = group("window", 4, 8, Some(8));
        window.prefix_cacheable = Some(true);
        let mut core = prefix_core(vec![window], 9 * 8, 8, 4, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        // The LRU holds the blocks that end at 96 and 92 of a finished request
        // of 100 tokens, then the block that ends at 4 of a finished request
        // of 8 tokens.
        let first: Vec<u32> = (0..100).collect();
        let second: Vec<u32> = (3000..3008).collect();
        submit_tokens(&mut core, 1401, first.clone(), 1);
        drain(&mut core, &mut trace, &mut step);
        submit_tokens(&mut core, 1402, second.clone(), 1);
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 3, "{version}");
        // As in grouped_prefix_waiting_hit_waits_when_its_first_chunk_does_not_fit,
        // two running requests hold 6 of the 9 blocks, and the request of 104
        // tokens fails the first-chunk test of its hit at 96.
        submit_tokens(&mut core, 1403, (1000..1008).collect(), 30);
        submit_tokens(&mut core, 1404, (2000..2008).collect(), 30);
        for _ in 0..3 {
            core.execute_pass(&mut trace, step as f64);
            step += 1;
        }
        let waiting = submit_tokens(&mut core, 1405, (0..104).collect(), 1);
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert!(pass.admissions.is_empty(), "{version}");
        assert_eq!(block_pool(&core).active_blocks(), 6, "{version}");
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 3, "{version}");
        core.apply_command(SchedulerCommand::CancelRequest {
            request_id: waiting,
        })
        .unwrap();
        // No block is free. A request of 4 new tokens takes one fresh block,
        // which evicts the oldest cached block: the one that ends at 96.
        let fresh = submit_tokens(&mut core, 1406, (4000..4004).collect(), 1);
        let pass = core.execute_pass(&mut trace, step as f64);
        assert_eq!(pass.admissions.len(), 1, "{version}");
        assert_eq!(pass.admissions[0].uuid, fresh, "{version}");
        assert!(!cached(&core, &first, 96), "{version}");
        assert!(cached(&core, &first, 92), "{version}");
        assert!(cached(&core, &second, 4), "{version}");
    }
}

#[test]
fn grouped_prefix_without_chunked_prefill_budgets_the_uncached_tokens() {
    // With chunked prefill off, vLLM compares the tokens after the prefix hit
    // with the step budget (v0.31 sched:1084, :1115-1121, v0.32 sched:1129,
    // :1160-1167), as the linear pool does. Two running requests decode one
    // token each and leave 14 of the 16 tokens. A prompt of 15 tokens with 12
    // cached tokens needs 3 of them.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let args = || {
            MockEngineArgs::builder()
                .block_size(4)
                .max_num_batched_tokens(Some(16))
                .max_num_seqs(Some(4))
                .max_model_len(Some(4096))
                .enable_prefix_caching(true)
                .enable_chunked_prefill(false)
                .speedup_ratio(0.0)
        };
        let linear = VllmCore::new(args().num_gpu_blocks(64).build().unwrap());
        let mut full = group("full", 4, 1, None);
        full.prefix_cacheable = Some(true);
        let grouped = VllmCore::new(
            args()
                .num_gpu_blocks(1)
                .prefix_match_unit(Some(4))
                .vllm_version(Some(version.to_string()))
                .kv_cache_groups(vec![full])
                .kv_cache_capacity_bytes(Some(64))
                .build()
                .unwrap(),
        );
        for (pool, mut core) in [("linear", linear), ("grouped", grouped)] {
            let mut trace = TraceCollector::default();
            let mut step = 0;
            submit_tokens(&mut core, 1301, (0..15).collect(), 1);
            drain(&mut core, &mut trace, &mut step);
            submit_tokens(&mut core, 1302, (1000..1004).collect(), 10);
            submit_tokens(&mut core, 1303, (2000..2004).collect(), 10);
            let pass = core.execute_pass(&mut trace, step as f64);
            step += 1;
            assert_eq!(pass.admissions.len(), 2, "{version} {pool}");
            let waiting = submit_tokens(&mut core, 1304, (0..15).collect(), 1);
            let pass = core.execute_pass(&mut trace, step as f64);
            let admitted: Vec<(Uuid, usize)> = pass
                .admissions
                .iter()
                .map(|admission| (admission.uuid, admission.reused_input_tokens))
                .collect();
            assert_eq!(admitted, vec![(waiting, 12)], "{version} {pool}");
        }
    }
}

/// Full-attention and state groups of 16-token blocks, a 4-token hash unit
/// and a pool of five blocks: the blocks of one 34-token request, as vLLM's
/// startup check counts them (3 + 2).
fn fine_grained_core(version: &str) -> VllmCore {
    let mut full = group("full", 16, 8, None);
    full.prefix_cacheable = Some(true);
    let mut state = state("state", 8, 1);
    state.block_size_tokens = 16;
    state.prefix_cacheable = Some(true);
    state.prefill_checkpoint_blocks = (version != "0.31.0").then_some(0);
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(4))
            .max_model_len(Some(34))
            .enable_prefix_caching(true)
            .prefix_match_unit(Some(4))
            .vllm_version(Some(version.to_string()))
            .kv_cache_groups(vec![full, state])
            .kv_cache_capacity_bytes(Some(5 * 8))
            .speedup_ratio(0.0)
            .build()
            .unwrap(),
    )
}

#[test]
fn grouped_prefix_hit_that_needs_more_blocks_than_the_pool_waits() {
    // A finished 13-token request leaves its full-attention block and its
    // state keyed at the hash boundary 12. A 33-token prompt that repeats it
    // hits 12 tokens inside a block. That hit needs 7 blocks: 2 new, the
    // inactive hit block and a copy for full attention (single:216-268), and
    // the running block, a copy and the inactive hit block for the state
    // (single:1716-1816). Cold the prompt needs 4 of the 5 blocks. vLLM keeps
    // such a request waiting and searches its hit again at the next step
    // (v0.31 sched:930-937, :1223-1230, v0.32 sched:960-967, :1275-1282).
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        // With nothing running, nothing changes the cache, so the request
        // waits in every pass. The replay reports that as a livelock
        // (runtime_contract.rs grouped_prefix_hit_above_the_pool_is_a_livelock).
        let mut core = fine_grained_core(version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1501, (0..13).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 2, "{version}");
        let waiting = submit_tokens(&mut core, 1502, (0..33).collect(), 1);
        for _ in 0..3 {
            let pass = core.execute_pass(&mut trace, step as f64);
            step += 1;
            assert!(pass.admissions.is_empty(), "{version}");
            assert!(pass.output_signals.is_empty(), "{version}");
            assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0, "{version}");
        }
        assert!(core.state.requests.contains_key(&waiting), "{version}");
        // Read after a pass: the last step of the finished request copied its
        // state at 12, which hides that key until the next step.
        let request = &core.state.requests[&waiting].sequence;
        let cost = core
            .kv_manager
            .get_native_prefill_cost(&request.sequence, &request.lease);
        assert_eq!((cost.new_blocks, cost.cached_tokens), (7, 12), "{version}");

        // A running request can evict the hit blocks. Its decode crosses a
        // block boundary at 16 tokens and takes two blocks while one is free,
        // which evicts the cached full-attention block of the hit. The
        // waiting request then hits nothing and starts cold once the running
        // request finishes.
        let mut core = fine_grained_core(version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1503, (0..13).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        let running = submit_tokens(&mut core, 1504, (1000..1004).collect(), 20);
        let waiting = submit_tokens(&mut core, 1505, (0..33).collect(), 1);
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        assert_eq!(pass.admissions.len(), 1, "{version}");
        assert_eq!(pass.admissions[0].uuid, running, "{version}");
        assert!(pass.output_signals.iter().all(|output| !output.rejected));
        let mut finished = Vec::new();
        let mut admissions = Vec::new();
        for _ in 0..40 {
            let pass = core.execute_pass(&mut trace, step as f64);
            step += 1;
            assert_eq!(pass.mocker_metrics.vllm_preemptions_total, 0, "{version}");
            admissions.extend(
                pass.admissions
                    .iter()
                    .map(|admission| (admission.uuid, admission.reused_input_tokens)),
            );
            for output in pass.output_signals {
                assert!(!output.rejected, "{version}");
                if output.completed {
                    finished.push(output.uuid);
                }
            }
            if core.is_drained() {
                break;
            }
        }
        assert_eq!(admissions, vec![(waiting, 0)], "{version}");
        assert_eq!(finished, vec![running, waiting], "{version}");
    }
}

#[test]
fn grouped_prefix_physical_usage_counts_cached_blocks() {
    // Metrics::physical_cache_usage includes the inactive reusable blocks and
    // Metrics::inactive_blocks counts them (protocol.rs). Active usage counts
    // only the blocks that requests hold.
    for (version, checkpoint) in GLM_CORES {
        let mut core = prefix_core(glm_groups(checkpoint), 64 * GLM_PAGE, 8192, 2176, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 1201, (0..9000).collect(), 4);
        drain(&mut core, &mut trace, &mut step);
        // G0 keeps its four full blocks, and each KDA group the state at the
        // replay boundary 8704.
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 8, "{version}");
        let metrics = core.mocker_metrics();
        assert_eq!(metrics.kv_cache_used_bytes, Some(0), "{version}");
        assert_eq!(
            metrics.kv_cache_cached_bytes,
            Some(8 * GLM_PAGE),
            "{version}"
        );
        assert_eq!(metrics.gpu_cache_usage_perc, 0.0, "{version}");
        assert_eq!(
            metrics.physical_gpu_cache_usage_perc,
            8.0 / 64.0,
            "{version}"
        );
        assert_eq!(metrics.active_decode_blocks, 0, "{version}");
        assert_eq!(metrics.inactive_decode_blocks, 8, "{version}");
    }
}

/// The tight-pool case: RC hits 3,264 tokens, inside a block.
fn tight_pool_case(
    version: &str,
    blocks: u64,
    max_concurrent_batches: usize,
) -> (usize, usize, Vec<usize>, usize) {
    pool_case(version, blocks, max_concurrent_batches, 3500, 4352, 3264)
}

/// A GLM rank of `blocks` pool blocks with a hash block of 1,088 tokens. RP
/// (`producer` tokens) runs alone and finishes. RC (RP's tokens and 300
/// more) and RN (1,000 new tokens) then arrive together, RC first, and RC
/// hits `hit` tokens. Returns the passes that admit RC and RN, counted from
/// their arrival, the active blocks after each of those passes and the peak
/// of active blocks in RC's pass.
fn pool_case(
    version: &str,
    blocks: u64,
    max_concurrent_batches: usize,
    producer: u32,
    max_model_len: usize,
    hit: usize,
) -> (usize, usize, Vec<usize>, usize) {
    let mut core = VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(8192))
            .max_num_seqs(Some(4))
            .max_model_len(Some(max_model_len))
            .enable_prefix_caching(true)
            .prefix_match_unit(Some(1088))
            .vllm_version(Some(version.to_string()))
            .kv_cache_groups(glm_groups(false))
            .kv_cache_capacity_bytes(Some(blocks * GLM_PAGE))
            .speedup_ratio(0.0)
            .max_concurrent_batches(max_concurrent_batches)
            .build()
            .unwrap(),
    );
    let mut now = 0.0;
    let rp: Vec<u32> = (44_000_000..44_000_000 + producer).collect();
    submit_tokens(&mut core, 1, rp.clone(), 4);
    for _ in 0..50 {
        if core.is_drained() {
            break;
        }
        async_step(&mut core, &mut now);
    }
    assert!(core.is_drained());
    let mut rc_tokens = rp;
    rc_tokens.extend(44_100_000..44_100_300);
    let rc = submit_tokens(&mut core, 2, rc_tokens, 4);
    let rn = submit_tokens(&mut core, 3, (44_200_000..44_201_000).collect(), 4);
    let (mut rc_pass, mut rn_pass, mut active, mut peak) = (None, None, Vec::new(), 0);
    for pass in 0..50 {
        block_pool(&core).reset_peak_active_blocks();
        let result = async_step(&mut core, &mut now);
        assert_eq!(result.mocker_metrics.vllm_preemptions_total, 0);
        for admission in &result.admissions {
            if admission.uuid == rc {
                rc_pass = Some(pass);
                peak = block_pool(&core).peak_active_blocks();
                assert_eq!(admission.reused_input_tokens, hit);
            }
            if admission.uuid == rn {
                rn_pass = Some(pass);
            }
        }
        active.push(block_pool(&core).active_blocks());
        if core.is_drained() {
            break;
        }
    }
    (rc_pass.unwrap(), rn_pass.unwrap(), active, peak)
}

#[test]
fn grouped_prefix_copy_sources_block_a_later_admission_of_the_pass() {
    // RC's hit inside a block copies the cached full-attention block and the
    // hit block of each of the four state groups. vLLM keeps these 5 sources
    // in use until the end of `schedule()` (v0.31 sched:1434-1442, v0.32
    // sched:1474-1482, v0.33 sched:1698-1706). So RC holds 12 blocks in its
    // step: 2 full-attention blocks, the ring block, 4 running state blocks
    // and the 5 sources. RN needs 6 blocks. On a pool of 15 blocks vLLM
    // refused RN in RC's step and admitted it in the next one. Replay
    // releases the sources at the end of the pass, so RN waits one pass on
    // pools of 13 to 17 blocks. On 18 blocks both fit in one pass. On 12
    // blocks RN waits until RC ends.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for blocks in 12..=18 {
            let expected = match blocks {
                12 => (0, 4, vec![7, 7, 7, 0, 6, 6, 6, 0], 12),
                18 => (0, 0, vec![13, 13, 13, 0], 18),
                _ => (0, 1, vec![7, 13, 13, 6, 0], 12),
            };
            assert_eq!(
                tight_pool_case(version, blocks, 1),
                expected,
                "{version} on {blocks} blocks"
            );
        }
    }
}

#[test]
fn grouped_prefix_boundary_hit_holds_its_state_blocks_until_the_next_pass() {
    // RP has 4,400 tokens, so its chunks end at 4,352 and 4,400. RC hits
    // 4,352 tokens, on a block boundary, and takes no copy. vLLM keeps the
    // hit block of each state group in RC's table until RC's next
    // `allocate_slots` (v0.31 kv_cache_manager.py:552-556,
    // single_type_kv_cache_manager.py:1626-1645, v0.32 :555-559,
    // :1592-1611, v0.33 :564-568, :1608-1627). So RC holds 12 blocks in its
    // step: 3 full-attention blocks, the ring block, and a hit block and a
    // running block in each state group. RN needs 6 blocks. Run on vLLM's
    // cache manager, this case refuses RN in RC's step and admits it in the
    // next one on pools of 14 to 17 blocks. Replay does the same. On 18
    // blocks both fit in one pass. On 12 and 13 blocks RN waits until RC
    // ends.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for blocks in 12..=18 {
            let expected = match blocks {
                12 | 13 => (0, 4, vec![12, 8, 8, 0, 6, 6, 6, 0], 12),
                18 => (0, 0, vec![18, 14, 14, 0], 18),
                _ => (0, 1, vec![12, 14, 14, 6, 0], 12),
            };
            assert_eq!(
                pool_case(version, blocks, 1, 4400, 6528, 4352),
                expected,
                "{version} on {blocks} blocks"
            );
        }
    }
}

#[test]
fn grouped_prefix_copy_sources_leave_in_group_order_at_the_end_of_the_pass() {
    // Two consumers with hits inside a block in one pass. vLLM frees their
    // copy sources at the end of `schedule()`, manager by manager in group
    // order (v0.31 kv_cache_manager.py:894-909, sched:1434-1442, v0.32
    // kv_cache_manager.py:854-869, sched:1474-1482, v0.33
    // kv_cache_manager.py:863-878, sched:1698-1706), and each cached source
    // goes to the tail of the free queue (v0.31 block_pool.py:777-808, v0.32
    // :774-805, v0.33 :816-849). So the full-attention sources of both
    // consumers enter the LRU first, then the sources of group 2 of both, and
    // so on. A cold request X then evicts the oldest cached blocks, and a
    // later request on either prefix finds the same hit. After the consumers
    // end, 22 blocks are cached: the 8 producer states at 2,176, the 10
    // sources, and the 4 full-attention blocks of the consumers, whose copies
    // keep the key of 3,264. 18 blocks are free. On the 0.31.0 rules X (26
    // blocks) computes its whole prompt in one chunk, since its last
    // cacheable stop is its length: 26 + 1 + 4 = 31 blocks, 13 evictions.
    // They take the 8 states, both full-attention sources, both sources of
    // group 2 and one of group 3, so neither prefix hits. X ends in that
    // pass, and its 26 full-attention blocks join the 9 cached blocks left:
    // 35. On the 0.32.0 and 0.33.0 rules the first chunk of X (24 blocks)
    // stops at (52,224 - 1) / 2,176 * 2,176 = 50,048: 23 + 1 + 4 = 28 blocks,
    // 10 evictions. They end with both full-attention sources, every state
    // source stays, both prefixes hit 3,264, and 12 blocks stay cached.
    // (version, X's blocks, inactive blocks after X's pass, the two hits)
    for (version, x_blocks, inactive_after_x, hits) in [
        ("0.31.0", 26, 35, (0, 0)),
        ("0.32.0", 24, 12, (3264, 3264)),
        ("0.33.0", 24, 12, (3264, 3264)),
    ] {
        let mut core = VllmCore::new(
            MockEngineArgs::builder()
                .block_size(4)
                .num_gpu_blocks(1)
                .max_num_batched_tokens(Some(65_536))
                .max_num_seqs(Some(4))
                .max_model_len(Some(131_072))
                .enable_prefix_caching(true)
                .prefix_match_unit(Some(1088))
                .vllm_version(Some(version.to_string()))
                .kv_cache_groups(glm_groups(false))
                .kv_cache_capacity_bytes(Some(40 * GLM_PAGE))
                .speedup_ratio(0.0)
                .build()
                .unwrap(),
        );
        let mut trace = TraceCollector::default();
        let mut step = 0;
        let first: Vec<u32> = (44_000_000..44_003_500).collect();
        let second: Vec<u32> = (45_000_000..45_003_500).collect();
        submit_tokens(&mut core, 1, first.clone(), 1);
        submit_tokens(&mut core, 2, second.clone(), 1);
        drain(&mut core, &mut trace, &mut step);
        let extend = |tokens: &[u32], base: u32| {
            let mut tokens = tokens.to_vec();
            tokens.extend(base..base + 300);
            tokens
        };
        let c1 = submit_tokens(&mut core, 3, extend(&first, 46_000_000), 1);
        let c2 = submit_tokens(&mut core, 4, extend(&second, 47_000_000), 1);
        let pass = core.execute_pass(&mut trace, step as f64);
        step += 1;
        let admitted: Vec<(Uuid, usize)> = pass
            .admissions
            .iter()
            .map(|admission| (admission.uuid, admission.reused_input_tokens))
            .collect();
        assert_eq!(admitted, [(c1, 3264), (c2, 3264)], "{version}");
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(block_pool(&core).inactive_cached_blocks(), 22, "{version}");
        let x_tokens = x_blocks * GLM_BLOCK;
        submit_tokens(
            &mut core,
            5,
            (48_000_000..48_000_000 + x_tokens).collect(),
            1,
        );
        core.execute_pass(&mut trace, step as f64);
        assert_eq!(
            block_pool(&core).inactive_cached_blocks(),
            inactive_after_x,
            "{version}"
        );
        let hit = |tokens: Vec<u32>| {
            let mut probe = super::request::RequestKvState::native(
                Uuid::from_u128(6),
                tokens.clone(),
                4,
                4,
                core.args.block_size,
                true,
                false,
                false,
                None,
            );
            probe.lease.configure_prefix_hashes(&tokens, 1088);
            core.kv_manager
                .get_native_prefill_cost(&probe.sequence, &probe.lease)
                .cached_tokens
        };
        assert_eq!(
            (
                hit(extend(&first, 49_000_000)),
                hit(extend(&second, 49_500_000))
            ),
            hits,
            "{version}"
        );
    }
}

#[test]
fn grouped_prefix_eviction_frees_cached_blocks_oldest_first() {
    // One full-attention group of 4-token blocks in a pool of 16 blocks.
    let mut full = group("full", 4, 8, None);
    full.prefix_cacheable = Some(true);
    let mut core = prefix_core(vec![full], 16 * 8, 64, 4, "0.31.0");
    let mut trace = TraceCollector::default();
    let mut step = 0;
    // Three finished requests of 16 tokens leave 12 cached blocks.
    for (id, base) in [(801, 0), (802, 100), (803, 200)] {
        submit_tokens(&mut core, id, (base..base + 16).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
    }
    assert_eq!(block_pool(&core).inactive_cached_blocks(), 12);
    // A fourth request of 40 tokens needs 10 fresh blocks: the 4 free ones
    // and the 6 evicted oldest first, in release order (tail first): all of
    // request one and the two last blocks of request two.
    let fourth = submit_tokens(&mut core, 804, (300..340).collect(), 1);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(fourth, 0)]);
    assert_eq!(block_pool(&core).inactive_cached_blocks(), 16);
    // A fifth request equal to the third one still hits, capped below the
    // prompt: (16 - 1) / 4 * 4 = 12.
    let fifth = submit_tokens(&mut core, 805, (200..216).collect(), 1);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(fifth, 12)]);
    // The fifth request's one fresh block evicted the older of request
    // two's two surviving blocks, so request two hits its first block only.
    let sixth = submit_tokens(&mut core, 806, (100..116).collect(), 1);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(sixth, 4)]);
    // Request one is gone.
    let seventh = submit_tokens(&mut core, 807, (0..16).collect(), 1);
    assert_eq!(drain(&mut core, &mut trace, &mut step), vec![(seventh, 0)]);
}

/// A rank with a full-attention group and a prefix-cacheable state group of
/// 4-token blocks, a hash block of 4 and `blocks` pool blocks.
fn async_prefix_core(version: &str, blocks: u64, max_concurrent_batches: usize) -> VllmCore {
    let mut attention = group("attention", 4, 8, None);
    attention.prefix_cacheable = Some(true);
    let mut kda = state("kda", 8, 1);
    kda.prefix_cacheable = Some(true);
    kda.prefill_checkpoint_blocks = (version != "0.31.0").then_some(0);
    VllmCore::new(
        MockEngineArgs::builder()
            .block_size(4)
            .num_gpu_blocks(1)
            .max_num_batched_tokens(Some(64))
            .max_num_seqs(Some(8))
            .max_model_len(Some(64))
            .enable_prefix_caching(true)
            .prefix_match_unit(Some(4))
            .vllm_version(Some(version.to_string()))
            .kv_cache_groups(vec![attention, kda])
            .kv_cache_capacity_bytes(Some(blocks * 8))
            .speedup_ratio(0.0)
            .max_concurrent_batches(max_concurrent_batches)
            .build()
            .unwrap(),
    )
}

/// One pass at `now_ms` and its boundary. `now_ms` moves to the end of the
/// pass, so the next pass is formed while this one is in flight.
fn async_step(core: &mut VllmCore, now_ms: &mut f64) -> crate::engine::scheduler::EnginePassResult {
    let mut trace = TraceCollector::default();
    let result = core.execute_pass(&mut trace, *now_ms);
    core.complete_engine_boundary(result.end_ms);
    *now_ms = result.end_ms;
    result
}

#[test]
fn grouped_prefix_blocks_of_a_finished_request_first_serve_pass_k_plus_2_with_async_scheduling() {
    // X (4 tokens) keeps the rank busy. R (17 tokens) prefills in passes 0
    // and 1 and finishes in pass 1 with 5 attention blocks and 2 state blocks.
    // W (21 tokens) needs 7 blocks of the 12 and fits only in R's blocks.
    // vLLM frees R's blocks when it processes the outputs of step 1 (v0.31
    // sched:2171, :2650-2679, v0.32 sched:2230, :2714-2744, v0.33
    // sched:2476, :3031-3062). Without async scheduling that comes before the
    // scheduling of step 2, so W starts in step 2. With async scheduling the
    // engine schedules step 2 before it processes the outputs of step 1 (v0.31
    // engine/core.py:697, :740-755, v0.32 engine/core.py:725, :768-783, v0.33
    // engine/core.py:735, :778-793), so W starts in step 3. The block counts
    // are replay's. After R's last pass X holds 2 attention blocks, its
    // running state block and the previous state block of its crossing at 4:
    // 4 blocks, and 11 with async scheduling, where R still holds its 7. Pass
    // 2 releases X's previous state block: 3 left with async scheduling.
    // Without async scheduling pass 2 also admits W, whose first chunk of 20
    // tokens takes 5 attention blocks and 1 state block: 9. With async
    // scheduling vLLM also holds X's previous state block for one more step
    // (v0.31 mgr:552-556, v0.32 mgr:555-559, v0.33 mgr:564-568), a known
    // difference.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for (max_concurrent_batches, admitting_pass, after_finish, after_next) in
            [(1, 2, 4, 9), (2, 3, 11, 3)]
        {
            let mut core = async_prefix_core(version, 12, max_concurrent_batches);
            let mut now = 0.0;
            submit_tokens(&mut core, 1, (1000..1004).collect(), 30);
            let r = submit_tokens(&mut core, 2, (2000..2017).collect(), 1);
            let w = submit_tokens(&mut core, 3, (3000..3021).collect(), 1);
            async_step(&mut core, &mut now);
            let finished = async_step(&mut core, &mut now);
            assert!(
                finished
                    .output_signals
                    .iter()
                    .any(|signal| signal.uuid == r && signal.completed),
                "{version}"
            );
            // With async scheduling R still holds its 7 blocks.
            assert_eq!(block_pool(&core).active_blocks(), after_finish, "{version}");
            let mut active = Vec::new();
            let admitted = (2..6).find(|_| {
                let result = async_step(&mut core, &mut now);
                active.push(block_pool(&core).active_blocks());
                assert_eq!(result.mocker_metrics.vllm_preemptions_total, 0, "{version}");
                result
                    .admissions
                    .iter()
                    .any(|admission| admission.uuid == w)
            });
            assert_eq!(admitted, Some(admitting_pass), "{version}");
            assert_eq!(active[0], after_next, "{version}");
        }
    }
}

#[test]
fn grouped_prefix_async_scheduling_frees_a_finished_request_after_a_running_request_releases_its_previous_state()
 {
    // This test pins a known difference from vLLM. G (10 tokens) publishes
    // its state at 8 in pass 0, crosses that boundary in pass 1 and releases
    // the block of that state one pass later, in pass 2. F (9 tokens)
    // prefills 8 tokens in pass 0 and 1 in pass 1, where it finishes with 3
    // attention blocks and 2 state blocks; its cached blocks are attention
    // at 4 and 8 and its state at 8. Without async scheduling F's blocks
    // enter the LRU in pass 1, tail first, before G's block. With async
    // scheduling replay frees F's blocks after it has formed pass 2, so
    // after G's block. After pass 2 G holds 4 blocks, 4 blocks are cached
    // and 3 are free. In pass 4 H (3 tokens) takes 2 blocks and G 2 more (its
    // fourth attention block and its next state block), so the pool evicts
    // its oldest cached block: F's attention block at 8 without async
    // scheduling, G's state block at 8 with it. A later prompt that starts
    // with G's tokens then hits 8 tokens against 0, and one that starts with
    // F's tokens 0 against 8.
    // By a reading of vLLM, F goes first in both modes. vLLM frees F when it
    // processes the outputs of step 1 (v0.31 sched:2171, :2650-2679, v0.32
    // sched:2230, :2714-2744, v0.33 sched:2476, :3031-3062). It frees G's
    // block in `allocate_slots` on the processed-token basis (v0.31
    // mgr:552-556, v0.32 mgr:555-559, v0.33 mgr:564-568), through the base
    // path of `MambaManager.remove_skipped_blocks` (v0.31 single:1647-1675,
    // base path :1655-1657, :687-725, :1626-1645, :1990-1995; v0.32
    // single:1613-1641, base path :1621-1623, :672-710, :1592-1611,
    // :1954-1959; v0.33 single:1629-1657, base path :1637-1639, :682-720,
    // :1608-1627, :1971-1976). With async scheduling the tokens of the step
    // in flight do not count, so vLLM frees G's block in the scheduling of
    // step 3, one step later than replay. So the outcome without async
    // scheduling equals vLLM's, and the outcome with async scheduling is
    // replay's own.
    let cached = |core: &VllmCore, group: u16, tokens: &[u32], end: usize| {
        let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
        lease.configure_prefix_hashes(tokens, 4);
        block_pool(core)
            .key_hit(group, lease.prefix_hash_at(end).unwrap())
            .is_some()
    };
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for max_concurrent_batches in [1usize, 2] {
            let mut core = async_prefix_core(version, 11, max_concurrent_batches);
            let mut now = 0.0;
            let g: Vec<u32> = (1000..1010).collect();
            let f: Vec<u32> = (2000..2009).collect();
            submit_tokens(&mut core, 1, g.clone(), 30);
            let f_id = submit_tokens(&mut core, 2, f.clone(), 1);
            // H arrives so that pass 4 admits it in both modes: without
            // async scheduling after pass 3, and with async scheduling while
            // pass 2 is in flight, when pass 3 is already queued behind it.
            let arrival = if max_concurrent_batches == 1 { 3 } else { 2 };
            let mut admitted = None;
            let mut f_finished = None;
            for pass in 0..5 {
                let result = async_step(&mut core, &mut now);
                if !result.admissions.is_empty() && pass > 0 {
                    admitted = Some(pass);
                }
                if result
                    .output_signals
                    .iter()
                    .any(|signal| signal.uuid == f_id && signal.completed)
                {
                    f_finished = Some(pass);
                }
                if pass == 2 {
                    // After pass 2: G holds 4 blocks and 4 are cached.
                    assert_eq!(block_pool(&core).active_blocks(), 4, "{version}");
                    assert_eq!(block_pool(&core).inactive_cached_blocks(), 4, "{version}");
                }
                if pass == arrival {
                    core.apply_command_effects(
                        SchedulerCommand::Submit(DirectRequest {
                            uuid: Some(Uuid::from_u128(3)),
                            tokens: (3000..3003).collect(),
                            max_output_tokens: 1,
                            ..Default::default()
                        }),
                        false,
                    )
                    .unwrap();
                }
            }
            assert_eq!(f_finished, Some(1), "{version}");
            assert_eq!(admitted, Some(4), "{version}");
            // [F attention at 4, F attention at 8, F state at 8, G state at 8]
            let left = [
                cached(&core, 0, &f, 4),
                cached(&core, 0, &f, 8),
                cached(&core, 1, &f, 8),
                cached(&core, 1, &g, 8),
            ];
            // Later prompts that start with G's and with F's tokens.
            let mut g_probe = waiting_probe(&core, 4, (1000..1016).collect());
            let mut f_probe = waiting_probe(&core, 5, (2000..2016).collect());
            let hits = (
                waiting_hit(&core, &mut g_probe),
                waiting_hit(&core, &mut f_probe),
            );
            if max_concurrent_batches == 1 {
                // vLLM's order in both modes: F's attention block at 8 goes
                // first, G's state at 8 stays.
                assert_eq!(left, [true, false, true, true], "{version}");
                assert_eq!(hits, (8, 0), "{version}");
            } else {
                // Replay with async scheduling: G's state at 8 goes first, so
                // the G prompt hits less and the F prompt more than in vLLM.
                assert_eq!(left, [true, true, true, false], "{version}");
                assert_eq!(hits, (0, 8), "{version}");
            }
        }
    }
}

#[test]
fn grouped_prefix_copy_sources_leave_at_the_end_of_the_pass_with_async_scheduling() {
    // The tight-pool case with async scheduling. Without a KV connector in a
    // consumer role vLLM does not defer its block frees (v0.31
    // sched:156, :175-181, v0.32 sched:167, :186-192, v0.33 sched:169,
    // :189-195), so the copy sources still leave at the end of `schedule()`
    // (v0.31 sched:1434-1442, v0.32 sched:1474-1482, v0.33 sched:1698-1706).
    // RN waits one pass on pools of 13 to 17 blocks, as
    // with one batch. RC's blocks leave one pass later, so on 12 blocks RN
    // starts one pass later than with one batch.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for blocks in 12..=18 {
            let expected = match blocks {
                12 => (0, 5, vec![7, 7, 7, 7, 0, 6, 6, 6, 0], 12),
                18 => (0, 0, vec![13, 13, 13, 0], 18),
                _ => (0, 1, vec![7, 13, 13, 13, 0], 12),
            };
            assert_eq!(
                tight_pool_case(version, blocks, 2),
                expected,
                "{version} on {blocks} blocks"
            );
        }
    }
}

#[test]
fn grouped_prefix_boundary_hit_holds_its_state_blocks_until_the_next_pass_with_async_scheduling() {
    // The boundary-hit case with async scheduling. Replay holds RC's hit
    // blocks until RC's next allocation, one pass, as with one batch. vLLM
    // frees them on the processed token count, so with a step in flight it
    // holds them one more step (v0.31 kv_cache_manager.py:552-556, v0.32
    // :555-559, v0.33 :564-568). Replay does not model that step. RC's
    // blocks leave one pass later than with one batch, so on 12 and 13 blocks
    // RN starts one pass later.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        for blocks in 12..=18 {
            let expected = match blocks {
                12 | 13 => (0, 5, vec![12, 8, 8, 8, 0, 6, 6, 6, 0], 12),
                18 => (0, 0, vec![18, 14, 14, 0], 18),
                _ => (0, 1, vec![12, 14, 14, 14, 0], 12),
            };
            assert_eq!(
                pool_case(version, blocks, 2, 4400, 6528, 4352),
                expected,
                "{version} on {blocks} blocks"
            );
        }
    }
}

/// A waiting request outside the core's queues, with the prefix hashes that
/// the core gives a request it receives.
fn waiting_probe(core: &VllmCore, id: u128, tokens: Vec<u32>) -> RequestKvState {
    let mut probe = RequestKvState::native(
        Uuid::from_u128(id),
        tokens.clone(),
        4,
        4,
        core.args.block_size,
        true,
        false,
        false,
        None,
    );
    probe
        .lease
        .configure_prefix_hashes(&tokens, core.args.prefix_match_unit.unwrap());
    probe
}

/// The cached tokens that the waiting-loop lookup gives `probe`, checked
/// against a fresh search of the same pool. The lookup that `schedule_request`
/// uses must give the same cost from the hit that the waiting lookup stored.
fn waiting_hit(core: &VllmCore, probe: &mut RequestKvState) -> usize {
    let lookup = core
        .kv_manager
        .get_native_waiting_prefill_cost(&probe.sequence, &mut probe.lease);
    let fresh = core
        .kv_manager
        .get_native_prefill_cost(&probe.sequence, &probe.lease);
    assert_eq!(
        (
            lookup.new_blocks,
            lookup.cached_tokens,
            lookup.active_cached_tokens
        ),
        (
            fresh.new_blocks,
            fresh.cached_tokens,
            fresh.active_cached_tokens
        )
    );
    let scheduled = core
        .kv_manager
        .lookup_native_prefill_cost(&probe.sequence, &mut probe.lease);
    assert_eq!(
        (
            scheduled.new_blocks,
            scheduled.cached_tokens,
            scheduled.active_cached_tokens
        ),
        (
            lookup.new_blocks,
            lookup.cached_tokens,
            lookup.active_cached_tokens
        )
    );
    lookup.cached_tokens
}

#[test]
fn grouped_prefix_waiting_hit_follows_keys_that_enter_and_leave_the_pool() {
    // One full-attention group of 4-token blocks in a pool of 16 blocks. A
    // waiting request of 16 tokens is looked up in every step. Its hit must
    // follow the pool: none, then the keys of a finished equal request, then
    // none again once a long request evicts them.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut full = group("full", 4, 8, None);
        full.prefix_cacheable = Some(true);
        let mut core = prefix_core(vec![full], 16 * 8, 64, 4, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        let mut probe = waiting_probe(&core, 2001, (0..16).collect());
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{version}");
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{version}");
        submit_tokens(&mut core, 2002, (0..16).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        // Capped below the prompt: (16 - 1) / 4 * 4 = 12.
        assert_eq!(waiting_hit(&core, &mut probe), 12, "{version}");
        assert_eq!(waiting_hit(&core, &mut probe), 12, "{version}");
        // 64 new tokens take all 16 blocks and evict the four cached ones.
        submit_tokens(&mut core, 2003, (1000..1064).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{version}");
    }
}

#[test]
fn grouped_prefix_waiting_hit_follows_the_state_keys_published_in_the_step() {
    // A state key published in a step is no hit in that step
    // (single:1726-1734). A 13-token request publishes its state at 12 in
    // its one pass. A waiting 33-token request repeats it: between that pass
    // and the next step its state lookup misses at 12, so it hits nothing.
    // The next step clears the set, and the same request hits 12 tokens with
    // no change of the resident keys.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut core = fine_grained_core(version);
        let mut trace = TraceCollector::default();
        let mut probe = waiting_probe(&core, 2101, (0..33).collect());
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{version}");
        submit_tokens(&mut core, 2102, (0..13).collect(), 1);
        let pass = core.execute_pass(&mut trace, 0.0);
        assert_eq!(pass.admissions.len(), 1, "{version}");
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{version}");
        core.kv_manager.begin_step();
        assert_eq!(waiting_hit(&core, &mut probe), 12, "{version}");
    }
}

#[test]
fn grouped_prefix_waiting_hit_follows_the_token_count_of_the_request() {
    // The hit stops below the last known token (mgr:289-295). A waiting
    // request of 8 tokens hits 4 of a cached 16-token prefix; one more token
    // lifts the cap, and the same pool gives 8.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut full = group("full", 4, 8, None);
        full.prefix_cacheable = Some(true);
        let mut core = prefix_core(vec![full], 16 * 8, 64, 4, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 2201, (0..16).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        let mut probe = waiting_probe(&core, 2202, (0..8).collect());
        assert_eq!(waiting_hit(&core, &mut probe), 4, "{version}");
        probe.generate_token();
        assert_eq!(probe.len(), 9, "{version}");
        assert_eq!(waiting_hit(&core, &mut probe), 8, "{version}");
    }
}

#[test]
fn grouped_prefix_waiting_hit_follows_new_prefix_hashes_of_the_request() {
    // The prefix hashes are an input of the search. A waiting request that
    // hits 12 tokens of a cached prefix hits nothing once its hashes are set
    // from other tokens, with no change of the pool.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let mut full = group("full", 4, 8, None);
        full.prefix_cacheable = Some(true);
        let mut core = prefix_core(vec![full], 16 * 8, 64, 4, version);
        let mut trace = TraceCollector::default();
        let mut step = 0;
        submit_tokens(&mut core, 2301, (0..16).collect(), 1);
        drain(&mut core, &mut trace, &mut step);
        let mut probe = waiting_probe(&core, 2302, (0..16).collect());
        assert_eq!(waiting_hit(&core, &mut probe), 12, "{version}");
        let other: Vec<u32> = (5000..5016).collect();
        probe.lease.configure_prefix_hashes(&other, 4);
        assert_eq!(waiting_hit(&core, &mut probe), 0, "{version}");
    }
}

/// `(id, submit pass, prompt token ranges, output tokens)`.
type PlannedRequest = (u128, usize, &'static [(u32, u32)], usize);

/// Twelve requests with shared prefixes, chunked prefill and eviction.
const SINGLE_GROUP_PLAN: [PlannedRequest; 12] = [
    (901, 0, &[(0, 300)], 3),
    (902, 0, &[(0, 300)], 3),
    (903, 3, &[(0, 256)], 2),
    (904, 6, &[(0, 100), (1000, 1150)], 4),
    (905, 8, &[(5000, 6600)], 5),
    (906, 9, &[(7000, 9000)], 6),
    (907, 10, &[(10_000, 12_500)], 3),
    (908, 30, &[(13_000, 14_600)], 2),
    (909, 45, &[(5000, 6600), (30_000, 30_030)], 3),
    (910, 60, &[(40_000, 40_192), (50_000, 50_040)], 2),
    (911, 60, &[(40_000, 40_192), (51_000, 51_010)], 2),
    (912, 70, &[(0, 300), (60_000, 60_500)], 4),
];

/// A linear core and a grouped core with one full-attention group of the
/// same block and capacity, with every other setting equal.
fn single_group_cores(version: &str) -> (VllmCore, VllmCore) {
    let args = || {
        MockEngineArgs::builder()
            .block_size(64)
            .max_num_batched_tokens(Some(256))
            .max_num_seqs(Some(4))
            .max_model_len(Some(4096))
            .enable_prefix_caching(true)
            .speedup_ratio(0.0)
    };
    let linear = VllmCore::new(args().num_gpu_blocks(128).build().unwrap());
    let mut full = group("full", 64, 1, None);
    full.prefix_cacheable = Some(true);
    let grouped = VllmCore::new(
        args()
            .num_gpu_blocks(1)
            .prefix_match_unit(Some(64))
            .vllm_version(Some(version.to_string()))
            .kv_cache_groups(vec![full])
            .kv_cache_capacity_bytes(Some(128))
            .build()
            .unwrap(),
    );
    (linear, grouped)
}

#[test]
fn single_full_attention_group_matches_the_linear_pool() {
    // Every rule row. One full-attention group at the hash block leaves every
    // row difference inert (the last cacheable stop, the partial tail and the
    // checkpoint block), so every row matches the linear pool.
    for version in ["0.31.0", "0.32.0", "0.33.0"] {
        let (mut linear, mut grouped) = single_group_cores(version);
        let mut linear_trace = TraceCollector::default();
        let mut grouped_trace = TraceCollector::default();
        let computed = |core: &VllmCore| {
            let mut computed: Vec<(Uuid, usize)> = core
                .state
                .requests
                .iter()
                .map(|(id, request)| (*id, request.num_computed_tokens))
                .collect();
            computed.sort();
            computed
        };
        let mut next = 0;
        let mut hits = Vec::new();
        for pass in 0..400 {
            while let Some(&(id, at, ranges, output)) = SINGLE_GROUP_PLAN.get(next)
                && at == pass
            {
                let tokens: Vec<u32> = ranges.iter().flat_map(|&(a, b)| a..b).collect();
                submit_tokens(&mut linear, id, tokens.clone(), output);
                submit_tokens(&mut grouped, id, tokens, output);
                next += 1;
            }
            let expected = linear.execute_pass(&mut linear_trace, pass as f64);
            let actual = grouped.execute_pass(&mut grouped_trace, pass as f64);
            let reused = |admissions: &[crate::engine::scheduler::AdmissionEvent]| {
                admissions
                    .iter()
                    .map(|admission| (admission.uuid, admission.reused_input_tokens))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                reused(&actual.admissions),
                reused(&expected.admissions),
                "{version} pass {pass}"
            );
            assert_eq!(
                computed(&grouped),
                computed(&linear),
                "{version} pass {pass}"
            );
            assert_eq!(
                grouped.kv_manager.num_active_blocks(),
                linear.kv_manager.num_active_blocks(),
                "{version} pass {pass}"
            );
            assert!(
                expected
                    .output_signals
                    .iter()
                    .chain(&actual.output_signals)
                    .all(|output| !output.rejected)
            );
            hits.extend(reused(&actual.admissions));
            if next == SINGLE_GROUP_PLAN.len() && linear.is_drained() && grouped.is_drained() {
                break;
            }
        }
        assert!(linear.is_drained() && grouped.is_drained(), "{version}");
        // The linear cap is `known - 1` floored to the block
        // (`apply_prefix_recompute`), the grouped one `num_tokens - 1` floored to
        // the alignment (`mgr:289-295`). With one group both are the block.
        let id = Uuid::from_u128;
        assert_eq!(
            hits,
            vec![
                (id(901), 0),
                // The four blocks that request 901 holds.
                (id(902), 256),
                // Four cached blocks, capped below the prompt: (256 - 1) / 64 * 64.
                (id(903), 192),
                (id(904), 64),
                (id(905), 0),
                (id(906), 0),
                (id(907), 0),
                (id(908), 0),
                (id(909), 1600),
                (id(910), 0),
                // Request 910's three blocks, in the pass that computes them.
                (id(911), 192),
                // Eviction left block 0 of request 901 only.
                (id(912), 64),
            ],
            "{version}"
        );
    }
}
