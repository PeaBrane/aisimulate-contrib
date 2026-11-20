// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rule and coordinator cases of the grouped prefix cache, with the vLLM
//! 0.31.0, 0.32.0 and 0.33.0 literals re-derived from the pinned trees, the
//! 0.33.0 ones at `4ba787ec9959e566d3b587a796dff07673324da5`. Some cases are
//! ported from the vLLM tests at `db9527a46873454610df6dbedf79a36d6bf1a7f6` and
//! `5df63a7203cdb9328d824a27b682e77c7b9928be`. Copyright contributors to the
//! vLLM project. See THIRD_PARTY_NOTICES.md. Apache-2.0.
//!
//! - https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py
//! - https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/tests/v1/core/test_mamba_align_chunk_split.py

use super::*;
use crate::engine::cache::vllm_block_pool::CacheKey;
use crate::engine::common::hashing::SequenceHash;
use crate::engine::kv_manager::grouped::{GroupLease, GroupedBlockPool, GroupedKvPool};
use crate::engine::kv_manager::vllm_backend::BlockRequestLease;
use crate::perfmodel::{FpmCacheGroup, FpmCacheKind};
use uuid::Uuid;

const BUDGET: usize = 8192;
const GLM_BLOCK: u32 = 2176;

fn group(
    name: &str,
    kind: FpmCacheKind,
    block: u32,
    window: Option<u32>,
    state: Option<(bool, u32)>,
) -> FpmCacheGroup {
    FpmCacheGroup {
        name: name.into(),
        kind,
        num_layers: 1,
        block_size_tokens: block,
        page_size_bytes: 1,
        sliding_window: window,
        blocks_per_request: state.map(|_| 1),
        prefix_cacheable: Some(state.is_none_or(|(cacheable, _)| cacheable)),
        transfer_bytes_per_block: None,
        prefill_checkpoint_blocks: state.and_then(|(cacheable, ckpt)| cacheable.then_some(ckpt)),
        prefill_checkpoint_alignment: state
            .and_then(|(cacheable, ckpt)| (cacheable && ckpt == 1).then_some(16)),
    }
}

fn attention(name: &str, block: u32, window: Option<u32>) -> FpmCacheGroup {
    group(name, FpmCacheKind::Attention, block, window, None)
}

fn state(name: &str, block: u32, cacheable: bool, checkpoint: bool) -> FpmCacheGroup {
    group(
        name,
        FpmCacheKind::State,
        block,
        None,
        Some((cacheable, u32::from(checkpoint))),
    )
}

/// The six GLM-5.3-Flash TP 4 cache groups that the vLLM 0.33.0 runtime
/// probe records (fixture `glm53flash_vllm_0330_probe`): MLA attention and
/// four KDA `MambaSpec` states with 2,176-token blocks, and a 4-token
/// `CircularBufferSpec` ring. The tests use them as hand-made profiles of
/// vLLM 0.31.0, 0.32.0 and 0.33.0, the versions that the rule table holds. The
/// helper lists them as MLA, ring, KDA. The record lists the four Mamba
/// groups first, then MLA, then the ring. With `checkpoint`, G2 to G5
/// carry `prefill_checkpoint_blocks: Some(1)` and alignment 16.
fn groups_glm(checkpoint: bool) -> Vec<FpmCacheGroup> {
    let mut groups = vec![
        attention("mla", GLM_BLOCK, None),
        state("kpool_tail", 4, false, false),
    ];
    for i in 0..4 {
        groups.push(state(&format!("kda_{i}"), GLM_BLOCK, true, checkpoint));
    }
    groups
}

fn rules_glm_hash(
    version: VllmRuleVersion,
    checkpoint: bool,
    hash_block: usize,
) -> GroupedPrefixRules {
    GroupedPrefixRules::new(&groups_glm(checkpoint), hash_block, version).unwrap()
}

fn rules_glm(version: VllmRuleVersion, checkpoint: bool) -> GroupedPrefixRules {
    rules_glm_hash(version, checkpoint, GLM_BLOCK as usize)
}

fn rules(groups: &[FpmCacheGroup], hash_block: usize) -> GroupedPrefixRules {
    GroupedPrefixRules::new(groups, hash_block, VllmRuleVersion::V0_31_0).unwrap()
}

/// A block pool with `blocks` pool blocks for `groups`.
fn kv_pool(groups: &[FpmCacheGroup], blocks: u64) -> GroupedKvPool {
    GroupedKvPool::with_prefix_cache(groups.to_vec(), blocks, 1)
}

/// A lease with chained hashes of the tokens `0..tokens` at `hash_block`.
fn lease(tokens: usize, hash_block: usize) -> BlockRequestLease {
    let tokens: Vec<u32> = (0..tokens as u32).collect();
    let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
    lease.configure_prefix_hashes(&tokens, hash_block);
    lease
}

fn hash_at(hash_block: usize, tokens: usize) -> SequenceHash {
    lease(tokens, hash_block).prefix_hash_at(tokens).unwrap()
}

fn key(group: u16, hash_block: usize, tokens: usize) -> CacheKey {
    CacheKey::Grouped {
        group,
        hash: hash_at(hash_block, tokens),
    }
}

/// Publish one block under `key` and leave it in the inactive LRU, as a
/// finished request leaves its cached blocks.
fn seed(pool: &mut GroupedBlockPool, key: CacheKey) {
    let mut reservation = pool.reserve(&[], 1).unwrap();
    let id = pool.take_private(&mut reservation);
    assert!(pool.cache(id, key));
    pool.release(id);
}

fn seed_many(pool: &mut GroupedBlockPool, group: u16, hash_block: usize, ends: &[usize]) {
    for &end in ends {
        seed(pool, key(group, hash_block, end));
    }
}

fn seed_states(pool: &mut GroupedBlockPool, groups: &[u16], hash_block: usize, end: usize) {
    for &group in groups {
        seed(pool, key(group, hash_block, end));
    }
}

fn hit_exists(pool: &GroupedBlockPool, group: u16, hash_block: usize, tokens: usize) -> bool {
    pool.key_hit(group, hash_at(hash_block, tokens)).is_some()
}

fn state_lease(lease: &GroupedBlockLease, group: usize) -> &GroupStateLease {
    match &lease.groups[group] {
        GroupLease::State(state) => state,
        other => panic!("group {group} is not a state lease: {other:?}"),
    }
}

fn attention_copies(lease: &GroupedBlockLease, group: usize) -> usize {
    match &lease.groups[group] {
        GroupLease::Attention { copies, .. } => copies.iter().flatten().count(),
        other => panic!("group {group} is not an attention lease: {other:?}"),
    }
}

fn split(
    rules: &GroupedPrefixRules,
    start: usize,
    prompt: usize,
    known: usize,
    desired: usize,
    junction: usize,
) -> usize {
    rules.align_split(start, prompt, known, desired, junction, BUDGET)
}

/// Run a whole prefill of `prompt` tokens from `hit`: the chunks of
/// `align_split` at `budget`, one step each. Returns the chunk ends.
fn prefill(
    coordinator: &mut GroupedPrefixCoordinator,
    pool: &mut GroupedBlockPool,
    lease: &mut GroupedBlockLease,
    request: &BlockRequestLease,
    prompt: usize,
    hit: &GroupedHit,
    budget: usize,
) -> Vec<usize> {
    let mut ends = Vec::new();
    let mut computed = hit.tokens;
    let mut step = 1;
    while computed < prompt {
        coordinator.begin_step();
        let chunk = coordinator.rules().align_split(
            computed,
            prompt,
            prompt,
            budget.min(prompt - computed),
            hit.tokens + hit.junction_tokens,
            budget,
        );
        assert!(chunk > 0, "the split must make progress at {computed}");
        let target = computed + chunk;
        assert!(coordinator.allocate(pool, lease, prompt, computed, target, step));
        coordinator.finalize(pool, lease, request, prompt, target, hit, step);
        coordinator.release_copy_sources(pool);
        computed = target;
        ends.push(target);
        step += 1;
    }
    coordinator.begin_step();
    ends
}

const GLM_STATES: [u16; 4] = [2, 3, 4, 5];

/// One batch of `BUDGET` tokens in flight and no model-length cap.
const LIMITS: AdmissionLimits = AdmissionLimits {
    max_in_flight_tokens: BUDGET,
    max_model_len: usize::MAX,
};

fn with_limits(rules: GroupedPrefixRules) -> GroupedPrefixCoordinator {
    GroupedPrefixCoordinator::new(rules, LIMITS)
}

#[test]
fn version_rows_differ_in_the_last_cacheable_stop_and_the_partial_tail() {
    // v0.31 sched:444 `num_tokens - num_tokens % block_size`; v0.32
    // sched:476-480 with iface:1110-1122 `(n - 1) // B * B`.
    let v31 = rules_glm(VllmRuleVersion::V0_31_0, false);
    let v32 = rules_glm(VllmRuleVersion::V0_32_0, false);
    assert_eq!(v31.last_cacheable_stop(9000), 8704);
    assert_eq!(v32.last_cacheable_stop(9000), 8704);
    assert_eq!(v31.last_cacheable_stop(8704), 8704);
    assert_eq!(v32.last_cacheable_stop(8704), 6528);
    assert_eq!(v32.last_cacheable_stop(0), 0);
    // v0.31 sched:482-487 `prompt // hash * hash`; v0.32 sched:517-525
    // `(prompt - 1) // hash * hash`.
    let v31 = rules_glm_hash(VllmRuleVersion::V0_31_0, false, 4);
    let v32 = rules_glm_hash(VllmRuleVersion::V0_32_0, false, 4);
    assert_eq!(v31.partial_tail(9000), 9000);
    assert_eq!(v32.partial_tail(9000), 8996);
    assert_eq!(v31.partial_tail(4356), 4356);
    assert_eq!(v32.partial_tail(4356), 4352);
    assert_eq!(
        VllmRuleVersion::parse("0.31.0"),
        Some(VllmRuleVersion::V0_31_0)
    );
    assert_eq!(
        VllmRuleVersion::parse("0.32.0"),
        Some(VllmRuleVersion::V0_32_0)
    );
    assert_eq!(VllmRuleVersion::parse("0.30.0"), None);
}

#[test]
fn vllm_0330_row_takes_the_0320_arm_in_every_version_branch() {
    assert_eq!(
        VllmRuleVersion::parse("0.33.0"),
        Some(VllmRuleVersion::V0_33_0)
    );
    assert_eq!(VllmRuleVersion::parse("0.33.1"), None);
    assert_eq!(VllmRuleVersion::V0_31_0.label(), "0.31.0");
    assert_eq!(VllmRuleVersion::V0_32_0.label(), "0.32.0");
    assert_eq!(VllmRuleVersion::V0_33_0.label(), "0.33.0");
    // v0.33 sched:579-584 with iface:1169-1186: `(n - max(1, L)) // B * B`.
    // The hash lookahead L is 0 without speculative decoding (v0.33
    // `vllm/config/vllm.py:677-692`), which grouped execution refuses.
    let v32 = rules_glm(VllmRuleVersion::V0_32_0, false);
    let v33 = rules_glm(VllmRuleVersion::V0_33_0, false);
    for (n, stop) in [(9000, 8704), (8704, 6528), (0, 0)] {
        assert_eq!(v33.last_cacheable_stop(n), stop);
        assert_eq!(v32.last_cacheable_stop(n), stop);
    }
    // v0.33 sched:622-631: `(prompt - 1) // hash * hash`.
    let fine = rules_glm_hash(VllmRuleVersion::V0_33_0, false, 4);
    assert_eq!(fine.partial_tail(9000), 8996);
    assert_eq!(fine.partial_tail(4356), 4352);
    // v0.33 sched:437-449: checkpoint mode only when every Mamba group has
    // a positive count.
    assert!(v33.checkpoint.is_none());
    let ckpt = rules_glm(VllmRuleVersion::V0_33_0, true);
    assert_eq!(ckpt.checkpoint, Some(CheckpointRule { alignment: 16 }));
    // The align split (v0.33 sched:546-662, iface:1189-1210).
    assert_eq!(split(&v33, 0, 9000, 9000, 8192, 0), 6528);
    assert_eq!(split(&v33, 6528, 9000, 9000, 2472, 0), 2176);
    assert_eq!(split(&v33, 4352, 8704, 8704, 4352, 0), 2176);
    assert_eq!(ckpt.checkpoint_position(9000), 8704);
    assert!(ckpt.checkpoint_valid(6528, 9000, 8704));
    assert_eq!(split(&ckpt, 6528, 9000, 9000, 2472, 0), 2472);
    assert_eq!(split(&ckpt, 4352, 8704, 8704, 4352, 0), 4352);
}

#[test]
fn vllm_0330_row_refuses_checkpoint_fields_with_its_own_texts() {
    let refusal = |groups: &[FpmCacheGroup], hash: usize| {
        GroupedPrefixRules::new(groups, hash, VllmRuleVersion::V0_33_0)
            .unwrap_err()
            .to_string()
    };
    let mut groups = groups_glm(false);
    groups[2].prefill_checkpoint_blocks = None;
    assert_eq!(
        refusal(&groups, 2176),
        "kv_cache_groups with vllm_version=0.33.0 require prefill_checkpoint_blocks on every prefix-cacheable state group; set the field (0 or 1) on each such group"
    );
    let mut groups = groups_glm(true);
    groups[3].prefill_checkpoint_blocks = Some(0);
    groups[3].prefill_checkpoint_alignment = None;
    assert_eq!(
        refusal(&groups, 2176),
        "kv_cache_groups with vllm_version=0.33.0 require the same prefill_checkpoint_blocks across state groups"
    );
    let mut groups = groups_glm(true);
    groups[3].prefill_checkpoint_alignment = Some(32);
    assert_eq!(
        refusal(&groups, 2176),
        "kv_cache_groups with vllm_version=0.33.0 require one prefill_checkpoint_alignment across state groups"
    );
    let small_attention = [attention("full", 16, None), state("s", 64, true, true)];
    assert_eq!(
        refusal(&small_attention, 16),
        "kv_cache_groups with vllm_version=0.33.0 require every state group with prefill checkpoint blocks to have the smallest block_size_tokens of the prefix_cacheable groups"
    );
}

#[test]
fn second_cache_step_of_a_keyed_state_block_defers_no_waiting_request_on_vllm_0330() {
    // vLLM 0.33.0 does not publish again a state block that already has the
    // key of its position (v0.33 single:2112-2115), so the key does not go
    // into the set of the step (v0.33 single:1994-1999) and a request that
    // hits it does not wait (v0.33 single:1704-1712). Replay publishes a
    // state key once, so the second cache step of the producer marks nothing.
    let groups = [attention("full", 16, None), state("s", 16, true, false)];
    let hash = 16;
    let rules = GroupedPrefixRules::new(&groups, hash, VllmRuleVersion::V0_33_0).unwrap();
    let mut coordinator = with_limits(rules);
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let producer = lease(40, hash);
    let hit = coordinator.find_longest_hit(pool, &producer, 40);
    let mut block_lease = coordinator.new_lease();
    // Step 1: the chunk ends at the last cacheable stop (40 - 1) / 16 * 16.
    coordinator.begin_step();
    assert_eq!(
        coordinator
            .rules()
            .align_split(0, 40, 40, BUDGET, 0, BUDGET),
        32
    );
    assert!(coordinator.allocate(pool, &mut block_lease, 40, 0, 32, 1));
    coordinator.finalize(pool, &mut block_lease, &producer, 40, 32, &hit, 1);
    // In the publishing step the key at 32 is no hit.
    assert_eq!(
        coordinator
            .find_longest_hit(pool, &lease(48, hash), 48)
            .tokens,
        0
    );
    // Step 2: the producer computes 32..40. Its cache step is still in the
    // block that ends at 32.
    coordinator.begin_step();
    assert_eq!(
        coordinator
            .rules()
            .align_split(32, 40, 40, BUDGET, 0, BUDGET),
        8
    );
    assert!(coordinator.allocate(pool, &mut block_lease, 40, 32, 40, 2));
    coordinator.finalize(pool, &mut block_lease, &producer, 40, 40, &hit, 2);
    // A request that waits in step 2 gets the hit of 32 tokens in step 2.
    assert_eq!(
        coordinator
            .find_longest_hit(pool, &lease(48, hash), 48)
            .tokens,
        32
    );
}

#[test]
fn rules_refuse_unsupported_checkpoint_fields() {
    // v0.32 sched:390-394 runs checkpoint mode only when every Mamba
    // group has a positive count.
    let err = GroupedPrefixRules::new(&groups_glm(true), 2176, VllmRuleVersion::V0_31_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with vllm_version=0.31.0 do not model prefill checkpoint blocks"
    );
    let mut groups = groups_glm(false);
    groups[2].prefill_checkpoint_blocks = None;
    let err = GroupedPrefixRules::new(&groups, 2176, VllmRuleVersion::V0_32_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with vllm_version=0.32.0 require prefill_checkpoint_blocks on every prefix-cacheable state group; set the field (0 or 1) on each such group"
    );
    let mut groups = groups_glm(true);
    groups[3].prefill_checkpoint_alignment = Some(32);
    let err = GroupedPrefixRules::new(&groups, 2176, VllmRuleVersion::V0_32_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with vllm_version=0.32.0 require one prefill_checkpoint_alignment across state groups"
    );
    let zero = rules_glm(VllmRuleVersion::V0_32_0, false);
    assert!(zero.checkpoint.is_none());
    let one = rules_glm(VllmRuleVersion::V0_32_0, true);
    assert_eq!(one.checkpoint, Some(CheckpointRule { alignment: 16 }));
    // Block mode holds one running block per state group, so another count
    // is refused.
    let mut groups = groups_glm(false);
    groups[4].blocks_per_request = Some(2);
    let err = GroupedPrefixRules::new(&groups, 2176, VllmRuleVersion::V0_31_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with enable_prefix_caching=true model one state block per request; blocks_per_request must be 1"
    );
    // utils:799-805 and coord:87-92.
    assert!(GroupedPrefixRules::new(&groups_glm(false), 3, VllmRuleVersion::V0_31_0).is_err());
}

#[test]
fn rules_refuse_a_checkpoint_block_on_a_ring() {
    // The count is a field of `MambaSpec` alone (v0.31 iface:1049-1050,
    // v0.32 iface:1042-1043, v0.33 iface:1101-1102). A ring that declares a
    // block is refused on every row. A ring with no count or a zero count
    // stays a ring.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut groups = vec![attention("full", 16, None), state("ring", 4, false, false)];
        for count in [None, Some(0)] {
            groups[1].prefill_checkpoint_blocks = count;
            let rules = GroupedPrefixRules::new(&groups, 16, version).unwrap();
            assert_eq!(rules.groups[1].kind, GroupRuleKind::Ring, "{version:?}");
        }
        groups[1].prefill_checkpoint_blocks = Some(1);
        groups[1].prefill_checkpoint_alignment = Some(16);
        assert!(groups[1].validate().is_ok());
        let err = GroupedPrefixRules::new(&groups, 16, version)
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "kv_cache_groups with enable_prefix_caching=true do not model a prefill checkpoint block on a state group with prefix_cacheable=false; set prefix_cacheable: true on a Mamba state group, or remove prefill_checkpoint_blocks and prefill_checkpoint_alignment from a ring buffer",
            "{version:?}"
        );
    }
}

#[test]
fn rules_refuse_missing_prefix_cacheability() {
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        for missing in 0..2 {
            let mut groups = [attention("full", 16, None), state("s", 16, true, false)];
            groups[missing].prefix_cacheable = None;
            // Legacy profiles permit absence, but it cannot identify a
            // state group as a ring when prefix caching is enabled.
            assert!(groups[missing].validate().is_ok());
            let err = GroupedPrefixRules::new(&groups, 16, version)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("require prefix_cacheable on every group"),
                "{err}"
            );
        }
    }
}

#[test]
fn rules_refuse_a_non_cacheable_sliding_window_group() {
    // The window specs that opt out of prefix caching: the v0.31 ring
    // `KpoolTailSpec` (iface:1015-1038), which GLM-5.3-Flash builds with
    // `block_size=ring` and `sliding_window=ring`
    // (glm5next/common/attention.py:205-212; v0.32 and v0.33 have no such
    // spec), and a bounded replay window (`SlidingWindowMLASpec`, v0.31
    // iface:949-955, v0.32 :968-970, v0.33 :1027-1029).
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut groups = [
            attention("full", 16, None),
            attention("other", 16, Some(17)),
        ];
        groups[1].prefix_cacheable = Some(false);
        let err = GroupedPrefixRules::new(&groups, 16, version)
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "kv_cache_groups with enable_prefix_caching=true do not model a sliding-window attention group with prefix_cacheable=false; describe a ring buffer as a state group with blocks_per_request: 1 and prefix_cacheable: false, and turn prefix caching off for a bounded replay window, which replay does not model",
            "{version:?}"
        );
        // The same ring as a state group is accepted.
        let ring = [attention("full", 16, None), state("ring", 16, false, false)];
        assert!(GroupedPrefixRules::new(&ring, 16, version).is_ok());
    }
}

#[test]
fn non_cacheable_full_attention_group_takes_no_part_in_the_hit() {
    // Two full-attention groups of block 16, the second with
    // `prefix_cacheable: false`. vLLM's hybrid coordinator skips a group
    // that opts out of prefix caching in the lookup, so its slot of the
    // hit stays empty (v0.31 coord:743-749, v0.32 coord:713-720, v0.33
    // coord:723-730). Its manager then counts `cdiv(17, 16) = 2` new blocks
    // with no computed block (v0.31 single:216-268, v0.32 single:213-265,
    // v0.33 single:214-266). The first group hits 16 tokens and needs one
    // new block and its inactive hit block.
    let mut groups = [attention("full", 16, None), attention("other", 16, None)];
    groups[1].prefix_cacheable = Some(false);
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 16, version).unwrap());
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[16]);
        let request = lease(17, 16);
        let hit = coordinator.find_longest_hit(pool, &request, 17);
        assert_eq!(hit.tokens, 16, "{version:?}");
        assert_eq!(hit.per_group[1], GroupHitBlocks::None, "{version:?}");
        assert_eq!(coordinator.admission_blocks(pool, &hit, 17), 2 + 2);
        let mut block_lease = coordinator.new_lease();
        coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
        assert!(coordinator.allocate(pool, &mut block_lease, 17, 16, 17, 1));
        assert_eq!(attention_copies(&block_lease, 0), 2);
        assert_eq!(attention_copies(&block_lease, 1), 2);
        coordinator.release_request(pool, &mut block_lease);
        assert_eq!(pool.active_blocks(), 0);
    }
}

#[test]
fn rules_refuse_convolution_groups() {
    // A convolution group keeps a window, and the table has no rule for it.
    let mut convolution = attention("convolution", 4, Some(4));
    convolution.kind = FpmCacheKind::Convolution;
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        for groups in [
            vec![convolution.clone()],
            vec![attention("full", 4, None), convolution.clone()],
        ] {
            let err = GroupedPrefixRules::new(&groups, 4, version)
                .unwrap_err()
                .to_string();
            assert_eq!(
                err,
                "grouped prefix caching does not support convolution cache groups; turn prefix caching off to replay them cold",
                "{version:?}"
            );
        }
    }
}

// The values below fit a 64-bit `usize` only.
#[cfg(target_pointer_width = "64")]
#[test]
fn rules_refuse_a_scheduler_block_above_usize() {
    // The group validation has no upper bound for block_size_tokens. The
    // LCM of these three blocks is 23,058,430,081,399,521,280, above
    // `usize::MAX`. vLLM computes `math.lcm` on Python integers, which
    // cannot overflow.
    let groups = [
        attention("a", 1 << 31, None),
        attention("b", (1 << 31) - 1, None),
        attention("c", 5, None),
    ];
    assert!(groups.iter().all(|group| group.validate().is_ok()));
    let err = GroupedPrefixRules::new(&groups, 1, VllmRuleVersion::V0_31_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with enable_prefix_caching=true require the least common multiple of every group's block_size_tokens (the scheduler block) to be at most 18446744073709551615; choose block_size_tokens with a smaller least common multiple"
    );
    // Two u32 blocks cannot overflow, and keep their exact LCM.
    let groups = [
        attention("a", u32::MAX, None),
        attention("b", u32::MAX - 1, None),
    ];
    assert_eq!(rules(&groups, 1).alignment, 18_446_744_060_824_649_730);
}

#[test]
fn alignment_is_the_hash_block_only_with_fine_grained_hits() {
    // coord:687-735. G2 to G5 are state rules with block 2176 and G0 is
    // full attention. At hash 2176 no state block exceeds the hash block.
    let glm = rules_glm(VllmRuleVersion::V0_31_0, false);
    assert_eq!(glm.alignment, 2176);
    assert_eq!(glm.cacheable_block, 2176);
    let fine = rules_glm_hash(VllmRuleVersion::V0_31_0, false, 4);
    assert_eq!(fine.alignment, 4);
    // Fine-grained hits need a Mamba group in align mode whose block exceeds
    // the hash block. Two attention groups have none, so they give the
    // scheduler block, with or without a window (coord:691-704).
    let windowed = [attention("a", 64, Some(100)), attention("b", 16, None)];
    assert_eq!(rules(&windowed, 16).alignment, 64);
    let plain = [attention("a", 64, None), attention("b", 16, None)];
    assert_eq!(rules(&plain, 16).alignment, 64);
    // A state group above the hash block turns fine-grained hits on when
    // every other cacheable rule is full attention or at the hash block.
    let with_state = [
        attention("a", 64, None),
        attention("b", 16, None),
        state("s", 64, true, false),
    ];
    assert_eq!(rules(&with_state, 16).alignment, 16);
    // A window at another block than the hash block turns them off again
    // (single:960 has no sub-block lookup).
    let with_window = [
        attention("a", 64, Some(100)),
        attention("b", 16, None),
        state("s", 64, true, false),
    ];
    assert_eq!(rules(&with_window, 16).alignment, 64);
}

#[test]
fn full_attention_hit_is_capped_below_the_prompt() {
    // mgr:289-295 caps the lookup at `num_tokens - 1`.
    let groups = [attention("full", 16, None)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 16, version).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[16, 32, 48, 64]);
        let hit = coordinator.find_longest_hit(pool, &lease(64, 16), 64);
        assert_eq!(hit.tokens, 48);
        assert_eq!(hit.junction_tokens, 0);
        assert_eq!(hit.per_group.len(), 1);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 3));
        let hit = coordinator.find_longest_hit(pool, &lease(65, 16), 65);
        assert_eq!(hit.tokens, 64);
    }
}

#[test]
fn sliding_window_hits_need_a_contiguous_run_before_an_aligned_end() {
    // single:960-1069: `cdiv(32, 16) == 2` contiguous blocks before an
    // aligned end, or a run from block 0.
    let groups = [
        attention("full", 16, None),
        attention("window", 16, Some(33)),
    ];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let r = GroupedPrefixRules::new(&groups, 16, version).unwrap();
        assert_eq!(r.alignment, 16);
        let coordinator = with_limits(r);
        let mut kv = kv_pool(&groups, 32);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[16, 32, 48, 64, 80, 96]);
        seed_many(pool, 1, 16, &[32, 48, 64, 96]);
        let hit = coordinator.find_longest_hit(pool, &lease(96, 16), 96);
        // The end 64 has blocks 2 and 3 before it. The end 80 lacks block 4.
        assert_eq!(hit.tokens, 64);
        match &hit.per_group[1] {
            GroupHitBlocks::Attention(keys) => {
                assert_eq!(keys.len(), 4);
                assert!(keys[0].is_none() && keys[1].is_none());
                assert_eq!(keys[2], Some(key(1, 16, 48)));
                assert_eq!(keys[3], Some(key(1, 16, 64)));
            }
            other => panic!("{other:?}"),
        }
        // Junction: the full-attention hit before the loop (80) minus 64.
        assert_eq!(hit.junction_tokens, 16);

        let mut kv = kv_pool(&groups, 32);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[16, 32, 48, 64, 80, 96]);
        seed_many(pool, 1, 16, &[16]);
        let hit = coordinator.find_longest_hit(pool, &lease(96, 16), 96);
        assert_eq!(hit.tokens, 16);
    }
}

#[test]
fn one_token_window_hits_only_at_a_cached_block() {
    // A window of one token needs `cdiv(0, 4) == 0` contiguous blocks
    // (v0.31 single:961-971, v0.32 :943-953). vLLM counts a block only
    // after it finds the block cached, so a hit still needs the cached
    // block at its end (v0.31 single:1021-1044, v0.32 :1003-1026).
    let groups = [attention("window", 4, Some(1))];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 32);
        let pool = kv.block_pool_mut().unwrap();
        let hit = coordinator.find_longest_hit(pool, &lease(100, 4), 100);
        assert_eq!((hit.tokens, hit.junction_tokens), (0, 0), "{version:?}");
        seed_many(pool, 0, 4, &[48]);
        let hit = coordinator.find_longest_hit(pool, &lease(100, 4), 100);
        assert_eq!((hit.tokens, hit.junction_tokens), (48, 0), "{version:?}");
        match &hit.per_group[0] {
            GroupHitBlocks::Attention(keys) => {
                assert_eq!(keys.len(), 12, "{version:?}");
                assert!(keys[..11].iter().all(Option::is_none), "{version:?}");
                assert_eq!(keys[11], Some(key(0, 4, 48)), "{version:?}");
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn sliding_window_admission_reserves_the_window_plus_the_in_flight_tokens() {
    // iface:859-866: `cdiv(min(window - 1 + max_in_flight_tokens,
    // max_model_len), block) + 1`, with `max_in_flight_tokens =
    // max_concurrent_batches * max_num_batched_tokens` (vllm.py:605-613).
    let groups = [
        attention("full", 16, None),
        attention("window", 16, Some(33)),
    ];
    let prompt: usize = 100_000;
    let hit = GroupedHit {
        tokens: 0,
        junction_tokens: 0,
        per_group: vec![
            GroupHitBlocks::Attention(Vec::new()),
            GroupHitBlocks::Attention(Vec::new()),
        ],
    };
    let full_blocks = prompt.div_ceil(16);
    let kv = kv_pool(&groups, 16);
    let pool = kv.block_pool().unwrap();
    // One batch of 8,192: (32 + 8192) / 16 + 1 = 515.
    let one_batch = with_limits(rules(&groups, 16));
    assert_eq!(
        one_batch.admission_blocks(pool, &hit, prompt),
        full_blocks + 515
    );
    // Two batches of 8,192: (32 + 16384) / 16 + 1 = 1027.
    let two_batches = GroupedPrefixCoordinator::new(
        rules(&groups, 16),
        AdmissionLimits {
            max_in_flight_tokens: 2 * BUDGET,
            max_model_len: usize::MAX,
        },
    );
    assert_eq!(
        two_batches.admission_blocks(pool, &hit, prompt),
        full_blocks + 1027
    );
    // The model length caps it: cdiv(4096, 16) + 1 = 257.
    let capped = GroupedPrefixCoordinator::new(
        rules(&groups, 16),
        AdmissionLimits {
            max_in_flight_tokens: 2 * BUDGET,
            max_model_len: 4096,
        },
    );
    assert_eq!(
        capped.admission_blocks(pool, &hit, prompt),
        full_blocks + 257
    );
    // A short prompt needs fewer blocks than the cap.
    assert_eq!(one_batch.admission_blocks(pool, &hit, 100), 7 + 7);
}

#[test]
fn sliding_window_admission_after_a_hit_is_vllms_whole_prompt_count() {
    // vLLM's whole-prompt test (v0.31 mgr:519-536, v0.32 mgr:522-539,
    // v0.33 mgr:531-548) for a sliding-window group (v0.31
    // single:216-268, v0.32 single:213-265, v0.33 single:214-266):
    //   required = min(cdiv(prompt, B), cap), with
    //   cap = cdiv(min(window - 1 + in_flight, max_model_len), B) + 1
    //         (v0.31 iface:859-866, v0.32 iface:878-885, v0.33
    //         iface:937-944);
    //   skipped = max(0, hit - window + 1) // B
    //         (v0.31 single:1185-1188, v0.32 single:1167-1170, v0.33
    //         single:1180-1183);
    //   new = max(required - max(skipped, len(hit blocks)), 0);
    //   plus the hit blocks from `skipped` on that no request holds.
    // The hit list has one entry per hit block, null blocks included.
    // vLLM compares the capped count with absolute block indices, so a
    // long hit gives no new block here. vLLM's second test, on the first
    // chunk and without the cap (v0.31 mgr:558-576, v0.32 mgr:561-579,
    // v0.33 mgr:570-588), is a separate count.
    //
    // C1, window 8, block 4, in flight 8, model length 200, prompt 108,
    // hit 96: cap = cdiv(min(7 + 8, 200), 4) + 1 = 5; required =
    // min(27, 5) = 5; skipped = 89 // 4 = 22; the hit list has 24
    // entries, so new = max(5 - 24, 0) = 0; the two hit blocks of the
    // window run count when no request holds them: 2, else 0.
    // (window, block, in flight, model length, prompt, hit, the cached
    // blocks that end at the hit, not held, held)
    type Case = (u32, u32, usize, usize, usize, usize, usize, usize, usize);
    let cases: [Case; 6] = [
        (8, 4, 8, 200, 108, 96, 2, 2, 0),
        (8, 4, 8, 200, 200, 96, 2, 2, 0),
        // C3: skipped = 5 // 4 = 1, 3 entries, new = 5 - 3 = 2.
        (8, 4, 8, 200, 108, 12, 2, 4, 2),
        // C4, cold: new = min(27, 5) = 5.
        (8, 4, 8, 200, 108, 0, 0, 5, 5),
        // C5: cap = cdiv(13, 4) + 1 = 5, skipped = 91 // 4 = 22.
        (6, 4, 8, 200, 108, 96, 2, 2, 0),
        // C6: cap = cdiv(min(127 + 16384, 131072), 16) + 1 = 1,033;
        // skipped = 49,873 // 16 = 3,117; 3,125 entries; new = 0; the
        // eight blocks of the window run.
        (128, 16, 16384, 131_072, 100_000, 50_000, 8, 8, 0),
    ];
    for (window, block, in_flight, model_len, prompt, hit_tokens, run, idle, held) in cases {
        let groups = [attention("window", block, Some(window))];
        for version in [
            VllmRuleVersion::V0_31_0,
            VllmRuleVersion::V0_32_0,
            VllmRuleVersion::V0_33_0,
        ] {
            let coordinator = GroupedPrefixCoordinator::new(
                GroupedPrefixRules::new(&groups, block as usize, version).unwrap(),
                AdmissionLimits {
                    max_in_flight_tokens: in_flight,
                    max_model_len: model_len,
                },
            );
            let mut kv = kv_pool(&groups, 16);
            let pool = kv.block_pool_mut().unwrap();
            let ends: Vec<usize> = (0..run).map(|i| hit_tokens - i * block as usize).collect();
            seed_many(pool, 0, block as usize, &ends);
            let request = lease(prompt, block as usize);
            let hit = coordinator.find_longest_hit(pool, &request, prompt);
            let case = format!("{version:?}, window {window}, prompt {prompt}");
            assert_eq!(hit.tokens, hit_tokens, "{case}");
            assert_eq!(
                coordinator.admission_blocks(pool, &hit, prompt),
                idle,
                "{case}"
            );
            let mut block_lease = coordinator.new_lease();
            coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
            assert_eq!(
                coordinator.admission_blocks(pool, &hit, prompt),
                held,
                "{case}"
            );
            coordinator.release_request(pool, &mut block_lease);
        }
    }
}

#[test]
fn state_group_hits_only_at_a_kept_boundary() {
    // single:1487-1564 scans aligned ends right to left for one cached
    // state block. pool:198-224: every group of the spec group must hold
    // the key.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
        seed_states(pool, &GLM_STATES, hash, 8704);
        let request = lease(9000, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        assert_eq!(hit.tokens, 8704);
        assert_eq!(hit.junction_tokens, 0);
        assert_eq!(
            hit.per_group[2],
            GroupHitBlocks::State(Some(key(2, hash, 8704)))
        );
        assert_eq!(hit.per_group[1], GroupHitBlocks::None);

        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
        seed_states(pool, &[2, 4, 5], hash, 8704);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        assert_eq!(hit.tokens, 0);
        assert_eq!(hit.per_group[0], GroupHitBlocks::Attention(Vec::new()));
        assert_eq!(hit.per_group[3], GroupHitBlocks::State(None));

        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
        seed_states(pool, &GLM_STATES, hash, 6528);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        assert_eq!(hit.tokens, 6528);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 3));
    }
}

#[test]
fn ring_group_never_hits_and_holds_one_block() {
    // single:1214-1230 claims one block per request; single:498-499
    // caches nothing for a group that is not prefix cacheable.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = groups_glm(false);
        let mut coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(2500, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 2500);
        assert_eq!(hit.tokens, 0);
        let mut block_lease = coordinator.new_lease();
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 2500, 0, 2176, 1));
        assert_eq!(
            block_lease.groups[1],
            GroupLease::Ring {
                copy: Some(ring_copy(&block_lease))
            }
        );
        coordinator.finalize(pool, &mut block_lease, &request, 2500, 2176, &hit, 1);
        assert_eq!(pool.inactive_cached_blocks(), 0);
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 2500, 2176, 2500, 2));
        coordinator.finalize(pool, &mut block_lease, &request, 2500, 2500, &hit, 2);
        coordinator.release_request(pool, &mut block_lease);
        coordinator.begin_step();

        let hit = coordinator.find_longest_hit(pool, &lease(2500, hash), 2500);
        assert_eq!(hit.tokens, 2176);
        assert_eq!(hit.per_group[1], GroupHitBlocks::None);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 1));
        assert_eq!(
            hit.per_group[2],
            GroupHitBlocks::State(Some(key(2, hash, 2176)))
        );
    }
}

fn ring_copy(lease: &GroupedBlockLease) -> crate::engine::cache::vllm_block_pool::BlockCopyId {
    match &lease.groups[1] {
        GroupLease::Ring { copy: Some(copy) } => *copy,
        other => panic!("{other:?}"),
    }
}

#[test]
fn hybrid_hit_is_the_longest_prefix_every_cacheable_group_serves() {
    // coord:842-977: the fixed point shortens to 4,352; coord:970-977
    // reports the uncached common prefix.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704, 10880]);
        seed_states(pool, &GLM_STATES, hash, 4352);
        let hit = coordinator.find_longest_hit(pool, &lease(11000, hash), 11000);
        assert_eq!(hit.tokens, 4352);
        assert_eq!(hit.junction_tokens, 10880 - 4352);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 2));
    }
}

#[test]
fn hybrid_partial_hash_truncates_full_attention_hit_length() {
    // Ported from v0.31 partial_test:1596. Values hand-derived: the
    // fine-grained probe of single:806-825 extends the two full blocks to
    // the partial entry at 4,356, and the state hit there keeps it.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm_hash(version, false, 4));
        let hash = 4;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 4356]);
        seed_states(pool, &GLM_STATES, hash, 4356);
        let request = lease(4500, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 4500);
        assert_eq!(hit.tokens, 4356);
        assert_eq!(hit.junction_tokens, 0);
        match &hit.per_group[0] {
            GroupHitBlocks::Attention(keys) => {
                assert_eq!(keys.len(), 3);
                assert_eq!(keys[2], Some(key(0, hash, 4356)));
            }
            other => panic!("{other:?}"),
        }

        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 4356]);
        seed_states(pool, &GLM_STATES, hash, 4352);
        let hit = coordinator.find_longest_hit(pool, &request, 4500);
        assert_eq!(hit.tokens, 4352);
        // coord:970-977: the full-attention hit of 4,356 minus 4,352.
        assert_eq!(hit.junction_tokens, 4);
        assert!(matches!(&hit.per_group[0], GroupHitBlocks::Attention(keys) if keys.len() == 2));
    }
}

/// Two full-attention groups of one spec and a state group, all at block 64.
fn two_full_groups_and_a_state() -> [FpmCacheGroup; 3] {
    [
        attention("full_a", 64, None),
        attention("full_b", 64, None),
        state("state", 64, true, false),
    ]
}

#[test]
fn groups_of_one_spec_hit_jointly_and_set_the_junction_together() {
    // coord:737-779 (v0.32 coord:708-750) looks up the groups of one spec
    // together, and pool:198-224 finds a block only when every group of the
    // spec holds it. The longest hit is the largest result of one spec's
    // lookup (coord:947, :970-977). Group a holds 192 tokens, group b 64 and
    // the state group the state at 64, so vLLM hits 64 with no junction.
    let groups = two_full_groups_and_a_state();
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 64, version).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 64, &[64, 128, 192]);
        seed_many(pool, 1, 64, &[64]);
        seed_many(pool, 2, 64, &[64]);
        let hit = coordinator.find_longest_hit(pool, &lease(401, 64), 401);
        assert_eq!((hit.tokens, hit.junction_tokens), (64, 0), "{version:?}");
        assert_eq!(
            hit.per_group[0],
            GroupHitBlocks::Attention(vec![Some(key(0, 64, 64))])
        );
        assert_eq!(
            hit.per_group[1],
            GroupHitBlocks::Attention(vec![Some(key(1, 64, 64))])
        );
        assert_eq!(
            hit.per_group[2],
            GroupHitBlocks::State(Some(key(2, 64, 64)))
        );
        // So the first chunk runs to the last cacheable stop 384 (v0.31
        // sched:444, v0.32 sched:476-480). A junction at 192 would end it
        // there.
        let junction = hit.tokens + hit.junction_tokens;
        assert_eq!(
            split(coordinator.rules(), 64, 401, 401, BUDGET, junction),
            320,
            "{version:?}"
        );
    }
}

#[test]
fn groups_of_one_spec_hit_inside_a_block_only_jointly() {
    // With fine-grained hits, single:794-828 probes the hash boundaries of the
    // first partial block for every group of the spec at once. Group a holds
    // two full blocks, group b one full block and the partial entry at 96.
    // No boundary inside the second block is cached in both, so vLLM hits 64.
    let groups = two_full_groups_and_a_state();
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let rules = GroupedPrefixRules::new(&groups, 16, version).unwrap();
        assert_eq!(rules.alignment, 16);
        let coordinator = with_limits(rules);
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 16, &[64, 128]);
        seed_many(pool, 1, 16, &[64, 96]);
        seed_many(pool, 2, 16, &[64, 96]);
        let hit = coordinator.find_longest_hit(pool, &lease(200, 16), 200);
        assert_eq!((hit.tokens, hit.junction_tokens), (64, 0), "{version:?}");
        assert_eq!(
            hit.per_group[0],
            GroupHitBlocks::Attention(vec![Some(key(0, 16, 64))])
        );
        assert_eq!(
            hit.per_group[1],
            GroupHitBlocks::Attention(vec![Some(key(1, 16, 64))])
        );
        assert_eq!(
            hit.per_group[2],
            GroupHitBlocks::State(Some(key(2, 16, 64)))
        );
    }
}

#[test]
fn state_groups_of_one_spec_hit_jointly() {
    // Without a full-attention group, the first spec looked up is the first
    // one in group order (coord:775-779), here the two state groups. They
    // hold the states at 64 and 192, and at 64. The joint lookup
    // (single:1540-1564) hits 64 and leaves no junction.
    let groups = [
        state("state_a", 64, true, false),
        state("state_b", 64, true, false),
        attention("window", 64, Some(65)),
    ];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let coordinator = with_limits(GroupedPrefixRules::new(&groups, 64, version).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, 64, &[64, 192]);
        seed_many(pool, 1, 64, &[64]);
        seed_many(pool, 2, 64, &[64]);
        let hit = coordinator.find_longest_hit(pool, &lease(401, 64), 401);
        assert_eq!((hit.tokens, hit.junction_tokens), (64, 0), "{version:?}");
        assert_eq!(
            hit.per_group[0],
            GroupHitBlocks::State(Some(key(0, 64, 64)))
        );
        assert_eq!(
            hit.per_group[1],
            GroupHitBlocks::State(Some(key(1, 64, 64)))
        );
    }
    // In this helper the full-attention group is G0. vLLM looks up the
    // full-attention spec first in any group order (`coord:775-779`), so
    // its hit sets the junction. The extra state of G2 at 8,704 changes
    // nothing, and the four KDA groups hit 4,352 together.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
        seed_states(pool, &GLM_STATES, hash, 4352);
        seed_states(pool, &[2], hash, 8704);
        let hit = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
        assert_eq!(
            (hit.tokens, hit.junction_tokens),
            (4352, 8704 - 4352),
            "{version:?}"
        );
        for group in GLM_STATES {
            assert_eq!(
                hit.per_group[usize::from(group)],
                GroupHitBlocks::State(Some(key(group, hash, 4352)))
            );
        }
    }
}

#[test]
fn admission_counts_inactive_hits_and_state_copies() {
    // single:216-268 counts inactive hit blocks again. The keys are seeded
    // directly, because a finished request of 4,352 tokens keeps its state
    // key only at its replay boundary, 2,176.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = groups_glm(false);
        let coordinator = with_limits(rules_glm(version, false));
        let hash = 2176;
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        seed_many(pool, 0, hash, &[2176, 4352]);
        seed_states(pool, &GLM_STATES, hash, 4352);
        let hit = coordinator.find_longest_hit(pool, &lease(4400, hash), 4400);
        assert_eq!(hit.tokens, 4352);
        // 2 inactive G0 blocks, 1 new G0 block, 1 ring, 4 x (running +
        // inactive cached state block).
        assert_eq!(coordinator.admission_blocks(pool, &hit, 4400), 12);
        // Another running request holds the two G0 blocks active.
        let other = [key(0, hash, 2176), key(0, hash, 4352)];
        let mut reservation = pool.reserve(&other, 0).unwrap();
        let held = pool.activate(&mut reservation);
        assert_eq!(held.len(), 2);
        assert_eq!(coordinator.admission_blocks(pool, &hit, 4400), 10);
        pool.cancel(reservation);
    }
}

#[test]
fn resumed_state_holds_cached_and_running_blocks_during_the_first_step() {
    // single:1858-1868: a resumed request holds the hit block and the
    // running block. vLLM frees the hit block at the request's next
    // `allocate_slots` (mgr:552-556, single:1626-1645), once
    // `cdiv(processed, block) - 1 > hit_index` (single:1667-1671):
    // `cdiv(6000, 2176) - 1 == 2 > 1`.
    let groups = groups_glm(false);
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let hash = 2176;
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[2176, 4352]);
    seed_states(pool, &GLM_STATES, hash, 4352);
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    assert_eq!(hit.tokens, 4352);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert_eq!(state.hit.map(|(_, index)| index), Some(1));
        assert!(state.running.is_none());
    }
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 4352, 6000, 1));
    let state_blocks: usize = GLM_STATES
        .iter()
        .map(|&group| {
            let state = state_lease(&block_lease, usize::from(group));
            assert!(state.hit.is_some() && state.running.is_some());
            assert!(state.previous.is_none() && state.copy.is_none());
            2
        })
        .sum();
    assert_eq!(state_blocks, 8);
    // G0: 2 hits + 1 new, ring 1, states 8.
    assert_eq!(pool.active_blocks(), 12);
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 6000, &hit, 1);
    // The hit blocks stay through the first step.
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.hit.is_some() && state.running.is_some());
    }
    assert_eq!(pool.active_blocks(), 12);
    assert_eq!(pool.inactive_cached_blocks(), 0);
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 6000, 6001, 2));
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.hit.is_none() && state.running.is_some());
    }
    assert_eq!(pool.active_blocks(), 8);
    assert_eq!(pool.inactive_cached_blocks(), 4);
}

#[test]
fn boundary_crossing_keeps_the_previous_running_block_for_one_step() {
    // single:1658-1675 and mamba_utils:576-584: the block of the earlier
    // boundary goes at the next allocation. One pass, as on the linear path,
    // where vLLM's async scheduling holds it one more.
    let groups = groups_glm(false);
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let hash = 2176;
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 0, 2176, 1));
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 2176, &hit, 1);
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 2176, 4352, 2));
    let state_and_ring = |lease: &GroupedBlockLease| -> usize {
        lease.groups[1..]
            .iter()
            .map(|g| match g {
                GroupLease::Attention { .. } => 0,
                GroupLease::Ring { copy } => usize::from(copy.is_some()),
                GroupLease::State(s) => [
                    s.running,
                    s.hit.map(|h| h.0),
                    s.previous.map(|p| p.0),
                    s.copy,
                    s.checkpoint.map(|c| c.0),
                ]
                .iter()
                .flatten()
                .count(),
            })
            .sum()
    };
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.running.is_some());
        assert_eq!(state.previous.map(|(_, step)| step), Some(2));
    }
    assert_eq!(state_and_ring(&block_lease), 9);
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 4352, &hit, 2);
    assert_eq!(state_and_ring(&block_lease), 9);
    coordinator.begin_step();
    let before = pool.active_blocks();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 4352, 6528, 3));
    // Four previous blocks go, four new running blocks and one G0 block come.
    assert_eq!(pool.active_blocks(), before + 1);
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert_eq!(state.previous.map(|(_, step)| step), Some(3));
    }
    assert_eq!(state_and_ring(&block_lease), 9);
}

#[test]
fn partial_hit_takes_a_copy_block_for_the_shared_block() {
    // single:387-409 and single:1770-1777: a hit that ends inside a block
    // takes one copy block. single:800-805: the full-attention probe needs
    // the full blocks below the partial entry, so blocks 0 and 1 are
    // cached too.
    let groups = [attention("full", 16, None), state("s", 32, true, false)];
    let r = rules(&groups, 4);
    assert_eq!(r.alignment, 4);
    let mut coordinator = with_limits(r);
    let hash = 4;
    let mut kv = kv_pool(&groups, 32);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[16, 32, 36]);
    seed_many(pool, 1, hash, &[36]);
    let request = lease(50, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 50);
    assert_eq!(hit.tokens, 36);
    // G0: 4 live - 3 hits + 3 inactive + 1 copy = 5. State
    // (single:1757-1777): `cdiv(50, 32) - cdiv(36, 32) == 0` new blocks
    // plus the copy block, plus the inactive hit = 2.
    assert_eq!(coordinator.admission_blocks(pool, &hit, 50), 7);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert_eq!(pool.active_blocks(), 4);
    assert!(coordinator.allocate(pool, &mut block_lease, 50, 36, 50, 1));
    // G0 holds blocks 0 to 3, with block 2 as a private copy. The target
    // 50 stays in the hit's block, so the state's copy-on-write block is
    // its running block (single:1891-1895, `_apply_cow` at
    // single:454-474). The shared block 2 and the state's hit block are
    // the copy sources. They leave the lease and stay active until the
    // scheduling pass ends (sched:1434-1442).
    assert_eq!(attention_copies(&block_lease, 0), 4);
    let state = state_lease(&block_lease, 1);
    assert!(state.hit.is_none() && state.running.is_some() && state.copy.is_none());
    assert_eq!(coordinator.copy_source_count(), 2);
    assert_eq!(pool.active_blocks(), 4 + 1 + 2);
    assert_eq!(pool.inactive_cached_blocks(), 0);
    coordinator.finalize(pool, &mut block_lease, &request, 50, 50, &hit, 1);
    let state = state_lease(&block_lease, 1);
    assert!(state.hit.is_none() && state.running.is_some());
    assert_eq!(coordinator.copy_source_count(), 2);
    assert_eq!(pool.active_blocks(), 4 + 1 + 2);
    assert_eq!(pool.inactive_cached_blocks(), 0);
    assert_eq!(coordinator.release_copy_sources(pool), 2);
    assert_eq!(pool.active_blocks(), 4 + 1);
    assert_eq!(pool.inactive_cached_blocks(), 2);

    // A target beyond the hit's block: the copy sits at the hit's index
    // and a running block follows. `remove_skipped_blocks` frees the
    // copy at the next allocation (single:1658-1675, index set at :1868).
    let mut kv = kv_pool(&groups, 32);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[16, 32, 36]);
    seed_many(pool, 1, hash, &[36]);
    let request = lease(70, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 70);
    assert_eq!(hit.tokens, 36);
    // G0: 5 live - 3 hits + 3 inactive + 1 copy = 6. State: `cdiv(70, 32)
    // - cdiv(36, 32) == 1` new block plus the copy block plus the
    // inactive hit = 3.
    assert_eq!(coordinator.admission_blocks(pool, &hit, 70), 9);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 70, 36, 70, 1));
    let state = state_lease(&block_lease, 1);
    assert!(state.hit.is_none() && state.running.is_some() && state.copy.is_some());
    assert_eq!(coordinator.copy_source_count(), 2);
    assert_eq!(pool.active_blocks(), 5 + 2 + 2);
    coordinator.finalize(pool, &mut block_lease, &request, 70, 70, &hit, 1);
    let state = state_lease(&block_lease, 1);
    assert!(state.hit.is_none() && state.copy.is_some());
    assert_eq!(pool.active_blocks(), 5 + 2 + 2);
    assert_eq!(coordinator.release_copy_sources(pool), 2);
    assert_eq!(pool.active_blocks(), 5 + 2);
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 70, 70, 71, 2));
    let state = state_lease(&block_lease, 1);
    assert!(state.copy.is_none() && state.running.is_some());
    assert_eq!(pool.active_blocks(), 5 + 1);
}

#[test]
fn admission_counts_vllms_whole_prompt_blocks_without_the_producer_copy() {
    // vLLM's whole-prompt test (v0.31 mgr:519-536, v0.32 mgr:522-539,
    // v0.33 mgr:531-548) counts an attention group as `cdiv(prompt, A)`
    // less its hit blocks, plus the hit blocks that no request holds and a
    // copy block for a hit inside a block (v0.31 single:216-268, v0.32
    // single:213-265, v0.33 single:214-266). It counts a state group as
    // `cdiv(prompt, S) - len(computed)`, plus one for a hit inside a block
    // that the request consumes, capped at `1 + partial hit + checkpoint`
    // (v0.31 single:1762-1777, :1806-1811; v0.32 single:1724-1739,
    // :1768-1773; v0.33 single:1740-1755, :1785-1790). A new request is
    // not yet the producer of a tail, so the copy that its own published
    // tail takes later is not counted.
    // (attention block, hash, prompt, checkpoint, vLLM's count)
    let cold = [
        // 1 + 1 for every prompt inside one block of 64.
        (64, 4, 4, false, 2),
        (64, 4, 48, false, 2),
        (64, 4, 50, false, 2),
        (64, 4, 64, false, 2),
        (64, 64, 50, false, 2),
        (64, 4, 48, true, 2),
        (64, 4, 50, true, 2),
        (64, 4, 64, true, 2),
        // cdiv(prompt, 4) + 1.
        (4, 4, 4, false, 1 + 1),
        (4, 4, 48, false, 12 + 1),
        (4, 4, 50, false, 13 + 1),
        (4, 4, 64, false, 16 + 1),
    ];
    for (attention_block, hash, prompt, checkpoint, count) in cold {
        for version in [
            VllmRuleVersion::V0_31_0,
            VllmRuleVersion::V0_32_0,
            VllmRuleVersion::V0_33_0,
        ] {
            if checkpoint && version == VllmRuleVersion::V0_31_0 {
                continue;
            }
            let groups = [
                attention("full", attention_block, None),
                state("s", 64, true, checkpoint),
            ];
            let coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
            let kv = kv_pool(&groups, 32);
            let pool = kv.block_pool().unwrap();
            let hit = coordinator.find_longest_hit(pool, &lease(prompt, hash), prompt);
            assert_eq!(hit.tokens, 0);
            assert_eq!(
                coordinator.admission_blocks(pool, &hit, prompt),
                count,
                "{version:?}, attention {attention_block}, hash {hash}, prompt {prompt}, checkpoint={checkpoint}"
            );
        }
    }
    // A hit at 32 or at the tail 48 inside the blocks of 64: one copy
    // block per group, and the two hit blocks when no request holds
    // them: 4, else 2.
    let groups = [attention("full", 64, None), state("s", 64, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        for hit_tokens in [32, 48] {
            for held in [false, true] {
                let coordinator =
                    with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
                let mut kv = kv_pool(&groups, 8);
                let pool = kv.block_pool_mut().unwrap();
                let keys = [key(0, 4, hit_tokens), key(1, 4, hit_tokens)];
                for key in keys {
                    seed(pool, key);
                }
                let other = if held {
                    let mut reservation = pool.reserve(&keys, 0).unwrap();
                    let copies = pool.activate(&mut reservation);
                    pool.cancel(reservation);
                    copies
                } else {
                    Vec::new()
                };
                let hit = coordinator.find_longest_hit(pool, &lease(50, 4), 50);
                assert_eq!(hit.tokens, hit_tokens);
                assert_eq!(
                    coordinator.admission_blocks(pool, &hit, 50),
                    if held { 2 } else { 4 },
                    "{version:?}, hit {hit_tokens}, held={held}"
                );
                pool.release_all(other);
            }
        }
    }
}

#[test]
fn producer_copy_needs_a_block_that_admission_does_not_count() {
    // Blocks of 64, a hash block of 4, a cold prompt of 50 tokens. The
    // split stops at the tail 48, and the write 48..50 takes the copy.
    // vLLM's admission counts 2 blocks; the allocation of the copy then
    // needs a third (v0.31 single:1891-1936, v0.32 single:1855-1900,
    // v0.33 single:1872-1917). When it does not fit, vLLM preempts the
    // last running request, the request itself included, and tries again
    // (v0.31 sched:745-813, v0.32 sched:776-844, v0.33 sched:912-980).
    // vLLM counts the pending copy in the blocks that a prefill in flight
    // still needs (v0.31 sched:3003-3017, v0.32 sched:3060-3076, v0.33
    // sched:3404-3420, with single:1770-1777, v0.32 :1732-1739 and v0.33
    // :1748-1755); that count is not part of this coordinator.
    let groups = [attention("full", 64, None), state("s", 64, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        for pool_blocks in [2, 3] {
            let mut coordinator =
                with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
            let mut kv = kv_pool(&groups, pool_blocks);
            let pool = kv.block_pool_mut().unwrap();
            let request = lease(50, 4);
            let hit = coordinator.find_longest_hit(pool, &request, 50);
            assert_eq!(coordinator.admission_blocks(pool, &hit, 50), 2);
            let mut block_lease = coordinator.new_lease();
            coordinator.begin_step();
            let chunk = coordinator
                .rules()
                .align_split(0, 50, 50, BUDGET, 0, BUDGET);
            assert_eq!(chunk, 48, "{version:?}");
            assert!(coordinator.allocate(pool, &mut block_lease, 50, 0, 48, 1));
            coordinator.finalize(pool, &mut block_lease, &request, 50, 48, &hit, 1);
            assert!(hit_exists(pool, 1, 4, 48));
            coordinator.begin_step();
            let fits = coordinator.allocate(pool, &mut block_lease, 50, 48, 50, 2);
            assert_eq!(fits, pool_blocks == 3, "{version:?}, pool {pool_blocks}");
            if fits {
                assert_eq!(pool.active_blocks(), 3);
            } else {
                assert_eq!(pool.active_blocks(), 2);
            }
            coordinator.release_request(pool, &mut block_lease);
        }
    }
}

#[test]
fn producer_partial_state_keeps_separate_capacity_until_eviction() {
    // Producer-tail CoW: v0.31 single:1903-1936, v0.32 single:1867-1900
    // and v0.33 single:1884-1917 preserve the cached boundary before
    // another write.
    let groups = [attention("full", 32, None), state("s", 32, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 5);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(50, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        let mut block_lease = coordinator.new_lease();
        let mut before = 0;
        for (step, target) in [(1, 32), (2, 48), (3, 50)] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 50, before, target, step));
            coordinator.finalize(pool, &mut block_lease, &request, 50, target, &hit, step);
            before = target;
        }
        // Two attention blocks, the retained state at 48, and the
        // writable state at 50. The state at 32 is already inactive.
        assert_eq!(pool.active_blocks(), 4, "{version:?}");
        let state = state_lease(&block_lease, 1);
        assert!(pool.is_private(state.running.unwrap()));
        assert!(state.previous.is_some());
        assert_eq!(pool.inactive_cached_blocks(), 1);
        coordinator.release_request(pool, &mut block_lease);
        assert_eq!(pool.active_blocks(), 0);
        assert_eq!(pool.inactive_cached_blocks(), 4);
        assert!(hit_exists(pool, 1, 4, 48));
        // One free block, then eviction of state 32 and both attention blocks;
        // the preserved partial state is still independently resident.
        let pressure = pool.reserve(&[], 4).unwrap();
        assert!(hit_exists(pool, 1, 4, 48));
        let last = pool.reserve(&[], 1).unwrap();
        assert!(!hit_exists(pool, 1, 4, 48));
        pool.cancel(last);
        pool.cancel(pressure);
    }
}

#[test]
fn producer_copy_is_taken_only_with_hits_inside_a_block() {
    // With hits inside a block on, a chunk that ends at the prompt's
    // partial tail inside a state block publishes the running block, and
    // the next allocation takes a fresh block and keeps the published
    // copy (v0.31 single:2094-2147, :1891-1936; v0.32 single:2067-2112,
    // :1855-1900; v0.33 single:2085-2135, :1872-1917). With checkpoint
    // blocks, a final chunk without a valid checkpoint follows the same
    // rule (v0.32 single:2019-2066, v0.33 single:2036-2084). The
    // chunk 48..50 cannot export the checkpoint at 48.
    for (version, checkpoint) in [
        (VllmRuleVersion::V0_31_0, false),
        (VllmRuleVersion::V0_32_0, false),
        (VllmRuleVersion::V0_32_0, true),
        (VllmRuleVersion::V0_33_0, false),
        (VllmRuleVersion::V0_33_0, true),
    ] {
        let groups = [
            attention("full", 32, None),
            state("s", 32, true, checkpoint),
        ];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        assert_eq!(coordinator.rules().alignment, 4);
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(50, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        let mut block_lease = coordinator.new_lease();
        let mut before = 0;
        for (step, target) in [(1, 32), (2, 48), (3, 50)] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 50, before, target, step));
            coordinator.finalize(pool, &mut block_lease, &request, 50, target, &hit, step);
            before = target;
        }
        let case = format!("{version:?}, checkpoint={checkpoint}");
        assert!(hit_exists(pool, 1, 4, 48), "{case}");
        assert_eq!(pool.active_blocks(), 2 + 2, "{case}");
        assert!(state_lease(&block_lease, 1).previous.is_some(), "{case}");
        coordinator.release_request(pool, &mut block_lease);
    }
    // With them off, vLLM rounds the cached length down to the scheduler
    // block before the state manager sees it (v0.31 coord:796-806,
    // :808-840; v0.32 coord:767-794, :354-377; v0.33 coord:777-804,
    // :358-381). The rounded length is a multiple of the state block, so
    // vLLM registers no tail and takes no copy (v0.31 single:2096-2097;
    // v0.32 single:2101; v0.33 single:2124). A window group of
    // block 8 has no sub-block lookup, so the alignment is 8 at a hash
    // block of 4. A budget of 4 makes a chunk end at the tail 12: the
    // prompt end on v0.31, one token before it on v0.32 and v0.33.
    for (version, prompt) in [
        (VllmRuleVersion::V0_31_0, 12),
        (VllmRuleVersion::V0_32_0, 13),
        (VllmRuleVersion::V0_33_0, 13),
    ] {
        let groups = [attention("window", 8, Some(8)), state("s", 8, true, false)];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        assert_eq!(coordinator.rules().alignment, 8);
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(prompt + 1, 4);
        let hit = coordinator.find_longest_hit(pool, &request, prompt);
        let mut block_lease = coordinator.new_lease();
        let mut before = 0;
        for (step, target) in [(1, 4), (2, 8), (3, 12), (4, 13)] {
            if before < prompt {
                let chunk = coordinator
                    .rules()
                    .align_split(before, prompt, prompt, 4, 0, 4);
                assert_eq!(before + chunk, target, "{version:?}");
            }
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, prompt, before, target, step));
            if target < 13 {
                coordinator.finalize(pool, &mut block_lease, &request, prompt, target, &hit, step);
            }
            before = target;
        }
        assert!(hit_exists(pool, 1, 4, 8), "{version:?}");
        assert!(!hit_exists(pool, 1, 4, 12), "{version:?}");
        // The write 12..13, the last prompt token on v0.32 and v0.33 or the
        // first decode token on v0.31, stays on the running state block: two
        // window blocks and one state block.
        assert_eq!(pool.active_blocks(), 2 + 1, "{version:?}");
        assert!(state_lease(&block_lease, 1).previous.is_none());
        coordinator.release_request(pool, &mut block_lease);
    }
}

#[test]
fn producer_copy_defers_hits_on_its_state_in_the_step_of_the_copy() {
    // vLLM moves the cache entry of the published state to the copy block
    // and adds its key to the keys of the step, so a waiting request hits
    // it only from the next step on (v0.31 single:1931-1934, v0.32
    // single:1895-1898, v0.33 single:1912-1915). Replay keeps the key on
    // the published block and hides it for the rest of the step of the
    // copy.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = [attention("full", 32, None), state("s", 32, true, false)];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(50, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        let mut block_lease = coordinator.new_lease();
        let mut before = 0;
        for (step, target) in [(1, 32), (2, 48)] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 50, before, target, step));
            coordinator.finalize(pool, &mut block_lease, &request, 50, target, &hit, step);
            before = target;
        }
        coordinator.begin_step();
        let sibling = lease(60, 4);
        assert_eq!(
            coordinator.find_longest_hit(pool, &sibling, 60).tokens,
            48,
            "{version:?}"
        );
        assert!(coordinator.allocate(pool, &mut block_lease, 50, 48, 50, 3));
        assert!(
            state_lease(&block_lease, 1).previous.is_some(),
            "{version:?}"
        );
        assert!(
            coordinator.find_longest_hit(pool, &sibling, 60).tokens < 48,
            "{version:?}"
        );
        coordinator.begin_step();
        assert_eq!(
            coordinator.find_longest_hit(pool, &sibling, 60).tokens,
            48,
            "{version:?}"
        );
        coordinator.release_request(pool, &mut block_lease);
    }
}

/// For each prompt in order on one pool: the hit, the junction, the
/// admission count, then each chunk end with the active blocks after its
/// allocation, the active blocks after one decode token, and the
/// inactive cached blocks after the release.
fn glm_trace(version: VllmRuleVersion, checkpoint: bool) -> Vec<Vec<usize>> {
    let mut coordinator = with_limits(rules_glm(version, checkpoint));
    let groups = groups_glm(checkpoint);
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let block = GLM_BLOCK as usize;
    let mut step = 1;
    let mut trace = Vec::new();
    // Cold, warm on the first prompt's prefix, and a block boundary.
    for prompt in [9000, 12_000, 4 * block] {
        let request = lease(prompt, block);
        let hit = coordinator.find_longest_hit(pool, &request, prompt);
        let junction = hit.tokens + hit.junction_tokens;
        let mut row = vec![
            hit.tokens,
            hit.junction_tokens,
            coordinator.admission_blocks(pool, &hit, prompt),
        ];
        let mut block_lease = coordinator.new_lease();
        coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
        let mut computed = hit.tokens;
        while computed <= prompt {
            coordinator.begin_step();
            let target = if computed < prompt {
                computed
                    + coordinator
                        .rules()
                        .align_split(computed, prompt, prompt, BUDGET, junction, BUDGET)
            } else {
                prompt + 1
            };
            assert!(coordinator.allocate(pool, &mut block_lease, prompt, computed, target, step));
            row.extend([target, pool.active_blocks()]);
            if target <= prompt {
                coordinator.finalize(pool, &mut block_lease, &request, prompt, target, &hit, step);
            }
            computed = target;
            step += 1;
        }
        coordinator.release_request(pool, &mut block_lease);
        row.push(pool.inactive_cached_blocks());
        trace.push(row);
    }
    trace
}

#[test]
fn glm_groups_at_the_hash_block_keep_their_counts() {
    // The six GLM-5.3-Flash groups with the hash block equal to the
    // block. A state is published only at a block boundary, where the
    // next write already takes a new block, so the producer copy never
    // adds one, and no group has a window or a tail inside a block. The
    // rows give the hit, the junction, the admission count, each chunk
    // end with the active blocks, one decode token, and the inactive
    // cached blocks after the release, for a cold prompt, a warm prompt
    // and a prompt that ends on a block boundary.
    let no_checkpoint = vec![
        vec![0, 0, 10, 6528, 8, 8704, 13, 9000, 14, 9001, 10, 8],
        vec![8704, 0, 15, 10880, 14, 12000, 15, 12001, 11, 13],
        vec![0, 6528, 9, 6528, 8, 8704, 13, 8705, 14, 21],
    ];
    let checkpoint = vec![
        vec![0, 0, 14, 6528, 8, 9000, 18, 9001, 10, 8],
        vec![8704, 0, 19, 12000, 19, 12001, 11, 13],
        vec![0, 6528, 13, 6528, 8, 8704, 13, 8705, 14, 21],
    ];
    let expected: [(VllmRuleVersion, bool, Vec<Vec<usize>>); 5] = [
        (VllmRuleVersion::V0_31_0, false, no_checkpoint.clone()),
        (VllmRuleVersion::V0_32_0, false, no_checkpoint.clone()),
        (VllmRuleVersion::V0_32_0, true, checkpoint.clone()),
        (VllmRuleVersion::V0_33_0, false, no_checkpoint),
        (VllmRuleVersion::V0_33_0, true, checkpoint),
    ];
    for (version, checkpoint, rows) in expected {
        assert_eq!(
            glm_trace(version, checkpoint),
            rows,
            "{version:?}, checkpoint={checkpoint}"
        );
    }
}

#[test]
fn finalize_publishes_states_at_the_replay_boundary_and_the_junction() {
    // single:1614-1622 with retention 0 (cache:160-166): the replay
    // boundary `(9000 - 1) // 2176 * 2176 == 8704` and the junction.
    let groups = groups_glm(false);
    let hash = 2176;
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    let ends = prefill(
        &mut coordinator,
        pool,
        &mut block_lease,
        &request,
        9000,
        &hit,
        2176,
    );
    assert_eq!(ends, vec![2176, 4352, 6528, 8704, 9000]);
    coordinator.release_request(pool, &mut block_lease);
    for group in GLM_STATES {
        for end in [2176, 4352, 6528] {
            assert!(
                !hit_exists(pool, group, hash, end),
                "group {group} at {end}"
            );
        }
        assert!(hit_exists(pool, group, hash, 8704));
    }
    // G0 blocks 0 to 3 and four state blocks stay cached.
    assert_eq!(pool.inactive_cached_blocks(), 8);
    assert_eq!(pool.active_blocks(), 0);

    // A hit of 4,352 with a junction of 2,176.
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[2176, 4352, 6528]);
    seed_states(pool, &GLM_STATES, hash, 4352);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    assert_eq!((hit.tokens, hit.junction_tokens), (4352, 2176));
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    let ends = prefill(
        &mut coordinator,
        pool,
        &mut block_lease,
        &request,
        9000,
        &hit,
        2176,
    );
    assert_eq!(ends, vec![6528, 8704, 9000]);
    coordinator.release_request(pool, &mut block_lease);
    for group in GLM_STATES {
        assert!(hit_exists(pool, group, hash, 6528), "group {group}");
        assert!(hit_exists(pool, group, hash, 8704), "group {group}");
        assert!(!hit_exists(pool, group, hash, 2176), "group {group}");
    }
}

#[test]
fn align_split_follows_vllm_0310() {
    // sched:414-527 at v0.31.
    let r = rules_glm(VllmRuleVersion::V0_31_0, false);
    // The floor of 8,192 to 2,176 is 6,528 (sched:460-468).
    assert_eq!(split(&r, 0, 9000, 9000, 8192, 0), 6528);
    // The last cacheable stop `9000 - 9000 % 2176 == 8704` (sched:444).
    assert_eq!(split(&r, 6528, 9000, 9000, 8192, 0), 2176);
    assert_eq!(split(&r, 8704, 9000, 9000, 8192, 0), 296);
    // A prompt of exactly 8,704 tokens has no stop at 6,528, because its
    // last cacheable stop `8704 - 8704 % 2176` is the prompt itself
    // (sched:444).
    assert_eq!(split(&r, 6528, 8704, 8704, 8192, 0), 2176);
    // With hash 4 the tail 9,000 equals the prompt and adds no stop
    // (sched:482-487, :512-514). With prompt 9,002 the tail is 9,000.
    let fine = rules_glm_hash(VllmRuleVersion::V0_31_0, false, 4);
    assert_eq!(split(&fine, 8704, 9000, 9000, 8192, 0), 296);
    assert_eq!(split(&fine, 8704, 9002, 9002, 8192, 0), 296);
    // The junction stop (sched:488-519).
    assert_eq!(split(&r, 4352, 9000, 9000, 8192, 6528), 2176);
    // Past the prefill the split returns the request (sched:434-436).
    assert_eq!(split(&r, 9000, 9000, 9001, 1, 0), 1);
}

#[test]
fn align_split_follows_vllm_0320() {
    // v0.32 sched:443-556.
    let r = rules_glm(VllmRuleVersion::V0_32_0, false);
    assert_eq!(split(&r, 0, 9000, 9000, 8192, 0), 6528);
    assert_eq!(split(&r, 6528, 9000, 9000, 2472, 0), 2176);
    // The last stop of 8,704 tokens is `(8704 - 1) // 2176 * 2176 == 6528`
    // (v0.32 sched:476-480).
    assert_eq!(split(&r, 4352, 8704, 8704, 4352, 0), 2176);
    let v31 = rules_glm(VllmRuleVersion::V0_31_0, false);
    assert_eq!(split(&v31, 4352, 8704, 8704, 4352, 0), 4352);
    // Checkpoint mode (v0.32 sched:482-501, iface:1125-1146). For
    // (6528, 9000, pos 8704): 6528 % 2176 == 0; cdiv(9000, 2176) - 2 == 3
    // > 6527 / 2176 == 2; 6528 + 2176 <= 8704; 6528 < 8704 < 9000;
    // 2176 % 16 == 0.
    let ckpt = rules_glm(VllmRuleVersion::V0_32_0, true);
    assert_eq!(ckpt.checkpoint_position(9000), 8704);
    assert!(ckpt.checkpoint_valid(6528, 9000, 8704));
    assert_eq!(split(&ckpt, 6528, 9000, 9000, 2472, 0), 2472);
    // For (4352, 8704, pos 6528): 4352 % 2176 == 0; cdiv(8704, 2176) - 2
    // == 2 > 4351 / 2176 == 1; 4352 + 2176 <= 6528; 4352 < 6528 < 8704;
    // 2176 % 16 == 0.
    assert_eq!(ckpt.checkpoint_position(8704), 6528);
    assert!(ckpt.checkpoint_valid(4352, 8704, 6528));
    assert_eq!(split(&ckpt, 4352, 8704, 8704, 4352, 0), 4352);
    // Without a checkpoint rule the check is false.
    assert!(!r.checkpoint_valid(6528, 9000, 8704));
}

#[test]
fn align_split_with_a_checkpoint_block_ports_vllm_0320() {
    // Ported from v0.32 split_test:127-180, which stubs `block_size` to
    // 1,600 and `hash_block_size` to 16 (split_test:139-153). The engine
    // shape with the same numbers: one full-attention rule and one state
    // rule at block 1,600, hash 16, checkpoint 1 at alignment 16. The
    // state block is above the hash block and the other rule is full
    // attention, so fine-grained hits are on and A is 16.
    let groups = [attention("full", 1600, None), state("s", 1600, true, true)];
    let r = GroupedPrefixRules::new(&groups, 16, VllmRuleVersion::V0_32_0).unwrap();
    assert_eq!(r.alignment, 16);
    assert_eq!(r.cacheable_block, 1600);
    // The engine's checkpoint position is `(2002 - 1) / 16 * 16 == 2000`.
    // The vLLM test stubs partial hits off and gets 1,600. Both positions
    // pass the five conditions (0 % 16 == 0; cdiv(2002, 1600) - 2 == 0 >
    // -1; 0 + 16 <= pos; 0 < pos < 2002; pos % 16 == 0), so both give
    // 2,002.
    assert_eq!(r.checkpoint_position(2002), 2000);
    assert!(r.checkpoint_valid(0, 2002, 2000));
    assert!(r.checkpoint_valid(0, 2002, 1600));
    assert_eq!(r.align_split(0, 2002, 2002, 2002, 0, 16384), 2002);
    // The end 2,000 is below the prompt, so the floor to 1,600 applies
    // and no checkpoint is taken.
    assert_eq!(r.align_split(0, 3602, 3602, 2000, 0, 16384), 1600);
    let groups = [attention("full", 1600, None), state("s", 1600, true, false)];
    let v31 = GroupedPrefixRules::new(&groups, 16, VllmRuleVersion::V0_31_0).unwrap();
    assert_eq!(v31.align_split(0, 2002, 2002, 2002, 0, 16384), 1600);
    assert_eq!(v31.align_split(0, 3602, 3602, 2000, 0, 16384), 1600);
}

#[test]
fn align_split_needs_a_mamba_state_group() {
    // vLLM splits only with Mamba layers in align mode (v0.31 `sched:354-356`,
    // `:695-697`, v0.32 `sched:377-379`, `:724-726`). Without a state rule
    // every chunk is the desired count, on every row.
    let full = [attention("full", 64, None)];
    let windowed = [attention("full", 64, None), attention("swa", 64, Some(128))];
    // A tail ring, such as the GLM `KpoolTailSpec`, is not a MambaSpec.
    let ring = [attention("full", 64, None), state("ring", 4, false, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        for groups in [&full[..], &windowed[..], &ring[..]] {
            let r = GroupedPrefixRules::new(groups, 64, version).unwrap();
            // The last block boundary of the prompt is no stop.
            assert_eq!(split(&r, 0, 300, 300, 300, 0), 300, "{version:?}");
            // A chunk that ends below the prompt keeps its length.
            assert_eq!(split(&r, 0, 1000, 1000, 500, 0), 500, "{version:?}");
            // A start inside a block runs past the next boundary.
            assert_eq!(split(&r, 100, 1000, 1000, 500, 0), 500, "{version:?}");
            // The junction is no stop.
            assert_eq!(split(&r, 64, 1000, 1000, 800, 640), 800, "{version:?}");
        }
        // A state group keeps the split: the floor of 8,192 to 2,176.
        assert_eq!(
            split(&rules_glm(version, false), 0, 9000, 9000, 8192, 0),
            6528,
            "{version:?}"
        );
    }
}

#[test]
fn partial_tail_moves_one_unit_down_on_vllm_0320() {
    // At hash 4 the 0.32.0 tail, `(prompt - 1) / hash * hash`, is 8,996,
    // between the last stop 8,704 and the prompt (v0.32 sched:517-525,
    // single:860-868, single:2069-2073). The 0.31.0 tail is 9,000.
    let groups = groups_glm(false);
    let hash = 4;
    for (version, expected_ends, expected_hit) in [
        (VllmRuleVersion::V0_32_0, vec![6528, 8704, 8996, 9000], 8996),
        (VllmRuleVersion::V0_33_0, vec![6528, 8704, 8996, 9000], 8996),
        (VllmRuleVersion::V0_31_0, vec![6528, 8704, 9000], 8704),
    ] {
        let mut coordinator = with_limits(rules_glm_hash(version, false, hash));
        let mut kv = kv_pool(&groups, 64);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(9000, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 9000);
        let mut block_lease = coordinator.new_lease();
        let ends = prefill(
            &mut coordinator,
            pool,
            &mut block_lease,
            &request,
            9000,
            &hit,
            BUDGET,
        );
        assert_eq!(ends, expected_ends, "{version:?}");
        if version != VllmRuleVersion::V0_31_0 {
            for group in GLM_STATES {
                assert!(hit_exists(pool, group, hash, 8704));
                assert!(hit_exists(pool, group, hash, 8996));
            }
        }
        let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
        assert_eq!(resend.tokens, expected_hit, "{version:?}");
        coordinator.release_request(pool, &mut block_lease);
        let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
        assert_eq!(resend.tokens, expected_hit, "{version:?}");
    }
}

#[test]
fn checkpoint_block_is_taken_in_the_final_chunk_and_published_at_the_prompt_checkpoint() {
    // v0.32 single:1643-1674, :1745-1773 (one more block in the chunk
    // that reaches the prompt), :2017-2066 (published at the prompt
    // checkpoint under retention 0).
    let groups = groups_glm(true);
    let hash = 2176;
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_32_0, true));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    assert_eq!(hit.tokens, 0);
    // 5 for G0, 1 for the ring, 2 for each KDA group.
    assert_eq!(coordinator.admission_blocks(pool, &hit, 9000), 14);
    let v31 = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    assert_eq!(v31.admission_blocks(pool, &hit, 9000), 10);

    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert_eq!(split(coordinator.rules(), 0, 9000, 9000, BUDGET, 0), 6528);
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 0, 6528, 1));
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.running.is_some());
        assert!(state.previous.is_none() && state.checkpoint.is_none() && state.hit.is_none());
    }
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 6528, &hit, 1);
    coordinator.begin_step();
    assert_eq!(
        split(coordinator.rules(), 6528, 9000, 9000, BUDGET, 0),
        2472
    );
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 6528, 9000, 2));
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.running.is_some() && state.previous.is_some());
        assert_eq!(state.checkpoint.map(|(_, position)| position), Some(8704));
    }
    // G0 5, ring 1, 4 x (running + previous + checkpoint).
    assert_eq!(pool.active_blocks(), 5 + 1 + 12);
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 9000, &hit, 2);
    for group in GLM_STATES {
        assert!(hit_exists(pool, group, hash, 8704), "group {group}");
        let state = state_lease(&block_lease, usize::from(group));
        let (checkpoint, _) = state.checkpoint.unwrap();
        assert!(!pool.is_private(checkpoint));
        assert!(pool.is_private(state.running.unwrap()));
    }
    coordinator.begin_step();
    let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
    assert_eq!(resend.tokens, 8704);
    coordinator.release_request(pool, &mut block_lease);
    // Four G0 blocks and the four checkpoint copies stay in the LRU.
    assert_eq!(pool.inactive_cached_blocks(), 8);
    assert_eq!(pool.active_blocks(), 0);
    let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
    assert_eq!(resend.tokens, 8704);
}

#[test]
fn checkpoint_block_goes_at_the_next_step_and_stays_cached_until_eviction() {
    // A Mamba manager skips every token before the last computed one (v0.32
    // single:1954-1959). So the request's next allocation first frees every
    // block below its last block, the checkpoint block included (v0.32
    // kv_cache_manager.py:546-559, single:672-710, :1592-1641). A freed
    // block keeps its key in the LRU (v0.32 block_pool.py:774-803).
    let groups = groups_glm(true);
    let hash = 2176;
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_32_0, true));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    let ends = prefill(
        &mut coordinator,
        pool,
        &mut block_lease,
        &request,
        9000,
        &hit,
        BUDGET,
    );
    assert_eq!(ends, [6528, 9000]);
    // G0 5, ring 1, 4 x (running + previous + checkpoint).
    assert_eq!(pool.active_blocks(), 5 + 1 + 12);
    // The next allocation releases blocks, so no step is quiet yet.
    assert_eq!(block_lease.quiet_until, 0);
    // The prefill ran steps 1 and 2. Each decode step holds G0 5, the ring
    // and the four running blocks only, and a resend still hits the
    // checkpoint keys.
    for (step, target) in [(3, 9001), (4, 9002)] {
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 9000, target - 1, target, step));
        assert_eq!(pool.active_blocks(), 5 + 1 + 4, "step {step}");
        for group in GLM_STATES {
            let state = state_lease(&block_lease, usize::from(group));
            assert!(state.checkpoint.is_none() && state.previous.is_none());
            let cached = pool.key_hit(group, hash_at(hash, 8704));
            assert!(cached.is_some_and(|hit| !hit.is_active), "group {group}");
        }
        coordinator.finalize(pool, &mut block_lease, &request, 9000, target, &hit, step);
        // Quiet up to the end of the fifth G0 block, which the request holds.
        assert_eq!(block_lease.quiet_until, 10880, "step {step}");
        let resend = coordinator.find_longest_hit(pool, &lease(9000, hash), 9000);
        assert_eq!(resend.tokens, 8704, "step {step}");
    }
    // The checkpoint keys go only when a new block evicts them, oldest first
    // in the order the groups freed them (v0.32 coord:447-450).
    assert_eq!(pool.inactive_cached_blocks(), 4);
    let free = pool.capacity_blocks() - pool.active_blocks() - 4;
    let mut reservation = pool.reserve(&[], free + 1).unwrap();
    for _ in 0..=free {
        pool.take_private(&mut reservation);
    }
    pool.cancel(reservation);
    assert!(!hit_exists(pool, 2, hash, 8704));
    for group in [3, 4, 5] {
        assert!(hit_exists(pool, group, hash, 8704), "group {group}");
    }
}

#[test]
fn checkpoint_block_publishes_the_prompt_checkpoint_under_retention_zero() {
    // Ported from v0.32 partial_test:775-807: the prompt-end checkpoint
    // `(prompt - 1) / 32 * 32` stays published under retention 0.
    let groups = [attention("full", 32, None), state("s", 32, true, true)];
    let hash = 32;
    for (prompt, expected) in [(120, 96), (104, 96), (96, 64)] {
        let mut coordinator =
            with_limits(GroupedPrefixRules::new(&groups, hash, VllmRuleVersion::V0_32_0).unwrap());
        let mut kv = kv_pool(&groups, 16);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(prompt, hash);
        let hit = coordinator.find_longest_hit(pool, &request, prompt);
        let mut block_lease = coordinator.new_lease();
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, prompt, 0, prompt, 1));
        assert_eq!(
            state_lease(&block_lease, 1).checkpoint.map(|(_, p)| p),
            Some(expected)
        );
        coordinator.finalize(pool, &mut block_lease, &request, prompt, prompt, &hit, 1);
        assert!(hit_exists(pool, 1, hash, expected), "prompt {prompt}");
        coordinator.release_request(pool, &mut block_lease);
        coordinator.begin_step();
        let resend = coordinator.find_longest_hit(pool, &lease(prompt, hash), prompt);
        assert_eq!(resend.tokens, expected, "prompt {prompt}");
    }
}

#[test]
fn checkpoint_beyond_the_prompt_checkpoint_is_not_published_under_retention_zero() {
    // A request with a prompt of 32 tokens resumes with 40 known tokens. Its
    // prefill ends at `max(32, 40 - 1) = 39` (v0.32 mgr:518-520). From a hit
    // at 16, the chunk 16..40 reaches it and reserves the checkpoint at 32
    // (v0.32 single:1643-1674, :1745-1773). Retention 0 keeps a checkpoint
    // only at the prompt checkpoint `(32 - 1) / 16 * 16 = 16` or at the
    // shared-prefix junction, and drops this one (v0.32 single:2025-2056,
    // cache.py:159-165). So a later prompt hits 16, not 32.
    let groups = [attention("full", 16, None), state("s", 16, true, true)];
    let hash = 16;
    let mut coordinator =
        with_limits(GroupedPrefixRules::new(&groups, hash, VllmRuleVersion::V0_32_0).unwrap());
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[16]);
    seed_states(pool, &[1], hash, 16);
    let request = lease(32, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 40);
    assert_eq!(hit.tokens, 16);
    assert_eq!(split(coordinator.rules(), 16, 32, 40, BUDGET, 0), 24);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
    assert!(coordinator.allocate(pool, &mut block_lease, 39, 16, 40, 1));
    assert_eq!(
        state_lease(&block_lease, 1).checkpoint.map(|(_, p)| p),
        Some(32)
    );
    coordinator.finalize(pool, &mut block_lease, &request, 32, 40, &hit, 1);
    assert!(!hit_exists(pool, 1, hash, 32));
    coordinator.release_request(pool, &mut block_lease);
    coordinator.begin_step();
    let later = coordinator.find_longest_hit(pool, &lease(48, hash), 48);
    assert_eq!(later.tokens, 16);
}

#[test]
fn state_key_published_in_a_step_is_not_a_hit_in_that_step() {
    // single:1726-1734: a state block cached in this step is not a hit
    // until the next step. The block pool stores no step, so the coordinator
    // keeps the set.
    let groups = groups_glm(false);
    let hash = 2176;
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    let mut kv = kv_pool(&groups, 64);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(2500, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 2500);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 2500, 0, 2176, 1));
    coordinator.finalize(pool, &mut block_lease, &request, 2500, 2176, &hit, 1);
    for group in GLM_STATES {
        assert!(hit_exists(pool, group, hash, 2176));
    }
    let same_step = coordinator.find_longest_hit(pool, &lease(2500, hash), 2500);
    assert_eq!(same_step.tokens, 0);
    coordinator.begin_step();
    let next_step = coordinator.find_longest_hit(pool, &lease(2500, hash), 2500);
    assert_eq!(next_step.tokens, 2176);
}

#[test]
fn allocate_fails_atomically() {
    let groups = groups_glm(false);
    let mut coordinator = with_limits(rules_glm(VllmRuleVersion::V0_31_0, false));
    // Room for the first chunk (G0 1, ring 1, 4 running) but not the
    // second (G0 1 more, 4 new running blocks while 4 previous stay).
    let mut kv = kv_pool(&groups, 9);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease(9000, 2176);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 0, 2176, 1));
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 2176, &hit, 1);
    assert_eq!(pool.active_blocks(), 6);
    coordinator.begin_step();
    let before = format!("{block_lease:?}");
    assert!(!coordinator.allocate(pool, &mut block_lease, 9000, 2176, 4352, 2));
    assert_eq!(pool.active_blocks(), 6);
    assert_eq!(format!("{block_lease:?}"), before);
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(pool.active_blocks(), 0);
}

#[test]
fn release_frees_groups_in_profile_order_each_tail_first() {
    // coord:387-395 frees the managers in group order, and each manager frees
    // its blocks tail first (single:581-589). Cached blocks join the LRU tail
    // in that order (pool:777-808). So the cached block of a state group that
    // the profile lists first is evicted before the attention blocks.
    let groups = [state("s", 64, true, false), attention("full", 64, None)];
    let hash = 64;
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 6);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(200, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 200);
        let mut block_lease = coordinator.new_lease();
        let ends = prefill(
            &mut coordinator,
            pool,
            &mut block_lease,
            &request,
            200,
            &hit,
            256,
        );
        assert_eq!(ends, vec![192, 200], "{version:?}");
        coordinator.release_request(pool, &mut block_lease);
        // The state at the replay boundary 192 and the attention blocks that
        // end at 64, 128 and 192 stay cached. Two blocks are free.
        assert_eq!(pool.inactive_cached_blocks(), 4, "{version:?}");
        assert_eq!(pool.active_blocks(), 0, "{version:?}");
        // Three fresh blocks evict one cached block: the state block.
        let reservation = pool.reserve(&[], 3).unwrap();
        pool.cancel(reservation);
        assert!(!hit_exists(pool, 0, hash, 192), "{version:?}");
        for end in [64, 128, 192] {
            assert!(hit_exists(pool, 1, hash, end), "{version:?} at {end}");
        }
        // Four fresh blocks evict one more: the attention tail block.
        let reservation = pool.reserve(&[], 4).unwrap();
        pool.cancel(reservation);
        assert!(!hit_exists(pool, 1, hash, 192), "{version:?}");
        for end in [64, 128] {
            assert!(hit_exists(pool, 1, hash, end), "{version:?} at {end}");
        }
    }
}

#[test]
fn checkpoint_rule_needs_state_groups_at_the_smallest_cacheable_block() {
    // vLLM 0.32.0 tests a checkpoint against `cache_config.block_size`, the
    // smallest prefix-cacheable block, in the scheduler (v0.32 sched:469,
    // :496) and against the Mamba block in the manager (v0.32 single:1652,
    // :1662). The rule table holds one block for both.
    let small_attention = [attention("full", 16, None), state("s", 64, true, true)];
    let err = GroupedPrefixRules::new(&small_attention, 16, VllmRuleVersion::V0_32_0)
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "kv_cache_groups with vllm_version=0.32.0 require every state group with prefill checkpoint blocks to have the smallest block_size_tokens of the prefix_cacheable groups"
    );
    // Without checkpoint blocks the same shape is modeled on every row.
    let no_checkpoint = [attention("full", 16, None), state("s", 64, true, false)];
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        assert!(GroupedPrefixRules::new(&no_checkpoint, 16, version).is_ok());
    }
    // A larger attention block keeps the state block the smallest. The
    // checkpoint position then moves in units of the scheduler block, the LCM
    // 128, in the manager (v0.32 coord:699-706) and in the scheduler (v0.32
    // sched:473-475, :209): `(300 - 1) / 128 * 128 == 256`.
    let large_attention = [attention("full", 128, None), state("s", 64, true, true)];
    let r = GroupedPrefixRules::new(&large_attention, 64, VllmRuleVersion::V0_32_0).unwrap();
    assert_eq!((r.alignment, r.cacheable_block), (128, 64));
    assert_eq!(r.checkpoint_position(300), 256);
}

#[test]
fn retired_window_blocks_are_released_tail_first() {
    // v0.31 single:660-685 and v0.32 single:645-670 free a retired window
    // range backward, and free_blocks appends its cached blocks to the free
    // queue in that order (v0.31 pool:777-806, v0.32 pool:774-803). So the
    // tail block of the range is evicted first.
    let groups = [attention("window", 4, Some(8))];
    let hash = 4;
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        // The run that a finished request of 100 tokens keeps before its
        // replay boundary 96: the blocks that end at 92 and 96.
        seed_many(pool, 0, hash, &[92, 96]);
        let request = lease(108, hash);
        let hit = coordinator.find_longest_hit(pool, &request, 108);
        assert_eq!(hit.tokens, 96, "{version:?}");
        let mut block_lease = coordinator.new_lease();
        coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
        assert!(coordinator.allocate(pool, &mut block_lease, 108, 96, 104, 1));
        coordinator.finalize(pool, &mut block_lease, &request, 108, 104, &hit, 1);
        // The next allocation retires both hit blocks: (104 - 7) / 4 = 24.
        assert!(coordinator.allocate(pool, &mut block_lease, 108, 104, 108, 2));
        assert_eq!(pool.inactive_cached_blocks(), 2, "{version:?}");
        assert_eq!(pool.active_blocks(), 3, "{version:?}");
        // Three blocks are free, so four fresh blocks evict one cached block:
        // the tail block of the retired range, which ends at 96.
        let reservation = pool.reserve(&[], 4).unwrap();
        pool.cancel(reservation);
        assert!(!hit_exists(pool, 0, hash, 96), "{version:?}");
        assert!(hit_exists(pool, 0, hash, 92), "{version:?}");
    }
}

/// The sliding-window search before the single traversal: every candidate
/// end scans its own run again.
fn rescanning_window_run(
    max: usize,
    block: usize,
    step: usize,
    needed: usize,
    mut resident: impl FnMut(usize) -> bool,
) -> (usize, usize) {
    let mut end = max / step * step;
    while end > 0 {
        let blocks = end / block;
        let mut run = 0;
        while run < blocks && run < needed && resident(blocks - run - 1) {
            run += 1;
        }
        if run >= needed || run == blocks {
            return (end, run);
        }
        end -= step;
    }
    (0, 0)
}

#[test]
fn window_search_looks_up_each_block_at_most_once() {
    // single:1018-1044 finds a sliding-window hit in one backward traversal.
    // 4,096 candidate blocks of which the last 2,048 are resident, and a
    // window that needs 4,096 contiguous blocks: no hit.
    let (block, needed) = (16, 4096);
    let resident = |i: usize| i >= 2048;
    let mut lookups = 0;
    let found = window_run(4096 * block, block, block, needed, |i| {
        lookups += 1;
        resident(i)
    });
    let mut rescans = 0;
    let rescanned = rescanning_window_run(4096 * block, block, block, needed, |i| {
        rescans += 1;
        resident(i)
    });
    assert_eq!(found, (0, 0));
    assert_eq!(rescanned, found);
    assert_eq!(rescans, 2_102_272);
    assert_eq!(lookups, 4096);
}

#[test]
fn window_search_finds_the_hits_of_the_rescanning_search() {
    // A fixed xorshift sequence: random residency, blocks, alignments,
    // windows and lengths. Each block is looked up at most once.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    for case in 0..20_000 {
        let block = 1 + next(8) as usize;
        let step = block * (1 + next(4) as usize);
        let needed = next(7) as usize;
        let max = next(400) as usize;
        let density = next(101);
        let blocks = max / block;
        let resident: Vec<bool> = (0..blocks).map(|_| next(100) < density).collect();
        let mut lookups = vec![0; blocks];
        let found = window_run(max, block, step, needed, |i| {
            lookups[i] += 1;
            resident[i]
        });
        let expected = rescanning_window_run(max, block, step, needed, |i| resident[i]);
        assert_eq!(
            found, expected,
            "case {case}: max {max} block {block} step {step} needed {needed} {resident:?}"
        );
        assert!(lookups.iter().all(|&count| count <= 1), "case {case}");
    }
}

/// Whether a scan of every completed attention block, as finalization did
/// before it kept a cursor, would still find a block to publish.
fn rescan_finds_a_block(
    coordinator: &GroupedPrefixCoordinator,
    pool: &GroupedBlockPool,
    lease: &GroupedBlockLease,
    request: &BlockRequestLease,
    prompt: usize,
    computed_after: usize,
    hit: &GroupedHit,
) -> bool {
    use crate::engine::cache::vllm_block_pool::BlockCopyId;
    let rules = coordinator.rules();
    let junction = (hit.junction_tokens > 0).then_some(hit.tokens + hit.junction_tokens);
    let publishable = |g: usize, copy: Option<&Option<BlockCopyId>>, end: usize| {
        copy.is_some_and(|copy| copy.is_some_and(|id| pool.is_private(id)))
            && grouped_key(g, request.prefix_hash_at(end)).is_some()
    };
    rules.groups.iter().enumerate().any(|(g, rule)| {
        let GroupLease::Attention { copies, .. } = &lease.groups[g] else {
            return false;
        };
        let block = rule.block;
        match rule.kind {
            GroupRuleKind::FullAttention => {
                (0..computed_after / block).any(|i| publishable(g, copies.get(i), (i + 1) * block))
            }
            GroupRuleKind::SlidingWindow { window } => [
                Some(rules.replay_boundary(prompt, block)),
                junction.map(|junction| rules.block_floor(junction, block)),
            ]
            .into_iter()
            .flatten()
            .filter(|&pos| pos > 0 && pos <= computed_after)
            .any(|pos| {
                let top = pos / block;
                (top.saturating_sub((window - 1).div_ceil(block))..top)
                    .any(|i| publishable(g, copies.get(i), (i + 1) * block))
            }),
            _ => false,
        }
    })
}

#[test]
fn finalize_publishes_what_a_full_rescan_would() {
    // Two requests share a prefix on a full-attention and a sliding-window
    // group, with chunked prefills and decode steps. After every step a scan
    // of all completed blocks finds nothing more to publish, so the cursor
    // skips only blocks that are cached or have no key.
    let mut state: u64 = 0xD1B5_4A32_D192_ED03;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % bound) as usize
    };
    let mut hits = 0;
    for case in 0..400 {
        let hash = [4, 8][next(2)];
        let groups = [
            attention("full", (hash * (1 + next(2))) as u32, None),
            attention(
                "window",
                (hash * (1 + next(2))) as u32,
                Some(2 + next(40) as u32),
            ),
        ];
        let version = [VllmRuleVersion::V0_31_0, VllmRuleVersion::V0_32_0][next(2)];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, hash, version).unwrap());
        let mut kv = kv_pool(&groups, 4096);
        let pool = kv.block_pool_mut().unwrap();
        let first = 1 + next(300);
        // Half of the cases resend the whole first prompt, which reaches its
        // replay boundary.
        let shared = if next(2) == 0 {
            first
        } else {
            next(first as u64 + 1)
        };
        let second: Vec<u32> = (0..shared as u32)
            .chain(10_000..10_000 + (1 + next(300)) as u32)
            .collect();
        let first: Vec<u32> = (0..first as u32).collect();
        let mut step = 1;
        for tokens in [first, second] {
            let mut request = BlockRequestLease::new(Uuid::nil(), Vec::new());
            request.configure_prefix_hashes(&tokens, hash);
            let prompt = tokens.len();
            let hit = coordinator.find_longest_hit(pool, &request, prompt);
            hits += usize::from(hit.tokens > 0);
            let mut block_lease = coordinator.new_lease();
            coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
            // A chunked prefill, then up to eight decode steps.
            let (budget, end) = (1 + next(64), prompt + 1 + next(8));
            let mut computed = hit.tokens;
            while computed < end {
                coordinator.begin_step();
                let target = (computed + budget).min(prompt).max(computed + 1);
                assert!(coordinator.allocate(
                    pool,
                    &mut block_lease,
                    prompt,
                    computed,
                    target,
                    step
                ));
                coordinator.finalize(pool, &mut block_lease, &request, prompt, target, &hit, step);
                assert!(
                    !rescan_finds_a_block(
                        &coordinator,
                        pool,
                        &block_lease,
                        &request,
                        prompt,
                        target,
                        &hit
                    ),
                    "case {case} {groups:?} prompt {prompt} hit {} at {target}",
                    hit.tokens
                );
                computed = target;
                step += 1;
            }
            coordinator.release_request(pool, &mut block_lease);
        }
    }
    assert!(hits > 150, "{hits} hits");
}

#[test]
fn first_allocation_test_counts_the_blocks_that_the_first_allocation_takes() {
    // The test before a first allocation (mgr:558-576) counts the inactive
    // hit blocks and the fresh blocks of the chunk. pin_hit and the first
    // allocate hold exactly these at their peak, so the test never refuses a
    // chunk that fits and never admits one that does not.
    let layouts = [
        (
            vec![
                attention("full", 16, None),
                attention("window", 16, Some(40)),
            ],
            16,
            VllmRuleVersion::V0_31_0,
        ),
        (
            vec![attention("full", 64, None), state("s", 64, true, false)],
            16,
            VllmRuleVersion::V0_31_0,
        ),
        (
            vec![attention("full", 64, None), state("s", 64, true, false)],
            16,
            VllmRuleVersion::V0_32_0,
        ),
        (
            vec![
                attention("full", 64, None),
                state("ring", 4, false, false),
                state("s", 64, true, true),
            ],
            64,
            VllmRuleVersion::V0_32_0,
        ),
    ];
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % bound) as usize
    };
    let mut hits = 0;
    for (groups, hash, version) in &layouts {
        for case in 0..150 {
            // A small step budget, so that the admission cap of the window
            // binds. The allocation test leaves it out.
            let mut coordinator = GroupedPrefixCoordinator::new(
                GroupedPrefixRules::new(groups, *hash, *version).unwrap(),
                AdmissionLimits {
                    max_in_flight_tokens: 16,
                    max_model_len: usize::MAX,
                },
            );
            let mut kv = kv_pool(groups, 512);
            let pool = kv.block_pool_mut().unwrap();
            let first = 1 + next(400);
            let request = lease(first, *hash);
            let hit = coordinator.find_longest_hit(pool, &request, first);
            let mut block_lease = coordinator.new_lease();
            let budget = 64 * (1 + next(4));
            prefill(
                &mut coordinator,
                pool,
                &mut block_lease,
                &request,
                first,
                &hit,
                budget,
            );
            coordinator.release_request(pool, &mut block_lease);
            // A second request shares a prefix of the first one.
            let shared = if next(2) == 0 {
                first
            } else {
                next(first as u64 + 1)
            };
            let tokens: Vec<u32> = (0..shared as u32)
                .chain(10_000..10_000 + (1 + next(300)) as u32)
                .collect();
            let mut request = BlockRequestLease::new(Uuid::nil(), Vec::new());
            request.configure_prefix_hashes(&tokens, *hash);
            let prompt = tokens.len();
            coordinator.begin_step();
            let hit = coordinator.find_longest_hit(pool, &request, prompt);
            hits += usize::from(hit.tokens > 0);
            let target = if next(2) == 0 {
                prompt
            } else {
                hit.tokens + 1 + next((prompt - hit.tokens) as u64)
            };
            let needed = coordinator.first_allocation_blocks(pool, &hit, prompt, target);
            let before = pool.active_blocks();
            pool.reset_peak_active_blocks();
            let mut block_lease = coordinator.new_lease();
            coordinator.pin_hit(pool, &mut block_lease, &hit).unwrap();
            assert!(coordinator.allocate(pool, &mut block_lease, prompt, hit.tokens, target, 100));
            // The copy of a shared block is taken while the hit block is
            // still held, so the peak counts both.
            assert_eq!(
                pool.peak_active_blocks() - before,
                needed,
                "{version:?} case {case}: prompt {prompt} hit {} target {target}",
                hit.tokens
            );
            coordinator.release_request(pool, &mut block_lease);
        }
    }
    assert!(hits > 200, "{hits} hits");
}

#[test]
fn hit_stamp_changes_exactly_when_a_search_input_changes() {
    // Full attention and a cacheable state, both of 4-token blocks, at hash
    // 4, in a pool of 3 blocks.
    let groups = [attention("full", 4, None), state("s", 4, true, false)];
    let mut coordinator = with_limits(rules(&groups, 4));
    let mut kv = kv_pool(&groups, 3);
    let pool = kv.block_pool_mut().unwrap();
    let start = coordinator.hit_stamp(pool, 8);
    assert_eq!(coordinator.hit_stamp(pool, 8), start);
    // The token count.
    assert_ne!(coordinator.hit_stamp(pool, 9), start);
    // Another pool with the same resident keys.
    let other = kv_pool(&groups, 3);
    assert_ne!(coordinator.hit_stamp(other.block_pool().unwrap(), 8), start);
    // Clearing an empty published set changes nothing.
    coordinator.begin_step();
    assert_eq!(coordinator.hit_stamp(pool, 8), start);

    // A key enters the index.
    seed(pool, key(0, 4, 4));
    let one = coordinator.hit_stamp(pool, 8);
    assert_ne!(one.residency, start.residency);
    // Pinning, activating and releasing it changes no residency.
    let mut reservation = pool.reserve(&[key(0, 4, 4)], 0).unwrap();
    let copies = pool.activate(&mut reservation);
    pool.cancel(reservation);
    pool.release_all(copies);
    assert_eq!(coordinator.hit_stamp(pool, 8), one);
    // Nor does a second copy of the same key, or the loss of one of two
    // copies.
    let mut reservation = pool.reserve(&[], 1).unwrap();
    let id = pool.take_private(&mut reservation);
    pool.cancel(reservation);
    assert!(!pool.cache(id, key(0, 4, 4)));
    pool.release(id);
    assert_eq!(coordinator.hit_stamp(pool, 8), one);
    let reservation = pool.reserve(&[], 2).unwrap();
    assert!(hit_exists(pool, 0, 4, 4));
    assert_eq!(coordinator.hit_stamp(pool, 8), one);
    pool.cancel(reservation);
    // The key leaves the index with its last copy.
    let reservation = pool.reserve(&[], 3).unwrap();
    assert!(!hit_exists(pool, 0, 4, 4));
    let gone = coordinator.hit_stamp(pool, 8);
    assert_ne!(gone.residency, one.residency);
    pool.cancel(reservation);

    // A state key published in the step, then the clear of the set.
    let request = lease(9, 4);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.allocate(pool, &mut block_lease, 9, 0, 8, 1));
    let before = coordinator.hit_stamp(pool, 9);
    let no_hit = coordinator.empty_hit();
    coordinator.finalize(pool, &mut block_lease, &request, 9, 8, &no_hit, 1);
    let published = coordinator.hit_stamp(pool, 9);
    assert!(coordinator.published_this_step.contains(&key(1, 4, 8)));
    assert_ne!(published.published_epoch, before.published_epoch);
    coordinator.begin_step();
    let cleared = coordinator.hit_stamp(pool, 9);
    assert_ne!(cleared.published_epoch, published.published_epoch);
    assert_eq!(cleared.residency, published.residency);
    coordinator.release_request(pool, &mut block_lease);
}

/// Publish in this step, through `finalize`, a second copy of a state key that
/// is already resident, and return whether a hit stored before that publish
/// would still be reused. The residency of every key stays the same, so only
/// the published set tells the waiting request that its hit changed.
fn reused_after_a_duplicate_state_publish(
    groups: &[FpmCacheGroup],
    version: VllmRuleVersion,
    hash: usize,
    seeded: usize,
    waiting_tokens: usize,
    running_prompt: usize,
    running_target: usize,
) -> bool {
    let mut coordinator = with_limits(GroupedPrefixRules::new(groups, hash, version).unwrap());
    let mut kv = kv_pool(groups, 32);
    let pool = kv.block_pool_mut().unwrap();
    // An earlier request left the keys of its first `seeded` tokens.
    let ends: Vec<usize> = (1..=seeded / hash).map(|i| i * hash).collect();
    seed_many(pool, 0, hash, &ends);
    seed(pool, key(1, hash, seeded));
    let waiting = lease(waiting_tokens, hash);
    let first = coordinator.find_longest_hit(pool, &waiting, waiting_tokens);
    assert_eq!(first.tokens, seeded);
    let stored =
        StoredGroupedHit::searched(first.clone(), coordinator.hit_stamp(pool, waiting_tokens));
    let residency = pool.residency();
    // A running request with the same tokens started cold. Its chunk up to
    // `running_target` publishes its state at `seeded` in this step.
    let request = lease(running_prompt, hash);
    let mut block_lease = coordinator.new_lease();
    let target = running_target;
    assert!(coordinator.allocate(pool, &mut block_lease, running_prompt, 0, target, 1));
    let no_hit = coordinator.empty_hit();
    coordinator.finalize(
        pool,
        &mut block_lease,
        &request,
        running_prompt,
        target,
        &no_hit,
        1,
    );
    assert!(
        coordinator
            .published_this_step
            .contains(&key(1, hash, seeded))
    );
    assert_eq!(pool.residency(), residency, "a second copy keeps every key");
    // A state key published in this step is no hit (single:1726-1734).
    let fresh = coordinator.find_longest_hit(pool, &waiting, waiting_tokens);
    assert_ne!(fresh, first);
    let reused = stored
        .reusable(&coordinator.hit_stamp(pool, waiting_tokens))
        .is_some();
    coordinator.release_request(pool, &mut block_lease);
    reused
}

#[test]
fn a_published_running_state_key_ends_the_reuse_of_a_stored_hit() {
    // vLLM 0.31.0, no checkpoint: the running state of a 9-token request is
    // kept at its replay boundary 8, which a waiting 9-token request hits.
    let groups = [attention("full", 4, None), state("s", 4, true, false)];
    assert!(!reused_after_a_duplicate_state_publish(
        &groups,
        VllmRuleVersion::V0_31_0,
        4,
        8,
        9,
        9,
        8
    ));
}

#[test]
fn a_published_checkpoint_state_key_ends_the_reuse_of_a_stored_hit() {
    // vLLM 0.32.0 with a checkpoint block: a 40-token prompt that runs in one
    // chunk exports its checkpoint at 32 (v0.32 iface:1110-1146), which a
    // waiting 41-token request hits.
    let groups = [attention("full", 16, None), state("s", 16, true, true)];
    assert!(!reused_after_a_duplicate_state_publish(
        &groups,
        VllmRuleVersion::V0_32_0,
        16,
        32,
        41,
        40,
        40
    ));
}

#[test]
fn a_moved_producer_state_key_ends_the_reuse_of_a_stored_hit() {
    // The copy of a state that the request published inside its running
    // block hides the key for the rest of the step (v0.31
    // single:1931-1934, v0.32 single:1895-1898, v0.33 single:1912-1915).
    // The residency of every key stays the same, so only the published set
    // tells the waiting request that its hit changed.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let groups = [attention("full", 32, None), state("s", 32, true, false)];
        let mut coordinator = with_limits(GroupedPrefixRules::new(&groups, 4, version).unwrap());
        let mut kv = kv_pool(&groups, 8);
        let pool = kv.block_pool_mut().unwrap();
        let request = lease(50, 4);
        let hit = coordinator.find_longest_hit(pool, &request, 50);
        let mut block_lease = coordinator.new_lease();
        let mut before = 0;
        for (step, target) in [(1, 32), (2, 48)] {
            coordinator.begin_step();
            assert!(coordinator.allocate(pool, &mut block_lease, 50, before, target, step));
            coordinator.finalize(pool, &mut block_lease, &request, 50, target, &hit, step);
            before = target;
        }
        coordinator.begin_step();
        let waiting = lease(60, 4);
        let first = coordinator.find_longest_hit(pool, &waiting, 60);
        assert_eq!(first.tokens, 48, "{version:?}");
        let stored = StoredGroupedHit::searched(first, coordinator.hit_stamp(pool, 60));
        let residency = pool.residency();
        assert!(coordinator.allocate(pool, &mut block_lease, 50, 48, 50, 3));
        assert_eq!(pool.residency(), residency, "{version:?}");
        assert!(
            stored.reusable(&coordinator.hit_stamp(pool, 60)).is_none(),
            "{version:?}"
        );
        coordinator.release_request(pool, &mut block_lease);
    }
}

/// The hash block of the copy-source tests. A hit can then end inside a
/// 2,176-token block of the GLM groups.
const FINE: usize = 1088;

/// A request over `tokens` tokens that start at `base`. Two bases give
/// two prefixes that share no hash.
fn lease_from(base: u32, tokens: usize) -> BlockRequestLease {
    let tokens: Vec<u32> = (base..base + tokens as u32).collect();
    let mut lease = BlockRequestLease::new(Uuid::nil(), Vec::new());
    lease.configure_prefix_hashes(&tokens, FINE);
    lease
}

/// Prefill a request of 3,500 tokens from no hit and release it. Its
/// chunks end at 2,176, 3,264 and 3,500, so it publishes a state at
/// 3,264, inside the second block of each 2,176-token group.
fn produce(coordinator: &mut GroupedPrefixCoordinator, pool: &mut GroupedBlockPool, base: u32) {
    let request = lease_from(base, 3500);
    let hit = coordinator.find_longest_hit(pool, &request, 3500);
    assert_eq!(hit.tokens, 0);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    let ends = prefill(
        coordinator,
        pool,
        &mut block_lease,
        &request,
        3500,
        &hit,
        BUDGET,
    );
    assert_eq!(ends, [2176, 3264, 3500]);
    coordinator.release_request(pool, &mut block_lease);
}

/// Admit a request of 3,800 tokens on the prefix at `base` and allocate
/// its whole prompt. It hits 3,264 tokens, inside a block.
fn consume(
    coordinator: &mut GroupedPrefixCoordinator,
    pool: &mut GroupedBlockPool,
    base: u32,
) -> (BlockRequestLease, GroupedBlockLease, GroupedHit) {
    let request = lease_from(base, 3800);
    let hit = coordinator.find_longest_hit(pool, &request, 3800);
    assert_eq!(hit.tokens, 3264);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 3800, 3264, 3800, 1));
    (request, block_lease, hit)
}

/// `None` when the key of `tokens` in `group` is not resident, else
/// whether a request holds its block.
fn key_state(
    pool: &GroupedBlockPool,
    group: u16,
    request: &BlockRequestLease,
    tokens: usize,
) -> Option<bool> {
    pool.key_hit(group, request.prefix_hash_at(tokens).unwrap())
        .map(|hit| hit.is_active)
}

#[test]
fn copy_sources_of_a_hit_inside_a_block_stay_pinned_until_the_pass_ends() {
    // A hit inside a block copies the cached block of the full-attention
    // group (v0.31 and v0.32 single:387-397, `_apply_cow` at :454-474,
    // v0.33 :388-398, :455-475) and the hit block of each state group
    // (v0.31 single:1935-1937, v0.32 :1899-1901, v0.33 :1916-1918). Each
    // source keeps the reference that `touch` gave it (v0.31 single:307,
    // v0.32 :304, v0.33 :305) until `schedule()` frees the copy endpoints
    // at its end (v0.31 sched:1434-1442, v0.32 :1474-1482, v0.33
    // :1698-1706). Until then a later request of the same step cannot
    // take or evict it.
    for version in [
        VllmRuleVersion::V0_31_0,
        VllmRuleVersion::V0_32_0,
        VllmRuleVersion::V0_33_0,
    ] {
        let mut coordinator = with_limits(rules_glm_hash(version, false, FINE));
        let mut kv = kv_pool(&groups_glm(false), 32);
        let pool = kv.block_pool_mut().unwrap();
        produce(&mut coordinator, pool, 0);
        assert_eq!(pool.active_blocks(), 0);
        assert_eq!(pool.inactive_cached_blocks(), 10);
        let (request, mut block_lease, hit) = consume(&mut coordinator, pool, 0);
        // The lease holds 2 full-attention blocks, the ring block and a
        // running block in each state group. The 5 sources are active
        // too: 7 + 5 = 12.
        assert_eq!(coordinator.copy_source_count(), 5);
        assert_eq!(pool.active_blocks(), 12);
        assert_eq!(pool.inactive_cached_blocks(), 4);
        assert_eq!(attention_copies(&block_lease, 0), 2);
        for group in GLM_STATES {
            let state = state_lease(&block_lease, usize::from(group));
            assert!(state.hit.is_none() && state.running.is_some() && state.copy.is_none());
        }
        for group in [0, 2, 3, 4, 5] {
            assert_eq!(key_state(pool, group, &request, 3264), Some(true));
        }
        // Only the blocks that are not active count as free.
        let capacity = pool.capacity_blocks();
        assert!(pool.reserve(&[], capacity - 12 + 1).is_none());
        assert_eq!(pool.active_blocks(), 12);
        assert_eq!(pool.inactive_cached_blocks(), 4);
        let reservation = pool.reserve(&[], capacity - 12).unwrap();
        pool.cancel(reservation);
        assert_eq!(pool.inactive_cached_blocks(), 0);
        assert_eq!(key_state(pool, 0, &request, 3264), Some(true));
        coordinator.finalize(pool, &mut block_lease, &request, 3800, 3800, &hit, 1);
        assert_eq!(coordinator.copy_source_count(), 5);
        assert_eq!(pool.active_blocks(), 12);
        assert_eq!(coordinator.release_copy_sources(pool), 5);
        assert_eq!(coordinator.copy_source_count(), 0);
        assert_eq!(pool.active_blocks(), 7);
        for group in [0, 2, 3, 4, 5] {
            assert_eq!(key_state(pool, group, &request, 3264), Some(false));
        }
    }
}

#[test]
fn copy_sources_leave_in_group_order_then_in_allocation_order() {
    // vLLM collects the copies manager by manager, in group order, and
    // each manager in the order of its allocations (v0.31
    // kv_cache_manager.py:894-909). `free_blocks` puts a cached block
    // that no request holds at the tail of the free queue, in that order
    // (v0.31 block_pool.py:777-808). So the sources of two requests of
    // one step leave group by group.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 40);
    let pool = kv.block_pool_mut().unwrap();
    produce(&mut coordinator, pool, 0);
    produce(&mut coordinator, pool, 1_000_000);
    let (first, mut first_lease, first_hit) = consume(&mut coordinator, pool, 0);
    coordinator.finalize(pool, &mut first_lease, &first, 3800, 3800, &first_hit, 1);
    let (second, mut second_lease, second_hit) = consume(&mut coordinator, pool, 1_000_000);
    coordinator.finalize(pool, &mut second_lease, &second, 3800, 3800, &second_hit, 1);
    assert_eq!(coordinator.copy_source_count(), 10);
    assert_eq!(pool.active_blocks(), 24);
    assert_eq!(pool.inactive_cached_blocks(), 8);
    assert_eq!(coordinator.release_copy_sources(pool), 10);
    assert_eq!(pool.active_blocks(), 14);
    assert_eq!(pool.inactive_cached_blocks(), 18);
    let free = pool.capacity_blocks() - 14 - 18;
    assert_eq!(free, 8);
    // The 8 free blocks and 11 evictions: the 8 states at 2,176 of the
    // two producers, then the full-attention sources of both requests,
    // then the source of group 2 of the first request, which allocated
    // first.
    let mut reservation = pool.reserve(&[], free + 11).unwrap();
    let mut taken: Vec<_> = (0..free + 11)
        .map(|_| pool.take_private(&mut reservation))
        .collect();
    pool.cancel(reservation);
    assert_eq!(
        GLM_STATES.map(|group| key_state(pool, group, &first, 3264).is_some()),
        [false, true, true, true]
    );
    assert_eq!(
        GLM_STATES.map(|group| key_state(pool, group, &second, 3264).is_some()),
        [true, true, true, true]
    );
    // The 12th eviction takes the source of group 2 of the second request.
    let mut reservation = pool.reserve(&[], 1).unwrap();
    taken.push(pool.take_private(&mut reservation));
    pool.cancel(reservation);
    for request in [&first, &second] {
        assert_eq!(
            GLM_STATES.map(|group| key_state(pool, group, request, 3264).is_some()),
            [false, true, true, true]
        );
    }
    pool.release_all(taken);
}

#[test]
fn copy_sources_outlive_a_request_that_leaves_in_its_admission_pass() {
    // Freeing a request leaves the pending copies alone (v0.31
    // single:562-579, :1971-1988), so its sources stay referenced until
    // `schedule()` ends (v0.31 sched:1434-1442).
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 32);
    let pool = kv.block_pool_mut().unwrap();
    produce(&mut coordinator, pool, 0);
    let (request, mut block_lease, hit) = consume(&mut coordinator, pool, 0);
    coordinator.finalize(pool, &mut block_lease, &request, 3800, 3800, &hit, 1);
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(coordinator.copy_source_count(), 5);
    assert_eq!(pool.active_blocks(), 5);
    assert_eq!(key_state(pool, 0, &request, 3264), Some(true));
    let capacity = pool.capacity_blocks();
    assert!(pool.reserve(&[], capacity - 5 + 1).is_none());
    assert_eq!(coordinator.release_copy_sources(pool), 5);
    assert_eq!(pool.active_blocks(), 0);
}

#[test]
fn a_hit_on_a_block_boundary_pins_no_copy_source() {
    // A hit on a block boundary takes no copy (v0.31 single:321-327), so
    // the hit block of a state group stays in the request until its next
    // allocation (v0.31 mgr:552-556, single:1626-1645).
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 32);
    let pool = kv.block_pool_mut().unwrap();
    let request = lease_from(0, 4400);
    let hit = coordinator.find_longest_hit(pool, &request, 4400);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    let ends = prefill(
        &mut coordinator,
        pool,
        &mut block_lease,
        &request,
        4400,
        &hit,
        BUDGET,
    );
    assert_eq!(ends, [4352, 4400]);
    coordinator.release_request(pool, &mut block_lease);
    let request = lease_from(0, 4700);
    let hit = coordinator.find_longest_hit(pool, &request, 4700);
    assert_eq!(hit.tokens, 4352);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert_eq!(pool.active_blocks(), 6);
    assert!(coordinator.allocate(pool, &mut block_lease, 4700, 4352, 4700, 1));
    assert_eq!(coordinator.copy_source_count(), 0);
    assert_eq!(pool.active_blocks(), 12);
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.hit.is_some() && state.running.is_some() && state.copy.is_none());
    }
    coordinator.finalize(pool, &mut block_lease, &request, 4700, 4700, &hit, 1);
    assert_eq!(coordinator.copy_source_count(), 0);
    assert_eq!(pool.active_blocks(), 12);
    assert_eq!(pool.inactive_cached_blocks(), 0);
    assert_eq!(coordinator.release_copy_sources(pool), 0);
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 4700, 4700, 4701, 2));
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(state.hit.is_none() && state.running.is_some());
    }
    assert_eq!(pool.active_blocks(), 8);
    assert_eq!(pool.inactive_cached_blocks(), 4);
}

#[test]
fn a_failed_first_allocation_pins_no_copy_source() {
    // vLLM tests the free blocks before it touches the hit
    // (v0.31 kv_cache_manager.py:572-589), so a refused request pins
    // nothing.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 11);
    let pool = kv.block_pool_mut().unwrap();
    produce(&mut coordinator, pool, 0);
    assert_eq!(pool.active_blocks(), 0);
    assert_eq!(pool.inactive_cached_blocks(), 6);
    let request = lease_from(0, 3800);
    let hit = coordinator.find_longest_hit(pool, &request, 3800);
    assert_eq!(hit.tokens, 3264);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert_eq!(pool.active_blocks(), 6);
    // 6 fresh blocks are needed and 5 are left.
    assert!(!coordinator.allocate(pool, &mut block_lease, 3800, 3264, 3800, 1));
    assert_eq!(coordinator.copy_source_count(), 0);
    assert_eq!(pool.active_blocks(), 6);
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(pool.active_blocks(), 0);
    assert_eq!(pool.inactive_cached_blocks(), 6);
}

/// Prefill a request of 4,400 tokens from no hit and release it. Its chunks
/// end at 4,352 and 4,400, so it publishes a state at 4,352, on a block
/// boundary of each 2,176-token group.
fn produce_on_a_boundary(
    coordinator: &mut GroupedPrefixCoordinator,
    pool: &mut GroupedBlockPool,
    base: u32,
) {
    let request = lease_from(base, 4400);
    let hit = coordinator.find_longest_hit(pool, &request, 4400);
    assert_eq!(hit.tokens, 0);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    let ends = prefill(
        coordinator,
        pool,
        &mut block_lease,
        &request,
        4400,
        &hit,
        BUDGET,
    );
    assert_eq!(ends, [4352, 4400]);
    coordinator.release_request(pool, &mut block_lease);
}

/// Evict the oldest cached block: take every free block and one more, and
/// give them back.
fn evict_one(pool: &mut GroupedBlockPool) {
    let free = pool.capacity_blocks() - pool.active_blocks() - pool.inactive_cached_blocks();
    let mut reservation = pool.reserve(&[], free + 1).unwrap();
    let taken: Vec<_> = (0..free + 1)
        .map(|_| pool.take_private(&mut reservation))
        .collect();
    pool.cancel(reservation);
    pool.release_all(taken);
}

#[test]
fn a_boundary_hit_block_leaves_after_the_copy_sources_of_its_pass() {
    // A hits 3,264 tokens, inside a block, and B hits 4,352, on a block
    // boundary. vLLM frees A's copy sources at the end of the step
    // (v0.31 sched:1434-1442) and B's hit blocks at B's next
    // `allocate_slots` (v0.31 mgr:552-556, single:1626-1645), in a later
    // step. So A's sources enter the free queue first, in either
    // admission order.
    for a_first in [true, false] {
        let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
        let mut kv = kv_pool(&groups_glm(false), 64);
        let pool = kv.block_pool_mut().unwrap();
        produce(&mut coordinator, pool, 0);
        produce_on_a_boundary(&mut coordinator, pool, 1_000_000);
        let b_request = lease_from(1_000_000, 4700);
        let admit_b = |coordinator: &mut GroupedPrefixCoordinator, pool: &mut GroupedBlockPool| {
            let hit = coordinator.find_longest_hit(pool, &b_request, 4700);
            assert_eq!(hit.tokens, 4352);
            let mut lease = coordinator.new_lease();
            assert!(coordinator.pin_hit(pool, &mut lease, &hit).is_ok());
            assert!(coordinator.allocate(pool, &mut lease, 4700, 4352, 4700, 1));
            coordinator.finalize(pool, &mut lease, &b_request, 4700, 4700, &hit, 1);
            (lease, hit)
        };
        let admit_a = |coordinator: &mut GroupedPrefixCoordinator, pool: &mut GroupedBlockPool| {
            let (request, mut lease, hit) = consume(coordinator, pool, 0);
            coordinator.finalize(pool, &mut lease, &request, 3800, 3800, &hit, 1);
            (request, lease, hit)
        };
        let ((a_request, mut a_lease, a_hit), (mut b_lease, b_hit)) = if a_first {
            let a = admit_a(&mut coordinator, pool);
            (a, admit_b(&mut coordinator, pool))
        } else {
            let b = admit_b(&mut coordinator, pool);
            (admit_a(&mut coordinator, pool), b)
        };
        // A: 7 blocks and 5 sources. B: 8 blocks and 4 hit blocks.
        assert_eq!(pool.active_blocks(), 24, "{a_first}");
        assert_eq!(coordinator.release_copy_sources(pool), 5);
        assert_eq!(pool.active_blocks(), 19, "{a_first}");
        // The next pass: both requests compute one token, in running order.
        coordinator.begin_step();
        for is_a in if a_first {
            [true, false]
        } else {
            [false, true]
        } {
            if is_a {
                assert!(coordinator.allocate(pool, &mut a_lease, 3800, 3800, 3801, 2));
                coordinator.finalize(pool, &mut a_lease, &a_request, 3800, 3801, &a_hit, 2);
            } else {
                assert!(coordinator.allocate(pool, &mut b_lease, 4700, 4700, 4701, 2));
                coordinator.finalize(pool, &mut b_lease, &b_request, 4700, 4701, &b_hit, 2);
            }
        }
        assert_eq!(coordinator.release_copy_sources(pool), 0);
        assert_eq!(pool.active_blocks(), 15, "{a_first}");
        assert_eq!(pool.inactive_cached_blocks(), 13, "{a_first}");
        // The four producer states at 2,176 go first, then A's sources in
        // group order, then B's hit blocks.
        let (mut a_gone, mut b_gone) = (None, None);
        for eviction in 1..=13 {
            evict_one(pool);
            if a_gone.is_none() && key_state(pool, 2, &a_request, 3264).is_none() {
                a_gone = Some(eviction);
            }
            if b_gone.is_none() && key_state(pool, 2, &b_request, 4352).is_none() {
                b_gone = Some(eviction);
            }
        }
        assert_eq!((a_gone, b_gone), (Some(6), Some(10)), "{a_first}");
    }
}

#[test]
fn a_request_that_ends_first_frees_its_boundary_hit_block_after_its_attention_blocks() {
    // A request that ends frees its blocks in group order, each group tail
    // first (v0.31 coord:387-395, single:581-589). Its hit blocks are still
    // in its state groups, so they go after its full-attention blocks.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 32);
    let pool = kv.block_pool_mut().unwrap();
    produce_on_a_boundary(&mut coordinator, pool, 0);
    let request = lease_from(0, 4700);
    let hit = coordinator.find_longest_hit(pool, &request, 4700);
    assert_eq!(hit.tokens, 4352);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 4700, 4352, 4700, 1));
    coordinator.finalize(pool, &mut block_lease, &request, 4700, 4700, &hit, 1);
    assert_eq!(pool.active_blocks(), 12);
    assert_eq!(coordinator.release_copy_sources(pool), 0);
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(pool.active_blocks(), 0);
    assert_eq!(pool.inactive_cached_blocks(), 6);
    evict_one(pool);
    assert_eq!(key_state(pool, 0, &request, 4352), None);
    assert_eq!(key_state(pool, 0, &request, 2176), Some(false));
    evict_one(pool);
    assert_eq!(key_state(pool, 0, &request, 2176), None);
    for group in GLM_STATES {
        assert_eq!(key_state(pool, group, &request, 4352), Some(false));
    }
}

#[test]
fn a_request_with_a_copy_source_and_a_boundary_hit_releases_the_source_first() {
    // The full-attention group (block 64) hits 48 tokens inside a block and
    // takes a copy. The state group (block 16) hits on its block boundary
    // at 48 and takes none. The copy source leaves at the end of the pass
    // (v0.31 sched:1434-1442), the state hit block at the request's next
    // allocation (v0.31 mgr:552-556).
    let groups = [attention("full", 64, None), state("kda", 16, true, false)];
    let hash = 8;
    let mut coordinator =
        with_limits(GroupedPrefixRules::new(&groups, hash, VllmRuleVersion::V0_31_0).unwrap());
    let mut kv = kv_pool(&groups, 16);
    let pool = kv.block_pool_mut().unwrap();
    seed_many(pool, 0, hash, &[48]);
    seed_states(pool, &[1], hash, 48);
    let request = lease(100, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 100);
    assert_eq!(hit.tokens, 48);
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 100, 48, 100, 1));
    assert_eq!(coordinator.copy_source_count(), 1);
    assert!(state_lease(&block_lease, 1).hit.is_some());
    coordinator.finalize(pool, &mut block_lease, &request, 100, 100, &hit, 1);
    assert!(state_lease(&block_lease, 1).hit.is_some());
    assert_eq!(coordinator.release_copy_sources(pool), 1);
    evict_one(pool);
    assert!(!hit_exists(pool, 0, hash, 48));
    assert!(hit_exists(pool, 1, hash, 48));
}

/// A request of 9,000 tokens on the prefix of `produce_on_a_boundary`: it
/// hits 4,352 tokens, on a block boundary, and its first chunk computes to
/// 8,704. It then holds 13 blocks: 4 full-attention blocks, the ring block,
/// and a hit block and a running block in each state group.
fn admit_past_a_boundary_hit(
    coordinator: &mut GroupedPrefixCoordinator,
    pool: &mut GroupedBlockPool,
) -> GroupedBlockLease {
    let request = lease_from(0, 9000);
    let hit = coordinator.find_longest_hit(pool, &request, 9000);
    assert_eq!(hit.tokens, 4352);
    let mut block_lease = coordinator.new_lease();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 4352, 8704, 1));
    coordinator.finalize(pool, &mut block_lease, &request, 9000, 8704, &hit, 1);
    assert_eq!(coordinator.release_copy_sources(pool), 0);
    assert_eq!(pool.active_blocks(), 13);
    block_lease
}

#[test]
fn a_next_allocation_can_take_the_blocks_of_its_boundary_hit() {
    // vLLM frees the skipped blocks before the free-block test of
    // `allocate_slots` (v0.31 kv_cache_manager.py:552-556, :570-576). The
    // next chunk, to 9,000, crosses the block boundary at 8,704 and needs 5
    // fresh blocks: one full-attention block and a running block in each
    // state group. On a pool of 14 blocks one block is free and the 4 hit
    // blocks leave first, so the allocation succeeds.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 14);
    let pool = kv.block_pool_mut().unwrap();
    produce_on_a_boundary(&mut coordinator, pool, 0);
    let mut block_lease = admit_past_a_boundary_hit(&mut coordinator, pool);
    coordinator.begin_step();
    assert!(coordinator.allocate(pool, &mut block_lease, 9000, 8704, 9000, 2));
    // 13 - 4 hit blocks + 5 fresh blocks.
    assert_eq!(pool.active_blocks(), 14);
    for group in GLM_STATES {
        assert!(state_lease(&block_lease, usize::from(group)).hit.is_none());
    }
    coordinator.release_request(pool, &mut block_lease);
}

#[test]
fn a_refused_next_allocation_still_releases_the_boundary_hit_block() {
    // On a pool of 13 blocks no block is free, and the 4 released hit blocks
    // do not cover the 5 fresh blocks, so the allocation is refused. vLLM
    // has freed the skipped blocks before its test, so they stay released.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 13);
    let pool = kv.block_pool_mut().unwrap();
    produce_on_a_boundary(&mut coordinator, pool, 0);
    let mut block_lease = admit_past_a_boundary_hit(&mut coordinator, pool);
    coordinator.begin_step();
    assert!(!coordinator.allocate(pool, &mut block_lease, 9000, 8704, 9000, 2));
    assert_eq!(pool.active_blocks(), 9);
    assert_eq!(pool.inactive_cached_blocks(), 4);
    for group in GLM_STATES {
        assert!(state_lease(&block_lease, usize::from(group)).hit.is_none());
    }
    // A second try changes nothing, and the request leaves cleanly.
    assert!(!coordinator.allocate(pool, &mut block_lease, 9000, 8704, 9000, 2));
    assert_eq!(pool.active_blocks(), 9);
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(pool.active_blocks(), 0);
}

/// A request of 13,000 tokens with checkpoint blocks: it hits 8,704 tokens,
/// on a block boundary, and computes its whole prompt in one chunk. It then
/// holds 19 blocks: 6 full-attention blocks, the ring block, and a hit, a
/// running and a checkpoint block in each state group. The checkpoint block
/// holds the state at 10,880 and is cached.
fn admit_with_a_checkpoint_block(
    coordinator: &mut GroupedPrefixCoordinator,
    pool: &mut GroupedBlockPool,
    version: VllmRuleVersion,
) -> GroupedBlockLease {
    let hash = 2176;
    seed_many(pool, 0, hash, &[2176, 4352, 6528, 8704]);
    seed_states(pool, &GLM_STATES, hash, 8704);
    let request = lease(13000, hash);
    let hit = coordinator.find_longest_hit(pool, &request, 13000);
    assert_eq!(hit.tokens, 8704, "{version:?}");
    let mut block_lease = coordinator.new_lease();
    coordinator.begin_step();
    assert!(coordinator.pin_hit(pool, &mut block_lease, &hit).is_ok());
    assert!(coordinator.allocate(pool, &mut block_lease, 13000, 8704, 13000, 1));
    coordinator.finalize(pool, &mut block_lease, &request, 13000, 13000, &hit, 1);
    assert_eq!(coordinator.release_copy_sources(pool), 0);
    for group in GLM_STATES {
        let state = state_lease(&block_lease, usize::from(group));
        assert!(
            state.hit.is_some() && state.checkpoint.is_some(),
            "{version:?}"
        );
    }
    assert_eq!(pool.active_blocks(), 19, "{version:?}");
    assert_eq!(pool.inactive_cached_blocks(), 0, "{version:?}");
    block_lease
}

#[test]
fn the_checkpoint_block_leaves_before_the_boundary_hit_block_of_its_group() {
    // vLLM frees the skipped blocks manager by manager, in group order (v0.32
    // kv_cache_coordinator.py:447-450, v0.33 :451-454), and the blocks of a
    // state group in descending index order (v0.32
    // single_type_kv_cache_manager.py:1604-1608, v0.33 :1620-1624). So the
    // checkpoint block of group 2 enters the free queue first, then its hit
    // block, then the blocks of group 3.
    for version in [VllmRuleVersion::V0_32_0, VllmRuleVersion::V0_33_0] {
        let mut coordinator = with_limits(rules_glm(version, true));
        let mut kv = kv_pool(&groups_glm(true), 64);
        let pool = kv.block_pool_mut().unwrap();
        let mut block_lease = admit_with_a_checkpoint_block(&mut coordinator, pool, version);
        coordinator.begin_step();
        assert!(coordinator.allocate(pool, &mut block_lease, 13000, 13000, 13001, 2));
        // The 4 hit and the 4 checkpoint blocks are cached now.
        assert_eq!(pool.active_blocks(), 11, "{version:?}");
        assert_eq!(pool.inactive_cached_blocks(), 8, "{version:?}");
        evict_one(pool);
        assert!(!hit_exists(pool, 2, 2176, 10880), "{version:?}");
        assert!(hit_exists(pool, 2, 2176, 8704), "{version:?}");
        evict_one(pool);
        assert!(!hit_exists(pool, 2, 2176, 8704), "{version:?}");
        assert!(hit_exists(pool, 3, 2176, 10880), "{version:?}");
        coordinator.release_request(pool, &mut block_lease);
    }
}

#[test]
fn a_request_that_ends_first_frees_its_checkpoint_block_before_its_boundary_hit_block() {
    // A request that ends frees its blocks manager by manager, in group order
    // (v0.32 kv_cache_coordinator.py:407-408, v0.33 :411-412), and each group's
    // table in reverse order (v0.32 single_type_kv_cache_manager.py:581-589,
    // v0.33 :591-599): in a state group the running block, the checkpoint
    // block, then the hit block, and then the next group.
    for version in [VllmRuleVersion::V0_32_0, VllmRuleVersion::V0_33_0] {
        let mut coordinator = with_limits(rules_glm(version, true));
        let mut kv = kv_pool(&groups_glm(true), 64);
        let pool = kv.block_pool_mut().unwrap();
        let mut block_lease = admit_with_a_checkpoint_block(&mut coordinator, pool, version);
        coordinator.release_request(pool, &mut block_lease);
        assert_eq!(pool.active_blocks(), 0, "{version:?}");
        // The five cached full-attention blocks go first, then the
        // checkpoint block of group 2.
        for _ in 0..6 {
            evict_one(pool);
        }
        assert!(!hit_exists(pool, 2, 2176, 10880), "{version:?}");
        assert!(hit_exists(pool, 2, 2176, 8704), "{version:?}");
        // Then the hit block of group 2, before the blocks of group 3.
        evict_one(pool);
        assert!(!hit_exists(pool, 2, 2176, 8704), "{version:?}");
        assert!(hit_exists(pool, 3, 2176, 10880), "{version:?}");
    }
}

#[test]
fn a_request_that_ends_first_frees_its_running_state_block_before_its_boundary_hit_block() {
    // The first chunk ends on the block boundary at 8,704, so the running
    // block of each state group holds the cached state at 8,704. A request
    // that ends frees each group's table in reverse order (v0.31
    // single_type_kv_cache_manager.py:581-589): the running block, at index
    // 3, before the hit block, at index 1. `release_request` has no rule that
    // depends on the vLLM version, so one rule row is enough.
    let mut coordinator = with_limits(rules_glm_hash(VllmRuleVersion::V0_31_0, false, FINE));
    let mut kv = kv_pool(&groups_glm(false), 64);
    let pool = kv.block_pool_mut().unwrap();
    produce_on_a_boundary(&mut coordinator, pool, 0);
    let request = lease_from(0, 9000);
    let mut block_lease = admit_past_a_boundary_hit(&mut coordinator, pool);
    for group in GLM_STATES {
        assert_eq!(key_state(pool, group, &request, 8704), Some(true));
        assert_eq!(key_state(pool, group, &request, 4352), Some(true));
    }
    coordinator.release_request(pool, &mut block_lease);
    assert_eq!(pool.active_blocks(), 0);
    // The four full-attention blocks, tail first, then the running block of
    // group 2.
    for _ in 0..5 {
        evict_one(pool);
    }
    for at in [2176, 4352, 6528, 8704] {
        assert_eq!(key_state(pool, 0, &request, at), None, "{at}");
    }
    assert_eq!(key_state(pool, 2, &request, 8704), None);
    assert_eq!(key_state(pool, 2, &request, 4352), Some(false));
}
