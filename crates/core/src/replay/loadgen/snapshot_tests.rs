// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Self-authored request-boundary fixtures. These test logical snapshot state;
//! the sibling replay suite qualifies physical cache reuse in native engines.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::*;

fn row(id: &str, conversation: &str, start: f64, duration: Option<f64>) -> AgenticMooncakeRow {
    AgenticMooncakeRow {
        request_id: id.into(),
        play_id: "play".into(),
        session_id: conversation.into(),
        model: "model".into(),
        input_length: Some(128),
        output_length: Some(3),
        hash_ids: Some(vec![10, 20]),
        not_before_ms: start,
        recorded_api_time_ms: duration,
        ..Default::default()
    }
}

fn edge(
    source: &str,
    relation: AgenticDependencyRelation,
    trigger: AgenticDependencyTrigger,
    delay: f64,
) -> AgenticDependency {
    AgenticDependency {
        request_id: source.into(),
        relation,
        trigger,
        delay_ms: delay,
    }
}

fn graph(rows: Vec<AgenticMooncakeRow>) -> ValidatedAgenticGraph {
    ValidatedAgenticGraph::from_agentic_mooncake_rows(
        AgenticMooncakeHeader {
            schema: AGENTIC_MOONCAKE_SCHEMA.into(),
            version: AGENTIC_MOONCAKE_VERSION,
            block_size: 64,
            hash_id_scope: AgenticHashIdScope::Local,
            source: AgenticSourceProvenance {
                format: "self-authored-test".into(),
                digest: "snapshot-fixture-v1".into(),
            },
        },
        rows,
    )
    .unwrap()
}

fn boundary_graph() -> ValidatedAgenticGraph {
    use AgenticDependencyRelation::{Sequence, Spawn};
    use AgenticDependencyTrigger::{Completion, Dispatch};
    let root = row("root", "main", 0.0, Some(120.0));
    let mut child = row("child", "worker", 20.0, Some(80.0));
    child.dependencies = vec![edge("root", Spawn, Dispatch, 20.0)];
    let mut background = row("background", "finished-conversation", 80.0, Some(500.0));
    background.dependencies = vec![edge("root", Spawn, Dispatch, 80.0)];
    let mut boundary = row("at-cut", "main", 100.0, Some(40.0));
    boundary.dependencies = vec![edge("root", Sequence, Completion, 10.0)];
    let mut child_next = row("child-next", "worker", 140.0, Some(20.0));
    child_next.dependencies = vec![edge("child", Sequence, Completion, 10.0)];
    let mut future = row("future", "main", 160.0, None);
    future.dependencies = vec![edge("at-cut", Sequence, Completion, 20.0)];
    graph(vec![root, child, background, boundary, child_next, future])
}

fn request<'a>(play: &'a AgenticPlaySnapshot, source: &str) -> &'a AgenticSnapshotRequest {
    play.evidence()
        .requests
        .iter()
        .find(|request| request.source_request_id == source)
        .unwrap()
}

fn prepare(graph: &ValidatedAgenticGraph, cut: f64) -> AgenticPlaySnapshot {
    graph
        .prepare_snapshots(
            1,
            AgenticSnapshotOptions {
                seed: 42,
                sampled: None,
            },
        )
        .unwrap()
        .context()
        .prepare_play(0, 0, Some(cut))
        .unwrap()
}

fn drain(mut driver: WorkloadDriver) -> BTreeMap<String, ReadyTurn> {
    let mut requests = BTreeMap::new();
    while !driver.is_drained() {
        let now = driver
            .next_ready_time_ms()
            .expect("fixture must have a future wakeup");
        let ready = driver.pop_ready(now, usize::MAX);
        assert!(!ready.is_empty());
        for ready in ready {
            driver.on_complete(ready.request_uuid, now).unwrap();
            requests.insert(ready.authored_request_id.clone().unwrap(), ready);
        }
    }
    requests
}

#[test]
fn full_play_start_handles_nonzero_source_clocks_without_changing_absolute_cuts() {
    let root = row("root", "main", 10_000.0, Some(250.0));
    let mut next = row("next", "main", 11_000.0, Some(250.0));
    next.dependencies = vec![edge(
        "root",
        AgenticDependencyRelation::Sequence,
        AgenticDependencyTrigger::Completion,
        750.0,
    )];
    let mut earlier_play = row("other-play", "other", 1_000.0, None);
    earlier_play.play_id = "earlier-play".into();
    let mooncake = graph(vec![earlier_play, root, next]);

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("nonzero-origin.json");
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "id": "nonzero-origin", "models": ["model"],
            "block_size": 64, "hash_id_scope": "local",
            "requests": [
                {"t": 10, "type": "s", "model": "model", "in": 128,
                 "out": 3, "api_time": 0.25, "hash_ids": [10, 20]},
                {"t": 11, "type": "s", "model": "model", "in": 192,
                 "out": 3, "api_time": 0.25, "hash_ids": [10, 20, 30]}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let weka = load_weka_agentic_graph(&path, Some(64)).unwrap();

    // Weka normalizes its nonzero source clock; Mooncake retains absolute starts.
    for (graph, first_ms) in [(mooncake, 10_000.0), (weka, 0.0)] {
        let prepared = graph
            .prepare_snapshots(
                1,
                AgenticSnapshotOptions {
                    seed: 42,
                    sampled: None,
                },
            )
            .unwrap();
        let context = prepared.context();
        let play = context.prepare_play_from_start(0, 1).unwrap();
        assert_eq!(play.evidence().t_star_ms, first_ms);
        assert_eq!(play.evidence().requests.len(), 2);
        assert!(play.evidence().requests.iter().all(|r| !r.historical));
        assert!(play.evidence().primers.is_empty());
        assert_eq!(
            play.evidence(),
            context
                .prepare_play(0, 1, Some(first_ms))
                .unwrap()
                .evidence()
        );
        assert_eq!(
            context
                .prepare_play(0, 1, Some(first_ms + 500.0))
                .unwrap()
                .evidence()
                .requests
                .iter()
                .filter(|r| r.historical)
                .count(),
            1
        );
        if first_ms > 0.0 {
            assert!(context.prepare_play(0, 1, Some(0.0)).is_err());
        }

        let mut driver = WorkloadDriver::new_agentic_snapshots(
            PreparedAgenticSnapshots::from_plays(vec![play]).unwrap(),
            64,
            true,
            2.0,
        )
        .unwrap();
        assert_eq!(driver.total_turns(), 2);
        assert_eq!(driver.next_ready_time_ms(), Some(0.0));
        let root = driver.pop_ready(0.0, usize::MAX);
        assert_eq!(root.len(), 1);
        assert!(driver.pop_ready(500.0, usize::MAX).is_empty());
        driver.on_complete(root[0].request_uuid, 600.0).unwrap();
        // Live completion plus the scaled edge delay controls the second turn.
        assert_eq!(driver.next_ready_time_ms(), Some(975.0));
        assert!(driver.pop_ready(974.0, usize::MAX).is_empty());
        let next = driver.pop_ready(975.0, usize::MAX);
        assert_eq!(next.len(), 1);
        driver.on_complete(next[0].request_uuid, 980.0).unwrap();
        assert!(driver.is_drained());
    }
}

#[test]
fn exact_cut_preserves_active_history_and_original_identity_with_residual_timers() {
    let graph = boundary_graph();
    let play = prepare(&graph, 100.0);
    assert_eq!(play.evidence().graph, graph.identity());
    assert_eq!(play.evidence().t_star_ms, 100.0);
    assert!(request(&play, "root").historical);
    assert_eq!(request(&play, "root").recorded_end_ms, Some(120.0));
    assert!(request(&play, "background").historical);
    assert_eq!(request(&play, "background").recorded_end_ms, Some(580.0));
    assert!(!request(&play, "at-cut").historical);
    assert_eq!(request(&play, "at-cut").remaining_delay_ms, 30.0);
    assert!(request(&play, "at-cut").pending_dependencies.is_empty());
    assert_eq!(request(&play, "child-next").remaining_delay_ms, 40.0);
    assert_eq!(request(&play, "future").remaining_delay_ms, 60.0);
    assert_eq!(
        request(&play, "future").pending_dependencies,
        vec![play.identity("at-cut").unwrap().request_id]
    );
    let mut primers = play
        .evidence()
        .primers
        .iter()
        .map(|primer| primer.source_request_id.as_str())
        .collect::<Vec<_>>();
    primers.sort_unstable();
    assert_eq!(primers, ["child", "root"]);
    assert!(
        play.evidence()
            .primers
            .iter()
            .all(|primer| primer.input_length == 128)
    );
    let root_id = play.identity("root").unwrap().request_id;
    assert_eq!(
        play.identity("child").unwrap().parent_id.as_deref(),
        Some(root_id.as_str())
    );
    assert_eq!(
        play.identity("future").unwrap().root_id.as_deref(),
        Some(root_id.as_str())
    );
    let turn_zero = play.context().prepare_play(0, 0, Some(0.0)).unwrap();
    assert!(Arc::ptr_eq(play.context(), turn_zero.context()));
    for node in graph.nodes() {
        assert_eq!(
            play.identity(node.request_id()).unwrap(),
            turn_zero.identity(node.request_id()).unwrap()
        );
    }
}

#[test]
fn snapshot_driver_scales_timers_and_waits_for_actual_live_completion() {
    let graph = boundary_graph();
    let play = prepare(&graph, 100.0);
    let expected_evidence = play.evidence().clone();
    let mut driver = WorkloadDriver::new_agentic_snapshots(
        PreparedAgenticSnapshots::from_plays(vec![play.clone()]).unwrap(),
        48,
        true,
        2.0,
    )
    .unwrap();
    assert_eq!(driver.total_turns(), 3);
    assert_eq!(
        driver.agentic_snapshot_evidence(),
        Some(std::slice::from_ref(&expected_evidence))
    );
    assert_eq!(driver.next_ready_time_ms(), Some(15.0));
    assert!(driver.pop_ready(14.0, usize::MAX).is_empty());
    let mut ready = driver.pop_ready_compact(15.0, usize::MAX);
    assert_eq!(ready.len(), 1);
    let first = ready.pop().unwrap();
    assert_eq!(first.authored_request_id.as_deref(), Some("at-cut"));
    assert!(first.request.materialized_tokens().is_none());
    assert_eq!(
        first
            .request
            .metadata()
            .replay_context
            .as_ref()
            .unwrap()
            .agentic,
        Some(play.identity("at-cut").unwrap())
    );
    assert_eq!(
        first.replay_hashes,
        Some(ReplayRequestHashes::from_tokens(
            &play.materialize_prefix("at-cut", 128).unwrap(),
            48,
        ))
    );
    assert_eq!(driver.next_ready_time_ms(), Some(20.0));
    let child = driver.pop_ready(20.0, usize::MAX).pop().unwrap();
    assert_eq!(child.authored_request_id.as_deref(), Some("child-next"));
    driver.on_complete(child.request_uuid, 25.0).unwrap();
    // The recorded completion was 140 ms, but the live executor completes at
    // 35 ms: its 20 / 2 ms edge delay releases the dependent at 45 ms.
    driver.on_complete(first.request_uuid, 35.0).unwrap();
    assert_eq!(driver.next_ready_time_ms(), Some(45.0));
    assert!(driver.pop_ready(44.0, usize::MAX).is_empty());
    let future = driver.pop_ready(45.0, usize::MAX).pop().unwrap();
    assert_eq!(future.authored_request_id.as_deref(), Some("future"));
    driver.on_complete(future.request_uuid, 50.0).unwrap();
    assert!(driver.is_drained());
    let transcript = driver.agentic_lifecycle_transcript().unwrap();
    let dispatched = transcript
        .events
        .iter()
        .filter(|event| event.event == AgenticLifecycleEventKind::Dispatch)
        .collect::<Vec<_>>();
    assert_eq!(dispatched.len(), 3);
    assert_eq!(
        dispatched
            .iter()
            .map(|event| event.at_ms)
            .collect::<Vec<_>>(),
        [15.0, 20.0, 45.0]
    );
}

#[test]
fn primer_uses_chronological_predecessor_not_lexicographic_request_order() {
    let mut rows = Vec::new();
    for (index, start, previous) in [
        (0, 0.0, None),
        (2, 20.0, Some(0)),
        (10, 100.0, Some(2)),
        (11, 110.0, Some(10)),
    ] {
        let mut next = row(&format!("outer:{index}"), "main", start, Some(1.0));
        if let Some(previous) = previous {
            next.dependencies.push(edge(
                &format!("outer:{previous}"),
                AgenticDependencyRelation::Sequence,
                AgenticDependencyTrigger::Completion,
                0.0,
            ));
        }
        rows.push(next);
    }
    let graph = graph(rows);
    assert_eq!(
        graph
            .nodes()
            .iter()
            .map(AgenticNode::request_id)
            .collect::<Vec<_>>(),
        ["outer:0", "outer:10", "outer:11", "outer:2"]
    );
    let play = prepare(&graph, 105.0);
    assert_eq!(play.evidence().primers.len(), 1);
    assert_eq!(play.evidence().primers[0].source_request_id, "outer:10");
    assert_eq!(
        play.evidence().primers[0].request_id,
        play.identity("outer:10").unwrap().request_id
    );
}

#[test]
fn equal_timestamp_primer_follows_sequence_dependency_order() {
    let root = row("outer:0", "main", 0.0, Some(0.0));
    let mut predecessor = row("outer:2", "main", 10.0, Some(0.0));
    predecessor.dependencies.push(edge(
        "outer:0",
        AgenticDependencyRelation::Sequence,
        AgenticDependencyTrigger::Completion,
        0.0,
    ));
    let mut latest = row("outer:10", "main", 10.0, Some(0.0));
    latest.dependencies.push(edge(
        "outer:2",
        AgenticDependencyRelation::Sequence,
        AgenticDependencyTrigger::Completion,
        0.0,
    ));
    let mut future = row("outer:11", "main", 20.0, Some(0.0));
    future.dependencies.push(edge(
        "outer:10",
        AgenticDependencyRelation::Sequence,
        AgenticDependencyTrigger::Completion,
        0.0,
    ));
    let graph = graph(vec![root, predecessor, latest, future]);
    let play = prepare(&graph, 15.0);
    assert_eq!(play.evidence().primers.len(), 1);
    assert_eq!(play.evidence().primers[0].source_request_id, "outer:10");
}

fn cycle_graph() -> ValidatedAgenticGraph {
    let mut rows = Vec::new();
    for play in ["a", "b"] {
        for (index, start) in [(0, 0.0), (1, 100.0)] {
            let mut request = row(&format!("{play}{index}"), play, start, Some(1.0));
            request.play_id = play.into();
            request.input_length = Some(64);
            request.hash_ids = Some(vec![10 + index]);
            rows.push(request);
        }
    }
    graph(rows)
}

#[test]
fn seeded_snapshots_are_repeatable_and_cycle_lanes_without_completion_order() {
    let graph = cycle_graph();
    let first = graph
        .prepare_snapshots(
            3,
            AgenticSnapshotOptions {
                seed: 42,
                sampled: None,
            },
        )
        .unwrap();
    let second = graph
        .prepare_snapshots(
            3,
            AgenticSnapshotOptions {
                seed: 42,
                sampled: None,
            },
        )
        .unwrap();
    assert_eq!(
        serde_json::to_vec(first.snapshots()).unwrap(),
        serde_json::to_vec(second.snapshots()).unwrap()
    );
    assert_eq!(
        first
            .snapshots()
            .iter()
            .map(|snapshot| snapshot.source_play_id.as_str())
            .collect::<Vec<_>>(),
        ["a", "b", "a"]
    );
    assert!(
        first
            .snapshots()
            .iter()
            .all(|snapshot| (25.0..75.0).contains(&snapshot.t_star_ms))
    );
    let changed = graph
        .prepare_snapshots(
            3,
            AgenticSnapshotOptions {
                seed: 43,
                sampled: None,
            },
        )
        .unwrap();
    assert_ne!(
        first.snapshots()[0].t_star_ms,
        changed.snapshots()[0].t_star_ms
    );
    let context = first.context();
    let late_first = context.prepare_play(2, 1, Some(0.0)).unwrap();
    let early_second = context.prepare_play(0, 1, Some(0.0)).unwrap();
    assert_eq!(
        late_first.evidence(),
        context.prepare_play(2, 1, Some(0.0)).unwrap().evidence()
    );
    assert_eq!(early_second.evidence().source_play_id, "b");
    assert_eq!(late_first.evidence().source_play_id, "b");
    assert_eq!(early_second.materialize_prefix("b0", 1).unwrap(), [6]);
    assert_eq!(late_first.materialize_prefix("b0", 1).unwrap(), [10]);
    assert_ne!(
        late_first.identity("b0").unwrap().cache_id,
        early_second.identity("b0").unwrap().cache_id
    );
    assert_eq!(
        first
            .snapshots()
            .iter()
            .map(|snapshot| snapshot.t_star_ms.to_bits())
            .collect::<Vec<_>>(),
        [
            4631854218153394059,
            4634544125269611050,
            4632905092361953415
        ],
        "versioned seed sampling must retain its exact binary64 cuts"
    );
}

#[test]
fn single_start_and_missing_or_zero_api_duration_preserve_provenance() {
    let single = graph(vec![row("only", "main", 0.0, None)]);
    let prepared = single
        .prepare_snapshots(
            2,
            AgenticSnapshotOptions {
                seed: 3,
                sampled: None,
            },
        )
        .unwrap();
    for snapshot in prepared.snapshots() {
        assert_eq!(snapshot.t_star_ms, 0.0);
        assert!(!snapshot.requests[0].historical);
        assert_eq!(snapshot.requests[0].recorded_end_ms, None);
        assert!(snapshot.primers.is_empty());
    }
    for duration in [None, Some(0.0)] {
        let predecessor = row("before", "main", 0.0, duration);
        let mut after = row("after", "main", 100.0, None);
        after.dependencies.push(edge(
            "before",
            AgenticDependencyRelation::Sequence,
            AgenticDependencyTrigger::Completion,
            150.0,
        ));
        let graph = graph(vec![predecessor, after]);
        let play = prepare(&graph, 50.0);
        assert_eq!(request(&play, "before").recorded_end_ms, duration);
        assert_eq!(request(&play, "after").remaining_delay_ms, 100.0);
    }
}

#[test]
fn snapshots_retain_full_graph_outputs_and_logical_prompt_views() {
    let graph = boundary_graph();
    let prepared = graph
        .prepare_snapshots(
            1,
            AgenticSnapshotOptions {
                seed: 8,
                sampled: None,
            },
        )
        .unwrap();
    let zero = prepared.context().prepare_play(0, 0, Some(0.0)).unwrap();
    let cut = prepared.context().prepare_play(0, 0, Some(100.0)).unwrap();
    let full = drain(WorkloadDriver::new_agentic_trace(graph.clone(), 64).unwrap());
    let at_zero = drain(
        WorkloadDriver::new_agentic_snapshots(
            PreparedAgenticSnapshots::from_plays(vec![zero]).unwrap(),
            64,
            true,
            1.0,
        )
        .unwrap(),
    );
    let suffix = drain(
        WorkloadDriver::new_agentic_snapshots(
            PreparedAgenticSnapshots::from_plays(vec![cut.clone()]).unwrap(),
            64,
            true,
            1.0,
        )
        .unwrap(),
    );
    assert_eq!(at_zero.len(), full.len());
    for (id, ready) in &full {
        assert_eq!(ready.request.tokens, at_zero[id].request.tokens);
        assert_eq!(ready.replay_hashes, at_zero[id].replay_hashes);
        assert_eq!(
            ready.request.output_token_ids,
            at_zero[id].request.output_token_ids
        );
    }
    for (id, ready) in &suffix {
        assert_eq!(
            ready.request.output_token_ids,
            full[id].request.output_token_ids
        );
        assert_eq!(ready.request_uuid, at_zero[id].request_uuid);
        assert_eq!(
            ready.request.tokens,
            cut.materialize_prefix(id, 128).unwrap()
        );
    }
}

#[test]
fn original_weka_prefix_keeps_shared_units_and_private_partial_tail() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("trace.json");
    let requests = [(223, vec![10, 20, 30, 40]), (80, vec![10, 20]), (223, vec![10, 20, 30, 40])].into_iter().enumerate().map(|(index, (length, hashes))| {
        serde_json::json!({"t": index, "type":"s", "model":"model", "in":length, "out":1, "hash_ids":hashes})
    }).collect::<Vec<_>>();
    std::fs::write(&path, serde_json::to_vec(&serde_json::json!({"id":"play", "models":["model"], "block_size":64, "hash_id_scope":"local", "requests":requests})).unwrap()).unwrap();
    let graph = load_weka_agentic_graph(&path, Some(64)).unwrap();
    let id = |outer| {
        graph
            .nodes()
            .iter()
            .find(|node| node.request_id().ends_with(&format!(":outer:{outer}")))
            .unwrap()
            .request_id()
            .to_owned()
    };
    let original = id(0);
    let shortened = id(1);
    let follower = id(2);
    let play = prepare(&graph, 1000.0);
    let full = play.materialize_prefix(&original, 223).unwrap();
    let independently_short = play.materialize_prefix(&shortened, 80).unwrap();
    assert_eq!(&full[..64], &independently_short[..64]);
    assert_ne!(&full[64..80], &independently_short[64..]);
    let following = play.materialize_prefix(&follower, 223).unwrap();
    assert_eq!(&full[..192], &following[..192]);
    assert_ne!(&full[192..], &following[192..]);
    for length in [0, 1, 63, 64, 65, 80, 127, 128, 129, 192, 193, 207, 222, 223] {
        assert_eq!(
            play.materialize_prefix(&original, length).unwrap(),
            full[..length]
        );
    }
}

#[test]
fn encounter_allocated_prompt_preserves_full_width_source_ids_and_lane_offsets() {
    let mut authored = row("golden", "main", 0.0, None);
    authored.input_length = Some(257);
    authored.hash_ids = Some(vec![u64::MAX - 1, 0, 1_u64 << 32, u64::MAX, 7]);
    let graph = graph(vec![authored]);
    let prepared = graph
        .prepare_snapshots(
            2,
            AgenticSnapshotOptions {
                seed: 42,
                sampled: None,
            },
        )
        .unwrap();
    let play = prepared.context().prepare_play(0, 0, Some(0.0)).unwrap();
    let mut expected = [vec![0; 64], vec![1; 64], vec![2; 64], vec![3; 64]].concat();
    expected.push(4);
    assert_eq!(play.materialize_prefix("golden", 257).unwrap(), expected);
    let next = prepared.context().prepare_play(1, 0, Some(0.0)).unwrap();
    assert_eq!(
        next.materialize_prefix("golden", 257).unwrap(),
        expected.iter().map(|token| token + 5).collect::<Vec<_>>()
    );
}

#[test]
fn invalid_snapshot_inputs_and_exhausted_identity_ranges_fail_before_execution() {
    let graph = cycle_graph();
    assert!(
        graph
            .prepare_snapshots(
                0,
                AgenticSnapshotOptions {
                    seed: 1,
                    sampled: None
                }
            )
            .is_err()
    );
    let prepared = graph
        .prepare_snapshots(
            2,
            AgenticSnapshotOptions {
                seed: 1,
                sampled: None,
            },
        )
        .unwrap();
    let context = prepared.context();
    for cut in [-1.0, 101.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(context.prepare_play(0, 0, Some(cut)).is_err());
    }
    assert!(context.prepare_play(2, 0, None).is_err());
    assert!(context.prepare_play(0, u64::MAX, None).is_err());
    let last = context
        .prepare_play(1, (1_u64 << 30) - 1, Some(0.0))
        .unwrap();
    assert_eq!(last.materialize_prefix("b1", 1).unwrap(), [u32::MAX]);
    assert!(context.prepare_play(0, 1_u64 << 30, Some(0.0)).is_err());
    let play = context.prepare_play(0, 0, Some(0.0)).unwrap();
    assert!(play.materialize_prefix("missing", 0).is_err());
    assert!(play.materialize_prefix("b0", 1).is_err());
    assert!(play.materialize_prefix("a0", 65).is_err());
    assert!(PreparedAgenticSnapshots::from_plays(Vec::new()).is_err());
    assert!(PreparedAgenticSnapshots::from_plays(vec![play.clone(), play.clone()]).is_err());
    let independent = graph
        .prepare_snapshots(
            2,
            AgenticSnapshotOptions {
                seed: 1,
                sampled: None,
            },
        )
        .unwrap();
    let other_context = independent.context().prepare_play(1, 0, Some(0.0)).unwrap();
    assert!(PreparedAgenticSnapshots::from_plays(vec![play.clone(), other_context]).is_err());
    for speedup in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MIN_POSITIVE] {
        let cohort = PreparedAgenticSnapshots::from_plays(vec![play.clone()]).unwrap();
        assert!(WorkloadDriver::new_agentic_snapshots(cohort, 64, true, speedup).is_err());
    }
    let mut misaligned = play.clone();
    misaligned.evidence.requests.reverse();
    let cohort = PreparedAgenticSnapshots::from_plays(vec![misaligned]).unwrap();
    let error = WorkloadDriver::new_agentic_snapshots(cohort, 64, true, 1.0).unwrap_err();
    assert!(error.to_string().contains("does not match its source play"));
}

#[test]
fn overflowing_recorded_intervals_and_dependency_deadlines_are_rejected() {
    let interval = graph(vec![
        row("first", "main", 1.0e308, Some(1.0e308)),
        row("last", "main", 1.5e308, None),
    ]);
    assert!(
        interval
            .prepare_snapshots(
                1,
                AgenticSnapshotOptions {
                    seed: 1,
                    sampled: None
                }
            )
            .is_err()
    );
    let root = row("first", "main", 0.0, Some(1.0e308));
    let mut last = row("last", "main", 10.0, None);
    last.dependencies.push(edge(
        "first",
        AgenticDependencyRelation::Sequence,
        AgenticDependencyTrigger::Completion,
        1.0e308,
    ));
    let deadline = graph(vec![root, last]);
    assert!(
        deadline
            .prepare_snapshots(
                1,
                AgenticSnapshotOptions {
                    seed: 1,
                    sampled: None
                }
            )
            .is_err()
    );
}

#[test]
fn historical_request_cannot_depend_on_a_request_not_started_at_the_cut() {
    let source = row("later-source", "source", 100.0, Some(1.0));
    let mut historical = row("earlier-target", "target", 0.0, None);
    historical.dependencies.push(edge(
        "later-source",
        AgenticDependencyRelation::Spawn,
        AgenticDependencyTrigger::Dispatch,
        0.0,
    ));
    let graph = graph(vec![source, historical]);
    let error = graph
        .prepare_snapshots(
            1,
            AgenticSnapshotOptions {
                seed: 1,
                sampled: None,
            },
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("historical request's unresolved dependency")
    );
}

fn live_warmup_chain() -> WorkloadDriver {
    let rows = (0..24)
        .map(|index| {
            let mut next = row(
                &format!("turn-{index:02}"),
                "main",
                index as f64 * 200.0,
                Some(1.0),
            );
            if index > 0 {
                next.dependencies.push(edge(
                    &format!("turn-{:02}", index - 1),
                    AgenticDependencyRelation::Sequence,
                    AgenticDependencyTrigger::Completion,
                    100.0,
                ));
            }
            next
        })
        .collect();
    let graph = graph(rows);
    let play = prepare(&graph, 0.0);
    let mut driver = WorkloadDriver::new_agentic_warmup_with_policy(
        PreparedAgenticSnapshots::from_plays(vec![play]).unwrap(),
        64,
        true,
        1.0,
        AgenticWarmupPolicy::LiveTrajectories,
    )
    .unwrap();
    driver
        .enable_agentic_profile(AgenticProfileOptions::default())
        .unwrap();
    driver.finish_agentic_preparation(0.0).unwrap();
    driver
}

#[test]
fn live_warmup_advances_frontier_and_waits_for_final_server_cleanup() {
    let mut driver = live_warmup_chain();
    let mut last = None;
    for index in 0..10 {
        let ready = driver.pop_ready_compact(index as f64, 100);
        assert_eq!(ready.len(), 1);
        let turn = &ready[0];
        assert_eq!(
            turn.authored_request_id.as_deref(),
            Some(format!("turn-{index:02}").as_str())
        );
        assert_eq!(turn.request.metadata().max_output_tokens, 1);
        assert_eq!(
            turn.request
                .metadata()
                .output_token_ids
                .as_ref()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            turn.replay_hashes
                .as_ref()
                .unwrap()
                .local_block_hashes
                .len(),
            2
        );
        driver.on_output_token(turn.request_uuid, 0).unwrap();
        driver
            .on_causal_terminal(
                turn.request_uuid,
                index as f64 + 1.0,
                crate::replay::ReplayTerminalStatus::Completed,
            )
            .unwrap();
        if index < 9 {
            driver
                .on_quiescent(turn.request_uuid, index as f64 + 1.0)
                .unwrap();
        } else {
            last = Some(turn.request_uuid);
        }
    }
    assert!(driver.pop_ready_compact(10.0, 100).is_empty());
    assert_eq!(driver.finish_agentic_preparation(10.0).unwrap(), None);
    assert!(driver.is_agentic_preparing());
    assert_eq!(driver.agentic_profile_report().unwrap().issued_requests, 0);
    driver.on_quiescent(last.unwrap(), 30.0).unwrap();
    assert_eq!(
        driver.finish_agentic_preparation(30.0).unwrap(),
        Some(AgenticPreparationTransition::OpenProfile)
    );
    // Timestamped handoff subtracts the earliest surviving stream offset:
    // the first profile request starts at the barrier, after server cleanup.
    assert_eq!(driver.next_ready_time_ms(), Some(30.0));
    let profile = driver.pop_ready_compact(30.0, 100).pop().unwrap();
    assert_eq!(profile.authored_request_id.as_deref(), Some("turn-10"));
    assert_eq!(profile.request.metadata().max_output_tokens, 3);
    let report = driver.agentic_profile_report().unwrap();
    assert_eq!(report.issued_requests, 1);
    assert_eq!(report.successful_responses, 0);
    assert_eq!(
        driver.agentic_phase_evidence().unwrap().lanes[0].warmup_completed,
        10
    );
}

#[test]
fn live_warmup_cancellation_aborts_without_profiling() {
    let mut driver = live_warmup_chain();
    let turn = driver.pop_ready_compact(0.0, 100).pop().unwrap();
    driver
        .on_causal_terminal(
            turn.request_uuid,
            1.0,
            crate::replay::ReplayTerminalStatus::Canceled,
        )
        .unwrap();
    assert_eq!(driver.finish_agentic_preparation(1.0).unwrap(), None);
    assert!(driver.pop_ready_compact(1.0, 100).is_empty());
    driver.on_quiescent(turn.request_uuid, 2.0).unwrap();
    assert_eq!(
        driver.finish_agentic_preparation(2.0).unwrap(),
        Some(AgenticPreparationTransition::Aborted)
    );
    assert!(driver.is_drained());
    assert_eq!(
        driver.agentic_profile_report().unwrap().profile_start_ms,
        None
    );
}

#[test]
fn live_warmup_restores_global_stream_spread_after_concurrent_quota_drain() {
    let graph = graph(
        (0..12)
            .map(|index| {
                row(
                    &format!("stream-{index:02}"),
                    &format!("conversation-{index:02}"),
                    index as f64 * 10.0,
                    Some(1.0),
                )
            })
            .collect(),
    );
    let mut driver = WorkloadDriver::new_agentic_warmup_with_policy(
        PreparedAgenticSnapshots::from_plays(vec![prepare(&graph, 0.0)]).unwrap(),
        64,
        true,
        1.0,
        AgenticWarmupPolicy::LiveTrajectories,
    )
    .unwrap();
    driver
        .enable_agentic_profile(AgenticProfileOptions::default())
        .unwrap();
    driver.finish_agentic_preparation(0.0).unwrap();
    let issued = driver.pop_ready_compact(0.0, 100);
    assert_eq!(
        issued.len(),
        10,
        "quota counts admissions, including concurrent requests"
    );
    assert!(driver.pop_ready_compact(0.0, 100).is_empty());
    for request in &issued {
        driver.on_output_token(request.request_uuid, 0).unwrap();
        driver
            .on_causal_terminal(
                request.request_uuid,
                1.0,
                crate::replay::ReplayTerminalStatus::Completed,
            )
            .unwrap();
    }
    assert_eq!(driver.finish_agentic_preparation(1.0).unwrap(), None);
    for request in issued {
        driver.on_quiescent(request.request_uuid, 5.0).unwrap();
    }
    driver.finish_agentic_preparation(5.0).unwrap();
    let first = driver.pop_ready_compact(5.0, 100);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].authored_request_id.as_deref(), Some("stream-10"));
    assert_eq!(driver.next_ready_time_ms(), Some(15.0));
    let second = driver.pop_ready_compact(15.0, 100);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].authored_request_id.as_deref(), Some("stream-11"));
    assert!(
        first
            .iter()
            .chain(&second)
            .all(|request| request.request.metadata().max_output_tokens == 3)
    );
}

fn exact_row(id: &str, tokens: Vec<u32>, output: Vec<u32>) -> AgenticMooncakeRow {
    AgenticMooncakeRow {
        input_length: Some(tokens.len()),
        input_token_ids: Some(tokens),
        output_length: Some(output.len()),
        output_token_ids: Some(output),
        hash_ids: None,
        ..row(id, "conversation", 0.0, Some(1.0))
    }
}

#[test]
fn exact_agentic_tokens_preserve_closed_loop_dependencies_and_native_prefix_hashes() {
    let input: Vec<u32> = (100..229).collect();
    let output = vec![900, 901, 902];
    let root = exact_row("root", input.clone(), output.clone());
    let mut child_input = input.clone();
    child_input.extend_from_slice(&output);
    let mut child = exact_row("child", child_input.clone(), vec![]);
    child.dependencies = vec![edge(
        "root",
        AgenticDependencyRelation::Sequence,
        AgenticDependencyTrigger::Completion,
        7.0,
    )];
    let mut driver =
        WorkloadDriver::new_agentic_trace_with_options(graph(vec![root, child]), 4, true, None)
            .unwrap();
    let first = driver.pop_ready(0.0, 2).pop().unwrap();
    assert_eq!(first.request.tokens, input);
    assert_eq!(first.request.output_token_ids, Some(output));
    assert_eq!(
        first.replay_hashes,
        Some(ReplayRequestHashes::from_tokens(&input, 4))
    );
    assert!(
        driver.pop_ready(100.0, 2).is_empty(),
        "child requires modeled completion"
    );
    driver.on_complete(first.request_uuid, 10.0).unwrap();
    assert!(driver.pop_ready(16.0, 2).is_empty());
    let second = driver.pop_ready(17.0, 2).pop().unwrap();
    assert_eq!(second.request.tokens, child_input);
    assert_eq!(second.request.max_output_tokens, 0);
    assert_eq!(second.request.output_token_ids, Some(vec![]));
    let hashes = second.replay_hashes.unwrap();
    assert_eq!(
        &hashes.sequence_hashes[..32],
        &first.replay_hashes.unwrap().sequence_hashes[..]
    );
    driver.on_complete(second.request_uuid, 20.0).unwrap();
    assert!(driver.is_drained());
}

#[test]
fn exact_agentic_tokens_survive_snapshot_recycling_and_primer_warmup() {
    let tokens: Vec<u32> = (100..228).collect();
    let mut history = exact_row("history", tokens.clone(), vec![700, 701]);
    history.not_before_ms = 0.0;
    let mut future = exact_row("future", tokens.clone(), vec![800, 801]);
    future.not_before_ms = 100.0;
    let graph = graph(vec![history, future]);
    let context = graph
        .prepare_snapshots(
            1,
            AgenticSnapshotOptions {
                seed: 42,
                sampled: None,
            },
        )
        .unwrap()
        .context()
        .clone();
    for ordinal in [0, 1] {
        let play = context.prepare_play(0, ordinal, Some(50.0)).unwrap();
        assert_eq!(
            play.materialize_prefix("history", 65).unwrap(),
            tokens[..65]
        );
        let mut profile = WorkloadDriver::new_agentic_snapshots(
            PreparedAgenticSnapshots::from_plays(vec![play.clone()]).unwrap(),
            64,
            true,
            1.0,
        )
        .unwrap();
        let ready_at = profile.next_ready_time_ms().unwrap();
        let request = profile.pop_ready(ready_at, 1).pop().unwrap();
        assert_eq!(request.request.tokens, tokens);
        assert_eq!(request.request.output_token_ids, Some(vec![800, 801]));
        assert_eq!(
            request.replay_hashes,
            Some(ReplayRequestHashes::from_tokens(&tokens, 64))
        );

        let mut driver = WorkloadDriver::new_agentic_warmup(
            PreparedAgenticSnapshots::from_plays(vec![play]).unwrap(),
            64,
            true,
            1.0,
        )
        .unwrap();
        let primer = driver.pop_ready(0.0, 1).pop().unwrap();
        assert_eq!(primer.request.tokens, tokens);
        assert_eq!(primer.request.max_output_tokens, 1);
        assert_eq!(primer.request.output_token_ids, Some(vec![700]));
        assert_eq!(
            primer.replay_hashes,
            Some(ReplayRequestHashes::from_tokens(&tokens, 64))
        );
    }
}

#[test]
fn exact_agentic_tokens_are_validated_serialized_and_part_of_graph_identity() {
    let original = exact_row("root", vec![10, 20], vec![30]);
    let serialized = serde_json::to_string(&original).unwrap();
    let restored: AgenticMooncakeRow = serde_json::from_str(&serialized).unwrap();
    let first = graph(vec![restored]);
    let mut changed = original.clone();
    changed.input_token_ids = Some(vec![10, 21]);
    assert_ne!(
        first.identity().graph_digest,
        graph(vec![changed]).identity().graph_digest
    );
    let header: AgenticMooncakeHeader = serde_json::from_value(serde_json::json!({
        "schema": AGENTIC_MOONCAKE_SCHEMA, "version": AGENTIC_MOONCAKE_VERSION,
        "block_size": 64, "hash_id_scope": "local", "source": {"format": "test", "digest": "test"}
    }))
    .unwrap();
    for bad in [
        AgenticMooncakeRow {
            hash_ids: Some(vec![1]),
            ..original.clone()
        },
        AgenticMooncakeRow {
            input_length: Some(3),
            ..original.clone()
        },
        AgenticMooncakeRow {
            output_length: Some(2),
            ..original.clone()
        },
    ] {
        assert!(
            ValidatedAgenticGraph::from_agentic_mooncake_rows(header.clone(), vec![bad]).is_err()
        );
    }
    for invalid in [
        serde_json::json!([true]),
        serde_json::json!([-1]),
        serde_json::json!([4294967296_u64]),
    ] {
        let mut value = serde_json::to_value(&original).unwrap();
        value["input_token_ids"] = invalid;
        assert!(serde_json::from_value::<AgenticMooncakeRow>(value).is_err());
    }
    let hash_only = row("legacy", "conversation", 0.0, None);
    assert!(
        !serde_json::to_value(&hash_only)
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("input_token_ids")
    );
}

#[test]
fn exact_agentic_tokens_reject_mixed_synthetic_identity_to_avoid_false_cache_hits() {
    let header: AgenticMooncakeHeader = serde_json::from_value(serde_json::json!({
        "schema": AGENTIC_MOONCAKE_SCHEMA, "version": AGENTIC_MOONCAKE_VERSION,
        "block_size": 64, "hash_id_scope": "local", "source": {"format": "test", "digest": "test"}
    }))
    .unwrap();
    let exact = exact_row("exact", vec![0; 128], vec![1]);
    let synthetic = row("synthetic", "other", 0.0, None);
    for rows in [
        vec![exact.clone(), synthetic.clone()],
        vec![synthetic, exact],
    ] {
        let error =
            ValidatedAgenticGraph::from_agentic_mooncake_rows(header.clone(), rows).unwrap_err();
        assert!(error.to_string().contains("cannot mix"));
    }
}

fn sampled_options(
    graph: &ValidatedAgenticGraph,
    lanes: Vec<AgenticSampledLane>,
) -> AgenticSnapshotOptions {
    AgenticSnapshotOptions {
        seed: 42,
        sampled: Some(AgenticSampledSnapshot {
            graph: graph.identity(),
            lanes,
            provenance: BTreeMap::from([(
                "producer_revision".into(),
                "immutable-fixture-pin".into(),
            )]),
        }),
    }
}

fn sampled_lane() -> AgenticSampledLane {
    AgenticSampledLane {
        lane_id: 0,
        play_ordinal: 3,
        source_play_id: "play".into(),
        t_star_ms: 50.0,
        session_ids: BTreeMap::from([("conversation".into(), "producer-session".into())]),
    }
}

#[test]
fn external_sampled_cut_preserves_pins_identity_timers_and_real_warmup() {
    let graph = graph(vec![
        row("history", "conversation", 0.0, Some(10.0)),
        row("future", "conversation", 100.0, Some(10.0)),
    ]);
    let options = sampled_options(&graph, vec![sampled_lane()]);
    let encoded = serde_json::to_string(&options).unwrap();
    let decoded: AgenticSnapshotOptions = serde_json::from_str(&encoded).unwrap();
    assert_eq!(options, decoded);
    let prepared = graph.prepare_snapshots(1, decoded).unwrap();
    let evidence = &prepared.snapshots()[0];
    assert_eq!(evidence.t_star_ms, 50.0);
    assert_eq!(evidence.play_ordinal, 3);
    assert_eq!(evidence.graph, graph.identity());
    assert_eq!(
        evidence.sampled.as_ref().unwrap().session_ids["conversation"],
        "producer-session"
    );
    assert_eq!(
        evidence.sampling_provenance.as_ref().unwrap()["producer_revision"],
        "immutable-fixture-pin"
    );
    assert!(
        evidence
            .requests
            .iter()
            .all(|request| request.identity.conversation_id == "producer-session")
    );
    assert_eq!(evidence.primers[0].conversation_id, "producer-session");
    let mut profile =
        WorkloadDriver::new_agentic_snapshots(prepared.clone(), 64, true, 1.0).unwrap();
    assert!(profile.pop_ready(49.0, 1).is_empty());
    let request = profile.pop_ready(50.0, 1).pop().unwrap();
    assert_eq!(request.session_id, "producer-session");
    assert_eq!(request.authored_request_id.as_deref(), Some("future"));
    let mut warmup = WorkloadDriver::new_agentic_warmup(prepared, 64, true, 1.0).unwrap();
    let primer = warmup.pop_ready(0.0, 1).pop().unwrap();
    assert_eq!(primer.session_id, "producer-session");
    assert_eq!(primer.request.max_output_tokens, 1);
    assert!(warmup.agentic_phase_evidence().is_some());
}

#[test]
fn external_sampled_cut_rejects_wrong_graph_play_lane_and_identity() {
    let graph = graph(vec![
        row("history", "conversation", 0.0, Some(1.0)),
        row("future", "conversation", 100.0, Some(1.0)),
    ]);
    let mut bad_graph = sampled_options(&graph, vec![sampled_lane()]);
    bad_graph.sampled.as_mut().unwrap().graph.source.digest = "wrong-pin".into();
    assert!(
        graph
            .prepare_snapshots(1, bad_graph)
            .unwrap_err()
            .to_string()
            .contains("graph identity")
    );
    for (lane, expected) in [
        (
            AgenticSampledLane {
                source_play_id: "missing".into(),
                ..sampled_lane()
            },
            "source_play_id",
        ),
        (
            AgenticSampledLane {
                lane_id: 1,
                ..sampled_lane()
            },
            "lane_id",
        ),
        (
            AgenticSampledLane {
                t_star_ms: f64::NAN,
                ..sampled_lane()
            },
            "cut must lie",
        ),
        (
            AgenticSampledLane {
                t_star_ms: 101.0,
                ..sampled_lane()
            },
            "cut must lie",
        ),
        (
            AgenticSampledLane {
                session_ids: BTreeMap::from([("wrong-play-session".into(), "id".into())]),
                ..sampled_lane()
            },
            "different source play",
        ),
        (
            AgenticSampledLane {
                session_ids: BTreeMap::from([("conversation".into(), " ".into())]),
                ..sampled_lane()
            },
            "nonempty",
        ),
        (
            AgenticSampledLane {
                session_ids: BTreeMap::from([(
                    "conversation".into(),
                    "agentic-v1:collision".into(),
                )]),
                ..sampled_lane()
            },
            "reserved native identity",
        ),
    ] {
        let error = graph
            .prepare_snapshots(1, sampled_options(&graph, vec![lane]))
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    let duplicate = vec![sampled_lane(), sampled_lane()];
    assert!(
        graph
            .prepare_snapshots(2, sampled_options(&graph, duplicate))
            .unwrap_err()
            .to_string()
            .contains("duplicate lane")
    );
    let alias = vec![
        sampled_lane(),
        AgenticSampledLane {
            lane_id: 1,
            ..sampled_lane()
        },
    ];
    assert!(
        graph
            .prepare_snapshots(2, sampled_options(&graph, alias))
            .unwrap_err()
            .to_string()
            .contains("unique across lanes")
    );
}

#[test]
fn external_sampled_cut_selects_source_play_and_leaves_unmapped_ids_explicit() {
    let first = row("first", "other", 0.0, Some(1.0));
    let mut second = row("second", "future-session", 0.0, Some(1.0));
    second.play_id = "second-play".into();
    let graph = graph(vec![first, second]);
    let lane = AgenticSampledLane {
        source_play_id: "second-play".into(),
        t_star_ms: 0.0,
        session_ids: BTreeMap::new(),
        ..sampled_lane()
    };
    let prepared = graph
        .prepare_snapshots(1, sampled_options(&graph, vec![lane]))
        .unwrap();
    let evidence = &prepared.snapshots()[0];
    assert_eq!(evidence.source_play_id, "second-play");
    assert!(evidence.sampled.as_ref().unwrap().session_ids.is_empty());
    assert!(
        evidence.requests[0]
            .identity
            .conversation_id
            .contains(":conversation:future-session")
    );
    let legacy = graph
        .prepare_snapshots(
            1,
            AgenticSnapshotOptions {
                seed: 42,
                sampled: None,
            },
        )
        .unwrap();
    let serialized = serde_json::to_value(&legacy.snapshots()[0]).unwrap();
    assert!(serialized.get("sampled").is_none());
    assert!(serialized.get("sampling_provenance").is_none());
    assert_eq!(
        serde_json::to_value(AgenticSnapshotOptions {
            seed: 42,
            sampled: None
        })
        .unwrap(),
        serde_json::json!({"seed":42})
    );
}
