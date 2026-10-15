# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lower complete captured initial lanes into canonical AgenticMooncake.

Original adapter for AIPerf754356e9 metadata; see COLLECTION.md for API sources.
Only source-template clocks are used. Runtime arrivals never become release times.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from agentx_capture import PRODUCER, extract_request, jsonl, token_ids


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return value


def lower_dependencies(templates):
    """Translate authored sequence, branch and join gates; reject unsupported shapes."""
    gates = {(cid, i): [] for cid, meta in templates.items() for i in range(len(meta["turns"]))}

    def add(target, source, relation, delay=0, trigger="completion"):
        if target not in gates or source not in gates:
            raise ValueError("dependency references an absent or different source play")
        edge = {"source": source, "relation": relation, "trigger": trigger, "delay_ms": finite(delay, "delay")}
        if edge not in gates[target]:
            gates[target].append(edge)

    for cid, meta in templates.items():
        branches = {branch["branch_id"]: branch for branch in meta.get("branches", [])}
        if len(branches) != len(meta.get("branches", [])):
            raise ValueError("duplicate branch identity")
        declarations = {}
        for i, turn in enumerate(meta["turns"]):
            finite(turn["timestamp_ms"], "source timestamp")
            finite(turn["api_time_ms"], "source API duration")
            if i:
                # Weka emits explicit end-to-start delays; do not infer a new clock.
                add((cid, i), (cid, i - 1), "sequence", finite(turn["delay_ms"], "turn delay"))
            for predecessor in turn.get("replay_predecessors", []):
                add((cid, i), (predecessor["conversation_id"], predecessor["turn_index"]), "replay_barrier")
            for branch_id in turn.get("branch_ids", []):
                if branch_id in declarations or branch_id not in branches:
                    raise ValueError("duplicate or unknown branch declaration")
                declarations[branch_id] = i
        for branch_id, branch in branches.items():
            if branch.get("dispatch_timing", "post") != "post" or branch_id not in declarations:
                raise ValueError("pre-session or undeclared branch requires separate qualification")
            if meta.get("agent_depth", 0) != 0:
                raise ValueError("nested branch timing requires separate qualification")
            parent_index = declarations[branch_id]
            parent = meta["turns"][parent_index]
            children = branch["child_conversation_ids"]
            if not children or len(set(children)) != len(children) or any(child not in templates for child in children):
                raise ValueError("invalid branch children")
            mode = branch["mode"]
            if mode not in ("fork", "spawn"):
                raise ValueError("unsupported branch mode")
            start = branch.get("start_timestamp_ms")
            if start is None:
                start = min(templates[child]["turns"][0]["timestamp_ms"] for child in children)
            finite(start, "branch start")
            overlap = (
                mode == "spawn"
                and meta.get("replay_scope_id") is not None
                and parent["api_time_ms"] > 0
                and start < parent["timestamp_ms"] + parent["api_time_ms"]
            )
            for child in children:
                if templates[child].get("parent_conversation_id") != cid:
                    raise ValueError("branch parent identity mismatch")
                origin = parent["timestamp_ms"] if overlap else start
                delay = max(0, templates[child]["turns"][0]["timestamp_ms"] - origin) if mode == "spawn" else 0
                add((child, 0), (cid, parent_index), "spawn", delay, "dispatch" if overlap else "completion")
        for i, turn in enumerate(meta["turns"]):
            for prerequisite in turn.get("prerequisites", []):
                if prerequisite.get("kind") != "spawn_join" or any(
                    prerequisite.get(key) is not None
                    for key in ("child_conversation_ids", "barrier_id", "timer_seconds", "event_name")
                ):
                    raise ValueError("unsupported producer prerequisite")
                branch = branches.get(prerequisite.get("branch_id"))
                if branch is None or branch.get("is_background", False):
                    raise ValueError("join references unknown or background branch")
                for child in branch["child_conversation_ids"]:
                    # Delay is an independent parent-completion timer (sequence edge),
                    # not a second wait after the last child finishes.
                    add((cid, i), (child, len(templates[child]["turns"]) - 1), "join")
    return gates


def materialize_raw_record(raw, inputs, formatter_sha256, runtime_image_digest):
    """Bind offline native formatting to the unchanged normal wire payload."""
    request = extract_request(raw)
    if request["failed_or_cancelled"]:
        raise ValueError("failed/cancelled requests require explicit lifecycle lowering")
    inputs = token_ids(inputs)
    if request["input_token_ids"] is not None and request["input_token_ids"] != inputs:
        raise ValueError("offline native prompt differs from echoed native IDs")
    usage = request["usage"]
    if usage is None or usage.get("prompt_tokens") != len(inputs):
        raise ValueError("offline native prompt differs from actual server usage")
    output_length = usage.get("completion_tokens")
    if type(output_length) is not int or output_length < 0:
        raise ValueError("actual completion-token usage is required")
    marker = request["cache_bust_marker"]

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    if not marker or not any(marker in value for value in strings(raw.get("payload"))):
        raise ValueError("actual wire payload does not contain its recorded cache marker")
    return {
        **request,
        "input_token_ids": inputs,
        "input_identity": "native_formatter",
        "formatter_sha256": formatter_sha256,
        "runtime_image_digest": runtime_image_digest,
        "output_length": output_length,
        "output_length_basis": "observed_usage",
        "output_token_ids": request["output_token_ids"] if len(request["output_token_ids"]) == output_length else None,
    }


def lower(capture, materializations, provenance, model, block_size=64, max_input_tokens=2_000_000):
    """Materializations include every source turn; output IDs may remain unknown."""
    capture = Path(capture)
    events = [row for path in sorted(capture.glob("events-*.jsonl")) for _, row in jsonl(path)]
    if any(row.get("producer_revision") != PRODUCER for row in events):
        raise ValueError("capture producer revision mismatch")
    initial = [row for row in events if (row["phase"], row["event"]) == ("warmup", "setup")]
    if len(initial) != 1:
        raise ValueError("exactly one initial warmup setup is required")
    initial = initial[0]
    if provenance.get("producer_revision") != PRODUCER or any(
        not isinstance(key, str) or not isinstance(value, str) or not key.strip() or not value.strip()
        for key, value in provenance.items()
    ):
        raise ValueError("immutable string provenance is required")
    for field in ("corpus_sha256", "runtime_image_digest", "formatter_sha256", "encoder_sha256", "tokenizer_revision"):
        if not provenance.get(field):
            raise ValueError(f"immutable materialization pin is required: {field}")
    if type(max_input_tokens) is not int or max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")
    by_key, input_count = {}, 0
    for row in materializations:
        input_count += len(token_ids(row["input_token_ids"]))
        if input_count > max_input_tokens:
            raise ValueError("selected token budget exceeded; use a bounded cohort, not whole-corpus expansion")
        key = (row["lane_id"], row["conversation_id"], row["turn_index"])
        if key in by_key:
            raise ValueError("duplicate materialized lane/conversation/turn")
        by_key[key] = row
    rows, lanes, identities = [], [], []
    marker_ledger = {}
    for event in events:
        for session, marker in event.get("session_markers", {}).items():
            if session in marker_ledger and marker_ledger[session] != marker:
                raise ValueError("producer changed a continuing session marker")
            marker_ledger[session] = marker
    used, producer_sessions = set(), set()
    for lane, trajectory in enumerate(initial["trajectories"]):
        root = trajectory["conversation_id"]
        snapshot = trajectory.get("snapshot")
        if snapshot is None:
            raise ValueError("timestamped initial snapshot is required")
        relative = initial["templates"][root]
        path = (capture / relative).resolve()
        if not path.is_relative_to(capture.resolve()):
            raise ValueError("template path escapes capture")
        templates = json.loads(path.read_text())
        gates = lower_dependencies(templates)
        play = f"captured-lane:{lane}:{root}"
        request_id = lambda key: json.dumps([play, *key], separators=(",", ":"))
        session_id = lambda cid: json.dumps([play, cid], separators=(",", ":"))
        supplied_sessions = {state["conversation_id"]: state["x_correlation_id"] for state in snapshot["states"]}
        if len(supplied_sessions) != len(snapshot["states"]):
            raise ValueError("duplicate initial conversation state")
        roots = {
            state.get("root_correlation_id")
            or (state["x_correlation_id"] if state.get("agent_depth", 0) == 0 else state.get("parent_correlation_id"))
            for state in snapshot["states"]
        }
        if len(roots) != 1 or None in roots:
            raise ValueError("initial lane has ambiguous producer root identity")
        root_session = next(iter(roots))
        expected_marker = marker_ledger.get(root_session)
        mapping = {}
        for cid, i in gates:
            key = (lane, cid, i)
            if key not in by_key:
                raise ValueError(f"missing exact source-turn materialization: {key}; no prompt is synthesized")
            item = by_key[key]
            used.add(key)
            tokens = token_ids(item["input_token_ids"])
            if not tokens or item.get("input_identity") != "native_formatter":
                raise ValueError("nonempty native-formatted prompt identity is required")
            if not item.get("payload_sha256") or not item.get("formatter_sha256"):
                raise ValueError("payload and formatter content identities are required")
            for field in ("runtime_image_digest", "formatter_sha256", "encoder_sha256", "tokenizer_revision"):
                if item.get(field) != provenance[field]:
                    raise ValueError(f"materialized identity mismatch: {field}")
            producer = item["session_id"]
            if (item.get("root_session_id") or producer) != root_session:
                raise ValueError("materialization belongs to a recycled or different producer tree")
            if not isinstance(producer, str) or not producer or producer.startswith("agentic-v1:"):
                raise ValueError("invalid actual producer session identity")
            if cid in supplied_sessions and producer != supplied_sessions[cid]:
                raise ValueError("materialized session differs from sampled producer identity")
            if session_id(cid) in mapping and mapping[session_id(cid)] != producer:
                raise ValueError("source session has multiple producer incarnations")
            mapping[session_id(cid)] = producer
            marker = item.get("cache_bust_marker")
            if (
                not isinstance(marker, str)
                or not marker
                or marker != expected_marker
                or item.get("cache_bust_target") != "first_turn_prefix"
            ):
                raise ValueError("exact marker-bearing first_turn_prefix materialization is required")
            length = item["output_length"]
            if type(length) is not int or length < 0:
                raise ValueError("invalid output stop length")
            if item.get("output_length_basis") not in ("observed_usage", "prescribed_source"):
                raise ValueError("output length must be observed or source-prescribed")
            turn = templates[cid]["turns"][i]
            if any(
                item.get("source", {}).get(key) != value
                for key, value in turn.items()
                if key.startswith("source_") and value is not None
            ):
                raise ValueError("source request identity differs from captured template")
            row = {
                "request_id": request_id((cid, i)),
                "play_id": play,
                "source_play_ordinal": lane,
                "session_id": session_id(cid),
                "model": model,
                "input_length": len(tokens),
                "input_token_ids": tokens,
                "output_length": length,
                "not_before_ms": turn["timestamp_ms"],
                "recorded_api_time_ms": turn["api_time_ms"],
                "dependencies": [
                    dict(request_id=request_id(edge["source"]), **{k: v for k, v in edge.items() if k != "source"})
                    for edge in gates[cid, i]
                ],
            }
            if item.get("output_token_ids") is not None:
                output = token_ids(item["output_token_ids"])
                if len(output) != length:
                    raise ValueError("output token IDs do not match stop length")
                row["output_token_ids"] = output
            rows.append(row)
            identities.append(
                {key: value for key, value in item.items() if key not in ("input_token_ids", "output_token_ids")}
                | {"canonical_request_id": row["request_id"], "input_token_sha256": digest(tokens)}
            )
        if len(set(mapping.values())) != len(mapping) or producer_sessions.intersection(mapping.values()):
            raise ValueError("producer session IDs alias across source conversations or lanes")
        producer_sessions.update(mapping.values())
        starts = [row["not_before_ms"] for row in rows if row["play_id"] == play]
        cut = finite(snapshot["t_star_ms"], "cut")
        if not min(starts) <= cut <= max(starts):
            raise ValueError("sampled cut outside source graph clock")
        lanes.append(dict(lane_id=lane, play_ordinal=0, source_play_id=play, t_star_ms=cut, session_ids=mapping))
    if not rows or used != set(by_key):
        raise ValueError("empty selection or materializations outside initial selected lanes")
    source_digest = digest({"initial": initial, "rows": rows, "identities": identities, "provenance": provenance})
    header = dict(
        schema="dynamo.agentic_mooncake",
        version=2,
        block_size=block_size,
        hash_id_scope="local",
        source=dict(format="pinned-agentx-initial-lanes", digest=source_digest),
    )
    manifest = dict(
        schema="dspark-agentx-lowered-v1",
        source=header["source"],
        block_size=block_size,
        node_count=len(rows),
        play_count=len(lanes),
        source_models=[model],
        lanes=lanes,
        provenance=provenance,
        request_identities=identities,
        scoring_eligible=False,
        scope="complete source graphs of selected initial lanes only",
        input_tokens=input_count,
        max_input_tokens=max_input_tokens,
        limitations=[
            "Initial lanes only; no recycling contract.",
            "Native warmup frontier must match producer handoff.",
            "Unknown generated IDs use opaque request-scoped cache; cross-request output reuse is unverified.",
            "Idle caps, cancellation/stop cohorts and complete raw counts still require qualification.",
        ],
    )
    return header, rows, manifest


def bind_sampled(manifest, graph_identity, seed=42):
    for key in ("source", "block_size", "node_count", "play_count", "source_models"):
        if graph_identity.get(key) != manifest[key]:
            raise ValueError(f"canonical graph identity differs from import manifest: {key}")
    if not graph_identity.get("graph_digest"):
        raise ValueError("canonical graph digest is required; Python does not invent it")
    return dict(
        seed=seed, sampled=dict(graph=graph_identity, lanes=manifest["lanes"], provenance=manifest["provenance"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--materializations", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-identity", type=Path)
    parser.add_argument("--max-input-tokens", type=int, default=2_000_000)
    args = parser.parse_args()
    header, rows, manifest = lower(
        args.capture,
        (row for _, row in jsonl(args.materializations)),
        json.loads(args.provenance.read_text()),
        args.model,
        max_input_tokens=args.max_input_tokens,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "agentic.jsonl").open("w") as stream:
        for row in [header, *rows]:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.output / "import.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.graph_identity:
        options = bind_sampled(manifest, json.loads(args.graph_identity.read_text()))
        (args.output / "agentic_snapshot.json").write_text(json.dumps(options, indent=2) + "\n")


if __name__ == "__main__":
    main()
