<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISimulate Core

`aisimulate-core` is the single public Rust package for engine-neutral
inference simulation. It preserves three explicit implementation and API
layers:

- `aisimulate_core::engine` owns scheduling, native GPU KV accounting,
  preemption, timing, and attention-DP composition.
- `aisimulate_core::replay` owns deterministic virtual time, logical-worker
  lifecycle, placement and scaling composition, and replay reports.
- `aisimulate_core::perfmodel` owns the imported AIConfigurator latency,
  memory, and compiled-model implementation. Its `EngineConfig` and `engine`
  stay in this namespace so they cannot collide with the replay engine.

The crate root promotes the common `EngineConfig`, `ReplaySpec`, `Replayer`,
canonical `ReplayReport`, and timing-provider contracts. Dynamo-specific Router, Planner,
transport, and live-runtime adapters remain outside this crate.

## Generated concurrency requests

Programmatic replay callers can provide
`replay::loadgen::GeneratedRequests::new(request_count, factory)` through
`ReplayRuntimeInput::GeneratedRequests` instead of constructing every
`DirectRequest` up front. Set `ReplaySpec::max_in_flight` to the concurrency
limit. Replay calls the factory with the original zero-based request index
only when a slot becomes available, and applies the admission timestamp just
as it does for an eager request queue. The factory should generate one request
without retaining previous requests or capturing a prebuilt request list.

This keeps pending prompt storage independent of the total request count;
live requests, engine queues, and explicitly captured per-request output still
consume memory. The engine receives the same ordinary `DirectRequest` and
does not participate in source generation. Open-loop and existing trace inputs
retain their current behavior. Integrations such as Dynamo must opt into the
generated source after adopting a crate release containing this API.

## Output identity at the request boundary

`engine::Request` and Replay's materialized `DirectRequest` carry an
`output_identity` policy. The serialized default is omitted, preserving existing
request payloads.

| Field value | Contract |
| --- | --- |
| `synthetic` (default) | Missing `output_token_ids` use the existing deterministic generated tokens and cache hashes. Explicit token plans retain their exact identity. |
| `unknown` | Real output IDs are unavailable; an explicit output plan is rejected. Generated KV uses opaque request-scoped hashes, remains resident and evictable, and can be reused by that request after preemption. It cannot match another request's authored tokens. |

For `unknown`, each distinct execution must have a distinct `request_id`, even
when executions belong to the same producer session. Reusing a request ID means
reusing its opaque execution identity. AgenticMooncake automatically selects this
policy for exact-input rows without output IDs and allocates separate execution
UUIDs. P/D preserves the same policy across both roles. Cache events retain opaque
hashes but omit token IDs for blocks containing unknown outputs. Ordinary inline,
length-only and hash-only workload defaults remain unchanged.

## Source layout and upstream syncs

The former AIConfigurator Rust crate is kept as a stable mirror subtree under
`src/perfmodel/`. The crate root contains only the small compatibility aliases
needed by that imported source and the public namespace boundary. Keeping the
implementation together instead of interleaving it with replay makes future
AIConfigurator commits path-rewritable and reviewable while still producing a
single crates.io package.

The default feature set is Rust-only and does not depend on PyO3. The `python`
feature exposes registration hooks for the unified `aisimulate._runtime`
extension, and `embed-python` additionally enables PyO3 interpreter
initialization for standalone Rust applications that call into Python during
one-time model compilation.

## Scheduler fidelity

The replay engine models vLLM's attention-DP prefill cadence through
`EngineConfig::prefill_schedule_interval`. See the
[configuration and validation notes](../../docs/replay/engine/workers.md#vllm-prefill-schedule-interval).

SGLang uses `EngineConfig::prefill_decode_interval` to block prefill for N
scheduler rounds after a group-wide EXTEND. Its default is zero and its semantics
are separate from vLLM's cadence; see the
[SGLang configuration and scope](../../docs/replay/engine/workers.md#sglang-prefilldecode-interval).
