<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV cache

`kv_cache` configures each worker role's KV cache tiers:

| Tier | Where | Configure with |
| --- | --- | --- |
| G1 | GPU memory | `block_size`, `prefix_caching`, `bytes_per_token`, `capacity` |
| G2 | Host (CPU) memory | `host_offload` |
| G3 | Storage shared by workers | `g3_offload` |
| G2 peer fetch (KVCR) | Another rank's G2 | `kvcr` |
| KVCR-owned memory | Host memory KVCR owns, behind each rank's G2 | `kvcr.local_g2` |
| KVCR-owned disk | Local disk KVCR owns, behind each rank's KVCR-owned memory | `kvcr.g3` |

Recurrent-state models also use `state_cache`; see
[state cache](#state-cache).

Merge this fragment into an existing `engine` block from the [engine overview](README.md):

```yaml
engine:
  workers:
    aggregated:                  # or prefill / decode
      kv_cache:
        block_size: 64
        prefix_caching: true
        bytes_per_token: auto
        capacity: {type: default, memory_fraction: 0.9}
        host_offload:            # optional G2
          num_host_blocks: 4096
        g3_offload:              # optional G3: aggregated only, requires host_offload
          scope: cluster_shared
          num_g3_blocks: 16384
        # kvcr: {}               # optional KVCR peer fetch: not on decode, requires private host_offload, not with g3_offload
```

## KV cache fields

All paths below are under `engine.workers.<role>.kv_cache`.

| Knob | Default | Recommend | Rules |
|---|---|---|---|
| `block_size` | vLLM `64`, SGLang `1`, TensorRT-LLM `32` | Searchable; fixed by default | Tokens per block. vLLM and TensorRT-LLM require at least 2. |
| `prefix_caching` | `true` | fixed | Keep completed prompt blocks for reuse by later requests. A grouped FPM cache can turn it on only under the conditions of [grouped caches](#other-cache-modes). |
| `swa_prefix_tails` | `auto` | fixed | SGLang only. Sequence tails per attention-DP rank that keep their DeepSeek-V4 sliding-window KV; `null` turns it off. `auto` uses four per running request for a DeepSeek-V4 checkpoint and is off for other models. See [DeepSeek-V4 SWA prefix reuse](#deepseek-v4-swa-prefix-reuse). |
| `decode_radix_cache` | `false` | fixed | SGLang `decode` only. Enables the decode worker's radix cache (`--disaggregation-decode-enable-radix-cache`); requires `prefix_caching: true` and no speculative decoding. See [SGLang decode radix cache](#sglang-decode-radix-cache). |
| `bytes_per_token` | `auto` | fixed | KV bytes per token on one TP shard. `auto` derives it from the model, KV precision and this role's parallelism. |
| `capacity.type` | `default` | fixed | `default` sizes G1 from GPU memory; `fixed` uses `blocks` or `bytes`. |
| `capacity.memory_fraction` | vLLM/TensorRT-LLM `0.9`, SGLang `0.88` | Searchable; fixed by default | In `(0, 1]`. `default` capacity only. Same meaning as vLLM `gpu_memory_utilization`, SGLang `mem_fraction_static`, TensorRT-LLM `free_gpu_memory_fraction`. |
| `capacity.cuda_graph_reserved_bytes` | `0` | predict only | Bytes held back from G1 for CUDA graphs. `default` capacity only. |
| `capacity.blocks` | Unset | fixed | G1 blocks per attention-DP rank. `fixed` capacity only. |
| `capacity.bytes` | Unset | predict only | G1 bytes per attention-DP rank, instead of `blocks`. Requires explicit `block_size` and numeric `bytes_per_token`, unless `state_cache: {}` infers them. |
| `host_offload` | Unset (off) | fixed | G2 block; see [host offload](#host-offload-g2). |
| `hicache` | Unset (off) | Rejected | SGLang host KV cache for aggregated prediction; see [SGLang HiCache](#sglang-hicache). |
| `g3_offload` | Unset (off) | Rejected | G3 block; see [G3 offload](#g3-offload). |
| `kvcr` | Unset (off) | Rejected | KVCR block; see [KVCR peer fetch](#kvcr-peer-fetch). |
| `state_cache` | Unset (off) | Rejected | Recurrent-state block; see [state cache](#state-cache). |
| `prefix_match_unit` | Unset | Rejected | Prefix-matching granularity with `state_cache`. For a grouped FPM cache with prefix caching, it is the profile's `hash_block_size_tokens`: unset uses it, and a set value must equal it. See [grouped caches](#other-cache-modes). |

The block-size defaults above are simulation defaults. When a Sweeper candidate's
role block size is absent or `null`, the deployment generator omits `--block-size`
(vLLM), `--page-size` (SGLang), or `kv_cache_config.tokens_per_block`
(TensorRT-LLM), allowing the engine to choose its runtime default. Explicit values
are preserved. For SGLang and TensorRT-LLM, generating a fixed `capacity.blocks`
requires an explicit `block_size` matching the runtime: the generator must
convert the block count to a token capacity and cannot infer runtime geometry
from a simulation default. vLLM accepts the block count directly. Set
`block_size` explicitly when simulation and deployment must use the same value.

## GPU cache (G1)

Each attention-DP rank has its own G1 cache and its own running requests.
Tensor parallelism splits each token's KV across GPUs. `bytes_per_token` and
`capacity` are therefore per TP shard and per DP rank.

With `capacity.type: default`, the number of G1 blocks comes from the
performance model's memory estimate, which applies `memory_fraction` the way
each backend does. The backends differ in what they subtract before and after
the fraction; see [memory accounting](../../perf-model/memory.md#kv-cache-capacity-reservation). Use `fixed` capacity to pin an
exact block count, for example to compare cache sizes or to match a measured
deployment.

With `prefix_caching: true`, a new request reuses complete blocks whose tokens
match the start of its prompt and are still resident on the rank it is placed
on. Partial blocks are never reused. When G1 is full, each backend applies its
own policy: requests wait or are preempted, and unused cached blocks are
evicted least recently used first.

For SGLang, complete prompt pages enter the radix cache after prefill; decoded
pages enter when a request finishes or a running request is cancelled. In overlap
mode, prefill reuse becomes visible when the forward result is observed. A
completed disaggregated source publishes its prompt before the transfer hold;
the hold retains its path lock and partial tail. A transferred destination keeps
its prompt private until running admission, or caches its full pages when a
ready request is cancelled.

SGLang KV events follow those radix insertions, rather than page allocation.
Compatible consecutive stores and removals are batched, with stores retaining
their parent links and session IDs. Events flush after prefill, every 40 decode
forwards and on idle. Under memory pressure, decode retraction stably orders
requests by descending output length and then ascending prompt length, retracts
from the end, and evicts unused cache toward a 20-token reserve per survivor.
These contracts follow [SGLang v0.5.20](https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e).

### SGLang simulator storage

Waiting-queue admission reserves the request's host-side page-ID vector for its
bounded prompt-plus-output target, including a partial final page. Chunked
prefill also reserves spare capacity on a private radix leaf when it can extend
in place. The leaf releases unused capacity when it gains a child or its last
holder exits. Shared leaves, HiCache edges, and sliding-window cache boundaries
cannot extend in place and do not receive this tail reservation.

Disaggregated destination activation takes ownership of the transferred prompt's
page vector and retains amortized growth during decode. It does not run chunked
prefill or promise completion-sized page-list capacity. In ordinary decode the
radix cache receives the prompt at running admission and the output at terminal
cleanup, rather than extending a leaf after every token. These host allocation
choices do not reserve simulated GPU pages or change predicted serving times.

### Native SGLang Mamba checkpoints

The Rust `EngineConfig.sglang.mamba_cache` option models the non-speculative
SGLang `extra_buffer` checkpoint cache. It is an opt-in **native Replay**
configuration, available in a Rust `ReplaySpec.engine.rank` or the equivalent
`run_replay_json` payload. The Python `KvCachePredictionConfig`, prediction YAML,
compiler and deployment generator do not expose or forward this option. The
native-only scope is deliberate: state-slot capacity must come from the measured
deployment, and automatic Python capacity inference is not qualified here.
This option is separate from the Python `kv_cache.state_cache` configuration.

All paths in this table are relative to native `EngineConfig`:

| Field | Default | Rules |
| --- | --- | --- |
| `sglang.mamba_cache` | Absent (off) | Optional `SglangMambaCacheConfig`; native aggregated SGLang Replay only. |
| `sglang.mamba_cache.prefill_checkpoint_interval` | `64` | Positive tokens; must be a multiple of `block_size`. |
| `sglang.mamba_cache.decode_checkpoint_interval` | `256` | Positive tokens; must be a multiple of `prefill_checkpoint_interval`. |
| `sglang.mamba_cache.max_states` | Required | Effective native `max_mamba_cache_size` in state slots, at least `4`; includes live state, retained waiting copies and resident checkpoints. |

For example, merge this fragment into the native rank descriptor:

```json
{
  "backend": "sglang",
  "block_size": 64,
  "enable_prefix_caching": true,
  "sglang": {
    "chunked_prefill_size": 16384,
    "mamba_cache": {
      "prefill_checkpoint_interval": 64,
      "decode_checkpoint_interval": 256,
      "max_states": 994
    }
  }
}
```

`994` is an example measured state-pool size, not a default. The native chunk
size must be divisible by `block_size`. The supported scope requires
`worker_type: aggregated`, prefix caching enabled, attention DP 1, prefill
context parallelism 1 and pipeline parallelism 1. Speculative decoding, KV
events, HiCache, native host/G3 offload, SWA prefix tails and Belady eviction
are rejected. Tensor parallelism may shard the worker normally.

A token prefix is reusable only at a resident recurrent-state checkpoint.
Splitting a radix edge does not create a checkpoint. Prefill selects an aligned
branch checkpoint when available, otherwise the last complete checkpoint
interval in the chunk; a short final chunk does not create a new checkpoint.
Decode tracks the configured decode interval. With `host_loop: true`, prefill
publication waits until its producing forward is observed, including
one-output completion. A matched waiting request retains one private state
copy across admission retries; admission transfers that reservation into the
active request's three state slots, and cancellation releases it.

Checkpoint eviction and state-pool reclamation under pressure are not modeled.
Exceeding `max_states` or requiring KV eviction returns an explicit runtime
error rather than an optimistic hit or an idle pass with stranded work. Choose
capacity sufficient for the whole replay; increasing the token pool alone does
not increase `max_states`.

The [historical Qwen diagnostic receipt](../../../benchmarks/evidence/sglang-mamba-cache/README.md)
records input requirements, scoring commands and accuracy limits. It does not
qualify automatic memory sizing or current-head end-to-end serving accuracy.

### SGLang decode radix cache

An SGLang decode server runs a chunk cache unless it is launched with
`--disaggregation-decode-enable-radix-cache`, which is off by default, so an
SGLang `decode` worker caches nothing unless `decode_radix_cache: true` is set.
Without it, transferred prompts and decoded output stay private to their
request, the worker publishes no KV events, and every transfer moves the whole
prompt. With it, the worker reuses cached prefixes and publishes a transferred
prompt at running admission, as described above. Like SGLang, the simulator
rejects the decode radix cache with speculative decoding. An SGLang decode
worker always serves its waiting queue first-come first-served. This follows
[SGLang v0.5.22](https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d)
(`arg_groups/pd_disaggregation_hook.py`, `disaggregation/decode.py`).

For raw Sweeper `SearchSpace` inputs, `decode_radix_cache: true` requires
`decode_enable_prefix_caching: true` and no speculation when the domain includes
SGLang disaggregated candidates. Validation rejects invalid combinations before
enumeration; deployment lowering also checks direct sample inputs. Analytical
AFD and AFD+P/D do not model this cache and reject the setting. A sweep may include other backends,
whose samples omit the SGLang-only radix flag. The recommendation path carries
both controls from the worker's `kv_cache` configuration. A search with no
SGLang backend rejects `decode_radix_cache: true`.

To see what was reused, run with `--capture-per-request` and read these fields
in `requests.jsonl`:

| Field | Meaning |
| --- | --- |
| `first_admission_reused_input_tokens` | Prompt tokens reused from all tiers when the request was first admitted; later readmissions do not change it. |
| `reused_input_tokens` | Largest reuse across the request's prefill admissions, or across all admissions without a prefill stage. |
| `first_admission_g1_reused_input_tokens` | Prompt tokens served from G1 when the request was first admitted. |
| `first_admission_host_reused_input_tokens` | Prompt tokens restored from G2, including G3 restores and KVCR pulls. |
| `admission_history` | Every admission of the request, including after preemption, with the pool it reused from. |

The Dynamo KV Router's overlap score is a routing estimate; these fields record
what Replay actually reused. A readmission after preemption can reuse KV the
request itself computed, which raises `reused_input_tokens` but not
`first_admission_reused_input_tokens`; the summary's
`first_admission_prefix_cache_reused_ratio` aggregates the latter.

### SGLang request cache salt

Native `Request`, authored Rust `ReplaySpec.requests`, and configured inline
Python requests accept this per-request field:

| Field | Default | Rules |
| --- | --- | --- |
| `cache_salt` | Absent | Optional string. Empty is equivalent to absent. Nonempty salts require the native SGLang backend. |

Requests reuse a prefix only within the same salt namespace, even when their
token IDs match. The namespace also applies to LPM scores, waiting-queue
deduplication, P/D destination matching, and Belady input-demand forecasts.
Session IDs remain attribution and do not separate cache contents.
Built-in trace formats keep their existing unsalted schema; custom deferred
replay requests carry salt in their `DirectRequest` metadata.

Stored KV events carry `cache_salt`; removals reference the same salted hashes.
AISim retains its XXH3 block/sequence representation, with unchanged identities
for unsalted requests. It models the namespace behavior of
[SGLang v0.5.22](https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d),
without reproducing SGLang's SHA256 event encoding. Placement adapters hashing
a salted prompt should use `ReplayRequestHashes::from_tokens_with_cache_salt`.

This is a simulator contract pinned to SGLang v0.5.22. AISim does not check
the installed serving runtime's support for `cache_salt` or the decode radix
flag; verify both capabilities when deploying an older SGLang version.

<a id="deepseek-v4-swa-prefix-reuse"></a>

## DeepSeek-V4 SWA prefix reuse

DeepSeek-V4 layers keep a sliding window of KV next to their compressed KV.
SGLang serves these models with 256-token KV pages and frees a running
request's SWA pages as the request moves past them, so a cached sequence holds
SWA KV only on the pages just before the points where it was inserted into the
radix cache. A prefix hit must end on a page whose SWA KV is cached
(`create_match_validator` in SGLang v0.5.21
`mem_cache/unified_cache/components/swa.py`). A request that forks elsewhere
inside another cached sequence, such as a subagent's first turn, therefore
recomputes from the last such page even though full KV covers more.

`kv_cache.swa_prefix_tails` turns this on for SGLang `aggregated` and `prefill`
workers. Replay keeps the SWA pages SGLang leaves cached:

- Each prefill chunk end and the prompt end is an insert. An insert that ends
  on a page boundary keeps the two SWA pages before it; any other insert keeps
  the last whole page before it.
- A request whose hit was cut back inserts at its fork point once it has
  recomputed that far, in place of that chunk's end, so the next request
  forking there reuses it (`finalize_match_result_in_tree_core`).
- Replay's generated output tokens differ from the next prompt in a trace, so a
  continuation diverges where the previous output began instead of matching
  through it to the tail. A finished request therefore keeps its SWA page
  before the end of its prompt.

A hit refreshes the SWA page it ended on and the one before it. The SWA pool
holds one window and one page (384 tokens) per `swa_prefix_tails`, per
attention-DP rank; beyond that, the least recently used SWA page is dropped.
Evicting a sequence's tail also drops its SWA pages. With HiCache, an SWA page
stays valid while its node is host-resident, and the host SWA pool adds room:
it is sized from the device SWA pool by the ratio of `hicache.num_host_blocks`
to device KV blocks. Under `write_through` the host copy duplicates the device
page, so the larger of the two pools bounds the cached SWA pages; under
`write_back` the two pools add up. `write_through_selective` copies a node to
the host only after repeated hits, so the device pool alone bounds the cached
SWA pages.

`auto` sizes the pool as SGLang's `cap` mode does: four tails per running
request (`max_sequences`). It applies to a checkpoint whose bundled or local
config has architecture `DeepseekV4ForCausalLM` or `DeepseekV41ForCausalLM`.
Set a number to match a server that sized its SWA pool another way (for
`swa_full_tokens_ratio`, the pool's tokens beyond 384 per running request,
divided by 384), or `null` to match full KV at any page. These are modeling
assumptions: SWA pages are tracked on replay's page grid rounded up to 256
tokens, a chunk end's partial page is not tracked, and the pool shares no
capacity with full KV.
A decode worker's radix cache matches full KV as before. Native engine
configurations set the resolved count as `sglang.swa_prefix_tails` (a positive
integer, SGLang only; unset turns it off). It requires a `block_size` that
divides the 256-token SWA page.

<a id="host-offload-g2"></a>
<a id="native-vllm-host-offload-prediction"></a>

## SGLang HiCache

Set `engine.workers.aggregated.kv_cache.hicache` to model SGLang v0.5.22
caching on a private host pool for each attention-DP rank.
Prefix caching must be enabled. This is supported for aggregated prediction
and native offline replay; disaggregated roles, recommendation, Belady eviction,
external storage, and peer KVCR fetch are not supported by this option.

| Field under `hicache` | Default | Meaning |
|---|---|---|
| `num_host_blocks` | Required | Positive host capacity per attention-DP rank. |
| `write_policy` | `write_through` | `write_through`, `write_through_selective`, or `write_back`; see below. |
| `d2h_bandwidth_gbps` | `32` | Device-to-host bandwidth in decimal GB/s; zero is unlimited. |
| `h2d_bandwidth_gbps` | `32` | Host-to-device bandwidth in decimal GB/s; zero is unlimited. |
| `latency_to_first_byte_ms` | `0` | Non-negative latency per transfer operation. |

Host pages use the GPU page geometry: `block_size * bytes_per_token` bytes.
The compiler resolves `bytes_per_token: auto` from the model and rank layout.
Native engine configurations place these fields under `sglang.hicache` and
require `kv_cache_bytes_per_token`.

With `write_through`, completed chunk and request inserts back up their canonical
radix pages, including materialized output pages. `write_through_selective` waits
for a node's second insertion observation; each request counts newly covered
prefixes once, including reused pages. These policies back up ancestors first.
Each step submits one merged D2H operation;
its acknowledgement publishes host residency and releases the independent GPU
page locks. GPU eviction preserves backed-up radix nodes. Host eviction removes
only unlocked, host-only leaves in last-access order and may reclaim old entries
even when the new node ultimately cannot fit.

`write_back` instead backs up the selected GPU eviction leaf and waits for its
D2H acknowledgement before releasing the device pages. This wait delays batch
selection and forward execution. Host pressure first reclaims redundant copies
that remain on the GPU, sparing evictable GPU leaves on the first pass, then
reclaims host-only leaves. In-flight transfers and host source locks protect
their copies. If a backup cannot fit, eviction drops the selected device leaf
and its eligible host-only descendants together.

Admission selects the prefill shape before queuing H2D and budgets both the host
prefix and new GPU work. Load-back restores the entire promised host prefix or
recomputes admission without it; host hits below ten tokens are recomputed.
Ordinary forward duration includes the maximum of compute and dependent H2D,
without a per-layer pipeline model. With `host_loop`, this duration belongs to
the launched forward; its results are observed by a later loop iteration.
Admission reports device and host reuse separately. Host transfers continue to
drain and emit cache events after the last request completes or is cancelled.

These semantics follow public SGLang v0.5.22 at
[`baa9befa`](https://github.com/sgl-project/sglang/tree/baa9befa381511cd4d56f83980951e5a45fffc6d).
The separate `host_offload` option below models vLLM connectors.

## Host offload (G2)

G2 keeps copies of KV blocks in host memory. When a prefix has been evicted
from G1, a later request can restore it over PCIe instead of recomputing it.
It follows vLLM's native CPU offload.

```yaml
# host-offload-prediction.yaml
engine:
  mode: aggregated
  model: Qwen/Qwen3-32B-FP8
  hardware: h200_sxm
  backend: vllm
  backend_version: "0.24.0"
  context_length: 8192
  workers:
    aggregated:
      scheduler: {max_batched_tokens: 8192, max_sequences: 16}
      kv_cache:
        block_size: 16
        capacity: {type: fixed, blocks: 2499}
        host_offload:
          num_host_blocks: 4096
          d2h_bandwidth_gbps: 32.0
          h2d_bandwidth_gbps: 32.0

traffic:
  source: {type: synthetic-session, session: {turns: 4}}
  load: {type: concurrency, concurrency: 4}
  stop: {sessions: 16}
```

```bash
aisim predict --stack engine --config host-offload-prediction.yaml --capture-per-request
```

| Knob (under `host_offload`) | Default | Rules |
|---|---|---|
| `num_host_blocks` | Required | Positive. Per DP rank with `dp_rank_local`; total pool size with `cluster_shared`. |
| `scope` | `dp_rank_local` | `dp_rank_local` or `cluster_shared`; see [scope](#g2-scope). |
| `d2h_bandwidth_gbps` | `32.0` | GPU→host bandwidth per DP rank, decimal GB/s. `0` means unlimited. |
| `h2d_bandwidth_gbps` | `32.0` | Host→GPU bandwidth per DP rank, decimal GB/s. `0` means unlimited. |
| `latency_to_first_byte_ms` | `0.0` | Fixed delay before each transfer starts moving bytes. |
| `shared_d2h_bandwidth_gbps` | `80.0` | Total GPU→host bandwidth of the pool. `cluster_shared` only. |
| `shared_h2d_bandwidth_gbps` | `80.0` | Total host→GPU bandwidth of the pool. `cluster_shared` only. |
| `connector` | `offloading_v0_24` | vLLM connector whose rules G2 follows: `offloading_v0_24`, `offloading_v0_32` or `simple_cpu_v0_32`. The last two need the aggregated worker, `dp_rank_local` and no `g3_offload` or `kvcr`, except that a sliding-window [state cache](#state-cache) takes `offloading_v0_32` on prefill and decode workers too; see [connector profiles](#g2-connector-profiles). |

`recommend` keeps `host_offload` fixed; its size and bandwidth are not searched.

### How G2 behaves

- A block's size is `block_size × bytes_per_token`, the same per-TP-shard bytes
  as G1. No TP multiplier is applied to host capacity or transfer time, so an
  explicit `bytes_per_token` must use the same unit as the bandwidths.
- Blocks are copied to G2 after they are computed in G1. A request that finds its
  prefix in G2 waits for the host→GPU copy before it is admitted, so restore
  time adds to TTFT.
- G1 blocks that are still being copied to G2 are not overwritten.
- A decode rank with G2 admits requests whose P/D handoff has arrived before
  other waiting requests, as vLLM's FCFS scheduler does.

<a id="g2-connector-profiles"></a>

### Connector profiles

`connector` picks the vLLM CPU-offload connector whose store, lookup and
eviction rules G2 follows:

| | `offloading_v0_24` (default) | `offloading_v0_32` | `simple_cpu_v0_32` |
| --- | --- | --- | --- |
| vLLM connector | `OffloadingConnector`, vLLM 0.24 | `OffloadingConnector` with the CPU backend, vLLM v0.32.0 | `SimpleCPUOffloadConnector` in eager mode, vLLM v0.32.0 |
| Stored blocks | Prompt blocks as they are computed, after any blocks loaded from G2 | The same; a finishing request also offers prompt blocks it could not store earlier | Every block one pass after it is computed, decode blocks included, scanned from block 0 at each admission; blocks already in G2, such as those just loaded, are skipped. A finishing request stores the rest. |
| Eviction order | Least recently used. Each lookup and store makes the prompt's resident blocks the newest, its first block newest of all. | Least recently used by rank. A block is ranked when its store starts; a finishing request re-ranks every block it stored, offered or loaded, its first block newest. | First in, first out by store completion. A completed load or a dropped lookup pin makes its blocks the newest; a block already in G2 is not refreshed. |
| A hit whose store is in flight | Defers the request | Defers the request | Misses |
| A hit another request is loading | Defers the request | Defers the request | Both requests load it |
| The request's own store in flight | Defers its lookup | Defers its lookup | Does not defer |
| A hit before its load | Not pinned | Not pinned | Pinned until the request loads it or looks it up again |
| G1 blocks being copied | Freed with their request; a new owner waits for the copy | The same | Held until the copy completes, then freed first block first |
| A preemption | Waits for the preempted request's stores | The same | Waits for every transfer in flight |
| A waiting request that holds G1 for its load | Keeps its place among the requests that have waited on G2, unless the rank has G3 or has pulled over KVCR | Once its load lands, is admitted before every waiting request that holds none | The same |

On a decode rank with G2, or while a destination reservation is pending, an
activated handoff is admitted before a fresh G2 restore, including one whose
load has landed. A pending reservation blocks fresh admission but not a
handoff whose KV has already been activated.

A rank-local pool that keeps the checkpoints of a sliding-window
[state cache](#state-cache) follows `offloading_v0_32` under either offloading
name, as vLLM has sliding-window groups from v0.32.0 on.

vLLM main has changed some of these rules since v0.32.0. Each profile models the
rules of the version it names; none was compared with a running vLLM. The model
also differs from vLLM v0.32.0 here:

- `offloading_v0_32` offers and re-ranks a finishing request's blocks when the
  request finishes. vLLM does both in its next step, after that step's other
  stores.
- After a preemption, `simple_cpu_v0_32` loads only prompt blocks from G2 and
  recomputes the request's output blocks, which vLLM reloads.

<a id="hybrid-g2"></a>

### Hybrid models (state cache)

With [`state_cache`](#state-cache), the two v0.32.0 profiles also keep recurrent
states, so a follow-up whose state left G1 can resume from host memory. This
needs `prefix_match_unit` equal to `block_size`.

- A state goes to G2 when G1 publishes it, before the pass's blocks, as one key
  per state block, for example one per Mamba group. `offloading_v0_32` stores
  only states inside the prompt, each key on its own. `simple_cpu_v0_32` stores
  a state only when all its keys fit, and otherwise drops it without a retry;
  vLLM stores each group's block while room remains.
- A lookup resumes at the deepest block boundary within the prompt, at most
  `(len - 1) // block_size` blocks, where G1 and G2 hold the attention prefix
  and G2 holds every key of the state there. A state missing one key is not a
  hit. The load restores both, and the restored state is the first pass's
  input.
- `offloading_v0_32` re-ranks the states a finishing request stored, offered or
  loaded together with its blocks.
- G1 keeps the blocks a hybrid G2 is copying until the copy completes, as
  SimpleCPU does. For `offloading_v0_32` this is a simplification: vLLM fences
  the copied blocks' reuse instead, and replay does not fence state blocks.
  Replay then frees them tail first, as vLLM frees a finished request's blocks.

<a id="g2-scope"></a>
<a id="g2-ownership"></a>

### Scope

| `scope` | Pools | `num_host_blocks` means | Bandwidth |
| --- | --- | --- | --- |
| `dp_rank_local` | One per DP rank, as in vLLM | Per rank | Each rank's own D2H and H2D limits |
| `cluster_shared` | One for the whole deployment | Total for the pool | Rank limits, also capped by the pool's shared limits |

For example, two replicas with `attention_data: 2` and `num_host_blocks: 4096`
get four private 4096-block caches with `dp_rank_local`, or one 4096-block
cache with `cluster_shared`. In a shared pool, transfers in the same direction
share bandwidth max-min fairly per transfer, not per rank: four ranks at
32 GB/s, each running one transfer under an 80 GB/s pool, each move at 20 GB/s.
Pool bandwidth a rank cannot use because its own limit binds goes to the other
ranks: with one rank running nine transfers and another running one, both at
32 GB/s, each rank moves 32 GB/s in total. Under a 20 GB/s pool the same ten
transfers each get 2 GB/s, so the busier rank moves 18 GB/s.

`cluster_shared` models a deployment-wide host cache such as Mooncake Store or
LMCache. vLLM's native offload has no equivalent, so it has no native parity
reference. All roles in one shared pool must use the same model, backend,
parallel shape, KV precision and block geometry, plus the same pool capacity
and shared bandwidths; otherwise the run is rejected. In `recommend` with
`cluster_shared` on both P/D roles, pin each role's `tensor` and `pipeline` with
`parallelism.preset: false`.

### Supported

G2 supports vLLM aggregated and disaggregated workers with prefix
caching and any `attention_data`. It also keeps the states of a
[state cache](#state-cache): a recurrent state on a v0.32.0 profile, see
[hybrid models](#hybrid-g2), and a sliding-window state under either offloading
connector name. G2
is rejected with SGLang, TensorRT-LLM, ngram speculation, Belady eviction
and a [grouped FPM cache](#other-cache-modes).

Canonical `engine.speculation.kind: mtp` can combine with G2 on aggregated and
disaggregated vLLM workers, with any `attention_data`, prefix caching and no
G3. An aggregated worker needs `scope: dp_rank_local`; prefill and decode
workers take either scope. Use explicit fixed KV capacity and default timing.
Scheduling is synchronous. Separate draft models and grouped caches are
unsupported. See
[speculation](speculation.md#registered-schemes) for the complete composition.
A hybrid G2 takes MTP only with [`disable_eagle_block_drop`](speculation.md#prefix-caching):
replay does not model the EAGLE block drop hiding the last block, and the state,
that a G2 load restored.

An aggregated worker with a uniform token cache models single-module MTP with
one shared FullAttention cache. This combined behavior is adapted and modified
from the
[vLLM v0.32.0 scheduler](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py)
and [OffloadingConnector](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py),
Copyright contributors to the vLLM project, Apache-2.0. Prompt stores exclude
unverified drafts, and G1 copy sources stay protected until their transfer
completes. A next step that can never fit physical KV capacity or the total
padded verification budget is rejected instead of retried indefinitely; this
is a simulation capacity guard. The scheduling baseline does not change the
version or provenance of the performance data used for timing.

Prefill and decode workers, and workers with a sliding-window state cache, keep
the speculative lifecycle they have without G2. G2 stores complete prompt
blocks only, so drafts never reach it. With a uniform token cache a restore
resumes at the loaded prefix, as on an aggregated worker. With a sliding-window
state cache it loads and resumes one block short of the G2 hit, as vLLM's
drafter KV group loads a connector hit without its last block; with
[`disable_eagle_block_drop`](speculation.md#prefix-caching) the hit follows the
non-speculative rules. Accepted drafts that take a G1 block a store still
reads make the forward that writes it wait for that copy, and so does
preempting a request whose store is in flight; a store prepared in that same
pass delays the rank's next pass. A preempted request resumes once G1 has room
for its tokens, its state and its next token; a shared pool, and a rank that
offloads state checkpoints, keep the G1 blocks they copy until each copy
completes, so their stores can delay that. The capacity guard above does not
cover these workers: when G1 holds fewer than about two requests, a replay can
end with a stall error or not terminate, with or without G2. Size G1 for
several concurrent requests.

For shared pools, the report's `g2_domains` gives capacity and resident blocks.
Workers that publish KV events report G2 changes with `"tier": "host_pinned"`;
see [KV events](../../adapters/native-composition.md#kv-events).

<a id="g3-offload"></a>
<a id="optional-g3-offload"></a>

## G3 offload

G3 adds a storage tier behind G2. It requires `host_offload`. Add it next to
`host_offload` in the example above:

```yaml
g3_offload:
  scope: cluster_shared
  num_g3_blocks: 8192
```

| Knob (under `g3_offload`) | Default | Rules |
|---|---|---|
| `scope` | Required | `worker_local` (one store per worker) or `cluster_shared` (one store for all workers). |
| `num_g3_blocks` | Required | Positive. Per worker with `worker_local`; total with `cluster_shared`. |
| `latency_to_first_byte_ms` | `0.1` | Fixed delay before each transfer moves bytes. |
| `read_bandwidth_gbps`, `write_bandwidth_gbps` | `10.0` | Per worker, decimal GB/s. `0` means unlimited. |
| `shared_read_bandwidth_gbps`, `shared_write_bandwidth_gbps` | `80.0` | Total for the store. `cluster_shared` only. |

These defaults are modeling assumptions, not measured values.

- Blocks written to G2 are also written to G3 in the background.
- A read restores a prefix from G3 into G2 and then into G1. G3 alone does not
  make blocks ready on the GPU.
- With `cluster_shared`, a block is stored once and any worker can read it,
  including blocks written by a worker that has since been removed. With
  `worker_local`, workers cannot read each other's blocks.
- G3 evicts least recently used blocks that are not being transferred.
- With `worker_local`, total bandwidth grows with the worker count (16 workers
  at 10 GB/s reach 160 GB/s). `cluster_shared` caps it at the shared limits and
  divides them max-min fairly, as in a [shared G2 pool](#g2-scope).

To compare scopes at equal total capacity with `N` workers, use
`num_g3_blocks: C` for `worker_local` and `N × C` for `cluster_shared`. Set the
shared bandwidths to `0` to isolate reuse from bandwidth contention.

The prediction summary includes a `g3_offload` section with lookup hits,
read/write jobs and bytes, evictions, resident blocks and
`cross_worker_read_blocks` (blocks read by a worker other than the writer).

Supported: aggregated vLLM with `attention_data: 1`, prefix caching, and either
G2 scope; fixed or Planner-scaled workers. Rejected: `recommend`, P/D,
native MTP, and a `host_offload.connector` other than `offloading_v0_24`.

When a request's prefix is larger than G2 can hold, restores can keep evicting
each other. If the rank makes no progress for 1,024 consecutive restore rounds,
the request recomputes the missing part instead. Each time this happens
`g3_offload.bypassed_restores` increments. vLLM has no such rule.

<a id="kvcr-peer-fetch"></a>

## KVCR peer fetch

KVCR (KV Cache Runner) lets a rank pull a prompt prefix it is missing from
another rank's G2 over the network, instead of recomputing it. A router attaches
a hint naming the source rank; without hints KVCR pulls nothing. With
`local_g2` it also keeps blocks in memory of its own on each rank and restores
them without a hint. It requires
private G2 (`host_offload` with the default `scope: dp_rank_local`). Add it next
to `host_offload` in the [host offload example](#host-offload-g2):

```yaml
kvcr:
  latency_to_first_byte_ms: 1.0
  bandwidth_gbps: 32.0
  # local_g2: {num_blocks: 4096}   # optional KVCR-owned memory per rank
  # g3: {num_blocks: 65536}        # optional KVCR-owned disk per rank; requires local_g2
```

| Knob (under `kvcr`) | Default | Rules |
|---|---|---|
| `latency_to_first_byte_ms` | `1.0` | Fixed delay before each pull moves bytes, for the control round trip and the source pin. |
| `bandwidth_gbps` | `32.0` | Per source rank (one attention-DP rank of a worker), decimal GB/s (a 400 Gb/s NIC is `50`). All GPUs of the rank share it, and so do concurrent pulls from it, equally. `0` means unlimited. |
| `local_g2` | Unset (off) | Host memory KVCR owns on each rank; see [KVCR-owned memory](#kvcr-owned-memory). |
| `local_g2.num_blocks` | Required | Capacity of each rank, in the rank's G2 blocks (vLLM `secondary_g2_slots`). Integer from 1 through 2^64 − 1; omit `local_g2` to turn it off. |
| `local_g2.policy` | `lru` | `lru` or `fifo`, the block to drop when the memory is full. |
| `local_g2.copy_bandwidth_gbps` | `6.0` | Rate of the copies between a rank's G2 and its KVCR-owned memory, decimal GB/s. `0` means the copies take no time. |
| `g3` | Unset (off) | Local disk KVCR owns on each rank; requires `local_g2`. See [KVCR-owned disk](#kvcr-owned-disk). |
| `g3.num_blocks` | Required | Capacity of each rank, in the rank's G2 blocks. Integer from 1 through 2^64 − 1; omit `g3` to turn it off. |
| `g3.latency_to_first_byte_ms` | `3.0` | Fixed delay of each read and each write. |
| `g3.read_bandwidth_gbps` | `5.0` | Rate of reads from disk into the rank's G2, decimal GB/s. `0` means the bytes take no time. |
| `g3.write_bandwidth_gbps` | `0.8` | Rate of writes from KVCR-owned memory to disk, decimal GB/s. `0` means the bytes take no time. |
| `g3.spill_timeout_ms` | `1000.0` | How long a block of a completed G2 store waits for room in KVCR-owned memory (KVCR `operation_timeout_ms`). Must be positive. |

`kvcr: {}` uses the defaults. The default `bandwidth_gbps` approximates pulls
over RDMA between B200 nodes; set the rate of your transport.
`latency_to_first_byte_ms` is a modeling assumption.

### How KVCR behaves

- When some of a hinted request's prompt blocks are in neither G1 nor the
  rank's own G2, the rank checks the source's G2 block by block, in prompt
  order, and pulls each missing block the source holds. It stops at the first
  block the source lacks.
- A pull starts when the source finishes its current forward pass, including
  a pass that starts at the same instant the pull is requested, waits
  `latency_to_first_byte_ms`, then moves bytes. Each pulled block carries the
  KV of every GPU of the source rank: G2 block bytes times the role's
  `parallelism.tensor` (native `tensor_parallel_size`). The ranks of one worker run
  their passes in lockstep, so a pull from another DP rank of the same worker
  starts when the worker's current pass ends. Pulled blocks land in the rank's
  own G2 and load to G1 like any G2 hit, so pull and load time add to TTFT.
- Requests on one rank that need the same blocks share one pull. A request that
  ends while its pull is in flight leaves the pulled blocks in G2.

The model simplifies KVCR:

- The rank sees the source's G2 contents when it looks up the prefix. The
  source's blocks are then protected from eviction until the rank takes the
  completed peer transfer: as soon as it lands if the rank's worker is idle,
  otherwise when that worker's current pass ends. For a standalone pull, the
  pulled blocks become usable in the rank's G2 at the same point. A pull joined
  with a local promotion can still be waiting for the local copy after this
  source protection ends.
- The rank reserves G2 slots only for the blocks the source holds, all in one
  reservation per pull. A pull never evicts from the rank's G2 a block of the
  hinted prompt that the rank's G1 does not hold; it stops at the first block
  it has no other room for. G2 copies of prompt blocks already in G1 stay
  evictable, since the request does not load them.
- While its pull is in flight a request is not looked up again, as vLLM skips
  a request that waits for remote KV.
- Pulls do not fail or time out, and the first pull between two ranks has no
  connection setup cost.
- A pull moves one block share per tensor-parallel GPU. KV that each GPU
  replicates (an MLA latent, or a KV head shared by several GPUs when
  `num_kv_heads` is smaller than `tensor`) is counted once per GPU.

When a request's prefix is larger than G2 can hold, pulls can keep evicting
each other. If the rank makes no progress for 1,024 consecutive pull rounds,
the request recomputes the missing part instead. Each time this happens
`kvcr.bypassed` increments. vLLM has no such rule.

The prediction summary includes a `kvcr` section: `pulls` by result (`success`,
or `partial` when the source held only part of the missing blocks or the rank
had no free G2 slot for the rest, including a block the same request already
pulled at that instant), pulled `blocks` and `bytes` (summed over
the GPUs of each source rank),
`remote_deliver_ms` (total time from submit to landing in G2), `source_wait_ms`
(total time spent waiting for the source's pass to end) and `loaded_blocks`
(pulled blocks, counted when their first load into G1 starts). `bypassed`
appears only when it is nonzero. The counters update as the pool's ranks run.
A run stopped by `traffic.stop.max_virtual_time_seconds` (native
`max_sim_time_ms`) can miss pulls that landed during a pass still running at
the cutoff; a run that completes counts every pull.

Ranks without incoming pulls skip shared KVCR deadline lookups. Other KVCR
bookkeeping still queries shared pool state from per-rank calls. Large worker
or attention-DP pools can substantially increase replay wall time. Further
overhead reductions are deferred; they must preserve CanonicalV1 replay output bytes.

<a id="kvcr-owned-memory"></a>

### KVCR-owned memory

With `kvcr.local_g2`, each rank has a second host cache that KVCR owns, behind
the rank's G2. Only G2 exchanges blocks with the GPU, so the two caches work
as a chain:

- **Store.** A completed G2 store's blocks are copied into KVCR-owned memory
  and are readable there once the copy ends. A block already there is
  skipped, and one that finds the memory full with nothing to drop is not
  copied.
- **Evict and order.** The two caches evict independently. `lru` drops the
  block used least recently: a request's stores, promotion and finish and a
  peer's read count as use, lookups do not. `fifo` drops the block stored
  longest ago; a promotion queues its blocks again. A block being copied or
  read by a peer is never dropped.
- **Promote.** A lookup that misses G2 checks KVCR-owned memory next, before
  the hinted peer and without a hint. A block found there is copied back into
  G2, which can evict another G2 block, and loads to G1 like any G2 hit; copy
  and load both add to TTFT.
- **Peer pulls.** A source serves a pull from both caches. If its KVCR-owned
  memory holds every requested block, the pull starts at once; otherwise it
  waits for the source's pass. Pulled blocks land in the target's G2 only.
- **Router events.** KVCR-owned memory publishes `HostPinned` events in the
  cache-owner domain (`"domain": "cache_owner"`), from the end of a block's
  copy until memory evicts it or its disk write ends. The worker's
  `HostPinned` events cover G2 alone.

A copy between the two caches takes a block's bytes, its KV for every GPU of
the rank, divided by `copy_bandwidth_gbps`; each direction runs a rank's
copies one after another. The default approximates a B200 node without RDMA;
set the rate of your transport. Promotions count toward the 1,024-round limit
above.

The model simplifies KVCR-owned memory:

- Copies do not fail or time out.
- G2 cannot evict a store's blocks until their copy into KVCR-owned memory
  ends, and then ranks them as most recently used.
- A promotion never evicts a G2 block of the same prompt and stops at the
  first block it has no other room for.

Under `lru`, the order can differ from vLLM's in corner cases, such
as a pull that lands without a promotion.

With `local_g2` the `kvcr` summary gains a `local_g2` section of block counts:
`stored_blocks` (counted when the copy completes), `dropped_blocks` (not
copied for lack of room), `evicted_blocks`, `promoted_blocks` and
`loaded_blocks` (promoted blocks, counted at their first load into G1).

A joined promotion and peer pull become readable in G2 together. The pull's
delivery counters update only then, and `remote_deliver_ms` includes any wait
for the local copy. Its source blocks are released when the target takes the
completed peer transfer, even before the local copy ends.

Guard ownership and custom policies are not modeled.

<a id="kvcr-owned-disk"></a>

### KVCR-owned disk

With `kvcr.g3`, each rank also has a disk cache that KVCR owns: a block that
KVCR-owned memory evicts is written there instead of dropped, and a lookup
that misses G2 and that memory reads it back into G2. The `local_g2.policy`
values `lru` and `fifo` then stand for vLLM's `g3_lru` and `g3_fifo`.

- **Write.** KVCR writes one block at a time. A block of a completed G2 store
  that finds KVCR-owned memory full waits for the write that frees its slot,
  up to `spill_timeout_ms`; after that it stays in G2 only. A write still
  running at that point fails and its block is dropped. A rank sees a
  finished write when its pass ends, so a busy rank finishes at most one
  write per pass. While a block waits, G2 cannot evict any block of its
  store, so a long `spill_timeout_ms` with a slow disk can hold most of G2.
  A store that times out holds its blocks until the write in flight ends, up
  to one write longer; KVCR releases them at the timeout.
- **Evict.** A full disk drops a block by the memory's policy, never one
  being read; when all are being read, the block memory evicts is dropped
  unsaved. Under `lru` a block keeps the rank it had in memory and the same
  uses count on disk; `fifo` drops the block unused longest since it was
  written or last read.
- **Read.** A block found only on disk is read into G2, like a promotion and
  before the hinted peer. It stays on disk and out of KVCR-owned memory.
- **Peer pulls.** A peer cannot pull a block that is only on disk.
- **Router events.** A block is published with tier `Disk` (`"tier": "disk"`),
  in the cache-owner domain, when its write ends and removed when the disk
  drops it.

The timing defaults describe a local NVMe drive; set the values of your
storage. With them a 9.4 MB block, its KV for every GPU of the rank, is
written in about 15 ms. Each rank's disk is modeled on its own, so lower
`write_bandwidth_gbps` for ranks whose disk caches share a drive.

The model simplifies the disk tier: reads do not fail or time out, where
KVCR fails a read that outlasts its operation timeout, 1 s by default. Under
`lru`, the order in memory and on disk can differ from KVCR's in corner
cases, such as a store that gives up before copying all of its blocks.

With `g3` the `kvcr` summary gains a `g3` section of block counts:
`stored_blocks` (written to disk), `evicted_blocks` (dropped from disk) and
`read_blocks` (read back into G2). In the `local_g2` section, `evicted_blocks`
then includes blocks written to disk, `dropped_blocks` those that waited too
long and `loaded_blocks` those read from disk.

`kvcr.g3` runs on `--stack engine`; another stack is rejected unless its
runner declares support.

### Supported

KVCR supports vLLM aggregated workers with any `attention_data`, and pulls
between the prefill ranks of P/D. Workers may be fixed or Planner-scaled; a
draining worker is removed only after every pull from or into it has landed and
been taken by its target. Rejected: missing or `cluster_shared` `host_offload`,
a `host_offload.connector` other than `offloading_v0_24`, `g3_offload` together with `kvcr`, `kvcr.g3` without `kvcr.local_g2`, `kvcr` on the decode role, MTP on a prefill role with `kvcr`, `parallelism.pipeline` above 1,
`parallelism.prefill_context` above 1 (native `prefill_context_parallel_size`), and `recommend`.
Everything G2 rejects is rejected too, including SGLang and TensorRT-LLM.

Hints come from the placement policy (see
[native composition](../../adapters/native-composition.md#placementpolicy)) or
from `kv_hints` on native ReplaySpec requests (see
[Rust API](../api/rust.md#serializable-spec)). Replay ignores a hint that names
the receiving rank. CLI traffic carries no hints, so
`aisim predict --stack engine` runs peer fetch idle: without `local_g2`, the
results equal G2 alone apart from the `kvcr` section.

For a complete CLI example with router-generated hints, follow
[Simulate KVCR peer reuse](../dynamo.md#simulate-kvcr-peer-reuse). It uses
`--stack dynamo`, two workers with private G2, and a Mooncake trace, and compares
KVCR-enabled replay against a G2-only baseline.

<a id="state-cache"></a>
<a id="manual-state-cache-sizing"></a>

## State cache

Hybrid-KV models keep a fixed-size state per request in addition to per-token
KV: the recurrent state of Kimi K3's KDA layers, or the sliding-window KV and
compressor states of DeepSeek V4. `state_cache` reserves that state in the same
G1 pool. The fragment below goes under
`engine.workers.aggregated`, or under both `prefill` and `decode` for P/D:

```yaml
kv_cache:
  block_size: 64
  bytes_per_token: 16
  capacity: {type: fixed, bytes: 8192}
  state_cache: {bytes_per_request: 1500}
```

This gives eight 1024-byte blocks. Each request's state takes two blocks
(1500 bytes rounded up) in addition to its token KV.

| Knob | Default | Rules |
|---|---|---|
| `state_cache.bytes_per_request` | Inferred | State bytes per request per rank. When set, `block_size` and numeric `bytes_per_token` must also be set. |
| `state_cache.num_speculative_states` | Unset | Explicit full-state scratch copies per request. Must match the draft depth, from positive `nextn` or `engine.speculation`. Requires numeric state bytes and token geometry. Under MTP it also applies vLLM's [EAGLE block drop](speculation.md#prefix-caching) to state lookups. |
| `state_cache.prefill_checkpoint` | `false` | vLLM's internal prefill checkpoint; see [below](#opt-in-align-rules). Needs `prefix_match_unit` equal to `block_size`. |
| `state_cache.junction_checkpoints` | `false` | vLLM v0.29.0's shared-prefix junction states; see [below](#opt-in-align-rules). Needs an explicit `prefix_match_unit`. |
| `state_cache.mamba_cache_dtype` | `auto` | Kimi K3 only. `auto` (model dtype), `float16` or `float32`. Used when bytes are inferred. |
| `state_cache.indexer_cache_dtype` | `auto` | DeepSeek V4 only. `auto` (vLLM's FP8), `fp8` or `mxfp4`; vLLM supports `mxfp4` only on Blackwell datacenter GPUs. Used when bytes are inferred. |
| `state_cache.retention_interval` | Unset (every block boundary) | DeepSeek V4 with inferred bytes only. vLLM's `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`: `0` or a multiple of the resolved `block_size`. DeepSeek V4 rejects `prefix_match_unit`. |
| `state_cache.sliding_window_lookahead_tokens` | Inferred (DeepSeek V4: 64) | Tokens past the full-attention hit that the MTP layer's sliding-window group reads; sizing fills it. Set it only with explicit `bytes_per_request`, to replay a state resolved earlier. |
| `prefix_match_unit` | Unset (`block_size`) | Positive divisor of `block_size`. Lets prefixes match at a finer granularity than the physical block. |

`state_cache: {}` infers Kimi K3 geometry from the model. `block_size` is then
a requested page size (default 64) that is enlarged to fit one state per layer.
For example, TP8 with FP8 KV and `block_size: 128` resolves to 1536-token blocks
of 13824 bytes per token. The same estimate is available in the SDK:

```python
from aisim_core.sdk import estimate_state_cache

state = estimate_state_cache("moonshotai/Kimi-K3", tp_size=8, kvcache_quant_mode="fp8", block_size=128)
```

For DeepSeek V4, `state_cache: {}` follows vLLM's allocation. vLLM stores
`fp8_ds_mla` KV and gives every cache group the same pool row, as wide as the
widest group. A request holds one row per 256 tokens of compressed KV and
indexer keys (`block_size` 256, `bytes_per_token` one row per 256 tokens), plus
a 26-row state for its sliding-window groups (SWA KV and compressor states). For
DeepSeek-V4-Flash, 1,002,240-byte rows give 3915 bytes per token and a
26,058,240-byte state on every TP rank.
Leave `block_size` and `bytes_per_token` unset. Speculative decoding (`nextn`
or `speculation`) widens each sliding window by the draft tokens, so one draft
token gives 30 rows. These are modeling assumptions: 26 rows is vLLM's decode
peak (it dips to 24 between window blocks), vLLM also holds sliding-window slots
for every token of a prefill chunk until the next step (3,360 rows for an
8,192-token chunk), and a prefix hit restores 22 of the 26 rows. None of these
is modeled.

DeepSeek V4's state is a sliding window that vLLM caches block by block, so a
later request can resume at a retained boundary the earlier request passed,
wherever its steps ended. `retention_interval` selects the boundaries, as
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` does in vLLM v0.32.0: unset keeps every
256-token boundary; `0` keeps only the last boundary below each prompt's end; a
positive value additionally keeps one boundary per interval. With speculative
decoding the group holding the MTP layer also needs the 64 tokens after a
boundary, and they must lie inside the block-aligned token hit, so a hit lands
one block lower and vLLM keeps the boundary one block lower instead. A 40,000-token
prompt with `retention_interval: 32768` keeps 32,768 and 39,936 without
speculative decoding, and 32,768 and 39,680 with it; its next turn resumes at
39,936 or 39,680. Each retained boundary holds one reclaimable state copy
(vLLM keeps about 22 of the 26 rows) until eviction. vLLM also keeps state
where a request's cache hit stopped short of a longer shared prefix; replay
does not model that. vLLM keeps the window in the request's own blocks, while
replay copies the state, so a full pool with nothing to evict skips the
boundary. This model needs inferred geometry: with `bytes_per_request` set,
DeepSeek V4 falls back to one recurrent-style checkpoint per request.

Kimi K3's recurrent state resumes only from a stored snapshot, so it does not
land on every matching token. With `block_size: 1536` and `prefix_match_unit: 128`, a cold
24,300-token prompt keeps snapshots at 23,040 and 24,192 tokens. A later request
sharing 23,700 tokens resumes at 23,040. This follows vLLM v0.29.0 align mode,
with one difference: when a prompt ends on a block boundary, replay also keeps
the state one block earlier (two under the EAGLE block drop), so an identical
resend resumes there. vLLM keeps a state there only if a prefill chunk ended
there.

### State pool from recorded runtime memory

An `engine.fpm_profile` whose deployment records `resources.runtime_memory`
rejects fixed capacity, because the pool is the one the backend allocated when
the profile was collected. With inferred state geometry (`state_cache` without
`bytes_per_request`), leave `capacity` at `type: default`. Each rank's pool is
then the recorded `runtime_memory.kv_cache_bytes` in whole blocks of the
resolved geometry: `kv_cache_bytes // (block_size * bytes_per_token)`.
`capacity.memory_fraction` must equal the recorded `gpu_memory_utilization`,
and the context length must not exceed the recorded `max_model_len`.
Leave `capacity.cuda_graph_reserved_bytes` unset, because the recorded pool
already excludes graph memory. A grouped FPM cache and `prefill_context` or
`decode_context` above 1 are not supported here. Every
`engine.speculation.kind` except `ngram` requires fixed capacity, so it cannot
use a recorded pool; set the draft depth with `engine.nextn` instead.
The block count follows the inferred row size, so it can differ from the count
the backend logged when that build packs its cache groups differently.

In `mode: disaggregated`, set `state_cache` on both `prefill` and `decode`;
each role sizes its own state. The decode worker reserves the prompt blocks
plus one state when it accepts a request, then continues from the transferred
state. With [`engine.kv_transfer`](kv-transfer.md), the transfer also moves
one state: the prefill role's `bytes_per_request`, charged once per request in
either `timing_mode`. A decode-side prefix hit skips resident token KV; the
state still travels with the rest of the prompt. These are modeling
assumptions: vLLM sends the raw convolution and recurrent state (about 8% less
than the padded `bytes_per_request` for Kimi K3 at TP8), transfers it after all
but the last prompt token and recomputes that token on the decode worker. Replay
transfers the state for the whole prompt and does not model the recompute.

<a id="opt-in-align-rules"></a>

Two align-mode behaviors of vLLM are opt-in and off by default:

- **`prefill_checkpoint`** is vLLM's `num_prefill_checkpoint_blocks = 1`, which
  vLLM 43b4aaea sets for Mamba2 layers. When the final prefill chunk starts on a
  block boundary and spans at least one block before
  `c = (P - 1) // block_size * block_size`, the chunk runs to the prompt end and
  its forward exports the state at `c` into one extra state reserved for that
  pass. Without it the chunk stops at `c`, which costs one more pass.
- **`junction_checkpoints`** is vLLM's default at v0.29.0, the revision the
  align model follows. When a lookup's attention
  hit runs deeper than its deepest state (`shared_prefix_boundary`), the
  prefill stops at that block-aligned position and keeps the state there; a
  sibling sharing the prefix then resumes from it.
  A lookup that would resume from a state published in the same pass waits one
  pass, as vLLM's allocation does. Without the switch, such states stay hidden
  and the lookup falls back to an older state.

With [G2 host offload](#host-offload-g2), a sliding-window state's retained checkpoints
follow the prompt blocks into the host cache, as vLLM's offloading connectors
store the sliding-window blocks that retention keeps. A checkpoint is one G2
entry that takes a block per state row (26 for DeepSeek-V4-Flash, 30 with one draft token), so
`num_host_blocks` must exceed one checkpoint. With the default retention every
prompt block retains a checkpoint. Replay copies a whole window per checkpoint
where vLLM's overlapping windows share blocks, so state then takes about a
checkpoint's rows times the G2 space and store time of the prompt blocks, more
than in vLLM. Set `retention_interval` as the deployment does (`0`, or an
interval of at least one checkpoint's tokens) to keep the pool for prompt
blocks. The checkpoints a prompt pass retains are stored in one cohort with
that pass's prompt blocks, as the connector issues one store per request and
step. G2 takes the whole cohort or none of it, and evicts none of its blocks
to make room for the rest. A cohort that found no room is offered again, with
what later passes added, when the pool's room changes. One larger than the pool
is not offered again before the request is readmitted; replay warns once and
names `retention_interval` and `num_host_blocks`. A boundary reached during
decode is not stored. A restore resumes at the highest boundary that has both
its token blocks and its whole checkpoint in G2, below the logits cap. With
speculative decoding that keeps the EAGLE block drop it is also one block below
the last prompt block G2 holds, as vLLM's drafter group drops the last chunk it
finds; a block-aligned prompt whose blocks are all in G2 therefore still
resumes at the logits cap. If G2 evicted a checkpoint, the restore uses a lower
boundary or recomputes. The load brings the token blocks up to that
boundary and the checkpoint, which becomes the request's state there, as the
state a prefill worker hands to a decode worker does. G1 also caches the window
at that boundary when it has room, so a later request resumes there from G1.

vLLM has sliding-window groups from v0.32.0 on, so a rank-local pool that
offloads a sliding-window state follows the `offloading_v0_32`
[profile](#connector-profiles), whether `connector` names it or keeps its
default. `simple_cpu_v0_32` is rejected: vLLM's `SimpleCPUOffloadConnector`
keeps sliding-window groups too, but replay does not model how it stores them
block by block. Blocks and checkpoints are ranked when their store starts. A finishing request offers a cohort it could not store
earlier and re-ranks what it stored, offered or loaded, each checkpoint just
above the block that ends at its boundary. A prefill worker does the
finish-time work when it releases the request after the KV transfer, later than
vLLM; for other models that timing keeps `offloading_v0_32` to aggregated
workers. A cluster-shared
pool has no vLLM counterpart and keeps its own order: a lookup makes the
prompt's blocks and the checkpoint it resumes at the newest, so checkpoints at
lower boundaries are evicted first.

The rank's stores keep the G1 token blocks they copy until the whole cohort is
copied, as a shared pool's do, and the checkpoints in the cohort lengthen that
copy. vLLM also flushes a store before a pass reuses one of its
sliding-window blocks, so that pass waits for the copy. Replay holds no G1
block for a checkpoint, so G2 can keep a checkpoint that G1 skipped or evicted. Ranks whose states differ in size (with
and without draft tokens) do not read each other's checkpoints from a shared
pool. G2 occupancy counts a checkpoint's rows as blocks; checkpoints publish no
`host_pinned` KV events.

Supported: aggregated and P/D vLLM, fixed `capacity` (or the
[recorded pool](#state-pool-from-recorded-runtime-memory) of an FPM profile), `predict --stack engine`;
inferred geometry for Kimi K3 and DeepSeek V4 at PP=1; G2 for a sliding-window state.
Speculative decoding supports `prefix_match_unit` equal to `block_size`.
For sub-block matching, supply `state_cache.num_speculative_states` equal to
the draft depth, from positive `nextn` or `engine.speculation`.
Each scratch copy consumes the same padded pages as the
working state. The pool must fit these copies plus one token page.
Without the field, MTP reserves no scratch copies and state lookups skip the
EAGLE block drop.
This explicit mode models full-state scratch storage, not ReplaySSM or
RecoverSSM rings or temporary records.

Rejected: `recommend`, G3 offload, G2 for a recurrent state except as a
[hybrid G2](#hybrid-g2), G2 for a sliding-window state with `kvcr` or a
`host_offload.connector` other than `offloading_v0_24`, and speculative
sub-block matching without an explicit full-state scratch count.

## Other cache modes

- **Observed Mamba pools.** An FPM profile with `cache_layout: mamba_align`
  supplies the shared pool and state reservations for aggregated vLLM; see
  [memory accounting](../../perf-model/memory.md#observed-mamba-align-pools).
- **Grouped caches.** Some whole-forward (FPM) profiles describe several KV
  groups, such as sliding-window and full attention, sharing one byte budget.
  Replay then allocates each group separately. A `state` group holds a fixed
  number of blocks for each running request. Grouped caches require aggregated
  vLLM, no offload, no P/D and default capacity. They run cold with
  `prefix_caching: false`. No offload means no `host_offload`, `g3_offload` or
  `kvcr`, with prefix caching on or off. A `sliding_window` group or a `state`
  group of a grouped cache is not the sliding-window [state cache](#state-cache)
  whose checkpoints [G2](#host-offload-g2) keeps. See
  [memory accounting](../../perf-model/memory.md).

  Grouped caches run with prefix caching on only under these conditions,
  which prediction, recommendation and planning check. Planning tests the
  vLLM version of the evidence, as the next paragraph says. The profile
  deployment's `backend_version` is `0.31.0`, `0.32.0` or `0.33.0`, and its
  runtime memory records `enable_prefix_caching: true` and
  `hash_block_size_tokens`.
  Every group declares `prefix_cacheable`. No group is a `convolution` group,
  because grouped prefix caching does not support convolution groups yet. No
  attention group with a `sliding_window` has `prefix_cacheable: false`: a
  ring must be a `state` group with `blocks_per_request: 1` and
  `prefix_cacheable: false`, and replay does not model a bounded replay window,
  so turn prefix caching off for it. All
  groups have the same `page_size_bytes`. A prefix-cacheable `state` group
  also needs `mamba_cache_mode: align` in the runtime memory. When the
  runtime memory records `runtime_framework_version`, the vLLM version of its
  evidence, that version equals the `backend_version`. A set
  `prefix_match_unit` equals `hash_block_size_tokens`. For GLM-5.3-Flash, the
  hash block and `max_batched_tokens` are multiples of 4. On vLLM 0.33.0,
  `scheduler.prefill_schedule_interval` is 1. vLLM 0.33.0 holds prefills at
  attention DP size 1 too, and replay does not model that; see
  [vLLM prefill schedule interval](workers.md#vllm-prefill-schedule-interval).
  A recommendation and an onboarding plan run with 1. A prediction or a native
  rank that you write by hand and that sets another value is refused.

  The runtime probe import takes prefix-caching evidence only from a vLLM
  0.33.0 record, and it records that version as
  `runtime_memory.runtime_framework_version`. `aisim onboard plan` and
  `aisim onboard finalize` test the vLLM version of the evidence: that field,
  else the request's `runtime_framework_version`, else the `backend_version`.
  They turn prefix caching on only when that version has grouped rules and
  equals the `backend_version`. So the plan of a profile from the runtime
  probe with the label `0.33.0` sets `prefix_caching: true` when the profile
  meets the conditions above, also when `onboard init --fpm-profile` reuses
  the profile. Under another label the plan sets `prefix_caching: false`.
  When the probe ran with prefix caching and the version of the evidence has
  no grouped rules or differs from the label, these commands print a note
  that names that version and, when it has grouped rules, the label. A
  prediction or recommendation that turns prefix caching on for a profile
  whose recorded version differs from its label is refused before replay.
  `onboard probe-runtime` keeps an explicit dataset label that differs from the
  observed runtime; keep the label equal to the probed version. A plan that
  was saved while replay had no grouped rules for vLLM 0.33.0 holds
  `prefix_caching: false`. The commands that verify a saved plan still accept
  it and keep its files. `aisim onboard plan --overwrite` writes the plan again
  with prefix caching on, and `aisim onboard finalize` writes the plan of the
  resolved profile with prefix caching on. A profile
  imported before the version was recorded holds no `prefix_cacheable` marks
  either, because the import writes the marks and the version together, so
  its plan is cold and a prediction or recommendation with prefix caching is
  refused. If you set the marks of such a profile by hand, also set
  `runtime_memory.runtime_framework_version` to the version that the probe ran
  on. With the marks and without the version, a label of `0.31.0` or `0.32.0`
  on vLLM 0.33.0 evidence can plan prefix caching on and passes these
  conditions, and native replay applies the rules of the label to that
  evidence, after the startup checks below. That use is not modeled. A
  hand-made profile of `backend_version`
  0.31.0, 0.32.0 or 0.33.0 with prefix-caching evidence, without a recorded
  version or with that version, can turn grouped prefix caching on, also in
  its plan.

  Native replay checks more at startup and refuses the rank otherwise. Every
  state group holds one block per request. At least one group is
  prefix-cacheable. The hash block divides the block of every
  prefix-cacheable group, and the scheduler block is a multiple of it. The
  scheduler block, the least common multiple of the group blocks, fits in
  the machine's unsigned integer size. On
  vLLM 0.31.0 no prefix-cacheable state group has a positive
  `prefill_checkpoint_blocks`. On vLLM 0.32.0 and 0.33.0, which replay checks
  the same way, every prefix-cacheable state group records
  `prefill_checkpoint_blocks`, and all of them record one count, 0 or 1. With
  a count of 1 they record one `prefill_checkpoint_alignment`, and each of
  them has the smallest block of the prefix-cacheable groups. With a count of
  0 no group has an alignment. On
  vLLM 0.33.0 `prefill_schedule_interval` is 1. On every modeled version no
  `state` group with `prefix_cacheable: false` has a positive
  `prefill_checkpoint_blocks`. `max_model_len` is set, and the pool holds one prompt of
  `max_model_len` plus the state blocks of one request. `preemption_mode` is
  `lifo`, the default, which is vLLM's order: it preempts the last running
  request. Planning runs only the
  conditions above, so a profile that fails a native check plans prefix
  caching on and is refused at replay. The native rank does not hold the
  Mamba cache mode, so native replay does not check align mode. It does not
  hold the profile's `runtime_framework_version` either: a replay spec or a
  native rank that you write by hand states `vllm_version` itself, and native
  replay does not check it against the version that the profile records.
  A native rank does not hold the runtime memory either, so native replay
  does not check the probe evidence there, that `prefix_match_unit` equals
  `hash_block_size_tokens`, or the multiples of 4 of GLM-5.3-Flash.

  Replay also assumes vLLM settings that it has no field for. The scheduling
  policy is FCFS. No aux-output connector and no HiSparse placement are in
  use. Requests are text requests without a LoRA adapter. Replay has no gate
  for these assumptions, and a deployment that differs in one of them is not
  modeled.

  Replay then follows the rules of the vLLM version in
  `engine.backend_version`. For the cases that replay models, vLLM 0.33.0 has
  the rules of vLLM 0.32.0:

  - Replay hashes prompts in blocks of the profile's `hash_block_size_tokens`.
    vLLM allows hits inside a block under two conditions. A prefix-cacheable
    `state` group has a block larger than the hash block. Every
    prefix-cacheable sliding-window group has a block equal to the hash
    block. A hit then ends on a multiple of the hash block. Otherwise it ends
    on a multiple of the least common multiple of the group block sizes.
  - State groups resume only at the boundaries that vLLM keeps under its
    default `prefix_cache_retention_interval` of 0, such as the last block
    boundary before the end of a prompt. A `state` group with
    `prefix_cacheable: false` is a ring, such as an indexer tail. It holds one
    block for the whole request and never hits. An attention group with
    `prefix_cacheable: false` and no `sliding_window` takes no part in hits
    and takes blocks for the whole prompt. No full-attention cache spec of
    vLLM 0.31.0, 0.32.0 or 0.33.0 opts out of prefix caching.
  - A request that resumes from a cached state on a block boundary holds the
    cached block and its running block during its first step. It releases the
    cached block at its next step, as vLLM does at the request's next
    allocation with `async_scheduling: false`. The block stays cached until
    eviction. See vLLM v0.31.0
    [`kv_cache_manager.py:552-556`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/kv_cache_manager.py#L552-L556)
    and
    [`single_type_kv_cache_manager.py:1626-1645`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/single_type_kv_cache_manager.py#L1626-L1645),
    vLLM v0.32.0
    [`kv_cache_manager.py:555-559`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_manager.py#L555-L559)
    and
    [`single_type_kv_cache_manager.py:1592-1611`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py#L1592-L1611),
    and vLLM v0.33.0
    [`kv_cache_manager.py:564-568`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_manager.py#L564-L568)
    and
    [`single_type_kv_cache_manager.py:1608-1627`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/single_type_kv_cache_manager.py#L1608-L1627).
    A hit inside a block of a full-attention or `state` group copies the
    cached block in its first step, and the copy replaces the cached block in
    the request. As in vLLM, the cached block stays in use until the
    scheduling pass ends, so that the copy can read it. A request that the
    same pass admits later cannot take it. Replay then releases these blocks
    group by group, in the order of the admissions. See vLLM v0.31.0
    [`single_type_kv_cache_manager.py:454-474`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/single_type_kv_cache_manager.py#L454-L474)
    and
    [`scheduler.py:1434-1442`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/core/sched/scheduler.py#L1434-L1442),
    vLLM v0.32.0
    [`single_type_kv_cache_manager.py:454-474`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py#L454-L474)
    and
    [`scheduler.py:1474-1482`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py#L1474-L1482),
    and vLLM v0.33.0
    [`single_type_kv_cache_manager.py:455-475`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/single_type_kv_cache_manager.py#L455-L475)
    and
    [`scheduler.py:1698-1706`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L1698-L1706).
  - Prefill chunks end on block boundaries, so that the state there can be
    cached. Some chunks end inside a block. The final chunk ends at the end of
    the prompt. If one block is larger than `max_batched_tokens`, chunks
    advance by up to that budget until they reach the next block boundary.
    With hits inside a block, a chunk can also stop at a hash boundary inside
    the last block of the prompt, so that a later hit can resume there.
    As in vLLM, this align split applies only to a rank with a
    prefix-cacheable `state` group. A rank with only full-attention and
    sliding-window groups is not split.
  - On vLLM 0.32.0 and 0.33.0 a prefix-cacheable `state` group can carry a
    prefill checkpoint block, such as the FlashKDA block of a KDA group. The
    request takes that block in the chunk that reaches the end of its prefill and
    releases it at its next step. As in vLLM, the prefill of a request that
    resumes after a preemption ends one token before its last known token,
    so it covers the output tokens too. The block stays cached until eviction
    when it holds the prompt's checkpoint, the last position before the end
    of the prompt where a hit can end, or the end of a shared prefix where
    the request's hit stopped short. vLLM drops any other
    checkpoint under its default retention of 0. Only a resumed request takes
    such a checkpoint.

  Replay differs from vLLM in these known ways:

  - Replay holds the previous state block after a block boundary, the cached
    block after a hit on a block boundary and the checkpoint block after the
    final prefill chunk for one scheduler step, and it releases the blocks of
    a sliding-window group once they leave the window. vLLM 0.31.0, 0.32.0 and
    0.33.0 turn async scheduling on by default and then hold these blocks for
    one more step. Replay models async scheduling in the arrival rule, in the
    admission cap of a sliding-window group and in the blocks of a finished
    request, which it frees after it has formed the next pass (see
    [vLLM async scheduling](workers.md#vllm-async-scheduling)). It does not
    model it in these holds. So block counts and admission can be optimistic
    for one step, by one block per state group and held block and by the
    window blocks that left the window. A held block, or a block of a
    sliding-window group that left the window, also enters the eviction order
    one step early: before the blocks of a request that finished in the pass
    before, where vLLM frees it after them. When the pool evicts, a different
    cached block can then go first, and a later hit can be shorter or longer
    than in vLLM. With `async_scheduling: false` these blocks are held for one
    step in vLLM too.
  - Replay caches blocks of prompt tokens only. vLLM also caches the
    full-attention blocks that output tokens fill. A request's hit is at most
    the shortest hit over its prefix-cacheable groups. Under the default
    retention, a state or sliding-window group keeps no output-token position.
    So when every prefix-cacheable group is full attention, a next turn that
    repeats recorded output tokens hits fewer tokens than in vLLM. With a state
    or sliding-window group, the difference changes only which blocks are cached
    and the eviction order.
  - Replay does not let a waiting request hit a state block that another
    request cached in the same scheduler step, or a state whose copy another
    request took in that step. The request gets the shorter
    hit and can start in that step. vLLM keeps the request, and the requests
    queued behind it, waiting for that step, and gives it the longer hit in
    the next step. vLLM 0.32.0 also holds such a request in the step or steps
    after the publishing step, when the next cache step of the publishing
    request marks the same block again. Replay does not model that. vLLM
    0.33.0 does not mark a block that already has its key, so it does not
    hold the request there. See vLLM v0.32.0
    [`single_type_kv_cache_manager.py:1977-1982`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py#L1977-L1982)
    and
    [`:1688-1696`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py#L1688-L1696),
    and vLLM v0.33.0
    [`single_type_kv_cache_manager.py:2112-2115`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/single_type_kv_cache_manager.py#L2112-L2115).
  - vLLM looks up prefix-cacheable groups together only when their cache specs
    are equal in every field. The profile records only a group's kind, block
    size, window and checkpoint setting, so replay looks up groups that agree
    on these together, also when vLLM keeps them apart because another field
    differs, such as the head size, the data type or the state shape. Replay
    can then miss a prefill stop where vLLM saves an extra state, and with hits
    inside a block it can give a shorter hit. The profile does not record which
    groups share a spec.
  - When the hash block is smaller than a full-attention block, vLLM 0.32.0
    and 0.33.0 can give a cached block two keys: the key of its block
    boundary and the key of the last hash boundary before the end of a
    prompt, inside the block. Replay gives the block one key, the key that it
    got first. vLLM 0.31.0 puts that key at the last hash boundary at or
    before the end of the prompt, in a block that prompt tokens do not fill,
    so it has no such case. The difference shows in two ways. With hits
    inside a block and a prefill checkpoint block, one chunk can end at the
    end of a prompt whose length is a multiple of the full-attention block.
    The last block then gets its full key only, and the state group has a key
    only at the checkpoint inside that block. A later request with the same
    tokens and a length inside the last hash block of that prompt, such as a
    repeat of the prompt, gets an older state or no hit. vLLM hits the prompt
    length less one hash block. With blocks of 32 tokens, a hash block of 4
    and a prompt of 64 tokens, a repeat hits 0 tokens in replay and 60 in
    vLLM. A longer prompt, or a prompt whose length is not a multiple of the
    block, is not affected. The other way needs no hits inside a block:
    every prefix-cacheable group is full attention. The hash block is then
    smaller than a block when the blocks differ, where it is their greatest
    common divisor by default, or when `prefix_match_unit` sets a smaller
    one. A block that gets the
    key of the last hash boundary first gets no full key when a later chunk
    fills it. A later, longer prompt can then hit one least common multiple
    of the group block sizes less than in vLLM. With blocks of 64 and 16
    tokens, a prompt of 128 tokens computed in chunks that end at 120 and 128
    makes a later prompt of 200 tokens hit 64 tokens in replay and 128 in
    vLLM. With hits inside a block this order gives the same hit as in vLLM,
    because the state group has its key at the same hash boundary. The
    published GLM-5.3-Flash profiles reach neither case, because they are
    vLLM 0.31.0 profiles. GLM-5.3-Flash at default serving reaches neither,
    because its hash block then equals its block. See vLLM v0.32.0
    [`single_type_kv_cache_manager.py:838-884`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py#L838-L884),
    [`block_pool.py:286-299`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/block_pool.py#L286-L299),
    [`:502-523`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/block_pool.py#L502-L523)
    and
    [`:626-646`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/block_pool.py#L626-L646).
  - With hits inside a block and a state group whose block is larger than the
    hash block, a request can write past a state that it published inside a
    block. Replay then takes a new block and keeps the published block
    cached, as vLLM does. As in vLLM, the admission count does not include
    that block, so this write can find no free block, and the last running
    request is preempted. Replay holds the published block until the
    request's next allocation, where vLLM frees it at the end of the
    scheduling pass that took the copy. So the block count can be higher by
    one block per state group until then.
  - With hits inside a block and a prefill checkpoint block, vLLM 0.32.0 and
    0.33.0 can give a later request a longer hit than replay. The case needs a
    request whose state hit ends inside a block, while its full-attention
    groups match at least one block boundary further. That boundary is the end
    of the shared prefix. With a checkpoint block the request's first chunk
    does not stop at that boundary. vLLM still gives the key of that boundary
    to the request's state block that ends at it. By a reading of vLLM's code,
    that block holds the state of a shorter prefix: the copy of the cached
    state, or the state at the place inside the block where an earlier chunk
    stopped. A later request with the same prefix then hits at the boundary in
    vLLM and continues from that state. Replay publishes a state key only
    where a chunk ended, so the later request hits the shorter prefix and
    computes the tokens up to the boundary again. Replay does not copy this
    behavior of vLLM, and it accepts the configuration. Without a checkpoint
    block a chunk stops at the boundary in vLLM and in replay, and the two
    agree. See vLLM v0.32.0
    [`scheduler.py:526-543`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/sched/scheduler.py#L526-L543),
    [`single_type_kv_cache_manager.py:510-511`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py#L510-L511),
    [`:1584-1588`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py#L1584-L1588)
    and
    [`block_pool.py:277-280`](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/block_pool.py#L277-L280),
    and vLLM v0.33.0
    [`scheduler.py:632-649`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py#L632-L649),
    [`single_type_kv_cache_manager.py:512-513`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/single_type_kv_cache_manager.py#L512-L513),
    [`:1600-1604`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/single_type_kv_cache_manager.py#L1600-L1604)
    and
    [`block_pool.py:281-284`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/block_pool.py#L281-L284).

- **Belady eviction.** The native ReplaySpec field
  `engine.kv_eviction_policy: belady` evicts the block whose next use in the
  trace is farthest away. It is not available in `predict` YAML. It requires an
  open-loop trace on fixed aggregated workers with `attention_data: 1` and no
  offload. Use it to bound how much a better eviction policy could help.

<a id="agentic-g2"></a>

## Agentic traffic with G2

AgentX snapshot, warmup and continuous-profile traffic supports G2 on vLLM with
static aggregated or prefill/decode workers of any count and `attention_data`,
with either G2 scope. A role may omit G2. Each role uses canonical MTP, within
the [combined scope](#supported) above, MTP on a [hybrid G2](#hybrid-g2), or no
speculation. G3 is not supported.
When the replay composition captures engine
KV events, each worker publishes its G2 residency as native `HostPinned` events,
so a KV-aware placement policy can follow a G2-resident prefix to its owner, or
to any replica of a `cluster_shared` pool. The built-in round-robin composition
leaves this capture disabled by default. Snapshot and profile plays
use play-local token identities, so G1 and G2 hits never cross a recycled play.
`kvcr` requires `dp_rank_local` G2 and the other [KVCR constraints](#kvcr-peer-fetch);
it is rejected with `cluster_shared`. In supported AgentX configurations it
never pulls, because agentic traffic carries no router hint. Results are
qualified `functional_only`.

A `cluster_shared` pool under AgentX follows this contract:

- **Ownership and discovery.** The replay deployment owns one pool. Every
  participating rank, across replicas and P/D roles, joins it; the first
  participant fixes layout, capacity, and shared bandwidth, and capacity is
  counted once in `g2_domains`. A block becomes resident when its D2H store
  completes. With engine KV-event capture enabled, the pool publishes a
  `HostPinned` store to every subscribed rank. Pool-wide LRU eviction publishes
  the matching removal to those subscribers.
- **Identity.** Conversations of one play share G2 prefixes. Snapshot and
  profile plays use play-local token identities and a distinct `cache_id`, so
  equal source prompts in recycled plays occupy separate entries and never hit
  each other.
- **Contention.** Concurrent restores of the same entries count residency once
  and split the pool's directional bandwidth max-min fairly, deterministically
  from the replay clock.
- **Stale entries.** If a placement scored residency that is evicted before the
  request is admitted, the request recomputes the missing prefix. Its routing
  overlap then exceeds the realized `host_reused_input_tokens`.
- **Membership.** AgentX G2 requires static pools, so worker failure and
  ownership handoff are outside its support boundary.
- **Disabling sharing.** `scope: dp_rank_local` reproduces the private-G2
  result exactly, even when shared-only bandwidth fields remain set.

Two small offline fixtures use fixed timing and need no GPU:

```bash
aisim predict --stack engine --config examples/cli/agentx-g2-local.yaml \
  --capture-per-request --output-dir ./agentx-g2-local --format json
aisim predict --stack engine --config examples/cli/agentx-g2-shared-pd.yaml \
  --capture-per-request --output-dir ./agentx-g2-shared --format json
```

In both, a child request evicts its parent's prefix from a three-block GPU
cache, and the parent later restores it from host memory. Change
`host_offload.scope`, `h2d_bandwidth_gbps` or `num_host_blocks` to compare.
