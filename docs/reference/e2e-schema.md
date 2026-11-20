<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# End-to-end (e2e) data schema

`aisim-e2e/v1` stores end-to-end serving measurements in the unified dataset: InferenceX points, AgentX
points (official and self-collected) and our own benchmark runs share one layout. The schema is built to
grow: new fields are expected, and old readers keep working.

The code lives in `aisim_core/data/e2e.py` (models and tables), `aisim_core/data/names.py` and
`names_registry.json` (canonical names), and `aisim_core/data/agentx_convert.py` (AgentX converter).
JSON Schema snapshots are generated into `aisim_core/data/schema_snapshots/`.

## Evolution contract

The `/v1` in each schema ID is the major version. Additive changes raise the minor version
(`SCHEMA_MINOR`, stamped as `schema_minor` in the manifest body and `aisim.schema_minor` in every Parquet file).

| Change | Within a major version |
|---|---|
| Add an optional field, column, metric name, phase or workload kind | Allowed; old readers ignore it |
| Add a must-have field | New major |
| Remove or rename a field, or change a type or unit | Not allowed; use a new field name |
| Unknown field in a file | Kept (`model_extra`, `ext`, `attrs`) and written back unchanged |
| Missing optional column in a table | Reads as null |
| File from a newer minor | Loads, with a warning; unknown fields are ignored |

Rules for readers and writers:

- Models accept unknown fields (`extra="allow"`); required fields are strictly typed. Tables are matched by
  column name, so order and extra columns do not matter.
- Consumers (the dashboard and gym) read only known fields. Unknown fields, `ext` and `attrs` are carried,
  never read. Unknown fields seen in real data are reviewed at each schema upgrade and promoted to known fields.
- Absent never means zero. A missing metric is a row with `status="missing"`.
- Source-specific facts go under `ext.<namespace>` in the manifest (for example `agentx/1`) or `attrs` on rows.

## Canonical names

One canonical name per concept; stored data uses only canonical names (`b300_sxm`, `sglang`, `ttft`, `median`,
`profile`). The registry lists the spellings sources use (`B300`, `dynamo-sglang`, `p50`, `preparation`) as
**aliases for converters, collectors and agents**; readers never translate them.

| Concept | Canonical examples | Notes |
|---|---|---|
| hardware | `b300_sxm`, `h200_sxm`, `gb300` | AISim system name, as op-latency and FPM identities use |
| framework | `sglang`, `vllm`, `trtllm` | Engine only; `dynamo-sglang` splits into `framework=sglang`, `orchestrator=dynamo` |
| precision | `bfloat16`, `fp8`, `nvfp4` | Spellings the published FPM identities use |
| model | Hugging Face repo ID | Unknown models are accepted unchanged |
| metric | `ttft`, `tpot`, `itl`, `e2el`, `output_throughput` | One canonical unit each (`ms`, `tokens/s`, ...) |
| statistic | `mean`, `median`, `p90`, `std` | `p50` is an alias of `median` |
| phase | `warmup`, `profile` | `preparation` is an alias of `warmup` |

Check a directory before a pull request with [`aisim data check-names`](cli.md#check-canonical-names). A name that
is not canonical is still stored, but it is reported with the canonical spelling to use, and consumers ignore it
until the data is fixed. External contributors convert to canonical names first; the dataset's `AGENTS.md` and
`CONTRIBUTING.md` instruct a contributor's agent to do it.

## Layers

| Layer | File | Grain | Required |
|---|---|---|---|
| Manifest | `manifest.json` (envelope with the body in `provenance["e2e"]`) | one per configuration | yes |
| Observations | `observations.parquet` | one per run of a point | yes |
| Metrics | `metrics.parquet` | metric × statistic × scope per observation | yes |
| Requests | `requests.parquet` | one per request | optional |
| Series | `series.parquet` | metric × time interval | optional |
| Artifacts | `provenance/*` | raw output and logs, hash-linked, tiered | optional |

One configuration (one `configuration_id`, hashed from identity only) holds many observations, so lanes 1, 4,
8 of one serving setup sit side by side.

### Manifest body (`aisim-e2e-manifest/v1`)

| Field | Required | Notes |
|---|---|---|
| `origin` | yes | `official` or `self-collected` |
| `model_id`, `hardware`, `framework`, `framework_version`, `deployment`, `parallelism`, `parallelism_complete`, `precision` | yes | identity core; `framework_version` may be `unknown` with an evidence gap |
| `source_kind` | yes | producing tool: `inferencex`, `agentx`, `developer` |
| `orchestrator` | no | `dynamo`, `llm-d`, `none`; hashed when set and not `none` |
| `speculation` | no | method, draft length, acceptance mode |
| `dp_attention`, `multinode`, `offload_mode` | no | identity hash v3 fields |
| `settings` | no | `{name: {value, authority}}`; authority is `observed`, `producer_declared`, `source_verified_default` or `unknown` |
| `workload` | no | `kind` (required inside), `concurrency`, token lengths, `dataset`, `duration_s`, `warmup`, `boundary`, `cache_state`, `seed` |
| `implementation` | no | image digest, git revisions, model and config revisions |
| `source_ids` | no | the source's own spellings and IDs |
| `evidence_gaps` | no | `{field, reason}` for facts the source did not record |
| `ext` | no | `{namespace: {schema, ...}}` |

### Observations (`aisim-e2e-observations/v1`)

Required: `observation_id`, `configuration_id`, `record_sha256`, `source_kind`. Optional: producer and tool
identity, `collected_at`, `image`, `benchmark_type` (canonical workload kind), `concurrency`, token lengths,
request counts, `outcome`, `valid_for_submission`, `qualification`, `observer` (who timed the metrics),
`time_origin` (what request times count from), raw-artifact references and `attrs` (canonical JSON for scalars
with no column yet).

### Metrics (`aisim-e2e-metrics/v1`)

Long table. Required: `observation_id`, `metric`, `statistic`, `unit`, `status`. A valid row has a finite
`value` in the canonical unit; a missing or invalid row has none. Optional: `scope` (`run` by default),
`observer` (`client` or `server`, so client and engine TTFT never mix), `normalization`, `device_count`, and the
source audit fields. Scoring needs `ttft`, `tpot` and `e2el` with `mean`, `median` and `p90`; absent ones are
explicit `missing` rows.

### Requests (`aisim-e2e-requests/v1`)

Required: `observation_id`, `request_id`, `status` (`completed`, `failed`, `cancelled`). Optional, all nullable:
`phase`, `start_ms`, `end_ms`, `input_tokens`, `output_tokens`, `cached_tokens`, `ttft_ms`, `tpot_ms`, `itl_ms`,
`e2el_ms`, `queue_ms`, `error`, `identity_valid`, lineage keys (`conversation_id`, `turn_index`,
`parent_request_id`, `source_trace_id`, `source_outer_idx`, `source_inner_idx`, `agent_depth`) and `attrs`.

### Series (`aisim-e2e-series/v1`)

Required: `observation_id`, `metric`, `start_ms`, `end_ms`. Optional: `value`, `coverage`, `unit`, `source`,
`phase` (times count from the start of that phase). An absent series means unavailable; an observed zero is a
row with value 0. Replicas of a tensor-parallel group are never summed; `source` names the reporting rank.

## Where each source lands

| Source | Lands in |
|---|---|
| InferenceX model, hardware, framework, precision, parallelism, spec method | manifest identity core, `orchestrator`, `speculation`; spellings in `source_ids` |
| InferenceX `metrics.*` | metrics rows in canonical units; unmapped keys stay in `unmapped_metrics_json` |
| AgentX lanes, corpus, boundary, warmup | `workload`, observation `concurrency` and `attrs` |
| AgentX per-request observations | requests table |
| AgentX server-side series | series table |
| AgentX effective runtime settings | manifest `settings` with `observed` authority, `implementation` |
| AgentX feature list | `ext["agentx/1"]` |
| AgentX observed rank-0 parallelism (tp, pp, attention_dp, cp, dcp) | manifest `parallelism`, complete only when every GPU-multiplying degree is observed; their product is the observation and total-throughput `device_count` (null with a `device_count` gap otherwise) |
| AgentX official gaps (no effective settings, `cached_tokens` null) | `evidence_gaps`, null columns |

`aisim_core.data.agentx_convert.agentx_bundles_to_configurations` converts AgentX bundles; the run-level
statistics it writes are derived from the request rows and say so in `source_name`.
