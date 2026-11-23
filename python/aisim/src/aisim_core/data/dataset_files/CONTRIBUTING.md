# Contributing to the AISim dataset

The dataset holds measured data in one layout for four categories: `op-latency`, `fpm`, `e2e` and `eval`.
This guide applies to all of them. Coding agents: follow `AGENTS.md`.

## The naming rule

One canonical name per concept, everywhere in stored data. The names come from the registry
(`names_registry.json`): `b300_sxm`, `sglang`, `ttft`, `median`, `profile`. The registry also lists the
spellings sources use, so a converter (or an agent) can translate them. Readers never translate: a stored
name that is not canonical is reported, and the dashboard and gym ignore it until the dataset is fixed.

Why: a name is a join key. Two spellings of one GPU split a plot in two and silently drop data from a score.

Before you open a pull request, convert your data to canonical names and run:

```bash
aisim data check-names <your-data-directory>
```

An external contribution that still contains aliases will be sent back with the checker's output.

`check-names` reads configuration manifests. It does not read content-addressed `aisim-eval/v1` records, so an
eval-only directory can pass it with a non-canonical hardware or framework in `truth.identity`. For eval
records, convert the names by hand and confirm them with `aisim_core.data.names.normalize` before you commit.

## Schemas and compatibility

Every file declares a schema ID such as `aisim-e2e-metrics/v1`. Within a major version (`v1`):

| Change | Allowed |
|---|---|
| Add an optional field, column, or registered name | Yes (the schema minor increases) |
| Add a must-have field | No (needs a new major) |
| Remove or rename a field, or change its type or unit | No |
| Add a field the schema does not know | Yes; it is kept and ignored by the dashboard and gym until promoted |

Unknown fields seen in real data are reviewed at each schema upgrade and promoted to known fields. Until
then, keep them under `ext.<namespace>` (manifest) or `attrs` (rows).

## Per-category checklist

**op-latency**: one Parquet table per version directory, stamped `aisim-op-latency/v1`. `hardware` is the AISim
system name, `framework` an engine or a communication library (`sglang`, `vllm`, `trtllm`, `nccl`).

**fpm**: execution profiles under `data/fpm/<org>--<model>/<hardware>/<parallelism>/`. Identity fields
(`hardware`, `framework`, `precision`) use canonical names; precision values are the format names the
registry lists (`bfloat16`, `fp8`, `fp8_block`, `nvfp4`, ...).

**e2e**: a manifest (`origin`, `model_id`, `hardware`, `framework`, `framework_version`, `deployment`,
`parallelism`, `precision` are required), an observations table, a metrics table, and optionally requests
and series tables. Metrics are rows (`metric`, `statistic`, `value`, `unit`, `status`); a missing value is a
`missing` row, never zero. State who timed a metric (`observer`: `client` or `server`) when both exist.

**eval**: evaluation records follow `aisim-eval/v1`; hardware, framework and model names are canonical.

## Redaction and licensing

Do not commit credentials, prompts, raw token streams, run URLs, container digests or internal host names
to the public dataset. Data derived from a third party needs its source, revision and license recorded, and
redistribution must be allowed by that license.
