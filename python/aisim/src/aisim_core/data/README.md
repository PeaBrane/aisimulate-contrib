# `aisim_core.data`

Library behind the unified AISim dataset (`ai-dynamo/aisim-data`). It defines the layout and schemas,
fetches and verifies data, and loads it into predictors. Never authenticates, publishes or runs stored scripts.

Categories: `op-latency`, `fpm`, `e2e`, `eval`. User-facing docs: [docs/reference/cli.md](../../../../../docs/reference/cli.md).

## Layout

| Group | Files | Role |
|---|---|---|
| Contract | `__init__.py`, `registry.py`, `parquet.py`, `schemas.py`, `schema_snapshots/` | Manifest, index and identity models; Parquet read/write; JSON Schema snapshots (generated) |
| Names | `names.py`, `names_registry.json`, `names_check.py` | One canonical name per concept, plus source aliases for converters |
| Catalog | `catalog.py`, `systems.py`, `repos.py` | Coverage catalog, content-addressed system specs, retired-repo aliases |
| Fetch | `fetch.py` | Plan, download, hash-verify and cache a selection at a pinned revision |
| Load | `ensure.py`, `loaders.py`, `op_tree.py`, `materialize.py`, `fpm_layout.py` | Turn verified data into predictor inputs; `ensure_data` is the hidden step in `best_available` |
| Op-latency | `op_schema.py`, `op_tree.py` | Row models and the composed `systems/` root |
| E2E and eval | `e2e.py`, `agentx_convert.py`, `eval.py`, `gpu_time.py` | Tolerant e2e tables, AgentX bundle conversion, accuracy records, GPU-time records |
| PR validation | `validate.py` | Trusted-workflow validation of a contributor's candidate |
| Contributor files | `dataset_files/` (`AGENTS.md`, `CONTRIBUTING.md`) | Shipped to the dataset repo root |

## Flow

```
HF dataset ──fetch──▶ local cache ──loader──▶ predictor input
 (pinned SHA)       (hash-verified)     (systems root / FPM config)
```

- `ensure_data(config, pin)`: derive a selection from the config, fetch, verify, load, and wire the result in.
  - No pin means a no-op.
  - A category named by the pin never falls back to packaged data.
- `op-latency` composes every operation family for one system, backend and version, with the system specs, into one root.
- `fpm` selects one entry; several matches need `AISIM_DATA_CONFIGURATION`.
- `materialize_systems_root` writes a self-contained root for Rust and other consumers that never run Python.

## Usage

| Task | Command |
|---|---|
| Download a selection | `aisim data fetch --category op-latency --repo ai-dynamo/aisim-data --hardware h200_sxm --framework vllm --framework-version <v> [--dry-run] [--offline]` |
| Find non-canonical names | `aisim data check-names <dataset root, category dir or manifest.json>` |
| Regenerate or check schema snapshots | `python -m aisim_core.data.schemas --write` / `--check` |
| Build the op-latency publish tree | `python python/aisim/tools/perf_database/op_latency_publish.py --help` |
| Validate a PR candidate (CI only) | `python -m aisim_core.data.validate --help` |
| Pin a predictor to a dataset | `AISIM_DATASET=<repo>@<full SHA>`, or `AISIM_DATA_LOCAL=<dir>`; `AISIM_DATA_CATEGORIES` (default `fpm`) |

Other environment variables: `AISIM_DATA_CONFIGURATION`, `AISIM_DATA_MAX_BYTES`, `AISIM_DATA_OFFLINE`, `AISIM_DATA_CACHE`.

## Rules for changes

- Change a schema, then run `schemas --write` and commit the snapshots; CI fails on drift.
- Stored data uses canonical names only. Add aliases to `names_registry.json`, not to readers.
- Additive e2e changes bump the minor version; see the evolution contract in `e2e.py`.
- Tests: `python/aisim/tests/unit/data`. Run with `PYTHONPATH=python/aisim/src` so the shared venv imports this checkout.
