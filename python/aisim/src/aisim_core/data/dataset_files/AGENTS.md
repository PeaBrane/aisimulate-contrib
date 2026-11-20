# AGENTS: instructions for coding agents contributing to this dataset

You are preparing a pull request that adds or changes data in `ai-dynamo/aisim-data` (op-latency, FPM, e2e or
eval). Follow these steps in order. Read `CONTRIBUTING.md` for the reasons and the per-category checklists.

## 1. Use canonical names only

Stored data uses exactly one spelling per concept. The registry is `names_registry.json` in the AISim
package (`aisim_core/data/names_registry.json`, installed with `pip install aisim`). Source tools use
other spellings (`B300`, `dynamo-sglang`, `p50`, `preparation`); those are **aliases for your converter**, never
a reason to store them.

- Translate every name before writing a file: hardware, framework, orchestrator, precision, model, metric,
  statistic, phase, workload kind, request status, observer, origin.
- In Python use `aisim_core.data.names.normalize(kind, name)` (raises on an unknown name of a closed kind) and
  `split_framework(name)` for fused spellings such as `dynamo-sglang`.
- Do not invent a new spelling for something the registry already names. If a concept is genuinely new, store
  it, and say so in the pull request description (step 4).

## 2. Run the checker and fix every finding

```bash
aisim data check-names <your-data-directory>
```

Each finding gives the location, the stored value and the canonical name to use. For a known alias (a finding
with a `"canonical"` name) fix the data, not the registry. A finding with `"canonical": null` is a name the
registry does not know: if it is genuinely new (for example a new hardware name of a closed kind), register it
first (`names_registry.json`: name, kind, unit for a metric, aliases) in the same pull request, then rerun the
checker until it prints `"ok": true`.

## 3. Keep unknown fields, do not rely on them

Extra fields are allowed and are preserved, but the dashboard and gym read only fields the schema defines.
Put source-specific facts under `ext.<namespace>` (manifest) or `attrs` (rows), never in a field you made up.
Never remove or rename an existing field, and never change a field's type or unit.

## 4. Write the pull request description

- State the category, the source of the data and the tool versions.
- List every name you could not map, as a proposed registry addition (name, kind, unit if a metric, aliases).
- Confirm that `aisim data check-names` printed `"ok": true`.
- Do not include secrets, prompts, tokens, internal URLs or run URLs; follow the redaction rules in
  `CONTRIBUTING.md`.
