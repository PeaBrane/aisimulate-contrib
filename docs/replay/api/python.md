<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Python Replay API

The Python API compiles public configuration into a data-only runner contract.
Its `ReplaySpec` is not the native Rust JSON structure with the same name.
Use the compiler rather than translating fields between them by hand.

## Public configuration to runner

```python
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.replay import ReplayOutputRequirements

config = CorePredictionConfig.from_yaml("prediction.yaml")
spec = prediction_to_replay_spec(config)
factory = EngineReplayRunnerFactory()
factory.capabilities().require_compatible(spec)
runner = factory.create(worker_id=0)
try:
    report = runner.run(
        spec,
        output_requirements=ReplayOutputRequirements(
            include_raw_report=True,
            capture_per_request=True,
        ),
    )
finally:
    runner.close()
```

The configuration above is the same core YAML used by `predict --stack engine`.
`CorePredictionConfig` does not accept optional Router/Planner top-level sections;
the CLI resolves and compiles those through their selected configuration adapters.
The compiler signature is:

```python
prediction_to_replay_spec(
    config: CorePredictionConfig,
    *,
    adapter_specs: dict[str, AdapterReplaySpec] | None = None,
    afd_performance_model: AFDPerformanceModel | None = None,
    execution_mode: str = "offline",
) -> ReplaySpec
```

Compilation resolves concrete traffic, topology, capacity and per-role estimator
identities. `ForwardPassEstimatorSpec.config` is the sole estimator identity;
its convenience properties are projections. Do not independently edit a
model/backend/version property after materialization. Configuration and
capability validation precede native execution but do not guarantee measured
accuracy or availability of every timing query.

## Inputs and reports

The [runner ABI](../../adapters/runner-abi.md) defines `ReplaySpec`,
`BackendDeploymentSpec`, `RunnerFactory`, `RunnerCapabilities`, and `Runner`.
`spec.workload` contains concrete trace or synthetic inputs; `spec.concurrency`
is a closed-loop cap or `None` for open loop. `spec.goal` carries evaluation
requirements. Adapter data and runtime hooks are serialized values, not live
policy objects.

`ReplayReport.metrics` supplies numeric values used by scoring. Non-power
metrics must be finite numbers; only designated power fields can be null.
`metadata` retains provenance and optional native detail. Engine runner raw
reports are retained under `metadata["native_report"]` when requested; do not assume
that every optional runner exposes identical metadata. Agentic results retain
`agentic_qualification: functional_only` and phase/snapshot/profile evidence.
See [result definitions](../../getting-started/understand-results.md) before
comparing latency or throughput across runs.

`ReplayOutputRequirements` defaults to summary output. It can request a raw
report, per-request capture, memory diagnostics or performance diagnostics.
These controls retain additional evidence without changing the workload.
Telemetry fields are part of the shared contract, but the built-in Engine
JSON runner currently rejects `capture_telemetry=True`; the native Rust
observer is a separate interface. Analytical EPD does not provide raw native
or per-request EPD reports. Large capture can require substantial memory.

## Exact token workloads

After compiling the deployment, replace `spec.workload` to replay measured
prompt and output token IDs through the same runner. This is a `ReplaySpec`
Python API input; it is not a new public YAML traffic source.

```python
from dataclasses import replace

# `spec` is the compiled ReplaySpec from the example above.
spec = replace(
    spec,
    workload={
        "requests": [
            {
                "id": "measured-request-0",
                "arrival_time_ms": 0.0,
                "input_token_ids": [42, 73, 91],
                "output_tokens": 2,
                "output_token_ids": [105, 106],
                "metadata": {"measurement_id": "record-0"},
            },
        ],
        "replay_concurrency": 2,
    },
    concurrency=2,
)
runner = EngineReplayRunnerFactory().create(worker_id=0)
try:
    report = runner.run(
        spec,
        output_requirements=ReplayOutputRequirements(
            include_raw_report=True,
            capture_per_request=True,
        ),
    )
finally:
    runner.close()
```

| Field | Contract |
| --- | --- |
| `requests` | Nonempty list of independent requests. Inline workloads accept only this field and optional `replay_concurrency`. |
| `id` | Optional unique, nonempty string; defaults to `request-<index>`. |
| `arrival_time_ms` | Required finite, nonnegative timestamp in milliseconds. Used for open-loop arrival scheduling. |
| `input_token_ids` | Nonempty list of unsigned 32-bit integers, preserved exactly. Input length is derived from the list. Supply exactly one of this field or positive `input_tokens`. |
| `output_tokens` | Required positive output length. |
| `output_token_ids` | Optional unsigned 32-bit token plan, preserved exactly. Its length must equal `output_tokens`; omitting it retains the existing generated-output behavior. |
| `replay_concurrency` | Optional positive integer limiting outstanding requests. Takes precedence over `spec.concurrency`; otherwise the runner uses `spec.concurrency`. |
| `session_id`, `metadata` | Optional correlation labels retained in per-request records. |

Token IDs reject booleans, fractional values, negative values, and values above
`2**32 - 1`. Both aggregated and P/D Engine Replay accept this workload.
With a concurrency cap, admission starts immediately and releases another
request when a slot becomes available; authored arrival times do not impose
additional delays. With neither cap set, arrival timestamps govern admission.
Use agentic trace inputs for session dependencies, forks, joins, and tool delays;
`session_id` on an inline request is correlation metadata.

Exact token plans preserve the supplied cache identity and generated sequence.
Measured comparison still requires matching deployment, acceptance, timing,
cache policy, and metric definitions. Replacing the workload retains the
compiled performance-model configuration and topology.

## Lifecycle and errors

One runner is reusable within one worker. The production Engine factory loads
`aisimulate._runtime` lazily and has no Dynamo dependency. A missing extension
raises `RunnerUnavailableError`; an invalid runtime or unsupported runtime
combination raises the runner's validation error. `close()` is currently a
no-op for this built-in runner; consumers must still call it for other runners.

Low-level calls do not automatically provide the CLI's subprocess host-memory
supervision. The caller owns resource budgets, output publication and failure
handling, especially for long agentic profiles. Use
[local resource controls](../../reference/local-resources.md) when executing
through the CLI or guarded Sweeper. The native JSON bridge and private runner
lowering helpers are implementation interfaces, not replacements for this API.

Source: [compiler](../../../python/aisimulate/src/aisimulate/compiler.py),
[runner](../../../python/aisimulate/src/aisimulate/runner.py), and
[contract definitions](../../../python/aisimulate/src/aisimulate/sweeper/replay.py).
