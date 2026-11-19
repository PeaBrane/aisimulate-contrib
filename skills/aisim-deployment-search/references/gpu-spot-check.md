<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU spot-check of AISim picks

Source: `docs/sweeper/deployment-generation.md` in ai-dynamo/aisim (0.14.0).
Step 1 runs on CPU. Steps 2 to 4 need GPUs and run only after the user approves the picks, GPU count and duration.
They were not run during testing.

## 1. Render serving artifacts for one pick (CPU)

`recommendations/*.yaml` are prediction inputs, not deployments. Render a
selected candidate into Dynamo launch scripts with the generator API that
ships in the `aisim` wheel. Save as `render.py` and run
`python3 render.py ./aisim-recommendation <candidate_id> <head-node-ip>`;
take the candidate ID from the summarizer output. Use the head node's address
reachable from every serving node and the benchmark host. `127.0.0.1` is only
appropriate when the deployment and benchmark client all run on one host.

```python
import json
import sys
from dataclasses import replace
from pathlib import Path

from aisim.generator.api import generate_from_request
from aisim.generator.request import from_sweeper_candidate

out = Path(sys.argv[1])
candidate_id = sys.argv[2]
head_node_ip = sys.argv[3]
result = json.loads((out / "recommendation.json").read_text())
candidate = next(r for r in result["candidates"] if r["candidate_id"] == candidate_id)
assert candidate["status"] == "feasible"
request = from_sweeper_candidate(
    candidate,
    workload=candidate["provenance"]["workload"],
    deployment_target="dynamo-j2",   # or dynamo-python, llm-d-helm, llm-d-kustomize, slurm
    output_dir=str(out.parent / f"generated-{candidate_id}"),
    generator_overrides={
        "ServiceConfig": {
            "model_path": "/workspace/models/<model-dir>",   # path inside the serving container
            "served_model_name": "<hf-model-id>",
            "head_node_ip": head_node_ip,
        },
    },
)
# Set this to the Dynamo version of the runtime image you will launch.
request = replace(request, backend=replace(request.backend, dynamo_version="<dynamo version>"))
print(candidate_id, sorted(generate_from_request(request)))
```

It writes deployment artifacts:

- `run_0.sh`, `run_1.sh`, ...: the launch script for each serving node;
- `k8s_deploy.yaml`;
- `sflow.yaml`;
- per-role CLI argument files.

For a fixed-length workload, it also writes `bench_run.sh` (an AIPerf run)
and `k8s_bench.yaml`. Trace workloads have different benchmark requirements;
see the limitation in step 3.

Show the user any warning such as "No exact CLI args template for 0.30.0,
using closest prior template". It means the launch flags come from an
older backend template; check them against the runtime image before
launching. Confirm the runtime image named in `k8s_deploy.yaml`.

The bridge keeps the evaluated parallelism, worker counts and token limits,
and rejects what it cannot lower. Token budgets must be multiples of 64.
Run `bash -n` on every generated script.

## 2. Launch (GPU, approval required)

Before starting discovery services, record which etcd and NATS containers
are already running in the Compose project. Reuse them. Record the exact
container IDs of any services this spot-check starts, including containers
that existed but were stopped. Keep this list for cleanup; if etcd was
already running and only NATS was started, the list contains only NATS.

- Single host: start etcd and NATS from the matching Dynamo release's
  `dev/docker-compose.yml`. Run the Dynamo runtime image with the model
  mounted read-only and the complete generated bundle mounted (or copied)
  into the container, check the Dynamo and backend versions inside the
  container, then run `bash run_0.sh` from the bundle directory. Confirm
  `GET /v1/models` before benchmarking.
- Multiple hosts: inspect every generated `run_N.sh` and assign each script
  to its corresponding node, preserving the generated GPU placement. Render
  with a reachable head-node address, not loopback. Start etcd and NATS from
  the matching Dynamo release on that head node and make their endpoints
  reachable from every serving container. Copy the complete generated bundle
  to the same container path on every node, and mount the model at the same
  configured path with matching runtime/backend versions and the assigned GPUs.
  In each serving container, set the same discovery endpoints before launch:

  ```bash
  export HEAD_NODE_IP="<head-node-ip-reachable-from-all-nodes>"
  export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
  export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
  ```

  On node 0 run `bash run_0.sh`; on node 1 run `bash run_1.sh`; continue with
  each generated `run_N.sh` on node N. Keep all of them running concurrently;
  do not run every script on node 0 or run `run_0.sh` on every node. Allow the
  backend's generated inter-node communication ports as well as discovery.
  Confirm all expected workers register, `GET /v1/models` succeeds from the
  benchmark host, and a real completion succeeds before benchmarking.
- Kubernetes: install the matching Dynamo platform and CRDs, set the
  namespace, model cache and image in the generator overrides, and review
  `k8s_deploy.yaml` with the user before applying it.

## 3. Benchmark a fixed-length workload with the same arrival process

This spot-check cannot validate trace workloads, including mixed-length
traces using concurrency load. The manual commands below use one fixed
ISL/OSL pair; they do not replay the trace's per-request lengths or arrival
process.

By default, a trace candidate gets no `bench_run.sh` or `k8s_bench.yaml`, and
`sflow.yaml` has no benchmark task. Explicitly setting both `BenchConfig.isl`
and `BenchConfig.osl` generates fixed-length benchmarks, not trace replay;
those measurements do not validate the original trace prediction. Trace
validation requires a separate replay procedure preserving the original
per-request lengths and arrival process, which this guide does not provide.

The generated `bench_run.sh` runs AIPerf in closed loop and, by default,
sweeps multiple concurrency points. Each point sends 20 requests per unit
of concurrency; the script does not use `BenchConfig.num_requests` to set
that count. `AICONFIGURATOR_BENCH_CONCURRENCY` and
`AICONFIGURATOR_BENCH_MULTI_ROUND` can override the points and multiplier.
Compare the point matching the simulated concurrency. This sweep does not
validate a request-rate (Poisson or constant) simulation because the arrival
process differs.

For a single point using the simulated request count, run:

Closed loop (simulation used `concurrency: C`):

```bash
aiperf profile -m <served-model-name> -u http://<head-node-ip>:8000 \
  --endpoint-type chat --streaming --tokenizer <tokenizer-path-or-id> \
  --isl <input_tokens> --osl <output_tokens> --extra-inputs ignore_eos:true \
  --concurrency <C> --num-requests <requests in the simulation> \
  --warmup-request-count <2 x C> --artifact-dir ./bench-<candidate_id>
```

Open loop (simulation used `poisson` or `constant_rate` at R requests per
second):

```bash
aiperf profile -m <served-model-name> -u http://<head-node-ip>:8000 \
  --endpoint-type chat --streaming --tokenizer <tokenizer-path-or-id> \
  --isl <input_tokens> --osl <output_tokens> --extra-inputs ignore_eos:true \
  --request-rate <R> --arrival-pattern <poisson|constant> \
  --num-requests <requests in the simulation> \
  --artifact-dir ./bench-<candidate_id>
```

These AIPerf flags were checked against an AIPerf source checkout from
August 2026. Run `aiperf profile --help` on the benchmark host before use,
and keep AIPerf's raw outputs. Check the completed request count before
comparing latency.

## 4. Compare, then clean up

- Compare like with like: output tokens per second per GPU, mean TTFT,
  per-request TPOT (AIPerf's per-request average ITL matches AISim's TPOT)
  and E2E, completed requests.
- The simulation ran `--stack engine`, without Dynamo's frontend, router or
  planner; the measurement includes them. Report that as a limitation of
  the comparison.
- Stop the serving processes or delete the deployment when done. Also stop
  discovery containers recorded in step 2, using `docker stop <container-id>`
  for each recorded ID. If the list is empty, skip discovery cleanup. Leave
  pre-existing running services, networks and volumes in place. Tell the
  user which processes, deployments and containers were stopped or removed.
