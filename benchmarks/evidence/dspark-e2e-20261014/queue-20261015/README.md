# Full fixed-length collection queue

## Status at 2026-10-15 23:14 UTC

- B200: all 37 jobs pending. The first job, 2267383, waits on priority;
  the other 36 wait on dependencies.
- B300: pilot 797431 failed after 1h40m55s when server startup exceeded
  6,000 seconds. Its last server messages are NCCL distributed initialization,
  before reported model-weight loading. The underlying cause is unresolved.
  It produced no measured requests or frozen K3 acceptance control.
- The other 18 B300 jobs remain pending, led by 797432 waiting on priority.
  Independent real/K2 cells can proceed; K3 synthetic cells still require the
  missing pilot control. No retries or topology substitutions are claimed.
- Total: 55 pending, one failed, none running, and no new measured pairs.
  The existing 13 paired runs / 1,664 requests and MAPE are unchanged.
  Raw logs, identity and scheduler output are under `status-20261015T2314Z/`.

This queue expands the approved matrix to all 196 configurations: input/output
1024/1024 and 8192/1024, concurrency 1/4/16/64, both draft widths and acceptance
modes, and three repetitions. Pro P/D additionally includes concurrency 8 at
8192/1024. Each window measures `max(128, 10 * concurrency)` requests.

The frozen plan excludes 13 already qualified runs (1,664 requests), leaving
575 production runs and 151,936 measured requests. Its 48 sweep jobs cover all
remaining cells; two independent 128-request real-acceptance pilots freeze the
V4.1 K3 and Kimi K3 synthetic controls. Pilots, startup smokes and warmup are
excluded from production totals. Pro K3 uses the independently frozen pilot
control in `../synthetic-control-v4-pro-k3.json`.

All 50 fixed-length jobs were accepted by Slurm on 2026-10-15. Four completed
configurations need no repeats; 192 configurations retain at least one pending run.

| Platform | Sweep + acceptance-pilot jobs | Job IDs | AgentX context anchors |
| --- | ---: | --- | --- |
| B200 | 33 | 2267383–2267415 | 2267427–2267430 |
| B300 | 17 | 797431–797447 | 797448–797449 |

`submission-verification.json` verifies all 56 jobs in the scheduler. At that
snapshot all were pending: the first job on each platform waited for priority
or resources; subsequent jobs waited on their predecessors. The signed source
commit and PR CI are independent of GPU collection completion. Twelve AgentX
pilot windows and 54 full profiles remain unsubmitted until their explicit
anchor, prefix/lifecycle and reviewed-pilot prerequisites are satisfied.

## Reproduce or resume

The source archive and `bundle-manifest.json` retain the exact submitted source
bytes, separately from the base Git revision. `plan.json` hashes every job script.
On each cluster, the campaign lives under the user's approved storage root:

```bash
python3 "$storage_root/jobs/dspark-full-20261015/bundle/submit_matrix.py" submit \
  --directory "$storage_root/jobs/dspark-full-20261015/queue" \
  --platform b200 --storage-root "$storage_root"
```

Use `b300` on the B300 cluster. The submitter preserves a receipt after every
accepted job and resumes only missing submissions. An uncertain submission is
blocked for scheduler reconciliation. Never delete its state to retry blindly.

Each platform uses one `afterany` chain, so at most one campaign GPU job runs per
platform. P/D roles share one allocation. A failed group does not block unrelated
groups. Every sweep first runs a native one-request smoke; missing additional-K
pilot controls fail before serving starts. Four-hour jobs preserve completed
windows when later windows fail or time out. Such windows must be reconciled
against actual raw records before preparing retries; submission is not completion.

## Qualification boundaries

CPU preflight verifies source hashes, exact checkpoint identities and every selected
window. The pinned image's CLI parser initializes a device and is unavailable on
CPU nodes; each GPU startup smoke checks the actual role CLI instead. CPU checks
do not establish GPU runtime support or timing accuracy. Initial preflight attempts
failed on a relocated vLLM CLI import and device inference; their scheduler/log
receipts are retained separately.

Missing simulator op tables do not prevent native E2E collection. High-concurrency
P/D records require fresh cache/lifecycle qualification before scoring; the prior
concurrency-1/4 source-pool bound is not a high-concurrency proof. AgentX context
anchors, pilots and profiles are tracked separately in
`../phaseb-queue-20261015/` and `../agentx-collection-readiness.json`.
