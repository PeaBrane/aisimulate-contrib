# AgentX context anchors

Six native fixed-length anchors were submitted on 2026-10-15, chained after the
full fixed-length campaign on each GPU platform. Each collects 128 requests at
concurrency 1 and 4, using the reference K and synthetic AL, prefix caching off,
and a 1,048,576-token server context limit. Actual input/output remains 1024/1024.
These 1,536 requests are separate from the 196-configuration Phase A matrix.

| Model | Aggregated job | P/D job | Platform |
| --- | --- | --- | --- |
| V4-Pro-0813 | 2267427 | 2267428 | B200 |
| V4.1-Flash | 2267429 | 2267430 | B200 |
| Kimi-K3 + Inferact DSpark | 797448 | 797449 | B300 |

`plan.json` records topology, role budgets, source hashes and scripts.
`anchor-source.tar.gz` retains the exact launch bundle. `submitted-b200.json`
and `submitted-b300.json` retain scheduler receipts. All jobs were pending at
the parent queue's submission-verification snapshot.

Re-running `submit_agentx_anchors.py` with the same directory, platform, storage
root and `--after-job` resumes missing submissions. Original predecessors are
2267415 for B200 and 797447 for B300. The submitter rejects a changed plan or
an uncertain prior submission; reconcile scheduler records before retrying.

The 12 actual AgentX pilots and 54 one-hour profiles are **not submitted**.
Pilots require complete matching anchors, a staged pinned producer/corpus, and
prefix-enabled server receipts. Profiles additionally require reviewed pilots
with the matched lifecycle/cache contract. The CPU formatter witnesses cover all
three model families but are not serving requests or lifecycle validation.
See `../agentx-collection-readiness.json`, `../agentx_run.py`, and `../COLLECTION.md`.
