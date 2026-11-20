<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR #896 CI evidence and limits

This is the source evidence record referenced by PR #896. It supersedes the
quantified savings estimates in the PR description: **approximately 18% and
16% less runner work, 1–3 seconds saved by skipping locks, and 1–4% less
elapsed time are unmeasured hypotheses, not demonstrated improvements.** No
net runner-work saving or elapsed-time speedup is established. The historical
69-second median and 93-second p95 are author-reported context; their original
95-run dataset is unavailable in this record and is not used to establish a
baseline or calculate an improvement.

Evidence was inspected on 2026-11-18 with the published implementation head at
`de273338aad2fdd5aae3811547b7d3ab16914408`, against base
`58ff4c94d25452a82612e6a266930d663fe67e21`.
[timings.json](timings.json) binds those revisions and the SHA-256 hashes of
the unchanged CI implementation and regression tests. Its historical samples
each name their own full head SHA, run ID, attempt, event, and runner. This
evidence-only correction does not claim hosted checks for its own later commit;
the supervisor must obtain those checks and fresh required reviews separately.
The earlier `daab5a5` evidence-only commit mentioned in the PR description was
absent from the published implementation head and is not used as proof here.

## Historical observations

The machine-readable record contains timestamp excerpts retrieved directly from
the GitHub Actions run and jobs APIs, plus derived totals and anomaly checks.
All four source runs completed successfully on attempt 1.

| Sample | Source runs | Head | Combined runner work | Fast CI job span |
| --- | --- | --- | ---: | ---: |
| Historical base | [Fast CI 44030668740](https://github.com/ai-dynamo/aisim/actions/runs/44030668740), [ownership 44030668730](https://github.com/ai-dynamo/aisim/actions/runs/44030668730) | `bcf75c84ac5a372eb71384aa225c8f0935fbd971` | 143 s | 62 s |
| Historical head 1 | [Fast CI 44033277784](https://github.com/ai-dynamo/aisim/actions/runs/44033277784) | `07e13fe41b03216f2024cd4b3f2f3275775eb247` | 110 s | 67 s |
| Historical head 2 | [Fast CI 44035761839](https://github.com/ai-dynamo/aisim/actions/runs/44035761839) | `1c1ae02e53dff05e1ccc205155f0165e2d3c1e37` | 130 s | 90 s |

Job duration is `completed_at - started_at` from the jobs API, at one-second
timestamp precision. Combined runner work sums every Fast CI job, including its
aggregate, plus the standalone ownership job for the historical base. Thus the
base is `34 + 56 + 23 + 4 + 26 = 143` seconds; the heads are
`23 + 61 + 22 + 4 = 110` and `20 + 23 + 84 + 3 = 130` seconds. These are
running-time observations, not billed time or attributable savings.

Fast CI job span runs from its earliest job start to its latest job completion.
It includes waits between dependent jobs and excludes initial dispatch delay;
it is neither the sum of job durations nor the API run-update interval. The
standalone base ownership job finished before the Fast CI aggregate.

Anomaly checks prevent interpreting the totals as a controlled comparison:

- Job creation-to-start delays range from 1 to 3 seconds. This is a dispatch/
  queue proxy, not a pure queue-time measurement; it cannot control runner speed.
- Repository-policy checkout varies from 13 to 25 seconds, Rust toolchain setup
  from 8 to 15 seconds, and workflow-contract execution from 26 to 33 seconds.
  Jobs used distinct runners labeled `ubuntu-24.04`; image versions, cache
  state, and setup conditions were not controlled.
- The historical base used `push`; the two heads used `pull_request`. Their
  contents also differ. The historical base is not the current PR base above.
  There are no repeated pairs using the same unrelated change.
- All three Fast CI samples ran lock regeneration successfully, taking 1, 1,
  and 3 seconds. Those observed step durations do not measure the net saving
  from skipping the step, including selector and added-test costs. There is
  no hosted sample of the skip path.

The hypothesis-labeling option in the review is used here. Supporting a future
quantified claim requires repeated paired base/head runs of the same unrelated
change, with identical events and runner/setup conditions, both full source
SHAs, run IDs, per-job/step timestamps, confirmed skip execution, and queueing/
runner-variance analysis. No such campaign has been run or inferred from
pass/fail tests.

To retrieve the original metadata, use the run IDs in the table:

```sh
gh api repos/ai-dynamo/aisim/actions/runs/RUN_ID
gh api repos/ai-dynamo/aisim/actions/runs/RUN_ID/attempts/1/jobs --paginate
```

## Behavioral verification and remaining gates

The existing [selector regression tests](../../../tests/test_fast_ci_policy.py)
remain unchanged. Their temporary Git histories provide expected skip/regenerate
outcomes for complete PR diffs, earlier commits, renamed/deleted locks, trusted
copies, missing/invalid history, manual runs, and lifecycle pushes. Ownership
tests cover identical/different merge trees and shadow ownership files. These
are behavioral evidence; a test pass is not timing evidence.

The focused source-contract command is:

```sh
python -m pytest -c /dev/null tests/test_fast_ci_policy.py tests/test_full_ci_fail_fast.py tests/test_ci_workflow_contracts.py -q -p no:cacheprovider --override-ini=addopts= --override-ini=filterwarnings=error
```

Use CPython 3.12 and the hash-verified `scripts/ci/requirements.txt` environment,
with its `bin` directory on `PATH` for test subprocesses. On workstations that
enable Git signing, fixture-only subprocesses need process-local
`GIT_CONFIG_COUNT=1`, `GIT_CONFIG_KEY_0=commit.gpgsign`,
`GIT_CONFIG_VALUE_0=false`; shared Git configuration and assertions stay intact.
No native wheel or model download is needed.

During this repair the command passed **615 tests in 12.23 seconds** with
CPython 3.12.3, the hash-verified CI dependencies, and process-local fixture
signing disabled as above. The implementation/test hashes in `timings.json`
match both the published implementation head and this correction's source
files. The raw timing excerpts, totals, anomaly ranges and hashes were checked
against the retrieved API payloads. These checks establish data integrity and
behavior, without measuring a performance improvement.

[Fast CI run 44378545546](https://github.com/ai-dynamo/aisim/actions/runs/44378545546)
was independently verified successful for the full implementation head
`de273338aad2fdd5aae3811547b7d3ab16914408`: `codeowners`,
`Python Static Checks`, `Rust Format`, and `Fast CI Success` all passed. Its
[policy job log](https://github.com/ai-dynamo/aisim/actions/runs/44378545546/job/135064149195)
records 821 workflow-contract tests and 107 ownership tests on each ownership
pass. This run is validation of that parent revision, not a paired benchmark or
validation of the corrective commit. No Full CI run for that head was returned
by the Actions API at inspection. The controller handles fresh reviews,
technical reapproval and Full CI admission; human approval and merge gates
remain in force.
