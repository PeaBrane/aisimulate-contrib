<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AISim review contract

Review for behavior and evidence, not just whether the diff looks reasonable.
Start at the changed input, follow the real consumer to the externally visible
output, and identify every contract crossed along the way. Prefer one precise
root-cause finding over several comments on its symptoms.

Use these priorities:

- **P0**: stop-ship critical security exposure, data loss, or broad outage,
  supported by concrete reachability and impact evidence.
- **P1**: a major regression in supported public behavior, materially wrong
  prediction, deadlock, release breakage, or unsafe compatibility change.
- **P2**: a bounded reachable defect, missing validation, performance regression,
  or insufficient evidence for a changed contract without demonstrated P0/P1
  consequences.
- **P3**: low-risk maintainability concern with a concrete future failure mode.

Do not report preference-only naming, formatting, or documentation nits already
covered by automated tools. Do not treat bot approval, `MERGEABLE`, CODEOWNERS,
DCO, or a small green check set as merge readiness. Technical findings and
governance status are separate; merge readiness is assessed on the exact head.

## Risk-tiered review and CI

Fast CI runs automatically for every PR, including drafts. CodeRabbit reviews
non-draft PRs automatically against **any target branch**, including feature,
stacked, and release branches, subject to its configured title and label
exclusions. Marking a PR ready enables review without waiting for its target to
become `main` or for a stack parent to merge. Neither requires a `review-ready` label. Medium- and high-risk
PRs also require a Codex review of the same commit. The risk level changes review
depth, not merge authority: every tier still requires the applicable CODEOWNER
approval.

### Risk selects reviewers

**Risk: Low / Medium / High** selects the review route. Align its meaning with
CodeRabbit's [Triage Risk](https://docs.coderabbit.ai/triage/your-triage-view#what-you-can-toggle):
an estimate combining review evidence with review effort. Consider review depth
alongside Risk when allocating human attention. CodeRabbit does not publish
precise Low/Medium/High scoring thresholds; do not invent vendor cutoffs or
replace this concept with a separate change-impact scale.

Use a verified current Triage assessment when available and identify its source.
Otherwise, CodeRabbit's PR summary must explicitly label its Risk as a
**review-routing estimate**, explain the evidence and effort behind it, and not
claim that the estimate is the native Triage score. Assess the complete PR
independently of the author's declared tier. Missing or inconclusive evidence
must not default to Low: require at least the Medium review route and ask the
relevant human owner to settle a possible High classification.

| Risk | Review before Full CI | Approval path when remaining Merge Risk is Minimal/Low |
| --- | --- | --- |
| Low | Fast CI and CodeRabbit | Eligible for automatic approval after required review |
| Medium | Fast CI, CodeRabbit, and current-head Codex | Eligible for automatic approval after both required reviews |
| High | Fast CI, CodeRabbit, and current-head Codex | Human approval from the relevant domain, architecture, security, or release owner |

These are AISim's routing and approval rules, not CodeRabbit's built-in actions.
Every tier still requires the applicable CODEOWNER approval and protected-branch
checks. An automatic approval counts toward a CODEOWNER requirement only if the
approving identity is actually eligible under those rules. High Risk continues
to require human approval even after remaining Merge Risk becomes Minimal/Low;
a low residual-risk rating cannot substitute for the required reviewers.

### Merge Risk selects the next action

[**Merge Risk**](https://docs.coderabbit.ai/pr-reviews/merge-risk#risk-levels)
assesses risk remaining in the whole PR at the reviewed commit. It is independent
of Low/Medium/High Risk. A High-risk change can have Minimal remaining Merge Risk
after adequate review and validation. Reconcile applicable inline findings,
custom pre-merge warnings, and uncovered behavior; no inline comments does not
establish absence of risk.

| Remaining Merge Risk | Required action |
| --- | --- |
| Minimal / Low | Eligible for the Risk-selected approval path after required reviews; then admit Full CI. Approval is not permission to merge. |
| Moderate | Hold approval. Use CodeRabbit fix commands or an authorized Codex agent to address findings, then rerun required bot reviews. |
| High / Critical | Hold approval. Obtain a human update on the issue and repair direction before agent fixes, then rerun required bot reviews. |

Use the highest supported remaining risk across the applicable reviews. As an
**AISim escalation policy**, an unresolved current-head Codex **P0** finding
sets a floor of **High** (or **Critical** when severe impact is demonstrated);
**P1** sets a floor of **Moderate**. These are conservative approval-blocking
floors, not an assertion that Codex priorities and CodeRabbit ratings are
identical. A more severe supported CodeRabbit assessment wins. Resolve all
P0/P1 findings before approval or Full CI admission; never inflate or downgrade
a finding just to fit a reporting surface or make the PR approvable.

After fixes, reassess the complete PR at its current head with CodeRabbit and,
for Medium/High Risk, Codex. A new commit invalidates the earlier review's
coverage. Missing, stale, or unavailable required review evidence holds approval;
it is not a passing result. For repair, `@coderabbitai fix comments` addresses
review findings, `@coderabbitai fix-ci commit` addresses CI failures, and
`@coderabbitai autofix` requests ongoing repair where available and authorized.
Instructions and fixability do not grant an agent authority to push or merge.

### Full CI failure revokes automatic approval

**Full CI fails → revoke auto-approval → fix → rerun bot review → reapprove.**

On Full CI failure for the reviewed PR revision, revoke/dismiss the automatic
approving review and withhold automatic reapproval. A late failure from an
older head must not revoke an approval for a newer revision. Do not dismiss
another reviewer's human approval as part of this automatic-approval rule.
After any fix, rerun CodeRabbit and the Risk-required Codex review on the current
commit.
Those reviews must finish **after the failure**, including when retrying CI
without changing the SHA. An earlier review or a green CI rerun alone cannot
reinstate automatic approval. `@coderabbitai full review` requests a new complete
review when the same commit has already been reviewed; use `@codex review` for
the required Codex GitHub review. Reapprove only after fresh reviews satisfy
the Risk-selected path with no blocking findings, then re-admit/rerun Full CI as
applicable. The final commit must pass all required checks before merge.

This section specifies the desired workflow. The repository instructions and
CodeRabbit prompts do not install a conditional Codex dispatcher, an approval
controller, or a Full CI failure/revocation handler. Until a trusted service
implements these transitions, a maintainer performs them and records exact-head
evidence. Do not report automatic routing, dismissal, repair, or approval as
completed without evidence from the responsible service or maintainer.

### Review evidence

Review evidence must identify its commit and source: verified current-head
checks, author-reported local results, pending work, or unavailable evidence.
Check accessible runs and linked results before claiming evidence is missing;
an inaccessible result is unverified, and an older-head result is stale. Full CI
pending initial-review admission is expected under the sequence below. Missing
Full CI or CODEOWNER approval remains a readiness condition; neither alone
proves a code defect. Concrete validation gaps for changed behavior still
count as technical risk. Bot ratings and warning checks advise reviewers;
they do not change the CI, CODEOWNER, or human merge gates above.
For review-policy-only changes, published vendor-schema validation, policy
checks, and link/attribution checks can provide the relevant evidence. Tests
that merely pin natural-language wording do not establish review quality.

CodeRabbit has no repository-specific path allowlist: current and future source
directories are eligible under its
[default path exclusions](https://docs.coderabbit.ai/configuration/path-instructions#default-ignored-paths).
Those defaults still skip binary, generated, lock, and some data files. Inspect
the ignored-files summary and require separate provenance, machine-readable
summaries, and relevant validation for skipped data or compatibility artifacts;
an eligible path is not proof that its changes were reviewed. The configured
[`fail_commit_status`](https://docs.coderabbit.ai/pr-reviews/status-check#fail-the-check-when-a-review-errors)
fails the CodeRabbit check on review errors. It does not fail skipped or
rate-limited reviews. Verify review completion and coverage at the current head
before approval; a successful or neutral status alone is insufficient.

Fast CI contains quick deterministic checks: source and legal policy, generated
CODEOWNERS integrity, lint, syntax compilation, whitespace, and Rust formatting.
Full CI contains the expensive multi-architecture dependency, Rust, Python,
public-API, feature-mode, engine-golden, platform-wheel, collector-data,
prediction-regression, build, and release-artifact tests. The FPE support
matrix remains a scheduled/manual product-support audit. Full CI's support-matrix
shard also exercises the required nightly FPE sentinels against its prebuilt wheel,
including all four phases and the release qualifier, to catch stale baselines and
native probe regressions before merge.
`Full CI Success` is required for every admitted PR, but trusted
`pull-request/*` copies select only the components affected by the pull
request's complete changed-file set. Renames classify both the old and new
paths. Documentation and review-policy-only changes may mark every expensive
component explicitly N/A; unknown paths and changes to CI execution contracts
run the complete matrix. Manual, `main`, and `release/*` runs also execute the
complete matrix. The aggregate gate accepts a skipped component only when the
selector explicitly marked that component N/A; missing selection outputs,
unexpected skips, failures, and cancellations fail closed.
The independently maintained mapping oracle in
`.github/full-ci-selection-cases.yml` records the job-consumer rationale and
representative expected plans; Fast CI verifies the implementation against
that complete component inventory.

Dispatch Full CI only after the required reviews have completed on the current
commit with no unresolved P0/P1 finding. Lower-priority findings and CODEOWNER
review may proceed while Full CI runs, but all required conversations,
approvals, and exact-head checks must be complete before merge.

`Fast CI Success` and `Full CI Success` are the stable merge-gate results. Both
run with `always()` semantics and fail when required evidence is missing,
skipped unexpectedly, canceled, or failed. Fast CI checks depend on job results,
not draft status or labels. Opening, updating, reopening, or marking a PR ready
for review triggers Fast CI; label changes do not. Standalone Fast CI runs
publish `Fast CI Success`. Full CI's
`Require Fast CI` job verifies a successful run and all substantive jobs on the
same branch and commit; it does not rerun Fast CI internally. Branch pushes
require matching Fast push evidence; manual Full CI accepts matching Fast push
or manual evidence. The bounded wait and API checks fail closed. Release staging
depends on successful validation and remains a separate protected step on `main` and `release/*`
lifecycle pushes. Waiting for staging approval does not hold `Full CI Success`
open; a green validation result does not certify staging or publication.
Require the direct `Fast CI Success` and aggregate `Full CI Success` results in
branch rules rather than
individual conditional or reusable-workflow jobs.

The additive ruleset payload and runner-image rollout procedure are in the
[CI guide](docs/ci/release.md#required-checks-and-release-approval). A committed ruleset
payload is not evidence that repository enforcement has been activated.

During the review-acceleration pilot, a maintainer dispatches Full CI after
verifying those conditions, supplying the reviewed full commit SHA through the
required `expected_sha` input. For manual validation, dispatch standalone Fast CI
with the same ref and SHA first, then dispatch Full CI. Trusted copy-pr-bot
`pull-request/*` branches also run Full CI automatically as a temporary coverage backstop while the `main`
ruleset does not require the exact-SHA checks; treat such a run as PR evidence
only after confirming that its copied SHA equals the PR head. Automatic runs on
`main` and `release/*` remain lifecycle validation outside the pre-merge
sequence. Do not remove the copy-branch backstop until the ruleset enforces the
Fast and Full CI checks and requires branches to be current. Do not claim
conditional Codex or post-review Full CI automation until an approved service
credential and exact-head dispatcher are installed.

For an admitted PR, no second Full CI launch is needed after Fast CI. The
trusted `pull-request/*` push starts standalone Fast CI and Full CI automatically,
and every expensive component waits for the exact-branch/SHA Fast CI prerequisite
and scope selector to pass before it can acquire a protected runner. Application
tests additionally build one wheel per architecture and then fan out contracts, unit, integration, CLI-build,
support-matrix, and tool-build shards. Admission itself remains the maintainer security gate;
do not replace it with PR-authored credentials or a `pull_request_target`
workflow.

The [application test inventory](docs/ci/README.md#application-test-inventory-and-exceptions)
maps collected cases to their Full CI shard and records explicit manual and
optional-dependency exceptions. The contracts shard fails when a collected test
has no assignment.

## Product invariants

- Python describes and orchestrates work. Rust computes per-operation latency,
  energy, and SOL values. Do not introduce a second performance oracle.
- Formulas, table selection, interpolation, quantization, fallback ordering, and
  performance data are product behavior. Changed answers require reproducible,
  explained before/after evidence; never refresh goldens merely to make tests
  pass.
- Public configuration, serialized schemas, CLI options, Python types/imports,
  Rust types/exports, defaults, docs, and fixtures must change together.
- Missing optional data may use an explicitly documented and observable absence
  path. Corrupt, ambiguous, or unsupported input must fail loudly rather than
  silently selecting another model, runtime, backend, kernel, or default.
- Keep provenance exact: distinguish measured values, estimates, proxies,
  synthetic fixtures, parity results, fake-runner results, and production
  traces. Do not turn parity into a predictive-accuracy claim.
- Binary parquet changes require a machine-readable summary of keys, shapes,
  units, row counts, coverage, and anomalies plus proof that a real consumer
  reaches the new data. A clean binary diff is not evidence.
- AISim is standalone. Dynamo is a downstream compatibility target, not a
  required runtime dependency. The release surface remains one `aisim`
  wheel and one `aisim-core` crate unless the artifact contract is changed
  deliberately.
- Only workflows under the repository-root `.github/workflows/` run for this
  repository. Imported workflows below `python/aisim/` are provenance, not
  hosted-CI evidence.

## High-value review paths

For schedulers and simulation state, inspect cancellation, preemption, draining,
retry, duplication, empty inputs, terminal events, queue fairness, time units,
and determinism. For performance modeling, compare cold, warm, and sweep paths;
selection rules and data-source precedence deserve the same scrutiny as
formulas. For cross-layer changes, require a test that exercises the final
consumer, not only object construction or an isolated mock.

Ask for the smallest evidence that would disprove the risky assumption: a
negative test, boundary case, exact command and result, explained golden diff,
held-out comparison, or production trace. When that evidence is unavailable,
say precisely what remains unmodeled or unverified.

## Codex review method

Review the complete PR against its actual base at the current head, including
earlier commits, renames, and deletions. Identify the reviewed head and base
where the reporting format allows it. Evidence for an older head is historical;
refresh it or state the limitation after a push or relevant base change.

Trace affected contracts through the final consumer and inspect nearby failure
paths and relevant tests. Check empty, invalid, boundary, and unsupported inputs;
state transitions and resource accounting; and producer/consumer compatibility
as applicable. Verify an apparent defect against the implementation and the
base behavior before reporting it. Existing problems, documented supported
absence paths, and intentional behavior changes are not introduced regressions
by themselves; flag a violated contract or concrete harmful consequence.

Each finding should identify the triggering conditions, affected consumer,
observable failure, and evidence supporting its priority. Anchor it to the
smallest useful changed line range and suggest the smallest adequate correction
or regression case. A direct code trace or independent numerical derivation can
establish a defect without executing it; do not claim a reproduction you did
not run. Prefer one root-cause finding over repeated symptoms, and do not repeat
another reviewer's finding without checking that it still applies.

A violated evidence requirement can itself justify a finding even when the
numerical or runtime failure is not yet established: name the required proof,
the changed contract, and the resulting validation gap. For example, missing
explained before/after evidence for a formula change can warrant P2. Missing
evidence alone does not establish P0/P1 impact; keep this distinction visible
on review surfaces that request the broader priority range.

Grade urgency by demonstrated impact and reachability. High-impact code is not
automatically a P1 defect, and a missing test alone does not establish P1 harm.
The [GitHub Codex reviewer](https://learn.chatgpt.com/docs/third-party/github#request-a-codex-review)
reports P0/P1 findings. Keep P2/P3 for review surfaces or tasks that request that
scope; never inflate priority to fit the GitHub reporting threshold. Follow the
surface's required output format rather than adding unsupported report fields.

Distinguish inspected code and verified current-head results from author claims,
pending checks, and unavailable evidence. Use proportional validation of the
affected consumer; tests that merely repeat implementation or prompt wording do
not establish correctness. If tools, dependencies, data, or CI logs are
unavailable, state the coverage limit instead of claiming a pass or a code
defect solely from the missing environment. Full CI pending initial-review
admission is expected under this contract. A report with no P0/P1 findings does
not certify absence of lower-priority defects, human approval, or merge readiness.
