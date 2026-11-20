# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Fast CI's skip decisions with real Git histories."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.ci.select_fast_ci_checks import main, select_locks

ROOT = Path(__file__).resolve().parents[1]
FAST = ROOT / ".github/workflows/fast-ci.yml"


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def commit(root, path, content="changed"):
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)
    git(root, "add", "--all")
    git(root, "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.name", "CI fixture")
    git(tmp_path, "config", "user.email", "fixture@example.com")
    base = commit(tmp_path, "README.md", "base")
    return tmp_path, base


def environment(base, target, *, event="pull_request", ref="refs/pull/42/merge"):
    return {"GITHUB_EVENT_NAME": event, "GITHUB_REF": ref, "BASE_SHA": base, "TARGET_SHA": target}


@pytest.mark.parametrize(
    "path",
    [
        "scripts/pyproject.toml",
        "scripts/ci/requirements.txt",
        "scripts/pages/requirements.txt",
        "scripts/new_group/requirements.txt",
        "scripts/ci/select_fast_ci_checks.py",
        "tests/test_fast_ci_policy.py",
        ".github/workflows/fast-ci.yml",
        "uv.toml",
        "scripts/uv.toml",
        "pyproject.toml",
    ],
)
def test_lock_inputs_always_validate(repository, path):
    root, base = repository
    head = commit(root, path)
    assert select_locks(root, environment(base, head))[0]


@pytest.mark.parametrize("path", ["docs/design.md", "crates/core/src/lib.rs", "scripts/ci/check_copyright.py"])
def test_unrelated_changes_skip_only_lock_regeneration(repository, path):
    root, base = repository
    head = commit(root, path)
    assert not select_locks(root, environment(base, head))[0]


def test_an_earlier_pr_commit_cannot_be_hidden_by_a_later_docs_push(repository):
    root, base = repository
    commit(root, "scripts/pyproject.toml")
    head = commit(root, "docs/design.md")
    assert select_locks(root, environment(base, head))[0]


def test_deleted_and_renamed_locks_still_validate(repository):
    root, _ = repository
    base = commit(root, "scripts/ci/requirements.txt")
    git(root, "mv", "scripts/ci/requirements.txt", "scripts/ci/old-lock.txt")
    git(root, "commit", "-qm", "rename")
    assert select_locks(root, environment(base, git(root, "rev-parse", "HEAD")))[0]


def test_trusted_copy_uses_the_whole_pr_instead_of_the_previous_push(repository):
    root, base = repository
    previous = commit(root, "scripts/pyproject.toml")
    head = commit(root, "docs/design.md")
    env = environment(previous, head, event="push", ref="refs/heads/pull-request/42")
    env["GITHUB_REPOSITORY"] = "ai-dynamo/aisim"
    calls = []

    def read_pr(repo, number):
        calls.append((repo, number))
        return {"head": {"sha": head}, "base": {"sha": base}}

    assert select_locks(root, env, pr_reader=read_pr)[0]
    assert calls == [("ai-dynamo/aisim", "42")]


@pytest.mark.parametrize("failure", ["api", "head-moved", "bad-base", "bad-ref"])
def test_unverifiable_copies_regenerate_instead_of_skipping(repository, failure):
    root, base = repository
    head = commit(root, "docs/design.md")
    env = environment(base, head, event="push", ref="refs/heads/pull-request/42")
    env["GITHUB_REPOSITORY"] = "ai-dynamo/aisim"
    if failure == "bad-ref":
        env["GITHUB_REF"] += "invalid"

    def read_pr(repo, number):
        if failure == "api":
            raise OSError("API unavailable")
        return {
            "head": {"sha": "a" * 40 if failure == "head-moved" else head},
            "base": {"sha": "missing" if failure == "bad-base" else base},
        }

    assert select_locks(root, env, pr_reader=read_pr)[0]


@pytest.mark.parametrize("base", ["", "0" * 40, "a" * 40, "main"])
def test_missing_invalid_or_unavailable_base_cannot_skip(repository, base):
    root, _ = repository
    head = commit(root, "docs/design.md")
    assert select_locks(root, environment(base, head))[0]


def test_manual_runs_regenerate_even_with_a_complete_unrelated_diff(repository):
    root, base = repository
    head = commit(root, "docs/design.md")
    assert select_locks(root, environment(base, head, event="workflow_dispatch"))[0]


def test_lifecycle_push_covers_every_commit_in_the_push(repository):
    root, base = repository
    commit(root, "scripts/pages/requirements.txt")
    head = commit(root, "docs/design.md")
    assert select_locks(root, environment(base, head, event="push", ref="refs/heads/main"))[0]


def test_output_is_appended_for_the_workflow(monkeypatch, tmp_path):
    output = tmp_path / "output"
    output.write_text("prior=value\n")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    main()
    assert output.read_text() == "prior=value\ndependency_locks=true\n"


def test_codeowners_is_provided_once_and_fast_ci_keeps_its_other_checks():
    jobs = yaml.safe_load(FAST.read_text())["jobs"]
    assert jobs["policy"]["name"] == "codeowners"
    assert not (ROOT / ".github/workflows/codeowners.yml").exists()
    assert set(jobs["readiness"]["needs"]) == {"policy", "python-static", "rust-format"}
    steps = {s.get("name"): s for s in jobs["policy"]["steps"] if "name" in s}
    assert steps["Check CODEOWNERS policy and generated artifacts"]["run"] == "bash scripts/ci/check_codeowners.sh"
    assert "if" not in steps["Check CODEOWNERS policy and generated artifacts"]
    assert steps["Check script dependency locks"]["if"] == "steps.script-locks.outputs.dependency_locks != 'false'"
    assert jobs["policy"]["steps"][0]["with"]["ref"] == "${{ env.PR_MERGE_SHA || env.TARGET_SHA }}"
    assert "git checkout --detach" in steps["Verify exact target"]["run"]
    for name in [
        "Check source and packaged legal files",
        "Check bundled configuration provenance",
        "Check documentation destinations",
        "Check active workflow contracts",
    ]:
        assert "if" not in steps[name]


@pytest.mark.parametrize("same_tree,fail", [(True, False), (False, False), (False, True)])
def test_proposed_merge_reuses_only_identical_contents(repository, same_tree, fail):
    root, base = repository
    check_log = root / "checks.log"
    head = commit(
        root,
        "scripts/ci/check_codeowners.sh",
        '#!/usr/bin/env bash\nset -e\ngit rev-parse HEAD >> "$CHECK_LOG"\n[[ ! -e bad-merge ]]\n',
    )
    if not same_tree:
        git(root, "checkout", "--detach", base)
        commit(root, "bad-merge" if fail else "base-update", "base changed")
        git(root, "merge", "--no-ff", "--no-edit", head)
    else:
        git(root, "commit", "--allow-empty", "-qm", "same contents")
    merge = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "--detach", head)
    steps = yaml.safe_load(FAST.read_text())["jobs"]["policy"]["steps"]
    step = next(s for s in steps if s.get("name") == "Check proposed merge CODEOWNERS")
    assert step["if"] == "github.event_name == 'pull_request'"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=root,
        env={**os.environ, "TARGET_SHA": head, "PR_MERGE_SHA": merge, "CHECK_LOG": str(check_log)},
        capture_output=True,
        text=True,
    )
    assert (result.returncode != 0) == fail, result.stderr
    if same_tree:
        assert not check_log.exists()
        assert git(root, "rev-parse", "HEAD") == head
    else:
        assert check_log.read_text().strip() == merge


@pytest.mark.parametrize("shadow", [".github/CODEOWNERS", "docs/CODEOWNERS"])
def test_shadow_ownership_files_fail_before_tests(repository, shadow):
    root, _ = repository
    script = root / "scripts/ci/check_codeowners.sh"
    script.parent.mkdir(parents=True)
    script.write_text((ROOT / "scripts/ci/check_codeowners.sh").read_text())
    path = root / shadow
    path.parent.mkdir(parents=True)
    path.write_text("* @wrong\n")
    result = subprocess.run(["bash", str(script)], cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    assert "only CODEOWNERS file" in result.stdout


def test_ownership_tool_directory_is_not_a_shadow_file(repository):
    root, _ = repository
    script = root / "scripts/ci/check_codeowners.sh"
    script.parent.mkdir(parents=True)
    script.write_text((ROOT / "scripts/ci/check_codeowners.sh").read_text())
    (root / ".github/codeowners").mkdir(parents=True)
    binaries = root / "bin"
    binaries.mkdir()
    python = binaries / "python"
    python.write_text('#!/bin/sh\necho "ownership tests reached"\nexit 23\n')
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        env={**os.environ, "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    # On a case-insensitive checkout, .github/CODEOWNERS resolves to this
    # directory. GitHub only discovers ownership files, not directories.
    assert result.returncode == 23
    assert "ownership tests reached" in result.stdout
