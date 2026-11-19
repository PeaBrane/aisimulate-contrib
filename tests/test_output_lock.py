# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One AISim run owns an output directory at a time."""

from __future__ import annotations

import contextvars
import json
import os
import select
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import aisim.output as output
from aisim.output import (
    OUTPUT_LOCK_FD_ENVIRONMENT,
    OutputDirectoryOwnership,
    adopt_inherited_output_lock,
    prepare_output_directory,
)

_LOCK_MESSAGE = "another AISim run is using output directory"

# Prepares (and so locks) a directory inside an ownership scope, reports, and
# holds it until a line arrives on stdin. Adopts a lock handed over by a parent.
_HOLDER = textwrap.dedent(
    """
    import sys
    from aisim.output import OutputDirectoryOwnership, adopt_inherited_output_lock, prepare_output_directory

    adopt_inherited_output_lock()
    with OutputDirectoryOwnership():
        try:
            prepare_output_directory(sys.argv[1], overwrite=False)
        except ValueError as exc:
            print("refused", exc, flush=True)
            sys.exit(2)
        print("locked", flush=True)
        sys.stdin.readline()
    """
)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    monkeypatch.delenv(OUTPUT_LOCK_FD_ENVIRONMENT, raising=False)
    monkeypatch.setattr(output, "_inherited_descriptor", None)


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop(OUTPUT_LOCK_FD_ENVIRONMENT, None)
    env.update(extra)
    return env


class _Holder:
    """Another process that runs prepare_output_directory on a directory."""

    def __init__(
        self,
        root: Path,
        *,
        env: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-c", _HOLDER, str(root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env if env is not None else _env(),
            pass_fds=pass_fds,
        )
        ready, _, _ = select.select([self.process.stdout], [], [], 60)
        assert ready, "holder did not report within 60 s"
        self.state = self.process.stdout.readline().strip()

    def release(self) -> None:
        self.process.stdin.write("\n")
        self.process.stdin.flush()
        assert self.process.wait(timeout=30) == 0

    def __enter__(self) -> _Holder:
        return self

    def __exit__(self, *exc) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=30)
        self.process.stdin.close()
        self.process.stdout.close()


def _blocked(root: Path) -> bool:
    """Whether another process is refused by the lock (a refusal for content means it got the lock)."""
    with _Holder(root) as other:
        if other.state == "locked":
            other.release()
        return other.state.startswith("refused") and _LOCK_MESSAGE in other.state


def test_a_running_holder_blocks_prepare_and_overwrite(tmp_path) -> None:
    root = tmp_path / "out"
    with _Holder(root) as holder:
        assert holder.state == "locked"
        (root / "prediction.json").write_text("{}\n")
        with pytest.raises(ValueError, match=_LOCK_MESSAGE):
            prepare_output_directory(root, overwrite=True)
        # The refused run removed nothing, even with --overwrite.
        assert (root / "prediction.json").read_text() == "{}\n"
        holder.release()
    prepare_output_directory(root, overwrite=True)
    assert not (root / "prediction.json").exists()


def test_a_held_lock_blocks_failed_run_cleanup(tmp_path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    (root / "execution-events.jsonl").write_text("")
    (root / "resource-runtime.json").write_text(json.dumps({"schema_version": 1, "status": "failed"}) + "\n")
    with OutputDirectoryOwnership():
        prepare_output_directory(root, overwrite=True)  # takes the lock; leaves nothing to reuse
        (root / "execution-events.jsonl").write_text("")
        (root / "resource-runtime.json").write_text(json.dumps({"schema_version": 1, "status": "failed"}) + "\n")
        assert _blocked(root)
        assert sorted(entry.name for entry in root.iterdir()) == [
            "execution-events.jsonl",
            "resource-runtime.json",
        ]
    # Released: the failed-run leftovers are now eligible for reuse.
    prepare_output_directory(root, overwrite=False)
    assert list(root.iterdir()) == []


def test_scope_holds_until_exit_and_is_reentrant(tmp_path) -> None:
    root = tmp_path / "out"
    with OutputDirectoryOwnership() as ownership:
        prepare_output_directory(root, overwrite=False)
        prepare_output_directory(root, overwrite=True)
        assert ownership.descriptor_for(root) is not None
        assert _blocked(root)
    assert not _blocked(root)


def test_another_scope_in_the_same_process_is_refused(tmp_path) -> None:
    root = tmp_path / "out"

    def unrelated_run() -> None:
        with OutputDirectoryOwnership(), pytest.raises(ValueError, match=_LOCK_MESSAGE):
            prepare_output_directory(root, overwrite=False)

    with OutputDirectoryOwnership():
        prepare_output_directory(root, overwrite=False)
        # A fresh context has no enclosing scope, like another run in this process.
        contextvars.Context().run(unrelated_run)


def test_nested_scope_reuses_the_enclosing_lock(tmp_path) -> None:
    # `onboard validate-fpm` prepares validation/prediction, then runs predict
    # in-process on it; the inner command must not refuse its own caller.
    root = tmp_path / "out"
    with OutputDirectoryOwnership() as outer:
        prepare_output_directory(root, overwrite=False)
        held = outer.descriptor_for(root)
        with OutputDirectoryOwnership() as inner:
            prepare_output_directory(root, overwrite=True)
            assert inner.descriptor_for(root) == held
        # The enclosing scope still owns the lock after the inner one ends.
        assert outer.descriptor_for(root) == held
        assert _blocked(root)
    assert not _blocked(root)


def test_refused_preparation_releases_its_lock(tmp_path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    (root / "prediction.json").write_text("{}\n")
    with OutputDirectoryOwnership() as ownership:
        with pytest.raises(ValueError, match="results from a finished AISim run"):
            prepare_output_directory(root, overwrite=False)
        assert ownership.descriptor_for(root) is None
        assert not _blocked(root)


def test_preparation_outside_a_scope_locks_only_while_preparing(tmp_path) -> None:
    root = tmp_path / "out"
    prepare_output_directory(root, overwrite=False)
    assert not _blocked(root)


def test_scope_tracks_each_directory_separately(tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    with OutputDirectoryOwnership() as ownership:
        prepare_output_directory(first, overwrite=False)
        prepare_output_directory(second, overwrite=False)
        prepare_output_directory(first, overwrite=False)
        held_first, held_second = (
            ownership.descriptor_for(first),
            ownership.descriptor_for(second),
        )
        assert held_first is not None and held_second is not None and held_first != held_second
        assert os.fstat(held_first).st_ino == os.stat(first).st_ino


def test_handed_over_lock_survives_the_parent(tmp_path) -> None:
    root = tmp_path / "out"
    ownership = OutputDirectoryOwnership()
    with ownership:
        prepare_output_directory(root, overwrite=False)
        descriptor = ownership.descriptor_for(root)
        os.set_inheritable(descriptor, True)
        worker = _Holder(
            root,
            env=_env(**{OUTPUT_LOCK_FD_ENVIRONMENT: str(descriptor)}),
            pass_fds=(descriptor,),
        )
    # The "supervisor" has released its copy; the worker still holds the lock.
    with worker:
        assert worker.state == "locked"
        assert _blocked(root)
        worker.release()
    assert not _blocked(root)


def test_variable_without_the_locked_descriptor_grants_nothing(tmp_path) -> None:
    root = tmp_path / "out"
    with OutputDirectoryOwnership() as ownership:
        prepare_output_directory(root, overwrite=False)
        descriptor = ownership.descriptor_for(root)
        # Same variable, but the descriptor was not passed: no exemption.
        with _Holder(root, env=_env(**{OUTPUT_LOCK_FD_ENVIRONMENT: str(descriptor)})) as stranger:
            assert stranger.state.startswith("refused")


def test_adopting_hides_the_variable_from_grandchildren(tmp_path, monkeypatch) -> None:
    root = tmp_path / "out"
    root.mkdir()
    descriptor = os.open(root, os.O_RDONLY)
    try:
        monkeypatch.setenv(OUTPUT_LOCK_FD_ENVIRONMENT, str(descriptor))
        adopt_inherited_output_lock()
        assert OUTPUT_LOCK_FD_ENVIRONMENT not in os.environ
        assert output._inherited_descriptor == descriptor
    finally:
        output._inherited_descriptor = None
        os.close(descriptor)


def test_symlinked_path_shares_the_lock(tmp_path) -> None:
    root = tmp_path / "out"
    alias = tmp_path / "alias"
    with _Holder(root) as holder:
        assert holder.state == "locked"
        alias.symlink_to(root, target_is_directory=True)
        with pytest.raises(ValueError, match=_LOCK_MESSAGE):
            prepare_output_directory(alias, overwrite=False)


def test_lock_adds_no_file_to_the_directory(tmp_path) -> None:
    root = tmp_path / "out"
    with OutputDirectoryOwnership():
        prepare_output_directory(root, overwrite=False)
        assert list(root.iterdir()) == []


@pytest.mark.parametrize("argv", [[], ["--help"]])
def test_cli_main_closes_an_inherited_descriptor_it_never_used(tmp_path, monkeypatch, argv) -> None:
    import aisim.main as cli

    root = tmp_path / "out"
    with OutputDirectoryOwnership() as ownership:
        prepare_output_directory(root, overwrite=False)
        descriptor = os.dup(ownership.descriptor_for(root))
    try:
        monkeypatch.setenv(OUTPUT_LOCK_FD_ENVIRONMENT, str(descriptor))
        with pytest.raises(SystemExit):
            cli.main(argv)
        with pytest.raises(OSError):
            os.fstat(descriptor)
        assert output._inherited_descriptor is None
        assert not _blocked(root)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
