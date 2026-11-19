# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find verified, matching RC build inputs; README tests always run afresh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import urlencode, urlsplit

ROOT = Path(__file__).resolve().parents[2]
OS_RELEASE = Path("/etc/os-release")
INPUTS = (
    ".github/workflows/readme-checks.yml",
    ".github/actions/wheel-storage/action.yml",
    ".github/actions/wheel-storage/transfer.py",
    "scripts/ci/readme_rc_wheels.py",
    "scripts/ci/build_readme_rc_wheels.sh",
    "scripts/readme/requirements.txt",
)


def api(endpoint: str) -> dict:
    result = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True, check=False)
    if result.returncode:
        # Do not echo authenticated CLI diagnostics or signed download URLs.
        raise RuntimeError("Unable to verify the RC wheel producer through GitHub")
    return json.loads(result.stdout)


def identity(root: Path = ROOT) -> dict:
    revisions = {key: os.environ[key] for key in ("AISIM_RC_REVISION", "DYNAMO_RC_REVISION")}
    if any(not re.fullmatch(r"[a-f0-9]{40}", value) for value in revisions.values()):
        raise ValueError("RC revisions must be full commit SHAs")
    image_reference = os.environ["RC_JOB_IMAGE"]
    if not image_reference or any(character in image_reference for character in "\r\n"):
        raise ValueError("RC job image must be identified")
    # Fingerprint the effective image after the native prerequisites are
    # installed, including OS/package changes under a mutable runner tag.
    image = {
        "reference": image_reference,
        "os_release_sha256": hashlib.sha256(OS_RELEASE.read_bytes()).hexdigest(),
        "packages_sha256": hashlib.sha256(
            subprocess.check_output(["dpkg-query", "--show", "--showformat=${Package}=${Version}\\n"])
        ).hexdigest(),
    }
    storage_repo = os.environ["RC_STORAGE_REPO"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", storage_repo):
        raise ValueError("RC storage repository must be configured")
    storage_url = os.environ["RC_STORAGE_URL"].rstrip("/")
    endpoint = urlsplit(storage_url)
    if (
        endpoint.scheme != "https"
        or not endpoint.netloc
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("RC storage URL must be an HTTPS service root")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("RC reuse requires a Linux AMD64 runner image")
    return {
        "schema": 1,
        "revisions": revisions,
        "image": image,
        "storage_repo": storage_repo,
        "storage_url": storage_url,
        "python": platform.python_version(),
        "rust": subprocess.check_output(["rustc", "-Vv"], text=True).strip(),
        "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "inputs": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in INPUTS},
    }


def handoff_name(build_identity: dict) -> str:
    key = hashlib.sha256(json.dumps(build_identity, sort_keys=True).encode()).hexdigest()
    return f"readme-rc-wheels-{key}"


def find_producer(name: str, env: dict, request=api) -> dict[str, str]:
    """Only main or this exact admitted PR head may provide a cross-run bundle."""
    repository = env["GITHUB_REPOSITORY"]
    repository_id = int(env["GITHUB_REPOSITORY_ID"])
    for page in range(1, 11):
        query = urlencode({"name": name + "-manifest", "per_page": 100, "page": page})
        artifacts = request(f"repos/{repository}/actions/artifacts?{query}")["artifacts"]
        for artifact in artifacts:
            producer = artifact.get("workflow_run", {})
            if (
                artifact.get("expired")
                or artifact.get("name") != name + "-manifest"
                or str(producer.get("id")) == env["GITHUB_RUN_ID"]
                or producer.get("head_repository_id") != repository_id
                or producer.get("repository_id") != repository_id
            ):
                continue
            run = request(f"repos/{repository}/actions/runs/{producer['id']}")
            if (
                run.get("status") != "completed"
                or run.get("conclusion") != "success"
                or run.get("path") != ".github/workflows/ci.yml"
                or run.get("event") not in {"push", "workflow_dispatch"}
                or run.get("repository", {}).get("id") != repository_id
                or run.get("head_repository", {}).get("id") != repository_id
                or run.get("head_sha") != producer.get("head_sha")
            ):
                continue
            if run.get("head_branch") == "main":
                return {"run-id": str(run["id"]), "scope": "preview", "pr-number": ""}
            if (
                run.get("head_branch") == env["GITHUB_REF_NAME"]
                and run.get("head_sha") == env["GITHUB_SHA"]
                and env["GITHUB_REF_NAME"] == f"pull-request/{env['RC_PR_NUMBER']}"
                and re.fullmatch(r"[1-9][0-9]*", env["RC_PR_NUMBER"])
            ):
                return {"run-id": str(run["id"]), "scope": "pr", "pr-number": env["RC_PR_NUMBER"]}
        if len(artifacts) < 100:
            break
    # No trusted producer, or a bounded discovery window was exhausted: rebuild.
    return {"run-id": "", "scope": "", "pr-number": ""}


def verify(directory: Path, expected: dict) -> None:
    if json.loads((directory / "rc-build-inputs.json").read_text()) != expected:
        raise ValueError("RC wheel build inputs do not match this job")
    wheels = list(directory.glob("*.whl"))
    names = []
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("RC wheel must contain exactly one package metadata file")
            names.append(BytesParser().parsebytes(archive.read(metadata[0]))["Name"])
    if len(wheels) != 3 or sorted(name.lower().replace("_", "-") for name in names) != [
        "ai-dynamo",
        "ai-dynamo-runtime",
        "aisim",
    ]:
        raise ValueError("Expected exactly the matching AISim, Dynamo and Dynamo runtime RC wheels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    build_identity = identity()
    if args.mode == "verify":
        verify(args.directory, build_identity)
        return
    name = handoff_name(build_identity)
    producer = find_producer(name, dict(os.environ))
    args.directory.mkdir(parents=True, exist_ok=True)
    (args.directory / "rc-build-inputs.json").write_text(json.dumps(build_identity, indent=2) + "\n")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        for key, value in {"name": name, **producer}.items():
            output.write(f"{key}={value}\n")
    print(
        f"Pinned RC wheels: {'reuse verified producer ' + producer['run-id'] if producer['run-id'] else 'cold build'}"
    )


if __name__ == "__main__":
    main()
