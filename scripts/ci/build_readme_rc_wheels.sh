#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

for entry in "aisim:${AISIM_RC_REVISION}" "dynamo:${DYNAMO_RC_REVISION}"; do
  repo="${entry%%:*}"
  revision="${entry#*:}"
  git init "$RUNNER_TEMP/$repo-rc"
  git -C "$RUNNER_TEMP/$repo-rc" fetch --depth 1 "https://github.com/ai-dynamo/$repo.git" "$revision"
  git -C "$RUNNER_TEMP/$repo-rc" checkout --detach FETCH_HEAD
done
uv build --wheel --out-dir "$AISIM_RC_WHEELHOUSE" --config-setting=build-args=--locked "$RUNNER_TEMP/aisim-rc/python/aisim"
uv build --wheel --out-dir "$AISIM_RC_WHEELHOUSE" "$RUNNER_TEMP/dynamo-rc"
uv build --wheel --out-dir "$AISIM_RC_WHEELHOUSE" --config-setting='build-args=--locked --features ais-forward-pass' \
  "$RUNNER_TEMP/dynamo-rc/lib/bindings/python"
