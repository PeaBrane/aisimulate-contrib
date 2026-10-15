# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the legacy Kimi draft import bound to the canonical implementation."""

from aisimulate.sdk._compat import alias_module

alias_module(__name__, "aisimulate_core.sdk.speculation.kimi_draft")
