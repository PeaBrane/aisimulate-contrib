# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original read-only capture hooks for the pinned producer's live DAG.

API sources: SemiAnalysisAI/aiperf@754356e9a39acc6cc6afb242d123bb57c3fb6f75,
src/aiperf/timing/{trajectory_source,strategies/agentic_replay}.py;
Apache-2.0, NVIDIA Corporation & Affiliates. No upstream scheduling code copied.
"""

import dataclasses
import enum
import hashlib
import json
import os
from pathlib import Path

from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy

PRODUCER = "754356e9a39acc6cc6afb242d123bb57c3fb6f75"


def encode(value):
    if dataclasses.is_dataclass(value):
        return {field.name: encode(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


class CapturedAgenticReplayStrategy(AgenticReplayStrategy):
    """Export phase boundaries and recycle choices; delegate all timing unchanged."""

    def _capture(self, event, session=None):
        destination = os.environ.get("DSPARK_AGENTX_CAPTURE_DIR")
        if not destination:
            return
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        source = self.conversation_source
        trajectories = source.trajectories
        trace_ids = {t.conversation_id for t in trajectories} if session is None else {session.conversation_id}
        templates = root / "templates"
        templates.mkdir(exist_ok=True)
        template_files = {}
        for trace_id in sorted(trace_ids):
            path = templates / (hashlib.sha256(trace_id.encode()).hexdigest() + ".json")
            if not path.exists():
                ids = source._collect_trace_conversation_ids(trace_id)
                data = {key: encode(source._metadata_lookup[key]) for key in sorted(ids)}
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")
                temporary.replace(path)
            template_files[trace_id] = str(path.relative_to(root))
        row = {
            "schema": "dspark-agentx-capture-v1",
            "producer_revision": PRODUCER,
            "event": event,
            "phase": encode(self.config.phase),
            "templates": template_files,
            "trajectories": encode(trajectories) if session is None else None,
            "recycled_session": encode(session) if session is not None else None,
            "session_markers": dict(self._session_marker),
        }
        # One timing-manager process owns each file. No write occurs on the
        # per-token or ordinary per-request path; only boundaries and recycling.
        with (root / f"events-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    async def setup_phase(self):
        await super().setup_phase()
        self._capture("setup")

    async def finalize_phase(self):
        await super().finalize_phase()
        self._capture("finalize")

    def _build_session_for_trace(self, trace_id):
        session = super()._build_session_for_trace(trace_id)
        if session is not None:
            self._capture("recycle", session)
        return session
