# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original fixture checks capture delegation against the actual pinned producer."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiperf.common.enums import CreditPhase
from aiperf.plugin import plugins
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import (
    ConversationState,
    Trajectory,
    TrajectorySnapshot,
)
from aisim_agentx_pin.capture import CapturedAgenticReplayStrategy


class HookTests(unittest.IsolatedAsyncioTestCase):
    async def test_delegates_and_captures_advanced_live_join_frontier(self):
        self.assertIs(
            plugins.get_class("timing_strategy", "agentic_replay"),
            CapturedAgenticReplayStrategy,
        )
        strategy = object.__new__(CapturedAgenticReplayStrategy)
        before = Trajectory(
            "root",
            1,
            TrajectorySnapshot(
                123.0,
                (
                    ConversationState("root", "root-session", 1, waiting_on_children=True),
                    ConversationState(
                        "child",
                        "child-session",
                        1,
                        parent_correlation_id="root-session",
                        join_gate_memberships=(("join", 2),),
                    ),
                ),
            ),
        )
        after = Trajectory(
            "root",
            2,
            TrajectorySnapshot(
                123.0,
                (
                    ConversationState(
                        "child",
                        "child-session",
                        2,
                        parent_correlation_id="root-session",
                        join_gate_memberships=(("join", 2),),
                    ),
                ),
            ),
        )
        source = SimpleNamespace(
            trajectories=[before],
            _metadata_lookup={
                "root": {"turns": [{}, {}, {}]},
                "child": {"turns": [{}, {}, {}]},
            },
            _collect_trace_conversation_ids=lambda _: {"root", "child"},
        )
        strategy.conversation_source = source
        strategy.config = SimpleNamespace(phase=CreditPhase.WARMUP)
        strategy._session_marker = {"root-session": "tree-prefix"}

        async def advance():
            source.trajectories = [after]

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, DSPARK_AGENTX_CAPTURE_DIR=folder),
        ):
            with patch.object(AgenticReplayStrategy, "setup_phase", new_callable=AsyncMock) as setup:
                await strategy.setup_phase()
                setup.assert_awaited_once()
            with patch.object(AgenticReplayStrategy, "finalize_phase", side_effect=advance) as finalize:
                await strategy.finalize_phase()
                finalize.assert_awaited_once()
            rows = [json.loads(line) for line in next(Path(folder).glob("events-*.jsonl")).read_text().splitlines()]
            self.assertEqual(rows[0]["trajectories"][0]["start_turn_index"], 1)
            self.assertEqual(rows[1]["trajectories"][0]["start_turn_index"], 2)
            state = rows[1]["trajectories"][0]["snapshot"]["states"][0]
            self.assertEqual(state["join_gate_memberships"], [["join", 2]])
            self.assertEqual(rows[1]["session_markers"], {"root-session": "tree-prefix"})
            self.assertEqual(len(list((Path(folder) / "templates").glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
