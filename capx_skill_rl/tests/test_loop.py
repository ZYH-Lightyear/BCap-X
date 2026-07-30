from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from capx_skill_rl.env import ToolEnv
from capx_skill_rl.loop import ToolExchange, run_episode
from capx_skill_rl.tests.fakes import FakeBackend


class ScriptedPolicy:
    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def act(
        self,
        *,
        task: str,
        rgb: np.ndarray,
        history: Sequence[ToolExchange],
        tools: Sequence[dict[str, Any]],
    ):
        self.seen.append(
            {
                "task": task,
                "rgb": rgb.copy(),
                "history": tuple(history),
                "tools": tuple(tools),
            }
        )
        if not history:
            return {
                "name": "vlm_point_detection",
                "arguments": {"query": "object"},
            }
        return {"name": "go_home", "arguments": {}}


def test_loop_keeps_reward_out_of_policy_context() -> None:
    policy = ScriptedPolicy()
    result = run_episode(ToolEnv(FakeBackend(), max_steps=2), policy, seed=11)

    assert len(result.transitions) == 2
    assert result.transitions[-1].done is True
    assert len(policy.seen) == 2
    assert set(policy.seen[0]) == {"task", "rgb", "history", "tools"}
    assert policy.seen[1]["history"][0].result == {"point": [1.0, 1.0]}
    assert policy.seen[1]["rgb"][0, 0, 0] == 0
    assert result.transitions[-1].next_observation.rgb[0, 0, 0] == 1
