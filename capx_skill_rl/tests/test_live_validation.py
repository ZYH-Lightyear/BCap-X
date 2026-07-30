from __future__ import annotations

from typing import Any

import numpy as np

from capx_skill_rl.backends import LiberoBackendConfig
from capx_skill_rl.env import ToolEnv
from capx_skill_rl.scripts.validate_live import run_validation
from capx_skill_rl.tests.fakes import FakeBackend


class _FakeApi:
    def get_observation(self) -> dict[str, Any]:
        return {"robot_joint_pos": np.arange(8, dtype=np.float64)}


def test_live_validator_covers_complete_chain_and_uses_no_op_motion() -> None:
    backend = FakeBackend()
    backend.api = _FakeApi()
    env = ToolEnv(backend)
    config = LiberoBackendConfig()

    report = run_validation(
        env,
        backend,  # type: ignore[arg-type]
        config=config,
        query="object",
        seed=7,
    )

    assert report["passed"] is True
    assert report["suite_name"] == "libero_object_swap"
    assert report["task_id"] == 0
    assert all(check["ok"] for check in report["checks"])
    assert backend.reset_seeds == [7]
    np.testing.assert_array_equal(backend.moves[-1], np.arange(7))
    assert backend.capture_count == 4
