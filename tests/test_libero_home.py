import numpy as np
import pytest

from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced


def _api_with_env(env) -> FrankaLiberoApiReduced:
    api = FrankaLiberoApiReduced.__new__(FrankaLiberoApiReduced)
    api._env = env
    return api


class _HomeEnv:
    def __init__(self, statuses):
        self.home_joint_position = np.arange(7, dtype=np.float64)
        self.statuses = list(statuses)
        self.calls = []

    def move_to_joints_blocking(self, joints, **kwargs):
        self.calls.append((np.asarray(joints).copy(), dict(kwargs)))
        if self.statuses:
            return self.statuses.pop(0)
        return {"converged": True, "final_error": 0.0}


def test_goto_home_joint_position_retries_until_converged() -> None:
    env = _HomeEnv([
        {"converged": False, "final_error": 0.04},
        {"converged": True, "final_error": 0.002},
    ])

    _api_with_env(env).goto_home_joint_position(tolerance=0.006, max_steps=123, retries=2)

    assert len(env.calls) == 2
    joints, kwargs = env.calls[0]
    assert joints.tolist() == env.home_joint_position.tolist()
    assert kwargs == {
        "tolerance": 0.006,
        "max_steps": 123,
        "settle_steps": 8,
        "strict": False,
    }


def test_goto_home_joint_position_raises_when_not_converged() -> None:
    env = _HomeEnv([
        {"converged": False, "final_error": 0.05},
        {"converged": False, "final_error": 0.03},
    ])

    with pytest.raises(RuntimeError, match="did not reach home"):
        _api_with_env(env).goto_home_joint_position(tolerance=0.006, max_steps=20, retries=1)

    assert len(env.calls) == 2
