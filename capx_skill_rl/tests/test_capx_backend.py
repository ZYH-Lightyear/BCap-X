from __future__ import annotations

from typing import Any

import numpy as np

from capx_skill_rl.backends.capx import (
    LIBERO_PRO_SUITES,
    CapXLiberoBackend,
    LiberoBackendConfig,
    create_libero_backend,
)


class FakeLiberoEnv:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False

    def reset(self, seed=None):
        return raw_observation(), {"task_prompt": "put the bowl on the stove"}

    def task_completed(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


class FakeLiberoApi:
    def __init__(self, env: FakeLiberoEnv) -> None:
        self.env = env
        self.vlm_config: dict[str, Any] | None = None

    def configure_vlm_backend(self, **kwargs: Any) -> None:
        self.vlm_config = kwargs

    def get_observation(self):
        return raw_observation()

    def solve_ik(self, position, quaternion_wxyz, *, return_info=False):
        assert return_info is True
        return np.arange(7), {"orientation_used": "requested"}


def raw_observation() -> dict[str, Any]:
    return {
        "agentview": {
            "images": {
                "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "depth": np.ones((4, 4, 1), dtype=np.float32),
            },
            "intrinsics": np.eye(3),
            "pose_mat": np.eye(4),
        },
        "privileged_object_pose": np.ones(7),
    }


def test_factory_passes_libero_pro_suite_and_task_without_privilege() -> None:
    seen: dict[str, Any] = {}

    def env_factory(**kwargs: Any):
        seen.update(kwargs)
        return FakeLiberoEnv(**kwargs)

    backend = create_libero_backend(
        LiberoBackendConfig(suite_name="libero_spatial_task", task_id=2),
        env_factory=env_factory,
        api_factory=FakeLiberoApi,
    )
    assert seen == {
        "suite_name": "libero_spatial_task",
        "task_id": 2,
        "privileged": False,
        "max_steps": 8000,
    }
    assert backend.api.vlm_config == {
        "model": "vapi/gpt-5.5",
        "server_url": "http://localhost:8110/chat/completions",
        "api_key": None,
        "coord_space": "pixel",
    }


def test_backend_exposes_only_agentview_frame_and_task_prompt() -> None:
    backend = CapXLiberoBackend(FakeLiberoEnv(), FakeLiberoApi(FakeLiberoEnv()))
    frame, task = backend.reset(seed=3)
    assert task == "put the bowl on the stove"
    assert frame.rgb.shape == (4, 4, 3)
    assert frame.depth.shape == (4, 4)
    assert not hasattr(frame, "privileged_object_pose")


def test_all_six_libero_pro_suites_are_supported() -> None:
    assert LIBERO_PRO_SUITES == (
        "libero_object_swap",
        "libero_object_task",
        "libero_goal_swap",
        "libero_goal_task",
        "libero_spatial_swap",
        "libero_spatial_task",
    )


def test_backend_rejects_silent_ik_orientation_fallback() -> None:
    api = FakeLiberoApi(FakeLiberoEnv())
    backend = CapXLiberoBackend(api.env, api)
    joints = backend.solve_ik(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(joints, np.arange(7))

    def fallback(position, quaternion_wxyz, *, return_info=False):
        return np.arange(7), {"orientation_used": "top-down"}

    api.solve_ik = fallback
    try:
        backend.solve_ik(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    except RuntimeError as exc:
        assert "backend fallback: top-down" in str(exc)
    else:
        raise AssertionError("silent IK fallback was accepted")
