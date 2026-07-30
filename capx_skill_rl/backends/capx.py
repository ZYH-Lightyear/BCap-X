"""Thin adapter over CaP-X's existing LIBERO-PRO environment and reduced API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from capx_skill_rl.context import SensorFrame

LIBERO_PRO_SUITES = (
    "libero_object_swap",
    "libero_object_task",
    "libero_goal_swap",
    "libero_goal_task",
    "libero_spatial_swap",
    "libero_spatial_task",
)


@dataclass(frozen=True, slots=True)
class LiberoBackendConfig:
    suite_name: str = "libero_object_swap"
    task_id: int = 0
    max_sim_steps: int = 8000
    vlm_model: str = "vapi/gpt-5.5"
    vlm_server_url: str = "http://localhost:8110/chat/completions"
    vlm_api_key: str | None = None
    vlm_coord_space: str = "pixel"

    def __post_init__(self) -> None:
        if self.suite_name not in LIBERO_PRO_SUITES:
            raise ValueError(
                f"suite_name must be one of {LIBERO_PRO_SUITES}, "
                f"got {self.suite_name!r}"
            )
        if not 0 <= self.task_id < 10:
            raise ValueError("LIBERO-PRO task_id must be between 0 and 9")
        if self.max_sim_steps <= 0:
            raise ValueError("max_sim_steps must be positive")
        if self.vlm_coord_space not in {"pixel", "norm1000", "fraction", "auto"}:
            raise ValueError(
                "vlm_coord_space must be pixel, norm1000, fraction, or auto"
            )


class CapXLiberoBackend:
    """Adapt already-constructed CaP-X objects to the minimal backend protocol."""

    camera_name = "agentview"

    def __init__(self, env: Any, api: Any) -> None:
        self.env = env
        self.api = api

    def reset(self, seed: int | None = None) -> tuple[SensorFrame, str]:
        observation, info = self.env.reset(seed=seed)
        task = str(info.get("task_prompt") or "")
        if not task:
            raise ValueError("LIBERO-PRO reset did not return task_prompt")
        return self._frame(observation, revision=0), task

    def capture(self, revision: int) -> SensorFrame:
        return self._frame(self.api.get_observation(), revision=revision)

    def task_completed(self) -> bool:
        return bool(self.env.task_completed())

    def vlm_bbox_detection(self, rgb: np.ndarray, query: str) -> list[float]:
        return list(self.api.vlm_bbox_detection(rgb, query))

    def vlm_point_detection(self, rgb: np.ndarray, query: str) -> list[float]:
        return list(self.api.vlm_point_detection(rgb, query))

    def sam3_text(self, rgb: np.ndarray, text: str) -> list[dict[str, Any]]:
        return list(self.api.segment_sam3_text_prompt(rgb, text))

    def sam3_box(
        self,
        rgb: np.ndarray,
        bbox: list[float],
    ) -> list[dict[str, Any]]:
        return list(self.api.segment_sam3_box_prompt(rgb, bbox))

    def sam3_point(
        self,
        rgb: np.ndarray,
        point: list[float],
    ) -> list[dict[str, Any]]:
        return list(self.api.segment_sam3_point_prompt(rgb, tuple(point)))

    def get_obb(self, points_base: np.ndarray) -> dict[str, Any]:
        return dict(self.api.get_oriented_bounding_box_from_3d_points(points_base))

    def plan_grasp(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        poses, scores = self.api.plan_grasp(depth, intrinsics, mask)
        return np.asarray(poses), np.asarray(scores)

    def solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> np.ndarray:
        joints, info = self.api.solve_ik(
            position,
            quaternion_wxyz,
            return_info=True,
        )
        orientation_used = str(info.get("orientation_used") or "")
        if orientation_used != "requested":
            raise RuntimeError(
                "IK could not solve the requested orientation "
                f"(backend fallback: {orientation_used or 'unknown'})"
            )
        return np.asarray(joints)

    def move_to_joints(self, joints: np.ndarray) -> None:
        self.api.move_to_joints(joints)

    def open_gripper(self) -> None:
        self.api.open_gripper()

    def close_gripper(self) -> None:
        self.api.close_gripper()

    def go_home(self) -> None:
        self.api.goto_home_joint_position()

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def _frame(self, observation: dict[str, Any], revision: int) -> SensorFrame:
        camera = observation[self.camera_name]
        images = camera["images"]
        depth = np.asarray(images["depth"], dtype=np.float64)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        return SensorFrame(
            rgb=np.asarray(images["rgb"], dtype=np.uint8).copy(),
            depth=depth.copy(),
            intrinsics=np.asarray(camera["intrinsics"], dtype=np.float64).copy(),
            base_from_camera=np.asarray(
                camera["pose_mat"],
                dtype=np.float64,
            ).copy(),
            revision=revision,
        )


EnvFactory = Callable[..., Any]
ApiFactory = Callable[[Any], Any]


def create_libero_backend(
    config: LiberoBackendConfig,
    *,
    env_factory: EnvFactory | None = None,
    api_factory: ApiFactory | None = None,
) -> CapXLiberoBackend:
    """Construct the real backend; injectable factories keep offline tests light."""

    if env_factory is None:
        from capx.envs.simulators.libero import FrankaLiberoTask

        env_factory = FrankaLiberoTask
    if api_factory is None:
        from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced

        api_factory = FrankaLiberoApiReduced

    env = env_factory(
        suite_name=config.suite_name,
        task_id=config.task_id,
        privileged=False,
        max_steps=config.max_sim_steps,
    )
    api = api_factory(env)
    api.configure_vlm_backend(
        model=config.vlm_model,
        server_url=config.vlm_server_url,
        api_key=config.vlm_api_key,
        coord_space=config.vlm_coord_space,
    )
    return CapXLiberoBackend(env, api)


__all__ = [
    "LIBERO_PRO_SUITES",
    "CapXLiberoBackend",
    "LiberoBackendConfig",
    "create_libero_backend",
]
