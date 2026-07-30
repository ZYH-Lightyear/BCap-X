from __future__ import annotations

from typing import Any

import numpy as np

from capx_skill_rl.context import SensorFrame


class FakeBackend:
    def __init__(self, *, success_on_move: bool = False) -> None:
        self.success_on_move = success_on_move
        self.success = False
        self.capture_count = 0
        self.reset_seeds: list[int | None] = []
        self.last_ik_quaternion: np.ndarray | None = None
        self.last_obb_points: np.ndarray | None = None
        self.moves: list[np.ndarray] = []
        self.fail_move = False

    def reset(self, seed: int | None = None) -> tuple[SensorFrame, str]:
        self.reset_seeds.append(seed)
        self.success = False
        return make_frame(0), "pick up the object"

    def capture(self, revision: int) -> SensorFrame:
        self.capture_count += 1
        return make_frame(revision)

    def task_completed(self) -> bool:
        return self.success

    def vlm_bbox_detection(self, rgb: np.ndarray, query: str) -> list[float]:
        assert rgb.shape == (4, 4, 3)
        assert query
        return [0.0, 0.0, 3.0, 3.0]

    def vlm_point_detection(self, rgb: np.ndarray, query: str) -> list[float]:
        assert rgb.shape == (4, 4, 3)
        assert query
        return [1.0, 1.0]

    def sam3_text(self, rgb: np.ndarray, text: str) -> list[dict[str, Any]]:
        return self._masks()

    def sam3_box(
        self,
        rgb: np.ndarray,
        bbox: list[float],
    ) -> list[dict[str, Any]]:
        return self._masks()

    def sam3_point(
        self,
        rgb: np.ndarray,
        point: list[float],
    ) -> list[dict[str, Any]]:
        return self._masks()

    def get_obb(self, points_base: np.ndarray) -> dict[str, Any]:
        self.last_obb_points = points_base.copy()
        return {
            "center": points_base.mean(axis=0),
            "extent": np.array([0.1, 0.2, 0.3]),
            "R": np.eye(3),
        }

    def plan_grasp(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert depth.shape == mask.shape == (4, 4)
        assert intrinsics.shape == (3, 3)
        first = np.eye(4)
        first[:3, 3] = [0.1, 0.2, 0.3]
        second = np.eye(4)
        second[:3, 3] = [0.4, 0.5, 0.6]
        return np.stack([first, second]), np.array([0.2, 0.9])

    def solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> np.ndarray:
        assert position.shape == (3,)
        self.last_ik_quaternion = quaternion_wxyz.copy()
        return np.arange(7, dtype=np.float64)

    def move_to_joints(self, joints: np.ndarray) -> None:
        self.moves.append(joints.copy())
        if self.fail_move:
            raise RuntimeError("motion failed")
        if self.success_on_move:
            self.success = True

    def open_gripper(self) -> None:
        pass

    def close_gripper(self) -> None:
        pass

    def go_home(self) -> None:
        pass

    @staticmethod
    def _masks() -> list[dict[str, Any]]:
        small = np.zeros((4, 4), dtype=bool)
        small[0, 0] = True
        large = np.ones((4, 4), dtype=bool)
        return [
            {"mask": small, "score": 0.1},
            {"mask": large, "score": 0.9},
        ]


def make_frame(revision: int) -> SensorFrame:
    rgb = np.full((4, 4, 3), revision, dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float64)
    intrinsics = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    base_from_camera = np.eye(4)
    base_from_camera[:3, 3] = [1.0, 2.0, 3.0]
    return SensorFrame(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        base_from_camera=base_from_camera,
        revision=revision,
    )
