"""Episode-local runtime state hidden from the policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """One synchronized camera frame in the robot-base coordinate system."""

    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    base_from_camera: np.ndarray
    revision: int

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"rgb must have shape (H, W, 3), got {self.rgb.shape}")
        if self.depth.ndim != 2 or self.depth.shape != self.rgb.shape[:2]:
            raise ValueError(
                f"depth must have shape {self.rgb.shape[:2]}, got {self.depth.shape}"
            )
        if self.intrinsics.shape != (3, 3):
            raise ValueError(
                f"intrinsics must have shape (3, 3), got {self.intrinsics.shape}"
            )
        if self.base_from_camera.shape != (4, 4):
            raise ValueError(
                "base_from_camera must have shape (4, 4), "
                f"got {self.base_from_camera.shape}"
            )
        if self.revision < 0:
            raise ValueError("revision must be non-negative")


@dataclass(frozen=True, slots=True)
class MaskArtifact:
    """A segmentation mask tied to exactly one sensor-frame revision."""

    mask: np.ndarray
    revision: int


@dataclass(slots=True)
class ArtifactStore:
    """Large intermediate values referenced by small policy-visible handles."""

    masks: dict[str, MaskArtifact] = field(default_factory=dict)
    _next_mask_index: int = 0

    def add_mask(self, mask: np.ndarray, revision: int) -> str:
        mask_id = f"m{self._next_mask_index}"
        self._next_mask_index += 1
        self.masks[mask_id] = MaskArtifact(
            mask=np.asarray(mask, dtype=bool).copy(),
            revision=revision,
        )
        return mask_id

    def get_mask(self, mask_id: str, revision: int) -> np.ndarray:
        artifact = self.masks.get(mask_id)
        if artifact is None:
            raise ValueError(f"unknown mask_id {mask_id!r}")
        if artifact.revision != revision:
            raise ValueError(f"mask_id {mask_id!r} belongs to a stale observation")
        return artifact.mask

    def clear(self) -> None:
        self.masks.clear()


class Backend(Protocol):
    """Narrow backend surface required by the ten public tools."""

    def reset(self, seed: int | None = None) -> tuple[SensorFrame, str]: ...

    def capture(self, revision: int) -> SensorFrame: ...

    def task_completed(self) -> bool: ...

    def vlm_bbox_detection(self, rgb: np.ndarray, query: str) -> list[float]: ...

    def vlm_point_detection(self, rgb: np.ndarray, query: str) -> list[float]: ...

    def sam3_text(self, rgb: np.ndarray, text: str) -> list[dict[str, Any]]: ...

    def sam3_box(
        self, rgb: np.ndarray, bbox: list[float]
    ) -> list[dict[str, Any]]: ...

    def sam3_point(
        self, rgb: np.ndarray, point: list[float]
    ) -> list[dict[str, Any]]: ...

    def get_obb(self, points_base: np.ndarray) -> dict[str, Any]: ...

    def plan_grasp(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> np.ndarray: ...

    def move_to_joints(self, joints: np.ndarray) -> None: ...

    def open_gripper(self) -> None: ...

    def close_gripper(self) -> None: ...

    def go_home(self) -> None: ...


@dataclass(slots=True)
class EnvContext:
    """Mutable execution context for one episode."""

    backend: Backend
    frame: SensorFrame
    artifacts: ArtifactStore = field(default_factory=ArtifactStore)

    def replace_frame(self, frame: SensorFrame) -> None:
        if frame.revision <= self.frame.revision:
            raise ValueError("new sensor frame must advance the revision")
        self.frame = frame
        self.artifacts.clear()


__all__ = [
    "ArtifactStore",
    "Backend",
    "EnvContext",
    "MaskArtifact",
    "SensorFrame",
]
