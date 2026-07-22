"""Deterministic, read-only point-cloud previews for Arena motion candidates.

The Arena bridge deliberately knows nothing about a concrete robot model.  A
trusted geometry provider supplies scene points and forward-kinematic TCP
positions for the exact sealed plan.  This renderer turns those arrays into
content-addressed evidence files; it never receives an action backend or an
LLM handle and therefore cannot move either an authoritative or shadow robot.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from robomex.orchestration.arena import ArenaRuntimeContextV1
from robomex.orchestration.arena_coding_provider import (
    MotionPreviewFrame,
    MotionPreviewResult,
)
from robomex.runtime.action_protocol import MotionPlan

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class MotionPreviewError(RuntimeError):
    """The trusted geometry or deterministic renderer violated its contract."""


class PointCloudPreviewConfig(BaseModel):
    """Manifest-pinnable rendering policy for one production renderer."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: Literal["robomex.pointcloud_preview_config.v1"] = (
        "robomex.pointcloud_preview_config.v1"
    )
    renderer_id: NonEmptyStr = "robomex.pointcloud_motion_preview.v1"
    max_scene_points: int = Field(default=4_000, ge=32, le=100_000)
    dpi: int = Field(default=140, ge=72, le=400)
    point_size: float = Field(default=2.0, gt=0.0, le=20.0, allow_inf_nan=False)
    views: tuple[Literal["perspective", "top_down", "side"], ...] = (
        "perspective",
        "top_down",
    )

    def model_post_init(self, __context: object) -> None:
        if not self.views or len(set(self.views)) != len(self.views):
            raise ValueError("preview views must be a non-empty unique sequence")


@runtime_checkable
class MotionPreviewGeometryProvider(Protocol):
    """Trusted read-only scene and FK boundary used by the renderer.

    Implementations may call CuRobo or a simulator for forward kinematics and
    may resolve a point-cloud artifact from the snapshot lineage.  They must
    not execute a trajectory or mutate a simulation world.
    """

    @property
    def provider_id(self) -> str: ...

    def scene_points(
        self,
        *,
        context: ArenaRuntimeContextV1,
    ) -> np.ndarray: ...

    def tcp_positions(
        self,
        *,
        plan: MotionPlan,
        context: ArenaRuntimeContextV1,
    ) -> np.ndarray: ...


class PointCloudMotionPreviewRenderer:
    """Render sealed motion plans over point clouds without execution authority."""

    def __init__(
        self,
        *,
        geometry: MotionPreviewGeometryProvider,
        output_root: str | Path,
        config: PointCloudPreviewConfig | None = None,
    ) -> None:
        if not isinstance(geometry, MotionPreviewGeometryProvider):
            raise TypeError("geometry must implement MotionPreviewGeometryProvider")
        provider_id = str(geometry.provider_id).strip()
        if not provider_id:
            raise ValueError("geometry provider_id must not be empty")
        self.geometry = geometry
        self.config = config or PointCloudPreviewConfig()
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @property
    def renderer_id(self) -> str:
        return self.config.renderer_id

    def render(
        self,
        *,
        plan: MotionPlan,
        context: ArenaRuntimeContextV1,
    ) -> MotionPreviewResult:
        if not isinstance(plan, MotionPlan):
            raise TypeError("plan must be a MotionPlan")
        if not isinstance(context, ArenaRuntimeContextV1):
            context = ArenaRuntimeContextV1.model_validate(context)
        points = _finite_xyz(
            self.geometry.scene_points(context=context),
            label="scene_points",
            allow_empty=True,
        )
        trajectory = _finite_xyz(
            self.geometry.tcp_positions(plan=plan, context=context),
            label="tcp_positions",
            allow_empty=False,
        )
        if len(trajectory) not in {len(plan.motion.positions_rad), len(plan.motion.positions_rad) + 1}:
            raise MotionPreviewError(
                "FK provider must return one TCP point per waypoint, optionally including "
                "the admitted start state"
            )
        sampled = _deterministic_sample(points, self.config.max_scene_points)
        frames: list[MotionPreviewFrame] = []
        with self._lock:
            for view in self.config.views:
                output = self._output_path(plan=plan, context=context, view=view)
                if not output.exists():
                    self._draw_atomic(
                        output=output,
                        points=sampled,
                        trajectory=trajectory,
                        view=view,
                        candidate_id=context.candidate_id,
                    )
                digest = _file_digest(output)
                frames.append(
                    MotionPreviewFrame(
                        view_id=view,
                        media_type="image/png",
                        media_digest=digest,
                        payload={
                            "path": str(output),
                            "geometry_provider_id": str(self.geometry.provider_id),
                            "scene_point_count": int(len(points)),
                            "rendered_point_count": int(len(sampled)),
                            "trajectory_point_count": int(len(trajectory)),
                            "view": view,
                        },
                    )
                )
        terminal = tuple(float(value) for value in trajectory[-1])
        return MotionPreviewResult(
            frames=tuple(frames),
            terminal_position_m=terminal,
        )

    def _output_path(
        self,
        *,
        plan: MotionPlan,
        context: ArenaRuntimeContextV1,
        view: str,
    ) -> Path:
        identity = "\0".join(
            (
                plan.content_digest,
                context.episode_id,
                context.workflow_id,
                context.graph_id,
                str(context.graph_revision),
                context.slot_id,
                context.arena_run_id,
                context.candidate_id,
                view,
                self.config.model_dump_json(),
                str(self.geometry.provider_id),
            )
        ).encode("utf-8")
        return self.output_root / f"{hashlib.sha256(identity).hexdigest()}_{view}.png"

    def _draw_atomic(
        self,
        *,
        output: Path,
        points: np.ndarray,
        trajectory: np.ndarray,
        view: str,
        candidate_id: str,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output.parent.mkdir(parents=True, exist_ok=True)
        if view == "perspective":
            figure = plt.figure(figsize=(6.4, 5.4))
            axes = figure.add_subplot(111, projection="3d")
            if len(points):
                axes.scatter(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    s=self.config.point_size,
                    c="#84909c",
                    alpha=0.30,
                )
            axes.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color="#ef7d00",
                linewidth=2.2,
                marker="o",
                markersize=3,
            )
            axes.scatter(
                [trajectory[-1, 0]],
                [trajectory[-1, 1]],
                [trajectory[-1, 2]],
                c="#d62728",
                s=45,
            )
            _equal_xyz_limits(axes, points, trajectory)
            axes.set_zlabel("z (m)")
        else:
            figure, axes = plt.subplots(figsize=(6.4, 5.4))
            coordinates = (0, 1) if view == "top_down" else (0, 2)
            labels = ("x (m)", "y (m)") if view == "top_down" else ("x (m)", "z (m)")
            if len(points):
                axes.scatter(
                    points[:, coordinates[0]],
                    points[:, coordinates[1]],
                    s=self.config.point_size,
                    c="#84909c",
                    alpha=0.30,
                )
            axes.plot(
                trajectory[:, coordinates[0]],
                trajectory[:, coordinates[1]],
                color="#ef7d00",
                linewidth=2.2,
                marker="o",
                markersize=3,
            )
            axes.scatter(
                [trajectory[-1, coordinates[0]]],
                [trajectory[-1, coordinates[1]]],
                c="#d62728",
                s=45,
            )
            axes.set_aspect("equal", adjustable="box")
            axes.set_xlabel(labels[0])
            axes.set_ylabel(labels[1])
        axes.set_title(f"Arena motion candidate: {candidate_id} ({view})")
        axes.set_xlabel("x (m)")
        if view == "perspective":
            axes.set_ylabel("y (m)")
        axes.grid(True, alpha=0.18)
        figure.tight_layout()
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.stem}.",
            suffix=".png.tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            figure.savefig(
                temporary,
                format="png",
                dpi=self.config.dpi,
                metadata={"Software": "RoboMEx-v2"},
            )
            plt.close(figure)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, output)
            _fsync_directory(output.parent)
        except Exception:
            plt.close(figure)
            temporary.unlink(missing_ok=True)
            raise


def _finite_xyz(value: object, *, label: str, allow_empty: bool) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MotionPreviewError(f"{label} is not a numeric XYZ array") from exc
    if array.size == 0:
        array = np.empty((0, 3), dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise MotionPreviewError(f"{label} must have shape (N, 3)")
    if not allow_empty and len(array) == 0:
        raise MotionPreviewError(f"{label} must not be empty")
    if not np.isfinite(array).all():
        raise MotionPreviewError(f"{label} contains NaN or infinity")
    return np.ascontiguousarray(array, dtype=float)


def _deterministic_sample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, num=limit, dtype=int)
    return points[indices]


def _equal_xyz_limits(axes: object, points: np.ndarray, trajectory: np.ndarray) -> None:
    combined = trajectory if not len(points) else np.vstack((points, trajectory))
    minimum = np.min(combined, axis=0)
    maximum = np.max(combined, axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 0.025)
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "MotionPreviewError",
    "MotionPreviewGeometryProvider",
    "PointCloudMotionPreviewRenderer",
    "PointCloudPreviewConfig",
]
