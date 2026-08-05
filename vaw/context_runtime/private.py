"""Episode-local sensor state that must never enter a model message.

The Context compiler is trusted to *read* this state and rasterise selected
evidence.  Neither :class:`PrivateEnvContext` nor its numpy arrays have a JSON
serializer on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RegionGeometryArtifact:
    """Private multiview geometry attached to one revision-local region."""

    agentview_mask: np.ndarray
    wrist_mask: np.ndarray | None
    object_points_base: np.ndarray
    scene_points_base: np.ndarray
    filtered_object_points_base: np.ndarray


@dataclass
class PrivateEnvContext:
    """Raw observation and artifacts for exactly one physical revision."""

    observation: dict[str, Any] | None = None
    region_masks: dict[str, np.ndarray] = field(default_factory=dict)
    region_geometry: dict[str, RegionGeometryArtifact] = field(default_factory=dict)
    # Values are MotionPlan instances, kept as Any here to avoid making the
    # private sensor container depend on planner implementation details.
    motion_plans: dict[str, Any] = field(default_factory=dict)
    # Scalar / JSON-safe diagnostics consumed by the trace logger after one
    # Function call.  They are never compiled into the policy-visible packet.
    trace_diagnostics: dict[str, Any] = field(default_factory=dict)

    def begin_revision(self, observation: dict[str, Any]) -> None:
        self.observation = observation
        self.region_masks.clear()
        self.region_geometry.clear()
        self.motion_plans.clear()
        self.trace_diagnostics.clear()

    def begin_function_call(self) -> None:
        self.trace_diagnostics.clear()

    def camera(self, name: str) -> dict[str, Any]:
        observation = self.observation
        if observation is None:
            raise RuntimeError("observation is unavailable")
        camera = observation.get(name)
        if not isinstance(camera, dict):
            raise RuntimeError(f"observation has no camera '{name}'")
        return camera


__all__ = ["PrivateEnvContext", "RegionGeometryArtifact"]
