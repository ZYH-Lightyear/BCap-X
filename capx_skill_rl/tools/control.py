"""Low-level robot control tools."""

from __future__ import annotations

from typing import Any

import numpy as np

from capx_skill_rl.context import EnvContext


def solve_ik(context: EnvContext, arguments: dict[str, Any]) -> dict[str, Any]:
    position = np.asarray(arguments["position"], dtype=np.float64)
    quaternion_xyzw = _unit_quaternion(arguments["quaternion"])
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    joints = np.asarray(
        context.backend.solve_ik(position, quaternion_wxyz),
        dtype=np.float64,
    ).reshape(-1)
    if len(joints) != 7 or not np.isfinite(joints).all():
        raise ValueError("IK solver must return exactly seven finite joints")
    return {"joints": [float(value) for value in joints]}


def move_to_joints(
    context: EnvContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    context.backend.move_to_joints(
        np.asarray(arguments["joints"], dtype=np.float64)
    )
    return {}


def open_gripper(
    context: EnvContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    del arguments
    context.backend.open_gripper()
    return {}


def close_gripper(
    context: EnvContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    del arguments
    context.backend.close_gripper()
    return {}


def go_home(
    context: EnvContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    del arguments
    context.backend.go_home()
    return {}


def _unit_quaternion(values: list[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("quaternion must be finite and non-zero")
    return quaternion / norm


__all__ = [
    "close_gripper",
    "go_home",
    "move_to_joints",
    "open_gripper",
    "solve_ik",
]
