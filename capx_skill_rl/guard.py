"""Non-model-facing admission checks for live model rollouts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from capx_skill_rl.env import ActionLike
from capx_skill_rl.loop import ToolExchange

PHYSICAL_TOOLS = {
    "move_to_joints",
    "open_gripper",
    "close_gripper",
    "go_home",
}


class ModelActionGuard:
    """Reject unsafe low-level model actions without changing the public API."""

    def validate(
        self,
        action: ActionLike,
        history: Sequence[ToolExchange],
    ) -> str | None:
        parsed = _action_mapping(action)
        if parsed is None:
            return None
        name = parsed.get("name")
        arguments = parsed.get("arguments")
        if not isinstance(arguments, Mapping):
            return None
        if name == "move_to_joints":
            return _validate_joint_provenance(arguments.get("joints"), history)
        return None

    def is_physical(self, action: ActionLike) -> bool:
        parsed = _action_mapping(action)
        return bool(parsed and parsed.get("name") in PHYSICAL_TOOLS)


def _action_mapping(action: ActionLike) -> Mapping[str, Any] | None:
    if isinstance(action, str):
        try:
            value = json.loads(action)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, Mapping) else None
    return action if isinstance(action, Mapping) else None


def _validate_joint_provenance(
    value: Any,
    history: Sequence[ToolExchange],
) -> str | None:
    if not history:
        return "move_to_joints requires joints from the immediately preceding solve_ik"
    previous = history[-1]
    previous_action = _action_mapping(previous.action)
    if previous_action is None or previous_action.get("name") != "solve_ik":
        return "move_to_joints requires joints from the immediately preceding solve_ik"
    approved = previous.result.get("joints")
    try:
        requested_array = np.asarray(value, dtype=np.float64).reshape(-1)
        approved_array = np.asarray(approved, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if requested_array.shape != (7,) or approved_array.shape != (7,):
        return None
    if not np.allclose(requested_array, approved_array, rtol=0.0, atol=1e-7):
        return "move_to_joints arguments differ from the preceding solve_ik result"
    return None


__all__ = ["ModelActionGuard", "PHYSICAL_TOOLS"]
