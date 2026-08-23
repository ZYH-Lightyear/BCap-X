"""Deterministic policy memory compiled from executed Function transactions.

This module is intentionally small.  It is not a transcript, a scene belief,
or a model-authored summary.  The runtime records only physical primitives
that may have changed the world; live perception and action references remain
in :mod:`model`, while full controller diagnostics remain in the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

PrimitiveStatus = Literal["executed", "effect_unknown"]


@dataclass(frozen=True)
class PhysicalPrimitive:
    """One compact physical command fact, never a task-effect claim."""

    op: Literal["move_to", "delta_move", "open_gripper", "close_gripper"]
    status: PrimitiveStatus
    intent: str | None = None
    frame: Literal["base", "tool"] | None = None
    delta_xyz_m: tuple[float, float, float] | None = None

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op, "status": self.status}
        if self.intent:
            result["intent"] = self.intent
        if self.frame is not None:
            result["frame"] = self.frame
        if self.delta_xyz_m is not None:
            result["delta_xyz_m"] = [
                round(float(value), 6) for value in self.delta_xyz_m
            ]
        return result


@dataclass
class TaskMemory:
    """Bounded semantic ledger of physical primitives for one episode."""

    capacity: int = 6
    physical_primitives: list[PhysicalPrimitive] = field(default_factory=list)

    def record(self, primitive: PhysicalPrimitive) -> None:
        """Append a primitive with deterministic, semantics-preserving compaction."""

        if self.physical_primitives:
            previous = self.physical_primitives[-1]
            if _supersedes_previous_move(previous, primitive):
                self.physical_primitives[-1] = primitive
                return
            if _same_gripper_command(previous, primitive):
                self.physical_primitives[-1] = primitive
                return
            merged = _merge_delta(previous, primitive)
            if merged is not None:
                self.physical_primitives[-1] = merged
                return
        self.physical_primitives.append(primitive)
        if len(self.physical_primitives) > self.capacity:
            del self.physical_primitives[: len(self.physical_primitives) - self.capacity]

    def summary(self) -> list[dict[str, Any]]:
        """Return the policy form directly as a compact chronological list."""

        return [primitive.summary() for primitive in self.physical_primitives]


@dataclass(frozen=True)
class FunctionEvent:
    """One policy-visible projection of the latest Function transaction."""

    function: str
    status: Literal["ok", "partial", "failed"]
    references: dict[str, Any] | None = None
    message: str | None = None

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "function": self.function,
            "status": self.status,
        }
        if self.references:
            result["references"] = dict(self.references)
        if self.message:
            result["message"] = self.message
        return result


_PHYSICAL_FUNCTIONS = frozenset(
    {"commit", "delta_move", "open_gripper", "close_gripper"}
)
_FAILURE_MESSAGES = {
    "detection_and_sam": "target could not be grounded in the current observation",
    "locate_point": "operation point could not be grounded in the current observation",
    "propose_grasps": "no usable grasp seeds were produced",
    "propose_pose": "a spatial action proposal could not be created",
    "select": "the selected seed did not create a spatial action proposal",
    "call_imagination": "local imagination did not deliver a refined action",
    "delta_move": "the physical TCP adjustment did not complete; its effect is uncertain",
    "commit": "the spatial command did not complete; its effect is uncertain",
    "open_gripper": "the gripper command did not complete; its effect is uncertain",
    "close_gripper": "the gripper command did not complete; its effect is uncertain",
    "reject_action": "the action proposal could not be rejected",
    "done": "the termination declaration was rejected",
}


def project_function_event(
    function_name: str,
    result: dict[str, Any],
    *,
    world_changed: bool | None = None,
) -> FunctionEvent:
    """Project raw handler output without leaking controller diagnostics."""

    failed = "error" in result or result.get("status") == "failed"
    if failed:
        reason = result.get("reason")
        if not isinstance(reason, str):
            reason = None
        references: dict[str, Any] = {}
        action_id = result.get("action_id")
        if isinstance(action_id, str):
            references["action_id"] = action_id
        if reason:
            references["reason"] = reason
        return FunctionEvent(
            function=function_name,
            status="failed",
            references=references or None,
            message=_failure_message(function_name, result, world_changed),
        )

    if result.get("status") == "partial":
        references = {}
        action_id = result.get("action_id")
        if isinstance(action_id, str):
            references["action_id"] = action_id
        reason = result.get("reason")
        if isinstance(reason, str):
            references["reason"] = reason
        advisory = result.get("advisory")
        return FunctionEvent(
            function=function_name,
            status="partial",
            references=references or None,
            message=advisory if isinstance(advisory, str) else None,
        )

    references = {}
    for key in ("region_id", "point_id", "action_id"):
        value = result.get(key)
        if isinstance(value, str):
            references[key] = value
    seed_ids = result.get("seed_ids")
    if isinstance(seed_ids, (list, tuple)):
        references["seed_ids"] = [str(value) for value in seed_ids]
    return FunctionEvent(
        function=function_name,
        status="ok",
        references=references or None,
        message=_success_message(result),
    )


def _success_message(result: dict[str, Any]) -> str | None:
    """Measured facts worth one line: change checks, reuse, advisories."""

    parts: list[str] = []
    changes = result.get("world_changes")
    if isinstance(changes, dict):
        formatted = _format_world_changes(changes)
        if formatted:
            parts.append(formatted)
    if result.get("reused") is True:
        parts.append(
            "reused an existing verified grounding; that spot is unchanged "
            "since it was last measured"
        )
    if result.get("plan_reused") is True:
        parts.append(
            "returned this seed's already-measured plan; the trajectory result "
            "is unchanged"
        )
    detail = result.get("detail")
    if isinstance(detail, str) and detail:
        parts.append(detail)
    advisory = result.get("advisory")
    if isinstance(advisory, str) and advisory:
        parts.append(advisory)
    return "; ".join(parts) or None


def _format_world_changes(changes: dict[str, Any]) -> str | None:
    parts: list[str] = []
    removed = changes.get("removed")
    if isinstance(removed, (list, tuple)) and removed:
        parts.append("changed and dropped: " + ", ".join(str(v) for v in removed))
    occluded = changes.get("occluded")
    if isinstance(occluded, (list, tuple)) and occluded:
        parts.append("occluded, kept unverified: " + ", ".join(str(v) for v in occluded))
    verified = changes.get("verified")
    if isinstance(verified, (list, tuple)) and verified:
        parts.append("verified unchanged: " + ", ".join(str(v) for v in verified))
    if not parts:
        return None
    return "world change check — " + "; ".join(parts)


def _failure_message(
    function_name: str,
    result: dict[str, Any],
    world_changed: bool | None,
) -> str:
    if function_name in _PHYSICAL_FUNCTIONS and world_changed is False:
        detail = result.get("error")
        suffix = f": {detail}" if isinstance(detail, str) and detail else ""
        return f"rejected before dispatch; the real world is unchanged{suffix}"
    if function_name == "call_imagination":
        reason = result.get("reason")
        base = _FAILURE_MESSAGES["call_imagination"]
        if isinstance(reason, str):
            return f"{base}; reason={reason}"
        return base
    return _FAILURE_MESSAGES.get(function_name, "function call was rejected")


def record_physical_transaction(
    memory: TaskMemory,
    *,
    function_name: str,
    arguments: dict[str, Any],
    intent: str | None,
    physical_outcome: str,
) -> None:
    """Reduce one dispatched physical Function into TaskMemory."""

    status: PrimitiveStatus = (
        "executed" if physical_outcome == "completed" else "effect_unknown"
    )
    if function_name == "commit":
        memory.record(
            PhysicalPrimitive(
                op="move_to",
                status=status,
                intent=_normalize_intent(intent),
            )
        )
        return
    if function_name == "delta_move":
        delta = arguments.get("delta_xyz_m")
        frame = arguments.get("frame")
        if (
            isinstance(delta, (list, tuple))
            and len(delta) == 3
            and frame in {"base", "tool"}
        ):
            memory.record(
                PhysicalPrimitive(
                    op="delta_move",
                    status=status,
                    frame=frame,
                    delta_xyz_m=tuple(float(value) for value in delta),
                )
            )
        return
    if function_name in {"open_gripper", "close_gripper"}:
        memory.record(PhysicalPrimitive(op=function_name, status=status))


def _normalize_intent(value: str | None) -> str | None:
    normalized = " ".join(str(value or "").split())
    return normalized or None


def _same_gripper_command(
    previous: PhysicalPrimitive,
    current: PhysicalPrimitive,
) -> bool:
    return (
        previous.op == current.op
        and current.op in {"open_gripper", "close_gripper"}
        and previous.status == current.status
    )


def _supersedes_previous_move(
    previous: PhysicalPrimitive,
    current: PhysicalPrimitive,
) -> bool:
    """Keep only the latest consecutive absolute arm target.

    A later ``move_to`` supersedes the arm pose established by the preceding
    ``move_to`` even when their natural-language intents differ.  Keeping both
    made the policy reason over stale approach descriptions although the live
    robot pose already encodes the only current arm state.  Gripper operations
    and deltas still form explicit causal boundaries and are not removed.
    """

    return (
        previous.op == current.op == "move_to"
        and current.status == "executed"
    )


def _merge_delta(
    previous: PhysicalPrimitive,
    current: PhysicalPrimitive,
) -> PhysicalPrimitive | None:
    if (
        previous.op != "delta_move"
        or current.op != "delta_move"
        or previous.status != "executed"
        or current.status != "executed"
        or previous.frame != current.frame
        or previous.delta_xyz_m is None
        or current.delta_xyz_m is None
    ):
        return None
    total = tuple(
        float(before + after)
        for before, after in zip(
            previous.delta_xyz_m,
            current.delta_xyz_m,
            strict=True,
        )
    )
    return replace(previous, delta_xyz_m=total)


__all__ = [
    "FunctionEvent",
    "PhysicalPrimitive",
    "TaskMemory",
    "project_function_event",
    "record_physical_transaction",
]
