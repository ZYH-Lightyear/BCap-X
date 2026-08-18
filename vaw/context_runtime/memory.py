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
            if _same_move_intent(previous, primitive):
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
    status: Literal["ok", "failed"]
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


_FAILURE_MESSAGES = {
    "detection_and_sam": "target could not be grounded in the current observation",
    "locate_point": "operation point could not be grounded in the current observation",
    "propose_grasps": "no usable grasp seeds were produced",
    "propose_pose": "a pending spatial action could not be created",
    "select": "the selected seed did not create a pending spatial action",
    "refine_action": (
        "local refinement did not produce a ready action; do not recreate the same "
        "point/offset task without new geometry or a materially different target"
    ),
    "delta_move": "the physical TCP adjustment did not complete; its effect is uncertain",
    "commit": "the spatial command did not complete; its effect is uncertain",
    "open_gripper": "the gripper command did not complete; its effect is uncertain",
    "close_gripper": "the gripper command did not complete; its effect is uncertain",
    "reject_action": "the pending action could not be rejected",
    "done": "the termination declaration was rejected",
}


def project_function_event(function_name: str, result: dict[str, Any]) -> FunctionEvent:
    """Project raw handler output without leaking controller diagnostics."""

    failed = "error" in result or result.get("status") == "failed"
    if failed:
        return FunctionEvent(
            function=function_name,
            status="failed",
            message=_FAILURE_MESSAGES.get(function_name, "function call was rejected"),
        )

    references: dict[str, Any] = {}
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
    )


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


def _same_move_intent(
    previous: PhysicalPrimitive,
    current: PhysicalPrimitive,
) -> bool:
    return (
        previous.op == current.op == "move_to"
        and previous.intent is not None
        and previous.intent == current.intent
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
