"""Pure projections used to build Main's small, fresh context each turn."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from vaw.context_runtime.memory import (
    DEFAULT_PROMPT_WINDOW,
    InteractionEvent,
    InteractionMemory,
)
from vaw.context_runtime.model import ContextState
from vaw.mmskill import MMSkill


@dataclass(frozen=True)
class ActionRecency:
    function: str
    arguments: dict[str, Any]
    outcome: str
    turns_ago: int
    revisions_ago: int

    def summary(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "arguments": dict(self.arguments),
            "outcome": self.outcome,
            "turns_ago": self.turns_ago,
            "revisions_ago": self.revisions_ago,
        }


@dataclass(frozen=True)
class EmbodiedStateCard:
    """Main-visible proprioception. Gripper aperture is canvas-only."""

    tcp_pose: dict[str, Any] | None
    last_action: ActionRecency | None
    last_gripper_action: ActionRecency | None
    active_action: dict[str, Any] | None

    def summary(self) -> dict[str, Any]:
        return {
            "tcp_pose": self.tcp_pose,
            "last_action": (
                self.last_action.summary() if self.last_action is not None else None
            ),
            "last_gripper_action": (
                self.last_gripper_action.summary()
                if self.last_gripper_action is not None
                else None
            ),
            "active_action": self.active_action,
        }


def project_embodied_state(
    state: ContextState,
    memory: InteractionMemory,
    *,
    decision_turn: int,
) -> EmbodiedStateCard:
    robot = state.robot
    action = state.action_proposal
    return EmbodiedStateCard(
        tcp_pose=(
            robot.tcp_pose.summary()
            if robot is not None and robot.tcp_pose is not None
            else None
        ),
        last_action=_recency(
            memory.last_action,
            decision_turn=decision_turn,
            revision=state.observation_revision,
        ),
        last_gripper_action=_recency(
            memory.last_gripper_action,
            decision_turn=decision_turn,
            revision=state.observation_revision,
        ),
        active_action=(
            {
                "action_id": action.action_id,
                "intent": action.intent,
                "state": "refined" if action.refined else "planned",
            }
            if action is not None
            else None
        ),
    )


def render_main_context(
    *,
    task: str,
    available_mmskills: str,
    live_references: dict[str, Any],
    embodied_state: EmbodiedStateCard,
    interaction_memory: InteractionMemory,
    active_mmskill: MMSkill | None = None,
    feedback: str | None = None,
) -> str:
    """Build the textual half of Main's multimodal context in one place."""

    lines = [
        f"用户任务：{task}",
        available_mmskills,
    ]
    if active_mmskill is not None:
        lines.extend(
            [
                f"当前加载技能：{active_mmskill.skill_id}",
                active_mmskill.prompt_block(),
            ]
        )
    lines.extend(
        [
            "当前有效引用：" + _json(live_references),
            "机器人本体状态：" + _json(embodied_state.summary()),
            "短期交互记忆：",
        ]
    )
    history = interaction_memory.prompt_lines(limit=DEFAULT_PROMPT_WINDOW)
    lines.extend(history if history else ["（空）"])
    if feedback is not None:
        lines.append(f"协议反馈：{feedback}")
    return "\n".join(lines)


def _recency(
    event: InteractionEvent | None,
    *,
    decision_turn: int,
    revision: int,
) -> ActionRecency | None:
    if event is None:
        return None
    return ActionRecency(
        function=event.function,
        arguments=dict(event.arguments),
        outcome=event.outcome,
        turns_ago=max(0, decision_turn - event.turn),
        revisions_ago=max(0, revision - event.revision_after),
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "ActionRecency",
    "EmbodiedStateCard",
    "project_embodied_state",
    "render_main_context",
]
