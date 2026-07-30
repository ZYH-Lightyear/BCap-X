"""Provider-agnostic one-tool-per-turn episode loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from capx_skill_rl.env import ActionLike, Observation, ToolEnv


@dataclass(frozen=True, slots=True)
class ToolExchange:
    action: ActionLike
    result: dict[str, Any]


class Policy(Protocol):
    """A policy adapter for inference servers or RL rollout workers."""

    def act(
        self,
        *,
        task: str,
        rgb: np.ndarray,
        history: Sequence[ToolExchange],
        tools: Sequence[dict[str, Any]],
    ) -> ActionLike: ...


@dataclass(frozen=True, slots=True)
class Transition:
    observation: Observation
    action: ActionLike
    result: dict[str, Any]
    next_observation: Observation
    reward: float
    done: bool


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    initial_observation: Observation
    transitions: tuple[Transition, ...]


def run_episode(env: ToolEnv, policy: Policy, *, seed: int | None = None) -> EpisodeResult:
    """Run until task success or the environment's tool-call horizon."""

    observation = env.reset(seed)
    initial_observation = observation
    history: list[ToolExchange] = []
    transitions: list[Transition] = []

    while True:
        action = policy.act(
            task=observation.task,
            rgb=observation.rgb.copy(),
            history=tuple(history),
            tools=env.tool_definitions,
        )
        step = env.step(action)
        next_observation = step.observation or observation
        transitions.append(
            Transition(
                observation=observation,
                action=action,
                result=step.result,
                next_observation=next_observation,
                reward=step.reward,
                done=step.done,
            )
        )
        history.append(ToolExchange(action=action, result=step.result))
        observation = next_observation
        if step.done:
            break

    return EpisodeResult(
        initial_observation=initial_observation,
        transitions=tuple(transitions),
    )


__all__ = [
    "EpisodeResult",
    "Policy",
    "ToolExchange",
    "Transition",
    "run_episode",
]
