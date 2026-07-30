"""Stateful single-tool environment used by inference and RL rollouts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from capx_skill_rl.context import Backend, EnvContext
from capx_skill_rl.tools import ToolRegistry, default_registry
from capx_skill_rl.tools.registry import ToolInputError

ActionLike = str | Mapping[str, Any] | list[Any]


@dataclass(frozen=True, slots=True)
class Observation:
    """The complete policy-visible environment observation."""

    rgb: np.ndarray
    task: str


@dataclass(frozen=True, slots=True)
class StepResult:
    result: dict[str, Any]
    observation: Observation | None
    reward: float
    done: bool


class ToolEnv:
    """One LIBERO-PRO episode with one tool call per RL step."""

    def __init__(
        self,
        backend: Backend,
        *,
        registry: ToolRegistry | None = None,
        max_steps: int = 32,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.backend = backend
        self.registry = registry or default_registry()
        self.max_steps = max_steps
        self.context: EnvContext | None = None
        self.task = ""
        self.step_count = 0
        self.done = False

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        return self.registry.definitions()

    def reset(self, seed: int | None = None) -> Observation:
        frame, task = self.backend.reset(seed)
        self.context = EnvContext(backend=self.backend, frame=frame)
        self.task = task
        self.step_count = 0
        self.done = False
        return self._observation()

    def step(self, action: ActionLike) -> StepResult:
        if self.context is None:
            raise RuntimeError("reset() must be called before step()")
        if self.done:
            raise RuntimeError("episode already ended; call reset()")

        self.step_count += 1
        result: dict[str, Any]
        observation: Observation | None = None
        physical = False

        try:
            name, raw_arguments = _parse_action(action)
            spec, arguments = self.registry.prepare(name, raw_arguments)
            physical = spec.physical
            try:
                result = spec.handler(self.context, arguments)
            except Exception as exc:  # external tool failures are policy-visible
                result = _error(exc)
            finally:
                if physical:
                    observation, refresh_error = self._refresh_after_physical()
                    if refresh_error is not None:
                        if "error" in result:
                            result = {
                                "error": (
                                    f"{result['error']}; observation refresh failed: "
                                    f"{refresh_error}"
                                )
                            }
                        else:
                            result = {
                                "error": f"observation refresh failed: {refresh_error}"
                            }
        except (ToolInputError, ValueError, TypeError, json.JSONDecodeError) as exc:
            result = _error(exc)

        success = False
        if physical:
            try:
                success = bool(self.backend.task_completed())
            except Exception as exc:
                if "error" not in result:
                    result = _error(exc)

        reward = 1.0 if success else 0.0
        self.done = success or self.step_count >= self.max_steps
        return StepResult(
            result=result,
            observation=observation,
            reward=reward,
            done=self.done,
        )

    def _refresh_after_physical(
        self,
    ) -> tuple[Observation | None, str | None]:
        assert self.context is not None
        self.context.artifacts.clear()
        try:
            frame = self.backend.capture(self.context.frame.revision + 1)
            self.context.replace_frame(frame)
        except Exception as exc:
            return None, str(exc)
        return self._observation(), None

    def _observation(self) -> Observation:
        assert self.context is not None
        return Observation(rgb=self.context.frame.rgb.copy(), task=self.task)


def _parse_action(action: ActionLike) -> tuple[str, Mapping[str, Any]]:
    parsed = json.loads(action) if isinstance(action, str) else action
    if isinstance(parsed, list):
        raise ValueError("exactly one tool call is required per step")
    if not isinstance(parsed, Mapping):
        raise ValueError("action must be an object")
    keys = set(parsed)
    if keys != {"name", "arguments"}:
        missing = {"name", "arguments"} - keys
        extra = keys - {"name", "arguments"}
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise ValueError("invalid action: " + ", ".join(details))
    name = parsed["name"]
    arguments = parsed["arguments"]
    if not isinstance(name, str) or not name:
        raise ValueError("action name must be a non-empty string")
    if not isinstance(arguments, Mapping):
        raise ValueError("action arguments must be an object")
    return name, arguments


def _error(exc: Exception) -> dict[str, str]:
    message = str(exc).strip() or type(exc).__name__
    return {"error": message}


__all__ = ["ActionLike", "Observation", "StepResult", "ToolEnv"]
