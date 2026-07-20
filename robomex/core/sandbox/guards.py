"""Deterministic safety guards shared by all Coding Agents."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any

from robomex.core.events import emit_event
from robomex.core.sandbox.action_block import BlockExecutionResult, SemanticActionBlock


STATE_CHANGING_CAPABILITIES = frozenset(
    {"robot_motion", "gripper_control", "object_manipulation"}
)

_ROBOT_STATE_OBS_KEYS = ("robot_cartesian_pos", "robot_joint_pos")


@dataclass
class RuntimeSafetyState:
    observation_epoch: int = 0
    motion_holder: str = ""
    # Terminal robot state captured by the runtime after the most recent
    # state-changing block. Adapters project it into execution_evidence so
    # executors never need perception_read to report their terminal state.
    last_terminal_robot_state: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        self._lock = Lock()


class MotionLeaseGuard:
    """Serialize state-changing robot blocks and advance observation epochs."""

    def __init__(self, inner: Any, state: RuntimeSafetyState, *, agent_id: str) -> None:
        self.inner = inner
        self.state = state
        self.agent_id = agent_id

    @property
    def env(self) -> Any:
        return getattr(self.inner, "env", None)

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        # Import lazily: core.sandbox is imported by the CodingAgent parser, while
        # authoring itself depends on CodingAgent. A module-level import would form a
        # package initialization cycle.
        from robomex.authoring.capabilities import CALL_EFFECTS, called_function_names

        effects = {
            CALL_EFFECTS.get(name)
            for name in called_function_names(block.code)
            if CALL_EFFECTS.get(name) is not None
        }
        changes_state = bool(effects & STATE_CHANGING_CAPABILITIES)
        if not changes_state:
            return self.inner.run_block(block)

        if not self.state._lock.acquire(blocking=False):
            holder = self.state.motion_holder or "unknown"
            raise RuntimeError(
                f"Robot motion lease is held by {holder!r}; {self.agent_id!r} cannot execute."
            )
        self.state.motion_holder = self.agent_id
        emit_event(
            "motion_lease_acquired",
            "Robot motion lease acquired",
            agent_id=self.agent_id,
            observation_epoch=self.state.observation_epoch,
        )
        try:
            result = self.inner.run_block(block)
        finally:
            # A block may move the robot before a later statement fails. Treat every
            # admitted state-changing attempt as an epoch boundary.
            self.state.observation_epoch += 1
            emit_event(
                "observation_epoch_advanced",
                "State-changing robot block attempted",
                agent_id=self.agent_id,
                observation_epoch=self.state.observation_epoch,
            )
            self.state.motion_holder = ""
            self.state._lock.release()
            emit_event(
                "motion_lease_released",
                "Robot motion lease released",
                agent_id=self.agent_id,
                observation_epoch=self.state.observation_epoch,
            )
        return self._attach_terminal_robot_state(result)

    def _attach_terminal_robot_state(
        self, result: BlockExecutionResult
    ) -> BlockExecutionResult:
        """Record the post-motion robot state on behalf of the executor.

        ``env.step`` already returns a fresh observation after every code block,
        so the runtime can publish proprioception (end-effector pose, joints,
        gripper) without granting the executor ``perception_read``. The captured
        state is stamped with the *post*-advance epoch it belongs to.
        """

        terminal_state = _terminal_robot_state(
            result, observation_epoch=self.state.observation_epoch
        )
        if terminal_state is None:
            return result
        self.state.last_terminal_robot_state = terminal_state
        emit_event(
            "terminal_robot_state_captured",
            "Runtime captured post-motion robot state",
            agent_id=self.agent_id,
            terminal_robot_state=terminal_state,
        )
        return replace(
            result,
            info={**result.info, "terminal_robot_state": terminal_state},
        )


def _terminal_robot_state(
    result: BlockExecutionResult, *, observation_epoch: int
) -> dict[str, Any] | None:
    """Extract compact proprioception from a block's post-execution observation."""

    observation = result.observation
    if not isinstance(observation, dict):
        return None
    state: dict[str, Any] = {}
    for key in _ROBOT_STATE_OBS_KEYS:
        values = _float_list(observation.get(key))
        if values is not None:
            state[key] = values
    if not state:
        return None
    cartesian = state.get("robot_cartesian_pos")
    if cartesian is not None and len(cartesian) >= 8:
        # Layout: xyz, quaternion wxyz, normalized gripper opening (0..1).
        state["gripper_open_ratio"] = cartesian[7]
    state["observation_epoch"] = observation_epoch
    state["source"] = "runtime"
    if result.terminated is not None:
        state["terminated"] = bool(result.terminated)
    if result.truncated is not None:
        state["truncated"] = bool(result.truncated)
    return state


def _float_list(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    values = raw.tolist() if hasattr(raw, "tolist") else raw
    if not isinstance(values, (list, tuple)):
        return None
    try:
        return [round(float(v), 5) for v in values]
    except (TypeError, ValueError):
        return None
