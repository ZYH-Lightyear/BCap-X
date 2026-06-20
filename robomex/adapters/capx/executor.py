from __future__ import annotations

from typing import Any

from robomex.execution import (
    ActionBlockStatus,
    BlockExecutionResult,
    ExecutionTraceEvent,
    SemanticActionBlock,
)


class CapXExecutorAdapter:
    """Run RoboMEx semantic action blocks through a CapX code execution env.

    The adapter deliberately uses duck typing instead of importing CapX classes.
    A compatible env should expose:

    - step(code: str) -> (obs, reward, terminated, truncated, info)
    - configure_line_trace(block_index, emit_callback=None), optional
    - consume_line_trace_events(), optional
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self._block_index = 0

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        """Execute one semantic action block in the wrapped CapX environment."""

        current_index = self._block_index
        self._block_index += 1

        if hasattr(self.env, "configure_line_trace"):
            self.env.configure_line_trace(current_index)

        obs: dict[str, Any] | None = None
        reward: float | None = None
        terminated: bool | None = None
        truncated: bool | None = None
        info: dict[str, Any] = {}

        try:
            obs, raw_reward, raw_terminated, raw_truncated, raw_info = self.env.step(block.code)
            reward = float(raw_reward) if raw_reward is not None else None
            terminated = bool(raw_terminated)
            truncated = bool(raw_truncated)
            info = dict(raw_info or {})
            ok = info.get("sandbox_rc", 0) == 0
        except Exception as exc:
            ok = False
            info = {"adapter_error": repr(exc)}

        raw_events = []
        if hasattr(self.env, "consume_line_trace_events"):
            raw_events = self.env.consume_line_trace_events()
        if hasattr(self.env, "configure_line_trace"):
            self.env.configure_line_trace(None)

        trace_events = tuple(self._normalize_trace_event(event, block.name) for event in raw_events)
        stdout = str(info.get("stdout", ""))
        stderr = str(info.get("stderr", info.get("adapter_error", "")))
        status = ActionBlockStatus.SUCCEEDED if ok else ActionBlockStatus.FAILED

        return BlockExecutionResult(
            block=block,
            ok=ok,
            status=status,
            stdout=stdout,
            stderr=stderr,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            observation=obs,
            result=info.get("result"),
            info=info,
            trace_events=trace_events,
        )

    @staticmethod
    def _normalize_trace_event(event: Any, block_name: str) -> ExecutionTraceEvent:
        if isinstance(event, dict):
            return ExecutionTraceEvent(
                event_type=str(event.get("event_type", event.get("type", "trace"))),
                message=str(event.get("message", "")),
                block_name=str(event.get("block_name", block_name)),
                line_no=event.get("line_no"),
                payload=dict(event),
            )
        return ExecutionTraceEvent(
            event_type="trace",
            message=str(event),
            block_name=block_name,
        )

