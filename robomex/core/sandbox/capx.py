from __future__ import annotations

from typing import Any

from robomex.core.sandbox.action_block import (
    ActionBlockStatus,
    BlockExecutionResult,
    ExecutionTraceEvent,
    SemanticActionBlock,
)


class CapXExecutorAdapter:
    """通过 CapX 代码执行环境运行 RoboMEx 的语义动作块。

    适配器刻意用鸭子类型而非 import CapX 的类。一个兼容的 env 应暴露:

    - step(code: str) -> (obs, reward, terminated, truncated, info)
    - configure_line_trace(block_index, emit_callback=None),可选
    - consume_line_trace_events(),可选
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self._block_index = 0

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        """在包装的 CapX 环境里执行一个语义动作块。"""

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

