"""Single-VLM agent loop for the revision-local VAW Context Runtime."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from vaw.agents.contracts import (
    EpisodeResult,
    ModelResponse,
    StepRecord,
    TerminateMode,
    ToolCall,
)
from vaw.agents.providers.base import ModelProvider
from vaw.context_runtime.history import ContextHistory
from vaw.context_runtime.packet import CONTEXT_SCHEMA, ContextCompiler, ContextPacket
from vaw.context_runtime.protocol import SYSTEM_PROMPT, function_definitions, parse_action
from vaw.context_runtime.trace import ContextTraceLogger
from vaw.context_runtime.workspace import ContextStepResult, ContextWorkspace

NO_CALL_FEEDBACK = (
    "本轮没有执行任何 Function。请且只调用一个 VAW Function。"
    "若任务已完成或无法继续，调用 done(success=...)。"
)
MULTI_CALL_ERROR = "未执行：当前 Runtime 要求每轮且只能调用一个 Function。"


class ContextRenderer(Protocol):
    name: str

    def render(self, packet: ContextPacket) -> Any: ...

    def close(self) -> None: ...


@dataclass
class ContextRunConfig:
    max_turns: int = 32
    max_time_s: float = 1800.0
    max_physical_ops: int = 30
    history_k: int = 3
    on_turn: Callable[[int, StepRecord], None] | None = None


class ContextRuntime:
    def __init__(
        self,
        provider: ModelProvider,
        workspace: ContextWorkspace,
        renderer: ContextRenderer,
        *,
        compiler: ContextCompiler | None = None,
        config: ContextRunConfig | None = None,
        trace: ContextTraceLogger | None = None,
        env_check: Callable[[], bool] | None = None,
    ) -> None:
        self.provider = provider
        self.workspace = workspace
        self.renderer = renderer
        self.compiler = compiler or ContextCompiler()
        self.config = config or ContextRunConfig()
        self.trace = trace
        self.env_check = env_check
        self.tools = function_definitions()
        self.history = ContextHistory(
            SYSTEM_PROMPT,
            workspace.state.task_prompt,
            max_transactions=self.config.history_k,
        )
        self.usage: dict[str, int] = {}
        self.env_success: bool | None = None
        self.physical_ops = 0
        if self.trace is not None:
            self.trace.log_meta(
                {
                    "context_schema": CONTEXT_SCHEMA,
                    "renderer": renderer.name,
                    "task_prompt": workspace.state.task_prompt,
                    "history_k": self.config.history_k,
                    "motion_backend": workspace.motion_backend_name,
                }
            )

    def run(self) -> EpisodeResult:
        started = time.monotonic()
        steps: list[StepRecord] = []
        turn = 0
        while True:
            if self.workspace.finished:
                return self._result(TerminateMode.GOAL, turn, steps)
            if turn >= self.config.max_turns:
                return self._finish(
                    TerminateMode.MAX_TURNS,
                    turn,
                    steps,
                    f"turn limit {self.config.max_turns} reached",
                )
            if time.monotonic() - started >= self.config.max_time_s:
                return self._finish(
                    TerminateMode.TIMEOUT,
                    turn,
                    steps,
                    f"time limit {self.config.max_time_s:.0f}s reached",
                )

            packet = self.compiler.compile(self.workspace)
            image = self.renderer.render(packet)
            messages = self.history.build_messages(
                manifest=self.workspace.state.manifest(), context_image=image
            )
            visible_before = self.history.visible_records()
            turn += 1
            try:
                response = self.provider.generate(messages, self.tools)
            except KeyboardInterrupt:
                return self._finish(
                    TerminateMode.CANCELLED, turn, steps, "interrupted by user"
                )
            except Exception as exc:
                return self._finish(
                    TerminateMode.ERROR,
                    turn,
                    steps,
                    f"{type(exc).__name__}: {exc}",
                )
            self._accumulate_usage(response)

            if not response.tool_calls:
                self.history.pending_feedback = NO_CALL_FEEDBACK
                self._log(
                    turn=turn,
                    image=image,
                    packet=packet,
                    call=None,
                    step=None,
                    thought=response.text,
                    response=response,
                    visible_recent_calls=visible_before,
                )
                continue

            if len(response.tool_calls) != 1:
                results = [{"error": MULTI_CALL_ERROR} for _ in response.tool_calls]
                self.history.add_response(response, results)
                call_summary = {
                    "name": "invalid_multiple_calls",
                    "calls": [_call_summary(call) for call in response.tool_calls],
                }
                step = self.workspace.reject(
                    "invalid_multiple_calls", call_summary, MULTI_CALL_ERROR
                )
                record = _step_record(turn, step, response.text, physical=False)
                steps.append(record)
                self._log(
                    turn,
                    image,
                    packet,
                    call_summary,
                    step,
                    response.text,
                    response=response,
                    visible_recent_calls=visible_before,
                )
                self._notify(turn, record)
                continue

            call = response.tool_calls[0]
            if call.name in ContextWorkspace.PHYSICAL_FUNCTIONS:
                if self.physical_ops >= self.config.max_physical_ops:
                    return self._finish(
                        TerminateMode.MAX_TURNS,
                        turn,
                        steps,
                        f"physical operation limit {self.config.max_physical_ops} reached",
                    )
                self.physical_ops += 1
            step = self._dispatch(call)
            self.history.add_response(response, [step.result])
            physical = call.name in ContextWorkspace.PHYSICAL_FUNCTIONS
            if physical or call.name == "done":
                self._update_env_success()
            record = _step_record(turn, step, response.text, physical=physical)
            steps.append(record)
            self._log(
                turn,
                image,
                packet,
                _call_summary(call),
                step,
                response.text,
                response=response,
                visible_recent_calls=visible_before,
            )
            self._notify(turn, record)

    def _dispatch(self, call: ToolCall) -> ContextStepResult:
        if call.parse_error:
            return self.workspace.reject(call.name, call.args, call.parse_error)
        try:
            name, arguments = parse_action({"name": call.name, "arguments": call.args})
        except ValueError as exc:
            return self.workspace.reject(call.name, call.args, str(exc))
        return self.workspace.execute(name, **arguments)

    def _finish(
        self,
        mode: TerminateMode,
        turn: int,
        steps: list[StepRecord],
        detail: str,
    ) -> EpisodeResult:
        if not self.workspace.finished:
            step = self.workspace.execute("done", success=False)
            steps.append(_step_record(turn, step, "", physical=False))
        self._update_env_success()
        return self._result(mode, turn, steps, detail)

    def _result(
        self,
        mode: TerminateMode,
        turns: int,
        steps: list[StepRecord],
        detail: str = "",
    ) -> EpisodeResult:
        if self.trace is not None:
            self.trace.log_meta(
                {
                    "terminate_mode": mode.value,
                    "turns": turns,
                    "claimed_success": self.workspace.claimed_success,
                    "env_success": self.env_success,
                    "usage": self.usage,
                }
            )
        return EpisodeResult(
            terminate_mode=mode,
            turns=turns,
            claimed_success=self.workspace.claimed_success,
            env_success=self.env_success,
            detail=detail,
            steps=list(steps),
            usage=dict(self.usage),
            trace_dir=str(self.trace.dir) if self.trace is not None else None,
        )

    def _update_env_success(self) -> None:
        if self.env_check is None:
            return
        try:
            self.env_success = bool(self.env_check())
        except Exception:
            self.env_success = None

    def _log(
        self,
        turn: int,
        image: Any,
        packet: ContextPacket,
        call: dict[str, Any] | None,
        step: ContextStepResult | None,
        thought: str,
        *,
        response: ModelResponse,
        visible_recent_calls: list[dict[str, Any]],
    ) -> None:
        if self.trace is None:
            return
        self.trace.log_turn(
            turn=turn,
            image=image,
            packet=packet,
            visible_recent_calls=visible_recent_calls,
            function_call=call,
            step=step,
            thought=thought,
            env_success=self.env_success,
            done=self.workspace.finished,
            raw_response_text=response.raw_response_text or response.text,
            provider_reasoning=response.provider_reasoning,
        )

    def _notify(self, turn: int, record: StepRecord) -> None:
        if self.config.on_turn is not None:
            self.config.on_turn(turn, record)

    def _accumulate_usage(self, response: ModelResponse) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = response.usage.get(key)
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value


def run_context_episode(
    provider: ModelProvider,
    api: Any,
    task_prompt: str,
    renderer: ContextRenderer,
    *,
    trace_dir: str | None = None,
    config: ContextRunConfig | None = None,
    env_check: Callable[[], bool] | None = None,
    motion_backend: str = "curobo",
) -> EpisodeResult:
    workspace = ContextWorkspace(api, task_prompt, motion_backend=motion_backend)
    trace = ContextTraceLogger(trace_dir) if trace_dir is not None else None
    try:
        return ContextRuntime(
            provider,
            workspace,
            renderer,
            config=config,
            trace=trace,
            env_check=env_check,
        ).run()
    finally:
        renderer.close()


def _call_summary(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": call.args}


def _step_record(
    turn: int,
    step: ContextStepResult,
    thought: str,
    *,
    physical: bool,
) -> StepRecord:
    return StepRecord(
        turn=turn,
        op=step.function_name,
        args=step.arguments,
        ok=step.ok,
        physical=physical,
        receipt=json.dumps(step.result, ensure_ascii=False, separators=(",", ":")),
        thought=thought.strip(),
    )


__all__ = [
    "ContextRenderer",
    "ContextRunConfig",
    "ContextRuntime",
    "MULTI_CALL_ERROR",
    "NO_CALL_FEEDBACK",
    "run_context_episode",
]
