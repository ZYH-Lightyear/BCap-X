"""History-free Main/Imagination orchestration for the VAW Context Runtime."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from vaw.agents.contracts import (
    EpisodeResult,
    Message,
    ModelResponse,
    StepRecord,
    TerminateMode,
    ToolCall,
)
from vaw.agents.providers.base import ModelProvider
from vaw.context_runtime.packet import (
    CONTEXT_SCHEMA,
    ContextCompiler,
    ContextPacket,
    encode_png_data_url,
)
from vaw.context_runtime.protocol import (
    FUNCTION_NAMES,
    IMAGINATION_FUNCTION_NAMES,
    IMAGINATION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    function_definitions,
    imagination_function_definitions,
    parse_action,
)
from vaw.context_runtime.trace import ContextTraceLogger
from vaw.context_runtime.workspace import ContextStepResult, ContextWorkspace

NO_CALL_FEEDBACK = "未执行：本轮必须且只能调用一个 Function。"
MULTI_CALL_ERROR = "未执行：一轮只能调用一个 Function。"
IMAGINATION_STARTERS = frozenset(
    {"select", "propose_pose", "delta_move", "rotate", "open_gripper", "close_gripper"}
)


class ContextRenderer(Protocol):
    name: str

    def render(self, packet: ContextPacket) -> Any: ...

    def close(self) -> None: ...


@dataclass
class ContextRunConfig:
    max_main_turns: int = 32
    max_imagination_turns: int = 6
    max_time_s: float = 1800.0
    max_physical_ops: int = 30
    on_turn: Callable[[int, StepRecord], None] | None = None


class ContextRuntime:
    """Alternate ownership without replaying either agent's transcript."""

    def __init__(
        self,
        main_provider: ModelProvider,
        workspace: ContextWorkspace,
        renderer: ContextRenderer,
        *,
        imagination_provider: ModelProvider | None = None,
        compiler: ContextCompiler | None = None,
        config: ContextRunConfig | None = None,
        trace: ContextTraceLogger | None = None,
        env_check: Callable[[], bool] | None = None,
    ) -> None:
        self.main_provider = main_provider
        self.imagination_provider = imagination_provider or main_provider
        self.workspace = workspace
        self.renderer = renderer
        self.compiler = compiler or ContextCompiler()
        self.config = config or ContextRunConfig()
        self.trace = trace
        self.env_check = env_check
        self.main_tools = function_definitions()
        self.imagination_tools = imagination_function_definitions()
        self.usage: dict[str, int] = {}
        self.env_success: bool | None = None
        self.physical_ops = 0
        self.main_turns = 0
        self.imagination_turns = 0
        self._feedback: dict[str, str | None] = {"main": None, "imagination": None}
        self._previous_observed: np.ndarray | None = None
        if self.trace is not None:
            self.trace.log_meta(
                {
                    "context_schema": CONTEXT_SCHEMA,
                    "renderer": renderer.name,
                    "task_prompt": workspace.state.task_prompt,
                    "orchestration": "main-imagination-no-history",
                    "max_imagination_turns": self.config.max_imagination_turns,
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
            if (
                self.workspace.state.owner == "main"
                and self.main_turns >= self.config.max_main_turns
            ):
                return self._finish(
                    TerminateMode.MAX_TURNS,
                    turn,
                    steps,
                    f"main turn limit {self.config.max_main_turns} reached",
                )
            if time.monotonic() - started >= self.config.max_time_s:
                return self._finish(
                    TerminateMode.TIMEOUT,
                    turn,
                    steps,
                    f"time limit {self.config.max_time_s:.0f}s reached",
                )

            owner = self.workspace.state.owner
            packet = self.compiler.compile(self.workspace)
            image = self.renderer.render(packet)
            messages = self._messages(owner, packet, image)
            provider = (
                self.imagination_provider if owner == "imagination" else self.main_provider
            )
            tools = self.imagination_tools if owner == "imagination" else self.main_tools
            turn += 1
            if owner == "main":
                self.main_turns += 1
            else:
                self.imagination_turns += 1
                self._set_private_imagination_turn(self.imagination_turns)
            try:
                response = provider.generate(messages, tools)
            except KeyboardInterrupt:
                return self._finish(TerminateMode.CANCELLED, turn, steps, "interrupted by user")
            except Exception as exc:
                return self._finish(
                    TerminateMode.ERROR, turn, steps, f"{type(exc).__name__}: {exc}"
                )
            self._accumulate_usage(response, owner)

            call, protocol_error = self._single_call(response)
            if protocol_error is not None:
                self._feedback[owner] = protocol_error
                self._log(turn, owner, image, packet, None, None, response)
                continue

            assert call is not None
            if owner == "main" and call.name == "commit":
                if self.physical_ops >= self.config.max_physical_ops:
                    return self._finish(
                        TerminateMode.MAX_TURNS,
                        turn,
                        steps,
                        f"physical operation limit {self.config.max_physical_ops} reached",
                    )
                self.physical_ops += 1
            if owner == "main" and call.name in IMAGINATION_STARTERS:
                self.workspace.set_refinement_goal(response.text)

            step = self._dispatch(owner, call)
            if not step.ok:
                self._feedback[owner] = str(step.result.get("error", "Function failed"))
            physical = owner == "main" and call.name == "commit"
            if physical:
                self._capture_previous_observed()
                self._update_env_success()
            elif owner == "main" and call.name == "done":
                self._update_env_success()

            record = _step_record(turn, owner, step, response.text, physical=physical)
            steps.append(record)
            self._log(turn, owner, image, packet, _call_summary(call), step, response)
            self._notify(turn, record)

            if (
                owner == "main"
                and self.workspace.state.owner == "imagination"
            ) or (
                owner == "imagination" and self.workspace.state.owner == "main"
            ):
                self.imagination_turns = 0
            elif (
                owner == "imagination"
                and self.imagination_turns >= self.config.max_imagination_turns
                and self.workspace.state.owner == "imagination"
            ):
                limit_step = self.workspace.limit_imagination()
                limit_record = _step_record(
                    turn, "runtime", limit_step, "imagination turn limit", physical=False
                )
                steps.append(limit_record)
                self.imagination_turns = 0

    def _messages(
        self, owner: str, packet: ContextPacket, context_image: np.ndarray
    ) -> list[Message]:
        feedback = self._feedback[owner]
        self._feedback[owner] = None
        if owner == "imagination":
            imagination = self.workspace.state.imagination
            assert imagination is not None
            text = (
                f"User Task：{self.workspace.state.task_prompt}\n"
                f"Refinement Goal：{imagination.refinement_goal}\n"
                "Current ActionTarget："
                + json.dumps(imagination.target.summary(), ensure_ascii=False, separators=(",", ":"))
            )
            if feedback:
                text += f"\n上轮协议错误：{feedback}"
            return _image_messages(IMAGINATION_SYSTEM_PROMPT, text, context_image)

        text = (
            f"User Task：{self.workspace.state.task_prompt}\n"
            "Current Policy State："
            + json.dumps(packet.manifest(), ensure_ascii=False, separators=(",", ":"))
        )
        handoff = self.workspace.state.last_handoff
        if handoff is not None:
            text += "\nLatest Imagination Handoff：" + json.dumps(
                handoff.summary(), ensure_ascii=False, separators=(",", ":")
            )
        if feedback:
            text += f"\n上轮协议错误：{feedback}"
        return _image_messages(
            SYSTEM_PROMPT,
            text,
            context_image,
            previous_observed=self._previous_observed,
        )

    def _single_call(
        self, response: ModelResponse
    ) -> tuple[ToolCall | None, str | None]:
        if not response.tool_calls:
            return None, NO_CALL_FEEDBACK
        if len(response.tool_calls) != 1:
            return None, MULTI_CALL_ERROR
        return response.tool_calls[0], None

    def _dispatch(self, owner: str, call: ToolCall) -> ContextStepResult:
        if call.parse_error:
            return self.workspace.reject(call.name, call.args, call.parse_error)
        allowed = IMAGINATION_FUNCTION_NAMES if owner == "imagination" else FUNCTION_NAMES
        try:
            name, arguments = parse_action(
                {"name": call.name, "arguments": call.args}, allowed=allowed
            )
        except ValueError as exc:
            return self.workspace.reject(call.name, call.args, str(exc))
        step = self.workspace.execute(name, **arguments)
        if owner == "main":
            self.workspace.state.last_handoff = None
        if owner == "main" and self._previous_observed is not None:
            self._previous_observed = None
            self.workspace._private.previous_observation = None
        return step

    def _capture_previous_observed(self) -> None:
        try:
            camera = self.workspace._private.camera(
                self.workspace.camera_name, previous=True
            )
            self._previous_observed = np.asarray(camera["images"]["rgb"], dtype=np.uint8).copy()
        except (KeyError, RuntimeError, TypeError, ValueError):
            self._previous_observed = None

    def _set_private_imagination_turn(self, count: int) -> None:
        artifacts = self.workspace._private.imagination_artifacts
        if artifacts is None:
            return
        from dataclasses import replace

        self.workspace._private.imagination_artifacts = replace(artifacts, turn_count=count)

    def _finish(
        self, mode: TerminateMode, turn: int, steps: list[StepRecord], detail: str
    ) -> EpisodeResult:
        if not self.workspace.finished:
            step = self.workspace.execute("done", success=False)
            steps.append(_step_record(turn, "runtime", step, "", physical=False))
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
                    "main_turns": self.main_turns,
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
        owner: str,
        image: np.ndarray,
        packet: ContextPacket,
        call: dict[str, Any] | None,
        step: ContextStepResult | None,
        response: ModelResponse,
    ) -> None:
        if self.trace is None:
            return
        self.trace.log_turn(
            turn=turn,
            agent_owner=owner,
            image=image,
            packet=packet,
            function_call=call,
            step=step,
            thought=response.text,
            env_success=self.env_success,
            done=self.workspace.finished,
            raw_response_text=response.raw_response_text or response.text,
            provider_reasoning=response.provider_reasoning,
            state_summary=self.workspace.state.trace_summary(),
        )

    def _notify(self, turn: int, record: StepRecord) -> None:
        if self.config.on_turn is not None:
            self.config.on_turn(turn, record)

    def _accumulate_usage(self, response: ModelResponse, owner: str) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = response.usage.get(key)
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value
                scoped = f"{owner}_{key}"
                self.usage[scoped] = self.usage.get(scoped, 0) + value


def _image_messages(
    system_prompt: str,
    text: str,
    image: np.ndarray,
    *,
    previous_observed: np.ndarray | None = None,
) -> list[Message]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": text},
        {"type": "text", "text": "CURRENT CONTEXT CANVAS"},
        {"type": "image_url", "image_url": {"url": encode_png_data_url(image)}},
    ]
    if previous_observed is not None:
        content.extend(
            [
                {"type": "text", "text": "PREVIOUS OBSERVED BEFORE LAST COMMIT"},
                {
                    "type": "image_url",
                    "image_url": {"url": encode_png_data_url(previous_observed)},
                },
            ]
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def run_context_episode(
    main_provider: ModelProvider,
    api: Any,
    task_prompt: str,
    renderer: ContextRenderer,
    *,
    imagination_provider: ModelProvider | None = None,
    trace_dir: str | None = None,
    config: ContextRunConfig | None = None,
    env_check: Callable[[], bool] | None = None,
    motion_backend: str = "curobo",
) -> EpisodeResult:
    workspace = ContextWorkspace(api, task_prompt, motion_backend=motion_backend)
    trace = ContextTraceLogger(trace_dir) if trace_dir is not None else None
    try:
        return ContextRuntime(
            main_provider,
            workspace,
            renderer,
            imagination_provider=imagination_provider,
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
    owner: str,
    step: ContextStepResult,
    thought: str,
    *,
    physical: bool,
) -> StepRecord:
    return StepRecord(
        turn=turn,
        op=f"{owner}:{step.function_name}",
        args=step.arguments,
        ok=step.ok,
        physical=physical,
        result=json.dumps(step.result, ensure_ascii=False, separators=(",", ":")),
        thought=thought.strip(),
    )


__all__ = [
    "ContextRenderer",
    "ContextRunConfig",
    "ContextRuntime",
    "run_context_episode",
]
