"""Bounded-state Main/Imagination orchestration for the VAW Context Runtime."""

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
from vaw.context_runtime.private import build_edit_summary
from vaw.context_runtime.protocol import (
    ACTION_REVIEW_SYSTEM_PROMPT,
    IMAGINATION_FUNCTION_NAMES,
    IMAGINATION_SYSTEM_PROMPT,
    REVIEW_FUNCTION_NAMES,
    STANDARD_MAIN_FUNCTION_NAMES,
    SYSTEM_PROMPT,
    imagination_function_definitions,
    main_function_definitions,
    parse_action,
    review_function_definitions,
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
        env_terminal_check: Callable[[], bool] | None = None,
    ) -> None:
        self.main_provider = main_provider
        self.imagination_provider = imagination_provider or main_provider
        self.workspace = workspace
        self.renderer = renderer
        self.compiler = compiler or ContextCompiler()
        self.config = config or ContextRunConfig()
        self.trace = trace
        self.env_check = env_check
        self.env_terminal_check = env_terminal_check
        self.main_tools = main_function_definitions()
        self.review_tools = review_function_definitions()
        self.imagination_tools = imagination_function_definitions()
        self.usage: dict[str, int] = {}
        self.env_success: bool | None = None
        self.physical_ops = 0
        self.main_turns = 0
        self.imagination_turns = 0
        self._feedback: dict[str, str | None] = {"main": None, "imagination": None}
        # One overwrite-only Main belief bridges adjacent semantic decisions.
        # This is intentionally not a transcript: no old images, tool payloads
        # or Imagination rationale are replayed.
        self._main_working_focus: str | None = None
        if self.trace is not None:
            self.trace.log_meta(
                {
                    "context_schema": CONTEXT_SCHEMA,
                    "renderer": renderer.name,
                    "task_prompt": workspace.state.task_prompt,
                    "orchestration": "main-imagination-single-working-focus",
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
            visible_main_focus = self._visible_main_working_focus(owner)
            messages = self._messages(owner, packet, image)
            provider = (
                self.imagination_provider if owner == "imagination" else self.main_provider
            )
            tools = (
                self.imagination_tools
                if owner == "imagination"
                else self.review_tools
                if self.workspace.state.action_review is not None
                else self.main_tools
            )
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
                self._log(
                    turn,
                    owner,
                    image,
                    packet,
                    None,
                    None,
                    response,
                    main_working_focus=visible_main_focus,
                )
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
            step = self._dispatch(owner, call)
            if not step.ok:
                self._feedback[owner] = str(step.result.get("error", "Function failed"))
            physical = owner == "main" and call.name == "commit"
            if physical or (owner == "main" and call.name == "done"):
                self._update_env_success()

            record = _step_record(turn, owner, step, response.text, physical=physical)
            steps.append(record)
            self._log(
                turn,
                owner,
                image,
                packet,
                _call_summary(call),
                step,
                response,
                main_working_focus=visible_main_focus,
            )
            if owner == "main":
                self._update_main_working_focus(response.text)
            self._notify(turn, record)

            if physical and self._environment_terminated():
                return self._result(
                    TerminateMode.ENV_TERMINATED,
                    turn,
                    steps,
                    "environment terminated during physical execution",
                )

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
                if self.trace is not None:
                    self.trace.log_event(
                        "imagination_handoff",
                        {
                            "turn": turn,
                            "termination_reason": "turn_limit",
                            "result": limit_step.result,
                            "runtime_diagnostics": limit_step.trace_diagnostics,
                        },
                    )
                self.imagination_turns = 0

    def _messages(
        self, owner: str, packet: ContextPacket, context_image: np.ndarray
    ) -> list[Message]:
        feedback = self._feedback[owner]
        self._feedback[owner] = None
        if owner == "imagination":
            imagination = self.workspace.state.imagination
            assert imagination is not None
            edit_summary = build_edit_summary(
                imagination.target,
                self.workspace._private.imagination_artifacts,
            )
            opening = (
                self.workspace.state.robot.gripper_opening
                if self.workspace.state.robot is not None
                else None
            )
            text = (
                f"Refinement Goal：{imagination.refinement_goal}\n"
                "Edit Summary："
                + json.dumps(
                    edit_summary.summary(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            if imagination.target.gripper is None and opening is not None:
                text += (
                    "\nTarget Gripper：inherit observed opening "
                    f"{float(opening):.3f}（0≈闭合，1≈张开）"
                )
            if feedback:
                text += f"\n上轮协议错误：{feedback}"
            return _image_messages(IMAGINATION_SYSTEM_PROMPT, text, context_image)

        text = (
            f"User Task：{self.workspace.state.task_prompt}\n"
            "Current Policy State："
            + json.dumps(packet.manifest(), ensure_ascii=False, separators=(",", ":"))
        )
        visible_focus = self._visible_main_working_focus(owner)
        if visible_focus is not None:
            text += (
                "\nMain Working Focus（上一轮 Main 的可覆盖 belief，不是真值）："
                + visible_focus
            )
        handoff = self.workspace.state.last_handoff
        if handoff is not None:
            text += "\nLatest Imagination Handoff：" + json.dumps(
                handoff.summary(), ensure_ascii=False, separators=(",", ":")
            )
        review = self.workspace.state.action_review
        if review is not None:
            review_artifacts = self.workspace._private.review_artifacts.get(
                review.action_id
            )
            if (
                review_artifacts is not None
                and review_artifacts.edit_summary is not None
            ):
                text += "\nAction Review Edit Summary：" + json.dumps(
                    review_artifacts.edit_summary.summary(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        last_physical = self.workspace.state.last_physical_action
        if (
            last_physical is not None
            and (
                (
                    self.workspace.state.imagination is None
                    and self.workspace.state.action_review is None
                )
                or last_physical.outcome == "completed"
            )
        ):
            text += "\nLast Physical Action：" + json.dumps(
                last_physical.summary(), ensure_ascii=False, separators=(",", ":")
            )
        verification = packet.world.physical_verification
        if verification is not None:
            text += "\nPhysical Effect Verification：" + json.dumps(
                verification.summary(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if feedback:
            text += f"\n上轮协议错误：{feedback}"
        system_prompt = (
            ACTION_REVIEW_SYSTEM_PROMPT
            if self.workspace.state.action_review is not None
            else SYSTEM_PROMPT
        )
        return _image_messages(system_prompt, text, context_image)

    def _visible_main_working_focus(self, owner: str) -> str | None:
        if owner != "main":
            return None
        return self._main_working_focus

    def _update_main_working_focus(self, text: str) -> None:
        normalized = " ".join(str(text).split())
        if normalized:
            self._main_working_focus = normalized

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
        allowed = (
            IMAGINATION_FUNCTION_NAMES
            if owner == "imagination"
            else REVIEW_FUNCTION_NAMES
            if self.workspace.state.action_review is not None
            else STANDARD_MAIN_FUNCTION_NAMES
        )
        arguments = dict(call.args)
        if owner == "main" and call.name in IMAGINATION_STARTERS:
            raw_goal = arguments.pop("refinement_goal", None)
            if not isinstance(raw_goal, str) or not raw_goal.strip():
                return self.workspace.reject(
                    call.name,
                    call.args,
                    "refinement_goal is required when Main starts Imagination",
                )
            self.workspace.set_refinement_goal(raw_goal)
        try:
            name, arguments = parse_action(
                {"name": call.name, "arguments": arguments}, allowed=allowed
            )
        except ValueError as exc:
            return self.workspace.reject(call.name, call.args, str(exc))
        review_before = self.workspace.state.action_review
        step = self.workspace.execute(name, **arguments)
        if owner == "main" and name != "commit" and step.ok:
            # A rejected Function is not a valid decision: retain the causal
            # comparison so Main can repair its arguments without losing what
            # just happened.  A reviewed action is a one-decision offer: any
            # other successful Main call explicitly declines it, so a stale
            # target cannot be committed on a later turn.  A successful commit
            # refreshes/replaces causal state atomically inside the workspace.
            if (
                review_before is not None
                and self.workspace.state.action_review is review_before
            ):
                self.workspace.discard_action_review()
            self.workspace.consume_main_context()
        return step

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

    def _environment_terminated(self) -> bool:
        if self.env_terminal_check is None:
            return False
        try:
            return bool(self.env_terminal_check())
        except Exception:
            return False

    def _log(
        self,
        turn: int,
        owner: str,
        image: np.ndarray,
        packet: ContextPacket,
        call: dict[str, Any] | None,
        step: ContextStepResult | None,
        response: ModelResponse,
        *,
        main_working_focus: str | None,
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
            main_working_focus=main_working_focus,
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
) -> list[Message]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": text},
        {"type": "text", "text": "CURRENT CONTEXT CANVAS"},
        {"type": "image_url", "image_url": {"url": encode_png_data_url(image)}},
    ]
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
    env_terminal_check: Callable[[], bool] | None = None,
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
            env_terminal_check=env_terminal_check,
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
