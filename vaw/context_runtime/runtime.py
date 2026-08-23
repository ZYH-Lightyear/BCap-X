"""Main-owned ReAct runtime with an encapsulated Imagination sub-agent."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
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
from vaw.context_runtime.memory import FunctionEvent, project_function_event
from vaw.context_runtime.model import LastPhysicalAction
from vaw.context_runtime.packet import (
    CONTEXT_SCHEMA,
    ContextCompiler,
    ContextPacket,
    encode_png_data_url,
)
from vaw.context_runtime.private import build_edit_summary
from vaw.context_runtime.protocol import (
    IMAGINATION_FUNCTION_NAMES,
    IMAGINATION_SYSTEM_PROMPT,
    MAIN_FUNCTION_NAMES,
    SYSTEM_PROMPT,
    imagination_function_definitions,
    main_function_definitions,
    parse_action,
)
from vaw.context_runtime.trace import ContextTraceLogger, SubagentTraceLogger
from vaw.context_runtime.workspace import ContextStepResult, ContextWorkspace

NO_CALL_FEEDBACK = "未执行：本轮必须且只能调用一个 Function。"
MULTI_CALL_ERROR = "未执行：一轮只能调用一个 Function。"
_NOTICE_LABEL = {
    "protocol": "上轮协议错误",
    "function": "上一轮执行失败",
    "advisory": "上一轮编辑未生效，Preview 未改变",
}

MAX_MAIN_TURNS = 32


class ContextRenderer(Protocol):
    name: str

    def render(self, packet: ContextPacket) -> Any: ...

    def close(self) -> None: ...


@dataclass
class ContextRunConfig:
    max_main_turns: int = MAX_MAIN_TURNS
    max_imagination_turns: int = 6
    max_time_s: float = 1800.0
    max_physical_ops: int = 30
    # How many extra model queries share one Main turn after a missing or
    # multi Function call.  0 restores the old "silence burns a turn" rule.
    max_no_call_retries: int = 2
    on_turn: Callable[[int, StepRecord], None] | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_main_turns <= MAX_MAIN_TURNS:
            raise ValueError(f"max_main_turns must be in [1, {MAX_MAIN_TURNS}]")
        if not 0 <= self.max_no_call_retries <= 3:
            raise ValueError("max_no_call_retries must be in [0, 3]")


@dataclass(frozen=True)
class ImaginationResult:
    status: str
    action_id: str | None
    turns: int
    result: dict[str, Any]


class ImaginationRunner:
    """Run one isolated local-control task synchronously for the Main Agent."""

    def __init__(
        self,
        provider: ModelProvider,
        workspace: ContextWorkspace,
        renderer: ContextRenderer,
        compiler: ContextCompiler,
        *,
        max_turns: int,
        trace: SubagentTraceLogger | None = None,
        usage_callback: Callable[[ModelResponse, str], None] | None = None,
    ) -> None:
        self.provider = provider
        self.workspace = workspace
        self.renderer = renderer
        self.compiler = compiler
        self.max_turns = max_turns
        self.trace = trace
        self.tools = imagination_function_definitions()
        self.usage_callback = usage_callback

    def run(self, instruction: str, action_id: str) -> ImaginationResult:
        try:
            self.workspace.begin_imagination(instruction, action_id)
        except Exception as exc:
            return self._close(ImaginationResult("failed", action_id, 0, {"error": str(exc)}))

        feedback: tuple[str, str] | None = None
        for turn in range(1, self.max_turns + 1):
            self._set_turn(turn)
            packet = self.compiler.compile_imagination(self.workspace)
            image = self.renderer.render(packet)
            messages = self._messages(packet, image, feedback)
            try:
                response = self.provider.generate(messages, self.tools)
            except Exception as exc:
                step = self.workspace.fail_imagination("subagent_error")
                result = dict(step.result)
                self._log(
                    turn,
                    image,
                    packet,
                    None,
                    step,
                    "",
                    {"error": f"{type(exc).__name__}: {exc}", **result},
                )
                return self._close(
                    ImaginationResult(
                        "failed",
                        str(result["action_id"]) if "action_id" in result else None,
                        turn,
                        result,
                    )
                )
            if self.usage_callback is not None:
                self.usage_callback(response, "imagination")
            call, error = _single_call(response)
            if error is not None:
                feedback = ("protocol", error)
                self._log(turn, image, packet, None, None, response.text, {"error": error})
                continue
            assert call is not None
            step = self._dispatch(call)
            self._log(turn, image, packet, _call_summary(call), step, response.text, step.result)
            if not step.ok:
                feedback = ("function", str(step.result.get("error", "Function failed")))
                continue
            if call.name == "finish_imagination":
                status = str(step.result.get("status", "failed"))
                return self._close(
                    ImaginationResult(
                        status,
                        str(step.result["action_id"]) if "action_id" in step.result else None,
                        turn,
                        dict(step.result),
                    )
                )
            # A call can succeed as a transaction and still leave the target
            # untouched.  Such a step carries a reason instead of an error;
            # without it the next turn would edit on from an unchanged Canvas
            # believing the previous edit landed.
            advisory = step.result.get("reason")
            feedback = ("advisory", str(advisory)) if advisory else None

        step = self.workspace.limit_imagination()
        # Budget exhaustion hands the last planner-checked edit back to Main
        # as a partial refinement; only a session with nothing executable
        # rolls back.  The last focused turn is already logged above; do not
        # fabricate another Imagination image from Main-owned state.
        return self._close(
            ImaginationResult(
                str(step.result.get("status", "failed")),
                str(step.result["action_id"]) if "action_id" in step.result else None,
                self.max_turns,
                dict(step.result),
            )
        )

    def _dispatch(self, call: ToolCall) -> ContextStepResult:
        if call.parse_error:
            return self.workspace.reject(call.name, call.args, call.parse_error)
        try:
            name, arguments = parse_action(
                {"name": call.name, "arguments": call.args},
                allowed=IMAGINATION_FUNCTION_NAMES,
            )
        except ValueError as exc:
            return self.workspace.reject(call.name, call.args, str(exc))
        return self.workspace.execute_imagination(name, **arguments)

    def _close(self, result: ImaginationResult) -> ImaginationResult:
        if self.trace is not None:
            meta: dict[str, Any] = {"status": result.status}
            if result.action_id is not None:
                meta["action_id"] = result.action_id
            reason = result.result.get("reason")
            if isinstance(reason, str):
                meta["reason"] = reason
            self.trace.log_meta(meta)
        return result

    def _messages(
        self,
        packet: ContextPacket,
        image: np.ndarray,
        feedback: tuple[str, str] | None,
    ) -> list[Message]:
        session = self.workspace.state.imagination
        action = self.workspace.state.action_proposal
        assert session is not None and action is not None
        summary = build_edit_summary(action.target, self.workspace._private.action_artifacts)
        artifacts = self.workspace._private.action_artifacts
        current_turn = artifacts.turn_count if artifacts is not None else 0
        target_basis = _target_basis(artifacts)
        carried_geometry = (
            "available_as_amber_proxy"
            if self.workspace._private.attachment_hypothesis is not None
            else "unavailable_use_gripper_only_fallback"
        )
        text = (
            f"Imagination Task：{session.instruction}\n"
            f"Target Basis：{target_basis}\n"
            f"Carried Geometry：{carried_geometry}\n"
            "Current Edit Summary："
            + json.dumps(summary.summary(), ensure_ascii=False, separators=(",", ":"))
            + f"\nRemaining Edit Turns：{max(0, self.max_turns - current_turn + 1)}"
        )
        if feedback is not None:
            text += f"\n{_NOTICE_LABEL[feedback[0]]}：{feedback[1]}"
        return _image_messages(IMAGINATION_SYSTEM_PROMPT, text, image, label="FOCUSED IMAGINATION CANVAS")

    def _set_turn(self, count: int) -> None:
        artifacts = self.workspace._private.action_artifacts
        if artifacts is not None:
            self.workspace._private.action_artifacts = replace(artifacts, turn_count=count)

    def _log(
        self,
        turn: int,
        image: np.ndarray,
        packet: ContextPacket,
        call: dict[str, Any] | None,
        step: ContextStepResult | None,
        thought: str,
        result: dict[str, Any],
    ) -> None:
        if self.trace is not None:
            self.trace.log_turn(
                turn=turn,
                image=image,
                packet=packet,
                function_call=call,
                step=step,
                thought=thought,
                result=result,
            )


class ContextRuntime:
    """Run one Main ReAct loop; sub-agent work is one synchronous Function."""

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
        self.usage: dict[str, int] = {}
        self.env_success: bool | None = None
        self.physical_ops = 0
        self.main_turns = 0
        self._current_event: FunctionEvent | None = None
        self._subagent_index = 0
        if trace is not None:
            trace.log_meta(
                {
                    "context_schema": CONTEXT_SCHEMA,
                    "renderer": renderer.name,
                    "task_prompt": workspace.state.task_prompt,
                    "orchestration": "main-react-with-imagination-subagent",
                    "max_imagination_turns": self.config.max_imagination_turns,
                    "motion_backend": workspace.motion_backend_name,
                    "local_motion_backend": workspace.local_motion_backend_name,
                    "preview_gripper": self.compiler.preview_gripper_style,
                }
            )

    def run(self) -> EpisodeResult:
        started = time.monotonic()
        steps: list[StepRecord] = []
        while True:
            if self.workspace.finished:
                return self._result(TerminateMode.GOAL, steps)
            if self.main_turns >= self.config.max_main_turns:
                return self._finish(TerminateMode.MAX_TURNS, steps, "main turn limit reached")
            if time.monotonic() - started >= self.config.max_time_s:
                return self._finish(TerminateMode.TIMEOUT, steps, "time limit reached")

            packet = self.compiler.compile(self.workspace)
            image = self.renderer.render(packet)
            messages = self._main_messages(packet, image)
            self.main_turns += 1
            turn = self.main_turns
            try:
                call, response = self._solicit_single_call(
                    packet, image, messages, turn
                )
            except KeyboardInterrupt:
                return self._finish(TerminateMode.CANCELLED, steps, "interrupted by user")
            except Exception as exc:
                return self._finish(TerminateMode.ERROR, steps, f"{type(exc).__name__}: {exc}")
            if call is None:
                continue
            requested_physical = call.name in self.workspace.PHYSICAL_FUNCTIONS
            if (
                requested_physical
                and self.physical_ops >= self.config.max_physical_ops
            ):
                return self._finish(
                    TerminateMode.MAX_TURNS,
                    steps,
                    "physical operation limit reached",
                )

            if call.name == "call_imagination":
                step = self._call_imagination(call)
            else:
                step = self._dispatch_main(call)
            physical = requested_physical and step.revision_after > step.revision_before
            if physical:
                self.physical_ops += 1
            if physical or call.name == "done":
                self._update_env_success()
            record = _step_record(turn, "main", step, response.text, physical=physical)
            steps.append(record)
            self._log(turn, image, packet, _call_summary(call), step, response)
            self._current_event = project_function_event(
                call.name,
                step.result,
                world_changed=step.revision_after > step.revision_before,
            )
            self._notify(turn, record)
            if physical and self._environment_terminated():
                return self._result(TerminateMode.ENV_TERMINATED, steps, "environment terminated")

    def _call_imagination(self, call: ToolCall) -> ContextStepResult:
        if call.parse_error:
            return self.workspace.reject(call.name, call.args, call.parse_error)
        try:
            _name, arguments = parse_action(
                {"name": call.name, "arguments": call.args}, allowed=("call_imagination",)
            )
            instruction = str(arguments["instruction"])
            action_id = str(arguments["action_id"])
        except (KeyError, ValueError) as exc:
            return self.workspace.reject(call.name, call.args, str(exc))
        before = self.workspace.state.observation_revision
        subtrace = None
        subtrace_rel = None
        if self.trace is not None:
            self._subagent_index += 1
            subtrace = self.trace.new_subagent_trace(self._subagent_index, instruction)
            subtrace_rel = str(subtrace.dir.relative_to(self.trace.dir))
        result = ImaginationRunner(
            self.imagination_provider,
            self.workspace,
            self.renderer,
            self.compiler,
            max_turns=self.config.max_imagination_turns,
            trace=subtrace,
            usage_callback=self._accumulate_usage,
        ).run(instruction, action_id)
        payload: dict[str, Any] = {"status": result.status}
        if result.action_id is not None:
            payload["action_id"] = result.action_id
        reason = result.result.get("reason")
        if isinstance(reason, str):
            payload["reason"] = reason
        advisory = result.result.get("advisory")
        if isinstance(advisory, str):
            payload["advisory"] = advisory
        if result.turns == 0 and "error" in result.result:
            payload["error"] = result.result["error"]
        diagnostics: dict[str, Any] = {
            "turns": result.turns,
            "status": result.status,
        }
        if subtrace_rel is not None:
            diagnostics["trace"] = subtrace_rel
        return ContextStepResult(
            function_name="call_imagination",
            arguments=dict(call.args),
            result=payload,
            revision_before=before,
            revision_after=self.workspace.state.observation_revision,
            manifest=self.workspace.state.manifest(),
            trace_diagnostics={"subagent": diagnostics},
        )

    def _dispatch_main(self, call: ToolCall) -> ContextStepResult:
        if call.parse_error:
            return self.workspace.reject(call.name, call.args, call.parse_error)
        try:
            name, arguments = parse_action(
                {"name": call.name, "arguments": call.args}, allowed=MAIN_FUNCTION_NAMES
            )
        except ValueError as exc:
            return self.workspace.reject(call.name, call.args, str(exc))
        return self.workspace.execute(name, **arguments)

    def _solicit_single_call(
        self,
        packet: ContextPacket,
        image: np.ndarray,
        messages: list[Message],
        turn: int,
    ) -> tuple[ToolCall | None, ModelResponse]:
        """Query Main until one Function call arrives or the retry budget ends.

        Intermediate misses do not write a Context turn.  They stay in the
        runtime event log so a later formal turn can still consume the slot.
        """

        attempts = 0
        while True:
            response = self.main_provider.generate(messages, self.main_tools)
            self._accumulate_usage(response, "main")
            call, error = _single_call(response)
            if error is None:
                assert call is not None
                return call, response
            attempts += 1
            if self.trace is not None:
                self.trace.log_event(
                    "no_call_retry",
                    {
                        "turn": turn,
                        "attempt": attempts,
                        "error": error,
                        "text": response.text[:2000],
                    },
                )
            if attempts > self.config.max_no_call_retries:
                self._current_event = FunctionEvent(
                    function="protocol",
                    status="failed",
                    message=error,
                )
                self._log(turn, image, packet, None, None, response)
                return None, response
            messages = self._main_messages(packet, image, feedback=error)

    def _main_messages(
        self,
        packet: ContextPacket,
        image: np.ndarray,
        feedback: str | None = None,
    ) -> list[Message]:
        text = (
            f"User Task：{self.workspace.state.task_prompt}\n"
            "Task Memory："
            + json.dumps(
                self.workspace.state.task_memory.summary(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\nLive References："
            + json.dumps(packet.manifest(), ensure_ascii=False, separators=(",", ":"))
        )
        attachment = self.workspace._private.attachment_hypothesis
        continuity = _control_continuity(
            self.workspace.state.last_physical_action,
            manipulation_subject=(attachment.query if attachment is not None else None),
        )
        if continuity is not None:
            text += "\nControl Continuity：" + json.dumps(
                continuity,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if self._current_event is not None:
            text += "\nCurrent Function Event：" + json.dumps(
                self._current_event.summary(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if feedback is not None:
            text += f"\n{_NOTICE_LABEL['protocol']}：{feedback}"
        return _image_messages(SYSTEM_PROMPT, text, image, label="CURRENT MAIN CONTEXT CANVAS")

    def _finish(self, mode: TerminateMode, steps: list[StepRecord], detail: str) -> EpisodeResult:
        if not self.workspace.finished:
            step = self.workspace.execute("done", success=False)
            steps.append(_step_record(self.main_turns, "runtime", step, "", physical=False))
        self._update_env_success()
        return self._result(mode, steps, detail)

    def _result(self, mode: TerminateMode, steps: list[StepRecord], detail: str = "") -> EpisodeResult:
        if self.trace is not None:
            self.trace.log_meta(
                {
                    "terminate_mode": mode.value,
                    "turns": self.main_turns,
                    "main_turns": self.main_turns,
                    "claimed_success": self.workspace.claimed_success,
                    "env_success": self.env_success,
                    "usage": self.usage,
                }
            )
        return EpisodeResult(
            terminate_mode=mode,
            turns=self.main_turns,
            claimed_success=self.workspace.claimed_success,
            env_success=self.env_success,
            detail=detail,
            steps=list(steps),
            usage=dict(self.usage),
            trace_dir=str(self.trace.dir) if self.trace is not None else None,
        )

    def _log(
        self,
        turn: int,
        image: np.ndarray,
        packet: ContextPacket,
        call: dict[str, Any] | None,
        step: ContextStepResult | None,
        response: ModelResponse,
    ) -> None:
        if self.trace is not None:
            self.trace.log_turn(
                turn=turn,
                agent_owner="main",
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
        if self.config.on_turn is None:
            return
        # The observer runs after the transaction committed and the trace was
        # written; it must never be able to kill an episode (a real crash was
        # a BlockingIOError from a print into a non-blocking stdout pipe).
        try:
            self.config.on_turn(turn, record)
        except Exception:
            pass

    def _accumulate_usage(self, response: ModelResponse, scope: str) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = response.usage.get(key)
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value
                scoped = f"{scope}_{key}"
                self.usage[scoped] = self.usage.get(scoped, 0) + value

    def _update_env_success(self) -> None:
        if self.env_check is not None:
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


def _single_call(response: ModelResponse) -> tuple[ToolCall | None, str | None]:
    if not response.tool_calls:
        return None, NO_CALL_FEEDBACK
    if len(response.tool_calls) != 1:
        return None, MULTI_CALL_ERROR
    return response.tool_calls[0], None


def _target_basis(artifacts: Any | None) -> str:
    """Describe the one geometric prior relevant to focused refinement."""

    context = artifacts.planning_context if artifacts is not None else None
    if context is None:
        return "relative pose from current TCP"
    if context.source_kind == "point":
        return (
            "coarse point + offset anchor; refine only from visible target geometry, "
            "not from assumed perfect centering"
        )
    if context.source_kind == "grasp":
        return "coarse grasp seed; verify and refine visible contact geometry"
    return f"coarse {context.source_kind} seed"


def _control_continuity(
    action: LastPhysicalAction | None,
    *,
    manipulation_subject: str | None = None,
) -> dict[str, Any] | None:
    """Expose only the causal focus of the latest physical command.

    This deliberately omits controller telemetry and does not assert that the
    intended grasp/place relation became true.  Its purpose is to keep a
    revision change from erasing which manipulation problem Main was solving.
    """

    if action is None and manipulation_subject is None:
        return None
    result: dict[str, Any] = {}
    if manipulation_subject:
        result["manipulation_subject"] = manipulation_subject
        result["subject_relation"] = "intended_attachment_unverified"
    if action is not None:
        result.update(
            {
                "last_intent": action.intent,
                "executed_stage": action.executed_stages,
                "command_status": action.outcome,
            }
        )
        if action.source_query:
            result["control_subject"] = action.source_query
        if action.target_gripper is not None:
            result["target_gripper"] = action.target_gripper
    return result


def _image_messages(system_prompt: str, text: str, image: np.ndarray, *, label: str) -> list[Message]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": encode_png_data_url(image)}},
            ],
        },
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
    "ImaginationRunner",
    "ImaginationResult",
    "run_context_episode",
]
