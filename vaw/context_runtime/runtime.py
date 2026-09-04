"""Main-owned ReAct runtime with an encapsulated Imagination sub-agent."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import suppress
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
from vaw.context_runtime.context_projection import (
    EmbodiedStateCard,
    project_embodied_state,
    render_main_context,
)
from vaw.context_runtime.memory import (
    InteractionEvent,
    InteractionMemory,
    interaction_outcome,
)
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
    main_function_registry,
    parse_action,
)
from vaw.context_runtime.trace import ContextTraceLogger, SubagentTraceLogger
from vaw.context_runtime.workspace import ContextStepResult, ContextWorkspace
from vaw.mmskill import MMSkill, MMSkillBuffer, MMSkillLibrary

NO_CALL_FEEDBACK = "未执行：本轮必须且只能调用一个 Function。"
MULTI_CALL_ERROR = "未执行：一轮只能调用一个 Function。"
_NOTICE_LABEL = {
    "protocol": "上轮协议错误",
    "function": "上一轮执行失败",
    "advisory": "上一轮编辑未生效，Preview 未改变",
}

class ContextRenderer(Protocol):
    name: str

    def render(self, packet: ContextPacket) -> Any: ...

    def close(self) -> None: ...


class ActionMediaRecorder(Protocol):
    """Optional observer around one physical Function transaction."""

    def start(
        self,
        *,
        turn: int,
        function: str,
        arguments: dict[str, Any],
        revision_before: int,
    ) -> Any: ...

    def finish(
        self,
        token: Any,
        *,
        outcome: str,
        revision_after: int,
    ) -> dict[str, Any] | None: ...


@dataclass
class ContextRunConfig:
    max_main_turns: int = 32
    max_imagination_turns: int = 6
    max_time_s: float = 1800.0
    max_physical_ops: int = 30
    # How many extra model queries share one Main turn after a missing or
    # multi Function call.  0 restores the old "silence burns a turn" rule.
    max_no_call_retries: int = 2
    on_turn: Callable[[int, StepRecord], None] | None = None


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
        turn_callback: Callable[[int], None] | None = None,
        active_mmskill: MMSkill | None = None,
        reference_mmskill: MMSkill | None = None,
    ) -> None:
        self.provider = provider
        self.workspace = workspace
        self.renderer = renderer
        self.compiler = compiler
        self.max_turns = max_turns
        self.trace = trace
        self.tools = imagination_function_definitions()
        self.usage_callback = usage_callback
        self.turn_callback = turn_callback
        self.active_mmskill = active_mmskill
        self.reference_mmskill = reference_mmskill

    def run(self, instruction: str, action_id: str | None) -> ImaginationResult:
        try:
            self.workspace.begin_imagination(instruction, action_id)
        except Exception as exc:
            return self._close(ImaginationResult("failed", action_id, 0, {"error": str(exc)}))

        feedback: tuple[str, str] | None = None
        for turn in range(1, self.max_turns + 1):
            self._set_turn(turn)
            packet = self.compiler.compile_imagination(
                self.workspace,
                active_mmskill=self.reference_mmskill,
            )
            image = self.renderer.render(packet)
            messages = self._messages(packet, image, feedback)
            visible_messages, visible_tools = _provider_visible_request(
                self.provider,
                messages,
                self.tools,
            )
            if self.trace is not None:
                self.trace.log_model_request(
                    turn=turn,
                    attempt=1,
                    owner="imagination",
                    messages=visible_messages,
                    tools=visible_tools,
                )
            try:
                response = self.provider.generate(messages, self.tools)
            except Exception as exc:
                if self.trace is not None:
                    self.trace.log_model_response(
                        turn=turn,
                        attempt=1,
                        owner="imagination",
                        error=f"{type(exc).__name__}: {exc}",
                    )
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
            if self.trace is not None:
                self.trace.log_model_response(
                    turn=turn,
                    attempt=1,
                    owner="imagination",
                    response=response,
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
            "可用：以琥珀色代理体显示"
            if self.workspace._private.attachment_hypothesis is not None
            else "不可用：仅依据夹爪几何判断"
        )
        text = (
            f"想象任务：{session.instruction}\n"
            f"目标依据：{target_basis}\n"
            f"携带物几何：{carried_geometry}\n"
            "当前编辑摘要："
            + json.dumps(summary.summary(), ensure_ascii=False, separators=(",", ":"))
            + f"\n剩余编辑轮数：{max(0, self.max_turns - current_turn + 1)}"
        )
        if feedback is not None:
            text += f"\n{_NOTICE_LABEL[feedback[0]]}：{feedback[1]}"
        if self.active_mmskill is not None:
            text += (
                f"\n本轮加载视觉技能：{self.active_mmskill.skill_id}\n"
                + self.active_mmskill.prompt_block()
            )
        return _image_messages(IMAGINATION_SYSTEM_PROMPT, text, image, label="当前局部想象画布")

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
        if self.turn_callback is not None:
            self.turn_callback(turn)


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
        action_media_recorder: ActionMediaRecorder | None = None,
        skill_library: MMSkillLibrary | None = None,
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
        self.function_registry = main_function_registry()
        self.main_tools = self.function_registry.definitions
        self.interaction_memory = InteractionMemory()
        self.mmskill_library = skill_library or MMSkillLibrary.builtin()
        self.mmskill_buffer = MMSkillBuffer()
        self.usage: dict[str, int] = {}
        self.env_success: bool | None = None
        self.physical_ops = 0
        self.main_turns = 0
        self._subagent_index = 0
        self.action_media_recorder = action_media_recorder
        self._system_prompt = SYSTEM_PROMPT
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
                    "mmskill_library": self.mmskill_library.summary(),
                    "mmskill": self.mmskill_library.provenance(),
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

            packet = self.compiler.compile(
                self.workspace,
                active_mmskill=self.mmskill_buffer.reference_skill,
            )
            image = self.renderer.render(packet)
            decision_turn = self.main_turns + 1
            context_card = self._embodied_state(decision_turn=decision_turn)
            context_snapshot = self._main_context_snapshot(
                packet,
                decision_turn=decision_turn,
                embodied_state=context_card,
            )
            messages = self._messages_from_context(image, context_snapshot)
            context_snapshot_ref = None
            interaction_memory_before = self.interaction_memory.snapshot()
            if self.trace is not None:
                context_snapshot_ref = self.trace.freeze_turn_context(
                    turn=decision_turn,
                    image=image,
                    snapshot=context_snapshot,
                )
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
            if self.trace is not None:
                self.trace.log_event(
                    "model_decision_ready",
                    {
                        "turn": turn,
                        "decision_basis": response.text,
                        "function": call.name,
                        "arguments": dict(call.args),
                    },
                )

            function_spec = self.function_registry.get(call.name)
            requested_physical = (
                function_spec is not None
                and function_spec.world_effect == "physical"
            )
            if (
                requested_physical
                and self.physical_ops >= self.config.max_physical_ops
            ):
                return self._finish(
                    TerminateMode.MAX_TURNS,
                    steps,
                    "physical operation limit reached",
                )

            previous_physical = self.workspace.state.last_physical_action
            if self.trace is not None:
                self.trace.log_event(
                    "function_started",
                    {
                        "turn": turn,
                        "function": call.name,
                        "arguments": dict(call.args),
                        "effect_kind": "action" if requested_physical else "call",
                        "revision": packet.revision,
                    },
                )
            media_token = None
            if requested_physical and self.action_media_recorder is not None:
                with suppress(Exception):
                    media_token = self.action_media_recorder.start(
                        turn=turn,
                        function=call.name,
                        arguments=dict(call.args),
                        revision_before=packet.revision,
                    )
            if call.name == "imagine_action":
                step = self._call_imagination(call)
            elif call.name == "consult_mmskill":
                step = self._consult_mmskill(call)
            else:
                step = self._dispatch_main(call)
            physical = requested_physical and step.revision_after > step.revision_before
            if physical and self.mmskill_buffer.reference_skill is not None:
                skill_id = self.mmskill_buffer.reference_skill.skill_id
                self.mmskill_buffer.hide_references()
                if self.trace is not None:
                    self.trace.log_event(
                        "mmskill_reference_hidden",
                        {
                            "turn": turn,
                            "skill_id": skill_id,
                            "reason": "physical_action",
                        },
                    )
            latest_physical = self.workspace.state.last_physical_action
            physical_outcome = (
                latest_physical.outcome
                if requested_physical
                and latest_physical is not None
                and latest_physical is not previous_physical
                else None
            )
            event = InteractionEvent(
                turn=turn,
                kind="action" if requested_physical else "call",
                function=call.name,
                arguments=dict(step.arguments),
                outcome=interaction_outcome(
                    step.result,
                    physical_outcome=physical_outcome,
                ),
                revision_before=step.revision_before,
                revision_after=step.revision_after,
            )
            action_segment = None
            if media_token is not None and self.action_media_recorder is not None:
                with suppress(Exception):
                    action_segment = self.action_media_recorder.finish(
                        media_token,
                        outcome=event.outcome,
                        revision_after=step.revision_after,
                    )
            self.interaction_memory.record(
                event,
                effect_channel=(
                    function_spec.effect_channel
                    if function_spec is not None
                    else None
                ),
            )
            if physical:
                self.physical_ops += 1
            if physical or call.name == "finish_task":
                self._update_env_success()
            record = _step_record(turn, "main", step, response.text, physical=physical)
            steps.append(record)
            self._log(
                turn,
                image,
                packet,
                _call_summary(call),
                step,
                response,
                embodied_state_card=context_card,
                function_effect_kind=event.kind,
                context_snapshot_ref=context_snapshot_ref,
                interaction_memory_before=interaction_memory_before,
                interaction_event=event.summary(),
                action_segment=action_segment,
            )
            self._notify(turn, record)
            if physical and self._environment_terminated():
                return self._result(TerminateMode.ENV_TERMINATED, steps, "environment terminated")

    def _call_imagination(self, call: ToolCall) -> ContextStepResult:
        if call.parse_error:
            return self.workspace.reject(call.name, call.args, call.parse_error)
        try:
            _name, arguments = parse_action(
                {"name": call.name, "arguments": call.args}, allowed=("imagine_action",)
            )
            instruction = str(arguments["instruction"])
            raw_action_id = arguments.get("action_id")
            action_id = (
                str(raw_action_id).strip()
                if raw_action_id is not None and str(raw_action_id).strip()
                else None
            )
        except (KeyError, ValueError) as exc:
            return self.workspace.reject(call.name, call.args, str(exc))
        before = self.workspace.state.observation_revision
        subtrace = None
        subtrace_rel = None
        if self.trace is not None:
            self._subagent_index += 1
            subtrace = self.trace.new_subagent_trace(self._subagent_index, instruction)
            subtrace_rel = str(subtrace.dir.relative_to(self.trace.dir))
            self.trace.log_event(
                "imagination_started",
                {
                    "turn": self.main_turns,
                    "trace": subtrace_rel,
                    "instruction": instruction,
                    "action_id": action_id,
                },
            )

        def on_imagination_turn(subagent_turn: int) -> None:
            if self.trace is not None and subtrace_rel is not None:
                self.trace.log_event(
                    "imagination_turn_closed",
                    {
                        "turn": self.main_turns,
                        "subagent_turn": int(subagent_turn),
                        "trace": subtrace_rel,
                    },
                )

        result = ImaginationRunner(
            self.imagination_provider,
            self.workspace,
            self.renderer,
            self.compiler,
            max_turns=self.config.max_imagination_turns,
            trace=subtrace,
            usage_callback=self._accumulate_usage,
            turn_callback=on_imagination_turn,
            active_mmskill=self.mmskill_buffer.active,
            reference_mmskill=self.mmskill_buffer.reference_skill,
        ).run(instruction, action_id)
        if self.trace is not None and subtrace_rel is not None:
            self.trace.log_event(
                "imagination_closed",
                {
                    "turn": self.main_turns,
                    "trace": subtrace_rel,
                    "status": result.status,
                    "turns": result.turns,
                    "action_id": result.action_id,
                },
            )
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
            function_name="imagine_action",
            arguments=dict(call.args),
            result=payload,
            revision_before=before,
            revision_after=self.workspace.state.observation_revision,
            manifest=self.workspace.state.manifest(),
            trace_diagnostics={"subagent": diagnostics},
        )

    def _consult_mmskill(self, call: ToolCall) -> ContextStepResult:
        before = self.workspace.state.observation_revision
        if call.parse_error:
            return self.workspace.reject(call.name, call.args, call.parse_error)
        try:
            _name, arguments = parse_action(
                {"name": call.name, "arguments": call.args},
                allowed=("consult_mmskill",),
            )
            skill = self.mmskill_library.get(str(arguments["skill_id"]))
        except (KeyError, ValueError) as exc:
            return self.workspace.reject(call.name, call.args, str(exc))
        reused = (
            self.mmskill_buffer.active is not None
            and self.mmskill_buffer.active.skill_id == skill.skill_id
        )
        if not reused:
            self.mmskill_buffer.load(skill)
        return ContextStepResult(
            function_name="consult_mmskill",
            arguments=dict(call.args),
            result={"skill_id": skill.skill_id, "loaded": True},
            revision_before=before,
            revision_after=before,
            manifest=self.workspace.state.manifest(),
            trace_diagnostics={"mmskill": skill.summary(), "reused": reused},
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
            attempt_number = attempts + 1
            if self.trace is not None:
                self.trace.log_event(
                    "model_started",
                    {"turn": turn, "attempt": attempt_number},
                )
                visible_messages, visible_tools = _provider_visible_request(
                    self.main_provider,
                    messages,
                    self.main_tools,
                )
                self.trace.log_model_request(
                    turn=turn,
                    attempt=attempt_number,
                    owner="main",
                    messages=visible_messages,
                    tools=visible_tools,
                )
            try:
                response = self.main_provider.generate(messages, self.main_tools)
            except Exception as exc:
                if self.trace is not None:
                    self.trace.log_model_response(
                        turn=turn,
                        attempt=attempt_number,
                        owner="main",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                raise
            if self.trace is not None:
                self.trace.log_model_response(
                    turn=turn,
                    attempt=attempt_number,
                    owner="main",
                    response=response,
                )
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
                self._log(turn, image, packet, None, None, response)
                return None, response
            messages = self._main_messages(
                packet,
                image,
                decision_turn=turn,
                feedback=error,
            )

    def _main_messages(
        self,
        packet: ContextPacket,
        image: np.ndarray,
        *,
        decision_turn: int | None = None,
        feedback: str | None = None,
    ) -> list[Message]:
        snapshot = self._main_context_snapshot(
            packet,
            decision_turn=decision_turn,
            feedback=feedback,
        )
        return self._messages_from_context(image, snapshot)

    def _main_context_snapshot(
        self,
        packet: ContextPacket,
        *,
        decision_turn: int | None = None,
        embodied_state: EmbodiedStateCard | None = None,
        feedback: str | None = None,
    ) -> dict[str, Any]:
        state_card = embodied_state or self._embodied_state(
            decision_turn=decision_turn
        )
        text = render_main_context(
            task=self.workspace.state.task_prompt,
            available_mmskills=self.mmskill_library.format_index(),
            live_references=packet.manifest(),
            embodied_state=state_card,
            interaction_memory=self.interaction_memory,
            active_mmskill=self.mmskill_buffer.active,
            feedback=feedback,
        )
        return {
            "schema": "vaw-agent-context-v3-mmskill",
            "task": self.workspace.state.task_prompt,
            "revision": packet.revision,
            "live_references": packet.manifest(),
            "embodied_state_card": state_card.summary(),
            "interaction_memory_before": self.interaction_memory.snapshot(),
            "interaction_memory_prompt": self.interaction_memory.prompt_lines(),
            "available_mmskills": self.mmskill_library.index(),
            "active_mmskill": self.mmskill_buffer.summary(),
            "protocol_feedback": feedback,
            "prompt_text": text,
        }

    def _messages_from_context(
        self,
        image: np.ndarray,
        snapshot: dict[str, Any],
    ) -> list[Message]:
        return _image_messages(
            self._system_prompt,
            str(snapshot["prompt_text"]),
            image,
            label="当前主智能体上下文画布",
        )

    def _embodied_state(
        self,
        *,
        decision_turn: int | None = None,
    ) -> EmbodiedStateCard:
        return project_embodied_state(
            self.workspace.state,
            self.interaction_memory,
            decision_turn=(
                self.main_turns + 1 if decision_turn is None else decision_turn
            ),
        )

    def _finish(self, mode: TerminateMode, steps: list[StepRecord], detail: str) -> EpisodeResult:
        if not self.workspace.finished:
            step = self.workspace.execute("finish_task", success=False)
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
            self.trace.log_event(
                "episode_closed",
                {
                    "terminate_mode": mode.value,
                    "turns": self.main_turns,
                    "claimed_success": self.workspace.claimed_success,
                    "env_success": self.env_success,
                },
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
        *,
        embodied_state_card: EmbodiedStateCard | None = None,
        function_effect_kind: str | None = None,
        context_snapshot_ref: str | None = None,
        interaction_memory_before: list[dict[str, Any]] | None = None,
        interaction_event: dict[str, Any] | None = None,
        action_segment: dict[str, Any] | None = None,
    ) -> None:
        if self.trace is not None:
            state_card = embodied_state_card or self._embodied_state(
                decision_turn=turn
            )
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
                embodied_state_card=state_card.summary(),
                interaction_memory_before=interaction_memory_before or [],
                interaction_event=interaction_event,
                interaction_memory_after=self.interaction_memory.snapshot(),
                function_effect_kind=function_effect_kind,
                context_snapshot_ref=context_snapshot_ref,
                action_segment=action_segment,
            )

    def _notify(self, turn: int, record: StepRecord) -> None:
        if self.config.on_turn is None:
            return
        # The observer runs after the transaction committed and the trace was
        # written; it must never be able to kill an episode (a real crash was
        # a BlockingIOError from a print into a non-blocking stdout pipe).
        with suppress(Exception):
            self.config.on_turn(turn, record)

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
        return "相对于当前 TCP 的局部位姿"
    if context.source_kind == "point":
        return (
            "粗略语义点与偏移锚点；只能依据可见目标几何微调，"
            "不能假设初始锚点已经完美居中"
        )
    if context.source_kind == "grasp":
        return "粗略抓取候选；依据可见接触几何验证并微调"
    return f"来源类型为 {context.source_kind} 的粗略候选"


def _image_messages(
    system_prompt: str,
    text: str,
    image: np.ndarray,
    *,
    label: str,
    extra_parts: list[dict[str, Any]] | None = None,
) -> list[Message]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if extra_parts:
        content.extend(extra_parts)
    content.extend(
        [
            {"type": "text", "text": label},
            {"type": "image_url", "image_url": {"url": encode_png_data_url(image)}},
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _provider_visible_request(
    provider: ModelProvider,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
) -> tuple[list[Message], list[dict[str, Any]] | None]:
    """展开文本协议，使 trace 与模型真正收到的请求保持一致。"""

    prepare = getattr(provider, "prepare_messages", None)
    if callable(prepare):
        return prepare(messages, tools), None
    return messages, tools


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
    grounding_model: str | None = None,
    point_model: str | None = None,
) -> EpisodeResult:
    from vaw.context_runtime.perception_defaults import (
        DEFAULT_GROUNDING_MODEL,
        DEFAULT_POINT_MODEL,
    )
    from vaw.context_runtime.point_grounding import resolve_point_coord_space
    from vaw.context_runtime.semantic_grounding import resolve_grounding_coord_space

    resolved_grounding_model = grounding_model or DEFAULT_GROUNDING_MODEL
    resolved_point_model = point_model or DEFAULT_POINT_MODEL
    workspace = ContextWorkspace(
        api,
        task_prompt,
        motion_backend=motion_backend,
        grounding_model=resolved_grounding_model,
        grounding_coord_space=resolve_grounding_coord_space(resolved_grounding_model),
        point_model=resolved_point_model,
        point_coord_space=resolve_point_coord_space(resolved_point_model),
    )
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
