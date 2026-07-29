"""VAWRuntime: the ReAct loop that drives one workspace episode.

Forked from ``agentx/core.py`` (itself a port of Qwen-Code's ``AgentCore``).
The skeleton survives — budget checks, one provider call per turn, results fed
back, failures returned as content rather than raised — but four things differ,
each because the workspace is a physical environment rather than a codebase.
See ``docs/vaw_implementation_plan.md`` §2 for the full rationale.

1. **One op per turn.** The original scheduled batches of parallel tool calls;
   the whole ``ToolScheduler`` is gone here. Workspace ops are strongly ordered
   and the canvas renders the state after exactly one of them, so a batch has
   no meaning. Extra calls in a reply are answered with an error, not executed.

2. **Observation is structural, not a hook.** ``RunConfig.observe`` existed so
   the loop could re-observe after tool execution. Unnecessary here:
   ``Workspace.step`` always returns a freshly rendered canvas, and physical
   ops refresh the observation before rendering. There is no code path in
   which the model acts without seeing the result.

3. **``done`` is an explicit op.** The original deliberately had no finish
   tool: text without a tool call meant completion. An episode boundary is a
   physical event here, and whether the agent believes it succeeded is a
   trainable output, so text alone ends nothing — it gets nudged. When the
   budget runs out the runtime commits ``done(success=False)`` itself, so every
   trace has a terminal step.

4. **Deterministic context.** No compression. The last K canvases stay as
   images (``ChatSession.max_images``), older turns keep their text receipts,
   and the full state summary rides along every turn. SFT samples and RL
   rollouts have to see identical context for identical state.

The hallucination guard is kept as-is: some routes intermittently drop out of
native tool-calling mode and narrate a fake conversation of calls and results.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image

from vaw.agents.chat import ChatSession
from vaw.agents.contracts import (
    EpisodeResult,
    Message,
    StepRecord,
    TerminateMode,
    ToolCall,
)
from vaw.agents.providers.base import ModelProvider
from vaw.protocol import SYSTEM_PROMPT, parse_action, tool_definitions
from vaw.types import StepResult
from vaw.workspace import Workspace

#: Sent when the model neither calls an op nor is obviously hallucinating.
#: An empty reply usually means the model is waiting for a signal it thinks
#: exists; terminating there would kill a run that was still recoverable.
NUDGE = (
    "You did not call any operation. Every turn must call exactly one workspace "
    "operation. If the task is finished or impossible, call `done`."
)

#: Matching these in a reply that carried no real tool call means the model is
#: writing a conversation script instead of acting. Regexes rather than
#: literals: the same model alternates between ``**Tool Call:``, ``**Tool:``
#: and ``<tool_call>`` in varying case, and missing one voids the guard.
HALLUCINATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"<\s*/?\s*tool_(call|use|result)\s*>",
        r"<\s*system\s*>",
        r"\*\*\s*tool[\s_]*(call)?\s*:",
        r"tool_call\s+error",
        r"^\s*(assistant|system|tool)\s*:",
    )
)

HALLUCINATION_REBUKE = (
    "Your last reply narrated operation calls and their results in prose, but "
    "**none of them ran**. Never write a receipt yourself: it is fiction, and "
    "the robot has not moved. Issue one real operation call and wait for the "
    "canvas and receipt that come back."
)

ONE_OP_PER_STEP = (
    "Not executed: exactly one operation runs per step, and this reply "
    "contained several. Only the first was executed; re-issue this one next "
    "turn if you still want it, after reading the receipt."
)


@dataclass
class RunConfig:
    """Budget and context policy for one episode."""

    max_turns: int = 40
    max_time_s: float = 1800.0
    #: How many canvases stay in history as images. Older turns keep their text
    #: receipts. Three covers "before / after / now" around the last physical
    #: op, which is what failure attribution needs.
    canvas_window_k: int = 3
    #: Called after each turn for progress printing or external logging. The
    #: episode trace itself is written by ``Workspace.TraceLogger``.
    on_turn: Callable[[int, StepRecord], None] | None = None


@dataclass
class _Turn:
    """Bookkeeping for one iteration."""

    index: int
    thought: str = ""
    records: list[StepRecord] = field(default_factory=list)


class VAWRuntime:
    """Drives one agent episode against one workspace."""

    def __init__(
        self,
        provider: ModelProvider,
        workspace: Workspace,
        config: RunConfig | None = None,
    ) -> None:
        self.provider = provider
        self.ws = workspace
        self.config = config or RunConfig()
        self.tools = tool_definitions()
        self.chat = ChatSession(
            provider,
            SYSTEM_PROMPT,
            max_images=self.config.canvas_window_k,
        )

    # ------------------------------------------------------------------ #
    def run(self) -> EpisodeResult:
        """Run until the agent calls ``done`` or a budget is exhausted."""

        self.chat.append({"role": "user", "content": f"Task: {self.ws.state.instruction}"})
        # Opening observation: the agent never has to spend a turn asking for
        # the first look at the scene.
        opening = self.ws.step("observe")
        self.chat.append(_observation_message(opening, turn=0))

        started = time.monotonic()
        steps: list[StepRecord] = []
        turn = 0

        while True:
            if self.ws.finished:
                return self._result(TerminateMode.GOAL, turn, steps)
            if turn >= self.config.max_turns:
                return self._finish_on_budget(
                    TerminateMode.MAX_TURNS, turn, steps,
                    f"turn limit {self.config.max_turns} reached",
                )
            elapsed = time.monotonic() - started
            if elapsed >= self.config.max_time_s:
                return self._finish_on_budget(
                    TerminateMode.TIMEOUT, turn, steps,
                    f"time limit {self.config.max_time_s:.0f}s reached",
                )

            turn += 1
            try:
                response = self.chat.send(self.tools)
            except KeyboardInterrupt:
                return self._finish_on_budget(
                    TerminateMode.CANCELLED, turn, steps, "interrupted by user"
                )
            except Exception as exc:  # noqa: BLE001 - all provider faults land here
                return self._finish_on_budget(
                    TerminateMode.ERROR, turn, steps, f"{type(exc).__name__}: {exc}"
                )

            if not response.tool_calls:
                self._handle_no_call(response.text)
                continue

            record = self._execute(turn, response.tool_calls, response.text)
            steps.append(record)
            if self.config.on_turn is not None:
                self.config.on_turn(turn, record)

    # ------------------------------------------------------------------ #
    def _handle_no_call(self, text: str) -> None:
        """Reply carried no op. Push back and keep going — never terminate.

        Unlike the coding agent this forks from, text is never a completion
        signal: only ``done`` ends an episode.
        """
        stripped = text.strip()
        rebuke = (
            HALLUCINATION_REBUKE
            if stripped and _looks_hallucinated(stripped)
            else NUDGE
        )
        self.chat.append({"role": "user", "content": rebuke})

    def _execute(self, turn: int, calls: tuple[ToolCall, ...], thought: str) -> StepRecord:
        """Run the first call; answer the rest with a refusal.

        Every call id must get a tool message even when we decline to run it —
        an unanswered id makes the endpoint reject the entire next request.
        """
        primary, extras = calls[0], calls[1:]
        result = self._dispatch(primary)

        messages: list[Message] = [
            {
                "role": "tool",
                "tool_call_id": primary.id,
                "content": _receipt_text(result),
            }
        ]
        for extra in extras:
            messages.append(
                {"role": "tool", "tool_call_id": extra.id, "content": ONE_OP_PER_STEP}
            )
        self.chat.extend(messages)
        self.chat.append(_observation_message(result, turn=turn))

        return StepRecord(
            turn=turn,
            op=result.op,
            args=result.args,
            ok=result.ok,
            physical=result.physical,
            receipt=result.receipt_text,
            thought=thought.strip(),
        )

    def _dispatch(self, call: ToolCall) -> StepResult:
        """Turn one model call into one workspace step.

        Routed through ``protocol.parse_action`` rather than calling
        ``Workspace.step`` with the raw name so that the native path, the text
        protocol and the RL rollout path all validate actions identically.
        Anything malformed becomes an error receipt: recovering from its own
        bad output is part of what the policy has to learn.
        """
        if call.parse_error:
            return self.ws.reject(call.name, call.args, call.parse_error)
        try:
            op_name, kwargs = parse_action({"name": call.name, "arguments": call.args})
        except ValueError as exc:
            return self.ws.reject(call.name, call.args, str(exc))
        return self.ws.step(op_name, **kwargs)

    # ------------------------------------------------------------------ #
    def _finish_on_budget(
        self,
        mode: TerminateMode,
        turn: int,
        steps: list[StepRecord],
        detail: str,
    ) -> EpisodeResult:
        """Close out an episode the agent did not end itself.

        The forced ``done`` keeps every trace terminated the same way, so data
        builders never have to special-case a truncated episode — and the
        environment gets told to stop rather than being left mid-motion.
        """
        if not self.ws.finished:
            result = self.ws.step("done", success=False)
            steps.append(
                StepRecord(
                    turn=turn,
                    op="done",
                    args={"success": False},
                    ok=result.ok,
                    physical=True,
                    receipt=f"{result.receipt_text} (forced: {detail})",
                )
            )
        return self._result(mode, turn, steps, detail)

    def _result(
        self,
        mode: TerminateMode,
        turns: int,
        steps: list[StepRecord],
        detail: str = "",
    ) -> EpisodeResult:
        return EpisodeResult(
            terminate_mode=mode,
            turns=turns,
            claimed_success=self.ws.claimed_success,
            env_success=self.ws.env_success,
            detail=detail,
            steps=list(steps),
            usage=dict(self.chat.usage),
            trace_dir=str(self.ws.trace.dir) if self.ws.trace else None,
        )


def run_episode(
    provider: ModelProvider,
    api: Any,
    instruction: str,
    *,
    trace_dir: str | None = None,
    config: RunConfig | None = None,
    max_physical_ops: int = 30,
    env_check: Callable[[], bool] | None = None,
) -> EpisodeResult:
    """Build a workspace over ``api`` and run one episode on it.

    The single entry point for both roles: teacher and student differ only in
    the provider handed in, which is what keeps their traces interchangeable.

    :param env_check: Privileged episode-end verdict (e.g. ``env.task_completed``);
        recorded in trace meta only, never shown to the agent.
    """
    workspace = Workspace(
        api,
        instruction,
        trace_dir=trace_dir,
        max_physical_ops=max_physical_ops,
        env_check=env_check,
    )
    return VAWRuntime(provider, workspace, config).run()


# ---------------------------------------------------------------------- #
def _receipt_text(result: StepResult) -> str:
    """Tool-message body: the receipt plus the full state summary.

    The summary is resent every turn instead of being diffed. It is small
    (ids and numbers, never masks or point clouds), and a diff would make the
    context depend on history rather than on state, which breaks the
    determinism SFT and RL rely on.
    """
    return (
        f"{result.receipt_text}\n\n"
        f"workspace state:\n```json\n"
        f"{json.dumps(result.state_summary, ensure_ascii=False, indent=None)}\n```"
    )


def _observation_message(result: StepResult, *, turn: int) -> Message:
    """The canvas, as a user message.

    Images cannot ride on tool messages on most endpoints, so the canvas comes
    as a separate user message right after the receipt. The header says what
    produced it, so the model does not read it as a new instruction from the
    user.
    """
    header = f"[canvas after turn {turn}: {result.op}]"
    parts: list[dict[str, Any]] = [{"type": "text", "text": header}]
    if result.canvas is not None:
        parts.append(_image_part(result.canvas))
    return {"role": "user", "content": parts}


def _image_part(canvas: np.ndarray) -> dict[str, Any]:
    """Encode an (H, W, 3) uint8 canvas as a PNG data URL.

    PNG, not JPEG: the canvas is line art and text where compression artefacts
    land exactly on the ids the model has to read back.
    """
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(canvas, dtype=np.uint8)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def _looks_hallucinated(text: str) -> bool:
    """Whether a reply with no real call is a fabricated transcript.

    Only used on the no-call path: these markers are perfectly normal in a
    reply that did call an op (the model explaining what it just did).

    A false positive costs one round trip; a false negative feeds the model
    its own fiction as if it were sensor data. Biased accordingly.
    """
    return any(pattern.search(text) for pattern in HALLUCINATION_PATTERNS)


__all__ = [
    "HALLUCINATION_PATTERNS",
    "HALLUCINATION_REBUKE",
    "NUDGE",
    "ONE_OP_PER_STEP",
    "RunConfig",
    "VAWRuntime",
    "run_episode",
]
