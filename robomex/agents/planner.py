"""外层反应式 planner:任务 + 当前场景 -> 逐步给出**下一个** sub-goal。

planner 刻意做得很薄(grounding 比 planning 更重要)。每一步它读取任务、*当前*场景图、
task 技能的规划指导、以及已完成的 sub-goals,然后做一次 LLM 调用,返回单个
Markdown 两字段 sub-goal——或在任务完成时返回 ``DONE``。

场景图只用于判断进度(是否 DONE、下一步该 pick 还是 place),**不**用于改写/
扩写对象外观。对象消歧与视觉 grounding 交给内层 Agent Swarm。Goal 应尽量沿用
任务用语,只做步骤粒度分解,不添加颜色、纹理或自创空间描述。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from robomex.perception.render import image_content_part

from robomex.core.coder.trace import AgentTrace
from robomex.core.logging import get_logger
from robomex.skills import SkillLibrary

_log = get_logger("planner")

_SYSTEM_PROMPT = (
    "You are a thin reactive robot task planner. Your job is to understand the task and "
    "emit the SINGLE next high-level sub-goal step — not to rewrite, paraphrase, or "
    "visually expand the task. "
    "Given a task, the current scene image, task skill guidance, and a concise execution "
    "history, first assess whether the task is already complete, then choose the next "
    "step at skill-pattern granularity (e.g. pick then place). Task skills are planning "
    "patterns only; do NOT merely name a skill or force the executor into one skill. "
    "Use the scene image ONLY as a progress check: decide DONE vs which next step is "
    "still needed. Do NOT invent or append appearance/spatial glosses that are absent "
    "from the task text — no color, texture, label, brand markup, or self-authored "
    "phrases like 'red/green labeled', 'blue can', 'front-center', 'behind the bottle'. "
    "Keep object names as they appear in the task (or a minimal imperative split of that "
    "wording). Visual disambiguation belongs to the downstream Agent Swarm, not you. "
    "Do not repeat a pick/place just because a prior checkpoint was uncertain; uncertainty "
    "is not observed failure. "
    "If the task is not complete, reply with exactly two Markdown-style fields and no "
    "extra prose:\n"
    "Goal: <short imperative next sub-goal using task wording; >\n"
    "Postcondition: <one short success condition in the same vocabulary; >\n"
    "If the task is already complete, reply with exactly the word DONE and nothing else."
)

_MD_FIELD_RE = re.compile(
    r"^\s*Goal\s*:\s*(?P<goal>.+?)(?:\n+\s*Postcondition\s*:\s*(?P<postcondition>.+))?\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class SubGoal:
    """一个反应式步骤:一个自然语言 sub-goal,外加“如何判断它成功了”。
    """

    goal: str
    postcondition: str = ""


@dataclass(frozen=True)
class SubGoalResult:
    """One subgoal result projected from the runtime-owned outcome."""

    subgoal: SubGoal
    trace: AgentTrace
    success: bool
    motion_attempted: bool = False
    authoring_status: str = "uncertain"
    verification_status: str = "not_run"
    note: str = ""


@dataclass(frozen=True)
class PlanExecution:
    """A reactive episode; planner completion is not environment success."""

    task: str
    subgoals: tuple[SubGoal, ...]
    results: tuple[SubGoalResult, ...] = ()
    planner_status: str = "exhausted"

    @property
    def success(self) -> bool:
        """Deprecated compatibility alias; use ``planner_status``."""

        return self.planner_status == "done"


class PlannerPolicy(Protocol):
    """把 chat 形式的 prompt 变成 planner 的原始回复(Markdown fields or DONE)。"""

    def propose(self, prompt: list[dict]) -> str: ...


class LLMPlannerPolicy:
    """基于 ``capx.llm.client.query_model`` 的真实 planner 策略(多模态)。"""

    def __init__(
        self,
        model: str = "openrouter/qwen/qwen3.6-plus",
        server_url: str = "http://localhost:8110/chat/completions",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 20480,  # 对齐 capx baseline(2048*10);含 reasoning 预算
        empty_retries: int = 1,
    ) -> None:
        from capx.llm.client import ModelQueryArgs, query_model

        self._query_model = query_model
        self._empty_retries = max(0, empty_retries)
        self._args = ModelQueryArgs(
            model=model,
            server_url=server_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def propose(self, prompt: list[dict]) -> str:
        active_prompt = prompt
        for attempt in range(self._empty_retries + 1):
            out = self._query_model(self._args, active_prompt)
            content = out.get("content") or ""
            if content.strip():
                return content
            if attempt >= self._empty_retries:
                return content
            active_prompt = [
                *prompt,
                {
                    "role": "user",
                    "content": (
                        "Your previous response contained no visible content. "
                        "Reply now with either DONE or two fields: "
                        "Goal: <next sub-goal> and Postcondition: <visual success condition>. "
                        "Do not leave the message empty."
                    ),
                },
            ]
            _log.warning("empty model content; retrying once with an explicit planner nudge")
        return ""


class ScriptedPlannerPolicy:
    """回放一组固定回复(每步一个),用尽后返回 ``DONE``。

    用于反应式循环的离线运行和测试:每个预期步骤传入一个 Markdown/DONE 回复,
    策略按序返回;序列用尽后返回 ``DONE`` 让循环终止。
    """

    def __init__(self, replies: str | list[str]) -> None:
        self._replies = [replies] if isinstance(replies, str) else list(replies)
        self._i = 0

    def propose(self, prompt: list[dict]) -> str:
        if self._i >= len(self._replies):
            return "DONE"
        reply = self._replies[self._i]
        self._i += 1
        return reply


def parse_next_subgoal(text: str) -> SubGoal | None:
    """Parse one strict planner response; invalid/empty replies return ``None``.

    Planner 的首选格式是轻量 Markdown 两字段::

        Goal: ...
        Postcondition: ...

    Strict ``DONE`` is the only completion signal. Arbitrary prose is not executed as
    robot intent; the session records it as an invalid planner response.
    """

    stripped = (text or "").strip()
    if not stripped:
        return None
    if stripped.upper() == "DONE":
        return None

    md = _parse_markdown_subgoal(stripped)
    if md is not None:
        return md

    return None


def _parse_markdown_subgoal(text: str) -> SubGoal | None:
    if not re.match(r"^\s*Goal\s*:", text, re.IGNORECASE):
        starts = list(re.finditer(r"(?im)^\s*Goal\s*:", text))
        if starts:
            text = text[starts[-1].start():].strip()
    match = _MD_FIELD_RE.match(text)
    if not match:
        return None
    goal = _clean_markdown_field(match.group("goal"))
    postcondition = _clean_markdown_field(match.group("postcondition") or "")
    if not goal:
        return None
    return SubGoal(goal=goal, postcondition=postcondition)


def _clean_markdown_field(value: str) -> str:
    lines = [line.strip() for line in str(value or "").strip().splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if re.match(r"^(Goal|Postcondition)\s*:", line, re.IGNORECASE):
            break
        cleaned.append(line)
    return " ".join(line for line in cleaned if line).strip()


def _image_part(path: str) -> dict:
    return image_content_part(path)


def _render_history(history: list[SubGoalResult]) -> str:
    if not history:
        return "(none yet)"
    lines = []
    for i, r in enumerate(history, start=1):
        meta = r.trace.metadata or {}
        if r.success:
            execution_status = "succeeded"
        elif r.authoring_status == "uncertain":
            # An undecided checkpoint is not a failure (M1.5 Fix D); surface it
            # as such so the planner reasons from the evidence, not from a
            # generic "not_succeeded".
            execution_status = "checkpoint_uncertain (actions executed; outcome unverified, not a failure)"
        else:
            execution_status = "not_succeeded"
        terminal = meta.get("terminal_result") if isinstance(meta, dict) else None
        claim = ""
        if isinstance(terminal, dict):
            claim = str(terminal.get("claim", "")).strip().replace("\n", " ")
            if len(claim) > 180:
                claim = claim[:177].rstrip() + "..."
        skills = ", ".join(r.trace.loaded_skill_ids) or "none"
        lines.append(
            f"{i}. {r.subgoal.goal}: execution {execution_status}; "
            f"motion_attempted={r.motion_attempted}; skills: {skills}"
            + (f"; claim: {claim}" if claim else "")
            + (f"; verification: {r.verification_status}" if r.verification_status else "")
            + (f"; note: {r.note[:180]}" if r.note else "")
            + "."
        )
    return "\n".join(lines)


class ReactivePlanner:
    """每一步参考高层技能指导,反应式地给出下一个自然语言 sub-goal。"""

    def __init__(
        self,
        library: SkillLibrary,
        policy: PlannerPolicy,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        self.library = library
        self.policy = policy
        self.system_prompt = system_prompt
        self.last_raw: str = ""  # 最近一次 planner 原始回复(供入口落盘/排查)

    def menu(self) -> str:
        """高层技能规划指导:每个 task 技能的 id + 用途。

        Planner 只用它们校准 sub-goal 粒度与常见任务模式;它输出的是自然语言
        sub-goal,不是必须绑定给 executor 的技能函数名。
        """

        lines = []
        for record in self.library.task_skills():
            lines.append(f"- {record.skill_id}: {record.skill.description}")
        return "\n".join(lines) or "(no task skill guidance available)"

    def next_subgoal(
        self,
        task: str,
        history: list[SubGoalResult] | None = None,
        scene_image_path: str | None = None,
    ) -> SubGoal | None:
        """依据 任务 + 高层技能指导 + 历史 + 当前场景,决定下一个 sub-goal。"""

        history = history or []
        parts: list[dict] = [{
            "type": "text",
            "text": (
                f"Task: {task}\n\n"
                f"High-level skill guidance (planning patterns, not forced executor choices):\n"
                f"{self.menu()}\n\n"
                f"Execution history (most recent last; checkpoint uncertainty is not failure):\n"
                f"{_render_history(history)}\n\n"
                "The current scene image is attached below when available. Use it only to "
                "decide whether the task is already complete and which next high-level step "
                "remains. Do not rewrite or expand the task with colors, textures, labels, or "
                "spatial glosses from the image; keep Goal/Postcondition in the task's own "
                "vocabulary so the Agent Swarm can do visual grounding. "
                "Do not repeat a pick/place solely because a previous checkpoint was uncertain. "
                "Output DONE if complete; otherwise output exactly two fields: "
                "Goal: <short next sub-goal> and Postcondition: <short success condition>."
            ),
        }]
        if scene_image_path:
            parts.append(_image_part(scene_image_path))
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": parts},
        ]
        self.last_raw = self.policy.propose(prompt) or ""
        _log.debug("planner 原始回复: %s", self.last_raw.strip()[:600])
        return parse_next_subgoal(self.last_raw)
