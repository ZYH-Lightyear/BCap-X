"""外层反应式 planner:任务 + 当前场景 -> 逐步给出**下一个** sub-goal。

planner 刻意做得很薄(grounding 比 planning 更重要)。每一步它读取任务、*当前*场景图、
高层(复合)技能的规划指导、以及已完成的 sub-goals,然后做一次 LLM 调用,返回单个
自然语言 sub-goal(一个 JSON 对象)——或在任务完成时返回 ``DONE``。内层 Code Agent
再根据这个 sub-goal 自主选择并组合技能,场景随之刷新,再次询问 planner。

这个反应式循环取代了旧的“一次性把整张 To-Do 表规划出来”设计:每步都在实时场景上
重新 grounding,正是恢复/终止能 work 的关键。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from robomex.perception.render import image_content_part

from robomex.agents.executor import CodeAsPolicyAgent
from robomex.core.coder.trace import AgentTrace
from robomex.core.logging import get_logger
from robomex.skills import SkillLibrary

_log = get_logger("planner")

_SYSTEM_PROMPT = (
    "You are a reactive robot task planner. Given a task, the current scene image, the "
    "high-level skill guidance, and a concise execution history, first assess the CURRENT "
    "scene state, then decide the SINGLE next natural-language sub-goal to do now. The "
    "high-level skills are planning "
    "patterns that help you choose the right granularity; do NOT merely name a skill or "
    "force the executor into one skill. "
    "Always inspect the current scene image before planning another manipulation. If the "
    "task goal is already visually satisfied, reply with exactly DONE. Do not repeat a "
    "pick/place just because a prior checkpoint was uncertain; uncertainty is not observed "
    "failure. "
    'Reply with ONLY one JSON object with keys: "goal" (the concrete imperative next '
    'sub-goal) and "postcondition" (a single visually-checkable condition that means it '
    "succeeded). "
    "If the task is already complete, reply with exactly the word DONE and nothing else."
)

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class SubGoal:
    """一个反应式步骤:一个自然语言 sub-goal,外加“如何判断它成功了”。

    ``skill`` 只为兼容旧 planner 回复/产物保留;新 planner prompt 不再要求它,
    执行器也不会被该字段强制绑定到某个高层技能。
    """

    goal: str
    postcondition: str = ""
    skill: str | None = None


@dataclass(frozen=True)
class SubGoalResult:
    """通过内层 Code Agent 执行一个 sub-goal 的结果。

    ``note`` 为子目标级验证器给出的裁决理由(若有),会喂回 planner 历史,让反应式
    planner 据此重试 / 改写 / 推进。
    """

    subgoal: SubGoal
    trace: AgentTrace
    success: bool
    note: str = ""


@dataclass(frozen=True)
class PlanExecution:
    """完整的两层 episode:走过的每个 sub-goal 及其结果。"""

    task: str
    subgoals: tuple[SubGoal, ...]
    results: tuple[SubGoalResult, ...] = ()
    success: bool = False


class PlannerPolicy(Protocol):
    """把 chat 形式的 prompt 变成 planner 的原始回复(一个 JSON 对象或 DONE)。"""

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
                        "Reply now with either DONE or one JSON object containing the next sub-goal. "
                        "Do not leave the message empty."
                    ),
                },
            ]
            print("[LLMPlannerPolicy] empty model content; retrying once with an explicit planner nudge")
        return ""


class ScriptedPlannerPolicy:
    """回放一组固定回复(每步一个),用尽后返回 ``DONE``。

    用于反应式循环的离线运行和测试:每个预期步骤传入一个回复(JSON 对象字符串),
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
    """把一条 planner 回复解析成下一个 SubGoal;DONE/空 则返回 ``None``。

    含 ``goal`` 的 JSON 对象优先;否则 ``DONE``(或无法解析的回复)表示没有下一个
    sub-goal。
    """

    match = _JSON_OBJ_RE.search(text or "")
    if match:
        try:
            item = json.loads(match.group(0))
        except json.JSONDecodeError:
            item = None
        if isinstance(item, dict):
            goal = str(item.get("goal") or item.get("goal_text") or "").strip()
            if goal:
                skill = item.get("skill") or item.get("skill_hint")
                return SubGoal(
                    goal=goal,
                    postcondition=str(item.get("postcondition", "") or ""),
                    skill=str(skill) if skill else None,
                )
    return None


def _image_part(path: str) -> dict:
    return image_content_part(path)


def _render_history(history: list[SubGoalResult]) -> str:
    if not history:
        return "(none yet)"
    lines = []
    for i, r in enumerate(history, start=1):
        meta = r.trace.metadata or {}
        act_status = meta.get("act_status") or ("finished" if r.success else "stopped")
        unresolved = meta.get("unresolved") if isinstance(meta, dict) else None
        note = ""
        if isinstance(unresolved, dict):
            note = str(unresolved.get("last_state_summary", "")).strip().replace("\n", " ")
            if len(note) > 180:
                note = note[:177].rstrip() + "..."
        skills = ", ".join(r.trace.loaded_skill_ids) or "none"
        lines.append(
            f"{i}. {r.subgoal.goal}: Act {act_status}; skills: {skills}"
            + (f"; note: {note}." if note else ".")
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
        """高层技能规划指导:每个复合技能的 id + 用途。

        Planner 只用它们校准 sub-goal 粒度与常见任务模式;它输出的是自然语言
        sub-goal,不是必须绑定给 executor 的技能函数名。
        """

        lines = []
        for record in self.library.compound_skills():
            lines.append(f"- {record.skill_id}: {record.skill.description}")
        return "\n".join(lines) or "(no high-level skill guidance available)"

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
                "The current scene image is attached below when available. First decide whether "
                "the task is already visually complete from the current scene and history. "
                "Output DONE if complete; otherwise output the single next sub-goal now."
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


class TwoLevelAgent:
    """外层反应式 planner + 内层 Code Agent:逐步执行直到 planner 说 DONE。

    每一步,planner 给出下一个自然语言 sub-goal(基于到目前为止的历史),内层
    ``CodeAsPolicyAgent`` 自主选择并组合技能执行它,其 trace 是否成功即为
    该 sub-goal 的结果。``max_subgoals`` 给循环封顶,这样一个从不说 DONE 的 planner
    也不会永远跑下去。场景图刷新交给持有 env 的真机入口;离线时场景固定不变。
    """

    def __init__(
        self,
        planner: ReactivePlanner,
        inner: CodeAsPolicyAgent,
        max_subgoals: int = 8,
    ) -> None:
        self.planner = planner
        self.inner = inner
        self.max_subgoals = max_subgoals

    def run(
        self,
        task: str,
        scene_image_path: str | None = None,
        observation_summary: str = "",
    ) -> PlanExecution:
        results: list[SubGoalResult] = []
        for _ in range(self.max_subgoals):
            sg = self.planner.next_subgoal(task, results, scene_image_path)
            if sg is None:
                break
            trace = self.inner.run(
                sg.goal,
                observation_summary,
                scene_image_path=scene_image_path,
            )
            results.append(SubGoalResult(subgoal=sg, trace=trace, success=trace.success))
        success = bool(results) and all(r.success for r in results)
        return PlanExecution(
            task=task,
            subgoals=tuple(r.subgoal for r in results),
            results=tuple(results),
            success=success,
        )
