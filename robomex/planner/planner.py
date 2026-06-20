"""Outer reactive planner: task + scene -> an ordered To-Do list of sub-goals.

The planner is deliberately thin (grounding beats planning). It reads the task,
the initial scene image, and the capability menu of high-level (compound) skills,
then makes ONE LLM call that returns a JSON To-Do list. Each item is a sub-goal a
single high-level skill can achieve, with a visually-checkable postcondition.

It does not rank/score skills and (for now) does not re-plan: the To-Do list is
produced once and handed to the inner Code Agent sub-goal by sub-goal. Re-planning
on inner failure is a later addition.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from robomex.agent.agent import CodeAsPolicyAgent
from robomex.agent.trace import AgentTrace
from robomex.library import SkillLibrary

_SYSTEM_PROMPT = (
    "You are a robot task planner. Given a task and the current scene, break the task "
    "into an ordered To-Do list of sub-goals, where each sub-goal can be achieved by ONE "
    "high-level skill from the menu. "
    "Reply with ONLY a JSON array; each element is an object with keys: "
    '"goal" (the imperative sub-goal), '
    '"skill" (the high-level skill name from the menu that fits, or null), '
    '"postcondition" (a single visually-checkable condition that means the sub-goal succeeded). '
    "Output the JSON array and nothing else."
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


@dataclass(frozen=True)
class SubGoal:
    """One item of the To-Do list: a sub-goal plus how to know it succeeded."""

    goal: str
    postcondition: str = ""
    skill: str | None = None


@dataclass(frozen=True)
class SubGoalResult:
    """Outcome of executing one sub-goal through the inner Code Agent."""

    subgoal: SubGoal
    trace: AgentTrace
    success: bool


@dataclass(frozen=True)
class PlanExecution:
    """Full two-level episode: the plan and each sub-goal's result."""

    task: str
    subgoals: tuple[SubGoal, ...]
    results: tuple[SubGoalResult, ...] = ()
    success: bool = False


class PlannerPolicy(Protocol):
    """Turns a chat-style prompt into the planner's raw (JSON) reply."""

    def propose(self, prompt: list[dict]) -> str: ...


class LLMPlannerPolicy:
    """Real planner policy backed by ``capx.llm.client.query_model`` (multimodal)."""

    def __init__(
        self,
        model: str = "openrouter/qwen/qwen3.6-plus",
        server_url: str = "http://localhost:8110/chat/completions",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        from capx.llm.client import ModelQueryArgs, query_model

        self._query_model = query_model
        self._args = ModelQueryArgs(
            model=model,
            server_url=server_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def propose(self, prompt: list[dict]) -> str:
        return self._query_model(self._args, prompt)["content"]


class ScriptedPlannerPolicy:
    """Returns a fixed reply (JSON string); for offline runs and tests."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def propose(self, prompt: list[dict]) -> str:
        return self._reply


def parse_subgoals(text: str) -> list[SubGoal]:
    """Extract the JSON To-Do list from the planner reply into SubGoals."""

    match = _JSON_ARRAY_RE.search(text or "")
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    subgoals: list[SubGoal] = []
    for item in items:
        if isinstance(item, str):
            subgoals.append(SubGoal(goal=item))
        elif isinstance(item, dict):
            goal = str(item.get("goal") or item.get("goal_text") or "").strip()
            if not goal:
                continue
            skill = item.get("skill") or item.get("skill_hint")
            subgoals.append(SubGoal(
                goal=goal,
                postcondition=str(item.get("postcondition", "") or ""),
                skill=str(skill) if skill else None,
            ))
    return subgoals


def _image_part(path: str) -> dict:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


class ReactivePlanner:
    """Plans a task into a To-Do list of sub-goals over the high-level skill menu."""

    def __init__(
        self,
        library: SkillLibrary,
        policy: PlannerPolicy,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        self.library = library
        self.policy = policy
        self.system_prompt = system_prompt

    def menu(self) -> str:
        """The capability menu: each high-level (compound) skill's name + purpose."""

        lines = []
        for record in self.library.compound_skills():
            skill = record.skill
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines) or "(no high-level skills available)"

    def plan(self, task: str, scene_image_path: str | None = None) -> list[SubGoal]:
        parts: list[dict] = [{
            "type": "text",
            "text": (
                f"Task: {task}\n\n"
                f"High-level skills available (the menu):\n{self.menu()}\n\n"
                "Produce the To-Do list now."
            ),
        }]
        if scene_image_path:
            parts.append(_image_part(scene_image_path))
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": parts},
        ]
        return parse_subgoals(self.policy.propose(prompt))


class TwoLevelAgent:
    """Outer planner + inner Code Agent: plan once, run each sub-goal in order.

    Minimal version: no re-planning. Each sub-goal is executed by the inner
    ``CodeAsPolicyAgent``; its trace success is the sub-goal outcome. The
    postcondition is carried on the SubGoal for later explicit gating.
    """

    def __init__(self, planner: ReactivePlanner, inner: CodeAsPolicyAgent) -> None:
        self.planner = planner
        self.inner = inner

    def run(
        self,
        task: str,
        scene_image_path: str | None = None,
        observation_summary: str = "",
    ) -> PlanExecution:
        subgoals = self.planner.plan(task, scene_image_path)
        results: list[SubGoalResult] = []
        for sg in subgoals:
            trace = self.inner.run(sg.goal, observation_summary)
            results.append(SubGoalResult(subgoal=sg, trace=trace, success=trace.success))
        success = bool(results) and all(r.success for r in results)
        return PlanExecution(
            task=task,
            subgoals=tuple(subgoals),
            results=tuple(results),
            success=success,
        )
