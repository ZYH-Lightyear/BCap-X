"""Skill routing: retrieve relevant skills and inject their markdown verbatim.

Skills are procedural-knowledge text, so routing is deliberately dumb: keyword-
retrieve the best-matching skills for the task and concatenate their bodies into
the prompt as-is. There is no claim dependency resolution -- sequencing and
verification are decided by the agent/planner reading the prose, not by the
router wiring typed claims together.
"""

from __future__ import annotations

from robomex.library import SkillLibrary, SkillRecord
from robomex.skills import Skill


def build_query(task: str, observation_summary: str = "") -> str:
    """Derive a retrieval query from the task and (optional) observation summary."""

    return f"{task} {observation_summary}".strip()


def build_guidance(skill: Skill, task: str) -> str:
    """Frame one skill's markdown body for the prompt (a small header + the body)."""

    header = f"## Skill: {skill.name}"
    if skill.description:
        header += f"\n_{skill.description}_"
    return f"{header}\n\n{skill.body.strip()}"


class SkillRouter:
    """Retrieves the top-k skills relevant to the task by keyword overlap."""

    def __init__(self, library: SkillLibrary, top_k: int = 3) -> None:
        self.library = library
        self.top_k = top_k

    def route(self, task: str, observation_summary: str = "") -> list[SkillRecord]:
        query = build_query(task, observation_summary)
        return self.library.retrieve(query, k=self.top_k)

    def guidance_for(self, records: list[SkillRecord], task: str) -> str:
        """Concatenate guidance for the chosen skills (empty if none)."""

        return "\n\n".join(build_guidance(r.skill, task) for r in records)
