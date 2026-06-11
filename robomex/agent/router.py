"""Skill routing: retrieve skills and assemble compact, prompt-ready guidance.

v1.1 routing is claim-driven: retrieve the best-matching action skills for the
task, then pull in the observation skills that produce the claims those action
skills require. Guidance passes the skill's markdown body through verbatim --
the body is authored (or distilled) specifically as prompt content, so what
the LLM sees is exactly what is on disk, framed by a one-line contract header.
"""

from __future__ import annotations

from robomex.library import SkillLibrary, SkillRecord
from robomex.skills import Skill, SkillKind


def build_query(task: str, observation_summary: str = "") -> str:
    """Derive a retrieval query from the task and (optional) observation summary."""

    return f"{task} {observation_summary}".strip()


def build_guidance(skill: Skill, task: str) -> str:
    """Frame one skill's markdown body with its claim contract for the prompt."""

    role = "perceive (produces claims)" if skill.kind == SkillKind.OBSERVATION else "act (requires claims)"
    lines = [
        f"## Skill [{skill.kind.value}] {skill.name} — {role}",
        f"use for: {skill.description}",
    ]
    if skill.requires:
        lines.append(f"requires claims: {', '.join(skill.requires)}")
    if skill.produces:
        lines.append(f"produces claims: {', '.join(skill.produces)}")
    lines.append("")
    lines.append(skill.guidance.strip())
    if skill.verify:
        lines.append("")
        lines.append("verify before trusting the result:")
        lines += [f"- {check}" for check in skill.verify]
    if skill.recovery:
        lines.append("on failure:")
        lines += [f"- {step}" for step in skill.recovery]
    return "\n".join(lines)


class SkillRouter:
    """Retrieves action skills and the observation skills backing their claims."""

    def __init__(self, library: SkillLibrary, top_k_actions: int = 1) -> None:
        self.library = library
        self.top_k_actions = top_k_actions

    def route(self, task: str, observation_summary: str = "") -> list[SkillRecord]:
        """Observation skills first (perceive before act), then action skills."""

        query = build_query(task, observation_summary)
        actions = self.library.retrieve(query, k=self.top_k_actions, kind=SkillKind.ACTION)

        required = tuple(claim for record in actions for claim in record.skill.requires)
        observations = {record.skill_id: record for record in self.library.producers_of(required)}
        # The task may also call for perception directly (e.g. "find the mug").
        for record in self.library.retrieve(query, k=1, kind=SkillKind.OBSERVATION):
            observations.setdefault(record.skill_id, record)

        return list(observations.values()) + actions

    def guidance_for(self, records: list[SkillRecord], task: str) -> str:
        """Concatenate guidance for the chosen skills (empty if none)."""

        return "\n\n".join(build_guidance(r.skill, task) for r in records)
