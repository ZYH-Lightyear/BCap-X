"""Distill execution traces into reusable multimodal executable skills.

RoboMEx's distillation substrate is the *code trace* (generated code + stdout/
stderr + verifier verdicts), not raw video or free text. A successful trace is
condensed into an action skill whose guidance carries the working code as a
reusable sketch; a failed trace contributes a recovery hint to the consulted
skills. Distilled skills use the same markdown representation as seeds.

An incremental-value gate (Skill1's ``r(tau) - U_hat`` idea) only admits a new
skill when the run succeeded and beats the best consulted skill's success rate.
"""

from __future__ import annotations

import re

from robomex.agent.trace import AgentTrace
from robomex.library import SkillLibrary
from robomex.skills import Skill, SkillKind


def _slug(task: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")[:48] or "skill"


class SkillDistiller:
    def __init__(self, library: SkillLibrary) -> None:
        self.library = library

    def evolve(self, trace: AgentTrace) -> Skill | None:
        """Own all lifecycle updates from a trace; return the admitted skill, if any.

        The incremental-value gate uses utilities *before* this run is folded in,
        so a successful trace is only distilled when prior coverage was unreliable.
        """

        prior_best = self._prior_best_success_rate(trace.loaded_skill_ids)
        failure_note = trace.last_error.strip().splitlines()[-1] if trace.last_error else ""
        for skill_id in trace.loaded_skill_ids:
            self.library.update_utility(skill_id, trace.success, failure_note=failure_note)

        if not trace.success:
            self._patch_failures(trace, failure_note)
            return None
        if prior_best >= 1.0:
            return None

        skill = self._build_skill(trace)
        self.library.admit(skill, source="self_trial")
        return skill

    def _prior_best_success_rate(self, skill_ids: tuple[str, ...]) -> float:
        return max((self.library.get(sid).utility.success_rate for sid in skill_ids), default=0.0)

    def _build_skill(self, trace: AgentTrace) -> Skill:
        slug = _slug(trace.task)
        guidance_parts = [
            "Verified code trace from a successful run; adapt each step to the current observation.",
        ]
        for i, code in enumerate(trace.successful_code):
            guidance_parts.append(f"Step {i}:\n\n```python\n{code}\n```")
        return Skill(
            kind=SkillKind.ACTION,
            skill_id=f"learned_{slug}",
            name=trace.task[:60],
            description=f"Distilled from a successful code trace for: {trace.task}",
            keywords=tuple(t for t in slug.split("_") if len(t) > 2),
            verify=("Re-run the verifier checks used during the successful trace.",),
            guidance="\n\n".join(guidance_parts),
        )

    def _patch_failures(self, trace: AgentTrace, error: str) -> None:
        """Append a recovery hint distilled from the failure to consulted skills."""

        if not error:
            return
        for skill_id in trace.loaded_skill_ids:
            record = self.library.get(skill_id)
            patched = record.skill.with_recovery(
                f"observed error: {error} -> re-perceive and re-plan before retrying"
            )
            self.library.admit(patched, source=record.utility.source)
