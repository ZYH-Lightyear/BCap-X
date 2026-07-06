"""Role-conditioned views over a RoboMEx SkillLibrary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robomex.dysc.contracts import SkillContract
from robomex.dysc.society import SkillAccessPolicy
from robomex.skills import SkillCategory, SkillRecord


@dataclass(frozen=True)
class SkillLibraryView:
    """A filtered SkillLibrary for one DySC role.

    The underlying SkillLibrary remains the source of truth.  This view only
    controls what a role can discover or load, based on a declarative policy and
    optional skill contracts.
    """

    library: Any
    access_policy: SkillAccessPolicy
    contracts: dict[str, SkillContract]

    def get(self, skill_id: str) -> SkillRecord:
        if not self.allows(skill_id):
            raise KeyError(f"skill {skill_id!r} is not visible under policy {self.access_policy.name!r}")
        return self.library.get(skill_id)

    def all(self, category: SkillCategory | None = None) -> list[SkillRecord]:
        return [
            record
            for record in self.library.all(category)
            if self.allows(record.skill_id)
        ]

    def task_skills(self) -> list[SkillRecord]:
        return [
            record
            for record in self.library.task_skills()
            if self.allows(record.skill_id)
        ]

    def allows(self, skill_id: str) -> bool:
        policy = self.access_policy
        if skill_id in policy.forbid:
            return False
        contract = self.contracts.get(skill_id)
        tags = set(contract.tags if contract is not None else ())
        if tags.intersection(policy.forbid_tags):
            return False
        if skill_id in policy.allow or skill_id in policy.prefer:
            return True
        if policy.allow_tags and tags.intersection(policy.allow_tags):
            return True
        if not policy.allow and not policy.allow_tags and not policy.prefer:
            return True
        return False
