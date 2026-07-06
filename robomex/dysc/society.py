"""Society specification for evolving multi-agent skill composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SkillAccessPolicy:
    """Declarative policy deciding which skills a role can see or prefer."""

    name: str = ""
    allow: tuple[str, ...] = ()
    allow_tags: tuple[str, ...] = ()
    prefer: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    forbid_tags: tuple[str, ...] = ()
    conditional: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any] | None) -> "SkillAccessPolicy":
        data = dict(data or {})
        return cls(
            name=name,
            allow=tuple(str(item) for item in (data.get("allow") or ())),
            allow_tags=tuple(str(item) for item in (data.get("allow_tags") or ())),
            prefer=tuple(str(item) for item in (data.get("prefer") or ())),
            forbid=tuple(str(item) for item in (data.get("forbid") or ())),
            forbid_tags=tuple(str(item) for item in (data.get("forbid_tags") or ())),
            conditional=tuple(dict(item) for item in (data.get("conditional") or ())),
            raw=data,
        )


@dataclass(frozen=True)
class RoleSpec:
    """A role-conditioned controller specification.

    This is runtime-neutral: the same role can later be backed by a SubAgent,
    the main Act agent, or an offline curator.
    """

    name: str
    objective: str = ""
    skill_access_policy: str = ""
    consumes: tuple[str, ...] = ()
    emits: tuple[str, ...] = ()
    execution_boundary: str = "read_only"
    model_policy: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any] | None) -> "RoleSpec":
        data = dict(data or {})
        return cls(
            name=name,
            objective=str(data.get("objective") or ""),
            skill_access_policy=str(data.get("skill_access_policy") or ""),
            consumes=tuple(str(item) for item in (data.get("consumes") or ())),
            emits=tuple(str(item) for item in (data.get("emits") or ())),
            execution_boundary=str(data.get("execution_boundary") or "read_only"),
            model_policy=str(data.get("model_policy") or ""),
            budget=dict(data.get("budget") or {}),
            raw=data,
        )


@dataclass(frozen=True)
class SocietySpec:
    """A DySC society genotype."""

    name: str
    roles: dict[str, RoleSpec] = field(default_factory=dict)
    skill_access_policies: dict[str, SkillAccessPolicy] = field(default_factory=dict)
    topology: tuple[dict[str, Any], ...] = ()
    budget_policy: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def policy_for_role(self, role_name: str) -> SkillAccessPolicy:
        role = self.roles[role_name]
        policy_name = role.skill_access_policy
        if policy_name not in self.skill_access_policies:
            return SkillAccessPolicy(name=policy_name)
        return self.skill_access_policies[policy_name]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SocietySpec":
        roles = {
            str(name): RoleSpec.from_mapping(str(name), value)
            for name, value in dict(data.get("roles") or {}).items()
        }
        policies = {
            str(name): SkillAccessPolicy.from_mapping(str(name), value)
            for name, value in dict(data.get("skill_access_policies") or {}).items()
        }
        return cls(
            name=str(data.get("name") or "society"),
            roles=roles,
            skill_access_policies=policies,
            topology=tuple(dict(item) for item in (data.get("topology") or ())),
            budget_policy=dict(data.get("budget_policy") or {}),
            raw=dict(data),
        )


def load_society_spec(path: str | Path) -> SocietySpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SocietySpec.from_mapping(data)
