"""LLM-proposed SocietySpec patches.

The patch layer is deliberately small: the LLM curator decides *what* to change,
while this module only validates and applies a stable set of operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from robomex.dysc.society import RoleSpec, SkillAccessPolicy, SocietySpec


ALLOWED_OPS = frozenset({
    "add_skill_preference",
    "remove_skill_preference",
    "add_forbid_skill",
    "remove_forbid_skill",
    "add_topology_edge",
    "remove_topology_edge",
    "add_motif",
    "revise_role_objective",
    "adjust_budget",
    "rollback",
})


@dataclass(frozen=True)
class SocietyPatch:
    """One LLM-proposed society mutation bundle."""

    rationale: str = ""
    mutations: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "SocietyPatch":
        data = dict(data or {})
        return cls(
            rationale=str(data.get("rationale") or ""),
            mutations=tuple(dict(item) for item in (data.get("mutations") or ())),
            raw=data,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "rationale": self.rationale,
            "mutations": [dict(item) for item in self.mutations],
        }


class SocietyPatchError(ValueError):
    """Raised when a curator patch is invalid or unsafe to apply."""


def apply_society_patch(spec: SocietySpec, patch: SocietyPatch) -> SocietySpec:
    current = spec
    for mutation in patch.mutations:
        op = str(mutation.get("op") or "")
        if op not in ALLOWED_OPS:
            raise SocietyPatchError(f"unknown society patch op: {op}")
        if op == "rollback":
            # Rollback is handled by SocietyVersionStore because it needs older versions.
            continue
        current = _apply_one(current, op, mutation)
    return current


def _apply_one(spec: SocietySpec, op: str, mutation: dict[str, Any]) -> SocietySpec:
    if op in {"add_skill_preference", "remove_skill_preference", "add_forbid_skill", "remove_forbid_skill"}:
        policy_name = _required(mutation, "policy")
        skill = _required(mutation, "skill")
        policy = _policy(spec, policy_name)
        if op == "add_skill_preference":
            return _replace_policy(spec, policy_name, replace(policy, prefer=_append(policy.prefer, skill)))
        if op == "remove_skill_preference":
            return _replace_policy(spec, policy_name, replace(policy, prefer=_remove(policy.prefer, skill)))
        if op == "add_forbid_skill":
            return _replace_policy(spec, policy_name, replace(policy, forbid=_append(policy.forbid, skill)))
        return _replace_policy(spec, policy_name, replace(policy, forbid=_remove(policy.forbid, skill)))

    if op == "add_topology_edge":
        edge = _edge_from_mutation(mutation)
        return replace(spec, topology=_append_mapping(spec.topology, edge))

    if op == "remove_topology_edge":
        edge = _edge_from_mutation(mutation)
        return replace(spec, topology=tuple(item for item in spec.topology if not _edge_matches(item, edge)))

    if op == "revise_role_objective":
        role_name = _required(mutation, "role")
        objective = _required(mutation, "objective")
        if role_name not in spec.roles:
            raise SocietyPatchError(f"unknown role: {role_name}")
        roles = dict(spec.roles)
        roles[role_name] = replace(roles[role_name], objective=objective)
        return replace(spec, roles=roles)

    if op == "adjust_budget":
        budget = dict(spec.budget_policy)
        for key, value in dict(mutation.get("budget") or {}).items():
            budget[str(key)] = value
        return replace(spec, budget_policy=budget)

    if op == "add_motif":
        # Motif package creation is handled elsewhere.  The society records the motif
        # as a preferred skill when the curator identifies a target policy.
        policy_name = str(mutation.get("policy") or "")
        skill_id = str(mutation.get("skill_id") or "")
        if not policy_name or not skill_id:
            return spec
        policy = _policy(spec, policy_name)
        return _replace_policy(spec, policy_name, replace(policy, prefer=_append(policy.prefer, skill_id)))

    return spec


def _policy(spec: SocietySpec, policy_name: str) -> SkillAccessPolicy:
    if policy_name not in spec.skill_access_policies:
        raise SocietyPatchError(f"unknown skill access policy: {policy_name}")
    return spec.skill_access_policies[policy_name]


def _replace_policy(spec: SocietySpec, policy_name: str, policy: SkillAccessPolicy) -> SocietySpec:
    policies = dict(spec.skill_access_policies)
    policies[policy_name] = policy
    return replace(spec, skill_access_policies=policies)


def _required(mutation: dict[str, Any], key: str) -> str:
    value = str(mutation.get(key) or "")
    if not value:
        raise SocietyPatchError(f"mutation {mutation.get('op')} requires {key}")
    return value


def _append(items: tuple[str, ...], value: str) -> tuple[str, ...]:
    return items if value in items else (*items, value)


def _remove(items: tuple[str, ...], value: str) -> tuple[str, ...]:
    return tuple(item for item in items if item != value)


def _edge_from_mutation(mutation: dict[str, Any]) -> dict[str, Any]:
    edge = dict(mutation.get("edge") or {})
    if not edge:
        edge = {key: mutation[key] for key in ("from", "to", "when") if key in mutation}
    if "from" not in edge or "to" not in edge:
        raise SocietyPatchError("topology edge mutation requires from and to")
    return {str(key): value for key, value in edge.items()}


def _append_mapping(items: tuple[dict[str, Any], ...], value: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    for item in items:
        if item == value:
            return items
    return (*items, value)


def _edge_matches(item: dict[str, Any], edge: dict[str, Any]) -> bool:
    return item.get("from") == edge.get("from") and item.get("to") == edge.get("to")
