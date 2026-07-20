"""Runtime invocation specifications for contracted Coding specialists."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from robomex.authoring.artifacts import PortSpec
from robomex.authoring.capabilities import KNOWN_CAPABILITIES
from robomex.dysc.contracts import SkillContract


_AGENT_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class SpecialistSpec:
    """One graph-owned specialist invocation.

    This is not an LLM-authored capability request. The Manager selects a
    contracted skill and authors topology; the compiler derives role, ports,
    capabilities and budget from that skill contract.
    """

    agent_id: str
    specialist_skill: str
    role: str
    objective: str
    task: str
    system_prompt: str = ""
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()
    required_skills: tuple[str, ...] = ()
    preferred_skills: tuple[str, ...] = ()
    skill_access_policy: str = ""
    requested_capabilities: frozenset[str] = frozenset()
    max_turns: int = 6
    verifier: bool = False
    changes_world: bool = False
    # Events this specialist's contract declares it can emit (M2). Empty means
    # the contract predates exit_conditions; edge validation then falls back
    # to the full closed vocabulary.
    exit_conditions: tuple[str, ...] = ()

    @classmethod
    def from_contract(
        cls,
        *,
        agent_id: str,
        contract: SkillContract,
        role: str = "",
        objective: str = "",
        task: str = "",
        max_turns: int | None = None,
    ) -> "SpecialistSpec":
        required = tuple(dict.fromkeys((contract.skill_id, *contract.required_skills)))
        spec = cls(
            agent_id=agent_id,
            specialist_skill=contract.skill_id,
            role=role or contract.role or "specialist",
            objective=objective or f"Execute the {contract.skill_id} specialist stage.",
            task=task or objective or f"Produce the declared outputs for {contract.skill_id}.",
            inputs=tuple(
                PortSpec(port.name, port.schema, port.required, port.frame)
                for port in contract.input_ports
            ),
            outputs=tuple(
                PortSpec(port.name, port.schema, port.required, port.frame)
                for port in contract.output_ports
            ),
            required_skills=required,
            preferred_skills=contract.preferred_skills,
            requested_capabilities=frozenset(contract.capabilities),
            max_turns=max_turns or int(contract.budget.get("max_turns", 6)),
            verifier=contract.role == "verifier",
            changes_world=contract.changes_world,
            exit_conditions=tuple(contract.exit_conditions),
        )
        spec.validate()
        return spec

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SpecialistSpec":
        spec = cls(
            agent_id=str(raw.get("id") or raw.get("agent_id") or ""),
            specialist_skill=str(raw.get("specialist_skill") or raw.get("skill") or ""),
            role=str(raw.get("role") or ""),
            objective=str(raw.get("objective") or ""),
            task=str(raw.get("task") or ""),
            system_prompt=str(raw.get("system_prompt") or ""),
            inputs=tuple(PortSpec.from_mapping(v) for v in raw.get("inputs", ())),
            outputs=tuple(PortSpec.from_mapping(v) for v in raw.get("outputs", ())),
            required_skills=tuple(str(v) for v in raw.get("required_skills", ())),
            preferred_skills=tuple(str(v) for v in raw.get("preferred_skills", ())),
            skill_access_policy=str(raw.get("skill_access_policy") or ""),
            requested_capabilities=frozenset(
                str(v) for v in raw.get("capabilities", raw.get("requested_capabilities", ()))
            ),
            max_turns=max(1, int(raw.get("max_turns", 6))),
            verifier=bool(raw.get("verifier", False)),
            changes_world=bool(raw.get("changes_world", False)),
            exit_conditions=tuple(str(v) for v in raw.get("exit_conditions", ())),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not _AGENT_ID.fullmatch(self.agent_id):
            raise ValueError(
                "Specialist id must start with a letter and contain only "
                "letters, digits, '_' or '-'."
            )
        if not self.specialist_skill.strip():
            raise ValueError("Specialist invocation requires a contracted skill id.")
        if not self.role.strip() or not self.objective.strip() or not self.task.strip():
            raise ValueError("Specialist invocation requires role, objective, and task.")
        unknown = self.requested_capabilities - KNOWN_CAPABILITIES
        if unknown:
            raise ValueError(f"Unknown capability name(s): {', '.join(sorted(unknown))}.")
        names = [port.name for port in (*self.inputs, *self.outputs)]
        if any(not port.name or not port.schema for port in (*self.inputs, *self.outputs)):
            raise ValueError("Dynamic SubAgent ports require non-empty name and schema.")
        if len(names) != len(set(names)):
            raise ValueError("Dynamic SubAgent input/output port names must be unique.")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.agent_id,
            "specialist_skill": self.specialist_skill,
            "role": self.role,
            "objective": self.objective,
            "task": self.task,
            "system_prompt": self.system_prompt,
            "inputs": [port.to_mapping() for port in self.inputs],
            "outputs": [port.to_mapping() for port in self.outputs],
            "required_skills": list(self.required_skills),
            "preferred_skills": list(self.preferred_skills),
            "skill_access_policy": self.skill_access_policy,
            "requested_capabilities": sorted(self.requested_capabilities),
            "max_turns": self.max_turns,
            "verifier": self.verifier,
            "changes_world": self.changes_world,
            "exit_conditions": list(self.exit_conditions),
        }
