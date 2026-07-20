"""Structured skill contracts for DySC.

RoboMEx skills remain Claude/Codex-style packages whose primary interface is
SKILL.md.  DySC optionally reads sidecar ``contract.yaml`` files to build
role-conditioned library views and validate dynamically composed graphs.

M2 upgrades the contract into the complete machine-authoritative face of a
skill (SKILL.md keeps only the *why*):

- ``exit_conditions``: the subset of the closed edge-event vocabulary this
  skill can actually emit, with per-event semantics.  The graph compiler uses
  it to reject dead recovery edges at compile time.
- ``functions``: canonical implementations shipped by the skill.  The runtime
  binds them into the leaf agent's sandbox namespace; they are high-level
  primitives, never a forced path.
- ``prompts``: curated prompt templates (e.g. VLM question templates) the
  runtime seeds into the sandbox so agents reuse polished wording instead of
  improvising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


CONTRACT_FILE = "contract.yaml"


@dataclass(frozen=True)
class ContractPort:
    """A lightweight typed port declared by a skill contract."""

    name: str
    schema: str
    required: bool = True
    frame: str = ""

    @classmethod
    def from_any(cls, value: str | dict[str, Any]) -> "ContractPort":
        if isinstance(value, str):
            return cls(name=value, schema=f"robomex.{value}.v1")
        return cls(
            name=str(value.get("name") or ""),
            schema=str(value.get("schema") or ""),
            required=bool(value.get("required", True)),
            frame=str(value.get("frame") or ""),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema": self.schema,
            "required": self.required,
            "frame": self.frame,
        }


@dataclass(frozen=True)
class ContractFunction:
    """One canonical function a skill ships as an injectable sandbox primitive.

    ``entry`` locates the implementation relative to the skill package root,
    in ``<relative_path>.py:<function_name>`` form (e.g.
    ``scripts/place_trajectory.py:build_place_trajectory``).  ``signature`` is
    optional documentation sugar; when omitted the loader derives it from the
    source AST so prompts always show a truthful call shape.
    """

    name: str
    entry: str
    signature: str = ""
    description: str = ""

    @classmethod
    def from_any(cls, value: dict[str, Any]) -> "ContractFunction":
        return cls(
            name=str(value.get("name") or ""),
            entry=str(value.get("entry") or ""),
            signature=str(value.get("signature") or ""),
            description=str(value.get("description") or ""),
        )

    @property
    def entry_path(self) -> str:
        """Relative source-file part of ``entry`` ('' when malformed)."""

        path, _, func = self.entry.rpartition(":")
        return path if path and func else ""

    @property
    def entry_function(self) -> str:
        """Function-name part of ``entry`` ('' when malformed)."""

        path, _, func = self.entry.rpartition(":")
        return func if path and func else ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entry": self.entry,
            "signature": self.signature,
            "description": self.description,
        }


@dataclass(frozen=True)
class ContractPrompt:
    """One curated prompt template shipped by a skill package."""

    name: str
    path: str
    description: str = ""

    @classmethod
    def from_any(cls, value: dict[str, Any]) -> "ContractPrompt":
        return cls(
            name=str(value.get("name") or ""),
            path=str(value.get("path") or ""),
            description=str(value.get("description") or ""),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "description": self.description}


@dataclass(frozen=True)
class SkillContract:
    """Machine-readable composition metadata for one skill."""

    skill_id: str
    tags: tuple[str, ...] = ()
    changes_world: bool = False
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    suggested_next: dict[str, tuple[str, ...]] = field(default_factory=dict)
    role: str = ""
    capabilities: tuple[str, ...] = ()
    forbidden_capabilities: tuple[str, ...] = ()
    required_skills: tuple[str, ...] = ()
    preferred_skills: tuple[str, ...] = ()
    input_ports: tuple[ContractPort, ...] = ()
    output_ports: tuple[ContractPort, ...] = ()
    budget: dict[str, Any] = field(default_factory=dict)
    # M2 machine-authoritative extensions. All optional: a contract without
    # them behaves exactly as before.
    exit_conditions: dict[str, str] = field(default_factory=dict)
    functions: tuple[ContractFunction, ...] = ()
    prompts: tuple[ContractPrompt, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, fallback_skill_id: str = "") -> "SkillContract":
        skill_id = str(data.get("skill_id") or fallback_skill_id)
        suggested_next = {
            str(key): tuple(str(item) for item in (value or ()))
            for key, value in dict(data.get("suggested_next") or {}).items()
        }
        input_values = tuple(data.get("input_ports") or data.get("inputs") or ())
        output_values = tuple(data.get("output_ports") or data.get("outputs") or ())
        input_ports = tuple(ContractPort.from_any(value) for value in input_values)
        output_ports = tuple(ContractPort.from_any(value) for value in output_values)
        exit_conditions = {
            str(event): str(meaning or "")
            for event, meaning in dict(data.get("exit_conditions") or {}).items()
        }
        functions = tuple(
            ContractFunction.from_any(dict(value))
            for value in (data.get("functions") or ())
            if isinstance(value, dict)
        )
        prompts = tuple(
            ContractPrompt.from_any(dict(value))
            for value in (data.get("prompts") or ())
            if isinstance(value, dict)
        )
        return cls(
            skill_id=skill_id,
            tags=tuple(str(item) for item in (data.get("tags") or ())),
            changes_world=bool(data.get("changes_world", False)),
            inputs=tuple(port.name for port in input_ports),
            outputs=tuple(port.name for port in output_ports),
            uncertainty=tuple(str(item) for item in (data.get("uncertainty") or ())),
            postconditions=tuple(str(item) for item in (data.get("postconditions") or ())),
            suggested_next=suggested_next,
            role=str(data.get("role") or ""),
            capabilities=tuple(str(item) for item in (data.get("capabilities") or ())),
            forbidden_capabilities=tuple(
                str(item) for item in (data.get("forbidden_capabilities") or ())
            ),
            required_skills=tuple(str(item) for item in (data.get("required_skills") or ())),
            preferred_skills=tuple(str(item) for item in (data.get("preferred_skills") or ())),
            input_ports=input_ports,
            output_ports=output_ports,
            budget=dict(data.get("budget") or {}),
            exit_conditions=exit_conditions,
            functions=functions,
            prompts=prompts,
            raw=dict(data),
        )


def load_contract_for_skill(skill_root: str | Path) -> SkillContract | None:
    """Load the sidecar contract of one skill package, if it ships one.

    No consistency validation here: the library-level loader already vets
    every contract at startup; this accessor serves runtime consumers (e.g.
    the coding agent binding canonical functions on skill load).
    """

    root = Path(skill_root)
    path = root / CONTRACT_FILE
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return SkillContract.from_mapping(data, fallback_skill_id=root.name)


def load_skill_contracts(
    library_root: str | Path,
    *,
    validate: bool = True,
) -> dict[str, SkillContract]:
    """Load every ``contract.yaml`` under a RoboMEx skill library root.

    With ``validate=True`` (the default) each contract's M2 declarations are
    checked against the skill package on disk — referenced files must exist,
    declared functions must be introspectable, and exit_conditions must stay
    inside the closed edge-event vocabulary.  A broken declaration raises at
    load time instead of surfacing as a dead edge or missing primitive mid-run.
    """

    # Local import: validation needs the edge-event vocabulary from core, and
    # keeping it out of module import time preserves dysc as a leaf package.
    from robomex.dysc.contract_checks import contract_consistency_errors

    root = Path(library_root)
    contracts: dict[str, SkillContract] = {}
    errors: list[str] = []
    for path in sorted(root.glob(f"*/*/{CONTRACT_FILE}")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        contract = SkillContract.from_mapping(data, fallback_skill_id=path.parent.name)
        if not contract.skill_id:
            continue
        if validate:
            errors.extend(contract_consistency_errors(contract, path.parent))
        contracts[contract.skill_id] = contract
    if errors:
        raise ValueError(
            "Skill contract validation failed:\n- " + "\n- ".join(errors)
        )
    return contracts
