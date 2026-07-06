"""Structured skill contracts for DySC.

RoboMEx skills remain Claude/Codex-style packages whose primary interface is
SKILL.md.  DySC optionally reads sidecar ``contract.yaml`` files to build
composition graphs and role-conditioned library views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


CONTRACT_FILE = "contract.yaml"


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
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, fallback_skill_id: str = "") -> "SkillContract":
        skill_id = str(data.get("skill_id") or fallback_skill_id)
        suggested_next = {
            str(key): tuple(str(item) for item in (value or ()))
            for key, value in dict(data.get("suggested_next") or {}).items()
        }
        return cls(
            skill_id=skill_id,
            tags=tuple(str(item) for item in (data.get("tags") or ())),
            changes_world=bool(data.get("changes_world", False)),
            inputs=tuple(str(item) for item in (data.get("inputs") or ())),
            outputs=tuple(str(item) for item in (data.get("outputs") or ())),
            uncertainty=tuple(str(item) for item in (data.get("uncertainty") or ())),
            postconditions=tuple(str(item) for item in (data.get("postconditions") or ())),
            suggested_next=suggested_next,
            raw=dict(data),
        )


def load_skill_contracts(library_root: str | Path) -> dict[str, SkillContract]:
    """Load every ``contract.yaml`` under a RoboMEx skill library root."""

    root = Path(library_root)
    contracts: dict[str, SkillContract] = {}
    for path in sorted(root.glob(f"*/*/{CONTRACT_FILE}")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        contract = SkillContract.from_mapping(data, fallback_skill_id=path.parent.name)
        if contract.skill_id:
            contracts[contract.skill_id] = contract
    return contracts
