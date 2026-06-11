"""Lightweight dual-species skill schema (v1.1).

A skill is one markdown file with YAML frontmatter:

- frontmatter = the machine-facing contract (kind, claim interface, verify,
  recovery, retrieval keywords);
- body = the guidance text injected into the LLM prompt as-is. Seed skills are
  hand-written; distilled skills are generated. One representation for both.

Two species share the schema, differing only in what their claims mean:

- ``observation`` skills produce claims about the world (gate 1 verifies them);
- ``action`` skills consume claims as preconditions and produce effect-claims
  that should become true after execution (gate 3 verifies them).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import yaml


class SkillKind(str, Enum):
    OBSERVATION = "observation"
    ACTION = "action"


# Typed interface between O-skills and A-skills. An action skill may only
# require claim types that some observation skill produces; effect-claims are
# what gate-3 verification checks after the action ran.
CLAIM_TYPES: dict[str, str] = {
    "object_mask": "2D segmentation mask of a named object in a camera view.",
    "object_points": "Noise-filtered 3D points of a named object.",
    "object_geometry": "Oriented bounding box / size estimate of an object.",
    "grasp_pose": "A grasp position + quaternion, ideally IK-feasible.",
    "object_grasped": "The object is held in the closed gripper (effect).",
    "object_in_container": "The object rests inside the target container (effect).",
}


@dataclass(frozen=True)
class Claim:
    """A typed runtime assertion about the world, produced by a skill."""

    claim_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    source_skill: str | None = None
    evidence: str = ""


_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

_TUPLE_FIELDS = ("requires", "produces", "apis", "verify", "recovery", "keywords")


@dataclass(frozen=True)
class Skill:
    """One multimodal executable skill: contract (frontmatter) + guidance (body)."""

    kind: SkillKind
    skill_id: str
    name: str
    description: str
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    apis: tuple[str, ...] = ()
    verify: tuple[str, ...] = ()
    recovery: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    guidance: str = ""
    version: str = "0.1"

    def with_recovery(self, note: str) -> Skill:
        """Return a copy with one recovery hint appended (failure patching)."""

        return replace(self, recovery=self.recovery + (note,))

    @classmethod
    def from_markdown(cls, text: str) -> Skill:
        match = _FRONTMATTER.match(text)
        if not match:
            raise ValueError("skill markdown must start with a YAML frontmatter block")
        meta = yaml.safe_load(match.group(1)) or {}
        body = text[match.end():].strip()
        return cls(
            kind=SkillKind(meta["kind"]),
            skill_id=meta["skill_id"],
            name=meta["name"],
            description=meta.get("description", ""),
            guidance=body,
            version=str(meta.get("version", "0.1")),
            **{f: tuple(meta.get(f, ()) or ()) for f in _TUPLE_FIELDS},
        )

    def to_markdown(self) -> str:
        meta: dict[str, Any] = {
            "kind": self.kind.value,
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }
        for f in _TUPLE_FIELDS:
            value = getattr(self, f)
            if value:
                meta[f] = list(value)
        frontmatter = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=100).strip()
        return f"---\n{frontmatter}\n---\n\n{self.guidance.strip()}\n"
