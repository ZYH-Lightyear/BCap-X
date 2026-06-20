"""Skill carrier: a skill is a *directory package*, MMSkills/jianying-style.

A skill is procedural-knowledge text plus optional sidecar assets, laid out as a
self-contained directory (progressive disclosure -- the agent reads ``SKILL.md``
first and loads sidecars only when relevant). Files are split by *reader*::

    <skill_id>/
      SKILL.md          # the executor agent: when / decompose / procedure / recovery
      ref/              # the Verifier Agent: how to judge success
        verify.md       #   authoritative pass/fail rubric (multimodal-friendly)
        success.png     #   optional visual references (success frame, good mask, ...)
      scripts/          # deterministic code (verifier-as-code, distiller-maintained)
        verify.py       #   optional executable gate

``SKILL.md`` itself stays prose-first: a short YAML frontmatter for the few things
code branches on, and a markdown body injected into the agent prompt verbatim.
There is no typed claim interface, no API whitelist, no validation -- chaining and
verification are the agent/planner's job, read from the prose (and, later, from the
sidecars), not enforced by a schema.

Frontmatter keys that code looks at (all optional):

- ``kind``: ``observation`` | ``action`` -- only organizes the library on disk.
- ``compound``: ``true`` for a high-level skill whose body orchestrates others.
- ``name`` / ``description``: the planner's capability-menu surface.

Any other frontmatter key is kept untouched in ``meta`` and never enforced. The
package layout carries structure; the frontmatter stays deliberately thin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class SkillCategory(str, Enum):
    """The three skill categories, mirroring the two-tier design.

    - ``high_level``: compound skills that orchestrate leaf skills (planner menu).
    - ``observation``: O-Skills that perceive/ground (segment, measure, vet grasps).
    - ``action``: leaf A-Skills that move the robot.
    """

    HIGH_LEVEL = "high_level"
    OBSERVATION = "observation"
    ACTION = "action"


_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

SKILL_FILE = "SKILL.md"
REF_DIR = "ref"
VERIFY_DOC = "verify.md"
SCRIPTS_DIR = "scripts"
VERIFIER_FILE = "verify.py"


def _parse_category(meta: dict[str, Any]) -> SkillCategory:
    """Read ``category`` from frontmatter, tolerating the legacy ``kind``/``compound``."""

    raw = meta.get("category")
    if raw:
        return SkillCategory(str(raw))
    if meta.get("compound"):  # legacy: compound action -> high level
        return SkillCategory.HIGH_LEVEL
    return SkillCategory(str(meta.get("kind", "action")))


@dataclass(frozen=True)
class Skill:
    """One skill package: a little metadata, the prose body, and (lazily) its sidecars.

    ``root`` is the on-disk package directory when the skill was loaded from one
    (``None`` for skills built in memory). Sidecar accessors resolve against it.
    """

    skill_id: str
    name: str
    category: SkillCategory = SkillCategory.ACTION
    description: str = ""
    body: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    root: Path | None = None

    @property
    def guidance(self) -> str:
        """The text injected into the prompt (the markdown body, verbatim)."""

        return self.body

    @property
    def compound(self) -> bool:
        """High-level skills orchestrate other skills (the planner's menu)."""

        return self.category is SkillCategory.HIGH_LEVEL

    def verify_doc_path(self) -> Path | None:
        """``ref/verify.md``: the Verifier Agent's success rubric, if present."""

        if self.root is None:
            return None
        path = self.root / REF_DIR / VERIFY_DOC
        return path if path.is_file() else None

    def reference_paths(self) -> list[Path]:
        """Visual/other reference assets under ``ref/`` (excluding ``verify.md``)."""

        if self.root is None:
            return []
        refs = self.root / REF_DIR
        if not refs.is_dir():
            return []
        return sorted(p for p in refs.iterdir() if p.is_file() and p.name != VERIFY_DOC)

    def verifier_path(self) -> Path | None:
        """``scripts/verify.py``: deterministic verifier-as-code, if it ships one."""

        if self.root is None:
            return None
        path = self.root / SCRIPTS_DIR / VERIFIER_FILE
        return path if path.is_file() else None

    def with_note(self, note: str) -> Skill:
        """Append a free-text note to the body (used to patch in failure lessons)."""

        body = f"{self.body.rstrip()}\n\n## Notes\n\n- {note}\n"
        return replace(self, body=body)

    @classmethod
    def from_dir(cls, path: str | Path) -> Skill:
        """Load a skill package: parse ``<path>/SKILL.md`` and attach ``root=path``."""

        path = Path(path)
        text = (path / SKILL_FILE).read_text()
        skill = cls.from_markdown(text, skill_id=path.name)
        return replace(skill, root=path)

    @classmethod
    def from_markdown(cls, text: str, skill_id: str | None = None) -> Skill:
        match = _FRONTMATTER.match(text)
        if match:
            meta = yaml.safe_load(match.group(1)) or {}
            body = text[match.end():].strip()
        else:
            meta, body = {}, text.strip()
        sid = skill_id or meta.get("id") or meta.get("skill_id") or meta.get("name") or "skill"
        return cls(
            skill_id=str(sid),
            name=str(meta.get("name", sid)),
            category=_parse_category(meta),
            description=str(meta.get("description", "") or ""),
            body=body,
            meta=dict(meta),
        )

    def to_markdown(self) -> str:
        meta = dict(self.meta)
        meta.setdefault("name", self.name)
        meta["category"] = self.category.value
        if self.description:
            meta.setdefault("description", self.description)
        # Drop keys that are derived or not part of the prose-first contract.
        for stale in ("id", "skill_id", "kind", "compound"):
            meta.pop(stale, None)
        frontmatter = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=100).strip()
        return f"---\n{frontmatter}\n---\n\n{self.body.strip()}\n"
