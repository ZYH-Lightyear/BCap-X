"""Seed skills shipped as markdown files; loaded straight from this directory."""

from __future__ import annotations

from pathlib import Path

from robomex.skills.schema import Skill

_SEED_DIR = Path(__file__).parent


def load_seed_skills() -> list[Skill]:
    """Parse every seed ``*.md`` in this package into a Skill."""

    return [Skill.from_markdown(p.read_text()) for p in sorted(_SEED_DIR.glob("*.md"))]


__all__ = ["load_seed_skills"]
