"""On-disk skill library: persistence, keyword retrieval, and utility tracking.

Skills are stored as directory packages, grouped by category::

    <root>/observation/<skill_id>/SKILL.md       # the skill prose body
    <root>/observation/<skill_id>/ref/           # optional verifier-agent assets
    <root>/observation/<skill_id>/scripts/       # optional verifier-as-code
    <root>/observation/<skill_id>/utility.json   # runtime learning metadata
    <root>/action/<skill_id>/...
    <root>/high_level/<skill_id>/...

Skill content and learning metadata are stored side by side but kept conceptually
separate: the package (SKILL.md + sidecars) is portable across agents/models, the
utility is local statistics the library uses for retrieval and retirement.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from robomex.skills import Skill, SkillCategory
from robomex.skills.schema import REF_DIR, SCRIPTS_DIR, SKILL_FILE

_UTILITY_FILE = "utility.json"


@dataclass
class SkillUtility:
    """Local learning metadata for one skill (not part of the skill content)."""

    call_count: int = 0
    success_count: int = 0
    last_failure: str = ""
    source: str = "seed"
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.call_count if self.call_count else 0.0


@dataclass
class SkillRecord:
    """A skill paired with its utility metadata."""

    skill: Skill
    utility: SkillUtility

    @property
    def skill_id(self) -> str:
        return self.skill.skill_id


def _keywords(skill: Skill) -> set[str]:
    text = " ".join((skill.name, skill.description, skill.body))
    return {token for token in text.lower().replace(",", " ").split() if len(token) > 2}


class SkillLibrary:
    """Disk-backed store of skill packages: persistence, retrieval, utility."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for category in SkillCategory:
            (self.root / category.value).mkdir(parents=True, exist_ok=True)

    def _dir(self, skill_id: str) -> Path:
        """Locate the on-disk package directory for a skill, across categories."""

        for category in SkillCategory:
            path = self.root / category.value / skill_id
            if (path / SKILL_FILE).exists():
                return path
        raise KeyError(f"unknown skill: {skill_id}")

    def admit(self, skill: Skill, source: str = "seed") -> SkillRecord:
        """Add or overwrite a skill package, initializing its utility if new."""

        dest = self.root / skill.category.value / skill.skill_id
        dest.mkdir(parents=True, exist_ok=True)
        (dest / SKILL_FILE).write_text(skill.to_markdown())
        self._copy_sidecars(skill, dest)

        utility_path = dest / _UTILITY_FILE
        utility = self._load_utility(utility_path) if utility_path.exists() else SkillUtility(source=source)
        utility_path.write_text(json.dumps(asdict(utility), indent=2))
        return SkillRecord(Skill.from_dir(dest), utility)

    def get(self, skill_id: str) -> SkillRecord:
        path = self._dir(skill_id)
        return SkillRecord(Skill.from_dir(path), self._load_utility(path / _UTILITY_FILE))

    def all(self, category: SkillCategory | None = None) -> list[SkillRecord]:
        categories = [category] if category else list(SkillCategory)
        records = []
        for c in categories:
            for skill_md in sorted((self.root / c.value).glob(f"*/{SKILL_FILE}")):
                records.append(self.get(skill_md.parent.name))
        return records

    def retrieve(self, query: str, k: int = 3, category: SkillCategory | None = None) -> list[SkillRecord]:
        """Rank skills by keyword overlap with the query, tie-broken by utility."""

        tokens = {t for t in query.lower().replace(",", " ").split() if len(t) > 2}
        scored = []
        for record in self.all(category):
            overlap = len(tokens & _keywords(record.skill))
            if overlap:
                scored.append((overlap, record.utility.success_rate, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[:k]]

    def compound_skills(self) -> list[SkillRecord]:
        """High-level (compound) skills -- the outer planner's capability menu."""

        return [record for record in self.all() if record.skill.compound]

    def update_utility(self, skill_id: str, success: bool, failure_note: str = "") -> SkillUtility:
        utility_path = self._dir(skill_id) / _UTILITY_FILE
        utility = self._load_utility(utility_path)
        utility.call_count += 1
        if success:
            utility.success_count += 1
        elif failure_note:
            utility.last_failure = failure_note
        utility_path.write_text(json.dumps(asdict(utility), indent=2))
        return utility

    @staticmethod
    def _copy_sidecars(skill: Skill, dest: Path) -> None:
        """Copy the ref/ and scripts/ sidecar dirs from the source package."""

        src = skill.root
        if src is None or src.resolve() == dest.resolve():
            return
        for sub in (REF_DIR, SCRIPTS_DIR):
            src_dir = src / sub
            if src_dir.is_dir():
                shutil.copytree(src_dir, dest / sub, dirs_exist_ok=True)

    @staticmethod
    def _load_utility(path: Path) -> SkillUtility:
        return SkillUtility(**json.loads(path.read_text()))
