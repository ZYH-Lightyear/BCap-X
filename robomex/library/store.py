"""On-disk skill library: persistence, keyword retrieval, and utility tracking.

Layout (skills are markdown content files, grouped by species)::

    <root>/observation/<skill_id>.md            # the skill (contract + guidance)
    <root>/observation/<skill_id>.utility.json  # runtime learning metadata
    <root>/action/<skill_id>.md
    <root>/action/<skill_id>.utility.json

Skill content and learning metadata are stored separately on purpose: the
knowledge is portable across agents/models, the utility is local statistics
the library uses for retrieval and retirement.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from robomex.skills import Skill, SkillKind


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
    text = " ".join((
        skill.name, skill.description,
        *skill.keywords, *skill.requires, *skill.produces,
    ))
    return {token for token in text.lower().replace(",", " ").split() if len(token) > 2}


class SkillLibrary:
    """Disk-backed store of dual-species multimodal executable skills."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for kind in SkillKind:
            (self.root / kind.value).mkdir(parents=True, exist_ok=True)

    def _path(self, skill_id: str) -> Path:
        for kind in SkillKind:
            path = self.root / kind.value / f"{skill_id}.md"
            if path.exists():
                return path
        raise KeyError(f"unknown skill: {skill_id}")

    def admit(self, skill: Skill, source: str = "seed") -> SkillRecord:
        """Add or overwrite a skill, initializing its utility if new."""

        path = self.root / skill.kind.value / f"{skill.skill_id}.md"
        path.write_text(skill.to_markdown())

        utility_path = path.with_suffix(".utility.json")
        utility = self._load_utility(utility_path) if utility_path.exists() else SkillUtility(source=source)
        utility_path.write_text(json.dumps(asdict(utility), indent=2))
        return SkillRecord(skill, utility)

    def get(self, skill_id: str) -> SkillRecord:
        path = self._path(skill_id)
        skill = Skill.from_markdown(path.read_text())
        return SkillRecord(skill, self._load_utility(path.with_suffix(".utility.json")))

    def all(self, kind: SkillKind | None = None) -> list[SkillRecord]:
        kinds = [kind] if kind else list(SkillKind)
        records = []
        for k in kinds:
            for path in sorted((self.root / k.value).glob("*.md")):
                records.append(self.get(path.stem))
        return records

    def retrieve(self, query: str, k: int = 3, kind: SkillKind | None = None) -> list[SkillRecord]:
        """Rank skills by keyword overlap with the query, tie-broken by utility."""

        tokens = {t for t in query.lower().replace(",", " ").split() if len(t) > 2}
        scored = []
        for record in self.all(kind):
            overlap = len(tokens & _keywords(record.skill))
            if overlap:
                scored.append((overlap, record.utility.success_rate, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[:k]]

    def producers_of(self, claim_types: tuple[str, ...]) -> list[SkillRecord]:
        """Observation skills whose produced claims intersect ``claim_types``."""

        wanted = set(claim_types)
        return [
            record for record in self.all(SkillKind.OBSERVATION)
            if wanted & set(record.skill.produces)
        ]

    def update_utility(self, skill_id: str, success: bool, failure_note: str = "") -> SkillUtility:
        utility_path = self._path(skill_id).with_suffix(".utility.json")
        utility = self._load_utility(utility_path)
        utility.call_count += 1
        if success:
            utility.success_count += 1
        elif failure_note:
            utility.last_failure = failure_note
        utility_path.write_text(json.dumps(asdict(utility), indent=2))
        return utility

    @staticmethod
    def _load_utility(path: Path) -> SkillUtility:
        return SkillUtility(**json.loads(path.read_text()))
