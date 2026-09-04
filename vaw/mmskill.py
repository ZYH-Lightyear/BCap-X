"""支持渐进式披露的只读 MMSkill 技能库。

每个技能目录只要求包含 ``SKILL.md``。Main 每轮看到轻量索引，只有显式调用
``consult_mmskill`` 后才会看到正文。技能库同时保存来源摘要，供 trace 复现实验，
但这些实现元数据不会进入 Agent Prompt。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib.resources.abc import Traversable
from pathlib import Path
from types import MappingProxyType
from typing import Any

from vaw.evolution.store import skill_tree_digest


def _parse_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    normalized = markdown.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    closing = normalized.find("\n---\n", 4)
    if closing < 0:
        raise ValueError("SKILL.md frontmatter is not closed")

    metadata: dict[str, str] = {}
    for raw_line in normalized[4:closing].splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            raise ValueError(f"invalid SKILL.md frontmatter line: {raw_line!r}")
        metadata[key.strip()] = value.strip().strip('"\'')
    return metadata, normalized[closing + 5 :].strip()


@dataclass(frozen=True)
class MMSkillReference:
    """随技能冻结的历史视觉参考；它不是当前环境观测。"""

    reference_id: str
    state: str
    view: str
    when_to_use: str
    visual_cue: str
    filename: str
    sha256: str
    image_bytes: bytes

    def summary(self) -> dict[str, str]:
        return {
            "reference_id": self.reference_id,
            "state": self.state,
            "view": self.view,
            "when_to_use": self.when_to_use,
            "visual_cue": self.visual_cue,
        }


@dataclass(frozen=True)
class MMSkill:
    """从 MMSkills 风格 ``SKILL.md`` 加载的不可变技能。"""

    skill_id: str
    name: str
    description: str
    body: str
    references: tuple[MMSkillReference, ...] = ()

    @classmethod
    def from_markdown(cls, skill_id: str, markdown: str) -> MMSkill:
        metadata, body = _parse_frontmatter(markdown)
        name = metadata.get("name", "").strip()
        description = metadata.get("description", "").strip()
        if not skill_id.strip():
            raise ValueError("MMSkill directory name must not be empty")
        if not name:
            raise ValueError(f"MMSkill {skill_id!r} is missing frontmatter name")
        if not description:
            raise ValueError(f"MMSkill {skill_id!r} is missing frontmatter description")
        if not body:
            raise ValueError(f"MMSkill {skill_id!r} has an empty SKILL.md body")
        return cls(
            skill_id=skill_id.strip(),
            name=name,
            description=description,
            body=body,
        )

    def index_entry(self) -> dict[str, str]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self.index_entry(),
            "references": [item.summary() for item in self.references],
        }

    def prompt_block(self) -> str:
        return self.body

    def get_reference(self, reference_id: str) -> MMSkillReference:
        for reference in self.references:
            if reference.reference_id == reference_id:
                return reference
        raise ValueError(f"unknown MMSkill reference: {self.skill_id}/{reference_id}")


class MMSkillLibrary:
    """不可变技能目录，并携带 trace-only 的来源信息。"""

    def __init__(
        self,
        skills: tuple[MMSkill, ...],
        *,
        source: str = "memory",
        generation_id: str | None = None,
        digest: str | None = None,
    ) -> None:
        ordered = tuple(sorted(skills, key=lambda skill: skill.skill_id))
        by_id = {skill.skill_id: skill for skill in ordered}
        if len(by_id) != len(ordered):
            raise ValueError("duplicate MMSkill id")
        self._skills = ordered
        self._by_id: Mapping[str, MMSkill] = MappingProxyType(by_id)
        self.source = str(source)
        self.generation_id = generation_id
        self.digest = digest or _skill_catalog_digest(ordered)

    @classmethod
    def builtin(cls) -> MMSkillLibrary:
        return cls.from_root(
            Path(__file__).with_name("mmskills"),
            source="builtin",
            generation_id="builtin",
        )

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        source: str | None = None,
        generation_id: str | None = None,
        expected_digest: str | None = None,
    ) -> MMSkillLibrary:
        """从一个技能快照加载，并在进入 Runtime 前核对内容摘要。"""

        directory = Path(root)
        if not directory.is_dir():
            raise FileNotFoundError(f"MMSkill 根目录不存在: {directory}")
        skills = tuple(
            cls._load_directory(entry)
            for entry in directory.iterdir()
            if entry.is_dir() and entry.joinpath("SKILL.md").is_file()
        )
        if not skills:
            raise ValueError(f"MMSkill 根目录不包含 skill/*/SKILL.md: {directory}")
        digest = skill_tree_digest(directory)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(f"MMSkill 快照摘要不匹配: {directory}")
        return cls(
            skills,
            source=source or str(directory.resolve()),
            generation_id=generation_id,
            digest=digest,
        )

    @staticmethod
    def _load_directory(directory: Traversable) -> MMSkill:
        markdown = directory.joinpath("SKILL.md").read_text(encoding="utf-8")
        skill = MMSkill.from_markdown(directory.name, markdown)
        references_root = directory.joinpath("references")
        index_path = references_root.joinpath("index.json")
        if not index_path.is_file():
            return skill
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "vaw-mmskill-references-v1":
            raise ValueError(f"不支持的 MMSkill reference schema: {directory.name}")
        items = payload.get("references")
        if not isinstance(items, list):
            raise ValueError(f"MMSkill reference index 必须包含 references list: {directory.name}")
        references: list[MMSkillReference] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError(f"MMSkill reference 必须是 object: {directory.name}")
            reference_id = str(item["reference_id"]).strip()
            filename = str(item["file"]).strip()
            metadata = {
                key: str(item[key]).strip()
                for key in ("state", "view", "when_to_use", "visual_cue")
            }
            if not reference_id or reference_id in seen:
                raise ValueError(f"MMSkill reference_id 为空或重复: {directory.name}")
            if not filename or Path(filename).name != filename:
                raise ValueError(f"MMSkill reference file 必须是单个文件名: {filename}")
            if not all(metadata.values()):
                raise ValueError(f"MMSkill reference 视觉语义不能为空: {directory.name}")
            image_bytes = references_root.joinpath(filename).read_bytes()
            digest = hashlib.sha256(image_bytes).hexdigest()
            if digest != str(item["sha256"]):
                raise ValueError(f"MMSkill reference 摘要不匹配: {directory.name}/{filename}")
            references.append(
                MMSkillReference(
                    reference_id=reference_id,
                    state=metadata["state"],
                    view=metadata["view"],
                    when_to_use=metadata["when_to_use"],
                    visual_cue=metadata["visual_cue"],
                    filename=filename,
                    sha256=digest,
                    image_bytes=image_bytes,
                )
            )
            seen.add(reference_id)
        return replace(skill, references=tuple(references))

    def get(self, skill_id: str) -> MMSkill:
        try:
            return self._by_id[skill_id]
        except KeyError as exc:
            raise ValueError(f"unknown MMSkill: {skill_id}") from exc

    def index(self) -> tuple[dict[str, str], ...]:
        return tuple(skill.index_entry() for skill in self._skills)

    def format_index(self, *, inspect_function: str = "consult_mmskill") -> str:
        """按 MMSkill-Agent 的形式生成完整轻量索引。

        同一份索引会被 DAY Runtime 和 NIGHT Evolver 使用；这里只注入各自真实可用的
        读取函数名，避免索引向 Agent 宣告不存在的工具。
        """

        if not self._skills:
            return "可用 MMSkills：无"
        lines = [
            "可用 MMSkills（skill_id - name - description）：",
            *(
                f"- {skill.skill_id} - {skill.name} - {skill.description}"
                for skill in self._skills
            ),
            f"需要完整技能内容时，调用 {inspect_function} 并使用上面的精确 skill_id。",
        ]
        return "\n".join(lines)

    def summary(self) -> list[dict[str, str]]:
        return list(self.index())

    def provenance(self) -> dict[str, Any]:
        """返回只写入 trace 的技能库来源，不进入策略上下文。"""

        return {
            "source": self.source,
            "generation_id": self.generation_id,
            "digest": self.digest,
            "skill_count": len(self._skills),
        }

    def get_reference(self, skill_id: str, reference_id: str) -> MMSkillReference:
        return self.get(skill_id).get_reference(reference_id)


@dataclass
class MMSkillBuffer:
    """当前已加载技能的 overwrite-only episode memory。

    技能正文负责跨物理动作维持策略连续性；历史参考图只服务于首次理解，
    发生物理动作后停止重复展示，避免同一参考图持续占用视觉 token。
    """

    active: MMSkill | None = None
    references_visible: bool = False

    def load(self, skill: MMSkill) -> None:
        self.active = skill
        self.references_visible = True

    def clear(self) -> None:
        self.active = None
        self.references_visible = False

    def hide_references(self) -> None:
        self.references_visible = False

    @property
    def reference_skill(self) -> MMSkill | None:
        return self.active if self.references_visible else None

    def summary(self) -> dict[str, Any] | None:
        return self.active.summary() if self.active is not None else None


def _skill_catalog_digest(skills: tuple[MMSkill, ...]) -> str:
    """为内存构造的技能库提供稳定摘要；文件快照使用完整目录摘要。"""

    digest = hashlib.sha256()
    for skill in skills:
        for value in (skill.skill_id, skill.name, skill.description, skill.body):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        for reference in skill.references:
            for value in (
                reference.reference_id,
                reference.state,
                reference.view,
                reference.when_to_use,
                reference.visual_cue,
                reference.sha256,
            ):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
            digest.update(reference.image_bytes)
    return digest.hexdigest()


__all__ = ["MMSkill", "MMSkillBuffer", "MMSkillLibrary", "MMSkillReference"]
