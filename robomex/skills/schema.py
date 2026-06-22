"""技能载体:一个技能就是一个*目录包*,MMSkills 风格。

一个技能 = 程序性知识文本 + 可选的 sidecar 资产,布局成一个自包含目录(渐进披露——
agent 先读 ``SKILL.md``,只在相关时才加载 sidecar)。文件按*读者*拆分::

    <skill_id>/
      SKILL.md          # 执行 agent:何时用 / 分解 / 过程 / 失败恢复
      ref/              # 参考资料
        verify.md       #   权威的 pass/fail rubric(对多模态友好)
        success.png     #   可选的视觉参考(成功帧、好的 mask 等)
      scripts/          # 确定性代码(verifier-as-code,由蒸馏器维护)
        verify.py       #   可选的可执行门

``SKILL.md`` 本身坚持 prose-first:用一小段 YAML frontmatter 承载少数代码会分支的字段,
正文 markdown 原样注入 agent 提示词。没有 typed claim 接口、没有 API 白名单、没有校验——
串联和验证是 agent/planner 的事,从 prose(以及之后从 sidecar)读取,而非由 schema 强制。

代码会看的 frontmatter 字段(全部可选):

- ``kind``:``observation`` | ``action`` —— 仅用于在磁盘上组织技能库。
- ``compound``:``true`` 表示高层技能,其正文编排其他技能。
- ``name`` / ``description``:planner 能力菜单的展示面。

任何其他 frontmatter 字段都原样留在 ``meta`` 里,永不强制。结构由包布局承载;
frontmatter 刻意保持轻薄。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class SkillCategory(str, Enum):
    """三类技能,对应两层设计。

    - ``high_level``:编排叶子技能的复合技能(planner 菜单)。
    - ``observation``:做感知/grounding 的 O-Skill(分割、测量、甄别抓取)。
    - ``action``:移动机器人的叶子 A-Skill。
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
    """从 frontmatter 读取 ``category``,同时兼容旧的 ``kind``/``compound``。"""

    raw = meta.get("category")
    if raw:
        return SkillCategory(str(raw))
    if meta.get("compound"):  # 兼容旧版:复合 action -> high level
        return SkillCategory.HIGH_LEVEL
    return SkillCategory(str(meta.get("kind", "action")))


@dataclass(frozen=True)
class Skill:
    """一个技能包:少量元数据、prose 正文,以及(惰性的)sidecar。

    ``root`` 是技能从磁盘加载时的包目录(内存中构造的技能为 ``None``)。sidecar
    访问器都相对它解析。
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
        """注入提示词的文本(即 markdown 正文,原样)。"""

        return self.body

    @property
    def compound(self) -> bool:
        """高层技能编排其他技能(即 planner 的菜单)。"""

        return self.category is SkillCategory.HIGH_LEVEL

    def verify_doc_path(self) -> Path | None:
        """``ref/verify.md``:验证 Agent 的成功 rubric(若存在)。"""

        if self.root is None:
            return None
        path = self.root / REF_DIR / VERIFY_DOC
        return path if path.is_file() else None

    def reference_paths(self) -> list[Path]:
        """``ref/`` 下的视觉/其他参考资产(不含 ``verify.md``)。"""

        if self.root is None:
            return []
        refs = self.root / REF_DIR
        if not refs.is_dir():
            return []
        return sorted(p for p in refs.iterdir() if p.is_file() and p.name != VERIFY_DOC)

    def verifier_path(self) -> Path | None:
        """``scripts/verify.py``:确定性的 verifier-as-code(若技能自带)。"""

        if self.root is None:
            return None
        path = self.root / SCRIPTS_DIR / VERIFIER_FILE
        return path if path.is_file() else None

    def with_note(self, note: str) -> Skill:
        """在正文末尾追加一条自由文本备注(用于打补丁记录失败教训)。"""

        body = f"{self.body.rstrip()}\n\n## Notes\n\n- {note}\n"
        return replace(self, body=body)

    @classmethod
    def from_dir(cls, path: str | Path) -> Skill:
        """加载一个技能包:解析 ``<path>/SKILL.md`` 并附上 ``root=path``。"""

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
        # 丢弃那些派生的、或不属于 prose-first 约定的键。
        for stale in ("id", "skill_id", "kind", "compound"):
            meta.pop(stale, None)
        frontmatter = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=100).strip()
        return f"---\n{frontmatter}\n---\n\n{self.body.strip()}\n"
