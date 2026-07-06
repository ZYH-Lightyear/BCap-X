"""技能载体:一个技能就是一个*目录包*,MMSkills 风格。

一个技能 = 程序性知识文本,布局成一个自包含目录(渐进披露——agent 先读
``SKILL.md`` 短清单,只在相关时才加载正文)::

    <skill_id>/
      SKILL.md          # 何时用 / 分解 / 过程 / 失败恢复

``SKILL.md`` 本身坚持 prose-first:用一小段 YAML frontmatter 承载少数代码会分支的字段,
正文 markdown 原样注入 agent 提示词。没有 typed claim 接口、没有 API 白名单、没有校验——
串联是 agent/planner 的事,从 prose 读取,而非由 schema 强制。

技能包可以携带 Claude-style 侧车目录 ``assets/``、``references/``、``scripts/``。
``scripts/`` 放普通可运行/可检查的 helper 文件;``references/`` 放非直接运行的参考材料。
运行时不会自动把侧车内容注入上下文;技能正文需要显式引用,agent 再根据加载 skill 时给出的
base directory 按需读取、导入或执行。

代码会看的 frontmatter 字段(全部可选):

- ``category``:``perception`` | ``affordance`` | ``motion`` | ``task`` —— 仅用于在磁盘上组织技能库。
- ``name`` / ``description``:planner 能力菜单的展示面。

技能 id 由目录名承载,不由 frontmatter 的 ``id`` / ``skill_id`` 路由。任何其他
frontmatter 字段都原样留在 ``meta`` 里,永不强制。结构由包布局承载;frontmatter
刻意保持轻薄。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class SkillCategory(str, Enum):
    """物理技能在 RoboMEx 闭环里的角色。

    - ``perception``:从观测得到物体/场景状态。
    - ``affordance``:从状态生成可操作点、位姿、约束或候选。
    - ``motion``:组织或执行机器人运动 primitives。
    - ``verification``:检查状态、谓词、关系误差或失败原因。
    - ``motif``:跨技能的闭环组合模式,供 DySC / Act 作为编排指导。
    - ``task``:任务/流程级技能,用于 planner 菜单和 Act 编排。
    """

    PERCEPTION = "perception"
    AFFORDANCE = "affordance"
    MOTION = "motion"
    VERIFICATION = "verification"
    MOTIF = "motif"
    TASK = "task"


_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

SKILL_FILE = "SKILL.md"
SKILL_PACKAGE_EXTRA_DIRS = ("assets", "references", "scripts")


def _parse_category(meta: dict[str, Any]) -> SkillCategory:
    """从 frontmatter 读取 ``category``."""

    raw = meta.get("category")
    if raw:
        return SkillCategory(str(raw))
    return SkillCategory.MOTION


@dataclass(frozen=True)
class Skill:
    """一个技能包:少量元数据、prose 正文,以及磁盘根目录。"""

    skill_id: str
    name: str
    category: SkillCategory = SkillCategory.MOTION
    description: str = ""
    body: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    root: Path | None = None

    @property
    def guidance(self) -> str:
        """注入提示词的文本(即 markdown 正文,原样)。"""

        return self.body

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
        sid = skill_id or meta.get("name") or "skill"
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
        frontmatter = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=100).strip()
        return f"---\n{frontmatter}\n---\n\n{self.body.strip()}\n"
