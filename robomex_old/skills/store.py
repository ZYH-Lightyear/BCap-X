"""磁盘上的技能库:持久化 + utility 统计。

技能选择完全靠 qwen-code 式的*渐进披露*(agent 看到完整的 ``<available_skills>``
名称+描述清单,并用 ``USE SKILL`` 按需拉取正文);技能库**不做**关键词/语义检索。

技能以目录包形式存储,按 category 分组::

    <root>/perception/<skill_id>/SKILL.md       # 技能 prose 正文
    <root>/perception/<skill_id>/utility.json   # 运行期学习元数据
    <root>/affordance/<skill_id>/...
    <root>/motion/<skill_id>/...
    <root>/task/<skill_id>/...

技能内容与学习元数据并排存放,但概念上分离:技能包(SKILL.md)可跨
agent/模型移植,而 utility 是技能库用于检索/退役的本地统计量。
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from robomex.skills.schema import (
    SKILL_FILE,
    SKILL_PACKAGE_EXTRA_DIRS,
    SKILL_PACKAGE_EXTRA_FILES,
    Skill,
    SkillCategory,
)

_UTILITY_FILE = "utility.json"


@dataclass
class SkillUtility:
    """单个技能的本地学习元数据(不属于技能内容本身)。"""

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
    """技能 + 其 utility 元数据的配对。"""

    skill: Skill
    utility: SkillUtility

    @property
    def skill_id(self) -> str:
        return self.skill.skill_id


class SkillLibrary:
    """技能包的磁盘后端存储:持久化 + utility。

    发现方式是渐进披露,而非检索:调用方通过 :meth:`all` / :meth:`task_skills`
    列出整库,通过 :meth:`get` 按 id 加载某个具体技能包。
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for category in SkillCategory:
            (self.root / category.value).mkdir(parents=True, exist_ok=True)

    def _dir(self, skill_id: str) -> Path:
        """跨所有 category 定位某技能在磁盘上的包目录。"""

        for category in SkillCategory:
            path = self.root / category.value / skill_id
            if (path / SKILL_FILE).exists():
                return path
        raise KeyError(f"unknown skill: {skill_id}")

    def admit(self, skill: Skill, source: str = "seed") -> SkillRecord:
        """新增或覆盖一个技能包;若是新技能则初始化其 utility。"""

        dest = self.root / skill.category.value / skill.skill_id
        dest.mkdir(parents=True, exist_ok=True)
        (dest / SKILL_FILE).write_text(skill.to_markdown())
        if skill.root is not None:
            for dirname in SKILL_PACKAGE_EXTRA_DIRS:
                src = skill.root / dirname
                if not src.exists():
                    continue
                dst = dest / dirname
                # Idempotent overwrite (M1.5 Fix G / B9): concurrent admits of the
                # same builtin race between rmtree and copytree; dirs_exist_ok
                # closes that window and __pycache__ noise is never copied.
                shutil.copytree(
                    src,
                    dst,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            for filename in SKILL_PACKAGE_EXTRA_FILES:
                src = skill.root / filename
                if src.is_file():
                    shutil.copy2(src, dest / filename)

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

    def task_skills(self) -> list[SkillRecord]:
        """任务级技能 —— 外层 planner 的能力菜单。"""

        return self.all(SkillCategory.TASK)

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
    def _load_utility(path: Path) -> SkillUtility:
        return SkillUtility(**json.loads(path.read_text()))
