"""磁盘上的技能库:持久化 + utility 统计。

技能选择完全靠 qwen-code 式的*渐进披露*(agent 看到完整的 ``<available_skills>``
名称+描述清单,并用 ``USE SKILL`` 按需拉取正文);技能库**不做**关键词/语义检索。

技能以目录包形式存储,按 category 分组::

    <root>/observation/<skill_id>/SKILL.md       # 技能 prose 正文
    <root>/observation/<skill_id>/ref/           # 可选:验证 agent 资产
    <root>/observation/<skill_id>/scripts/       # 可选:verifier-as-code
    <root>/observation/<skill_id>/utility.json   # 运行期学习元数据
    <root>/action/<skill_id>/...
    <root>/high_level/<skill_id>/...

技能内容与学习元数据并排存放,但概念上分离:技能包(SKILL.md + sidecar)可跨
agent/模型移植,而 utility 是技能库用于检索/退役的本地统计量。
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from robomex.skills.schema import REF_DIR, SCRIPTS_DIR, SKILL_FILE, Skill, SkillCategory

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

    发现方式是渐进披露,而非检索:调用方通过 :meth:`all` / :meth:`compound_skills`
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

    def compound_skills(self) -> list[SkillRecord]:
        """高层(复合)技能 —— 外层 planner 的能力菜单。"""

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
        """从源技能包拷贝 ref/ 和 scripts/ 这两个 sidecar 目录。"""

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
