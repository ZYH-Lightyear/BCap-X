"""技能:载体(:mod:`schema`)、磁盘后端存储(:mod:`store`),以及随包附带的
:mod:`builtin` 技能包。

一个技能就是一个目录包(``SKILL.md`` + 可选的 ``ref/`` 与 ``scripts/`` sidecar)。
store 负责持久化技能包并记录每个技能的 utility;发现方式是 qwen-code 式的渐进披露,
而非检索。
"""

from robomex.skills.builtin import load_builtin_skills, render_inventory
from robomex.skills.schema import Skill, SkillCategory
from robomex.skills.store import SkillLibrary, SkillRecord, SkillUtility

__all__ = [
    "Skill",
    "SkillCategory",
    "SkillLibrary",
    "SkillRecord",
    "SkillUtility",
    "load_builtin_skills",
    "render_inventory",
]
