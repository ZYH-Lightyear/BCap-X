"""技能:载体(:mod:`schema`)、磁盘后端存储(:mod:`store`),以及随包附带的
:mod:`builtin` 技能包。

一个技能就是一个目录包(``SKILL.md`` + 可选的 ``references/``、``assets/`` 与
``scripts/`` sidecar)。Sidecar 是普通文件,由 Agent 按 skill base directory 自行读取
或执行,不是 RoboMEx 专用 wrapper 函数。
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
