"""技能蒸馏 —— 占位(尚未接入最小循环)。

预期的原料是*代码轨迹*(生成的代码 + stdout/stderr + 验证器裁决):成功的轨迹会被
凝练成可复用的 action 技能,失败的轨迹会把恢复提示打补丁到咨询过的技能上,并由一条
“增量价值”规则把关。这些目前都还没实现——蒸馏被搁置,直到最小 Agentic 闭环跑通、
技能设计稳定下来。

``SkillDistiller`` 保留为 no-op,好让那些在运行后调用 ``evolve`` 的入口仍接得上线;
真正的蒸馏留到以后单独改。
"""

from __future__ import annotations

from robomex.core.coder.trace import AgentTrace
from robomex.skills import Skill, SkillLibrary


class SkillDistiller:
    """No-op 占位:接收一条 trace,什么也不学,什么也不录入。"""

    def __init__(self, library: SkillLibrary) -> None:
        self.library = library

    def evolve(self, trace: AgentTrace) -> Skill | None:
        """占位实现。返回 ``None``——不更新 utility,也不录入技能。"""

        return None
