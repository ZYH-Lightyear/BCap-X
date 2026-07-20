"""角色 agent,全部建在共享 coder(:mod:`robomex.core.coder`)之上。

- :class:`CodeAsPolicyAgent`(``executor``)—— 写执行代码。
- :class:`ReactivePlanner`(``planner``)—— 逐步给出下一个高层 sub-goal。
- :class:`SkillDistiller`(``evolve``)—— 把轨迹蒸馏成技能(占位)。
"""

from robomex.agents.evolve import SkillDistiller
from robomex.agents.executor import CodeAsPolicyAgent
from robomex.agents.planner import (
    LLMPlannerPolicy,
    PlanExecution,
    PlannerPolicy,
    ReactivePlanner,
    ScriptedPlannerPolicy,
    SubGoal,
    SubGoalResult,
    parse_next_subgoal,
)
from robomex.agents.subagents import (
    CodingAgentSubAgent,
    SubAgentRequest,
    SubAgentResult,
    render_subagent_system_prompt,
)
__all__ = [
    "CodeAsPolicyAgent",
    "CodingAgentSubAgent",
    "LLMPlannerPolicy",
    "PlanExecution",
    "PlannerPolicy",
    "ReactivePlanner",
    "ScriptedPlannerPolicy",
    "SkillDistiller",
    "SubGoal",
    "SubGoalResult",
    "SubAgentRequest",
    "SubAgentResult",
    "parse_next_subgoal",
    "render_subagent_system_prompt",
]
