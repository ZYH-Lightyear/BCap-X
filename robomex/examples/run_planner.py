"""离线两层 demo:反应式 ReactivePlanner -> 内层 Code Agent,逐步推进。

展示外层 planner 参考 task 技能指导给出**下一个自然语言** sub-goal,再由
``RoboMExAgent`` 通过统一 authoring runtime 执行它,如此循环直到 planner 说 DONE。
全部是脚本/mock(无 env、LLM 或网络)。

要跑真机:把 ``ScriptedPlannerPolicy`` 换成 ``LLMPlannerPolicy``,``MockExecutor``
换成 ``CapXExecutorAdapter(env)``,内层 ``ScriptedCodePolicy`` 换成
``LLMCodePolicy(...)``(见 ``run_planner_live.py``)。
"""

from __future__ import annotations

import tempfile

import numpy as np

from robomex.agents import ScriptedPlannerPolicy
from robomex.core.session import RoboMExAgent, RoboMExConfig
from robomex.core.coder import ScriptedCodePolicy
from robomex.core.logging import configure_logging
from robomex.core.sandbox import ActionBlockStatus, BlockExecutionResult, SemanticActionBlock
from robomex.skills import SkillLibrary, load_builtin_skills

# planner LLM 针对 "pick up the black bowl" 会给出的一个反应式步骤,之后 DONE
#(脚本回复用尽后自动返回 DONE)。
PLANNER_REPLIES = [
    "Goal: pick up the black bowl from the table\n"
    "Postcondition: the black bowl is held in the closed gripper",
]


class MockExecutor:
    """每个块都直接完成当前 sub-goal(terminated=True)。"""

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        return BlockExecutionResult(
            block=block,
            ok=True,
            status=ActionBlockStatus.SUCCEEDED,
            stdout=f"[mock] executed {block.name}",
            reward=1.0,
            terminated=True,
            truncated=False,
            observation={"agentview": {"images": {"rgb": np.zeros((4, 4, 3), np.uint8)}}},
            info={"task_completed": True},
        )


def main() -> None:
    configure_logging()
    with tempfile.TemporaryDirectory() as tmp:
        library = SkillLibrary(f"{tmp}/library")
        for skill in load_builtin_skills():
            library.admit(skill, source="builtin")

        task = "pick up the black bowl"
        agent = RoboMExAgent(
            RoboMExConfig(
                library=library,
                planner_policy=ScriptedPlannerPolicy(PLANNER_REPLIES),
                executor=MockExecutor(),
                code_policy=ScriptedCodePolicy([
                '{"tool":"use_skill","args":{"name":"pick_object"}}',
                '{"tool":"run_python","args":{"code":"obs = get_observation()","intent":"inspect scene"}}',
                ]),
            )
        )
        execution = agent.run(task).execution

        print(f"\nplanner status={execution.planner_status}")
        for r in execution.results:
            flag = "OK  " if r.success else "FAIL"
            print(f"  [{flag}] {r.subgoal.goal}  "
                  f"(inner turns={len(r.trace.turns)}, loaded={list(r.trace.loaded_skill_ids)})")


if __name__ == "__main__":
    main()
