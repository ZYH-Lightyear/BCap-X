"""Offline two-level demo: ReactivePlanner -> To-Do list -> inner Code Agent.

Shows the outer planner turning a task into a JSON To-Do list of sub-goals over
the high-level (compound) skill menu, then the ``TwoLevelAgent`` running each
sub-goal through the existing inner ``CodeAsPolicyAgent`` (no re-planning).

To drive a real run, swap ``ScriptedPlannerPolicy`` for ``LLMPlannerPolicy``,
``MockExecutor`` for ``CapXExecutorAdapter(env)``, and the inner ``OneShotPolicy``
for ``LLMCodePolicy(...)`` with a multimodal verifier.
"""

from __future__ import annotations

import tempfile

import numpy as np

from robomex.agent import CodeAsPolicyAgent
from robomex.execution import ActionBlockStatus, BlockExecutionResult, SemanticActionBlock
from robomex.library import SkillLibrary
from robomex.planner import ReactivePlanner, ScriptedPlannerPolicy, TwoLevelAgent
from robomex.skills.skills_library import load_skills_library

# What a planner LLM would return for "put the black bowl on the plate":
PLAN_JSON = """[
  {"goal": "pick up the black bowl from the table",
   "skill": "Pick Object",
   "postcondition": "the black bowl is held in the closed gripper"},
  {"goal": "place the held black bowl on the plate",
   "skill": "Place Held Object In Container",
   "postcondition": "the black bowl rests on the plate and the gripper is open"}
]"""


class MockExecutor:
    """Each block completes the current sub-goal (terminated=True)."""

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


class OneShotPolicy:
    """Trivial inner policy: emit one code block; the executor then terminates."""

    def act(self, prompt: list[dict]) -> str:
        return "obs = get_observation()"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        library = SkillLibrary(f"{tmp}/library")
        for skill in load_skills_library():
            library.admit(skill, source="builtin")

        planner = ReactivePlanner(library, ScriptedPlannerPolicy(PLAN_JSON))
        print("capability menu (high-level skills):")
        print(planner.menu())

        task = "put the black bowl on the plate"
        subgoals = planner.plan(task)
        print(f"\nTo-Do list for: {task!r}")
        for i, sg in enumerate(subgoals, 1):
            print(f"  {i}. {sg.goal}")
            print(f"     skill={sg.skill} | done-when: {sg.postcondition}")

        inner = CodeAsPolicyAgent(executor=MockExecutor(), policy=OneShotPolicy(), library=library)
        execution = TwoLevelAgent(planner, inner).run(task)

        print(f"\nexecution success={execution.success}")
        for r in execution.results:
            flag = "OK  " if r.success else "FAIL"
            print(f"  [{flag}] {r.subgoal.goal}  (inner turns={len(r.trace.turns)}, "
                  f"loaded={list(r.trace.loaded_skill_ids)})")


if __name__ == "__main__":
    main()
