"""End-to-end RoboMEx Skill Agent loop, runnable offline.

Flow: seed the library -> agent retrieves a skill and gets compact guidance ->
the (scripted) code policy emits code grounded in feedback -> a mock executor
returns CapX-style signals -> the agent verifies and iterates -> the distiller
turns the successful trace into a new skill and updates utilities.

To drive a real run, swap ``MockExecutor`` for ``CapXExecutorAdapter(env)`` and
``ScriptedCodePolicy`` for ``LLMCodePolicy(model=...)``.
"""

from __future__ import annotations

import tempfile

from robomex.agent import CodeAsPolicyAgent, ScriptedCodePolicy
from robomex.distill import SkillDistiller
from robomex.execution import ActionBlockStatus, BlockExecutionResult, ExecutionTraceEvent, SemanticActionBlock
from robomex.library import SkillLibrary
from robomex.skills.seeds import load_seed_skills


class MockExecutor:
    """Simulates CapX ``step`` signals for the agent's generated code offline.

    A turn whose code releases the object (``open_gripper``) terminates the task.
    """

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        terminal = "open_gripper" in block.code
        return BlockExecutionResult(
            block=block,
            ok=True,
            status=ActionBlockStatus.SUCCEEDED,
            stdout=f"[mock] executed {block.name}",
            reward=1.0 if terminal else 0.0,
            terminated=terminal,
            truncated=False,
            info={"sandbox_rc": 0, "task_completed": terminal},
            trace_events=(ExecutionTraceEvent("block", block.code.splitlines()[0], block.name),),
        )


SCRIPTED_RESPONSES = [
    "```python\n"
    "mask_pc = get_object_3d_points_and_masks_from_language('milk carton')\n"
    "obb = get_oriented_bounding_box_from_3d_points(filter_noise(mask_pc['points_3d'])[0])\n"
    "grasp_pos, grasp_quat = get_top_down_grasp_from_obb(obb)\n"
    "ok, traj, _ = plan_grasp_trajectory('milk carton', object_mask=mask_pc['agentview_mask'], grasp_poses=[(grasp_pos, grasp_quat)])\n"
    "execute_joint_trajectory(traj, subsample=2)\n"
    "close_gripper()\n"
    "```",
    "```python\n"
    "basket = get_object_3d_points_and_masks_from_language('basket')\n"
    "basket_pos = get_oriented_bounding_box_from_3d_points(filter_noise(basket['points_3d'])[0])['center']\n"
    "ok, traj = plan_with_grasped_object((basket_pos + np.array([0,0,0.3]), np.array([0.,1.,0.,0.])), 'milk carton')\n"
    "execute_joint_trajectory(traj, subsample=2)\n"
    "open_gripper()\n"
    "```",
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        library = SkillLibrary(tmp)
        for skill in load_seed_skills():
            library.admit(skill, source="seed")
        print(f"library seeded with: {[(r.skill.kind.value, r.skill_id) for r in library.all()]}")

        agent = CodeAsPolicyAgent(
            executor=MockExecutor(),
            policy=ScriptedCodePolicy(SCRIPTED_RESPONSES),
            library=library,
        )
        trace = agent.run(task="pick up the milk carton and place it into the basket")

        print(f"\nretrieval query: {trace.skill_query}")
        print(f"loaded skills:   {list(trace.loaded_skill_ids)}")
        print(f"turns:           {len(trace.turns)}  success={trace.success}")
        for t in trace.turns:
            print(f"  turn {t.turn}: exec={t.execution.status.value} verdict={t.verification.status.value}")

        distiller = SkillDistiller(library)
        learned = distiller.evolve(trace)
        print(f"\ndistilled skill: {learned.skill_id if learned else None}")

        print("\nlibrary after evolution:")
        for record in library.all():
            u = record.utility
            print(f"  [{record.skill.kind.value:11s}] {record.skill_id:42s} source={u.source:10s} "
                  f"calls={u.call_count} success_rate={u.success_rate:.2f}")


if __name__ == "__main__":
    main()
