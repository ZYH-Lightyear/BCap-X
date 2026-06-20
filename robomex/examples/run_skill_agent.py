"""End-to-end RoboMEx Skill Agent loop, runnable offline.

Flow: seed the library with markdown skills -> the router retrieves the most
relevant skills and injects their bodies -> the (scripted) code policy emits
code grounded in feedback -> a mock executor returns CapX-style signals and
observations -> the evidence collector renders before/after comparisons -> the
agent verifies and iterates -> the distiller turns the successful trace into a
new skill and updates utilities.

To drive a real run, swap ``MockExecutor`` for ``CapXExecutorAdapter(env)``,
``ScriptedCodePolicy`` for ``LLMCodePolicy(model=...)``, and the verifier for
``CompositeVerifier(TaskSignalVerifier(), VLMJudgeVerifier(model=...))``.
"""

from __future__ import annotations

import tempfile

import numpy as np

from robomex.agent import CodeAsPolicyAgent, ScriptedCodePolicy
from robomex.distill import SkillDistiller
from robomex.execution import ActionBlockStatus, BlockExecutionResult, ExecutionTraceEvent, SemanticActionBlock
from robomex.library import SkillLibrary
from robomex.perception import EvidenceCollector
from robomex.skills.skills_library import load_skills_library


class MockExecutor:
    """Simulates CapX ``step`` signals and observations for offline runs.

    A turn whose code releases the object (``open_gripper`` after a grasp
    turn) terminates the task.
    """

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)
        self._grasped = False

    def _observation(self) -> dict:
        rgb = self._rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
        return {"agentview": {"images": {"rgb": rgb}}}

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        if "close_gripper" in block.code:
            self._grasped = True
        terminal = self._grasped and "open_gripper" in block.code and "close_gripper" not in block.code
        return BlockExecutionResult(
            block=block,
            ok=True,
            status=ActionBlockStatus.SUCCEEDED,
            stdout=f"[mock] executed {block.name}",
            reward=1.0 if terminal else 0.0,
            terminated=terminal,
            truncated=False,
            observation=self._observation(),
            info={"sandbox_rc": 0, "task_completed": terminal},
            trace_events=(ExecutionTraceEvent("block", block.code.splitlines()[0], block.name),),
        )


# Scripted responses follow the seeded claim chain:
# segment -> geometry -> grasp candidates -> grasp -> place.
SCRIPTED_RESPONSES = [
    "```python\n"
    "obs = get_observation()\n"
    "cam = obs['agentview']\n"
    "results = segment_sam3_text_prompt(cam['images']['rgb'], text_prompt='milk carton')\n"
    "mask = max(results, key=lambda r: r['score'])['mask']\n"
    "points = mask_to_world_points(mask, cam['images']['depth'], cam['intrinsics'], cam['pose_mat'])\n"
    "points, _ = filter_noise(points)\n"
    "grasps_cam, scores = plan_grasp(cam['images']['depth'], cam['intrinsics'], mask.astype(np.int32))\n"
    "g_world, _ = select_top_down_grasp(grasps_cam, scores, cam['pose_mat'])\n"
    "grasp_pos, grasp_quat = decompose_transform(g_world)\n"
    "joints = solve_ik(grasp_pos, grasp_quat)\n"
    "```",
    "```python\n"
    "open_gripper()\n"
    "goto_pose(grasp_pos, grasp_quat, z_approach=0.075)\n"
    "close_gripper()\n"
    "lift_pos = grasp_pos.copy(); lift_pos[2] += 0.10\n"
    "goto_pose(lift_pos, grasp_quat)\n"
    "```",
    "```python\n"
    "basket_results = segment_sam3_text_prompt(cam['images']['rgb'], text_prompt='basket')\n"
    "basket_mask = max(basket_results, key=lambda r: r['score'])['mask']\n"
    "basket_points = mask_to_world_points(basket_mask, cam['images']['depth'], cam['intrinsics'], cam['pose_mat'])\n"
    "basket_points, _ = filter_noise(basket_points)\n"
    "basket_center = get_oriented_bounding_box_from_3d_points(basket_points)['center']\n"
    "basket_top_z = basket_points[:, 2].max()\n"
    "release_z = basket_top_z + (points[:, 2].max() - points[:, 2].min()) / 2 + 0.03\n"
    "target = np.array([basket_center[0], basket_center[1], release_z + 0.10])\n"
    "goto_pose(target, np.array([0.0, 1.0, 0.0, 0.0]), z_approach=0.05)\n"
    "open_gripper()\n"
    "```",
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        library = SkillLibrary(f"{tmp}/library")
        for skill in load_skills_library():
            library.admit(skill, source="builtin")
        print(f"library loaded with: {[(r.skill.category.value, r.skill_id) for r in library.all()]}")

        agent = CodeAsPolicyAgent(
            executor=MockExecutor(),
            policy=ScriptedCodePolicy(SCRIPTED_RESPONSES),
            library=library,
            collector=EvidenceCollector(f"{tmp}/evidence"),
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
            print(f"  [{record.skill.category.value:11s}] {record.skill_id:42s} source={u.source:10s} "
                  f"calls={u.call_count} success_rate={u.success_rate:.2f}")


if __name__ == "__main__":
    main()
