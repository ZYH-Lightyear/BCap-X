"""Offline tests for the unified CodingAgent core (no env / LLM / network).

Drives the executor (CodeAsPolicyAgent) and verifier (VerifyCodeAgent) with a
scripted policy and a fake sandbox executor to assert the shared loop: skill
progressive disclosure (USE SKILL), python turns + feedback, terminal parsing,
repeated-action / force-terminal safety.
"""

from __future__ import annotations

from robomex.agent import CodeAsPolicyAgent, ScriptedCodePolicy
from robomex.execution import ActionBlockStatus, BlockExecutionResult, SemanticActionBlock
from robomex.skills import Skill
from robomex.verification import VerifierContext, VerifyCodeAgent


class FakeRecord:
    def __init__(self, skill: Skill) -> None:
        self.skill = skill
        self.skill_id = skill.skill_id


class FakeLibrary:
    """Minimal SkillLibrary stand-in backed by in-memory Skill objects."""

    def __init__(self, skills: list[Skill]) -> None:
        self._by_id = {s.skill_id: FakeRecord(s) for s in skills}

    def all(self) -> list[FakeRecord]:
        return list(self._by_id.values())

    def get(self, skill_id: str) -> FakeRecord:
        return self._by_id[skill_id]


class FakeExecutor:
    """Records executed blocks; returns canned stdout, optional terminate rule."""

    def __init__(self, terminate_when=None) -> None:
        self.blocks: list[SemanticActionBlock] = []
        self._terminate_when = terminate_when or (lambda code: False)

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        self.blocks.append(block)
        terminal = self._terminate_when(block.code)
        return BlockExecutionResult(
            block=block,
            ok=True,
            status=ActionBlockStatus.SUCCEEDED,
            stdout=f"ran {block.name}",
            stderr="",
            reward=1.0 if terminal else 0.0,
            terminated=terminal,
            truncated=False,
            observation={"agentview": {}},
            info={"sandbox_rc": 0, "task_completed": terminal},
        )


def _skill(skill_id: str, desc: str, category: str = "observation") -> Skill:
    return Skill.from_markdown(
        f"---\nname: {skill_id}\ncategory: {category}\ndescription: {desc}\n---\n\nBody of {skill_id}.",
        skill_id=skill_id,
    )


def test_verifier_use_skill_then_python_then_verdict() -> None:
    lib = FakeLibrary([_skill("estimate_geometry", "estimate object geometry")])
    ctx = VerifierContext(sub_goal="estimate height", skills_used=("estimate_geometry",))
    policy = ScriptedCodePolicy([
        "USE SKILL: estimate_geometry",
        "```python\nx = 1  # measure\nprint('measured')\n```",
        '{"verdict": "passed", "confidence": 0.9, "reason": "matches", "evidence": {"overlay": "o.png"}}',
    ])
    ex = FakeExecutor()
    agent = VerifyCodeAgent(executor=ex, policy=policy, context=ctx, library=lib, max_turns=6)
    vtrace = agent.verify()

    assert vtrace.verdict.verdict == "passed"
    assert vtrace.verdict.confidence == 0.9
    assert vtrace.verdict.evidence.get("overlay") == "o.png"
    assert len(vtrace.turns) == 1  # one python judge turn
    assert vtrace.op_trace and "x = 1" in vtrace.op_trace[0]
    # seed block ran first, then the one python turn
    assert ex.blocks[0].name == "verify_seed"
    res = vtrace.result
    assert res.passed


def test_verifier_unknown_skill_does_not_crash() -> None:
    lib = FakeLibrary([_skill("estimate_geometry", "estimate object geometry")])
    ctx = VerifierContext(sub_goal="x", skills_used=("estimate_geometry",))
    policy = ScriptedCodePolicy([
        "USE SKILL: not_a_real_skill",
        '{"verdict": "uncertain", "confidence": 0.2, "reason": "n/a"}',
    ])
    agent = VerifyCodeAgent(executor=FakeExecutor(), policy=policy, context=ctx, library=lib)
    vtrace = agent.verify()
    assert vtrace.verdict.verdict == "uncertain"


def test_verifier_force_terminal_on_exhaust() -> None:
    lib = FakeLibrary([_skill("estimate_geometry", "g")])
    ctx = VerifierContext(sub_goal="x", skills_used=("estimate_geometry",))
    # Always emits the same python block -> never terminal; loop must force-finish.
    policy = ScriptedCodePolicy(["```python\nprint('again')\n```"] * 10)
    agent = VerifyCodeAgent(executor=FakeExecutor(), policy=policy, context=ctx, library=lib, max_turns=3)
    vtrace = agent.verify()
    # Out of steps with no JSON verdict -> a (best-effort) uncertain verdict, no crash.
    assert vtrace.verdict.verdict == "uncertain"
    assert len(vtrace.turns) == 3


def test_executor_terminates_on_env_signal() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="action")])
    policy = ScriptedCodePolicy([
        "```python\nclose_gripper()\n```",
        "```python\nopen_gripper()\n```",
    ])
    ex = FakeExecutor(terminate_when=lambda c: "open_gripper" in c)
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)
    trace = agent.run(task="pick and place")
    assert trace.success
    assert len(trace.turns) == 2
    assert trace.loaded_skill_ids == ()  # scripted policy consulted no skills


def test_executor_loads_skill_via_use_skill() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="action")])
    policy = ScriptedCodePolicy([
        "USE SKILL: grasp",
        "```python\nclose_gripper()\n```",
        "FINISH",
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)
    trace = agent.run(task="grasp the cube")
    assert "grasp" in trace.loaded_skill_ids
    assert len(trace.turns) == 1  # only the python turn is recorded


def _run_all() -> None:
    test_verifier_use_skill_then_python_then_verdict()
    test_verifier_unknown_skill_does_not_crash()
    test_verifier_force_terminal_on_exhaust()
    test_executor_terminates_on_env_signal()
    test_executor_loads_skill_via_use_skill()
    print("all coding_agent offline tests passed")


if __name__ == "__main__":
    _run_all()
