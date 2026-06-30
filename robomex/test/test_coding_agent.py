"""统一 CodingAgent 内核的离线测试(无 env / LLM / 网络)。

用脚本策略 + 假沙箱执行器驱动执行器(CodeAsPolicyAgent)和验证器(VerifyCodeAgent),
断言共享循环:技能渐进披露(use_skill)、python 轮 + 反馈、终止解析、
重复动作 / 强制终止 的安全护栏。
"""

from __future__ import annotations

import json

from robomex.agents import CodeAsPolicyAgent
from robomex.agents.verifier import VerifyCodeAgent
from robomex.core.coder import ScriptedCodePolicy
from robomex.core.sandbox import ActionBlockStatus, BlockExecutionResult, SemanticActionBlock
from robomex.skills import Skill
from robomex.verification import VerifierContext, VerifyResource


class FakeRecord:
    def __init__(self, skill: Skill) -> None:
        self.skill = skill
        self.skill_id = skill.skill_id


class FakeLibrary:
    """最小的 SkillLibrary 替身,用内存里的 Skill 对象支撑。"""

    def __init__(self, skills: list[Skill]) -> None:
        self._by_id = {s.skill_id: FakeRecord(s) for s in skills}

    def all(self) -> list[FakeRecord]:
        return list(self._by_id.values())

    def get(self, skill_id: str) -> FakeRecord:
        return self._by_id[skill_id]


class FakeExecutor:
    """记录执行过的块;返回预设 stdout,可选的终止规则。"""

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


class FailingEvidenceExecutor(FakeExecutor):
    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        if block.name == "verify_seed":
            return super().run_block(block)
        self.blocks.append(block)
        return BlockExecutionResult(
            block=block,
            ok=False,
            status=ActionBlockStatus.FAILED,
            stdout="",
            stderr="boom",
            reward=0.0,
            terminated=False,
            truncated=False,
            observation={"agentview": {}},
            info={"sandbox_rc": 1},
        )


class RecordingPolicy:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[list[dict]] = []
        self._index = 0

    def complete(self, prompt: list[dict]) -> str:
        self.prompts.append(prompt)
        if self._index >= len(self._responses):
            return _finish("recording policy exhausted")
        response = self._responses[self._index]
        self._index += 1
        return response


def _skill(skill_id: str, desc: str, category: str = "observation") -> Skill:
    return Skill.from_markdown(
        f"---\nname: {skill_id}\ncategory: {category}\ndescription: {desc}\n---\n\nBody of {skill_id}.",
        skill_id=skill_id,
    )


def _use_skill(name: str) -> str:
    return json.dumps({"tool": "use_skill", "args": {"name": name}})


def _run_python(code: str, intent: str = "test code") -> str:
    return json.dumps({"tool": "run_python", "args": {"code": code, "intent": intent}})


def _finish(claim: str = "done") -> str:
    return json.dumps({"tool": "finish", "args": {"claim": claim}})


def test_parse_json_action_variants() -> None:
    from robomex.core.coder import parse_action

    action = parse_action(_run_python("print('x')"), lambda _raw: False)
    assert action.kind == "run_python"
    assert action.args["code"] == "print('x')"

    invalid = parse_action("not json", lambda _raw: False)
    assert invalid.kind == "invalid"
    assert invalid.error

    terminal = parse_action('{"status":"pass","confidence":0.9}', lambda _raw: True)
    assert terminal.kind == "finish"


def test_verifier_use_skill_then_python_then_verdict() -> None:
    lib = FakeLibrary([_skill("estimate_geometry", "estimate object geometry")])
    ctx = VerifierContext(sub_goal="estimate height", skills_used=("estimate_geometry",))
    policy = ScriptedCodePolicy([
        _use_skill("estimate_geometry"),
        _run_python("assert OBS_AFTER is not None\nx = EVIDENCE.get('height', 1)\nprint('measured')"),
        '{"verdict": "passed", "confidence": 0.9, "reason": "matches", "evidence": {"overlay": "o.png"}}',
    ])
    ex = FakeExecutor()
    agent = VerifyCodeAgent(executor=ex, policy=policy, context=ctx, library=lib, max_turns=6)
    vtrace = agent.verify()

    assert vtrace.verdict.verdict == "passed"
    assert vtrace.verdict.confidence == 0.9
    assert vtrace.verdict.evidence.get("overlay") == "o.png"
    assert len(vtrace.turns) == 1  # 仅一个 python judge 轮
    assert vtrace.op_trace and "OBS_AFTER" in vtrace.op_trace[0]
    # seed 块先跑,然后才是那一个 python 轮
    assert ex.blocks[0].name == "verify_seed"
    res = vtrace.result
    assert res.passed


def test_verifier_unknown_skill_does_not_crash() -> None:
    lib = FakeLibrary([_skill("estimate_geometry", "estimate object geometry")])
    ctx = VerifierContext(sub_goal="x", skills_used=("estimate_geometry",))
    policy = ScriptedCodePolicy([
        _use_skill("not_a_real_skill"),
        _run_python("assert OBS_AFTER is not None\nprint('inspected')"),
        '{"verdict": "uncertain", "confidence": 0.2, "reason": "n/a"}',
    ])
    agent = VerifyCodeAgent(executor=FakeExecutor(), policy=policy, context=ctx, library=lib)
    vtrace = agent.verify()
    assert vtrace.verdict.verdict == "uncertain"
    assert len(vtrace.turns) == 1


def test_verifier_evidence_then_judge_uncertain() -> None:
    lib = FakeLibrary([_skill("estimate_geometry", "g")])
    ctx = VerifierContext(sub_goal="x", skills_used=("estimate_geometry",))
    policy = ScriptedCodePolicy([
        _run_python("assert OBS_AFTER is not None\nprint('again')"),
        '{"status":"uncertain","confidence":0.3,"feedback":"evidence unclear"}',
    ])
    agent = VerifyCodeAgent(executor=FakeExecutor(), policy=policy, context=ctx, library=lib, max_turns=3)
    vtrace = agent.verify()
    assert vtrace.verdict.verdict == "uncertain"
    assert len(vtrace.turns) == 1


def test_verifier_evidence_failure_is_verifier_error() -> None:
    lib = FakeLibrary([_skill("estimate_geometry", "g")])
    ctx = VerifierContext(sub_goal="x", skills_used=("estimate_geometry",))
    policy = ScriptedCodePolicy([_run_python("assert OBS_AFTER is not None\nprint('again')")])
    agent = VerifyCodeAgent(
        executor=FailingEvidenceExecutor(),
        policy=policy,
        context=ctx,
        library=lib,
        max_turns=1,
    )
    vtrace = agent.verify()
    assert vtrace.verdict.verdict == "verifier_error"
    assert len(vtrace.turns) == 1


def test_executor_terminates_on_env_signal() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="action")])
    policy = ScriptedCodePolicy([
        _use_skill("grasp"),
        _run_python("close_gripper()"),
        _run_python("open_gripper()"),
    ])
    ex = FakeExecutor(terminate_when=lambda c: "open_gripper" in c)
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)
    trace = agent.run(task="pick and place")
    assert trace.success
    assert len(trace.turns) == 2
    assert trace.loaded_skill_ids == ("grasp",)


def test_executor_loads_skill_via_use_skill() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="action")])
    policy = ScriptedCodePolicy([
        _use_skill("grasp"),
        _run_python("close_gripper()"),
        _finish(),
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)
    trace = agent.run(task="grasp the cube")
    assert "grasp" in trace.loaded_skill_ids
    assert len(trace.turns) == 1  # 只记录 python 轮


def test_meta_turns_do_not_consume_action_budget() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="action")])
    policy = ScriptedCodePolicy([
        _use_skill("grasp"),
        _run_python("close_gripper()"),
        _finish(),
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=1)

    trace = agent.run(task="grasp the cube")

    assert not trace.success
    assert "grasp" in trace.loaded_skill_ids
    assert len(trace.turns) == 1
    assert [b.name for b in ex.blocks if b.name != "evidence_seed"] == ["turn_1"]


def test_finish_action_returns_control_without_verifier_gate() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="action")])
    policy = ScriptedCodePolicy([
        _use_skill("grasp"),
        _run_python("close_gripper()"),
        _finish("I cannot run more code, and the object is not lifted."),
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(
        executor=ex,
        policy=policy,
        library=lib,
        max_turns=2,
    )

    trace = agent.run(task="grasp the cube")

    assert trace.success
    assert len(trace.turns) == 1
    meta = trace.metadata or {}
    assert meta.get("act_status") == "finished"
    assert meta.get("final_review") is None
    assert not meta.get("unresolved")
    assert "turn_1" in [b.name for b in ex.blocks]
    assert not any(b.name == "verify_seed" for b in ex.blocks)


def test_executor_blocks_python_until_skill_loaded() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="action")])
    policy = ScriptedCodePolicy([
        _run_python("close_gripper()"),
        _use_skill("grasp"),
        _run_python("close_gripper()"),
        _finish(),
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)

    trace = agent.run(task="grasp the cube")

    assert "grasp" in trace.loaded_skill_ids
    assert len(trace.turns) == 1
    assert [b.name for b in ex.blocks if b.name != "evidence_seed"] == ["turn_2"]


def test_act_treats_verify_as_invalid_action() -> None:
    lib = FakeLibrary([_skill("segment_object", "segment objects", category="observation")])
    policy = RecordingPolicy([
        _use_skill("segment_object"),
        json.dumps({"tool": "verify", "args": {"scope": "checkpoint"}}),
        _finish("planner should inspect the refreshed scene"),
    ])
    agent = CodeAsPolicyAgent(executor=FakeExecutor(), policy=policy, library=lib)

    trace = agent.run(task="localize the can")

    assert trace.success
    assert len(trace.turns) == 0
    assert any(
        "Unknown tool" in str(msg.get("content", ""))
        for prompt in policy.prompts
        for msg in prompt
        if isinstance(msg, dict)
    )


def test_verifier_context_marks_executor_wording_untrusted() -> None:
    ctx = VerifierContext(
        sub_goal="Pick up the alphabet soup can.",
        scope="checkpoint",
        question="Does the saved box select the object named by the sub-goal?",
        expected="The evidence should identify exactly the object named by the sub-goal.",
        act_question="Does the green/red can pass and not the blue can?",
        act_expected="The green/red can should pass.",
        act_claim="I localized the target.",
    )
    rendered = ctx.render()

    assert "Framework-generated review question" in rendered
    assert "Executor-requested wording (untrusted" in rendered
    assert "green/red can" in rendered
    assert "Executor claim / hypothesis" in rendered


def test_sanitize_code_redacts_vlm_prompt_text() -> None:
    from robomex.verification.context import sanitize_code

    code = '''
q = "Find the alphabet soup can: it is the front-center red/orange and green/blue cylindrical can, not the blue can in the back."
reply = query_vlm(q, images=rgb)
short = "target_box"
'''
    cleaned = sanitize_code(code)

    assert "front-center" not in cleaned
    assert "red/orange" not in cleaned
    assert "<redacted natural-language prompt>" in cleaned
    assert "target_box" in cleaned


def test_verifier_no_code_pass_is_reprompted_inside_verifier() -> None:
    lib = FakeLibrary([_skill("segment_object", "segment objects", category="observation")])
    ctx = VerifierContext(
        sub_goal="Pick up the alphabet soup can.",
        scope="checkpoint",
        question="Does the saved box select the object named by the sub-goal?",
        resources={},
    )
    policy = ScriptedCodePolicy([
        '{"status":"pass","confidence":0.9,"feedback":"looks correct","evidence":{}}',
        _run_python("assert OBS_AFTER is not None\nprint('inspected OBS_AFTER')"),
        '{"status":"pass","confidence":0.9,"feedback":"visual evidence inspected","evidence":{}}',
    ])
    agent = VerifyCodeAgent(executor=FakeExecutor(), policy=policy, context=ctx, library=lib)

    trace = agent.verify()

    assert trace.verdict.verdict == "passed"
    assert trace.verdict.confidence == 0.9
    assert len(trace.turns) == 1
    assert "OBS_AFTER" in trace.turns[0].code


def test_verifier_no_code_uncertain_is_reprompted_inside_verifier() -> None:
    lib = FakeLibrary([_skill("segment_object", "segment objects", category="observation")])
    ctx = VerifierContext(
        sub_goal="Pick up the alphabet soup can.",
        scope="checkpoint",
        question="Does the saved box select the object named by the sub-goal?",
    )
    policy = ScriptedCodePolicy([
        '{"status":"uncertain","confidence":0.2,"feedback":"not enough visual evidence","evidence":{}}',
        _run_python("assert OBS_AFTER is not None\nprint('inspected OBS_AFTER')"),
        '{"status":"uncertain","confidence":0.2,"feedback":"visual evidence still unclear","evidence":{}}',
    ])
    agent = VerifyCodeAgent(executor=FakeExecutor(), policy=policy, context=ctx, library=lib)

    trace = agent.verify()

    assert trace.verdict.verdict == "uncertain"
    assert len(trace.turns) == 1
    assert "OBS_AFTER" in trace.turns[0].code


def test_verifier_no_code_fail_is_reprompted_inside_verifier() -> None:
    lib = FakeLibrary([_skill("segment_object", "segment objects", category="observation")])
    ctx = VerifierContext(
        sub_goal="Pick up the alphabet soup can.",
        scope="checkpoint",
        question="Does the saved box select the object named by the sub-goal?",
    )
    policy = ScriptedCodePolicy([
        '{"status":"fail","confidence":0.8,"feedback":"wrong object","evidence":{}}',
        _run_python("assert OBS_AFTER is not None\nprint('inspected OBS_AFTER')"),
        '{"status":"fail","confidence":0.8,"feedback":"visual evidence shows wrong object","evidence":{}}',
    ])
    agent = VerifyCodeAgent(executor=FakeExecutor(), policy=policy, context=ctx, library=lib)

    trace = agent.verify()

    assert trace.verdict.verdict == "failed"
    assert len(trace.turns) == 1


def test_verifier_judge_receives_verify_rubric_and_evidence_package() -> None:
    lib = FakeLibrary([_skill("grasp_object", "grasp objects", category="action")])
    ctx = VerifierContext(
        sub_goal="Pick up the alphabet soup can.",
        scope="checkpoint",
        question="After the grasp attempt and lift, is the object held?",
        expected="The object should move with the gripper.",
        resources={
            "grasp_object": VerifyResource(
                skill_id="grasp_object",
                rubric_text="Pass only if the target rises with the gripper.",
            )
        },
    )
    policy = RecordingPolicy([
        _run_python("assert OBS_AFTER is not None\nprint('vlm says held true')"),
        '{"status":"pass","confidence":0.9,"feedback":"rubric satisfied"}',
    ])
    agent = VerifyCodeAgent(executor=FakeExecutor(), policy=policy, context=ctx, library=lib)

    trace = agent.verify()

    assert trace.verdict.verdict == "passed"
    judge_prompt = policy.prompts[-1][1]["content"]
    assert "Pass only if the target rises with the gripper." in judge_prompt
    assert "vlm says held true" in judge_prompt
    assert "Evidence package produced by the Evidence Coder" in judge_prompt


def test_act_no_longer_applies_verifier_specific_code_gate() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="action")])
    policy = ScriptedCodePolicy([
        _use_skill("grasp"),
        _run_python("open('verifier_judge.py', 'w').write('pass')"),
        _run_python("close_gripper()"),
        _finish(),
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)

    trace = agent.run(task="grasp the cube")

    executed_codes = [b.code for b in ex.blocks if b.name != "evidence_seed"]
    assert any("verifier_judge.py" in c for c in executed_codes)
    assert any("close_gripper" in c for c in executed_codes)
    assert len(trace.turns) == 2


def test_human_event_lines_identify_roles_and_decisions() -> None:
    from robomex.core.events import _human_event_line

    assert (
        _human_event_line({
            "event": "agent_action",
            "agent_role": "act",
            "turn": 3,
            "action_turns": 1,
            "max_action_turns": 10,
            "action": "run_python",
            "payload_preview": "move arm",
        })
        == "[ACT] t03 a1/10 action=run_python: move arm"
    )
    terminal_line = _human_event_line({
        "event": "terminal_review",
        "agent_role": "act",
        "turn": 0,
        "should_stop": True,
    })
    assert terminal_line == "[ACT] t00 terminal accepted"


def _run_all() -> None:
    test_parse_json_action_variants()
    test_verifier_use_skill_then_python_then_verdict()
    test_verifier_unknown_skill_does_not_crash()
    test_verifier_evidence_then_judge_uncertain()
    test_verifier_evidence_failure_is_verifier_error()
    test_executor_terminates_on_env_signal()
    test_executor_loads_skill_via_use_skill()
    test_meta_turns_do_not_consume_action_budget()
    test_finish_action_returns_control_without_verifier_gate()
    test_executor_blocks_python_until_skill_loaded()
    test_act_treats_verify_as_invalid_action()
    test_verifier_context_marks_executor_wording_untrusted()
    test_sanitize_code_redacts_vlm_prompt_text()
    test_verifier_no_code_pass_is_reprompted_inside_verifier()
    test_verifier_no_code_uncertain_is_reprompted_inside_verifier()
    test_verifier_no_code_fail_is_reprompted_inside_verifier()
    test_verifier_judge_receives_verify_rubric_and_evidence_package()
    test_act_no_longer_applies_verifier_specific_code_gate()
    test_human_event_lines_identify_roles_and_decisions()
    test_executor_initial_prompt_includes_scene_image()
    test_executor_feedback_includes_observation_image()
    test_preview_content_strips_base64()
    print("all coding_agent offline tests passed")


def test_executor_initial_prompt_includes_scene_image() -> None:
    """Act Agent initial prompt returns multimodal content when scene_image_path is set."""
    import tempfile
    import numpy as np
    from robomex.perception.render import save_rgb

    lib = FakeLibrary([_skill("grasp", "grasp objects", category="action")])
    policy = ScriptedCodePolicy([_finish()])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        save_rgb(f.name, np.zeros((4, 4, 3), dtype=np.uint8))
        agent._task = "test task"
        agent._observation_summary = ""
        agent._primary_skill_id = None
        agent._feedback = ""
        agent._scene_image_path = f.name
        msg = agent._initial_user_message()
        assert isinstance(msg, list), "Expected multimodal content list"
        assert msg[0]["type"] == "text"
        assert "occludes" in msg[0]["text"]
        assert msg[1]["type"] == "image_url"
        assert msg[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_executor_feedback_includes_observation_image() -> None:
    """Feedback after execution includes observation image when available."""
    import tempfile
    import numpy as np
    from pathlib import Path

    lib = FakeLibrary([_skill("grasp", "grasp objects", category="action")])
    policy = ScriptedCodePolicy([_use_skill("grasp"), _run_python("print('hi')"), _finish()])

    rgb = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
    obs_with_rgb = {"agentview": {"images": {"rgb": rgb}}}

    class FakeExecWithObs:
        def __init__(self):
            self.blocks = []
        def run_block(self, block):
            self.blocks.append(block)
            return BlockExecutionResult(
                block=block, ok=True, status=ActionBlockStatus.SUCCEEDED,
                stdout="done", stderr="", reward=0.0, terminated=False, truncated=False,
                observation=obs_with_rgb, info={"sandbox_rc": 0},
            )

    with tempfile.TemporaryDirectory() as td:
        ex = FakeExecWithObs()
        agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)
        trace = agent.run(task="test", video_dir=td)

        feedback = agent._feedback_message(BlockExecutionResult(
            block=SemanticActionBlock(name="t", intent="t", code="x"),
            ok=True, status=ActionBlockStatus.SUCCEEDED,
            stdout="out", stderr="", reward=0.0, terminated=False, truncated=False,
            observation=obs_with_rgb, info={},
        ))
        assert isinstance(feedback, list), "Expected multimodal feedback"
        assert feedback[0]["type"] == "text"
        assert feedback[1]["type"] == "image_url"


def test_preview_content_strips_base64() -> None:
    """_preview_content must not include raw base64 data."""
    from robomex.core.coder.agent import _preview_content

    long_b64 = "data:image/png;base64," + "A" * 5000
    content = [
        {"type": "text", "text": "hello world"},
        {"type": "image_url", "image_url": {"url": long_b64}},
    ]
    result = _preview_content(content)
    assert "AAAA" not in result
    assert "image(s)" in result
    assert "hello world" in result


if __name__ == "__main__":
    _run_all()
