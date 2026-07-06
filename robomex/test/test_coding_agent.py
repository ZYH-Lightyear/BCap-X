"""Act-only CodingAgent 离线测试(无 env / LLM / 网络)。"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np

from robomex.agents import CodeAsPolicyAgent
from robomex.agents.subagents import (
    CodingAgentSubAgent,
    PolicyBoundBlockExecutor,
    SubAgentExecutionPolicy,
    SubAgentRegistry,
    SubAgentRequest,
    SubAgentResult,
)
from robomex.core.coder import ScriptedCodePolicy
from robomex.core.context import (
    AttemptRecord,
    Diagnosis,
    EvidencePacket,
    LocalVerdict,
    PrimitiveTrace,
    StateFact,
    StatePatch,
    WorkspaceArtifact,
    WorldState,
)
from robomex.core.sandbox import ActionBlockStatus, BlockExecutionResult, SemanticActionBlock
from robomex.skills import Skill


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


class NativeTurnPolicy:
    def __init__(self, turns) -> None:
        self._turns = list(turns)
        self.prompts: list[list[dict]] = []
        self._index = 0

    def complete_turn(self, prompt: list[dict]):
        self.prompts.append(prompt)
        if self._index >= len(self._turns):
            from robomex.core.coder import ModelTurn

            return ModelTurn(raw="", text="done")
        turn = self._turns[self._index]
        self._index += 1
        return turn

    def complete(self, prompt: list[dict]) -> str:
        raise AssertionError("NativeTurnPolicy should be consumed through complete_turn")


def _skill(skill_id: str, desc: str, category: str = "perception") -> Skill:
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


def _finish_result(claim: str, result: dict) -> str:
    return json.dumps({"tool": "finish", "args": {"claim": claim, "result": result}})


def _call_subagent(
    task: str,
    inputs: dict | None = None,
    task_id: str | None = None,
) -> str:
    args = {"task": task, "inputs": inputs or {}}
    if task_id:
        args["task_id"] = task_id
    return json.dumps({"tool": "call_subagent", "args": args})


def _finish_with_state_patch(claim: str, key: str, value, confidence: float = 1.0) -> str:
    return json.dumps({
        "tool": "finish",
        "args": {
            "claim": claim,
            "result": {"checked": True},
            "state_patch": {
                "upsert_facts": [
                    {"key": key, "value": value, "confidence": confidence, "provenance": {"test": True}}
                ]
            },
        },
    })


def test_world_state_merge_preserves_higher_confidence_fact() -> None:
    world = WorldState()
    world.merge_patch(StatePatch(upsert_facts=(StateFact("robot.held_object", "can", confidence=0.9),)))
    world.merge_patch(StatePatch(upsert_facts=(StateFact("robot.held_object", None, confidence=0.2),)))

    assert world.facts["robot.held_object"].value == "can"
    assert world.history[-1]["ignored_low_confidence"] == ["robot.held_object"]


def test_world_state_invalidated_facts_remove_promoted_beliefs() -> None:
    world = WorldState()
    world.merge_patch(StatePatch(upsert_facts=(StateFact("robot.held_object", "can", confidence=0.9),)))
    world.merge_patch(StatePatch(invalidated_facts=("robot.held_object",)), source="diagnosis")

    assert "robot.held_object" not in world.facts
    assert world.history[-1]["invalidated"] == ["robot.held_object"]


def test_world_state_invalidates_observation_scoped_grounding_after_motion() -> None:
    world = WorldState()
    world.merge_patch(StatePatch(upsert_facts=(
        StateFact(
            "target_object.generic_item",
            {"bbox_px": [1, 2, 3, 4], "world_centroid": [0.1, 0.2, 0.3]},
            confidence=0.8,
        ),
        StateFact("held_object.generic_item", {"state": "held_in_gripper"}, confidence=0.9),
    )))

    assert world.facts["target_object.generic_item"].validity["scope"] == "observation"

    invalidated = world.invalidate_observation_scoped_facts(source="test:motion")

    assert invalidated == ("target_object.generic_item",)
    assert "target_object.generic_item" not in world.facts
    assert "held_object.generic_item" in world.facts


def test_evidence_packet_compacts_large_payloads_for_trace_metadata() -> None:
    packet = EvidencePacket.from_any(
        {
            "claim": "localized object candidates",
            "confidence": 0.73,
            "evidence": {
                "points": np.ones((1000, 3), dtype=np.float32),
                "long_log": "x" * 2000,
                "candidate": {"center": [0.1, 0.2, 0.3]},
            },
            "verdict": {"status": "uncertain", "reason": "two similar cans"},
        },
        default_source="test_subagent",
        default_turn="subagent:0",
    )

    payload = packet.to_json_dict()

    assert payload["schema"] == "robomex.evidence_packet.v1"
    assert payload["packet_id"].startswith("test_subagent:")
    assert payload["evidence"]["points"]["type"] == "ndarray"
    assert payload["evidence"]["points"]["shape"] == [1000, 3]
    assert payload["evidence"]["long_log"]["type"] == "str"
    assert payload["verdict"]["status"] == "uncertain"


def test_parse_json_action_variants() -> None:
    from robomex.core.coder import parse_action, parse_model_turn

    action = parse_action(_run_python("print('x')"))
    assert action.kind == "run_python"
    assert action.args["code"] == "print('x')"

    turn = parse_model_turn(_run_python("print('x')"))
    assert not turn.is_error
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "run_python"
    assert turn.tool_calls[0].args["code"] == "print('x')"

    invalid = parse_action("not json")
    assert invalid.kind == "invalid"

    subagent = parse_action(_call_subagent("localize target"))
    assert subagent.kind == "call_subagent"
    assert subagent.args["task"] == "localize target"

    profiled = parse_action(
        json.dumps({"tool": "call_subagent", "args": {"task": "find grasp", "profile": "affordance"}}),
    )
    assert profiled.kind == "invalid"
    assert "no longer accepts args.profile" in (profiled.error or "")

    old_named = parse_action(
        json.dumps({"tool": "call_subagent", "args": {"name": "grounding_object", "task": "find can"}}),
    )
    assert old_named.kind == "invalid"
    assert "no longer accepts args.name" in (old_named.error or "")

    phased = parse_action(
        json.dumps({"tool": "call_subagent", "args": {"task": "find can", "phase": "grounding"}}),
    )
    assert phased.kind == "invalid"
    assert "does not accept args.phase" in (phased.error or "")
    assert invalid.error

    invalid_turn = parse_model_turn("not json")
    assert invalid_turn.is_error
    assert not invalid_turn.tool_calls

    prose_done = parse_model_turn("Done.")
    assert prose_done.is_error
    assert not prose_done.tool_calls


def test_parse_prose_wrapped_json_action_with_json_repair() -> None:
    if importlib.util.find_spec("json_repair") is None:
        return

    from robomex.core.coder import parse_model_turn

    raw = (
        "Found an IK-feasible grasp candidate. Let me execute the grasp.\n\n"
        + _run_python("print('grasp')", intent="execute grasp")
    )
    turn = parse_model_turn(raw)

    assert not turn.is_error
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "run_python"
    assert turn.tool_calls[0].args["code"] == "print('grasp')"


def test_parse_run_python_with_extra_trailing_brace_does_not_repair_code_dict_literal() -> None:
    from robomex.core.coder import parse_model_turn

    raw = (
        '{"tool":"run_python","args":{"code":"'
        "check = 'ok'\\n"
        "print({'executed_strategy':'pca_side_body','vlm_check':check})"
        '"}}}'
    )

    turn = parse_model_turn(raw)

    assert not turn.is_error
    assert len(turn.tool_calls) == 1
    args = turn.tool_calls[0].args
    assert set(args) == {"code"}
    assert "final_robot_pos" not in args
    assert args["code"].endswith("'vlm_check':check})")


def test_native_model_turn_final_text_is_reprompted_not_terminal() -> None:
    from robomex.core.coder import ModelTurn, ToolCall

    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = NativeTurnPolicy([
        ModelTurn(raw="", text="sub-goal attempt complete"),
        ModelTurn(raw=_finish("done"), tool_calls=(ToolCall(name="finish", args={"claim": "done", "raw": _finish("done")}),)),
    ])
    agent = CodeAsPolicyAgent(executor=FakeExecutor(), policy=policy, library=lib, max_turns=6)

    trace = agent.run(task="grasp the cube")

    assert trace.success
    assert trace.turns == ()
    assert trace.metadata["terminal_raw"] == _finish("done")
    assert len(policy.prompts) == 2


def test_executor_terminates_on_env_signal() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
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
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = ScriptedCodePolicy([
        _use_skill("grasp"),
        _run_python("close_gripper()"),
        _finish(),
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)
    trace = agent.run(task="grasp the cube")
    assert "grasp" in trace.loaded_skill_ids
    assert len(trace.turns) == 1


def test_executor_blocks_open_gripper_after_goto_home_in_same_block() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = RecordingPolicy([
        _use_skill("grasp"),
        _run_python("goto_home_joint_position()\nopen_gripper()"),
        _run_python("goto_home_joint_position()\nobs = get_observation()"),
        _finish(),
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)

    trace = agent.run(task="reobserve without dropping held object")

    executed = [b.code for b in ex.blocks if b.name != "evidence_seed"]
    assert executed == ["goto_home_joint_position()\nobs = get_observation()"]
    assert len(trace.turns) == 1
    assert any(
        "must preserve any held object" in str(message.get("content", ""))
        for prompt in policy.prompts
        for message in prompt
    )


def test_meta_turns_do_not_consume_action_budget() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
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


def test_finish_action_returns_control_without_review_gate() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
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
    assert not meta.get("unresolved")
    assert "turn_1" in [b.name for b in ex.blocks]


def test_executor_allows_read_only_python_before_skill_loaded() -> None:
    lib = FakeLibrary([_skill("segment_object", "segment objects", category="perception")])
    policy = ScriptedCodePolicy([
        _run_python("obs = get_observation()\nEVIDENCE['checked'] = True"),
        _finish(),
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)

    trace = agent.run(task="inspect the scene")

    assert trace.success
    assert trace.loaded_skill_ids == ()
    assert len(trace.turns) == 1
    assert [b.name for b in ex.blocks if b.name != "evidence_seed"] == ["turn_0"]


def test_executor_blocks_state_changing_python_until_skill_loaded() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = RecordingPolicy([
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
    assert any(
        "would change robot/environment state before any RoboMEx skill was loaded" in str(msg.get("content", ""))
        for prompt in policy.prompts
        for msg in prompt
        if isinstance(msg, dict)
    )


def test_executor_does_not_special_case_sidecar_imports() -> None:
    lib = FakeLibrary([_skill("segment_object", "segment objects", category="perception")])
    policy = RecordingPolicy([
        _use_skill("segment_object"),
        _run_python("import importlib.util\nprint('ordinary sidecar loading is model-controlled')"),
        _finish(),
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)

    trace = agent.run(task="ground the can")

    assert trace.success
    executed = [b.code for b in ex.blocks if b.name != "evidence_seed"]
    assert executed == ["import importlib.util\nprint('ordinary sidecar loading is model-controlled')"]


def test_act_treats_verify_as_invalid_action() -> None:
    lib = FakeLibrary([_skill("segment_object", "segment objects", category="perception")])
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
        "not available" in str(msg.get("content", ""))
        for prompt in policy.prompts
        for msg in prompt
        if isinstance(msg, dict)
    )


def test_act_can_call_subagent_without_consuming_action_budget() -> None:
    class FakeSubAgent:
        name = "verifier_subagent"

        def __init__(self) -> None:
            self.requests: list[SubAgentRequest] = []

        def run(self, request: SubAgentRequest) -> SubAgentResult:
            self.requests.append(request)
            return SubAgentResult(
                name=self.name,
                ok=True,
                result={"verdict": {"status": "uncertain", "reason": "object is not visibly held"}},
                turns=1,
            )

    subagent = FakeSubAgent()
    registry = SubAgentRegistry(default_agent=subagent)
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = ScriptedCodePolicy([
        _use_skill("grasp"),
        _call_subagent("verify whether the can is visibly held by the gripper", {"object_name": "can"}),
        _run_python("close_gripper()"),
        _finish(),
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(
        executor=ex,
        policy=policy,
        library=lib,
        max_turns=1,
        subagents=registry,
    )

    trace = agent.run(task="grasp the can")

    assert not trace.success
    assert len(subagent.requests) == 1
    assert subagent.requests[0].inputs["object_name"] == "can"
    assert "verify whether" in subagent.requests[0].task
    assert [b.name for b in ex.blocks if b.name != "evidence_seed"] == ["turn_2"]


def test_act_finish_state_patch_is_not_promoted_to_trace_context() -> None:
    lib = FakeLibrary([_skill("observe", "observe state", category="perception")])
    policy = ScriptedCodePolicy([
        _use_skill("observe"),
        _finish_with_state_patch("can is held", "robot.held_object", "alphabet_soup_can", 0.91),
    ])
    agent = CodeAsPolicyAgent(executor=FakeExecutor(), policy=policy, library=lib)

    trace = agent.run(task="check held object")

    assert trace.success
    terminal = trace.metadata["terminal_result"]
    assert terminal["claim"] == "can is held"
    assert "state_patches" not in trace.metadata


def test_subagent_state_patch_is_ignored_by_act_context() -> None:
    class FakeSubAgent:
        name = "verifier_subagent"

        def __init__(self) -> None:
            self.requests: list[SubAgentRequest] = []

        def run(self, request: SubAgentRequest) -> SubAgentResult:
            self.requests.append(request)
            return SubAgentResult(
                name=self.name,
                ok=True,
                claim="object is visible on table",
                result={"state": "on_table"},
                state_patch=StatePatch(upsert_facts=(
                    StateFact("objects.alphabet_soup_can.state", "on_table", confidence=0.8),
                )),
                turns=1,
                task_id=request.task_id,
            )

    subagent = FakeSubAgent()
    lib = FakeLibrary([_skill("observe", "observe state", category="perception")])
    policy = ScriptedCodePolicy([
        _use_skill("observe"),
        _call_subagent("verify whether the alphabet soup can is on the table", {"object": "alphabet_soup_can"}, task_id="track-can"),
        _finish(),
    ])
    agent = CodeAsPolicyAgent(
        executor=FakeExecutor(),
        policy=policy,
        library=lib,
        subagents=SubAgentRegistry(default_agent=subagent),
    )

    trace = agent.run(task="check can")

    assert trace.success
    assert subagent.requests[0].task_id == "track-can"
    assert "world_state" not in subagent.requests[0].context
    assert subagent.requests[0].context["act_task"] == "check can"
    call = trace.metadata["subagent_calls"][0]
    assert call["claim"] == "object is visible on table"
    assert call["state_patch"]["upsert_facts"][0]["key"] == "objects.alphabet_soup_can.state"
    assert "state_patches" not in trace.metadata


def test_subagent_structured_outputs_are_returned_to_act_trace() -> None:
    class FakeSubAgent:
        name = "verifier_subagent"

        def run(self, request: SubAgentRequest) -> SubAgentResult:
            return SubAgentResult(
                name=self.name,
                ok=True,
                result_type="verification",
                claim="side grasp candidate is visually aligned",
                result={"target": "bowl"},
                local_verdict=LocalVerdict("candidate_alignment", "pass", confidence=0.8, reason="clear side rim"),
                artifact_refs=(WorkspaceArtifact("verifier_overlay", path="overlay.png", producer=self.name),),
                primitive_traces=(
                    PrimitiveTrace(
                        trace_id="verifier:t0:block",
                        primitive_name="verify_candidate",
                        status="succeeded",
                        producer=self.name,
                    ),
                ),
                attempt_records=(
                    AttemptRecord(
                        attempt_id="grasp:bowl:1",
                        object_key="bowl",
                        strategy="top_down_rim",
                        outcome="candidate_selected",
                    ),
                ),
                diagnoses=(
                    Diagnosis(
                        diagnosis_id="diag:placement-offset",
                        failed_primitive="release_at",
                        failure_type="center_offset_missing",
                        next_route="estimate_held_object_frame",
                    ),
                ),
                turns=1,
                task_id=request.task_id,
            )

    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = ScriptedCodePolicy([
        _use_skill("grasp"),
        _call_subagent(
            "verify whether the proposed bowl grasp is visually aligned with the rim",
            {"object": "bowl"},
            task_id="bowl-grasp-check",
        ),
        _finish(),
    ])
    agent = CodeAsPolicyAgent(
        executor=FakeExecutor(),
        policy=policy,
        library=lib,
        subagents=SubAgentRegistry(default_agent=FakeSubAgent()),
    )

    trace = agent.run(task="pick the bowl")

    assert trace.success
    assert trace.metadata["subagent_calls"][0]["result_type"] == "verification"
    packet = trace.metadata["subagent_calls"][0]["evidence_packet"]
    assert packet["claim"] == "side grasp candidate is visually aligned"
    assert packet["verdict"]["status"] == "pass"
    assert trace.metadata["evidence_packets"][0]["packet"]["claim"] == "side grasp candidate is visually aligned"
    assert trace.metadata["evidence_timeline"][0]["packet"]["claim"] == "side grasp candidate is visually aligned"
    assert trace.metadata["evidence_timeline"][0]["source"] == "subagent:0"
    assert trace.metadata["artifact_refs"][0]["artifact_id"] == "verifier_overlay"
    assert trace.metadata["primitive_traces"][0]["trace_id"] == "verifier:t0:block"
    assert trace.metadata["attempt_records"][0]["attempt_id"] == "grasp:bowl:1"
    assert trace.metadata["diagnoses"][0]["diagnosis_id"] == "diag:placement-offset"
    assert trace.metadata["local_verdicts"][0]["status"] == "pass"


def test_act_auto_records_physical_attempt_and_exhaustion_diagnosis() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = ScriptedCodePolicy([
        _use_skill("grasp"),
        _run_python("close_gripper()", intent="try closing on target"),
    ])
    agent = CodeAsPolicyAgent(executor=FakeExecutor(), policy=policy, library=lib, max_turns=1)

    trace = agent.run(task="pick target object")

    assert not trace.success
    attempts = trace.metadata["attempt_records"]
    assert attempts[0]["strategy"] == "try closing on target"
    assert attempts[0]["outcome"] == "executed"
    assert attempts[0]["pose_or_target"]["state_changing_calls"] == ["close_gripper"]
    diagnoses = trace.metadata["diagnoses"]
    assert diagnoses[0]["failure_type"] == "action_budget_exhausted"


def test_act_persists_subagent_call_manifests(tmp_path) -> None:
    class FakeSubAgent:
        name = "verifier_subagent"

        def run(self, request: SubAgentRequest) -> SubAgentResult:
            return SubAgentResult(
                name=self.name,
                ok=True,
                result={"verdict": {"status": "pass", "reason": "can is visibly held"}},
                turns=1,
                artifacts_dir=request.artifacts_dir,
            )

    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = ScriptedCodePolicy([
        _use_skill("grasp"),
        _call_subagent("verify whether the can is visibly held", {"object_name": "can"}),
        _finish(),
    ])
    agent = CodeAsPolicyAgent(
        executor=FakeExecutor(),
        policy=policy,
        library=lib,
        subagents=SubAgentRegistry(default_agent=FakeSubAgent()),
    )

    trace = agent.run(task="grasp the can", video_dir=tmp_path)

    assert trace.success
    calls = list((tmp_path / "subagents").glob("00_verify_whether_the_can_is_visibly_held/subagent_call.json"))
    assert calls
    call_payload = json.loads(calls[0].read_text(encoding="utf-8"))
    assert call_payload["schema"] == "robomex.subagent_call.v1"
    assert call_payload["call_index"] == 0
    assert call_payload["request"]["inputs"]["object_name"] == "can"
    assert call_payload["result"]["verdict"]["status"] == "pass"
    result_payload = json.loads((calls[0].parent / "result.json").read_text(encoding="utf-8"))
    assert result_payload["schema"] == "robomex.subagent_result.v1"
    assert result_payload["request"]["task"] == "verify whether the can is visibly held"


def test_act_persists_failed_subagent_call_manifest(tmp_path) -> None:
    class FailingSubAgent:
        name = "verifier_subagent"

        def run(self, request: SubAgentRequest) -> SubAgentResult:
            raise RuntimeError("subagent backend failed")

    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = ScriptedCodePolicy([
        _use_skill("grasp"),
        _call_subagent("verify whether the can is visibly held"),
        _finish(),
    ])
    agent = CodeAsPolicyAgent(
        executor=FakeExecutor(),
        policy=policy,
        library=lib,
        subagents=SubAgentRegistry(default_agent=FailingSubAgent()),
    )

    trace = agent.run(task="grasp the can", video_dir=tmp_path)

    assert trace.success
    call_path = tmp_path / "subagents" / "00_verify_whether_the_can_is_visibly_held" / "subagent_call.json"
    call_payload = json.loads(call_path.read_text(encoding="utf-8"))
    assert call_payload["ok"] is False
    assert "subagent backend failed" in call_payload["error"]
    result_payload = json.loads((call_path.parent / "result.json").read_text(encoding="utf-8"))
    assert result_payload["result"]["ok"] is False
    assert "subagent backend failed" in result_payload["result"]["error"]


def test_session_writes_subagent_runtime_manifest(tmp_path) -> None:
    from robomex import RoboMExAgent, RoboMExConfig
    from robomex.agents import ScriptedPlannerPolicy
    from robomex.agents.subagents import SubAgentExecutionPolicy
    from robomex.skills import SkillLibrary

    library = SkillLibrary(tmp_path / "library")
    library.admit(_skill("pick_object", "pick objects", category="task"), source="test")
    policy = ScriptedCodePolicy([
        _use_skill("pick_object"),
        _finish("done"),
    ])
    agent = RoboMExAgent(
        RoboMExConfig(
            library=library,
            planner_policy=ScriptedPlannerPolicy(
                "Goal: Pick the can\nPostcondition: The can is held."
            ),
            code_policy=policy,
            executor=FakeExecutor(),
            artifacts_dir=str(tmp_path / "episode"),
            subagent_max_turns=7,
            subagent_execution_policy=SubAgentExecutionPolicy(
                denied_calls=frozenset({"custom_motion"}),
            ),
            code_policy_kind="json_action_adapter",
        )
    )

    result = agent.run("Pick the can")

    assert result.success
    manifest = json.loads((tmp_path / "episode" / "subagents.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "robomex.subagents.v1"
    assert manifest["subagent_runtime_enabled"] is True
    assert manifest["subagent_runtime"] == "coding_agent"
    assert manifest["subagent_max_turns"] == 7
    assert manifest["execution_policy"]["denied_calls"] == ["custom_motion"]


def test_session_keeps_structured_trace_without_promoting_world_facts(tmp_path) -> None:
    from robomex import RoboMExAgent, RoboMExConfig
    from robomex.agents import ScriptedPlannerPolicy
    from robomex.skills import SkillLibrary

    library = SkillLibrary(tmp_path / "library")
    library.admit(_skill("pick_object", "pick objects", category="task"), source="test")
    finish = json.dumps({
        "tool": "finish",
        "args": {
            "claim": "grasp candidate failed; retry with a different rim point",
            "result": {
                "state_patch": {
                    "upsert_facts": [
                        {
                            "key": "robot.held_object",
                            "value": "bowl",
                            "confidence": 0.9,
                        },
                        {
                            "key": "robot.last_grasp_strategy",
                            "value": "rim_top_down",
                            "confidence": 0.8,
                        },
                    ]
                },
                "primitive_traces": [
                    {
                        "trace_id": "act:t0:grasp",
                        "primitive_name": "grasp",
                        "status": "failed",
                    }
                ],
                "attempt_records": [
                    {
                        "attempt_id": "attempt:bowl:rim:1",
                        "object_key": "bowl",
                        "strategy": "rim_top_down",
                        "outcome": "failed",
                        "failure_reason": "slipped during lift",
                        "invalidated_facts": ["robot.held_object"],
                    }
                ],
                "diagnoses": [
                    {
                        "diagnosis_id": "diag:bowl:slip",
                        "failed_primitive": "grasp",
                        "failure_type": "grasp_slip",
                        "next_route": "affordance_replan",
                    }
                ],
            },
        },
    })
    agent = RoboMExAgent(
        RoboMExConfig(
            library=library,
            planner_policy=ScriptedPlannerPolicy("Goal: Pick the bowl\nPostcondition: The bowl is held."),
            code_policy=ScriptedCodePolicy([_use_skill("pick_object"), finish]),
            executor=FakeExecutor(),
            artifacts_dir=str(tmp_path / "episode"),
            enable_default_subagents=False,
        )
    )

    result = agent.run("Pick the bowl")

    assert result.success
    root = tmp_path / "episode"
    assert not (root / "world_state.json").exists()
    attempts = json.loads((root / "attempt_history.json").read_text(encoding="utf-8"))
    assert attempts["records"][0]["attempt_id"] == "attempt:bowl:rim:1"
    traces = json.loads((root / "trace_store.json").read_text(encoding="utf-8"))
    assert "act:t0:grasp" in traces["traces"]
    diagnoses = json.loads((root / "diagnoses.json").read_text(encoding="utf-8"))
    assert diagnoses["diagnoses"][0]["failure_type"] == "grasp_slip"
    sg_meta = json.loads((root / "subgoal_00" / "meta.json").read_text(encoding="utf-8"))
    assert sg_meta["primitive_trace_count"] == 1
    assert sg_meta["evidence_timeline"][0]["packet"]["claim"] == (
        "grasp candidate failed; retry with a different rim point"
    )
    sg_timeline = json.loads((root / "subgoal_00" / "evidence_timeline.json").read_text(encoding="utf-8"))
    assert sg_timeline["schema"] == "robomex.evidence_timeline.v1"
    assert sg_timeline["records"][0]["source"] == "act:finish"
    episode_timeline = json.loads((root / "evidence_timeline.json").read_text(encoding="utf-8"))
    assert episode_timeline["records"][0]["subgoal_goal"] == "Pick the bowl"
    assert (root / "evidence_timeline.md").exists()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["subgoals"][0]["evidence_timeline"][0]["packet"]["claim"] == (
        "grasp candidate failed; retry with a different rim point"
    )
    report = (root / "report.html").read_text(encoding="utf-8")
    assert "RoboMEx Debug Report" in report
    assert "grasp candidate failed" in report
    candidates = json.loads((root / "skill_evolution_candidates.json").read_text(encoding="utf-8"))
    assert candidates["schema"] == "robomex.skill_evolution_candidates.v1"
    assert candidates["subgoals"][0]["goal"] == "Pick the bowl"
    assert candidates["subgoals"][0]["evidence_timeline"][0]["source"] == "act:finish"


def test_missing_subagent_runtime_returns_recoverable_feedback() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = RecordingPolicy([
        _use_skill("grasp"),
        _call_subagent("do something"),
        _finish(),
    ])
    agent = CodeAsPolicyAgent(
        executor=FakeExecutor(),
        policy=policy,
        library=lib,
        subagents=SubAgentRegistry(),
    )

    trace = agent.run(task="test delegation")

    assert trace.success
    assert any(
        "No SubAgent registry" in str(msg.get("content", ""))
        for prompt in policy.prompts
        for msg in prompt
        if isinstance(msg, dict)
    )


def test_subagent_guard_blocks_motion_calls() -> None:
    ex = FakeExecutor()
    guarded = PolicyBoundBlockExecutor(ex)
    block = SemanticActionBlock(
        name="sub",
        intent="bad motion",
        code="close_gripper()\ngoto_pose([0, 0, 0])",
    )

    result = guarded.run_block(block)

    assert not result.ok
    assert result.status is ActionBlockStatus.SKIPPED
    assert "close_gripper" in result.stderr
    assert "goto_pose" in result.stderr
    assert ex.blocks == []


def test_subagent_guard_allows_ordinary_sidecar_code() -> None:
    ex = FakeExecutor()
    guarded = PolicyBoundBlockExecutor(ex)
    block = SemanticActionBlock(
        name="sub",
        intent="sidecar code",
        code="import importlib.util\nprint('load sidecar explicitly')",
    )

    result = guarded.run_block(block)

    assert result.ok
    assert len(ex.blocks) == 1


def test_subagent_execution_policy_can_customize_denied_calls() -> None:
    ex = FakeExecutor()
    guarded = PolicyBoundBlockExecutor(
        ex,
        SubAgentExecutionPolicy(denied_calls=frozenset({"custom_motion"})),
    )

    blocked = guarded.run_block(SemanticActionBlock(name="sub", intent="custom", code="custom_motion()"))
    allowed = guarded.run_block(SemanticActionBlock(name="sub", intent="motion not denied here", code="goto_pose([0, 0, 0])"))

    assert not blocked.ok
    assert "custom_motion" in blocked.stderr
    assert allowed.ok
    assert len(ex.blocks) == 1


def test_coding_subagent_returns_finish_result_json() -> None:
    lib = FakeLibrary([_skill("segment_object", "segment objects", category="perception")])
    policy = ScriptedCodePolicy([
        _use_skill("segment_object"),
        _run_python("EVIDENCE['grounding.can.points'] = [[0, 0, 0]]\nprint('grounded')"),
        _finish_result("grounded can", {"ok": True, "target": "can", "points_key": "grounding.can.points"}),
    ])
    agent = CodingAgentSubAgent(
        executor=FakeExecutor(),
        policy=policy,
        library=lib,
        max_turns=2,
    )

    result = agent.run(SubAgentRequest(task="ground can"))

    assert result.ok
    assert result.result["target"] == "can"
    assert result.result["points_key"] == "grounding.can.points"
    assert result.loaded_skill_ids == ("segment_object",)


def test_act_finish_with_preface_preserves_structured_result() -> None:
    lib = FakeLibrary([_skill("observe", "observe state", category="perception")])
    finish = _finish_result(
        "held state checked",
        {"confidence": 0.82, "evidence": {"held": True}},
    )
    policy = ScriptedCodePolicy([f"The state is clear now.\n\n{finish}"])
    agent = CodeAsPolicyAgent(executor=FakeExecutor(), policy=policy, library=lib)

    trace = agent.run(task="check held state")

    assert trace.success
    assert trace.metadata["terminal_result"]["claim"] == "held state checked"
    assert trace.metadata["terminal_result"]["confidence"] == 0.82
    assert trace.metadata["terminal_result"]["evidence"]["held"] is True


def test_subagent_finish_with_fenced_json_preserves_evidence_packet() -> None:
    lib = FakeLibrary([_skill("segment_object", "segment objects", category="perception")])
    finish = _finish_result(
        "grounded can",
        {
            "confidence": 0.9,
            "evidence": {"bbox": [1, 2, 3, 4]},
            "recommended_next": "use bbox for affordance planning",
        },
    )
    policy = ScriptedCodePolicy([f"```json\n{finish}\n```"])
    agent = CodingAgentSubAgent(executor=FakeExecutor(), policy=policy, library=lib)

    result = agent.run(SubAgentRequest(task="ground can"))

    assert result.ok
    assert result.claim == "grounded can"
    assert result.evidence_packet.confidence == 0.9
    assert result.evidence_packet.evidence["bbox"] == [1, 2, 3, 4]
    assert result.recommended_next == "use bbox for affordance planning"


def test_verifier_subagent_does_not_apply_affordance_candidate_policy() -> None:
    lib = FakeLibrary([_skill("grasp_open_bowl", "propose bowl grasps", category="affordance")])
    finish = _finish_result(
        "computed grasp affordance",
        {
            "confidence": 0.8,
            "evidence": {"selected_candidate": {"pos": [0.1, 0.2, 0.3]}},
            "artifacts": ["outputs/run/subagents/affordance_dir"],
        },
    )
    agent = CodingAgentSubAgent(executor=FakeExecutor(), policy=ScriptedCodePolicy([finish]), library=lib)

    result = agent.run(SubAgentRequest(task="verify whether the proposed grasp appears aligned"))

    assert result.ok
    assert not any("review image artifact" in item for item in result.uncertainty)


def test_coding_subagent_writes_python_turn_artifacts(tmp_path) -> None:
    lib = FakeLibrary([_skill("segment_object", "segment objects", category="perception")])
    policy = ScriptedCodePolicy([
        _run_python("print('subagent visible code')"),
        _finish_result("done", {"ok": True}),
    ])
    agent = CodingAgentSubAgent(
        executor=FakeExecutor(),
        policy=policy,
        library=lib,
        max_turns=2,
    )

    result = agent.run(
        SubAgentRequest(
            task="ground can",
            artifacts_dir=str(tmp_path),
        )
    )

    assert result.ok
    assert (tmp_path / "turn_00.py").read_text(encoding="utf-8") == "print('subagent visible code')"
    out = (tmp_path / "turn_00.out.txt").read_text(encoding="utf-8")
    assert "## stdout" in out
    assert "ran turn_0" in out


def test_default_subagent_runtime_uses_eight_turn_budget_and_bounded_prompt() -> None:
    lib = FakeLibrary([_skill("segment_object", "segment objects", category="perception")])
    policy = RecordingPolicy([_finish_result("done", {"ok": True})])
    agent = CodingAgentSubAgent(
        executor=FakeExecutor(),
        policy=policy,
        library=lib,
    )

    assert agent.max_turns == 8

    result = agent.run(SubAgentRequest(task="verify whether the can is visible"))

    assert result.ok
    assert policy.prompts
    prompt_text = "\n".join(
        str(msg.get("content", ""))
        for prompt in policy.prompts
        for msg in prompt
        if isinstance(msg, dict)
    )
    assert "Do not take over Act's role" in prompt_text
    assert "do not localize a new target for execution" in prompt_text
    assert "Treat request inputs as claims" in prompt_text
    assert "sidecar" in prompt_text


def test_act_prompt_and_loaded_skill_content_explain_sidecar_paths() -> None:
    from robomex.core.coder import build_skill_llm_content
    from robomex.prompts import LIBERO_ACT_SYSTEM_PROMPT

    skill_text = build_skill_llm_content("/tmp/skill", "Body")

    assert "base directory" in LIBERO_ACT_SYSTEM_PROMPT
    assert "usage patterns named in the skill" in LIBERO_ACT_SYSTEM_PROMPT
    assert "Loaded skills provide a base directory" in LIBERO_ACT_SYSTEM_PROMPT
    assert "No RoboMEx-specific skill wrapper function is provided" in skill_text
    assert "Do not spend a turn printing sidecar source" in skill_text
    assert "Skill sidecar files:" in skill_text


def test_subagent_rejects_role_selection_in_prompt_contract() -> None:
    lib = FakeLibrary([_skill("place_object", "place objects", category="task")])
    policy = RecordingPolicy([_finish_result("checked", {"verdict": "fail", "state": "misaligned"})])
    agent = CodingAgentSubAgent(
        executor=FakeExecutor(),
        policy=policy,
        library=lib,
    )

    result = agent.run(SubAgentRequest(task="Check whether the bowl is centered on the plate"))

    assert result.ok
    prompt_text = "\n".join(
        str(msg.get("content", ""))
        for prompt in policy.prompts
        for msg in prompt
        if isinstance(msg, dict)
    )
    assert '"profile"' not in prompt_text
    assert "You are the RoboMEx Verifier SubAgent" in prompt_text
    assert "Verifier output requirement" not in prompt_text
    assert "Grounding output requirement" not in prompt_text
    assert "Affordance output requirement" not in prompt_text


def test_subagent_uses_task_first_prompt_without_profile() -> None:
    lib = FakeLibrary([_skill("place_object", "place objects", category="task")])
    policy = RecordingPolicy([_finish_result("checked", {"ok": True})])
    agent = CodingAgentSubAgent(
        executor=FakeExecutor(),
        policy=policy,
        library=lib,
    )

    result = agent.run(SubAgentRequest(task="Check whether the bowl is centered on the plate"))

    assert result.ok
    prompt_text = "\n".join(
        str(msg.get("content", ""))
        for prompt in policy.prompts
        for msg in prompt
        if isinstance(msg, dict)
    )
    assert '"profile"' not in prompt_text
    assert "Check whether the bowl is centered on the plate" in prompt_text
    assert "Verifier output requirement" not in prompt_text
    assert "Grounding output requirement" not in prompt_text
    assert "Affordance output requirement" not in prompt_text


def test_act_prompt_limits_subagent_to_verification() -> None:
    from robomex.prompts import BASE_ACT_SYSTEM_PROMPT, LIBERO_ACT_SYSTEM_PROMPT

    assert "Use call_subagent only as a read-only verifier/diagnoser" in LIBERO_ACT_SYSTEM_PROMPT
    assert "Do not delegate grounding" in LIBERO_ACT_SYSTEM_PROMPT
    assert "Before writing any Python" not in BASE_ACT_SYSTEM_PROMPT
    assert "Before writing any Python" not in LIBERO_ACT_SYSTEM_PROMPT
    assert "Before physical execution" in BASE_ACT_SYSTEM_PROMPT
    assert "before motion or gripper execution" in LIBERO_ACT_SYSTEM_PROMPT
    assert "read-only observation" in BASE_ACT_SYSTEM_PROMPT
    assert "read-only observation" in LIBERO_ACT_SYSTEM_PROMPT
    assert "state_patch" not in BASE_ACT_SYSTEM_PROMPT
    assert "persistent world state" in LIBERO_ACT_SYSTEM_PROMPT


def test_subagent_prompt_prefers_sidecar_paths_over_schema_probing() -> None:
    from robomex.agents.subagents import _SUBAGENT_SYSTEM_PROMPT, render_subagent_system_prompt

    assert "sidecar scripts, references, or assets" in _SUBAGENT_SYSTEM_PROMPT
    assert "entry points and usage patterns" in _SUBAGENT_SYSTEM_PROMPT
    assert "Do not read or print a whole sidecar" in _SUBAGENT_SYSTEM_PROMPT
    assert "do not spend turns printing obs.keys()" in _SUBAGENT_SYSTEM_PROMPT
    assert "inspect.signature" in _SUBAGENT_SYSTEM_PROMPT

    rendered = render_subagent_system_prompt("Core APIs:\nget_observation()")
    assert "Available sandbox API functions" in rendered
    assert "get_observation()" in rendered


def test_segment_object_sidecar_uses_live_observation_contract(tmp_path) -> None:
    script = Path("robomex/skills/builtin/perception/segment_object/scripts/segment_object.py")
    spec = importlib.util.spec_from_file_location("segment_object_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    obs = {
        "agentview": {
            "images": {"rgb": rgb, "depth": depth},
            "intrinsics": np.eye(3),
            "pose_mat": np.eye(4),
        }
    }
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    points = np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.3]], dtype=float)

    def vlm_bbox_detection(_rgb, _target):
        return [2, 2, 5, 5]

    def segment_sam3_box_prompt(_rgb, _box):
        return [{"mask": mask, "score": 0.9}]

    def mask_to_world_points(_mask, _depth, _intrinsics, _extrinsics):
        return points

    def filter_noise(values):
        return values

    evidence = {}
    result = module.ground_object_from_observation(
        obs,
        target_name="test bowl",
        artifacts_dir=str(tmp_path),
        evidence=evidence,
        apis={
            "vlm_bbox_detection": vlm_bbox_detection,
            "segment_sam3_box_prompt": segment_sam3_box_prompt,
            "mask_to_world_points": mask_to_world_points,
            "filter_noise": filter_noise,
        },
    )

    assert result["target"] == "test bowl"
    assert result["center_xyz"] == [0.15000000000000002, 0.2, 0.3]
    assert evidence["object_grounding"]["points_key"] == "grounding.points"
    assert np.array_equal(evidence["grounding.points"], points)
    assert (tmp_path / "segment_vlm_box.png").exists()
    assert (tmp_path / "segment_sam3_mask.png").exists()


def test_segment_object_sidecar_auto_selects_available_camera(tmp_path) -> None:
    script = Path("robomex/skills/builtin/perception/segment_object/scripts/segment_object.py")
    spec = importlib.util.spec_from_file_location("segment_object_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    obs = {
        "robot0_eye_in_hand": {
            "images": {"rgb": rgb, "depth": depth},
            "intrinsics": np.eye(3),
            "pose_mat": np.eye(4),
        }
    }
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    points = np.array([[0.1, 0.2, 0.3]], dtype=float)

    evidence = {}
    result = module.ground_object_from_observation(
        obs,
        target_name="test bowl",
        artifacts_dir=str(tmp_path),
        evidence=evidence,
        apis={
            "vlm_bbox_detection": lambda _rgb, _target: [1, 1, 4, 4],
            "segment_sam3_box_prompt": lambda _rgb, _box: [{"mask": mask, "score": 0.9}],
            "mask_to_world_points": lambda _mask, _depth, _intrinsics, _extrinsics: points,
            "filter_noise": lambda values: values,
        },
    )

    assert result["bbox"] == [1.0, 1.0, 4.0, 4.0]
    assert result["center_xyz"] == [0.1, 0.2, 0.3]
    assert np.array_equal(evidence["grounding.points"], points)


def test_skill_library_admit_copies_claude_style_sidecars() -> None:
    import tempfile
    from pathlib import Path

    from robomex.skills import SkillLibrary

    with tempfile.TemporaryDirectory() as src_td, tempfile.TemporaryDirectory() as dst_td:
        src = Path(src_td) / "segment_object"
        (src / "references").mkdir(parents=True)
        (src / "SKILL.md").write_text(
            "---\nname: segment_object\ncategory: perception\n---\n\nBody.",
            encoding="utf-8",
        )
        (src / "references" / "grounding_object.md").write_text("reference", encoding="utf-8")

        lib = SkillLibrary(dst_td)
        lib.admit(Skill.from_dir(src))

        copied = Path(dst_td) / "perception" / "segment_object" / "references" / "grounding_object.md"
        assert copied.read_text(encoding="utf-8") == "reference"


def test_act_has_no_review_specific_code_gate() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = ScriptedCodePolicy([
        _use_skill("grasp"),
        _run_python("open('review_note.py', 'w').write('pass')"),
        _run_python("close_gripper()"),
        _finish(),
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)

    trace = agent.run(task="grasp the cube")

    executed_codes = [b.code for b in ex.blocks if b.name != "evidence_seed"]
    assert any("review_note.py" in c for c in executed_codes)
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


def test_executor_initial_prompt_includes_scene_image() -> None:
    """Act Agent initial prompt returns multimodal content when scene_image_path is set."""
    import tempfile

    import numpy as np

    from robomex.perception.render import save_rgb

    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = ScriptedCodePolicy([_finish()])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=6)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        save_rgb(f.name, np.zeros((4, 4, 3), dtype=np.uint8))
        agent._task = "test task"
        agent._observation_summary = ""
        agent._feedback = ""
        agent._scene_image_path = f.name
        msg = agent._initial_user_message()
        assert isinstance(msg, list), "Expected multimodal content list"
        assert msg[0]["type"] == "text"
        assert "occludes" in msg[0]["text"]
        assert msg[1]["type"] == "image_url"
        assert msg[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_executor_initial_prompt_includes_expected_postcondition() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    agent = CodeAsPolicyAgent(executor=FakeExecutor(), policy=ScriptedCodePolicy([_finish()]), library=lib)

    agent._task = "pick up the bowl"
    agent._expected_postcondition = "The bowl is visibly held by the gripper."
    agent._observation_summary = ""
    agent._feedback = ""
    agent._scene_image_path = None
    msg = agent._initial_user_message()

    assert isinstance(msg, str)
    assert "Expected postcondition: The bowl is visibly held by the gripper." in msg


def test_executor_feedback_includes_observation_image() -> None:
    """Feedback after execution includes observation image when available."""
    import tempfile

    import numpy as np

    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = ScriptedCodePolicy([_use_skill("grasp"), _run_python("print('hi')"), _finish()])

    rgb = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
    obs_with_rgb = {"robot0_eye_in_hand": {"images": {"rgb": rgb}}}

    class FakeExecWithObs:
        def __init__(self):
            self.blocks = []

        def run_block(self, block):
            self.blocks.append(block)
            return BlockExecutionResult(
                block=block,
                ok=True,
                status=ActionBlockStatus.SUCCEEDED,
                stdout="done",
                stderr="",
                reward=0.0,
                terminated=False,
                truncated=False,
                observation=obs_with_rgb,
                info={"sandbox_rc": 0},
            )

    with tempfile.TemporaryDirectory() as td:
        ex = FakeExecWithObs()
        agent = CodeAsPolicyAgent(
            executor=ex,
            policy=policy,
            library=lib,
            max_turns=6,
            observation_camera="robot0_eye_in_hand",
        )
        agent.run(task="test", video_dir=td)

        feedback = agent._feedback_message(BlockExecutionResult(
            block=SemanticActionBlock(name="t", intent="t", code="x"),
            ok=True,
            status=ActionBlockStatus.SUCCEEDED,
            stdout="out",
            stderr="",
            reward=0.0,
            terminated=False,
            truncated=False,
            observation=obs_with_rgb,
            info={},
        ))
        assert isinstance(feedback, list), "Expected multimodal feedback"
        assert feedback[0]["type"] == "text"
        assert feedback[1]["type"] == "image_url"


def test_executor_seed_uses_configured_observation_camera() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = ScriptedCodePolicy([_finish()])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(
        executor=ex,
        policy=policy,
        library=lib,
        observation_camera="robot0_eye_in_hand",
    )

    agent.run(task="test")

    seed_blocks = [b.code for b in ex.blocks if b.name == "evidence_seed"]
    assert seed_blocks
    assert "OBS_CAMERA = 'robot0_eye_in_hand'" in seed_blocks[0]
    assert "['agentview']" not in seed_blocks[0]


def test_executor_seed_auto_selects_observation_camera_by_default() -> None:
    lib = FakeLibrary([_skill("grasp", "grasp objects", category="motion")])
    policy = ScriptedCodePolicy([_finish()])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(
        executor=ex,
        policy=policy,
        library=lib,
        max_turns=6,
    )

    agent.run("inspect")

    seed_blocks = [b.code for b in ex.blocks if b.name == "evidence_seed"]
    assert seed_blocks
    assert "OBS_CAMERA = None" in seed_blocks[0]
    assert "['agentview']" not in seed_blocks[0]


def test_evidence_collector_auto_selects_available_rgb_camera(tmp_path) -> None:
    from robomex.perception.collector import EvidenceCollector

    rgb_before = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_after = np.ones((4, 4, 3), dtype=np.uint8) * 255
    obs_before = {"robot0_eye_in_hand": {"images": {"rgb": rgb_before}}}
    obs_after = {"robot0_eye_in_hand": {"images": {"rgb": rgb_after}}}

    bundle = EvidenceCollector(tmp_path).bundle_for_block("turn_00", obs_before, obs_after)

    assert len(bundle.artifacts) == 3
    assert (tmp_path / "turn_00" / "before.png").exists()
    assert (tmp_path / "turn_00" / "after.png").exists()
    assert (tmp_path / "turn_00" / "before_after.png").exists()


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
