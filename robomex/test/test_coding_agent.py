"""Act-only CodingAgent 离线测试(无 env / LLM / 网络)。"""

from __future__ import annotations

import json
import importlib.util
import copy
from pathlib import Path

import numpy as np
import pytest

from robomex.agents import CodeAsPolicyAgent
from robomex.authoring.capabilities import CapabilityBoundBlockExecutor, CapabilityPolicy
from robomex.authoring.capabilities import CapabilityBoundBlockExecutor, CapabilityPolicy
from robomex.agents.subagents import (
    CodingAgentSubAgent,
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
    ArtifactRef,
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
        self.prompts.append(copy.deepcopy(prompt))
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
    payload = {
        **result,
        "outputs": {
            "verifier_report": {
                "payload": dict(result),
                "confidence": result.get("confidence", 0.0),
            }
        },
    }
    return json.dumps({"tool": "finish", "args": {"claim": claim, "result": payload}})


def _call_subagent(
    task: str,
    inputs: dict | None = None,
    task_id: str | None = None,
) -> str:
    args = {"task": task, "task_kind": "verify", "inputs": inputs or {}}
    if task_id:
        args["task_id"] = task_id
    return json.dumps({"tool": "call_subagent", "args": args})


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

    spawned = parse_action(
        json.dumps(
            {
                "tool": "spawn_subagent",
                "args": {
                    "spec": {
                        "id": "localizer",
                        "role": "visual-localizer",
                        "objective": "Localize the target.",
                        "task": "localize target",
                    }
                },
            }
        )
    )
    assert spawned.kind == "invalid"
    assert "dynamic specialist graph" in spawned.error

    incomplete = parse_action(
        '{"tool":"spawn_subagent","args":{"spec":{"id":"x","role":"r"}}}'
    )
    assert incomplete.kind == "invalid"
    assert "dynamic specialist graph" in (incomplete.error or "")
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


def test_first_action_frame_quarantines_hallucinated_turns() -> None:
    from robomex.core.coder import parse_model_turn

    raw = (
        _run_python("print('move')", intent="execute")
        + '\n\nuser{"status":"succeeded"}'
        + '\nassistant{"tool":"finish","args":{"claim":"fabricated"}}'
    )

    turn = parse_model_turn(raw)

    assert not turn.is_error
    assert turn.tool_calls[0].name == "run_python"
    assert json.loads(turn.canonical_json) == json.loads(
        _run_python("print('move')", intent="execute")
    )
    assert "fabricated" in turn.quarantined_suffix
    assert "fabricated" not in turn.canonical_json


def test_coding_agent_history_contains_only_canonical_action_frame() -> None:
    fake_tail = '\nuser{"status":"succeeded"}\nassistant{"tool":"finish","args":{}}'
    policy = RecordingPolicy([
        _run_python("print('real')") + fake_tail,
        _finish(),
    ])
    agent = CodeAsPolicyAgent(
        executor=FakeExecutor(),
        policy=policy,
        library=FakeLibrary([]),
        max_turns=2,
    )

    trace = agent.run(task="inspect")

    assert trace.success
    second_prompt = json.dumps(policy.prompts[1], ensure_ascii=False)
    assert "print('real')" in second_prompt
    assert '"status":"succeeded"' not in second_prompt
    assert "assistant{" not in second_prompt


def test_turn_engine_stops_on_explicit_protocol_error_budget() -> None:
    from robomex.core.coder import TurnBudget, TurnEngine

    engine = TurnEngine(
        ScriptedCodePolicy(["not-json", _finish()]),
        allowed_tools={"finish"},
        budget=TurnBudget(max_model_calls=5, max_protocol_errors=1),
    )
    prompt = [{"role": "user", "content": "finish"}]

    first = engine.next(prompt)

    assert first is not None and first.tool_call is None
    assert engine.ledger.model_calls == 1
    assert engine.ledger.protocol_errors == 1
    assert not engine.can_call_model
    assert engine.next(prompt) is None


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


def _skill_package(tmp_path: Path, skill_id: str, module: str) -> Skill:
    root = tmp_path / skill_id
    (root / "scripts").mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {skill_id}\ncategory: verification\ndescription: verify things\n---\n\n"
        f"Use `{module}` helpers from scripts/.",
        encoding="utf-8",
    )
    (root / "scripts" / f"{module}.py").write_text(
        "def helper():\n    return {'ok': True}\n",
        encoding="utf-8",
    )
    return Skill.from_dir(root)


def test_skill_load_injects_scripts_sys_path_without_budget(tmp_path: Path) -> None:
    """M1.5 Fix C: the runtime wires scripts/ imports; the agent never pays for it."""

    skill = _skill_package(tmp_path, "verify_grasp", "verify_helpers")
    lib = FakeLibrary([skill])
    policy = RecordingPolicy([
        _use_skill("verify_grasp"),
        _run_python("import verify_helpers\nprint(verify_helpers.helper())"),
        _finish(),
    ])
    ex = FakeExecutor()
    agent = CodeAsPolicyAgent(executor=ex, policy=policy, library=lib, max_turns=1)

    trace = agent.run(task="verify the grasp")

    setup_blocks = [b for b in ex.blocks if b.metadata.get("runtime_setup")]
    assert len(setup_blocks) == 1
    scripts_dir = str((tmp_path / "verify_grasp" / "scripts").resolve())
    assert scripts_dir in setup_blocks[0].code
    assert "sys.path.insert" in setup_blocks[0].code
    # The agent's own python action still fits in a max_turns=1 budget: the
    # setup block did not consume it.
    assert len(trace.turns) == 1
    # The loaded-skill message tells the agent to import directly.
    prompt_text = "\n".join(
        str(msg.get("content", ""))
        for prompt in policy.prompts
        for msg in prompt
        if isinstance(msg, dict)
    )
    assert "already on sys.path" in prompt_text
    assert "import verify_helpers" in prompt_text


def test_preloaded_skill_scripts_are_injected_once(tmp_path: Path) -> None:
    skill = _skill_package(tmp_path, "verify_grasp", "verify_helpers")
    lib = FakeLibrary([skill])
    policy = ScriptedCodePolicy([
        _use_skill("verify_grasp"),  # re-loading the same skill must not re-inject
        _finish_result("done", {"ok": True}),
    ])
    ex = FakeExecutor()
    agent = CodingAgentSubAgent(
        executor=ex,
        policy=policy,
        library=lib,
        max_turns=2,
        preloaded_skills=("verify_grasp",),
    )

    result = agent.run(SubAgentRequest(task="verify the grasp"))

    assert result.ok
    # Seed also uses runtime_setup; skill scripts must still inject only once.
    skill_setup_blocks = [
        b for b in ex.blocks if b.name.startswith("skill_setup_")
    ]
    assert len(skill_setup_blocks) == 1


def test_runtime_setup_block_bypasses_capability_policy() -> None:
    ex = FakeExecutor()
    guarded = CapabilityBoundBlockExecutor(
        ex,
        CapabilityPolicy(allowed=frozenset(), unknown_calls="deny"),
        node_id="verify",
    )
    block = SemanticActionBlock(
        name="skill_setup_verify",
        intent="runtime skill setup",
        code="import sys\nsys.path.insert(0, '/tmp/x')",
        metadata={"runtime_setup": True},
    )

    result = guarded.run_block(block)

    assert result.ok
    assert len(ex.blocks) == 1


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


@pytest.mark.skip(reason="Act inline delegation was replaced by SubgoalSwarmManager")
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


@pytest.mark.skip(reason="Act inline delegation was replaced by SubgoalSwarmManager")
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
                artifact_refs=(ArtifactRef("verifier_overlay", path="overlay.png", producer=self.name),),
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


@pytest.mark.skip(reason="SwarmRuntime owns dynamic SubAgent manifests")
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


@pytest.mark.skip(reason="SwarmRuntime owns dynamic SubAgent manifests")
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


def test_session_does_not_write_legacy_inline_subagent_manifest(tmp_path) -> None:
    from robomex import RoboMExAgent, RoboMExConfig
    from robomex.agents import ScriptedPlannerPolicy
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
            code_policy_kind="json_action_adapter",
        )
    )

    result = agent.run("Pick the can")

    assert result.success
    assert not (tmp_path / "episode" / "subagents.json").exists()
    summary = json.loads((tmp_path / "episode" / "summary.json").read_text(encoding="utf-8"))
    assert summary["authoring_strategy"] == "universal"


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
                "facts": [
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
                ],
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
    assert not (root / "report.html").exists()
    candidates = json.loads((root / "skill_evolution_candidates.json").read_text(encoding="utf-8"))
    assert candidates["schema"] == "robomex.skill_evolution_candidates.v1"
    assert candidates["subgoals"][0]["goal"] == "Pick the bowl"
    assert candidates["subgoals"][0]["evidence_timeline"][0]["source"] == "act:finish"


@pytest.mark.skip(reason="Act no longer exposes call_subagent")
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


def test_capability_guard_blocks_ungranted_motion_calls() -> None:
    ex = FakeExecutor()
    guarded = CapabilityBoundBlockExecutor(
        ex,
        CapabilityPolicy(allowed=frozenset({"perception_read"})),
        node_id="observer",
    )
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


def test_capability_guard_allows_ordinary_sidecar_code() -> None:
    ex = FakeExecutor()
    guarded = CapabilityBoundBlockExecutor(
        ex,
        CapabilityPolicy(allowed=frozenset({"perception_read"})),
        node_id="observer",
    )
    block = SemanticActionBlock(
        name="sub",
        intent="sidecar code",
        code="import importlib.util\nprint('load sidecar explicitly')",
    )

    result = guarded.run_block(block)

    assert result.ok
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
    assert "Treat input artifacts as claims" in prompt_text
    assert "capability classes" in prompt_text
    assert "typed hand-off" in prompt_text
    assert "base directory" in prompt_text


def test_act_prompt_and_loaded_skill_content_explain_sidecar_paths() -> None:
    from robomex.core.coder import build_skill_llm_content
    from robomex.prompts import LIBERO_ACT_SYSTEM_PROMPT

    skill_text = build_skill_llm_content("/tmp/skill", "Body")

    assert "base directory" in LIBERO_ACT_SYSTEM_PROMPT
    assert "resolve their relative scripts" in LIBERO_ACT_SYSTEM_PROMPT
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

    assert not result.ok
    prompt_text = "\n".join(
        str(msg.get("content", ""))
        for prompt in policy.prompts
        for msg in prompt
        if isinstance(msg, dict)
    )
    assert '"profile"' not in prompt_text
    assert "You are the RoboMEx verifier node." in prompt_text
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


def test_act_prompt_matches_universal_authoring_contract() -> None:
    from robomex.prompts import BASE_ACT_SYSTEM_PROMPT, LIBERO_ACT_SYSTEM_PROMPT

    assert "stage-sized block" in BASE_ACT_SYSTEM_PROMPT
    assert "stage-sized block" in LIBERO_ACT_SYSTEM_PROMPT
    assert "Never print raw arrays" in LIBERO_ACT_SYSTEM_PROMPT
    assert "Before motion or gripper execution" in BASE_ACT_SYSTEM_PROMPT
    assert "capability classes" in LIBERO_ACT_SYSTEM_PROMPT
    assert "state_patch" not in BASE_ACT_SYSTEM_PROMPT
    assert "persistent world facts" in LIBERO_ACT_SYSTEM_PROMPT


def test_execution_feedback_truncates_verbose_streams_for_llm() -> None:
    from robomex.core.coder.agent import CodingAgent

    block = SemanticActionBlock(name="t", intent="test", code="print('x')")
    result = BlockExecutionResult(
        block=block,
        ok=True,
        status=ActionBlockStatus.SUCCEEDED,
        stdout="A" * 2500,
        stderr="",
    )

    feedback = CodingAgent._feedback_message(result)

    assert isinstance(feedback, str)
    assert "stdout truncated for LLM feedback" in feedback
    assert "full stream is saved" in feedback
    assert "A" * 2500 not in feedback
    assert "large arrays, masks, candidates" in feedback


def test_subagent_prompt_uses_shared_runtime_contract() -> None:
    from robomex.agents.subagents import _SUBAGENT_SYSTEM_PROMPT, render_subagent_system_prompt

    assert "stage-sized block" in _SUBAGENT_SYSTEM_PROMPT
    assert "Treat input artifacts as claims" in _SUBAGENT_SYSTEM_PROMPT
    assert "perception_read" in _SUBAGENT_SYSTEM_PROMPT
    assert "output key `verifier_report`" in _SUBAGENT_SYSTEM_PROMPT
    assert "schema `robomex.verifier.v1`" in _SUBAGENT_SYSTEM_PROMPT

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


def test_skill_library_admit_is_idempotent_with_existing_sidecars(tmp_path) -> None:
    """M1.5 Fix G (B9): concurrent/repeated admits must not crash on existing dirs."""

    from robomex.skills import SkillLibrary

    src = tmp_path / "src" / "verify_x"
    (src / "scripts" / "__pycache__").mkdir(parents=True)
    (src / "SKILL.md").write_text(
        "---\nname: verify_x\ncategory: verification\n---\n\nBody.", encoding="utf-8"
    )
    (src / "scripts" / "helper.py").write_text("X = 1\n", encoding="utf-8")
    (src / "scripts" / "__pycache__" / "helper.cpython-311.pyc").write_bytes(b"junk")

    lib = SkillLibrary(tmp_path / "lib")
    skill = Skill.from_dir(src)
    lib.admit(skill)
    # Second admit hits the already-populated destination — must not raise.
    lib.admit(skill)

    dest = tmp_path / "lib" / "verification" / "verify_x" / "scripts"
    assert (dest / "helper.py").read_text(encoding="utf-8") == "X = 1\n"
    assert not (dest / "__pycache__").exists()


def test_compact_json_is_idempotent_on_its_own_summaries() -> None:
    """M1.5 Fix G: re-compacting a manifest never nests {"type":"dict","repr":...}."""

    from robomex.core.context import compact_json

    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"points": np.ones((100, 3))}}}}}}}
    once = compact_json(deep)
    twice = compact_json(once)
    assert twice == once

    arr_summary = compact_json({"points": np.ones((50, 3), dtype=np.float32)})
    assert arr_summary["points"]["type"] == "ndarray"
    again = compact_json(arr_summary, max_depth=1)
    assert again["points"] == arr_summary["points"]
    assert "repr" not in json.dumps(again)


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
