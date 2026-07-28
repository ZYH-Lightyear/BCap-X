"""Reactive Planner tests.

Covers: single ActionIntent per call, the react feedback loop, the done
branch, tolerant parsing, missing-field errors, and the prompt declarations
that lock the single-step discipline in place.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from robomex.contracts import ActionIntent, IntentFeedback, PlannerStep
from robomex.planner import PLANNER_SYSTEM, ReactivePlanner
from robomex.trace import PlannerTracer


class _ScriptedPolicy:
    """Return pre-scripted responses; record all prompts."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[list[dict]] = []

    def complete(self, prompt: list[dict]) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            return '{"done": true, "thought": "exhausted", "reason": "no more responses"}'
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
#  Test 1: First call decides exactly one ActionIntent
# ---------------------------------------------------------------------------

def test_first_step_produces_one_intent() -> None:
    policy = _ScriptedPolicy([
        json.dumps({
            "thought": "The robot needs to open its gripper first.",
            "intent": {
                "instruction": "open the gripper",
                "expected_effect": "gripper is fully open",
            },
        }),
    ])
    planner = ReactivePlanner(policy)

    step = planner.step(task="put the black bowl on the plate")

    assert not step.done
    assert step.intent is not None
    assert step.intent.intent_id == "intent-1"
    assert step.intent.instruction == "open the gripper"
    assert step.intent.expected_effect == "gripper is fully open"
    assert step.thought == "The robot needs to open its gripper first."


# ---------------------------------------------------------------------------
#  Test 2: React feedback loop
# ---------------------------------------------------------------------------

def test_react_feedback_drives_different_next_step() -> None:
    policy = _ScriptedPolicy([
        json.dumps({
            "thought": "Retry with more force.",
            "intent": {
                "instruction": "grasp the bowl with increased grip force",
                "expected_effect": "bowl is securely grasped",
            },
        }),
    ])
    planner = ReactivePlanner(policy)

    i1 = ActionIntent(
        intent_id="intent-1",
        instruction="grasp the bowl",
        expected_effect="bowl is grasped",
    )
    fb1 = IntentFeedback(
        intent_id="intent-1",
        status="failed",
        summary="gripper never closed",
        detail="Traceback: gripper timeout after 2s",
    )

    step = planner.step(task="put the black bowl on the plate", history=((i1, fb1),))

    assert not step.done
    assert step.intent is not None
    assert step.intent.intent_id == "intent-2"
    assert step.intent.instruction != i1.instruction

    prompt = policy.prompts[0]
    user_content = json.loads(prompt[1]["content"])
    assert len(user_content["issued_intents"]) == 1
    issued = user_content["issued_intents"][0]
    assert issued["feedback_status"] == "failed"
    assert issued["feedback_detail"] == "Traceback: gripper timeout after 2s"


# ---------------------------------------------------------------------------
#  Test 3: Done branch
# ---------------------------------------------------------------------------

def test_done_branch() -> None:
    policy = _ScriptedPolicy([
        json.dumps({
            "thought": "The bowl is on the plate.",
            "done": True,
            "reason": "task complete",
        }),
    ])
    planner = ReactivePlanner(policy)

    i1 = ActionIntent("intent-1", "pick up the bowl", "bowl is held")
    fb1 = IntentFeedback("intent-1", "succeeded", "bowl picked up")

    step = planner.step(
        task="put the black bowl on the plate",
        history=((i1, fb1),),
    )

    assert step.done
    assert step.intent is None
    assert step.reason == "task complete"


# ---------------------------------------------------------------------------
#  Test 4: Tolerant parsing
# ---------------------------------------------------------------------------

def test_tolerant_parsing_markdown_wrapped_json() -> None:
    """JSON wrapped in markdown fences should still parse."""
    policy = _ScriptedPolicy([
        '```json\n{"thought": "ok", "intent": {"instruction": "move above bowl", "expected_effect": "arm is above bowl"}}\n```',
    ])
    planner = ReactivePlanner(policy)

    step = planner.step(task="pick up bowl")

    assert not step.done
    assert step.intent is not None
    assert step.intent.instruction == "move above bowl"


def test_tolerant_parsing_accepts_legacy_subgoal_key() -> None:
    """A model that still says "subgoal" is understood rather than rejected."""
    policy = _ScriptedPolicy([
        json.dumps({
            "thought": "ok",
            "subgoal": {
                "instruction": "close gripper",
                "expected_effect": "gripper closed on object",
            },
        }),
    ])
    planner = ReactivePlanner(policy)

    step = planner.step(task="grasp object")

    assert step.intent is not None
    assert step.intent.instruction == "close gripper"


def test_tolerant_parsing_extra_fields() -> None:
    """Extra fields in JSON are silently ignored."""
    policy = _ScriptedPolicy([
        json.dumps({
            "thought": "ok",
            "confidence": 0.95,
            "intent": {
                "instruction": "close gripper",
                "expected_effect": "gripper closed on object",
                "priority": "high",
            },
        }),
    ])
    planner = ReactivePlanner(policy)

    step = planner.step(task="grasp object")

    assert not step.done
    assert step.intent is not None
    assert step.intent.instruction == "close gripper"


def test_missing_expected_effect_raises_readable_error() -> None:
    policy = _ScriptedPolicy([
        json.dumps({"thought": "ok", "intent": {"instruction": "move to bowl"}}),
    ])
    planner = ReactivePlanner(policy)

    with pytest.raises(ValueError, match="expected_effect"):
        planner.step(task="pick up bowl")


def test_missing_instruction_raises_readable_error() -> None:
    policy = _ScriptedPolicy([
        json.dumps({"thought": "ok", "intent": {"expected_effect": "arm above bowl"}}),
    ])
    planner = ReactivePlanner(policy)

    with pytest.raises(ValueError, match="instruction"):
        planner.step(task="pick up bowl")


# ---------------------------------------------------------------------------
#  Test 5: Prompt locks the paradigm
# ---------------------------------------------------------------------------

def test_system_prompt_declares_execution_not_connected() -> None:
    """Locks the Phase 1 empty-slot declaration so it is neither silently
    dropped nor forgotten once execution is wired up."""
    assert "执行层尚未接通" in PLANNER_SYSTEM
    assert "not_executed" in PLANNER_SYSTEM

    policy = _ScriptedPolicy([
        json.dumps({"thought": "ok", "done": True, "reason": "check prompt"}),
    ])
    planner = ReactivePlanner(policy)
    planner.step(task="test")

    system_msg = policy.prompts[0][0]["content"]
    assert "执行层尚未接通" in system_msg
    assert "not_executed" in system_msg


def test_system_prompt_enforces_single_step_discipline() -> None:
    """The planner must ask for exactly one next ActionIntent.

    Guards the core proposition: Code is a VLA-aligned intermediate action
    language, so the planner closes the loop each tick instead of pre-expanding
    a fixed pipeline. An earlier revision framed the job as "decompose into a
    sequence" / "author an instruction sequence", and the model duly produced
    open-loop plans that chained off its own imagined prior steps.
    """
    assert "单步纪律" in PLANNER_SYSTEM
    assert "只决定**下一个**动作意图" in PLANNER_SYSTEM
    assert "你不是任务拆解器" in PLANNER_SYSTEM

    for banned in ("拆解成一串", "编写一份供将来执行的指令序列", "指令序列"):
        assert banned not in PLANNER_SYSTEM, banned


def test_system_prompt_keeps_machine_contract_keys_in_english() -> None:
    """JSON keys and the status token map 1:1 onto robomex.contracts and must
    stay English even though the surrounding prompt is Chinese."""
    for token in (
        "thought",
        "intent",
        "instruction",
        "expected_effect",
        "done",
        "reason",
        "not_executed",
    ):
        assert token in PLANNER_SYSTEM, token


# ---------------------------------------------------------------------------
#  Test 6: Max intents boundary
# ---------------------------------------------------------------------------

def test_max_intents_returns_done() -> None:
    policy = _ScriptedPolicy([])
    planner = ReactivePlanner(policy, max_intents=2)

    i1 = ActionIntent("intent-1", "a", "b")
    i2 = ActionIntent("intent-2", "c", "d")
    fb = IntentFeedback("x", "not_executed")

    step = planner.step(task="task", history=((i1, fb), (i2, fb)))

    assert step.done
    assert "max_intents" in step.reason
    # 熔断路径不得调用模型。
    assert policy.prompts == []


# ---------------------------------------------------------------------------
#  Test 7: Trace recording
# ---------------------------------------------------------------------------

def test_trace_records_step_files(tmp_path: Any) -> None:
    raw_response = json.dumps({
        "thought": "first step",
        "intent": {
            "instruction": "open gripper",
            "expected_effect": "gripper open",
        },
    })
    policy = _ScriptedPolicy([raw_response])
    planner = ReactivePlanner(policy)
    tracer = PlannerTracer(tmp_path)

    step = planner.step(task="test task")
    step_dir = tracer.record(0, planner, step)

    data = json.loads((step_dir / "step.json").read_text())
    assert data["intent"]["instruction"] == "open gripper"
    assert data["done"] is False

    # prompt.json must hold the real prompt the model saw, not an empty stub.
    prompt = json.loads((step_dir / "prompt.json").read_text())
    assert prompt[0]["role"] == "system"
    assert "执行层尚未接通" in prompt[0]["content"]
    assert json.loads(prompt[1]["content"])["task"] == "test task"

    # response.txt must hold the raw model text verbatim.
    assert (step_dir / "response.txt").read_text() == raw_response


def test_trace_captures_raw_response_even_when_unparsed(tmp_path: Any) -> None:
    """A malformed reply still lands in response.txt, so runs are debuggable."""
    policy = _ScriptedPolicy(["I cannot help with that."])
    planner = ReactivePlanner(policy)
    tracer = PlannerTracer(tmp_path)

    with pytest.raises(ValueError):
        planner.step(task="test task")

    step_dir = tracer.record(
        0,
        planner,
        PlannerStep(thought="", intent=None, done=False, reason="parse failed"),
    )
    assert (step_dir / "response.txt").read_text() == "I cannot help with that."
    assert json.loads((step_dir / "prompt.json").read_text())[0]["role"] == "system"
