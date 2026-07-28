"""Reactive Planner tests.

Covers: single ActionIntent per call, the observation channel, the react
feedback loop, the done branch, tolerant parsing, missing-field errors, and
the prompt declarations that lock the single-step discipline in place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robomex.contracts import ActionIntent, IntentFeedback, Observation, PlannerStep
from robomex.env import NOT_EXECUTED_NOTE
from robomex.images import save_rgb
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


def _fake_observation(tmp_path: Path, name: str = "obs-000.png") -> Observation:
    """A real on-disk PNG, so image encoding runs for real in tests."""
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    rgb[:, :6] = (200, 30, 30)
    path = save_rgb(tmp_path / name, rgb)
    return Observation(image_path=path, camera="agentview", note="任务初始状态。")


def _intent_response(instruction: str, effect: str) -> str:
    return json.dumps({
        "thought": "ok",
        "intent": {"instruction": instruction, "expected_effect": effect},
    })


def _text_part(prompt: list[dict]) -> str:
    """Pull the text half out of a multimodal user message."""
    return next(p["text"] for p in prompt[1]["content"] if p["type"] == "text")


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
    assert step.intent.instruction == "open the gripper"
    assert step.intent.expected_effect == "gripper is fully open"
    assert step.thought == "The robot needs to open its gripper first."


# ---------------------------------------------------------------------------
#  Test 2: Observation channel
# ---------------------------------------------------------------------------

def test_observation_is_attached_as_image_part(tmp_path: Path) -> None:
    """The current frame must reach the model as a real image part.

    This is the material precondition for single-step discipline: with nothing
    to look at, the model has no option but to invent a pipeline.
    """
    policy = _ScriptedPolicy([_intent_response("move above the bowl", "arm is above bowl")])
    planner = ReactivePlanner(policy)

    planner.step(task="pick up the bowl", observation=_fake_observation(tmp_path))

    content = policy.prompts[0][1]["content"]
    assert isinstance(content, list)
    kinds = [part["type"] for part in content]
    assert kinds == ["text", "image_url"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    # Camera and the environment's note travel alongside the pixels, so the
    # model knows which viewpoint it is looking from.
    observation = json.loads(_text_part(policy.prompts[0]))["observation"]
    assert observation["camera"] == "agentview"
    assert observation["note"] == "任务初始状态。"


def test_prompt_stays_plain_text_without_observation() -> None:
    """No image means no multimodal wrapper — backends that only accept string
    content should not have to unpack a one-element list."""
    policy = _ScriptedPolicy([_intent_response("open gripper", "gripper open")])
    planner = ReactivePlanner(policy)

    planner.step(task="pick up the bowl")

    content = policy.prompts[0][1]["content"]
    assert isinstance(content, str)
    assert "observation" not in json.loads(content)


def test_oversized_observation_is_downscaled(tmp_path: Path) -> None:
    """Observations are re-sent every tick, so the in-prompt copy is bounded."""
    rgb = np.random.default_rng(0).integers(0, 255, size=(600, 900, 3), dtype=np.uint8)
    path = save_rgb(tmp_path / "big.png", rgb)
    observation = Observation(image_path=path, camera="agentview", note="")

    policy = _ScriptedPolicy([_intent_response("look", "looked")])
    planner = ReactivePlanner(policy, image_max_edge=128)
    planner.step(task="t", observation=observation)

    small = policy.prompts[0][1]["content"][1]["image_url"]["url"]

    policy_full = _ScriptedPolicy([_intent_response("look", "looked")])
    ReactivePlanner(policy_full, image_max_edge=900).step(task="t", observation=observation)
    large = policy_full.prompts[0][1]["content"][1]["image_url"]["url"]

    assert len(small) < len(large)
    # The artifact on disk is trace evidence and must never be rewritten.
    assert Path(path).stat().st_size > 0


# ---------------------------------------------------------------------------
#  Test 3: React feedback loop
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
        instruction="grasp the bowl",
        expected_effect="bowl is grasped",
    )
    fb1 = IntentFeedback(
        status="failed",
        summary="gripper never closed",
        detail="Traceback: gripper timeout after 2s",
    )

    step = planner.step(task="put the black bowl on the plate", history=((i1, fb1),))

    assert not step.done
    assert step.intent is not None
    assert step.intent.instruction != i1.instruction

    prompt = policy.prompts[0]
    user_content = json.loads(prompt[1]["content"])
    assert len(user_content["issued_intents"]) == 1
    issued = user_content["issued_intents"][0]
    assert issued["feedback_status"] == "failed"
    assert issued["feedback_detail"] == "Traceback: gripper timeout after 2s"


# ---------------------------------------------------------------------------
#  Test 4: Done branch
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

    i1 = ActionIntent("pick up the bowl", "bowl is held")
    fb1 = IntentFeedback("succeeded", "bowl picked up")

    step = planner.step(
        task="put the black bowl on the plate",
        history=((i1, fb1),),
    )

    assert step.done
    assert step.intent is None
    assert step.reason == "task complete"


# ---------------------------------------------------------------------------
#  Test 5: Tolerant parsing
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
#  Test 6: Prompt locks the paradigm
# ---------------------------------------------------------------------------

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
    """The JSON keys map 1:1 onto robomex.contracts and must stay English even
    though the surrounding prompt is Chinese."""
    for token in (
        "thought",
        "intent",
        "instruction",
        "expected_effect",
        "done",
        "reason",
    ):
        assert token in PLANNER_SYSTEM, token


# ---------------------------------------------------------------------------
#  Test 7: Max intents boundary
# ---------------------------------------------------------------------------

def test_max_intents_returns_done() -> None:
    policy = _ScriptedPolicy([])
    planner = ReactivePlanner(policy, max_intents=2)

    i1 = ActionIntent("a", "b")
    i2 = ActionIntent("c", "d")
    fb = IntentFeedback("not_executed")

    step = planner.step(task="task", history=((i1, fb), (i2, fb)))

    assert step.done
    assert "max_intents" in step.reason
    # 熔断路径不得调用模型。
    assert policy.prompts == []


# ---------------------------------------------------------------------------
#  Test 8: Trace recording
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
    assert "单步纪律" in prompt[0]["content"]
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


def test_trace_elides_base64_and_records_observation(tmp_path: Path) -> None:
    """prompt.json stays human-readable; the image lives on at its own path."""
    observation = _fake_observation(tmp_path)
    policy = _ScriptedPolicy([_intent_response("open gripper", "gripper open")])
    planner = ReactivePlanner(policy)
    tracer = PlannerTracer(tmp_path / "run")

    step = planner.step(task="test task", observation=observation)
    step_dir = tracer.record(0, planner, step, observation)

    prompt = json.loads((step_dir / "prompt.json").read_text())
    image_part = prompt[1]["content"][1]
    assert "base64" not in image_part["image_url"]["url"]
    assert "elided" in image_part["image_url"]["url"]

    recorded = json.loads((step_dir / "observation.json").read_text())
    assert recorded["image_path"] == observation.image_path
    assert Path(recorded["image_path"]).is_file()


# ---------------------------------------------------------------------------
#  Test 9: The loop against a stub environment
# ---------------------------------------------------------------------------

class _StubEnv:
    """An environment that renders honestly and executes nothing.

    Mirrors :class:`robomex.env.LiberoEnv` without pulling in the simulator:
    every ``apply`` returns ``not_executed`` plus a freshly numbered frame.
    """

    task_prompt = "put the black bowl on the plate"

    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self._index = 0
        self.emitted: list[str] = []

    def reset(self) -> Observation:
        return self._observe("任务初始状态。")

    def apply(self, intent: ActionIntent) -> tuple[IntentFeedback, Observation]:
        feedback = IntentFeedback(status="not_executed", summary=NOT_EXECUTED_NOTE)
        return feedback, self._observe(NOT_EXECUTED_NOTE)

    def _observe(self, note: str) -> Observation:
        observation = _fake_observation(self._tmp_path, f"obs-{self._index:03d}.png")
        self._index += 1
        self.emitted.append(observation.image_path)
        return Observation(
            image_path=observation.image_path,
            camera=observation.camera,
            note=note,
        )


def test_loop_feeds_not_executed_and_a_fresh_frame_back(tmp_path: Path) -> None:
    """Two ticks of the real loop shape used by the CLI.

    The second prompt must carry both halves of the closed loop: the honest
    ``not_executed`` verdict on what was issued, and a freshly rendered frame.
    """
    policy = _ScriptedPolicy([
        _intent_response("move above the bowl", "arm is above bowl"),
        _intent_response("move above the bowl", "arm is above bowl"),
    ])
    planner = ReactivePlanner(policy)
    env = _StubEnv(tmp_path)

    observation = env.reset()
    history: list[tuple[ActionIntent, IntentFeedback]] = []
    for _ in range(2):
        step = planner.step(task=env.task_prompt, observation=observation, history=tuple(history))
        assert step.intent is not None
        feedback, observation = env.apply(step.intent)
        history.append((step.intent, feedback))

    second = json.loads(_text_part(policy.prompts[1]))
    issued = second["issued_intents"]
    assert len(issued) == 1
    assert issued[0]["feedback_status"] == "not_executed"
    assert "没有执行成功" in issued[0]["feedback_summary"]
    assert second["observation"]["note"] == NOT_EXECUTED_NOTE

    # Each tick looks at its own freshly rendered frame rather than reusing the
    # first one — that is the half of the loop this milestone connects.
    assert [Path(p).name for p in env.emitted] == ["obs-000.png", "obs-001.png", "obs-002.png"]
