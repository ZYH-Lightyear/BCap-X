from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.test_vaw_context_runtime import FakeContextApi
from vaw.agents.contracts import ModelResponse, ToolCall
from vaw.context_runtime.runtime import ContextRunConfig, ContextRuntime
from vaw.context_runtime.protocol import (
    REVIEW_FUNCTION_NAMES,
    STANDARD_MAIN_FUNCTION_NAMES,
)
from vaw.context_runtime.trace import ContextTraceLogger
from vaw.context_runtime.workspace import ContextWorkspace


def _response(index: int, name: str, **arguments) -> ModelResponse:
    return ModelResponse(
        text=f"reason {index}",
        tool_calls=(ToolCall(id=f"call-{index}", name=name, args=arguments),),
    )


class RecordingProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.messages = []
        self.tools = []

    def generate(self, messages, tools=None):
        self.messages.append(messages)
        self.tools.append(tools)
        return self.responses.pop(0)


class SolidRenderer:
    name = "test-renderer"

    def render(self, packet):
        return np.full((1080, 1920, 3), packet.revision, dtype=np.uint8)

    def close(self):
        return None


def _image_count(messages) -> int:
    return sum(
        part.get("type") == "image_url"
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict)
    )


def _user_text(messages) -> str:
    return str(messages[1]["content"][0]["text"])


def test_dual_agent_runtime_rebuilds_every_request_without_history(tmp_path: Path) -> None:
    main = RecordingProvider(
        [
            _response(
                1,
                "close_gripper",
                refinement_goal="设置闭合目标并检查两指通道",
            ),
            _response(3, "commit", action_id="a1"),
            _response(4, "done", success=False),
        ]
    )
    imagination = RecordingProvider(
        [_response(2, "finish_imagination", status="ready")]
    )
    runtime = ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "test task", motion_backend="pyroki"),
        SolidRenderer(),
        imagination_provider=imagination,
        config=ContextRunConfig(max_main_turns=8, max_imagination_turns=4),
        trace=ContextTraceLogger(tmp_path),
    )

    result = runtime.run()

    assert result.turns == 4
    assert len(main.messages) == 3 and len(imagination.messages) == 1
    for messages in [*main.messages, *imagination.messages]:
        assert [message["role"] for message in messages] == ["system", "user"]
        assert not any(message.get("role") in {"assistant", "tool"} for message in messages)
        assert _image_count(messages) >= 1
    assert all(_image_count(messages) == 1 for messages in main.messages)
    post_commit_text = _user_text(main.messages[2])
    assert "Last Physical Action" in post_commit_text
    assert "设置闭合目标并检查两指通道" in post_commit_text
    assert "Main Working Focus" in post_commit_text
    assert "reason 3" in post_commit_text
    assert "Refinement Goal：设置闭合目标并检查两指通道" in json.dumps(
        imagination.messages[0], ensure_ascii=False
    )
    assert "reason 1" not in json.dumps(imagination.messages[0], ensure_ascii=False)
    review_text = _user_text(main.messages[1])
    assert "Main Working Focus" in review_text
    assert "reason 1" in review_text
    assert "Edit Summary" in json.dumps(imagination.messages[0], ensure_ascii=False)
    assert "reason 2" not in json.dumps(main.messages[1], ensure_ascii=False)

    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [row["agent_owner"] for row in rows] == [
        "main",
        "imagination",
        "main",
        "main",
    ]
    assert all("visible_recent_calls" not in row for row in rows)
    assert all("execution_receipt" not in row for row in rows)
    assert rows[0]["main_working_focus"] is None
    assert rows[2]["main_working_focus"] == "reason 1"
    assert rows[3]["main_working_focus"] == "reason 3"


def test_main_keeps_only_one_overwrite_only_working_focus() -> None:
    main = RecordingProvider(
        [
            ModelResponse(
                text="当前抬升核验显示目标没有随动；重新定位目标以重抓。",
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="detection_and_sam",
                        args={"query": "can"},
                    ),
                ),
            ),
            ModelResponse(
                text="region 已确认；生成不同抓取起点。",
                tool_calls=(
                    ToolCall(
                        id="call-2",
                        name="propose_grasps",
                        args={"region_id": "region1"},
                    ),
                ),
            ),
            _response(3, "done", success=False),
        ]
    )
    runtime = ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        SolidRenderer(),
    )

    runtime.run()

    second = _user_text(main.messages[1])
    third = _user_text(main.messages[2])
    assert "目标没有随动；重新定位目标以重抓" in second
    assert "region 已确认；生成不同抓取起点" in third
    assert "目标没有随动；重新定位目标以重抓" not in third
    assert "function_result" not in third and "call-2" not in third


def test_runtime_stops_after_environment_terminates_during_commit() -> None:
    main = RecordingProvider(
        [
            _response(
                1,
                "open_gripper",
                refinement_goal="只设置张开目标",
            ),
            _response(3, "commit", action_id="a1"),
            _response(4, "done", success=False),
        ]
    )
    imagination = RecordingProvider(
        [_response(2, "finish_imagination", status="ready")]
    )
    terminated = False

    def terminal_check() -> bool:
        return terminated

    api = FakeContextApi()
    original_open = api.open_gripper

    def terminating_open() -> None:
        nonlocal terminated
        original_open()
        terminated = True

    api.open_gripper = terminating_open  # type: ignore[method-assign]
    result = ContextRuntime(
        main,
        ContextWorkspace(api, "task", motion_backend="pyroki"),
        SolidRenderer(),
        imagination_provider=imagination,
        env_terminal_check=terminal_check,
    ).run()

    assert result.terminate_mode.value == "env_terminated"
    assert result.turns == 3
    assert len(main.messages) == 2
    assert [step.op for step in result.steps] == [
        "main:open_gripper",
        "imagination:finish_imagination",
        "main:commit",
    ]


def test_last_physical_action_persists_across_main_perception_calls() -> None:
    main = RecordingProvider(
        [
            _response(
                1,
                "open_gripper",
                refinement_goal="只设置真实执行后的张开目标",
            ),
            _response(3, "commit", action_id="a1"),
            ModelResponse(text="no function this time", tool_calls=()),
            _response(
                5,
                "delta_move",
                delta_xyz_m=[0.0, 0.0, 0.05],
                frame="base",
                refinement_goal="抬升并检查物体是否随动",
            ),
            _response(6, "detection_and_sam", query="can"),
            _response(7, "done", success=False),
        ]
    )
    imagination = RecordingProvider(
        [_response(2, "finish_imagination", status="ready")]
    )
    ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        SolidRenderer(),
        imagination_provider=imagination,
    ).run()

    assert "Last Physical Action" in _user_text(main.messages[2])
    assert "只设置真实执行后的张开目标" in _user_text(main.messages[2])
    assert "Last Physical Action" in _user_text(main.messages[3])
    assert "Last Physical Action" in _user_text(main.messages[4])
    assert "within [-0.03, 0.03]" in _user_text(main.messages[4])
    assert "Last Physical Action" in _user_text(main.messages[5])
    assert all(_image_count(messages) == 1 for messages in main.messages)


def test_completed_gripper_action_remains_visible_during_lift_review() -> None:
    main = RecordingProvider(
        [
            _response(1, "close_gripper", refinement_goal="闭合真实夹爪"),
            _response(3, "commit", action_id="a1"),
            _response(
                4,
                "delta_move",
                delta_xyz_m=[0.0, 0.0, 0.03],
                frame="base",
                refinement_goal="保持闭合并上抬 3cm 核验随动",
            ),
            _response(6, "done", success=False),
        ]
    )
    imagination = RecordingProvider(
        [
            _response(2, "finish_imagination", status="ready"),
            _response(5, "finish_imagination", status="ready"),
        ]
    )
    ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        SolidRenderer(),
        imagination_provider=imagination,
    ).run()

    review_text = _user_text(main.messages[3])
    assert "Last Physical Action" in review_text
    assert '"target_gripper":"closed"' in review_text
    assert '"outcome":"completed"' in review_text
    assert '"total_translation_base_m":[0.0,0.0,0.03]' in review_text


def test_review_tool_surface_requires_explicit_commit_revise_or_reject() -> None:
    main = RecordingProvider(
        [
            _response(
                1,
                "delta_move",
                delta_xyz_m=[0.0, 0.0, 0.01],
                frame="base",
                refinement_goal="检查目标位置",
            ),
            _response(3, "detection_and_sam", query="basket"),
            _response(4, "reject_action", action_id="a1"),
            _response(5, "detection_and_sam", query="basket"),
            _response(6, "done", success=False),
        ]
    )
    imagination = RecordingProvider(
        [_response(2, "finish_imagination", status="ready")]
    )
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    runtime = ContextRuntime(
        main,
        workspace,
        SolidRenderer(),
        imagination_provider=imagination,
    )

    result = runtime.run()

    invalid_perception = result.steps[2]
    assert invalid_perception.op == "main:detection_and_sam"
    assert not invalid_perception.ok
    assert "unknown function 'detection_and_sam'" in invalid_perception.result
    assert [item["function"]["name"] for item in main.tools[0]] == list(
        STANDARD_MAIN_FUNCTION_NAMES
    )
    assert [item["function"]["name"] for item in main.tools[1]] == list(
        REVIEW_FUNCTION_NAMES
    )
    assert [item["function"]["name"] for item in main.tools[2]] == list(
        REVIEW_FUNCTION_NAMES
    )
    assert [item["function"]["name"] for item in main.tools[3]] == list(
        STANDARD_MAIN_FUNCTION_NAMES
    )
    assert "当前只负责审查一个" in main.messages[1][0]["content"]
    assert "unknown function 'detection_and_sam'" in _user_text(main.messages[2])
    assert any(step.op == "main:reject_action" and step.ok for step in result.steps)
    assert any(step.op == "main:detection_and_sam" and step.ok for step in result.steps)
    assert workspace.state.regions
    assert workspace.state.action_review is None


def test_imagination_turn_limit_returns_neutral_review_to_main(tmp_path: Path) -> None:
    main = RecordingProvider(
        [
            _response(
                1,
                "open_gripper",
                refinement_goal="仅预览张开夹爪",
            ),
            _response(4, "done", success=False),
        ]
    )
    imagination = RecordingProvider(
        [
            _response(2, "close_gripper"),
            _response(3, "open_gripper"),
        ]
    )
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    result = ContextRuntime(
        main,
        workspace,
        SolidRenderer(),
        imagination_provider=imagination,
        config=ContextRunConfig(max_main_turns=4, max_imagination_turns=2),
        trace=ContextTraceLogger(tmp_path),
    ).run()

    assert result.turns == 4
    assert "Latest Imagination Handoff" in json.dumps(
        main.messages[1], ensure_ascii=False
    )
    visible = json.dumps(main.messages[1], ensure_ascii=False)
    assert "review_required" in visible
    assert "budget_exhausted" not in visible
    assert "turn_limit" not in visible
    events = [
        json.loads(line)
        for line in (tmp_path / "runtime_events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["termination_reason"] == "turn_limit"


def test_invalid_imagination_tool_is_one_shot_feedback_not_history() -> None:
    main = RecordingProvider(
        [
            _response(
                1,
                "open_gripper",
                refinement_goal="仅预览张开夹爪",
            ),
            _response(4, "done", success=False),
        ]
    )
    imagination = RecordingProvider(
        [
            _response(2, "detection_and_sam", query="can"),
            _response(3, "finish_imagination", status="failed"),
        ]
    )
    ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        SolidRenderer(),
        imagination_provider=imagination,
        config=ContextRunConfig(max_main_turns=4, max_imagination_turns=4),
    ).run()

    second = json.dumps(imagination.messages[1], ensure_ascii=False)
    assert "unknown function 'detection_and_sam'" in second
    assert "call-2" not in second


def test_failed_imagination_tells_main_which_seed_was_rejected() -> None:
    main = RecordingProvider(
        [
            _response(1, "detection_and_sam", query="can"),
            _response(2, "propose_grasps", region_id="region1"),
            _response(
                3,
                "select",
                seed_id="s1",
                refinement_goal="check whether the fingers can surround the can",
            ),
            _response(5, "done", success=False),
        ]
    )
    imagination = RecordingProvider(
        [_response(4, "finish_imagination", status="failed")]
    )

    ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        SolidRenderer(),
        imagination_provider=imagination,
    ).run()

    handoff_text = _user_text(main.messages[3])
    assert '"status":"failed"' in handoff_text
    assert '"source_ref":"s1"' in handoff_text


def test_main_starter_requires_explicit_refinement_goal() -> None:
    main = RecordingProvider(
        [
            _response(1, "close_gripper"),
            _response(2, "done", success=False),
        ]
    )
    imagination = RecordingProvider([])
    ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        SolidRenderer(),
        imagination_provider=imagination,
    ).run()

    assert imagination.messages == []
    assert "refinement_goal is required" in _user_text(main.messages[1])


def test_imagination_receives_cumulative_edit_summary_not_transcript() -> None:
    main = RecordingProvider(
        [
            _response(
                1,
                "delta_move",
                delta_xyz_m=[0.01, 0.0, 0.0],
                frame="base",
                refinement_goal="把 TCP 移到目标几何中心",
            ),
            _response(5, "done", success=False),
        ]
    )
    imagination = RecordingProvider(
        [
            _response(
                2,
                "delta_move",
                delta_xyz_m=[0.0, 0.02, 0.0],
                frame="base",
            ),
            _response(
                3,
                "delta_move",
                delta_xyz_m=[0.0, -0.01, 0.0],
                frame="base",
            ),
            _response(4, "finish_imagination", status="ready"),
        ]
    )
    ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        SolidRenderer(),
        imagination_provider=imagination,
    ).run()

    first = _user_text(imagination.messages[0])
    second = _user_text(imagination.messages[1])
    third = _user_text(imagination.messages[2])
    assert "把 TCP 移到目标几何中心" in first
    assert '"total_translation_base_m":[0.01,0.0,0.0]' in first
    assert "quaternion_xyzw" not in first
    assert "Target Gripper：inherit observed opening" in first
    assert '"previous_edit"' in second and '"last_edit"' in second
    assert '"total_translation_base_m":[0.01,0.02,0.0]' in second
    assert '"total_translation_base_m":[0.01,0.01,0.0]' in third
    assert "reason 2" not in second and "reason 3" not in third
    main_review = _user_text(main.messages[1])
    assert "Action Review Edit Summary" in main_review
    assert '"total_translation_base_m":[0.01,0.01,0.0]' in main_review
