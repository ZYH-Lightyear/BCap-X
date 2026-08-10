from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.test_vaw_context_runtime import FakeContextApi
from vaw.agents.contracts import ModelResponse, ToolCall
from vaw.context_runtime.runtime import ContextRunConfig, ContextRuntime
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
    assert "Refinement Goal：设置闭合目标并检查两指通道" in json.dumps(
        imagination.messages[0], ensure_ascii=False
    )
    assert "reason 1" not in json.dumps(imagination.messages[0], ensure_ascii=False)
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


def test_last_physical_action_is_visible_for_one_valid_main_decision() -> None:
    main = RecordingProvider(
        [
            _response(
                1,
                "open_gripper",
                refinement_goal="只设置真实执行后的张开目标",
            ),
            _response(3, "commit", action_id="a1"),
            ModelResponse(text="no function this time", tool_calls=()),
            _response(5, "detection_and_sam", query="can"),
            _response(6, "done", success=False),
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
    assert "Last Physical Action" not in _user_text(main.messages[4])
    assert all(_image_count(messages) == 1 for messages in main.messages)


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
    assert '"previous_edit"' in second and '"last_edit"' in second
    assert '"total_translation_base_m":[0.01,0.02,0.0]' in second
    assert '"total_translation_base_m":[0.01,0.01,0.0]' in third
    assert "reason 2" not in second and "reason 3" not in third
