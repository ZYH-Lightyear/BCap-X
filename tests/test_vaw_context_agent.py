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
        return np.full((1440, 1920, 3), packet.revision, dtype=np.uint8)

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


def test_dual_agent_runtime_rebuilds_every_request_without_history(tmp_path: Path) -> None:
    main = RecordingProvider(
        [
            _response(1, "close_gripper"),
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
    assert _image_count(main.messages[2]) == 2
    assert "Refinement Goal：reason 1" in json.dumps(
        imagination.messages[0], ensure_ascii=False
    )
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


def test_imagination_turn_limit_returns_control_to_main() -> None:
    main = RecordingProvider(
        [_response(1, "open_gripper"), _response(4, "done", success=False)]
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
    ).run()

    assert result.turns == 4
    assert "Latest Imagination Handoff" in json.dumps(
        main.messages[1], ensure_ascii=False
    )
    assert "budget_exhausted" in json.dumps(main.messages[1], ensure_ascii=False)


def test_invalid_imagination_tool_is_one_shot_feedback_not_history() -> None:
    main = RecordingProvider([_response(1, "open_gripper"), _response(4, "done", success=False)])
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
