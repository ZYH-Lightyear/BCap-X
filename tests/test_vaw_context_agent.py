from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.test_vaw_context_runtime import FakeContextApi
from vaw.agents.contracts import ModelResponse, ToolCall
from vaw.context_runtime.protocol import (
    IMAGINATION_FUNCTION_NAMES,
    MAIN_FUNCTION_NAMES,
)
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
        self.messages: list = []
        self.tools: list = []

    def generate(self, messages, tools=None):
        self.messages.append(messages)
        self.tools.append(tools)
        return self.responses.pop(0)


class RecordingRenderer:
    name = "test-renderer"

    def __init__(self) -> None:
        self.projections: list[str] = []

    def render(self, packet):
        self.projections.append(packet.projection)
        return np.full((1280, 2048, 3), packet.revision, dtype=np.uint8)

    def close(self):
        return None


def _user_text(messages) -> str:
    return str(messages[1]["content"][0]["text"])


def _image_count(messages) -> int:
    return sum(
        part.get("type") == "image_url"
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict)
    )


def _tool_names(definitions) -> list[str]:
    return [item["function"]["name"] for item in definitions]


def test_main_react_calls_imagination_as_one_nested_function(tmp_path: Path) -> None:
    main = RecordingProvider(
        [
            _response(1, "detection_and_sam", query="can"),
            _response(2, "propose_grasps", region_id="region1"),
            _response(3, "select", seed_id="s1"),
            _response(
                4,
                "refine_action",
                action_id="a1",
                instruction="两指对称包夹罐体，保持张开",
            ),
            _response(5, "commit", action_id="a1"),
            _response(6, "done", success=False),
        ]
    )
    imagination = RecordingProvider(
        [
            _response(1, "delta_move", delta_xyz_m=[0.0, 0.0, -0.01], frame="base"),
            _response(2, "finish_imagination", status="ready"),
        ]
    )
    renderer = RecordingRenderer()
    runtime = ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "test task", motion_backend="pyroki"),
        renderer,
        imagination_provider=imagination,
        config=ContextRunConfig(max_main_turns=8, max_imagination_turns=4),
        trace=ContextTraceLogger(tmp_path),
    )

    result = runtime.run()

    assert result.turns == 6
    assert len(main.messages) == 6
    assert len(imagination.messages) == 2
    assert all(_tool_names(tools) == list(MAIN_FUNCTION_NAMES) for tools in main.tools)
    assert all(
        _tool_names(tools) == list(IMAGINATION_FUNCTION_NAMES)
        for tools in imagination.tools
    )
    assert all(_image_count(messages) == 1 for messages in main.messages)
    assert all(_image_count(messages) == 1 for messages in imagination.messages)
    assert all(
        "Carried Geometry：unavailable_use_gripper_only_fallback"
        in _user_text(messages)
        for messages in imagination.messages
    )
    assert renderer.projections.count("imagination") == 2
    assert renderer.projections.count("main") == 6

    # Internal edit calls live only in a nested trace, not Main's top-level log.
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [row["function_call"]["name"] for row in rows] == [
        "detection_and_sam",
        "propose_grasps",
        "select",
        "refine_action",
        "commit",
        "done",
    ]
    nested = tmp_path / "subagents" / "imagination_0001" / "steps.jsonl"
    nested_rows = [json.loads(line) for line in nested.read_text().splitlines()]
    assert [row["function_call"]["name"] for row in nested_rows] == [
        "delta_move",
        "finish_imagination",
    ]
    refine_result = rows[3]["function_result"]
    assert refine_result == {"status": "ready", "action_id": "a1"}
    assert "subtrace" not in _user_text(main.messages[4])


def test_refinement_requires_an_explicit_live_action() -> None:
    main = RecordingProvider(
        [
            _response(
                1,
                "refine_action",
                instruction="从当前 TCP 向上移动 1cm",
            ),
            _response(2, "done", success=False),
        ]
    )
    imagination = RecordingProvider([])
    api = FakeContextApi()

    result = ContextRuntime(
        main,
        ContextWorkspace(api, "task", motion_backend="pyroki"),
        RecordingRenderer(),
        imagination_provider=imagination,
    ).run()

    assert [step.op for step in result.steps] == [
        "main:refine_action",
        "main:done",
    ]
    assert not result.steps[0].ok
    assert imagination.messages == []
    assert api.operation_log == []


def test_imagination_limit_retires_action_and_returns_one_clean_failure(tmp_path: Path) -> None:
    main = RecordingProvider(
        [
            _response(1, "refine_action", instruction="检查旋转方向"),
            _response(2, "done", success=False),
        ]
    )
    imagination = RecordingProvider(
        [
            _response(1, "show_rotation_gizmo", frame="base", axis="z"),
            _response(2, "show_rotation_gizmo", frame="tool", axis="y"),
        ]
    )
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    action_id = workspace.execute("select", seed_id=seed).result["action_id"]
    main.responses[0] = _response(
        1,
        "refine_action",
        action_id=action_id,
        instruction="检查旋转方向",
    )

    ContextRuntime(
        main,
        workspace,
        RecordingRenderer(),
        imagination_provider=imagination,
        config=ContextRunConfig(max_main_turns=4, max_imagination_turns=2),
        trace=ContextTraceLogger(tmp_path),
    ).run()

    assert workspace.state.refinement is None
    assert workspace.state.pending_action is None
    assert '"status":"failed"' in _user_text(main.messages[1])
    assert "show_rotation_gizmo" not in _user_text(main.messages[1])


def test_main_context_is_rebuilt_without_transcript_history() -> None:
    main = RecordingProvider(
        [
            _response(1, "detection_and_sam", query="can"),
            _response(2, "propose_grasps", region_id="region1"),
            _response(3, "done", success=False),
        ]
    )

    ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        RecordingRenderer(),
    ).run()

    assert all([message["role"] for message in messages] == ["system", "user"] for messages in main.messages)
    assert "Task Memory" in _user_text(main.messages[1])
    assert "Live References" in _user_text(main.messages[1])
    assert "Current Function Event" in _user_text(main.messages[1])
    assert '"function":"detection_and_sam"' in _user_text(main.messages[1])
    assert '"function":"propose_grasps"' in _user_text(main.messages[2])
    assert '"function":"detection_and_sam"' not in _user_text(main.messages[2])
    assert "Last Function Outcome" not in _user_text(main.messages[1])
    assert "receipt" not in _user_text(main.messages[1]).lower()


def test_task_memory_survives_perception_while_current_event_is_overwritten() -> None:
    main = RecordingProvider(
        [
            _response(1, "open_gripper"),
            _response(2, "detection_and_sam", query="can"),
            _response(3, "done", success=False),
        ]
    )
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")

    ContextRuntime(main, workspace, RecordingRenderer()).run()

    after_open = _user_text(main.messages[1])
    after_detection = _user_text(main.messages[2])
    assert '"op":"open_gripper","status":"executed"' in after_open
    assert '"function":"open_gripper"' in after_open
    assert '"op":"open_gripper","status":"executed"' in after_detection
    assert '"function":"detection_and_sam"' in after_detection
    assert '"function":"open_gripper"' not in after_detection
    assert "Last Physical Action" not in after_open
    assert "Physical Effect Verification" not in after_open
    assert workspace.state.last_physical_action is not None


def test_runtime_stops_after_environment_terminates_during_direct_gripper_action() -> None:
    main = RecordingProvider(
        [
            _response(1, "open_gripper"),
            _response(2, "done", success=False),
        ]
    )
    terminated = False
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
        RecordingRenderer(),
        env_terminal_check=lambda: terminated,
    ).run()

    assert result.terminate_mode.value == "env_terminated"
    assert result.turns == 1
    assert [step.op for step in result.steps] == ["main:open_gripper"]
