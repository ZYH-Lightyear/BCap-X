from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.test_vaw_context_runtime import FakeContextApi
from vaw.agents.contracts import ModelResponse, ToolCall
from vaw.agents.providers.text_protocol import parse_tool_calls
from vaw.context_runtime.protocol import (
    IMAGINATION_FUNCTION_NAMES,
    MAIN_FUNCTION_NAMES,
)
from vaw.context_runtime.runtime import (
    ContextRunConfig,
    ContextRuntime,
    NO_CALL_FEEDBACK,
)
from vaw.context_runtime.trace import ContextTraceLogger
from vaw.context_runtime.workspace import ContextWorkspace


def _response(index: int, name: str, **arguments) -> ModelResponse:
    return ModelResponse(
        text=f"reason {index}",
        tool_calls=(ToolCall(id=f"call-{index}", name=name, args=arguments),),
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _empty_response(text: str = "thinking") -> ModelResponse:
    return ModelResponse(
        text=text,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


class RecordingProvider:
    def __init__(
        self,
        responses: list[ModelResponse],
        *,
        bootstrap_response: ModelResponse | None = None,
    ) -> None:
        self.responses = list(responses)
        self.messages: list = []
        self.tools: list = []
        self.bootstrap_messages: list = []
        self.bootstrap_response = bootstrap_response or ModelResponse(
            text='{"selected_card_ids":[]}',
            raw_response_text='{"selected_card_ids":[]}',
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    def generate(self, messages, tools=None):
        system_text = str(messages[0].get("content", "")) if messages else ""
        if tools is None and "任务知识引导选择器" in system_text:
            self.bootstrap_messages.append(messages)
            return self.bootstrap_response
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
            _response(1, "detect_region", query="can"),
            _response(2, "propose_grasps", region_id="region1"),
            _response(3, "preview_grasp", seed_id="s1"),
            _response(
                4,
                "imagine_action",
                action_id="a1",
                instruction="两指对称包夹罐体，保持张开",
            ),
            _response(5, "execute_action", action_id="a1"),
            _response(6, "finish_task", success=False),
        ]
    )
    imagination = RecordingProvider(
        [
            _response(1, "shift_preview", delta_xyz_m=[0.0, 0.0, -0.01], frame="base"),
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
        "携带物几何：不可用：仅依据夹爪几何判断"
        in _user_text(messages)
        for messages in imagination.messages
    )
    assert renderer.projections.count("imagination") == 2
    assert renderer.projections.count("main") == 6

    # Internal edit calls live only in a nested trace, not Main's top-level log.
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [row["function_call"]["name"] for row in rows] == [
        "detect_region",
        "propose_grasps",
        "preview_grasp",
        "imagine_action",
        "execute_action",
        "finish_task",
    ]
    nested = tmp_path / "subagents" / "imagination_0001" / "steps.jsonl"
    nested_rows = [json.loads(line) for line in nested.read_text().splitlines()]
    assert [row["function_call"]["name"] for row in nested_rows] == [
        "shift_preview",
        "finish_imagination",
    ]
    refine_result = rows[3]["function_result"]
    assert refine_result == {"status": "ready", "action_id": "a1"}
    assert "subtrace" not in _user_text(main.messages[4])
    post_commit = _user_text(main.messages[5])
    assert "机器人本体状态：" in post_commit
    assert "gripper_opening" not in post_commit
    assert '"function":"execute_action"' in post_commit
    assert '"outcome":"completed"' in post_commit
    assert "position_error_m" not in post_commit


def test_imagination_can_start_from_the_current_tcp() -> None:
    main = RecordingProvider(
        [
            _response(
                1,
                "imagine_action",
                instruction="从当前 TCP 向上移动 1cm",
            ),
            _response(2, "finish_task", success=False),
        ]
    )
    imagination = RecordingProvider(
        [
            _response(1, "shift_preview", delta_xyz_m=[0.0, 0.0, 0.01], frame="base"),
            _response(2, "finish_imagination", status="ready"),
        ]
    )
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")

    result = ContextRuntime(
        main,
        workspace,
        RecordingRenderer(),
        imagination_provider=imagination,
        config=ContextRunConfig(max_main_turns=4, max_imagination_turns=4),
    ).run()

    assert [step.op for step in result.steps] == [
        "main:imagine_action",
        "main:finish_task",
    ]
    assert result.steps[0].ok
    assert json.loads(result.steps[0].result)["status"] == "ready"
    assert workspace.state.action_proposal is not None
    assert workspace.state.action_proposal.intent == "refine from current TCP"
    assert len(imagination.messages) == 2


def test_a_crashing_turn_observer_cannot_kill_the_episode() -> None:
    main = RecordingProvider(
        [
            _response(1, "detect_region", query="can"),
            _response(2, "finish_task", success=False),
        ]
    )
    seen: list[int] = []

    def exploding_observer(turn: int, record: Any) -> None:
        seen.append(turn)
        # A print into a non-blocking stdout pipe raises exactly this.
        raise BlockingIOError(11, "write could not complete without blocking")

    result = ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        RecordingRenderer(),
        config=ContextRunConfig(max_main_turns=4, on_turn=exploding_observer),
    ).run()

    assert [step.op for step in result.steps] == [
        "main:detect_region",
        "main:finish_task",
    ]
    assert seen == [1, 2]


def test_imagination_limit_hands_the_partial_state_back_to_main(tmp_path: Path) -> None:
    main = RecordingProvider(
        [
            _response(1, "imagine_action", instruction="检查旋转方向"),
            _response(2, "finish_task", success=False),
        ]
    )
    imagination = RecordingProvider(
        [
            _response(1, "inspect_rotation", frame="base", axis="z"),
            _response(2, "inspect_rotation", frame="tool", axis="y"),
        ]
    )
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region = workspace.execute("detect_region", query="can").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    action_id = workspace.execute("preview_grasp", seed_id=seed).result["action_id"]
    main.responses[0] = _response(
        1,
        "imagine_action",
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

    assert workspace.state.imagination is None
    # The budget-exhausted session hands its planner-checked state back for
    # Main's Preview review instead of rolling the proposal back.
    assert workspace.state.action_proposal.action_id == action_id
    assert workspace.state.action_proposal.refined is True
    assert "imagine_action" in _user_text(main.messages[1])
    assert "partial (reason=turn_limit" in _user_text(main.messages[1])
    assert "review the Preview" in _user_text(main.messages[1])
    assert "inspect_rotation" not in _user_text(main.messages[1])
    meta = json.loads(
        (tmp_path / "subagents" / "imagination_0001" / "meta.json").read_text()
    )
    assert meta["status"] == "partial"
    assert meta["reason"] == "turn_limit"
    assert meta["instruction"] == "检查旋转方向"


def test_main_context_is_rebuilt_without_transcript_history() -> None:
    main = RecordingProvider(
        [
            _response(1, "detect_region", query="can"),
            _response(2, "propose_grasps", region_id="region1"),
            _response(3, "finish_task", success=False),
        ]
    )

    ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        RecordingRenderer(),
    ).run()

    assert all([message["role"] for message in messages] == ["system", "user"] for messages in main.messages)
    assert "用户任务：task" in _user_text(main.messages[1])
    assert "当前目标：" not in _user_text(main.messages[1])
    assert "当前有效引用" in _user_text(main.messages[1])
    assert "机器人本体状态" in _user_text(main.messages[1])
    assert "短期交互记忆" in _user_text(main.messages[1])
    assert "t1 [函数调用] detect_region" in _user_text(main.messages[1])
    assert "t2 [函数调用] propose_grasps" in _user_text(main.messages[2])
    assert "t1 [函数调用] detect_region" in _user_text(main.messages[2])
    assert "Task Memory" not in _user_text(main.messages[1])
    assert "Current Function Event" not in _user_text(main.messages[1])
    assert "Control Continuity" not in _user_text(main.messages[1])
    assert "receipt" not in _user_text(main.messages[1]).lower()


def test_interaction_memory_keeps_physical_and_perception_calls_in_one_timeline() -> None:
    main = RecordingProvider(
        [
            _response(1, "open_gripper"),
            _response(2, "detect_region", query="can"),
            _response(3, "finish_task", success=False),
        ]
    )
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")

    ContextRuntime(main, workspace, RecordingRenderer()).run()

    after_open = _user_text(main.messages[1])
    after_detection = _user_text(main.messages[2])
    assert "t1 [物理动作] open_gripper({}) -> 结果=completed" in after_open
    assert "t1 [物理动作] open_gripper({}) -> 结果=completed" in after_detection
    assert 't2 [函数调用] detect_region({"query":"can"}) -> 结果=ok' in after_detection
    assert "grasped" not in after_detection
    assert "placed" not in after_detection
    assert workspace.state.last_physical_action is not None


def test_runtime_stops_after_environment_terminates_during_direct_gripper_action() -> None:
    main = RecordingProvider(
        [
            _response(1, "open_gripper"),
            _response(2, "finish_task", success=False),
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


def test_imagination_provider_error_is_classified_as_subagent_error(tmp_path: Path) -> None:
    class ExplodingProvider:
        def generate(self, messages, tools=None):
            del messages, tools
            raise RuntimeError("provider down")

    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region = workspace.execute("detect_region", query="can").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    action_id = workspace.execute("preview_grasp", seed_id=seed).result["action_id"]
    original = workspace.state.action_proposal
    main = RecordingProvider(
        [
            _response(1, "imagine_action", action_id=action_id, instruction="检查"),
            _response(2, "finish_task", success=False),
        ]
    )

    ContextRuntime(
        main,
        workspace,
        RecordingRenderer(),
        imagination_provider=ExplodingProvider(),
        config=ContextRunConfig(max_main_turns=4, max_imagination_turns=2),
        trace=ContextTraceLogger(tmp_path),
    ).run()

    assert workspace.state.action_proposal is original
    assert "reason=subagent_error" in _user_text(main.messages[1])
    meta = json.loads(
        (tmp_path / "subagents" / "imagination_0001" / "meta.json").read_text()
    )
    assert meta["status"] == "failed"
    assert meta["reason"] == "subagent_error"
    assert "RuntimeError" not in _user_text(main.messages[1])

def test_runtime_budgets_are_profile_values_without_arbitrary_upper_caps() -> None:
    assert ContextRunConfig().max_main_turns == 32
    assert ContextRunConfig(max_main_turns=65).max_main_turns == 65
    assert ContextRunConfig().max_no_call_retries == 2
    assert ContextRunConfig(max_no_call_retries=4).max_no_call_retries == 4


def test_no_call_retries_share_one_main_turn(tmp_path: Path) -> None:
    main = RecordingProvider(
        [
            _empty_response("hesitate 1"),
            _empty_response("hesitate 2"),
            _response(1, "detect_region", query="can"),
            _response(2, "finish_task", success=False),
        ]
    )

    result = ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        RecordingRenderer(),
        config=ContextRunConfig(max_main_turns=4),
        trace=ContextTraceLogger(tmp_path),
    ).run()

    assert result.turns == 2
    assert [step.op for step in result.steps] == [
        "main:detect_region",
        "main:finish_task",
    ]
    assert len(main.messages) == 4
    assert "协议反馈" not in _user_text(main.messages[0])
    assert f"协议反馈：{NO_CALL_FEEDBACK}" in _user_text(main.messages[1])
    assert f"协议反馈：{NO_CALL_FEEDBACK}" in _user_text(main.messages[2])
    rows = [
        json.loads(line)
        for line in (tmp_path / "steps.jsonl").read_text().splitlines()
    ]
    assert [row["function_call"]["name"] for row in rows] == [
        "detect_region",
        "finish_task",
    ]
    events = [
        json.loads(line)
        for line in (tmp_path / "runtime_events.jsonl").read_text().splitlines()
    ]
    retries = [event for event in events if event["event_type"] == "no_call_retry"]
    assert [event["attempt"] for event in retries] == [1, 2]
    assert all(event["turn"] == 1 for event in retries)
    assert result.usage["main_total_tokens"] == 8


def test_exhausted_no_call_retries_consume_one_failed_turn(tmp_path: Path) -> None:
    main = RecordingProvider(
        [
            _empty_response("a"),
            _empty_response("b"),
            _empty_response("c"),
            _response(1, "finish_task", success=False),
        ]
    )

    result = ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        RecordingRenderer(),
        config=ContextRunConfig(max_main_turns=4),
        trace=ContextTraceLogger(tmp_path),
    ).run()

    assert result.turns == 2
    assert [step.op for step in result.steps] == ["main:finish_task"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "steps.jsonl").read_text().splitlines()
    ]
    assert rows[0]["function_call"] is None
    assert rows[0]["turn"] == 1
    assert rows[1]["function_call"]["name"] == "finish_task"
    assert "protocol" not in _user_text(main.messages[3])
    assert "短期交互记忆：\n（空）" in _user_text(main.messages[3])


def test_zero_no_call_retries_keeps_legacy_burn(tmp_path: Path) -> None:
    main = RecordingProvider(
        [
            _empty_response("silence"),
            _response(1, "finish_task", success=False),
        ]
    )

    result = ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        RecordingRenderer(),
        config=ContextRunConfig(max_main_turns=4, max_no_call_retries=0),
        trace=ContextTraceLogger(tmp_path),
    ).run()

    assert result.turns == 2
    assert len(main.messages) == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "steps.jsonl").read_text().splitlines()
    ]
    assert rows[0]["function_call"] is None
    assert rows[1]["function_call"]["name"] == "finish_task"
    events = (tmp_path / "runtime_events.jsonl").read_text().strip()
    assert events
    event_rows = [json.loads(line) for line in events.splitlines()]
    assert any(row["event_type"] == "no_call_retry" for row in event_rows)


def test_text_protocol_preserves_reason_before_parsing_the_call() -> None:
    prose, calls = parse_tool_calls(
        "The grasp looks stable.\n"
        '<tool_call>{"name":"move_tcp_delta","arguments":'
        '{"delta_xyz_m":[0,0,0.02],"frame":"base"}}</tool_call>'
    )

    assert prose == "The grasp looks stable."
    assert len(calls) == 1
    assert calls[0].name == "move_tcp_delta"


def test_main_context_and_trace_have_no_model_authored_goal(tmp_path: Path) -> None:
    main = RecordingProvider(
        [
            _response(1, "detect_region", query="can"),
            _response(2, "finish_task", success=False),
        ]
    )

    ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "pick the can", motion_backend="pyroki"),
        RecordingRenderer(),
        trace=ContextTraceLogger(tmp_path),
    ).run()

    assert all("当前目标：" not in _user_text(messages) for messages in main.messages)
    assert all("context_update" not in str(messages) for messages in main.messages)
    frozen = json.loads(
        (tmp_path / "contexts" / "turn_0001" / "context.json").read_text()
    )
    assert frozen["schema"] == "vaw-agent-context-v2-no-goal"
    assert "current_goal" not in frozen
    rows = [
        json.loads(line)
        for line in (tmp_path / "steps.jsonl").read_text().splitlines()
    ]
    for row in rows:
        assert "current_goal_before" not in row
        assert "current_goal_after" not in row
        assert "context_update" not in row
        assert "context_update_error" not in row


def test_identical_episode_transactions_replay_to_the_same_context_memory() -> None:
    def run_once() -> tuple[list[dict], dict]:
        provider = RecordingProvider(
            [
                _response(1, "open_gripper"),
                _response(2, "detect_region", query="can"),
                _response(3, "finish_task", success=False),
            ]
        )
        runtime = ContextRuntime(
            provider,
            ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
            RecordingRenderer(),
        )
        runtime.run()
        return (
            runtime.interaction_memory.snapshot(),
            runtime._embodied_state(decision_turn=4).summary(),
        )

    assert run_once() == run_once()
