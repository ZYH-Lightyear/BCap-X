from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from tests.test_vaw_context_runtime import FakeContextApi
from vaw.agents.contracts import ModelResponse, ToolCall
from vaw.agents.providers.openai import _parse_response
from vaw.agents.providers.text_protocol import TextProtocolProvider
from vaw.context_runtime.history import ContextHistory
from vaw.context_runtime.packet import ContextCompiler
from vaw.context_runtime.protocol import SYSTEM_PROMPT, function_definitions
from vaw.context_runtime.runtime import ContextRunConfig, ContextRuntime, MULTI_CALL_ERROR
from vaw.context_runtime.trace import ContextTraceLogger
from vaw.context_runtime.video import save_episode_videos
from vaw.context_runtime.workspace import ContextWorkspace


def _response(index: int, name: str, **arguments) -> ModelResponse:
    return ModelResponse(
        text=f"thought {index}",
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


class SolidContextRenderer:
    name = "solid-context"

    def __init__(self) -> None:
        self.packets = []
        self.closed = False

    def render(self, packet):
        self.packets.append(packet)
        return np.full((1080, 1440, 3), packet.revision, dtype=np.uint8)

    def close(self):
        self.closed = True


class FakeVideoEnv:
    def __init__(self) -> None:
        self.agentview = [
            np.full((16, 24, 3), value, dtype=np.uint8) for value in (20, 40, 60)
        ]
        self.wrist = [
            np.full((16, 24, 3), value, dtype=np.uint8) for value in (80, 100)
        ]

    def get_video_frames(self, *, clear: bool = False):
        frames = list(self.agentview)
        if clear:
            self.agentview.clear()
        return frames

    def get_wrist_video_frames(self, *, clear: bool = False):
        frames = list(self.wrist)
        if clear:
            self.wrist.clear()
        return frames


def _image_count(messages) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    )


def _assert_protocol_pairs(messages) -> None:
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        ids = [call["id"] for call in message["tool_calls"]]
        replies = []
        scan = index + 1
        while scan < len(messages) and messages[scan].get("role") == "tool":
            replies.append(messages[scan].get("tool_call_id"))
            scan += 1
        assert replies == ids


def test_agent_prompt_is_chinese_and_keeps_wire_identifiers() -> None:
    assert "每轮必须且只能调用一个 Function" in SYSTEM_PROMPT
    assert "唯一的视觉观测" in SYSTEM_PROMPT
    assert "不得只读文字" in SYSTEM_PROMPT
    assert "AGENTVIEW · PRIMARY" in SYSTEM_PROMPT
    assert "GRIPPER-LOCAL" in SYSTEM_PROMPT
    assert "空白表示未观测区域" in SYSTEM_PROMPT
    assert "世界知识只能生成受当前视觉证据约束的假设" in SYSTEM_PROMPT
    assert "不得假定任何 Function 存在默认的下一个 Function" in SYSTEM_PROMPT
    assert "预测所选 Function 的直接物理后果" in SYSTEM_PROMPT
    assert "可逆、小幅、能够获取信息或改善几何关系" in SYSTEM_PROMPT
    assert "candidate 和 Action Proposal 是待验证的几何/动作假设" in SYSTEM_PROMPT
    assert "commit 只执行指定 active Action Proposal" in SYSTEM_PROMPT
    assert "不移动 TCP，也不保证接触或夹持" in SYSTEM_PROMPT
    assert "0=closed、1=open" in SYSTEM_PROMPT
    assert "每次 Function call 前必须输出一条简短的“决策依据”" in SYSTEM_PROMPT
    assert "小幅向上 delta_move" not in SYSTEM_PROMPT
    assert "至少一个未选候选" not in SYSTEM_PROMPT
    definitions = function_definitions()
    assert [item["function"]["name"] for item in definitions] == [
        "inspect",
        "locate_point",
        "propose_grasps",
        "propose_pose",
        "select",
        "delta_move",
        "rotate",
        "commit",
        "open_gripper",
        "close_gripper",
        "done",
    ]


def test_text_protocol_preserves_raw_response_and_provider_reasoning() -> None:
    raw = (
        "决策依据：g2 的竖直 approach 比 g1 留有更大间隙。\n"
        '<tool_call>{"name":"select","arguments":{"candidate_id":"g2"}}</tool_call>'
    )
    inner = RecordingProvider(
        [
            ModelResponse(
                text=raw,
                raw_response_text=raw,
                provider_reasoning="provider-side diagnostic",
            )
        ]
    )

    response = TextProtocolProvider(inner).generate([], [])

    assert response.text.startswith("决策依据：g2")
    assert response.tool_calls[0].args == {"candidate_id": "g2"}
    assert response.raw_response_text == raw
    assert response.provider_reasoning == "provider-side diagnostic"


def test_openai_provider_captures_separate_reasoning_channel() -> None:
    response = _parse_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "visible decision",
                        "reasoning_content": "separate provider reasoning",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )

    assert response.text == "visible decision"
    assert response.raw_response_text == "visible decision"
    assert response.provider_reasoning == "separate provider reasoning"


def test_history_keeps_three_transactions_and_one_current_image() -> None:
    history = ContextHistory("system", "task", max_transactions=3)
    for index in range(5):
        response = _response(index, "inspect", query=f"item {index}")
        history.add_response(response, [{"region_id": f"region{index}"}])

    messages = history.build_messages(
        manifest={"revision": 1},
        context_image=np.zeros((20, 30, 3), dtype=np.uint8),
    )

    assistants = [message for message in messages if message["role"] == "assistant"]
    assert len(assistants) == 3
    assert assistants[0]["tool_calls"][0]["id"] == "call-2"
    assert all(message["content"] is None for message in assistants)
    assert "thought" not in json.dumps(messages)
    assert _image_count(messages) == 1
    _assert_protocol_pairs(messages)


def test_context_runtime_sends_only_current_image_and_bounded_history(tmp_path: Path) -> None:
    provider = RecordingProvider(
        [
            _response(1, "inspect", query="mug"),
            _response(2, "inspect", query="basket"),
            _response(3, "locate_point", query="basket center"),
            _response(4, "inspect", query="handle"),
            _response(5, "done", success=False),
        ]
    )
    workspace = ContextWorkspace(FakeContextApi(), "test task")
    renderer = SolidContextRenderer()
    runtime = ContextRuntime(
        provider,
        workspace,
        renderer,
        compiler=ContextCompiler(),
        config=ContextRunConfig(max_turns=8, history_k=3),
        trace=ContextTraceLogger(tmp_path),
        env_check=lambda: True,
    )

    result = runtime.run()

    assert result.ended_by_agent
    assert len(provider.messages) == 5
    assert all(_image_count(messages) == 1 for messages in provider.messages)
    assert [
        len([message for message in messages if message["role"] == "assistant"])
        for messages in provider.messages
    ] == [0, 1, 2, 3, 3]
    for messages in provider.messages:
        _assert_protocol_pairs(messages)
        encoded = json.dumps(messages).lower()
        assert "env_success" not in encoded
        assert "depth" not in encoded
        assert "intrinsics" not in encoded
        assert "receipt_id" not in encoded
        assert "context_schema" not in encoded
        assert not any(
            message.get("content", "").startswith("thought ")
            for message in messages
            if message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        )
    assert len(list(tmp_path.glob("context_*.png"))) == 5
    assert json.loads((tmp_path / "meta.json").read_text())["env_success"] is True
    trace_records = [
        json.loads(line)
        for line in (tmp_path / "steps.jsonl").read_text().splitlines()
    ]
    for record in trace_records:
        assert record["decision_mode"] in {
            "idle", "grounding", "candidates", "proposal", "receipt", "error", "terminal"
        }
        assert record["context_manifest"]["revision"] == record["context_packet"][
            "revision"
        ]
        assert record["result_manifest"]["revision"] == record["revision_after"]
        assert record["decision_basis"] == record["thought"]
        assert "execution_receipt" in record
        assert "runtime_diagnostics" in record
        assert "raw_response_text" in record
        assert "provider_reasoning" in record


def test_physical_receipt_is_trace_only(tmp_path: Path) -> None:
    provider = RecordingProvider(
        [_response(1, "open_gripper"), _response(2, "done", success=False)]
    )
    runtime = ContextRuntime(
        provider,
        ContextWorkspace(FakeContextApi(), "task"),
        SolidContextRenderer(),
        trace=ContextTraceLogger(tmp_path),
        config=ContextRunConfig(max_turns=3),
    )

    runtime.run()

    second_request = json.dumps(provider.messages[1]).lower()
    assert "receipt_id" not in second_request
    assert "context_schema" not in second_request
    current_context_text = provider.messages[1][-1]["content"][0]["text"]
    assert '"revision":2' in current_context_text
    records = [
        json.loads(line)
        for line in (tmp_path / "steps.jsonl").read_text().splitlines()
    ]
    receipt = records[0]["execution_receipt"]
    assert receipt["receipt_id"] == "receipt1"
    assert receipt["revision_before"] == 1
    assert receipt["revision_after"] == 2


def test_episode_videos_include_simulator_views_and_context_timeline(
    tmp_path: Path,
) -> None:
    for index, value in enumerate((120, 140)):
        Image.fromarray(np.full((18, 26, 3), value, dtype=np.uint8)).save(
            tmp_path / f"context_{index:04d}.png"
        )
    env = FakeVideoEnv()

    result = save_episode_videos(
        tmp_path,
        env,
        environment_fps=30,
        context_fps=2,
    )

    assert result == {
        "artifacts": {
            "agentview": {"path": "video_agentview.mp4", "fps": 30, "frames": 3},
            "wrist": {"path": "video_wrist.mp4", "fps": 30, "frames": 2},
            "context": {"path": "video_context.mp4", "fps": 2, "frames": 2},
        }
    }
    assert env.agentview == []
    assert env.wrist == []
    for artifact in result["artifacts"].values():
        path = tmp_path / artifact["path"]
        assert path.exists()
        assert path.stat().st_size > 0


def test_multiple_calls_execute_nothing_and_return_every_protocol_reply() -> None:
    multi = ModelResponse(
        tool_calls=(
            ToolCall(id="one", name="inspect", args={"query": "mug"}),
            ToolCall(id="two", name="inspect", args={"query": "basket"}),
        )
    )
    provider = RecordingProvider([multi, _response(2, "done", success=False)])
    workspace = ContextWorkspace(FakeContextApi(), "task")
    runtime = ContextRuntime(
        provider,
        workspace,
        SolidContextRenderer(),
        config=ContextRunConfig(max_turns=3),
    )

    runtime.run()

    assert workspace.state.regions == {}
    transaction = runtime.history.transactions[0]
    assert len(transaction.tool_messages) == 2
    assert all(MULTI_CALL_ERROR in message["content"] for message in transaction.tool_messages)
