from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from capx_skill_rl.env import ToolEnv
from capx_skill_rl.guard import ModelActionGuard
from capx_skill_rl.loop import ToolExchange
from capx_skill_rl.policies import ChatCompletionsPolicy, ChatPolicyConfig
from capx_skill_rl.policies.chat_completions import _parse_tool_calls
from capx_skill_rl.scripts.model_rollout import run_model_rollout
from capx_skill_rl.tests.fakes import FakeBackend


class _Response:
    status_code = 200
    text = ""

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body

    def json(self) -> dict[str, Any]:
        return self.body


def _body(*calls: tuple[str, str]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            },
                        }
                        for index, (name, arguments) in enumerate(calls)
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def test_chat_policy_sends_current_rgb_and_parses_one_native_tool_call() -> None:
    captured: dict[str, Any] = {}

    def request_fn(url: str, **kwargs: Any) -> _Response:
        captured["url"] = url
        captured.update(kwargs)
        return _Response(
            _body(
                (
                    "vlm_point_detection",
                    '{"query":"alphabet soup can"}',
                )
            )
        )

    policy = ChatCompletionsPolicy(
        ChatPolicyConfig(),
        request_fn=request_fn,
    )
    action = policy.act(
        task="pick the object",
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        history=(
            ToolExchange(
                action={"name": "go_home", "arguments": {}},
                result={},
            ),
        ),
        tools=ToolEnv(FakeBackend()).tool_definitions,
    )

    assert action == {
        "name": "vlm_point_detection",
        "arguments": {"query": "alphabet soup can"},
    }
    payload = captured["json"]
    assert "tool_choice" not in payload
    assert payload["parallel_tool_calls"] is False
    image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    assert "reward" not in payload["messages"][1]["content"][0]["text"]
    assert "depth" not in payload["messages"][1]["content"][0]["text"].lower()
    assert "base64" not in json.dumps(policy.records)


def test_multiple_or_missing_tool_calls_become_invalid_environment_actions() -> None:
    multiple, multiple_error = _parse_tool_calls(
        _body(
            ("open_gripper", "{}"),
            ("go_home", "{}"),
        )
    )
    missing, missing_error = _parse_tool_calls({"choices": [{"message": {"content": "I am done"}}]})

    assert isinstance(multiple, list)
    assert multiple_error == "model returned 2 tool calls"
    assert missing == {"name": "", "arguments": {}}
    assert missing_error == "model returned no tool call"

    env = ToolEnv(FakeBackend(), max_steps=2)
    env.reset()
    assert "error" in env.step(multiple).result
    assert "error" in env.step(missing).result


def test_model_guard_requires_ik_joint_provenance() -> None:
    guard = ModelActionGuard()
    joints = [float(index) for index in range(7)]
    move = {"name": "move_to_joints", "arguments": {"joints": joints}}
    solve = ToolExchange(
        action={
            "name": "solve_ik",
            "arguments": {
                "position": [0.4, 0.0, 0.2],
                "quaternion": [0.0, 0.0, 0.0, 1.0],
            },
        },
        result={"joints": joints},
    )

    assert guard.validate(move, ()) is not None
    assert guard.validate(move, (solve,)) is None
    changed = {"name": "move_to_joints", "arguments": {"joints": [0.0] * 7}}
    assert guard.validate(changed, (solve,)) is not None
    out_of_workspace_ik = {
        "name": "solve_ik",
        "arguments": {
            "position": [2.0, 0.0, 0.2],
            "quaternion": [0.0, 0.0, 0.0, 1.0],
        },
    }
    assert guard.validate(out_of_workspace_ik, ()) is None


class _ShadowPolicy:
    config = ChatPolicyConfig()

    def __init__(self) -> None:
        self.actions = [
            {
                "name": "vlm_point_detection",
                "arguments": {"query": "object"},
            },
            {"name": "go_home", "arguments": {}},
        ]
        self.records: list[dict[str, Any]] = []

    def act(
        self,
        *,
        task: str,
        rgb: np.ndarray,
        history: Sequence[ToolExchange],
        tools: Sequence[dict[str, Any]],
    ):
        del task, rgb, history, tools
        action = self.actions[len(self.records)]
        self.records.append({"elapsed_seconds": 0.0, "action": action})
        return action


def test_shadow_rollout_stops_before_first_physical_action(tmp_path) -> None:
    backend = FakeBackend()
    report = run_model_rollout(
        ToolEnv(backend),
        _ShadowPolicy(),  # type: ignore[arg-type]
        output_dir=tmp_path,
        seed=3,
        mode="shadow",
    )

    assert report["status"] == "shadow_ready"
    assert report["blocked_action"] == {"name": "go_home", "arguments": {}}
    assert len(report["transitions"]) == 1
    assert backend.capture_count == 0
