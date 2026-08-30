from __future__ import annotations

import json

from vaw.agents.contracts import ModelResponse
from vaw.agents.providers.openai import OpenAIProvider
from vaw.agents.providers.text_protocol import TextProtocolProvider
from vaw.context_runtime.protocol import (
    MAIN_FUNCTION_NAMES,
    main_function_definitions,
)
from vaw.context_runtime.trace_read import canonical_trace_function


class _HttpResponse:
    status_code = 200
    text = ""

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "目标区域清晰。",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "detect_region",
                                    "arguments": json.dumps(
                                        {"query": "红色杯子"}, ensure_ascii=False
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }


def test_native_provider_forwards_canonical_tools_without_prompt_rewriting(
    monkeypatch,
) -> None:
    captured: dict = {}

    def fake_post(url, *, json, headers, timeout):
        captured.update(url=url, payload=json, headers=headers, timeout=timeout)
        return _HttpResponse()

    monkeypatch.setattr("vaw.agents.providers.openai.requests.post", fake_post)
    tools = main_function_definitions()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]

    response = OpenAIProvider(model="local/qwen3.5-27b").generate(messages, tools)

    assert captured["payload"]["messages"] == messages
    assert captured["payload"]["tools"] == tools
    assert captured["payload"]["tool_choice"] == "auto"
    assert captured["payload"]["parallel_tool_calls"] is False
    assert response.tool_calls[0].name == "detect_region"
    assert response.tool_calls[0].args == {"query": "红色杯子"}


class _RecordingCompletion:
    def __init__(self, text: str | None = None) -> None:
        self.messages = None
        self.tools = "unset"
        self.text = text or (
            "目标区域清晰。\n"
            '<tool_call>{"name":"detect_region","arguments":'
            '{"query":"红色杯子"}}</tool_call>'
        )

    def generate(self, messages, tools=None) -> ModelResponse:
        self.messages = messages
        self.tools = tools
        return ModelResponse(text=self.text, raw_response_text=self.text)


def test_text_protocol_is_an_explicit_compact_fallback() -> None:
    inner = _RecordingCompletion()
    provider = TextProtocolProvider(inner)
    response = provider.generate(
        [{"role": "system", "content": "system"}],
        main_function_definitions(),
    )

    assert inner.tools is None
    rendered = str(inner.messages[0]["content"])
    assert "<tools>" in rendered
    assert "## detect_region" not in rendered
    assert len(rendered) < 5_000
    assert response.text == "目标区域清晰。"
    assert response.tool_calls[0].name == "detect_region"


def test_text_protocol_leaves_no_tool_bootstrap_request_untouched() -> None:
    inner = _RecordingCompletion('{"selected_card_ids":["container-placement"]}')
    provider = TextProtocolProvider(inner)
    messages = [
        {"role": "system", "content": "任务知识引导选择器"},
        {"role": "user", "content": "task + canvas"},
    ]

    response = provider.generate(messages, tools=None)

    assert inner.messages == messages
    assert inner.tools is None
    assert response.text == '{"selected_card_ids":["container-placement"]}'
    assert response.tool_calls == ()


def test_public_function_catalog_is_small_and_unambiguous() -> None:
    definitions = main_function_definitions()
    names = [item["function"]["name"] for item in definitions]

    assert names == list(MAIN_FUNCTION_NAMES)
    assert len(names) == len(set(names))
    assert max(len(item["function"]["description"]) for item in definitions) < 40
    assert len(json.dumps(definitions, ensure_ascii=False)) < 4_500
    assert "detection_and_sam" not in names
    assert "commit" not in names


def test_historical_function_names_are_normalized_only_for_trace_reading() -> None:
    assert canonical_trace_function("commit") == "execute_action"
    assert canonical_trace_function("delta_move") == "move_tcp_delta"
    assert (
        canonical_trace_function("delta_move", owner="imagination")
        == "shift_preview"
    )
    assert canonical_trace_function("future_unknown") == "future_unknown"
