from capx.llm.client import ModelQueryArgs, _build_chat_payload


def test_vapi_reasoning_model_uses_completion_budget() -> None:
    args = ModelQueryArgs(
        model="vapi/gpt-5.5",
        server_url="http://localhost:8110/chat/completions",
        max_tokens=123,
        reasoning_effort="low",
    )

    payload = _build_chat_payload(args, [{"role": "user", "content": "hello"}])

    assert payload["model"] == "vapi/gpt-5.5"
    assert payload["max_completion_tokens"] == 123
    assert payload["reasoning_effort"] == "low"
    assert "max_tokens" not in payload


def test_vapi_non_reasoning_model_keeps_chat_budget() -> None:
    args = ModelQueryArgs(
        model="vapi/gemini-2.5-pro",
        server_url="http://localhost:8110/chat/completions",
        max_tokens=456,
        reasoning_effort="low",
    )

    payload = _build_chat_payload(args, [{"role": "user", "content": "hello"}])

    assert payload["model"] == "vapi/gemini-2.5-pro"
    assert payload["max_tokens"] == 456
    assert "max_completion_tokens" not in payload
    assert "reasoning_effort" not in payload


def test_openrouter_model_keeps_openrouter_payload_surface() -> None:
    args = ModelQueryArgs(
        model="openrouter/qwen/qwen3.6-plus",
        server_url="http://localhost:8110/chat/completions",
        max_tokens=789,
        temperature=0.3,
        reasoning_effort="low",
    )

    payload = _build_chat_payload(args, [{"role": "user", "content": "hello"}])

    assert payload["model"] == "openrouter/qwen/qwen3.6-plus"
    assert payload["max_tokens"] == 789
    assert payload["temperature"] == 0.3
    assert "max_completion_tokens" not in payload
    assert "reasoning_effort" not in payload
