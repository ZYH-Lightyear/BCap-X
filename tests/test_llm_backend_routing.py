from capx.llm import client as llm_client
from capx.llm.client import ModelQueryArgs, _build_chat_payload
from capx.llm.routing import RouteProfile, normalize_proxy_payload


def test_client_payload_is_provider_neutral() -> None:
    args = ModelQueryArgs(
        model="vapi/gpt-5.5",
        server_url="http://localhost:8110/chat/completions",
        max_tokens=123,
        reasoning_effort="low",
    )

    payload = _build_chat_payload(args, [{"role": "user", "content": "hello"}])

    assert payload["model"] == "vapi/gpt-5.5"
    assert payload["max_tokens"] == 123
    assert payload["reasoning_effort"] == "low"
    assert "max_completion_tokens" not in payload


def test_proxy_route_normalizes_openrouter_request() -> None:
    payload = {
        "model": "openrouter/qwen/qwen3.6-plus",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 456,
    }
    profile = RouteProfile(
        reasoning_effort="low",
        token_field="auto",
        reasoning_style="openrouter",
    )

    kwargs, extra_body = normalize_proxy_payload(
        payload,
        profile=profile,
        native_model="qwen/qwen3.6-plus",
    )

    assert kwargs["model"] == "qwen/qwen3.6-plus"
    assert kwargs["max_tokens"] == 456
    assert "reasoning_effort" not in kwargs
    assert extra_body == {"reasoning": {"effort": "low"}}


def test_proxy_route_normalizes_openai_reasoning_request() -> None:
    payload = {
        "model": "vapi/gpt-5.5",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 789,
        "temperature": 0.3,
        "reasoning_effort": "low",
    }
    profile = RouteProfile(
        reasoning_effort="none",
        token_field="auto",
        reasoning_style="openai",
    )

    kwargs, extra_body = normalize_proxy_payload(
        payload,
        profile=profile,
        native_model="gpt-5.5",
    )

    assert kwargs["max_completion_tokens"] == 789
    assert kwargs["reasoning_effort"] == "low"
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs
    assert extra_body is None


def test_query_model_uses_configured_endpoint_for_openrouter(monkeypatch) -> None:
    captured = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(server_url, headers, payload):
        captured["server_url"] = server_url
        captured["payload"] = payload
        return Response()

    monkeypatch.setattr(llm_client, "_post_with_retries", fake_post)
    endpoint = "http://proxy.example.test/chat/completions"

    result = llm_client.query_model(
        ModelQueryArgs(
            model="openrouter/qwen/qwen3.6-plus",
            server_url=endpoint,
        ),
        [{"role": "user", "content": "hello"}],
    )

    assert result["content"] == "ok"
    assert captured["server_url"] == endpoint
