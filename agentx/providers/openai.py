"""OpenAI 兼容端点的 provider。

对接任何暴露 ``POST /chat/completions`` 的服务:OpenAI 本体、vLLM、以及本仓库
``capx.serving.openrouter_server`` 那个统一代理。
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from agentx.contracts import ModelResponse, ToolCall
from agentx.providers.base import ProviderError

#: 代理侧 ``normalize_proxy_payload`` 有一句 ``budget = max_completion_tokens or
#: max_tokens or 256``。也就是说不显式给上限,输出会被悄悄压到 256 token,现象是
#: 模型「话说一半就停」而没有任何报错。所以 max_tokens 永远显式发送。
DEFAULT_MAX_TOKENS = 8192


class OpenAIProvider:
    """基于 ``requests`` 的同步 OpenAI 兼容 provider。

    M1 只做非流式。流式的难点不在 SSE 本身,而在 ``tool_calls`` 的 delta 会按
    index 切成任意碎片、还可能乱序,需要一个独立的重组器 —— 那是后续里程碑的事,
    在这里留一个干净的接缝比提前写错强。
    """

    def __init__(
        self,
        model: str = "vapi/claude-opus-4-8",
        server_url: str = "http://localhost:8110/chat/completions",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_s: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.server_url = server_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = self._post_with_retry(payload)
        return _parse_response(data)

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.server_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                last_error = exc
            else:
                # 4xx 基本是请求本身有问题,重试只是浪费时间和额度;5xx 与限流才退避重试。
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as exc:
                        last_error = ProviderError(f"响应不是合法 JSON: {response.text[:300]}")
                        del exc
                elif response.status_code < 500 and response.status_code != 429:
                    raise ProviderError(
                        f"模型端点返回 {response.status_code}: {response.text[:500]}"
                    )
                else:
                    last_error = ProviderError(
                        f"模型端点返回 {response.status_code}: {response.text[:300]}"
                    )

            if attempt < self.max_retries - 1:
                time.sleep(2.0 * (attempt + 1))

        raise ProviderError(f"模型请求在 {self.max_retries} 次尝试后仍失败: {last_error}")


def _parse_response(data: dict[str, Any]) -> ModelResponse:
    """把 OpenAI 响应体规范化成 :class:`ModelResponse`。"""

    choices = data.get("choices") or []
    if not choices:
        raise ProviderError(f"响应里没有 choices: {json.dumps(data)[:300]}")

    choice = choices[0] or {}
    message = choice.get("message") or {}

    text = message.get("content") or ""
    if isinstance(text, list):
        # 少数上游会把 assistant 的 content 也拆成片段列表。
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))

    return ModelResponse(
        text=str(text),
        tool_calls=tuple(_parse_tool_calls(message.get("tool_calls"))),
        finish_reason=str(choice.get("finish_reason") or ""),
        usage=data.get("usage") or {},
    )


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    """解析 ``tool_calls``,把坏 JSON 记录下来而不是抛出。

    参数是模型写的,写坏是常态。坏 JSON 要作为一条 error 结果回灌给模型让它重写,
    所以这里必须产出一个带 ``parse_error`` 的 :class:`ToolCall` 而不是异常 ——
    异常会连带丢掉 call id,后面就补不出配对的 tool 消息了。
    """

    if not isinstance(raw, list):
        return []

    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        function = item.get("function") or {}
        name = str(function.get("name") or "")
        if not name:
            continue
        # id 缺失时自己补一个:后续所有配对都靠它,没有 id 整条链路会断。
        call_id = str(item.get("id") or f"call_{index}")

        raw_args = function.get("arguments")
        if isinstance(raw_args, dict):
            calls.append(ToolCall(id=call_id, name=name, args=raw_args))
            continue

        text = str(raw_args or "").strip()
        if not text:
            calls.append(ToolCall(id=call_id, name=name, args={}))
            continue

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    parse_error=f"arguments 不是合法 JSON ({exc}): {text[:200]}",
                )
            )
            continue

        if isinstance(parsed, dict):
            calls.append(ToolCall(id=call_id, name=name, args=parsed))
        else:
            calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    parse_error=f"arguments 必须是 JSON 对象,收到 {type(parsed).__name__}",
                )
            )

    return calls


__all__ = ["DEFAULT_MAX_TOKENS", "OpenAIProvider"]
