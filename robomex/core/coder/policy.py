"""Coding Agent 的补全策略(completion policy)。

策略把一段 chat 形式的 prompt 变成模型回合。当前 JSON action 策略仍可只实现
``complete`` 返回原始文本;真正 provider-native tool-call 策略可实现
``complete_turn`` 直接返回 :class:`robomex.core.coder.action.ModelTurn`。
"""

from __future__ import annotations

import json
from typing import Any
from typing import Protocol

import requests

from robomex.core.coder.action import ModelTurn, ToolCall


ROBO_MEX_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "use_skill",
            "description": "Load a RoboMEx skill's full SKILL.md guidance before using it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill id from the available_skills block.",
                    }
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute one Python code block in the RoboMEx sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                    "intent": {
                        "type": "string",
                        "description": "Short purpose of this code block.",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Return control to the Planner when this sub-goal attempt is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "Brief statement of what was completed or why control should return.",
                    }
                },
                "required": ["claim"],
                "additionalProperties": False,
            },
        },
    },
]


class CompletionPolicy(Protocol):
    """返回模型原始回复或规范化模型回合。"""

    def complete(self, prompt: list[dict]) -> str: ...

    def complete_turn(self, prompt: list[dict]) -> ModelTurn: ...


class LLMCodePolicy:
    """基于 ``capx.llm.client.query_model`` 的真实 LLM 策略。

    ``model``/``server_url`` 沿用 CapX 约定;OpenRouter 模型由底层 client
    自动经本地代理转发。
    """

    def __init__(
        self,
        model: str = "openrouter/qwen/qwen3.6-plus",
        server_url: str = "http://localhost:8110/chat/completions",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 20480,  # 对齐 capx baseline(2048*10);含 reasoning 预算
        empty_retries: int = 1,
    ) -> None:
        from capx.llm.client import ModelQueryArgs, query_model

        self._query_model = query_model
        self._empty_retries = max(0, empty_retries)
        self._args = ModelQueryArgs(
            model=model,
            server_url=server_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete(self, prompt: list[dict]) -> str:
        active_prompt = prompt
        for attempt in range(self._empty_retries + 1):
            out = self._query_model(self._args, active_prompt)
            content = out.get("content") or ""
            if content.strip():
                return content
            if attempt >= self._empty_retries:
                return content
            active_prompt = [
                *prompt,
                {
                    "role": "user",
                    "content": (
                        "Your previous response contained no visible content. "
                        "Reply now with exactly one JSON action object using tool use_skill, "
                        "run_python, or finish. Do not leave the message empty."
                    ),
                },
            ]
            print("[LLMCodePolicy] empty model content; retrying once with an explicit action nudge")
        return ""


class VAPIToolCallPolicy:
    """OpenAI-compatible native tool-call policy for ``vapi_tool_server.py``.

    This policy expects a proxy that preserves raw ``tool_calls`` fields, such as
    :mod:`capx.serving.vapi_tool_server`. It returns ``ModelTurn`` directly, so
    :class:`robomex.core.coder.agent.CodingAgent` does not parse JSON text.
    """

    def __init__(
        self,
        model: str = "vapi/gpt-5.5",
        server_url: str = "http://localhost:8110/chat/completions",
        api_key: str | None = None,
        max_tokens: int = 20480,
        temperature: float | None = None,
        timeout: float = 600.0,
        token_field: str = "max_completion_tokens",
    ) -> None:
        self.model = model
        self.server_url = server_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.token_field = token_field

    def complete(self, prompt: list[dict]) -> str:
        turn = self.complete_turn(prompt)
        return turn.raw or turn.text

    def complete_turn(self, prompt: list[dict]) -> ModelTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": prompt,
            "tools": ROBO_MEX_TOOL_SCHEMAS,
            "tool_choice": "auto",
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.token_field == "max_completion_tokens":
            payload["max_completion_tokens"] = self.max_tokens
        elif self.token_field == "max_tokens":
            payload["max_tokens"] = self.max_tokens
        else:
            raise ValueError(f"Unsupported token_field: {self.token_field}")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.server_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return _model_turn_from_chat_completion(data)


def _model_turn_from_chat_completion(data: dict[str, Any]) -> ModelTurn:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ModelTurn(raw=json.dumps(data, ensure_ascii=False), error="chat completion returned no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ModelTurn(raw=json.dumps(data, ensure_ascii=False), error="first choice has no message")

    calls: list[ToolCall] = []
    for i, call in enumerate(message.get("tool_calls") or ()):
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "")
        if not name:
            continue
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            return ModelTurn(
                raw=json.dumps(data, ensure_ascii=False),
                error=f"tool call {name!r} arguments are not valid JSON",
            )
        if not isinstance(args, dict):
            return ModelTurn(
                raw=json.dumps(data, ensure_ascii=False),
                error=f"tool call {name!r} arguments must decode to an object",
            )
        calls.append(
            ToolCall(
                name=name,
                args=args,
                id=str(call.get("id") or f"call_{i}"),
                raw=json.dumps(call, ensure_ascii=False),
            )
        )

    content = message.get("content")
    text = content if isinstance(content, str) else ""
    return ModelTurn(
        raw=json.dumps(data, ensure_ascii=False),
        text=text,
        tool_calls=tuple(calls),
    )


class ScriptedCodePolicy:
    """回放一组固定的原始回复;用尽后返回 JSON ``finish`` action。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def complete(self, prompt: list[dict]) -> str:
        if self._index >= len(self._responses):
            return '{"tool":"finish","args":{"claim":"scripted policy exhausted"}}'
        response = self._responses[self._index]
        self._index += 1
        return response
