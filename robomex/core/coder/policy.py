"""Coding Agent 的补全策略(completion policy)。

策略把一段 chat 形式的 prompt 变成模型回合。在线主路径使用 JSON action:
模型返回普通文本 JSON,运行时再解析成内部 :class:`robomex.core.coder.action.ToolCall`。
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Protocol

from robomex.core.coder.action import ModelTurn, parse_model_turn
from robomex.core.coder.protocol import StructuredOutputConfig


class CompletionPolicy(Protocol):
    """返回模型原始回复或规范化模型回合。"""

    def complete(self, prompt: list[dict]) -> str: ...

    def complete_turn(
        self, prompt: list[dict], tool_names: set[str] | None = None
    ) -> ModelTurn: ...


class BoundedCompletionPolicy(Protocol):
    """Completion boundary that enforces an output-token ceiling at request time."""

    def complete_bounded(
        self,
        prompt: list[dict],
        *,
        max_tokens: int,
        deadline_monotonic_s: float | None = None,
    ) -> str: ...

    def complete_turn_bounded(
        self,
        prompt: list[dict],
        *,
        max_tokens: int,
        deadline_monotonic_s: float | None = None,
        tool_names: set[str] | None = None,
    ) -> ModelTurn: ...


class LLMCodePolicy:
    """基于 ``capx.llm.client.query_model`` 的真实 LLM 策略。

    ``model``/``server_url`` 沿用 CapX 约定；provider 路由由统一代理配置处理。
    """

    def __init__(
        self,
        model: str = "openrouter/qwen/qwen3.6-plus",
        server_url: str = "http://localhost:8110/chat/completions",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 20480,  # 对齐 capx baseline(2048*10);含 reasoning 预算
        structured_output: StructuredOutputConfig | None = None,
    ) -> None:
        from capx.llm.client import ModelQueryArgs, query_model

        self._query_model = query_model
        self.structured_output = structured_output or StructuredOutputConfig()
        self._args = ModelQueryArgs(
            model=model,
            server_url=server_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete(self, prompt: list[dict]) -> str:
        out = self._query_model(self._args, prompt)
        return out.get("content") or ""

    def complete_bounded(
        self,
        prompt: list[dict],
        *,
        max_tokens: int,
        deadline_monotonic_s: float | None = None,
    ) -> str:
        if isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        timeout_s = None
        if deadline_monotonic_s is not None:
            timeout_s = deadline_monotonic_s - time.monotonic()
            if timeout_s <= 0:
                raise TimeoutError("bounded completion deadline expired before API entry")
        args = replace(
            self._args,
            max_tokens=min(self._args.max_tokens, max_tokens),
            request_timeout_s=timeout_s,
            max_retries=1 if timeout_s is not None else None,
            deadline_monotonic_s=deadline_monotonic_s,
        )
        out = self._query_model(args, prompt)
        return out.get("content") or ""

    def complete_turn(
        self,
        prompt: list[dict],
        tool_names: set[str] | None = None,
    ) -> ModelTurn:
        # text_json is the only active transport. Other modes are intentionally
        # configuration placeholders until the proxy forwards response_format.
        return parse_model_turn(self.complete(prompt))

    def complete_turn_bounded(
        self,
        prompt: list[dict],
        *,
        max_tokens: int,
        deadline_monotonic_s: float | None = None,
        tool_names: set[str] | None = None,
    ) -> ModelTurn:
        del tool_names
        return parse_model_turn(
            self.complete_bounded(
                prompt,
                max_tokens=max_tokens,
                deadline_monotonic_s=deadline_monotonic_s,
            )
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

    def complete_bounded(
        self,
        prompt: list[dict],
        *,
        max_tokens: int,
        deadline_monotonic_s: float | None = None,
    ) -> str:
        if isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
            raise TimeoutError("bounded scripted completion deadline expired")
        return self.complete(prompt)

    def complete_turn(
        self,
        prompt: list[dict],
        tool_names: set[str] | None = None,
    ) -> ModelTurn:
        return parse_model_turn(self.complete(prompt))

    def complete_turn_bounded(
        self,
        prompt: list[dict],
        *,
        max_tokens: int,
        deadline_monotonic_s: float | None = None,
        tool_names: set[str] | None = None,
    ) -> ModelTurn:
        del tool_names
        return parse_model_turn(
            self.complete_bounded(
                prompt,
                max_tokens=max_tokens,
                deadline_monotonic_s=deadline_monotonic_s,
            )
        )
