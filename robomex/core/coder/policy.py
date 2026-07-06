"""Coding Agent 的补全策略(completion policy)。

策略把一段 chat 形式的 prompt 变成模型回合。在线主路径使用 JSON action:
模型返回普通文本 JSON,运行时再解析成内部 :class:`robomex.core.coder.action.ToolCall`。
"""

from __future__ import annotations

from typing import Protocol

from robomex.core.coder.action import ModelTurn
from robomex.core.logging import get_logger

_log = get_logger("coder.policy")


class CompletionPolicy(Protocol):
    """返回模型原始回复或规范化模型回合。"""

    def complete(self, prompt: list[dict]) -> str: ...

    def complete_turn(self, prompt: list[dict], tool_names: set[str] | None = None) -> ModelTurn: ...


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
                        "call_subagent, run_python, or finish. Do not leave the message empty."
                    ),
                },
            ]
            _log.warning("empty model content; retrying once with an explicit action nudge")
        return ""


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
