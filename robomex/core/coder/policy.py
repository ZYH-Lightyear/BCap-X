"""Coding Agent 的补全策略(completion policy)。

策略把一段 chat 形式的 prompt 变成模型的*原始*回复;动作解析(python /
USE SKILL / terminal)由 :class:`~robomex.core.coder.agent.CodingAgent` 循环自己做,
所以策略只需实现 ``complete``。``LLMCodePolicy`` 封装 CapX 的 ``query_model``,
用真实 LLM 驱动 agent;``ScriptedCodePolicy`` 回放预设回复,用于离线运行和测试。
"""

from __future__ import annotations

from typing import Protocol

FINISH = "FINISH"


class CompletionPolicy(Protocol):
    """返回模型的原始回复;由 agent 循环负责把它路由成动作。"""

    def complete(self, prompt: list[dict]) -> str: ...


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
    ) -> None:
        from capx.llm.client import ModelQueryArgs, query_model

        self._query_model = query_model
        self._args = ModelQueryArgs(
            model=model,
            server_url=server_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete(self, prompt: list[dict]) -> str:
        return self._query_model(self._args, prompt)["content"]


class ScriptedCodePolicy:
    """回放一组固定的原始回复;用尽后返回 ``FINISH``。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def complete(self, prompt: list[dict]) -> str:
        if self._index >= len(self._responses):
            return FINISH
        response = self._responses[self._index]
        self._index += 1
        return response
