"""LLM 补全边界。

planner 与模型之间只有一个方法:``complete(prompt) -> str``。prompt 采用 OpenAI
chat 格式,``content`` 既可以是字符串,也可以是 ``[{"type": "text"...},
{"type": "image_url"...}]`` 这样的多模态片段列表 —— 观测图片就是这样送进去的。

保持这个边界只有一个方法,是为了让单测能用十几行的假策略替换掉真实模型。
"""

from __future__ import annotations

from typing import Protocol


class CompletionPolicy(Protocol):
    """把一段 chat prompt 变成模型回复文本。"""

    def complete(self, prompt: list[dict]) -> str: ...


class LLMPolicy:
    """走 ``capx.llm.client.query_model`` 的真实模型策略。

    ``model`` / ``server_url`` 沿用 CapX 约定,provider 路由交给统一代理处理。
    """

    def __init__(
        self,
        model: str = "vapi/claude-opus-4-8",
        server_url: str = "http://localhost:8110/chat/completions",
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        # 延迟到构造时才 import:capx.llm 会拖进一串网络与配置依赖,
        # 不用真实模型的单测不该为此付出代价。
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
        out = self._query_model(self._args, prompt)
        return out.get("content") or ""


__all__ = ["CompletionPolicy", "LLMPolicy"]
