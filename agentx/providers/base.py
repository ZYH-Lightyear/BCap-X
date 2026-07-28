"""模型端点抽象。

只要求一个方法:把 messages + tool schemas 变成一个
:class:`~agentx.contracts.ModelResponse`。上层(chat / core)不知道底下是
OpenAI、Anthropic 还是本地代理。
"""

from __future__ import annotations

from typing import Any, Protocol

from agentx.contracts import Message, ModelResponse


class ModelProvider(Protocol):
    """模型补全边界。"""

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        """发一次请求。

        :param messages: OpenAI wire format 的完整对话。
        :param tools: OpenAI function-calling 的 tool schema 列表;``None`` 或空
            列表表示这一轮不给工具。
        """
        ...


class ProviderError(RuntimeError):
    """模型端点返回了无法使用的结果。"""


__all__ = ["ModelProvider", "ProviderError"]
