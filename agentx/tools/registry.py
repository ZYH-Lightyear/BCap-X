"""工具注册表。"""

from __future__ import annotations

from typing import Any

from agentx.tools.base import Tool


class ToolRegistry:
    """按名字存放工具,并导出发给模型的 schema 列表。"""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError(f"{type(tool).__name__} 没有设置 name")
        if tool.name in self._tools:
            raise ValueError(f"工具重名:{tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def schemas(
        self,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """导出 schema 列表。

        :param allow: 只暴露这些工具;``None`` 表示全部。
        :param deny: 在 allow 之后再剔除这些。顺序很重要 —— 先 allow 后 deny 才能
            表达「继承全部但排除某几个」这种最常用的子 agent 配置。
        """
        selected = self.names() if allow is None else tuple(n for n in allow if n in self._tools)
        if deny:
            blocked = set(deny)
            selected = tuple(name for name in selected if name not in blocked)
        return [self._tools[name].schema() for name in selected]


__all__ = ["ToolRegistry"]
