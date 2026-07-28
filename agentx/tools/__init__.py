"""内置工具集。"""

from agentx.tools.base import Invocation, Tool, ToolContext
from agentx.tools.files import EditTool, ReadFileTool, WriteFileTool
from agentx.tools.registry import ToolRegistry
from agentx.tools.search import GlobTool, GrepTool
from agentx.tools.shell import ShellTool
from agentx.tools.skill import SkillTool


def default_registry() -> ToolRegistry:
    """M1 的默认工具集:读、写、改、搜、跑。"""
    return ToolRegistry([
        ReadFileTool(),
        WriteFileTool(),
        EditTool(),
        GlobTool(),
        GrepTool(),
        ShellTool(),
    ])


__all__ = [
    "EditTool",
    "GlobTool",
    "GrepTool",
    "Invocation",
    "ReadFileTool",
    "ShellTool",
    "SkillTool",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "WriteFileTool",
    "default_registry",
]
