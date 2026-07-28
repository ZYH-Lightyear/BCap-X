"""文件查找与内容检索工具。"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from agentx.contracts import ToolKind, ToolResult, ToolValidationError
from agentx.tools.base import Invocation, Tool, ToolContext

MAX_GLOB_RESULTS = 200
MAX_GREP_MATCHES = 200
SEARCH_TIMEOUT_S = 60


class GlobTool(Tool):
    name = "glob"
    description = (
        "Find files by glob pattern (e.g. '**/*.py'). Returns paths sorted by "
        "modification time, most recent first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.ts'."},
            "path": {"type": "string", "description": "Directory to search in. Defaults to workspace root."},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    kind = ToolKind.SEARCH

    def create_invocation(self, params: dict[str, Any]) -> Invocation:
        return _GlobInvocation(params)


class _GlobInvocation(Invocation):
    def describe(self) -> str:
        return f"glob {self.params['pattern']}"

    def execute(self, ctx: ToolContext) -> ToolResult:
        try:
            base = ctx.resolve(self.params.get("path") or ".")
        except ToolValidationError as exc:
            return ToolResult(llm_content=str(exc), error=str(exc))

        if not base.is_dir():
            message = f"{base} 不是目录"
            return ToolResult(llm_content=message, error=message)

        matches = [p for p in base.glob(self.params["pattern"]) if p.is_file()]
        # 按修改时间倒序:找「最近动过什么」是这个工具最常见的用途。
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        if not matches:
            return ToolResult(llm_content=f"没有文件匹配 {self.params['pattern']}")

        shown = matches[:MAX_GLOB_RESULTS]
        body = "\n".join(str(p) for p in shown)
        if len(matches) > len(shown):
            body += f"\n... 另有 {len(matches) - len(shown)} 个结果未显示 ..."
        return ToolResult(llm_content=body, display=body)


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents with a regular expression. Returns matching lines "
        "with file paths and line numbers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression."},
            "path": {"type": "string", "description": "File or directory to search. Defaults to workspace root."},
            "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'."},
            "ignore_case": {"type": "boolean", "default": False},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    kind = ToolKind.SEARCH

    def create_invocation(self, params: dict[str, Any]) -> Invocation:
        return _GrepInvocation(params)


class _GrepInvocation(Invocation):
    def describe(self) -> str:
        return f"grep {self.params['pattern']}"

    def execute(self, ctx: ToolContext) -> ToolResult:
        try:
            target = ctx.resolve(self.params.get("path") or ".")
        except ToolValidationError as exc:
            return ToolResult(llm_content=str(exc), error=str(exc))

        # 只走 ripgrep。自己用 Python 遍历在大仓库上慢一到两个数量级,而且要重新
        # 实现 .gitignore 语义 —— 与其做个半吊子,不如明确要求装 rg。
        rg = shutil.which("rg")
        if rg is None:
            message = "找不到 ripgrep(rg),grep 工具不可用。"
            return ToolResult(llm_content=message, error=message)

        command = [rg, "--line-number", "--no-heading", "--color", "never"]
        if self.params.get("ignore_case"):
            command.append("--ignore-case")
        if self.params.get("glob"):
            command += ["--glob", self.params["glob"]]
        command += ["--regexp", self.params["pattern"], str(target)]

        try:
            completed = subprocess.run(
                command, capture_output=True, text=True,
                errors="replace", timeout=SEARCH_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            message = f"检索超时({SEARCH_TIMEOUT_S}s)"
            return ToolResult(llm_content=message, error=message)

        # rg 用退出码 1 表示「没匹配」,这不是错误。
        if completed.returncode == 1:
            return ToolResult(llm_content=f"没有内容匹配 {self.params['pattern']!r}")
        if completed.returncode != 0:
            message = f"ripgrep 失败: {completed.stderr[:500]}"
            return ToolResult(llm_content=message, error=message)

        lines = completed.stdout.splitlines()
        shown = lines[:MAX_GREP_MATCHES]
        body = "\n".join(shown)
        if len(lines) > len(shown):
            body += f"\n... 另有 {len(lines) - len(shown)} 处匹配未显示,请缩小检索范围 ..."
        return ToolResult(llm_content=body, display=body)


__all__ = ["GlobTool", "GrepTool"]
