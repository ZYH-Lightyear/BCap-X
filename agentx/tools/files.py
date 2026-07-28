"""文件读写与编辑工具。"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from agentx.contracts import ToolKind, ToolResult, ToolValidationError
from agentx.tools.base import Invocation, Tool, ToolContext

#: 按扩展名识别的图片。命中后 read_file 返回 image part 而不是乱码字节。
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
#: 单张图片进 prompt 的字节上限。超过就退化成文字说明 —— 让模型知道有这张图、
#: 但不要用几 MB 的 base64 把上下文冲垮。
MAX_INLINE_IMAGE_BYTES = 3 * 1024 * 1024
DEFAULT_READ_LINES = 2000


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a file from the workspace. Text files are returned with line numbers. "
        "Image files are returned as viewable images. Use offset/limit to page "
        "through large files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "offset": {
                "type": "integer",
                "description": "1-based line number to start from.",
                "minimum": 1,
            },
            "limit": {
                "type": "integer",
                "description": f"Max lines to read (default {DEFAULT_READ_LINES}).",
                "minimum": 1,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    kind = ToolKind.READ
    # read_file 自管分页,交给 scheduler 再截一刀只会把行号切断。
    max_output_chars = None

    def create_invocation(self, params: dict[str, Any]) -> Invocation:
        return _ReadFileInvocation(params)


class _ReadFileInvocation(Invocation):
    def describe(self) -> str:
        return f"read {self.params['path']}"

    def execute(self, ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.resolve(self.params["path"])
        except ToolValidationError as exc:
            return ToolResult(llm_content=str(exc), error=str(exc))

        if not path.exists():
            message = f"文件不存在: {path}"
            return ToolResult(llm_content=message, error=message)
        if path.is_dir():
            message = f"{path} 是目录,不是文件。用 glob 或 list 查看目录内容。"
            return ToolResult(llm_content=message, error=message)

        if path.suffix.lower() in IMAGE_SUFFIXES:
            return _read_image(path)
        return _read_text(path, self.params)


def _read_image(path: Path) -> ToolResult:
    raw = path.read_bytes()
    if len(raw) > MAX_INLINE_IMAGE_BYTES:
        note = (
            f"图片 {path.name} 有 {len(raw) / 1024 / 1024:.1f} MB,"
            f"超过内联上限 {MAX_INLINE_IMAGE_BYTES // 1024 // 1024} MB,未加载。"
        )
        return ToolResult(llm_content=note, display=note)

    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(raw).decode()
    return ToolResult(
        llm_content=[
            {"type": "text", "text": f"Image: {path}"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
        ],
        display=f"[image] {path} ({len(raw)} bytes)",
    )


def _read_text(path: Path, params: dict[str, Any]) -> ToolResult:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        message = f"{path} 不是 UTF-8 文本文件(可能是二进制)。"
        return ToolResult(llm_content=message, error=message)

    lines = text.splitlines()
    offset = int(params.get("offset") or 1)
    limit = int(params.get("limit") or DEFAULT_READ_LINES)
    start = max(offset - 1, 0)
    window = lines[start : start + limit]

    # 带行号返回:后续的 edit 全靠模型准确引用原文,行号让它能定位并在报错时对齐。
    numbered = "\n".join(f"{start + i + 1:6d}|{line}" for i, line in enumerate(window))
    shown_end = start + len(window)
    if shown_end < len(lines):
        numbered += (
            f"\n... 还有 {len(lines) - shown_end} 行未显示,"
            f"用 offset={shown_end + 1} 继续读 ..."
        )
    if not window:
        numbered = "(文件为空,或 offset 超出了文件长度)"

    return ToolResult(llm_content=numbered, display=numbered)


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write content to a file, creating parent directories as needed. "
        "Overwrites the file if it already exists."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }
    kind = ToolKind.EDIT

    def create_invocation(self, params: dict[str, Any]) -> Invocation:
        return _WriteFileInvocation(params)


class _WriteFileInvocation(Invocation):
    def describe(self) -> str:
        return f"write {self.params['path']}"

    def execute(self, ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.resolve(self.params["path"])
        except ToolValidationError as exc:
            return ToolResult(llm_content=str(exc), error=str(exc))

        content = self.params["content"]
        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        verb = "覆盖" if existed else "创建"
        summary = f"已{verb} {path}({len(content.splitlines())} 行)"
        return ToolResult(llm_content=summary, display=summary)


class EditTool(Tool):
    name = "edit"
    description = (
        "Replace an exact string in a file. old_string must appear exactly once "
        "unless replace_all is true. Include enough surrounding context to make "
        "old_string unique."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    }
    kind = ToolKind.EDIT

    def create_invocation(self, params: dict[str, Any]) -> Invocation:
        if params["old_string"] == params["new_string"]:
            raise ToolValidationError("old_string 与 new_string 相同,这次编辑没有意义")
        return _EditInvocation(params)


class _EditInvocation(Invocation):
    def describe(self) -> str:
        return f"edit {self.params['path']}"

    def execute(self, ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.resolve(self.params["path"])
        except ToolValidationError as exc:
            return ToolResult(llm_content=str(exc), error=str(exc))

        if not path.is_file():
            message = f"文件不存在: {path}"
            return ToolResult(llm_content=message, error=message)

        original = path.read_text(encoding="utf-8")
        old = self.params["old_string"]
        new = self.params["new_string"]
        occurrences = original.count(old)

        if occurrences == 0:
            message = (
                f"在 {path} 里找不到 old_string。文件可能已被改动,"
                "先用 read_file 重新读一遍再编辑。"
            )
            return ToolResult(llm_content=message, error=message)

        # 多处命中时拒绝而不是改第一处:猜错位置产生的静默错误远比多一轮往返昂贵。
        if occurrences > 1 and not self.params.get("replace_all"):
            message = (
                f"old_string 在 {path} 里出现了 {occurrences} 次。"
                "请补充上下文让它唯一,或设置 replace_all=true。"
            )
            return ToolResult(llm_content=message, error=message)

        path.write_text(original.replace(old, new), encoding="utf-8")
        summary = f"已编辑 {path}(替换 {occurrences} 处)"
        return ToolResult(llm_content=summary, display=summary)


__all__ = ["EditTool", "ReadFileTool", "WriteFileTool"]
