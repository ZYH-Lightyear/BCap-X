"""Coding Agent 的模型回合与工具调用抽象。

RoboMEx 的第一版 Qwen-Code-style runtime 不接 provider 原生 tool calls;模型在
普通 assistant content 中输出一个 JSON action。本模块把这种文本协议适配成
Qwen-Code-like ``ModelTurn`` / ``ToolCall``:

- :func:`parse_model_turn` —— 把原始回复路由为 ``ToolCall`` 或 parser error。
- :func:`parse_action` —— 旧兼容入口,把原始回复路由为 ``run_python`` / ``use_skill`` /
  ``finish`` / ``empty`` / ``invalid``。
- :class:`SkillEntry` + :func:`render_available_skills` —— 开场的
  ``<available_skills>`` 感知清单(仅名称 + 描述)。
- :func:`build_skill_llm_content` —— 按需拉取某技能正文时返回的文本
  (对应 qwen-code 的 ``buildSkillLlmContent``)。
"""

from __future__ import annotations

import json
from html import escape
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from robomex.core.sandbox import BlockExecutionResult, SemanticActionBlock


class BlockExecutor(Protocol):
    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult: ...


@dataclass(frozen=True)
class AgentAction:
    """Legacy single structured model action."""

    kind: str
    args: dict[str, Any]
    raw: str = ""
    error: str = ""

    @property
    def signature(self) -> tuple[str, str]:
        """Stable repeat-detection signature."""

        return (self.kind, json.dumps(self.args, sort_keys=True, ensure_ascii=False))

    @property
    def payload_preview(self) -> str:
        if self.error:
            return self.error
        return json.dumps(self.args, ensure_ascii=False)


@dataclass(frozen=True)
class ToolCall:
    """A normalized tool call produced by one model turn."""

    name: str
    args: dict[str, Any]
    id: str = "call_0"
    raw: str = ""

    @property
    def signature(self) -> tuple[str, str]:
        return (self.name, json.dumps(self.args, sort_keys=True, ensure_ascii=False))

    @property
    def payload_preview(self) -> str:
        return json.dumps(self.args, ensure_ascii=False)


@dataclass(frozen=True)
class ModelTurn:
    """One normalized model turn.

    ``tool_calls`` mirrors provider-native function-calling shape. For the
    current JSON-in-text adapter it contains at most one call; future VAPI
    tool-call policies can populate it directly from upstream ``tool_calls``.
    ``error`` means the text adapter failed to produce a valid tool call and
    the runtime should reprompt instead of treating the text as a final answer.
    """

    raw: str
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    error: str = ""

    @property
    def is_error(self) -> bool:
        return bool(self.error)


@dataclass(frozen=True)
class SkillEntry:
    """``<available_skills>`` 感知清单中的一行(不含正文)。"""

    name: str
    description: str
    category: str = ""


def render_available_skills(entries: list[SkillEntry]) -> str:
    """渲染 qwen-code 式的 ``<skill>`` 块:只含名称 + 描述,不含正文。"""

    rows = []
    for e in entries:
        desc = f"{e.description} ({e.category})" if e.category else e.description
        rows.append(
            "<skill>\n"
            f"<name>{escape(e.name)}</name>\n"
            f"<description>{escape(desc)}</description>\n"
            "</skill>"
        )
    return "\n".join(rows)


def build_skill_llm_content(base_dir: Any, body: str) -> str:
    """加载某技能时返回的文本(对应 qwen-code 的 ``buildSkillLlmContent``)。"""

    base = str(base_dir) if base_dir else "(in-memory skill; no base directory)"
    return (
        f"Loaded skill. Base directory for this skill: {base}\n"
        "Resolve any referenced sidecar files (e.g. reference/verify.md, scripts/verify.py) "
        "as absolute paths under this base directory.\n\n"
        f"{body.strip()}\n"
    )


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip() in {"```json", "```"} and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _invalid(raw: str, error: str) -> AgentAction:
    return AgentAction(kind="invalid", args={}, raw=raw, error=error)


def parse_action(raw: str, is_terminal: Callable[[str], bool]) -> AgentAction:
    """Parse one model reply into one structured action.

    Expected shape:
    ``{"tool": "run_python", "args": {"code": "...", "intent": "..."}}``.
    ``finish`` may also be produced by role-specific terminal detectors; legacy
    verifier code uses that to route bare JSON verdicts to its terminal hook.
    """

    text = raw or ""
    if not text.strip():
        return AgentAction(kind="empty", args={}, raw=text)

    data = _parse_json_object(text)
    if data is None:
        if is_terminal(text):
            return AgentAction(kind="finish", args={"raw": text}, raw=text)
        return _invalid(text, "Expected exactly one JSON object action.")

    tool = data.get("tool", data.get("action"))
    args = data.get("args", {})
    if not isinstance(tool, str) or not tool.strip():
        if is_terminal(text):
            return AgentAction(kind="finish", args={"raw": text}, raw=text)
        return _invalid(text, 'JSON action must include a string "tool" field.')
    if not isinstance(args, dict):
        return _invalid(text, 'JSON action "args" must be an object.')

    aliases = {
        "python": "run_python",
        "terminal": "finish",
    }
    kind = aliases.get(tool.strip(), tool.strip())
    allowed = {"use_skill", "run_python", "finish"}
    if kind not in allowed:
        return _invalid(text, f'Unknown tool "{tool}".')

    if kind == "use_skill" and not isinstance(args.get("name"), str):
        return _invalid(text, 'use_skill requires args.name as a string.')
    if kind == "run_python" and not isinstance(args.get("code"), str):
        return _invalid(text, 'run_python requires args.code as a string.')
    if kind == "finish" and "raw" not in args:
        args = {**args, "raw": text}

    return AgentAction(kind=kind, args=args, raw=text)


def parse_model_turn(raw: str, is_terminal: Callable[[str], bool]) -> ModelTurn:
    """Parse current JSON action text into a normalized ``ModelTurn``.

    This is the adapter layer that lets the runtime operate on Qwen-Code-like
    tool calls while retaining the existing JSON action protocol until VAPI
    native tool calling is wired in.
    """

    action = parse_action(raw, is_terminal)
    if action.kind == "empty":
        return ModelTurn(raw=raw or "", error="Model returned empty content.")
    if action.kind == "invalid":
        return ModelTurn(raw=raw or "", error=action.error or "Invalid model action.")
    return ModelTurn(
        raw=raw or "",
        text="",
        tool_calls=(ToolCall(name=action.kind, args=action.args, raw=action.raw),),
    )
