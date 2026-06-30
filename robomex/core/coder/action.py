"""Coding Agent 的动作空间:把模型回复解析成结构化 action。

RoboMEx 的第一版 Qwen-Code-style runtime 不接 provider 原生 tool calls;模型在
普通 assistant content 中输出一个 JSON action,本模块负责解析和校验:

- :func:`parse_action` —— 把原始回复路由为 ``run_python`` / ``use_skill`` /
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
    """A single structured model action."""

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
