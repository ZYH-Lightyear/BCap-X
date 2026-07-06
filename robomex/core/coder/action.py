"""Coding Agent 的模型回合与工具调用抽象。

RoboMEx 的 Qwen-Code-style runtime 不接 provider 原生 tool calls;模型在
普通 assistant content 中输出一个严格 JSON action。本模块把这种文本协议适配成
Qwen-Code-like ``ModelTurn`` / ``ToolCall``:

- :func:`parse_model_turn` —— 把原始回复路由为 ``ToolCall`` 或 parser error。
- :func:`parse_action` —— 把原始 JSON action 回复路由为一个结构化工具动作。
- :class:`SkillEntry` + :func:`render_available_skills` —— 开场的
  ``<available_skills>`` 感知清单(仅名称 + 描述)。
- :func:`build_skill_llm_content` —— 按需拉取某技能正文时返回的文本
  (对应 qwen-code 的 ``buildSkillLlmContent``)。
"""

from __future__ import annotations

import json
from html import escape
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from robomex.core.sandbox import BlockExecutionResult, SemanticActionBlock


class BlockExecutor(Protocol):
    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult: ...


@dataclass(frozen=True)
class AgentAction:
    """Single structured model action."""

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

    ``tool_calls`` mirrors function-calling shape inside RoboMEx. For the
    current JSON-in-text adapter it contains at most one call.
    ``error`` means the JSON action adapter failed to produce a valid tool call and
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
    files = _render_skill_sidecar_files(base_dir)
    return (
        f"Loaded skill. Base directory for this skill: {base}\n"
        "If the guidance references scripts/, references/, or assets/, resolve those "
        "paths as absolute paths under this base directory. Scripts are ordinary helper "
        "files, not hidden framework tools. Prefer the entry points and usage pattern "
        "named in SKILL.md. Do not spend a turn printing sidecar source or probing API "
        "signatures; the runtime prompt already documents available sandbox APIs. Open "
        "sidecar source only after a concrete execution error requires a short targeted "
        "diagnostic. No RoboMEx-specific skill wrapper function is provided.\n"
        f"{files}\n\n"
        f"{body.strip()}\n"
    )


def _render_skill_sidecar_files(base_dir: Any, *, limit: int = 24) -> str:
    if not base_dir:
        return "Skill sidecar files: (none; in-memory skill)"
    root = Path(str(base_dir))
    if not root.exists():
        return "Skill sidecar files: (base directory not found)"
    rows: list[str] = []
    for child_dir in ("scripts", "references", "assets"):
        folder = root / child_dir
        if not folder.exists():
            continue
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            rows.append(f"- {path.relative_to(root).as_posix()}: {path}")
            if len(rows) >= limit:
                rows.append("- ...")
                return "Skill sidecar files:\n" + "\n".join(rows)
    return "Skill sidecar files:\n" + ("\n".join(rows) if rows else "- (none)")


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip() in {"```json", "```"} and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = _strip_json_fence(text)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = _decode_first_json_object(stripped)
    if isinstance(data, dict):
        return data
    data = _repair_json_object(text)
    return data if isinstance(data, dict) else None


def _decode_first_json_object(text: str) -> dict[str, Any] | None:
    """Decode the first complete JSON object without repairing strings inside code.

    This handles common LLM wrappers such as prose before/after a JSON action and a
    harmless extra trailing brace. It deliberately runs before ``json_repair`` because
    repair libraries can reinterpret Python dict literals inside ``args.code`` as JSON
    structure and truncate executable code.
    """

    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        data, end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    tail = text[start + end :].strip()
    if tail and any(ch not in "}" for ch in tail):
        return None
    return data


def _repair_json_object(text: str) -> Any:
    try:
        import json_repair
    except ImportError:
        return None
    try:
        return json_repair.loads(text)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _invalid(raw: str, error: str) -> AgentAction:
    return AgentAction(kind="invalid", args={}, raw=raw, error=error)


def parse_action(raw: str) -> AgentAction:
    """Parse one model reply into one structured action.

    Expected shape:
    ``{"tool": "run_python", "args": {"code": "...", "intent": "..."}}``.
    Termination is explicit: the model must produce ``{"tool":"finish", ...}``.
    """

    text = raw or ""
    if not text.strip():
        return AgentAction(kind="empty", args={}, raw=text)

    data = _parse_json_object(text)
    if data is None:
        return _invalid(text, "Expected exactly one JSON object action.")

    tool = data.get("tool")
    args = data.get("args", {})
    if not isinstance(tool, str) or not tool.strip():
        return _invalid(text, 'JSON action must include a string "tool" field.')
    if not isinstance(args, dict):
        return _invalid(text, 'JSON action "args" must be an object.')

    kind = tool.strip()
    if kind == "use_skill" and not isinstance(args.get("name"), str):
        return _invalid(text, 'use_skill requires args.name as a string.')
    if kind == "run_python" and not isinstance(args.get("code"), str):
        return _invalid(text, 'run_python requires args.code as a string.')
    if kind == "call_subagent":
        if not isinstance(args.get("task"), str):
            return _invalid(text, 'call_subagent requires args.task as a string.')
        if "name" in args:
            return _invalid(text, 'call_subagent no longer accepts args.name; describe the delegated work in args.task.')
        if "profile" in args:
            return _invalid(text, 'call_subagent no longer accepts args.profile; describe the delegated work in args.task.')
        if "phase" in args:
            return _invalid(text, 'call_subagent does not accept args.phase; describe the delegated work in args.task.')
        if "inputs" in args and not isinstance(args.get("inputs"), dict):
            return _invalid(text, 'call_subagent args.inputs must be an object when provided.')
        if "task_id" in args and not isinstance(args.get("task_id"), str):
            return _invalid(text, 'call_subagent args.task_id must be a string when provided.')
    if kind == "finish" and "raw" not in args:
        args = {**args, "raw": text}

    return AgentAction(kind=kind, args=args, raw=text)


def normalized_action_json(tool: str, args: dict[str, Any]) -> str:
    """Serialize a parsed action without parser-only transport fields."""

    clean_args = {k: v for k, v in dict(args).items() if k != "raw"}
    return json.dumps({"tool": tool, "args": clean_args}, ensure_ascii=False)


def parse_action_payload(raw: str | None) -> dict[str, Any] | None:
    """Return the JSON action object from raw model text, if one can be recovered."""

    if not raw:
        return None
    return _parse_json_object(raw)


def parse_model_turn(raw: str) -> ModelTurn:
    """Parse current JSON action text into a normalized ``ModelTurn``.

    This is the adapter layer that lets the runtime operate on Qwen-Code-like
    tool calls while using JSON action text at the provider boundary.
    """

    action = parse_action(raw)
    if action.kind == "empty":
        return ModelTurn(raw=raw or "", error="Model returned empty content.")
    if action.kind == "invalid":
        return ModelTurn(raw=raw or "", error=action.error or "Invalid model action.")
    return ModelTurn(
        raw=raw or "",
        text="",
        tool_calls=(ToolCall(name=action.kind, args=action.args, raw=action.raw),),
    )
