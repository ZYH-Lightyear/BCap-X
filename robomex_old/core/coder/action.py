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

from robomex.core.coder.protocol import ParsedActionFrame, parse_action_frame
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
    prefix: str = ""
    suffix: str = ""

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
    canonical_json: str = ""
    quarantined_prefix: str = ""
    quarantined_suffix: str = ""

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


def build_skill_llm_content(
    base_dir: Any,
    body: str,
    *,
    scripts_on_sys_path: bool = False,
    bound_functions: tuple[tuple[str, str, str], ...] = (),
    prompt_names: tuple[str, ...] = (),
) -> str:
    """加载某技能时返回的文本(对应 qwen-code 的 ``buildSkillLlmContent``)。

    qwen-code 只需注入 base directory,因为它的 agent 跑 bash、绝对路径即可执行;
    这里的 agent 在同一个 exec 解释器里跑,所以 runtime 会先把 ``scripts/`` 挂上
    ``sys.path``(M1.5 Fix C),让 agent 直接 ``import <module>`` 而不是每次手写
    importlib 样板。M2 起,契约声明的 ``functions:`` 已直接绑定为沙箱内可调用名,
    ``prompts:`` 模板放进 ``PROMPTS`` 字典;两者都随本消息如实告知。
    """

    base = str(base_dir) if base_dir else "(in-memory skill; no base directory)"
    files = _render_skill_sidecar_files(base_dir)
    modules = skill_script_modules(base_dir)
    if scripts_on_sys_path and modules:
        rendered = ", ".join(f"`import {m}`" for m in modules)
        import_line = (
            f"The skill's scripts/ directory is already on sys.path: use {rendered} "
            "directly. Do NOT write importlib/spec_from_file_location boilerplate "
            "and do not re-add sys.path entries.\n"
        )
    elif modules:
        rendered = ", ".join(modules)
        import_line = (
            f"Python modules under scripts/ ({rendered}) must be loaded from the "
            "absolute base directory above.\n"
        )
    else:
        import_line = ""
    bindings_block = _render_contract_bindings(bound_functions, prompt_names)
    return (
        f"Loaded skill. Base directory for this skill: {base}\n"
        "If the guidance references scripts/, references/, or assets/, resolve those "
        "paths as absolute paths under this base directory. Scripts are ordinary helper "
        "files, not hidden framework tools. Prefer the entry points and usage pattern "
        f"named in SKILL.md. {import_line}"
        "Do not spend a turn printing sidecar source or probing API "
        "signatures; the runtime prompt already documents available sandbox APIs. Open "
        "sidecar source only after a concrete execution error requires a short targeted "
        "diagnostic. No RoboMEx-specific skill wrapper function is provided.\n"
        f"{bindings_block}"
        f"{files}\n\n"
        f"{body.strip()}\n"
    )


def _render_contract_bindings(
    bound_functions: tuple[tuple[str, str, str], ...],
    prompt_names: tuple[str, ...],
) -> str:
    """Render the canonical-function / prompt-template section of a skill load."""

    if not bound_functions and not prompt_names:
        return ""
    parts: list[str] = []
    if bound_functions:
        rows = "\n".join(
            f"- {signature}" + (f" — {description}" if description else "")
            for _, signature, description in bound_functions
        )
        parts.append(
            "Canonical skill functions are already defined in your sandbox "
            "namespace — call them directly (no import, no importlib):\n"
            f"{rows}\n"
            "Prefer these over re-implementing the same computation; write "
            "custom code only where the canonical function does not cover the "
            "current situation."
        )
    if prompt_names:
        rendered = ", ".join(f"PROMPTS[{name!r}]" for name in prompt_names)
        parts.append(
            f"Curated prompt templates are available in the sandbox: {rendered}. "
            "Reuse them (e.g. for query_vlm) instead of writing new question "
            "wording from scratch."
        )
    return "\n".join(parts) + "\n"


def skill_script_modules(base_dir: Any) -> tuple[str, ...]:
    """Importable module names contributed by a skill's ``scripts/`` sidecar."""

    if not base_dir:
        return ()
    folder = Path(str(base_dir)) / "scripts"
    if not folder.is_dir():
        return ()
    return tuple(
        sorted(
            p.stem
            for p in folder.glob("*.py")
            if p.is_file() and not p.name.startswith("_")
        )
    )


def _render_skill_sidecar_files(base_dir: Any, *, limit: int = 24) -> str:
    if not base_dir:
        return "Skill sidecar files: (none; in-memory skill)"
    root = Path(str(base_dir))
    if not root.exists():
        return "Skill sidecar files: (base directory not found)"
    rows: list[str] = []
    for child_dir in ("scripts", "references", "assets", "prompts"):
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


def _invalid(
    raw: str,
    error: str,
    *,
    frame: ParsedActionFrame | None = None,
) -> AgentAction:
    return AgentAction(
        kind="invalid",
        args={},
        raw=raw,
        error=error,
        prefix=frame.prefix if frame else "",
        suffix=frame.suffix if frame else "",
    )


def parse_action(raw: str) -> AgentAction:
    """Parse one model reply into one structured action.

    Expected shape:
    ``{"tool": "run_python", "args": {"code": "...", "intent": "..."}}``.
    Termination is explicit: the model must produce ``{"tool":"finish", ...}``.
    """

    text = raw or ""
    if not text.strip():
        return AgentAction(kind="empty", args={}, raw=text)

    frame = parse_action_frame(text)
    if frame.envelope is None:
        return _invalid(text, frame.error or "Expected one JSON object action.", frame=frame)

    kind = frame.envelope.tool
    args = frame.envelope.args
    if kind == "use_skill" and not isinstance(args.get("name"), str):
        return _invalid(text, 'use_skill requires args.name as a string.', frame=frame)
    if kind == "run_python" and not isinstance(args.get("code"), str):
        return _invalid(text, 'run_python requires args.code as a string.', frame=frame)
    if kind == "spawn_subagent":
        return _invalid(
            text,
            "spawn_subagent was removed; submit a validated dynamic specialist graph.",
            frame=frame,
        )
    if kind == "call_subagent":
        return _invalid(
            text,
            "call_subagent was removed; submit a validated dynamic specialist graph.",
            frame=frame,
        )
    if kind == "finish":
        result = args.get("result")
        if "state_patch" in args or (
            isinstance(result, dict) and "state_patch" in result
        ):
            return _invalid(
                text,
                "finish no longer accepts state_patch; return typed outputs, evidence, "
                "artifacts, and verdicts instead.",
                frame=frame,
            )
        if "raw" not in args:
            args = {**args, "raw": text}

    return AgentAction(
        kind=kind,
        args=args,
        raw=text,
        prefix=frame.prefix,
        suffix=frame.suffix,
    )


def normalized_action_json(tool: str, args: dict[str, Any]) -> str:
    """Serialize a parsed action without parser-only transport fields."""

    clean_args = {k: v for k, v in dict(args).items() if k != "raw"}
    return json.dumps({"tool": tool, "args": clean_args}, ensure_ascii=False)


def parse_action_payload(raw: str | None) -> dict[str, Any] | None:
    """Return the JSON action object from raw model text, if one can be recovered."""

    if not raw:
        return None
    frame = parse_action_frame(raw)
    return frame.envelope.to_mapping() if frame.envelope is not None else None


_FENCED_CODE_RE = __import__("re").compile(
    r"```(?:python|py)?\s*\n(.*?)```", __import__("re").DOTALL
)


def parse_model_turn(raw: str) -> ModelTurn:
    """Parse current JSON action text into a normalized ``ModelTurn``.

    This is the adapter layer that lets the runtime operate on Qwen-Code-like
    tool calls while using JSON action text at the provider boundary.

    Tolerant fallback: when the model returns a bare fenced code block instead
    of a JSON action, treat it as ``run_python`` rather than a protocol error.
    """

    action = parse_action(raw)
    if action.kind == "empty":
        return ModelTurn(raw=raw or "", error="Model returned empty content.")
    if action.kind == "invalid":
        code = _extract_single_fenced_block(raw or "")
        if code is not None:
            args = {"code": code, "intent": "bare-code-block-fallback"}
            return ModelTurn(
                raw=raw or "",
                text="",
                tool_calls=(ToolCall(name="run_python", args=args, raw=raw or ""),),
                canonical_json=normalized_action_json("run_python", args),
            )
        return ModelTurn(
            raw=raw or "",
            error=action.error or "Invalid model action.",
            quarantined_prefix=action.prefix,
            quarantined_suffix=action.suffix,
        )
    return ModelTurn(
        raw=raw or "",
        text="",
        tool_calls=(ToolCall(name=action.kind, args=action.args, raw=action.raw),),
        canonical_json=normalized_action_json(action.kind, action.args),
        quarantined_prefix=action.prefix,
        quarantined_suffix=action.suffix,
    )


def _extract_single_fenced_block(text: str) -> str | None:
    """Return code from a single fenced block, or None."""
    matches = _FENCED_CODE_RE.findall(text)
    if len(matches) != 1:
        return None
    code = matches[0].strip()
    return code if code else None
