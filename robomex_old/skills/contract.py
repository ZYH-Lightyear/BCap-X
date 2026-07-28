"""Skill contract loading and function signature resolution.

Extracted from robomex.dysc during the clean-slate rebuild. Only the parts
needed by the Coding Agent's progressive skill disclosure are kept:
load_contract_for_skill and resolve_function_signature. The old exit_conditions
vocabulary and prose/capability cross-checks are dropped.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONTRACT_FILE = "contract.yaml"


@dataclass(frozen=True)
class ContractPort:
    name: str
    schema: str
    required: bool = True
    frame: str = ""

    @classmethod
    def from_any(cls, value: str | dict[str, Any]) -> ContractPort:
        if isinstance(value, str):
            return cls(name=value, schema=f"robomex.{value}.v1")
        return cls(
            name=str(value.get("name") or ""),
            schema=str(value.get("schema") or ""),
            required=bool(value.get("required", True)),
            frame=str(value.get("frame") or ""),
        )


@dataclass(frozen=True)
class ContractFunction:
    name: str
    entry: str
    signature: str = ""
    description: str = ""

    @classmethod
    def from_any(cls, value: dict[str, Any]) -> ContractFunction:
        return cls(
            name=str(value.get("name") or ""),
            entry=str(value.get("entry") or ""),
            signature=str(value.get("signature") or ""),
            description=str(value.get("description") or ""),
        )

    @property
    def entry_path(self) -> str:
        path, _, func = self.entry.rpartition(":")
        return path if path and func else ""

    @property
    def entry_function(self) -> str:
        path, _, func = self.entry.rpartition(":")
        return func if path and func else ""


@dataclass(frozen=True)
class ContractPrompt:
    name: str
    path: str
    description: str = ""

    @classmethod
    def from_any(cls, value: dict[str, Any]) -> ContractPrompt:
        return cls(
            name=str(value.get("name") or ""),
            path=str(value.get("path") or ""),
            description=str(value.get("description") or ""),
        )


@dataclass(frozen=True)
class SkillContract:
    skill_id: str
    functions: tuple[ContractFunction, ...] = ()
    prompts: tuple[ContractPrompt, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, fallback_skill_id: str = "") -> SkillContract:
        skill_id = str(data.get("skill_id") or fallback_skill_id)
        functions = tuple(
            ContractFunction.from_any(dict(value))
            for value in (data.get("functions") or ())
            if isinstance(value, dict)
        )
        prompts = tuple(
            ContractPrompt.from_any(dict(value))
            for value in (data.get("prompts") or ())
            if isinstance(value, dict)
        )
        return cls(
            skill_id=skill_id,
            functions=functions,
            prompts=prompts,
            raw=dict(data),
        )


def load_contract_for_skill(skill_root: str | Path) -> SkillContract | None:
    root = Path(skill_root)
    path = root / CONTRACT_FILE
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return SkillContract.from_mapping(data, fallback_skill_id=root.name)


def resolve_function_signature(
    function: ContractFunction,
    skill_root: Path,
) -> str:
    """Truthful call signature for one contract function.

    A declared ``signature`` wins; otherwise derived from the source AST.
    """
    if function.signature.strip():
        return function.signature.strip()
    node = _entry_function_def(function, skill_root)
    if node is None:
        return ""
    return f"{function.name}{_render_signature(node)}"


def _entry_function_def(
    function: ContractFunction,
    skill_root: Path,
) -> ast.FunctionDef | None:
    source_path = skill_root / function.entry_path
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function.entry_function:
            return node
    return None


def _render_signature(node: ast.FunctionDef) -> str:
    args = node.args
    parts: list[str] = []
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        parts.append(_render_arg(arg, default))
    if args.posonlyargs and len(parts) >= len(args.posonlyargs):
        parts.insert(len(args.posonlyargs), "/")
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(_render_arg(arg, default))
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")
    return "(" + ", ".join(parts) + ")"


def _render_arg(arg: ast.arg, default: ast.expr | None) -> str:
    rendered = arg.arg
    if default is not None:
        rendered += f"={ast.unparse(default)}"
    return rendered
