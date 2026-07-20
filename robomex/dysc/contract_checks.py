"""Load-time consistency checks for M2 skill contracts.

The contract is the machine-authoritative face of a skill.  These checks run
when a library's contracts are loaded, so a broken declaration (missing file,
unknown exit event, prose that contradicts the capability grant) fails fast
with an actionable message instead of surfacing mid-episode as a dead
recovery edge or a missing sandbox primitive.

Every check reports *all* problems it finds rather than stopping at the
first, because a skill author fixing a contract wants the complete list.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from robomex.core.edge_events import (
    EDGE_EVENT_SUCCESS,
    EDGE_EVENTS,
    KNOWN_EDGE_EVENTS,
    normalize_edge_event,
)
from robomex.dysc.contracts import ContractFunction, SkillContract

_PYTHON_FENCE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def contract_consistency_errors(
    contract: SkillContract,
    skill_root: Path,
) -> list[str]:
    """Return every consistency problem between one contract and its package."""

    errors: list[str] = []
    prefix = f"skill {contract.skill_id!r}"
    errors.extend(
        f"{prefix}: {msg}" for msg in _exit_condition_errors(contract)
    )
    for function in contract.functions:
        errors.extend(
            f"{prefix}, function {function.name!r}: {msg}"
            for msg in _function_errors(function, skill_root)
        )
    seen_function_names = [f.name for f in contract.functions]
    if len(seen_function_names) != len(set(seen_function_names)):
        errors.append(f"{prefix}: duplicate function names in `functions:`.")
    for prompt in contract.prompts:
        if not prompt.name.strip():
            errors.append(f"{prefix}: a prompt entry is missing `name`.")
            continue
        if not prompt.path.strip():
            errors.append(f"{prefix}, prompt {prompt.name!r}: missing `path`.")
            continue
        resolved = skill_root / prompt.path
        if not resolved.is_file():
            errors.append(
                f"{prefix}, prompt {prompt.name!r}: file {prompt.path!r} not found "
                f"under the skill package ({resolved})."
            )
    seen_prompt_names = [p.name for p in contract.prompts]
    if len(seen_prompt_names) != len(set(seen_prompt_names)):
        errors.append(f"{prefix}: duplicate prompt names in `prompts:`.")
    errors.extend(
        f"{prefix}: {msg}"
        for msg in _prose_capability_errors(contract, skill_root)
    )
    return errors


def resolve_function_signature(
    function: ContractFunction,
    skill_root: Path,
) -> str:
    """The truthful call signature for one contract function.

    A declared ``signature`` wins; otherwise it is derived from the source
    AST so the prompt never shows an invented call shape.  Returns '' when
    the entry cannot be resolved (the loader check reports that separately).
    """

    if function.signature.strip():
        return function.signature.strip()
    node = _entry_function_def(function, skill_root)
    if node is None:
        return ""
    return f"{function.name}{_render_signature(node)}"


# ---- exit conditions ---------------------------------------------------------


def _exit_condition_errors(contract: SkillContract) -> list[str]:
    errors: list[str] = []
    for event in contract.exit_conditions:
        normalized = normalize_edge_event(event)
        if normalized not in KNOWN_EDGE_EVENTS:
            errors.append(
                f"exit condition {event!r} is not in the closed edge-event "
                f"vocabulary ({', '.join(EDGE_EVENTS)})."
            )
        elif normalized != event:
            errors.append(
                f"exit condition {event!r} must use the canonical spelling "
                f"{normalized!r}."
            )
    if contract.exit_conditions and EDGE_EVENT_SUCCESS not in contract.exit_conditions:
        errors.append(
            "exit_conditions must declare `success` (every routable node needs "
            "a success exit)."
        )
    return errors


# ---- functions ---------------------------------------------------------------


def _function_errors(function: ContractFunction, skill_root: Path) -> list[str]:
    errors: list[str] = []
    if not function.name.isidentifier():
        errors.append(f"name {function.name!r} is not a valid Python identifier.")
    if not function.entry_path or not function.entry_function:
        errors.append(
            f"entry {function.entry!r} must have the form "
            "'<relative_path>.py:<function_name>'."
        )
        return errors
    source_path = skill_root / function.entry_path
    if not source_path.is_file():
        errors.append(
            f"entry file {function.entry_path!r} not found under the skill "
            f"package ({source_path})."
        )
        return errors
    node = _entry_function_def(function, skill_root)
    if node is None:
        errors.append(
            f"function {function.entry_function!r} is not defined at module "
            f"level in {function.entry_path!r}."
        )
    return errors


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
    """Render ``(a, b, *, c=1)`` from one function-def AST node."""

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


# ---- prose / capability consistency ------------------------------------------


def _prose_capability_errors(
    contract: SkillContract,
    skill_root: Path,
) -> list[str]:
    """Flag SKILL.md reference code that calls a forbidden env API.

    Best-effort by design: only fenced ```python blocks are inspected (via the
    same AST call scan the runtime enforces), because prose mentions of an API
    name are frequently negative instructions ("do not call get_observation").
    A reference-code block that *calls* a forbidden API is the exact
    contradiction that produced M1's B2 bug, and is always a real error.
    """

    if not contract.forbidden_capabilities:
        return []
    skill_md = skill_root / "SKILL.md"
    if not skill_md.is_file():
        return []
    # Local import: capabilities lives in robomex.authoring; importing it at
    # module scope would couple the dysc leaf package upward at import time.
    from robomex.authoring.capabilities import CALL_EFFECTS, called_function_names

    forbidden = set(contract.forbidden_capabilities)
    errors: list[str] = []
    text = skill_md.read_text(encoding="utf-8")
    for index, block in enumerate(_PYTHON_FENCE.findall(text)):
        try:
            calls = called_function_names(block)
        except SyntaxError:
            # Illustrative fragments are allowed to be non-parseable.
            continue
        offending = sorted(
            call
            for call in calls
            if CALL_EFFECTS.get(call.rsplit(".", 1)[-1]) in forbidden
        )
        if offending:
            errors.append(
                f"SKILL.md python block #{index} calls {', '.join(offending)}, "
                "which the contract lists under forbidden_capabilities — the "
                "runtime will deny it, so the reference code teaches a dead "
                "path. Fix the code or the contract."
            )
    return errors
