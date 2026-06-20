"""Context construction for the (independent) Reference-Anchored Verifier.

Design (see docs §5.6): the Verifier is an independent role that may write its own
judge code, but its context is built on the *facts vs interpretation* line — it
gets everything FACTUAL about what the executor did (which skills, the claimed
result, a sanitized op-trace, the authored rubrics + reference primitives) and
NONE of the executor's reasoning / self-assessment, so its blind spots stay
uncorrelated with the executor's.

This module provides the data plane:

- ``sanitize_code`` / ``build_op_trace``: turn executor code into a comment-free,
  CoT-free operation trace (the "actual flow", default-on, low taint).
- ``collect_verify_resources``: the VerifyRouter — map the skills the executor
  used to their ``ref/verify.md`` rubric + ``scripts/verify.py`` primitives.
- ``VerifierContext``: the assembled, render-able context.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


class _DropStringExprs(ast.NodeTransformer):
    """Remove bare string-literal statements (docstrings / pseudo-comments)."""

    def visit_Expr(self, node: ast.Expr) -> Any:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return node


def sanitize_code(code: str) -> str:
    """Strip comments, docstrings and CoT prose from executor code → ops only.

    ``ast.unparse`` already drops comments; we additionally remove bare string
    statements (where agents often stash natural-language reasoning). If the code
    does not parse, fall back to dropping ``#`` comment lines so we still avoid
    leaking inline reasoning.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "\n".join(
            line for line in code.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ).strip()
    tree = _DropStringExprs().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).strip()


def build_op_trace(turns: Iterable[Any], only_successful: bool = True) -> list[str]:
    """Sanitized per-turn op-trace from an ``AgentTrace``'s turns.

    Each turn is expected to expose ``.code`` and ``.execution.ok``. Failed turns
    are dropped by default (they did not contribute to the final artifacts).
    """
    trace: list[str] = []
    for turn in turns:
        if only_successful and not getattr(getattr(turn, "execution", None), "ok", True):
            continue
        code = getattr(turn, "code", "") or ""
        cleaned = sanitize_code(code)
        if cleaned:
            trace.append(cleaned)
    return trace


@dataclass(frozen=True)
class VerifyResource:
    """The authored verification assets routed in for one skill."""

    skill_id: str
    rubric_text: str = ""
    rubric_path: str | None = None
    verifier_path: str | None = None


def collect_verify_resources(
    skills: Iterable[Any],
    skill_ids: Sequence[str],
) -> dict[str, VerifyResource]:
    """VerifyRouter: for each used skill, gather its rubric + verifier primitives.

    ``skills`` is any iterable of ``Skill`` objects (e.g. ``[r.skill for r in
    library.all()]``); only those whose ``skill_id`` is in ``skill_ids`` are kept.
    """
    wanted = set(skill_ids)
    by_id = {s.skill_id: s for s in skills if s.skill_id in wanted}
    resources: dict[str, VerifyResource] = {}
    for skill_id in skill_ids:
        skill = by_id.get(skill_id)
        if skill is None:
            continue
        rubric_path = skill.verify_doc_path()
        verifier_path = skill.verifier_path()
        resources[skill_id] = VerifyResource(
            skill_id=skill_id,
            rubric_text=Path(rubric_path).read_text() if rubric_path else "",
            rubric_path=str(rubric_path) if rubric_path else None,
            verifier_path=str(verifier_path) if verifier_path else None,
        )
    return resources


@dataclass(frozen=True)
class VerifierContext:
    """Everything an independent Verifier is allowed to see — facts, not narrative.

    Deliberately excludes the executor's chain-of-thought / self-assessment. The
    raw full code is *not* embedded here; it is fetched on demand (progressive
    disclosure) only if the Verifier asks.
    """

    sub_goal: str
    skills_used: tuple[str, ...] = ()
    claim: dict[str, Any] = field(default_factory=dict)
    op_trace: tuple[str, ...] = ()
    resources: dict[str, VerifyResource] = field(default_factory=dict)
    expected_decomposition: str = ""
    executor_stdout: str = ""

    def rubrics_text(self) -> str:
        """Concatenated authored rubrics for all routed skills."""
        chunks = [
            f"### {r.skill_id}\n{r.rubric_text.strip()}"
            for r in self.resources.values() if r.rubric_text.strip()
        ]
        return "\n\n".join(chunks)

    def render(self) -> str:
        """A human/LLM-readable layout of the (fact-only) verification context."""
        lines = [
            f"Sub-goal to verify: {self.sub_goal}",
            f"Skills the executor used: {', '.join(self.skills_used) or '(unknown)'}",
            "",
            "Executor's CLAIM (verify or refute — do NOT assume it is true):",
            _format_claim(self.claim),
        ]
        if self.executor_stdout.strip():
            lines += ["", "Executor's printed output (real stdout — facts, not its reasoning):",
                      self.executor_stdout.strip()]
        if self.expected_decomposition.strip():
            lines += ["", "Expected flow (authored decomposition):",
                      self.expected_decomposition.strip()]
        if self.op_trace:
            lines += ["", "Actual flow (sanitized op-trace):"]
            lines += [f"  [{i}] " + op.replace("\n", "\n      ")
                      for i, op in enumerate(self.op_trace)]
        rubrics = self.rubrics_text()
        if rubrics:
            lines += ["", "Authored success rubrics (ref/verify.md):", rubrics]
        return "\n".join(lines)


def _format_claim(claim: dict[str, Any]) -> str:
    if not claim:
        return "  (no manifest claim provided)"
    return "\n".join(f"  {k}: {v}" for k, v in claim.items())
