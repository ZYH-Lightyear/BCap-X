"""Restricted code-authored monitors for RoboMEx v2.

A Monitor Author may write a small Python ``evaluate(sample)`` function, but
the function is compiled against a deliberately tiny language.  It receives a
plain, read-only JSON mapping and may only return a structured finding.  It
cannot import modules, inspect objects, access files, call robot APIs, loop, or
mutate runtime state.  The deterministic runtime remains the only component
that can turn a critical finding into a stop request.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import CodeType, MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from robomex.runtime.events import (
    FindingSeverity,
    MonitorFinding,
    MonitorFindingKind,
)


class MonitorCompileError(ValueError):
    """Monitor source violates the restricted-language contract."""


class MonitorRuntimeError(RuntimeError):
    """A compiled monitor was invoked outside its sealed action context."""


class MonitorHook(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    PHASE = "phase"
    WAYPOINT = "waypoint"
    CONTROL = "control"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class MonitorProgramSpec(_StrictModel):
    schema_id: Literal["robomex.monitor_program.v1"] = "robomex.monitor_program.v1"
    monitor_id: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=16_384)
    hook: MonitorHook = MonitorHook.PHASE
    allowed_signals: tuple[str, ...] = Field(min_length=1)
    debounce_count: int = Field(default=1, ge=1, le=100)
    max_ast_nodes: int = Field(default=256, ge=8, le=2_048)
    max_runtime_ms: float = Field(default=5.0, gt=0.0, le=100.0, allow_inf_nan=False)


class MonitorDecision(_StrictModel):
    finding: MonitorFindingKind
    severity: FindingSeverity = FindingSeverity.WARNING
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    details: dict[str, JsonValue] = Field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledMonitorProgram:
    spec: MonitorProgramSpec
    digest: str
    code: CodeType


@dataclass(frozen=True)
class MonitorEvaluation:
    finding: MonitorFinding | None
    stop_requested: bool
    observable: bool
    elapsed_ms: float
    debounce_progress: int


_SAFE_CALLS: Mapping[str, Any] = MappingProxyType(
    {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "float": float,
        "int": int,
        "len": len,
        "max": max,
        "min": min,
        "round": round,
    }
)

_ALLOWED_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.AnnAssign,
    ast.If,
    ast.Return,
    ast.Expr,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Dict,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Subscript,
    ast.Slice,
    ast.UnaryOp,
    ast.BinOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.keyword,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
)


class MonitorCompiler:
    """Compile the code-shaped monitor DSL after exhaustive AST checks."""

    def compile(self, spec: MonitorProgramSpec | Mapping[str, Any]) -> CompiledMonitorProgram:
        validated = (
            spec if isinstance(spec, MonitorProgramSpec) else MonitorProgramSpec.model_validate(spec)
        )
        if len(set(validated.allowed_signals)) != len(validated.allowed_signals):
            raise MonitorCompileError("allowed_signals contains duplicates")
        if any(not name.strip() or name.startswith("_") for name in validated.allowed_signals):
            raise MonitorCompileError("signal names must be public, non-empty identifiers")
        try:
            tree = ast.parse(validated.source, mode="exec")
        except SyntaxError as exc:
            raise MonitorCompileError(f"invalid monitor syntax: {exc.msg}") from exc
        nodes = list(ast.walk(tree))
        if len(nodes) > validated.max_ast_nodes:
            raise MonitorCompileError(
                f"monitor AST has {len(nodes)} nodes; limit is {validated.max_ast_nodes}"
            )
        unsupported = next((node for node in nodes if not isinstance(node, _ALLOWED_NODES)), None)
        if unsupported is not None:
            raise MonitorCompileError(
                f"monitor syntax {type(unsupported).__name__} is not allowed"
            )
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise MonitorCompileError("source must contain exactly one evaluate(sample) function")
        function = tree.body[0]
        if function.name != "evaluate" or function.decorator_list:
            raise MonitorCompileError("the only function must be undecorated evaluate(sample)")
        args = function.args
        if (
            len(args.args) != 1
            or args.args[0].arg != "sample"
            or args.posonlyargs
            or args.kwonlyargs
            or args.vararg is not None
            or args.kwarg is not None
            or args.defaults
            or args.kw_defaults
        ):
            raise MonitorCompileError("evaluate must accept exactly one argument named sample")
        for node in nodes:
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise MonitorCompileError("dunder names are forbidden")
            if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_CALLS
            ):
                raise MonitorCompileError(
                    "only deterministic allowlisted function calls are permitted"
                )
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (bytes, complex)):
                    raise MonitorCompileError("bytes and complex constants are forbidden")
                if isinstance(node.value, float) and not math.isfinite(node.value):
                    raise MonitorCompileError("non-finite constants are forbidden")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
                    raise MonitorCompileError("power exponent must be a small integer literal")
                if abs(node.right.value) > 8:
                    raise MonitorCompileError("power exponent exceeds the monitor bound")
        canonical = json.dumps(
            validated.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = "sha256:" + hashlib.sha256(canonical).hexdigest()
        return CompiledMonitorProgram(
            spec=validated,
            digest=digest,
            code=compile(tree, f"<robomex-monitor:{validated.monitor_id}>", "exec"),
        )


class MonitorRuntime:
    """Evaluate one program sealed to one action/plan identity.

    The runtime does not expose a callback that can move the robot.  It returns
    ``stop_requested`` for the Action Supervisor to act upon.  Exceptions,
    missing signals, non-finite telemetry, and budget overruns become a
    fail-closed ``unobservable`` finding.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        workflow_id: str,
        action_id: str,
        plan_digest: str,
        program: CompiledMonitorProgram,
        stop_severities: frozenset[FindingSeverity] = frozenset({FindingSeverity.CRITICAL}),
    ) -> None:
        for name, value in {
            "episode_id": episode_id,
            "workflow_id": workflow_id,
            "action_id": action_id,
            "plan_digest": plan_digest,
        }.items():
            if not str(value).strip():
                raise ValueError(f"{name} must not be empty")
        self.episode_id = episode_id
        self.workflow_id = workflow_id
        self.action_id = action_id
        self.plan_digest = plan_digest
        self.program = program
        self.stop_severities = frozenset(FindingSeverity(value) for value in stop_severities)
        self._last_signature: tuple[str, str] | None = None
        self._consecutive = 0
        self._last_sequence = -1

    def evaluate(
        self,
        sample: Mapping[str, Any],
        *,
        sequence: int,
        hook: MonitorHook,
        action_id: str,
        plan_digest: str,
    ) -> MonitorEvaluation:
        if action_id != self.action_id or plan_digest != self.plan_digest:
            raise MonitorRuntimeError("stale monitor invocation does not match the sealed action")
        if hook != self.program.spec.hook:
            raise MonitorRuntimeError("monitor invoked at an undeclared hook")
        if sequence <= self._last_sequence:
            raise MonitorRuntimeError("monitor samples must have a strictly increasing sequence")
        self._last_sequence = sequence
        started = time.perf_counter_ns()
        decision: MonitorDecision | None
        try:
            safe_sample = self._validate_sample(sample)
            globals_dict = {"__builtins__": {}, **_SAFE_CALLS}
            locals_dict: dict[str, Any] = {}
            exec(self.program.code, globals_dict, locals_dict)  # noqa: S102 - AST-restricted DSL
            raw = locals_dict["evaluate"](safe_sample)
            decision = None if raw is None else MonitorDecision.model_validate(raw)
        except (ArithmeticError, KeyError, TypeError, ValueError, ValidationError) as exc:
            decision = MonitorDecision(
                finding=MonitorFindingKind.UNOBSERVABLE,
                severity=FindingSeverity.CRITICAL,
                details={"reason": type(exc).__name__},
            )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if elapsed_ms > self.program.spec.max_runtime_ms:
            decision = MonitorDecision(
                finding=MonitorFindingKind.UNOBSERVABLE,
                severity=FindingSeverity.CRITICAL,
                details={"reason": "runtime_budget_exceeded"},
            )
        if decision is None:
            self._last_signature = None
            self._consecutive = 0
            return MonitorEvaluation(
                finding=None,
                stop_requested=False,
                observable=True,
                elapsed_ms=elapsed_ms,
                debounce_progress=0,
            )
        signature = (decision.finding.value, decision.severity.value)
        if signature == self._last_signature:
            self._consecutive += 1
        else:
            self._last_signature = signature
            self._consecutive = 1
        if self._consecutive < self.program.spec.debounce_count:
            return MonitorEvaluation(
                finding=None,
                stop_requested=False,
                observable=decision.finding is not MonitorFindingKind.UNOBSERVABLE,
                elapsed_ms=elapsed_ms,
                debounce_progress=self._consecutive,
            )
        finding = MonitorFinding(
            episode_id=self.episode_id,
            workflow_id=self.workflow_id,
            source="monitor_runtime",
            monitor_id=self.program.spec.monitor_id,
            finding=decision.finding,
            severity=decision.severity,
            confidence=decision.confidence,
            evidence_refs=decision.evidence_refs,
            action_id=self.action_id,
            details={
                **decision.details,
                "hook": hook.value,
                "sequence": sequence,
                "monitor_digest": self.program.digest,
                "plan_digest": self.plan_digest,
            },
        )
        return MonitorEvaluation(
            finding=finding,
            stop_requested=decision.severity in self.stop_severities,
            observable=decision.finding is not MonitorFindingKind.UNOBSERVABLE,
            elapsed_ms=elapsed_ms,
            debounce_progress=self._consecutive,
        )

    def _validate_sample(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(sample, Mapping):
            raise TypeError("sample must be a mapping")
        missing = set(self.program.spec.allowed_signals) - set(sample)
        unknown = set(sample) - set(self.program.spec.allowed_signals)
        if missing:
            raise KeyError(f"missing signals: {', '.join(sorted(missing))}")
        if unknown:
            raise ValueError(f"undeclared signals: {', '.join(sorted(unknown))}")
        normalized = _json_finite(dict(sample))
        return MappingProxyType(normalized)


def _json_finite(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("monitor telemetry contains NaN or infinity")
        return value
    if isinstance(value, list | tuple):
        return tuple(_json_finite(item) for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("monitor telemetry keys must be strings")
        return MappingProxyType({key: _json_finite(item) for key, item in value.items()})
    raise TypeError(f"unsupported monitor telemetry type: {type(value).__name__}")


__all__ = [
    "CompiledMonitorProgram",
    "MonitorCompileError",
    "MonitorCompiler",
    "MonitorDecision",
    "MonitorEvaluation",
    "MonitorHook",
    "MonitorProgramSpec",
    "MonitorRuntime",
    "MonitorRuntimeError",
]
