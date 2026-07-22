"""Restricted v1-CodingAgent provider for the RoboMEx v2 actor runtime.

The provider is the compatibility seam between the existing skill-retrieval
``CodingAgentSubAgent`` loop and the v2 episode data plane.  A coding worker may
read admitted typed artifacts and author new typed proposals.  It never owns a
physical-world effect: exact motion and gripper specs still have to pass through
the trusted ``system_action`` runner.

The boundary is deliberately enforced twice.  ``ActorHandle`` checks the
profile/invocation ceilings, while this provider rejects physical capability
ceilings, non-empty effect requests, unknown calls, and dynamically-shaped
calls before generated Python reaches the configured block executor.
"""

from __future__ import annotations

import ast
import copy
import fcntl
import hashlib
import inspect
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from robomex.agents.subagents import CodingAgentSubAgent, SubAgentRequest, SubAgentResult
from robomex.authoring.capabilities import (
    ACTIVE_PERCEPTION,
    ARTIFACT_WRITE,
    CALL_EFFECTS,
    DYNAMIC_OR_PROCESS_CALLS,
    GEOMETRY_COMPUTE,
    GRIPPER_CONTROL,
    OBJECT_MANIPULATION,
    PERCEPTION_READ,
    ROBOT_MOTION,
    SAFE_BUILTINS,
    SIMULATION_PROBE,
)
from robomex.core.coder import CompletionPolicy
from robomex.core.sandbox import ActionBlockStatus, BlockExecutionResult, SemanticActionBlock
from robomex.data import EpisodeDataPlane, ResolvedArtifactRef
from robomex.orchestration.actors import (
    ActorAuthorityError,
    ActorConflictError,
    ActorIsolation,
    ActorProfile,
    ActorStateError,
    InvocationSpec,
)
from robomex.orchestration.episode import (
    ActivationExecutionResult,
    ArtifactEmission,
    InvocationUsage,
)
from robomex.prompts.authoring import build_agent_system_prompt, output_contract_for_ports
from robomex.runtime.events import ControlOutcome
from robomex.skills import SkillLibrary

_LEDGER_SCHEMA = "robomex.coding_provider_invocation.v1"
_RUNTIME_CONTEXT_SCHEMA = "robomex.coding_runtime_context.v1"
_NODE_CONFIG_SCHEMA = "robomex.coding_node_config.v1"
_RESERVED_CONTEXT_NAMES = frozenset({"RUNTIME_CONTEXT_V1", "NODE_CONFIG_V1"})
_PHYSICAL_EFFECTS = frozenset(
    {ACTIVE_PERCEPTION, ROBOT_MOTION, OBJECT_MANIPULATION, GRIPPER_CONTROL}
)
_SIDECAR_DENIED_EFFECTS = frozenset(
    {ARTIFACT_WRITE, PERCEPTION_READ, SIMULATION_PROBE, *_PHYSICAL_EFFECTS}
)
_PHYSICAL_API_NAMES = frozenset(
    name for name, effect in CALL_EFFECTS.items() if effect in _PHYSICAL_EFFECTS
)
_PROCESS_MODULES = frozenset({"ctypes", "multiprocessing", "subprocess"})
_SAFE_IMPORT_ROOTS = frozenset(
    {
        "collections",
        "dataclasses",
        "decimal",
        "fractions",
        "functools",
        "itertools",
        "json",
        "math",
        "numpy",
        "operator",
        "pathlib",
        "scipy",
        "statistics",
    }
)
# The legacy general-purpose policy treats introspection as ordinary Python.
# This stricter provider does not: passing a physical callable through
# ``vars``/``getattr`` into a safe builtin would otherwise hide the effect.
_DENIED_INTROSPECTION_CALLS = frozenset({"delattr", "dir", "getattr", "setattr", "vars"})

# Coding workers may calculate exact action specs, but they may not execute an
# action or move a sensor.  ``active_perception`` is intentionally absent: in
# the legacy capability table it contains a real robot movement primitive.
PROPOSAL_CAPABILITIES = frozenset(
    {
        PERCEPTION_READ,
        ARTIFACT_WRITE,
        GEOMETRY_COMPUTE,
        SIMULATION_PROBE,
    }
)

# Graph capabilities describe semantic authority; executor capabilities
# describe which legacy CapX API families a generated block may call. Keeping
# this translation closed prevents ``motion.plan`` from ever becoming
# physical ``robot_motion`` authority.
DEFAULT_CODING_CAPABILITY_GRANTS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        PERCEPTION_READ: frozenset({PERCEPTION_READ}),
        ARTIFACT_WRITE: frozenset({ARTIFACT_WRITE}),
        GEOMETRY_COMPUTE: frozenset({GEOMETRY_COMPUTE}),
        SIMULATION_PROBE: frozenset({SIMULATION_PROBE}),
        "motion.plan": frozenset({GEOMETRY_COMPUTE}),
        "monitor.author": frozenset(),
        "perception.observe": frozenset({PERCEPTION_READ}),
        "perception.geometry": frozenset({PERCEPTION_READ, GEOMETRY_COMPUTE}),
        # State proposals are immutable typed artifacts.  The runtime-owned
        # reducer remains the sole state writer, so this semantic capability
        # grants no legacy executor/API authority.
        "state.propose": frozenset(),
    }
)

_PURE_NUMPY_CALLS = frozenset(
    {
        "numpy.abs",
        "numpy.all",
        "numpy.any",
        "numpy.arange",
        "numpy.argmax",
        "numpy.argmin",
        "numpy.argsort",
        "numpy.array",
        "numpy.asarray",
        "numpy.arctan2",
        "numpy.clip",
        "numpy.concatenate",
        "numpy.cos",
        "numpy.cross",
        "numpy.dot",
        "numpy.eye",
        "numpy.hstack",
        "numpy.isfinite",
        "numpy.linalg.det",
        "numpy.linalg.eigh",
        "numpy.linalg.eigvalsh",
        "numpy.linalg.inv",
        "numpy.linalg.norm",
        "numpy.linalg.solve",
        "numpy.linalg.svd",
        "numpy.linspace",
        "numpy.matmul",
        "numpy.max",
        "numpy.mean",
        "numpy.median",
        "numpy.min",
        "numpy.ones",
        "numpy.sin",
        "numpy.sort",
        "numpy.sqrt",
        "numpy.stack",
        "numpy.std",
        "numpy.sum",
        "numpy.tan",
        "numpy.unique",
        "numpy.vstack",
        "numpy.where",
        "numpy.zeros",
    }
)
_PURE_MATH_CALLS = frozenset(
    {
        "math.acos",
        "math.asin",
        "math.atan",
        "math.atan2",
        "math.ceil",
        "math.cos",
        "math.degrees",
        "math.dist",
        "math.exp",
        "math.fabs",
        "math.floor",
        "math.fmod",
        "math.fsum",
        "math.hypot",
        "math.isclose",
        "math.isfinite",
        "math.isinf",
        "math.isnan",
        "math.log",
        "math.log10",
        "math.prod",
        "math.radians",
        "math.sin",
        "math.sqrt",
        "math.tan",
        "statistics.fmean",
        "statistics.mean",
        "statistics.median",
        "statistics.pstdev",
        "statistics.stdev",
    }
)
_PURE_DATA_METHODS = frozenset(
    {
        "append",
        "astype",
        "clear",
        "clip",
        "copy",
        "count",
        "endswith",
        "extend",
        "flatten",
        "get",
        "index",
        "insert",
        "items",
        "join",
        "keys",
        "lower",
        "max",
        "mean",
        "min",
        "pop",
        "ravel",
        "remove",
        "reshape",
        "reverse",
        "setdefault",
        "sort",
        "split",
        "startswith",
        "std",
        "strip",
        "sum",
        "tolist",
        "transpose",
        "update",
        "upper",
        "values",
    }
)
_PURE_EXTRA_BUILTINS = frozenset({"all", "any"})


class CodingProviderError(RuntimeError):
    """Base error for a malformed or unsafe coding-worker invocation."""


class CodingProviderContractError(CodingProviderError, ValueError):
    """Raised when typed input/output or durable replay validation fails."""


class CodingRuntimeContextV1(BaseModel):
    """Provider-authored invocation identity exposed read-only to coding workers."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["robomex.coding_runtime_context.v1"] = _RUNTIME_CONTEXT_SCHEMA
    episode_id: str = Field(min_length=1, max_length=256)
    run_id: str = Field(min_length=1, max_length=256)
    node_id: str = Field(min_length=1, max_length=256)
    attempt: int = Field(ge=1)
    invocation_id: str = Field(min_length=1, max_length=256)
    idempotency_key: str = Field(min_length=1, max_length=256)
    graph_id: str = Field(min_length=1, max_length=256)
    graph_revision: int = Field(ge=1)
    graph_digest: str = Field(min_length=1, max_length=256)


class CodingNodeConfigV1(BaseModel):
    """Closed projection of graph params a proposal worker may consume."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal["robomex.coding_node_config.v1"] = _NODE_CONFIG_SCHEMA
    objective: str | None = Field(default=None, min_length=1, max_length=4096)
    monitor_hook: Literal["control"] | None = None
    allowed_signals: tuple[str, ...] = ()
    debounce_count: int | None = Field(default=None, ge=1, le=100)
    target_status: Literal["verified_held", "not_held"] | None = None
    plan_kind: (
        Literal[
            "transport_to_safe_hover",
            "bounded_correction",
            "descend_to_release",
            "safe_retreat",
        ]
        | None
    ) = None
    tcp_frame_id: str | None = Field(default=None, min_length=1, max_length=128)
    planner_backend: str | None = Field(default=None, min_length=1, max_length=128)
    robot_model_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    planner_configuration_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    required_alignment: Literal["within_tolerance"] | None = None
    error_space: Literal["translation_plus_yaw"] | None = None
    fail_on_mixed_snapshot: bool | None = None
    phase: Literal["pre_release"] | None = None
    predicate: Literal["supported_by"] | None = None
    value: Literal["asserted"] | None = None
    tolerance_xy_m: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    tolerance_z_m: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    tolerance_yaw_rad: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _closed_groups(self) -> CodingNodeConfigV1:
        monitor_values = (self.monitor_hook, self.debounce_count, self.allowed_signals)
        if any(value not in (None, (), "") for value in monitor_values) and (
            self.monitor_hook != "control"
            or self.debounce_count is None
            or self.allowed_signals
            != ("attachment_status", "held_entity_visible", "identity_match")
        ):
            raise ValueError("monitor config must use the audited control signal contract")
        tolerances = (
            self.tolerance_xy_m,
            self.tolerance_z_m,
            self.tolerance_yaw_rad,
        )
        if any(value is not None for value in tolerances) and any(
            value is None for value in tolerances
        ):
            raise ValueError("alignment tolerances must be supplied as one complete triple")
        if self.plan_kind is not None and (
            self.tcp_frame_id is None or self.planner_backend is None
        ):
            raise ValueError("sealed motion config requires TCP frame and planner backend")
        return self

    def canonical_mapping(self) -> dict[str, Any]:
        values = self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
        values["schema_version"] = self.schema_version
        return dict(sorted(values.items()))


@dataclass(frozen=True)
class _CodingContextBinding:
    runtime_context: CodingRuntimeContextV1
    node_config: CodingNodeConfigV1
    content_digest: str


@dataclass(frozen=True)
class CodingInvocationAudit:
    """Small provider-owned audit record; traces remain in the worker directory."""

    invocation_id: str
    idempotency_key: str
    outcome: ControlOutcome
    loaded_skill_ids: tuple[str, ...] = ()
    turns: int = 0
    replayed: bool = False
    context_digest: str = ""


@dataclass
class CodingWorkerRuntime:
    """Provider state isolated to one ActorHandle."""

    profile: ActorProfile
    isolation: ActorIsolation
    executor: Any
    policy: CompletionPolicy
    audits: list[CodingInvocationAudit] = field(default_factory=list)
    suspended: bool = False
    retired: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


ExecutorFactory = Callable[[ActorProfile, ActorIsolation], Any]
PolicyFactory = Callable[[ActorProfile, ActorIsolation], CompletionPolicy]


def _audit_trusted_skill_sidecars(
    library: SkillLibrary,
    skill_ids: frozenset[str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Resolve an operator allowlist and reject executable import-time side effects."""

    from robomex.dysc.contracts import load_contract_for_skill

    function_names: set[str] = set()
    binding_names: set[str] = set()
    allowed_imports = frozenset(
        {"__future__", "collections", "hashlib", "json", "robomex", "typing"}
    )
    for skill_id in sorted(skill_ids):
        record = library.get(skill_id)
        root = record.skill.root
        if root is None:
            raise CodingProviderContractError(
                f"trusted skill {skill_id!r} has no filesystem package root"
            )
        contract = load_contract_for_skill(root)
        if contract is None or not contract.functions:
            raise CodingProviderContractError(
                f"trusted skill {skill_id!r} has no contracted canonical function"
            )
        binding_names.add(f"skill_bindings_{skill_id}")
        for function in contract.functions:
            source = (root / function.entry_path).resolve()
            try:
                source.relative_to(root.resolve())
            except ValueError as exc:
                raise CodingProviderContractError(
                    f"trusted function {function.name!r} escapes its skill package"
                ) from exc
            try:
                tree = ast.parse(source.read_text(encoding="utf-8"))
            except (OSError, SyntaxError) as exc:
                raise CodingProviderContractError(
                    f"trusted function source is unreadable: {source!s}"
                ) from exc
            for statement in tree.body:
                if isinstance(statement, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
                    continue
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                    continue
                if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    value = statement.value
                    try:
                        ast.literal_eval(value) if value is not None else None
                    except (TypeError, ValueError) as exc:
                        raise CodingProviderContractError(
                            f"trusted sidecar {source!s} has executable module assignment"
                        ) from exc
                    continue
                raise CodingProviderContractError(
                    f"trusted sidecar {source!s} has executable top-level "
                    f"{type(statement).__name__}"
                )
            imports = {
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in (
                    node.names
                    if isinstance(node, ast.Import)
                    else [ast.alias(name=str(node.module or ""))]
                )
            }
            denied_imports = imports - allowed_imports
            if denied_imports:
                raise CodingProviderContractError(
                    f"trusted sidecar {source!s} imports unapproved roots: "
                    + ", ".join(sorted(denied_imports))
                )
            aliases = _ProposalCallPolicy._aliases(tree)
            denied_calls = sorted(
                {
                    name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    for name in (_qualified_ast_name(node.func, aliases) or "<dynamic>",)
                    if name in DYNAMIC_OR_PROCESS_CALLS
                    or _call_effect(name) in _SIDECAR_DENIED_EFFECTS
                }
            )
            if denied_calls:
                raise CodingProviderContractError(
                    f"trusted sidecar {source!s} contains effectful/process calls: "
                    + ", ".join(denied_calls)
                )
            function_names.add(function.name)
    return frozenset(function_names), frozenset(binding_names)


def _qualified_ast_name(node: ast.expr, aliases: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if not isinstance(node, ast.Attribute):
        return None
    parent = _qualified_ast_name(node.value, aliases)
    return f"{parent}.{node.attr}" if parent else None


def _call_effect(name: str) -> str | None:
    return CALL_EFFECTS.get(name) or CALL_EFFECTS.get(name.rsplit(".", 1)[-1])


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for value in target.elts for name in _assigned_names(value)}
    return set()


@dataclass(frozen=True)
class _ProposalCallPolicy:
    """Deny-by-default call policy with an explicit pure-computation surface."""

    allowed: frozenset[str]
    trusted_calls: frozenset[str] = frozenset()
    unknown_calls: str = "deny"

    def violations(self, code: str) -> tuple[tuple[str, str], ...]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return (("<syntax>", "invalid_syntax"),)
        aliases = self._aliases(tree)
        local_functions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        safe_names = self._safe_data_names(tree, aliases, local_functions)
        violations: set[tuple[str, str]] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.id in _RESERVED_CONTEXT_NAMES
            ):
                violations.add((node.id, "reserved_runtime_context"))
            if not isinstance(node, ast.Call):
                continue
            allowed, label, name = self._classify_call(
                node,
                aliases=aliases,
                local_functions=local_functions,
                safe_names=safe_names,
            )
            if not allowed:
                violations.add((name, label))
        return tuple(sorted(violations))

    @staticmethod
    def _aliases(tree: ast.AST) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for item in node.names:
                    aliases[item.asname or item.name.split(".", 1)[0]] = item.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for item in node.names:
                    aliases[item.asname or item.name] = f"{node.module}.{item.name}"
        return aliases

    def _safe_data_names(
        self,
        tree: ast.AST,
        aliases: Mapping[str, str],
        local_functions: set[str],
    ) -> set[str]:
        safe = {
            "EVIDENCE",
            "INPUTS",
            "NODE_RESULT",
            "PROMPTS",
            "RUNTIME_CONTEXT_V1",
            "NODE_CONFIG_V1",
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                safe.update(arg.arg for arg in node.args.args)
                safe.update(arg.arg for arg in node.args.kwonlyargs)
            elif isinstance(node, ast.comprehension):
                safe.update(_assigned_names(node.target))
        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                value: ast.expr | None = None
                if isinstance(node, ast.Assign):
                    value = node.value
                    targets = node.targets
                elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                    value = node.value
                    targets = [node.target]
                elif isinstance(node, (ast.For, ast.AsyncFor)):
                    value = node.iter
                    targets = [node.target]
                else:
                    continue
                if value is None or not self._safe_expression(
                    value, aliases, local_functions, safe
                ):
                    continue
                names = {name for target in targets for name in _assigned_names(target)}
                if not names.issubset(safe):
                    safe.update(names)
                    changed = True
        return safe

    def _safe_expression(
        self,
        node: ast.expr,
        aliases: Mapping[str, str],
        local_functions: set[str],
        safe_names: set[str],
    ) -> bool:
        if isinstance(
            node,
            (
                ast.Constant,
                ast.Dict,
                ast.List,
                ast.Tuple,
                ast.Set,
                ast.ListComp,
                ast.DictComp,
                ast.SetComp,
                ast.GeneratorExp,
            ),
        ):
            return True
        if isinstance(node, ast.Name):
            return node.id in safe_names
        if isinstance(node, ast.Subscript):
            return self._safe_expression(node.value, aliases, local_functions, safe_names)
        if isinstance(node, ast.Attribute):
            return self._safe_expression(node.value, aliases, local_functions, safe_names)
        if isinstance(node, ast.Call):
            allowed, _, _ = self._classify_call(
                node,
                aliases=aliases,
                local_functions=local_functions,
                safe_names=safe_names,
            )
            return allowed
        if isinstance(node, (ast.BinOp, ast.BoolOp, ast.Compare)):
            return True
        if isinstance(node, ast.UnaryOp):
            return self._safe_expression(node.operand, aliases, local_functions, safe_names)
        return False

    def _classify_call(
        self,
        node: ast.Call,
        *,
        aliases: Mapping[str, str],
        local_functions: set[str],
        safe_names: set[str],
    ) -> tuple[bool, str, str]:
        name = _qualified_ast_name(node.func, aliases) or "<dynamic>"
        effect = _call_effect(name)
        if effect is not None:
            return effect in self.allowed, effect, name
        if isinstance(node.func, ast.Name):
            raw_name = node.func.id
            if raw_name in self.trusted_calls:
                return True, "trusted_skill_function", name
            if raw_name in local_functions:
                return True, "pure_local", name
            if raw_name in SAFE_BUILTINS or raw_name in _PURE_EXTRA_BUILTINS:
                return True, "pure_builtin", name
        if name in _PURE_MATH_CALLS:
            return True, "pure_numeric", name
        if name in _PURE_NUMPY_CALLS:
            return True, "pure_numeric", name
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _PURE_DATA_METHODS
            and self._safe_expression(node.func.value, aliases, local_functions, safe_names)
        ):
            return True, "pure_data_method", name
        return False, "unknown", name


class _ProposalCapabilityExecutor:
    """Apply the provider-local pure/effect call policy before the sandbox."""

    def __init__(self, inner: Any, policy: _ProposalCallPolicy, *, node_id: str) -> None:
        self.inner = inner
        self.policy = policy
        self.node_id = node_id

    @property
    def env(self) -> Any:
        return getattr(self.inner, "env", None)

    @property
    def sandbox_namespace(self) -> Any:
        return getattr(self.inner, "sandbox_namespace", None)

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        if block.metadata.get("runtime_setup"):
            return self.inner.run_block(block)
        violations = self.policy.violations(block.code)
        if not violations:
            return self.inner.run_block(block)
        parts = []
        for name, label in violations:
            if label == "unknown":
                parts.append(f"{name} is not in the pure-call or capability registry")
            elif label == "invalid_syntax":
                parts.append("generated Python is invalid")
            else:
                parts.append(f"{name} requires ungranted capability {label}")
        reason = "Proposal capability policy rejected this block: " + "; ".join(parts)
        return BlockExecutionResult(
            block=block,
            ok=False,
            status=ActionBlockStatus.SKIPPED,
            stderr=reason,
            info={
                "authoring_node": self.node_id,
                "capability_violations": [
                    {"call": name, "effect": label} for name, label in violations
                ],
            },
        )


class _StrictCallShapeExecutor:
    """Reject call indirection that would make the capability scan ambiguous.

    ``CapabilityPolicy`` classifies direct names such as ``goto_pose`` and
    ``robot.goto_pose``.  A construct such as ``vars()[name](...)`` has no
    statically knowable callee and must therefore be denied at this proposal
    boundary.  Runtime-authored setup blocks are trusted and retain the legacy
    skill-loading behavior.
    """

    def __init__(
        self,
        inner: Any,
        *,
        node_id: str,
        trusted_import_roots: frozenset[str] = frozenset(),
        trusted_skill_binding_names: frozenset[str] = frozenset(),
    ) -> None:
        self.inner = inner
        self.node_id = node_id
        self.trusted_import_roots = trusted_import_roots
        self.trusted_skill_binding_names = trusted_skill_binding_names

    @property
    def env(self) -> Any:
        return getattr(self.inner, "env", None)

    @property
    def sandbox_namespace(self) -> Any:
        return getattr(self.inner, "sandbox_namespace", None)

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        if block.metadata.get("runtime_setup"):
            # Skill contract binding executes a sidecar module at import time.
            # That is not safe merely because the surrounding block was
            # runtime-authored.  Proposal workers retain prose retrieval and
            # sys.path setup, but canonical sidecar functions must be exposed
            # as audited env APIs (or run in a separate pure sandbox).
            if block.name.startswith("skill_bindings_"):
                if block.name in self.trusted_skill_binding_names:
                    return self.inner.run_block(block)
                return self._denied(
                    block,
                    "proposal-only provider does not execute skill sidecar modules "
                    "during runtime setup",
                )
            allowed_setup = (
                block.name.endswith("_seed")
                or block.name.startswith("skill_setup_")
                or "_materialize_" in block.name
            )
            if not allowed_setup:
                return self._denied(block, "unknown runtime setup block")
            return self.inner.run_block(block)
        try:
            tree = ast.parse(block.code)
        except SyntaxError as exc:
            return self._denied(block, f"invalid generated Python: {exc.msg}")
        forbidden_imports = sorted(
            {
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in (
                    node.names
                    if isinstance(node, ast.Import)
                    else [ast.alias(name=str(node.module or ""))]
                )
                if alias.name.split(".", 1)[0] in _PROCESS_MODULES
            }
        )
        if forbidden_imports:
            return self._denied(
                block,
                "proposal-only policy denied process/FFI module import(s): "
                + ", ".join(forbidden_imports),
            )
        imported_roots = sorted(
            {
                alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in (
                    node.names
                    if isinstance(node, ast.Import)
                    else [ast.alias(name=str(node.module or ""))]
                )
                if alias.name
            }
        )
        untrusted_imports = sorted(
            set(imported_roots) - _SAFE_IMPORT_ROOTS - self.trusted_import_roots
        )
        if untrusted_imports:
            return self._denied(
                block,
                "proposal-only policy denied untrusted import root(s): "
                + ", ".join(untrusted_imports),
            )
        dynamic = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and not isinstance(node.func, (ast.Name, ast.Attribute))
        ]
        if dynamic:
            lines = sorted({int(getattr(node, "lineno", 0)) for node in dynamic})
            return self._denied(
                block,
                "proposal-only policy denied dynamically-shaped call(s) at line(s) "
                + ", ".join(str(line) for line in lines),
            )
        denied_references: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                name = node.attr
            else:
                continue
            if (
                name in _PHYSICAL_API_NAMES
                or name in DYNAMIC_OR_PROCESS_CALLS
                or any(value.endswith(f".{name}") for value in DYNAMIC_OR_PROCESS_CALLS)
            ):
                denied_references.add(name)
        if denied_references:
            return self._denied(
                block,
                "proposal-only policy denied physical/process callable reference(s): "
                + ", ".join(sorted(denied_references)),
            )
        introspection_calls = sorted(
            {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _DENIED_INTROSPECTION_CALLS
            }
        )
        builtins_access = any(
            isinstance(node, ast.Name) and node.id == "__builtins__" for node in ast.walk(tree)
        )
        if introspection_calls or builtins_access:
            detail = introspection_calls + (["__builtins__"] if builtins_access else [])
            return self._denied(
                block,
                "proposal-only policy denied capability-indirection primitive(s): "
                + ", ".join(detail),
            )
        return self.inner.run_block(block)

    def _denied(self, block: SemanticActionBlock, reason: str) -> BlockExecutionResult:
        return BlockExecutionResult(
            block=block,
            ok=False,
            status=ActionBlockStatus.SKIPPED,
            stderr=reason,
            info={
                "authoring_node": self.node_id,
                "proposal_only_violation": True,
            },
        )


class SkillCodingAgentProvider:
    """Run skill-retrieval Coding Agents as read-only v2 proposal actors.

    ``executor_factory`` supplies the real persistent Python sandbox (for
    example a CapX executor adapter).  Generated blocks are always wrapped in a
    deny-by-default capability guard.  ``policy`` is convenient for stateless
    online policies; ``policy_factory`` should be used when each actor requires
    independent provider-owned policy state.  ``trusted_import_roots`` is an
    explicit operator assertion that those Python modules are pure; arbitrary
    skill sidecars remain non-executable and skill contract bindings are
    rejected.  Semantic ``capability_grants`` may be extended only with
    proposal-safe executor grants.
    """

    def __init__(
        self,
        *,
        data_plane: EpisodeDataPlane | None = None,
        executor_factory: ExecutorFactory,
        library: SkillLibrary,
        policy: CompletionPolicy | None = None,
        policy_factory: PolicyFactory | None = None,
        artifacts_root: str | Path | None = None,
        max_turns: int = 6,
        max_model_calls: int = 32,
        max_tokens: int = 262_144,
        max_wall_time_ms: int = 60_000,
        environment: str = "",
        api_docs: str = "",
        trusted_import_roots: frozenset[str] = frozenset(),
        trusted_skill_sidecars: frozenset[str] = frozenset(),
        capability_grants: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        if data_plane is not None and not isinstance(data_plane, EpisodeDataPlane):
            raise TypeError("data_plane must be an EpisodeDataPlane or None")
        if not callable(executor_factory):
            raise TypeError("executor_factory must be callable")
        if (policy is None) == (policy_factory is None):
            raise ValueError("Supply exactly one of policy or policy_factory.")
        if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if (
            not isinstance(max_model_calls, int)
            or isinstance(max_model_calls, bool)
            or max_model_calls < 1
        ):
            raise ValueError("max_model_calls must be a positive integer")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if (
            isinstance(max_wall_time_ms, bool)
            or not isinstance(max_wall_time_ms, int)
            or max_wall_time_ms < 1
        ):
            raise ValueError("max_wall_time_ms must be a positive integer")
        self.executor_factory = executor_factory
        self.library = library
        self._shared_policy = policy
        self._policy_factory = policy_factory
        self.max_turns = max_turns
        self.max_model_calls = max_model_calls
        self.max_tokens = max_tokens
        self.max_wall_time_ms = max_wall_time_ms
        self.environment = str(environment)
        self.api_docs = str(api_docs)
        normalized_imports = frozenset(str(value).strip() for value in trusted_import_roots)
        if "" in normalized_imports or any(
            not value.isidentifier() for value in normalized_imports
        ):
            raise ValueError("trusted_import_roots must contain plain module root names")
        self.trusted_import_roots = normalized_imports
        normalized_skill_ids = frozenset(str(value).strip() for value in trusted_skill_sidecars)
        if "" in normalized_skill_ids:
            raise ValueError("trusted_skill_sidecars must contain non-empty skill IDs")
        (
            self.trusted_canonical_calls,
            self.trusted_skill_binding_names,
        ) = _audit_trusted_skill_sidecars(library, normalized_skill_ids)
        self.trusted_skill_sidecars = normalized_skill_ids
        grants = dict(DEFAULT_CODING_CAPABILITY_GRANTS)
        grants.update(capability_grants or {})
        normalized_grants: dict[str, frozenset[str]] = {}
        for semantic_id, executor_grants in grants.items():
            semantic = str(semantic_id).strip()
            if not semantic:
                raise ValueError("semantic capability IDs must be non-empty")
            translated = frozenset(str(value).strip() for value in executor_grants)
            if "" in translated:
                raise ValueError("executor capability grants must be non-empty")
            unsafe = translated - PROPOSAL_CAPABILITIES
            if unsafe:
                raise ValueError(
                    f"Semantic capability {semantic!r} maps to unsafe executor grants: "
                    + ", ".join(sorted(unsafe))
                )
            normalized_grants[semantic] = translated
        self.capability_grants: Mapping[str, frozenset[str]] = MappingProxyType(normalized_grants)
        self._configured_artifacts_root = (
            Path(artifacts_root).resolve() if artifacts_root is not None else None
        )
        self.artifacts_root: Path | None = None
        self.ledger_root: Path | None = None
        self.worker_root: Path | None = None
        self._data_plane: EpisodeDataPlane | None = None
        self._runtimes: dict[str, CodingWorkerRuntime] = {}
        self._lock = threading.RLock()
        self._shared_policy_lock = threading.RLock()
        if data_plane is not None:
            self.bind_episode_data_plane(data_plane)

    @property
    def data_plane(self) -> EpisodeDataPlane:
        plane = self._data_plane
        if plane is None:
            raise CodingProviderContractError(
                "Coding provider is not bound to the EpisodeRuntime data plane."
            )
        return plane

    @property
    def invocation_audits(self) -> tuple[CodingInvocationAudit, ...]:
        """Return a stable episode-wide view of skill/coding invocations."""

        with self._lock:
            return tuple(
                audit
                for actor_id in sorted(self._runtimes)
                for audit in self._runtimes[actor_id].audits
            )

    def bind_episode_data_plane(self, data_plane: EpisodeDataPlane) -> None:
        """Bind the exact plane created by ``V2RuntimeFactory`` once.

        Providers can therefore be admitted as dependencies before the episode
        exists without constructing a stale second ``EpisodeDataPlane`` view.
        """

        if not isinstance(data_plane, EpisodeDataPlane):
            raise TypeError("data_plane must be an EpisodeDataPlane")
        with self._lock:
            if self._data_plane is data_plane:
                return
            if self._data_plane is not None:
                raise CodingProviderContractError(
                    "Coding provider is already bound to a different data-plane instance."
                )
            if self._runtimes:
                raise CodingProviderContractError(
                    "Coding provider cannot bind a data plane after actors were spawned."
                )
            self._data_plane = data_plane
            root = self._configured_artifacts_root or (
                data_plane.episode_root / "providers" / "coding_agent"
            )
            self.artifacts_root = root.resolve()
            self.ledger_root = self.artifacts_root / "idempotency"
            self.worker_root = self.artifacts_root / "workers"
            self.ledger_root.mkdir(parents=True, exist_ok=True)
            self.worker_root.mkdir(parents=True, exist_ok=True)

    def spawn(self, profile: ActorProfile, isolation: ActorIsolation) -> CodingWorkerRuntime:
        _ = self.data_plane  # fail before allocating an unbound provider runtime
        self._validate_profile(profile)
        actor_id = isolation.owner_actor_id
        with self._lock:
            previous = self._runtimes.get(actor_id)
            if previous is not None:
                if previous.profile != profile or previous.isolation != isolation:
                    raise ActorConflictError(
                        f"Coding provider actor {actor_id!r} was rebound to a different profile."
                    )
                return previous
            executor = self.executor_factory(profile, isolation)
            if not callable(getattr(executor, "run_block", None)):
                raise TypeError("executor_factory must return a block executor")
            policy = (
                self._shared_policy
                if self._policy_factory is None
                else self._policy_factory(profile, isolation)
            )
            if policy is None or not (
                callable(getattr(policy, "complete", None))
                or callable(getattr(policy, "complete_turn", None))
                or callable(getattr(policy, "complete_bounded", None))
                or callable(getattr(policy, "complete_turn_bounded", None))
            ):
                raise TypeError("policy must implement a completion boundary")
            runtime = CodingWorkerRuntime(
                profile=profile,
                isolation=isolation,
                executor=executor,
                policy=policy,
            )
            self._runtimes[actor_id] = runtime
            return runtime

    def invoke(
        self, runtime: CodingWorkerRuntime, spec: InvocationSpec
    ) -> ActivationExecutionResult:
        with runtime.lock:
            self._require_active(runtime)
            granted = self._validate_invocation(runtime.profile, spec)
            context_binding = self._context_binding(runtime.profile, spec)
            resolved_inputs, lineage = self._resolve_inputs(spec.inputs)
            record_path = self._record_path(runtime.isolation.owner_actor_id, spec)
            with self._invocation_lock(record_path):
                previous = self._read_record(record_path)
                if previous is not None:
                    self._validate_record_identity(previous, runtime, spec)
                    if previous.get("status") == "committed":
                        result = self._result_from_record(previous)
                        self._validate_result(result, spec)
                        runtime.audits.append(
                            CodingInvocationAudit(
                                invocation_id=spec.invocation_id,
                                idempotency_key=spec.effective_idempotency_key,
                                outcome=result.outcome or ControlOutcome.FAILED,
                                loaded_skill_ids=tuple(
                                    str(value) for value in previous.get("loaded_skill_ids", ())
                                ),
                                turns=int(previous.get("turns", 0)),
                                replayed=True,
                                context_digest=str(previous.get("context_digest") or ""),
                            )
                        )
                        return result
                    if previous.get("status") == "pending":
                        raise CodingProviderContractError(
                            "Durable coding invocation is pending after an uncertain "
                            "model boundary and cannot be re-entered"
                        )
                    if previous.get("status") != "pending":
                        raise CodingProviderContractError(
                            f"Unknown durable invocation status {previous.get('status')!r}."
                        )
                else:
                    self._validate_bounded_policy(runtime.policy)
                    # Fail before writing the started record for malformed grants.
                    self._max_model_calls(runtime.profile, spec)
                    self._max_tokens(runtime.profile, spec)
                    self._wall_time_ms(runtime.profile, spec)
                    self._write_record(
                        record_path,
                        self._pending_record(runtime, spec),
                    )

                result, subagent_result = self._run_agent(
                    runtime,
                    spec,
                    resolved_inputs=resolved_inputs,
                    lineage=lineage,
                    granted=granted,
                    context_binding=context_binding,
                )
                self._validate_result(result, spec)
                self._write_record(
                    record_path,
                    self._committed_record(
                        runtime,
                        spec,
                        result,
                        subagent_result,
                        context_binding=context_binding,
                    ),
                )
                runtime.audits.append(
                    CodingInvocationAudit(
                        invocation_id=spec.invocation_id,
                        idempotency_key=spec.effective_idempotency_key,
                        outcome=result.outcome or ControlOutcome.FAILED,
                        loaded_skill_ids=subagent_result.loaded_skill_ids,
                        turns=subagent_result.turns,
                        replayed=False,
                        context_digest=(context_binding.content_digest if context_binding else ""),
                    )
                )
                return result

    def suspend(self, runtime: CodingWorkerRuntime) -> None:
        with runtime.lock:
            if runtime.retired:
                raise ActorStateError("Cannot suspend a retired coding worker.")
            runtime.suspended = True

    def resume(self, runtime: CodingWorkerRuntime) -> None:
        with runtime.lock:
            if runtime.retired:
                raise ActorStateError("Cannot resume a retired coding worker.")
            runtime.suspended = False

    def retire(self, runtime: CodingWorkerRuntime) -> None:
        with runtime.lock:
            runtime.retired = True
            runtime.suspended = False

    def _run_agent(
        self,
        runtime: CodingWorkerRuntime,
        spec: InvocationSpec,
        *,
        resolved_inputs: dict[str, Any],
        lineage: tuple[ResolvedArtifactRef, ...],
        granted: frozenset[str],
        context_binding: _CodingContextBinding | None,
    ) -> tuple[ActivationExecutionResult, SubAgentResult]:
        if context_binding is not None:
            self._seed_runtime_context(runtime, context_binding)
        actor_id = runtime.isolation.owner_actor_id
        guarded = _ProposalCapabilityExecutor(
            runtime.executor,
            _ProposalCallPolicy(
                allowed=granted,
                trusted_calls=self.trusted_canonical_calls,
            ),
            node_id=actor_id,
        )
        executor = _StrictCallShapeExecutor(
            guarded,
            node_id=actor_id,
            trusted_import_roots=self.trusted_import_roots,
            trusted_skill_binding_names=self.trusted_skill_binding_names,
        )
        output_ports = tuple(
            (name, schema) for name, schema in sorted(spec.output_contract.items())
        )
        task_kind = self._metadata_text(runtime.profile, "task_kind", "author")
        prompt = build_agent_system_prompt(
            role=task_kind,
            objective=(
                spec.objective
                + "\nAuthor proposal artifacts only. Never execute robot motion, gripper, "
                "environment steps, or active-perception motion; trusted system_action "
                "runners exclusively own those effects. Skill prose is available, but this "
                "worker does not execute unaudited skill sidecar modules."
            ),
            capabilities=granted,
            output_contract=output_contract_for_ports(output_ports),
            environment=self.environment,
            api_docs=self.api_docs,
        )
        max_turns = self._max_turns(runtime.profile, spec)
        max_model_calls = self._max_model_calls(runtime.profile, spec)
        max_tokens = self._max_tokens(runtime.profile, spec)
        wall_time_ms = self._wall_time_ms(runtime.profile, spec)
        deadline = time.monotonic() + wall_time_ms / 1000.0
        if spec.deadline_monotonic_s is not None:
            deadline = min(deadline, spec.deadline_monotonic_s)
        agent = CodingAgentSubAgent(
            executor=executor,
            policy=runtime.policy,
            library=self.library,
            max_turns=max_turns,
            max_model_calls=max_model_calls,
            max_tokens=max_tokens,
            deadline_monotonic_s=deadline,
            require_bounded_model_calls=True,
            system_prompt=prompt,
            name=runtime.profile.profile_id,
            description="v2 typed proposal coding worker",
            objective=spec.objective,
            task_kind=task_kind,
            output_ports=output_ports,
            preloaded_skills=self._preloaded_skills(runtime.profile),
        )
        request = SubAgentRequest(
            task=spec.objective,
            task_kind=task_kind,
            inputs=resolved_inputs,
            artifacts_dir=str(
                self._required_worker_root() / actor_id / spec.effective_idempotency_key
            ),
            task_id=spec.invocation_id,
            observation_epoch=0,
        )
        policy_guard = self._shared_policy_lock if self._policy_factory is None else nullcontext()
        started_at = time.monotonic()
        with policy_guard:
            subagent_result = agent.run(request)
        wall_time_s = max(time.monotonic() - started_at, 0.0)
        usage = self._invocation_usage(
            subagent_result,
            invocation_fingerprint=f"sha256:{spec.fingerprint()}",
            wall_time_s=wall_time_s,
        )
        if not subagent_result.ok:
            outcome = self._failure_outcome(subagent_result)
            reason = subagent_result.error or subagent_result.claim or "coding worker failed"
            return (
                ActivationExecutionResult(
                    outcome=outcome,
                    reason=reason,
                    usage=usage,
                ),
                subagent_result,
            )

        terminal = self._terminal_mapping(subagent_result.result)
        outcome = self._declared_outcome(terminal)
        artifacts = self._emissions(
            terminal,
            output_contract=spec.output_contract,
            lineage=lineage,
        )
        return (
            ActivationExecutionResult(
                outcome=outcome,
                artifacts=artifacts,
                reason=str(terminal.get("reason") or ""),
                usage=usage,
            ),
            subagent_result,
        )

    @staticmethod
    def _invocation_usage(
        result: SubAgentResult,
        *,
        invocation_fingerprint: str,
        wall_time_s: float,
    ) -> InvocationUsage | None:
        """Extract the CodingAgent ledger, never infer tokens from text length."""

        trace = result.trace
        metadata = getattr(trace, "metadata", None)
        raw = metadata.get("runtime_budget") if isinstance(metadata, Mapping) else None
        if not isinstance(raw, Mapping):
            # Legacy/custom SubAgent implementations remain safe: absence of a
            # trusted ledger makes EpisodeRuntime charge the complete grant.
            return None
        model_calls = raw.get("model_calls")
        tokens = raw.get("tokens_committed")
        if (
            isinstance(model_calls, bool)
            or not isinstance(model_calls, int)
            or model_calls < 0
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens < 0
        ):
            raise CodingProviderContractError(
                "CodingAgent runtime budget ledger is malformed"
            )
        return InvocationUsage(
            invocation_fingerprint=invocation_fingerprint,
            model_calls=model_calls,
            tokens=tokens,
            wall_time_s=wall_time_s,
        )

    def _resolve_inputs(
        self, values: Mapping[str, Any]
    ) -> tuple[dict[str, Any], tuple[ResolvedArtifactRef, ...]]:
        inputs: dict[str, Any] = {}
        lineage: list[ResolvedArtifactRef] = []
        seen: set[tuple[str, str]] = set()
        for name, value in sorted(values.items()):
            if not isinstance(value, Mapping):
                raise CodingProviderContractError(
                    f"Input {name!r} is not an explicit typed artifact ref."
                )
            ref = ResolvedArtifactRef.from_any(value)
            resolved = self.data_plane.resolve(ref)
            if not isinstance(resolved.payload, Mapping):
                raise CodingProviderContractError(
                    f"Input {name!r} resolved to a non-mapping payload."
                )
            record = resolved.record
            record_lineage: list[ResolvedArtifactRef] = []
            for item in record.get("lineage", ()):
                try:
                    causal_ref = ResolvedArtifactRef.from_any(item)
                    self.data_plane.resolve(causal_ref)
                except (TypeError, ValueError) as exc:
                    raise CodingProviderContractError(
                        f"Input {name!r} carries malformed or unresolved lineage."
                    ) from exc
                record_lineage.append(causal_ref)
            inputs[str(name)] = {
                "ref": ref.to_mapping(),
                "schema": resolved.schema,
                "payload": copy.deepcopy(dict(resolved.payload)),
                "lineage": [value.to_mapping() for value in record_lineage],
                "producer": str(record.get("activation_id") or ""),
                "port": str(record.get("port") or ""),
                "attempt": int(record.get("attempt") or 0),
                "generation": int(record.get("generation") or 0),
            }
            for admitted_ref in (ref, *record_lineage):
                identity = (admitted_ref.artifact_id, admitted_ref.content_digest)
                if identity not in seen:
                    seen.add(identity)
                    lineage.append(admitted_ref)
        return inputs, tuple(lineage)

    def _emissions(
        self,
        terminal: Mapping[str, Any],
        *,
        output_contract: Mapping[str, str],
        lineage: tuple[ResolvedArtifactRef, ...],
    ) -> tuple[ArtifactEmission, ...]:
        raw_outputs = terminal.get("outputs")
        if not isinstance(raw_outputs, Mapping):
            raw_outputs = {}
        normalized: dict[str, Any] = {}
        for raw_name, value in raw_outputs.items():
            name = str(raw_name)
            if name not in output_contract:
                name = name.split(":", 1)[0]
            if name in normalized:
                raise CodingProviderContractError(
                    f"Coding worker emitted duplicate output port {name!r}."
                )
            normalized[name] = value
        unknown = set(normalized) - set(output_contract)
        missing = set(output_contract) - set(normalized)
        if unknown or missing:
            details: list[str] = []
            if unknown:
                details.append(f"undeclared={sorted(unknown)!r}")
            if missing:
                details.append(f"missing={sorted(missing)!r}")
            raise CodingProviderContractError(
                "Coding worker output contract mismatch: " + ", ".join(details)
            )

        emissions: list[ArtifactEmission] = []
        for name, schema_id in sorted(output_contract.items()):
            value = normalized[name]
            if not isinstance(value, Mapping):
                raise CodingProviderContractError(f"Output {name!r} must be a typed mapping.")
            claimed_schema = value.get("schema_id") or value.get("schema")
            if claimed_schema is not None and str(claimed_schema) != schema_id:
                raise CodingProviderContractError(
                    f"Output {name!r} claimed schema {claimed_schema!r}, expected {schema_id!r}."
                )
            payload = value.get("payload")
            if not isinstance(payload, Mapping):
                raise CodingProviderContractError(f"Output {name!r} requires a mapping payload.")
            normalized_payload = self.data_plane.validate_payload(schema_id, payload)
            emissions.append(
                ArtifactEmission(
                    port=name,
                    schema_id=schema_id,
                    payload=normalized_payload,
                    lineage=lineage,
                )
            )
        return tuple(emissions)

    def _validate_result(self, result: ActivationExecutionResult, spec: InvocationSpec) -> None:
        if result.outcome is None:
            raise CodingProviderContractError("Coding result has no control outcome.")
        declared = dict(spec.output_contract)
        emitted = {value.port: value for value in result.artifacts}
        if len(emitted) != len(result.artifacts):
            raise CodingProviderContractError("Coding result repeats an output port.")
        unknown = set(emitted) - set(declared)
        if unknown:
            raise CodingProviderContractError(
                f"Coding result contains undeclared outputs: {sorted(unknown)!r}."
            )
        if result.outcome is ControlOutcome.SUCCESS:
            missing = set(declared) - set(emitted)
            if missing:
                raise CodingProviderContractError(
                    f"Successful coding result omitted outputs: {sorted(missing)!r}."
                )
        for name, emission in emitted.items():
            if emission.schema_id != declared[name]:
                raise CodingProviderContractError(
                    f"Output {name!r} schema drifted during durable replay."
                )
            self.data_plane.validate_payload(emission.schema_id, emission.payload)
            for ref in emission.lineage:
                self.data_plane.resolve(ref)

    def _context_binding(
        self,
        profile: ActorProfile,
        spec: InvocationSpec,
    ) -> _CodingContextBinding | None:
        shadowed_inputs = set(spec.inputs).intersection(_RESERVED_CONTEXT_NAMES)
        if shadowed_inputs:
            raise CodingProviderContractError(
                "artifact inputs cannot shadow provider runtime context: "
                + ", ".join(sorted(shadowed_inputs))
            )
        shadowed_metadata = set(spec.metadata).intersection(
            {
                *_RESERVED_CONTEXT_NAMES,
                "runtime_context",
                "node_config",
            }
        )
        if shadowed_metadata:
            raise CodingProviderContractError(
                "invocation metadata cannot inject reserved runtime context: "
                + ", ".join(sorted(shadowed_metadata))
            )
        strict = profile.metadata.get("strict_runtime_context", False)
        if not isinstance(strict, bool):
            raise CodingProviderContractError("profile strict_runtime_context must be a boolean")
        if not strict:
            if profile.metadata.get("node_config_v1") not in (None, (), []):
                raise CodingProviderContractError("node_config_v1 requires strict_runtime_context")
            return None

        expected_configs = self._profile_node_configs(profile)
        node_id = str(spec.metadata.get("activation_id") or "").strip()
        raw_node_config = spec.metadata.get("node_params")
        if not isinstance(raw_node_config, Mapping):
            raise CodingProviderContractError(
                "strict coding invocation requires graph-owned node_params"
            )
        try:
            node_config = CodingNodeConfigV1.model_validate(
                {"schema_version": _NODE_CONFIG_SCHEMA, **dict(raw_node_config)}
            )
        except (TypeError, ValueError) as exc:
            raise CodingProviderContractError(
                "graph node_params violate CodingNodeConfigV1"
            ) from exc
        expected = expected_configs.get(node_id)
        if expected is None:
            raise CodingProviderContractError(
                f"activation {node_id!r} is not bound by the coding profile"
            )
        if node_config.canonical_mapping() != expected.canonical_mapping():
            raise CodingProviderContractError(
                "graph-owned node config differs from the admitted actor profile"
            )
        try:
            runtime_context = CodingRuntimeContextV1(
                episode_id=str(spec.metadata["episode_id"]),
                run_id=str(spec.metadata["workflow_id"]),
                node_id=node_id,
                attempt=spec.metadata["attempt"],
                invocation_id=spec.invocation_id,
                idempotency_key=spec.effective_idempotency_key,
                graph_id=str(spec.metadata["graph_id"]),
                graph_revision=spec.metadata["graph_revision"],
                graph_digest=str(spec.metadata["graph_digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CodingProviderContractError(
                "strict coding invocation is missing valid runtime identity metadata"
            ) from exc
        canonical = json.dumps(
            {
                "runtime_context": runtime_context.model_dump(mode="json"),
                "node_config": node_config.canonical_mapping(),
            },
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return _CodingContextBinding(
            runtime_context=runtime_context,
            node_config=node_config,
            content_digest="sha256:" + hashlib.sha256(canonical).hexdigest(),
        )

    @staticmethod
    def _profile_node_configs(profile: ActorProfile) -> dict[str, CodingNodeConfigV1]:
        rows = profile.metadata.get("node_config_v1")
        if not isinstance(rows, (tuple, list)) or not rows:
            raise CodingProviderContractError("strict coding profile requires node_config_v1 rows")
        configs: dict[str, CodingNodeConfigV1] = {}
        for row in rows:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise CodingProviderContractError(
                    "node_config_v1 rows must be (activation_id, canonical_json) pairs"
                )
            activation_id, encoded = row
            node_id = str(activation_id).strip()
            if not node_id or node_id in configs or not isinstance(encoded, str):
                raise CodingProviderContractError(
                    "node_config_v1 activation IDs must be unique non-empty strings"
                )
            try:
                raw = json.loads(encoded)
                if not isinstance(raw, dict):
                    raise TypeError("node config must decode to a mapping")
                configs[node_id] = CodingNodeConfigV1.model_validate(
                    {"schema_version": _NODE_CONFIG_SCHEMA, **raw}
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise CodingProviderContractError(
                    f"profile node config for {node_id!r} is invalid"
                ) from exc
        return configs

    @staticmethod
    def _seed_runtime_context(
        runtime: CodingWorkerRuntime,
        binding: _CodingContextBinding,
    ) -> None:
        runtime_mapping = binding.runtime_context.model_dump(mode="json")
        node_mapping = binding.node_config.canonical_mapping()
        block = SemanticActionBlock(
            name="coding_runtime_context_seed",
            intent="seed provider-owned immutable coding context",
            code=(
                "from types import MappingProxyType as _robomex_mapping_proxy\n"
                f"RUNTIME_CONTEXT_V1 = _robomex_mapping_proxy({runtime_mapping!r})\n"
                f"NODE_CONFIG_V1 = _robomex_mapping_proxy({node_mapping!r})\n"
            ),
            metadata={"runtime_setup": True, "provider_owned_context": True},
        )
        execution = runtime.executor.run_block(block)
        if not execution.ok:
            raise CodingProviderContractError(
                "coding sandbox rejected provider-owned runtime context setup: " + execution.stderr
            )
        namespace = getattr(runtime.executor, "sandbox_namespace", None)
        if isinstance(namespace, dict):
            namespace["RUNTIME_CONTEXT_V1"] = MappingProxyType(runtime_mapping)
            namespace["NODE_CONFIG_V1"] = MappingProxyType(node_mapping)

    @staticmethod
    def _terminal_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
        nested = value.get("result")
        return nested if isinstance(nested, Mapping) else value

    @staticmethod
    def _declared_outcome(terminal: Mapping[str, Any]) -> ControlOutcome:
        raw = terminal.get("control_outcome") or terminal.get("outcome")
        if raw is None:
            return ControlOutcome.SUCCESS
        try:
            return ControlOutcome(str(raw))
        except ValueError as exc:
            raise CodingProviderContractError(
                f"Coding worker declared unknown control outcome {raw!r}."
            ) from exc

    @staticmethod
    def _failure_outcome(result: SubAgentResult) -> ControlOutcome:
        verdict = result.local_verdict
        status = str(verdict.status if verdict is not None else "").lower()
        return (
            ControlOutcome.UNCERTAIN
            if status in {"uncertain", "unknown"}
            else ControlOutcome.FAILED
        )

    def _validate_profile(self, profile: ActorProfile) -> None:
        if profile.runner_kind != "coding_worker":
            raise ActorAuthorityError(
                "SkillCodingAgentProvider only accepts coding_worker profiles."
            )
        if profile.effect_ceiling:
            raise ActorAuthorityError("A proposal coding worker must have an empty effect ceiling.")
        for semantic_id in profile.capability_ceiling:
            self._translate_capability(semantic_id)
        strict = profile.metadata.get("strict_runtime_context", False)
        if not isinstance(strict, bool):
            raise CodingProviderContractError("profile strict_runtime_context must be a boolean")
        if strict:
            self._profile_node_configs(profile)

    def _validate_invocation(self, profile: ActorProfile, spec: InvocationSpec) -> frozenset[str]:
        if spec.requested_effects:
            raise ActorAuthorityError(
                "Coding workers author proposals and cannot request world effects."
            )
        semantic_excess = spec.requested_capabilities - profile.capability_ceiling
        if semantic_excess:
            raise ActorAuthorityError(
                "Coding invocation exceeds its semantic capability ceiling: "
                + ", ".join(sorted(semantic_excess))
            )
        translated: set[str] = set()
        for semantic_id in spec.requested_capabilities:
            translated.update(self._translate_capability(semantic_id))
        return frozenset(translated)

    def _translate_capability(self, semantic_id: str) -> frozenset[str]:
        try:
            return self.capability_grants[semantic_id]
        except KeyError as exc:
            raise ActorAuthorityError(
                f"Unknown or unsafe coding capability {semantic_id!r}."
            ) from exc

    def _max_turns(self, profile: ActorProfile, spec: InvocationSpec) -> int:
        configured = profile.metadata.get("max_turns", self.max_turns)
        try:
            value = int(configured)
        except (TypeError, ValueError) as exc:
            raise CodingProviderContractError("profile max_turns must be an integer") from exc
        if value < 1:
            raise CodingProviderContractError("profile max_turns must be positive")
        value = min(value, self.max_turns)
        requested = spec.budget.get("action_turns")
        if requested is not None and requested > 0:
            value = min(value, max(1, int(requested)))
        return value

    def _max_model_calls(self, profile: ActorProfile, spec: InvocationSpec) -> int:
        configured = profile.metadata.get("max_model_calls", self.max_model_calls)
        try:
            value = int(configured)
        except (TypeError, ValueError) as exc:
            raise CodingProviderContractError("profile max_model_calls must be an integer") from exc
        if value < 1:
            raise CodingProviderContractError("profile max_model_calls must be positive")
        value = min(value, self.max_model_calls)
        requested = spec.budget.get("model_calls")
        if requested is not None:
            if requested < 1 or int(requested) != requested:
                raise CodingProviderContractError(
                    "coding invocation requires a positive integer model_calls grant"
                )
            value = min(value, int(requested))
        return value

    def _max_tokens(self, profile: ActorProfile, spec: InvocationSpec) -> int:
        configured = profile.metadata.get("max_tokens", self.max_tokens)
        try:
            value = int(configured)
        except (TypeError, ValueError) as exc:
            raise CodingProviderContractError("profile max_tokens must be an integer") from exc
        if value < 1:
            raise CodingProviderContractError("profile max_tokens must be positive")
        value = min(value, self.max_tokens)
        requested = spec.budget.get("tokens")
        if requested is not None:
            if requested < 1 or int(requested) != requested:
                raise CodingProviderContractError(
                    "coding invocation requires a positive integer tokens grant"
                )
            value = min(value, int(requested))
        return value

    def _wall_time_ms(self, profile: ActorProfile, spec: InvocationSpec) -> int:
        configured = profile.metadata.get("max_wall_time_ms", self.max_wall_time_ms)
        try:
            value = int(configured)
        except (TypeError, ValueError) as exc:
            raise CodingProviderContractError(
                "profile max_wall_time_ms must be an integer"
            ) from exc
        if value < 1:
            raise CodingProviderContractError("profile max_wall_time_ms must be positive")
        value = min(value, self.max_wall_time_ms)
        requested = spec.budget.get("wall_time_ms")
        if requested is not None:
            if requested < 1 or int(requested) != requested:
                raise CodingProviderContractError(
                    "coding invocation requires a positive integer wall_time_ms grant"
                )
            value = min(value, int(requested))
        return value

    @staticmethod
    def _validate_bounded_policy(policy: Any) -> None:
        boundary = getattr(policy, "complete_turn_bounded", None)
        if not callable(boundary):
            boundary = getattr(policy, "complete_bounded", None)
        if not callable(boundary):
            raise CodingProviderContractError(
                "production coding policy must implement a bounded completion boundary"
            )
        try:
            parameters = inspect.signature(boundary).parameters.values()
        except (TypeError, ValueError) as exc:
            raise CodingProviderContractError(
                "bounded coding policy signature is not inspectable"
            ) from exc
        if not any(
            parameter.name == "deadline_monotonic_s"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            raise CodingProviderContractError(
                "bounded coding policy must enforce deadline_monotonic_s"
            )

    @staticmethod
    def _metadata_text(profile: ActorProfile, key: str, default: str) -> str:
        value = str(profile.metadata.get(key) or default).strip()
        if not value:
            raise CodingProviderContractError(f"profile metadata {key!r} must be non-empty")
        return value

    @staticmethod
    def _preloaded_skills(profile: ActorProfile) -> tuple[str, ...]:
        value = profile.metadata.get("preloaded_skills", ())
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise CodingProviderContractError("profile preloaded_skills must be a list or tuple")
        skills = tuple(str(item).strip() for item in value)
        if any(not item for item in skills):
            raise CodingProviderContractError("preloaded skill IDs must be non-empty")
        return tuple(dict.fromkeys(skills))

    @staticmethod
    def _require_active(runtime: CodingWorkerRuntime) -> None:
        if runtime.retired:
            raise ActorStateError("Coding worker is retired.")
        if runtime.suspended:
            raise ActorStateError("Coding worker is suspended.")

    def _record_path(self, actor_id: str, spec: InvocationSpec) -> Path:
        return self._required_ledger_root() / actor_id / f"{spec.effective_idempotency_key}.json"

    def _required_ledger_root(self) -> Path:
        if self.ledger_root is None:
            raise CodingProviderContractError("Coding provider storage is not bound.")
        return self.ledger_root

    def _required_worker_root(self) -> Path:
        if self.worker_root is None:
            raise CodingProviderContractError("Coding provider storage is not bound.")
        return self.worker_root

    def _pending_record(self, runtime: CodingWorkerRuntime, spec: InvocationSpec) -> dict[str, Any]:
        binding = self._context_binding(runtime.profile, spec)
        value = {
            "schema": _LEDGER_SCHEMA,
            "status": "pending",
            "episode_id": self.data_plane.episode_id,
            "actor_id": runtime.isolation.owner_actor_id,
            "profile_id": runtime.profile.profile_id,
            "invocation_id": spec.invocation_id,
            "idempotency_key": spec.effective_idempotency_key,
            "invocation_fingerprint": spec.fingerprint(),
        }
        if binding is not None:
            value.update(
                {
                    "runtime_context": binding.runtime_context.model_dump(mode="json"),
                    "node_config": binding.node_config.canonical_mapping(),
                    "context_digest": binding.content_digest,
                }
            )
        return value

    def _committed_record(
        self,
        runtime: CodingWorkerRuntime,
        spec: InvocationSpec,
        result: ActivationExecutionResult,
        subagent_result: SubAgentResult,
        *,
        context_binding: _CodingContextBinding | None,
    ) -> dict[str, Any]:
        value = self._pending_record(runtime, spec)
        if context_binding is not None and value.get("context_digest") != (
            context_binding.content_digest
        ):
            raise CodingProviderContractError("coding context changed during one invocation")
        value.update(
            {
                "status": "committed",
                "result": self._result_to_mapping(result),
                "loaded_skill_ids": list(subagent_result.loaded_skill_ids),
                "turns": subagent_result.turns,
            }
        )
        return value

    def _validate_record_identity(
        self,
        record: Mapping[str, Any],
        runtime: CodingWorkerRuntime,
        spec: InvocationSpec,
    ) -> None:
        expected = self._pending_record(runtime, spec)
        for key, value in expected.items():
            if key == "status":
                continue
            if record.get(key) != value:
                raise ActorConflictError(
                    f"Durable coding invocation {spec.effective_idempotency_key!r} "
                    f"was rebound at field {key!r}."
                )

    @staticmethod
    def _result_to_mapping(result: ActivationExecutionResult) -> dict[str, Any]:
        if result.outcome is None:
            raise CodingProviderContractError("Cannot persist a result without outcome.")
        return {
            "outcome": result.outcome.value,
            "reason": result.reason,
            "usage": result.usage.to_mapping() if result.usage is not None else None,
            "artifacts": [
                {
                    "port": value.port,
                    "schema_id": value.schema_id,
                    "payload": dict(value.payload),
                    "lineage": [ref.to_mapping() for ref in value.lineage],
                }
                for value in result.artifacts
            ],
        }

    @staticmethod
    def _result_from_record(record: Mapping[str, Any]) -> ActivationExecutionResult:
        raw = record.get("result")
        if not isinstance(raw, Mapping):
            raise CodingProviderContractError("Committed coding record has no typed result.")
        artifacts_raw = raw.get("artifacts", ())
        if not isinstance(artifacts_raw, list):
            raise CodingProviderContractError("Committed coding artifacts must be a list.")
        artifacts: list[ArtifactEmission] = []
        for value in artifacts_raw:
            if not isinstance(value, Mapping) or not isinstance(value.get("payload"), Mapping):
                raise CodingProviderContractError("Malformed committed coding artifact.")
            artifacts.append(
                ArtifactEmission(
                    port=str(value.get("port") or ""),
                    schema_id=str(value.get("schema_id") or ""),
                    payload=dict(value["payload"]),
                    lineage=tuple(
                        ResolvedArtifactRef.from_any(item) for item in value.get("lineage", ())
                    ),
                )
            )
        try:
            outcome = ControlOutcome(str(raw["outcome"]))
        except (KeyError, ValueError) as exc:
            raise CodingProviderContractError("Committed coding outcome is invalid.") from exc
        raw_usage = raw.get("usage")
        if raw_usage is not None and not isinstance(raw_usage, Mapping):
            raise CodingProviderContractError("Committed coding usage is malformed.")
        try:
            usage = (
                InvocationUsage.from_mapping(raw_usage)
                if isinstance(raw_usage, Mapping)
                else None
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CodingProviderContractError("Committed coding usage is invalid.") from exc
        return ActivationExecutionResult(
            outcome=outcome,
            artifacts=tuple(artifacts),
            reason=str(raw.get("reason") or ""),
            usage=usage,
        )

    @staticmethod
    def _read_record(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CodingProviderContractError(
                f"Durable coding record is unreadable: {path!s}."
            ) from exc
        if not isinstance(value, dict) or value.get("schema") != _LEDGER_SCHEMA:
            raise CodingProviderContractError("Durable coding record schema is invalid.")
        return value

    @staticmethod
    def _write_record(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temp_path = Path(raw_temp)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @contextmanager
    def _invocation_lock(self, record_path: Path):
        lock_path = record_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


__all__ = [
    "CodingInvocationAudit",
    "CodingNodeConfigV1",
    "CodingProviderContractError",
    "CodingProviderError",
    "CodingRuntimeContextV1",
    "CodingWorkerRuntime",
    "DEFAULT_CODING_CAPABILITY_GRANTS",
    "PROPOSAL_CAPABILITIES",
    "SkillCodingAgentProvider",
]
