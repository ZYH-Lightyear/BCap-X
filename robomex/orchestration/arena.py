"""Risk-adaptive, effect-isolated Swarm Arena for RoboMEx v2.

The Arena explores hypotheses, never the authoritative robot world.  Candidate
workers are spawned through :mod:`robomex.orchestration.actors`, normalized to
one ``ActionHypothesis`` contract, checked by deterministic hard gates, and
retired independently.  Only the selected hypothesis may later be submitted to
the separate action-admission path.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from robomex.data import ResolvedArtifact, ResolvedArtifactRef
from robomex.elastic.graph_spec import EffectScope, ExecutionBudget
from robomex.orchestration.actors import (
    ActorLifecycle,
    ActorProfile,
    ActorRegistry,
    ActorState,
    InvocationSpec,
)
from robomex.orchestration.manager import ManagerSignal
from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    FeasibilityCertificate,
    FeasibilityStatus,
    MotionPlan,
    ShadowRolloutReceipt,
    ShadowRolloutStatus,
    WorldKind,
    canonical_model_digest,
    validate_action_spec,
)
from robomex.runtime.authority import (
    ActionBackend,
    FeasibilityChecker,
    ShadowRolloutRunner,
)
from robomex.runtime.events import RosterOperation, RosterUpdate

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_CANDIDATE_EFFECTS = frozenset({"shadow_world.write", "render.write"})
_REQUIRED_FEASIBILITY_GATE_IDS = (
    "kinematic_feasibility",
    "collision",
    "joint_limits",
)
_DIRECT_KINEMATIC_CHECKS = frozenset({"kinematic_feasibility", "kinematics", "ik"})
_SEALED_PATH_KINEMATIC_CHECKS = (
    "exact_joint_path_interface",
    "joint_order",
    "robot_model",
)
_COLLISION_CHECKS = frozenset({"collision", "collision_free"})
_SCHEMA_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\.v[1-9][0-9]*$")
_INPUT_PORT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ARENA_RUNTIME_CONTEXT_METADATA_KEY = "arena_runtime_context_v1"
_ARENA_RESERVED_METADATA_KEYS = frozenset(
    {
        ARENA_RUNTIME_CONTEXT_METADATA_KEY,
        "activation_id",
        "arena_run_id",
        "attempt",
        "candidate_id",
        "coding_activation_id",
        "coding_node_params",
        "command_attempt",
        "config_digest",
        "context_input_refs",
        "context_input_schemas",
        "episode_id",
        "expected_frame",
        "graph_digest",
        "graph_id",
        "graph_revision",
        "hypothesis_config",
        "node_params",
        "preview_config",
        "resource_id",
        "risk_ref",
        "robot_model_digest",
        "run_id",
        "slot_id",
        "snapshot_ref",
        "strategy",
        "workflow_id",
        "world_id",
    }
)


class ArenaError(RuntimeError):
    """Base error for Arena configuration and execution."""


class ArenaPolicyError(ArenaError):
    """Raised when a candidate could reach unauthorized effects."""


class ArenaLedgerIntegrityError(ArenaError):
    """Raised when durable Arena consumption state is corrupt or conflicting."""


class ArenaStaleContextError(ArenaError):
    """Raised when graph identity changes before a promotion is committed."""


class ShadowBindingError(ArenaPolicyError):
    """Raised when a declared shadow world lacks runtime backend proof."""


class CheckStatus(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    LOW = "low"
    HIGH = "high"


class CandidateStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class RiskInputs(_StrictModel):
    """Deterministic, manifest-visible inputs to the Arena risk gate."""

    grounding_confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    target_margin_m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    ik_status: CheckStatus = CheckStatus.UNKNOWN
    collision_status: CheckStatus = CheckStatus.UNKNOWN
    clearance_m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    held_pose_uncertainty_m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    candidate_disagreement_m: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    prior_failures: int = Field(default=0, ge=0)
    monitor_observable: bool = True


class RiskPolicy(_StrictModel):
    """Fixed thresholds recorded in the run manifest."""

    max_candidates: int = Field(default=3, ge=1, le=8)
    high_risk_score: float = Field(default=0.35, ge=0.0, le=1.0)
    min_grounding_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    min_target_margin_m: float = Field(default=0.015, ge=0.0, allow_inf_nan=False)
    min_clearance_m: float = Field(default=0.01, ge=0.0, allow_inf_nan=False)
    max_pose_uncertainty_m: float = Field(default=0.015, ge=0.0, allow_inf_nan=False)
    max_candidate_disagreement_m: float = Field(default=0.025, ge=0.0, allow_inf_nan=False)
    expand_on_unknown: bool = True


class RiskReport(_StrictModel):
    """Deterministic risk decision; low risk always means one candidate."""

    schema_version: Literal["robomex.risk_report.v1"] = "robomex.risk_report.v1"
    level: RiskLevel
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reasons: tuple[NonEmptyStr, ...] = ()
    recommended_candidates: int = Field(ge=1, le=8)
    inputs: RiskInputs
    policy: RiskPolicy

    @classmethod
    def assess(cls, inputs: RiskInputs, policy: RiskPolicy | None = None) -> RiskReport:
        policy = policy or RiskPolicy()
        reasons: list[str] = []
        score = 0.0

        if inputs.grounding_confidence < policy.min_grounding_confidence:
            reasons.append("low_grounding_confidence")
            score += 0.20
        if inputs.target_margin_m is None:
            reasons.append("target_margin_unknown")
            score += 0.12 if policy.expand_on_unknown else 0.0
        elif inputs.target_margin_m < policy.min_target_margin_m:
            reasons.append("narrow_target_margin")
            score += 0.18
        if inputs.ik_status is CheckStatus.FAIL:
            reasons.append("ik_failed")
            score += 0.40
        elif inputs.ik_status is CheckStatus.UNKNOWN:
            reasons.append("ik_unknown")
            score += 0.15 if policy.expand_on_unknown else 0.0
        if inputs.collision_status is CheckStatus.FAIL:
            reasons.append("collision_failed")
            score += 0.45
        elif inputs.collision_status is CheckStatus.UNKNOWN:
            reasons.append("collision_unknown")
            score += 0.15 if policy.expand_on_unknown else 0.0
        if inputs.clearance_m is None:
            reasons.append("clearance_unknown")
            score += 0.10 if policy.expand_on_unknown else 0.0
        elif inputs.clearance_m < policy.min_clearance_m:
            reasons.append("low_clearance")
            score += 0.16
        if inputs.held_pose_uncertainty_m is None:
            reasons.append("held_pose_uncertainty_unknown")
            score += 0.10 if policy.expand_on_unknown else 0.0
        elif inputs.held_pose_uncertainty_m > policy.max_pose_uncertainty_m:
            reasons.append("high_held_pose_uncertainty")
            score += 0.16
        if inputs.candidate_disagreement_m > policy.max_candidate_disagreement_m:
            reasons.append("candidate_disagreement")
            score += 0.18
        if inputs.prior_failures:
            reasons.append("prior_failure")
            score += min(0.10 * inputs.prior_failures, 0.30)
        if not inputs.monitor_observable:
            reasons.append("monitor_unobservable")
            score += 0.20

        score = min(round(score, 6), 1.0)
        hard_failure = (
            inputs.ik_status is CheckStatus.FAIL or inputs.collision_status is CheckStatus.FAIL
        )
        level = RiskLevel.HIGH if hard_failure or score >= policy.high_risk_score else RiskLevel.LOW
        return cls(
            level=level,
            score=score,
            reasons=tuple(reasons),
            recommended_candidates=(policy.max_candidates if level is RiskLevel.HIGH else 1),
            inputs=inputs,
            policy=policy,
        )


class ActionHypothesis(_StrictModel):
    """Unified compact card emitted by every heterogeneous candidate adapter.

    Artifact references are immutable ID+digest pairs.  A proposal worker
    cannot nominate a path, symbolic alias, or mutable ``latest`` handle for
    promotion.
    """

    schema_version: Literal["robomex.action_hypothesis.v1"] = "robomex.action_hypothesis.v1"
    candidate_id: NonEmptyStr
    strategy: NonEmptyStr
    plan_ref: ResolvedArtifactRef
    snapshot_ref: ResolvedArtifactRef
    frame: NonEmptyStr
    expected_effect: NonEmptyStr
    preconditions: tuple[NonEmptyStr, ...] = ()
    estimated_risk: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    utility: float = Field(default=0.0, allow_inf_nan=False)
    clearance_m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    path_length: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    terminal_position_m: tuple[float, float, float] | None = None
    render_refs: tuple[ResolvedArtifactRef, ...] = ()
    evidence_refs: tuple[ResolvedArtifactRef, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("terminal_position_m")
    @classmethod
    def _finite_terminal_position(
        cls, value: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if value is not None and not all(math.isfinite(component) for component in value):
            raise ValueError("terminal_position_m must contain finite values.")
        return value


@runtime_checkable
class HypothesisAdapter(Protocol):
    """Cataloged adapter from heterogeneous provider output to one card."""

    def adapt(self, value: Any, candidate: ArenaCandidateSpec) -> ActionHypothesis:
        """Normalize a provider result without executing world-changing code."""


class MappingHypothesisAdapter:
    """Strict adapter for mappings or already validated ActionHypothesis values."""

    def adapt(self, value: Any, candidate: ArenaCandidateSpec) -> ActionHypothesis:
        if isinstance(value, ActionHypothesis):
            hypothesis = value
        elif isinstance(value, Mapping):
            payload = dict(value)
            payload.setdefault("candidate_id", candidate.candidate_id)
            payload.setdefault("strategy", candidate.strategy)
            hypothesis = ActionHypothesis.model_validate(payload)
        else:
            raise TypeError("Candidate output is not an ActionHypothesis or mapping.")
        if hypothesis.candidate_id != candidate.candidate_id:
            raise ValueError("Adapter output candidate_id does not match its actor.")
        if hypothesis.strategy != candidate.strategy:
            raise ValueError("Adapter output strategy does not match its candidate spec.")
        return hypothesis


class ArenaHypothesisConfig(_StrictModel):
    """Manifest-pinned fields that a proposal model may never score itself."""

    schema_version: Literal["robomex.arena_hypothesis_config.v1"] = (
        "robomex.arena_hypothesis_config.v1"
    )
    expected_effect: NonEmptyStr = "execute_motion_plan"
    preconditions: tuple[NonEmptyStr, ...] = ()
    estimated_risk: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    utility: float = Field(default=0.0, allow_inf_nan=False)
    clearance_m: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    required_tcp_frame_id: NonEmptyStr | None = None


class ArenaPreviewConfig(_StrictModel):
    """Manifest-pinned policy for an optional read-only motion preview."""

    schema_version: Literal["robomex.arena_preview_config.v1"] = (
        "robomex.arena_preview_config.v1"
    )
    enabled: bool = False
    renderer_id: NonEmptyStr | None = None
    failure_mode: Literal["fail_closed", "omit"] = "fail_closed"

    @model_validator(mode="after")
    def _renderer_when_enabled(self) -> ArenaPreviewConfig:
        if self.enabled and self.renderer_id is None:
            raise ValueError("enabled Arena preview requires a manifest-pinned renderer_id")
        return self


class ArenaRuntimeContextV1(_StrictModel):
    """Runtime-authored candidate identity injected into invocation metadata."""

    schema_version: Literal["robomex.arena_runtime_context.v1"] = (
        "robomex.arena_runtime_context.v1"
    )
    episode_id: NonEmptyStr
    workflow_id: NonEmptyStr
    graph_id: NonEmptyStr
    graph_revision: int = Field(ge=1)
    graph_digest: Annotated[
        str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")
    ] | None = None
    command_attempt: int = Field(default=1, ge=1)
    slot_id: NonEmptyStr
    arena_run_id: NonEmptyStr
    candidate_id: NonEmptyStr
    strategy: NonEmptyStr
    snapshot_ref: ResolvedArtifactRef
    risk_ref: ResolvedArtifactRef | None = None
    expected_frame: NonEmptyStr
    world_id: NonEmptyStr
    resource_id: NonEmptyStr
    robot_model_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    config_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    coding_activation_id: NonEmptyStr | None = None
    coding_node_params: dict[str, JsonValue] = Field(default_factory=dict)
    context_input_schemas: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    context_input_refs: dict[NonEmptyStr, ResolvedArtifactRef] = Field(default_factory=dict)
    hypothesis_config: ArenaHypothesisConfig
    preview_config: ArenaPreviewConfig

    @model_validator(mode="after")
    def _closed_runtime_bindings(self) -> ArenaRuntimeContextV1:
        if set(self.context_input_schemas) != set(self.context_input_refs):
            raise ValueError("Arena context input schemas and refs must have exact keys")
        if {"snapshot", "risk"}.intersection(self.context_input_schemas):
            raise ValueError("Arena context inputs cannot redefine snapshot or risk")
        for name, schema_id in self.context_input_schemas.items():
            if _INPUT_PORT_RE.fullmatch(name) is None:
                raise ValueError(f"invalid Arena context input port {name!r}")
            if _SCHEMA_ID_RE.fullmatch(schema_id) is None:
                raise ValueError(f"invalid Arena context schema {schema_id!r}")
        if self.coding_activation_id is None:
            if self.coding_node_params:
                raise ValueError("coding node params require a coding activation ID")
        elif self.graph_digest is None:
            raise ValueError("coding Arena context requires an exact graph digest")
        return self


class ArenaMotionPreviewArtifact(_StrictModel):
    """Typed envelope for one deterministic, read-only rendered preview."""

    schema_version: Literal["robomex.motion_preview.v1"] = "robomex.motion_preview.v1"
    renderer_id: NonEmptyStr
    candidate_id: NonEmptyStr
    plan_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    view_id: NonEmptyStr
    media_type: NonEmptyStr
    media_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    terminal_position_m: tuple[float, float, float] | None = None

    @field_validator("terminal_position_m")
    @classmethod
    def _finite_preview_terminal(
        cls,
        value: tuple[float, float, float] | None,
    ) -> tuple[float, float, float] | None:
        if value is not None and not all(math.isfinite(component) for component in value):
            raise ValueError("preview terminal_position_m must contain finite values")
        return value


class HardGateResult(_StrictModel):
    gate_id: NonEmptyStr
    passed: bool
    reason: str = ""


@runtime_checkable
class HypothesisHardGate(Protocol):
    """Runtime-owned deterministic gate evaluated before ranking."""

    gate_id: str

    def evaluate(self, hypothesis: ActionHypothesis) -> HardGateResult:
        """Return a deterministic pass/fail record."""


@dataclass(frozen=True)
class BasicHypothesisGate:
    """Frame/freshness gate; numeric/schema checks happen during model parsing."""

    expected_frame: str
    expected_snapshot_ref: ResolvedArtifactRef
    gate_id: str = "schema_frame_freshness"

    def evaluate(self, hypothesis: ActionHypothesis) -> HardGateResult:
        if hypothesis.frame != self.expected_frame:
            return HardGateResult(
                gate_id=self.gate_id,
                passed=False,
                reason=f"frame {hypothesis.frame!r} != {self.expected_frame!r}",
            )
        if hypothesis.snapshot_ref != self.expected_snapshot_ref:
            return HardGateResult(
                gate_id=self.gate_id,
                passed=False,
                reason="hypothesis is stale relative to the Arena snapshot",
            )
        return HardGateResult(gate_id=self.gate_id, passed=True)


@dataclass(frozen=True)
class MetadataStatusGate:
    """Optional ranking/advisory gate over untrusted candidate metadata.

    This gate is never sufficient for motion promotion.  The mandatory
    :class:`RuntimeMotionPromotionAuthority` independently resolves the exact
    action and snapshot artifacts and runs trusted feasibility checks.
    """

    gate_id: str
    metadata_key: str
    accepted_values: frozenset[str] = frozenset({"pass"})

    def evaluate(self, hypothesis: ActionHypothesis) -> HardGateResult:
        raw = hypothesis.metadata.get(self.metadata_key)
        value = str(raw).lower() if raw is not None else "unknown"
        passed = value in self.accepted_values
        return HardGateResult(
            gate_id=self.gate_id,
            passed=passed,
            reason="" if passed else f"{self.metadata_key}={value}",
        )


@runtime_checkable
class ArtifactLookup(Protocol):
    """Episode-scoped fail-closed artifact resolver used by the authority."""

    def resolve(self, ref: ResolvedArtifactRef | Mapping[str, Any]) -> ResolvedArtifact: ...


class MotionSafetyEvaluation(_StrictModel):
    """Runtime-owned, effect-free evaluation of one exact motion proposal."""

    schema_version: Literal["robomex.motion_safety_evaluation.v1"] = (
        "robomex.motion_safety_evaluation.v1"
    )
    candidate_id: NonEmptyStr
    action_spec_ref: ResolvedArtifactRef
    snapshot_ref: ResolvedArtifactRef
    action_spec_digest: str | None = None
    world_id: str | None = None
    resource_id: str | None = None
    robot_model_digest: str | None = None
    config_digest: str | None = None
    feasibility_certificate: FeasibilityCertificate | None = None
    gate_results: tuple[HardGateResult, ...]
    eligible: bool

    @model_validator(mode="after")
    def _closed_eligibility(self) -> MotionSafetyEvaluation:
        reduced = bool(self.gate_results) and all(item.passed for item in self.gate_results)
        if self.eligible != reduced:
            raise ValueError("eligible must equal the closed hard-gate reduction")
        if self.eligible and self.feasibility_certificate is None:
            raise ValueError("eligible motion requires a runtime feasibility certificate")
        return self


class PromotionReceipt(_StrictModel):
    """Selection receipt; it grants no physical lease and commits no effect."""

    schema_version: Literal["robomex.arena_promotion_receipt.v1"] = (
        "robomex.arena_promotion_receipt.v1"
    )
    receipt_authority: Literal["runtime"] = "runtime"
    arena_run_id: NonEmptyStr
    candidate_id: NonEmptyStr
    graph_id: NonEmptyStr
    graph_revision: int = Field(ge=1)
    action_spec_ref: ResolvedArtifactRef
    snapshot_ref: ResolvedArtifactRef
    action_spec_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    world_id: NonEmptyStr
    resource_id: NonEmptyStr
    robot_model_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    config_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    feasibility_certificate: FeasibilityCertificate
    gate_results: tuple[HardGateResult, ...]
    promotion_status: Literal["eligible_for_action_admission"] = "eligible_for_action_admission"
    authoritative_effect_committed: Literal[False] = False

    @model_validator(mode="after")
    def _passing_receipt(self) -> PromotionReceipt:
        if not self.gate_results or not all(item.passed for item in self.gate_results):
            raise ValueError("promotion receipt requires every hard gate to pass")
        certificate = self.feasibility_certificate
        if certificate.action_spec_digest != self.action_spec_digest:
            raise ValueError("promotion certificate does not bind the action spec")
        if certificate.overall_status is not FeasibilityStatus.PASS:
            raise ValueError("promotion certificate must pass")
        return self


class RuntimeMotionPromotionAuthority:
    """Trusted core gate for an exact artifact-backed motion plan.

    Candidate metadata, risk reports, renders, and shadow-rollout summaries are
    evidence only.  They cannot replace these checks or turn UNKNOWN into PASS.
    The authority is deliberately read-only: promotion creates a receipt that
    may later be submitted to ``ActionSupervisor`` but never executes a robot.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        artifacts: ArtifactLookup,
        snapshot_providers: Mapping[tuple[str, str], Callable[[str, str], AdmissionSnapshot]],
        feasibility_checkers: Mapping[tuple[str, str], FeasibilityChecker],
    ) -> None:
        if not episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if not isinstance(artifacts, ArtifactLookup):
            raise TypeError("artifacts must implement the fail-closed ArtifactLookup")
        self.episode_id = episode_id
        self._artifacts = artifacts
        self._snapshot_providers = dict(snapshot_providers)
        self._checkers = dict(feasibility_checkers)
        if set(self._snapshot_providers) != set(self._checkers):
            raise ValueError("snapshot providers and feasibility checkers must bind equally")
        for key in self._checkers:
            if len(key) != 2 or not all(str(part).strip() for part in key):
                raise ValueError("motion authority keys must be (world_id, resource_id)")
            if not isinstance(self._checkers[key], FeasibilityChecker):
                raise TypeError(f"checker for {key!r} does not implement FeasibilityChecker")

    def assert_bound_to(self, *, episode_id: str, artifacts: ArtifactLookup) -> None:
        """Prove that an Episode is using its own resolver as the trust root."""

        if episode_id != self.episode_id or artifacts is not self._artifacts:
            raise ArenaPolicyError(
                "Arena promotion authority is not bound to this EpisodeDataPlane."
            )

    @property
    def artifacts(self) -> ArtifactLookup:
        """Exact episode resolver used by runtime-owned auxiliary gates."""

        return self._artifacts

    def evaluate(
        self, *, context: ArenaContext, hypothesis: ActionHypothesis
    ) -> MotionSafetyEvaluation:
        """Resolve, bind, and certify one proposal without physical effects."""

        gates: list[HardGateResult] = []
        plan: MotionPlan | None = None
        snapshot: AdmissionSnapshot | None = None
        certificate: FeasibilityCertificate | None = None
        try:
            if context.episode_id != self.episode_id:
                raise ValueError("Arena context belongs to another episode")
            plan_artifact = self._artifacts.resolve(hypothesis.plan_ref)
            snapshot_artifact = self._artifacts.resolve(hypothesis.snapshot_ref)
            for ref in (*hypothesis.render_refs, *hypothesis.evidence_refs):
                self._artifacts.resolve(ref)
            if plan_artifact.schema != "robomex.motion_plan.v2":
                raise ValueError(
                    f"plan artifact schema is {plan_artifact.schema!r}, not a motion plan"
                )
            if snapshot_artifact.schema != "robomex.admission_snapshot.v1":
                raise ValueError("snapshot artifact schema is not robomex.admission_snapshot.v1")
            parsed = validate_action_spec(plan_artifact.payload)
            if not isinstance(parsed, MotionPlan):
                raise ValueError("Arena promotion supports exact MotionPlan artifacts only")
            plan = parsed
            snapshot = AdmissionSnapshot.model_validate(snapshot_artifact.payload)
            gates.append(HardGateResult(gate_id="artifact_integrity", passed=True))
        except Exception as exc:
            gates.append(
                HardGateResult(
                    gate_id="artifact_integrity",
                    passed=False,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            return self._failed_evaluation(hypothesis, gates)

        snapshot_passed, snapshot_reason = self._snapshot_binding(
            context=context,
            hypothesis=hypothesis,
            plan=plan,
            snapshot=snapshot,
        )
        gates.append(
            HardGateResult(
                gate_id="snapshot_binding",
                passed=snapshot_passed,
                reason=snapshot_reason,
            )
        )
        robot_passed = (
            plan.robot_model_digest == context.robot_model_digest
            and snapshot.config_digest == context.config_digest
            and plan.world_id == context.world_id
            and plan.resource_id == context.resource_id
        )
        gates.append(
            HardGateResult(
                gate_id="robot_config_binding",
                passed=robot_passed,
                reason=(
                    ""
                    if robot_passed
                    else "motion plan robot/world/resource or snapshot config is not bound"
                ),
            )
        )

        key = (context.world_id, context.resource_id)
        provider = self._snapshot_providers.get(key)
        checker = self._checkers.get(key)
        if provider is None or checker is None:
            gates.append(
                HardGateResult(
                    gate_id="current_snapshot_freshness",
                    passed=False,
                    reason=f"no trusted runtime binding for {key!r}",
                )
            )
            return self._failed_evaluation(hypothesis, gates, plan=plan, snapshot=snapshot)
        try:
            current = AdmissionSnapshot.model_validate(provider(*key).model_dump(mode="python"))
            fresh, reason = self._current_snapshot_matches(plan, current)
        except Exception as exc:
            fresh = False
            reason = f"{type(exc).__name__}: {exc}"
        gates.append(
            HardGateResult(gate_id="current_snapshot_freshness", passed=fresh, reason=reason)
        )

        binding_passed = all(item.passed for item in gates)
        if binding_passed:
            try:
                raw_certificate = checker.certify(plan, current)
                certificate = FeasibilityCertificate.model_validate(
                    raw_certificate.model_dump(mode="python")
                )
                if certificate.action_spec_digest != plan.content_digest:
                    raise ValueError("checker certificate action digest mismatch")
                current_digest = canonical_model_digest(current)
                if certificate.admission_snapshot_digest != current_digest:
                    raise ValueError("checker certificate snapshot digest mismatch")
                gates.extend(self._certificate_gates(certificate))
            except Exception as exc:
                gates.extend(
                    HardGateResult(
                        gate_id=gate_id,
                        passed=False,
                        reason=f"checker error/unknown: {type(exc).__name__}: {exc}",
                    )
                    for gate_id in _REQUIRED_FEASIBILITY_GATE_IDS
                )
        else:
            gates.extend(
                HardGateResult(
                    gate_id=gate_id,
                    passed=False,
                    reason="not evaluated because artifact/binding/freshness gate failed",
                )
                for gate_id in _REQUIRED_FEASIBILITY_GATE_IDS
            )

        eligible = all(item.passed for item in gates)
        return MotionSafetyEvaluation(
            candidate_id=hypothesis.candidate_id,
            action_spec_ref=hypothesis.plan_ref,
            snapshot_ref=hypothesis.snapshot_ref,
            action_spec_digest=plan.content_digest,
            world_id=plan.world_id,
            resource_id=plan.resource_id,
            robot_model_digest=plan.robot_model_digest,
            config_digest=snapshot.config_digest,
            feasibility_certificate=certificate,
            gate_results=tuple(gates),
            eligible=eligible,
        )

    @staticmethod
    def _snapshot_binding(
        *,
        context: ArenaContext,
        hypothesis: ActionHypothesis,
        plan: MotionPlan,
        snapshot: AdmissionSnapshot,
    ) -> tuple[bool, str]:
        mismatches: list[str] = []
        if hypothesis.snapshot_ref != context.snapshot_ref:
            mismatches.append("context_snapshot_ref")
        if hypothesis.frame != context.expected_frame:
            mismatches.append("frame")
        if plan.expected_snapshot != snapshot:
            mismatches.append("plan_expected_snapshot")
        if snapshot.world_kind is not WorldKind.AUTHORITATIVE:
            mismatches.append("world_kind")
        return not mismatches, ("" if not mismatches else ", ".join(mismatches))

    @staticmethod
    def _current_snapshot_matches(plan: MotionPlan, current: AdmissionSnapshot) -> tuple[bool, str]:
        expected = plan.expected_snapshot
        exact_fields = (
            "world_id",
            "world_kind",
            "resource_id",
            "robot_revision",
            "scene_revision",
            "attachment_revision",
            "config_revision",
            "config_digest",
            "collision_world_digest",
            "attachment_status",
            "joint_names",
        )
        mismatches = [
            name for name in exact_fields if getattr(current, name) != getattr(expected, name)
        ]
        if current.controller_state.value not in {"ready", "quiescent"}:
            mismatches.append("controller_state")
        if len(current.joint_positions_rad) != len(expected.joint_positions_rad):
            mismatches.append("joint_positions_rad_width")
        elif any(
            abs(now - planned) > plan.max_start_deviation_rad
            for now, planned in zip(
                current.joint_positions_rad,
                expected.joint_positions_rad,
                strict=True,
            )
        ):
            mismatches.append("joint_positions_rad_deviation")
        return not mismatches, ("" if not mismatches else ", ".join(mismatches))

    @staticmethod
    def _certificate_gates(
        certificate: FeasibilityCertificate,
    ) -> tuple[HardGateResult, ...]:
        checks = certificate.checks
        direct_kinematics = {
            name: status for name, status in checks.items() if name in _DIRECT_KINEMATIC_CHECKS
        }
        sealed_path_kinematics = {
            name: checks[name] for name in _SEALED_PATH_KINEMATIC_CHECKS if name in checks
        }
        direct_passed = bool(direct_kinematics) and all(
            status is FeasibilityStatus.PASS for status in direct_kinematics.values()
        )
        sealed_path_passed = len(sealed_path_kinematics) == len(
            _SEALED_PATH_KINEMATIC_CHECKS
        ) and all(status is FeasibilityStatus.PASS for status in sealed_path_kinematics.values())
        kinematics_passed = direct_passed or sealed_path_passed

        collision = {name: status for name, status in checks.items() if name in _COLLISION_CHECKS}
        collision_passed = bool(collision) and all(
            status is FeasibilityStatus.PASS for status in collision.values()
        )
        limits_status = checks.get("joint_limits", FeasibilityStatus.UNKNOWN)
        limits_passed = limits_status is FeasibilityStatus.PASS
        results = [
            HardGateResult(
                gate_id="kinematic_feasibility",
                passed=kinematics_passed,
                reason=(
                    ""
                    if kinematics_passed
                    else "requires PASS from direct IK/kinematics or the exact-path "
                    "interface+joint-order+robot-model checks"
                ),
            ),
            HardGateResult(
                gate_id="collision",
                passed=collision_passed,
                reason=(
                    ""
                    if collision_passed
                    else "runtime collision certificate is missing, FAIL, or UNKNOWN"
                ),
            ),
            HardGateResult(
                gate_id="joint_limits",
                passed=limits_passed,
                reason=(
                    "" if limits_passed else f"runtime certificate status={limits_status.value}"
                ),
            ),
        ]
        if certificate.overall_status is not FeasibilityStatus.PASS:
            return tuple(
                item.model_copy(
                    update={
                        "passed": False,
                        "reason": item.reason or "overall certificate did not pass",
                    }
                )
                for item in results
            )
        return tuple(results)

    @staticmethod
    def _failed_evaluation(
        hypothesis: ActionHypothesis,
        gates: list[HardGateResult],
        *,
        plan: MotionPlan | None = None,
        snapshot: AdmissionSnapshot | None = None,
    ) -> MotionSafetyEvaluation:
        existing_ids = {item.gate_id for item in gates}
        for gate_id in (
            "snapshot_binding",
            "robot_config_binding",
            "current_snapshot_freshness",
            *_REQUIRED_FEASIBILITY_GATE_IDS,
        ):
            if gate_id not in existing_ids:
                gates.append(
                    HardGateResult(
                        gate_id=gate_id,
                        passed=False,
                        reason="not evaluated after an earlier fail-closed rejection",
                    )
                )
        return MotionSafetyEvaluation(
            candidate_id=hypothesis.candidate_id,
            action_spec_ref=hypothesis.plan_ref,
            snapshot_ref=hypothesis.snapshot_ref,
            action_spec_digest=(plan.content_digest if plan is not None else None),
            world_id=(plan.world_id if plan is not None else None),
            resource_id=(plan.resource_id if plan is not None else None),
            robot_model_digest=(plan.robot_model_digest if plan is not None else None),
            config_digest=(snapshot.config_digest if snapshot is not None else None),
            gate_results=tuple(gates),
            eligible=False,
        )


def _normalize_context_input_schemas(values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("Arena context_input_schemas must be a mapping")
    normalized: dict[str, str] = {}
    for raw_name, raw_schema in values.items():
        if not isinstance(raw_name, str) or not isinstance(raw_schema, str):
            raise TypeError("Arena context input ports and schemas must be strings")
        name = str(raw_name).strip()
        schema_id = str(raw_schema).strip()
        if _INPUT_PORT_RE.fullmatch(name) is None:
            raise ValueError(f"invalid Arena context input port {raw_name!r}")
        if name in {"snapshot", "risk"}:
            raise ValueError("Arena context_input_schemas cannot redefine snapshot or risk")
        if _SCHEMA_ID_RE.fullmatch(schema_id) is None:
            raise ValueError(f"invalid versioned Arena context schema {raw_schema!r}")
        normalized[name] = schema_id
    return MappingProxyType(dict(sorted(normalized.items())))


def _profile_coding_node_params(
    profile: ActorProfile,
    activation_id: str,
) -> dict[str, JsonValue]:
    """Resolve one immutable node-config row from the manifest-pinned profile."""

    rows = profile.metadata.get("node_config_v1")
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ArenaPolicyError(
            "Arena coding candidate requires manifest-pinned profile node_config_v1 rows"
        )
    matches = [row for row in rows if isinstance(row, (tuple, list)) and len(row) == 2 and str(row[0]).strip() == activation_id]
    if len(matches) != 1:
        raise ArenaPolicyError(
            f"coding activation {activation_id!r} must match exactly one profile node config"
        )
    encoded = matches[0][1]
    if not isinstance(encoded, str):
        raise ArenaPolicyError("profile node_config_v1 payload must be canonical JSON text")
    try:
        raw = json.loads(encoded)
        if not isinstance(raw, dict):
            raise TypeError("node config must decode to a mapping")
        # Re-encoding with allow_nan=False rejects non-finite or non-JSON rows.
        json.dumps(raw, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArenaPolicyError(
            f"profile node config for {activation_id!r} is not finite JSON"
        ) from exc
    return raw


@dataclass(frozen=True)
class ArenaCandidateSpec:
    """One ordered candidate template inside an already authorized Arena slot."""

    candidate_id: str
    strategy: str
    profile: ActorProfile
    objective: str
    effect_scope: EffectScope = EffectScope.READ_ONLY
    inputs: Mapping[str, Any] = field(default_factory=dict)
    output_contract: Mapping[str, str] = field(
        default_factory=lambda: {"hypothesis": "robomex.action_hypothesis.v1"}
    )
    requested_capabilities: frozenset[str] = field(default_factory=frozenset)
    requested_effects: frozenset[str] = field(default_factory=frozenset)
    shadow_backend_id: str | None = None
    shadow_resource_id: str | None = None
    adapter_id: str = "mapping_v1"
    estimated_budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    invocation_metadata: Mapping[str, Any] = field(default_factory=dict)
    coding_activation_id: str | None = None
    context_input_schemas: Mapping[str, str] = field(default_factory=dict)
    hypothesis_config: ArenaHypothesisConfig = field(default_factory=ArenaHypothesisConfig)
    preview_config: ArenaPreviewConfig = field(default_factory=ArenaPreviewConfig)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in self.candidate_id
        ):
            raise ValueError("candidate_id must be a path-safe identifier.")
        if not self.strategy.strip() or not self.objective.strip():
            raise ValueError("strategy and objective must not be empty.")
        object.__setattr__(self, "effect_scope", EffectScope(self.effect_scope))
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "output_contract", MappingProxyType(dict(self.output_contract)))
        object.__setattr__(
            self,
            "requested_capabilities",
            frozenset(self.requested_capabilities),
        )
        object.__setattr__(self, "requested_effects", frozenset(self.requested_effects))
        invocation_metadata = dict(self.invocation_metadata)
        reserved_metadata = sorted(set(invocation_metadata) & _ARENA_RESERVED_METADATA_KEYS)
        if reserved_metadata:
            raise ValueError(
                "Arena candidate metadata cannot override runtime context: "
                + ", ".join(reserved_metadata)
            )
        try:
            json.dumps(
                invocation_metadata,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Arena invocation_metadata must be finite JSON data") from exc
        object.__setattr__(
            self,
            "invocation_metadata",
            MappingProxyType(invocation_metadata),
        )
        coding_activation_id = self.coding_activation_id
        if coding_activation_id is not None:
            coding_activation_id = str(coding_activation_id).strip()
            if _INPUT_PORT_RE.fullmatch(coding_activation_id) is None:
                raise ValueError("coding_activation_id must be a path-safe graph node ID")
            if self.profile.metadata.get("strict_runtime_context") is not True:
                raise ValueError(
                    "Arena coding activation requires profile strict_runtime_context=True"
                )
            _profile_coding_node_params(self.profile, coding_activation_id)
            object.__setattr__(self, "coding_activation_id", coding_activation_id)
        object.__setattr__(
            self,
            "context_input_schemas",
            _normalize_context_input_schemas(self.context_input_schemas),
        )
        hypothesis_config = (
            self.hypothesis_config
            if isinstance(self.hypothesis_config, ArenaHypothesisConfig)
            else ArenaHypothesisConfig.model_validate(self.hypothesis_config)
        )
        preview_config = (
            self.preview_config
            if isinstance(self.preview_config, ArenaPreviewConfig)
            else ArenaPreviewConfig.model_validate(self.preview_config)
        )
        object.__setattr__(self, "hypothesis_config", hypothesis_config)
        object.__setattr__(self, "preview_config", preview_config)
        budget = (
            self.estimated_budget
            if isinstance(self.estimated_budget, ExecutionBudget)
            else ExecutionBudget.model_validate(self.estimated_budget)
        )
        if budget.authoritative_actions or budget.shadow_rollouts or budget.actor_spawns:
            raise ValueError(
                "Arena candidate estimated_budget may contain only model_calls, "
                "tokens, and wall_time_ms; Arena owns effects and spawns"
            )
        object.__setattr__(self, "estimated_budget", budget)
        for name in ("shadow_backend_id", "shadow_resource_id"):
            value = getattr(self, name)
            if value is not None:
                normalized = str(value).strip()
                if not normalized:
                    raise ValueError(f"{name} must be non-empty when supplied")
                object.__setattr__(self, name, normalized)


def _profile_payload(profile: ActorProfile) -> dict[str, Any]:
    """Canonical, JSON-safe profile material included in an Arena binding pin."""

    return {
        "profile_id": profile.profile_id,
        "provider_id": profile.provider_id,
        "runner_kind": profile.runner_kind,
        "lifecycle": profile.lifecycle.value,
        "model": profile.model,
        "capability_ceiling": sorted(profile.capability_ceiling),
        "effect_ceiling": sorted(profile.effect_ceiling),
        "isolation": {
            "namespace_prefix": profile.isolation.namespace_prefix,
            "workspace_mode": profile.isolation.workspace_mode.value,
            "workspace_key": profile.isolation.workspace_key,
            "world_id": profile.isolation.world_id,
            "metadata": dict(profile.isolation.metadata),
        },
        "metadata": dict(profile.metadata),
    }


def _trusted_component_payload(value: Any) -> dict[str, Any]:
    """Bind configured trusted gates/adapters without serializing live handles."""

    configuration: Any = None
    if isinstance(value, BaseModel):
        configuration = value.model_dump(mode="json")
    elif is_dataclass(value):
        configuration = asdict(value)
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "configuration": configuration,
    }


@dataclass(frozen=True)
class ArenaBinding:
    """Trusted, manifest-pinnable implementation behind one graph Arena node.

    Graph authors may select ``binding_id`` through ``runner_ref`` but cannot
    manufacture candidate profiles, adapters, effect ceilings, or budgets in
    the graph's open-ended ``params`` mapping.
    """

    binding_id: str
    candidates: tuple[ArenaCandidateSpec, ...]
    expected_frame: str
    robot_model_digest: str
    candidate_budget_limit: int
    candidate_budget_id: str | None = None
    context_input_schemas: Mapping[str, str] = field(default_factory=dict)
    gates: tuple[HypothesisHardGate, ...] = ()
    adapters: Mapping[str, HypothesisAdapter] = field(default_factory=dict)
    policy: ArenaPolicy | None = None
    risk_policy: RiskPolicy | None = None
    content_digest: str = field(init=False)

    def __post_init__(self) -> None:
        binding_id = str(self.binding_id).strip()
        if not binding_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in binding_id
        ):
            raise ValueError("Arena binding_id must be a path-safe identifier")
        if not self.expected_frame.strip():
            raise ValueError("Arena expected_frame must not be empty")
        if not (
            isinstance(self.robot_model_digest, str)
            and len(self.robot_model_digest) == 71
            and self.robot_model_digest.startswith("sha256:")
            and all(
                character in "0123456789abcdef"
                for character in self.robot_model_digest.removeprefix("sha256:")
            )
        ):
            raise ValueError("Arena robot_model_digest must be canonical sha256")
        if isinstance(self.candidate_budget_limit, bool) or self.candidate_budget_limit < 1:
            raise ValueError("Arena candidate_budget_limit must be positive")
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("Arena binding requires at least one candidate")
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Arena binding candidate IDs must be unique")
        context_input_schemas = _normalize_context_input_schemas(
            self.context_input_schemas
        )
        for candidate in candidates:
            if candidate.context_input_schemas:
                raise ValueError(
                    "Arena candidate templates cannot override binding-owned "
                    "context_input_schemas"
                )
        budget_id = (self.candidate_budget_id or f"{binding_id}.candidates").strip()
        if not budget_id:
            raise ValueError("Arena candidate_budget_id must not be empty")
        adapters = dict(self.adapters)
        if any(not key.strip() for key in adapters):
            raise ValueError("Arena adapter IDs must not be empty")
        available_adapters = set(adapters) | {"mapping_v1"}
        unknown_adapters = sorted(
            candidate.adapter_id
            for candidate in candidates
            if candidate.adapter_id not in available_adapters
        )
        if unknown_adapters:
            raise ValueError(
                "Arena candidates reference unregistered adapters: " + ", ".join(unknown_adapters)
            )
        for candidate in candidates:
            SwarmArena._validate_candidate(candidate)
        policy = self.policy or ArenaPolicy()
        risk_policy = self.risk_policy or RiskPolicy(max_candidates=policy.max_candidates)
        if policy.max_candidates > self.candidate_budget_limit:
            raise ValueError("Arena policy max_candidates exceeds its trusted candidate budget")
        if risk_policy.max_candidates > policy.max_candidates:
            raise ValueError("Arena risk policy cannot expand beyond the trusted Arena policy")
        object.__setattr__(self, "binding_id", binding_id)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "candidate_budget_id", budget_id)
        object.__setattr__(self, "context_input_schemas", context_input_schemas)
        object.__setattr__(self, "gates", tuple(self.gates))
        object.__setattr__(self, "adapters", MappingProxyType(adapters))
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "risk_policy", risk_policy)
        payload = {
            "binding_id": binding_id,
            "expected_frame": self.expected_frame,
            "robot_model_digest": self.robot_model_digest,
            "candidate_budget_limit": self.candidate_budget_limit,
            "candidate_budget_id": budget_id,
            "context_input_schemas": dict(context_input_schemas),
            "policy": policy.model_dump(mode="json"),
            "risk_policy": risk_policy.model_dump(mode="json"),
            "gates": [_trusted_component_payload(gate) for gate in self.gates],
            "adapters": {
                key: _trusted_component_payload(adapter)
                for key, adapter in sorted(adapters.items())
            },
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "strategy": candidate.strategy,
                    "profile": _profile_payload(candidate.profile),
                    "objective": candidate.objective,
                    "effect_scope": candidate.effect_scope.value,
                    "inputs": dict(candidate.inputs),
                    "output_contract": dict(candidate.output_contract),
                    "requested_capabilities": sorted(candidate.requested_capabilities),
                    "requested_effects": sorted(candidate.requested_effects),
                    "shadow_backend_id": candidate.shadow_backend_id,
                    "shadow_resource_id": candidate.shadow_resource_id,
                    "adapter_id": candidate.adapter_id,
                    "estimated_budget": candidate.estimated_budget.model_dump(mode="json"),
                    "invocation_metadata": dict(candidate.invocation_metadata),
                    "coding_activation_id": candidate.coding_activation_id,
                    "context_input_schemas": dict(candidate.context_input_schemas),
                    "hypothesis_config": candidate.hypothesis_config.model_dump(mode="json"),
                    "preview_config": candidate.preview_config.model_dump(mode="json"),
                }
                for candidate in candidates
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        object.__setattr__(self, "content_digest", f"sha256:{hashlib.sha256(encoded).hexdigest()}")

    def candidates_for(
        self,
        *,
        snapshot_ref: ResolvedArtifactRef,
        risk_ref: ResolvedArtifactRef,
        context_refs: Mapping[
            str, ResolvedArtifactRef | Mapping[str, Any]
        ] | None = None,
    ) -> tuple[ArenaCandidateSpec, ...]:
        """Inject admitted graph inputs into trusted candidate templates."""

        try:
            normalized_snapshot_ref = ResolvedArtifactRef.from_any(snapshot_ref)
            normalized_risk_ref = ResolvedArtifactRef.from_any(risk_ref)
        except (TypeError, ValueError) as exc:
            raise ArenaPolicyError(
                "Arena snapshot and risk inputs must be exact artifact refs"
            ) from exc
        supplied_context = dict(context_refs or {})
        expected_keys = set(self.context_input_schemas)
        supplied_keys = set(supplied_context)
        if supplied_keys != expected_keys:
            missing = sorted(expected_keys - supplied_keys)
            unknown = sorted(supplied_keys - expected_keys)
            details: list[str] = []
            if missing:
                details.append(f"missing={missing!r}")
            if unknown:
                details.append(f"unknown={unknown!r}")
            raise ArenaPolicyError(
                "Arena context refs must exactly match the manifest-pinned schema map: "
                + ", ".join(details)
            )
        normalized_context_refs: dict[str, ResolvedArtifactRef] = {}
        for name in sorted(expected_keys):
            try:
                normalized_context_refs[name] = ResolvedArtifactRef.from_any(
                    supplied_context[name]
                )
            except (TypeError, ValueError) as exc:
                raise ArenaPolicyError(
                    f"Arena context input {name!r} is not an exact artifact ref"
                ) from exc

        materialized: list[ArenaCandidateSpec] = []
        for candidate in self.candidates:
            runtime_ports = {"snapshot", "risk", *expected_keys}
            overridden = sorted(runtime_ports.intersection(candidate.inputs))
            if overridden:
                raise ArenaPolicyError(
                    "Arena candidate templates cannot override runtime input bindings: "
                    + ", ".join(overridden)
                )
            materialized.append(
                ArenaCandidateSpec(
                    candidate_id=candidate.candidate_id,
                    strategy=candidate.strategy,
                    profile=candidate.profile,
                    objective=candidate.objective,
                    effect_scope=candidate.effect_scope,
                    inputs={
                        **dict(candidate.inputs),
                        "snapshot": normalized_snapshot_ref.to_mapping(),
                        "risk": normalized_risk_ref.to_mapping(),
                        **{
                            name: ref.to_mapping()
                            for name, ref in normalized_context_refs.items()
                        },
                    },
                    output_contract=candidate.output_contract,
                    requested_capabilities=candidate.requested_capabilities,
                    requested_effects=candidate.requested_effects,
                    shadow_backend_id=candidate.shadow_backend_id,
                    shadow_resource_id=candidate.shadow_resource_id,
                    adapter_id=candidate.adapter_id,
                    estimated_budget=candidate.estimated_budget,
                    invocation_metadata=candidate.invocation_metadata,
                    coding_activation_id=candidate.coding_activation_id,
                    context_input_schemas=self.context_input_schemas,
                    hypothesis_config=candidate.hypothesis_config,
                    preview_config=candidate.preview_config,
                )
            )
        return tuple(materialized)

    def candidate_provider_budget(self, count: int) -> ExecutionBudget:
        """Aggregate the trusted prefix that the risk policy may invoke."""

        if isinstance(count, bool) or count < 0 or count > len(self.candidates):
            raise ValueError("Arena candidate budget count is outside the binding")
        selected = self.candidates[:count]
        return ExecutionBudget(
            model_calls=sum(item.estimated_budget.model_calls for item in selected),
            tokens=sum(item.estimated_budget.tokens for item in selected),
            wall_time_ms=sum(item.estimated_budget.wall_time_ms for item in selected),
        )


class ArenaBindingRegistry:
    """Immutable-by-ID runtime inventory for graph-native Arena runners."""

    def __init__(self, bindings: Sequence[ArenaBinding] = ()) -> None:
        self._bindings: dict[str, ArenaBinding] = {}
        for binding in bindings:
            self.register(binding)

    def register(self, binding: ArenaBinding) -> None:
        if not isinstance(binding, ArenaBinding):
            raise TypeError("Arena registry accepts only ArenaBinding values")
        existing = self._bindings.get(binding.binding_id)
        if existing is not None and existing != binding:
            raise ArenaPolicyError(
                f"Arena binding {binding.binding_id!r} is already registered differently"
            )
        self._bindings[binding.binding_id] = binding

    def resolve(self, binding_id: str) -> ArenaBinding:
        try:
            return self._bindings[binding_id]
        except KeyError as exc:
            raise ArenaPolicyError(
                f"Arena binding {binding_id!r} is not in the trusted runtime registry"
            ) from exc

    @property
    def bindings(self) -> tuple[ArenaBinding, ...]:
        return tuple(self._bindings[key] for key in sorted(self._bindings))


class ArenaContext(_StrictModel):
    arena_run_id: NonEmptyStr
    episode_id: NonEmptyStr
    workflow_id: NonEmptyStr
    graph_id: NonEmptyStr
    graph_revision: int = Field(ge=1)
    graph_digest: Annotated[
        str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")
    ] | None = None
    command_attempt: int = Field(default=1, ge=1)
    slot_id: NonEmptyStr
    snapshot_ref: ResolvedArtifactRef
    expected_frame: NonEmptyStr
    world_id: NonEmptyStr
    resource_id: NonEmptyStr
    robot_model_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    config_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    candidate_budget_id: NonEmptyStr
    candidate_budget_limit: int = Field(ge=0)
    source: NonEmptyStr = "swarm_arena"


class RuntimeArenaContextGuard:
    """Runtime-owned graph revision oracle checked before and after exploration."""

    def __init__(
        self,
        *,
        episode_id: str,
        current_revision: Callable[[ArenaContext], tuple[str, int]],
        binding_token: object | None = None,
    ) -> None:
        if not episode_id.strip():
            raise ValueError("episode_id must not be empty")
        self.episode_id = episode_id
        self._current_revision = current_revision
        self._binding_token = binding_token

    def assert_bound_to(self, *, episode_id: str, binding_token: object) -> None:
        if (
            self.episode_id != episode_id
            or self._binding_token is None
            or self._binding_token is not binding_token
        ):
            raise ArenaPolicyError("Arena context guard is not owned by this EpisodeRuntime.")

    def assert_current(self, context: ArenaContext) -> None:
        if context.episode_id != self.episode_id:
            raise ArenaStaleContextError("Arena context belongs to another episode")
        graph_id, revision = self._current_revision(context)
        if graph_id != context.graph_id or revision != context.graph_revision:
            raise ArenaStaleContextError(
                "Arena context became stale: "
                f"expected {context.graph_id!r}@{context.graph_revision}, "
                f"current {graph_id!r}@{revision}."
            )


class ArenaReservationRecord(_StrictModel):
    schema_version: Literal["robomex.arena_reservation.v1"] = "robomex.arena_reservation.v1"
    record_kind: Literal["reservation"] = "reservation"
    sequence: int = Field(ge=1)
    arena_run_id: NonEmptyStr
    episode_id: NonEmptyStr
    workflow_id: NonEmptyStr
    graph_id: NonEmptyStr
    graph_revision: int = Field(ge=1)
    candidate_budget_id: NonEmptyStr
    candidate_budget_limit: int = Field(ge=0)
    requested_candidates: int = Field(ge=1)
    caller_remaining: int = Field(default=0, ge=0)
    reserved_candidates: int = Field(ge=0)
    context_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")] | None = (
        None
    )
    run_binding_digest: (
        Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")] | None
    ) = None

    @model_validator(mode="after")
    def _bounded_reservation(self) -> ArenaReservationRecord:
        if self.reserved_candidates > self.requested_candidates:
            raise ValueError("reserved candidates exceed this run's request")
        if self.reserved_candidates > self.candidate_budget_limit:
            raise ValueError("reserved candidates exceed the durable budget limit")
        return self


class ArenaCompletionRecord(_StrictModel):
    schema_version: Literal["robomex.arena_completion.v1"] = "robomex.arena_completion.v1"
    record_kind: Literal["completion"] = "completion"
    sequence: int = Field(ge=1)
    arena_run_id: NonEmptyStr
    reservation_sequence: int = Field(ge=1)
    considered_candidate_ids: tuple[NonEmptyStr, ...]
    selected_candidate_id: str | None = None
    promotion_action_digest: (
        Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")] | None
    ) = None
    result_payload: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _bind_completion(self) -> ArenaCompletionRecord:
        if len(set(self.considered_candidate_ids)) != len(self.considered_candidate_ids):
            raise ValueError("considered candidate IDs must be unique")
        if self.selected_candidate_id is not None:
            if self.selected_candidate_id not in self.considered_candidate_ids:
                raise ValueError("selected candidate was not considered")
            if self.promotion_action_digest is None:
                raise ValueError("selected candidate requires promotion action digest")
        elif self.promotion_action_digest is not None:
            raise ValueError("unselected completion cannot carry a promotion digest")
        return self


class ArenaConsumptionLedger:
    """Append-only reserve-before-run ledger for IDs and candidate budgets.

    A reservation is durable before the first proposal actor is spawned.  A
    crash therefore leaves a consumed run ID and consumed candidate quota; it
    is never interpreted as permission to replay proposal work.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.lock_path = (
            self.path.with_name(self.path.name + ".lock") if self.path is not None else None
        )
        self._lock = threading.RLock()
        self._records: list[ArenaReservationRecord | ArenaCompletionRecord] = []
        self._reservations: dict[str, ArenaReservationRecord] = {}
        self._completions: dict[str, ArenaCompletionRecord] = {}
        self._scope_limits: dict[tuple[str, str, str], int] = {}
        self._scope_used: dict[tuple[str, str, str], int] = {}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)
            assert self.lock_path is not None
            self.lock_path.touch(exist_ok=True)
            with self._disk_lock():
                self._reload_from_disk()

    @property
    def persistent(self) -> bool:
        return self.path is not None

    @property
    def records(self) -> tuple[ArenaReservationRecord | ArenaCompletionRecord, ...]:
        with self._lock, self._disk_lock():
            self._reload_from_disk()
            return tuple(self._records)

    def reserve(
        self,
        *,
        context: ArenaContext,
        requested_candidates: int,
        caller_remaining: int,
        allow_exact_replay: bool = False,
        run_binding_digest: str | None = None,
    ) -> ArenaReservationRecord:
        if requested_candidates < 1:
            raise ValueError("requested_candidates must be positive")
        if caller_remaining < 0:
            raise ValueError("candidate_budget_remaining cannot be negative")
        with self._lock, self._disk_lock():
            self._reload_from_disk()
            previous = self._reservations.get(context.arena_run_id)
            context_digest = self._context_digest(context)
            if previous is not None and allow_exact_replay:
                if (
                    previous.episode_id != context.episode_id
                    or previous.workflow_id != context.workflow_id
                    or previous.graph_id != context.graph_id
                    or previous.graph_revision != context.graph_revision
                    or previous.candidate_budget_id != context.candidate_budget_id
                    or previous.candidate_budget_limit != context.candidate_budget_limit
                    or previous.requested_candidates != requested_candidates
                    or previous.caller_remaining != caller_remaining
                    or previous.context_digest != context_digest
                    or previous.run_binding_digest != run_binding_digest
                ):
                    raise ArenaLedgerIntegrityError(
                        "Arena run ID replay attempted to rebind durable context"
                    )
                return previous
            if previous is not None:
                raise ArenaPolicyError(
                    f"Arena run ID {context.arena_run_id!r} has already been consumed."
                )
            scope = self._scope(context)
            known_limit = self._scope_limits.get(scope)
            if known_limit is not None and known_limit != context.candidate_budget_limit:
                raise ArenaLedgerIntegrityError(
                    "candidate budget limit changed for an existing durable scope"
                )
            used = self._scope_used.get(scope, 0)
            available = max(context.candidate_budget_limit - used, 0)
            reserved = min(requested_candidates, caller_remaining, available)
            record = ArenaReservationRecord(
                sequence=len(self._records) + 1,
                arena_run_id=context.arena_run_id,
                episode_id=context.episode_id,
                workflow_id=context.workflow_id,
                graph_id=context.graph_id,
                graph_revision=context.graph_revision,
                candidate_budget_id=context.candidate_budget_id,
                candidate_budget_limit=context.candidate_budget_limit,
                requested_candidates=requested_candidates,
                caller_remaining=caller_remaining,
                reserved_candidates=reserved,
                context_digest=context_digest,
                run_binding_digest=run_binding_digest,
            )
            self._append(record)
            return record

    def complete(
        self,
        *,
        arena_run_id: str,
        considered_candidate_ids: tuple[str, ...],
        selected_candidate_id: str | None,
        promotion_action_digest: str | None,
        result_payload: Mapping[str, Any] | None = None,
        allow_exact_replay: bool = False,
    ) -> ArenaCompletionRecord:
        with self._lock, self._disk_lock():
            self._reload_from_disk()
            reservation = self._reservations.get(arena_run_id)
            if reservation is None:
                raise ArenaLedgerIntegrityError("Arena completion has no reservation")
            previous = self._completions.get(arena_run_id)
            normalized_result = (
                json.loads(
                    json.dumps(
                        result_payload,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                if result_payload is not None
                else None
            )
            if previous is not None and allow_exact_replay:
                if (
                    previous.considered_candidate_ids != considered_candidate_ids
                    or previous.selected_candidate_id != selected_candidate_id
                    or previous.promotion_action_digest != promotion_action_digest
                    or previous.result_payload != normalized_result
                ):
                    raise ArenaLedgerIntegrityError(
                        "Arena completion replay attempted to rebind durable result"
                    )
                return previous
            if previous is not None:
                raise ArenaLedgerIntegrityError("Arena run already has a completion")
            record = ArenaCompletionRecord(
                sequence=len(self._records) + 1,
                arena_run_id=arena_run_id,
                reservation_sequence=reservation.sequence,
                considered_candidate_ids=considered_candidate_ids,
                selected_candidate_id=selected_candidate_id,
                promotion_action_digest=promotion_action_digest,
                result_payload=normalized_result,
            )
            self._append(record)
            return record

    def completed_result(self, arena_run_id: str) -> Mapping[str, JsonValue] | None:
        """Return the fsynced result checkpoint for graph-command recovery."""

        with self._lock, self._disk_lock():
            self._reload_from_disk()
            completion = self._completions.get(arena_run_id)
            if completion is None or completion.result_payload is None:
                return None
            return json.loads(
                json.dumps(completion.result_payload, sort_keys=True, separators=(",", ":"))
            )

    def _append(self, record: ArenaReservationRecord | ArenaCompletionRecord) -> None:
        payload = json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(payload + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        self._apply(record)

    def _reload_from_disk(self) -> None:
        if self.path is None:
            return
        self._records.clear()
        self._reservations.clear()
        self._completions.clear()
        self._scope_limits.clear()
        self._scope_used.clear()
        self._replay()

    def _replay(self) -> None:
        assert self.path is not None
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ArenaLedgerIntegrityError("Arena ledger is unreadable") from exc
        for sequence, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("record must be an object")
                kind = payload.get("record_kind")
                if kind == "reservation":
                    record = ArenaReservationRecord.model_validate(payload)
                elif kind == "completion":
                    record = ArenaCompletionRecord.model_validate(payload)
                else:
                    raise ValueError(f"unknown record_kind {kind!r}")
            except Exception as exc:
                raise ArenaLedgerIntegrityError(
                    f"invalid Arena ledger record at line {sequence}"
                ) from exc
            if record.sequence != len(self._records) + 1:
                raise ArenaLedgerIntegrityError("Arena ledger sequence is not contiguous")
            self._apply(record)

    @contextmanager
    def _disk_lock(self) -> Iterator[None]:
        if self.lock_path is None:
            yield
            return
        with self.lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _apply(self, record: ArenaReservationRecord | ArenaCompletionRecord) -> None:
        if isinstance(record, ArenaReservationRecord):
            if record.arena_run_id in self._reservations:
                raise ArenaLedgerIntegrityError("duplicate Arena reservation ID")
            scope = (
                record.episode_id,
                record.workflow_id,
                record.candidate_budget_id,
            )
            known_limit = self._scope_limits.get(scope)
            if known_limit is not None and known_limit != record.candidate_budget_limit:
                raise ArenaLedgerIntegrityError("conflicting durable candidate budget limit")
            used = self._scope_used.get(scope, 0) + record.reserved_candidates
            if used > record.candidate_budget_limit:
                raise ArenaLedgerIntegrityError("durable candidate budget was overspent")
            self._scope_limits[scope] = record.candidate_budget_limit
            self._scope_used[scope] = used
            self._reservations[record.arena_run_id] = record
        else:
            reservation = self._reservations.get(record.arena_run_id)
            if reservation is None or reservation.sequence != record.reservation_sequence:
                raise ArenaLedgerIntegrityError("completion references no exact reservation")
            if record.arena_run_id in self._completions:
                raise ArenaLedgerIntegrityError("duplicate Arena completion")
            self._completions[record.arena_run_id] = record
        self._records.append(record)

    @staticmethod
    def _scope(context: ArenaContext) -> tuple[str, str, str]:
        return (
            context.episode_id,
            context.workflow_id,
            context.candidate_budget_id,
        )

    @staticmethod
    def _context_digest(context: ArenaContext) -> str:
        encoded = json.dumps(
            context.model_dump(mode="json"),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class RegisteredShadowBackend:
    backend_id: str
    backend: ActionBackend
    world_resource_bindings: frozenset[tuple[str, str]]

    def __post_init__(self) -> None:
        if not self.backend_id.strip() or not self.world_resource_bindings:
            raise ValueError("shadow backend ID and world/resource bindings are required")
        if not isinstance(self.backend, ActionBackend):
            raise TypeError("shadow backend does not implement ActionBackend")
        if self.backend.descriptor.backend_id != self.backend_id:
            raise ShadowBindingError("shadow registry ID differs from backend descriptor")
        normalized = frozenset(
            (str(world_id).strip(), str(resource_id).strip())
            for world_id, resource_id in self.world_resource_bindings
        )
        if any(not world_id or not resource_id for world_id, resource_id in normalized):
            raise ValueError("shadow world/resource IDs must not be empty")
        object.__setattr__(self, "world_resource_bindings", normalized)


class ShadowWorldProof(_StrictModel):
    backend_id: NonEmptyStr
    world_id: NonEmptyStr
    resource_id: NonEmptyStr
    world_kind: Literal[WorldKind.SHADOW] = WorldKind.SHADOW
    snapshot_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


class ShadowBackendRegistry:
    """Manifest-admitted backend inventory that proves isolated shadow worlds."""

    def __init__(self, bindings: Sequence[RegisteredShadowBackend] = ()) -> None:
        indexed: dict[str, RegisteredShadowBackend] = {}
        for binding in bindings:
            if binding.backend_id in indexed:
                raise ShadowBindingError(f"duplicate shadow backend {binding.backend_id!r}")
            indexed[binding.backend_id] = binding
        self._bindings = MappingProxyType(indexed)

    def prove(self, candidate: ArenaCandidateSpec) -> ShadowWorldProof:
        backend_id = candidate.shadow_backend_id
        resource_id = candidate.shadow_resource_id
        world_id = candidate.profile.isolation.world_id
        if backend_id is None or resource_id is None or world_id is None:
            raise ShadowBindingError(
                "shadow candidate requires backend, resource, and isolated world IDs"
            )
        binding = self._bindings.get(backend_id)
        if binding is None:
            raise ShadowBindingError(
                f"shadow backend {backend_id!r} is not in the runtime registry"
            )
        if (world_id, resource_id) not in binding.world_resource_bindings:
            raise ShadowBindingError(
                f"world/resource {(world_id, resource_id)!r} is not admitted for "
                f"shadow backend {backend_id!r}"
            )
        if binding.backend.descriptor.backend_id != backend_id:
            raise ShadowBindingError("shadow backend descriptor changed after registration")
        snapshot = AdmissionSnapshot.model_validate(
            binding.backend.snapshot(world_id, resource_id).model_dump(mode="python")
        )
        if (
            snapshot.world_id != world_id
            or snapshot.resource_id != resource_id
            or snapshot.world_kind is not WorldKind.SHADOW
        ):
            raise ShadowBindingError(
                "runtime backend snapshot does not prove the declared shadow world"
            )
        return ShadowWorldProof(
            backend_id=backend_id,
            world_id=world_id,
            resource_id=resource_id,
            snapshot_digest=canonical_model_digest(snapshot),
        )

    def rollout(
        self,
        *,
        candidate: ArenaCandidateSpec,
        hypothesis: ActionHypothesis,
        artifacts: ArtifactLookup,
    ) -> ShadowRolloutReceipt:
        """Replay one exact proposed joint path in its isolated shadow clone.

        The candidate proposes an authoritative-world plan because that is the
        artifact eventually checked for promotion.  The runtime—not the
        candidate—rebinds the same exact joint path to a fresh, registered
        shadow snapshot.  No authoritative backend is reachable here.
        """

        proof = self.prove(candidate)
        binding = self._bindings[proof.backend_id]
        artifact = artifacts.resolve(hypothesis.plan_ref)
        if artifact.schema != "robomex.motion_plan.v2":
            raise ShadowBindingError("shadow rollout requires a sealed MotionPlan")
        parsed = validate_action_spec(artifact.payload)
        if not isinstance(parsed, MotionPlan):
            raise ShadowBindingError("shadow rollout accepts MotionPlan artifacts only")
        snapshot = AdmissionSnapshot.model_validate(
            binding.backend.snapshot(proof.world_id, proof.resource_id).model_dump(mode="python")
        )
        if snapshot.world_kind is not WorldKind.SHADOW:
            raise ShadowBindingError("shadow backend returned an authoritative snapshot")
        if snapshot.joint_names != parsed.motion.joint_names:
            raise ShadowBindingError("shadow clone joint order differs from proposed plan")
        if snapshot.config_digest != parsed.expected_snapshot.config_digest:
            raise ShadowBindingError("shadow clone configuration differs from proposal")
        payload = parsed.model_dump(mode="json")
        payload["expected_snapshot"] = snapshot.model_dump(mode="json")
        payload["plan_id"] = f"{parsed.plan_id}.shadow.{candidate.candidate_id}"
        payload["content_digest"] = ""
        shadow_plan = MotionPlan.model_validate(payload)
        return ShadowRolloutRunner(binding.backend).run(
            shadow_plan, candidate_id=candidate.candidate_id
        )


class ArenaPolicy(_StrictModel):
    max_candidates: int = Field(default=3, ge=1, le=8)
    disagreement_distance_m: float = Field(default=0.025, ge=0.0, allow_inf_nan=False)


class ArenaCandidateResult(_StrictModel):
    candidate_id: NonEmptyStr
    actor_id: NonEmptyStr
    strategy: NonEmptyStr
    status: CandidateStatus
    provider_entered: bool = False
    namespace_id: str | None = None
    workspace_id: str | None = None
    world_id: str | None = None
    shadow_world_proof: ShadowWorldProof | None = None
    shadow_rollout_receipt: ShadowRolloutReceipt | None = None
    hypothesis: ActionHypothesis | None = None
    motion_safety: MotionSafetyEvaluation | None = None
    gate_results: tuple[HardGateResult, ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def _closed_candidate_status(self) -> ArenaCandidateResult:
        if self.motion_safety is not None and self.hypothesis is not None:
            if self.motion_safety.candidate_id != self.candidate_id:
                raise ValueError("motion safety evaluation belongs to another candidate")
            if self.motion_safety.action_spec_ref != self.hypothesis.plan_ref:
                raise ValueError("motion safety evaluation changed the proposed action ref")
        if self.shadow_rollout_receipt is not None:
            if self.shadow_rollout_receipt.candidate_id != self.candidate_id:
                raise ValueError("shadow rollout receipt belongs to another candidate")
            if self.shadow_rollout_receipt.receipt_authority != "shadow_only":
                raise ValueError("Arena may contain only shadow-only rollout receipts")
        if self.status is CandidateStatus.ACCEPTED and (
            self.hypothesis is None
            or self.motion_safety is None
            or not self.motion_safety.eligible
            or not self.gate_results
            or not all(item.passed for item in self.gate_results)
        ):
            raise ValueError("accepted candidate requires every runtime hard gate")
        return self


class ArenaHypothesisSet(_StrictModel):
    """One graph-bindable artifact containing every materialized hypothesis."""

    schema_version: Literal["robomex.arena_hypotheses.v1"] = "robomex.arena_hypotheses.v1"
    arena_run_id: NonEmptyStr
    graph_id: NonEmptyStr
    graph_revision: int = Field(ge=1)
    hypotheses: tuple[ActionHypothesis, ...] = ()

    @model_validator(mode="after")
    def _unique_candidates(self) -> ArenaHypothesisSet:
        candidate_ids = [item.candidate_id for item in self.hypotheses]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Arena hypothesis set repeats a candidate ID")
        return self


class ArenaResult(_StrictModel):
    schema_version: Literal["robomex.arena_result.v1"] = "robomex.arena_result.v1"
    arena_run_id: NonEmptyStr
    graph_id: NonEmptyStr
    graph_revision: int = Field(ge=1)
    risk_report: RiskReport
    consumption_reservation_sequence: int = Field(ge=1)
    requested_candidates: int = Field(ge=1)
    candidate_budget_used: int = Field(ge=0)
    candidate_results: tuple[ArenaCandidateResult, ...]
    quota_exhausted: bool = False
    selected_candidate_id: str | None = None
    selected_hypothesis: ActionHypothesis | None = None
    selected_action_spec_ref: ResolvedArtifactRef | None = None
    promotion_receipt: PromotionReceipt | None = None
    selection_reason: str = ""
    disagreement: bool = False
    manager_signals: tuple[ManagerSignal, ...] = ()
    roster_updates: tuple[RosterUpdate, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _upgrade_zero_reservation_result(cls, value: Any) -> Any:
        """Recover pre-field v1 checkpoints without changing their schema ID.

        A completed Arena result with a positive request and no consumed or
        materialized candidate can only have come from a zero-sized durable
        reservation.  Inferring the new flag from those already-durable fields
        keeps crash recovery deterministic while all newly serialized results
        carry the signal explicitly.
        """

        if not isinstance(value, Mapping) or "quota_exhausted" in value:
            return value
        if (
            value.get("requested_candidates", 0) > 0
            and value.get("candidate_budget_used") == 0
            and not value.get("candidate_results")
            and value.get("selected_candidate_id") is None
        ):
            return {**value, "quota_exhausted": True}
        return value

    @model_validator(mode="after")
    def _bind_single_promotion(self) -> ArenaResult:
        selected_fields = (
            self.selected_candidate_id,
            self.selected_hypothesis,
            self.selected_action_spec_ref,
            self.promotion_receipt,
        )
        if self.quota_exhausted:
            if self.candidate_budget_used != 0 or self.candidate_results:
                raise ValueError(
                    "quota-exhausted Arena result cannot contain consumed candidates"
                )
            if self.selected_candidate_id is not None:
                raise ValueError("quota-exhausted Arena result cannot select a candidate")
            if not self.selection_reason.strip():
                raise ValueError("quota-exhausted Arena result requires a selection reason")
        elif self.candidate_budget_used == 0:
            raise ValueError(
                "zero-candidate Arena result must be marked quota_exhausted"
            )
        if self.selected_candidate_id is None:
            if any(value is not None for value in selected_fields[1:]):
                raise ValueError("an unselected Arena result cannot contain a promotion")
            return self
        if any(value is None for value in selected_fields[1:]):
            raise ValueError("a selected Arena result requires one promotion receipt")
        assert self.selected_hypothesis is not None
        assert self.selected_action_spec_ref is not None
        assert self.promotion_receipt is not None
        if self.selected_hypothesis.candidate_id != self.selected_candidate_id:
            raise ValueError("selected hypothesis belongs to another candidate")
        if self.selected_hypothesis.plan_ref != self.selected_action_spec_ref:
            raise ValueError("selected action ref differs from selected hypothesis")
        if self.promotion_receipt.candidate_id != self.selected_candidate_id:
            raise ValueError("promotion receipt belongs to another candidate")
        if (
            self.promotion_receipt.graph_id != self.graph_id
            or self.promotion_receipt.graph_revision != self.graph_revision
        ):
            raise ValueError("promotion receipt changed the selected graph revision")
        if self.promotion_receipt.action_spec_ref != self.selected_action_spec_ref:
            raise ValueError("promotion receipt changed the selected action ref")
        return self


class SwarmArena:
    """Effect-isolated candidate swarm with runtime-owned motion promotion."""

    def __init__(
        self,
        registry: ActorRegistry,
        *,
        promotion_authority: RuntimeMotionPromotionAuthority,
        context_guard: RuntimeArenaContextGuard,
        consumption_ledger: ArenaConsumptionLedger,
        shadow_backends: ShadowBackendRegistry,
        gates: Sequence[HypothesisHardGate],
        adapters: Mapping[str, HypothesisAdapter] | None = None,
        policy: ArenaPolicy | None = None,
    ) -> None:
        self._registry = registry
        if not isinstance(promotion_authority, RuntimeMotionPromotionAuthority):
            raise TypeError("Arena requires the concrete runtime-owned motion promotion authority.")
        self._promotion_authority = promotion_authority
        if not isinstance(context_guard, RuntimeArenaContextGuard):
            raise TypeError("Arena requires the concrete runtime graph context guard.")
        if not isinstance(consumption_ledger, ArenaConsumptionLedger):
            raise TypeError("Arena requires an append-only consumption ledger.")
        if not isinstance(shadow_backends, ShadowBackendRegistry):
            raise TypeError("Arena requires the runtime shadow backend registry.")
        self._context_guard = context_guard
        self._consumption_ledger = consumption_ledger
        self._shadow_backends = shadow_backends
        self._gates = tuple(gates)
        self._adapters = dict(adapters or {"mapping_v1": MappingHypothesisAdapter()})
        self._policy = policy or ArenaPolicy()

    def assert_bound_to(
        self,
        *,
        episode_id: str,
        artifacts: ArtifactLookup,
        context_guard_token: object,
        consumption_ledger: ArenaConsumptionLedger,
        shadow_backends: ShadowBackendRegistry,
    ) -> None:
        """Fail unless promotion resolves through the caller's exact data plane."""

        self._promotion_authority.assert_bound_to(episode_id=episode_id, artifacts=artifacts)
        self._context_guard.assert_bound_to(
            episode_id=episode_id, binding_token=context_guard_token
        )
        if self._consumption_ledger is not consumption_ledger:
            raise ArenaPolicyError("Arena consumption ledger is not owned by this EpisodeRuntime.")
        if self._shadow_backends is not shadow_backends:
            raise ArenaPolicyError("Arena shadow registry is not owned by this EpisodeRuntime.")
        root = getattr(artifacts, "episode_root", None)
        ledger_path = self._consumption_ledger.path
        if root is None or ledger_path is None:
            raise ArenaPolicyError(
                "Episode Arena requires a persistent episode-local consumption ledger."
            )
        try:
            ledger_path.resolve().relative_to(Path(root).resolve())
        except ValueError as exc:
            raise ArenaPolicyError("Arena consumption ledger is outside the Episode root.") from exc

    def run(
        self,
        *,
        context: ArenaContext,
        risk_report: RiskReport,
        candidates: Sequence[ArenaCandidateSpec],
        candidate_budget_remaining: int,
        recovery_safe: bool = False,
        run_binding_digest: str | None = None,
        deadline_monotonic_s: float | None = None,
    ) -> ArenaResult:
        """Evaluate up to the risk- and budget-bounded candidate count."""

        self._context_guard.assert_current(context)
        if deadline_monotonic_s is not None:
            deadline_monotonic_s = float(deadline_monotonic_s)
            if not math.isfinite(deadline_monotonic_s):
                raise ValueError("deadline_monotonic_s must be finite")
        ids = [candidate.candidate_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ArenaPolicyError("Arena candidate IDs must be unique.")

        requested, _potential, _shadow = self.budget_request(
            risk_report=risk_report,
            candidates=candidates,
            candidate_budget_remaining=candidate_budget_remaining,
        )
        reservation = self._consumption_ledger.reserve(
            context=context,
            requested_candidates=requested,
            caller_remaining=min(candidate_budget_remaining, len(candidates)),
            allow_exact_replay=recovery_safe,
            run_binding_digest=run_binding_digest,
        )
        if recovery_safe:
            checkpoint = self._consumption_ledger.completed_result(context.arena_run_id)
            if checkpoint is not None:
                recovered = ArenaResult.model_validate(checkpoint)
                if (
                    recovered.arena_run_id != context.arena_run_id
                    or recovered.graph_id != context.graph_id
                    or recovered.graph_revision != context.graph_revision
                    or recovered.risk_report != risk_report
                ):
                    raise ArenaLedgerIntegrityError(
                        "Durable Arena result is not bound to the replay request"
                    )
                self._context_guard.assert_current(context)
                return recovered
        selected_specs = tuple(candidates[: reservation.reserved_candidates])
        quota_exhausted = reservation.reserved_candidates == 0
        self._validate_shadow_world_isolation(selected_specs)

        records: list[ArenaCandidateResult] = []
        roster_updates: list[RosterUpdate] = []
        for candidate in selected_specs:
            actor_id = f"{context.arena_run_id}-{candidate.candidate_id}"
            handle = None
            spawned = False
            provider_entered = False
            gate_results: tuple[HardGateResult, ...] = ()
            hypothesis: ActionHypothesis | None = None
            motion_safety: MotionSafetyEvaluation | None = None
            shadow_world_proof: ShadowWorldProof | None = None
            shadow_rollout_receipt: ShadowRolloutReceipt | None = None
            try:
                self._validate_candidate(candidate)
                candidate_deadline = self._candidate_deadline_monotonic_s(
                    candidate,
                    run_deadline_monotonic_s=deadline_monotonic_s,
                )
                if (
                    candidate_deadline is not None
                    and time.monotonic() >= candidate_deadline
                ):
                    raise TimeoutError(
                        f"Arena candidate {candidate.candidate_id!r} expired before provider entry"
                    )
                if candidate.effect_scope is EffectScope.SHADOW_WORLD:
                    shadow_world_proof = self._shadow_backends.prove(candidate)
                adapter = self._adapters.get(candidate.adapter_id)
                if adapter is None:
                    raise ArenaPolicyError(
                        f"Unknown cataloged hypothesis adapter {candidate.adapter_id!r}."
                    )
                handle = self._registry.spawn(candidate.profile, actor_id=actor_id)
                spawned = True
                roster_updates.append(
                    self._roster_event(
                        context,
                        operation=RosterOperation.SPAWN,
                        actor_id=actor_id,
                        profile_id=candidate.profile.profile_id,
                        reason=f"arena candidate spawned: {candidate.strategy}",
                    )
                )
                # Crossing AgentHandle.invoke may reach an opaque provider.  A
                # failure from this point is conservatively charged in full.
                provider_entered = True
                arena_runtime_context = self._candidate_runtime_context(
                    context=context,
                    candidate=candidate,
                )
                raw = handle.invoke(
                    InvocationSpec(
                        invocation_id=f"{context.arena_run_id}-{candidate.candidate_id}-invoke",
                        objective=candidate.objective,
                        inputs=candidate.inputs,
                        output_contract=candidate.output_contract,
                        requested_capabilities=candidate.requested_capabilities,
                        requested_effects=candidate.requested_effects,
                        budget={
                            name: float(value)
                            for name, value in candidate.estimated_budget.model_dump(
                                mode="python"
                            ).items()
                        },
                        deadline_monotonic_s=candidate_deadline,
                        idempotency_key=f"{context.arena_run_id}-{candidate.candidate_id}",
                        metadata={
                            **dict(candidate.invocation_metadata),
                            ARENA_RUNTIME_CONTEXT_METADATA_KEY: (
                                arena_runtime_context.model_dump(mode="json")
                            ),
                        },
                    )
                )
                hypothesis = adapter.adapt(raw, candidate)
                motion_safety = self._promotion_authority.evaluate(
                    context=context, hypothesis=hypothesis
                )
                evaluated: list[HardGateResult] = list(motion_safety.gate_results)
                if candidate.effect_scope is EffectScope.SHADOW_WORLD:
                    shadow_rollout_receipt = self._shadow_backends.rollout(
                        candidate=candidate,
                        hypothesis=hypothesis,
                        artifacts=self._promotion_authority.artifacts,
                    )
                    shadow_passed = (
                        shadow_rollout_receipt.status is ShadowRolloutStatus.COMPLETED
                        and shadow_rollout_receipt.converged is not False
                    )
                    evaluated.append(
                        HardGateResult(
                            gate_id="shadow_rollout",
                            passed=shadow_passed,
                            reason=(
                                ""
                                if shadow_passed
                                else shadow_rollout_receipt.reason
                                or "shadow rollout did not converge"
                            ),
                        )
                    )
                for gate in self._gates:
                    try:
                        evaluated.append(gate.evaluate(hypothesis))
                    except Exception as exc:
                        evaluated.append(
                            HardGateResult(
                                gate_id=gate.gate_id,
                                passed=False,
                                reason=f"gate error: {type(exc).__name__}: {exc}",
                            )
                        )
                gate_results = tuple(evaluated)
                status = (
                    CandidateStatus.ACCEPTED
                    if all(result.passed for result in gate_results)
                    else CandidateStatus.REJECTED
                )
                records.append(
                    ArenaCandidateResult(
                        candidate_id=candidate.candidate_id,
                        actor_id=actor_id,
                        strategy=candidate.strategy,
                        status=status,
                        provider_entered=provider_entered,
                        namespace_id=handle.isolation.namespace_id,
                        workspace_id=handle.isolation.workspace_id,
                        world_id=handle.isolation.world_id,
                        shadow_world_proof=shadow_world_proof,
                        shadow_rollout_receipt=shadow_rollout_receipt,
                        hypothesis=hypothesis,
                        motion_safety=motion_safety,
                        gate_results=gate_results,
                    )
                )
            except Exception as exc:
                records.append(
                    ArenaCandidateResult(
                        candidate_id=candidate.candidate_id,
                        actor_id=actor_id,
                        strategy=candidate.strategy,
                        status=CandidateStatus.FAILED,
                        provider_entered=provider_entered,
                        namespace_id=(handle.isolation.namespace_id if handle else None),
                        workspace_id=(handle.isolation.workspace_id if handle else None),
                        world_id=(handle.isolation.world_id if handle else None),
                        shadow_world_proof=shadow_world_proof,
                        shadow_rollout_receipt=shadow_rollout_receipt,
                        hypothesis=hypothesis,
                        motion_safety=motion_safety,
                        gate_results=gate_results,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            finally:
                if handle is not None:
                    if handle.state is not ActorState.RETIRED:
                        handle.retire()
                    if spawned and handle.state is ActorState.RETIRED:
                        roster_updates.append(
                            self._roster_event(
                                context,
                                operation=RosterOperation.RETIRE,
                                actor_id=actor_id,
                                profile_id=candidate.profile.profile_id,
                                reason="arena proposal worker completed",
                            )
                        )

        accepted = [
            record
            for record in records
            if record.status is CandidateStatus.ACCEPTED and record.hypothesis is not None
        ]
        disagreement = self._has_disagreement(accepted)
        ranked = sorted(accepted, key=self._rank_key)
        selected = ranked[0] if ranked else None
        # Candidate execution can overlap manager graph repair.  Recheck the
        # runtime-owned graph identity after exploration/ranking and immediately
        # before a promotion receipt can be created.
        self._context_guard.assert_current(context)
        promotion_receipt = (
            self._promotion_receipt(context, selected) if selected is not None else None
        )

        signals: list[ManagerSignal] = []
        if risk_report.level is RiskLevel.HIGH and risk_report.recommended_candidates > 1:
            signals.append(ManagerSignal.RISK_EXPANSION)
        if not accepted and not quota_exhausted:
            signals.append(ManagerSignal.ALL_CANDIDATES_REJECTED)
        elif disagreement:
            signals.append(ManagerSignal.DISAGREEMENT)

        result = ArenaResult(
            arena_run_id=context.arena_run_id,
            graph_id=context.graph_id,
            graph_revision=context.graph_revision,
            risk_report=risk_report,
            consumption_reservation_sequence=reservation.sequence,
            requested_candidates=requested,
            candidate_budget_used=len(selected_specs),
            candidate_results=tuple(records),
            quota_exhausted=quota_exhausted,
            selected_candidate_id=(selected.candidate_id if selected else None),
            selected_hypothesis=(selected.hypothesis if selected else None),
            selected_action_spec_ref=(
                promotion_receipt.action_spec_ref if promotion_receipt is not None else None
            ),
            promotion_receipt=promotion_receipt,
            selection_reason=(
                "deterministic rank: utility desc, risk asc, clearance desc, "
                "path length asc, candidate_id asc"
                if selected
                else (
                    "durable candidate quota exhausted: reservation admitted "
                    f"0 of {requested} requested candidates"
                    if quota_exhausted
                    else "all candidate hypotheses failed or were rejected by hard gates"
                )
            ),
            disagreement=disagreement,
            manager_signals=tuple(signals),
            roster_updates=tuple(roster_updates),
        )
        self._consumption_ledger.complete(
            arena_run_id=context.arena_run_id,
            considered_candidate_ids=tuple(item.candidate_id for item in records),
            selected_candidate_id=result.selected_candidate_id,
            promotion_action_digest=(
                result.promotion_receipt.action_spec_digest
                if result.promotion_receipt is not None
                else None
            ),
            result_payload=result.model_dump(mode="json"),
            allow_exact_replay=recovery_safe,
        )
        return result

    def budget_request(
        self,
        *,
        risk_report: RiskReport,
        candidates: Sequence[ArenaCandidateSpec],
        candidate_budget_remaining: int,
    ) -> tuple[int, int, int]:
        """Return requested, potential candidate, and shadow-rollout counts.

        The runtime uses this pure preview to reserve its sealed run-wide
        budget before ``run`` can touch a candidate provider or shadow backend.
        Arena's own scoped ledger may admit fewer candidates; completion then
        settles the run-wide reservation to the actual count.
        """

        if candidate_budget_remaining < 0:
            raise ValueError("candidate_budget_remaining cannot be negative")
        requested = 1 if risk_report.level is RiskLevel.LOW else risk_report.recommended_candidates
        requested = min(requested, self._policy.max_candidates)
        potential = min(requested, candidate_budget_remaining, len(candidates))
        shadow = sum(
            candidate.effect_scope is EffectScope.SHADOW_WORLD
            for candidate in candidates[:potential]
        )
        return requested, potential, shadow

    @staticmethod
    def _candidate_deadline_monotonic_s(
        candidate: ArenaCandidateSpec,
        *,
        run_deadline_monotonic_s: float | None,
    ) -> float | None:
        """Clamp a candidate to both its typed wall grant and the run deadline."""

        local_deadline = (
            time.monotonic() + candidate.estimated_budget.wall_time_ms / 1000.0
            if candidate.estimated_budget.wall_time_ms > 0
            else None
        )
        if run_deadline_monotonic_s is None:
            return local_deadline
        if local_deadline is None:
            return run_deadline_monotonic_s
        return min(local_deadline, run_deadline_monotonic_s)

    @staticmethod
    def _candidate_runtime_context(
        *,
        context: ArenaContext,
        candidate: ArenaCandidateSpec,
    ) -> ArenaRuntimeContextV1:
        if set(candidate.invocation_metadata) & _ARENA_RESERVED_METADATA_KEYS:
            raise ArenaPolicyError(
                "Arena candidate template attempted to override runtime metadata"
            )
        risk_ref: ResolvedArtifactRef | None = None
        if "risk" in candidate.inputs:
            try:
                risk_ref = ResolvedArtifactRef.from_any(candidate.inputs["risk"])
            except (TypeError, ValueError) as exc:
                raise ArenaPolicyError(
                    "Arena candidate risk input is not an exact artifact ref"
                ) from exc
        context_refs: dict[str, ResolvedArtifactRef] = {}
        for name in candidate.context_input_schemas:
            if name not in candidate.inputs:
                raise ArenaPolicyError(
                    f"Arena candidate is missing admitted context input {name!r}"
                )
            try:
                context_refs[name] = ResolvedArtifactRef.from_any(candidate.inputs[name])
            except (TypeError, ValueError) as exc:
                raise ArenaPolicyError(
                    f"Arena candidate context input {name!r} is not an exact artifact ref"
                ) from exc
        node_params = (
            _profile_coding_node_params(
                candidate.profile,
                candidate.coding_activation_id,
            )
            if candidate.coding_activation_id is not None
            else {}
        )
        return ArenaRuntimeContextV1(
            episode_id=context.episode_id,
            workflow_id=context.workflow_id,
            graph_id=context.graph_id,
            graph_revision=context.graph_revision,
            graph_digest=context.graph_digest,
            command_attempt=context.command_attempt,
            slot_id=context.slot_id,
            arena_run_id=context.arena_run_id,
            candidate_id=candidate.candidate_id,
            strategy=candidate.strategy,
            snapshot_ref=context.snapshot_ref,
            risk_ref=risk_ref,
            expected_frame=context.expected_frame,
            world_id=context.world_id,
            resource_id=context.resource_id,
            robot_model_digest=context.robot_model_digest,
            config_digest=context.config_digest,
            coding_activation_id=candidate.coding_activation_id,
            coding_node_params=node_params,
            context_input_schemas=dict(candidate.context_input_schemas),
            context_input_refs=context_refs,
            hypothesis_config=candidate.hypothesis_config,
            preview_config=candidate.preview_config,
        )

    @staticmethod
    def _promotion_receipt(
        context: ArenaContext, selected: ArenaCandidateResult
    ) -> PromotionReceipt:
        safety = selected.motion_safety
        if safety is None or not safety.eligible:
            raise ArenaPolicyError("selected hypothesis has no passing motion safety evaluation")
        certificate = safety.feasibility_certificate
        if (
            certificate is None
            or safety.action_spec_digest is None
            or safety.world_id is None
            or safety.resource_id is None
            or safety.robot_model_digest is None
            or safety.config_digest is None
        ):
            raise ArenaPolicyError("passing safety evaluation is missing promotion bindings")
        return PromotionReceipt(
            arena_run_id=context.arena_run_id,
            candidate_id=selected.candidate_id,
            graph_id=context.graph_id,
            graph_revision=context.graph_revision,
            action_spec_ref=safety.action_spec_ref,
            snapshot_ref=safety.snapshot_ref,
            action_spec_digest=safety.action_spec_digest,
            world_id=safety.world_id,
            resource_id=safety.resource_id,
            robot_model_digest=safety.robot_model_digest,
            config_digest=safety.config_digest,
            feasibility_certificate=certificate,
            gate_results=selected.gate_results,
        )

    @staticmethod
    def _validate_candidate(candidate: ArenaCandidateSpec) -> None:
        if candidate.profile.lifecycle is not ActorLifecycle.EPHEMERAL:
            raise ArenaPolicyError("Arena proposal candidates must be ephemeral workers.")
        if candidate.effect_scope is EffectScope.AUTHORITATIVE_WORLD:
            raise ArenaPolicyError("Arena candidates cannot use authoritative_world effects.")
        illegal_ceiling = candidate.profile.effect_ceiling - _CANDIDATE_EFFECTS
        illegal_request = candidate.requested_effects - _CANDIDATE_EFFECTS
        if illegal_ceiling or illegal_request:
            raise ArenaPolicyError(
                "Arena candidate declares effects outside read_only/shadow_world: "
                f"{sorted(illegal_ceiling | illegal_request)!r}."
            )
        if candidate.effect_scope is EffectScope.READ_ONLY and candidate.requested_effects:
            raise ArenaPolicyError("A read_only candidate cannot request write effects.")
        if candidate.effect_scope is EffectScope.READ_ONLY and (
            candidate.shadow_backend_id is not None or candidate.shadow_resource_id is not None
        ):
            raise ArenaPolicyError("A read_only candidate cannot claim a shadow backend.")
        if candidate.effect_scope is EffectScope.SHADOW_WORLD:
            if "shadow_world.write" not in candidate.requested_effects:
                raise ArenaPolicyError(
                    "A shadow_world candidate must explicitly request shadow_world.write."
                )
            if candidate.profile.isolation.world_id is None:
                raise ArenaPolicyError("A shadow_world candidate must name an isolated world_id.")
            if candidate.shadow_backend_id is None or candidate.shadow_resource_id is None:
                raise ArenaPolicyError(
                    "A shadow_world candidate must bind a registered backend/resource."
                )

    @staticmethod
    def _validate_shadow_world_isolation(
        candidates: Sequence[ArenaCandidateSpec],
    ) -> None:
        world_ids = [
            candidate.profile.isolation.world_id
            for candidate in candidates
            if candidate.effect_scope is EffectScope.SHADOW_WORLD
            and candidate.profile.isolation.world_id is not None
        ]
        if len(world_ids) != len(set(world_ids)):
            raise ArenaPolicyError(
                "Concurrent Arena shadow candidates must use distinct world_id values."
            )

    def _has_disagreement(self, accepted: Sequence[ArenaCandidateResult]) -> bool:
        positions = [
            record.hypothesis.terminal_position_m
            for record in accepted
            if record.hypothesis is not None and record.hypothesis.terminal_position_m is not None
        ]
        for left, right in combinations(positions, 2):
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))
            if distance > self._policy.disagreement_distance_m:
                return True
        return False

    @staticmethod
    def _rank_key(record: ArenaCandidateResult) -> tuple[float, float, float, float, str]:
        hypothesis = record.hypothesis
        assert hypothesis is not None
        clearance = hypothesis.clearance_m if hypothesis.clearance_m is not None else -math.inf
        path_length = hypothesis.path_length if hypothesis.path_length is not None else math.inf
        return (
            -hypothesis.utility,
            hypothesis.estimated_risk,
            -clearance,
            path_length,
            record.candidate_id,
        )

    @staticmethod
    def _roster_event(
        context: ArenaContext,
        *,
        operation: RosterOperation,
        actor_id: str,
        profile_id: str,
        reason: str,
    ) -> RosterUpdate:
        return RosterUpdate(
            episode_id=context.episode_id,
            workflow_id=context.workflow_id,
            source=context.source,
            operation=operation,
            actor_id=actor_id,
            slot_id=context.slot_id,
            actor_profile_ref=profile_id,
            reason=reason,
        )


ARENA_SCHEMA_MODELS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "robomex.risk_inputs.v1": RiskInputs,
        "robomex.risk_policy.v1": RiskPolicy,
        "robomex.risk_report.v1": RiskReport,
        "robomex.action_hypothesis.v1": ActionHypothesis,
        "robomex.arena_hypothesis_config.v1": ArenaHypothesisConfig,
        "robomex.arena_preview_config.v1": ArenaPreviewConfig,
        "robomex.arena_runtime_context.v1": ArenaRuntimeContextV1,
        "robomex.motion_preview.v1": ArenaMotionPreviewArtifact,
        "robomex.arena_hypotheses.v1": ArenaHypothesisSet,
        "robomex.motion_safety_evaluation.v1": MotionSafetyEvaluation,
        "robomex.arena_promotion_receipt.v1": PromotionReceipt,
        "robomex.arena_result.v1": ArenaResult,
    }
)


__all__ = [
    "ARENA_SCHEMA_MODELS",
    "ARENA_RUNTIME_CONTEXT_METADATA_KEY",
    "ActionHypothesis",
    "ArtifactLookup",
    "ArenaBinding",
    "ArenaBindingRegistry",
    "ArenaCandidateResult",
    "ArenaCandidateSpec",
    "ArenaCompletionRecord",
    "ArenaConsumptionLedger",
    "ArenaContext",
    "ArenaError",
    "ArenaHypothesisSet",
    "ArenaHypothesisConfig",
    "ArenaLedgerIntegrityError",
    "ArenaMotionPreviewArtifact",
    "ArenaPolicy",
    "ArenaPolicyError",
    "ArenaPreviewConfig",
    "ArenaReservationRecord",
    "ArenaResult",
    "ArenaRuntimeContextV1",
    "ArenaStaleContextError",
    "BasicHypothesisGate",
    "CandidateStatus",
    "CheckStatus",
    "HardGateResult",
    "HypothesisAdapter",
    "HypothesisHardGate",
    "MappingHypothesisAdapter",
    "MetadataStatusGate",
    "MotionSafetyEvaluation",
    "PromotionReceipt",
    "RiskInputs",
    "RiskLevel",
    "RiskPolicy",
    "RiskReport",
    "RegisteredShadowBackend",
    "RuntimeArenaContextGuard",
    "RuntimeMotionPromotionAuthority",
    "ShadowBackendRegistry",
    "ShadowBindingError",
    "ShadowWorldProof",
    "SwarmArena",
]
