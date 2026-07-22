"""Production assembly for the bowl-correction skill-coding Arena.

This is the only application-side factory for correction candidates.  The
protocol owns graph topology and typed ports; this module owns the concrete
proposal workers, provider bridge, physical configuration pins, cumulative
candidate quota, and optional deterministic point-cloud renderer.

Candidates are deliberately read-only.  A shadow rollout is legal only after
an isolated backend has been registered and admitted by the runtime's
``ShadowBackendRegistry``; this factory does not accept a shadow mode or a
backend handle and therefore cannot silently turn a planning worker into a
simulator or physical writer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from robomex.contracts import (
    DigestStr,
    canonical_payload_digest,
    revalidate_sealed,
)
from robomex.contracts.common import SealedContract, StrictContract
from robomex.data import EpisodeDataPlane
from robomex.elastic import EffectScope, RunnerKind
from robomex.orchestration.actors import ActorLifecycle, ActorProfile, AgentProvider
from robomex.orchestration.arena import (
    ActionHypothesis,
    ArenaBinding,
    ArenaCandidateSpec,
    ArenaHypothesisConfig,
    ArenaPolicy,
    ArenaPreviewConfig,
    HardGateResult,
    RiskPolicy,
)
from robomex.orchestration.arena_coding_provider import ArenaCodingAgentProvider
from robomex.orchestration.bootstrap import actor_profile_digest
from robomex.orchestration.coding_provider import (
    CodingNodeConfigV1,
    SkillCodingAgentProvider,
)
from robomex.orchestration.motion_preview import (
    MotionPreviewGeometryProvider,
    PointCloudMotionPreviewRenderer,
    PointCloudPreviewConfig,
)
from robomex.orchestration.risk_provider import MotionRiskProviderConfig
from robomex.protocols.bowl_place import CodingPhaseBudgetConfig
from robomex.protocols.risk_adaptive_bowl_place import (
    CORRECTION_CONTEXT_SCHEMAS,
    DEFAULT_CORRECTION_STRATEGIES,
    BowlCorrectionCandidateStrategy,
    RiskAdaptiveBowlPlaceProtocolConfig,
)

NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4096),
]
RuntimeId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    ),
]

MOTION_PLAN_SCHEMA_ID = "robomex.motion_plan.v2"
REQUIRED_BOWL_ARENA_SKILL_IDS = ("author_sealed_phase_motion",)
_CANDIDATE_SET_SCHEMA = "robomex.bowl_correction_candidate_set.v1"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class BowlArenaFactoryError(RuntimeError):
    """A production Arena dependency does not match its sealed configuration."""


class BowlArenaComponentProvenance(StrictContract):
    """Operator-attested provenance for renderer or geometry implementation."""

    schema_version: Literal["robomex.bowl_arena_component_provenance.v1"] = (
        "robomex.bowl_arena_component_provenance.v1"
    )
    component_id: RuntimeId
    implementation_digest: DigestStr
    configuration_digest: DigestStr
    version: NonEmptyStr


class BowlArenaPreviewRuntimeProvenance(StrictContract):
    """Exact runtime provenance that must match the manifest-side preview pin."""

    schema_version: Literal["robomex.bowl_arena_preview_runtime_provenance.v1"] = (
        "robomex.bowl_arena_preview_runtime_provenance.v1"
    )
    renderer: BowlArenaComponentProvenance
    geometry: BowlArenaComponentProvenance


def pointcloud_preview_configuration_digest(config: PointCloudPreviewConfig) -> str:
    """Canonical digest for the complete deterministic renderer configuration."""

    parsed = PointCloudPreviewConfig.model_validate(config.model_dump(mode="python"))
    return canonical_payload_digest(
        parsed.schema_version,
        parsed.model_dump(mode="json"),
    )


class BowlArenaPointCloudPreviewPin(SealedContract):
    """Manifest-side preview policy and its external implementation provenance."""

    schema_version: Literal["robomex.bowl_arena_pointcloud_preview_pin.v1"] = (
        "robomex.bowl_arena_pointcloud_preview_pin.v1"
    )
    enabled: bool = False
    failure_mode: Literal["fail_closed", "omit"] = "fail_closed"
    renderer_config: PointCloudPreviewConfig | None = None
    provenance: BowlArenaPreviewRuntimeProvenance | None = None

    @model_validator(mode="after")
    def _closed_preview(self) -> BowlArenaPointCloudPreviewPin:
        if not self.enabled:
            if self.renderer_config is not None or self.provenance is not None:
                raise ValueError(
                    "disabled preview cannot carry renderer configuration or provenance"
                )
            return self
        if self.renderer_config is None or self.provenance is None:
            raise ValueError(
                "enabled preview requires point-cloud config and exact runtime provenance"
            )
        if self.provenance.renderer.component_id != self.renderer_config.renderer_id:
            raise ValueError("renderer provenance ID differs from renderer config")
        expected_digest = pointcloud_preview_configuration_digest(self.renderer_config)
        if self.provenance.renderer.configuration_digest != expected_digest:
            raise ValueError(
                "renderer provenance configuration digest does not seal renderer config"
            )
        return self


class BowlMotionPlanningPin(SealedContract):
    """Closed physical/planner identity consumed by every correction candidate."""

    schema_version: Literal["robomex.bowl_motion_planning_pin.v1"] = (
        "robomex.bowl_motion_planning_pin.v1"
    )
    plan_kind: Literal["bounded_correction"] = "bounded_correction"
    robot_model_digest: DigestStr
    robot_config_digest: DigestStr
    expected_frame: NonEmptyStr
    tcp_frame_id: NonEmptyStr
    planner_backend: RuntimeId
    planner_configuration_digest: DigestStr


class BowlCorrectionCandidateConfig(SealedContract):
    """One heterogeneous semantic role with a positive, typed coding grant.

    Ranking priors are intentionally absent.  Until a trusted per-plan scorer
    exists, every candidate receives equal utility/risk/clearance priors and
    differs only in planning objective.  Runtime feasibility, actual sealed
    path length, and optional render evidence then decide among proposals.
    """

    schema_version: Literal["robomex.bowl_correction_candidate_config.v1"] = (
        "robomex.bowl_correction_candidate_config.v1"
    )
    candidate_id: RuntimeId
    strategy: NonEmptyStr
    objective: NonEmptyStr
    preconditions: tuple[NonEmptyStr, ...] = Field(min_length=1)
    budget: CodingPhaseBudgetConfig = Field(default_factory=CodingPhaseBudgetConfig)
    model: str = ""

    @field_validator("preconditions")
    @classmethod
    def _unique_preconditions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("candidate preconditions must be unique")
        return value

    @classmethod
    def from_strategy(
        cls,
        strategy: BowlCorrectionCandidateStrategy,
        *,
        budget: CodingPhaseBudgetConfig,
        model: str = "",
    ) -> BowlCorrectionCandidateConfig:
        parsed = (
            strategy
            if isinstance(strategy, BowlCorrectionCandidateStrategy)
            else BowlCorrectionCandidateStrategy.model_validate(strategy)
        )
        # Do not propagate strategy.utility/estimated_risk/clearance_m.  Those
        # fields preselect a winner before a candidate has authored any plan.
        return cls(
            candidate_id=parsed.candidate_id,
            strategy=parsed.strategy,
            objective=parsed.objective,
            preconditions=parsed.preconditions,
            budget=budget,
            model=model,
        )


class BowlCorrectionArenaFactoryConfig(SealedContract):
    """Complete manifest-pinnable input to the production factory."""

    schema_version: Literal["robomex.bowl_correction_arena_factory_config.v1"] = (
        "robomex.bowl_correction_arena_factory_config.v1"
    )
    graph_id: NonEmptyStr
    graph_revision: int = Field(ge=1)
    protocol_config_digest: DigestStr
    risk_config_digest: DigestStr
    binding_id: RuntimeId
    provider_id: RuntimeId = "bowl_correction_arena_coding"
    candidate_budget_id: RuntimeId
    max_alignment_iterations: int = Field(ge=1)
    total_candidate_budget_limit: int = Field(ge=1)
    candidate_budget_ceiling: CodingPhaseBudgetConfig
    candidates: tuple[BowlCorrectionCandidateConfig, ...] = Field(
        min_length=1,
        max_length=8,
    )
    arena_policy: ArenaPolicy
    risk_policy: RiskPolicy
    planning: BowlMotionPlanningPin
    preview: BowlArenaPointCloudPreviewPin = Field(
        default_factory=BowlArenaPointCloudPreviewPin
    )
    execution_mode: Literal["read_only"] = "read_only"

    @model_validator(mode="after")
    def _closed_candidate_set(self) -> BowlCorrectionArenaFactoryConfig:
        count = len(self.candidates)
        if self.arena_policy.max_candidates != count:
            raise ValueError("ArenaPolicy.max_candidates must equal candidate count K")
        if self.risk_policy.max_candidates != count:
            raise ValueError("RiskPolicy.max_candidates must equal candidate count K")
        candidate_ids = [item.candidate_id for item in self.candidates]
        strategies = [item.strategy for item in self.candidates]
        objectives = [item.objective for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("correction candidate IDs must be unique")
        if len(strategies) != len(set(strategies)):
            raise ValueError("correction candidates must use heterogeneous strategies")
        if len(objectives) != len(set(objectives)):
            raise ValueError("correction candidates must use heterogeneous objectives")
        maximum_total = self.max_alignment_iterations * count
        if self.total_candidate_budget_limit < count:
            raise ValueError("total candidate budget must admit one complete high-risk round")
        if self.total_candidate_budget_limit > maximum_total:
            raise ValueError(
                "total candidate budget exceeds graph loop envelope (iterations * K)"
            )
        ceiling = self.candidate_budget_ceiling
        for candidate in self.candidates:
            budget = candidate.budget
            if (
                budget.model_calls > ceiling.model_calls
                or budget.tokens > ceiling.tokens
                or budget.wall_time_ms > ceiling.wall_time_ms
            ):
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} exceeds graph coding budget ceiling"
                )
        return self

    @classmethod
    def from_protocol(
        cls,
        *,
        protocol_config: RiskAdaptiveBowlPlaceProtocolConfig,
        risk_config: MotionRiskProviderConfig,
        planning: BowlMotionPlanningPin,
        provider_id: str = "bowl_correction_arena_coding",
        strategies: Sequence[BowlCorrectionCandidateStrategy] | None = None,
        candidate_budgets: Sequence[CodingPhaseBudgetConfig] | None = None,
        candidate_models: Sequence[str] | None = None,
        total_candidate_budget_limit: int | None = None,
        arena_policy: ArenaPolicy | None = None,
        preview: BowlArenaPointCloudPreviewPin | None = None,
    ) -> BowlCorrectionArenaFactoryConfig:
        protocol = RiskAdaptiveBowlPlaceProtocolConfig.model_validate(
            protocol_config.model_dump(mode="python")
        )
        risk = MotionRiskProviderConfig.model_validate(
            risk_config.model_dump(mode="python")
        )
        count = protocol.max_arena_candidates
        if risk.risk_policy.max_candidates != count:
            raise ValueError(
                "risk provider max_candidates must equal protocol candidate count K"
            )
        if strategies is None:
            if count > len(DEFAULT_CORRECTION_STRATEGIES):
                raise ValueError(
                    "candidate counts above audited defaults require explicit strategies"
                )
            selected_strategies = DEFAULT_CORRECTION_STRATEGIES[:count]
        else:
            selected_strategies = tuple(
                item
                if isinstance(item, BowlCorrectionCandidateStrategy)
                else BowlCorrectionCandidateStrategy.model_validate(item)
                for item in strategies
            )
        if len(selected_strategies) != count:
            raise ValueError("production Arena requires exactly K candidate strategies")
        if candidate_budgets is None:
            budgets = tuple(protocol.arena_candidate_budget for _ in range(count))
        else:
            budgets = tuple(
                item
                if isinstance(item, CodingPhaseBudgetConfig)
                else CodingPhaseBudgetConfig.model_validate(item)
                for item in candidate_budgets
            )
        if len(budgets) != count:
            raise ValueError("production Arena requires exactly K typed candidate budgets")
        models = (
            tuple("" for _ in range(count))
            if candidate_models is None
            else tuple(candidate_models)
        )
        if len(models) != count:
            raise ValueError("production Arena requires exactly K candidate model entries")
        candidates = tuple(
            BowlCorrectionCandidateConfig.from_strategy(
                strategy,
                budget=budget,
                model=model,
            )
            for strategy, budget, model in zip(
                selected_strategies,
                budgets,
                models,
                strict=True,
            )
        )
        resolved_total = (
            protocol.max_alignment_iterations * count
            if total_candidate_budget_limit is None
            else total_candidate_budget_limit
        )
        return cls(
            graph_id=protocol.graph_id,
            graph_revision=protocol.revision,
            protocol_config_digest=_protocol_config_digest(protocol),
            risk_config_digest=risk.content_digest,
            binding_id=protocol.correction_arena_binding_id,
            provider_id=provider_id,
            candidate_budget_id=f"{protocol.correction_arena_binding_id}.candidate_budget",
            max_alignment_iterations=protocol.max_alignment_iterations,
            total_candidate_budget_limit=resolved_total,
            candidate_budget_ceiling=protocol.arena_candidate_budget,
            candidates=candidates,
            arena_policy=arena_policy or ArenaPolicy(max_candidates=count),
            risk_policy=risk.risk_policy,
            planning=revalidate_sealed(planning),
            preview=(
                revalidate_sealed(preview)
                if preview is not None
                else BowlArenaPointCloudPreviewPin()
            ),
        )


@dataclass(frozen=True)
class BowlMotionConfigurationGate:
    """Fail closed when bridge-authored hypothesis metadata drifts from pins."""

    robot_model_digest: str
    robot_config_digest: str
    expected_frame: str
    tcp_frame_id: str
    planner_backend: str
    planner_configuration_digest: str
    plan_kind: str = "bounded_correction"
    gate_id: str = "bowl_motion_configuration_pin"

    def __post_init__(self) -> None:
        for name in (
            "robot_model_digest",
            "robot_config_digest",
            "planner_configuration_digest",
        ):
            if _DIGEST_RE.fullmatch(str(getattr(self, name))) is None:
                raise ValueError(f"{name} must be canonical sha256")
        for name in (
            "expected_frame",
            "tcp_frame_id",
            "planner_backend",
            "plan_kind",
            "gate_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")

    def evaluate(self, hypothesis: ActionHypothesis) -> HardGateResult:
        expected: Mapping[str, str] = {
            "robot_model_digest": self.robot_model_digest,
            "config_digest": self.robot_config_digest,
            "expected_frame": self.expected_frame,
            "required_tcp_frame_id": self.tcp_frame_id,
            "planner_backend": self.planner_backend,
            "planner_configuration_digest": self.planner_configuration_digest,
            "plan_kind": self.plan_kind,
        }
        mismatches = [
            name
            for name, value in expected.items()
            if hypothesis.metadata.get(name) != value
        ]
        if hypothesis.frame != self.expected_frame:
            mismatches.append("frame")
        return HardGateResult(
            gate_id=self.gate_id,
            passed=not mismatches,
            reason=(
                ""
                if not mismatches
                else "configuration pin mismatch: " + ", ".join(sorted(set(mismatches)))
            ),
        )


@dataclass(frozen=True)
class BowlCorrectionArenaFactoryAssembly:
    """Dependencies ready to merge into the later Swarm application assembly."""

    config: BowlCorrectionArenaFactoryConfig
    providers: Mapping[str, AgentProvider]
    profiles: Mapping[str, ActorProfile]
    profile_digests: Mapping[str, str]
    binding: ArenaBinding
    candidate_config_digest: str
    required_skill_ids: tuple[str, ...]
    renderer: PointCloudMotionPreviewRenderer | None = None
    preview_runtime_provenance: BowlArenaPreviewRuntimeProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "providers", MappingProxyType(dict(self.providers)))
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        object.__setattr__(
            self,
            "profile_digests",
            MappingProxyType(dict(self.profile_digests)),
        )
        if set(self.providers) != {self.config.provider_id}:
            raise ValueError("Arena assembly provider mapping changed its sealed provider ID")
        if set(self.profiles) != set(self.profile_digests):
            raise ValueError("Arena assembly profile digest keys are incomplete")
        if self.binding.binding_id != self.config.binding_id:
            raise ValueError("Arena assembly binding ID drifted from factory config")
        if self.required_skill_ids != REQUIRED_BOWL_ARENA_SKILL_IDS:
            raise ValueError("Arena assembly required skill set drifted")

    @property
    def provider(self) -> ArenaCodingAgentProvider:
        provider = self.providers[self.config.provider_id]
        if not isinstance(provider, ArenaCodingAgentProvider):
            raise BowlArenaFactoryError("sealed Arena provider mapping is corrupt")
        return provider


def build_bowl_correction_arena_factory(
    config: BowlCorrectionArenaFactoryConfig,
    *,
    protocol_config: RiskAdaptiveBowlPlaceProtocolConfig,
    risk_config: MotionRiskProviderConfig,
    coding_provider: SkillCodingAgentProvider,
    data_plane: EpisodeDataPlane | None = None,
    artifacts_root: str | Path | None = None,
    preview_geometry: MotionPreviewGeometryProvider | None = None,
    preview_output_root: str | Path | None = None,
    preview_runtime_provenance: BowlArenaPreviewRuntimeProvenance | None = None,
) -> BowlCorrectionArenaFactoryAssembly:
    """Build the strict skill-coding provider, K profiles, and Arena binding.

    The function revalidates every sealed input from serialized content.  A
    stale ``model_copy`` or object-level mutation with an old digest is
    rejected instead of being trusted as an already-validated Pydantic object.
    """

    if not isinstance(config, BowlCorrectionArenaFactoryConfig):
        config = BowlCorrectionArenaFactoryConfig.model_validate(config)
    sealed = revalidate_sealed(config)
    protocol = RiskAdaptiveBowlPlaceProtocolConfig.model_validate(
        protocol_config.model_dump(mode="python")
    )
    risk = MotionRiskProviderConfig.model_validate(risk_config.model_dump(mode="python"))
    _assert_protocol_and_risk_pins(sealed, protocol=protocol, risk=risk)
    if not isinstance(coding_provider, SkillCodingAgentProvider):
        raise TypeError("coding_provider must be a SkillCodingAgentProvider")
    _assert_coding_provider_ready(coding_provider, sealed.candidates)

    renderer, admitted_preview_provenance = _build_preview_renderer(
        sealed.preview,
        geometry=preview_geometry,
        output_root=preview_output_root,
        runtime_provenance=preview_runtime_provenance,
    )
    bridge = ArenaCodingAgentProvider(
        coding_provider,
        data_plane=data_plane,
        renderer=renderer,
        artifacts_root=artifacts_root,
    )
    arena_preview = ArenaPreviewConfig(
        enabled=sealed.preview.enabled,
        renderer_id=(
            sealed.preview.renderer_config.renderer_id
            if sealed.preview.renderer_config is not None
            else None
        ),
        failure_mode=sealed.preview.failure_mode,
    )
    node_config = _coding_node_config(sealed)
    node_config_json = _canonical_json(node_config)
    candidate_config_digest = _candidate_config_digest(sealed)
    profiles: dict[str, ActorProfile] = {}
    candidates: list[ArenaCandidateSpec] = []
    for candidate_config in sealed.candidates:
        coding_activation_id = f"arena_correction_{candidate_config.candidate_id}"
        profile = _candidate_profile(
            config=sealed,
            candidate=candidate_config,
            candidate_config_digest=candidate_config_digest,
            coding_activation_id=coding_activation_id,
            node_config_json=node_config_json,
        )
        profile_key = f"{sealed.binding_id}.candidate.{candidate_config.candidate_id}"
        profiles[profile_key] = profile
        candidates.append(
            ArenaCandidateSpec(
                candidate_id=candidate_config.candidate_id,
                strategy=candidate_config.strategy,
                profile=profile,
                objective=candidate_config.objective,
                effect_scope=EffectScope.READ_ONLY,
                output_contract={"action_spec": MOTION_PLAN_SCHEMA_ID},
                requested_capabilities=frozenset({"motion.plan"}),
                requested_effects=frozenset(),
                estimated_budget=candidate_config.budget.execution_budget(),
                coding_activation_id=coding_activation_id,
                hypothesis_config=ArenaHypothesisConfig(
                    expected_effect="bounded_alignment_correction",
                    preconditions=candidate_config.preconditions,
                    estimated_risk=0.5,
                    utility=0.0,
                    clearance_m=None,
                    required_tcp_frame_id=sealed.planning.tcp_frame_id,
                ),
                preview_config=arena_preview,
            )
        )
    pin_gate = BowlMotionConfigurationGate(
        robot_model_digest=sealed.planning.robot_model_digest,
        robot_config_digest=sealed.planning.robot_config_digest,
        expected_frame=sealed.planning.expected_frame,
        tcp_frame_id=sealed.planning.tcp_frame_id,
        planner_backend=sealed.planning.planner_backend,
        planner_configuration_digest=sealed.planning.planner_configuration_digest,
        plan_kind=sealed.planning.plan_kind,
    )
    binding = ArenaBinding(
        binding_id=sealed.binding_id,
        candidates=tuple(candidates),
        expected_frame=sealed.planning.expected_frame,
        robot_model_digest=sealed.planning.robot_model_digest,
        candidate_budget_limit=sealed.total_candidate_budget_limit,
        candidate_budget_id=sealed.candidate_budget_id,
        context_input_schemas=CORRECTION_CONTEXT_SCHEMAS,
        gates=(pin_gate,),
        policy=sealed.arena_policy,
        risk_policy=sealed.risk_policy,
    )
    profile_digests = {
        key: actor_profile_digest(profile) for key, profile in profiles.items()
    }
    return BowlCorrectionArenaFactoryAssembly(
        config=sealed,
        providers=MappingProxyType({sealed.provider_id: bridge}),
        profiles=MappingProxyType(profiles),
        profile_digests=MappingProxyType(profile_digests),
        binding=binding,
        candidate_config_digest=candidate_config_digest,
        required_skill_ids=REQUIRED_BOWL_ARENA_SKILL_IDS,
        renderer=renderer,
        preview_runtime_provenance=admitted_preview_provenance,
    )


def _protocol_config_digest(config: RiskAdaptiveBowlPlaceProtocolConfig) -> str:
    return canonical_payload_digest(
        config.schema_version,
        config.model_dump(mode="json"),
    )


def _assert_protocol_and_risk_pins(
    config: BowlCorrectionArenaFactoryConfig,
    *,
    protocol: RiskAdaptiveBowlPlaceProtocolConfig,
    risk: MotionRiskProviderConfig,
) -> None:
    mismatches: list[str] = []
    if config.protocol_config_digest != _protocol_config_digest(protocol):
        mismatches.append("protocol_config_digest")
    if config.risk_config_digest != risk.content_digest:
        mismatches.append("risk_config_digest")
    if config.graph_id != protocol.graph_id:
        mismatches.append("graph_id")
    if config.graph_revision != protocol.revision:
        mismatches.append("graph_revision")
    if config.binding_id != protocol.correction_arena_binding_id:
        mismatches.append("binding_id")
    if config.max_alignment_iterations != protocol.max_alignment_iterations:
        mismatches.append("max_alignment_iterations")
    if config.candidate_budget_ceiling != protocol.arena_candidate_budget:
        mismatches.append("candidate_budget_ceiling")
    if len(config.candidates) != protocol.max_arena_candidates:
        mismatches.append("candidate_count")
    if config.risk_policy != risk.risk_policy:
        mismatches.append("risk_policy")
    if risk.risk_policy.max_candidates != len(config.candidates):
        mismatches.append("risk_max_candidates")
    if mismatches:
        raise BowlArenaFactoryError(
            "factory configuration drifted from graph/risk dependencies: "
            + ", ".join(mismatches)
        )


def _assert_coding_provider_ready(
    provider: SkillCodingAgentProvider,
    candidates: Sequence[BowlCorrectionCandidateConfig],
) -> None:
    for skill_id in REQUIRED_BOWL_ARENA_SKILL_IDS:
        try:
            provider.library.get(skill_id)
        except KeyError as exc:
            raise BowlArenaFactoryError(
                f"coding provider library is missing required skill {skill_id!r}"
            ) from exc
        if skill_id not in provider.trusted_skill_sidecars:
            raise BowlArenaFactoryError(
                f"coding provider has not admitted required sidecar {skill_id!r}"
            )
    for candidate in candidates:
        budget = candidate.budget
        if (
            budget.model_calls > provider.max_model_calls
            or budget.model_calls > provider.max_turns
            or budget.tokens > provider.max_tokens
            or budget.wall_time_ms > provider.max_wall_time_ms
        ):
            raise BowlArenaFactoryError(
                f"coding provider ceiling is below candidate {candidate.candidate_id!r} budget"
            )


def _build_preview_renderer(
    pin: BowlArenaPointCloudPreviewPin,
    *,
    geometry: MotionPreviewGeometryProvider | None,
    output_root: str | Path | None,
    runtime_provenance: BowlArenaPreviewRuntimeProvenance | None,
) -> tuple[
    PointCloudMotionPreviewRenderer | None,
    BowlArenaPreviewRuntimeProvenance | None,
]:
    if not pin.enabled:
        if geometry is not None or output_root is not None or runtime_provenance is not None:
            raise BowlArenaFactoryError(
                "disabled preview rejects geometry, output root, and runtime provenance"
            )
        return None, None
    if geometry is None or output_root is None or runtime_provenance is None:
        raise BowlArenaFactoryError(
            "enabled preview requires geometry, output root, and runtime provenance"
        )
    if not isinstance(geometry, MotionPreviewGeometryProvider):
        raise TypeError("preview_geometry must implement MotionPreviewGeometryProvider")
    runtime = BowlArenaPreviewRuntimeProvenance.model_validate(
        runtime_provenance.model_dump(mode="python")
    )
    if runtime != pin.provenance:
        raise BowlArenaFactoryError(
            "runtime preview provenance differs from manifest-side renderer/geometry pins"
        )
    if str(geometry.provider_id).strip() != runtime.geometry.component_id:
        raise BowlArenaFactoryError(
            "geometry provider ID differs from admitted geometry provenance"
        )
    assert pin.renderer_config is not None
    renderer = PointCloudMotionPreviewRenderer(
        geometry=geometry,
        output_root=output_root,
        config=pin.renderer_config,
    )
    if renderer.renderer_id != runtime.renderer.component_id:
        raise BowlArenaFactoryError(
            "constructed renderer ID differs from admitted renderer provenance"
        )
    return renderer, runtime


def _coding_node_config(config: BowlCorrectionArenaFactoryConfig) -> dict[str, Any]:
    node_config = CodingNodeConfigV1(
        plan_kind=config.planning.plan_kind,
        tcp_frame_id=config.planning.tcp_frame_id,
        planner_backend=config.planning.planner_backend,
        robot_model_digest=config.planning.robot_model_digest,
        planner_configuration_digest=config.planning.planner_configuration_digest,
    )
    return node_config.canonical_mapping()


def _candidate_profile(
    *,
    config: BowlCorrectionArenaFactoryConfig,
    candidate: BowlCorrectionCandidateConfig,
    candidate_config_digest: str,
    coding_activation_id: str,
    node_config_json: str,
) -> ActorProfile:
    suffix = candidate.content_digest.removeprefix("sha256:")[:16]
    return ActorProfile(
        profile_id=f"bowl-arena-{candidate.candidate_id}-{suffix}",
        provider_id=config.provider_id,
        runner_kind=RunnerKind.CODING_WORKER.value,
        lifecycle=ActorLifecycle.EPHEMERAL,
        model=candidate.model,
        capability_ceiling=frozenset({"motion.plan"}),
        effect_ceiling=frozenset(),
        metadata={
            "task_kind": "bounded_correction_planning",
            "preloaded_skills": REQUIRED_BOWL_ARENA_SKILL_IDS,
            "strict_runtime_context": True,
            "node_config_v1": ((coding_activation_id, node_config_json),),
            "max_turns": candidate.budget.model_calls,
            "max_model_calls": candidate.budget.model_calls,
            "max_tokens": candidate.budget.tokens,
            "max_wall_time_ms": candidate.budget.wall_time_ms,
            "arena_binding_id": config.binding_id,
            "arena_factory_config_digest": config.content_digest,
            "candidate_config_digest": candidate.content_digest,
            "candidate_set_digest": candidate_config_digest,
            "candidate_id": candidate.candidate_id,
            "execution_mode": config.execution_mode,
        },
    )


def _candidate_config_digest(config: BowlCorrectionArenaFactoryConfig) -> str:
    return canonical_payload_digest(
        _CANDIDATE_SET_SCHEMA,
        {
            "binding_id": config.binding_id,
            "provider_id": config.provider_id,
            "single_round_k": len(config.candidates),
            "total_candidate_budget_limit": config.total_candidate_budget_limit,
            "candidate_budget_id": config.candidate_budget_id,
            "arena_policy": config.arena_policy.model_dump(mode="json"),
            "risk_policy": config.risk_policy.model_dump(mode="json"),
            "planning_pin_digest": config.planning.content_digest,
            "preview_pin_digest": config.preview.content_digest,
            "candidates": [
                candidate.model_dump(mode="json") for candidate in config.candidates
            ],
        },
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "BowlArenaComponentProvenance",
    "BowlArenaFactoryError",
    "BowlArenaPointCloudPreviewPin",
    "BowlArenaPreviewRuntimeProvenance",
    "BowlCorrectionArenaFactoryAssembly",
    "BowlCorrectionArenaFactoryConfig",
    "BowlCorrectionCandidateConfig",
    "BowlMotionConfigurationGate",
    "BowlMotionPlanningPin",
    "REQUIRED_BOWL_ARENA_SKILL_IDS",
    "build_bowl_correction_arena_factory",
    "pointcloud_preview_configuration_digest",
]
