"""Closed production assembly for the risk-adaptive bowl-place Agent Swarm.

This module is the runnable v2 baseline, not a demo wrapper.  It composes the
complete bowl protocol, deterministic fresh-evidence risk actor, graph-native
one-to-K Arena, shared SkillCoding provider, optional point-cloud preview, and
the fixed application's physical/backend admission into one immutable run.

Evolution remains disabled by :class:`~robomex.evolution.RunManifest`; the
component/configuration digests emitted here are nevertheless stable inputs to
the existing evolve-ready registry and evaluation contracts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from robomex.contracts import canonical_payload_digest
from robomex.core.coder import CompletionPolicy
from robomex.data import SchemaRegistry
from robomex.evolution import ModelPin, RunManifest
from robomex.orchestration.actors import ActorProfile, AgentProvider
from robomex.orchestration.application import RoboMExV2Agent
from robomex.orchestration.arena import ArenaPolicy
from robomex.orchestration.bootstrap import (
    ActionBackendBinding,
    ObservationBackendBinding,
    ShadowBackendBinding,
    V2Application,
    actor_profile_digest,
)
from robomex.orchestration.bowl_application import (
    BOWL_PLACE_SKILL_IDS,
    BowlApplicationAssemblyError,
    FixedBowlPlaceApplication,
    FixedBowlPlaceApplicationConfig,
    TrustedBowlApplicationExtensions,
    build_bowl_skill_coding_provider,
    build_episode_bowl_skill_library,
    build_fixed_bowl_place_application,
    preflight_bowl_place_handoff,
)
from robomex.orchestration.bowl_arena import (
    BowlArenaPointCloudPreviewPin,
    BowlArenaPreviewRuntimeProvenance,
    BowlCorrectionArenaFactoryAssembly,
    BowlCorrectionArenaFactoryConfig,
    BowlMotionPlanningPin,
    build_bowl_correction_arena_factory,
)
from robomex.orchestration.coding_provider import ExecutorFactory, PolicyFactory
from robomex.orchestration.motion_preview import MotionPreviewGeometryProvider
from robomex.orchestration.risk_provider import (
    MOTION_RISK_INPUT_SCHEMAS,
    MOTION_RISK_PROVIDER_ID,
    DeterministicMotionRiskProvider,
    MotionRiskActorBindings,
    MotionRiskProviderConfig,
    build_motion_risk_actor_bindings,
)
from robomex.protocols.bowl_place import CodingPhaseBudgetConfig
from robomex.protocols.risk_adaptive_bowl_place import (
    CORRECTION_CONTEXT_SCHEMAS,
    CORRECTION_EXECUTE_ACTIVATION_ID,
    BowlCorrectionCandidateStrategy,
    RiskAdaptiveBowlPlaceProtocol,
    RiskAdaptiveBowlPlaceProtocolConfig,
    build_risk_adaptive_bowl_place_protocol,
)
from robomex.runtime.action_protocol import AdmissionSnapshot, WorldKind

_RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ASSEMBLY_SCHEMA = "robomex.risk_adaptive_bowl_swarm_application.v1"


class BowlSwarmApplicationAssemblyError(BowlApplicationAssemblyError):
    """The Swarm dependencies cannot form one closed production run."""


@dataclass(frozen=True)
class RiskAdaptiveBowlPlaceApplicationConfig:
    """Sealed cross-component inputs for one production Swarm run."""

    base: FixedBowlPlaceApplicationConfig
    risk: MotionRiskProviderConfig
    arena: BowlCorrectionArenaFactoryConfig
    candidate_model_pins: Mapping[str, ModelPin] = field(default_factory=dict)
    risk_provider_id: str = MOTION_RISK_PROVIDER_ID
    arena_artifacts_subdirectory: str = "bowl_arena_bridge.v1"
    preview_artifacts_subdirectory: str = "bowl_motion_previews.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.base, FixedBowlPlaceApplicationConfig):
            raise TypeError("base must be a FixedBowlPlaceApplicationConfig")
        if not isinstance(self.base.protocol, RiskAdaptiveBowlPlaceProtocolConfig):
            raise TypeError(
                "base.protocol must be a RiskAdaptiveBowlPlaceProtocolConfig"
            )
        if not isinstance(self.risk, MotionRiskProviderConfig):
            raise TypeError("risk must be a MotionRiskProviderConfig")
        if not isinstance(self.arena, BowlCorrectionArenaFactoryConfig):
            raise TypeError("arena must be a BowlCorrectionArenaFactoryConfig")
        if not isinstance(self.candidate_model_pins, Mapping):
            raise TypeError("candidate_model_pins must be a candidate/model mapping")
        model_pins = {
            str(candidate_id): (
                pin if isinstance(pin, ModelPin) else ModelPin.model_validate(pin)
            )
            for candidate_id, pin in self.candidate_model_pins.items()
        }
        object.__setattr__(self, "candidate_model_pins", MappingProxyType(model_pins))
        if _RUNTIME_ID_RE.fullmatch(self.risk_provider_id) is None:
            raise ValueError("risk_provider_id must be a runtime identifier")
        for label, value in (
            ("arena_artifacts_subdirectory", self.arena_artifacts_subdirectory),
            ("preview_artifacts_subdirectory", self.preview_artifacts_subdirectory),
        ):
            relative = Path(value)
            if relative.is_absolute() or len(relative.parts) != 1 or value in {".", ".."}:
                raise ValueError(f"{label} must be one relative path segment")
        episode_subdirectories = (
            self.base.skill_library_subdirectory,
            self.base.coding_artifacts_subdirectory,
            self.arena_artifacts_subdirectory,
            self.preview_artifacts_subdirectory,
        )
        if len(episode_subdirectories) != len(set(episode_subdirectories)):
            raise ValueError("Skill, coding, Arena, and preview roots must be distinct")

        protocol = self.protocol
        provider = self.base.provider
        expected_entities = (provider.bowl_entity_id, provider.plate_entity_id)
        if (
            self.risk.expected_bowl_entity_id,
            self.risk.expected_target_entity_id,
        ) != expected_entities:
            raise ValueError("risk provider entities differ from the bowl provider")
        if self.risk.risk_policy != self.arena.risk_policy:
            raise ValueError("risk provider and Arena must share one exact RiskPolicy")
        if dict(CORRECTION_CONTEXT_SCHEMAS) != dict(MOTION_RISK_INPUT_SCHEMAS):
            raise ValueError("Arena context schemas differ from motion-risk input schemas")
        if self.arena.graph_id != protocol.graph_id:
            raise ValueError("Arena graph_id differs from the risk-adaptive protocol")
        if self.arena.graph_revision != protocol.revision:
            raise ValueError("Arena graph revision differs from the protocol")
        if self.arena.binding_id != protocol.correction_arena_binding_id:
            raise ValueError("Arena binding_id differs from the graph activation")
        if len(self.arena.candidates) != protocol.max_arena_candidates:
            raise ValueError("Arena candidate count differs from protocol K")
        if self.arena.max_alignment_iterations != protocol.max_alignment_iterations:
            raise ValueError("Arena iteration bound differs from the protocol loop")
        if (
            self.arena.total_candidate_budget_limit
            != protocol.correction_candidate_budget_limit
        ):
            raise ValueError(
                "production Swarm requires the full durable candidate quota iterations*K"
            )
        if self.arena.planning.robot_model_digest != self.base.robot_model_digest:
            raise ValueError("Arena robot model pin differs from the run pin")
        if self.arena.planning.expected_frame != provider.frame_id:
            raise ValueError("Arena planning frame differs from bowl observation frame")
        provider_ids = (
            self.base.deterministic_provider_id,
            self.base.coding_provider_id,
            self.risk_provider_id,
            self.arena.provider_id,
        )
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("fixed, risk, and Arena provider IDs must be distinct")
        if self.base.budgets.max_candidates < protocol.correction_candidate_budget_limit:
            raise ValueError("run max_candidates is below the complete Swarm loop quota")
        candidate_ids = {candidate.candidate_id for candidate in self.arena.candidates}
        if set(model_pins).difference(candidate_ids):
            raise ValueError("candidate_model_pins contains an unknown candidate ID")
        for candidate in self.arena.candidates:
            effective_model_id = candidate.model or self.base.model.model_id
            pin = model_pins.get(candidate.candidate_id, self.base.model)
            if pin.model_id != effective_model_id:
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} model differs from its ModelPin"
                )
            if effective_model_id == self.base.model.model_id and pin != self.base.model:
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} reuses the base model ID "
                    "with different provenance"
                )
            if candidate.model and candidate.candidate_id not in model_pins:
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} requires an explicit ModelPin"
                )

    @property
    def protocol(self) -> RiskAdaptiveBowlPlaceProtocolConfig:
        value = self.base.protocol
        if not isinstance(value, RiskAdaptiveBowlPlaceProtocolConfig):  # defensive narrowing
            raise TypeError("base protocol lost its risk-adaptive type")
        return value

    @property
    def content_digest(self) -> str:
        return canonical_payload_digest(
            _ASSEMBLY_SCHEMA,
            {
                "protocol_config_digest": canonical_payload_digest(
                    self.protocol.schema_version,
                    self.protocol.model_dump(mode="json"),
                ),
                "risk_config_digest": self.risk.content_digest,
                "arena_factory_config_digest": self.arena.content_digest,
                "candidate_model_pins": {
                    candidate_id: pin.model_dump(mode="json")
                    for candidate_id, pin in sorted(self.candidate_model_pins.items())
                },
                "risk_provider_id": self.risk_provider_id,
                "arena_artifacts_subdirectory": self.arena_artifacts_subdirectory,
                "preview_artifacts_subdirectory": self.preview_artifacts_subdirectory,
            },
        )

    @classmethod
    def from_components(
        cls,
        *,
        base: FixedBowlPlaceApplicationConfig,
        risk: MotionRiskProviderConfig,
        planning: BowlMotionPlanningPin,
        candidate_model_pins: Mapping[str, ModelPin] | None = None,
        arena_provider_id: str = "bowl_correction_arena_coding",
        strategies: Sequence[BowlCorrectionCandidateStrategy] | None = None,
        candidate_budgets: Sequence[CodingPhaseBudgetConfig] | None = None,
        candidate_models: Sequence[str] | None = None,
        total_candidate_budget_limit: int | None = None,
        arena_policy: ArenaPolicy | None = None,
        preview: BowlArenaPointCloudPreviewPin | None = None,
    ) -> RiskAdaptiveBowlPlaceApplicationConfig:
        """Construct the exactly pinned Arena config from protocol/risk inputs."""

        if not isinstance(base.protocol, RiskAdaptiveBowlPlaceProtocolConfig):
            raise TypeError("base.protocol must be risk-adaptive")
        arena = BowlCorrectionArenaFactoryConfig.from_protocol(
            protocol_config=base.protocol,
            risk_config=risk,
            planning=planning,
            provider_id=arena_provider_id,
            strategies=strategies,
            candidate_budgets=candidate_budgets,
            candidate_models=candidate_models,
            total_candidate_budget_limit=total_candidate_budget_limit,
            arena_policy=arena_policy,
            preview=preview,
        )
        return cls(
            base=base,
            risk=risk,
            arena=arena,
            candidate_model_pins=candidate_model_pins or {},
        )


@dataclass(frozen=True)
class RiskAdaptiveBowlPlaceApplication:
    """Complete admitted Swarm application and its exact construction inventory."""

    fixed: FixedBowlPlaceApplication
    config: RiskAdaptiveBowlPlaceApplicationConfig
    protocol: RiskAdaptiveBowlPlaceProtocol
    risk_provider: DeterministicMotionRiskProvider
    risk_bindings: MotionRiskActorBindings
    arena: BowlCorrectionArenaFactoryAssembly
    extension_providers: Mapping[str, AgentProvider]
    extension_profiles: Mapping[str, ActorProfile]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "extension_providers",
            MappingProxyType(dict(self.extension_providers)),
        )
        object.__setattr__(
            self,
            "extension_profiles",
            MappingProxyType(dict(self.extension_profiles)),
        )

    @property
    def agent(self) -> RoboMExV2Agent:
        return self.fixed.agent

    @property
    def application(self) -> V2Application:
        return self.fixed.application

    @property
    def manifest(self) -> RunManifest:
        return self.fixed.manifest


def build_risk_adaptive_bowl_place_application(
    *,
    config: RiskAdaptiveBowlPlaceApplicationConfig,
    action_backends: Sequence[ActionBackendBinding],
    observation_backends: Sequence[ObservationBackendBinding],
    capx_executor_factory: ExecutorFactory,
    coding_policy: CompletionPolicy | None = None,
    coding_policy_factory: PolicyFactory | None = None,
    shadow_backends: Sequence[ShadowBackendBinding] = (),
    schema_registry: SchemaRegistry | None = None,
    preview_geometry: MotionPreviewGeometryProvider | None = None,
    preview_runtime_provenance: BowlArenaPreviewRuntimeProvenance | None = None,
) -> RiskAdaptiveBowlPlaceApplication:
    """Build and admit the complete risk-adaptive bowl-place Swarm baseline."""

    if not isinstance(config, RiskAdaptiveBowlPlaceApplicationConfig):
        raise TypeError("config must be a RiskAdaptiveBowlPlaceApplicationConfig")
    base = config.base
    action_bindings = tuple(action_backends)
    observation_bindings = tuple(observation_backends)
    shadow_bindings = tuple(shadow_backends)
    heterogeneous_models = {
        candidate.model
        for candidate in config.arena.candidates
        if candidate.model and candidate.model != base.model.model_id
    }
    if heterogeneous_models and coding_policy_factory is None:
        raise BowlSwarmApplicationAssemblyError(
            "heterogeneous candidate models require a per-actor coding_policy_factory"
        )

    # This read-only check happens before Skill/provider directories are
    # materialized.  The common builder repeats it and checks again after
    # runtime construction to close both mutation and TOCTOU gaps.
    preflight_bowl_place_handoff(base)
    protocol = build_risk_adaptive_bowl_place_protocol(config.protocol)
    _validate_swarm_motion_pins(
        protocol,
        config.protocol,
        config.arena.planning,
    )
    _preflight_arm_configuration(
        base,
        config.arena.planning,
        action_bindings,
    )
    library = build_episode_bowl_skill_library(
        base.episode_root,
        subdirectory=base.skill_library_subdirectory,
    )
    coding_provider = build_bowl_skill_coding_provider(
        config=base,
        library=library,
        capx_executor_factory=capx_executor_factory,
        coding_policy=coding_policy,
        coding_policy_factory=coding_policy_factory,
    )

    risk_provider = DeterministicMotionRiskProvider(config.risk)
    risk_bindings = build_motion_risk_actor_bindings(
        config.risk,
        provider=risk_provider,
        provider_id=config.risk_provider_id,
    )
    preview_enabled = config.arena.preview.enabled
    arena = build_bowl_correction_arena_factory(
        config.arena,
        protocol_config=config.protocol,
        risk_config=config.risk,
        coding_provider=coding_provider,
        artifacts_root=base.episode_root / config.arena_artifacts_subdirectory,
        preview_geometry=preview_geometry,
        preview_output_root=(
            base.episode_root / config.preview_artifacts_subdirectory
            if preview_enabled
            else None
        ),
        preview_runtime_provenance=preview_runtime_provenance,
    )
    extension_providers = _merge_unique(
        "provider ID",
        risk_bindings.providers,
        arena.providers,
    )
    extension_profiles = _merge_unique(
        "runner ref",
        risk_bindings.profiles,
        arena.profiles,
    )
    arena_profile_ids = tuple(
        sorted(profile.profile_id for profile in arena.profiles.values())
    )
    skill_actor_profiles = dict.fromkeys(arena.required_skill_ids, arena_profile_ids)
    unknown_arena_skills = set(skill_actor_profiles).difference(BOWL_PLACE_SKILL_IDS)
    if unknown_arena_skills:
        raise BowlSwarmApplicationAssemblyError(
            "Arena requests skills outside the bowl catalog: "
            + ", ".join(sorted(unknown_arena_skills))
        )

    metadata = _swarm_manifest_metadata(
        config=config,
        protocol=protocol,
        risk_bindings=risk_bindings,
        arena=arena,
    )
    fixed = build_fixed_bowl_place_application(
        config=base,
        action_backends=action_bindings,
        observation_backends=observation_bindings,
        capx_executor_factory=capx_executor_factory,
        coding_policy=coding_policy,
        coding_policy_factory=coding_policy_factory,
        schema_registry=schema_registry,
        _trusted_extensions=TrustedBowlApplicationExtensions(
            protocol_override=protocol,
            skill_library_override=library,
            coding_provider_override=coding_provider,
            actor_providers=extension_providers,
            actor_profiles=extension_profiles,
            shadow_backends=shadow_bindings,
            arena_bindings=(arena.binding,),
            skill_actor_profiles=skill_actor_profiles,
            manifest_metadata=metadata,
            candidate_config_digest=arena.candidate_config_digest,
        ),
    )
    _validate_admitted_swarm(
        fixed=fixed,
        config=config,
        protocol=protocol,
        risk_provider=risk_provider,
        risk_bindings=risk_bindings,
        arena=arena,
        extension_providers=extension_providers,
        extension_profiles=extension_profiles,
    )
    return RiskAdaptiveBowlPlaceApplication(
        fixed=fixed,
        config=config,
        protocol=protocol,
        risk_provider=risk_provider,
        risk_bindings=risk_bindings,
        arena=arena,
        extension_providers=extension_providers,
        extension_profiles=extension_profiles,
    )


def _swarm_manifest_metadata(
    *,
    config: RiskAdaptiveBowlPlaceApplicationConfig,
    protocol: RiskAdaptiveBowlPlaceProtocol,
    risk_bindings: MotionRiskActorBindings,
    arena: BowlCorrectionArenaFactoryAssembly,
) -> dict[str, Any]:
    risk_profiles = {
        runner_ref: {
            "profile_id": profile.profile_id,
            "content_digest": actor_profile_digest(profile),
        }
        for runner_ref, profile in sorted(risk_bindings.profiles.items())
    }
    arena_profiles = {
        runner_ref: {
            "profile_id": profile.profile_id,
            "content_digest": arena.profile_digests[runner_ref],
        }
        for runner_ref, profile in sorted(arena.profiles.items())
    }
    return {
        "swarm_assembly_schema_version": _ASSEMBLY_SCHEMA,
        "swarm_application_config_digest": config.content_digest,
        "risk_adaptive_protocol_config": config.protocol.model_dump(mode="json"),
        "risk_adaptive_protocol_config_digest": canonical_payload_digest(
            config.protocol.schema_version,
            config.protocol.model_dump(mode="json"),
        ),
        "risk_provider_config": config.risk.model_dump(mode="json"),
        "risk_provider_config_digest": config.risk.content_digest,
        "risk_actor_profiles": risk_profiles,
        "arena_factory_config": config.arena.model_dump(mode="json"),
        "arena_factory_config_digest": config.arena.content_digest,
        "arena_candidate_config_digest": arena.candidate_config_digest,
        "arena_actor_profiles": arena_profiles,
        "arena_candidate_model_pins": {
            candidate.candidate_id: (
                config.candidate_model_pins.get(candidate.candidate_id, config.base.model)
            ).model_dump(mode="json")
            for candidate in config.arena.candidates
        },
        "arena_binding_id": arena.binding.binding_id,
        "arena_binding_digest": arena.binding.content_digest,
        "arena_single_round_k": len(arena.binding.candidates),
        "arena_total_candidate_budget": arena.binding.candidate_budget_limit,
        "arena_candidate_budget_id": arena.binding.candidate_budget_id,
        "arena_preview_provenance": (
            arena.preview_runtime_provenance.model_dump(mode="json")
            if arena.preview_runtime_provenance is not None
            else None
        ),
        "compiled_swarm_graph_digest": f"sha256:{protocol.compiled.digest}",
        "authoritative_correction_activation": CORRECTION_EXECUTE_ACTIVATION_ID,
        "candidate_effect_authority": "read_only",
        "physical_writer_policy": "runtime_sealed_action_only",
        "baseline_mutation_policy": "disabled",
    }


def _validate_swarm_motion_pins(
    protocol: RiskAdaptiveBowlPlaceProtocol,
    protocol_config: RiskAdaptiveBowlPlaceProtocolConfig,
    planning: BowlMotionPlanningPin,
) -> None:
    tcp_frames = {
        str(node.params["tcp_frame_id"])
        for node in protocol.spec.activations
        if "tcp_frame_id" in node.params
    }
    planner_backends = {
        str(node.params["planner_backend"])
        for node in protocol.spec.activations
        if "planner_backend" in node.params
    }
    if tcp_frames != {planning.tcp_frame_id}:
        raise BowlSwarmApplicationAssemblyError(
            "Arena TCP differs from transport/descend/retreat motion workers"
        )
    if planner_backends != {planning.planner_backend}:
        raise BowlSwarmApplicationAssemblyError(
            "Arena planner backend differs from the remaining bowl motion workers"
        )
    nodes = {node.activation_id: node for node in protocol.spec.activations}
    correction = nodes.get(CORRECTION_EXECUTE_ACTIVATION_ID)
    if correction is None or (
        correction.authority_world_id,
        correction.authoritative_resource,
    ) != (protocol_config.authority_world_id, protocol_config.arm_resource_id):
        raise BowlSwarmApplicationAssemblyError(
            "execute_correction is not bound to the exact authoritative arm"
        )


def _preflight_arm_configuration(
    base: FixedBowlPlaceApplicationConfig,
    planning: BowlMotionPlanningPin,
    action_bindings: tuple[ActionBackendBinding, ...],
) -> None:
    matches = [
        binding
        for binding in action_bindings
        if (binding.world_id, binding.resource_id)
        == (base.protocol.authority_world_id, base.protocol.arm_resource_id)
    ]
    if len(matches) != 1:
        raise BowlSwarmApplicationAssemblyError(
            "Swarm preflight requires one exact authoritative arm binding"
        )
    binding = matches[0]
    try:
        raw = binding.backend.snapshot(binding.world_id, binding.resource_id)
        snapshot = AdmissionSnapshot.model_validate(
            raw.model_dump(mode="python") if isinstance(raw, AdmissionSnapshot) else raw
        )
    except Exception as exc:  # noqa: BLE001 - dependency admission boundary
        raise BowlSwarmApplicationAssemblyError(
            f"authoritative arm snapshot preflight failed: {type(exc).__name__}"
        ) from exc
    if (
        snapshot.world_id != binding.world_id
        or snapshot.resource_id != binding.resource_id
        or snapshot.world_kind is not WorldKind.AUTHORITATIVE
    ):
        raise BowlSwarmApplicationAssemblyError(
            "authoritative arm snapshot is bound to another world/resource"
        )
    if snapshot.config_digest != planning.robot_config_digest:
        raise BowlSwarmApplicationAssemblyError(
            "authoritative arm config digest differs from the Arena planning pin"
        )


def _validate_admitted_swarm(
    *,
    fixed: FixedBowlPlaceApplication,
    config: RiskAdaptiveBowlPlaceApplicationConfig,
    protocol: RiskAdaptiveBowlPlaceProtocol,
    risk_provider: DeterministicMotionRiskProvider,
    risk_bindings: MotionRiskActorBindings,
    arena: BowlCorrectionArenaFactoryAssembly,
    extension_providers: Mapping[str, AgentProvider],
    extension_profiles: Mapping[str, ActorProfile],
) -> None:
    application = fixed.application
    plane = application.episode.data_plane
    _preflight_arm_configuration(
        config.base,
        config.arena.planning,
        fixed.dependencies.action_backends,
    )
    if fixed.protocol is not protocol:
        raise BowlSwarmApplicationAssemblyError("runtime admitted another protocol object")
    if fixed.coding_provider is not arena.provider.coding_provider:
        raise BowlSwarmApplicationAssemblyError(
            "Arena and fixed workers do not share the same SkillCoding provider"
        )
    if fixed.coding_provider.data_plane is not plane:
        raise BowlSwarmApplicationAssemblyError("coding provider is not runtime-bound")
    if risk_provider.data_plane is not plane:
        raise BowlSwarmApplicationAssemblyError("risk provider is not runtime-bound")
    if arena.provider.data_plane is not plane:
        raise BowlSwarmApplicationAssemblyError("Arena provider is not runtime-bound")
    for provider_id, provider in extension_providers.items():
        if fixed.dependencies.actor_providers.get(provider_id) is not provider:
            raise BowlSwarmApplicationAssemblyError(
                f"runtime dependency changed extension provider {provider_id!r}"
            )
    for runner_ref, profile in extension_profiles.items():
        if fixed.dependencies.actor_profiles.get(runner_ref) != profile:
            raise BowlSwarmApplicationAssemblyError(
                f"runtime dependency changed extension profile {runner_ref!r}"
            )
    if set(risk_bindings.providers) != {config.risk_provider_id}:
        raise BowlSwarmApplicationAssemblyError("risk provider mapping drifted")
    if fixed.dependencies.arena_bindings != (arena.binding,):
        raise BowlSwarmApplicationAssemblyError("runtime Arena binding inventory drifted")
    if fixed.manifest.candidate_config_digest != arena.candidate_config_digest:
        raise BowlSwarmApplicationAssemblyError("manifest candidate digest drifted")
    if fixed.manifest.mutation_policy != "disabled":
        raise BowlSwarmApplicationAssemblyError("baseline unexpectedly enabled mutation")
    expected_total = config.protocol.correction_candidate_budget_limit
    if fixed.graph_budget_envelope.candidates != expected_total:
        raise BowlSwarmApplicationAssemblyError(
            "compiled graph candidate envelope is not max_iterations*K"
        )
    if arena.binding.candidate_budget_limit != expected_total:
        raise BowlSwarmApplicationAssemblyError("Arena durable quota drifted")


def _merge_unique(
    label: str,
    *mappings: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        overlap = set(merged).intersection(mapping)
        if overlap:
            raise BowlSwarmApplicationAssemblyError(
                f"duplicate {label}: " + ", ".join(sorted(overlap))
            )
        merged.update(mapping)
    return merged


__all__ = [
    "BowlSwarmApplicationAssemblyError",
    "RiskAdaptiveBowlPlaceApplication",
    "RiskAdaptiveBowlPlaceApplicationConfig",
    "build_risk_adaptive_bowl_place_application",
]
