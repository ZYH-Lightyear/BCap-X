"""Production assembly for the fixed bowl-on-plate RoboMEx v2 baseline.

The protocol, deterministic provider and coding provider deliberately remain
separate modules.  This module is the closed application boundary that proves
that one concrete run has all of them bound to the same immutable identity.
It contains no backend, policy, executor, provenance or digest fixtures: those
identities are supplied by the robot installation/operator and are admitted by
the normal :class:`~robomex.orchestration.bootstrap.V2RuntimeFactory`.

This is a *place-only* application.  A preceding grounding/pick workflow must
already have registered the bowl and plate and durably recorded the admitted
grasp action.  Assembly validates that handoff and never fabricates state.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from robomex.authoring.monitoring import CompiledMonitorProgram, MonitorHook
from robomex.contracts import (
    ContractCatalogSnapshot,
    canonical_payload_digest,
)
from robomex.contracts.skill_catalog_builder import build_skill_contract_catalog
from robomex.core.coder import CompletionPolicy
from robomex.data import (
    AttachmentStatus,
    EmbodiedStateReducer,
    SchemaRegistry,
)
from robomex.elastic import CompiledElasticGraph, ElasticGraphSpec
from robomex.elastic.graph_patch import ClosedSlot, ComposableFrontier
from robomex.evolution import (
    ModelPin,
    PromptPin,
    RunBudgets,
    RunManifest,
    TaskSnapshot,
)
from robomex.orchestration.actors import ActorProfile, AgentProvider
from robomex.orchestration.application import (
    ManagerWorkflowAuthor,
    PinnedBaselineManagerInvoker,
    RoboMExV2Agent,
    TaskOrchestrationLimits,
    V2AgentConfig,
)
from robomex.orchestration.arena import ArenaBinding
from robomex.orchestration.bootstrap import (
    ActionBackendBinding,
    ObservationBackendBinding,
    ShadowBackendBinding,
    V2Application,
    V2RuntimeConfig,
    V2RuntimeDependencies,
    V2RuntimeFactory,
)
from robomex.orchestration.bowl_provider import (
    BowlPlaceActorBindings,
    BowlPlaceActorProvider,
    BowlPlaceCodingBindings,
    BowlPlaceProviderConfig,
    BowlTrackingMode,
    build_bowl_place_actor_bindings,
    build_bowl_place_coding_profiles,
)
from robomex.orchestration.coding_provider import (
    DEFAULT_CODING_CAPABILITY_GRANTS,
    ExecutorFactory,
    PolicyFactory,
    SkillCodingAgentProvider,
)
from robomex.orchestration.episode import install_runtime_schemas
from robomex.orchestration.intent import SubgoalIntent
from robomex.orchestration.manager import ManagerLimits
from robomex.orchestration.production_manifest import (
    GraphBudgetEnvelope,
    assert_run_budgets_cover,
    catalog_function_pins,
    catalog_skill_pins,
    graph_budget_envelope,
    runtime_actor_profile_pins,
    runtime_backend_pins,
)
from robomex.orchestration.task_orchestrator import (
    OutcomeAwareFixedIntentPlanner,
    PlannerCallBudget,
)
from robomex.protocols.bowl_place import (
    BowlPlaceProtocolConfig,
    build_fixed_bowl_place_protocol,
)
from robomex.runtime.action_protocol import (
    BackendMotionInterface,
    MonitorTelemetryCapabilities,
    MonitorTelemetryHook,
)
from robomex.runtime.authority import (
    CooperativeMotionBackend,
    MonitorTelemetryBackend,
    WatchdogControllableBackend,
)
from robomex.skills import Skill, SkillLibrary

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

BOWL_PLACE_SKILL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("monitor", "author_attachment_monitor"),
    ("state", "propose_attachment_transition"),
    ("state", "propose_relation_transition"),
    ("affordance", "estimate_support_alignment"),
    ("motion", "author_sealed_phase_motion"),
)
BOWL_PLACE_SKILL_IDS: tuple[str, ...] = tuple(
    sorted(skill_id for _category, skill_id in BOWL_PLACE_SKILL_PACKAGES)
)
BOWL_PLACE_FUNCTION_IDS: tuple[str, ...] = tuple(
    sorted(
        (
            "function.author_attachment_monitor.build_attachment_monitor_program",
            "function.propose_attachment_transition.build_attachment_transition_proposal",
            "function.propose_relation_transition.build_relation_transition_proposal",
            "function.estimate_support_alignment.estimate_support_alignment",
            "function.author_sealed_phase_motion.build_sealed_phase_motion",
        )
    )
)


class BowlApplicationAssemblyError(RuntimeError):
    """A fixed bowl application cannot be admitted as one closed run."""


class BowlPlaceHandoffError(BowlApplicationAssemblyError):
    """The durable pick-to-place handoff is absent or unsafe."""


@runtime_checkable
class BowlPlaceProtocolAssembly(Protocol):
    """Structural protocol bundle consumed by the common application assembly."""

    @property
    def spec(self) -> ElasticGraphSpec: ...

    @property
    def compiled(self) -> CompiledElasticGraph: ...

    @property
    def recovery_slot(self) -> ClosedSlot: ...

    @property
    def recovery_frontier(self) -> ComposableFrontier: ...

    @property
    def attachment_monitor_program(self) -> CompiledMonitorProgram: ...


class FeasibilityCheckerPin(BaseModel):
    """Operator-owned provenance for one installed feasibility checker."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    checker_id: str = Field(min_length=1, max_length=256)
    implementation_digest: str
    configuration_digest: str
    version: str = Field(min_length=1, max_length=256)

    @field_validator("implementation_digest", "configuration_digest")
    @classmethod
    def _canonical_digest(cls, value: str) -> str:
        _require_digest("feasibility checker digest", value)
        return value


@dataclass(frozen=True)
class BowlPlaceHandoff:
    """Validated durable state inherited from a preceding pick workflow."""

    episode_id: str
    state_revision: int
    bowl_entity_revision: int
    plate_entity_revision: int
    attachment_revision: int
    attachment_status: AttachmentStatus
    action_id: str
    attempted_effect_id: str


def _default_task_limits() -> TaskOrchestrationLimits:
    return TaskOrchestrationLimits(
        max_intents=1,
        max_planner_calls=2,
        max_workflow_frontiers=1_000,
    )


def _default_manager_limits() -> ManagerLimits:
    # The pinned Manager spends no model calls/tokens.  One token of session
    # capacity keeps the structured initial call admissible; tokens_used is 0.
    return ManagerLimits(
        max_initial_calls=1,
        max_reactivations=1,
        max_tokens=1,
        max_tokens_per_call=1,
        max_candidates=0,
    )


@dataclass(frozen=True)
class FixedBowlPlaceApplicationConfig:
    """Operator-owned identity and bounded configuration for one fixed run."""

    run_id: str
    episode_id: str
    episode_root: Path
    task: TaskSnapshot
    intent: SubgoalIntent
    seed: int
    model: ModelPin
    prompts: tuple[PromptPin, ...]
    budgets: RunBudgets
    runtime_code_digest: str
    robot_model_digest: str
    feasibility_checker_pins: Mapping[str, FeasibilityCheckerPin]
    created_at: datetime
    protocol: BowlPlaceProtocolConfig = field(default_factory=BowlPlaceProtocolConfig)
    provider: BowlPlaceProviderConfig = field(default_factory=BowlPlaceProviderConfig)
    task_limits: TaskOrchestrationLimits = field(default_factory=_default_task_limits)
    planner_budget: PlannerCallBudget = field(default_factory=PlannerCallBudget)
    manager_limits: ManagerLimits = field(default_factory=_default_manager_limits)
    deterministic_provider_id: str = "bowl_place"
    coding_provider_id: str = "skill_coding"
    skill_library_subdirectory: str = "bowl_skill_library.v1"
    coding_artifacts_subdirectory: str = "bowl_coding_provider.v1"
    actor_namespace_root: str | None = None
    coding_environment: str = ""
    coding_api_docs: str = ""
    trusted_import_roots: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.feasibility_checker_pins, Mapping):
            raise TypeError("feasibility_checker_pins must be a resource mapping")
        raw_episode_root = Path(self.episode_root)
        if raw_episode_root in {Path("."), Path(".."), Path("/")}:
            raise ValueError("episode_root must identify a dedicated episode directory")
        object.__setattr__(self, "episode_root", raw_episode_root.resolve())
        object.__setattr__(self, "prompts", tuple(self.prompts))
        object.__setattr__(self, "trusted_import_roots", frozenset(self.trusted_import_roots))
        object.__setattr__(
            self,
            "feasibility_checker_pins",
            MappingProxyType(
                {
                    str(resource_id): (
                        pin
                        if isinstance(pin, FeasibilityCheckerPin)
                        else FeasibilityCheckerPin.model_validate(pin)
                    )
                    for resource_id, pin in self.feasibility_checker_pins.items()
                }
            ),
        )
        for label, value in (
            ("run_id", self.run_id),
            ("episode_id", self.episode_id),
            ("deterministic_provider_id", self.deterministic_provider_id),
            ("coding_provider_id", self.coding_provider_id),
        ):
            if not isinstance(value, str) or not _RUNTIME_ID_RE.fullmatch(value):
                raise ValueError(f"{label} must be a non-empty runtime identifier")
        if self.deterministic_provider_id == self.coding_provider_id:
            raise ValueError("deterministic and coding provider IDs must differ")
        for label, value in (
            ("skill_library_subdirectory", self.skill_library_subdirectory),
            ("coding_artifacts_subdirectory", self.coding_artifacts_subdirectory),
        ):
            path = Path(value)
            if path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
                raise ValueError(f"{label} must be one relative path segment")
        if not isinstance(self.task, TaskSnapshot):
            raise TypeError("task must be a TaskSnapshot")
        if not isinstance(self.intent, SubgoalIntent):
            raise TypeError("intent must be a SubgoalIntent")
        if not isinstance(self.model, ModelPin):
            raise TypeError("model must be a ModelPin")
        if not self.prompts or any(not isinstance(item, PromptPin) for item in self.prompts):
            raise TypeError("prompts must contain at least one PromptPin")
        if len({item.prompt_id for item in self.prompts}) != len(self.prompts):
            raise ValueError("prompt pins must have unique prompt_id values")
        if not isinstance(self.budgets, RunBudgets):
            raise TypeError("budgets must be RunBudgets")
        if not isinstance(self.protocol, BowlPlaceProtocolConfig):
            raise TypeError("protocol must be a BowlPlaceProtocolConfig")
        if not isinstance(self.provider, BowlPlaceProviderConfig):
            raise TypeError("provider must be a BowlPlaceProviderConfig")
        expected_entities = {self.provider.bowl_entity_id, self.provider.plate_entity_id}
        intent_entities = {item.entity_id for item in self.intent.entity_refs}
        if intent_entities != expected_entities:
            raise ValueError(
                "fixed bowl intent entity_refs must exactly name the configured bowl and plate"
            )
        if not isinstance(self.task_limits, TaskOrchestrationLimits):
            raise TypeError("task_limits must be TaskOrchestrationLimits")
        if not isinstance(self.planner_budget, PlannerCallBudget):
            raise TypeError("planner_budget must be PlannerCallBudget")
        if not isinstance(self.manager_limits, ManagerLimits):
            raise TypeError("manager_limits must be ManagerLimits")
        _require_digest("runtime_code_digest", self.runtime_code_digest)
        _require_digest("robot_model_digest", self.robot_model_digest)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.task_limits.max_intents != 1 or self.task_limits.max_planner_calls != 2:
            raise ValueError(
                "the outcome-aware fixed baseline requires exactly one intent and two "
                "planner decisions"
            )
        if self.manager_limits.max_initial_calls != 1:
            raise ValueError("the fixed baseline requires exactly one initial Manager call")
        if self.manager_limits.max_tokens < 1 or self.manager_limits.max_tokens_per_call < 1:
            raise ValueError(
                "the structured pinned Manager call requires positive session capacity"
            )
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be an operator-supplied datetime")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")


@dataclass(frozen=True)
class TrustedBowlApplicationExtensions:
    """Trusted assembly-only additions for the future Swarm/Arena variant.

    This value is injected by the application entrypoint before the immutable
    manifest exists.  It is never exposed as an Agent, graph, or Manager input.
    Fixed metadata remains protected; extensions can only add separately
    pinned namespaces.
    """

    protocol_override: BowlPlaceProtocolAssembly | None = None
    skill_library_override: SkillLibrary | None = None
    coding_provider_override: SkillCodingAgentProvider | None = None
    actor_providers: Mapping[str, AgentProvider] = field(default_factory=dict)
    actor_profiles: Mapping[str, ActorProfile] = field(default_factory=dict)
    shadow_backends: tuple[ShadowBackendBinding, ...] = ()
    arena_bindings: tuple[ArenaBinding, ...] = ()
    skill_actor_profiles: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    manifest_metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_config_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.skill_actor_profiles, Mapping):
            raise TypeError("skill_actor_profiles must be a skill/profile mapping")
        if self.protocol_override is not None and not isinstance(
            self.protocol_override, BowlPlaceProtocolAssembly
        ):
            raise TypeError("protocol_override must implement BowlPlaceProtocolAssembly")
        if self.skill_library_override is not None and not isinstance(
            self.skill_library_override, SkillLibrary
        ):
            raise TypeError("skill_library_override must be a SkillLibrary")
        if self.coding_provider_override is not None and not isinstance(
            self.coding_provider_override, SkillCodingAgentProvider
        ):
            raise TypeError("coding_provider_override must be a SkillCodingAgentProvider")
        if (self.skill_library_override is None) != (
            self.coding_provider_override is None
        ):
            raise ValueError(
                "skill_library_override and coding_provider_override must be supplied together"
            )
        if (
            self.coding_provider_override is not None
            and self.coding_provider_override.library is not self.skill_library_override
        ):
            raise ValueError("coding_provider_override must use the exact overridden library")
        providers = dict(self.actor_providers)
        profiles = dict(self.actor_profiles)
        if any(not isinstance(key, str) or not key.strip() for key in providers):
            raise TypeError("extension actor provider IDs must be non-empty strings")
        if any(not isinstance(value, AgentProvider) for value in providers.values()):
            raise TypeError("extension actor providers must implement AgentProvider")
        if any(not isinstance(key, str) or not key.strip() for key in profiles):
            raise TypeError("extension runner refs must be non-empty strings")
        if any(not isinstance(value, ActorProfile) for value in profiles.values()):
            raise TypeError("extension actor profiles must contain ActorProfile values")
        shadows = tuple(self.shadow_backends)
        arenas = tuple(self.arena_bindings)
        if any(not isinstance(value, ShadowBackendBinding) for value in shadows):
            raise TypeError("shadow_backends must contain ShadowBackendBinding values")
        if any(not isinstance(value, ArenaBinding) for value in arenas):
            raise TypeError("arena_bindings must contain ArenaBinding values")
        skill_actor_profiles: dict[str, tuple[str, ...]] = {}
        for skill_id, profile_ids in self.skill_actor_profiles.items():
            normalized_skill_id = str(skill_id).strip()
            normalized_profiles = tuple(str(value).strip() for value in profile_ids)
            if (
                not normalized_skill_id
                or not normalized_profiles
                or any(not value for value in normalized_profiles)
            ):
                raise ValueError("skill_actor_profiles requires non-empty skill/profile IDs")
            if len(normalized_profiles) != len(set(normalized_profiles)):
                raise ValueError("skill_actor_profiles cannot repeat a profile ID")
            skill_actor_profiles[normalized_skill_id] = normalized_profiles
        metadata = {str(key): _json_value(value) for key, value in self.manifest_metadata.items()}
        if any(not key.strip() for key in metadata):
            raise ValueError("extension manifest metadata keys must not be empty")
        if self.candidate_config_digest is not None:
            _require_digest("candidate_config_digest", self.candidate_config_digest)
        object.__setattr__(self, "actor_providers", MappingProxyType(providers))
        object.__setattr__(self, "actor_profiles", MappingProxyType(profiles))
        object.__setattr__(self, "shadow_backends", shadows)
        object.__setattr__(self, "arena_bindings", arenas)
        object.__setattr__(
            self,
            "skill_actor_profiles",
            MappingProxyType(skill_actor_profiles),
        )
        object.__setattr__(self, "manifest_metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class FixedBowlPlaceApplication:
    """Complete fixed baseline plus its admitted construction inventory."""

    agent: RoboMExV2Agent
    application: V2Application
    manifest: RunManifest
    protocol: BowlPlaceProtocolAssembly
    deterministic_provider: BowlPlaceActorProvider
    coding_provider: SkillCodingAgentProvider
    deterministic_bindings: BowlPlaceActorBindings
    coding_bindings: BowlPlaceCodingBindings
    actor_profiles: Mapping[str, ActorProfile]
    dependencies: V2RuntimeDependencies
    skill_library: SkillLibrary
    contract_catalog: ContractCatalogSnapshot
    handoff: BowlPlaceHandoff
    graph_budget_envelope: GraphBudgetEnvelope


def build_episode_bowl_skill_library(
    episode_root: str | Path,
    *,
    subdirectory: str = "bowl_skill_library.v1",
) -> SkillLibrary:
    """Materialize the exact five builtin bowl skills as an episode-owned view."""

    root = Path(episode_root).resolve()
    relative = Path(subdirectory)
    if relative.is_absolute() or len(relative.parts) != 1 or subdirectory in {".", ".."}:
        raise ValueError("subdirectory must be one relative path segment")
    library = SkillLibrary(root / relative)
    builtin_root = Path(__file__).resolve().parents[1] / "skills" / "builtin"
    for category, skill_id in BOWL_PLACE_SKILL_PACKAGES:
        package = builtin_root / category / skill_id
        if not package.is_dir():
            raise BowlApplicationAssemblyError(
                f"missing builtin bowl skill package {category}/{skill_id}"
            )
        library.admit(Skill.from_dir(package), source="builtin-fixed-bowl-v1")
    admitted = tuple(sorted(record.skill_id for record in library.all()))
    if admitted != BOWL_PLACE_SKILL_IDS:
        raise BowlApplicationAssemblyError(
            "episode bowl skill view must contain exactly the five admitted skills"
        )
    return library


def build_bowl_skill_coding_provider(
    *,
    config: FixedBowlPlaceApplicationConfig,
    library: SkillLibrary,
    capx_executor_factory: ExecutorFactory,
    coding_policy: CompletionPolicy | None = None,
    coding_policy_factory: PolicyFactory | None = None,
) -> SkillCodingAgentProvider:
    """Build the one coding provider shared by fixed and Arena workers.

    Sharing is intentional: every bowl worker retrieves from one episode-owned
    Skill view, uses one crash-safe invocation ledger, and receives the same
    proposal-only capability guard.  Per-actor policy state remains available
    through ``coding_policy_factory``.
    """

    if not isinstance(config, FixedBowlPlaceApplicationConfig):
        raise TypeError("config must be a FixedBowlPlaceApplicationConfig")
    if not isinstance(library, SkillLibrary):
        raise TypeError("library must be a SkillLibrary")
    if not callable(capx_executor_factory):
        raise TypeError("capx_executor_factory must be callable")
    if (coding_policy is None) == (coding_policy_factory is None):
        raise ValueError("supply exactly one bounded coding policy or policy factory")
    _validate_episode_bowl_skill_library(config, library)
    ceiling = config.protocol.coding_budget
    return SkillCodingAgentProvider(
        data_plane=None,
        executor_factory=capx_executor_factory,
        library=library,
        policy=coding_policy,
        policy_factory=coding_policy_factory,
        artifacts_root=config.episode_root / config.coding_artifacts_subdirectory,
        max_turns=ceiling.model_calls,
        max_model_calls=ceiling.model_calls,
        max_tokens=ceiling.tokens,
        max_wall_time_ms=ceiling.wall_time_ms,
        environment=config.coding_environment,
        api_docs=_effective_coding_api_docs(config),
        trusted_import_roots=config.trusted_import_roots,
        trusted_skill_sidecars=frozenset(BOWL_PLACE_SKILL_IDS),
    )


def build_bowl_place_contract_catalog(
    library: SkillLibrary,
    coding_bindings: BowlPlaceCodingBindings,
    *,
    additional_compatible_actor_profiles: Mapping[str, Sequence[str]] | None = None,
) -> ContractCatalogSnapshot:
    """Compile and validate the exact skill/function catalog for the coding swarm."""

    required = tuple(sorted(coding_bindings.required_skill_ids))
    if required != BOWL_PLACE_SKILL_IDS:
        raise BowlApplicationAssemblyError(
            "bowl coding profiles do not request the exact builtin skill inventory"
        )
    compatible: dict[str, list[str]] = {skill_id: [] for skill_id in required}
    for profile in coding_bindings.profiles.values():
        preloaded = profile.metadata.get("preloaded_skills", ())
        if not isinstance(preloaded, (tuple, list)):
            raise BowlApplicationAssemblyError(
                f"coding profile {profile.profile_id!r} has malformed preloaded_skills"
            )
        for skill_id in preloaded:
            if skill_id not in compatible:
                raise BowlApplicationAssemblyError(
                    f"coding profile {profile.profile_id!r} references an unadmitted skill"
                )
            compatible[skill_id].append(profile.profile_id)
    for skill_id, profile_ids in (additional_compatible_actor_profiles or {}).items():
        if skill_id not in compatible:
            raise BowlApplicationAssemblyError(
                f"extension compatibility references unadmitted skill {skill_id!r}"
            )
        values = tuple(str(profile_id).strip() for profile_id in profile_ids)
        if not values or any(not value for value in values):
            raise BowlApplicationAssemblyError(
                f"extension compatibility for {skill_id!r} requires profile IDs"
            )
        if len(values) != len(set(values)):
            raise BowlApplicationAssemblyError(
                f"extension compatibility for {skill_id!r} contains duplicates"
            )
        compatible[skill_id].extend(values)
    if any(not profile_ids for profile_ids in compatible.values()):
        raise BowlApplicationAssemblyError("every bowl skill must be reachable by a coding profile")
    catalog = build_skill_contract_catalog(
        library,
        required,
        compatible_actor_profiles={
            skill_id: tuple(sorted(set(profile_ids)))
            for skill_id, profile_ids in compatible.items()
        },
    )
    actual_skills = tuple(sorted(skill.skill_id for skill in catalog.skills))
    actual_functions = tuple(
        sorted(export.function_id for skill in catalog.skills for export in skill.functions)
    )
    if actual_skills != BOWL_PLACE_SKILL_IDS:
        raise BowlApplicationAssemblyError("contract catalog skill coverage is not exact")
    if actual_functions != BOWL_PLACE_FUNCTION_IDS:
        raise BowlApplicationAssemblyError("contract catalog function coverage is not exact")
    return catalog


def verify_bowl_place_handoff(
    reducer: EmbodiedStateReducer,
    *,
    provider_config: BowlPlaceProviderConfig,
) -> BowlPlaceHandoff:
    """Validate the durable pick result required by this place-only protocol.

    Both ``attempted`` and ``verified_held`` are valid boundaries.  In either
    case the reducer ledger must contain the matching ATTEMPTED transition and
    the current attachment must retain its admitted action ID.  No proposal is
    committed here.
    """

    if not isinstance(reducer, EmbodiedStateReducer):
        raise TypeError("reducer must be an EmbodiedStateReducer")
    if not isinstance(provider_config, BowlPlaceProviderConfig):
        raise TypeError("provider_config must be a BowlPlaceProviderConfig")
    state = reducer.state
    bowl = state.entity(provider_config.bowl_entity_id)
    plate = state.entity(provider_config.plate_entity_id)
    if bowl is None or plate is None:
        missing = [
            entity_id
            for entity_id, entity in (
                (provider_config.bowl_entity_id, bowl),
                (provider_config.plate_entity_id, plate),
            )
            if entity is None
        ]
        raise BowlPlaceHandoffError(
            "place-only handoff is missing registered entities: " + ", ".join(missing)
        )
    if bowl.track_id != provider_config.bowl_track_id:
        raise BowlPlaceHandoffError("registered bowl track differs from provider configuration")
    if plate.track_id != provider_config.plate_track_id:
        raise BowlPlaceHandoffError("registered plate track differs from provider configuration")
    attachment = state.attachment
    if attachment.entity_id != provider_config.bowl_entity_id:
        raise BowlPlaceHandoffError("current attachment is not bound to the admitted bowl")
    if attachment.status not in {
        AttachmentStatus.ATTEMPTED,
        AttachmentStatus.VERIFIED_HELD,
    }:
        raise BowlPlaceHandoffError(
            "place-only handoff requires attempted or verified_held attachment state"
        )
    if not attachment.action_id.strip():
        raise BowlPlaceHandoffError("place-only handoff attachment has no admitted action_id")

    attempted_effect_id = ""
    for event in reducer.events():
        proposal = event.get("proposal")
        if not isinstance(proposal, Mapping):
            continue
        if (
            proposal.get("kind") == "set_attachment"
            and proposal.get("attachment_status") == AttachmentStatus.ATTEMPTED.value
            and proposal.get("entity_id") == provider_config.bowl_entity_id
            and proposal.get("action_id") == attachment.action_id
        ):
            attempted_effect_id = str(proposal.get("effect_id") or "")
    if not attempted_effect_id:
        raise BowlPlaceHandoffError(
            "attachment state is not backed by a matching durable attempted transition"
        )
    return BowlPlaceHandoff(
        episode_id=state.episode_id,
        state_revision=state.revision,
        bowl_entity_revision=bowl.revision,
        plate_entity_revision=plate.revision,
        attachment_revision=attachment.revision,
        attachment_status=attachment.status,
        action_id=attachment.action_id,
        attempted_effect_id=attempted_effect_id,
    )


def preflight_bowl_place_handoff(
    config: FixedBowlPlaceApplicationConfig,
) -> BowlPlaceHandoff:
    """Read and validate the place-only handoff before materializing providers."""

    if not isinstance(config, FixedBowlPlaceApplicationConfig):
        raise TypeError("config must be a FixedBowlPlaceApplicationConfig")
    return verify_bowl_place_handoff(
        _load_existing_handoff_reducer(config),
        provider_config=config.provider,
    )


def build_fixed_bowl_place_application(
    *,
    config: FixedBowlPlaceApplicationConfig,
    action_backends: Sequence[ActionBackendBinding],
    observation_backends: Sequence[ObservationBackendBinding],
    capx_executor_factory: ExecutorFactory,
    coding_policy: CompletionPolicy | None = None,
    coding_policy_factory: PolicyFactory | None = None,
    schema_registry: SchemaRegistry | None = None,
    _trusted_extensions: TrustedBowlApplicationExtensions | None = None,
) -> FixedBowlPlaceApplication:
    """Build a complete, runnable and immutable fixed bowl-place baseline."""

    if not isinstance(config, FixedBowlPlaceApplicationConfig):
        raise TypeError("config must be a FixedBowlPlaceApplicationConfig")
    if not callable(capx_executor_factory):
        raise TypeError("capx_executor_factory must be callable")
    if (coding_policy is None) == (coding_policy_factory is None):
        raise ValueError("supply exactly one bounded coding policy or policy factory")
    action_bindings = tuple(action_backends)
    observation_bindings = tuple(observation_backends)
    extensions = _trusted_extensions or TrustedBowlApplicationExtensions()
    if not isinstance(extensions, TrustedBowlApplicationExtensions):
        raise TypeError("_trusted_extensions must be TrustedBowlApplicationExtensions")

    _validate_protocol_provider_config(config.protocol, config.provider)
    protocol = extensions.protocol_override or build_fixed_bowl_place_protocol(config.protocol)
    _validate_action_backend_coverage(
        config.protocol,
        protocol,
        action_bindings,
        checker_pins=config.feasibility_checker_pins,
    )
    _validate_observation_backend_coverage(config.provider, observation_bindings)
    backend_pins = runtime_backend_pins(
        action_bindings=action_bindings,
        observation_bindings=observation_bindings,
        shadow_bindings=extensions.shadow_backends,
    )

    deterministic_provider = BowlPlaceActorProvider(config.provider)
    deterministic_bindings = build_bowl_place_actor_bindings(
        protocol.spec,
        provider=deterministic_provider,
        provider_id=config.deterministic_provider_id,
    )
    coding_bindings = build_bowl_place_coding_profiles(
        protocol.spec,
        provider_id=config.coding_provider_id,
    )
    provider_overlap = {
        config.deterministic_provider_id,
        config.coding_provider_id,
    }.intersection(extensions.actor_providers)
    if provider_overlap:
        raise BowlApplicationAssemblyError(
            "extension providers overwrite fixed providers: " + ", ".join(sorted(provider_overlap))
        )
    profiles = _merge_profiles(
        deterministic_bindings,
        coding_bindings,
        extra_profiles=extensions.actor_profiles,
    )
    _validate_extension_skill_profiles(extensions.skill_actor_profiles, profiles)

    envelope = graph_budget_envelope(protocol.compiled)
    assert_run_budgets_cover(
        config.budgets,
        envelope,
        supplemental_wall_time_s=(
            config.task_limits.max_planner_calls * config.planner_budget.wall_time_ms / 1000.0
        ),
        required_recoveries=config.manager_limits.max_reactivations,
    )

    # Validate the inherited reducer before creating run identity or provider
    # state.  Absence is an error, not permission to seed a synthetic pick.
    reducer = _load_existing_handoff_reducer(config)
    handoff = verify_bowl_place_handoff(reducer, provider_config=config.provider)

    library = extensions.skill_library_override or build_episode_bowl_skill_library(
        config.episode_root,
        subdirectory=config.skill_library_subdirectory,
    )
    _validate_episode_bowl_skill_library(config, library)
    catalog = build_bowl_place_contract_catalog(
        library,
        coding_bindings,
        additional_compatible_actor_profiles=extensions.skill_actor_profiles,
    )
    effective_api_docs = _effective_coding_api_docs(config)
    coding_provider = extensions.coding_provider_override or build_bowl_skill_coding_provider(
        config=config,
        library=library,
        capx_executor_factory=capx_executor_factory,
        coding_policy=coding_policy,
        coding_policy_factory=coding_policy_factory,
    )
    _validate_bowl_skill_coding_provider(
        config=config,
        library=library,
        provider=coding_provider,
        capx_executor_factory=capx_executor_factory,
        coding_policy=coding_policy,
        coding_policy_factory=coding_policy_factory,
    )

    schemas = schema_registry or SchemaRegistry()
    if schema_registry is None:
        from robomex.data import core_schema_registry

        schemas = core_schema_registry()
    install_runtime_schemas(schemas)

    graph_digest = f"sha256:{protocol.compiled.digest}"
    metadata = _manifest_metadata(
        config=config,
        protocol=protocol,
        profiles=profiles,
        catalog=catalog,
        handoff=handoff,
        effective_api_docs=effective_api_docs,
        coding_policy_binding=("shared" if coding_policy is not None else "per_actor_factory"),
    )
    metadata["arena_binding_digests"] = {
        binding.binding_id: binding.content_digest
        for binding in sorted(extensions.arena_bindings, key=lambda value: value.binding_id)
    }
    metadata_overlap = set(metadata).intersection(extensions.manifest_metadata)
    if metadata_overlap:
        raise BowlApplicationAssemblyError(
            "trusted extension metadata cannot overwrite fixed manifest keys: "
            + ", ".join(sorted(metadata_overlap))
        )
    metadata.update(extensions.manifest_metadata)
    manifest_values: dict[str, Any] = {
        "run_id": config.run_id,
        "task": config.task,
        "seed": config.seed,
        "model": config.model,
        "prompts": config.prompts,
        "skills": catalog_skill_pins(catalog),
        "functions": catalog_function_pins(catalog),
        "budgets": config.budgets,
        "backends": backend_pins,
        "graph_digest": graph_digest,
        "contract_catalog_digest": catalog.content_digest,
        "schema_registry_digest": schemas.content_digest,
        "runtime_code_digest": config.runtime_code_digest,
        "actor_profile_pins": runtime_actor_profile_pins(profiles),
        "metadata": metadata,
    }
    if extensions.candidate_config_digest is not None:
        manifest_values["candidate_config_digest"] = extensions.candidate_config_digest
    manifest_values["created_at"] = config.created_at
    manifest = RunManifest(**manifest_values)
    dependencies = V2RuntimeDependencies(
        contract_catalog=catalog,
        actor_providers={
            config.deterministic_provider_id: deterministic_provider,
            config.coding_provider_id: coding_provider,
            **extensions.actor_providers,
        },
        actor_profiles=profiles,
        action_backends=action_bindings,
        observation_backends=observation_bindings,
        shadow_backends=extensions.shadow_backends,
        arena_bindings=extensions.arena_bindings,
        schema_registry=schemas,
        freshness_context_provider_id=config.deterministic_provider_id,
    )
    runtime_config = V2RuntimeConfig(
        run_id=config.run_id,
        episode_id=config.episode_id,
        episode_root=config.episode_root,
        graph_digest=graph_digest,
        runtime_code_digest=config.runtime_code_digest,
        actor_namespace_root=config.actor_namespace_root,
    )
    application = V2RuntimeFactory().build(
        config=runtime_config,
        manifest=manifest,
        dependencies=dependencies,
    )
    # Close the TOCTOU gap between the preflight reducer load and application
    # construction.  This remains read-only and catches a concurrent drop/open.
    current_handoff = verify_bowl_place_handoff(
        application.episode.state_reducer,
        provider_config=config.provider,
    )
    if current_handoff != handoff:
        raise BowlPlaceHandoffError("place-only handoff changed during application assembly")

    manager = PinnedBaselineManagerInvoker(
        protocol.compiled,
        frontier=protocol.recovery_frontier,
    )
    author = ManagerWorkflowAuthor(
        manager,
        limits=config.manager_limits,
        catalog_refs=(catalog.content_digest,),
    )
    planner = OutcomeAwareFixedIntentPlanner(config.intent)
    agent = RoboMExV2Agent(
        V2AgentConfig(
            application=application,
            planner=planner,
            author=author,
            max_intents=config.task_limits.max_intents,
            max_planner_calls=config.task_limits.max_planner_calls,
            max_workflow_frontiers=config.task_limits.max_workflow_frontiers,
            planner_budget=config.planner_budget,
        )
    )
    return FixedBowlPlaceApplication(
        agent=agent,
        application=application,
        manifest=application.manifest,
        protocol=protocol,
        deterministic_provider=deterministic_provider,
        coding_provider=coding_provider,
        deterministic_bindings=deterministic_bindings,
        coding_bindings=coding_bindings,
        actor_profiles=profiles,
        dependencies=dependencies,
        skill_library=library,
        contract_catalog=catalog,
        handoff=current_handoff,
        graph_budget_envelope=envelope,
    )


def _require_digest(label: str, value: str) -> None:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical sha256 digest")


def _validate_episode_bowl_skill_library(
    config: FixedBowlPlaceApplicationConfig,
    library: SkillLibrary,
) -> None:
    expected_root = (
        config.episode_root / config.skill_library_subdirectory
    ).resolve()
    if library.root.resolve() != expected_root:
        raise BowlApplicationAssemblyError(
            "bowl SkillLibrary root differs from the sealed episode configuration"
        )
    admitted = tuple(sorted(record.skill_id for record in library.all()))
    if admitted != BOWL_PLACE_SKILL_IDS:
        raise BowlApplicationAssemblyError(
            "bowl SkillLibrary must contain exactly the five admitted production skills"
        )


def _validate_bowl_skill_coding_provider(
    *,
    config: FixedBowlPlaceApplicationConfig,
    library: SkillLibrary,
    provider: SkillCodingAgentProvider,
    capx_executor_factory: ExecutorFactory,
    coding_policy: CompletionPolicy | None,
    coding_policy_factory: PolicyFactory | None,
) -> None:
    """Reject a preassembled provider whose authority or identity drifted."""

    ceiling = config.protocol.coding_budget
    expected_artifacts_root = (
        config.episode_root / config.coding_artifacts_subdirectory
    ).resolve()
    mismatches: list[str] = []
    if provider.library is not library:
        mismatches.append("library_identity")
    if provider.executor_factory is not capx_executor_factory:
        mismatches.append("executor_factory_identity")
    if provider._shared_policy is not coding_policy:  # noqa: SLF001 - trusted assembly
        mismatches.append("shared_policy_identity")
    if provider._policy_factory is not coding_policy_factory:  # noqa: SLF001
        mismatches.append("policy_factory_identity")
    if provider.max_turns != ceiling.model_calls:
        mismatches.append("max_turns")
    if provider.max_model_calls != ceiling.model_calls:
        mismatches.append("max_model_calls")
    if provider.max_tokens != ceiling.tokens:
        mismatches.append("max_tokens")
    if provider.max_wall_time_ms != ceiling.wall_time_ms:
        mismatches.append("max_wall_time_ms")
    if provider.environment != config.coding_environment:
        mismatches.append("environment")
    if provider.api_docs != _effective_coding_api_docs(config):
        mismatches.append("api_docs")
    if provider.trusted_import_roots != config.trusted_import_roots:
        mismatches.append("trusted_import_roots")
    if provider.trusted_skill_sidecars != frozenset(BOWL_PLACE_SKILL_IDS):
        mismatches.append("trusted_skill_sidecars")
    expected_grants = {
        key: frozenset(value) for key, value in DEFAULT_CODING_CAPABILITY_GRANTS.items()
    }
    if dict(provider.capability_grants) != expected_grants:
        mismatches.append("capability_grants")
    if provider._configured_artifacts_root != expected_artifacts_root:  # noqa: SLF001
        mismatches.append("artifacts_root")
    if provider._data_plane is not None:  # noqa: SLF001
        mismatches.append("prebound_data_plane")
    if provider._runtimes:  # noqa: SLF001
        mismatches.append("existing_actor_runtimes")
    if mismatches:
        raise BowlApplicationAssemblyError(
            "shared bowl coding provider drifted from sealed configuration: "
            + ", ".join(mismatches)
        )


def _validate_protocol_provider_config(
    protocol: BowlPlaceProtocolConfig,
    provider: BowlPlaceProviderConfig,
) -> None:
    mismatches: list[str] = []
    for label, left, right in (
        ("authority_world_id", protocol.authority_world_id, provider.authority_world_id),
        ("gripper_resource_id", protocol.gripper_resource_id, provider.gripper_resource_id),
        (
            "controller_resource_id",
            protocol.controller_resource_id,
            provider.controller_resource_id,
        ),
        ("settle_duration_s", protocol.settle_duration_s, provider.settle_duration_s),
        ("tolerance_xy_m", protocol.tolerance_xy_m, provider.tolerance.translation_xy_m),
        ("tolerance_z_m", protocol.tolerance_z_m, provider.tolerance.translation_z_m),
        ("tolerance_yaw_rad", protocol.tolerance_yaw_rad, provider.tolerance.yaw_rad),
        (
            "max_step_translation_m",
            protocol.max_step_translation_m,
            provider.correction_limits.max_step_translation_m,
        ),
        (
            "max_step_yaw_rad",
            protocol.max_step_yaw_rad,
            provider.correction_limits.max_step_yaw_rad,
        ),
        (
            "max_cumulative_translation_m",
            protocol.max_cumulative_translation_m,
            provider.correction_limits.max_cumulative_translation_m,
        ),
        (
            "max_cumulative_yaw_rad",
            protocol.max_cumulative_yaw_rad,
            provider.correction_limits.max_cumulative_yaw_rad,
        ),
        (
            "max_alignment_iterations",
            protocol.max_alignment_iterations,
            provider.correction_limits.max_iterations,
        ),
    ):
        if left != right:
            mismatches.append(label)
    if provider.tracking_mode is not BowlTrackingMode.CONTINUOUS:
        mismatches.append("tracking_mode")
    if mismatches:
        raise BowlApplicationAssemblyError(
            "protocol/provider configuration drift: " + ", ".join(sorted(mismatches))
        )


def _validate_action_backend_coverage(
    config: BowlPlaceProtocolConfig,
    protocol: BowlPlaceProtocolAssembly,
    bindings: tuple[ActionBackendBinding, ...],
    *,
    checker_pins: Mapping[str, FeasibilityCheckerPin],
) -> None:
    expected = {
        (config.authority_world_id, config.arm_resource_id),
        (config.authority_world_id, config.gripper_resource_id),
        (config.authority_world_id, config.controller_resource_id),
    }
    graph_resources = {
        (node.authority_world_id, node.authoritative_resource)
        for node in protocol.spec.activations
        if node.authority_world_id is not None or node.authoritative_resource is not None
    }
    if graph_resources != expected:
        raise BowlApplicationAssemblyError(
            "protocol override changes the fixed authoritative resource inventory"
        )
    actual = [(binding.world_id, binding.resource_id) for binding in bindings]
    if len(actual) != len(set(actual)):
        raise BowlApplicationAssemblyError("authoritative action bindings contain duplicates")
    if set(actual) != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(set(actual).difference(expected))
        raise BowlApplicationAssemblyError(
            f"fixed bowl protocol requires exact authoritative resource coverage; "
            f"missing={missing}, extra={extra}"
        )
    expected_resources = {
        config.arm_resource_id,
        config.gripper_resource_id,
        config.controller_resource_id,
    }
    if set(checker_pins) != expected_resources:
        raise BowlApplicationAssemblyError(
            "feasibility checker pins must exactly cover arm, gripper and controller resources"
        )
    identities: dict[str, FeasibilityCheckerPin] = {}
    for binding in bindings:
        pin = checker_pins[binding.resource_id]
        actual_checker_id = getattr(binding.feasibility_checker, "checker_id", None)
        if actual_checker_id != pin.checker_id:
            raise BowlApplicationAssemblyError(
                f"feasibility checker for {binding.resource_id!r} does not match its "
                "operator-supplied checker_id"
            )
        previous = identities.get(pin.checker_id)
        if previous is not None and previous != pin:
            raise BowlApplicationAssemblyError(
                f"feasibility checker {pin.checker_id!r} has conflicting provenance pins"
            )
        identities[pin.checker_id] = pin
    arm = next(
        binding
        for binding in bindings
        if (binding.world_id, binding.resource_id)
        == (config.authority_world_id, config.arm_resource_id)
    )
    if arm.backend.descriptor.motion_interface is not BackendMotionInterface.EXACT_JOINT_PATH:
        raise BowlApplicationAssemblyError(
            "bowl arm backend must execute the sealed exact joint path without pose reinterpretation"
        )
    backend = arm.backend
    if not isinstance(backend, MonitorTelemetryBackend):
        raise BowlApplicationAssemblyError("arm backend lacks typed monitor telemetry")
    if not isinstance(backend, CooperativeMotionBackend):
        raise BowlApplicationAssemblyError("arm backend lacks cooperative exact-path execution")
    if not isinstance(backend, WatchdogControllableBackend):
        raise BowlApplicationAssemblyError("arm backend lacks the runtime-owned stop surface")
    try:
        raw = backend.monitor_telemetry_capabilities(
            world_id=config.authority_world_id,
            resource_id=config.arm_resource_id,
        )
        capabilities = MonitorTelemetryCapabilities.model_validate(
            raw.model_dump(mode="python") if isinstance(raw, MonitorTelemetryCapabilities) else raw
        )
    except Exception as exc:
        raise BowlApplicationAssemblyError(
            f"arm monitor capability lookup failed: {type(exc).__name__}"
        ) from exc
    if (capabilities.world_id, capabilities.resource_id) != (
        config.authority_world_id,
        config.arm_resource_id,
    ):
        raise BowlApplicationAssemblyError("arm monitor capabilities bind another resource")
    program = protocol.attachment_monitor_program.spec
    if program.hook is not MonitorHook.CONTROL:
        raise BowlApplicationAssemblyError("bowl attachment monitor must use the control hook")
    required_hook = MonitorTelemetryHook(program.hook.value)
    if required_hook not in capabilities.supported_hooks:
        raise BowlApplicationAssemblyError("arm backend does not support the monitor control hook")
    missing = sorted(set(program.allowed_signals).difference(capabilities.always_available_signals))
    if missing:
        raise BowlApplicationAssemblyError(
            "arm backend does not guarantee monitor signals: " + ", ".join(missing)
        )
    if not capabilities.cooperative_stop_guaranteed:
        raise BowlApplicationAssemblyError(
            "control-hook monitoring requires callback-triggered physical stop"
        )


def _validate_observation_backend_coverage(
    config: BowlPlaceProviderConfig,
    bindings: tuple[ObservationBackendBinding, ...],
) -> None:
    ids = [binding.provenance.backend_id for binding in bindings]
    if ids != [config.observation_backend_id]:
        raise BowlApplicationAssemblyError(
            "fixed bowl provider requires exactly its configured observation backend"
        )


def _merge_profiles(
    deterministic: BowlPlaceActorBindings,
    coding: BowlPlaceCodingBindings,
    *,
    extra_profiles: Mapping[str, ActorProfile] | None = None,
) -> dict[str, ActorProfile]:
    overlap = set(deterministic.profiles).intersection(coding.profiles)
    if overlap:
        raise BowlApplicationAssemblyError(
            "deterministic/coding runner coverage overlaps: " + ", ".join(sorted(overlap))
        )
    base = {**deterministic.profiles, **coding.profiles}
    extra = extra_profiles or {}
    extension_overlap = set(base).intersection(extra)
    if extension_overlap:
        raise BowlApplicationAssemblyError(
            "extension runner refs overwrite fixed profiles: "
            + ", ".join(sorted(extension_overlap))
        )
    profiles = {**base, **extra}
    if not profiles:
        raise BowlApplicationAssemblyError("fixed bowl graph has no actor profiles")
    return profiles


def _validate_extension_skill_profiles(
    skill_actor_profiles: Mapping[str, tuple[str, ...]],
    profiles: Mapping[str, ActorProfile],
) -> None:
    unknown_skills = sorted(set(skill_actor_profiles).difference(BOWL_PLACE_SKILL_IDS))
    if unknown_skills:
        raise BowlApplicationAssemblyError(
            "extension skill compatibility references unknown skills: " + ", ".join(unknown_skills)
        )
    admitted_profile_ids = {profile.profile_id for profile in profiles.values()}
    unknown_profiles = sorted(
        {
            profile_id
            for profile_ids in skill_actor_profiles.values()
            for profile_id in profile_ids
            if profile_id not in admitted_profile_ids
        }
    )
    if unknown_profiles:
        raise BowlApplicationAssemblyError(
            "extension skill compatibility references unavailable actor profiles: "
            + ", ".join(unknown_profiles)
        )


def _load_existing_handoff_reducer(
    config: FixedBowlPlaceApplicationConfig,
) -> EmbodiedStateReducer:
    required = (
        config.episode_root / "episode_manifest.v2.json",
        config.episode_root / "embodied_state_events.jsonl",
    )
    if any(not path.is_file() for path in required):
        raise BowlPlaceHandoffError(
            "place-only assembly requires an existing durable reducer from grounding/pick"
        )
    try:
        return EmbodiedStateReducer(
            config.episode_root,
            episode_id=config.episode_id,
            strict_evidence=True,
        )
    except Exception as exc:
        raise BowlPlaceHandoffError(
            f"cannot validate prior reducer state: {type(exc).__name__}: {exc}"
        ) from exc


def _effective_coding_api_docs(config: FixedBowlPlaceApplicationConfig) -> str:
    pin = (
        "RoboMEx immutable run context:\n"
        f"robot_model_digest={config.robot_model_digest}\n"
        "MotionPlan.robot_model_digest must equal that exact value."
    )
    if not config.coding_api_docs.strip():
        return pin
    return f"{config.coding_api_docs.rstrip()}\n\n{pin}"


def _manifest_metadata(
    *,
    config: FixedBowlPlaceApplicationConfig,
    protocol: BowlPlaceProtocolAssembly,
    profiles: Mapping[str, ActorProfile],
    catalog: ContractCatalogSnapshot,
    handoff: BowlPlaceHandoff,
    effective_api_docs: str,
    coding_policy_binding: str,
) -> dict[str, Any]:
    protocol_config = _json_value(config.protocol)
    provider_config = _json_value(config.provider)
    task_limits = config.task_limits.model_dump(mode="json")
    planner_budget = config.planner_budget.model_dump(mode="json")
    manager_limits = config.manager_limits.model_dump(mode="json")
    intent = config.intent.model_dump(mode="json")
    tracking_config = {
        "mode": config.provider.tracking_mode.value,
        "poll_interval_s": config.provider.tracking_poll_interval_s,
        "max_silence_s": config.provider.tracking_max_silence_s,
        "capture_wait_timeout_s": config.provider.tracking_capture_wait_timeout_s,
        "max_pair_skew_s": config.provider.tracking_max_pair_skew_s,
        "pair_buffer_size": config.provider.tracking_pair_buffer_size,
        "bowl_track_id": config.provider.bowl_track_id,
        "plate_track_id": config.provider.plate_track_id,
        "observation_backend_id": config.provider.observation_backend_id,
    }
    coding_config = {
        "provider_id": config.coding_provider_id,
        "required_skill_ids": list(BOWL_PLACE_SKILL_IDS),
        "required_function_ids": list(BOWL_PLACE_FUNCTION_IDS),
        "profile_ids": sorted({profile.profile_id for profile in profiles.values()}),
        "max_turns": config.protocol.coding_budget.model_calls,
        "max_model_calls": config.protocol.coding_budget.model_calls,
        "max_tokens": config.protocol.coding_budget.tokens,
        "max_wall_time_ms": config.protocol.coding_budget.wall_time_ms,
        "policy_binding": coding_policy_binding,
        "executor_boundary": "capx_executor_factory",
        "environment_digest": _text_digest(config.coding_environment),
        "api_docs_digest": _text_digest(effective_api_docs),
        "trusted_import_roots": sorted(config.trusted_import_roots),
        "skill_catalog_digest": catalog.content_digest,
    }
    task_config = {
        "task_snapshot_digest": config.task.content_digest,
        "intent": intent,
        "intent_digest": canonical_payload_digest("robomex.fixed_bowl.intent.v1", intent),
        "limits": task_limits,
    }
    manager_config = {
        "kind": "pinned_baseline",
        "uses_model_calls": False,
        "limits": manager_limits,
        "graph_digest": f"sha256:{protocol.compiled.digest}",
        "recovery_frontier_digest": canonical_payload_digest(
            "robomex.composable_frontier.v1",
            protocol.recovery_frontier.model_dump(mode="json"),
        ),
    }
    planner_config = {
        "kind": "outcome_aware_fixed_intent",
        "uses_model_calls": False,
        "call_budget": planner_budget,
        "max_calls": config.task_limits.max_planner_calls,
        "intent_digest": task_config["intent_digest"],
    }
    checker_identities: dict[str, dict[str, Any]] = {}
    checker_bindings: dict[str, str] = {}
    for resource_id, pin in sorted(config.feasibility_checker_pins.items()):
        identity = pin.model_dump(mode="json")
        previous = checker_identities.get(pin.checker_id)
        if previous is not None and previous != identity:
            raise BowlApplicationAssemblyError(
                f"feasibility checker {pin.checker_id!r} has conflicting provenance pins"
            )
        checker_identities[pin.checker_id] = identity
        checker_bindings[f"{config.protocol.authority_world_id}/{resource_id}"] = pin.checker_id
    checker_config = {
        "identities": checker_identities,
        "resource_bindings": checker_bindings,
    }
    return {
        "assembly_schema_version": "robomex.fixed_bowl_application.v1",
        "allowed_initial_graph_digests": [],
        "arena_binding_digests": {},
        "robot_model_digest": config.robot_model_digest,
        "feasibility_checker_config": checker_config,
        "feasibility_checker_config_digest": canonical_payload_digest(
            "robomex.fixed_bowl.feasibility_checkers.v1", checker_config
        ),
        "protocol_config": protocol_config,
        "protocol_config_digest": canonical_payload_digest(
            "robomex.fixed_bowl.protocol_config.v1", protocol_config
        ),
        "provider_config": provider_config,
        "provider_config_digest": canonical_payload_digest(
            "robomex.fixed_bowl.provider_config.v1", provider_config
        ),
        "tracking_config": tracking_config,
        "tracking_config_digest": canonical_payload_digest(
            "robomex.fixed_bowl.tracking_config.v1", tracking_config
        ),
        "coding_config": coding_config,
        "coding_config_digest": canonical_payload_digest(
            "robomex.fixed_bowl.coding_config.v1", coding_config
        ),
        "task_config": task_config,
        "task_config_digest": canonical_payload_digest(
            "robomex.fixed_bowl.task_config.v1", task_config
        ),
        "manager_config": manager_config,
        "manager_config_digest": canonical_payload_digest(
            "robomex.fixed_bowl.manager_config.v1", manager_config
        ),
        "planner_config": planner_config,
        "planner_config_digest": canonical_payload_digest(
            "robomex.fixed_bowl.planner_config.v1", planner_config
        ),
        "task_orchestration_limits": task_limits,
        "planner_call_budget": planner_budget,
        "manager_limits": manager_limits,
        "place_handoff": _json_value(handoff),
    }


def _text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(item) for item in value)
    return value


__all__ = [
    "BOWL_PLACE_FUNCTION_IDS",
    "BOWL_PLACE_SKILL_IDS",
    "BOWL_PLACE_SKILL_PACKAGES",
    "BowlApplicationAssemblyError",
    "BowlPlaceHandoff",
    "BowlPlaceHandoffError",
    "BowlPlaceProtocolAssembly",
    "FeasibilityCheckerPin",
    "FixedBowlPlaceApplication",
    "FixedBowlPlaceApplicationConfig",
    "TrustedBowlApplicationExtensions",
    "build_bowl_place_contract_catalog",
    "build_bowl_skill_coding_provider",
    "build_episode_bowl_skill_library",
    "build_fixed_bowl_place_application",
    "preflight_bowl_place_handoff",
    "verify_bowl_place_handoff",
]
