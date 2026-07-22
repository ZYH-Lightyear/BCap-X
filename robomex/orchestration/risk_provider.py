"""Deterministic, fail-closed motion-risk provider for RoboMEx v2.

The provider is intentionally smaller than the Arena.  It consumes one exact,
already-admitted physical evidence chain, validates that the chain still names
one bowl, one target, and one synchronized observation, and emits the canonical
``RiskReport`` that controls the Arena's one-to-K expansion.  It never consults
an alias or ``latest`` value and has no model, tool, or physical-effect path.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from robomex.data.artifact_resolver import (
    ResolvedArtifactRef,
    compute_content_digest,
)
from robomex.data.embodied_state import AttachmentStatus
from robomex.data.episode_plane import ArtifactRecord, EpisodeDataPlane
from robomex.data.physical_schema import (
    AdmissionPurpose,
    AttachmentEvidence,
)
from robomex.manipulation.bowl_place import (
    AlignmentError,
    AlignmentStatus,
    BowlPlaceObservation,
    ServoDecision,
    ServoOutcome,
    VisibilityStatus,
    compute_alignment_error,
)
from robomex.orchestration.actors import (
    ActorIsolation,
    ActorLifecycle,
    ActorProfile,
    InvocationSpec,
)
from robomex.orchestration.arena import CheckStatus, RiskInputs, RiskPolicy, RiskReport
from robomex.orchestration.episode import ActivationExecutionResult, ArtifactEmission
from robomex.runtime.events import ControlOutcome

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

DETERMINISTIC_MOTION_RISK_RUNNER_REF = "robomex.motion_risk.assess"
"""Stable graph runner reference for the deterministic risk provider."""

MOTION_RISK_RUNNER_REF = DETERMINISTIC_MOTION_RISK_RUNNER_REF
MOTION_RISK_PROVIDER_ID = "motion_risk"
MOTION_RISK_READ_CAPABILITY = "episode.artifact.read"
MOTION_RISK_REPORT_SCHEMA_ID = "robomex.risk_report.v1"

_OBSERVATION_SCHEMA_ID = "robomex.bowl_place_observation.v1"
_ATTACHMENT_SCHEMA_ID = "robomex.attachment_evidence.v1"
_ALIGNMENT_SCHEMA_ID = "robomex.alignment_error.v1"
_SERVO_SCHEMA_ID = "robomex.servo_decision.v1"
_INPUT_ORDER = (
    "observation",
    "attachment_evidence",
    "alignment_error",
    "servo_decision",
)
MOTION_RISK_INPUT_SCHEMAS: Mapping[str, str] = MappingProxyType(
    {
        "observation": _OBSERVATION_SCHEMA_ID,
        "attachment_evidence": _ATTACHMENT_SCHEMA_ID,
        "alignment_error": _ALIGNMENT_SCHEMA_ID,
        "servo_decision": _SERVO_SCHEMA_ID,
    }
)
MOTION_RISK_CAPABILITIES = frozenset({MOTION_RISK_READ_CAPABILITY})


class MotionRiskProviderError(RuntimeError):
    """Base error for deterministic risk-provider admission and lifecycle."""


class MotionRiskContractError(MotionRiskProviderError):
    """One or more inputs cannot prove a single coherent physical state."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        str_strip_whitespace=True,
        validate_default=True,
    )


class MotionRiskProviderConfig(_StrictModel):
    """Manifest-pinnable deterministic inputs not supplied as artifacts.

    IK, collision, and clearance values are deliberately fixed here.  A graph
    author or Agent cannot replace them through invocation metadata.  The
    profile builder records the complete config and policy digests in the
    ``ActorProfile`` that is pinned by the run manifest.
    """

    schema_version: Literal["robomex.motion_risk_provider_config.v1"] = (
        "robomex.motion_risk_provider_config.v1"
    )
    expected_bowl_entity_id: NonEmptyStr
    expected_target_entity_id: NonEmptyStr
    risk_policy: RiskPolicy = Field(default_factory=RiskPolicy)
    fixed_ik_status: CheckStatus = CheckStatus.UNKNOWN
    fixed_collision_status: CheckStatus = CheckStatus.UNKNOWN
    fixed_clearance_m: float | None = Field(
        default=None,
        ge=0.0,
        allow_inf_nan=False,
    )
    candidate_disagreement_m: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
    )
    prior_failures_before_servo: int = Field(default=0, ge=0)
    monitor_observable: bool = True

    @model_validator(mode="after")
    def _distinct_entities(self) -> MotionRiskProviderConfig:
        if self.expected_bowl_entity_id == self.expected_target_entity_id:
            raise ValueError("motion risk requires distinct bowl and target entities")
        return self

    @property
    def content_digest(self) -> str:
        return compute_content_digest(
            self.schema_version,
            self.model_dump(mode="json"),
        )

    @property
    def risk_policy_digest(self) -> str:
        return compute_content_digest(
            "robomex.risk_policy.v1",
            self.risk_policy.model_dump(mode="json"),
        )


@dataclass
class _MotionRiskRuntime:
    profile: ActorProfile
    isolation: ActorIsolation
    suspended: bool = False
    retired: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


@dataclass(frozen=True)
class _ResolvedRiskEvidence:
    refs: Mapping[str, ResolvedArtifactRef]
    records: Mapping[str, ArtifactRecord]
    observation: BowlPlaceObservation
    attachment: AttachmentEvidence
    alignment: AlignmentError
    servo: ServoDecision


@dataclass(frozen=True)
class MotionRiskActorBindings:
    """Bootstrap-ready provider/profile maps with no effect-bearing authority."""

    provider_id: str
    providers: Mapping[str, DeterministicMotionRiskProvider]
    profiles: Mapping[str, ActorProfile]


class DeterministicMotionRiskProvider:
    """Read-only ``AgentProvider`` that derives one canonical ``RiskReport``."""

    def __init__(self, config: MotionRiskProviderConfig) -> None:
        if not isinstance(config, MotionRiskProviderConfig):
            raise TypeError("config must be a MotionRiskProviderConfig")
        self.config = config
        self._data_plane: EpisodeDataPlane | None = None
        self._lock = threading.RLock()

    @property
    def data_plane(self) -> EpisodeDataPlane:
        """Return the exact episode plane admitted by the runtime."""

        return self._require_data_plane()

    def bind_episode_data_plane(self, data_plane: EpisodeDataPlane) -> None:
        if not isinstance(data_plane, EpisodeDataPlane):
            raise TypeError("data_plane must be an EpisodeDataPlane")
        with self._lock:
            if self._data_plane is not None and self._data_plane is not data_plane:
                raise MotionRiskProviderError(
                    "motion-risk provider is already bound to another EpisodeDataPlane"
                )
            self._data_plane = data_plane

    def spawn(
        self,
        profile: ActorProfile,
        isolation: ActorIsolation,
    ) -> _MotionRiskRuntime:
        self._validate_profile(profile, isolation)
        return _MotionRiskRuntime(profile=profile, isolation=isolation)

    def invoke(
        self,
        runtime: _MotionRiskRuntime,
        spec: InvocationSpec,
    ) -> ActivationExecutionResult:
        if not isinstance(runtime, _MotionRiskRuntime):
            raise TypeError("runtime was not spawned by DeterministicMotionRiskProvider")
        with runtime.lock:
            if runtime.retired:
                raise MotionRiskProviderError("motion-risk actor is retired")
            if runtime.suspended:
                raise MotionRiskProviderError("motion-risk actor is suspended")
            try:
                self._validate_invocation(runtime.profile, spec)
                evidence = self._resolve_evidence(spec)
                self._validate_evidence(evidence, spec)
                report = self._assess(evidence)
            except MotionRiskProviderError:
                raise
            except Exception as exc:  # noqa: BLE001 - fail-closed provider boundary
                raise MotionRiskContractError(
                    f"motion-risk evidence admission failed: {exc}"
                ) from exc

            lineage = tuple(evidence.refs[name] for name in _INPUT_ORDER)
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(
                    ArtifactEmission(
                        port="risk_report",
                        schema_id=MOTION_RISK_REPORT_SCHEMA_ID,
                        payload=report.model_dump(mode="json"),
                        lineage=lineage,
                    ),
                ),
                reason=(
                    "deterministic motion risk assessed from one synchronized "
                    "physical-evidence chain"
                ),
            )

    def suspend(self, runtime: _MotionRiskRuntime) -> None:
        with runtime.lock:
            if runtime.retired:
                raise MotionRiskProviderError("cannot suspend a retired motion-risk actor")
            runtime.suspended = True

    def resume(self, runtime: _MotionRiskRuntime) -> None:
        with runtime.lock:
            if runtime.retired:
                raise MotionRiskProviderError("cannot resume a retired motion-risk actor")
            runtime.suspended = False

    def retire(self, runtime: _MotionRiskRuntime) -> None:
        with runtime.lock:
            runtime.retired = True
            runtime.suspended = False

    def _validate_profile(
        self,
        profile: ActorProfile,
        isolation: ActorIsolation,
    ) -> None:
        if profile.runner_kind != "deterministic_gate":
            raise MotionRiskProviderError(
                "motion-risk profile runner_kind must be deterministic_gate"
            )
        if profile.lifecycle is not ActorLifecycle.EPHEMERAL:
            raise MotionRiskProviderError("motion-risk actors must be invocation-scoped")
        if profile.model:
            raise MotionRiskProviderError("motion-risk actors cannot bind an LLM model")
        if profile.effect_ceiling:
            raise MotionRiskProviderError("motion-risk actors require an empty effect ceiling")
        if profile.capability_ceiling != MOTION_RISK_CAPABILITIES:
            raise MotionRiskProviderError(
                "motion-risk actors may hold only the exact artifact-read capability"
            )
        if isolation.world_id is not None:
            raise MotionRiskProviderError("motion-risk actors cannot bind a writable world")
        expected_metadata = {
            "runner_ref": DETERMINISTIC_MOTION_RISK_RUNNER_REF,
            "provider_config_digest": self.config.content_digest,
            "risk_policy_digest": self.config.risk_policy_digest,
        }
        for name, expected in expected_metadata.items():
            if profile.metadata.get(name) != expected:
                raise MotionRiskProviderError(
                    f"motion-risk ActorProfile has an invalid {name!r} manifest pin"
                )
        if profile.metadata.get("risk_policy") != self.config.risk_policy.model_dump(mode="json"):
            raise MotionRiskProviderError(
                "motion-risk ActorProfile policy differs from provider config"
            )
        if profile.metadata.get("provider_config") != self.config.model_dump(mode="json"):
            raise MotionRiskProviderError(
                "motion-risk ActorProfile config differs from the installed provider"
            )

    def _validate_invocation(self, profile: ActorProfile, spec: InvocationSpec) -> None:
        if spec.requested_effects:
            raise MotionRiskContractError("motion-risk invocation cannot request effects")
        if spec.requested_capabilities != MOTION_RISK_CAPABILITIES:
            raise MotionRiskContractError(
                "motion-risk invocation requires only the exact artifact-read capability"
            )
        if profile.effect_ceiling or profile.model:
            raise MotionRiskContractError("motion-risk profile acquired forbidden authority")
        if set(spec.inputs) != set(_INPUT_ORDER):
            raise MotionRiskContractError(
                "motion-risk invocation requires exactly observation, attachment_evidence, "
                "alignment_error, and servo_decision refs"
            )
        if dict(spec.output_contract) != {"risk_report": MOTION_RISK_REPORT_SCHEMA_ID}:
            raise MotionRiskContractError(
                "motion-risk output contract must contain one risk_report.v1 artifact"
            )
        if any(
            float(spec.budget.get(name, 0.0)) != 0.0
            for name in ("model_calls", "tokens", "llm_calls")
        ):
            raise MotionRiskContractError("motion-risk invocation cannot receive an LLM budget")
        node_params = spec.metadata.get("node_params")
        if node_params not in (None, {}):
            raise MotionRiskContractError(
                "graph node params cannot override manifest-pinned motion-risk inputs"
            )

    def _resolve_evidence(self, spec: InvocationSpec) -> _ResolvedRiskEvidence:
        plane = self._require_data_plane()
        refs: dict[str, ResolvedArtifactRef] = {}
        records: dict[str, ArtifactRecord] = {}
        payloads: dict[str, Any] = {}
        seen: set[tuple[str, str]] = set()
        models = {
            "observation": BowlPlaceObservation,
            "attachment_evidence": AttachmentEvidence,
            "alignment_error": AlignmentError,
            "servo_decision": ServoDecision,
        }
        for name in _INPUT_ORDER:
            ref = self._exact_ref(spec.inputs[name], name=name)
            identity = self._ref_identity(ref)
            if identity in seen:
                raise MotionRiskContractError(
                    f"motion-risk input {name!r} reuses another port's artifact ref"
                )
            seen.add(identity)
            resolved = plane.resolve(ref)
            expected_schema = MOTION_RISK_INPUT_SCHEMAS[name]
            if resolved.ref != ref or resolved.schema != expected_schema:
                raise MotionRiskContractError(
                    f"motion-risk input {name!r} is not exact schema {expected_schema!r}"
                )
            record = plane.artifact_record(ref.artifact_id)
            if record.ref != ref or record.schema != expected_schema:
                raise MotionRiskContractError(
                    f"motion-risk input {name!r} record identity does not match its ref"
                )
            refs[name] = ref
            records[name] = record
            payloads[name] = models[name].model_validate(resolved.payload)
        return _ResolvedRiskEvidence(
            refs=MappingProxyType(refs),
            records=MappingProxyType(records),
            observation=payloads["observation"],
            attachment=payloads["attachment_evidence"],
            alignment=payloads["alignment_error"],
            servo=payloads["servo_decision"],
        )

    def _validate_evidence(
        self,
        evidence: _ResolvedRiskEvidence,
        spec: InvocationSpec,
    ) -> None:
        plane = self._require_data_plane()
        episode_id = spec.metadata.get("episode_id")
        workflow_id = spec.metadata.get("workflow_id")
        if episode_id != plane.episode_id:
            raise MotionRiskContractError(
                "motion-risk invocation episode identity differs from its data plane"
            )
        if not isinstance(workflow_id, str) or not workflow_id.strip():
            raise MotionRiskContractError("motion-risk invocation requires workflow_id metadata")
        if any(
            record.episode_id != plane.episode_id or record.workflow_id != workflow_id
            for record in evidence.records.values()
        ):
            raise MotionRiskContractError(
                "motion-risk inputs must belong to one invocation workflow and episode"
            )

        self._validate_lineage(evidence, workflow_id=workflow_id)
        self._validate_physical_identity(evidence)

    def _validate_lineage(
        self,
        evidence: _ResolvedRiskEvidence,
        *,
        workflow_id: str,
    ) -> None:
        plane = self._require_data_plane()
        for name, record in evidence.records.items():
            identities: set[tuple[str, str]] = set()
            for ref in record.lineage:
                if self._ref_identity(ref) in identities:
                    raise MotionRiskContractError(
                        f"motion-risk input {name!r} contains duplicate lineage"
                    )
                identities.add(self._ref_identity(ref))
                plane.resolve(ref)
                if plane.artifact_record(ref.artifact_id).workflow_id != workflow_id:
                    raise MotionRiskContractError(
                        f"motion-risk input {name!r} crosses workflow lineage"
                    )

        observation_ref = evidence.refs["observation"]
        observation_record = evidence.records["observation"]
        attachment_record = evidence.records["attachment_evidence"]
        expected_attachment_lineage = self._dedupe_refs(
            (observation_ref,),
            observation_record.lineage,
        )
        if attachment_record.lineage != expected_attachment_lineage:
            raise MotionRiskContractError(
                "attachment evidence lineage is not the observation's exact causal chain"
            )
        if evidence.attachment.evidence_refs != tuple(
            ref.artifact_id for ref in expected_attachment_lineage
        ):
            raise MotionRiskContractError(
                "attachment evidence_refs differ from its immutable artifact lineage"
            )

        expected_alignment = {
            self._ref_identity(evidence.refs["observation"]),
            self._ref_identity(evidence.refs["attachment_evidence"]),
        }
        actual_alignment = {
            self._ref_identity(ref) for ref in evidence.records["alignment_error"].lineage
        }
        if actual_alignment != expected_alignment or len(
            evidence.records["alignment_error"].lineage
        ) != len(expected_alignment):
            raise MotionRiskContractError(
                "alignment error lineage must cover exactly observation and attachment evidence"
            )

        required_servo = expected_alignment | {self._ref_identity(evidence.refs["alignment_error"])}
        actual_servo = {
            self._ref_identity(ref) for ref in evidence.records["servo_decision"].lineage
        }
        if actual_servo != required_servo or len(evidence.records["servo_decision"].lineage) != len(
            required_servo
        ):
            raise MotionRiskContractError(
                "servo decision lineage must cover exactly its physical decision inputs"
            )

    def _validate_physical_identity(self, evidence: _ResolvedRiskEvidence) -> None:
        observation = evidence.observation
        attachment = evidence.attachment
        alignment = evidence.alignment
        servo = evidence.servo
        config = self.config

        if (
            observation.expected_bowl_entity_id != config.expected_bowl_entity_id
            or observation.expected_target_entity_id != config.expected_target_entity_id
        ):
            raise MotionRiskContractError(
                "observation entity contract differs from the manifest-pinned risk contract"
            )
        if observation.held is None or observation.target is None:
            raise MotionRiskContractError(
                "motion risk requires visible held-bowl and target geometry"
            )
        held = observation.held
        target = observation.target
        if (
            observation.held_visibility is not VisibilityStatus.VISIBLE
            or observation.target_visibility is not VisibilityStatus.VISIBLE
            or held.entity_id != config.expected_bowl_entity_id
            or target.entity_id != config.expected_target_entity_id
        ):
            raise MotionRiskContractError(
                "visible geometry does not match the manifest-pinned entity identities"
            )
        if (
            observation.attachment_status is not AttachmentStatus.VERIFIED_HELD
            or attachment.status != AttachmentStatus.VERIFIED_HELD.value
            or attachment.entity_id != config.expected_bowl_entity_id
        ):
            raise MotionRiskContractError(
                "motion risk requires matching verified-held attachment evidence"
            )
        if attachment.source_observation_id != observation.snapshot_id:
            raise MotionRiskContractError(
                "attachment evidence belongs to a different observation snapshot"
            )
        try:
            observed_camera_revision = observation.revisions.domain_revision(
                attachment.observation_domain
            )
        except Exception as exc:  # noqa: BLE001 - normalize physical schema failure
            raise MotionRiskContractError(
                "attachment evidence names an unavailable observation domain"
            ) from exc
        if attachment.source_observation_revision != observed_camera_revision:
            raise MotionRiskContractError(
                "attachment evidence camera revision differs from the observation"
            )
        expected_validity_revisions = {
            attachment.observation_domain: attachment.source_observation_revision
        }
        if attachment.validity.depends_on_revisions != expected_validity_revisions:
            raise MotionRiskContractError(
                "attachment validity does not bind the exact observation revision"
            )
        if not math.isclose(
            attachment.confidence,
            attachment.validity.confidence,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise MotionRiskContractError("attachment evidence and validity confidence differ")
        attachment.validity.assert_admissible(
            observation.revisions,
            purpose=AdmissionPurpose.VERIFICATION,
            current_observation_id=observation.snapshot_id,
            current_state_revision=attachment.base_state_revision,
        )

        common_clock = (
            observation.snapshot_id,
            observation.observation_generation,
            observation.revisions,
        )
        alignment_clock = (
            alignment.snapshot_id,
            alignment.observation_generation,
            alignment.revisions,
        )
        if alignment_clock != common_clock:
            raise MotionRiskContractError(
                "alignment error is stale or belongs to a different physical revision"
            )
        if (
            alignment.bowl_entity_id != config.expected_bowl_entity_id
            or alignment.target_entity_id != config.expected_target_entity_id
            or alignment.held_estimate_id != held.estimate_id
            or alignment.target_id != target.target_id
            or alignment.source_ref != held.estimate_id
            or alignment.target_ref != target.target_id
            or alignment.expressed_in_frame != observation.frame_id
        ):
            raise MotionRiskContractError(
                "alignment identity does not match the synchronized geometry"
            )
        canonical_alignment = compute_alignment_error(
            held,
            target,
            tolerance=alignment.tolerance,
            expected_bowl_entity_id=config.expected_bowl_entity_id,
            expected_target_entity_id=config.expected_target_entity_id,
        ).model_copy(update={"alignment_id": alignment.alignment_id})
        if canonical_alignment != alignment:
            raise MotionRiskContractError(
                "alignment payload is not the canonical measurement of its observation"
            )

        if servo.observation_generation != observation.observation_generation:
            raise MotionRiskContractError(
                "servo decision generation differs from the synchronized observation"
            )
        servo_alignment = servo.alignment_error
        if servo_alignment is None or (
            servo_alignment.model_copy(update={"alignment_id": alignment.alignment_id}) != alignment
        ):
            raise MotionRiskContractError(
                "servo decision does not embed the admitted alignment measurement"
            )
        if servo.outcome is not ServoOutcome.CORRECTION_REQUIRED:
            raise MotionRiskContractError(
                "only correction_required may enter motion-risk expansion"
            )
        if alignment.status is not AlignmentStatus.CORRECTION_REQUIRED:
            raise MotionRiskContractError(
                "servo correction contradicts a within-tolerance alignment"
            )
        correction = servo.correction
        if correction is None or (
            correction.expressed_in_frame != observation.frame_id
            or correction.source_snapshot_id != observation.snapshot_id
            or correction.source_generation != observation.observation_generation
            or correction.source_revisions != observation.revisions
            or correction.required_next_generation != observation.observation_generation + 1
        ):
            raise MotionRiskContractError(
                "servo correction is not bound to the admitted physical clock"
            )

    def _assess(self, evidence: _ResolvedRiskEvidence) -> RiskReport:
        observation = evidence.observation
        held = observation.held
        target = observation.target
        if held is None or target is None:  # Defensive narrowing after validation.
            raise MotionRiskContractError("validated observation lost its geometry")
        correction = evidence.servo.correction
        if correction is None:  # Defensive narrowing after validation.
            raise MotionRiskContractError("validated servo decision lost its correction")
        iteration = correction.iteration
        inputs = RiskInputs(
            grounding_confidence=min(
                held.confidence,
                target.confidence,
                evidence.attachment.confidence,
            ),
            target_margin_m=max(0.0, evidence.alignment.support_clearance_m),
            ik_status=self.config.fixed_ik_status,
            collision_status=self.config.fixed_collision_status,
            clearance_m=self.config.fixed_clearance_m,
            held_pose_uncertainty_m=held.uncertainty.translation_std_m.norm,
            candidate_disagreement_m=self.config.candidate_disagreement_m,
            prior_failures=(self.config.prior_failures_before_servo + max(0, iteration - 1)),
            monitor_observable=(
                self.config.monitor_observable
                and observation.held_visibility is VisibilityStatus.VISIBLE
                and observation.target_visibility is VisibilityStatus.VISIBLE
                and attachment_is_observable(evidence.attachment)
            ),
        )
        return RiskReport.assess(inputs, self.config.risk_policy)

    def _require_data_plane(self) -> EpisodeDataPlane:
        with self._lock:
            if self._data_plane is None:
                raise MotionRiskProviderError(
                    "motion-risk provider is not bound to an EpisodeDataPlane"
                )
            return self._data_plane

    @staticmethod
    def _exact_ref(value: Any, *, name: str) -> ResolvedArtifactRef:
        if isinstance(value, ResolvedArtifactRef):
            return value
        if not isinstance(value, Mapping) or set(value) != {
            "artifact_id",
            "content_digest",
        }:
            raise MotionRiskContractError(
                f"motion-risk input {name!r} requires an exact artifact_id+content_digest ref"
            )
        return ResolvedArtifactRef.from_any(value)

    @staticmethod
    def _ref_identity(ref: ResolvedArtifactRef) -> tuple[str, str]:
        return (ref.artifact_id, ref.content_digest)

    @classmethod
    def _dedupe_refs(
        cls,
        *groups: Sequence[ResolvedArtifactRef],
    ) -> tuple[ResolvedArtifactRef, ...]:
        values: list[ResolvedArtifactRef] = []
        seen: set[tuple[str, str]] = set()
        for group in groups:
            for ref in group:
                identity = cls._ref_identity(ref)
                if identity not in seen:
                    seen.add(identity)
                    values.append(ref)
        return tuple(values)


def attachment_is_observable(evidence: AttachmentEvidence) -> bool:
    """Closed, deterministic observability projection for held-object motion."""

    return (
        evidence.status == AttachmentStatus.VERIFIED_HELD.value
        and evidence.confidence > 0.0
        and evidence.validity.confidence > 0.0
    )


def build_motion_risk_actor_profile(
    config: MotionRiskProviderConfig,
    *,
    provider_id: str = MOTION_RISK_PROVIDER_ID,
) -> ActorProfile:
    """Build the exact manifest-pinnable profile for the risk runner."""

    if not isinstance(config, MotionRiskProviderConfig):
        raise TypeError("config must be a MotionRiskProviderConfig")
    if not isinstance(provider_id, str) or not provider_id.strip():
        raise ValueError("provider_id must be a non-empty string")
    digest_suffix = config.content_digest.removeprefix("sha256:")[:16]
    return ActorProfile(
        profile_id=f"motion-risk-{digest_suffix}",
        provider_id=provider_id,
        runner_kind="deterministic_gate",
        lifecycle=ActorLifecycle.EPHEMERAL,
        model="",
        capability_ceiling=MOTION_RISK_CAPABILITIES,
        effect_ceiling=frozenset(),
        metadata={
            "runner_ref": DETERMINISTIC_MOTION_RISK_RUNNER_REF,
            "provider_config_digest": config.content_digest,
            "provider_config": config.model_dump(mode="json"),
            "risk_policy_digest": config.risk_policy_digest,
            "risk_policy": config.risk_policy.model_dump(mode="json"),
            "llm_access": False,
            "action_authority": False,
        },
    )


def build_motion_risk_actor_bindings(
    config: MotionRiskProviderConfig,
    *,
    provider: DeterministicMotionRiskProvider | None = None,
    provider_id: str = MOTION_RISK_PROVIDER_ID,
) -> MotionRiskActorBindings:
    """Return explicit provider/profile mappings for ``V2RuntimeDependencies``."""

    actual_provider = provider or DeterministicMotionRiskProvider(config)
    if not isinstance(actual_provider, DeterministicMotionRiskProvider):
        raise TypeError("provider must be a DeterministicMotionRiskProvider")
    if actual_provider.config != config:
        raise MotionRiskProviderError(
            "motion-risk binding config differs from the provider's sealed config"
        )
    profile = build_motion_risk_actor_profile(config, provider_id=provider_id)
    return MotionRiskActorBindings(
        provider_id=provider_id,
        providers=MappingProxyType({provider_id: actual_provider}),
        profiles=MappingProxyType({DETERMINISTIC_MOTION_RISK_RUNNER_REF: profile}),
    )


__all__ = [
    "DETERMINISTIC_MOTION_RISK_RUNNER_REF",
    "MOTION_RISK_CAPABILITIES",
    "MOTION_RISK_INPUT_SCHEMAS",
    "MOTION_RISK_PROVIDER_ID",
    "MOTION_RISK_READ_CAPABILITY",
    "MOTION_RISK_REPORT_SCHEMA_ID",
    "MOTION_RISK_RUNNER_REF",
    "DeterministicMotionRiskProvider",
    "MotionRiskActorBindings",
    "MotionRiskContractError",
    "MotionRiskProviderConfig",
    "MotionRiskProviderError",
    "attachment_is_observable",
    "build_motion_risk_actor_bindings",
    "build_motion_risk_actor_profile",
]
