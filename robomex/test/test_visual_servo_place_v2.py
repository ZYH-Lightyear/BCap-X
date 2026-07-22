from __future__ import annotations

import importlib.util
import json
import random
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from robomex.authoring.monitoring import MonitorHook, MonitorRuntime
from robomex.core.sandbox import (
    ActionBlockStatus,
    BlockExecutionResult,
    SemanticActionBlock,
)
from robomex.data import (
    AdmissionFreshnessContext,
    AdmissionPurpose,
    AttachmentEvidence,
    AttachmentStatus,
    PhysicalStateTrigger,
    RelationEvidence,
    RelationPredicate,
    RelationValue,
    ResolvedArtifactRef,
    RevisionVector,
    SchemaRegistry,
    StateCommitReceipt,
    StateTransitionProposal,
    StateTransitionProposalWire,
    ValidityLifecycle,
    ValidityVector,
)
from robomex.elastic import EffectScope, LifecycleScope, RunnerKind
from robomex.manipulation import (
    AlignmentError,
    AlignmentStatus,
    AlignmentTolerance,
    AttachmentGuard,
    AttachmentGuardPhase,
    AttachmentStateSnapshot,
    BowlPlaceObservation,
    CheckpointPhase,
    CheckpointStatus,
    CorrectionLimits,
    EntityMismatchError,
    FrameMismatchError,
    HeldBowlEstimate,
    PhaseCheckpoint,
    PlacementPhase,
    PlacementVerdict,
    PlacementVerdictStatus,
    PlateSupportTarget,
    PoseUncertainty,
    QuaternionWXYZ,
    RecoveryDisposition,
    RecoveryEffectScope,
    RecoverySafetyDecision,
    RelationAssessment,
    RevisionMismatchError,
    ServoDecision,
    ServoOutcome,
    SupportFootprint,
    UnitVector3,
    Vector3,
    VisibilityStatus,
    VisualServoPlacer,
    apply_correction_to_estimate,
    compute_alignment_error,
    fixed_offset_baseline,
    observation_revision_from_physical,
    register_bowl_place_schemas,
    revision_vector_from_observation,
)
from robomex.orchestration.actors import (
    ActorLifecycle,
    ActorProfile,
    ActorRegistry,
    InMemoryAgentProvider,
    InvocationSpec,
)
from robomex.orchestration.bowl_provider import (
    BowlPlaceActorProvider,
    BowlPlaceProviderConfig,
    BowlPlaceProviderError,
    BowlTrackingMode,
    SynchronizedBowlTrackSampler,
    build_bowl_place_actor_bindings,
    build_bowl_place_coding_profiles,
)
from robomex.orchestration.coding_provider import (
    CodingNodeConfigV1,
    CodingProviderContractError,
    SkillCodingAgentProvider,
)
from robomex.orchestration.episode import (
    ActivationExecutionResult,
    ArtifactEmission,
    EpisodeRuntime,
)
from robomex.orchestration.intent import SubgoalIntent
from robomex.protocols.bowl_place import (
    EXECUTION_RECEIPT,
    MONITOR_PROGRAM,
    BowlPlaceProtocolConfig,
    build_bowl_attachment_monitor_program,
    build_fixed_bowl_place_protocol,
)
from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    BackendCallResult,
    BackendDescriptor,
    BackendMotionInterface,
    ExecutionPolicy,
    ExecutionReceipt,
    ExecutionStatus,
    FeasibilityStatus,
    GripperCommand,
    JointPath,
    MonitorTelemetryCapabilities,
    MonitorTelemetryHook,
    MotionPlan,
    WaitSpec,
)
from robomex.runtime.authority import build_feasibility_certificate
from robomex.runtime.events import ControlOutcome, MonitorFindingKind, NodeOutcomeEvent
from robomex.runtime.observation import (
    BackendObservation,
    EntityIdentity,
    InMemoryObservationBackend,
    ObservationQuality,
    ObservationRegistry,
    ObservationRevisionVector,
    TrackRequest,
    TrackState,
)
from robomex.skills import Skill, SkillLibrary

_BUILTIN_SKILL_ROOT = Path(__file__).parents[1] / "skills" / "builtin"


def _load_builtin_sidecar(relative_path: str, module_name: str):
    path = _BUILTIN_SKILL_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _revisions(value: int) -> RevisionVector:
    return RevisionVector(
        scene=value,
        arm=value,
        gripper=value,
        attachment=value,
        camera={"front": value},
    )


def _footprint(center: Vector3, radius: float = 0.035) -> SupportFootprint:
    return SupportFootprint(
        vertices_xy_m=(
            (center.x - radius, center.y - radius),
            (center.x + radius, center.y - radius),
            (center.x + radius, center.y + radius),
            (center.x - radius, center.y + radius),
        )
    )


def _uncertainty() -> PoseUncertainty:
    return PoseUncertainty(
        translation_std_m=Vector3(x=0.0008, y=0.0008, z=0.001),
        orientation_std_rad=0.006,
    )


def _held(
    *,
    center: Vector3,
    yaw: float,
    generation: int = 1,
    snapshot: str | None = None,
    revisions: RevisionVector | None = None,
    entity_id: str = "bowl-1",
    frame_id: str = "world",
) -> HeldBowlEstimate:
    return HeldBowlEstimate(
        entity_id=entity_id,
        frame_id=frame_id,
        snapshot_id=snapshot or f"snapshot-{generation}",
        observation_generation=generation,
        revisions=revisions or _revisions(generation),
        bottom_center_m=center,
        support_footprint=_footprint(center),
        orientation=QuaternionWXYZ.from_yaw(yaw),
        uncertainty=_uncertainty(),
        confidence=0.94,
        evidence_refs=(f"frame-{generation}",),
    )


def _target(
    *,
    center: Vector3,
    yaw: float = 0.0,
    generation: int = 1,
    snapshot: str | None = None,
    revisions: RevisionVector | None = None,
    entity_id: str = "plate-1",
    frame_id: str = "world",
) -> PlateSupportTarget:
    return PlateSupportTarget(
        entity_id=entity_id,
        frame_id=frame_id,
        snapshot_id=snapshot or f"snapshot-{generation}",
        observation_generation=generation,
        revisions=revisions or _revisions(generation),
        support_center_m=center,
        support_footprint=_footprint(center, radius=0.09),
        surface_normal=UnitVector3(x=0.0, y=0.0, z=1.0),
        orientation=QuaternionWXYZ.from_yaw(yaw),
        uncertainty=_uncertainty(),
        safe_margin_m=0.01,
        confidence=0.96,
        evidence_refs=(f"plate-frame-{generation}",),
    )


def _observation(
    held: HeldBowlEstimate | None,
    target: PlateSupportTarget | None,
    *,
    attachment: AttachmentStatus = AttachmentStatus.VERIFIED_HELD,
    held_visibility: VisibilityStatus = VisibilityStatus.VISIBLE,
    target_visibility: VisibilityStatus = VisibilityStatus.VISIBLE,
    expected_bowl: str = "bowl-1",
    expected_target: str = "plate-1",
    generation: int | None = None,
    snapshot: str | None = None,
    revisions: RevisionVector | None = None,
    frame_id: str = "world",
) -> BowlPlaceObservation:
    reference = held or target
    assert reference is not None or generation is not None
    actual_generation = generation or reference.observation_generation
    actual_snapshot = snapshot or reference.snapshot_id
    actual_revisions = revisions or reference.revisions
    return BowlPlaceObservation(
        snapshot_id=actual_snapshot,
        observation_generation=actual_generation,
        revisions=actual_revisions,
        frame_id=frame_id,
        expected_bowl_entity_id=expected_bowl,
        expected_target_entity_id=expected_target,
        attachment_status=attachment,
        held_visibility=held_visibility,
        target_visibility=target_visibility,
        held=held,
        target=target,
    )


def _refresh_target(
    target: PlateSupportTarget,
    *,
    generation: int,
    revisions: RevisionVector,
    center: Vector3 | None = None,
) -> PlateSupportTarget:
    next_center = center or target.support_center_m
    return PlateSupportTarget(
        target_id=target.target_id,
        entity_id=target.entity_id,
        frame_id=target.frame_id,
        snapshot_id=f"snapshot-{generation}",
        observation_generation=generation,
        revisions=revisions,
        support_center_m=next_center,
        support_footprint=_footprint(next_center, radius=0.09),
        surface_normal=target.surface_normal,
        orientation=target.orientation,
        uncertainty=target.uncertainty,
        safe_margin_m=target.safe_margin_m,
        confidence=target.confidence,
        evidence_refs=target.evidence_refs,
    )


def _sha256(char: str) -> str:
    return "sha256:" + char * 64


def _admission_snapshot(resource_id: str) -> AdmissionSnapshot:
    return AdmissionSnapshot(
        world_id="authoritative",
        resource_id=resource_id,
        robot_revision=1,
        scene_revision=1,
        attachment_revision=1,
        config_revision=1,
        joint_names=("joint-a", "joint-b"),
        joint_positions_rad=(0.0, 0.1),
        config_digest=_sha256("a"),
        collision_world_digest=_sha256("b"),
        attachment_status="verified_held",
        controller_state="ready",
        # Admission evidence is intentionally live: EpisodeRuntime enforces a
        # short freshness window before an authoritative action can run.
        captured_at=datetime.now(UTC),
    )


class _BowlFakeBackend:
    descriptor = BackendDescriptor(
        backend_id="bowl-replay-backend",
        motion_interface=BackendMotionInterface.EXACT_JOINT_PATH,
    )

    def __init__(self, *, scenario: str = "normal") -> None:
        self.scenario = scenario
        self.snapshots = {
            resource: _admission_snapshot(resource)
            for resource in ("robot.arm", "robot.gripper", "robot.controller")
        }
        self.calls: list[tuple[str, str]] = []
        self.cooperative_actions = 0
        self.monitor_sample_requests: list[tuple[str, int]] = []

    def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot:
        assert world_id == "authoritative"
        snapshot = self.snapshots[resource_id].model_copy(update={"captured_at": datetime.now(UTC)})
        self.snapshots[resource_id] = snapshot
        return snapshot

    def monitor_telemetry_capabilities(
        self, *, world_id: str, resource_id: str
    ) -> MonitorTelemetryCapabilities:
        return MonitorTelemetryCapabilities(
            world_id=world_id,
            resource_id=resource_id,
            always_available_signals=(
                "attachment_status",
                "held_entity_visible",
                "identity_match",
            ),
            supported_hooks=(MonitorTelemetryHook.CONTROL,),
            cooperative_stop_guaranteed=True,
        )

    def _advance(
        self,
        resource_id: str,
        *,
        joints: tuple[float, ...] | None = None,
        attachment_status: str | None = None,
    ) -> None:
        previous = self.snapshots[resource_id]
        self.snapshots[resource_id] = AdmissionSnapshot(
            world_id=previous.world_id,
            world_kind=previous.world_kind,
            resource_id=previous.resource_id,
            robot_revision=previous.robot_revision + 1,
            scene_revision=previous.scene_revision + 1,
            attachment_revision=previous.attachment_revision + 1,
            config_revision=previous.config_revision,
            joint_names=previous.joint_names,
            joint_positions_rad=joints or previous.joint_positions_rad,
            config_digest=previous.config_digest,
            collision_world_digest=previous.collision_world_digest,
            attachment_status=attachment_status or previous.attachment_status,
            controller_state="ready",
            captured_at=datetime.now(UTC),
        )

    def execute_joint_path(self, **kwargs: object) -> BackendCallResult:
        resource_id = str(kwargs["resource_id"])
        positions = kwargs["positions_rad"]
        assert isinstance(positions, tuple)
        endpoint = tuple(float(value) for value in positions[-1])
        self.calls.append(("execute_joint_path", resource_id))
        self._advance(resource_id, joints=endpoint)
        return BackendCallResult(converged=True)

    @staticmethod
    def _held_sample() -> dict[str, object]:
        return {
            "attachment_status": "verified_held",
            "held_entity_visible": True,
            "identity_match": True,
        }

    def _fault_sample(self) -> dict[str, object] | None:
        if self.scenario == "dropped":
            return {
                "attachment_status": "not_held",
                "held_entity_visible": True,
                "identity_match": True,
            }
        if self.scenario == "occluded":
            return {
                "attachment_status": "verified_held",
                "held_entity_visible": False,
                "identity_match": True,
            }
        return None

    def monitor_sample(self, **kwargs: object) -> dict[str, object]:
        phase = str(kwargs["phase"])
        sequence = int(kwargs["sequence"])
        self.monitor_sample_requests.append((phase, sequence))
        return self._held_sample()

    def stop_and_hold(self, *, resource_id: str) -> None:
        """Record the controller's two-step cooperative interruption protocol."""

        self.calls.append(("stop", resource_id))
        self.calls.append(("hold", resource_id))

    def stop_and_wait_quiescent(
        self,
        *,
        world_id: str,
        resource_id: str,
        timeout_s: float,
    ) -> AdmissionSnapshot:
        assert world_id == "authoritative" and timeout_s > 0
        self.stop_and_hold(resource_id=resource_id)
        snapshot = self.snapshots[resource_id].model_copy(
            update={
                "controller_state": "quiescent",
                "captured_at": datetime.now(UTC),
            }
        )
        self.snapshots[resource_id] = snapshot
        return snapshot

    def execute_joint_path_cooperative(
        self, *, progress_callback, **kwargs: object
    ) -> BackendCallResult:
        resource_id = str(kwargs["resource_id"])
        positions = kwargs["positions_rad"]
        assert isinstance(positions, tuple)
        endpoint = tuple(float(value) for value in positions[-1])
        self.cooperative_actions += 1
        self.calls.append(("execute_joint_path_cooperative", resource_id))

        samples = [self._held_sample()]
        fault = self._fault_sample() if self.cooperative_actions == 1 else None
        samples.append(fault or self._held_sample())
        for index, sample in enumerate(samples, start=1):
            if not progress_callback(sample):
                self.stop_and_hold(resource_id=resource_id)
                if sample["attachment_status"] == "not_held":
                    self._advance(resource_id, attachment_status="not_held")
                return BackendCallResult(
                    converged=False,
                    interrupted=True,
                    telemetry={"stopped_at_control_sample": index},
                )

        self._advance(resource_id, joints=endpoint)
        return BackendCallResult(
            converged=True,
            telemetry={"control_samples": len(samples)},
        )

    def set_gripper(self, **kwargs: object) -> BackendCallResult:
        resource_id = str(kwargs["resource_id"])
        self.calls.append(("set_gripper", resource_id))
        self._advance(resource_id, attachment_status="unknown")
        return BackendCallResult(converged=True)

    def wait(self, **kwargs: object) -> BackendCallResult:
        resource_id = str(kwargs["resource_id"])
        self.calls.append(("wait", resource_id))
        self._advance(resource_id)
        return BackendCallResult(converged=True)


class _AlwaysFeasible:
    def certify(self, spec, snapshot):
        return build_feasibility_certificate(
            spec=spec,
            snapshot=snapshot,
            checker_id="bowl-fake-ik-collision-checker",
            checks={
                "exact_joint_path_interface": FeasibilityStatus.PASS,
                "controller_admissible": FeasibilityStatus.PASS,
                "resource_binding": FeasibilityStatus.PASS,
                "ik": FeasibilityStatus.PASS,
                "collision": FeasibilityStatus.PASS,
                "joint_order": FeasibilityStatus.PASS,
                "joint_limits": FeasibilityStatus.PASS,
                "robot_model": FeasibilityStatus.PASS,
                "gripper_target_range": FeasibilityStatus.PASS,
                "gripper_effort_support": FeasibilityStatus.PASS,
                "wait_timeout_bound": FeasibilityStatus.PASS,
            },
        )


class _BowlEpisodeDriver:
    """Deterministic provider backed by the real observation registry boundary."""

    _BOWL_SIGNALS = (
        "attachment_status",
        "center_x",
        "center_y",
        "center_z",
        "identity_match",
        "visibility",
        "yaw",
    )
    _PLATE_SIGNALS = (
        "center_x",
        "center_y",
        "center_z",
        "identity_match",
        "visibility",
        "yaw",
    )

    def __init__(
        self,
        backend: _BowlFakeBackend,
        observation_backend: InMemoryObservationBackend,
        *,
        scenario: str,
    ) -> None:
        self.backend = backend
        self.observation_backend = observation_backend
        self.scenario = scenario
        self.runtime: EpisodeRuntime | None = None
        self.servo = VisualServoPlacer(
            bowl_entity_id="bowl-1",
            target_entity_id="plate-1",
            frame_id="world",
            limits=CorrectionLimits(max_iterations=4),
        )
        self.held = _held(center=Vector3(x=0.52, y=0.0, z=0.2), yaw=0.10)
        self.target = _target(center=Vector3(x=0.5, y=0.0, z=0.2))
        self.observation = _observation(self.held, self.target)
        self.decision = None
        self.pending_correction = None
        self.action_sequence = 0
        self.observation_generation = 0
        self.alignment_captures = 0
        self.observation_clock = time.monotonic()
        self.invoked: list[str] = []
        self.bowl_track = None
        self.plate_track = None

    def bind_runtime(self, runtime: EpisodeRuntime) -> None:
        self.runtime = runtime

    def freshness_context(self, purpose: AdmissionPurpose) -> AdmissionFreshnessContext:
        runtime = self._runtime()
        return AdmissionFreshnessContext(
            current_revisions=self.observation.revisions,
            purpose=purpose,
            current_observation_id=self.observation.snapshot_id,
            current_state_revision=runtime.state_reducer.state.revision,
        )

    def _runtime(self) -> EpisodeRuntime:
        assert self.runtime is not None
        return self.runtime

    @staticmethod
    def _input_refs(invocation) -> dict[str, ResolvedArtifactRef]:
        return {
            name: ResolvedArtifactRef.from_any(value) for name, value in invocation.inputs.items()
        }

    def _resolved_model(self, invocation, name: str, model):
        ref = self._input_refs(invocation)[name]
        return model.model_validate(self._runtime().data_plane.resolve(ref).payload)

    @staticmethod
    def _dedupe_refs(*groups) -> tuple[ResolvedArtifactRef, ...]:
        result: list[ResolvedArtifactRef] = []
        seen: set[tuple[str, str]] = set()
        for group in groups:
            for ref in group:
                identity = (ref.artifact_id, ref.content_digest)
                if identity not in seen:
                    seen.add(identity)
                    result.append(ref)
        return tuple(result)

    @staticmethod
    def _emission(port: str, model, *, lineage=()) -> ArtifactEmission:
        schema_id = getattr(model, "schema_version", None)
        if schema_id is None:
            schema_id = {
                AttachmentEvidence: "robomex.attachment_evidence.v1",
                RelationEvidence: "robomex.relation_evidence.v1",
            }[type(model)]
        return ArtifactEmission(
            port=port,
            schema_id=schema_id,
            payload=model.model_dump(mode="json"),
            lineage=tuple(lineage),
        )

    def _motion(self, plan_kind: str, snapshot: AdmissionSnapshot) -> MotionPlan:
        self.action_sequence += 1
        start = snapshot.joint_positions_rad
        endpoint = (start[0] + 0.01, start[1] + 0.01)
        return MotionPlan(
            plan_id=f"{plan_kind}-{self.action_sequence}",
            plan_kind=plan_kind,
            tcp_frame_id="panda-hand",
            planner_backend="deterministic-bowl-replay",
            robot_model_digest=_sha256("c"),
            expected_snapshot=snapshot,
            motion=JointPath(
                joint_names=snapshot.joint_names,
                positions_rad=(start, endpoint),
                execution_policy=ExecutionPolicy(timeout_s=2.0),
            ),
            possibly_affected_revisions=("robot.arm", "scene", "attachment"),
        )

    def _open(self, snapshot: AdmissionSnapshot) -> GripperCommand:
        self.action_sequence += 1
        return GripperCommand(
            command_id=f"open-{self.action_sequence}",
            expected_snapshot=snapshot,
            mode="open",
            target_width_m=0.08,
            timeout_s=2.0,
            possibly_affected_revisions=("robot.gripper", "attachment", "scene"),
        )

    def _wait(self, snapshot: AdmissionSnapshot) -> WaitSpec:
        self.action_sequence += 1
        return WaitSpec(
            wait_id=f"settle-{self.action_sequence}",
            expected_snapshot=snapshot,
            duration_s=0.05,
            timeout_s=0.5,
            possibly_affected_revisions=("scene",),
        )

    def _guard(self, phase: AttachmentGuardPhase) -> AttachmentGuard:
        runtime = self._runtime()
        return AttachmentGuard(
            phase=phase,
            allowed=True,
            status=AttachmentStatus.VERIFIED_HELD,
            entity_id="bowl-1",
            state_revision=runtime.state_reducer.state.revision,
            observation_generation=self.observation.observation_generation,
            reason="fresh typed evidence confirms the tracked held bowl",
            evidence_refs=(self.observation.snapshot_id,),
        )

    def _advance_observation_after_correction(self, next_generation: int) -> None:
        assert self.pending_correction is not None
        next_revisions = _revisions(next_generation)
        self.held = apply_correction_to_estimate(
            self.held,
            self.pending_correction,
            next_snapshot_id=f"snapshot-{next_generation}",
            next_generation=next_generation,
            next_revisions=next_revisions,
        )
        self.target = _refresh_target(
            self.target,
            generation=next_generation,
            revisions=next_revisions,
        )
        self.pending_correction = None

    def _next_capture_generation(self, activation_id: str) -> int:
        if activation_id == "capture_alignment":
            self.alignment_captures += 1
            if self.alignment_captures > 1 and self.scenario == "stale_generation":
                return self.observation_generation
        return self.observation_generation + 1

    def _latest_alignment_action_ref(self) -> ResolvedArtifactRef:
        runtime = self._runtime()
        records = [
            record
            for record in runtime.data_plane.artifacts
            if record.schema == EXECUTION_RECEIPT
            and record.activation_id in {"execute_transport", "execute_correction"}
        ]
        assert records
        return records[-1].ref

    def _capture_lineage(self, activation_id: str, invocation) -> tuple[ResolvedArtifactRef, ...]:
        refs = tuple(self._input_refs(invocation).values())
        if activation_id == "capture_alignment":
            refs = (self._latest_alignment_action_ref(),)
        return self._dedupe_refs(refs)

    def _push_observation_pair(
        self,
        *,
        generation: int,
        attachment: AttachmentStatus,
        held_visibility: VisibilityStatus,
    ):
        revisions = ObservationRevisionVector(
            scene_revision=generation,
            arm_revision=generation,
            gripper_revision=generation,
            attachment_revision=generation,
            camera_revision=generation,
        )
        now = datetime.now(UTC)
        self.observation_clock += 1.0
        if held_visibility is VisibilityStatus.VISIBLE:
            bowl = BackendObservation(
                entity_id="bowl-1",
                quality=ObservationQuality.TRACKED,
                revisions=revisions,
                signals={
                    "attachment_status": attachment.value,
                    "center_x": self.held.bottom_center_m.x,
                    "center_y": self.held.bottom_center_m.y,
                    "center_z": self.held.bottom_center_m.z,
                    "identity_match": True,
                    "visibility": "visible",
                    "yaw": self.held.orientation.yaw_rad,
                },
                observed_at=now,
                monotonic_time_s=self.observation_clock,
            )
        else:
            bowl = BackendObservation(
                entity_id="bowl-1",
                quality=ObservationQuality.LOST,
                revisions=revisions,
                signals={},
                reason="held_bowl_occluded_at_primary_checkpoint",
                observed_at=now,
                monotonic_time_s=self.observation_clock,
            )
        self.observation_clock += 1.0
        plate = BackendObservation(
            entity_id="plate-1",
            quality=ObservationQuality.TRACKED,
            revisions=revisions,
            signals={
                "center_x": self.target.support_center_m.x,
                "center_y": self.target.support_center_m.y,
                "center_z": self.target.support_center_m.z,
                "identity_match": True,
                "visibility": "visible",
                "yaw": self.target.orientation.yaw_rad,
            },
            observed_at=now,
            monotonic_time_s=self.observation_clock,
        )
        self.observation_backend.push(bowl)
        self.observation_backend.push(plate)
        assert self.bowl_track is not None and self.plate_track is not None
        return self.bowl_track.poll(), self.plate_track.poll()

    def _capture(self, activation_id: str, invocation) -> ActivationExecutionResult:
        next_generation = self._next_capture_generation(activation_id)
        if (
            self.pending_correction is not None
            and activation_id == "capture_alignment"
            and self.scenario != "stale_generation"
        ):
            self._advance_observation_after_correction(next_generation)
            if self.scenario == "target_drift":
                drifted = Vector3(x=0.54, y=0.0, z=0.2)
                self.target = _refresh_target(
                    self.target,
                    generation=next_generation,
                    revisions=_revisions(next_generation),
                    center=drifted,
                )
        attachment = (
            AttachmentStatus.NOT_HELD
            if activation_id in {"capture_release", "capture_final_relation"}
            else AttachmentStatus.VERIFIED_HELD
        )
        visibility = (
            VisibilityStatus.OCCLUDED
            if self.scenario == "checkpoint_occluded" and activation_id == "capture_alignment"
            else VisibilityStatus.VISIBLE
        )
        bowl_sample, plate_sample = self._push_observation_pair(
            generation=next_generation,
            attachment=attachment,
            held_visibility=visibility,
        )
        physical_revisions = revision_vector_from_observation(
            bowl_sample.revisions, camera_id="front"
        )
        snapshot_id = f"sync-{next_generation}-{bowl_sample.sequence}-{plate_sample.sequence}"
        if visibility is VisibilityStatus.VISIBLE:
            bowl_signals = bowl_sample.signals
            held = _held(
                center=Vector3(
                    x=float(bowl_signals["center_x"]),
                    y=float(bowl_signals["center_y"]),
                    z=float(bowl_signals["center_z"]),
                ),
                yaw=float(bowl_signals["yaw"]),
                generation=next_generation,
                snapshot=snapshot_id,
                revisions=physical_revisions,
            )
        else:
            held = None
        plate_signals = plate_sample.signals
        target = _target(
            center=Vector3(
                x=float(plate_signals["center_x"]),
                y=float(plate_signals["center_y"]),
                z=float(plate_signals["center_z"]),
            ),
            yaw=float(plate_signals["yaw"]),
            generation=next_generation,
            snapshot=snapshot_id,
            revisions=physical_revisions,
        )
        self.observation_generation = next_generation
        if held is not None:
            self.held = held
        self.target = target
        self.observation = _observation(
            held,
            target,
            attachment=attachment,
            held_visibility=visibility,
            generation=next_generation,
            snapshot=snapshot_id,
            revisions=physical_revisions,
        )
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                self._emission(
                    "observation",
                    self.observation,
                    lineage=self._capture_lineage(activation_id, invocation),
                ),
            ),
        )

    def _causal_refs_for_observation(
        self, observation_ref: ResolvedArtifactRef
    ) -> tuple[ResolvedArtifactRef, ...]:
        record = self._runtime().data_plane.artifact_record(observation_ref.artifact_id)
        return self._dedupe_refs((observation_ref,), record.lineage)

    def _action_id_for_refs(self, refs: tuple[ResolvedArtifactRef, ...]) -> str:
        runtime = self._runtime()
        action_records = [
            runtime.data_plane.artifact_record(ref.artifact_id)
            for ref in refs
            if runtime.data_plane.artifact_record(ref.artifact_id).schema == EXECUTION_RECEIPT
        ]
        if not action_records:
            return runtime.state_reducer.state.attachment.action_id
        open_records = [
            record for record in action_records if record.activation_id == "execute_open"
        ]
        selected = open_records[-1] if open_records else action_records[-1]
        receipt = ExecutionReceipt.model_validate(runtime.data_plane.resolve(selected.ref).payload)
        return receipt.action_id

    def _attachment_evidence(self, invocation, *, required_status: AttachmentStatus):
        runtime = self._runtime()
        refs = self._input_refs(invocation)
        observation_ref = refs["observation"]
        observation = BowlPlaceObservation.model_validate(
            runtime.data_plane.resolve(observation_ref).payload
        )
        if observation.held_visibility is not VisibilityStatus.VISIBLE:
            return ActivationExecutionResult(
                outcome=ControlOutcome.UNCERTAIN,
                reason="primary checkpoint cannot see the bowl",
            )
        if observation.attachment_status is not required_status:
            return ActivationExecutionResult(
                outcome=ControlOutcome.ATTACHMENT_NOT_CONFIRMED,
                reason="fresh observation contradicts the required attachment status",
            )
        causal_refs = self._causal_refs_for_observation(observation_ref)
        action_id = self._action_id_for_refs(causal_refs)
        revision = observation.revisions.camera["front"]
        evidence = AttachmentEvidence(
            evidence_id=f"attachment-{observation.snapshot_id}",
            entity_id="bowl-1",
            entity_track_id="track-bowl-1",
            status=required_status.value,
            source_observation_id=observation.snapshot_id,
            source_observation_revision=revision,
            observation_domain="camera.front",
            base_state_revision=runtime.state_reducer.state.revision,
            action_id=action_id,
            evidence_refs=tuple(ref.artifact_id for ref in causal_refs),
            method="synchronized_tracking_and_gripper_guard",
            reason="fresh synchronized bowl evidence matches the required attachment",
            confidence=0.98,
            validity=ValidityVector(
                lifecycle=ValidityLifecycle.DERIVED,
                depends_on_revisions={"camera.front": revision},
                observation_id=observation.snapshot_id,
                state_revision=runtime.state_reducer.state.revision,
                method="exact_observation_and_state_revision",
                reason="evidence is valid only for this checkpoint and state",
                confidence=0.98,
            ),
        )
        outputs = [self._emission("attachment_evidence", evidence, lineage=causal_refs)]
        if required_status is AttachmentStatus.VERIFIED_HELD:
            phase = {
                "verify_initial_attachment": AttachmentGuardPhase.TRANSPORT,
                "verify_alignment_attachment": AttachmentGuardPhase.CORRECTION,
                "verify_pre_release_attachment": AttachmentGuardPhase.OPEN,
            }[str(invocation.metadata["activation_id"])]
            guard = self._guard(phase)
            outputs.append(self._emission("attachment_guard", guard, lineage=causal_refs))
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=tuple(outputs),
        )

    def _attachment_proposal(self, invocation) -> ActivationExecutionResult:
        runtime = self._runtime()
        refs = self._input_refs(invocation)
        evidence = AttachmentEvidence.model_validate(
            runtime.data_plane.resolve(refs["attachment_evidence"]).payload
        )
        evidence_record = runtime.data_plane.artifact_record(
            refs["attachment_evidence"].artifact_id
        )
        proposal_refs = self._dedupe_refs(
            (refs["attachment_evidence"], refs["observation"]),
            evidence_record.lineage,
        )
        proposal = StateTransitionProposal.set_attachment(
            episode_id=runtime.episode_id,
            effect_id=(
                f"confirm-{evidence.source_observation_revision}"
                if evidence.status == AttachmentStatus.VERIFIED_HELD.value
                else f"release-{evidence.source_observation_revision}"
            ),
            before_revision=runtime.state_reducer.state.revision,
            source="bowl_attachment_verifier",
            evidence_refs=proposal_refs,
            entity_id="bowl-1",
            status=AttachmentStatus(evidence.status),
            observation_ref=refs["observation"],
            action_id=evidence.action_id or "",
            trigger=PhysicalStateTrigger.EVIDENCE,
            source_observation_id=evidence.source_observation_id,
            source_observation_revision=evidence.source_observation_revision,
            source_observation_domain=evidence.observation_domain,
            track_id="track-bowl-1",
        )
        wire = StateTransitionProposalWire.from_domain(proposal)
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                ArtifactEmission(
                    port="proposal",
                    schema_id="robomex.state_transition_proposal.v1",
                    payload=wire.model_dump(mode="json"),
                    lineage=proposal_refs,
                ),
            ),
        )

    def _relation_verdict(self, invocation) -> ActivationExecutionResult:
        runtime = self._runtime()
        observation_ref = self._input_refs(invocation)["observation"]
        observation = BowlPlaceObservation.model_validate(
            runtime.data_plane.resolve(observation_ref).payload
        )
        assert observation.held is not None and observation.target is not None
        error = compute_alignment_error(observation.held, observation.target)
        if not error.support_contained:
            return ActivationExecutionResult(
                outcome=ControlOutcome.FAILED_PLACEMENT,
                reason="independent final observation does not show support containment",
            )
        causal_refs = self._causal_refs_for_observation(observation_ref)
        action_id = self._action_id_for_refs(causal_refs)
        revision = observation.revisions.camera["front"]
        evidence = RelationEvidence(
            evidence_id=f"relation-{observation.snapshot_id}",
            subject_entity_id="bowl-1",
            subject_track_id="track-bowl-1",
            predicate=RelationPredicate.SUPPORTED_BY,
            target_entity_id="plate-1",
            target_track_id="track-plate-1",
            value=RelationValue.ASSERTED,
            source_observation_id=observation.snapshot_id,
            source_observation_revision=revision,
            observation_domain="camera.front",
            base_state_revision=runtime.state_reducer.state.revision,
            action_id=action_id,
            evidence_refs=tuple(ref.artifact_id for ref in causal_refs),
            method="independent_support_footprint_verifier",
            reason="fresh bowl footprint is contained by the plate support region",
            confidence=0.99,
            validity=ValidityVector(
                lifecycle=ValidityLifecycle.DERIVED,
                depends_on_revisions={"camera.front": revision},
                observation_id=observation.snapshot_id,
                state_revision=runtime.state_reducer.state.revision,
                method="exact_observation_and_state_revision",
                reason="relation is scoped to the independent final checkpoint",
                confidence=0.99,
            ),
        )
        verdict = PlacementVerdict(
            status=PlacementVerdictStatus.SUCCEEDED,
            bowl_entity_id="bowl-1",
            target_entity_id="plate-1",
            relation=RelationAssessment.ASSERTED,
            source_observation_id=observation.snapshot_id,
            state_revision=runtime.state_reducer.state.revision,
            support_clearance_m=error.support_clearance_m,
            confidence=0.99,
            evidence_refs=tuple(ref.artifact_id for ref in causal_refs),
            reason="independent final observation confirms support",
        )
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                self._emission("relation_evidence", evidence, lineage=causal_refs),
                self._emission("verdict", verdict, lineage=causal_refs),
            ),
        )

    def _relation_proposal(self, invocation) -> ActivationExecutionResult:
        runtime = self._runtime()
        refs = self._input_refs(invocation)
        evidence = RelationEvidence.model_validate(
            runtime.data_plane.resolve(refs["relation_evidence"]).payload
        )
        evidence_record = runtime.data_plane.artifact_record(refs["relation_evidence"].artifact_id)
        proposal_refs = self._dedupe_refs(
            (refs["relation_evidence"], refs["observation"]),
            evidence_record.lineage,
        )
        proposal = StateTransitionProposal.set_relation(
            episode_id=runtime.episode_id,
            effect_id=f"support-{evidence.source_observation_revision}",
            before_revision=runtime.state_reducer.state.revision,
            source="bowl_relation_verifier",
            evidence_refs=proposal_refs,
            subject_entity_id="bowl-1",
            predicate=RelationPredicate.SUPPORTED_BY,
            target_entity_id="plate-1",
            value=RelationValue.ASSERTED,
            source_observation_id=evidence.source_observation_id,
            source_observation_revision=evidence.source_observation_revision,
            source_observation_domain=evidence.observation_domain,
            observation_ref=refs["observation"],
            action_id=evidence.action_id or "",
            subject_track_id="track-bowl-1",
            target_track_id="track-plate-1",
        )
        wire = StateTransitionProposalWire.from_domain(proposal)
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                ArtifactEmission(
                    port="proposal",
                    schema_id="robomex.state_transition_proposal.v1",
                    payload=wire.model_dump(mode="json"),
                    lineage=proposal_refs,
                ),
            ),
        )

    def __call__(self, profile, invocation, isolation):
        del profile, isolation
        activation_id = str(invocation.metadata["activation_id"])
        self.invoked.append(activation_id)
        if activation_id in {
            "held_bowl_tracker",
            "plate_tracker",
        }:
            return None
        if activation_id == "author_attachment_monitor":
            program = build_bowl_attachment_monitor_program().spec
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(
                    ArtifactEmission(
                        port="monitor_program",
                        schema_id=program.schema_id,
                        payload=program.model_dump(mode="json"),
                    ),
                ),
            )
        if activation_id in {
            "capture_initial_attachment",
            "capture_alignment",
            "capture_pre_release",
            "capture_release",
            "capture_final_relation",
        }:
            return self._capture(activation_id, invocation)
        if activation_id in {
            "verify_initial_attachment",
            "verify_alignment_attachment",
            "verify_pre_release_attachment",
        }:
            return self._attachment_evidence(
                invocation, required_status=AttachmentStatus.VERIFIED_HELD
            )
        if activation_id == "verify_release_attachment":
            return self._attachment_evidence(invocation, required_status=AttachmentStatus.NOT_HELD)
        if activation_id in {
            "propose_initial_attachment",
            "propose_release_attachment",
        }:
            return self._attachment_proposal(invocation)
        if activation_id == "plan_transport":
            snapshot = self._resolved_model(invocation, "snapshot", AdmissionSnapshot)
            plan = self._motion("transport_to_hover", snapshot)
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(self._emission("action_spec", plan),),
            )
        if activation_id in {"estimate_alignment", "estimate_pre_release_alignment"}:
            observation = self._resolved_model(invocation, "observation", BowlPlaceObservation)
            assert observation.held is not None and observation.target is not None
            error = compute_alignment_error(observation.held, observation.target)
            if activation_id == "estimate_pre_release_alignment":
                return ActivationExecutionResult(
                    outcome=ControlOutcome.SUCCESS,
                    artifacts=(self._emission("alignment_error", error),),
                )
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(
                    self._emission("held_bowl", observation.held),
                    self._emission("plate_target", observation.target),
                    self._emission("alignment_error", error),
                ),
            )
        if activation_id == "alignment_gate":
            observation = self._resolved_model(invocation, "observation", BowlPlaceObservation)
            self.decision = self.servo.assess(observation)
            artifacts = [self._emission("servo_decision", self.decision)]
            if self.decision.correction is not None:
                self.pending_correction = self.decision.correction
            return ActivationExecutionResult(
                outcome=self.decision.control_outcome,
                artifacts=tuple(artifacts),
                reason=self.decision.reason,
            )
        if activation_id == "plan_correction":
            decision = self._resolved_model(invocation, "servo_decision", ServoDecision)
            assert decision.correction is not None
            self.pending_correction = decision.correction
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(
                    self._emission(
                        "action_spec",
                        self._motion(
                            "bounded_correction",
                            self._resolved_model(invocation, "snapshot", AdmissionSnapshot),
                        ),
                    ),
                ),
            )
        if activation_id == "plan_descend":
            plan = self._motion(
                "descend_to_release",
                self._resolved_model(invocation, "snapshot", AdmissionSnapshot),
            )
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(self._emission("action_spec", plan),),
            )
        if activation_id == "checkpoint_pre_release":
            observation = self._resolved_model(invocation, "observation", BowlPlaceObservation)
            error = self._resolved_model(invocation, "alignment_error", AlignmentError)
            if error.status is not AlignmentStatus.WITHIN_TOLERANCE:
                return ActivationExecutionResult(
                    outcome=ControlOutcome.STALE_OBSERVATION,
                    reason="post-descend alignment is outside tolerance",
                )
            refs = self._input_refs(invocation)
            checkpoint = PhaseCheckpoint(
                phase=CheckpointPhase.PRE_RELEASE,
                status=CheckpointStatus.PASSED,
                action_id=self._action_id_for_refs((refs["receipt"],)),
                receipt_ref=refs["receipt"].artifact_id,
                attachment_status=AttachmentStatus.VERIFIED_HELD,
                bowl_entity_id="bowl-1",
                alignment_id=error.alignment_id,
                observation_generation=observation.observation_generation,
                revisions=observation.revisions,
                evidence_refs=tuple(ref.artifact_id for ref in refs.values()),
                reason="fresh alignment and attachment checks passed",
            )
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(self._emission("checkpoint", checkpoint, lineage=tuple(refs.values())),),
            )
        if activation_id == "plan_open":
            command = self._open(self._resolved_model(invocation, "snapshot", AdmissionSnapshot))
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(self._emission("action_spec", command),),
            )
        if activation_id == "plan_settle":
            wait = self._wait(self._resolved_model(invocation, "snapshot", AdmissionSnapshot))
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(self._emission("action_spec", wait),),
            )
        if activation_id == "checkpoint_post_release":
            observation = self._resolved_model(invocation, "observation", BowlPlaceObservation)
            refs = self._input_refs(invocation)
            state_receipt = StateCommitReceipt.model_validate(
                self._runtime().data_plane.resolve(refs["state_receipt"]).payload
            )
            checkpoint = PhaseCheckpoint(
                phase=CheckpointPhase.POST_RELEASE,
                status=CheckpointStatus.PASSED,
                action_id=self._action_id_for_refs((refs["open_receipt"],)),
                receipt_ref=refs["settle_receipt"].artifact_id,
                attachment_status=AttachmentStatus.NOT_HELD,
                bowl_entity_id="bowl-1",
                alignment_id=(
                    self.decision.alignment_error.alignment_id
                    if self.decision and self.decision.alignment_error
                    else None
                ),
                observation_generation=observation.observation_generation,
                revisions=observation.revisions,
                evidence_refs=tuple(ref.artifact_id for ref in refs.values()),
                reason=f"fresh release evidence committed at state {state_receipt.after_revision}",
            )
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(self._emission("checkpoint", checkpoint, lineage=tuple(refs.values())),),
            )
        if activation_id == "plan_retreat":
            plan = self._motion(
                "safe_retreat",
                self._resolved_model(invocation, "snapshot", AdmissionSnapshot),
            )
            return ActivationExecutionResult(
                outcome=ControlOutcome.SUCCESS,
                artifacts=(self._emission("action_spec", plan),),
            )
        if activation_id == "verify_placement":
            return self._relation_verdict(invocation)
        if activation_id == "propose_final_relation":
            return self._relation_proposal(invocation)
        if activation_id == "placement_complete":
            return ActivationExecutionResult(outcome=ControlOutcome.SUCCESS)
        if activation_id == "recovery_frontier":
            return ActivationExecutionResult(
                outcome=ControlOutcome.FAILED,
                reason="closed recovery frontier reached",
            )
        raise AssertionError(f"unhandled fake bowl activation {activation_id!r}")


class _AlignmentProposalExecutor:
    """Pure persistent namespace used by the real SkillCodingAgentProvider."""

    def __init__(self) -> None:
        self.sandbox_namespace: dict[str, object] = {}

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        try:
            exec(  # noqa: S102 - proposal-only test sandbox
                block.code,
                self.sandbox_namespace,
                self.sandbox_namespace,
            )
        except Exception as exc:  # noqa: BLE001 - executor reports typed failure
            return BlockExecutionResult(
                block=block,
                ok=False,
                status=ActionBlockStatus.FAILED,
                stderr=f"{type(exc).__name__}: {exc}",
            )
        return BlockExecutionResult(
            block=block,
            ok=True,
            status=ActionBlockStatus.SUCCEEDED,
        )


def _alignment_skill_code(*, include_geometry: bool) -> str:
    geometry_outputs = (
        "outputs['held_bowl'] = {'payload': measured['held_bowl']}\n"
        "outputs['plate_target'] = {'payload': measured['plate_target']}\n"
        if include_geometry
        else ""
    )
    return (
        "measured = estimate_support_alignment(\n"
        "    INPUTS['observation']['payload'],\n"
        "    INPUTS['attachment_evidence']['payload'],\n"
        "    tolerance_xy_m=NODE_CONFIG_V1['tolerance_xy_m'],\n"
        "    tolerance_z_m=NODE_CONFIG_V1['tolerance_z_m'],\n"
        "    tolerance_yaw_rad=NODE_CONFIG_V1['tolerance_yaw_rad'],\n"
        ")\n"
        "outputs = {'alignment_error': {'payload': measured['alignment_error']}}\n"
        + geometry_outputs
        + "NODE_RESULT = {'outputs': outputs}\n"
    )


class _AlignmentCodingPolicy:
    def __init__(self) -> None:
        self._turn = 0
        self._include_geometry = False

    def complete_bounded(
        self,
        prompt: list[dict],
        *,
        max_tokens: int,
        deadline_monotonic_s: float | None = None,
    ) -> str:
        assert max_tokens > 0
        if deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s:
            raise TimeoutError("alignment policy deadline expired")
        self._turn += 1
        if self._turn == 1:
            self._include_geometry = "- output key `held_bowl`" in json.dumps(prompt)
            return json.dumps(
                {
                    "tool": "run_python",
                    "args": {
                        "intent": "estimate support alignment from admitted geometry",
                        "code": _alignment_skill_code(include_geometry=self._include_geometry),
                    },
                }
            )
        return json.dumps(
            {
                "tool": "finish",
                "args": {
                    "claim": "alignment estimated from admitted bowl/plate geometry",
                    "result_var": "NODE_RESULT",
                },
            }
        )


_BOWL_SKILL_DIRECTORIES = (
    ("monitor", "author_attachment_monitor"),
    ("state", "propose_attachment_transition"),
    ("state", "propose_relation_transition"),
    ("affordance", "estimate_support_alignment"),
    ("motion", "author_sealed_phase_motion"),
)


def _bowl_skill_library(path: Path) -> SkillLibrary:
    library = SkillLibrary(path)
    builtin = Path(__file__).resolve().parents[1] / "skills" / "builtin"
    for category, name in _BOWL_SKILL_DIRECTORIES:
        library.admit(Skill.from_dir(builtin / category / name))
    return library


def _alignment_coding_provider(tmp_path) -> SkillCodingAgentProvider:
    library = _bowl_skill_library(tmp_path / "alignment-skills")
    return SkillCodingAgentProvider(
        data_plane=None,
        executor_factory=lambda *_args: _AlignmentProposalExecutor(),
        library=library,
        policy_factory=lambda *_args: _AlignmentCodingPolicy(),
        artifacts_root=tmp_path / "alignment-coding-provider",
        max_turns=3,
        trusted_skill_sidecars=frozenset(name for _, name in _BOWL_SKILL_DIRECTORIES),
    )


def _queue_production_checkpoints(backend: InMemoryObservationBackend, *, scenario: str) -> None:
    """Queue synchronized sensor checkpoints; the provider owns every poll."""

    stages = [
        # revision, bowl x, bowl yaw, attachment, visibility, plate x
        (1, 0.520, 0.10, AttachmentStatus.VERIFIED_HELD, True, 0.500),
        (
            2,
            0.520,
            0.10,
            AttachmentStatus.VERIFIED_HELD,
            scenario != "checkpoint_occluded",
            0.500,
        ),
        (
            2 if scenario == "stale_generation" else 3,
            0.505,
            0.00,
            AttachmentStatus.VERIFIED_HELD,
            True,
            0.540 if scenario == "target_drift" else 0.500,
        ),
        (4, 0.505, 0.00, AttachmentStatus.VERIFIED_HELD, True, 0.500),
        (5, 0.505, 0.00, AttachmentStatus.NOT_HELD, True, 0.500),
        (6, 0.505, 0.00, AttachmentStatus.NOT_HELD, True, 0.500),
    ]
    clock = time.monotonic() + 1.0
    for index, (revision, bowl_x, bowl_yaw, attachment, visible, plate_x) in enumerate(stages):
        revisions = ObservationRevisionVector(
            scene_revision=revision,
            arm_revision=revision,
            gripper_revision=revision,
            attachment_revision=revision,
            camera_revision=revision,
        )
        observed_at = datetime.now(UTC)
        bowl = BackendObservation(
            entity_id="bowl-1",
            quality=(ObservationQuality.TRACKED if visible else ObservationQuality.LOST),
            revisions=revisions,
            signals=(
                {
                    "attachment_status": attachment.value,
                    "center_x": bowl_x,
                    "center_y": 0.0,
                    "center_z": 0.2,
                    "identity_match": True,
                    "visibility": "visible",
                    "yaw": bowl_yaw,
                }
                if visible
                else {}
            ),
            reason=None if visible else "held_bowl_occluded_at_primary_checkpoint",
            observed_at=observed_at,
            monotonic_time_s=clock + index,
        )
        plate = BackendObservation(
            entity_id="plate-1",
            quality=ObservationQuality.TRACKED,
            revisions=revisions,
            signals={
                "center_x": plate_x,
                "center_y": 0.0,
                "center_z": 0.2,
                "identity_match": True,
                "visibility": "visible",
                "yaw": 0.0,
            },
            observed_at=observed_at,
            monotonic_time_s=clock + index,
        )
        backend.push(bowl)
        backend.push(plate)


def _continuous_sampler_harness(
    *,
    poll_interval_s: float = 0.005,
    max_silence_s: float = 0.05,
):
    registry = ObservationRegistry(episode_id="continuous-tracking")
    backend = InMemoryObservationBackend("continuous-backend")
    registry.register_backend(backend)
    bowl = registry.create(
        TrackRequest(
            episode_id=registry.episode_id,
            track_id="continuous-bowl",
            entity=EntityIdentity(entity_id="bowl-1", semantic_label="bowl"),
            backend_id=backend.backend_id,
            declared_signals=BowlPlaceActorProvider.BOWL_SIGNALS,
            camera_ids=("front",),
        ),
        start=True,
    )
    plate = registry.create(
        TrackRequest(
            episode_id=registry.episode_id,
            track_id="continuous-plate",
            entity=EntityIdentity(entity_id="plate-1", semantic_label="plate"),
            backend_id=backend.backend_id,
            declared_signals=BowlPlaceActorProvider.PLATE_SIGNALS,
            camera_ids=("front",),
        ),
        start=True,
    )
    sampler = SynchronizedBowlTrackSampler(
        bowl_track=bowl,
        plate_track=plate,
        poll_interval_s=poll_interval_s,
        max_silence_s=max_silence_s,
        max_pair_skew_s=0.05,
        pair_buffer_size=4,
    )
    return registry, backend, bowl, plate, sampler


def _push_continuous_sample(
    backend: InMemoryObservationBackend,
    *,
    entity_id: str,
    revision: int,
    monotonic_time_s: float,
    observed_at: datetime,
    quality: ObservationQuality = ObservationQuality.TRACKED,
) -> None:
    is_bowl = entity_id == "bowl-1"
    signals = (
        {
            **({"attachment_status": AttachmentStatus.VERIFIED_HELD.value} if is_bowl else {}),
            "center_x": 0.51 if is_bowl else 0.5,
            "center_y": 0.0,
            "center_z": 0.2,
            "identity_match": True,
            "visibility": "visible",
            "yaw": 0.0,
        }
        if quality is ObservationQuality.TRACKED
        else {}
    )
    backend.push(
        BackendObservation(
            entity_id=entity_id,
            quality=quality,
            revisions=ObservationRevisionVector(
                scene_revision=revision,
                arm_revision=revision,
                gripper_revision=revision,
                attachment_revision=revision,
                camera_revision=revision,
            ),
            signals=signals,
            reason=None if quality is ObservationQuality.TRACKED else "object_dropped",
            observed_at=observed_at,
            monotonic_time_s=monotonic_time_s,
        )
    )


def _wait_until(predicate, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def _direct_capture_harness(tmp_path, backend: InMemoryObservationBackend):
    registry = ObservationRegistry(episode_id="capture-crash")
    registry.register_backend(backend)
    provider = BowlPlaceActorProvider(
        BowlPlaceProviderConfig(
            observation_backend_id=backend.backend_id,
            tracking_mode=BowlTrackingMode.MANUAL_CHECKPOINT,
        )
    )
    actors = ActorRegistry(
        {"bowl_place": provider},
        namespace_root="capture-crash",
        workspace_root=tmp_path / "actors-first",
    )
    runtime = EpisodeRuntime(
        episode_id="capture-crash",
        episode_root=tmp_path / "episode",
        actors=actors,
        observation_registry=registry,
        freshness_context_provider=provider.freshness_context,
    )
    provider.bind_episode_runtime(runtime)
    runtime.data_plane.open_workflow("capture-workflow")
    profile = ActorProfile(
        profile_id="direct-capture",
        provider_id="bowl_place",
        runner_kind=RunnerKind.DETERMINISTIC_GATE.value,
        lifecycle=ActorLifecycle.EPHEMERAL,
        capability_ceiling=frozenset({"perception.observe"}),
        effect_ceiling=frozenset(),
        metadata={"runner_ref": "robomex.bowl_place.capture_synchronized_checkpoint"},
    )
    invocation = InvocationSpec(
        invocation_id="capture-command-1",
        idempotency_key="capture-command-1",
        objective="capture one synchronized bowl/plate checkpoint",
        output_contract={"observation": "robomex.bowl_place_observation.v1"},
        requested_capabilities=frozenset({"perception.observe"}),
        metadata={
            "episode_id": runtime.episode_id,
            "workflow_id": "capture-workflow",
            "activation_id": "capture_initial_attachment",
            "attempt": 1,
        },
    )
    handle = actors.spawn(profile, actor_id="capture-actor-first")
    return runtime, provider, profile, invocation, handle


def _restart_direct_capture_provider(
    tmp_path,
    runtime: EpisodeRuntime,
    profile: ActorProfile,
):
    provider = BowlPlaceActorProvider(
        BowlPlaceProviderConfig(
            observation_backend_id="capture-checkpoints",
            tracking_mode=BowlTrackingMode.MANUAL_CHECKPOINT,
        )
    )
    provider.bind_episode_runtime(runtime)
    actors = ActorRegistry(
        {"bowl_place": provider},
        namespace_root="capture-crash-restart",
        workspace_root=tmp_path / "actors-restart",
    )
    return provider, actors.spawn(profile, actor_id="capture-actor-restarted")


def _episode_fixture(tmp_path, *, scenario: str):
    backend = _BowlFakeBackend(scenario=scenario)
    observation_backend = InMemoryObservationBackend("bowl-checkpoint-backend")
    _queue_production_checkpoints(observation_backend, scenario=scenario)
    observation_registry = ObservationRegistry(episode_id=f"bowl-{scenario}")
    observation_registry.register_backend(observation_backend)
    driver = _BowlEpisodeDriver(backend, observation_backend, scenario=scenario)
    coding_provider = InMemoryAgentProvider(driver)
    alignment_provider = _alignment_coding_provider(tmp_path)
    bowl_provider = BowlPlaceActorProvider(
        BowlPlaceProviderConfig(
            observation_backend_id=observation_backend.backend_id,
            correction_limits=CorrectionLimits(max_iterations=4),
            tracking_mode=BowlTrackingMode.MANUAL_CHECKPOINT,
        )
    )
    actors = ActorRegistry(
        {
            "in_memory": coding_provider,
            "skill_coding": alignment_provider,
            "bowl_place": bowl_provider,
        },
        namespace_root=f"bowl-{scenario}",
        workspace_root=tmp_path / "actors",
    )
    checker = _AlwaysFeasible()
    runtime = EpisodeRuntime(
        episode_id=f"bowl-{scenario}",
        episode_root=tmp_path / "episode",
        actors=actors,
        observation_registry=observation_registry,
        freshness_context_provider=bowl_provider.freshness_context,
        action_backends={
            ("authoritative", resource): backend
            for resource in ("robot.arm", "robot.gripper", "robot.controller")
        },
        feasibility_checkers={
            ("authoritative", resource): checker
            for resource in ("robot.arm", "robot.gripper", "robot.controller")
        },
    )
    driver.bind_runtime(runtime)
    alignment_provider.bind_episode_data_plane(runtime.data_plane)
    bowl_provider.bind_episode_runtime(runtime)
    protocol = build_fixed_bowl_place_protocol(BowlPlaceProtocolConfig(max_alignment_iterations=4))
    bowl_bindings = build_bowl_place_actor_bindings(
        protocol.spec,
        provider=bowl_provider,
        allow_manual_tracking_for_tests=True,
    )
    coding_bindings = build_bowl_place_coding_profiles(protocol.spec)
    registered_runner_refs: set[str] = set()
    for node in protocol.spec.activations:
        if node.runner_kind in {
            RunnerKind.ACTION_SNAPSHOT,
            RunnerKind.SYSTEM_ACTION,
            RunnerKind.REDUCER,
        }:
            continue
        if node.runner_ref in registered_runner_refs:
            continue
        registered_runner_refs.add(node.runner_ref)
        production_profile = bowl_bindings.profiles.get(node.runner_ref)
        alignment_profile = (
            coding_bindings.profiles[node.runner_ref]
            if node.runner_ref == "robomex.bowl_place.estimate_support_alignment"
            else None
        )
        runtime.register_actor_profile(
            node.runner_ref,
            production_profile
            or alignment_profile
            or ActorProfile(
                profile_id=f"profile-{len(registered_runner_refs)}",
                runner_kind=node.runner_kind.value,
                lifecycle=(
                    ActorLifecycle.EPHEMERAL
                    if node.lifecycle is LifecycleScope.INVOCATION
                    else ActorLifecycle.SERVICE
                ),
                capability_ceiling=frozenset(node.required_capabilities),
                effect_ceiling=(
                    frozenset()
                    if node.effect_scope is EffectScope.READ_ONLY
                    else frozenset({node.effect_scope.value})
                ),
            ),
        )
    runtime.data_plane.open_workflow("seed")
    seed = runtime.data_plane.publish(
        workflow_id="seed",
        activation_id="prior_pick",
        attempt=1,
        port="evidence",
        schema="test.prior_pick_evidence.v1",
        payload={"action_id": "grasp-action"},
    )
    for entity_id, label, track_id in (
        ("bowl-1", "bowl", "track-bowl-1"),
        ("plate-1", "plate", "track-plate-1"),
    ):
        runtime.state_reducer.commit(
            StateTransitionProposal.register_entity(
                episode_id=runtime.episode_id,
                effect_id=f"register-{entity_id}",
                before_revision=runtime.state_reducer.state.revision,
                source="prior_grounding",
                evidence_refs=(seed.ref,),
                entity_id=entity_id,
                semantic_label=label,
                track_id=track_id,
            )
        )
    runtime.state_reducer.commit(
        StateTransitionProposal.set_attachment(
            episode_id=runtime.episode_id,
            effect_id="attempt-grasp",
            before_revision=runtime.state_reducer.state.revision,
            source="prior_pick_action",
            evidence_refs=(seed.ref,),
            entity_id="bowl-1",
            status=AttachmentStatus.ATTEMPTED,
            action_id="grasp-action",
            trigger=PhysicalStateTrigger.EVIDENCE,
            track_id="track-bowl-1",
        )
    )
    runtime.data_plane.close_workflow("seed")
    workflow_id = runtime.open_workflow(
        workflow_id="place",
        intent=SubgoalIntent(
            intent_id="place-bowl",
            instruction="put the bowl on the plate",
            success_rubric="bowl is supported by the plate",
        ),
        graph=protocol.compiled,
        external_refs={},
        frontier=protocol.recovery_frontier,
    )
    return (
        runtime,
        workflow_id,
        backend,
        driver,
        bowl_provider,
        alignment_provider,
    )


def test_geometry_schemas_reject_nan_and_degenerate_shapes() -> None:
    with pytest.raises(ValidationError, match="finite"):
        Vector3(x=float("nan"), y=0.0, z=0.0)
    with pytest.raises(ValidationError, match="unit length"):
        QuaternionWXYZ(w=2.0, x=0.0, y=0.0, z=0.0)
    with pytest.raises(ValidationError, match="non-zero area"):
        SupportFootprint(vertices_xy_m=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    with pytest.raises(ValidationError, match="finite"):
        SupportFootprint(vertices_xy_m=((0.0, 0.0), (1.0, 0.0), (0.0, float("inf"))))


def test_domain_schema_ids_are_bound_to_real_validators() -> None:
    registry = SchemaRegistry()
    register_bowl_place_schemas(registry)
    held = _held(center=Vector3(x=0.4, y=0.0, z=0.2), yaw=0.0)

    parsed = registry.validate("robomex.held_bowl_estimate.v1", held)

    assert isinstance(parsed, HeldBowlEstimate)
    assert registry.is_registered("robomex.alignment_error.v1")
    with pytest.raises(ValueError, match="already registered"):
        register_bowl_place_schemas(registry)


def test_protocol_control_schemas_are_strict_registered_and_fail_closed() -> None:
    registry = SchemaRegistry()
    register_bowl_place_schemas(registry)
    attachment = AttachmentStateSnapshot(
        episode_id="episode",
        state_revision=4,
        attachment_revision=2,
        status=AttachmentStatus.VERIFIED_HELD,
        entity_id="bowl-1",
        source_observation_id="obs-4",
        evidence_refs=("attachment-evidence",),
    )
    guard = AttachmentGuard(
        phase=AttachmentGuardPhase.OPEN,
        allowed=True,
        status=AttachmentStatus.VERIFIED_HELD,
        entity_id="bowl-1",
        state_revision=4,
        observation_generation=3,
        reason="fresh evidence confirms held bowl",
        evidence_refs=("attachment-evidence",),
    )
    checkpoint = PhaseCheckpoint(
        phase=CheckpointPhase.PRE_RELEASE,
        status=CheckpointStatus.PASSED,
        action_id="descend-action",
        receipt_ref="descend-receipt",
        attachment_status=AttachmentStatus.VERIFIED_HELD,
        bowl_entity_id="bowl-1",
        alignment_id="alignment-3",
        observation_generation=3,
        revisions=_revisions(3),
        evidence_refs=("pre-release-frame",),
        reason="alignment and attachment passed",
    )
    verdict = PlacementVerdict(
        status=PlacementVerdictStatus.SUCCEEDED,
        bowl_entity_id="bowl-1",
        target_entity_id="plate-1",
        relation=RelationAssessment.ASSERTED,
        source_observation_id="obs-final",
        state_revision=5,
        support_clearance_m=0.02,
        confidence=0.97,
        evidence_refs=("final-frame",),
        reason="bowl support footprint is on the plate",
    )
    recovery = RecoverySafetyDecision(
        disposition=RecoveryDisposition.REOBSERVE_REQUIRED,
        patch_authorized=False,
        graph_revision=1,
        state_revision=5,
        attachment_status=AttachmentStatus.UNKNOWN,
        allowed_effect_scopes=(RecoveryEffectScope.READ_ONLY,),
        blocked_activation_ids=("execute_descend", "execute_open"),
        required_refreshes=("attachment", "held_bowl", "plate_target"),
        evidence_refs=("drop-finding",),
        reason="attachment must be re-observed before any release action",
    )

    for schema_id, payload in (
        ("robomex.attachment_state.v1", attachment),
        ("robomex.attachment_guard.v1", guard),
        ("robomex.phase_checkpoint.v1", checkpoint),
        ("robomex.placement_verdict.v1", verdict),
        ("robomex.recovery_safety_decision.v1", recovery),
    ):
        assert registry.validate(schema_id, payload).__class__ is payload.__class__

    with pytest.raises(ValidationError, match="verified_held"):
        AttachmentGuard(
            phase=AttachmentGuardPhase.OPEN,
            allowed=True,
            status=AttachmentStatus.UNKNOWN,
            state_revision=4,
            observation_generation=3,
            reason="unsafe",
        )
    with pytest.raises(ValidationError, match="block execute_open"):
        RecoverySafetyDecision(
            disposition=RecoveryDisposition.SAFE_STOP,
            patch_authorized=False,
            graph_revision=1,
            state_revision=5,
            attachment_status=AttachmentStatus.NOT_HELD,
            allowed_effect_scopes=(RecoveryEffectScope.READ_ONLY,),
            evidence_refs=("drop-finding",),
            reason="stop",
        )


def test_tracking_and_physical_revision_types_cross_only_explicit_adapter() -> None:
    tracking = ObservationRevisionVector(
        scene_revision=3,
        arm_revision=4,
        gripper_revision=5,
        attachment_revision=6,
        camera_revision=7,
    )

    physical = revision_vector_from_observation(tracking, camera_id="front")

    assert physical == RevisionVector(
        scene=3,
        arm=4,
        gripper=5,
        attachment=6,
        camera={"front": 7},
    )
    assert observation_revision_from_physical(physical, camera_id="front") == tracking
    with pytest.raises(RevisionMismatchError, match="no camera"):
        observation_revision_from_physical(physical, camera_id="wrist")


def test_alignment_rejects_entity_frame_snapshot_and_revision_mismatch() -> None:
    center = Vector3(x=0.4, y=0.0, z=0.2)
    held = _held(center=center, yaw=0.0)
    target = _target(center=center)

    with pytest.raises(EntityMismatchError, match="held entity"):
        compute_alignment_error(held, target, expected_bowl_entity_id="other-bowl")
    with pytest.raises(FrameMismatchError, match="frame"):
        compute_alignment_error(
            held,
            _target(center=center, frame_id="camera", snapshot="snapshot-1"),
        )
    with pytest.raises(RevisionMismatchError, match="snapshot/generation/revision"):
        compute_alignment_error(
            held,
            _target(center=center, generation=2, snapshot="snapshot-2"),
        )

    raw = _observation(held, target).model_dump(mode="python")
    raw["revisions"] = _revisions(3)
    with pytest.raises(ValidationError, match="revisions do not match"):
        BowlPlaceObservation.model_validate(raw)


def test_translation_and_yaw_error_are_target_minus_held() -> None:
    held = _held(center=Vector3(x=0.42, y=-0.03, z=0.22), yaw=0.3)
    target = _target(center=Vector3(x=0.5, y=0.02, z=0.2), yaw=-0.1)

    error = compute_alignment_error(
        held,
        target,
        tolerance=AlignmentTolerance(
            translation_xy_m=0.001,
            translation_z_m=0.001,
            yaw_rad=0.001,
        ),
    )

    assert error.translation_error_m.as_tuple() == pytest.approx((0.08, 0.05, -0.02))
    assert error.yaw_error_rad == pytest.approx(-0.4)
    assert not error.support_contained
    assert error.support_clearance_m < 0.0
    assert error.status is AlignmentStatus.CORRECTION_REQUIRED
    assert error.expressed_in_frame == "world"
    assert error.revisions == held.revisions == target.revisions


def test_support_footprint_is_a_real_alignment_gate_not_unused_metadata() -> None:
    center = Vector3(x=0.5, y=0.0, z=0.2)
    held = _held(center=center, yaw=0.0)
    target = _target(center=center)
    too_small = PlateSupportTarget(
        target_id=target.target_id,
        entity_id=target.entity_id,
        frame_id=target.frame_id,
        snapshot_id=target.snapshot_id,
        observation_generation=target.observation_generation,
        revisions=target.revisions,
        support_center_m=target.support_center_m,
        support_footprint=_footprint(center, radius=0.02),
        surface_normal=target.surface_normal,
        orientation=target.orientation,
        uncertainty=target.uncertainty,
        safe_margin_m=0.005,
        confidence=target.confidence,
    )

    error = compute_alignment_error(held, too_small)
    servo = VisualServoPlacer(bowl_entity_id="bowl-1", target_entity_id="plate-1", frame_id="world")
    decision = servo.assess(_observation(held, too_small))

    assert not error.support_contained
    assert error.support_clearance_m < 0.0
    assert error.status is AlignmentStatus.CORRECTION_REQUIRED
    # Centers and yaw already coincide, so no bounded motion can make the
    # physically-too-small support region valid; fail closed into recovery.
    assert decision.outcome is ServoOutcome.LOOP_EXHAUSTED
    assert decision.correction is None


def test_within_tolerance_stops_without_moving() -> None:
    target_center = Vector3(x=0.5, y=0.1, z=0.2)
    held = _held(center=Vector3(x=0.502, y=0.098, z=0.201), yaw=0.02)
    target = _target(center=target_center)
    servo = VisualServoPlacer(bowl_entity_id="bowl-1", target_entity_id="plate-1", frame_id="world")

    decision = servo.assess(_observation(held, target))

    assert decision.outcome is ServoOutcome.WITHIN_TOLERANCE
    assert decision.correction is None
    assert servo.iterations == 0
    assert servo.cumulative_translation_m == 0.0
    assert servo.authorize_phase(
        PlacementPhase.DESCEND,
        attachment_status=AttachmentStatus.VERIFIED_HELD,
        decision=decision,
    ).allowed


def test_randomized_rim_grasp_offsets_and_yaws_converge_with_fresh_observations() -> None:
    rng = random.Random(20260722)
    target_center = Vector3(x=0.52, y=-0.08, z=0.19)
    tolerance = AlignmentTolerance(
        translation_xy_m=0.001,
        translation_z_m=0.001,
        yaw_rad=0.01,
    )
    limits = CorrectionLimits(
        max_step_translation_m=0.02,
        max_step_yaw_rad=0.15,
        max_cumulative_translation_m=0.13,
        max_cumulative_yaw_rad=0.8,
        max_iterations=10,
        max_target_drift_m=0.015,
    )

    for _ in range(64):
        offset = Vector3(
            x=rng.uniform(-0.05, 0.05),
            y=rng.uniform(-0.05, 0.05),
            z=rng.uniform(-0.018, 0.018),
        )
        held = _held(center=target_center.plus(offset), yaw=rng.uniform(-0.55, 0.55))
        target = _target(center=target_center)
        servo = VisualServoPlacer(
            bowl_entity_id="bowl-1",
            target_entity_id="plate-1",
            frame_id="world",
            tolerance=tolerance,
            limits=limits,
        )

        for generation in range(1, 12):
            decision = servo.assess(_observation(held, target))
            if decision.outcome is ServoOutcome.WITHIN_TOLERANCE:
                break
            assert decision.outcome is ServoOutcome.CORRECTION_REQUIRED
            assert decision.correction is not None
            assert decision.correction.delta_translation_m.norm <= 0.02 + 1e-12
            assert abs(decision.correction.delta_yaw_rad) <= 0.15 + 1e-12
            assert servo.authorize_phase(
                PlacementPhase.CORRECTION,
                attachment_status=AttachmentStatus.VERIFIED_HELD,
                decision=decision,
            ).allowed
            next_generation = generation + 1
            revisions = _revisions(next_generation)
            held = apply_correction_to_estimate(
                held,
                decision.correction,
                next_snapshot_id=f"snapshot-{next_generation}",
                next_generation=next_generation,
                next_revisions=revisions,
            )
            target = _refresh_target(
                target,
                generation=next_generation,
                revisions=revisions,
            )
        else:  # pragma: no cover - explicit diagnostic if convergence regresses
            pytest.fail("bounded visual servo did not converge")

        assert decision.outcome is ServoOutcome.WITHIN_TOLERANCE
        assert servo.iterations <= limits.max_iterations
        assert servo.cumulative_translation_m <= limits.max_cumulative_translation_m
        assert servo.cumulative_yaw_rad <= limits.max_cumulative_yaw_rad


def test_each_correction_requires_new_generation_and_advanced_arm_revision() -> None:
    held = _held(center=Vector3(x=0.45, y=0.0, z=0.2), yaw=0.2)
    target = _target(center=Vector3(x=0.5, y=0.0, z=0.2))
    servo = VisualServoPlacer(bowl_entity_id="bowl-1", target_entity_id="plate-1", frame_id="world")
    first_observation = _observation(held, target)
    first = servo.assess(first_observation)
    assert first.outcome is ServoOutcome.CORRECTION_REQUIRED

    reused = servo.assess(first_observation)
    assert reused.outcome is ServoOutcome.STALE_OBSERVATION
    assert reused.correction is None
    assert servo.iterations == 1

    assert first.correction is not None
    with pytest.raises(RevisionMismatchError, match="advanced arm revision"):
        apply_correction_to_estimate(
            held,
            first.correction,
            next_snapshot_id="snapshot-2",
            next_generation=2,
            next_revisions=RevisionVector(
                scene=2,
                arm=1,
                gripper=2,
                attachment=2,
                camera={"front": 2},
            ),
        )


def test_step_cumulative_rotation_and_iteration_limits_fail_closed() -> None:
    held = _held(center=Vector3(x=0.35, y=0.0, z=0.2), yaw=0.8)
    target = _target(center=Vector3(x=0.5, y=0.0, z=0.2))
    limits = CorrectionLimits(
        max_step_translation_m=0.01,
        max_step_yaw_rad=0.05,
        max_cumulative_translation_m=0.015,
        max_cumulative_yaw_rad=0.06,
        max_iterations=10,
        max_target_drift_m=0.02,
    )
    servo = VisualServoPlacer(
        bowl_entity_id="bowl-1",
        target_entity_id="plate-1",
        frame_id="world",
        tolerance=AlignmentTolerance(
            translation_xy_m=0.001,
            translation_z_m=0.001,
            yaw_rad=0.001,
        ),
        limits=limits,
    )

    first = servo.assess(_observation(held, target))
    assert first.correction is not None
    assert first.correction.delta_translation_m.norm == pytest.approx(0.01)
    assert abs(first.correction.delta_yaw_rad) == pytest.approx(0.05)
    held = apply_correction_to_estimate(
        held,
        first.correction,
        next_snapshot_id="snapshot-2",
        next_generation=2,
        next_revisions=_revisions(2),
    )
    target = _refresh_target(target, generation=2, revisions=_revisions(2))
    second = servo.assess(_observation(held, target))
    assert second.correction is not None
    assert second.correction.delta_translation_m.norm == pytest.approx(0.005)
    assert abs(second.correction.delta_yaw_rad) == pytest.approx(0.01)
    held = apply_correction_to_estimate(
        held,
        second.correction,
        next_snapshot_id="snapshot-3",
        next_generation=3,
        next_revisions=_revisions(3),
    )
    target = _refresh_target(target, generation=3, revisions=_revisions(3))

    exhausted = servo.assess(_observation(held, target))
    assert exhausted.outcome is ServoOutcome.LOOP_EXHAUSTED
    assert exhausted.control_outcome is ControlOutcome.EXHAUSTED
    assert exhausted.correction is None
    assert servo.cumulative_translation_m == pytest.approx(0.015)
    assert servo.cumulative_yaw_rad == pytest.approx(0.06)

    one_step = VisualServoPlacer(
        bowl_entity_id="bowl-1",
        target_entity_id="plate-1",
        frame_id="world",
        limits=CorrectionLimits(max_iterations=1),
    )
    original_held = _held(center=Vector3(x=0.42, y=0.0, z=0.2), yaw=0.3)
    original_target = _target(center=Vector3(x=0.5, y=0.0, z=0.2))
    issued = one_step.assess(_observation(original_held, original_target))
    assert issued.correction is not None
    next_held = apply_correction_to_estimate(
        original_held,
        issued.correction,
        next_snapshot_id="snapshot-2",
        next_generation=2,
        next_revisions=_revisions(2),
    )
    next_target = _refresh_target(original_target, generation=2, revisions=_revisions(2))
    assert (
        one_step.assess(_observation(next_held, next_target)).outcome is ServoOutcome.LOOP_EXHAUSTED
    )


def test_occlusion_drop_identity_swap_and_target_drift_are_closed_outcomes() -> None:
    center = Vector3(x=0.5, y=0.0, z=0.2)
    held = _held(center=Vector3(x=0.46, y=0.0, z=0.2), yaw=0.0)
    target = _target(center=center)

    occluded_servo = VisualServoPlacer(
        bowl_entity_id="bowl-1", target_entity_id="plate-1", frame_id="world"
    )
    occluded = occluded_servo.assess(
        _observation(
            None,
            target,
            held_visibility=VisibilityStatus.OCCLUDED,
        )
    )
    assert occluded.outcome is ServoOutcome.OCCLUDED
    assert occluded.control_outcome is ControlOutcome.UNCERTAIN

    dropped_servo = VisualServoPlacer(
        bowl_entity_id="bowl-1", target_entity_id="plate-1", frame_id="world"
    )
    dropped = dropped_servo.assess(_observation(held, target, attachment=AttachmentStatus.NOT_HELD))
    assert dropped.outcome is ServoOutcome.DROPPED
    assert dropped.control_outcome is ControlOutcome.ATTACHMENT_NOT_CONFIRMED

    swapped_servo = VisualServoPlacer(
        bowl_entity_id="bowl-1", target_entity_id="plate-1", frame_id="world"
    )
    swapped_held = _held(center=held.bottom_center_m, yaw=0.0, entity_id="bowl-2")
    swapped = swapped_servo.assess(_observation(swapped_held, target))
    assert swapped.outcome is ServoOutcome.IDENTITY_SWAP
    assert swapped.control_outcome is ControlOutcome.WRONG_GROUNDING

    drift_servo = VisualServoPlacer(
        bowl_entity_id="bowl-1",
        target_entity_id="plate-1",
        frame_id="world",
        limits=CorrectionLimits(max_target_drift_m=0.01),
    )
    first = drift_servo.assess(_observation(held, target))
    assert first.correction is not None
    held2 = apply_correction_to_estimate(
        held,
        first.correction,
        next_snapshot_id="snapshot-2",
        next_generation=2,
        next_revisions=_revisions(2),
    )
    target2 = _refresh_target(
        target,
        generation=2,
        revisions=_revisions(2),
        center=Vector3(x=0.53, y=0.0, z=0.2),
    )
    drifted = drift_servo.assess(_observation(held2, target2))
    assert drifted.outcome is ServoOutcome.TARGET_DRIFT
    assert drifted.control_outcome is ControlOutcome.TARGET_DRIFT


def test_drop_or_unverified_attachment_never_authorizes_descend_or_open() -> None:
    center = Vector3(x=0.5, y=0.0, z=0.2)
    held = _held(center=center, yaw=0.0)
    target = _target(center=center)
    servo = VisualServoPlacer(bowl_entity_id="bowl-1", target_entity_id="plate-1", frame_id="world")
    aligned = servo.assess(_observation(held, target))
    assert aligned.outcome is ServoOutcome.WITHIN_TOLERANCE

    for phase in (PlacementPhase.CORRECTION, PlacementPhase.DESCEND, PlacementPhase.OPEN):
        dropped = servo.authorize_phase(
            phase,
            attachment_status=AttachmentStatus.NOT_HELD,
            decision=aligned,
        )
        unknown = servo.authorize_phase(
            phase,
            attachment_status=AttachmentStatus.UNKNOWN,
            decision=aligned,
        )
        assert not dropped.allowed
        assert dropped.outcome is ServoOutcome.DROPPED
        assert not unknown.allowed
        assert unknown.outcome is ServoOutcome.ATTACHMENT_NOT_CONFIRMED


def test_fixed_offset_baseline_remains_explicit_open_loop_ablation() -> None:
    result = fixed_offset_baseline(
        current_tcp_position_m=Vector3(x=0.4, y=0.1, z=0.3),
        target_support_center_m=Vector3(x=0.5, y=0.0, z=0.2),
        assumed_tcp_to_bottom_offset_world_m=Vector3(x=0.02, y=0.0, z=-0.1),
    )

    assert result.desired_tcp_position_m.as_tuple() == pytest.approx((0.48, 0.0, 0.3))
    assert result.translation_delta_m.as_tuple() == pytest.approx((0.08, -0.1, 0.0))


def test_fixed_protocol_compiles_actions_are_sealed_and_recovery_slot_is_bound() -> None:
    bundle = build_fixed_bowl_place_protocol(BowlPlaceProtocolConfig(max_alignment_iterations=4))
    spec = bundle.spec
    compiled = bundle.compiled
    node_map = {node.activation_id: node for node in spec.activations}

    assert compiled.loop_limits["alignment_visual_servo"] == 4
    assert set(
        next(
            loop.activation_ids
            for loop in spec.bounded_loops
            if loop.loop_id == "alignment_visual_servo"
        )
    ) == {
        "capture_alignment",
        "verify_alignment_attachment",
        "estimate_alignment",
        "alignment_gate",
        "snapshot_correction",
        "plan_correction",
        "execute_correction",
    }
    assert (
        compiled.next_activation("execute_correction", ControlOutcome.SUCCESS)
        == "capture_alignment"
    )
    assert (
        compiled.next_activation("alignment_gate", ControlOutcome.EXHAUSTED) == "recovery_frontier"
    )
    assert (
        compiled.next_activation("alignment_gate", ControlOutcome.TARGET_DRIFT)
        == "recovery_frontier"
    )
    assert (
        compiled.next_activation("alignment_gate", ControlOutcome.WRONG_GROUNDING)
        == "recovery_frontier"
    )

    system_actions = [
        node for node in spec.activations if node.runner_kind is RunnerKind.SYSTEM_ACTION
    ]
    assert {node.activation_id for node in system_actions} == {
        "execute_transport",
        "execute_correction",
        "execute_descend",
        "execute_open",
        "execute_settle",
        "execute_retreat",
    }
    snapshot_bindings = {
        "snapshot_transport": ("plan_transport", "robot.arm"),
        "snapshot_correction": ("plan_correction", "robot.arm"),
        "snapshot_descend": ("plan_descend", "robot.arm"),
        "snapshot_open": ("plan_open", "robot.gripper"),
        "snapshot_settle": ("plan_settle", "robot.controller"),
        "snapshot_retreat": ("plan_retreat", "robot.arm"),
    }
    for snapshot_id, (author_id, resource_id) in snapshot_bindings.items():
        snapshot_node = node_map[snapshot_id]
        author_node = node_map[author_id]
        assert snapshot_node.runner_kind is RunnerKind.ACTION_SNAPSHOT
        assert snapshot_node.authority_world_id == "authoritative"
        assert snapshot_node.authoritative_resource == resource_id
        assert not snapshot_node.inputs
        assert [(port.name, port.schema_id) for port in snapshot_node.outputs] == [
            ("snapshot", "robomex.admission_snapshot.v1")
        ]
        assert not any(snapshot_node.estimated_budget.model_dump().values())
        assert any(
            binding.input_port == "snapshot"
            and binding.source_activation == snapshot_id
            and binding.source_port == "snapshot"
            for binding in author_node.bindings
        )
    for tracker_id in ("held_bowl_tracker", "plate_tracker"):
        assert not any(node_map[tracker_id].estimated_budget.model_dump().values())
    monitored_actions = {
        "execute_transport",
        "execute_correction",
        "execute_descend",
    }
    for node in system_actions:
        assert node.effect_scope is EffectScope.AUTHORITATIVE_WORLD
        expected_inputs = (
            ["action_spec", "monitor_program"]
            if node.activation_id in monitored_actions
            else ["action_spec"]
        )
        assert [port.name for port in node.inputs] == expected_inputs
        assert [(port.name, port.schema_id) for port in node.outputs] == [
            ("receipt", EXECUTION_RECEIPT)
        ]
        assert node.bindings[0].input_port == "action_spec"
        if node.activation_id in monitored_actions:
            assert node.params["monitor_input_port"] == "monitor_program"
            assert node.inputs[1].schema_id == MONITOR_PROGRAM
            assert node.bindings[1].input_port == "monitor_program"
            assert node.bindings[1].source_activation == "author_attachment_monitor"
            assert node.bindings[1].source_port == "monitor_program"
        else:
            assert "monitor_input_port" not in node.params

    author = node_map["author_attachment_monitor"]
    assert spec.entry_activation == author.activation_id
    assert author.runner_kind is RunnerKind.CODING_WORKER
    assert [(port.name, port.schema_id) for port in author.outputs] == [
        ("monitor_program", MONITOR_PROGRAM)
    ]
    assert "monitor_program_digest" not in author.params
    assert compiled.next_activation(author.activation_id, ControlOutcome.SUCCESS) == (
        "capture_initial_attachment"
    )

    capture_sources = {
        "capture_pre_release": {"receipt": "execute_descend"},
        "capture_release": {
            "open_receipt": "execute_open",
            "settle_receipt": "execute_settle",
        },
        "capture_final_relation": {"receipt": "execute_retreat"},
    }
    for capture_id, expected_sources in capture_sources.items():
        capture = node_map[capture_id]
        assert capture.runner_kind is RunnerKind.DETERMINISTIC_GATE
        assert {
            binding.input_port: binding.source_activation for binding in capture.bindings
        } == expected_sources
    assert not any(
        binding.kind == "external" for node in spec.activations for binding in node.bindings
    )
    reducer_ids = {
        node.activation_id for node in spec.activations if node.runner_kind is RunnerKind.REDUCER
    }
    assert reducer_ids == {
        "commit_initial_attachment",
        "commit_release_attachment",
        "commit_final_relation",
    }
    for reducer_id in reducer_ids:
        reducer = node_map[reducer_id]
        assert [(port.name, port.schema_id) for port in reducer.inputs] == [
            ("proposal", "robomex.state_transition_proposal.v1")
        ]
        assert [(port.name, port.schema_id) for port in reducer.outputs] == [
            ("receipt", "robomex.state_commit_receipt.v1")
        ]

    for service_id in ("held_bowl_tracker", "plate_tracker"):
        assert node_map[service_id].lane.value == "service"
        assert node_map[service_id].effect_scope is EffectScope.READ_ONLY
        assert node_map[service_id].subscriptions == ("action_outcome",)

    frontier = bundle.recovery_frontier
    assert frontier.graph_id == spec.graph_id
    assert frontier.revision == spec.revision
    assert frontier.graph_digest == compiled.digest
    assert frontier.slots == (bundle.recovery_slot,)
    assert bundle.recovery_slot.target_activation_ids == ("recovery_frontier",)
    assert bundle.recovery_slot.effect_ceiling is EffectScope.AUTHORITATIVE_WORLD
    assert bundle.recovery_slot.verifier_obligations[0].verifier_tag == ("recovery-safety-gate")


def test_protocol_attachment_monitor_stops_drop_occlusion_and_identity_swap() -> None:
    program = build_fixed_bowl_place_protocol().attachment_monitor_program

    def evaluate(sample: dict[str, object]):
        runtime = MonitorRuntime(
            episode_id="episode",
            workflow_id="place",
            action_id="transport-action",
            plan_digest="sha256:plan",
            program=program,
        )
        return runtime.evaluate(
            sample,
            sequence=1,
            hook=MonitorHook.CONTROL,
            action_id="transport-action",
            plan_digest="sha256:plan",
        )

    normal = evaluate(
        {
            "attachment_status": "verified_held",
            "held_entity_visible": True,
            "identity_match": True,
        }
    )
    dropped = evaluate(
        {
            "attachment_status": "not_held",
            "held_entity_visible": True,
            "identity_match": True,
        }
    )
    occluded = evaluate(
        {
            "attachment_status": "verified_held",
            "held_entity_visible": False,
            "identity_match": True,
        }
    )
    swapped = evaluate(
        {
            "attachment_status": "verified_held",
            "held_entity_visible": True,
            "identity_match": False,
        }
    )

    assert normal.finding is None and not normal.stop_requested
    assert dropped.stop_requested and dropped.finding is not None
    assert dropped.finding.finding is MonitorFindingKind.ATTACHMENT_ANOMALY
    assert occluded.stop_requested and occluded.finding is not None
    assert occluded.finding.finding is MonitorFindingKind.UNOBSERVABLE
    assert swapped.stop_requested and swapped.finding is not None
    assert swapped.finding.finding is MonitorFindingKind.UNSAFE_DEVIATION


def test_all_bowl_coding_profiles_spawn_with_contract_skills_and_strict_context(
    tmp_path,
) -> None:
    protocol = build_fixed_bowl_place_protocol()
    bindings = build_bowl_place_coding_profiles(protocol.spec)
    coding_nodes = {
        node.runner_ref
        for node in protocol.spec.activations
        if node.runner_kind is RunnerKind.CODING_WORKER
    }
    assert set(bindings.profiles) == coding_nodes
    assert "author_sealed_phase_motion" in bindings.required_skill_ids
    assert "plan_bounded_motion" not in bindings.required_skill_ids
    assert "safe_return_home" not in bindings.required_skill_ids

    provider = _alignment_coding_provider(tmp_path)
    actors = ActorRegistry(
        {"skill_coding": provider},
        namespace_root="bowl-coding-bindings",
        workspace_root=tmp_path / "coding-actors",
    )
    runtime = EpisodeRuntime(
        episode_id="bowl-coding-bindings",
        episode_root=tmp_path / "coding-episode",
        actors=actors,
    )
    provider.bind_episode_data_plane(runtime.data_plane)

    for skill_id in bindings.required_skill_ids:
        record = provider.library.get(skill_id)
        assert (record.skill.root / "contract.yaml").is_file()
    for index, (_runner_ref, profile) in enumerate(sorted(bindings.profiles.items())):
        assert profile.effect_ceiling == frozenset()
        assert profile.metadata["strict_runtime_context"] is True
        rows = profile.metadata["node_config_v1"]
        assert rows
        for _activation_id, encoded in rows:
            CodingNodeConfigV1.model_validate(
                {"schema_version": "robomex.coding_node_config.v1", **json.loads(encoded)}
            )
        handle = actors.spawn(profile, actor_id=f"bowl-coding-{index}")
        assert handle.profile.provider_id == "skill_coding"
        assert tuple(profile.metadata["preloaded_skills"])[0] in bindings.required_skill_ids


def test_bowl_coding_runtime_context_is_versioned_unshadowable_and_fingerprinted(
    tmp_path,
) -> None:
    protocol = build_fixed_bowl_place_protocol()
    node = next(
        value
        for value in protocol.spec.activations
        if value.activation_id == "author_attachment_monitor"
    )
    profile = build_bowl_place_coding_profiles(protocol.spec).profiles[node.runner_ref]
    provider = _alignment_coding_provider(tmp_path)
    metadata = {
        "episode_id": "context-episode",
        "workflow_id": "context-run",
        "activation_id": node.activation_id,
        "attempt": 1,
        "graph_id": protocol.spec.graph_id,
        "graph_revision": protocol.spec.revision,
        "graph_digest": protocol.compiled.digest,
        "node_params": dict(node.params),
    }
    spec = InvocationSpec(
        invocation_id="context-command-1",
        idempotency_key="context-command-1",
        objective="author the admitted attachment monitor",
        output_contract={"monitor_program": MONITOR_PROGRAM},
        requested_capabilities=frozenset(node.required_capabilities),
        metadata=metadata,
    )

    first = provider._context_binding(profile, spec)
    replay = provider._context_binding(profile, spec)
    assert first is not None and replay is not None
    assert first.content_digest == replay.content_digest
    assert first.runtime_context.model_dump() == replay.runtime_context.model_dump()

    changed = replace(spec, metadata={**metadata, "attempt": 2})
    changed_binding = provider._context_binding(profile, changed)
    assert changed_binding is not None
    assert changed_binding.content_digest != first.content_digest
    assert changed.fingerprint() != spec.fingerprint()

    with pytest.raises(CodingProviderContractError, match="shadow"):
        provider._context_binding(
            profile,
            replace(spec, inputs={"RUNTIME_CONTEXT_V1": {"forged": True}}),
        )
    with pytest.raises(CodingProviderContractError, match="CodingNodeConfigV1"):
        provider._context_binding(
            profile,
            replace(
                spec,
                metadata={
                    **metadata,
                    "node_params": {**dict(node.params), "unknown_context_field": True},
                },
            ),
        )


def test_strict_context_committed_invocation_replays_without_model_call(tmp_path) -> None:
    protocol = build_fixed_bowl_place_protocol()
    node = next(
        value for value in protocol.spec.activations if value.activation_id == "estimate_alignment"
    )
    profile = build_bowl_place_coding_profiles(protocol.spec).profiles[node.runner_ref]
    provider = _alignment_coding_provider(tmp_path / "strict-replay")
    actors = ActorRegistry(
        {"skill_coding": provider},
        namespace_root="strict-context-replay",
        workspace_root=tmp_path / "strict-replay-actors-first",
    )
    runtime = EpisodeRuntime(
        episode_id="strict-context-replay",
        episode_root=tmp_path / "strict-replay-episode",
        actors=actors,
    )
    provider.bind_episode_data_plane(runtime.data_plane)
    runtime.data_plane.open_workflow("strict-replay-workflow")
    observation = _observation(
        _held(center=Vector3(x=0.51, y=0.0, z=0.2), yaw=0.04),
        _target(center=Vector3(x=0.5, y=0.0, z=0.2)),
    )
    observation_record = runtime.data_plane.publish(
        workflow_id="strict-replay-workflow",
        activation_id="capture_alignment",
        attempt=1,
        port="observation",
        schema="robomex.bowl_place_observation.v1",
        payload=observation.model_dump(mode="json"),
    )
    attachment = AttachmentEvidence(
        evidence_id="strict-replay-attachment",
        entity_id="bowl-1",
        entity_track_id="track-bowl-1",
        status=AttachmentStatus.VERIFIED_HELD.value,
        source_observation_id=observation.snapshot_id,
        source_observation_revision=observation.observation_generation,
        observation_domain="camera.front",
        base_state_revision=0,
        action_id="transport-action",
        evidence_refs=(observation_record.artifact_id,),
        method="synchronized_tracking_and_gripper_guard",
        reason="strict replay fixture confirms attachment",
        confidence=0.98,
        validity=ValidityVector(
            lifecycle=ValidityLifecycle.DERIVED,
            depends_on_revisions={"camera.front": observation.observation_generation},
            observation_id=observation.snapshot_id,
            state_revision=0,
            method="exact_observation_and_state_revision",
            reason="strict replay fixture",
            confidence=0.98,
        ),
    )
    attachment_record = runtime.data_plane.publish(
        workflow_id="strict-replay-workflow",
        activation_id="verify_alignment_attachment",
        attempt=1,
        port="attachment_evidence",
        schema="robomex.attachment_evidence.v1",
        payload=attachment.model_dump(mode="json"),
        lineage=(observation_record.ref,),
    )
    spec = InvocationSpec(
        invocation_id="strict-alignment-command",
        idempotency_key="strict-alignment-command",
        objective="estimate support alignment from the admitted checkpoint",
        inputs={
            "observation": observation_record.ref.to_mapping(),
            "attachment_evidence": attachment_record.ref.to_mapping(),
        },
        output_contract={port.name: port.schema_id for port in node.outputs},
        requested_capabilities=frozenset(node.required_capabilities),
        budget=node.estimated_budget.model_dump(mode="python"),
        metadata={
            "episode_id": runtime.episode_id,
            "workflow_id": "strict-replay-workflow",
            "activation_id": node.activation_id,
            "attempt": 1,
            "graph_id": protocol.spec.graph_id,
            "graph_revision": protocol.spec.revision,
            "graph_digest": protocol.compiled.digest,
            "node_params": dict(node.params),
        },
    )
    expected_handle = actors.spawn(profile, actor_id="strict-alignment-author")
    expected = expected_handle.invoke(spec)
    first_audit = expected_handle.runtime.audits[-1]

    class NoReplayCallPolicy:
        def __init__(self) -> None:
            self.calls = 0

        def complete_bounded(
            self,
            _prompt,
            *,
            max_tokens: int,
            deadline_monotonic_s: float | None = None,
        ) -> str:
            del max_tokens, deadline_monotonic_s
            self.calls += 1
            raise AssertionError("committed strict invocation must not call the model")

    replay_policy = NoReplayCallPolicy()
    restarted = SkillCodingAgentProvider(
        data_plane=runtime.data_plane,
        executor_factory=lambda *_args: _AlignmentProposalExecutor(),
        library=provider.library,
        policy=replay_policy,
        artifacts_root=provider.artifacts_root,
        max_turns=3,
        trusted_skill_sidecars=frozenset(name for _category, name in _BOWL_SKILL_DIRECTORIES),
    )
    replay_handle = ActorRegistry(
        {"skill_coding": restarted},
        namespace_root="strict-context-replay",
        workspace_root=tmp_path / "strict-replay-actors-second",
    ).spawn(profile, actor_id="strict-alignment-author")

    actual = replay_handle.invoke(spec)

    assert actual == expected
    assert replay_policy.calls == 0
    assert replay_handle.runtime.audits[-1].replayed
    assert replay_handle.runtime.audits[-1].context_digest == first_audit.context_digest
    assert first_audit.context_digest.startswith("sha256:")


def test_production_capture_replays_published_artifact_without_repoll(tmp_path) -> None:
    backend = InMemoryObservationBackend("capture-checkpoints")
    _queue_production_checkpoints(backend, scenario="normal")
    runtime, _provider, profile, invocation, handle = _direct_capture_harness(tmp_path, backend)
    result = handle.invoke(invocation)
    assert result.outcome is ControlOutcome.SUCCESS
    emission = result.artifacts[0]
    record = runtime.data_plane.publish_once(
        workflow_id="capture-workflow",
        activation_id="capture_initial_attachment",
        attempt=1,
        port=emission.port,
        schema=emission.schema_id,
        payload=emission.payload,
        lineage=emission.lineage,
    )
    history_size = len(runtime.observations.stream.history)

    _restarted_provider, restarted = _restart_direct_capture_provider(tmp_path, runtime, profile)
    replayed = restarted.invoke(invocation)

    assert replayed.outcome is ControlOutcome.SUCCESS
    assert replayed.artifacts[0].payload == runtime.data_plane.resolve(record.ref).payload
    assert len(runtime.observations.stream.history) == history_size


def test_canonical_monitor_and_motion_sidecars_emit_strict_sealed_contracts() -> None:
    monitor_module = _load_builtin_sidecar(
        "monitor/author_attachment_monitor/scripts/attachment_monitor.py",
        "robomex_test_attachment_monitor_sidecar",
    )
    motion_module = _load_builtin_sidecar(
        "motion/author_sealed_phase_motion/scripts/sealed_phase_motion.py",
        "robomex_test_sealed_motion_sidecar",
    )
    monitor_payload = monitor_module.build_attachment_monitor_program()
    expected_monitor = build_bowl_attachment_monitor_program()
    assert monitor_payload == expected_monitor.spec.model_dump(mode="json")

    snapshot = _admission_snapshot("robot.arm")
    waypoint_rows = (snapshot.joint_positions_rad, (0.05, 0.15))
    first = motion_module.build_sealed_phase_motion(
        snapshot.model_dump(mode="json"),
        plan_kind="bounded_correction",
        joint_waypoints=waypoint_rows,
    )
    second = motion_module.build_sealed_phase_motion(
        snapshot.model_dump(mode="json"),
        plan_kind="bounded_correction",
        joint_waypoints=waypoint_rows,
    )
    plan = MotionPlan.model_validate(first)
    assert first == second
    assert plan.motion.positions_rad[0] == snapshot.joint_positions_rad
    assert plan.tcp_frame_id == "panda_hand"
    assert plan.planner_backend == "curobo"
    with pytest.raises(ValueError, match="exactly equal"):
        motion_module.build_sealed_phase_motion(
            snapshot.model_dump(mode="json"),
            plan_kind="bounded_correction",
            joint_waypoints=((0.01, 0.1), (0.05, 0.15)),
        )


def test_production_bindings_require_continuous_tracking() -> None:
    protocol = build_fixed_bowl_place_protocol()
    provider = BowlPlaceActorProvider(
        BowlPlaceProviderConfig(tracking_mode=BowlTrackingMode.MANUAL_CHECKPOINT)
    )

    with pytest.raises(BowlPlaceProviderError, match="require continuous tracking"):
        build_bowl_place_actor_bindings(protocol.spec, provider=provider)

    bindings = build_bowl_place_actor_bindings(
        protocol.spec,
        provider=provider,
        allow_manual_tracking_for_tests=True,
    )
    assert bindings.providers[bindings.provider_id] is provider


def test_synchronized_sampler_pairs_by_revision_across_async_arrival() -> None:
    _registry, backend, _bowl, _plate, sampler = _continuous_sampler_harness()
    observed_at = datetime.now(UTC)
    clock = time.monotonic() + 1.0
    _push_continuous_sample(
        backend,
        entity_id="bowl-1",
        revision=1,
        monotonic_time_s=clock,
        observed_at=observed_at,
    )
    _push_continuous_sample(
        backend,
        entity_id="plate-1",
        revision=2,
        monotonic_time_s=clock + 0.01,
        observed_at=observed_at,
    )
    _push_continuous_sample(
        backend,
        entity_id="bowl-1",
        revision=2,
        monotonic_time_s=clock + 0.01,
        observed_at=observed_at,
    )

    sampler.acquire("held-bowl-service")
    _wait_until(
        lambda: (
            sampler.latest_pair is not None
            and sampler.latest_pair.bowl.revisions.camera_revision == 2
        )
    )
    pair = sampler.latest_pair
    assert pair is not None
    assert pair.bowl.revisions == pair.plate.revisions
    assert pair.bowl.revisions.camera_revision == 2
    assert pair.bowl.sample_id != pair.plate.sample_id
    sampler.release("held-bowl-service")
    sampler.close()


def test_continuous_sampler_publishes_bounded_silence_and_drop_as_lost() -> None:
    registry, backend, _bowl, _plate, sampler = _continuous_sampler_harness(
        poll_interval_s=0.005,
        max_silence_s=0.04,
    )
    observed_at = datetime.now(UTC)
    clock = time.monotonic() + 1.0
    for entity_id in ("bowl-1", "plate-1"):
        _push_continuous_sample(
            backend,
            entity_id=entity_id,
            revision=1,
            monotonic_time_s=clock,
            observed_at=observed_at,
        )
    sampler.acquire("monitor-service")
    _wait_until(
        lambda: (
            registry.stream.latest("continuous-bowl") is not None
            and registry.stream.latest("continuous-bowl").quality is ObservationQuality.TRACKED
        )
    )
    _wait_until(
        lambda: (
            registry.stream.latest("continuous-bowl").quality is ObservationQuality.LOST
            and registry.stream.latest("continuous-bowl").reason
            == "continuous_tracking_silence_exceeded"
        )
    )
    silence_sequence = registry.stream.latest("continuous-bowl").sequence
    time.sleep(0.06)
    assert registry.stream.latest("continuous-bowl").sequence == silence_sequence

    next_observed_at = datetime.now(UTC)
    _push_continuous_sample(
        backend,
        entity_id="bowl-1",
        revision=2,
        monotonic_time_s=clock + 1.0,
        observed_at=next_observed_at,
        quality=ObservationQuality.LOST,
    )
    _push_continuous_sample(
        backend,
        entity_id="plate-1",
        revision=2,
        monotonic_time_s=clock + 1.0,
        observed_at=next_observed_at,
    )
    _wait_until(lambda: registry.stream.latest("continuous-bowl").reason == "object_dropped")
    assert registry.stream.latest("continuous-bowl").signals == {}
    sampler.release("monitor-service")
    sampler.close()


def test_bowl_provider_shares_sampler_and_obeys_actor_lifecycle(tmp_path) -> None:
    registry = ObservationRegistry(episode_id="tracking-lifecycle")
    backend = InMemoryObservationBackend("tracking-lifecycle-backend")
    registry.register_backend(backend)
    provider = BowlPlaceActorProvider(
        BowlPlaceProviderConfig(
            observation_backend_id=backend.backend_id,
            tracking_mode=BowlTrackingMode.CONTINUOUS,
            tracking_poll_interval_s=0.005,
            tracking_max_silence_s=0.04,
        )
    )
    actors = ActorRegistry(
        {"bowl_place": provider},
        namespace_root="tracking-lifecycle",
        workspace_root=tmp_path / "tracking-actors",
    )
    runtime = EpisodeRuntime(
        episode_id="tracking-lifecycle",
        episode_root=tmp_path / "tracking-episode",
        actors=actors,
        observation_registry=registry,
        freshness_context_provider=provider.freshness_context,
    )
    provider.bind_episode_runtime(runtime)

    def profile(profile_id: str, runner_ref: str) -> ActorProfile:
        return ActorProfile(
            profile_id=profile_id,
            provider_id="bowl_place",
            runner_kind=RunnerKind.TRACKING_SERVICE.value,
            lifecycle=ActorLifecycle.SERVICE,
            capability_ceiling=frozenset({"perception.track"}),
            effect_ceiling=frozenset(),
            metadata={"runner_ref": runner_ref},
        )

    def invocation(invocation_id: str, activation_id: str) -> InvocationSpec:
        return InvocationSpec(
            invocation_id=invocation_id,
            idempotency_key=invocation_id,
            objective="maintain continuous read-only tracking",
            requested_capabilities=frozenset({"perception.track"}),
            metadata={
                "episode_id": runtime.episode_id,
                "workflow_id": "tracking-workflow",
                "activation_id": activation_id,
                "attempt": 1,
            },
        )

    held = actors.spawn(
        profile(
            "held-tracker-profile",
            "robomex.bowl_place.track_held_bowl",
        ),
        actor_id="held-tracker",
    )
    plate = actors.spawn(
        profile("plate-tracker-profile", "robomex.bowl_place.track_plate"),
        actor_id="plate-tracker",
    )
    assert held.invoke(invocation("held-start", "held_bowl_tracker")) is None
    assert plate.invoke(invocation("plate-start", "plate_tracker")) is None
    sampler = provider.tracking_sampler
    assert sampler is not None
    _wait_until(lambda: sampler.running)
    assert sampler.owner_count == 2

    held.suspend()
    assert sampler.sampling_enabled
    plate.suspend()
    _wait_until(lambda: not sampler.sampling_enabled)
    assert sampler.bowl_track.state is TrackState.SUSPENDED
    assert sampler.plate_track.state is TrackState.SUSPENDED

    plate.resume()
    assert sampler.sampling_enabled
    assert sampler.bowl_track.state is TrackState.ACTIVE
    held.retire()
    assert sampler.owner_count == 1
    plate.retire()
    _wait_until(lambda: not sampler.running)
    assert sampler.owner_count == 0
    assert sampler.bowl_track.state is TrackState.SUSPENDED
    restarted_tracker = actors.spawn(
        profile(
            "restarted-tracker-profile",
            "robomex.bowl_place.track_held_bowl",
        ),
        actor_id="restarted-tracker",
    )
    restarted_tracker.invoke(invocation("restarted-start", "held_bowl_tracker"))
    _wait_until(lambda: sampler.running)
    assert sampler.owner_count == 1
    runtime.close_episode()
    _wait_until(lambda: not sampler.running)
    assert sampler.owner_count == 0
    assert sampler.bowl_track.state is TrackState.STOPPED
    assert sampler.plate_track.state is TrackState.STOPPED
    assert provider.close()


def test_continuous_capture_consumes_sampler_owned_synchronized_pair(tmp_path) -> None:
    registry = ObservationRegistry(episode_id="continuous-capture")
    backend = InMemoryObservationBackend("continuous-capture-backend")
    registry.register_backend(backend)
    observed_at = datetime.now(UTC)
    clock = time.monotonic() + 1.0
    for entity_id in ("bowl-1", "plate-1"):
        _push_continuous_sample(
            backend,
            entity_id=entity_id,
            revision=1,
            monotonic_time_s=clock,
            observed_at=observed_at,
        )
    provider = BowlPlaceActorProvider(
        BowlPlaceProviderConfig(
            observation_backend_id=backend.backend_id,
            tracking_mode=BowlTrackingMode.CONTINUOUS,
            tracking_poll_interval_s=0.01,
            tracking_max_silence_s=1.0,
            tracking_capture_wait_timeout_s=0.5,
        )
    )
    actors = ActorRegistry(
        {"bowl_place": provider},
        namespace_root="continuous-capture",
        workspace_root=tmp_path / "continuous-capture-actors",
    )
    runtime = EpisodeRuntime(
        episode_id="continuous-capture",
        episode_root=tmp_path / "continuous-capture-episode",
        actors=actors,
        observation_registry=registry,
        freshness_context_provider=provider.freshness_context,
    )
    provider.bind_episode_runtime(runtime)
    runtime.data_plane.open_workflow("capture-workflow")
    tracker = actors.spawn(
        ActorProfile(
            profile_id="continuous-capture-tracker",
            provider_id="bowl_place",
            runner_kind=RunnerKind.TRACKING_SERVICE.value,
            lifecycle=ActorLifecycle.SERVICE,
            capability_ceiling=frozenset({"perception.track"}),
            effect_ceiling=frozenset(),
            metadata={"runner_ref": "robomex.bowl_place.track_held_bowl"},
        ),
        actor_id="continuous-capture-tracker",
    )
    tracker.invoke(
        InvocationSpec(
            invocation_id="continuous-tracker-start",
            idempotency_key="continuous-tracker-start",
            objective="start continuous tracking",
            requested_capabilities=frozenset({"perception.track"}),
            metadata={
                "episode_id": runtime.episode_id,
                "workflow_id": "capture-workflow",
                "activation_id": "held_bowl_tracker",
                "attempt": 1,
            },
        )
    )
    sampler = provider.tracking_sampler
    assert sampler is not None
    _wait_until(lambda: sampler.latest_pair is not None)

    capture = actors.spawn(
        ActorProfile(
            profile_id="continuous-capture-gate",
            provider_id="bowl_place",
            runner_kind=RunnerKind.DETERMINISTIC_GATE.value,
            lifecycle=ActorLifecycle.EPHEMERAL,
            capability_ceiling=frozenset({"perception.observe"}),
            effect_ceiling=frozenset(),
            metadata={"runner_ref": "robomex.bowl_place.capture_synchronized_checkpoint"},
        ),
        actor_id="continuous-capture-gate",
    )
    result = capture.invoke(
        InvocationSpec(
            invocation_id="continuous-capture-command",
            idempotency_key="continuous-capture-command",
            objective="capture the sampler-owned synchronized pair",
            output_contract={"observation": "robomex.bowl_place_observation.v1"},
            requested_capabilities=frozenset({"perception.observe"}),
            metadata={
                "episode_id": runtime.episode_id,
                "workflow_id": "capture-workflow",
                "activation_id": "capture_initial_attachment",
                "attempt": 1,
            },
        )
    )
    assert result.outcome is ControlOutcome.SUCCESS
    observation = BowlPlaceObservation.model_validate(result.artifacts[0].payload)
    assert observation.observation_generation == 1
    assert observation.held is not None
    assert observation.held.evidence_refs[0] == sampler.latest_pair.bowl.sample_id
    assert observation.target.evidence_refs[0] == sampler.latest_pair.plate.sample_id
    tracker.retire()
    provider.close()


def test_production_capture_started_without_artifact_never_repolls(tmp_path) -> None:
    backend = InMemoryObservationBackend("capture-checkpoints")
    revisions = ObservationRevisionVector(
        scene_revision=1,
        arm_revision=1,
        gripper_revision=1,
        attachment_revision=1,
        camera_revision=1,
    )
    backend.push(
        BackendObservation(
            entity_id="bowl-1",
            quality=ObservationQuality.TRACKED,
            revisions=revisions,
            signals={
                "attachment_status": "verified_held",
                "center_x": 0.52,
                "center_y": 0.0,
                "center_z": 0.2,
                "identity_match": True,
                "visibility": "visible",
                "yaw": 0.1,
            },
            monotonic_time_s=time.monotonic() + 1.0,
        )
    )
    runtime, _provider, profile, invocation, handle = _direct_capture_harness(tmp_path, backend)
    first = handle.invoke(invocation)
    assert first.outcome is ControlOutcome.STALE_OBSERVATION
    history_size = len(runtime.observations.stream.history)
    assert history_size == 2  # tracked bowl plus fail-closed LOST plate sample

    _restarted_provider, restarted = _restart_direct_capture_provider(tmp_path, runtime, profile)
    replayed = restarted.invoke(invocation)

    assert replayed.outcome is ControlOutcome.STALE_INPUT
    assert "sensor polling is not retried" in (replayed.reason or "")
    assert len(runtime.observations.stream.history) == history_size


def test_restarted_provider_rejects_backend_stale_checkpoint_replay(tmp_path) -> None:
    backend = InMemoryObservationBackend("capture-checkpoints")
    _queue_production_checkpoints(backend, scenario="stale_generation")
    runtime, provider, profile, invocation, handle = _direct_capture_harness(tmp_path, backend)

    def capture_and_publish(spec: InvocationSpec, actor_handle) -> None:
        result = actor_handle.invoke(spec)
        assert result.outcome is ControlOutcome.SUCCESS
        emission = result.artifacts[0]
        runtime.data_plane.publish_once(
            workflow_id="capture-workflow",
            activation_id="capture_initial_attachment",
            attempt=int(spec.metadata["attempt"]),
            port=emission.port,
            schema=emission.schema_id,
            payload=emission.payload,
            lineage=emission.lineage,
        )

    capture_and_publish(invocation, handle)
    second = replace(
        invocation,
        invocation_id="capture-command-2",
        idempotency_key="capture-command-2",
        metadata={**invocation.metadata, "attempt": 2},
    )
    second_handle = ActorRegistry(
        {"bowl_place": provider},
        namespace_root="capture-crash-second",
        workspace_root=tmp_path / "actors-second",
    ).spawn(profile, actor_id="capture-actor-second")
    capture_and_publish(second, second_handle)
    history_size = len(runtime.observations.stream.history)

    _restarted_provider, restarted = _restart_direct_capture_provider(tmp_path, runtime, profile)
    third = replace(
        invocation,
        invocation_id="capture-command-3",
        idempotency_key="capture-command-3",
        metadata={**invocation.metadata, "attempt": 3},
    )
    stale = restarted.invoke(third)

    assert stale.outcome is ControlOutcome.STALE_OBSERVATION
    assert "action-invalidated revision" in (stale.reason or "")
    assert len(runtime.observations.stream.history) == history_size + 2


def test_production_capture_accepts_new_camera_frame_in_stable_world() -> None:
    provider = BowlPlaceActorProvider(BowlPlaceProviderConfig())
    floor = _observation(
        _held(center=Vector3(x=0.52, y=0.0, z=0.2), yaw=0.1),
        _target(center=Vector3(x=0.5, y=0.0, z=0.2)),
    )
    stable_world_new_camera = RevisionVector(
        scene=1,
        arm=1,
        gripper=1,
        attachment=1,
        camera={"front": 2},
    )

    assert provider._strictly_advances_checkpoint(
        generation=2,
        revisions=stable_world_new_camera,
        floor=floor,
        required_domains=frozenset(),
    )
    assert not provider._strictly_advances_checkpoint(
        generation=2,
        revisions=stable_world_new_camera,
        floor=floor,
        required_domains=frozenset({"arm"}),
    )


def test_episode_runtime_runs_bounded_bowl_loop_through_final_verify(tmp_path) -> None:
    (
        runtime,
        workflow_id,
        backend,
        driver,
        bowl_provider,
        alignment_provider,
    ) = _episode_fixture(tmp_path, scenario="normal")

    terminal = runtime.run_until_terminal(workflow_id)

    assert terminal.status.value == "succeeded"
    assert terminal.terminal_activation == "placement_complete"
    assert terminal.loop_iterations["alignment_visual_servo"] >= 1
    gate_outcomes = [
        event.outcome
        for event in runtime.event_bus.history
        if isinstance(event, NodeOutcomeEvent) and event.activation_id == "alignment_gate"
    ]
    assert gate_outcomes == [ControlOutcome.NEEDS_ADJUSTMENT, ControlOutcome.SUCCESS]
    assert [primitive for primitive, _ in backend.calls].count(
        "execute_joint_path_cooperative"
    ) == 3
    assert [primitive for primitive, _ in backend.calls].count("execute_joint_path") == 1
    assert [primitive for primitive, _ in backend.calls].count("set_gripper") == 1
    assert [primitive for primitive, _ in backend.calls].count("wait") == 1
    assert "author_attachment_monitor" in driver.invoked
    assert {
        "capture_initial_attachment",
        "capture_alignment",
        "capture_pre_release",
        "capture_release",
        "capture_final_relation",
        "checkpoint_pre_release",
        "checkpoint_post_release",
    }.issubset(bowl_provider.invocations)
    assert "verify_placement" in bowl_provider.invocations
    assert "placement_complete" in bowl_provider.invocations
    assert "plan_open" in bowl_provider.invocations
    assert "plan_settle" in bowl_provider.invocations
    assert "plan_open" not in driver.invoked
    assert "plan_settle" not in driver.invoked
    assert len(alignment_provider.invocation_audits) == 3
    assert all(
        audit.loaded_skill_ids == ("estimate_support_alignment",)
        for audit in alignment_provider.invocation_audits
    )
    assert all(
        audit.context_digest.startswith("sha256:") for audit in alignment_provider.invocation_audits
    )
    assert "estimate_alignment" not in bowl_provider.invocations
    assert "estimate_pre_release_alignment" not in bowl_provider.invocations
    published_schemas = {record.schema for record in runtime.data_plane.artifacts}
    assert {
        "robomex.attachment_evidence.v1",
        "robomex.attachment_guard.v1",
        "robomex.bowl_place_observation.v1",
        "robomex.held_bowl_estimate.v1",
        "robomex.plate_support_target.v1",
        "robomex.alignment_error.v1",
        "robomex.servo_decision.v1",
        "robomex.phase_checkpoint.v1",
        "robomex.placement_verdict.v1",
        "robomex.execution_receipt.v2",
        "robomex.relation_evidence.v1",
        "robomex.state_transition_proposal.v1",
        "robomex.state_commit_receipt.v1",
        MONITOR_PROGRAM,
    }.issubset(published_schemas)

    alignment_records = [
        record
        for record in runtime.data_plane.artifacts
        if record.activation_id == "capture_alignment"
    ]
    alignment_observations = [
        BowlPlaceObservation.model_validate(runtime.data_plane.resolve(record.ref).payload)
        for record in alignment_records
    ]
    assert [item.observation_generation for item in alignment_observations] == [2, 3]
    assert [item.revisions.arm for item in alignment_observations] == [2, 3]

    def sole(activation_id: str, port: str):
        return next(
            record
            for record in runtime.data_plane.artifacts
            if record.activation_id == activation_id and record.port == port
        )

    open_receipt = sole("execute_open", "receipt")
    settle_receipt = sole("execute_settle", "receipt")
    release_observation = sole("capture_release", "observation")
    release_evidence = sole("verify_release_attachment", "attachment_evidence")
    release_proposal = sole("propose_release_attachment", "proposal")
    release_commit = sole("commit_release_attachment", "receipt")
    assert {open_receipt.ref, settle_receipt.ref}.issubset(release_observation.lineage)
    assert release_observation.ref in release_evidence.lineage
    assert {release_evidence.ref, release_observation.ref}.issubset(release_proposal.lineage)
    assert {release_proposal.ref, release_evidence.ref}.issubset(release_commit.lineage)

    final_observation = sole("capture_final_relation", "observation")
    final_evidence = sole("verify_placement", "relation_evidence")
    final_verdict = sole("verify_placement", "verdict")
    final_proposal = sole("propose_final_relation", "proposal")
    final_commit = sole("commit_final_relation", "receipt")
    assert final_observation.ref in final_evidence.lineage
    assert {final_evidence.ref, final_observation.ref}.issubset(final_proposal.lineage)
    assert {final_proposal.ref, final_evidence.ref}.issubset(final_commit.lineage)
    assert runtime.state_reducer.state.revision == 7
    assert runtime.state_reducer.state.attachment.status is AttachmentStatus.NOT_HELD
    assert runtime.state_reducer.state.relations[0].value is RelationValue.ASSERTED

    monitor_record = next(
        record for record in runtime.data_plane.artifacts if record.schema == MONITOR_PROGRAM
    )
    expected_monitor = build_bowl_attachment_monitor_program()
    assert runtime.data_plane.resolve(monitor_record.ref).payload == (
        expected_monitor.spec.model_dump(mode="json")
    )
    monitored_receipt_records = [
        record
        for record in runtime.data_plane.artifacts
        if record.schema == EXECUTION_RECEIPT
        and record.activation_id in {"execute_transport", "execute_correction", "execute_descend"}
    ]
    assert all(monitor_record.ref in record.lineage for record in monitored_receipt_records)
    monitored_receipts = [
        ExecutionReceipt.model_validate(runtime.data_plane.resolve(record.ref).payload)
        for record in monitored_receipt_records
    ]
    assert len(monitored_receipts) == 3
    assert {receipt.monitor_digest for receipt in monitored_receipts} == {expected_monitor.digest}

    def coding_input(record):
        return {
            "payload": runtime.data_plane.resolve(record.ref).payload,
            "ref": record.ref.to_mapping(),
            "lineage": tuple(ref.to_mapping() for ref in record.lineage),
        }

    attachment_module = _load_builtin_sidecar(
        "state/propose_attachment_transition/scripts/attachment_transition.py",
        "robomex_test_attachment_transition_sidecar",
    )
    canonical_attachment = StateTransitionProposalWire.model_validate(
        attachment_module.build_attachment_transition_proposal(
            coding_input(release_observation),
            coding_input(release_evidence),
            episode_id=runtime.episode_id,
        )
    ).to_domain()
    assert canonical_attachment.attachment_status is AttachmentStatus.NOT_HELD
    assert canonical_attachment.observation_ref == release_observation.ref
    assert {release_observation.ref, release_evidence.ref}.issubset(
        canonical_attachment.evidence_refs
    )

    relation_module = _load_builtin_sidecar(
        "state/propose_relation_transition/scripts/relation_transition.py",
        "robomex_test_relation_transition_sidecar",
    )
    canonical_relation = StateTransitionProposalWire.model_validate(
        relation_module.build_relation_transition_proposal(
            coding_input(final_observation),
            coding_input(final_evidence),
            coding_input(final_verdict),
            episode_id=runtime.episode_id,
        )
    ).to_domain()
    assert canonical_relation.relation_predicate is RelationPredicate.SUPPORTED_BY
    assert canonical_relation.relation_value is RelationValue.ASSERTED
    assert canonical_relation.observation_ref == final_observation.ref
    assert {final_observation.ref, final_evidence.ref, final_verdict.ref}.issubset(
        canonical_relation.evidence_refs
    )


@pytest.mark.parametrize("scenario", ["dropped", "occluded"])
def test_episode_runtime_failure_evidence_never_reaches_release(tmp_path, scenario: str) -> None:
    runtime, workflow_id, backend, driver, bowl_provider, _alignment_provider = _episode_fixture(
        tmp_path, scenario=scenario
    )

    terminal = runtime.run_until_terminal(workflow_id)

    assert terminal.status.value == "failed"
    assert terminal.terminal_activation == "recovery_frontier"
    primitives = [primitive for primitive, _ in backend.calls]
    assert primitives == ["execute_joint_path_cooperative", "stop", "hold"]
    assert "plan_open" not in driver.invoked
    assert "plan_settle" not in driver.invoked
    assert "verify_placement" not in driver.invoked
    assert "capture_alignment" not in driver.invoked
    assert "plan_open" not in bowl_provider.invocations
    assert "plan_settle" not in bowl_provider.invocations
    assert "verify_placement" not in bowl_provider.invocations
    assert "capture_alignment" not in bowl_provider.invocations
    action_nodes = {
        event.activation_id
        for event in runtime.event_bus.history
        if isinstance(event, NodeOutcomeEvent)
    }
    assert "execute_open" not in action_nodes
    assert "execute_settle" not in action_nodes
    assert "recovery_frontier" in action_nodes
    interrupted = next(
        ExecutionReceipt.model_validate(runtime.data_plane.resolve(record.ref).payload)
        for record in runtime.data_plane.artifacts
        if record.schema == EXECUTION_RECEIPT
    )
    assert interrupted.runtime_status is ExecutionStatus.INTERRUPTED
    assert interrupted.triggering_finding_id is not None
    assert interrupted.monitor_digest == build_bowl_attachment_monitor_program().digest


@pytest.mark.parametrize(
    ("scenario", "activation_id", "outcome"),
    [
        ("checkpoint_occluded", "verify_alignment_attachment", ControlOutcome.UNCERTAIN),
        ("stale_generation", "capture_alignment", ControlOutcome.STALE_OBSERVATION),
        ("target_drift", "alignment_gate", ControlOutcome.TARGET_DRIFT),
    ],
)
def test_checkpoint_failure_never_reaches_open(
    tmp_path, scenario: str, activation_id: str, outcome: ControlOutcome
) -> None:
    runtime, workflow_id, _backend, driver, bowl_provider, _alignment_provider = _episode_fixture(
        tmp_path, scenario=scenario
    )

    terminal = runtime.run_until_terminal(workflow_id)

    assert terminal.status.value == "failed"
    assert terminal.terminal_activation == "recovery_frontier"
    assert "plan_open" not in driver.invoked
    assert "plan_open" not in bowl_provider.invocations
    assert "execute_open" not in {
        event.activation_id
        for event in runtime.event_bus.history
        if isinstance(event, NodeOutcomeEvent)
    }
    assert any(
        isinstance(event, NodeOutcomeEvent)
        and event.activation_id == activation_id
        and event.outcome is outcome
        for event in runtime.event_bus.history
    )


def test_protocol_digest_changes_when_servo_budget_changes() -> None:
    first = build_fixed_bowl_place_protocol(BowlPlaceProtocolConfig(max_step_translation_m=0.01))
    second = build_fixed_bowl_place_protocol(BowlPlaceProtocolConfig(max_step_translation_m=0.02))

    assert first.compiled.digest != second.compiled.digest
    assert len(first.compiled.digest) == 64
    assert first.spec.metadata["closed_servo_outcomes"]["dropped"] == (
        ControlOutcome.ATTACHMENT_NOT_CONFIRMED.value
    )
