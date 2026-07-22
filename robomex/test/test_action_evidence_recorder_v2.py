from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from robomex.authoring.monitoring import (
    MonitorCompiler,
    MonitorHook,
    MonitorProgramSpec,
    MonitorRuntime,
)
from robomex.data.artifact_resolver import ArtifactIntegrityError
from robomex.data.episode_plane import EpisodeDataPlane
from robomex.data.schema_registry import SchemaRegistry
from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    BackendCallResult,
    BackendDescriptor,
    BackendMotionInterface,
    ExecutionPolicy,
    ExecutionStatus,
    JointPath,
    MonitorTelemetryCapabilities,
    MonitorTelemetryHook,
    MotionPlan,
)
from robomex.runtime.authority import (
    ActionSupervisor,
    InMemoryActionWAL,
    SealedActionRunner,
)
from robomex.runtime.evidence_recorder import (
    ACTION_EVIDENCE_FRAME_SCHEMA,
    ACTION_EVIDENCE_VIDEO_SCHEMA,
    ACTION_RUNTIME_SAMPLE_SCHEMA,
    ActionEvidenceFrame,
    ActionEvidenceVideo,
    ActionRuntimeEvidenceSample,
    EncodedEvidenceMedia,
    EpisodeActionEvidenceRecorder,
    EvidenceAlreadyFinalizedError,
    EvidenceFrameInput,
    decode_evidence_ref,
    install_action_evidence_schemas,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


class _SeedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["robomex.evidence_seed.v1"] = "robomex.evidence_seed.v1"
    plan: str


def _registry() -> SchemaRegistry:
    registry = SchemaRegistry()
    registry.ensure("robomex.evidence_seed.v1", _SeedArtifact)
    install_action_evidence_schemas(registry)
    return registry


def _plane(path: Path) -> EpisodeDataPlane:
    return EpisodeDataPlane(
        path,
        episode_id="episode-evidence",
        schema_registry=_registry(),
        strict_schema_prefixes=("robomex.",),
    )


def _seed(plane: EpisodeDataPlane):
    plane.open_workflow("workflow")
    return plane.publish_once(
        workflow_id="workflow",
        activation_id="plan_author",
        attempt=1,
        port="sealed_plan",
        schema="robomex.evidence_seed.v1",
        payload=_SeedArtifact(plan="exact-joint-path").model_dump(mode="json"),
    )


class _FrameEncoder:
    encoder_id = "test-frame-png-v1"

    def encode(self, **kwargs: object) -> EncodedEvidenceMedia:
        payload = f"png:{kwargs['sequence']}:{kwargs['phase']}".encode()
        return EncodedEvidenceMedia(
            media_type="image/png",
            data=payload,
            codec="test-png",
            metadata={"deterministic": True},
        )


class _VideoEncoder:
    encoder_id = "test-video-mp4-v1"

    def encode(
        self, *, action_id: str, frames: Sequence[EvidenceFrameInput]
    ) -> EncodedEvidenceMedia:
        payload = "\n".join(
            [action_id, *(frame.ref.content_digest for frame in frames)]
        ).encode()
        return EncodedEvidenceMedia(
            media_type="video/mp4",
            data=payload,
            codec="test-h264",
            metadata={"fps": 10},
        )


class _UnavailableVideoEncoder:
    encoder_id = "optional-ffmpeg-v1"

    def encode(self, **kwargs: object) -> EncodedEvidenceMedia:
        raise ModuleNotFoundError("optional codec unavailable")


def test_frames_are_digest_bound_lineaged_and_never_derive_paths_from_input(
    tmp_path: Path,
) -> None:
    plane = _plane(tmp_path / "episode")
    seed = _seed(plane)
    recorder = EpisodeActionEvidenceRecorder(
        plane,
        workflow_id="workflow",
        base_lineage=(seed.ref,),
        frame_encoder=_FrameEncoder(),
    )
    hostile_action_id = "../../outside/action"

    encoded_ref = recorder.record(
        action_id=hostile_action_id,
        sequence=1,
        phase="../../frames/pre",
        sample={"visible": True, "confidence": 0.9},
    )
    assert encoded_ref is not None
    ref = decode_evidence_ref(encoded_ref)
    resolved = plane.resolve(ref)
    frame = ActionEvidenceFrame.model_validate(resolved.payload)
    record = plane.artifact_record(ref.artifact_id)

    assert resolved.schema == ACTION_EVIDENCE_FRAME_SCHEMA
    assert frame.action_id == hostile_action_id
    assert frame.phase == "../../frames/pre"
    assert frame.media is not None
    assert frame.encoder_id == "test-frame-png-v1"
    assert frame.media.decode() == b"png:1:../../frames/pre"
    assert frame.media.byte_digest == "sha256:" + hashlib.sha256(
        frame.media.decode()
    ).hexdigest()
    assert record.lineage == (seed.ref,)
    assert record.activation_id.startswith("action_evidence_")
    assert hostile_action_id not in record.activation_id
    assert record.content_path.startswith("objects/sha256/")
    assert not (tmp_path / "outside").exists()

    # Exact retry is idempotent; changing the immutable sequence payload fails closed.
    assert recorder.record(
        action_id=hostile_action_id,
        sequence=1,
        phase="../../frames/pre",
        sample={"visible": True, "confidence": 0.9},
    ) == encoded_ref
    with pytest.raises(ArtifactIntegrityError, match="cannot be rebound"):
        recorder.record(
            action_id=hostile_action_id,
            sequence=1,
            phase="post",
            sample={"visible": False},
        )


def test_finalize_is_restart_recoverable_idempotent_and_closes_the_sequence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "episode"
    first_plane = _plane(root)
    seed = _seed(first_plane)
    first = EpisodeActionEvidenceRecorder(
        first_plane,
        workflow_id="workflow",
        base_lineage=(seed.ref,),
    )
    second_ref = first.record(
        action_id="action-restart",
        sequence=2,
        phase="post",
        sample={"position": [0.2, 0.3]},
    )
    first_ref = first.record(
        action_id="action-restart",
        sequence=1,
        phase="pre",
        sample={"position": [0.0, 0.1]},
    )

    # Simulate a process restart before finalize.  The new recorder has no
    # private checkpoint or configured lineage; it recovers both from the ledger.
    restarted_plane = _plane(root)
    restarted = EpisodeActionEvidenceRecorder(
        restarted_plane,
        workflow_id="workflow",
    )
    frame_refs, video_ref = restarted.finalize(action_id="action-restart")
    assert video_ref is not None
    assert frame_refs == (first_ref, second_ref)

    video_artifact_ref = decode_evidence_ref(video_ref)
    video_resolved = restarted_plane.resolve(video_artifact_ref)
    video = ActionEvidenceVideo.model_validate(video_resolved.payload)
    video_record = restarted_plane.artifact_record(video_artifact_ref.artifact_id)
    assert video_resolved.schema == ACTION_EVIDENCE_VIDEO_SCHEMA
    assert video.representation == "deterministic_manifest"
    assert video.fallback_reason == "encoder_not_configured"
    assert tuple(item.ref for item in video.source_lineage) == (seed.ref,)
    assert video.frame_sequences == (1, 2)
    assert video.frame_phases == ("pre", "post")
    assert video_record.lineage == (
        seed.ref,
        decode_evidence_ref(first_ref),
        decode_evidence_ref(second_ref),
    )

    # Simulate a crash immediately after the video publication but before its
    # caller consumed the return.  Replay returns the exact immutable refs.
    after_publish_plane = _plane(root)
    after_publish = EpisodeActionEvidenceRecorder(
        after_publish_plane,
        workflow_id="workflow",
    )
    assert after_publish.finalize(action_id="action-restart") == (
        frame_refs,
        video_ref,
    )
    with pytest.raises(EvidenceAlreadyFinalizedError):
        after_publish.record(
            action_id="action-restart",
            sequence=3,
            phase="late",
            sample={"position": [1.0, 1.0]},
        )


def test_video_encoder_and_dependency_free_fallback_share_the_complete_interface(
    tmp_path: Path,
) -> None:
    plane = _plane(tmp_path / "episode")
    _seed(plane)
    encoded = EpisodeActionEvidenceRecorder(
        plane,
        workflow_id="workflow",
        video_encoder=_VideoEncoder(),
    )
    encoded.record(
        action_id="action-video",
        sequence=1,
        phase="control",
        sample={"waypoint": 1},
    )
    _, encoded_video_ref = encoded.finalize(action_id="action-video")
    assert encoded_video_ref is not None
    encoded_video = ActionEvidenceVideo.model_validate(
        encoded.resolve_receipt_ref(encoded_video_ref).payload
    )
    assert encoded_video.representation == "encoded_video"
    assert encoded_video.media is not None
    assert encoded_video.media.media_type == "video/mp4"
    assert encoded_video.media.decode().startswith(b"action-video\nsha256:")

    fallback = EpisodeActionEvidenceRecorder(
        plane,
        workflow_id="workflow",
        video_encoder=_UnavailableVideoEncoder(),
    )
    fallback.record(
        action_id="action-fallback",
        sequence=1,
        phase="control",
        sample={"waypoint": 1},
    )
    _, fallback_video_ref = fallback.finalize(action_id="action-fallback")
    assert fallback_video_ref is not None
    fallback_video = ActionEvidenceVideo.model_validate(
        fallback.resolve_receipt_ref(fallback_video_ref).payload
    )
    assert fallback_video.representation == "deterministic_manifest"
    assert fallback_video.fallback_reason == "dependency_unavailable"
    assert fallback_video.encoder_id == "optional-ffmpeg-v1"


def _snapshot() -> AdmissionSnapshot:
    return AdmissionSnapshot(
        world_id="live-world",
        resource_id="arm",
        robot_revision=1,
        scene_revision=2,
        attachment_revision=3,
        config_revision=4,
        joint_names=("joint_a", "joint_b"),
        joint_positions_rad=(0.0, 0.1),
        config_digest=_digest("a"),
        collision_world_digest=_digest("b"),
        attachment_status="verified_held",
        controller_state="ready",
        captured_at=datetime(
            2026, 7, 22, tzinfo=timezone.utc  # noqa: UP017 - Python 3.10 compatibility
        ),
    )


def _motion(snapshot: AdmissionSnapshot) -> MotionPlan:
    return MotionPlan(
        plan_id="evidence-motion",
        plan_kind="bounded_alignment",
        tcp_frame_id="panda_hand",
        planner_backend="test-planner",
        robot_model_digest=_digest("c"),
        expected_snapshot=snapshot,
        motion=JointPath(
            joint_names=snapshot.joint_names,
            positions_rad=(snapshot.joint_positions_rad, (0.2, 0.3)),
            execution_policy=ExecutionPolicy(timeout_s=2.0),
        ),
        possibly_affected_revisions=("robot.arm", "scene", "attachment"),
    )


class _PhaseBackend:
    descriptor = BackendDescriptor(
        backend_id="evidence-test-backend",
        motion_interface=BackendMotionInterface.EXACT_JOINT_PATH,
    )

    def __init__(self, snapshot: AdmissionSnapshot, *, fail: bool) -> None:
        self._snapshot = snapshot
        self._fail = fail
        self._samples = [{"visible": True}, {"visible": True}]
        self.monitor_sequences: list[int] = []

    def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot:
        assert (world_id, resource_id) == (
            self._snapshot.world_id,
            self._snapshot.resource_id,
        )
        return self._snapshot

    def monitor_sample(self, **kwargs: object) -> Mapping[str, object]:
        if kwargs["phase"] == "compatibility_preflight":
            return self._samples[0]
        self.monitor_sequences.append(int(kwargs["sequence"]))
        return self._samples.pop(0)

    def monitor_telemetry_capabilities(
        self, *, world_id: str, resource_id: str
    ) -> MonitorTelemetryCapabilities:
        return MonitorTelemetryCapabilities(
            world_id=world_id,
            resource_id=resource_id,
            always_available_signals=("visible",),
            supported_hooks=(MonitorTelemetryHook.PHASE,),
        )

    def execute_joint_path(self, **kwargs: object) -> BackendCallResult:
        if self._fail:
            raise RuntimeError("controller transport failed")
        return BackendCallResult(converged=True, telemetry={"finished": True})

    def set_gripper(self, **kwargs: object) -> BackendCallResult:
        raise AssertionError("unexpected gripper command")

    def wait(self, **kwargs: object) -> BackendCallResult:
        raise AssertionError("unexpected wait command")


def _monitor(action_id: str, plan: MotionPlan) -> tuple[str, MonitorRuntime]:
    program = MonitorCompiler().compile(
        MonitorProgramSpec(
            monitor_id="evidence-monitor",
            source="""
def evaluate(sample):
    if sample["visible"] == False:
        return {"finding": "attachment_anomaly", "severity": "critical"}
    return None
""",
            hook=MonitorHook.PHASE,
            allowed_signals=("visible",),
            max_runtime_ms=50,
        )
    )
    return program.digest, MonitorRuntime(
        episode_id="episode-evidence",
        workflow_id="workflow",
        action_id=action_id,
        plan_digest=plan.content_digest,
        program=program,
    )


@pytest.mark.parametrize("backend_fails", [False, True])
def test_sealed_runner_writes_frames_and_video_into_receipt_even_on_failure(
    tmp_path: Path, backend_fails: bool
) -> None:
    plane = _plane(tmp_path / ("failure" if backend_fails else "success"))
    _seed(plane)
    recorder = EpisodeActionEvidenceRecorder(plane, workflow_id="workflow")
    snapshot = _snapshot()
    plan = _motion(snapshot)
    action_id = "action-failure" if backend_fails else "action-success"
    monitor_digest, monitor = _monitor(action_id, plan)
    supervisor = ActionSupervisor(InMemoryActionWAL())
    admitted = supervisor.admit(
        plan,
        snapshot,
        action_id=action_id,
        monitor_digest=monitor_digest,
    )

    backend = _PhaseBackend(snapshot, fail=backend_fails)
    receipt = SealedActionRunner(supervisor, backend).run(
        admitted, monitor=monitor, evidence_recorder=recorder
    )

    assert receipt.runtime_status is (
        ExecutionStatus.INDETERMINATE_AFTER_CRASH
        if backend_fails
        else ExecutionStatus.COMPLETED
    )
    expected_phases = (
        (
            "execution/pre_primitive",
            "monitor/pre_primitive",
            "primitive/exception",
            "execution/terminal",
        )
        if backend_fails
        else (
            "execution/pre_primitive",
            "monitor/pre_primitive",
            "primitive/post",
            "monitor/post_primitive",
            "execution/terminal",
        )
    )
    assert len(receipt.frame_refs) == len(expected_phases)
    assert receipt.video_ref is not None
    frames = tuple(
        ActionEvidenceFrame.model_validate(recorder.resolve_receipt_ref(ref).payload)
        for ref in receipt.frame_refs
    )
    assert tuple(frame.sequence for frame in frames) == tuple(
        range(1, len(expected_phases) + 1)
    )
    assert tuple(frame.phase for frame in frames) == expected_phases
    assert backend.monitor_sequences == ([1] if backend_fails else [1, 2])
    for frame in frames:
        if frame.phase.startswith("monitor/"):
            assert "schema_version" not in frame.sample
        else:
            assert frame.sample["schema_version"] == ACTION_RUNTIME_SAMPLE_SCHEMA
            ActionRuntimeEvidenceSample.model_validate(frame.sample)
    video = ActionEvidenceVideo.model_validate(
        recorder.resolve_receipt_ref(receipt.video_ref).payload
    )
    assert tuple(item.ref for item in video.frame_refs) == tuple(
        decode_evidence_ref(ref) for ref in receipt.frame_refs
    )
    assert recorder.finalize(action_id=action_id) == (
        receipt.frame_refs,
        receipt.video_ref,
    )


def test_sealed_runner_records_complete_runtime_boundaries_without_monitor(
    tmp_path: Path,
) -> None:
    plane = _plane(tmp_path / "no-monitor")
    _seed(plane)
    recorder = EpisodeActionEvidenceRecorder(plane, workflow_id="workflow")
    snapshot = _snapshot()
    plan = _motion(snapshot)
    supervisor = ActionSupervisor(InMemoryActionWAL())
    admitted = supervisor.admit(plan, snapshot, action_id="action-no-monitor")

    receipt = SealedActionRunner(
        supervisor,
        _PhaseBackend(snapshot, fail=False),
    ).run(admitted, evidence_recorder=recorder)

    assert receipt.runtime_status is ExecutionStatus.COMPLETED
    assert receipt.video_ref is not None
    frames = tuple(
        ActionEvidenceFrame.model_validate(recorder.resolve_receipt_ref(ref).payload)
        for ref in receipt.frame_refs
    )
    assert tuple(frame.phase for frame in frames) == (
        "execution/pre_primitive",
        "primitive/post",
        "execution/terminal",
    )
    assert tuple(frame.sequence for frame in frames) == (1, 2, 3)
    samples = tuple(
        ActionRuntimeEvidenceSample.model_validate(frame.sample) for frame in frames
    )
    assert tuple(sample.sample_type for sample in samples) == (
        "execution_pre_primitive",
        "primitive_post",
        "execution_terminal",
    )
    assert all(
        sample.primitive_args_digest
        == receipt.primitive_receipts[0].exact_args_digest
        for sample in samples
    )


class _CrashSignal(BaseException):
    pass


class _CrashBackend(_PhaseBackend):
    def __init__(self, snapshot: AdmissionSnapshot) -> None:
        super().__init__(snapshot, fail=False)
        self.executions = 0

    def execute_joint_path(self, **kwargs: object) -> BackendCallResult:
        self.executions += 1
        raise _CrashSignal("simulated process loss")


def test_pre_primitive_evidence_survives_crash_and_restart_without_replay(
    tmp_path: Path,
) -> None:
    root = tmp_path / "crash-restart"
    first_plane = _plane(root)
    _seed(first_plane)
    first_recorder = EpisodeActionEvidenceRecorder(
        first_plane, workflow_id="workflow"
    )
    snapshot = _snapshot()
    plan = _motion(snapshot)
    supervisor = ActionSupervisor(InMemoryActionWAL())
    admitted = supervisor.admit(plan, snapshot, action_id="action-crash")
    backend = _CrashBackend(snapshot)

    with pytest.raises(_CrashSignal, match="simulated process loss"):
        SealedActionRunner(supervisor, backend).run(
            admitted, evidence_recorder=first_recorder
        )

    assert backend.executions == 1
    assert supervisor.blocked_resources == {("live-world", "arm")}
    assert [record.record_kind for record in supervisor.wal.records()] == [
        "action_attempt"
    ]

    restarted_plane = _plane(root)
    restarted_recorder = EpisodeActionEvidenceRecorder(
        restarted_plane, workflow_id="workflow"
    )
    frame_refs, video_ref = restarted_recorder.finalize(action_id="action-crash")
    assert video_ref is not None
    assert restarted_recorder.finalize(action_id="action-crash") == (
        frame_refs,
        video_ref,
    )
    assert len(frame_refs) == 1
    frame = ActionEvidenceFrame.model_validate(
        restarted_recorder.resolve_receipt_ref(frame_refs[0]).payload
    )
    assert frame.phase == "execution/pre_primitive"
    ActionRuntimeEvidenceSample.model_validate(frame.sample)

    reconciled = supervisor.reconcile_orphans()
    assert len(reconciled) == 1
    assert reconciled[0].runtime_status is ExecutionStatus.INDETERMINATE_AFTER_CRASH
    assert backend.executions == 1
