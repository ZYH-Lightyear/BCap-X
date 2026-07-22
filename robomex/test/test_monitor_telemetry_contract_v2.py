from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from robomex.authoring.monitoring import (
    MonitorCompiler,
    MonitorHook,
    MonitorProgramSpec,
    MonitorRuntime,
)
from robomex.data.embodied_state import AttachmentStatus
from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    BackendCallResult,
    ControllerState,
    ExecutionStatus,
    MonitorTelemetryCapabilities,
    MonitorTelemetryHook,
)
from robomex.runtime.authority import (
    ActionSupervisor,
    AdmissionRejectedError,
    InMemoryActionWAL,
    SealedActionRunner,
)
from robomex.runtime.capx_action_backend import (
    CapXActionBackendError,
    CapXBackendConfigurationError,
    CapXSealedActionBackend,
    LiberoControlPort,
    ResourceRole,
)
from robomex.runtime.monitor_telemetry import (
    AuthoritativeAttachmentTelemetry,
    BowlMonitorTelemetryBridge,
    MonitorTelemetryUnavailableError,
)
from robomex.runtime.observation import (
    BackendObservation,
    EntityIdentity,
    InMemoryObservationBackend,
    ObservationQuality,
    ObservationRegistry,
    ObservationRevisionVector,
    TrackRequest,
)
from robomex.test.test_action_protocol_v2 import FakeBackend, _motion
from robomex.test.test_capx_action_backend_v2 import FakeLiberoEnv, Harness

_VISIBLE_SOURCE = """
def evaluate(sample):
    if sample["object_visible"] == False:
        return {
            "finding": "unobservable",
            "severity": "critical",
            "details": {"reason": "not_visible"},
        }
    return None
"""


def _monitor(
    *,
    hook: MonitorHook,
    action_id: str,
    plan_digest: str,
) -> tuple[object, MonitorRuntime]:
    program = MonitorCompiler().compile(
        MonitorProgramSpec(
            monitor_id=f"telemetry-{hook.value}",
            source=_VISIBLE_SOURCE,
            hook=hook,
            allowed_signals=("object_visible",),
            max_runtime_ms=50,
        )
    )
    return program, MonitorRuntime(
        episode_id="episode",
        workflow_id="workflow",
        action_id=action_id,
        plan_digest=plan_digest,
        program=program,
    )


class DeclaredMonitorBackend(FakeBackend):
    def __init__(
        self,
        snapshot: AdmissionSnapshot,
        *,
        signals: tuple[str, ...] = ("object_visible",),
        hooks: tuple[MonitorTelemetryHook, ...] = (MonitorTelemetryHook.CONTROL,),
        preflight: dict[str, object] | None = None,
        progress: tuple[dict[str, object], ...] = (),
        ignore_callback_stop: bool = False,
    ) -> None:
        super().__init__(snapshot)
        self.signals = signals
        self.hooks = hooks
        self.preflight = preflight or {"object_visible": True}
        self.progress = progress or ({"object_visible": True},)
        self.ignore_callback_stop = ignore_callback_stop
        self.stop_calls = 0

    def monitor_telemetry_capabilities(self, *, world_id, resource_id):
        return MonitorTelemetryCapabilities(
            world_id=world_id,
            resource_id=resource_id,
            always_available_signals=self.signals,
            supported_hooks=self.hooks,
            cooperative_stop_guaranteed=True,
        )

    def monitor_sample(self, **kwargs):
        del kwargs
        return self.preflight

    def execute_joint_path_cooperative(self, *, progress_callback, **kwargs):
        self.calls.append(("execute_joint_path_cooperative", kwargs))
        interrupted = False
        for sample in self.progress:
            if not progress_callback(sample):
                interrupted = True
                if not self.ignore_callback_stop:
                    break
        if self.ignore_callback_stop:
            return BackendCallResult(converged=True, interrupted=False)
        return BackendCallResult(converged=not interrupted, interrupted=interrupted)

    def stop_and_wait_quiescent(self, *, world_id, resource_id, timeout_s):
        del timeout_s
        self.stop_calls += 1
        return self.snapshot(world_id, resource_id).model_copy(
            update={"controller_state": ControllerState.QUIESCENT}
        )


class RecordingEvidence:
    def __init__(self) -> None:
        self.samples: list[tuple[str, dict[str, object]]] = []

    def record(self, *, action_id, sequence, phase, sample):
        del action_id, sequence
        self.samples.append((phase, dict(sample)))
        return None

    def finalize(self, *, action_id):
        del action_id
        return (), None


def _admitted_monitor(hook: MonitorHook):
    plan = _motion(timeout_s=1.0)
    wal = InMemoryActionWAL()
    supervisor = ActionSupervisor(wal)
    program, monitor = _monitor(
        hook=hook,
        action_id="telemetry-action",
        plan_digest=plan.content_digest,
    )
    admitted = supervisor.admit(
        plan,
        plan.expected_snapshot,
        action_id="telemetry-action",
        monitor_digest=program.digest,
    )
    return plan, wal, supervisor, admitted, monitor


@pytest.mark.parametrize(
    ("signals", "hooks", "message"),
    [
        (("joint_positions_rad",), (MonitorTelemetryHook.CONTROL,), "guarantee"),
        (("object_visible",), (MonitorTelemetryHook.WAYPOINT,), "does not support"),
    ],
)
def test_incompatible_monitor_is_rejected_before_wal_or_primitive(
    signals, hooks, message
) -> None:
    plan, wal, supervisor, admitted, monitor = _admitted_monitor(MonitorHook.CONTROL)
    backend = DeclaredMonitorBackend(
        plan.expected_snapshot,
        signals=signals,
        hooks=hooks,
    )

    with pytest.raises(AdmissionRejectedError, match=message):
        SealedActionRunner(supervisor, backend).run(admitted, monitor=monitor)

    assert wal.records() == ()
    assert backend.calls == []
    assert not supervisor.has_lease(plan.world_id, plan.resource_id)


def test_raw_progress_superset_is_evidence_only_and_monitor_gets_projection() -> None:
    plan, _wal, supervisor, admitted, monitor = _admitted_monitor(MonitorHook.CONTROL)
    backend = DeclaredMonitorBackend(
        plan.expected_snapshot,
        progress=(
            {
                "object_visible": True,
                "joint_positions_rad": [0.0, 0.1],
                "control_step": 1,
                "cooperative_granularity": "control",
            },
        ),
    )
    evidence = RecordingEvidence()

    receipt = SealedActionRunner(supervisor, backend).run(
        admitted,
        monitor=monitor,
        evidence_recorder=evidence,
    )

    assert receipt.runtime_status is ExecutionStatus.COMPLETED
    monitor_sample = next(
        sample for phase, sample in evidence.samples if phase == "monitor/control"
    )
    assert monitor_sample["control_step"] == 1
    assert monitor_sample["joint_positions_rad"] == [0.0, 0.1]


def test_signal_disappearing_after_preflight_becomes_critical_stop() -> None:
    plan, _wal, supervisor, admitted, monitor = _admitted_monitor(MonitorHook.CONTROL)
    backend = DeclaredMonitorBackend(
        plan.expected_snapshot,
        preflight={"object_visible": True},
        progress=({"control_step": 1},),
    )

    receipt = SealedActionRunner(supervisor, backend).run(admitted, monitor=monitor)

    assert receipt.runtime_status is ExecutionStatus.INTERRUPTED
    assert receipt.triggering_finding_id is not None


def test_backend_ignoring_stop_callback_is_quarantined_not_falsely_interrupted() -> None:
    plan, _wal, supervisor, admitted, monitor = _admitted_monitor(MonitorHook.CONTROL)
    backend = DeclaredMonitorBackend(
        plan.expected_snapshot,
        progress=({"object_visible": False}, {"object_visible": True}),
        ignore_callback_stop=True,
    )

    receipt = SealedActionRunner(supervisor, backend).run(admitted, monitor=monitor)

    assert receipt.runtime_status is ExecutionStatus.INDETERMINATE_AFTER_TIMEOUT
    assert receipt.runtime_status is not ExecutionStatus.INTERRUPTED
    assert receipt.terminal_telemetry["cooperative_protocol_violation"] is True
    assert backend.stop_calls == 1
    assert (plan.world_id, plan.resource_id) in supervisor.blocked_resources


class NonThreadSafeHangingBackend(FakeBackend):
    def __init__(self, snapshot: AdmissionSnapshot) -> None:
        super().__init__(snapshot)
        self.release = threading.Event()
        self.stop_calls = 0

    def execute_joint_path(self, **kwargs):
        self.calls.append(("execute_joint_path", kwargs))
        self.release.wait(timeout=1.0)
        return BackendCallResult(converged=False)

    def stop_and_wait_quiescent(self, **kwargs):
        del kwargs
        self.stop_calls += 1
        self.release.set()
        raise AssertionError("non-thread-safe stop must not be called concurrently")


def test_watchdog_does_not_concurrently_stop_non_thread_safe_controller() -> None:
    plan = _motion(timeout_s=0.03)
    backend = NonThreadSafeHangingBackend(plan.expected_snapshot)
    supervisor = ActionSupervisor(InMemoryActionWAL())

    receipt = SealedActionRunner(supervisor, backend).run(
        supervisor.admit(plan, plan.expected_snapshot)
    )

    assert receipt.runtime_status is ExecutionStatus.INDETERMINATE_AFTER_TIMEOUT
    assert receipt.terminal_telemetry["controller_stop_thread_safe"] is False
    assert receipt.terminal_telemetry["controller_stop_attempted"] is False
    assert backend.stop_calls == 0
    backend.release.set()


def test_libero_provider_requires_exact_declared_non_reserved_json_signals() -> None:
    env = FakeLiberoEnv()
    with pytest.raises(CapXBackendConfigurationError, match="explicitly declare"):
        LiberoControlPort(
            env,
            max_gripper_width_m=0.08,
            sample_provider=lambda: {"attachment_status": "verified_held"},
        )
    with pytest.raises(CapXBackendConfigurationError, match="reserved"):
        LiberoControlPort(
            env,
            max_gripper_width_m=0.08,
            sample_provider=lambda: {"joint_positions_rad": [0.0, 0.1]},
            sample_signal_names=("joint_positions_rad",),
            sample_world_resource_bindings=(("live-world", "arm"),),
        )
    with pytest.raises(CapXBackendConfigurationError, match="world/resource"):
        LiberoControlPort(
            env,
            max_gripper_width_m=0.08,
            sample_provider=lambda: {"object_visible": True},
            sample_signal_names=("object_visible",),
        )
    port = LiberoControlPort(
        env,
        max_gripper_width_m=0.08,
        sample_provider=lambda: {"wrong": True},
        sample_signal_names=("object_visible",),
        sample_world_resource_bindings=(("live-world", "arm"),),
    )
    with pytest.raises(CapXActionBackendError, match="exactly sample_signal_names"):
        port.sample()


def test_capx_declares_control_hook_only_when_tracking_step_exists() -> None:
    tracked = Harness().backend.monitor_telemetry_capabilities(
        world_id="live-world", resource_id="arm"
    )
    assert MonitorTelemetryHook.CONTROL in tracked.supported_hooks

    harness = Harness()
    harness.env._tracking_step = None  # type: ignore[method-assign]
    port = LiberoControlPort(harness.env, max_gripper_width_m=0.08)
    backend = CapXSealedActionBackend(
        backend_id="waypoint-only",
        control=port,
        snapshots=harness.provider,
        resource_roles={"arm": ResourceRole.ARM},
        allowed_world_ids=("live-world",),
    )
    waypoint_only = backend.monitor_telemetry_capabilities(
        world_id="live-world", resource_id="arm"
    )
    assert MonitorTelemetryHook.WAYPOINT in waypoint_only.supported_hooks
    assert MonitorTelemetryHook.CONTROL not in waypoint_only.supported_hooks


def _tracking_bridge(*, now_mono: float, now_utc: datetime):
    registry = ObservationRegistry(episode_id="episode")
    backend = InMemoryObservationBackend("tracker")
    registry.register_backend(backend)
    request = TrackRequest(
        episode_id="episode",
        track_id="bowl-track",
        entity=EntityIdentity(entity_id="bowl", semantic_label="bowl"),
        backend_id="tracker",
        declared_signals=("identity_match", "visibility"),
    )
    handle = registry.create(request, start=True)
    attachment = {
        "value": AuthoritativeAttachmentTelemetry(
            world_id="live-world",
            resource_id="arm",
            entity_id="bowl",
            track_id="bowl-track",
            attachment_status=AttachmentStatus.VERIFIED_HELD,
            attachment_revision=7,
            observed_at=now_utc,
            monotonic_time_s=now_mono,
        )
    }
    bridge = BowlMonitorTelemetryBridge(
        registry=registry,
        track_id="bowl-track",
        expected_entity_id="bowl",
        world_id="live-world",
        resource_id="arm",
        attachment_reader=lambda _world, _resource: attachment["value"],
        max_observation_age_s=0.25,
        max_attachment_age_s=0.25,
        monotonic_clock=lambda: now_mono,
        utc_clock=lambda: now_utc,
    )
    return backend, handle, attachment, bridge


def _push_tracking(
    backend,
    handle,
    *,
    now_mono: float,
    now_utc: datetime,
    identity_match: bool = True,
    visibility: str = "visible",
    quality: ObservationQuality = ObservationQuality.TRACKED,
):
    backend.push(
        BackendObservation(
            entity_id="bowl",
            quality=quality,
            revisions=ObservationRevisionVector(
                scene_revision=3,
                arm_revision=4,
                gripper_revision=5,
                attachment_revision=7,
                camera_revision=9,
            ),
            signals=(
                {"identity_match": identity_match, "visibility": visibility}
                if quality is ObservationQuality.TRACKED
                else {}
            ),
            reason=None if quality is ObservationQuality.TRACKED else "tracker_lost",
            observed_at=now_utc,
            monotonic_time_s=now_mono,
        )
    )
    return handle.poll()


def test_bowl_bridge_rejects_stale_or_lost_tracking_without_reusing_signals() -> None:
    now_utc = datetime(2026, 7, 22, 9, 0, tzinfo=timezone.utc)  # noqa: UP017
    backend, handle, _attachment, bridge = _tracking_bridge(
        now_mono=100.0,
        now_utc=now_utc,
    )
    _push_tracking(
        backend,
        handle,
        now_mono=99.0,
        now_utc=now_utc - timedelta(seconds=1),
    )
    with pytest.raises(MonitorTelemetryUnavailableError, match="stale"):
        bridge()

    _push_tracking(
        backend,
        handle,
        now_mono=100.0,
        now_utc=now_utc,
        quality=ObservationQuality.LOST,
    )
    with pytest.raises(MonitorTelemetryUnavailableError, match="tracker is lost"):
        bridge()


def test_bowl_bridge_exposes_identity_visibility_and_authoritative_drop() -> None:
    now_utc = datetime(2026, 7, 22, 9, 0, tzinfo=timezone.utc)  # noqa: UP017
    backend, handle, attachment, bridge = _tracking_bridge(
        now_mono=100.0,
        now_utc=now_utc,
    )
    _push_tracking(
        backend,
        handle,
        now_mono=100.0,
        now_utc=now_utc,
        identity_match=False,
        visibility="occluded",
    )
    assert bridge() == {
        "attachment_status": "verified_held",
        "held_entity_visible": False,
        "identity_match": False,
    }

    attachment["value"] = attachment["value"].model_copy(
        update={"attachment_status": AttachmentStatus.NOT_HELD}
    )
    assert bridge()["attachment_status"] == "not_held"


def test_bowl_bridge_rejects_stale_or_revision_mismatched_attachment() -> None:
    now_utc = datetime(2026, 7, 22, 9, 0, tzinfo=timezone.utc)  # noqa: UP017
    backend, handle, attachment, bridge = _tracking_bridge(
        now_mono=100.0,
        now_utc=now_utc,
    )
    _push_tracking(
        backend,
        handle,
        now_mono=100.0,
        now_utc=now_utc,
    )
    attachment["value"] = attachment["value"].model_copy(
        update={
            "observed_at": now_utc - timedelta(seconds=1),
            "monotonic_time_s": 99.0,
        }
    )
    with pytest.raises(MonitorTelemetryUnavailableError, match="attachment.*stale"):
        bridge()

    attachment["value"] = AuthoritativeAttachmentTelemetry(
        world_id="live-world",
        resource_id="arm",
        entity_id="bowl",
        track_id="bowl-track",
        attachment_status=AttachmentStatus.VERIFIED_HELD,
        attachment_revision=8,
        observed_at=now_utc,
        monotonic_time_s=100.0,
    )
    with pytest.raises(MonitorTelemetryUnavailableError, match="revisions differ"):
        bridge()
