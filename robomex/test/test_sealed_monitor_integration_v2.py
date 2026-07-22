from __future__ import annotations

import pytest

from robomex.authoring.monitoring import (
    MonitorCompiler,
    MonitorHook,
    MonitorProgramSpec,
    MonitorRuntime,
)
from robomex.runtime.action_protocol import (
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
from robomex.test.test_action_protocol_v2 import FakeBackend, _motion

SOURCE = """
def evaluate(sample):
    if sample["object_visible"] == False:
        return {
            "finding": "attachment_anomaly",
            "severity": "critical",
            "details": {"visible": False},
        }
    return None
"""


def _program(hook: MonitorHook):
    return MonitorCompiler().compile(
        MonitorProgramSpec(
            monitor_id=f"guard-{hook.value}",
            source=SOURCE,
            hook=hook,
            allowed_signals=("object_visible",),
            max_runtime_ms=50,
        )
    )


def _admit(program):
    plan = _motion()
    supervisor = ActionSupervisor(InMemoryActionWAL())
    admitted = supervisor.admit(
        plan,
        plan.expected_snapshot,
        action_id="action-monitor",
        monitor_digest=program.digest,
    )
    monitor = MonitorRuntime(
        episode_id="ep",
        workflow_id="wf",
        action_id=admitted.action_id,
        plan_digest=plan.content_digest,
        program=program,
    )
    return plan, supervisor, admitted, monitor


class PhaseBackend(FakeBackend):
    def __init__(self, snapshot, samples):
        super().__init__(snapshot)
        self.monitor_samples = list(samples)

    def monitor_sample(self, **kwargs):
        if kwargs["phase"] == "compatibility_preflight":
            return self.monitor_samples[0]
        return self.monitor_samples.pop(0)

    def monitor_telemetry_capabilities(self, *, world_id, resource_id):
        return MonitorTelemetryCapabilities(
            world_id=world_id,
            resource_id=resource_id,
            always_available_signals=("object_visible",),
            supported_hooks=(MonitorTelemetryHook.PHASE,),
        )


class CooperativeBackend(FakeBackend):
    def __init__(self, snapshot, samples):
        super().__init__(snapshot)
        self.progress_samples = list(samples)
        self.callback_results = []

    def monitor_sample(self, **kwargs):
        del kwargs
        return self.progress_samples[0]

    def monitor_telemetry_capabilities(self, *, world_id, resource_id):
        return MonitorTelemetryCapabilities(
            world_id=world_id,
            resource_id=resource_id,
            always_available_signals=("object_visible",),
            supported_hooks=(MonitorTelemetryHook.WAYPOINT,),
            cooperative_stop_guaranteed=True,
        )

    def stop_and_wait_quiescent(self, *, world_id, resource_id, timeout_s):
        del timeout_s
        return self.snapshot(world_id, resource_id).model_copy(
            update={"controller_state": ControllerState.QUIESCENT}
        )

    def execute_joint_path_cooperative(self, *, progress_callback, **kwargs):
        self.calls.append(("execute_joint_path_cooperative", kwargs))
        for sample in self.progress_samples:
            should_continue = progress_callback(sample)
            self.callback_results.append(should_continue)
            if not should_continue:
                return BackendCallResult(converged=False, interrupted=True)
        return BackendCallResult(converged=True)


def test_phase_monitor_can_interrupt_before_any_world_changing_primitive() -> None:
    program = _program(MonitorHook.PHASE)
    plan, supervisor, admitted, monitor = _admit(program)
    backend = PhaseBackend(plan.expected_snapshot, [{"object_visible": False}])
    findings = []

    receipt = SealedActionRunner(supervisor, backend).run(
        admitted,
        monitor=monitor,
        finding_sink=findings.append,
    )

    assert receipt.runtime_status is ExecutionStatus.INTERRUPTED
    assert receipt.primitive_receipts == ()
    assert receipt.triggering_finding_id == findings[0].finding_id
    assert receipt.monitor_digest == program.digest
    assert backend.calls == []


def test_phase_monitor_post_check_prevents_success_continuation() -> None:
    program = _program(MonitorHook.PHASE)
    plan, supervisor, admitted, monitor = _admit(program)
    backend = PhaseBackend(
        plan.expected_snapshot,
        [{"object_visible": True}, {"object_visible": False}],
    )

    receipt = SealedActionRunner(supervisor, backend).run(admitted, monitor=monitor)

    assert receipt.runtime_status is ExecutionStatus.INTERRUPTED
    assert len(receipt.primitive_receipts) == 1
    assert len(backend.calls) == 1


def test_waypoint_monitor_stops_backend_before_later_progress_samples() -> None:
    program = _program(MonitorHook.WAYPOINT)
    plan, supervisor, admitted, monitor = _admit(program)
    backend = CooperativeBackend(
        plan.expected_snapshot,
        [
            {"object_visible": True},
            {"object_visible": False},
            {"object_visible": True},
        ],
    )

    receipt = SealedActionRunner(supervisor, backend).run(admitted, monitor=monitor)

    assert receipt.runtime_status is ExecutionStatus.INTERRUPTED
    assert backend.callback_results == [True, False]
    assert len(receipt.primitive_receipts) == 1


def test_waypoint_monitor_without_cooperative_backend_fails_closed_without_motion() -> None:
    program = _program(MonitorHook.WAYPOINT)
    plan, supervisor, admitted, monitor = _admit(program)
    backend = FakeBackend(plan.expected_snapshot)

    with pytest.raises(AdmissionRejectedError, match="typed monitor telemetry"):
        SealedActionRunner(supervisor, backend).run(admitted, monitor=monitor)

    assert backend.calls == []
    assert supervisor.wal.records() == ()
    assert not supervisor.has_lease(plan.world_id, plan.resource_id)
