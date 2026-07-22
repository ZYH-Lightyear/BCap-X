from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context
from pathlib import Path

import pytest
from pydantic import ValidationError

from robomex.runtime.action_protocol import (
    ActionAttempt,
    AdmissionSnapshot,
    BackendCallResult,
    BackendDescriptor,
    BackendMotionInterface,
    ControllerState,
    ExecutionPolicy,
    ExecutionReceipt,
    ExecutionStatus,
    FeasibilityStatus,
    GripperCommand,
    JointPath,
    MotionPlan,
    PrimitiveReceipt,
    PrimitiveStatus,
    ShadowRolloutReceipt,
    WaitSpec,
    WorldKind,
    validate_action_spec,
)
from robomex.runtime.authority import (
    ActionLeaseConflictError,
    ActionSupervisor,
    AdmissionRejectedError,
    InMemoryActionWAL,
    JsonlActionWAL,
    RecoveryBlockedError,
    SealedActionMismatchError,
    SealedActionRunner,
    ShadowIsolationError,
    ShadowRolloutRunner,
    WalConflictError,
    build_feasibility_certificate,
)


def _digest(char: str) -> str:
    return "sha256:" + char * 64


CAPTURED_AT = datetime(
    2026, 7, 22, 4, 0, tzinfo=timezone.utc  # noqa: UP017 - Python 3.10
)


def _snapshot(
    *,
    world_id: str = "live-world",
    world_kind: WorldKind = WorldKind.AUTHORITATIVE,
    resource_id: str = "arm",
    robot_revision: int = 1,
    scene_revision: int = 2,
    attachment_revision: int = 3,
    config_revision: int = 4,
    config_digest: str | None = None,
    joints: tuple[float, ...] = (0.0, 0.1),
    captured_at: datetime = CAPTURED_AT,
) -> AdmissionSnapshot:
    return AdmissionSnapshot(
        world_id=world_id,
        world_kind=world_kind,
        resource_id=resource_id,
        robot_revision=robot_revision,
        scene_revision=scene_revision,
        attachment_revision=attachment_revision,
        config_revision=config_revision,
        joint_names=("joint_a", "joint_b"),
        joint_positions_rad=joints,
        config_digest=config_digest or _digest("a"),
        collision_world_digest=_digest("b"),
        attachment_status="verified_held",
        controller_state="ready",
        captured_at=captured_at,
    )


def _motion(
    snapshot: AdmissionSnapshot | None = None,
    *,
    plan_id: str = "motion-1",
    subsample: int = 1,
    endpoint: tuple[float, float] = (0.2, 0.3),
    timeout_s: float = 30.0,
) -> MotionPlan:
    snap = snapshot or _snapshot()
    return MotionPlan(
        plan_id=plan_id,
        plan_kind="bounded_alignment",
        tcp_frame_id="panda_hand",
        planner_backend="fake-curobo",
        robot_model_digest=_digest("c"),
        expected_snapshot=snap,
        max_start_deviation_rad=0.02,
        motion=JointPath(
            joint_names=snap.joint_names,
            positions_rad=(snap.joint_positions_rad, endpoint),
            execution_policy=ExecutionPolicy(
                subsample=subsample, timeout_s=timeout_s
            ),
        ),
        possibly_affected_revisions=("robot.arm", "scene", "attachment"),
    )


def _gripper(snapshot: AdmissionSnapshot | None = None) -> GripperCommand:
    return GripperCommand(
        command_id="gripper-open-1",
        expected_snapshot=snapshot or _snapshot(resource_id="gripper"),
        mode="open",
        target_width_m=0.08,
        timeout_s=2.0,
        possibly_affected_revisions=("robot.gripper", "attachment", "scene"),
    )


def _wait(snapshot: AdmissionSnapshot | None = None) -> WaitSpec:
    return WaitSpec(
        wait_id="settle-1",
        expected_snapshot=snapshot or _snapshot(resource_id="controller"),
        duration_s=0.25,
        timeout_s=1.0,
        possibly_affected_revisions=("scene",),
    )


class SimulatedProcessCrash(BaseException):
    pass


class FakeBackend:
    def __init__(
        self,
        snapshots: AdmissionSnapshot | tuple[AdmissionSnapshot, ...],
        *,
        wal: InMemoryActionWAL | None = None,
        motion_interface: BackendMotionInterface = BackendMotionInterface.EXACT_JOINT_PATH,
        result: object | None = None,
        failure: BaseException | None = None,
        delay_s: float = 0.0,
    ) -> None:
        values = snapshots if isinstance(snapshots, tuple) else (snapshots,)
        self.snapshots = {(item.world_id, item.resource_id): item for item in values}
        self.descriptor = BackendDescriptor(
            backend_id="fake-backend", motion_interface=motion_interface
        )
        self.wal = wal
        self.result = result if result is not None else BackendCallResult(converged=True)
        self.failure = failure
        self.delay_s = delay_s
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._lock = threading.Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot:
        return self.snapshots[(world_id, resource_id)]

    def _invoke(self, primitive: str, kwargs: dict[str, object]) -> BackendCallResult:
        if self.wal is not None:
            records = self.wal.records()
            assert records and isinstance(records[-1], ActionAttempt)
        with self._lock:
            self.calls.append((primitive, kwargs))
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.delay_s:
                time.sleep(self.delay_s)
            if self.failure is not None:
                raise self.failure
            return self.result  # type: ignore[return-value]
        finally:
            with self._lock:
                self.active_calls -= 1

    def execute_joint_path(self, **kwargs: object) -> BackendCallResult:
        return self._invoke("execute_joint_path", kwargs)

    def set_gripper(self, **kwargs: object) -> BackendCallResult:
        return self._invoke("set_gripper", kwargs)

    def wait(self, **kwargs: object) -> BackendCallResult:
        return self._invoke("wait", kwargs)


class StoppableHangingBackend(FakeBackend):
    def __init__(self, snapshot: AdmissionSnapshot) -> None:
        super().__init__(snapshot)
        self.descriptor = self.descriptor.model_copy(
            update={"watchdog_stop_thread_safe": True}
        )
        self.stop_requested = threading.Event()

    def _invoke(self, primitive: str, kwargs: dict[str, object]) -> BackendCallResult:
        self.calls.append((primitive, kwargs))
        self.stop_requested.wait(timeout=2.0)
        return BackendCallResult(converged=False, interrupted=True)

    def stop_and_wait_quiescent(
        self, *, world_id: str, resource_id: str, timeout_s: float
    ) -> AdmissionSnapshot:
        self.stop_requested.set()
        snapshot = self.snapshot(world_id, resource_id)
        return snapshot.model_copy(
            update={"controller_state": ControllerState.QUIESCENT}
        )


def _attempt_process_lease(
    plan_payload: dict[str, object], lock_root: str, result_queue
) -> None:
    plan = MotionPlan.model_validate(plan_payload)
    supervisor = ActionSupervisor(
        InMemoryActionWAL(), interprocess_lock_root=lock_root
    )
    try:
        admitted = supervisor.admit(plan, plan.expected_snapshot)
    except Exception as exc:  # pragma: no cover - assertion happens in parent
        result_queue.put(type(exc).__name__)
    else:  # pragma: no cover - safety regression path
        result_queue.put("admitted")
        supervisor.release(admitted)


def test_motion_plan_digest_is_canonical_and_covers_exact_execution_policy() -> None:
    plan = _motion()
    same = MotionPlan.model_validate(plan.model_dump(mode="json"))
    changed_path = _motion(endpoint=(0.21, 0.3))
    changed_subsample = _motion(subsample=2)
    changed_config = _motion(
        _snapshot(config_revision=5, config_digest=_digest("d"))
    )

    assert same.content_digest == plan.content_digest
    assert changed_path.content_digest != plan.content_digest
    assert changed_subsample.content_digest != plan.content_digest
    assert changed_config.content_digest != plan.content_digest

    tampered = plan.model_dump(mode="json")
    tampered["motion"]["positions_rad"][-1][0] = 0.99
    with pytest.raises(ValidationError, match="content_digest"):
        MotionPlan.model_validate(tampered)


def test_action_specs_are_strict_versioned_and_not_pose_programs() -> None:
    plan = _motion()
    payload = plan.model_dump(mode="json")
    payload["schema_version"] = "robomex.motion_plan.v3"
    with pytest.raises(ValidationError, match="schema_version"):
        MotionPlan.model_validate(payload)

    payload = plan.model_dump(mode="json")
    payload["cartesian_goal_for_runner_to_solve"] = [0.4, 0.1, 0.2]
    with pytest.raises(ValidationError, match="cartesian_goal"):
        MotionPlan.model_validate(payload)

    with pytest.raises(ValidationError, match="exactly one"):
        WaitSpec(
            wait_id="bad-wait",
            expected_snapshot=_snapshot(),
            duration_s=1.0,
            control_steps=20,
            timeout_s=2.0,
            possibly_affected_revisions=("scene",),
        )


def test_model_copy_tampering_is_revalidated_instead_of_trusted() -> None:
    plan = _motion()
    tampered_path = plan.motion.model_copy(
        update={"positions_rad": ((0.0, 0.1), (0.9, 0.9))}
    )
    tampered = plan.model_copy(update={"motion": tampered_path})

    assert tampered.content_digest == plan.content_digest
    with pytest.raises(ValidationError, match="content_digest"):
        validate_action_spec(tampered)


def test_supervisor_lease_is_scoped_per_authoritative_world_and_resource() -> None:
    wal = InMemoryActionWAL()
    supervisor = ActionSupervisor(wal)
    arm = _motion()
    arm_admission = supervisor.admit(arm, arm.expected_snapshot)

    with pytest.raises(ActionLeaseConflictError):
        supervisor.admit(arm, arm.expected_snapshot)

    gripper = _gripper()
    gripper_admission = supervisor.admit(gripper, gripper.expected_snapshot)
    other_world = _motion(_snapshot(world_id="live-world-2"), plan_id="other")
    other_admission = supervisor.admit(other_world, other_world.expected_snapshot)

    assert supervisor.has_lease("live-world", "arm")
    assert supervisor.has_lease("live-world", "gripper")
    assert supervisor.has_lease("live-world-2", "arm")
    supervisor.release(arm_admission)
    supervisor.release(gripper_admission)
    supervisor.release(other_admission)


def test_admission_checks_revisions_config_joint_order_and_start_tolerance() -> None:
    plan = _motion()
    supervisor = ActionSupervisor(InMemoryActionWAL())
    within_tolerance = _snapshot(joints=(0.01, 0.09))
    admitted = supervisor.admit(plan, within_tolerance)
    supervisor.release(admitted)

    with pytest.raises(AdmissionRejectedError, match="start joints"):
        supervisor.admit(plan, _snapshot(joints=(0.03, 0.1)))
    with pytest.raises(AdmissionRejectedError, match="scene_revision"):
        supervisor.admit(plan, _snapshot(scene_revision=99))
    with pytest.raises(AdmissionRejectedError, match="config"):
        supervisor.admit(
            plan,
            _snapshot(config_revision=5, config_digest=_digest("d")),
        )


def test_motion_safety_checks_cannot_be_not_applicable() -> None:
    plan = _motion()
    checks = {
        "exact_joint_path_interface": FeasibilityStatus.PASS,
        "controller_admissible": FeasibilityStatus.PASS,
        "resource_binding": FeasibilityStatus.PASS,
        "joint_order": FeasibilityStatus.PASS,
        "joint_limits": FeasibilityStatus.NOT_APPLICABLE,
        "robot_model": FeasibilityStatus.PASS,
        "collision": FeasibilityStatus.NOT_APPLICABLE,
    }
    certificate = build_feasibility_certificate(
        spec=plan,
        snapshot=plan.expected_snapshot,
        checker_id="unsafe-na-checker",
        checks=checks,
    )

    with pytest.raises(AdmissionRejectedError, match="collision, joint_limits"):
        ActionSupervisor(InMemoryActionWAL()).admit(
            plan,
            plan.expected_snapshot,
            feasibility_certificate=certificate,
        )


def test_two_processes_cannot_write_the_same_physical_resource(tmp_path: Path) -> None:
    plan = _motion()
    lock_root = tmp_path / "authority-locks"
    owner = ActionSupervisor(
        InMemoryActionWAL(), interprocess_lock_root=lock_root
    )
    admitted = owner.admit(plan, plan.expected_snapshot)
    context = get_context("spawn")
    result_queue = context.Queue()
    contender = context.Process(
        target=_attempt_process_lease,
        args=(plan.model_dump(mode="python"), str(lock_root), result_queue),
    )
    contender.start()
    contender.join(timeout=10)
    try:
        assert contender.exitcode == 0
        assert result_queue.get(timeout=2) == "ActionLeaseConflictError"
    finally:
        owner.release(admitted)


def test_authoritative_admission_rejects_shadow_specs() -> None:
    shadow_snapshot = _snapshot(world_id="shadow-a", world_kind=WorldKind.SHADOW)
    shadow_plan = _motion(shadow_snapshot)
    with pytest.raises(AdmissionRejectedError, match="shadow"):
        ActionSupervisor(InMemoryActionWAL()).admit(shadow_plan, shadow_snapshot)


def test_runner_writes_attempt_before_exact_primitive_and_owns_receipts() -> None:
    plan = _motion(subsample=2)
    wal = InMemoryActionWAL()
    supervisor = ActionSupervisor(wal)
    backend = FakeBackend(plan.expected_snapshot, wal=wal)
    admitted = supervisor.admit(plan, plan.expected_snapshot, action_id="action-1")

    receipt = SealedActionRunner(supervisor, backend).run(admitted)

    records = wal.records()
    assert [type(item) for item in records] == [
        ActionAttempt,
        PrimitiveReceipt,
        ExecutionReceipt,
    ]
    assert wal.flush_count == 3
    assert receipt.runtime_status is ExecutionStatus.COMPLETED
    assert receipt.execution_context == "authoritative"
    assert receipt.receipt_authority == "runtime"
    assert receipt.spec_digest == plan.content_digest
    primitive, kwargs = backend.calls[0]
    assert primitive == "execute_joint_path"
    assert kwargs["positions_rad"] == plan.motion.positions_rad
    assert kwargs["joint_names"] == plan.motion.joint_names
    assert kwargs["subsample"] == 2
    assert not supervisor.has_lease("live-world", "arm")


@pytest.mark.parametrize("replacement", ["path", "subsample"])
def test_runner_rejects_valid_but_nonadmitted_plan_replacement(replacement: str) -> None:
    plan = _motion()
    alternative = (
        _motion(endpoint=(0.8, 0.8)) if replacement == "path" else _motion(subsample=2)
    )
    wal = InMemoryActionWAL()
    supervisor = ActionSupervisor(wal)
    backend = FakeBackend(plan.expected_snapshot)
    admitted = supervisor.admit(plan, plan.expected_snapshot)

    with pytest.raises(SealedActionMismatchError, match="exact admitted"):
        SealedActionRunner(supervisor, backend).run(
            admitted, supplied_spec=alternative
        )

    assert backend.calls == []
    assert wal.records() == ()


def test_runner_rechecks_config_after_admission_before_wal_or_primitive() -> None:
    plan = _motion()
    wal = InMemoryActionWAL()
    supervisor = ActionSupervisor(wal)
    backend = FakeBackend(plan.expected_snapshot)
    admitted = supervisor.admit(plan, plan.expected_snapshot)
    backend.snapshots[(plan.world_id, plan.resource_id)] = _snapshot(
        config_revision=5, config_digest=_digest("d")
    )

    with pytest.raises(AdmissionRejectedError, match="config"):
        SealedActionRunner(supervisor, backend).run(admitted)

    assert backend.calls == []
    assert wal.records() == ()


def test_runner_rejects_backend_that_would_reinterpret_plan_through_pose_or_ik() -> None:
    plan = _motion()
    wal = InMemoryActionWAL()
    supervisor = ActionSupervisor(wal)
    backend = FakeBackend(
        plan.expected_snapshot,
        motion_interface=BackendMotionInterface.POSE_REINTERPRETATION,
    )
    admitted = supervisor.admit(plan, plan.expected_snapshot)

    with pytest.raises(SealedActionMismatchError, match="reinterpret"):
        SealedActionRunner(supervisor, backend).run(admitted)

    assert backend.calls == []
    assert wal.records() == ()


@pytest.mark.parametrize(
    ("spec", "primitive"),
    [(_gripper(), "set_gripper"), (_wait(), "wait")],
)
def test_gripper_and_wait_are_independently_sealed_effects(
    spec: GripperCommand | WaitSpec,
    primitive: str,
) -> None:
    wal = InMemoryActionWAL()
    supervisor = ActionSupervisor(wal)
    backend = FakeBackend(spec.expected_snapshot, wal=wal)
    admitted = supervisor.admit(spec, spec.expected_snapshot)

    receipt = SealedActionRunner(supervisor, backend).run(admitted)

    assert receipt.runtime_status is ExecutionStatus.COMPLETED
    assert backend.calls[0][0] == primitive
    assert isinstance(wal.records()[0], ActionAttempt)


def test_nonconvergence_and_backend_exception_are_not_reported_completed() -> None:
    plan = _motion()

    wal = InMemoryActionWAL()
    supervisor = ActionSupervisor(wal)
    backend = FakeBackend(plan.expected_snapshot, result=BackendCallResult(converged=False))
    partial = SealedActionRunner(supervisor, backend).run(
        supervisor.admit(plan, plan.expected_snapshot)
    )
    assert partial.runtime_status is ExecutionStatus.PARTIAL
    assert partial.abort_reason == "primitive_not_converged"

    wal2 = InMemoryActionWAL()
    supervisor2 = ActionSupervisor(wal2)
    backend2 = FakeBackend(plan.expected_snapshot, failure=RuntimeError("controller fault"))
    unknown = SealedActionRunner(supervisor2, backend2).run(
        supervisor2.admit(plan, plan.expected_snapshot)
    )
    assert unknown.runtime_status is ExecutionStatus.INDETERMINATE_AFTER_CRASH
    primitive = cast_primitive(wal2.records()[1])
    assert primitive.status is PrimitiveStatus.RAISED
    assert primitive.error_type == "RuntimeError"


def test_watchdog_timeout_stops_controller_and_waits_for_quiescence() -> None:
    plan = _motion(timeout_s=0.05)
    backend = StoppableHangingBackend(plan.expected_snapshot)
    supervisor = ActionSupervisor(InMemoryActionWAL())

    receipt = SealedActionRunner(supervisor, backend).run(
        supervisor.admit(plan, plan.expected_snapshot)
    )

    assert receipt.runtime_status is ExecutionStatus.INDETERMINATE_AFTER_TIMEOUT
    assert backend.stop_requested.is_set()
    assert receipt.terminal_telemetry["controller_stop_confirmed"] is True
    assert receipt.terminal_telemetry["backend_call_terminated"] is True
    assert (plan.world_id, plan.resource_id) in supervisor.blocked_resources


def cast_primitive(record: object) -> PrimitiveReceipt:
    assert isinstance(record, PrimitiveReceipt)
    return record


def test_backend_cannot_return_an_authoritative_receipt() -> None:
    plan = _motion()
    fake_receipt = ExecutionReceipt(
        action_id="forged",
        spec_type="motion_plan",
        spec_id=plan.spec_id,
        spec_digest=plan.content_digest,
        world_id=plan.world_id,
        resource_id=plan.resource_id,
        runtime_status="unknown",
        possibly_affected_revisions=plan.possibly_affected_revisions,
        started_at=CAPTURED_AT,
        finished_at=CAPTURED_AT,
    )
    wal = InMemoryActionWAL()
    supervisor = ActionSupervisor(wal)
    backend = FakeBackend(plan.expected_snapshot, result=fake_receipt)
    receipt = SealedActionRunner(supervisor, backend).run(
        supervisor.admit(plan, plan.expected_snapshot)
    )

    assert receipt.runtime_status is ExecutionStatus.INDETERMINATE_AFTER_CRASH
    assert receipt.abort_reason == "primitive_exception:TypeError"
    assert receipt.action_id != "forged"


def test_orphan_attempt_reconciles_indeterminate_without_automatic_replay() -> None:
    plan = _motion()
    wal = InMemoryActionWAL()
    supervisor = ActionSupervisor(wal)
    backend = FakeBackend(plan.expected_snapshot, wal=wal, failure=SimulatedProcessCrash())
    admitted = supervisor.admit(plan, plan.expected_snapshot, action_id="crashed-action")

    with pytest.raises(SimulatedProcessCrash):
        SealedActionRunner(supervisor, backend).run(admitted)

    assert len(backend.calls) == 1
    assert [type(item) for item in wal.records()] == [ActionAttempt]
    assert (plan.world_id, plan.resource_id) in supervisor.blocked_resources

    restarted = ActionSupervisor(wal)
    assert (plan.world_id, plan.resource_id) in restarted.blocked_resources
    reconciled = restarted.reconcile_orphans()

    assert len(reconciled) == 1
    assert reconciled[0].runtime_status is ExecutionStatus.INDETERMINATE_AFTER_CRASH
    assert reconciled[0].abort_reason == "orphan_attempt_reconciled_without_replay"
    assert len(backend.calls) == 1
    assert restarted.reconcile_orphans() == ()
    assert (plan.world_id, plan.resource_id) in restarted.blocked_resources
    with pytest.raises(RecoveryBlockedError):
        restarted.admit(plan, plan.expected_snapshot)

    recovery_snapshot = plan.expected_snapshot.model_copy(
        update={
            "robot_revision": plan.expected_snapshot.robot_revision + 1,
            "controller_state": ControllerState.QUIESCENT,
            "captured_at": CAPTURED_AT + timedelta(seconds=1),
        }
    )
    restarted.acknowledge_recovery(
        plan.world_id,
        plan.resource_id,
        current_snapshot=recovery_snapshot,
        evidence_refs=("artifact:controller-quiescence-proof",),
        reason="operator verified the controller stop and refreshed state",
    )
    recovered_plan = _motion(recovery_snapshot, plan_id="post-recovery")
    recovered = restarted.admit(recovered_plan, recovery_snapshot)
    restarted.release(recovered)


def test_jsonl_wal_is_append_only_reloadable_and_conflict_checked(tmp_path: Path) -> None:
    plan = _motion()
    wal = JsonlActionWAL(tmp_path / "actions.jsonl")
    supervisor = ActionSupervisor(wal)
    receipt = SealedActionRunner(supervisor, FakeBackend(plan.expected_snapshot)).run(
        supervisor.admit(plan, plan.expected_snapshot)
    )
    loaded = JsonlActionWAL(tmp_path / "actions.jsonl")

    assert loaded.records() == wal.records()
    assert loaded.append(receipt) is False
    exposed = loaded.records()[-1]
    assert isinstance(exposed, ExecutionReceipt)
    exposed.terminal_telemetry["caller_mutation"] = True
    assert "caller_mutation" not in loaded.records()[-1].terminal_telemetry
    conflict = receipt.model_copy(update={"terminal_telemetry": {"changed": True}})
    with pytest.raises(WalConflictError):
        loaded.append(conflict)


def test_shadow_rollout_is_separate_and_never_yields_authoritative_receipt() -> None:
    snapshot = _snapshot(world_id="shadow-a", world_kind=WorldKind.SHADOW)
    plan = _motion(snapshot)
    backend = FakeBackend(snapshot)
    receipt = ShadowRolloutRunner(backend).run(plan, candidate_id="candidate-a")

    assert isinstance(receipt, ShadowRolloutReceipt)
    assert not isinstance(receipt, ExecutionReceipt)
    assert receipt.execution_context == "shadow"
    assert receipt.receipt_authority == "shadow_only"
    assert receipt.shadow_world_id == "shadow-a"

    with pytest.raises(ShadowIsolationError, match="authoritative"):
        ShadowRolloutRunner(FakeBackend(_snapshot())).run(
            _motion(), candidate_id="candidate-live"
        )


def test_isolated_shadow_worlds_can_roll_out_in_parallel() -> None:
    snapshot_a = _snapshot(world_id="shadow-a", world_kind=WorldKind.SHADOW)
    snapshot_b = _snapshot(world_id="shadow-b", world_kind=WorldKind.SHADOW)
    plan_a = _motion(snapshot_a, plan_id="plan-a")
    plan_b = _motion(snapshot_b, plan_id="plan-b")
    backend = FakeBackend((snapshot_a, snapshot_b), delay_s=0.08)
    runner = ShadowRolloutRunner(backend)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(runner.run, plan_a, candidate_id="candidate-a")
        future_b = pool.submit(runner.run, plan_b, candidate_id="candidate-b")
        receipts = (future_a.result(), future_b.result())

    assert all(item.status.value == "completed" for item in receipts)
    assert backend.max_active_calls == 2
