from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from robomex.data import EpisodeDataPlane
from robomex.data.embodied_state import AttachmentStatus
from robomex.orchestration.actors import (
    ActorLifecycle,
    ActorProfile,
    ActorRegistry,
    InMemoryAgentProvider,
)
from robomex.orchestration.arena import (
    ArenaCandidateSpec,
    ArenaConsumptionLedger,
    ArenaContext,
    CheckStatus,
    RiskInputs,
    RiskPolicy,
    RiskReport,
    RuntimeArenaContextGuard,
    RuntimeMotionPromotionAuthority,
    ShadowBackendRegistry,
    SwarmArena,
)
from robomex.runtime.action_protocol import (
    BackendMotionInterface,
    ControllerState,
    ExecutionPolicy,
    ExecutionStatus,
    FeasibilityStatus,
    JointPath,
    MotionPlan,
    WorldKind,
)
from robomex.runtime.authority import (
    ActionSupervisor,
    AdmissionRejectedError,
    InMemoryActionWAL,
    SealedActionRunner,
)
from robomex.runtime.capx_action_backend import (
    CallbackAdmissionSnapshotProvider,
    CapXBackendConfigurationError,
    CapXFeasibilityChecker,
    CapXSealedActionBackend,
    FeasibilityProbeOutcome,
    JointState,
    LiberoControlPort,
    ResourceRole,
    SnapshotRevisions,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


CAPTURED_AT = datetime(2026, 7, 22, 8, 0, tzinfo=timezone.utc)  # noqa: UP017
JOINT_NAMES = ("panda_joint1", "panda_joint2")


class FakeLiberoEnv:
    def __init__(self) -> None:
        self._control_freq = 20.0
        self.joints = np.array([0.0, 0.1], dtype=np.float64)
        self._current_joints = self.joints.copy()
        self.gripper_fraction = 1.0
        self.blocking_targets: list[tuple[float, ...]] = []
        self.tracking_references: list[tuple[float, ...]] = []
        self.gripper_targets: list[float] = []
        self.hold_steps = 0
        self.force_timeout = False
        self.goto_pose_calls = 0
        self.solve_ik_calls = 0

    def _current_arm_joint_positions(self) -> np.ndarray:
        return self.joints.copy()

    def move_to_joints_blocking(
        self,
        joints: np.ndarray,
        *,
        tolerance: float,
        max_steps: int,
        settle_steps: int,
    ) -> dict[str, object]:
        del tolerance, max_steps, settle_steps
        target = np.asarray(joints, dtype=np.float64).copy()
        self.blocking_targets.append(tuple(float(value) for value in target))
        if self.force_timeout:
            return {
                "converged": False,
                "timed_out": True,
                "stalled": False,
                "current": self.joints.copy(),
            }
        self.joints = target
        return {
            "converged": True,
            "timed_out": False,
            "stalled": False,
            "current": self.joints.copy(),
        }

    def _tracking_step(self, reference: np.ndarray) -> None:
        self.joints = np.asarray(reference, dtype=np.float64).copy()
        self.tracking_references.append(tuple(float(value) for value in self.joints))

    def _set_gripper(self, fraction: float) -> None:
        self.gripper_fraction = float(fraction)
        self.gripper_targets.append(self.gripper_fraction)

    def _step_once(self) -> None:
        self.hold_steps += 1

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "robot_joint_pos": np.concatenate(
                [self.joints, np.array([self.gripper_fraction])]
            )
        }

    def goto_pose(self, *_args: object, **_kwargs: object) -> None:
        self.goto_pose_calls += 1
        raise AssertionError("sealed runtime must never call goto_pose")

    def solve_ik(self, *_args: object, **_kwargs: object) -> None:
        self.solve_ik_calls += 1
        raise AssertionError("sealed runtime must never call solve_ik")


class Harness:
    def __init__(self) -> None:
        self.env = FakeLiberoEnv()
        self.revisions = SnapshotRevisions(robot=3, scene=5, attachment=7, config=11)
        self.config_digest = _digest("a")
        self.collision_world_digest = _digest("b")
        self.provider = CallbackAdmissionSnapshotProvider(
            joint_state=lambda _world, _resource: JointState(
                JOINT_NAMES,
                tuple(float(value) for value in self.env.joints),
            ),
            revisions=lambda _world, _resource: self.revisions,
            config_digest=lambda _world, _resource: self.config_digest,
            collision_world_digest=lambda _world, _resource: self.collision_world_digest,
            world_kind=lambda _world: WorldKind.AUTHORITATIVE,
            attachment_status=lambda _world, _resource: AttachmentStatus.VERIFIED_HELD,
            controller_state=lambda _world, _resource: ControllerState.READY,
            clock=lambda: CAPTURED_AT,
        )
        self.control = LiberoControlPort(
            self.env,
            max_gripper_width_m=0.08,
            open_settle_steps=4,
            close_settle_steps=6,
            final_settle_steps=1,
        )
        self.backend = CapXSealedActionBackend(
            backend_id="capx-libero-exact-v2",
            control=self.control,
            snapshots=self.provider,
            resource_roles={
                "arm": ResourceRole.ARM,
                "gripper": ResourceRole.GRIPPER,
                "controller": ResourceRole.CONTROLLER,
            },
            allowed_world_ids=("live-world",),
        )

    def motion(self, *, subsample: int = 1, timeout_s: float = 5.0) -> MotionPlan:
        snapshot = self.provider.snapshot("live-world", "arm")
        return MotionPlan(
            plan_id="sealed-path",
            plan_kind="bounded_alignment",
            tcp_frame_id="panda_hand",
            planner_backend="curobo",
            robot_model_digest=_digest("c"),
            expected_snapshot=snapshot,
            max_start_deviation_rad=0.02,
            motion=JointPath(
                joint_names=JOINT_NAMES,
                positions_rad=(
                    (0.0, 0.1),
                    (0.1, 0.2),
                    (0.2, 0.3),
                    (0.3, 0.4),
                ),
                execution_policy=ExecutionPolicy(
                    subsample=subsample,
                    timeout_s=timeout_s,
                ),
            ),
            possibly_affected_revisions=("robot.arm", "scene", "attachment"),
        )


def test_sealed_backend_executes_exact_subsampled_joint_path_without_ik() -> None:
    harness = Harness()
    plan = harness.motion(subsample=2)
    supervisor = ActionSupervisor(InMemoryActionWAL())
    admitted = supervisor.admit(plan, plan.expected_snapshot)

    receipt = SealedActionRunner(supervisor, harness.backend).run(admitted)

    assert receipt.runtime_status is ExecutionStatus.COMPLETED
    assert harness.backend.descriptor.motion_interface is BackendMotionInterface.EXACT_JOINT_PATH
    assert harness.env.blocking_targets == [
        plan.motion.positions_rad[0],
        plan.motion.positions_rad[2],
        plan.motion.positions_rad[3],
    ]
    assert harness.env.goto_pose_calls == 0
    assert harness.env.solve_ik_calls == 0


def test_pose_only_environment_is_rejected_instead_of_reinterpreting_path() -> None:
    class PoseOnlyEnv:
        def goto_pose(self, *_args: object) -> None:
            raise AssertionError

        def solve_ik(self, *_args: object) -> None:
            raise AssertionError

    with pytest.raises(CapXBackendConfigurationError, match="move_to_joints_blocking"):
        LiberoControlPort(PoseOnlyEnv(), max_gripper_width_m=0.08)


def test_execution_gate_rejects_revision_drift_before_controller_call() -> None:
    harness = Harness()
    plan = harness.motion()
    supervisor = ActionSupervisor(InMemoryActionWAL())
    admitted = supervisor.admit(plan, plan.expected_snapshot)
    harness.revisions = SnapshotRevisions(
        robot=3,
        scene=6,
        attachment=7,
        config=11,
    )

    with pytest.raises(AdmissionRejectedError, match="scene_revision"):
        SealedActionRunner(supervisor, harness.backend).run(admitted)

    assert harness.env.blocking_targets == []
    assert harness.env.tracking_references == []


@pytest.mark.parametrize(
    ("changed_guard", "message"),
    [
        ("config_digest", "config"),
        ("collision_world_digest", "collision_world_digest"),
    ],
)
def test_execution_gate_rejects_semantic_digest_drift(
    changed_guard: str,
    message: str,
) -> None:
    harness = Harness()
    plan = harness.motion()
    supervisor = ActionSupervisor(InMemoryActionWAL())
    admitted = supervisor.admit(plan, plan.expected_snapshot)
    setattr(harness, changed_guard, _digest("d"))

    with pytest.raises(AdmissionRejectedError, match=message):
        SealedActionRunner(supervisor, harness.backend).run(admitted)

    assert harness.env.blocking_targets == []


def test_cooperative_control_callback_interrupts_before_later_waypoints() -> None:
    harness = Harness()
    plan = harness.motion()
    callback_samples: list[dict[str, object]] = []

    def callback(sample: dict[str, object]) -> bool:
        callback_samples.append(sample)
        return False

    result = harness.backend.execute_joint_path_cooperative(
        progress_callback=callback,
        world_id="live-world",
        resource_id="arm",
        joint_names=plan.motion.joint_names,
        positions_rad=plan.motion.positions_rad,
        mode="blocking_waypoint",
        subsample=1,
        timeout_s=5.0,
    )

    assert result.interrupted is True
    assert result.timed_out is False
    assert result.converged is False
    assert len(callback_samples) == 1
    assert callback_samples[0]["cooperative_granularity"] == "control"
    assert len(harness.env.tracking_references) == 1
    assert harness.env.blocking_targets == []
    assert np.array_equal(harness.env._current_joints, harness.env.joints)


def test_controller_timeout_is_explicit_and_runner_marks_indeterminate() -> None:
    harness = Harness()
    plan = harness.motion()
    harness.env.force_timeout = True
    supervisor = ActionSupervisor(InMemoryActionWAL())

    receipt = SealedActionRunner(supervisor, harness.backend).run(
        supervisor.admit(plan, plan.expected_snapshot)
    )

    assert receipt.runtime_status is ExecutionStatus.INDETERMINATE_AFTER_TIMEOUT
    assert receipt.abort_reason == "primitive_timeout"
    assert receipt.terminal_telemetry["timeout_semantics"].startswith("indeterminate")


def test_gripper_and_wait_use_independent_controller_surfaces() -> None:
    harness = Harness()

    gripper = harness.backend.set_gripper(
        world_id="live-world",
        resource_id="gripper",
        mode="close",
        target_width_m=0.0,
        max_effort_n=None,
        timeout_s=1.0,
    )
    steps_after_gripper = harness.env.hold_steps
    wait = harness.backend.wait(
        world_id="live-world",
        resource_id="controller",
        duration_s=None,
        control_steps=3,
        hold_command="hold_current",
        timeout_s=1.0,
    )

    assert gripper.converged is True
    assert wait.converged is True
    assert harness.env.gripper_targets == [0.0]
    assert steps_after_gripper == 6
    assert harness.env.hold_steps == 9
    assert harness.env.blocking_targets == []


def test_feasibility_checker_is_pass_with_pinned_evidence_and_unknown_without_it() -> None:
    harness = Harness()
    plan = harness.motion()
    limits = dict.fromkeys(JOINT_NAMES, (-2.0, 2.0))
    checker = CapXFeasibilityChecker(
        checker_id="capx-curobo-checker-v2",
        backend=harness.backend,
        robot_model_digest=plan.robot_model_digest,
        joint_limits_rad=limits,
        collision_probe=lambda _spec, _snapshot: FeasibilityProbeOutcome(
            FeasibilityStatus.PASS,
            ("artifact://collision-report",),
        ),
    )

    passed = checker.certify(plan, plan.expected_snapshot)
    missing_collision = CapXFeasibilityChecker(
        checker_id="uncertified-checker-v2",
        backend=harness.backend,
        robot_model_digest=plan.robot_model_digest,
        joint_limits_rad=limits,
        collision_probe=None,
    ).certify(plan, plan.expected_snapshot)

    assert passed.overall_status is FeasibilityStatus.PASS
    assert passed.checks["collision"] is FeasibilityStatus.PASS
    assert passed.evidence_refs == ("artifact://collision-report",)
    assert missing_collision.overall_status is FeasibilityStatus.UNKNOWN


def test_native_capx_checker_can_promote_exact_motion_without_an_ik_check(
    tmp_path,
) -> None:
    harness = Harness()
    plan = harness.motion()
    plane = EpisodeDataPlane(tmp_path / "episode", episode_id="episode-capx")
    plane.open_workflow("workflow")
    snapshot_record = plane.publish(
        workflow_id="workflow",
        activation_id="arena-input",
        attempt=1,
        port="snapshot",
        schema=plan.expected_snapshot.schema_version,
        payload=plan.expected_snapshot.model_dump(mode="json"),
    )
    plan_record = plane.publish(
        workflow_id="workflow",
        activation_id="arena-input",
        attempt=1,
        port="motion-plan",
        schema=plan.schema_version,
        payload=plan.model_dump(mode="json"),
        lineage=(snapshot_record.ref,),
    )
    provider = InMemoryAgentProvider(
        lambda *_: {
            "plan_ref": plan_record.ref,
            "snapshot_ref": snapshot_record.ref,
            "frame": "world",
            "expected_effect": "bounded exact-path alignment",
            "utility": 1.0,
            # A proposal claim is deliberately irrelevant to the runtime gate.
            "metadata": {"ik": "not_run_for_exact_joint_path"},
        }
    )
    registry = ActorRegistry(
        {"memory": provider},
        namespace_root="episode-capx",
        workspace_root=tmp_path / "actors",
    )
    checker = CapXFeasibilityChecker(
        checker_id="native-capx-checker",
        backend=harness.backend,
        robot_model_digest=plan.robot_model_digest,
        joint_limits_rad=dict.fromkeys(JOINT_NAMES, (-2.0, 2.0)),
        collision_probe=lambda *_: FeasibilityStatus.PASS,
    )
    arena = SwarmArena(
        registry,
        promotion_authority=RuntimeMotionPromotionAuthority(
            episode_id="episode-capx",
            artifacts=plane,
            snapshot_providers={
                (plan.world_id, plan.resource_id): harness.backend.snapshot
            },
            feasibility_checkers={(plan.world_id, plan.resource_id): checker},
        ),
        context_guard=RuntimeArenaContextGuard(
            episode_id="episode-capx",
            current_revision=lambda _context: ("capx-motion", 1),
        ),
        consumption_ledger=ArenaConsumptionLedger(
            tmp_path / "episode" / "arena_consumption.v1.jsonl"
        ),
        shadow_backends=ShadowBackendRegistry(),
        gates=(),
    )
    context = ArenaContext(
        arena_run_id="native-capx-promotion",
        episode_id="episode-capx",
        workflow_id="workflow",
        graph_id="capx-motion",
        graph_revision=1,
        slot_id="motion-candidates",
        snapshot_ref=snapshot_record.ref,
        expected_frame="world",
        world_id=plan.world_id,
        resource_id=plan.resource_id,
        robot_model_digest=plan.robot_model_digest,
        config_digest=plan.expected_snapshot.config_digest,
        candidate_budget_id="native-capx-budget",
        candidate_budget_limit=1,
    )
    risk = RiskReport.assess(
        RiskInputs(
            grounding_confidence=0.99,
            target_margin_m=0.05,
            ik_status=CheckStatus.PASS,
            collision_status=CheckStatus.PASS,
            clearance_m=0.05,
            held_pose_uncertainty_m=0.001,
        ),
        RiskPolicy(max_candidates=1),
    )

    result = arena.run(
        context=context,
        risk_report=risk,
        candidates=(
            ArenaCandidateSpec(
                candidate_id="capx-exact-path",
                strategy="native-capx",
                profile=ActorProfile(
                    profile_id="capx-proposal",
                    provider_id="memory",
                    lifecycle=ActorLifecycle.EPHEMERAL,
                ),
                objective="propose the exact Cap-X joint path",
            ),
        ),
        candidate_budget_remaining=1,
    )

    assert result.selected_candidate_id == "capx-exact-path"
    assert result.promotion_receipt is not None
    checks = result.promotion_receipt.feasibility_certificate.checks
    assert "ik" not in checks
    assert checks["exact_joint_path_interface"] is FeasibilityStatus.PASS
    assert result.promotion_receipt.authoritative_effect_committed is False


def test_code_execution_env_resolution_requires_explicit_api_identity() -> None:
    low_level = FakeLiberoEnv()

    class Api:
        _env = low_level

    class CodeEnv:
        _apis = {"FrankaLiberoApiReduced": Api()}

    port = LiberoControlPort.from_code_execution_env(
        CodeEnv(),
        api_name="FrankaLiberoApiReduced",
        max_gripper_width_m=0.08,
    )
    assert port.current_joint_positions() == (0.0, 0.1)

    with pytest.raises(CapXBackendConfigurationError, match="no configured API"):
        LiberoControlPort.from_code_execution_env(
            CodeEnv(),
            api_name="wrong-api",
            max_gripper_width_m=0.08,
        )
