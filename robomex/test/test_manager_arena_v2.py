"""Deterministic offline tests for the v2 bounded Manager and Swarm Arena."""

from __future__ import annotations

from datetime import UTC, datetime, timezone

import pytest

from robomex.data import ResolvedArtifact, ResolvedArtifactRef, compute_content_digest
from robomex.elastic.graph_spec import EffectScope
from robomex.orchestration.actors import (
    ActorLifecycle,
    ActorProfile,
    ActorRegistry,
    InMemoryAgentProvider,
    IsolationPolicy,
)
from robomex.orchestration.arena import (
    ArenaCandidateSpec,
    ArenaConsumptionLedger,
    ArenaContext,
    ArenaPolicy,
    ArenaPolicyError,
    ArenaResult,
    ArenaStaleContextError,
    BasicHypothesisGate,
    CandidateStatus,
    CheckStatus,
    MetadataStatusGate,
    RegisteredShadowBackend,
    RiskInputs,
    RiskLevel,
    RiskPolicy,
    RiskReport,
    RuntimeArenaContextGuard,
    RuntimeMotionPromotionAuthority,
    ShadowBackendRegistry,
    SwarmArena,
)
from robomex.orchestration.manager import (
    ManagerAction,
    ManagerDecision,
    ManagerLimits,
    ManagerSessionStatus,
    ManagerSignal,
    ManagerSnapshot,
    ManagerStateError,
    ScriptedManagerInvoker,
    SwarmManagerSession,
)
from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    BackendCallResult,
    BackendDescriptor,
    FeasibilityStatus,
    JointPath,
    MotionPlan,
    WorldKind,
)
from robomex.runtime.authority import build_feasibility_certificate
from robomex.runtime.events import RosterOperation

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64
_SNAPSHOT = AdmissionSnapshot(
    world_id="robot-world",
    resource_id="arm",
    robot_revision=2,
    scene_revision=4,
    attachment_revision=1,
    config_revision=3,
    joint_names=("j1", "j2"),
    joint_positions_rad=(0.0, 0.0),
    config_digest=_DIGEST_A,
    collision_world_digest=_DIGEST_B,
    captured_at=datetime(2026, 1, 1, tzinfo=UTC),
)
_SNAPSHOT_PAYLOAD = _SNAPSHOT.model_dump(mode="json")
_SNAPSHOT_REF = ResolvedArtifactRef(
    "art:episode-1:snapshot",
    compute_content_digest(_SNAPSHOT.schema_version, _SNAPSHOT_PAYLOAD),
)


class _Artifacts:
    def __init__(self) -> None:
        self.values: dict[str, ResolvedArtifact] = {
            _SNAPSHOT_REF.artifact_id: ResolvedArtifact(
                ref=_SNAPSHOT_REF,
                schema=_SNAPSHOT.schema_version,
                payload=_SNAPSHOT_PAYLOAD,
                record={},
            )
        }

    def add_plan(self, plan: MotionPlan) -> ResolvedArtifactRef:
        payload = plan.model_dump(mode="json")
        ref = ResolvedArtifactRef(
            f"art:episode-1:{plan.plan_id}",
            compute_content_digest(plan.schema_version, payload),
        )
        self.values[ref.artifact_id] = ResolvedArtifact(
            ref=ref,
            schema=plan.schema_version,
            payload=payload,
            record={},
        )
        return ref

    def resolve(self, ref: ResolvedArtifactRef | object) -> ResolvedArtifact:
        parsed = ResolvedArtifactRef.from_any(ref)
        artifact = self.values[parsed.artifact_id]
        if artifact.ref != parsed:
            raise ValueError("artifact digest mismatch")
        return artifact


class _Checker:
    def __init__(self, statuses: dict[str, FeasibilityStatus]) -> None:
        self.statuses = statuses

    def certify(self, spec: object, snapshot: AdmissionSnapshot):
        status = self.statuses[spec.content_digest]
        return build_feasibility_certificate(
            spec=spec,
            snapshot=snapshot,
            checker_id="trusted-test-checker",
            checks={
                "kinematic_feasibility": status,
                "collision": status,
                "joint_limits": status,
            },
        )


class _ShadowBackend:
    def __init__(self, backend_id: str, *, reports_shadow: bool = True) -> None:
        self.descriptor = BackendDescriptor(backend_id=backend_id)
        self.reports_shadow = reports_shadow

    def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot:
        return AdmissionSnapshot.model_validate(
            {
                **_SNAPSHOT.model_dump(mode="python"),
                "world_id": world_id,
                "world_kind": (
                    WorldKind.SHADOW
                    if self.reports_shadow
                    else WorldKind.AUTHORITATIVE
                ),
                "resource_id": resource_id,
            }
        )

    def execute_joint_path(self, **_kwargs: object) -> BackendCallResult:
        return BackendCallResult(converged=True)

    def set_gripper(self, **_kwargs: object) -> BackendCallResult:
        return BackendCallResult(converged=True)

    def wait(self, **_kwargs: object) -> BackendCallResult:
        return BackendCallResult(converged=True)


def _snapshot(sequence: int, *, graph_revision: int = 1) -> ManagerSnapshot:
    return ManagerSnapshot(
        snapshot_id=f"snapshot-{sequence}",
        episode_id="episode-1",
        workflow_id="workflow-1",
        graph_id="bowl-place",
        graph_revision=graph_revision,
        state_revision=sequence,
        frontier_id="proposal-slot",
        triggering_event={"kind": "risk_expansion"},
        compact_state={"attachment": "verified_held"},
        candidate_cards=({"candidate_id": "c1", "risk": 0.4},),
        artifact_refs=("artifact:bowl-track-v3",),
        catalog_refs=("fragment:motion-review-v1",),
    )


def _session(limits: ManagerLimits | None = None) -> SwarmManagerSession:
    return SwarmManagerSession(
        session_id="manager-1",
        episode_id="episode-1",
        workflow_id="workflow-1",
        intent_id="intent-place-bowl",
        limits=limits or ManagerLimits(),
    )


def _decision(
    action: ManagerAction = ManagerAction.AUTHOR_SCAFFOLD,
    *,
    tokens: int = 20,
    candidates: int = 0,
) -> ManagerDecision:
    return ManagerDecision(
        action=action,
        summary="bounded scripted decision",
        payload={"fragment_ref": "fragment:fixed-v2"},
        tokens_used=tokens,
        candidate_delta=candidates,
    )


def test_manager_session_is_versioned_serializable_record_without_chat_history() -> None:
    invoker = ScriptedManagerInvoker((_decision(tokens=24),))
    result = _session().author(_snapshot(1), invoker)

    assert result.invoked is True
    assert result.session.record_revision == 2
    assert result.session.usage.initial_calls == 1
    assert result.session.usage.tokens == 24
    assert len(result.session.receipts) == 1
    assert result.session.receipts[0].snapshot_digest == _snapshot(1).digest()

    restored = SwarmManagerSession.model_validate_json(result.session.model_dump_json())
    assert restored == result.session
    request_payload = invoker.requests[0].model_dump()
    assert "messages" not in request_payload
    assert "chat" not in request_payload
    assert request_payload["snapshot"]["snapshot_id"] == "snapshot-1"
    assert request_payload["token_limit"] == result.session.limits.max_tokens_per_call


@pytest.mark.parametrize(
    "signal",
    [
        ManagerSignal.NODE_SUCCESS,
        ManagerSignal.FRAME_TICK,
        ManagerSignal.NORMAL_CORRECTION,
        ManagerSignal.ACTOR_LIFECYCLE,
    ],
)
def test_routine_signals_never_wake_manager(signal: ManagerSignal) -> None:
    invoker = ScriptedManagerInvoker((_decision(),))
    session = _session()

    result = session.wake(signal, _snapshot(1), invoker)

    assert result.invoked is False
    assert result.session is session
    assert invoker.requests == []


@pytest.mark.parametrize(
    "signal",
    [
        ManagerSignal.RISK_EXPANSION,
        ManagerSignal.ALL_CANDIDATES_REJECTED,
        ManagerSignal.DISAGREEMENT,
        ManagerSignal.RECOVERY,
        ManagerSignal.LOOP_EXHAUSTED,
        ManagerSignal.PATCH_REJECTED,
    ],
)
def test_only_declared_exception_signals_reactivate_manager(signal: ManagerSignal) -> None:
    invoker = ScriptedManagerInvoker(
        (_decision(ManagerAction.REPAIR_FRONTIER, tokens=3),)
    )
    result = _session().wake(signal, _snapshot(1), invoker)

    assert result.invoked is True
    assert result.session.usage.reactivations == 1
    assert invoker.requests[0].signal is signal


def test_manager_uses_fresh_snapshots_and_rejects_reuse_or_revision_regression() -> None:
    invoker = ScriptedManagerInvoker(
        (
            _decision(tokens=1),
            _decision(ManagerAction.EXPAND_ROSTER, tokens=1),
        )
    )
    first = _session().author(_snapshot(1, graph_revision=2), invoker).session

    with pytest.raises(ManagerStateError, match="fresh snapshot_id"):
        first.wake(ManagerSignal.RISK_EXPANSION, _snapshot(1, graph_revision=2), invoker)
    with pytest.raises(ManagerStateError, match="moved backwards"):
        first.wake(ManagerSignal.RISK_EXPANSION, _snapshot(2, graph_revision=1), invoker)

    second = first.wake(
        ManagerSignal.RISK_EXPANSION, _snapshot(2, graph_revision=2), invoker
    )
    assert second.invoked is True
    assert [request.snapshot.snapshot_id for request in invoker.requests] == [
        "snapshot-1",
        "snapshot-2",
    ]


def test_manager_enforces_total_call_token_and_candidate_budgets() -> None:
    limits = ManagerLimits(
        max_initial_calls=1,
        max_reactivations=1,
        max_tokens=10,
        max_tokens_per_call=6,
        max_candidates=2,
    )
    invoker = ScriptedManagerInvoker(
        (
            _decision(tokens=4, candidates=1),
            _decision(ManagerAction.EXPAND_ROSTER, tokens=6, candidates=1),
        )
    )
    initial = _session(limits).author(_snapshot(1), invoker).session
    final = initial.wake(
        ManagerSignal.RISK_EXPANSION, _snapshot(2), invoker
    ).session

    assert final.usage.initial_calls == 1
    assert final.usage.reactivations == 1
    assert final.usage.tokens == 10
    assert final.usage.candidates == 2
    assert final.status is ManagerSessionStatus.EXHAUSTED

    denied = final.wake(ManagerSignal.RECOVERY, _snapshot(3), invoker)
    assert denied.invoked is False
    assert len(invoker.requests) == 2


def test_initial_call_limit_does_not_disable_remaining_reactivations() -> None:
    invoker = ScriptedManagerInvoker(
        (
            _decision(tokens=1),
            _decision(ManagerAction.REPAIR_FRONTIER, tokens=1),
        )
    )
    authored = _session().author(_snapshot(1), invoker).session

    extra_author = authored.author(_snapshot(2), invoker)
    assert extra_author.invoked is False
    assert extra_author.session.status is ManagerSessionStatus.ACTIVE

    wake = authored.wake(ManagerSignal.RECOVERY, _snapshot(2), invoker)
    assert wake.invoked is True
    assert wake.session.usage.reactivations == 1


def test_candidate_budget_overrequest_is_a_failed_bounded_receipt() -> None:
    limits = ManagerLimits(max_candidates=1)
    invoker = ScriptedManagerInvoker(
        (_decision(ManagerAction.EXPAND_ROSTER, tokens=2, candidates=2),)
    )

    result = _session(limits).wake(
        ManagerSignal.RISK_EXPANSION, _snapshot(1), invoker
    )

    assert result.invoked is True
    assert result.decision is None
    assert "ManagerBudgetError" in result.reason
    assert result.session.usage.reactivations == 1
    assert result.session.usage.candidates == 0
    assert result.session.receipts[-1].succeeded is False


def test_manager_invoker_failure_is_bounded_and_audited() -> None:
    invoker = ScriptedManagerInvoker((RuntimeError("provider unavailable"),))
    result = _session().author(_snapshot(1), invoker)

    assert result.invoked is True
    assert result.decision is None
    assert result.session.usage.initial_calls == 1
    assert result.session.receipts[-1].error == "RuntimeError: provider unavailable"


def _safe_risk(*, max_candidates: int = 3) -> RiskReport:
    return RiskReport.assess(
        RiskInputs(
            grounding_confidence=0.95,
            target_margin_m=0.05,
            ik_status=CheckStatus.PASS,
            collision_status=CheckStatus.PASS,
            clearance_m=0.03,
            held_pose_uncertainty_m=0.003,
            monitor_observable=True,
        ),
        RiskPolicy(max_candidates=max_candidates),
    )


def _high_risk(*, max_candidates: int = 3) -> RiskReport:
    return RiskReport.assess(
        RiskInputs(
            grounding_confidence=0.55,
            target_margin_m=0.005,
            ik_status=CheckStatus.UNKNOWN,
            collision_status=CheckStatus.UNKNOWN,
            clearance_m=0.004,
            held_pose_uncertainty_m=0.03,
            prior_failures=1,
        ),
        RiskPolicy(max_candidates=max_candidates),
    )


def _candidate(candidate_id: str, *, world_id: str | None = None) -> ArenaCandidateSpec:
    return ArenaCandidateSpec(
        candidate_id=candidate_id,
        strategy=f"strategy-{candidate_id}",
        profile=ActorProfile(
            profile_id=f"profile-{candidate_id}",
            provider_id="memory",
            lifecycle=ActorLifecycle.EPHEMERAL,
            isolation=IsolationPolicy(world_id=world_id),
        ),
        objective=f"propose {candidate_id}",
    )


def _context(
    run_id: str = "arena-001",
    *,
    revision: int = 7,
    budget_id: str = "arena-test-budget",
    budget_limit: int = 10,
) -> ArenaContext:
    return ArenaContext(
        arena_run_id=run_id,
        episode_id="episode-1",
        workflow_id="workflow-1",
        graph_id="place-bowl",
        graph_revision=revision,
        slot_id="motion-proposals",
        snapshot_ref=_SNAPSHOT_REF,
        expected_frame="world",
        world_id="robot-world",
        resource_id="arm",
        robot_model_digest=_DIGEST_C,
        config_digest=_DIGEST_A,
        candidate_budget_id=budget_id,
        candidate_budget_limit=budget_limit,
    )


def _card(
    *,
    utility: float,
    risk: float = 0.2,
    clearance: float = 0.02,
    endpoint: tuple[float, float, float] = (0.1, 0.2, 0.3),
    certificate: str = "pass",
) -> dict[str, object]:
    return {
        "plan_ref": f"plan:{utility}",
        "snapshot_ref": "snapshot:scene-4",
        "frame": "world",
        "expected_effect": "bowl alignment proposal",
        "estimated_risk": risk,
        "utility": utility,
        "clearance_m": clearance,
        "path_length": 0.4,
        "terminal_position_m": endpoint,
        "metadata": {"feasibility": certificate},
    }


def _arena(
    outputs: list[dict[str, object] | BaseException],
    *,
    trusted_statuses: list[FeasibilityStatus] | None = None,
    current_snapshot: AdmissionSnapshot = _SNAPSHOT,
    corrupt_plan_ref_at: frozenset[int] = frozenset(),
    consumption_ledger: ArenaConsumptionLedger | None = None,
    context_guard: RuntimeArenaContextGuard | None = None,
    shadow_backends: ShadowBackendRegistry | None = None,
) -> tuple[SwarmArena, InMemoryAgentProvider]:
    artifacts = _Artifacts()
    statuses: dict[str, FeasibilityStatus] = {}
    queue: list[dict[str, object] | BaseException] = []
    for index, output in enumerate(outputs):
        if isinstance(output, BaseException):
            queue.append(output)
            continue
        payload = dict(output)
        utility = float(payload["utility"])
        plan = MotionPlan(
            plan_id=f"plan-{index}-{utility}",
            plan_kind="arena-test",
            tcp_frame_id="tool0",
            planner_backend="trusted-test-planner",
            robot_model_digest=_DIGEST_C,
            expected_snapshot=_SNAPSHOT,
            max_start_deviation_rad=0.02,
            possibly_affected_revisions=("robot", "scene"),
            motion=JointPath(
                joint_names=_SNAPSHOT.joint_names,
                positions_rad=((0.01 + index * 0.001, 0.02),),
            ),
        )
        plan_ref = artifacts.add_plan(plan)
        if index in corrupt_plan_ref_at:
            plan_ref = ResolvedArtifactRef(plan_ref.artifact_id, "sha256:" + "f" * 64)
        payload["plan_ref"] = plan_ref
        payload["snapshot_ref"] = _SNAPSHOT_REF
        claimed = str(payload.get("metadata", {}).get("feasibility", "unknown"))
        statuses[plan.content_digest] = (
            trusted_statuses[index]
            if trusted_statuses is not None
            else FeasibilityStatus(claimed)
        )
        queue.append(payload)

    def handler(profile: ActorProfile, spec: object, isolation: object) -> dict[str, object]:
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    provider = InMemoryAgentProvider(handler)
    registry = ActorRegistry(
        {"memory": provider},
        namespace_root="episode-1",
        workspace_root="/tmp/arena-tests",
    )
    arena = SwarmArena(
        registry,
        promotion_authority=RuntimeMotionPromotionAuthority(
            episode_id="episode-1",
            artifacts=artifacts,
            snapshot_providers={("robot-world", "arm"): lambda *_: current_snapshot},
            feasibility_checkers={("robot-world", "arm"): _Checker(statuses)},
        ),
        context_guard=context_guard
        or RuntimeArenaContextGuard(
            episode_id="episode-1",
            current_revision=lambda context: (
                context.graph_id,
                context.graph_revision,
            ),
        ),
        consumption_ledger=consumption_ledger or ArenaConsumptionLedger(),
        shadow_backends=shadow_backends or ShadowBackendRegistry(),
        gates=(
            BasicHypothesisGate(
                expected_frame="world", expected_snapshot_ref=_SNAPSHOT_REF
            ),
            MetadataStatusGate("feasibility", "feasibility"),
        ),
        policy=ArenaPolicy(max_candidates=3, disagreement_distance_m=0.025),
    )
    return arena, provider


def test_risk_report_is_deterministic_and_controls_one_to_k_expansion() -> None:
    low_first = _safe_risk(max_candidates=3)
    low_second = _safe_risk(max_candidates=3)
    high = _high_risk(max_candidates=3)

    assert low_first == low_second
    assert low_first.level is RiskLevel.LOW
    assert low_first.recommended_candidates == 1
    assert high.level is RiskLevel.HIGH
    assert high.recommended_candidates == 3
    assert high.reasons == (
        "low_grounding_confidence",
        "narrow_target_margin",
        "ik_unknown",
        "collision_unknown",
        "low_clearance",
        "high_held_pose_uncertainty",
        "prior_failure",
    )


def test_low_risk_runs_one_candidate_and_high_risk_runs_bounded_k() -> None:
    candidates = (_candidate("c1"), _candidate("c2"), _candidate("c3"))
    low_arena, low_provider = _arena([_card(utility=1.0)])
    low = low_arena.run(
        context=_context("arena-low"),
        risk_report=_safe_risk(),
        candidates=candidates,
        candidate_budget_remaining=3,
    )
    assert low.candidate_budget_used == 1
    assert low.selected_candidate_id == "c1"
    assert [call[0] for call in low_provider.calls] == ["spawn", "invoke", "retire"]

    high_arena, high_provider = _arena(
        [_card(utility=1.0), _card(utility=3.0), _card(utility=2.0)]
    )
    high = high_arena.run(
        context=_context("arena-high"),
        risk_report=_high_risk(),
        candidates=candidates,
        candidate_budget_remaining=10,
    )
    assert high.candidate_budget_used == 3
    assert high.selected_candidate_id == "c2"
    assert len([call for call in high_provider.calls if call[0] == "invoke"]) == 3
    assert high.manager_signals[0] is ManagerSignal.RISK_EXPANSION


def test_arena_candidate_budget_caps_high_risk_expansion() -> None:
    arena, _ = _arena([_card(utility=1.0), _card(utility=2.0)])
    result = arena.run(
        context=_context("arena-budget"),
        risk_report=_high_risk(),
        candidates=(_candidate("c1"), _candidate("c2"), _candidate("c3")),
        candidate_budget_remaining=2,
    )

    assert result.requested_candidates == 3
    assert result.candidate_budget_used == 2
    assert len(result.candidate_results) == 2


def test_candidate_failure_is_isolated_and_other_candidate_can_win() -> None:
    arena, provider = _arena([RuntimeError("candidate crashed"), _card(utility=2.0)])
    result = arena.run(
        context=_context("arena-failure"),
        risk_report=_high_risk(max_candidates=2),
        candidates=(_candidate("bad"), _candidate("good")),
        candidate_budget_remaining=2,
    )

    assert [record.status for record in result.candidate_results] == [
        CandidateStatus.FAILED,
        CandidateStatus.ACCEPTED,
    ]
    assert result.selected_candidate_id == "good"
    assert provider.runtime_for("arena-failure-bad").retired is True
    assert provider.runtime_for("arena-failure-good").retired is True


def test_hard_gate_rejection_routes_all_rejected_without_selection() -> None:
    arena, _ = _arena([_card(utility=1.0, certificate="unknown")])
    result = arena.run(
        context=_context("arena-rejected"),
        risk_report=_safe_risk(),
        candidates=(_candidate("c1"),),
        candidate_budget_remaining=1,
    )

    assert result.candidate_results[0].status is CandidateStatus.REJECTED
    assert result.selected_candidate_id is None
    assert ManagerSignal.ALL_CANDIDATES_REJECTED in result.manager_signals


def test_candidate_metadata_cannot_forge_runtime_motion_safety() -> None:
    arena, _ = _arena(
        [_card(utility=99.0, certificate="pass")],
        trusted_statuses=[FeasibilityStatus.FAIL],
    )

    result = arena.run(
        context=_context("arena-forged-metadata"),
        risk_report=_safe_risk(),
        candidates=(_candidate("forged"),),
        candidate_budget_remaining=1,
    )

    candidate = result.candidate_results[0]
    assert candidate.status is CandidateStatus.REJECTED
    assert result.selected_candidate_id is None
    core = {item.gate_id: item for item in candidate.gate_results}
    assert core["kinematic_feasibility"].passed is False
    assert core["collision"].passed is False
    assert core["joint_limits"].passed is False


@pytest.mark.parametrize("risk", [_safe_risk(), _high_risk(max_candidates=1)])
def test_risk_report_never_bypasses_unknown_core_gate(risk: RiskReport) -> None:
    arena, _ = _arena(
        [_card(utility=2.0, certificate="pass")],
        trusted_statuses=[FeasibilityStatus.UNKNOWN],
    )

    result = arena.run(
        context=_context(f"arena-risk-{risk.level.value}"),
        risk_report=risk,
        candidates=(_candidate("unknown"),),
        candidate_budget_remaining=1,
    )

    assert result.selected_candidate_id is None
    assert result.promotion_receipt is None


def test_stale_current_snapshot_and_artifact_digest_mismatch_fail_closed() -> None:
    stale = _SNAPSHOT.model_copy(update={"scene_revision": 5})
    stale_arena, _ = _arena([_card(utility=1.0)], current_snapshot=stale)
    stale_result = stale_arena.run(
        context=_context("arena-stale-current"),
        risk_report=_safe_risk(),
        candidates=(_candidate("stale"),),
        candidate_budget_remaining=1,
    )
    stale_gates = {
        item.gate_id: item for item in stale_result.candidate_results[0].gate_results
    }
    assert stale_result.selected_candidate_id is None
    assert stale_gates["current_snapshot_freshness"].passed is False

    corrupt_arena, _ = _arena(
        [_card(utility=1.0)], corrupt_plan_ref_at=frozenset({0})
    )
    corrupt_result = corrupt_arena.run(
        context=_context("arena-corrupt-ref"),
        risk_report=_safe_risk(),
        candidates=(_candidate("corrupt"),),
        candidate_budget_remaining=1,
    )
    corrupt_gates = {
        item.gate_id: item for item in corrupt_result.candidate_results[0].gate_results
    }
    assert corrupt_result.selected_candidate_id is None
    assert corrupt_gates["artifact_integrity"].passed is False


def test_selected_hypothesis_yields_effect_free_promotion_receipt() -> None:
    arena, _ = _arena([_card(utility=1.0)])
    result = arena.run(
        context=_context("arena-promotion-receipt"),
        risk_report=_safe_risk(),
        candidates=(_candidate("selected"),),
        candidate_budget_remaining=1,
    )

    receipt = result.promotion_receipt
    assert receipt is not None
    assert result.selected_action_spec_ref == receipt.action_spec_ref
    assert receipt.authoritative_effect_committed is False
    assert receipt.promotion_status == "eligible_for_action_admission"
    assert receipt.feasibility_certificate.overall_status is FeasibilityStatus.PASS


def test_authoritative_candidate_is_rejected_before_provider_spawn() -> None:
    arena, provider = _arena([])
    forbidden = ArenaCandidateSpec(
        candidate_id="unsafe",
        strategy="direct-robot-action",
        profile=ActorProfile(
            profile_id="unsafe-profile",
            provider_id="memory",
            lifecycle=ActorLifecycle.EPHEMERAL,
            effect_ceiling=frozenset({"authoritative_world.write"}),
        ),
        objective="move the real robot",
        effect_scope=EffectScope.AUTHORITATIVE_WORLD,
        requested_effects=frozenset({"authoritative_world.write"}),
    )

    result = arena.run(
        context=_context("arena-unsafe"),
        risk_report=_safe_risk(),
        candidates=(forbidden,),
        candidate_budget_remaining=1,
    )

    assert result.candidate_results[0].status is CandidateStatus.FAILED
    assert "cannot use authoritative_world" in result.candidate_results[0].error
    assert provider.calls == ()


def test_roster_updates_record_only_actual_spawn_and_retire_not_selection() -> None:
    arena, _ = _arena([_card(utility=1.0), _card(utility=2.0)])
    result = arena.run(
        context=_context("arena-roster", revision=11),
        risk_report=_high_risk(max_candidates=2),
        candidates=(_candidate("c1"), _candidate("c2")),
        candidate_budget_remaining=2,
    )

    assert result.graph_revision == 11
    assert result.selected_candidate_id == "c2"
    assert [event.operation for event in result.roster_updates] == [
        RosterOperation.SPAWN,
        RosterOperation.RETIRE,
        RosterOperation.SPAWN,
        RosterOperation.RETIRE,
    ]
    assert all("selected" not in (event.reason or "") for event in result.roster_updates)


def test_candidates_get_separate_namespace_workspace_and_shadow_world() -> None:
    shadow_backend = _ShadowBackend("shadow-sim")
    arena, _ = _arena(
        [_card(utility=1.0), _card(utility=2.0)],
        shadow_backends=ShadowBackendRegistry(
            (
                RegisteredShadowBackend(
                    backend_id="shadow-sim",
                    backend=shadow_backend,
                    world_resource_bindings=frozenset(
                        {("shadow-a", "arm"), ("shadow-b", "arm")}
                    ),
                ),
            )
        ),
    )
    shadow1 = _candidate("c1", world_id="shadow-a")
    shadow1 = ArenaCandidateSpec(
        **{
            **shadow1.__dict__,
            "effect_scope": EffectScope.SHADOW_WORLD,
            "requested_effects": frozenset({"shadow_world.write"}),
            "shadow_backend_id": "shadow-sim",
            "shadow_resource_id": "arm",
            "profile": ActorProfile(
                profile_id="profile-c1",
                provider_id="memory",
                lifecycle=ActorLifecycle.EPHEMERAL,
                effect_ceiling=frozenset({"shadow_world.write"}),
                isolation=IsolationPolicy(world_id="shadow-a"),
            ),
        }
    )
    shadow2 = ArenaCandidateSpec(
        candidate_id="c2",
        strategy="strategy-c2",
        profile=ActorProfile(
            profile_id="profile-c2",
            provider_id="memory",
            lifecycle=ActorLifecycle.EPHEMERAL,
            effect_ceiling=frozenset({"shadow_world.write"}),
            isolation=IsolationPolicy(world_id="shadow-b"),
        ),
        objective="propose c2",
        effect_scope=EffectScope.SHADOW_WORLD,
        requested_effects=frozenset({"shadow_world.write"}),
        shadow_backend_id="shadow-sim",
        shadow_resource_id="arm",
    )
    result = arena.run(
        context=_context("arena-shadow"),
        risk_report=_high_risk(max_candidates=2),
        candidates=(shadow1, shadow2),
        candidate_budget_remaining=2,
    )

    first, second = result.candidate_results
    assert first.namespace_id != second.namespace_id
    assert first.workspace_id != second.workspace_id
    assert {first.world_id, second.world_id} == {"shadow-a", "shadow-b"}
    assert all(item.shadow_world_proof is not None for item in (first, second))


def test_deterministic_rank_selects_and_reports_geometric_disagreement() -> None:
    arena, _ = _arena(
        [
            _card(utility=1.0, endpoint=(0.0, 0.0, 0.0)),
            _card(utility=2.0, endpoint=(0.1, 0.0, 0.0)),
        ]
    )
    result = arena.run(
        context=_context("arena-disagreement"),
        risk_report=_high_risk(max_candidates=2),
        candidates=(_candidate("c1"), _candidate("c2")),
        candidate_budget_remaining=2,
    )

    assert result.selected_candidate_id == "c2"
    assert result.disagreement is True
    assert ManagerSignal.DISAGREEMENT in result.manager_signals


def test_arena_run_id_cannot_silently_reuse_retired_actor_handles() -> None:
    arena, _ = _arena([_card(utility=1.0)])
    context = _context("arena-once")
    arena.run(
        context=context,
        risk_report=_safe_risk(),
        candidates=(_candidate("c1"),),
        candidate_budget_remaining=1,
    )

    with pytest.raises(ArenaPolicyError, match="already been consumed"):
        arena.run(
            context=context,
            risk_report=_safe_risk(),
            candidates=(_candidate("c1"),),
            candidate_budget_remaining=1,
        )


def test_crash_reservation_survives_restart_and_is_never_replayed(tmp_path) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    ledger_path = tmp_path / "arena_consumption.v1.jsonl"
    first_ledger = ArenaConsumptionLedger(ledger_path)
    crashed_arena, crashed_provider = _arena(
        [SimulatedProcessCrash("power loss")],
        consumption_ledger=first_ledger,
    )
    context = _context("arena-crash-once", budget_limit=1)

    with pytest.raises(SimulatedProcessCrash):
        crashed_arena.run(
            context=context,
            risk_report=_safe_risk(),
            candidates=(_candidate("crash"),),
            candidate_budget_remaining=1,
        )

    assert [record.record_kind for record in first_ledger.records] == ["reservation"]
    assert crashed_provider.runtime_for("arena-crash-once-crash").retired is True

    restarted_ledger = ArenaConsumptionLedger(ledger_path)
    restarted_arena, restarted_provider = _arena(
        [_card(utility=1.0)],
        consumption_ledger=restarted_ledger,
    )
    with pytest.raises(ArenaPolicyError, match="already been consumed"):
        restarted_arena.run(
            context=context,
            risk_report=_safe_risk(),
            candidates=(_candidate("crash"),),
            candidate_budget_remaining=1,
        )
    assert restarted_provider.calls == ()


def test_candidate_budget_is_ledger_owned_across_restart(tmp_path) -> None:
    ledger_path = tmp_path / "arena_consumption.v1.jsonl"
    first, _ = _arena(
        [_card(utility=1.0), _card(utility=2.0)],
        consumption_ledger=ArenaConsumptionLedger(ledger_path),
    )
    first_result = first.run(
        context=_context("arena-budget-first", budget_id="shared", budget_limit=3),
        risk_report=_high_risk(max_candidates=2),
        candidates=(_candidate("a"), _candidate("b")),
        candidate_budget_remaining=99,
    )
    assert first_result.candidate_budget_used == 2

    replay, replay_provider = _arena(
        [_card(utility=9.0)],
        consumption_ledger=ArenaConsumptionLedger(ledger_path),
    )
    with pytest.raises(ArenaPolicyError, match="already been consumed"):
        replay.run(
            context=_context(
                "arena-budget-first", budget_id="shared", budget_limit=3
            ),
            risk_report=_safe_risk(),
            candidates=(_candidate("replay"),),
            candidate_budget_remaining=99,
        )
    assert replay_provider.calls == ()

    restarted, provider = _arena(
        [_card(utility=3.0), _card(utility=4.0)],
        consumption_ledger=ArenaConsumptionLedger(ledger_path),
    )
    second_result = restarted.run(
        context=_context("arena-budget-second", budget_id="shared", budget_limit=3),
        risk_report=_high_risk(max_candidates=2),
        candidates=(_candidate("c"), _candidate("d")),
        candidate_budget_remaining=99,
    )

    assert second_result.candidate_budget_used == 1
    assert len([call for call in provider.calls if call[0] == "invoke"]) == 1
    reservations = [
        record
        for record in ArenaConsumptionLedger(ledger_path).records
        if record.record_kind == "reservation"
    ]
    assert [record.reserved_candidates for record in reservations] == [2, 1]


def test_zero_durable_reservation_is_explicit_quota_exhaustion_and_replays(
    tmp_path,
) -> None:
    ledger_path = tmp_path / "arena_consumption.v1.jsonl"
    first, _ = _arena(
        [_card(utility=1.0)],
        consumption_ledger=ArenaConsumptionLedger(ledger_path),
    )
    first.run(
        context=_context("arena-quota-first", budget_id="shared", budget_limit=1),
        risk_report=_safe_risk(),
        candidates=(_candidate("first"),),
        candidate_budget_remaining=1,
    )

    exhausted_arena, exhausted_provider = _arena(
        [_card(utility=2.0)],
        consumption_ledger=ArenaConsumptionLedger(ledger_path),
    )
    exhausted_context = _context(
        "arena-quota-exhausted", budget_id="shared", budget_limit=1
    )
    exhausted = exhausted_arena.run(
        context=exhausted_context,
        risk_report=_safe_risk(),
        candidates=(_candidate("never-spawned"),),
        candidate_budget_remaining=1,
        recovery_safe=True,
        run_binding_digest=_DIGEST_C,
    )

    assert exhausted.quota_exhausted is True
    assert exhausted.requested_candidates == 1
    assert exhausted.candidate_budget_used == 0
    assert exhausted.candidate_results == ()
    assert exhausted.selected_candidate_id is None
    assert exhausted.selection_reason == (
        "durable candidate quota exhausted: reservation admitted 0 of 1 "
        "requested candidates"
    )
    assert ManagerSignal.ALL_CANDIDATES_REJECTED not in exhausted.manager_signals
    assert exhausted_provider.calls == ()

    durable = ArenaConsumptionLedger(ledger_path)
    reservations = [
        record for record in durable.records if record.record_kind == "reservation"
    ]
    assert [record.reserved_candidates for record in reservations] == [1, 0]
    completion = durable.records[-1]
    assert completion.record_kind == "completion"
    assert completion.result_payload is not None
    assert completion.result_payload["quota_exhausted"] is True

    legacy_payload = exhausted.model_dump(mode="json")
    legacy_payload.pop("quota_exhausted")
    assert ArenaResult.model_validate(legacy_payload).quota_exhausted is True
    with pytest.raises(ValueError, match="must be marked quota_exhausted"):
        ArenaResult.model_validate(
            {
                **legacy_payload,
                "quota_exhausted": False,
            }
        )

    restarted, restarted_provider = _arena(
        [_card(utility=9.0)],
        consumption_ledger=ArenaConsumptionLedger(ledger_path),
    )
    replayed = restarted.run(
        context=exhausted_context,
        risk_report=_safe_risk(),
        candidates=(_candidate("never-spawned"),),
        candidate_budget_remaining=1,
        recovery_safe=True,
        run_binding_digest=_DIGEST_C,
    )

    assert replayed == exhausted
    assert restarted_provider.calls == ()


def test_graph_revision_is_rechecked_after_candidates_before_promotion(tmp_path) -> None:
    calls = 0

    def revision_guard(context: ArenaContext) -> tuple[str, int]:
        nonlocal calls
        calls += 1
        return (
            context.graph_id,
            context.graph_revision if calls == 1 else context.graph_revision + 1,
        )

    ledger = ArenaConsumptionLedger(tmp_path / "arena_consumption.v1.jsonl")
    arena, provider = _arena(
        [_card(utility=1.0)],
        consumption_ledger=ledger,
        context_guard=RuntimeArenaContextGuard(
            episode_id="episode-1", current_revision=revision_guard
        ),
    )

    with pytest.raises(ArenaStaleContextError, match="became stale"):
        arena.run(
            context=_context("arena-stale-after-exploration"),
            risk_report=_safe_risk(),
            candidates=(_candidate("candidate"),),
            candidate_budget_remaining=1,
        )

    assert calls == 2
    assert provider.runtime_for(
        "arena-stale-after-exploration-candidate"
    ).retired is True
    assert [record.record_kind for record in ledger.records] == ["reservation"]


def test_shadow_world_declaration_cannot_spoof_authoritative_backend() -> None:
    authoritative = _ShadowBackend("spoofed-shadow", reports_shadow=False)
    arena, provider = _arena(
        [_card(utility=1.0)],
        shadow_backends=ShadowBackendRegistry(
            (
                RegisteredShadowBackend(
                    backend_id="spoofed-shadow",
                    backend=authoritative,
                    world_resource_bindings=frozenset(
                        {("claimed-shadow", "arm")}
                    ),
                ),
            )
        ),
    )
    candidate = ArenaCandidateSpec(
        candidate_id="spoof",
        strategy="spoof-shadow-world",
        profile=ActorProfile(
            profile_id="spoof-profile",
            provider_id="memory",
            lifecycle=ActorLifecycle.EPHEMERAL,
            effect_ceiling=frozenset({"shadow_world.write"}),
            isolation=IsolationPolicy(world_id="claimed-shadow"),
        ),
        objective="attempt an unsafe rollout",
        effect_scope=EffectScope.SHADOW_WORLD,
        requested_effects=frozenset({"shadow_world.write"}),
        shadow_backend_id="spoofed-shadow",
        shadow_resource_id="arm",
    )

    result = arena.run(
        context=_context("arena-shadow-spoof"),
        risk_report=_safe_risk(),
        candidates=(candidate,),
        candidate_budget_remaining=1,
    )

    assert result.selected_candidate_id is None
    assert result.candidate_results[0].status is CandidateStatus.FAILED
    assert "does not prove the declared shadow world" in (
        result.candidate_results[0].error or ""
    )
    assert provider.calls == ()
