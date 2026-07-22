from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from robomex.data import (
    AttachmentEvidence,
    AttachmentStatus,
    EpisodeDataPlane,
    RevisionVector,
    SchemaRegistry,
    ValidityLifecycle,
    ValidityVector,
)
from robomex.manipulation import (
    AlignmentError,
    BoundedCorrection,
    BowlPlaceObservation,
    HeldBowlEstimate,
    PlateSupportTarget,
    PoseUncertainty,
    QuaternionWXYZ,
    ServoDecision,
    ServoOutcome,
    SupportFootprint,
    UnitVector3,
    Vector3,
    VisibilityStatus,
    compute_alignment_error,
    register_bowl_place_schemas,
)
from robomex.orchestration.actors import (
    ActorIsolation,
    InvocationSpec,
    WorkspaceMode,
)
from robomex.orchestration.arena import CheckStatus, RiskLevel, RiskPolicy, RiskReport
from robomex.orchestration.risk_provider import (
    DETERMINISTIC_MOTION_RISK_RUNNER_REF,
    MOTION_RISK_CAPABILITIES,
    MOTION_RISK_REPORT_SCHEMA_ID,
    DeterministicMotionRiskProvider,
    MotionRiskContractError,
    MotionRiskProviderConfig,
    build_motion_risk_actor_bindings,
)
from robomex.runtime.events import ControlOutcome


def _revisions(value: int) -> RevisionVector:
    return RevisionVector(
        scene=value,
        arm=value,
        gripper=value,
        attachment=value,
        camera={"front": value},
    )


def _footprint(center: Vector3, radius: float) -> SupportFootprint:
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


def _observation(*, offset_x_m: float = 0.0) -> BowlPlaceObservation:
    generation = 7
    revisions = _revisions(generation)
    held_center = Vector3(x=0.50, y=0.0, z=0.20)
    target_center = Vector3(x=0.50 + offset_x_m, y=0.0, z=0.20)
    held = HeldBowlEstimate(
        estimate_id="held-7",
        entity_id="bowl-1",
        frame_id="world",
        snapshot_id="snapshot-7",
        observation_generation=generation,
        revisions=revisions,
        bottom_center_m=held_center,
        support_footprint=_footprint(held_center, 0.03),
        orientation=QuaternionWXYZ.from_yaw(0.0),
        uncertainty=_uncertainty(),
        confidence=0.95,
        evidence_refs=("bowl-frame-7",),
    )
    target = PlateSupportTarget(
        target_id="target-7",
        entity_id="plate-1",
        frame_id="world",
        snapshot_id="snapshot-7",
        observation_generation=generation,
        revisions=revisions,
        support_center_m=target_center,
        support_footprint=_footprint(target_center, 0.10),
        surface_normal=UnitVector3(x=0.0, y=0.0, z=1.0),
        orientation=QuaternionWXYZ.from_yaw(0.0),
        uncertainty=_uncertainty(),
        safe_margin_m=0.01,
        confidence=0.96,
        evidence_refs=("plate-frame-7",),
    )
    return BowlPlaceObservation(
        snapshot_id="snapshot-7",
        observation_generation=generation,
        revisions=revisions,
        frame_id="world",
        expected_bowl_entity_id="bowl-1",
        expected_target_entity_id="plate-1",
        attachment_status=AttachmentStatus.VERIFIED_HELD,
        held_visibility=VisibilityStatus.VISIBLE,
        target_visibility=VisibilityStatus.VISIBLE,
        held=held,
        target=target,
    )


@dataclass(frozen=True)
class _Chain:
    plane: EpisodeDataPlane
    refs: dict[str, object]


def _publish_chain(
    root: Path,
    *,
    correction_iteration: int | None = None,
    alignment_mutator: Callable[[AlignmentError], AlignmentError] | None = None,
    attachment_entity_id: str = "bowl-1",
    alignment_lineage_complete: bool = True,
    servo_extra_lineage: bool = False,
) -> _Chain:
    registry = SchemaRegistry()
    register_bowl_place_schemas(registry)
    plane = EpisodeDataPlane(
        root,
        episode_id="episode-risk",
        schema_registry=registry,
        strict_schema_prefixes=("robomex.",),
    )
    plane.open_workflow("workflow-risk")
    causal = plane.publish(
        workflow_id="workflow-risk",
        activation_id="execute_previous",
        attempt=1,
        port="receipt",
        schema="test.causal_receipt.v1",
        payload={"action_id": "action-1"},
    )
    observation = _observation(offset_x_m=0.02 if correction_iteration is not None else 0.0)
    observation_record = plane.publish(
        workflow_id="workflow-risk",
        activation_id="capture_alignment",
        attempt=1,
        port="observation",
        schema="robomex.bowl_place_observation.v1",
        payload=observation.model_dump(mode="json"),
        lineage=(causal.ref,),
    )
    attachment = AttachmentEvidence(
        evidence_id="attachment-7",
        entity_id=attachment_entity_id,
        entity_track_id="track-bowl-1",
        status="verified_held",
        source_observation_id=observation.snapshot_id,
        source_observation_revision=7,
        observation_domain="camera.front",
        base_state_revision=4,
        action_id="action-1",
        evidence_refs=(observation_record.artifact_id, causal.artifact_id),
        method="synchronized_tracking_and_gripper_guard",
        reason="held evidence for the synchronized checkpoint",
        confidence=0.98,
        validity=ValidityVector(
            lifecycle=ValidityLifecycle.DERIVED,
            depends_on_revisions={"camera.front": 7},
            observation_id=observation.snapshot_id,
            state_revision=4,
            method="exact_observation_and_state_revision",
            reason="valid only for this observation and state revision",
            confidence=0.98,
        ),
    )
    attachment_record = plane.publish(
        workflow_id="workflow-risk",
        activation_id="verify_alignment_attachment",
        attempt=1,
        port="attachment_evidence",
        schema="robomex.attachment_evidence.v1",
        payload=attachment.model_dump(mode="json"),
        lineage=(observation_record.ref, causal.ref),
    )
    assert observation.held is not None and observation.target is not None
    alignment = compute_alignment_error(observation.held, observation.target)
    if alignment_mutator is not None:
        alignment = alignment_mutator(alignment)
    alignment_record = plane.publish(
        workflow_id="workflow-risk",
        activation_id="estimate_alignment",
        attempt=1,
        port="alignment_error",
        schema="robomex.alignment_error.v1",
        payload=alignment.model_dump(mode="json"),
        lineage=(
            (observation_record.ref, attachment_record.ref)
            if alignment_lineage_complete
            else (observation_record.ref,)
        ),
    )
    correction = None
    if correction_iteration is not None:
        correction = BoundedCorrection(
            iteration=correction_iteration,
            expressed_in_frame="world",
            delta_translation_m=Vector3(x=0.01, y=0.0, z=0.0),
            delta_yaw_rad=0.0,
            source_snapshot_id=observation.snapshot_id,
            source_generation=observation.observation_generation,
            source_revisions=observation.revisions,
            required_next_generation=observation.observation_generation + 1,
            cumulative_translation_m=0.01 * correction_iteration,
            cumulative_yaw_rad=0.0,
        )
    outcome = (
        ServoOutcome.CORRECTION_REQUIRED
        if correction is not None
        else ServoOutcome.WITHIN_TOLERANCE
    )
    # The production servo gate independently recomputes the same measurement,
    # so its correlation ID differs while all physical content stays identical.
    decision_alignment = (
        compute_alignment_error(observation.held, observation.target)
        if alignment_mutator is None
        else alignment
    )
    servo = ServoDecision(
        outcome=outcome,
        control_outcome=(
            ControlOutcome.NEEDS_ADJUSTMENT if correction is not None else ControlOutcome.SUCCESS
        ),
        observation_generation=observation.observation_generation,
        reason="bounded correction" if correction is not None else "within tolerance",
        alignment_error=decision_alignment,
        correction=correction,
    )
    servo_record = plane.publish(
        workflow_id="workflow-risk",
        activation_id="alignment_gate",
        attempt=1,
        port="servo_decision",
        schema="robomex.servo_decision.v1",
        payload=servo.model_dump(mode="json"),
        lineage=(
            observation_record.ref,
            alignment_record.ref,
            attachment_record.ref,
            *((causal.ref,) if servo_extra_lineage else ()),
        ),
    )
    return _Chain(
        plane=plane,
        refs={
            "observation": observation_record.ref,
            "attachment_evidence": attachment_record.ref,
            "alignment_error": alignment_record.ref,
            "servo_decision": servo_record.ref,
        },
    )


def _invoke(
    chain: _Chain,
    config: MotionRiskProviderConfig,
) -> tuple[RiskReport, object]:
    bindings = build_motion_risk_actor_bindings(config)
    provider = bindings.providers[bindings.provider_id]
    profile = bindings.profiles[DETERMINISTIC_MOTION_RISK_RUNNER_REF]
    provider.bind_episode_data_plane(chain.plane)
    runtime = provider.spawn(
        profile,
        ActorIsolation(
            owner_actor_id="risk-actor",
            namespace_id="actors/risk-actor",
            workspace_id="actors/risk-actor",
            workspace_mode=WorkspaceMode.ISOLATED,
        ),
    )
    spec = InvocationSpec(
        invocation_id="risk-command-1",
        objective="deterministically assess motion risk",
        inputs={name: ref.to_mapping() for name, ref in chain.refs.items()},
        output_contract={"risk_report": MOTION_RISK_REPORT_SCHEMA_ID},
        requested_capabilities=MOTION_RISK_CAPABILITIES,
        requested_effects=frozenset(),
        budget={"model_calls": 0, "tokens": 0},
        metadata={
            "episode_id": "episode-risk",
            "workflow_id": "workflow-risk",
            "node_params": {},
        },
    )
    result = provider.invoke(runtime, spec)
    assert result.outcome is ControlOutcome.SUCCESS
    assert len(result.artifacts) == 1
    emission = result.artifacts[0]
    return RiskReport.model_validate(emission.payload), emission


def _safe_config(**updates: object) -> MotionRiskProviderConfig:
    values = {
        "expected_bowl_entity_id": "bowl-1",
        "expected_target_entity_id": "plate-1",
        "fixed_ik_status": CheckStatus.PASS,
        "fixed_collision_status": CheckStatus.PASS,
        "fixed_clearance_m": 0.03,
    }
    values.update(updates)
    return MotionRiskProviderConfig(**values)


def test_low_risk_emits_one_canonical_report_with_complete_lineage(tmp_path: Path) -> None:
    chain = _publish_chain(tmp_path / "episode", correction_iteration=1)

    report, emission = _invoke(chain, _safe_config())

    assert report.level is RiskLevel.LOW
    assert report.recommended_candidates == 1
    assert report.inputs.grounding_confidence == pytest.approx(0.95)
    assert report.inputs.target_margin_m is not None
    assert report.inputs.target_margin_m > 0.015
    assert report.inputs.monitor_observable
    assert tuple(emission.lineage) == tuple(chain.refs.values())
    assert emission.port == "risk_report"
    assert emission.schema_id == MOTION_RISK_REPORT_SCHEMA_ID


def test_within_tolerance_decision_cannot_enter_correction_risk_gate(
    tmp_path: Path,
) -> None:
    chain = _publish_chain(tmp_path / "episode")

    with pytest.raises(MotionRiskContractError, match="only correction_required"):
        _invoke(chain, _safe_config())


def test_correction_history_expands_high_risk_to_manifest_pinned_k(tmp_path: Path) -> None:
    chain = _publish_chain(tmp_path / "episode", correction_iteration=4)
    policy = RiskPolicy(max_candidates=5, high_risk_score=0.25)

    report, _ = _invoke(chain, _safe_config(risk_policy=policy))

    assert report.level is RiskLevel.HIGH
    assert report.inputs.prior_failures == 3
    assert "prior_failure" in report.reasons
    assert report.recommended_candidates == 5
    assert report.policy == policy


@pytest.mark.parametrize(
    "mutator",
    [
        lambda error: error.model_copy(update={"snapshot_id": "snapshot-stale"}),
        lambda error: error.model_copy(update={"observation_generation": 6}),
        lambda error: error.model_copy(update={"revisions": _revisions(6)}),
    ],
    ids=("snapshot", "generation", "revisions"),
)
def test_stale_alignment_clock_fails_closed(
    tmp_path: Path,
    mutator: Callable[[AlignmentError], AlignmentError],
) -> None:
    chain = _publish_chain(tmp_path / "episode", alignment_mutator=mutator)

    with pytest.raises(MotionRiskContractError, match="stale|revision"):
        _invoke(chain, _safe_config())


def test_entity_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    chain = _publish_chain(
        tmp_path / "episode",
        attachment_entity_id="different-bowl",
    )

    with pytest.raises(MotionRiskContractError, match="verified-held attachment"):
        _invoke(chain, _safe_config())


def test_missing_direct_alignment_lineage_fails_closed(tmp_path: Path) -> None:
    chain = _publish_chain(
        tmp_path / "episode",
        alignment_lineage_complete=False,
    )

    with pytest.raises(MotionRiskContractError, match="lineage"):
        _invoke(chain, _safe_config())


def test_unadmitted_extra_servo_lineage_fails_closed(tmp_path: Path) -> None:
    chain = _publish_chain(
        tmp_path / "episode",
        correction_iteration=1,
        servo_extra_lineage=True,
    )

    with pytest.raises(MotionRiskContractError, match="exactly"):
        _invoke(chain, _safe_config())


def test_bindings_pin_policy_and_expose_no_llm_or_effect_authority() -> None:
    config = _safe_config(risk_policy=RiskPolicy(max_candidates=4))

    bindings = build_motion_risk_actor_bindings(config)
    profile = bindings.profiles[DETERMINISTIC_MOTION_RISK_RUNNER_REF]

    assert isinstance(bindings.providers[bindings.provider_id], DeterministicMotionRiskProvider)
    assert profile.model == ""
    assert profile.effect_ceiling == frozenset()
    assert profile.capability_ceiling == MOTION_RISK_CAPABILITIES
    assert profile.metadata["provider_config_digest"] == config.content_digest
    assert profile.metadata["provider_config"] == config.model_dump(mode="json")
    assert profile.metadata["risk_policy_digest"] == config.risk_policy_digest
    assert profile.metadata["llm_access"] is False
    assert profile.metadata["action_authority"] is False
