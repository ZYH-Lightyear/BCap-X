"""Offline tests for frozen v2 contracts and evolve-ready admission surfaces."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from robomex.contracts import (
    ActorProfile,
    ContentPin,
    ContractHashDriftError,
    ContractRegistry,
    DeclaredEffect,
    DuplicateContractError,
    EffectContract,
    EffectScope,
    FunctionExport,
    ObligationSeverity,
    ProtocolSpec,
    SkillAsset,
    SkillAssetKind,
    SkillManifest,
    SlotPolicy,
    StatePredicate,
    VerificationPhase,
    VerifierObligation,
    revalidate_sealed,
)
from robomex.evolution import (
    BackendPin,
    BackendRole,
    CandidateAdmissionMode,
    CandidateComponentSnapshot,
    CandidateConfigSnapshot,
    DuplicateEvolutionIdError,
    EvaluationGateRecord,
    EvaluationReport,
    EvaluationRequest,
    Evaluator,
    EvolutionDisabledError,
    EvolutionHashDriftError,
    EvolvableComponentKind,
    EvolvableComponentRegistry,
    EvolvableComponentSpec,
    FunctionPin,
    GateStatus,
    MetricAggregation,
    MetricDefinition,
    MetricDirection,
    MetricRecord,
    ModelPin,
    PromotionRecord,
    PromotionStage,
    PromptPin,
    RunBudgets,
    RunManifest,
    SafetyBoundary,
    SafetyBoundaryViolationError,
    SkillPin,
    TaskSnapshot,
)
from robomex.orchestration.actors import ActorProfile as RuntimeActorProfile
from robomex.runtime.events import ControlOutcome

NOW = datetime(2026, 7, 22, 8, 0, tzinfo=timezone.utc)  # noqa: UP017


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _effect() -> EffectContract:
    return EffectContract(
        contract_id="effect.place",
        requires=(
            StatePredicate(
                predicate_id="bowl.attached",
                expression="attachment_status == 'verified_held'",
            ),
        ),
        effects=(
            DeclaredEffect(
                effect_id="arm.motion",
                scope=EffectScope.AUTHORITATIVE_WORLD,
                operation="execute sealed joint path",
                resource_selector="robot.arm",
            ),
        ),
        invalidates=(
            StatePredicate(
                predicate_id="bowl.pose.old",
                expression="bowl pose at admission revision",
            ),
        ),
        establishes=(
            StatePredicate(
                predicate_id="bowl.on.plate",
                expression="bowl bottom is supported by plate region",
            ),
        ),
    )


def _protocol() -> ProtocolSpec:
    return ProtocolSpec(
        protocol_id="protocol.place_bowl",
        summary="Place an already attached bowl using bounded physical actions.",
        inputs=(
            SlotPolicy(slot_id="bowl.track", schema_id="robomex.track.v1"),
            SlotPolicy(slot_id="plate.region", schema_id="robomex.region.v1"),
        ),
        outputs=(
            SlotPolicy(slot_id="execution.evidence", schema_id="robomex.evidence.v1"),
        ),
        outcomes=(
            ControlOutcome.SUCCESS,
            ControlOutcome.NEEDS_ADJUSTMENT,
            ControlOutcome.FAILED_PLACEMENT,
        ),
        effect_contract=_effect(),
        verifier_obligations=(
            VerifierObligation(
                obligation_id="verify.support",
                phase=VerificationPhase.POSTCONDITION,
                verifier_ref="verifier.bowl_support",
                predicate="bowl bottom is supported and gripper is clear",
                evidence_slots=("execution.evidence",),
                severity=ObligationSeverity.HARD_GATE,
            ),
        ),
        required_capabilities=("motion.execute", "perception.read"),
    )


def _skill(protocol: ProtocolSpec) -> SkillManifest:
    effect = protocol.effect_contract
    return SkillManifest(
        skill_id="skill.place_bowl",
        name="Place bowl with visual servoing",
        summary="Knowledge, API, and canonical code for precise bowl placement.",
        assets=(
            SkillAsset(
                asset_id="knowledge.place",
                kind=SkillAssetKind.KNOWLEDGE,
                relative_path="SKILL.md",
                content_digest=_digest("a"),
                media_type="text/markdown",
            ),
            SkillAsset(
                asset_id="api.place",
                kind=SkillAssetKind.API,
                relative_path="contract.yaml",
                content_digest=_digest("b"),
                media_type="application/yaml",
            ),
            SkillAsset(
                asset_id="code.servo",
                kind=SkillAssetKind.CODE,
                relative_path="scripts/servo.py",
                content_digest=_digest("c"),
                media_type="text/x-python",
            ),
        ),
        functions=(
            FunctionExport(
                function_id="function.servo_step",
                source_asset_id="code.servo",
                entrypoint="scripts/servo.py:servo_step",
                function_digest=_digest("d"),
                interface_digest=_digest("e"),
            ),
        ),
        protocol_pins=(
            ContentPin(
                component_id=protocol.protocol_id,
                revision=protocol.revision,
                content_digest=protocol.content_digest,
            ),
        ),
        effect_contract_pins=(
            ContentPin(
                component_id=effect.contract_id,
                revision=effect.revision,
                content_digest=effect.content_digest,
            ),
        ),
        compatible_actor_profiles=("actor.motion",),
        tags=("placement", "visual_servo"),
    )


def test_contracts_are_closed_frozen_and_canonically_sealed() -> None:
    first = _protocol()
    second_payload = first.model_dump(mode="python", exclude={"content_digest"})
    second_payload["metadata"] = {"z": 1, "a": 2}
    second = ProtocolSpec.model_validate(second_payload)
    third_payload = first.model_dump(mode="python", exclude={"content_digest"})
    third_payload["metadata"] = {"a": 2, "z": 1}
    third = ProtocolSpec.model_validate(third_payload)
    assert second.content_digest == third.content_digest
    assert first.content_digest.startswith("sha256:")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ProtocolSpec.model_validate({**first.model_dump(), "surprise": True})
    with pytest.raises(ValidationError, match="frozen"):
        first.summary = "mutated"  # type: ignore[misc]

    tampered = first.model_copy(update={"summary": "unsafe copy"})
    with pytest.raises(ValidationError, match="content_digest"):
        revalidate_sealed(tampered)


def test_effect_protocol_and_skill_reject_ambiguous_or_unsafe_inventory() -> None:
    with pytest.raises(ValidationError, match="both invalidated and established"):
        EffectContract(
            contract_id="effect.bad",
            invalidates=(StatePredicate(predicate_id="same", expression="old"),),
            establishes=(StatePredicate(predicate_id="same", expression="new"),),
        )
    with pytest.raises(ValidationError, match="safe package-relative"):
        SkillAsset(
            asset_id="escape",
            kind=SkillAssetKind.CODE,
            relative_path="../outside.py",
            content_digest=_digest("a"),
            media_type="text/x-python",
        )

    protocol = _protocol()
    skill = _skill(protocol)
    assert [asset.asset_id for asset in skill.knowledge_assets] == ["knowledge.place"]
    assert [asset.asset_id for asset in skill.api_assets] == ["api.place"]
    assert [asset.asset_id for asset in skill.code_assets] == ["code.servo"]
    bad_export = skill.functions[0].model_copy(update={"source_asset_id": "knowledge.place"})
    payload = skill.model_dump(mode="python", exclude={"content_digest"})
    payload["functions"] = (bad_export,)
    with pytest.raises(ValidationError, match="source must be a code asset"):
        SkillManifest.model_validate(payload)


def test_contract_registry_rejects_duplicate_identity_and_hash_drift() -> None:
    protocol = _protocol()
    skill = _skill(protocol)
    registry = ContractRegistry()
    registry.register_protocol(protocol)
    registry.register_skill(skill)
    snapshot = registry.snapshot()
    restored = ContractRegistry(snapshot)
    assert restored.skill(skill.skill_id, 1).content_digest == skill.content_digest
    assert restored.snapshot().content_digest == snapshot.content_digest

    with pytest.raises(DuplicateContractError, match="Duplicate protocol"):
        registry.register_protocol(protocol)
    changed_payload = protocol.model_dump(mode="python", exclude={"content_digest"})
    changed_payload["summary"] = "Same logical revision with different content."
    changed = ProtocolSpec.model_validate(changed_payload)
    with pytest.raises(ContractHashDriftError, match="changed content digest"):
        registry.register_protocol(changed)


def test_actor_contract_exports_runtime_authority_type_without_copying_it() -> None:
    assert ActorProfile is RuntimeActorProfile


def _run_manifest(catalog_digest: str) -> RunManifest:
    return RunManifest(
        run_id="run.bowl.001",
        task=TaskSnapshot(
            task_id="task.bowl_on_plate",
            instruction="Put the bowl beside the plate onto the plate.",
            success_rubric="The bowl is stably supported by the plate.",
            episode_spec_digest=_digest("1"),
        ),
        seed=1234,
        model=ModelPin(
            model_id="model.small_coder",
            provider_id="provider.local",
            weights_digest=_digest("2"),
            generation_config_digest=_digest("3"),
        ),
        prompts=(PromptPin(prompt_id="prompt.manager", content_digest=_digest("4")),),
        skills=(
            SkillPin(
                skill_id="skill.place_bowl",
                revision=1,
                manifest_digest=_digest("5"),
            ),
        ),
        functions=(
            FunctionPin(
                function_id="function.servo_step",
                skill_id="skill.place_bowl",
                implementation_digest=_digest("6"),
                interface_digest=_digest("7"),
            ),
        ),
        budgets=RunBudgets(
            max_model_calls=20,
            max_tokens=20_000,
            max_wall_time_s=300,
            max_physical_actions=12,
            max_shadow_rollouts=8,
            max_candidates=4,
            max_recoveries=2,
        ),
        backends=(
            BackendPin(
                backend_id="backend.robot",
                role=BackendRole.AUTHORITATIVE,
                implementation_digest=_digest("8"),
                configuration_digest=_digest("9"),
                version="1.0.0",
            ),
            BackendPin(
                backend_id="backend.shadow",
                role=BackendRole.SHADOW,
                implementation_digest=_digest("a"),
                configuration_digest=_digest("b"),
                version="1.0.0",
            ),
        ),
        graph_digest=_digest("c"),
        contract_catalog_digest=catalog_digest,
        schema_registry_digest=_digest("f"),
        runtime_code_digest=_digest("d"),
        actor_profile_pins=(
            ContentPin(component_id="actor.motion", content_digest=_digest("e")),
        ),
        created_at=NOW,
    )


def test_run_manifest_freezes_all_reproducibility_inputs_and_disables_mutation() -> None:
    catalog = ContractRegistry()
    protocol = _protocol()
    catalog.register_protocol(protocol)
    catalog.register_skill(_skill(protocol))
    manifest = _run_manifest(catalog.snapshot().content_digest)
    assert manifest.mutation_policy == "disabled"
    assert manifest.model.weights_digest == _digest("2")
    assert manifest.functions[0].implementation_digest == _digest("6")

    changed_payload = manifest.model_dump(mode="python", exclude={"content_digest"})
    changed_payload["budgets"] = {
        **manifest.budgets.model_dump(),
        "max_physical_actions": 13,
    }
    changed = RunManifest.model_validate(changed_payload)
    assert changed.content_digest != manifest.content_digest
    changed_payload["mutation_policy"] = "enabled"
    with pytest.raises(ValidationError, match="disabled"):
        RunManifest.model_validate(changed_payload)


def _boundary() -> SafetyBoundary:
    return SafetyBoundary(
        boundary_id="safety.action_authority",
        description="Physical authority and hard safety gates are immutable.",
        protected_paths=("/authority", "/hard_gates"),
        required_verifier_ids=("verifier.collision", "verifier.attachment"),
    )


def _component(boundary: SafetyBoundary) -> EvolvableComponentSpec:
    return EvolvableComponentSpec(
        component_id="manager.prompt",
        kind=EvolvableComponentKind.PROMPT,
        base_content_digest=_digest("1"),
        mutable_paths=("/temperature", "/instructions"),
        safety_boundary_ids=(boundary.boundary_id,),
    )


def _baseline(boundary: SafetyBoundary) -> CandidateConfigSnapshot:
    return CandidateConfigSnapshot(
        candidate_id="candidate.main",
        stage=PromotionStage.BASELINE,
        components=(
            CandidateComponentSnapshot(
                component_id="manager.prompt",
                base_content_digest=_digest("1"),
                candidate_content_digest=_digest("1"),
                configuration={"temperature": 0.0},
            ),
        ),
        safety_boundary_pins=(
            ContentPin(
                component_id=boundary.boundary_id,
                revision=boundary.revision,
                content_digest=boundary.content_digest,
            ),
        ),
        created_by="baseline.builder",
        created_at=NOW,
    )


def _draft(
    boundary: SafetyBoundary, baseline: CandidateConfigSnapshot
) -> CandidateConfigSnapshot:
    return CandidateConfigSnapshot(
        candidate_id=baseline.candidate_id,
        revision=2,
        parent_config_digest=baseline.content_digest,
        stage=PromotionStage.DRAFT,
        components=(
            CandidateComponentSnapshot(
                component_id="manager.prompt",
                base_content_digest=_digest("1"),
                candidate_content_digest=_digest("2"),
                changed_paths=("/temperature",),
                configuration={"temperature": 0.1},
            ),
        ),
        safety_boundary_pins=(
            ContentPin(
                component_id=boundary.boundary_id,
                revision=boundary.revision,
                content_digest=boundary.content_digest,
            ),
        ),
        created_by="external.candidate_builder",
        created_at=NOW,
    )


def test_baseline_registry_is_evolve_ready_but_performs_no_mutation() -> None:
    boundary = _boundary()
    component = _component(boundary)
    baseline = _baseline(boundary)
    registry = EvolvableComponentRegistry()
    registry.register_safety_boundary(boundary)
    registry.register_component(component)
    registry.register_candidate(baseline)
    restored = EvolvableComponentRegistry(snapshot=registry.snapshot())
    assert restored.snapshot().content_digest == registry.snapshot().content_digest

    with pytest.raises(EvolutionDisabledError, match="baseline-only"):
        registry.register_candidate(_draft(boundary, baseline))
    with pytest.raises(DuplicateEvolutionIdError, match="Duplicate safety boundary"):
        registry.register_safety_boundary(boundary)
    changed_boundary_payload = boundary.model_dump(
        mode="python", exclude={"content_digest"}
    )
    changed_boundary_payload["description"] = "Drifted policy text."
    changed_boundary = SafetyBoundary.model_validate(changed_boundary_payload)
    with pytest.raises(EvolutionHashDriftError, match="hash drift"):
        registry.register_safety_boundary(changed_boundary)


def test_external_candidates_are_only_admitted_inside_mutable_non_safety_paths() -> None:
    boundary = _boundary()
    component = _component(boundary)
    baseline = _baseline(boundary)
    registry = EvolvableComponentRegistry(
        admission_mode=CandidateAdmissionMode.EXTERNAL_SNAPSHOTS
    )
    registry.register_safety_boundary(boundary)
    registry.register_component(component)
    registry.register_candidate(baseline)
    draft = registry.register_candidate(_draft(boundary, baseline))
    assert draft.stage is PromotionStage.DRAFT

    unsafe_component = CandidateComponentSnapshot(
        component_id="manager.prompt",
        base_content_digest=_digest("1"),
        candidate_content_digest=_digest("3"),
        changed_paths=("/authority/lease",),
    )
    unsafe_payload = draft.model_dump(mode="python", exclude={"content_digest"})
    unsafe_payload["revision"] = 3
    unsafe_payload["parent_config_digest"] = draft.content_digest
    unsafe_payload["components"] = (unsafe_component,)
    unsafe = CandidateConfigSnapshot.model_validate(unsafe_payload)
    with pytest.raises(SafetyBoundaryViolationError, match="undeclared path"):
        registry.validate_candidate(unsafe)

    overlapping = EvolvableComponentSpec(
        component_id="unsafe.component",
        kind=EvolvableComponentKind.MANAGER_POLICY,
        base_content_digest=_digest("4"),
        mutable_paths=("/authority/lease",),
        safety_boundary_ids=(boundary.boundary_id,),
    )
    with pytest.raises(SafetyBoundaryViolationError, match="overlaps safety boundary"):
        registry.register_component(overlapping)


def test_promotion_records_are_ordered_and_never_deploy() -> None:
    record = PromotionRecord(
        candidate_id="candidate.main",
        candidate_config_digest=_digest("1"),
        from_stage=PromotionStage.DRAFT,
        to_stage=PromotionStage.OFFLINE_EVALUATED,
        decision_by="evaluation.controller",
        evidence_refs=("artifact.metric.1",),
        reason="Offline regression gates passed.",
        decided_at=NOW,
    )
    assert record.to_stage is PromotionStage.OFFLINE_EVALUATED
    with pytest.raises(ValidationError, match="Illegal promotion transition"):
        PromotionRecord(
            candidate_id="candidate.main",
            candidate_config_digest=_digest("1"),
            from_stage=PromotionStage.DRAFT,
            to_stage=PromotionStage.APPROVED,
            decision_by="automatic.evaluator",
            evidence_refs=("artifact.metric.1",),
            reason="Illegal jump.",
            decided_at=NOW,
        )


def test_evaluator_interface_records_pinned_metrics_without_auto_approval() -> None:
    metric = MetricDefinition(
        metric_id="metric.task_success",
        description="Fraction of episodes satisfying the success rubric.",
        unit="ratio",
        direction=MetricDirection.MAXIMIZE,
        aggregation=MetricAggregation.RATE,
        valid_min=0.0,
        valid_max=1.0,
        qualification_threshold=0.8,
        safety_critical=True,
    )
    evaluator_digest = _digest("a")
    record = MetricRecord(
        record_id="metric_record.1",
        metric_id=metric.metric_id,
        metric_definition_digest=metric.content_digest,
        evaluator_id="evaluator.regression",
        evaluator_digest=evaluator_digest,
        run_id="run.bowl.001",
        run_manifest_digest=_digest("b"),
        candidate_id="candidate.main",
        candidate_revision=2,
        candidate_config_digest=_digest("c"),
        value=0.9,
        sample_count=10,
        evidence_refs=("artifact.evaluation.1",),
        measured_at=NOW,
    )
    request = EvaluationRequest(
        request_id="evaluation.request.1",
        candidate_id="candidate.main",
        candidate_revision=2,
        candidate_config_digest=_digest("c"),
        run_manifest_pins=(
            ContentPin(component_id="run.bowl.001", content_digest=_digest("b")),
        ),
        metric_definition_pins=(
            ContentPin(component_id=metric.metric_id, content_digest=metric.content_digest),
        ),
        evaluator_id="evaluator.regression",
        evaluator_config_digest=evaluator_digest,
    )
    gate = EvaluationGateRecord(
        gate_id="gate.task_success",
        status=GateStatus.PASSED,
        metric_record_ids=(record.record_id,),
        evidence_refs=("artifact.evaluation.1",),
        reason="Task-success threshold passed.",
    )
    report = EvaluationReport(
        report_id="evaluation.report.1",
        request_id=request.request_id,
        request_digest=request.content_digest,
        evaluator_id="evaluator.regression",
        evaluator_digest=evaluator_digest,
        candidate_id="candidate.main",
        candidate_revision=2,
        candidate_config_digest=_digest("c"),
        metrics=(record,),
        gates=(gate,),
        recommended_stage=PromotionStage.OFFLINE_EVALUATED,
        summary="Offline evaluation passed; no promotion was executed.",
        completed_at=NOW,
    )

    class FakeEvaluator:
        evaluator_id = "evaluator.regression"
        content_digest = evaluator_digest

        def evaluate(self, evaluation_request: EvaluationRequest) -> EvaluationReport:
            assert evaluation_request == request
            return report

    evaluator = FakeEvaluator()
    assert isinstance(evaluator, Evaluator)
    assert evaluator.evaluate(request) == report
    with pytest.raises(ValidationError, match="never baseline, draft, or human approval"):
        EvaluationReport.model_validate(
            {
                **report.model_dump(exclude={"content_digest"}),
                "recommended_stage": PromotionStage.APPROVED,
            }
        )
    with pytest.raises(ValidationError):
        MetricRecord.model_validate(
            {**record.model_dump(exclude={"content_digest"}), "value": float("nan")}
        )
