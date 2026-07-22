from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from robomex.data import ResolvedArtifactRef
from robomex.elastic import (
    ACTION_SNAPSHOT_RUNNER_REF,
    ACTION_SNAPSHOT_SCHEMA_ID,
    ActivationSpec,
    ArtifactBinding,
    EffectScope,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    ExecutionBudget,
    PortSpecV2,
    RunnerKind,
    TransitionSpec,
)
from robomex.orchestration.actors import (
    ActorProfile,
    ActorRegistry,
    InMemoryAgentProvider,
)
from robomex.orchestration.episode import (
    ActivationExecutionResult,
    ArtifactEmission,
    EpisodeRuntime,
)
from robomex.orchestration.intent import SubgoalIntent
from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    FeasibilityStatus,
    MotionPlan,
)
from robomex.runtime.authority import build_feasibility_certificate
from robomex.runtime.events import ControlOutcome, NodeOutcomeEvent
from robomex.test.test_action_protocol_v2 import FakeBackend, _motion, _snapshot


class _CountingBackend(FakeBackend):
    def __init__(
        self,
        snapshot: AdmissionSnapshot,
        *,
        response_override: AdmissionSnapshot | None = None,
    ) -> None:
        super().__init__(snapshot)
        self.snapshot_calls: list[tuple[str, str]] = []
        self.response_override = response_override

    def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot:
        self.snapshot_calls.append((world_id, resource_id))
        if self.response_override is not None:
            return self.response_override
        return super().snapshot(world_id, resource_id)


class _Checker:
    def certify(self, spec, snapshot):
        return build_feasibility_certificate(
            spec=spec,
            snapshot=snapshot,
            checker_id="snapshot-e2e-checker",
            checks={
                "exact_joint_path_interface": FeasibilityStatus.PASS,
                "controller_admissible": FeasibilityStatus.PASS,
                "resource_binding": FeasibilityStatus.PASS,
                "joint_order": FeasibilityStatus.PASS,
                "collision": FeasibilityStatus.PASS,
                "joint_limits": FeasibilityStatus.PASS,
                "robot_model": FeasibilityStatus.PASS,
            },
        )


def _intent() -> SubgoalIntent:
    return SubgoalIntent(
        intent_id="fresh-plan",
        instruction="capture, author, and execute one sealed action",
        success_rubric="the action is admitted against its fresh snapshot",
    )


def _snapshot_node(
    *,
    world_id: str = "live-world",
    resource_id: str = "arm",
    params: dict[str, object] | None = None,
) -> ActivationSpec:
    return ActivationSpec(
        activation_id="capture",
        runner_kind=RunnerKind.ACTION_SNAPSHOT,
        runner_ref=ACTION_SNAPSHOT_RUNNER_REF,
        authority_world_id=world_id,
        authoritative_resource=resource_id,
        outputs=(
            PortSpecV2(name="snapshot", schema_id=ACTION_SNAPSHOT_SCHEMA_ID),
        ),
        estimated_budget=ExecutionBudget(),
        params=params or {},
    )


def _capture_only_graph() -> object:
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="capture-only",
            entry_activation="capture",
            terminal_activations=("capture",),
            activations=(_snapshot_node(),),
        )
    )


def _capture_author_execute_graph() -> object:
    capture = _snapshot_node()
    author = ActivationSpec(
        activation_id="author",
        runner_kind=RunnerKind.CODING_WORKER,
        runner_ref="test.snapshot_author",
        inputs=(
            PortSpecV2(name="snapshot", schema_id=ACTION_SNAPSHOT_SCHEMA_ID),
        ),
        outputs=(PortSpecV2(name="action_spec", schema_id="robomex.motion_plan.v2"),),
        bindings=(
            ArtifactBinding(
                input_port="snapshot",
                source_activation="capture",
                source_port="snapshot",
            ),
        ),
        required_capabilities=("motion.plan",),
        estimated_budget=ExecutionBudget(model_calls=1, tokens=100),
    )
    execute = ActivationSpec(
        activation_id="execute",
        runner_kind=RunnerKind.SYSTEM_ACTION,
        runner_ref="runtime.sealed_action",
        effect_scope=EffectScope.AUTHORITATIVE_WORLD,
        authority_world_id="live-world",
        authoritative_resource="arm",
        inputs=(
            PortSpecV2(name="action_spec", schema_id="robomex.motion_plan.v2"),
        ),
        outputs=(
            PortSpecV2(
                name="receipt", schema_id="robomex.execution_receipt.v2"
            ),
        ),
        bindings=(
            ArtifactBinding(
                input_port="action_spec",
                source_activation="author",
                source_port="action_spec",
            ),
        ),
        estimated_budget=ExecutionBudget(authoritative_actions=1),
    )
    return ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="capture-author-execute",
            entry_activation="capture",
            terminal_activations=("execute",),
            activations=(capture, author, execute),
            transitions=(
                TransitionSpec(
                    source="capture", outcome=ControlOutcome.SUCCESS, target="author"
                ),
                TransitionSpec(
                    source="author", outcome=ControlOutcome.SUCCESS, target="execute"
                ),
            ),
        )
    )


def _empty_actors(root: Path) -> ActorRegistry:
    return ActorRegistry(
        {"in_memory": InMemoryAgentProvider()},
        namespace_root="snapshot",
        workspace_root=root / "actors",
    )


def test_fresh_snapshot_is_the_only_physical_input_to_coding_author_then_action(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(captured_at=datetime.now(timezone.utc))  # noqa: UP017
    backend = _CountingBackend(snapshot)
    holder: dict[str, EpisodeRuntime] = {}
    authored_from: list[ResolvedArtifactRef] = []

    def handler(_profile, invocation, _isolation):
        ref = ResolvedArtifactRef.from_any(invocation.inputs["snapshot"])
        authored_from.append(ref)
        payload = holder["runtime"].data_plane.resolve(ref).payload
        admitted_snapshot = AdmissionSnapshot.model_validate(payload)
        plan = _motion(admitted_snapshot, plan_id="fresh-snapshot-plan")
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                ArtifactEmission(
                    port="action_spec",
                    schema_id=plan.schema_version,
                    payload=plan.model_dump(mode="json"),
                    lineage=(ref,),
                ),
            ),
        )

    actors = ActorRegistry(
        {"in_memory": InMemoryAgentProvider(handler)},
        namespace_root="snapshot-e2e",
        workspace_root=tmp_path / "actors",
    )
    runtime = EpisodeRuntime(
        episode_id="snapshot-e2e",
        episode_root=tmp_path / "episode",
        actors=actors,
        action_backends={(snapshot.world_id, snapshot.resource_id): backend},
        feasibility_checkers={(snapshot.world_id, snapshot.resource_id): _Checker()},
    )
    holder["runtime"] = runtime
    runtime.register_actor_profile(
        "test.snapshot_author",
        ActorProfile(
            profile_id="snapshot-author",
            runner_kind=RunnerKind.CODING_WORKER.value,
            capability_ceiling=frozenset({"motion.plan"}),
        ),
    )
    workflow_id = runtime.open_workflow(
        workflow_id="workflow",
        intent=_intent(),
        graph=_capture_author_execute_graph(),
    )

    terminal = runtime.run_until_terminal(workflow_id)

    assert terminal.status.value == "succeeded"
    assert len(authored_from) == 1
    capture_artifact = runtime.data_plane.resolve(authored_from[0])
    captured = AdmissionSnapshot.model_validate(capture_artifact.payload)
    plan_record = next(
        record
        for record in runtime.data_plane.artifacts
        if record.activation_id == "author" and record.port == "action_spec"
    )
    plan = MotionPlan.model_validate(runtime.data_plane.resolve(plan_record.ref).payload)
    assert plan.expected_snapshot == captured == snapshot
    assert plan_record.lineage == (authored_from[0],)
    assert backend.calls and backend.calls[0][0] == "execute_joint_path"
    assert backend.snapshot_calls[0] == (snapshot.world_id, snapshot.resource_id)


def test_action_snapshot_recovery_reuses_published_attempt_without_recapture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(captured_at=datetime.now(timezone.utc))  # noqa: UP017
    backend = _CountingBackend(snapshot)
    graph = _capture_only_graph()
    first = EpisodeRuntime(
        episode_id="snapshot-recovery",
        episode_root=tmp_path / "episode",
        actors=_empty_actors(tmp_path),
        action_backends={(snapshot.world_id, snapshot.resource_id): backend},
    )
    first.open_workflow(workflow_id="workflow", intent=_intent(), graph=graph)
    scheduler = first._workflow("workflow").scheduler  # noqa: SLF001
    original_on_event = scheduler.on_event

    class _Crash(BaseException):
        pass

    def crash_after_publication(event):
        if isinstance(event, NodeOutcomeEvent) and event.source == "action_snapshot_runner":
            raise _Crash
        return original_on_event(event)

    monkeypatch.setattr(scheduler, "on_event", crash_after_publication)
    with pytest.raises(_Crash):
        first.execute_ready("workflow")
    assert len(backend.snapshot_calls) == 1
    published = [
        record
        for record in first.data_plane.artifacts
        if record.activation_id == "capture" and record.port == "snapshot"
    ]
    assert len(published) == 1

    recovered = EpisodeRuntime(
        episode_id="snapshot-recovery",
        episode_root=tmp_path / "episode",
        actors=_empty_actors(tmp_path / "recovered"),
        action_backends={(snapshot.world_id, snapshot.resource_id): backend},
    )
    recovered.recover_workflow("workflow")
    terminal = recovered.run_until_terminal("workflow")

    assert terminal.status.value == "succeeded"
    assert len(backend.snapshot_calls) == 1
    assert len(
        [
            record
            for record in recovered.data_plane.artifacts
            if record.activation_id == "capture" and record.port == "snapshot"
        ]
    ) == 1


def test_action_snapshot_missing_backend_and_world_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    graph = _capture_only_graph()
    missing = EpisodeRuntime(
        episode_id="snapshot-missing",
        episode_root=tmp_path / "missing",
        actors=_empty_actors(tmp_path / "missing"),
    )
    workflow_id = missing.open_workflow(intent=_intent(), graph=graph)
    terminal = missing.run_until_terminal(workflow_id)
    assert terminal.status.value == "failed"
    assert terminal.terminal_outcome is ControlOutcome.INFEASIBLE

    expected = _snapshot(captured_at=datetime.now(timezone.utc))  # noqa: UP017
    wrong = expected.model_copy(update={"world_id": "another-world"})
    backend = _CountingBackend(expected, response_override=wrong)
    mismatched = EpisodeRuntime(
        episode_id="snapshot-mismatch",
        episode_root=tmp_path / "mismatch",
        actors=_empty_actors(tmp_path / "mismatch"),
        action_backends={(expected.world_id, expected.resource_id): backend},
    )
    workflow_id = mismatched.open_workflow(intent=_intent(), graph=graph)
    terminal = mismatched.run_until_terminal(workflow_id)
    assert terminal.status.value == "failed"
    assert terminal.terminal_outcome is ControlOutcome.STALE_INPUT
    assert not mismatched.data_plane.artifacts


def test_params_world_id_cannot_be_used_as_authority_channel() -> None:
    with pytest.raises(ValidationError, match="params.world_id"):
        _snapshot_node(params={"world_id": "injected-world"})

    with pytest.raises(ValidationError, match="authority_world_id"):
        ActivationSpec(
            activation_id="execute",
            runner_kind=RunnerKind.SYSTEM_ACTION,
            runner_ref="runtime.sealed_action",
            effect_scope=EffectScope.AUTHORITATIVE_WORLD,
            authoritative_resource="arm",
            params={"world_id": "injected-world"},
        )
