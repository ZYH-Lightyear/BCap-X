from __future__ import annotations

import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from robomex.authoring.capabilities import GEOMETRY_COMPUTE, ROBOT_MOTION
from robomex.contracts import ContentPin
from robomex.core.coder import ScriptedCodePolicy
from robomex.core.sandbox import ActionBlockStatus, BlockExecutionResult, SemanticActionBlock
from robomex.core.token_budget import conservative_chat_prompt_tokens
from robomex.data import EpisodeDataPlane, core_schema_registry
from robomex.elastic import (
    ActivationSpec,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    ExecutionBudget,
    ExternalBinding,
    PortSpecV2,
    RunnerKind,
)
from robomex.orchestration.actors import (
    ActorAuthorityError,
    ActorConflictError,
    ActorLifecycle,
    ActorProfile,
    ActorRegistry,
    InvocationSpec,
)
from robomex.orchestration.bootstrap import (
    V2RuntimeDependencies,
    V2RuntimeFactory,
    actor_profile_digest,
)
from robomex.orchestration.coding_provider import (
    DEFAULT_CODING_CAPABILITY_GRANTS,
    CodingProviderContractError,
    SkillCodingAgentProvider,
)
from robomex.orchestration.episode import install_runtime_schemas
from robomex.orchestration.intent import SubgoalIntent
from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    ExecutionPolicy,
    JointPath,
    MotionPlan,
)
from robomex.runtime.events import ControlOutcome
from robomex.skills import Skill, SkillLibrary
from robomex.test.test_v2_bootstrap_manifest import _fixtures, _rebuild_manifest


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def test_state_propose_default_grants_no_executor_or_physical_capability() -> None:
    assert DEFAULT_CODING_CAPABILITY_GRANTS["state.propose"] == frozenset()


def _snapshot() -> AdmissionSnapshot:
    return AdmissionSnapshot(
        world_id="live_world",
        resource_id="arm",
        robot_revision=3,
        scene_revision=7,
        attachment_revision=2,
        config_revision=1,
        joint_names=("joint_a", "joint_b"),
        joint_positions_rad=(0.0, 0.1),
        config_digest=_digest("a"),
        collision_world_digest=_digest("b"),
        attachment_status="verified_held",
        controller_state="ready",
    )


def _motion(snapshot: AdmissionSnapshot) -> MotionPlan:
    return MotionPlan(
        plan_id="plan_from_skill",
        plan_kind="bounded_alignment",
        tcp_frame_id="panda_hand",
        planner_backend="curobo",
        robot_model_digest=_digest("c"),
        expected_snapshot=snapshot,
        max_start_deviation_rad=0.02,
        motion=JointPath(
            joint_names=snapshot.joint_names,
            positions_rad=(snapshot.joint_positions_rad, (0.2, 0.3)),
            execution_policy=ExecutionPolicy(subsample=1, timeout_s=5.0),
        ),
        possibly_affected_revisions=("robot.arm", "scene", "attachment"),
    )


class _NamespaceExecutor:
    """Persistent test sandbox that makes attempted physical calls observable."""

    def __init__(self) -> None:
        self.blocks: list[SemanticActionBlock] = []
        self.physical_calls: list[object] = []
        self.sandbox_namespace = {
            "goto_pose": lambda value: self.physical_calls.append(value),
        }

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        self.blocks.append(block)
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(block.code, self.sandbox_namespace, self.sandbox_namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001 - sandbox fixture reports failures as data
            return BlockExecutionResult(
                block=block,
                ok=False,
                status=ActionBlockStatus.FAILED,
                stdout=stdout.getvalue(),
                stderr=f"{type(exc).__name__}: {exc}",
            )
        return BlockExecutionResult(
            block=block,
            ok=True,
            status=ActionBlockStatus.SUCCEEDED,
            stdout=stdout.getvalue(),
        )


def _library(path: Path) -> SkillLibrary:
    library = SkillLibrary(path)
    library.admit(
        Skill.from_markdown(
            """---
name: author_exact_joint_path
category: motion
description: Author a bounded exact joint-path MotionPlan from an admitted snapshot.
---

Read the admitted snapshot from INPUTS. Produce a sealed joint path as a typed
proposal; never invoke a robot API.
""",
            skill_id="author_exact_joint_path",
        )
    )
    return library


def _plane(path: Path) -> tuple[EpisodeDataPlane, object]:
    schemas = install_runtime_schemas(core_schema_registry())
    plane = EpisodeDataPlane(
        path,
        episode_id="episode_coding",
        schema_registry=schemas,
        strict_schema_prefixes=("robomex.",),
    )
    plane.open_workflow("workflow_1")
    snapshot = _snapshot()
    record = plane.publish(
        workflow_id="workflow_1",
        activation_id="observe",
        attempt=1,
        port="snapshot",
        schema="robomex.admission_snapshot.v1",
        payload=snapshot.model_dump(mode="json"),
    )
    return plane, record


def _profile(*, capabilities: frozenset[str] | None = None) -> ActorProfile:
    return ActorProfile(
        profile_id="motion_author",
        provider_id="skill_coding",
        runner_kind="coding_worker",
        lifecycle=ActorLifecycle.EPHEMERAL,
        capability_ceiling=capabilities or frozenset({GEOMETRY_COMPUTE}),
        effect_ceiling=frozenset(),
        metadata={"task_kind": "motion_author"},
    )


def _spec(record, *, invocation_id: str = "command_1") -> InvocationSpec:
    return InvocationSpec(
        invocation_id=invocation_id,
        idempotency_key=invocation_id,
        objective="Author an exact bounded transport MotionPlan.",
        inputs={"snapshot": record.ref.to_mapping()},
        output_contract={"motion_plan": "robomex.motion_plan.v2"},
        requested_capabilities=frozenset({GEOMETRY_COMPUTE}),
    )


def _authoring_responses(plan: MotionPlan) -> list[str]:
    payload = plan.model_dump(mode="json")
    # The generated code must consume the mechanically resolved input rather
    # than retyping the admission snapshot from the prompt.
    payload.pop("expected_snapshot")
    code = (
        f"plan = {payload!r}\n"
        "plan['expected_snapshot'] = INPUTS['snapshot']['payload']\n"
        "NODE_RESULT = {'outputs': {'motion_plan': {'payload': plan}}}\n"
    )
    return [
        json.dumps({"tool": "use_skill", "args": {"name": "author_exact_joint_path"}}),
        json.dumps({"tool": "run_python", "args": {"intent": "author plan", "code": code}}),
        json.dumps(
            {
                "tool": "finish",
                "args": {"claim": "exact plan authored", "result_var": "NODE_RESULT"},
            }
        ),
    ]


def _geometry_authoring_responses(plan: MotionPlan) -> list[str]:
    payload = plan.model_dump(mode="json")
    payload.pop("expected_snapshot")
    code = (
        "import math\n"
        "import numpy as np\n"
        "snapshot = INPUTS.get('snapshot', {}).get('payload', {})\n"
        "joints = np.asarray(snapshot.get('joint_positions_rad', []), dtype=float)\n"
        "metrics = []\n"
        "metrics.append(float(math.sqrt(float(np.sum(joints * joints)))))\n"
        "endpoint = [round(float(value) + 0.2, 6) for value in joints]\n"
        f"plan = {payload!r}\n"
        "plan['expected_snapshot'] = snapshot\n"
        "plan['motion']['positions_rad'] = [joints.tolist(), endpoint]\n"
        "NODE_RESULT = {'outputs': {'motion_plan': {'payload': plan}}}\n"
    )
    return [
        json.dumps({"tool": "use_skill", "args": {"name": "plan_bounded_motion"}}),
        json.dumps({"tool": "run_python", "args": {"intent": "compute endpoint", "code": code}}),
        json.dumps(
            {
                "tool": "finish",
                "args": {"claim": "geometry-derived plan", "result_var": "NODE_RESULT"},
            }
        ),
    ]


def _provider(
    *,
    plane: EpisodeDataPlane | None,
    library: SkillLibrary,
    policy,
    root: Path,
    executors: list[_NamespaceExecutor],
) -> SkillCodingAgentProvider:
    def executor_factory(*_args) -> _NamespaceExecutor:
        executor = _NamespaceExecutor()
        executors.append(executor)
        return executor

    return SkillCodingAgentProvider(
        data_plane=plane,
        executor_factory=executor_factory,
        library=library,
        policy=policy,
        artifacts_root=root,
        max_turns=3,
    )


def test_skill_retrieval_and_code_produce_typed_motion_plan(tmp_path: Path) -> None:
    plane, snapshot_record = _plane(tmp_path / "episode")
    library = _library(tmp_path / "skills")
    snapshot = AdmissionSnapshot.model_validate(plane.resolve(snapshot_record.ref).payload)
    executors: list[_NamespaceExecutor] = []
    provider = _provider(
        plane=plane,
        library=library,
        policy=ScriptedCodePolicy(_authoring_responses(_motion(snapshot))),
        root=tmp_path / "provider",
        executors=executors,
    )
    registry = ActorRegistry(
        {"skill_coding": provider},
        namespace_root="episode_coding",
        workspace_root=tmp_path / "actors",
    )
    handle = registry.spawn(_profile(), actor_id="motion_author_1")

    result = handle.invoke(_spec(snapshot_record))

    assert result.outcome is ControlOutcome.SUCCESS
    assert len(result.artifacts) == 1
    emission = result.artifacts[0]
    assert emission.port == "motion_plan"
    assert emission.schema_id == "robomex.motion_plan.v2"
    plan = MotionPlan.model_validate(emission.payload)
    assert plan.expected_snapshot == snapshot
    assert plan.motion.positions_rad[-1] == (0.2, 0.3)
    assert emission.lineage == (snapshot_record.ref,)
    assert handle.runtime.audits[-1].loaded_skill_ids == ("author_exact_joint_path",)
    assert not executors[0].physical_calls


def test_v2_motion_plan_capability_runs_real_builtin_skill_geometry_code(
    tmp_path: Path,
) -> None:
    plane, snapshot_record = _plane(tmp_path / "episode")
    snapshot = AdmissionSnapshot.model_validate(plane.resolve(snapshot_record.ref).payload)
    builtin_root = Path(__file__).resolve().parents[1] / "skills" / "builtin"
    library = SkillLibrary(tmp_path / "builtin_skill_view")
    library.admit(Skill.from_dir(builtin_root / "motion" / "plan_bounded_motion"))
    executors: list[_NamespaceExecutor] = []
    provider = _provider(
        plane=plane,
        library=library,
        policy=ScriptedCodePolicy(_geometry_authoring_responses(_motion(snapshot))),
        root=tmp_path / "provider",
        executors=executors,
    )
    profile = _profile(capabilities=frozenset({"motion.plan"}))
    registry = ActorRegistry({"skill_coding": provider}, workspace_root=tmp_path / "actors")
    handle = registry.spawn(profile, actor_id="semantic_motion_author")
    spec = InvocationSpec(
        invocation_id="semantic_motion_command",
        objective="Use current joints to compute a bounded exact plan.",
        inputs={"snapshot": snapshot_record.ref.to_mapping()},
        output_contract={"motion_plan": "robomex.motion_plan.v2"},
        requested_capabilities=frozenset({"motion.plan"}),
    )

    result = handle.invoke(spec)

    assert result.outcome is ControlOutcome.SUCCESS
    plan = MotionPlan.model_validate(result.artifacts[0].payload)
    assert plan.motion.positions_rad == ((0.0, 0.1), (0.2, 0.3))
    assert handle.runtime.audits[-1].loaded_skill_ids == ("plan_bounded_motion",)
    assert not executors[0].physical_calls


def test_v2_factory_late_binds_real_provider_and_episode_publishes_plan(
    tmp_path: Path,
) -> None:
    config, manifest, original = _fixtures(tmp_path)
    snapshot = _snapshot()
    library = _library(tmp_path / "skills")
    executors: list[_NamespaceExecutor] = []
    provider = _provider(
        plane=None,
        library=library,
        policy=ScriptedCodePolicy(_authoring_responses(_motion(snapshot))),
        root=config.episode_root / "coding_provider",
        executors=executors,
    )
    profile = ActorProfile(
        profile_id="actor.worker",
        provider_id="skill_coding",
        runner_kind="coding_worker",
        lifecycle=ActorLifecycle.EPHEMERAL,
        capability_ceiling=frozenset({"motion.plan"}),
        effect_ceiling=frozenset(),
        metadata={"task_kind": "motion_author"},
    )
    graph = ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="coding_provider_e2e",
            entry_activation="author_plan",
            terminal_activations=("author_plan",),
            activations=(
                ActivationSpec(
                    activation_id="author_plan",
                    runner_kind=RunnerKind.CODING_WORKER,
                    runner_ref="worker",
                    inputs=(
                        PortSpecV2(
                            name="snapshot",
                            schema_id="robomex.admission_snapshot.v1",
                        ),
                    ),
                    outputs=(
                        PortSpecV2(
                            name="motion_plan",
                            schema_id="robomex.motion_plan.v2",
                        ),
                    ),
                    bindings=(
                        ExternalBinding(
                            input_port="snapshot",
                            ref="snapshot_ref",
                            schema_id="robomex.admission_snapshot.v1",
                        ),
                    ),
                    required_capabilities=("motion.plan",),
                    estimated_budget=ExecutionBudget(
                        model_calls=3,
                        tokens=200_000,
                        wall_time_ms=5_000,
                    ),
                    params={"objective": "Author an exact bounded transport plan."},
                ),
            ),
        )
    )
    manifest = _rebuild_manifest(
        manifest,
        actor_profile_pins=(
            ContentPin(
                component_id=profile.profile_id,
                revision=1,
                content_digest=actor_profile_digest(profile),
            ),
        ),
        metadata={
            "allowed_initial_graph_digests": [f"sha256:{graph.digest}"],
        },
        budgets=manifest.budgets.model_copy(update={"max_tokens": 200_000}),
    )
    dependencies = V2RuntimeDependencies(
        contract_catalog=original.contract_catalog,
        actor_providers={"skill_coding": provider},
        actor_profiles={"worker": profile},
        action_backends=original.action_backends,
        observation_backends=original.observation_backends,
        shadow_backends=original.shadow_backends,
        schema_registry=original.schema_registry,
    )

    application = V2RuntimeFactory().build(
        config=config,
        manifest=manifest,
        dependencies=dependencies,
    )

    assert provider.data_plane is application.episode.data_plane
    plane = application.episode.data_plane
    plane.open_workflow("seed_inputs")
    snapshot_record = plane.publish(
        workflow_id="seed_inputs",
        activation_id="observe",
        attempt=1,
        port="snapshot",
        schema="robomex.admission_snapshot.v1",
        payload=snapshot.model_dump(mode="json"),
    )
    plane.close_workflow("seed_inputs")
    workflow_id = application.episode.open_workflow(
        workflow_id="factory_coding_workflow",
        intent=SubgoalIntent(
            intent_id="author_transport",
            instruction="Author a bounded transport plan.",
            success_rubric="A schema-valid exact MotionPlan is published.",
        ),
        graph=graph,
        external_refs={"snapshot_ref": snapshot_record.ref},
    )

    terminal = application.episode.run_until_terminal(workflow_id)

    assert terminal.status.value == "succeeded"
    published = next(
        record
        for record in plane.artifacts
        if record.workflow_id == workflow_id and record.port == "motion_plan"
    )
    plan = MotionPlan.model_validate(plane.resolve(published.ref).payload)
    assert plan.expected_snapshot == snapshot
    assert published.lineage == (snapshot_record.ref,)
    assert not executors[0].physical_calls


@pytest.mark.parametrize(
    "bad_code",
    [
        "goto_pose([0.1, 0.2, 0.3])",
        "vars()['goto_pose']([0.1, 0.2, 0.3])",
        "sorted([[0.1, 0.2, 0.3]], key=goto_pose)",
        "import unaudited_skill_sidecar",
    ],
)
def test_generated_physical_or_dynamic_calls_never_reach_executor(
    tmp_path: Path, bad_code: str
) -> None:
    plane, _ = _plane(tmp_path / "episode")
    executors: list[_NamespaceExecutor] = []
    policy = ScriptedCodePolicy(
        [
            json.dumps({"tool": "run_python", "args": {"intent": "unsafe", "code": bad_code}}),
            json.dumps(
                {
                    "tool": "finish",
                    "args": {
                        "claim": "unsafe request rejected",
                        "result": {"ok": False, "failure_kind": "execution_error"},
                    },
                }
            ),
        ]
    )
    provider = _provider(
        plane=plane,
        library=_library(tmp_path / "skills"),
        policy=policy,
        root=tmp_path / "provider",
        executors=executors,
    )
    profile = _profile()
    registry = ActorRegistry({"skill_coding": provider}, workspace_root=tmp_path / "actors")
    handle = registry.spawn(profile, actor_id="unsafe_author")
    spec = InvocationSpec(
        invocation_id="unsafe_command",
        objective="Attempt an unsafe action.",
        output_contract={},
        requested_capabilities=frozenset({GEOMETRY_COMPUTE}),
    )

    result = handle.invoke(spec)

    assert result.outcome is ControlOutcome.FAILED
    assert not executors[0].physical_calls
    assert all(bad_code != block.code for block in executors[0].blocks)


def test_profile_and_invocation_physical_authority_fail_closed(tmp_path: Path) -> None:
    plane, _ = _plane(tmp_path / "episode")
    provider = _provider(
        plane=plane,
        library=_library(tmp_path / "skills"),
        policy=ScriptedCodePolicy([]),
        root=tmp_path / "provider",
        executors=[],
    )
    registry = ActorRegistry({"skill_coding": provider}, workspace_root=tmp_path / "actors")

    with pytest.raises(ActorAuthorityError, match="robot_motion"):
        registry.spawn(
            _profile(capabilities=frozenset({GEOMETRY_COMPUTE, ROBOT_MOTION})),
            actor_id="overpowered_author",
        )

    handle = registry.spawn(_profile(), actor_id="proposal_author")
    with pytest.raises(ActorAuthorityError, match="effects"):
        # Direct provider invocation proves the provider fails closed even if a
        # caller bypasses ActorHandle's first authority check.
        provider.invoke(
            handle.runtime,
            InvocationSpec(
                invocation_id="physical_effect",
                objective="Write a proposal.",
                requested_capabilities=frozenset({GEOMETRY_COMPUTE}),
                requested_effects=frozenset({"authoritative_world"}),
            ),
        )


def test_committed_command_replays_after_provider_restart_without_model_call(
    tmp_path: Path,
) -> None:
    plane, snapshot_record = _plane(tmp_path / "episode")
    library = _library(tmp_path / "skills")
    snapshot = AdmissionSnapshot.model_validate(plane.resolve(snapshot_record.ref).payload)
    provider_root = tmp_path / "provider"
    first_policy = ScriptedCodePolicy(_authoring_responses(_motion(snapshot)))
    first_executors: list[_NamespaceExecutor] = []
    first = _provider(
        plane=plane,
        library=library,
        policy=first_policy,
        root=provider_root,
        executors=first_executors,
    )
    registry = ActorRegistry({"skill_coding": first}, workspace_root=tmp_path / "actors_first")
    expected = registry.spawn(_profile(), actor_id="restart_author").invoke(_spec(snapshot_record))

    class _NoCallPolicy:
        calls = 0

        def complete(self, _prompt):
            self.calls += 1
            raise AssertionError("committed invocation must not call the model")

    replay_policy = _NoCallPolicy()
    replay_executors: list[_NamespaceExecutor] = []
    restarted = _provider(
        plane=plane,
        library=library,
        policy=replay_policy,
        root=provider_root,
        executors=replay_executors,
    )
    replay_registry = ActorRegistry(
        {"skill_coding": restarted}, workspace_root=tmp_path / "actors_restarted"
    )
    handle = replay_registry.spawn(_profile(), actor_id="restart_author")

    actual = handle.invoke(_spec(snapshot_record))

    assert actual == expected
    assert replay_policy.calls == 0
    assert handle.runtime.audits[-1].replayed
    assert handle.runtime.audits[-1].loaded_skill_ids == ("author_exact_joint_path",)


def test_same_durable_key_cannot_be_rebound_after_restart(tmp_path: Path) -> None:
    plane, snapshot_record = _plane(tmp_path / "episode")
    library = _library(tmp_path / "skills")
    snapshot = AdmissionSnapshot.model_validate(plane.resolve(snapshot_record.ref).payload)
    root = tmp_path / "provider"
    first = _provider(
        plane=plane,
        library=library,
        policy=ScriptedCodePolicy(_authoring_responses(_motion(snapshot))),
        root=root,
        executors=[],
    )
    registry = ActorRegistry({"skill_coding": first}, workspace_root=tmp_path / "actors")
    registry.spawn(_profile(), actor_id="stable_author").invoke(_spec(snapshot_record))

    restarted = _provider(
        plane=plane,
        library=library,
        policy=ScriptedCodePolicy([]),
        root=root,
        executors=[],
    )
    replay_registry = ActorRegistry(
        {"skill_coding": restarted}, workspace_root=tmp_path / "actors_replay"
    )
    changed = _spec(snapshot_record)
    changed = InvocationSpec(
        invocation_id=changed.invocation_id,
        idempotency_key=changed.idempotency_key,
        objective="A different objective under the same durable key.",
        inputs=changed.inputs,
        output_contract=changed.output_contract,
        requested_capabilities=changed.requested_capabilities,
    )

    with pytest.raises(ActorConflictError, match="rebound"):
        replay_registry.spawn(_profile(), actor_id="stable_author").invoke(changed)


def test_pending_coding_call_is_fail_closed_after_model_boundary_crash(
    tmp_path: Path,
) -> None:
    plane, snapshot_record = _plane(tmp_path / "episode")
    library = _library(tmp_path / "skills")
    root = tmp_path / "provider"

    class CrashPolicy:
        calls = 0

        def complete(self, _prompt):
            raise AssertionError("bounded completion must be used")

        def complete_bounded(self, _prompt, *, max_tokens, deadline_monotonic_s):
            assert max_tokens > 0
            assert deadline_monotonic_s > 0
            self.calls += 1
            raise KeyboardInterrupt("synthetic crash after model entry")

    crashing_policy = CrashPolicy()
    first = _provider(
        plane=plane,
        library=library,
        policy=crashing_policy,
        root=root,
        executors=[],
    )
    first_handle = ActorRegistry(
        {"skill_coding": first}, workspace_root=tmp_path / "actors-first"
    ).spawn(_profile(), actor_id="crash-author")

    with pytest.raises(KeyboardInterrupt, match="synthetic crash"):
        first_handle.invoke(_spec(snapshot_record))
    assert crashing_policy.calls == 1

    class NoCallPolicy:
        calls = 0

        def complete(self, _prompt):
            self.calls += 1
            raise AssertionError("pending invocation must never re-enter the model")

    no_call = NoCallPolicy()
    restarted = _provider(
        plane=plane,
        library=library,
        policy=no_call,
        root=root,
        executors=[],
    )
    replay_handle = ActorRegistry(
        {"skill_coding": restarted}, workspace_root=tmp_path / "actors-restarted"
    ).spawn(_profile(), actor_id="crash-author")

    with pytest.raises(CodingProviderContractError, match="uncertain model boundary"):
        replay_handle.invoke(_spec(snapshot_record))
    assert no_call.calls == 0


def test_coding_multiturn_output_ceilings_share_one_typed_token_grant(
    tmp_path: Path,
) -> None:
    plane, snapshot_record = _plane(tmp_path / "episode")
    library = _library(tmp_path / "skills")
    snapshot = AdmissionSnapshot.model_validate(plane.resolve(snapshot_record.ref).payload)

    class RecordingPolicy(ScriptedCodePolicy):
        def __init__(self, responses):
            super().__init__(responses)
            self.ceilings: list[int] = []
            self.prompt_tokens: list[int] = []

        def complete_bounded(self, prompt, *, max_tokens, deadline_monotonic_s):
            self.ceilings.append(max_tokens)
            self.prompt_tokens.append(conservative_chat_prompt_tokens(prompt))
            return super().complete_bounded(
                prompt,
                max_tokens=max_tokens,
                deadline_monotonic_s=deadline_monotonic_s,
            )

    policy = RecordingPolicy(_authoring_responses(_motion(snapshot)))
    provider = _provider(
        plane=plane,
        library=library,
        policy=policy,
        root=tmp_path / "provider",
        executors=[],
    )
    handle = ActorRegistry({"skill_coding": provider}, workspace_root=tmp_path / "actors").spawn(
        _profile(), actor_id="bounded-author"
    )
    spec = replace(
        _spec(snapshot_record),
        budget={
            "model_calls": 3,
            "tokens": 262_144,
            "wall_time_ms": 5_000,
        },
    )

    result = handle.invoke(spec)

    assert result.outcome is ControlOutcome.SUCCESS
    assert len(policy.ceilings) == 3
    assert policy.prompt_tokens[1] > policy.prompt_tokens[0]
    assert policy.prompt_tokens[2] > policy.prompt_tokens[1]
    assert sum(policy.ceilings) + sum(policy.prompt_tokens) <= spec.budget["tokens"]
