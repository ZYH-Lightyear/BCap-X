from __future__ import annotations

import json

import pytest

from robomex.authoring.adapters import SubAgentFactory
from robomex.authoring.artifacts import TypedArtifact
from robomex.authoring.graph import SubgoalGraphCompiler
from robomex.authoring.graph_executor import SubgoalGraphExecutor
from robomex.authoring.result import (
    AuthoringCost,
    AuthoringNodeResult,
    AuthoringStatus,
    NodeStatus,
    SubgoalAuthoringContext,
    SubgoalOutcome,
    VerificationStatus,
)
from robomex.authoring.swarm_creator import SubgoalSwarmManager
from robomex.authoring.swarm_spec import SpecialistSpec
from robomex.core.coder import parse_action
from robomex.core.context import EvidencePacket
from robomex.core.sandbox import (
    ActionBlockStatus,
    BlockExecutionResult,
    MotionLeaseGuard,
    RuntimeSafetyState,
    SemanticActionBlock,
)
from robomex.dysc.contracts import load_skill_contracts
from robomex.skills import SkillLibrary, load_builtin_skills


class _Policy:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def complete(self, prompt) -> str:
        return self.responses.pop(0)


class _Executor:
    def __init__(self) -> None:
        self.blocks = []

    def run_block(self, block):
        self.blocks.append(block)
        return BlockExecutionResult(
            block=block,
            ok=True,
            status=ActionBlockStatus.SUCCEEDED,
            observation={"agentview": {}},
        )


def _library(tmp_path) -> tuple[SkillLibrary, dict]:
    library = SkillLibrary(tmp_path / "library")
    for skill in load_builtin_skills():
        library.admit(skill, source="builtin")
    return library, load_skill_contracts(library.root)


def _pick_graph() -> dict:
    return {
        "entry": "ground_target",
        "success_node": "verify_pick",
        "max_recoveries": 2,
        "nodes": [
            {"id": "ground_target", "skill": "segment_object"},
            {
                "id": "plan_grasp",
                "skill": "grasp_graspnet",
                "inputs": {
                    "object_mask": {"$ref": "ground_target.object_mask"},
                    "object_points": {"$ref": "ground_target.object_points"},
                },
            },
            {
                "id": "plan_motion",
                "skill": "plan_bounded_motion",
                "inputs": {"affordance": {"$ref": "plan_grasp.grasp_affordance"}},
            },
            {
                "id": "execute_pick",
                "skill": "grasp_object",
                "inputs": {"trajectory": {"$ref": "plan_motion.trajectory"}},
            },
            {
                "id": "verify_pick",
                "skill": "verify_grasp_and_lift_via_robot_state",
                "checkpoint": "hard",
                "inputs": {
                    "execution_evidence": {
                        "$ref": "execute_pick.execution_evidence"
                    }
                },
            },
        ],
        "edges": [
            {"from": "ground_target", "to": "plan_grasp", "on": "success"},
            {"from": "plan_grasp", "to": "plan_motion", "on": "success"},
            {"from": "plan_motion", "to": "execute_pick", "on": "success"},
            {"from": "execute_pick", "to": "verify_pick", "on": "success"},
            {
                "from": "ground_target",
                "to": "ground_target",
                "on": "stale_observation",
            },
            {"from": "plan_grasp", "to": "ground_target", "on": "wrong_grounding"},
            {"from": "plan_grasp", "to": "ground_target", "on": "stale_observation"},
            {"from": "plan_motion", "to": "plan_grasp", "on": "infeasible"},
            {"from": "verify_pick", "to": "ground_target", "on": "failed_grasp"},
            {"from": "verify_pick", "to": "ground_target", "on": "failed"},
        ],
    }


def _place_graph() -> dict:
    return {
        "entry": "ground_destination",
        "success_node": "verify_place",
        "max_recoveries": 2,
        "nodes": [
            {"id": "ground_destination", "skill": "segment_object"},
            {
                "id": "plan_placement",
                "skill": "find_placement",
                "inputs": {
                    "target_points": {"$ref": "ground_destination.object_points"}
                },
            },
            {
                "id": "plan_motion",
                "skill": "plan_bounded_motion",
                "inputs": {
                    "affordance": {"$ref": "plan_placement.placement_affordance"}
                },
            },
            {
                "id": "execute_place",
                "skill": "release_at",
                "inputs": {"trajectory": {"$ref": "plan_motion.trajectory"}},
            },
            {
                "id": "verify_place",
                "skill": "verify_placement",
                "checkpoint": "hard",
                "inputs": {
                    "execution_evidence": {
                        "$ref": "execute_place.execution_evidence"
                    }
                },
            },
        ],
        "edges": [
            {"from": "ground_destination", "to": "plan_placement", "on": "success"},
            {"from": "plan_placement", "to": "plan_motion", "on": "success"},
            {"from": "plan_motion", "to": "execute_place", "on": "success"},
            {"from": "execute_place", "to": "verify_place", "on": "success"},
            {
                "from": "ground_destination",
                "to": "ground_destination",
                "on": "stale_observation",
            },
            {
                "from": "plan_placement",
                "to": "ground_destination",
                "on": "wrong_grounding",
            },
            {
                "from": "plan_placement",
                "to": "ground_destination",
                "on": "stale_observation",
            },
            {"from": "plan_motion", "to": "plan_placement", "on": "infeasible"},
            {"from": "verify_place", "to": "ground_destination", "on": "failed"},
        ],
    }


def _compile_graph(contracts: dict, task_skill: str = "pick_object"):
    draft = _pick_graph() if task_skill == "pick_object" else _place_graph()
    return SubgoalGraphCompiler(contracts).compile(
        draft,
        task_skill=task_skill,
        goal="complete manipulation",
        postcondition="relation holds",
    )


def _canonical_payload(schema: str, role: str) -> dict:
    """Minimal payloads satisfying the canonical specs enforced at publish time."""

    if schema == "robomex.trajectory.v1":
        return {
            "feasible": True,
            "waypoints": [
                {
                    "name": "pregrasp",
                    "phase": "pregrasp",
                    "position_xyz": [0.7, 0.0, 0.2],
                    "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
                    "gripper": "open",
                },
                {
                    "name": "lift",
                    "phase": "lift",
                    "position_xyz": [0.7, 0.0, 0.3],
                    "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
                    "gripper": "close",
                },
            ],
        }
    if schema == "robomex.affordance.v1":
        return {
            "position": [0.7, 0.0, 0.07],
            "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
        }
    if schema == "robomex.execution_evidence.v1":
        return {
            "primitives": [{"name": "goto_pose", "status": "succeeded"}],
            "all_primitives_ok": True,
        }
    return {"stage": role}


class _ContractFactory:
    def __init__(self, state: RuntimeSafetyState, *, fail_first_verify: bool = False) -> None:
        self.state = state
        self.fail_first_verify = fail_first_verify
        self.verify_calls = 0
        self.seen: list[SpecialistSpec] = []

    def run(self, spec, **kwargs):
        self.seen.append(spec)
        if spec.changes_world:
            self.state.observation_epoch += 1
        if spec.verifier:
            self.verify_calls += 1
            if self.fail_first_verify and self.verify_calls == 1:
                return AuthoringNodeResult(
                    node_id=spec.agent_id,
                    status=NodeStatus.FAILED,
                    evidence=EvidencePacket.from_any(
                        {
                            "claim": "grasp miss",
                            "verdict": {"status": "fail", "reason": "object not held"},
                        },
                        default_source=spec.agent_id,
                    ),
                    verification=VerificationStatus.FAILED,
                )
        outputs = tuple(
            TypedArtifact(
                port=port.name,
                schema=port.schema,
                producer=spec.agent_id,
                payload=_canonical_payload(port.schema, spec.role),
                frame=port.frame,
                observation_epoch=self.state.observation_epoch,
            )
            for port in spec.outputs
        )
        evidence = (
            EvidencePacket.from_any(
                {"verdict": {"status": "pass", "reason": "fresh evidence"}},
                default_source=spec.agent_id,
            )
            if spec.verifier
            else EvidencePacket()
        )
        return AuthoringNodeResult(
            node_id=spec.agent_id,
            status=NodeStatus.SUCCEEDED,
            outputs=outputs,
            evidence=evidence,
            verification=VerificationStatus.PASSED
            if spec.verifier
            else VerificationStatus.NOT_RUN,
            cost=AuthoringCost(llm_calls=1, action_turns=1),
        )


def test_manager_catalog_exposes_budget_as_declared_data(tmp_path) -> None:
    """M1.5 Fix E: skill budgets and recommended_min_actions are data the
    Manager sees; they are never re-authored at runtime."""

    library, contracts = _library(tmp_path)
    state = RuntimeSafetyState()
    manager = SubgoalSwarmManager(
        policy=_Policy([]),
        library=library,
        factory=_ContractFactory(state),  # type: ignore[arg-type]
        capability_ceiling=frozenset(
            {
                "perception_read",
                "geometry_compute",
                "artifact_write",
                "robot_motion",
                "gripper_control",
                "object_manipulation",
            }
        ),
        safety_state=state,
        contracts=contracts,
    )

    catalog = manager._specialist_catalog()
    verify = catalog["verify_grasp_and_lift_via_robot_state"]
    assert verify["budget"] == {"max_turns": 4}
    assert verify["recommended_min_actions"] == 2


def test_skill_frontmatter_parses_recommended_min_actions() -> None:
    from robomex.skills import Skill

    skill = Skill.from_markdown(
        "---\nname: v\ncategory: verification\nrecommended_min_actions: 3\n---\n\nBody.",
        skill_id="v",
    )
    assert skill.recommended_min_actions == 3
    plain = Skill.from_markdown("---\nname: p\ncategory: motion\n---\n\nBody.", skill_id="p")
    assert plain.recommended_min_actions == 0


def test_dynamic_graph_compiles_strict_specialist_contracts(tmp_path) -> None:
    _, contracts = _library(tmp_path)
    graph = _compile_graph(contracts)

    assert [node.specialist.role for node in graph.nodes] == [
        "grounding",
        "affordance",
        "motion_planner",
        "action_executor",
        "verifier",
    ]
    assert graph.node_map["execute_pick"].specialist.requested_capabilities == frozenset(
        {"robot_motion", "gripper_control", "object_manipulation", "artifact_write"}
    )
    for node_id in ("ground_target", "plan_grasp", "plan_motion", "verify_pick"):
        assert "robot_motion" not in graph.node_map[node_id].specialist.requested_capabilities
    assert graph.node_map["verify_pick"].checkpoint == "hard"


def test_dynamic_graph_can_insert_geometry_and_choose_bowl_affordance(tmp_path) -> None:
    _, contracts = _library(tmp_path)
    draft = _pick_graph()
    draft["nodes"].insert(
        1,
        {
            "id": "analyze_geometry",
            "skill": "estimate_object_geometry",
            "inputs": {"object_points": {"$ref": "ground_target.object_points"}},
        },
    )
    draft["nodes"][2] = {
        "id": "plan_grasp",
        "skill": "grasp_open_bowl",
        "inputs": {
            "object_mask": {"$ref": "ground_target.object_mask"},
            "object_points": {"$ref": "ground_target.object_points"},
            "object_geometry": {"$ref": "analyze_geometry.object_geometry"},
        },
    }
    draft["edges"] = [
        {"from": "ground_target", "to": "analyze_geometry", "on": "success"},
        {"from": "analyze_geometry", "to": "plan_grasp", "on": "success"},
        *draft["edges"][1:],
    ]

    graph = SubgoalGraphCompiler(contracts).compile(
        draft,
        task_skill="pick_object",
        goal="pick up the open bowl",
        postcondition="bowl is held",
    )

    assert [node.specialist.specialist_skill for node in graph.nodes[:3]] == [
        "segment_object",
        "estimate_object_geometry",
        "grasp_open_bowl",
    ]
    assert graph.node_map["analyze_geometry"].specialist.role == "geometry_analyzer"


def test_dynamic_graph_cannot_override_contract_role_or_budget(tmp_path) -> None:
    _, contracts = _library(tmp_path)
    draft = _pick_graph()
    draft["nodes"][0]["role"] = "action_executor"
    draft["nodes"][0]["max_turns"] = 100

    graph = SubgoalGraphCompiler(contracts).compile(
        draft,
        task_skill="pick_object",
        goal="pick can",
        postcondition="can is held",
    )

    grounding = graph.node_map["ground_target"].specialist
    assert grounding.role == "grounding"
    assert grounding.max_turns == contracts["segment_object"].budget["max_turns"]


def test_edge_event_alias_normalizes_at_compile_time(tmp_path) -> None:
    _, contracts = _library(tmp_path)
    draft = _pick_graph()
    # "failure" is a common Manager alias for the canonical "failed" event.
    draft["edges"].append({"from": "execute_pick", "to": "ground_target", "on": "failure"})

    graph = SubgoalGraphCompiler(contracts).compile(
        draft,
        task_skill="pick_object",
        goal="pick can",
        postcondition="can is held",
    )

    assert any(
        edge.source == "execute_pick" and edge.target == "ground_target" and edge.on == "failed"
        for edge in graph.edges
    )


def test_unknown_edge_event_is_a_compile_error(tmp_path) -> None:
    _, contracts = _library(tmp_path)
    draft = _pick_graph()
    draft["edges"].append(
        {"from": "execute_pick", "to": "ground_target", "on": "retry_grasp"}
    )

    with pytest.raises(ValueError, match="unknown event .*declared edge events"):
        SubgoalGraphCompiler(contracts).compile(
            draft,
            task_skill="pick_object",
            goal="pick can",
            postcondition="can is held",
        )


def test_free_text_error_no_longer_routes_recovery(tmp_path) -> None:
    """Undeclared failures route on the generic failed edge, not on prose."""

    _, contracts = _library(tmp_path)
    graph = _compile_graph(contracts)
    state = RuntimeSafetyState()

    class ProseFailureFactory(_ContractFactory):
        failed = False

        def run(self, spec, **kwargs):
            if spec.agent_id == "plan_motion" and not self.failed:
                self.failed = True
                self.seen.append(spec)
                # The prose mentions "infeasible", but no failure_kind is
                # declared: the runtime must NOT infer an event from text.
                return AuthoringNodeResult(
                    node_id=spec.agent_id,
                    status=NodeStatus.FAILED,
                    error="motion looked infeasible in free text",
                )
            return super().run(spec, **kwargs)

    factory = ProseFailureFactory(state)
    result = SubgoalGraphExecutor(
        factory=factory, safety_state=state  # type: ignore[arg-type]
    ).run(
        graph,
        SubgoalAuthoringContext("task", 0, "pick", "held", artifact_dir=tmp_path),
    )

    # plan_motion only declares an "infeasible" recovery edge; without a
    # declared failure_kind the node exits "failed" and the graph terminates.
    assert result.status == AuthoringStatus.FAILED
    assert factory.seen[-1].agent_id == "plan_motion"


class _UncertainVerifyFactory(_ContractFactory):
    """Verifier always exits `uncertain` (e.g. VLM cannot decide)."""

    def run(self, spec, **kwargs):
        if spec.verifier:
            self.seen.append(spec)
            self.verify_calls += 1
            return AuthoringNodeResult(
                node_id=spec.agent_id,
                status=NodeStatus.SUCCEEDED,
                evidence=EvidencePacket.from_any(
                    {
                        "claim": "cannot decide whether the can is held",
                        "verdict": {"status": "uncertain", "reason": "occluded gripper"},
                    },
                    default_source=spec.agent_id,
                ),
                verification=VerificationStatus.UNCERTAIN,
            )
        return super().run(spec, **kwargs)


def test_uncertain_never_borrows_the_failed_edge(tmp_path) -> None:
    """M1.5 Fix D: an undecided checkpoint exits UNCERTAIN instead of silently
    rerouting onto the failed edge and replaying the whole (world-changing) chain."""

    _, contracts = _library(tmp_path)
    # _pick_graph declares `verify_pick -> ground_target on: failed`, but no
    # `on: uncertain` edge.
    graph = _compile_graph(contracts)
    state = RuntimeSafetyState()
    factory = _UncertainVerifyFactory(state)

    result = SubgoalGraphExecutor(
        factory=factory, safety_state=state  # type: ignore[arg-type]
    ).run(
        graph,
        SubgoalAuthoringContext("task", 0, "pick", "held", artifact_dir=tmp_path),
    )

    assert result.status == AuthoringStatus.UNCERTAIN
    assert result.verification == VerificationStatus.UNCERTAIN
    # The failed edge was NOT taken: the verifier ran once and nothing replayed.
    assert factory.verify_calls == 1
    assert [spec.agent_id for spec in factory.seen] == [
        "ground_target",
        "plan_grasp",
        "plan_motion",
        "execute_pick",
        "verify_pick",
    ]
    # The partial evidence survives for the planner to reason over.
    assert result.terminal_evidence.verdict is not None
    assert result.terminal_evidence.verdict.status == "uncertain"


def test_declared_uncertain_edge_routes_one_bounded_recheck(tmp_path) -> None:
    """A Manager may still declare `on: uncertain` explicitly; then it routes."""

    _, contracts = _library(tmp_path)
    draft = _pick_graph()
    draft["edges"].append(
        {"from": "verify_pick", "to": "verify_pick", "on": "uncertain"}
    )
    graph = SubgoalGraphCompiler(contracts).compile(
        draft,
        task_skill="pick_object",
        goal="pick can",
        postcondition="can is held",
    )
    state = RuntimeSafetyState()

    class UncertainOnceFactory(_ContractFactory):
        uncertain_once = False

        def run(self, spec, **kwargs):
            if spec.verifier and not self.uncertain_once:
                self.uncertain_once = True
                self.seen.append(spec)
                self.verify_calls += 1
                return AuthoringNodeResult(
                    node_id=spec.agent_id,
                    status=NodeStatus.SUCCEEDED,
                    evidence=EvidencePacket.from_any(
                        {"verdict": {"status": "uncertain", "reason": "blurred frame"}},
                        default_source=spec.agent_id,
                    ),
                    verification=VerificationStatus.UNCERTAIN,
                )
            return super().run(spec, **kwargs)

    factory = UncertainOnceFactory(state)
    result = SubgoalGraphExecutor(
        factory=factory, safety_state=state  # type: ignore[arg-type]
    ).run(
        graph,
        SubgoalAuthoringContext("task", 0, "pick", "held", artifact_dir=tmp_path),
    )

    assert result.status == AuthoringStatus.SUCCEEDED
    assert factory.verify_calls == 2
    # Only the verifier re-ran; the upstream chain did not replay.
    assert [spec.agent_id for spec in factory.seen[-2:]] == [
        "verify_pick",
        "verify_pick",
    ]


def test_dynamic_graph_rejects_forward_artifact_binding(tmp_path) -> None:
    _, contracts = _library(tmp_path)
    draft = _pick_graph()
    draft["nodes"][1]["inputs"]["object_points"] = {
        "$ref": "verify_pick.verifier_report"
    }

    with pytest.raises(ValueError, match="before its producer|expects"):
        SubgoalGraphCompiler(contracts).compile(
            draft,
            task_skill="pick_object",
            goal="pick can",
            postcondition="can is held",
        )


def test_manager_progressively_loads_task_skill_and_composes_graph(tmp_path) -> None:
    library, contracts = _library(tmp_path)
    state = RuntimeSafetyState()

    class RecordingGraphExecutor:
        def __init__(self):
            self.graph = None

        def run(self, graph, context):
            self.graph = graph
            return SubgoalOutcome(
                graph_name="test",
                status=AuthoringStatus.SUCCEEDED,
                verification=VerificationStatus.PASSED,
                node_results=(),
                artifacts=(),
                cost=AuthoringCost(),
            )

    graph_executor = RecordingGraphExecutor()
    manager = SubgoalSwarmManager(
        policy=_Policy(
            [
                '{"tool":"use_skill","args":{"name":"unknown"}}',
                '{"tool":"use_skill","args":{"name":"pick_object"}}',
                json.dumps(
                    {
                        "tool": "submit_graph",
                        "args": {
                            "task_skill": "pick_object",
                            "graph": {"nodes": []},
                        },
                    }
                ),
                json.dumps(
                    {
                        "tool": "submit_graph",
                        "args": {
                            "task_skill": "pick_object",
                            "graph": _pick_graph(),
                        },
                    }
                ),
            ]
        ),
        library=library,
        factory=_ContractFactory(state),  # type: ignore[arg-type]
        capability_ceiling=frozenset(
            {
                "perception_read",
                "geometry_compute",
                "artifact_write",
                "robot_motion",
                "gripper_control",
                "object_manipulation",
            }
        ),
        safety_state=state,
        contracts=contracts,
        graph_executor=graph_executor,  # type: ignore[arg-type]
    )
    result = manager.run(
        SubgoalAuthoringContext(
            task="pick soup",
            subgoal_index=0,
            goal="pick the soup can",
            postcondition="can is held",
            artifact_dir=tmp_path,
        )
    )

    assert result.status == AuthoringStatus.SUCCEEDED
    assert graph_executor.graph.task_skill == "pick_object"
    assert result.node_results[0].node_id == "swarm_manager"
    initial_request = (
        tmp_path
        / "authoring"
        / "swarm"
        / "manager_trace"
        / "llm_io"
        / "turn_00_request.json"
    ).read_text(encoding="utf-8")
    first_retry = (
        tmp_path
        / "authoring"
        / "swarm"
        / "manager_trace"
        / "llm_io"
        / "turn_01_request.json"
    ).read_text(encoding="utf-8")
    validation_retry = (
        tmp_path
        / "authoring"
        / "swarm"
        / "manager_trace"
        / "llm_io"
        / "turn_03_request.json"
    ).read_text(encoding="utf-8")
    assert "Unknown task skill" in first_retry
    assert "Dynamic graph validation failed" in validation_retry
    assert "Loaded skill" in validation_retry
    assert "generate a graph that fits the live scene" not in initial_request
    assert "generate a graph that fits the live scene" in validation_retry
    assert "spawn_subagent" not in validation_retry
    assert (
        tmp_path / "authoring" / "swarm" / "manager_trace" / "submitted_graph.json"
    ).is_file()


def test_graph_executor_runs_typed_pick_chain_and_hard_verifier(tmp_path) -> None:
    _, contracts = _library(tmp_path)
    graph = _compile_graph(contracts)
    state = RuntimeSafetyState()
    factory = _ContractFactory(state)
    executor = SubgoalGraphExecutor(factory=factory, safety_state=state)  # type: ignore[arg-type]

    result = executor.run(
        graph,
        SubgoalAuthoringContext("task", 0, "pick can", "can is held", artifact_dir=tmp_path),
    )

    assert result.status == AuthoringStatus.SUCCEEDED
    assert result.verification == VerificationStatus.PASSED
    assert [spec.role for spec in factory.seen] == [
        "grounding",
        "affordance",
        "motion_planner",
        "action_executor",
        "verifier",
    ]
    assert result.motion_attempted
    assert (tmp_path / "authoring" / "swarm" / "subgoal_graph.json").is_file()
    assert (tmp_path / "authoring" / "swarm" / "node_events.jsonl").is_file()


@pytest.mark.parametrize("task_skill", ["pick_object", "place_object"])
def test_pick_and_place_golden_paths(task_skill, tmp_path) -> None:
    _, contracts = _library(tmp_path)
    graph = _compile_graph(contracts, task_skill)
    state = RuntimeSafetyState()
    factory = _ContractFactory(state)

    result = SubgoalGraphExecutor(
        factory=factory, safety_state=state  # type: ignore[arg-type]
    ).run(
        graph,
        SubgoalAuthoringContext(
            "task", 0, "complete manipulation", "relation holds", artifact_dir=tmp_path
        ),
    )

    assert result.status == AuthoringStatus.SUCCEEDED
    assert factory.seen[-2].role == "action_executor"
    assert factory.seen[-1].role == "verifier"


def test_verifier_failure_routes_only_along_declared_recovery_edge(tmp_path) -> None:
    _, contracts = _library(tmp_path)
    graph = _compile_graph(contracts)
    state = RuntimeSafetyState()
    factory = _ContractFactory(state, fail_first_verify=True)
    executor = SubgoalGraphExecutor(factory=factory, safety_state=state)  # type: ignore[arg-type]

    result = executor.run(
        graph,
        SubgoalAuthoringContext("task", 0, "pick can", "can is held", artifact_dir=tmp_path),
    )

    assert result.status == AuthoringStatus.SUCCEEDED
    assert factory.verify_calls == 2
    assert [spec.agent_id for spec in factory.seen][5:] == [
        "ground_target",
        "plan_grasp",
        "plan_motion",
        "execute_pick",
        "verify_pick",
    ]


@pytest.mark.parametrize(
    ("failed_node", "failure_kind", "expected_recovery"),
    [
        ("plan_grasp", "wrong_grounding", "ground_target"),
        ("plan_motion", "infeasible", "plan_grasp"),
    ],
)
def test_declared_recovery_events_are_bounded(
    failed_node, failure_kind, expected_recovery, tmp_path
) -> None:
    _, contracts = _library(tmp_path)
    graph = _compile_graph(contracts)
    state = RuntimeSafetyState()

    class RecoverOnceFactory(_ContractFactory):
        failed = False

        def run(self, spec, **kwargs):
            if spec.agent_id == failed_node and not self.failed:
                self.failed = True
                self.seen.append(spec)
                return AuthoringNodeResult(
                    node_id=spec.agent_id,
                    status=NodeStatus.FAILED,
                    error="specialist reported a typed failure",
                    failure_kind=failure_kind,
                )
            return super().run(spec, **kwargs)

    factory = RecoverOnceFactory(state)
    result = SubgoalGraphExecutor(
        factory=factory, safety_state=state  # type: ignore[arg-type]
    ).run(
        graph,
        SubgoalAuthoringContext("task", 0, "pick", "held", artifact_dir=tmp_path),
    )

    failed_index = next(
        index for index, spec in enumerate(factory.seen) if spec.agent_id == failed_node
    )
    assert result.status == AuthoringStatus.SUCCEEDED
    assert factory.seen[failed_index + 1].agent_id == expected_recovery


def test_stale_publish_recovers_via_declared_reobserve_edge(tmp_path) -> None:
    _, contracts = _library(tmp_path)
    graph = _compile_graph(contracts)
    state = RuntimeSafetyState()

    class StaleOnceFactory(_ContractFactory):
        stale_once = False

        def run(self, spec, **kwargs):
            result = super().run(spec, **kwargs)
            if spec.agent_id == "ground_target" and not self.stale_once:
                self.stale_once = True
                self.state.observation_epoch += 1
            return result

    factory = StaleOnceFactory(state)
    result = SubgoalGraphExecutor(
        factory=factory, safety_state=state  # type: ignore[arg-type]
    ).run(
        graph,
        SubgoalAuthoringContext("task", 0, "pick", "held", artifact_dir=tmp_path),
    )

    assert result.status == AuthoringStatus.SUCCEEDED
    assert [spec.agent_id for spec in factory.seen[:2]] == [
        "ground_target",
        "ground_target",
    ]


def test_factory_rederives_capabilities_from_contract(tmp_path) -> None:
    library, contracts = _library(tmp_path)
    state = RuntimeSafetyState()
    inner = _Executor()
    factory = SubAgentFactory(
        executor=inner,
        policy=_Policy(
            [
                '{"tool":"run_python","args":{"code":"goto_pose(target)","intent":"move"}}',
                '{"tool":"finish","args":{"claim":"grounded","result":{"outputs":'
                '{"object_mask":{"payload":{},"frame":"image"},'
                '"object_points":{"payload":{},"frame":"world"}}}}}',
            ]
        ),
        library=library,
        capability_ceiling=frozenset(
            {"perception_read", "geometry_compute", "artifact_write", "robot_motion"}
        ),
        safety_state=state,
        contracts=contracts,
    )
    tampered = SpecialistSpec.from_mapping(
        {
            "id": "ground",
            "specialist_skill": "segment_object",
            "role": "action_executor",
            "objective": "ground",
            "task": "ground",
            "capabilities": ["robot_motion"],
        }
    )

    result = factory.run(
        tampered,
        context=SubgoalAuthoringContext("task", 0, "ground", "visible"),
        inputs={},
        artifact_dir=None,
    )

    assert result.ok
    assert state.observation_epoch == 0
    assert not any("goto_pose" in block.code for block in inner.blocks)
    assert result.trace is not None
    assert result.trace.loaded_skill_ids == ("segment_object",)


def test_contract_budget_is_not_clamped_by_factory_default(tmp_path) -> None:
    library, contracts = _library(tmp_path)
    inner = _Executor()
    factory = SubAgentFactory(
        executor=inner,
        policy=_Policy(
            [
                '{"tool":"run_python","args":{"code":"print(1)"}}',
                '{"tool":"run_python","args":{"code":"print(2)"}}',
                '{"tool":"finish","args":{"claim":"planned","result":{"outputs":'
                '{"trajectory":{"payload":{"feasible":true,"waypoints":'
                '[{"name":"pregrasp","phase":"pregrasp","position_xyz":[0.7,0.0,0.2],'
                '"quaternion_wxyz":[0.0,1.0,0.0,0.0],"gripper":"open"}]},'
                '"frame":"world"}}}}}',
            ]
        ),
        library=library,
        capability_ceiling=frozenset(
            {"perception_read", "geometry_compute", "artifact_write"}
        ),
        default_max_turns=1,
        contracts=contracts,
    )
    spec = SpecialistSpec.from_contract(
        agent_id="plan_motion",
        contract=contracts["plan_bounded_motion"],
        objective="plan",
        task="plan",
    )
    affordance = TypedArtifact(
        port="affordance",
        schema="robomex.affordance.v1",
        producer="affordance",
        frame="world",
    )

    result = factory.run(
        spec,
        context=SubgoalAuthoringContext("task", 0, "plan", "feasible"),
        inputs={"affordance": affordance},
        artifact_dir=None,
    )

    assert result.ok
    assert [block.code for block in inner.blocks if block.code.startswith("print(")] == [
        "print(1)",
        "print(2)",
    ]


def test_leaf_rejects_incomplete_or_inline_array_finish(tmp_path) -> None:
    library, contracts = _library(tmp_path)
    factory = SubAgentFactory(
        executor=_Executor(),
        policy=_Policy(
            [
                '{"tool":"finish","args":{"result":{"outputs":'
                '{"object_mask":{"payload":[]}}}}}',
                '{"tool":"finish","args":{"result":{"outputs":'
                '{"object_mask":{"payload":{}},"object_points":'
                '{"payload":{},"frame":"world"}}}}}',
            ]
        ),
        library=library,
        capability_ceiling=frozenset(
            {"perception_read", "geometry_compute", "artifact_write"}
        ),
        contracts=contracts,
    )
    spec = SpecialistSpec.from_contract(
        agent_id="ground",
        contract=contracts["segment_object"],
        objective="ground",
        task="ground",
    )

    result = factory.run(
        spec,
        context=SubgoalAuthoringContext("task", 0, "ground", "visible"),
        inputs={},
        artifact_dir=tmp_path,
    )

    assert result.ok
    assert {output.port for output in result.outputs} == {
        "object_mask",
        "object_points",
    }


def test_verifier_promotes_typed_report_verdict(tmp_path) -> None:
    library, contracts = _library(tmp_path)
    factory = SubAgentFactory(
        executor=_Executor(),
        policy=_Policy(
            [
                '{"tool":"finish","args":{"result":{"outputs":{"verifier_report":'
                '{"payload":{"verdict":{"status":"fail","reason":"object not held"}}}},'
                '"recommended_next":"reground"}}}'
            ]
        ),
        library=library,
        capability_ceiling=frozenset({"perception_read", "artifact_write"}),
        contracts=contracts,
    )
    spec = SpecialistSpec.from_contract(
        agent_id="verify",
        contract=contracts["verify_grasp_and_lift_via_robot_state"],
        objective="verify",
        task="verify",
    )
    execution = TypedArtifact(
        port="execution_evidence",
        schema="robomex.execution_evidence.v1",
        producer="execute",
    )

    result = factory.run(
        spec,
        context=SubgoalAuthoringContext("task", 0, "verify", "held"),
        inputs={"execution_evidence": execution},
        artifact_dir=tmp_path,
    )

    assert result.status == NodeStatus.FAILED
    assert result.verification == VerificationStatus.FAILED
    assert result.outputs[0].port == "verifier_report"
    assert result.evidence.verdict is not None
    assert result.evidence.verdict.reason == "object not held"


def test_motion_lease_advances_observation_epoch() -> None:
    executor = _Executor()
    state = RuntimeSafetyState(observation_epoch=4)
    guard = MotionLeaseGuard(executor, state, agent_id="action_executor")
    block = SemanticActionBlock("move", "move robot", "goto_pose(target)")

    result = guard.run_block(block)

    assert result.ok
    assert state.observation_epoch == 5


class _RobotStateExecutor:
    """Executor whose post-block observation carries proprioception."""

    def run_block(self, block):
        return BlockExecutionResult(
            block=block,
            ok=True,
            status=ActionBlockStatus.SUCCEEDED,
            terminated=False,
            truncated=False,
            observation={
                "agentview": {},
                "robot_cartesian_pos": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.42],
                "robot_joint_pos": [0.0] * 7,
            },
        )


def test_motion_lease_captures_terminal_robot_state() -> None:
    state = RuntimeSafetyState(observation_epoch=4)
    guard = MotionLeaseGuard(_RobotStateExecutor(), state, agent_id="action_executor")
    block = SemanticActionBlock("move", "move robot", "goto_pose(target)")

    result = guard.run_block(block)

    captured = result.info["terminal_robot_state"]
    assert captured["robot_cartesian_pos"][7] == pytest.approx(0.42)
    assert captured["gripper_open_ratio"] == pytest.approx(0.42)
    assert captured["observation_epoch"] == 5
    assert captured["source"] == "runtime"
    assert state.last_terminal_robot_state == captured


def test_non_motion_block_does_not_capture_terminal_state() -> None:
    state = RuntimeSafetyState(observation_epoch=4)
    guard = MotionLeaseGuard(_RobotStateExecutor(), state, agent_id="observer")
    block = SemanticActionBlock("look", "read scene", "print('inspect')")

    result = guard.run_block(block)

    assert "terminal_robot_state" not in result.info
    assert state.last_terminal_robot_state is None
    assert state.observation_epoch == 4


def test_runtime_terminal_state_projects_into_execution_evidence(tmp_path) -> None:
    library, contracts = _library(tmp_path)
    safety_state = RuntimeSafetyState(observation_epoch=3)
    safety_state.last_terminal_robot_state = {
        "robot_cartesian_pos": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.02],
        "gripper_open_ratio": 0.02,
        "observation_epoch": 3,
        "source": "runtime",
    }
    factory = SubAgentFactory(
        executor=_Executor(),
        policy=_Policy(
            [
                '{"tool":"finish","args":{"result":{"outputs":{"execution_evidence":'
                '{"payload":{"primitives":[{"name":"goto_pose","status":"succeeded"}],'
                '"all_primitives_ok":true}}}}}}'
            ]
        ),
        library=library,
        capability_ceiling=frozenset(
            {"robot_motion", "gripper_control", "object_manipulation", "artifact_write"}
        ),
        safety_state=safety_state,
        contracts=contracts,
    )
    spec = SpecialistSpec.from_contract(
        agent_id="execute",
        contract=contracts["grasp_object"],
        objective="grasp",
        task="grasp",
    )

    result = factory.run(
        spec,
        context=SubgoalAuthoringContext("task", 0, "grasp", "held"),
        inputs={},
        artifact_dir=tmp_path,
    )

    evidence = result.outputs[0]
    assert evidence.port == "execution_evidence"
    assert evidence.payload["all_primitives_ok"] is True
    assert evidence.payload["terminal_robot_state"]["gripper_open_ratio"] == pytest.approx(0.02)


def test_stale_terminal_state_is_not_projected(tmp_path) -> None:
    library, contracts = _library(tmp_path)
    safety_state = RuntimeSafetyState(observation_epoch=5)
    safety_state.last_terminal_robot_state = {
        "gripper_open_ratio": 0.9,
        "observation_epoch": 3,
        "source": "runtime",
    }
    factory = SubAgentFactory(
        executor=_Executor(),
        policy=_Policy(
            [
                '{"tool":"finish","args":{"result":{"outputs":{"execution_evidence":'
                '{"payload":{"primitives":[{"name":"goto_pose","status":"succeeded"}],'
                '"all_primitives_ok":true}}}}}}'
            ]
        ),
        library=library,
        capability_ceiling=frozenset(
            {"robot_motion", "gripper_control", "object_manipulation", "artifact_write"}
        ),
        safety_state=safety_state,
        contracts=contracts,
    )
    spec = SpecialistSpec.from_contract(
        agent_id="execute",
        contract=contracts["grasp_object"],
        objective="grasp",
        task="grasp",
    )

    result = factory.run(
        spec,
        context=SubgoalAuthoringContext("task", 0, "grasp", "held"),
        inputs={},
        artifact_dir=tmp_path,
    )

    assert "terminal_robot_state" not in result.outputs[0].payload


def test_action_alias_is_accepted_for_provider_robustness() -> None:
    action = parse_action(
        '{"action":"finish","args":{"claim":"done","result":{"ok":true}}}'
    )

    assert action.kind == "finish"
