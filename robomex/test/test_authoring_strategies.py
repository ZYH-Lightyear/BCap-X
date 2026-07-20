from __future__ import annotations

import json

from robomex.authoring.result import (
    AuthoringCost,
    AuthoringNodeResult,
    AuthoringStatus,
    NodeStatus,
    SubgoalOutcome,
    VerificationStatus,
)
from robomex.core.coder.trace import AgentTrace, TurnRecord
from robomex.core.context import AttemptHistory, DiagnosisStore, TraceStore
from robomex.core.sandbox import (
    ActionBlockStatus,
    BlockExecutionResult,
    ExecutionTraceEvent,
    SemanticActionBlock,
)
from robomex.core.session import RoboMExAgent, RoboMExConfig


class _Library:
    root = None

    def all(self, category=None):
        return []

    def task_skills(self):
        return []

    def get(self, name):
        raise KeyError(name)


class _PlannerPolicy:
    def propose(self, prompt):
        return "DONE"


class _CodePolicy:
    def complete(self, prompt):
        return '{"tool":"finish","args":{"claim":"done"}}'


class _Executor:
    def run_block(self, block):
        raise AssertionError("strategy construction smoke must not execute code")


def _config(**overrides) -> RoboMExConfig:
    values = {
        "library": _Library(),
        "planner_policy": _PlannerPolicy(),
        "code_policy": _CodePolicy(),
        "executor": _Executor(),
    }
    values.update(overrides)
    return RoboMExConfig(**values)


def test_universal_strategy_constructs_official_baseline() -> None:
    agent = RoboMExAgent(_config(authoring_strategy="universal"))

    assert agent.authoring.strategy == "universal"
    assert agent.authoring.graph_name == "universal_v2"


def test_dynamic_swarm_strategy_constructs_subgoal_mas_manager() -> None:
    agent = RoboMExAgent(_config(authoring_strategy="dynamic_swarm"))

    assert agent.authoring.strategy == "dynamic_swarm"
    assert agent.authoring.graph_name == "subgoal_mas_dynamic_v1"


def test_role_specific_policy_wiring_separates_manager_and_subagents() -> None:
    manager_policy = _CodePolicy()
    subagent_policy = _CodePolicy()
    agent = RoboMExAgent(
        _config(
            authoring_strategy="dynamic_swarm",
            code_policy=manager_policy,
            subagent_policy=subagent_policy,
        )
    )

    assert agent.authoring.policy is manager_policy
    assert agent.authoring.factory.policy is subagent_policy
    assert agent.executor_agent.policy is subagent_policy


def _node_trace(node_id: str, *, terminated: bool = False, task_completed: bool = False) -> AgentTrace:
    block = SemanticActionBlock(name="turn_0", intent="act", code="goto_pose(t)")
    execution = BlockExecutionResult(
        block=block,
        ok=True,
        status=ActionBlockStatus.SUCCEEDED,
        terminated=terminated,
        info={"task_completed": task_completed} if task_completed else {},
        trace_events=(
            ExecutionTraceEvent(
                event_type="primitive_trace",
                message="goto_pose succeeded",
                payload={
                    "primitive_traces": [
                        {
                            "trace_id": f"{node_id}:goto_pose:1",
                            "primitive_name": "goto_pose",
                            "status": "succeeded",
                        }
                    ]
                },
            ),
        ),
    )
    return AgentTrace(
        task=node_id,
        loaded_skill_ids=(),
        turns=(TurnRecord(0, "goto_pose(t)", execution),),
        success=True,
        metadata={},
    )


def _outcome(node_results) -> SubgoalOutcome:
    return SubgoalOutcome(
        graph_name="subgoal_mas:test:v1",
        status=AuthoringStatus.FAILED,
        verification=VerificationStatus.NOT_RUN,
        node_results=tuple(node_results),
        artifacts=(),
        cost=AuthoringCost(),
        strategy="dynamic_swarm",
    )


def test_swarm_node_results_project_into_episode_memory() -> None:
    trace_store = TraceStore()
    attempt_history = AttemptHistory()
    diagnosis_store = DiagnosisStore()
    outcome = _outcome(
        [
            AuthoringNodeResult(
                node_id="execute_grasp",
                status=NodeStatus.SUCCEEDED,
                trace=_node_trace("execute_grasp"),
            ),
            AuthoringNodeResult(
                node_id="verify_lift",
                status=NodeStatus.FAILED,
                verification=VerificationStatus.FAILED,
                failure_kind="failed_grasp",
                error="object not held",
            ),
        ]
    )

    RoboMExAgent._merge_swarm_node_results(
        trace_store,
        attempt_history,
        diagnosis_store,
        outcome,
        subgoal_index=0,
        subgoal_goal="pick the can",
    )

    assert "execute_grasp:goto_pose:1" in trace_store.traces
    assert len(attempt_history.records) == 2
    executed, verified = attempt_history.records
    assert executed.attempt_id == "subgoal_00:execute_grasp:a1"
    assert executed.outcome == "succeeded"
    assert "execute_grasp:goto_pose:1" in executed.related_trace_ids
    assert verified.outcome == "failed/verified_failed"
    assert verified.failure_reason == "failed_grasp"
    assert len(diagnosis_store.diagnoses) == 1
    assert diagnosis_store.diagnoses[0].failure_type == "failed_grasp"
    assert diagnosis_store.diagnoses[0].failed_primitive == "verify_lift"


def test_empty_outcome_projects_nothing() -> None:
    trace_store = TraceStore()
    attempt_history = AttemptHistory()
    diagnosis_store = DiagnosisStore()

    RoboMExAgent._merge_swarm_node_results(
        trace_store,
        attempt_history,
        diagnosis_store,
        _outcome([]),
        subgoal_index=0,
        subgoal_goal="pick",
    )

    assert not trace_store.traces
    assert not attempt_history.records
    assert not diagnosis_store.diagnoses


def test_env_termination_signal_short_circuits() -> None:
    assert RoboMExAgent._env_termination_signal(
        _node_trace("execute", terminated=True)
    ) == {"terminated": True, "turn": 0}
    assert RoboMExAgent._env_termination_signal(
        _node_trace("execute", task_completed=True)
    ) == {"task_completed": True, "turn": 0}
    assert RoboMExAgent._env_termination_signal(_node_trace("execute")) is None


def test_dynamic_swarm_session_summary_does_not_report_finish_as_execution_success(
    tmp_path,
) -> None:
    class Planner:
        def __init__(self):
            self.calls = 0

        def propose(self, prompt):
            self.calls += 1
            return (
                "Goal: inspect the object\nPostcondition: object is visible"
                if self.calls == 1
                else "DONE"
            )

    agent = RoboMExAgent(
        _config(
            planner_policy=Planner(),
            authoring_strategy="dynamic_swarm",
            artifacts_dir=str(tmp_path),
        )
    )

    result = agent.run("inspect")
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert result.execution.results[0].authoring_status == "exhausted"
    assert summary["subgoals"][0]["success"] is False
    assert "execution_finished" not in summary["subgoals"][0]
