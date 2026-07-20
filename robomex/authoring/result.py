"""Runtime values produced by subgoal authoring graphs.

The authoring layer deliberately separates control-flow completion from physical
success.  A node can finish normally while its verifier reports failure, and an
episode can reach planner ``DONE`` without the environment accepting the task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from robomex.authoring.artifacts import TypedArtifact
from robomex.core.coder.trace import AgentTrace
from robomex.core.context import EvidencePacket


class NodeStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    EXHAUSTED = "exhausted"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    NOT_RUN = "not_run"


class AuthoringStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class AuthoringCost:
    llm_calls: int = 0
    action_turns: int = 0
    duration_s: float = 0.0

    def __add__(self, other: "AuthoringCost") -> "AuthoringCost":
        return AuthoringCost(
            llm_calls=self.llm_calls + other.llm_calls,
            action_turns=self.action_turns + other.action_turns,
            duration_s=self.duration_s + other.duration_s,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "action_turns": self.action_turns,
            "duration_s": self.duration_s,
        }


@dataclass(frozen=True)
class AuthoringNodeResult:
    node_id: str
    status: NodeStatus
    outputs: tuple[TypedArtifact, ...] = ()
    evidence: EvidencePacket = field(default_factory=EvidencePacket)
    trace: AgentTrace | None = None
    verification: VerificationStatus = VerificationStatus.NOT_RUN
    cost: AuthoringCost = field(default_factory=AuthoringCost)
    error: str = ""
    attempt: int = 1
    # Structured routing hint: one of the declared graph failure kinds
    # (e.g. "failed_grasp", "stale_observation"). Empty means "no specific
    # kind"; the runtime then routes on status/verification alone.
    failure_kind: str = ""

    @property
    def ok(self) -> bool:
        return self.status == NodeStatus.SUCCEEDED

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "verification": self.verification.value,
            "failure_kind": self.failure_kind,
            "outputs": [item.to_json_dict() for item in self.outputs],
            "evidence": self.evidence.to_json_dict(),
            "cost": self.cost.to_json_dict(),
            "error": self.error,
            "attempt": self.attempt,
            "trace": None
            if self.trace is None
            else {
                "task": self.trace.task,
                "loaded_skill_ids": list(self.trace.loaded_skill_ids),
                "turns": len(self.trace.turns),
                "metadata": self.trace.metadata,
            },
        }


@dataclass(frozen=True)
class SubgoalAuthoringContext:
    task: str
    subgoal_index: int
    goal: str
    postcondition: str
    scene_image_path: str | None = None
    observation_summary: str = ""
    artifact_dir: Path | None = None


@dataclass(frozen=True)
class SubgoalOutcome:
    graph_name: str
    status: AuthoringStatus
    verification: VerificationStatus
    node_results: tuple[AuthoringNodeResult, ...]
    artifacts: tuple[TypedArtifact, ...]
    cost: AuthoringCost
    terminal_node: str = ""
    note: str = ""
    strategy: str = ""
    creator_status: str = ""
    motion_attempted: bool = False
    terminal_evidence: EvidencePacket = field(default_factory=EvidencePacket)

    @property
    def executable_trace(self) -> AgentTrace | None:
        traces = [result.trace for result in self.node_results if result.trace is not None]
        if not traces:
            return None
        if len(traces) == 1:
            return traces[0]
        return AgentTrace(
            task=traces[-1].task,
            loaded_skill_ids=tuple(
                dict.fromkeys(
                    skill_id
                    for trace in traces
                    for skill_id in trace.loaded_skill_ids
                )
            ),
            turns=tuple(turn for trace in traces for turn in trace.turns),
            success=self.success,
            metadata={
                "authoring_strategy": self.strategy,
                "graph_name": self.graph_name,
                "creator_status": self.creator_status,
                "motion_attempted": self.motion_attempted,
                "verification_status": self.verification.value,
                "authoring_cost": self.cost.to_json_dict(),
                "node_timeline": [
                    {
                        "node_id": result.node_id,
                        "status": result.status.value,
                        "verification": result.verification.value,
                        "attempt": result.attempt,
                        "loaded_skill_ids": []
                        if result.trace is None
                        else list(result.trace.loaded_skill_ids),
                        "error": result.error,
                    }
                    for result in self.node_results
                ],
                "terminal_result": self.terminal_evidence.to_json_dict(),
                "evidence_packets": [
                    {
                        "source": f"authoring:{result.node_id}",
                        "packet": result.evidence.to_json_dict(),
                    }
                    for result in self.node_results
                    if not result.evidence.is_empty
                ],
                "artifact_refs": [
                    ref.to_json_dict()
                    for artifact in self.artifacts
                    for ref in artifact.refs
                ],
                "agent_traces": [
                    {
                        "task": trace.task,
                        "turns": len(trace.turns),
                        "success": trace.success,
                        "metadata": trace.metadata,
                    }
                    for trace in traces
                ],
            },
        )

    @property
    def success(self) -> bool:
        return self.status == AuthoringStatus.SUCCEEDED

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": "robomex.subgoal_outcome.v1",
            "graph_name": self.graph_name,
            "status": self.status.value,
            "verification": self.verification.value,
            "terminal_node": self.terminal_node,
            "note": self.note,
            "strategy": self.strategy,
            "creator_status": self.creator_status,
            "motion_attempted": self.motion_attempted,
            "success": self.success,
            "terminal_evidence": self.terminal_evidence.to_json_dict(),
            "cost": self.cost.to_json_dict(),
            "artifacts": [item.to_json_dict() for item in self.artifacts],
            "node_results": [item.to_json_dict() for item in self.node_results],
        }


AuthoringRunResult = SubgoalOutcome
