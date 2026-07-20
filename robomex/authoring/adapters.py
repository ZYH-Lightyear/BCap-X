"""Execution adapters for the universal baseline and dynamic leaf agents."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from robomex.agents.executor import CodeAsPolicyAgent
from robomex.agents.subagents import CodingAgentSubAgent, SubAgentRequest
from robomex.authoring.artifacts import (
    TypedArtifact,
    add_grounding_overlay,
    outputs_from_finish,
)
from robomex.authoring.capabilities import (
    CapabilityBoundBlockExecutor,
    CapabilityPolicy,
    effective_capabilities,
)
from robomex.core.edge_events import FAILURE_KINDS, normalize_edge_event
from robomex.authoring.result import (
    AuthoringCost,
    AuthoringNodeResult,
    AuthoringStatus,
    NodeStatus,
    SubgoalAuthoringContext,
    SubgoalOutcome,
    VerificationStatus,
)
from robomex.authoring.swarm_spec import SpecialistSpec
from robomex.core.context import EvidencePacket
from robomex.core.events import emit_event
from robomex.core.sandbox import MotionLeaseGuard, RuntimeSafetyState
from robomex.prompts.authoring import build_agent_system_prompt, output_contract_for_ports
from robomex.dysc.contracts import SkillContract, load_skill_contracts
from robomex.dysc.views import specialist_skill_view
from robomex.skills import SkillLibrary


def declared_failure_kind(parsed: dict[str, Any] | None) -> str:
    """Extract a specialist-declared failure kind from one finish result.

    Only values from the closed graph vocabulary survive; anything else is
    dropped so downstream routing never sees an undeclared event.
    """

    if not isinstance(parsed, dict):
        return ""
    nested = parsed.get("result") if isinstance(parsed.get("result"), dict) else {}
    raw = parsed.get("failure_kind") or nested.get("failure_kind")
    if not raw:
        return ""
    event = normalize_edge_event(raw)
    return event if event in FAILURE_KINDS else ""


def verification_from_packet(packet: EvidencePacket) -> VerificationStatus:
    status = str(packet.verdict.status if packet.verdict is not None else "").lower()
    if status in {"pass", "passed", "success", "succeeded"}:
        return VerificationStatus.PASSED
    if status in {"fail", "failed", "error"}:
        return VerificationStatus.FAILED
    if status in {"uncertain", "unknown"}:
        return VerificationStatus.UNCERTAIN
    return VerificationStatus.NOT_RUN


class UniversalRunner:
    """Independent one-agent baseline with the same outcome contract as swarm."""

    strategy = "universal"
    graph_name = "universal_v2"

    def __init__(self, agent: CodeAsPolicyAgent, safety_state: RuntimeSafetyState) -> None:
        self.agent = agent
        self.safety_state = safety_state

    def run(self, context: SubgoalAuthoringContext) -> SubgoalOutcome:
        entry_epoch = self.safety_state.observation_epoch
        trace = self.agent.run(
            context.goal,
            context.observation_summary,
            expected_postcondition=context.postcondition,
            video_dir=context.artifact_dir,
            feedback="",
            scene_image_path=context.scene_image_path,
        )
        packet = EvidencePacket.from_any(
            (trace.metadata or {}).get("terminal_result") or {},
            default_source="universal",
            default_turn=f"subgoal:{context.subgoal_index}",
        )
        verification = verification_from_packet(packet)
        status = _outcome_status(
            terminal=bool((trace.metadata or {}).get("terminal_result")),
            verification=verification,
        )
        node = AuthoringNodeResult(
            node_id="universal",
            status=NodeStatus.SUCCEEDED if status == AuthoringStatus.SUCCEEDED else NodeStatus.FAILED,
            evidence=packet,
            trace=trace,
            verification=verification,
            cost=AuthoringCost(llm_calls=len(trace.turns) + 1, action_turns=len(trace.turns)),
        )
        return SubgoalOutcome(
            graph_name=self.graph_name,
            status=status,
            verification=verification,
            node_results=(node,),
            artifacts=(),
            cost=node.cost,
            terminal_node="universal",
            strategy=self.strategy,
            creator_status=status.value,
            motion_attempted=self.safety_state.observation_epoch > entry_epoch,
            terminal_evidence=packet,
        )


class SubAgentFactory:
    """Build and execute one focused Coding Agent leaf."""

    def __init__(
        self,
        *,
        executor: Any,
        policy: Any,
        library: SkillLibrary,
        capability_ceiling: frozenset[str],
        default_max_turns: int = 6,
        environment: str = "",
        api_docs: str = "",
        safety_state: RuntimeSafetyState | None = None,
        contracts: dict[str, SkillContract] | None = None,
    ) -> None:
        self.executor = executor
        self.policy = policy
        self.library = library
        self.capability_ceiling = capability_ceiling
        self.default_max_turns = default_max_turns
        self.environment = environment
        self.api_docs = api_docs
        self.safety_state = safety_state or RuntimeSafetyState()
        root = getattr(library, "root", None)
        self.contracts = (
            dict(contracts)
            if contracts is not None
            else (load_skill_contracts(root) if root is not None else {})
        )

    def run(
        self,
        spec: SpecialistSpec,
        *,
        context: SubgoalAuthoringContext,
        inputs: dict[str, TypedArtifact],
        artifact_dir: Path | None,
    ) -> AuthoringNodeResult:
        contract = self.contracts.get(spec.specialist_skill)
        if contract is None:
            raise ValueError(
                f"Specialist skill {spec.specialist_skill!r} has no machine-readable contract."
            )
        spec = SpecialistSpec.from_contract(
            agent_id=spec.agent_id,
            contract=contract,
            objective=spec.objective,
            task=spec.task,
            max_turns=spec.max_turns,
        )
        forbidden = set(contract.forbidden_capabilities).intersection(
            spec.requested_capabilities
        )
        if forbidden:
            raise PermissionError(
                f"Specialist contract grants forbidden capabilities: {', '.join(sorted(forbidden))}."
            )
        granted = effective_capabilities(spec.requested_capabilities, self.capability_ceiling)
        emit_event(
            "capability_granted",
            "Dynamic SubAgent capabilities granted",
            agent_id=spec.agent_id,
            effective=sorted(granted),
        )
        executor = CapabilityBoundBlockExecutor(
            MotionLeaseGuard(
                self.executor, self.safety_state, agent_id=spec.agent_id
            ),
            CapabilityPolicy(allowed=granted),
            node_id=spec.agent_id,
        )
        output_ports = tuple((p.name, p.schema, p.frame) for p in spec.outputs)
        base_prompt = build_agent_system_prompt(
            role=spec.role,
            objective=spec.objective,
            capabilities=granted,
            output_contract=output_contract_for_ports(output_ports),
            environment=self.environment,
            api_docs=self.api_docs,
        )
        prompt = (
            f"{base_prompt}\n\nRole-specific instructions:\n{spec.system_prompt.strip()}"
            if spec.system_prompt.strip()
            else base_prompt
        )
        agent = CodingAgentSubAgent(
            executor=executor,
            policy=self.policy,
            library=specialist_skill_view(
                self.library,
                self.contracts,
                required=spec.required_skills,
                preferred=spec.preferred_skills,
            ),  # type: ignore[arg-type]
            max_turns=spec.max_turns or self.default_max_turns,
            system_prompt=prompt,
            name=spec.agent_id,
            description=spec.role,
            objective=spec.objective,
            task_kind=spec.role,
            output_ports=output_ports,
            preloaded_skills=spec.required_skills,
        )
        request = SubAgentRequest(
            task=spec.task,
            task_kind=spec.role,
            inputs={name: artifact.to_json_dict() for name, artifact in inputs.items()},
            artifacts_dir=str(artifact_dir) if artifact_dir else None,
            task_id=f"{context.subgoal_index}:{spec.agent_id}",
            scene_image_path=context.scene_image_path,
            observation_epoch=self.safety_state.observation_epoch,
        )
        result = agent.run(request)
        packet = result.resolved_evidence_packet()
        verification = (
            verification_from_packet(packet)
            if spec.verifier
            else VerificationStatus.NOT_RUN
        )
        error = (
            ""
            if spec.verifier and verification != VerificationStatus.NOT_RUN
            else result.error
        )
        outputs: tuple[TypedArtifact, ...] = ()
        if result.ok or (
            spec.verifier and verification != VerificationStatus.NOT_RUN
        ):
            try:
                outputs = outputs_from_finish(
                    result.result,
                    producer=spec.agent_id,
                    ports=spec.outputs,
                    observation_epoch=self.safety_state.observation_epoch,
                    artifact_dir=artifact_dir,
                )
                outputs = add_grounding_overlay(
                    outputs,
                    artifact_dir=artifact_dir,
                    fallback_rgb_path=context.scene_image_path,
                )
                outputs = _project_runtime_terminal_state(outputs, self.safety_state)
            except ValueError as exc:
                error = str(exc)
        ok = result.ok and not error
        runtime_budget = (
            result.trace.metadata.get("runtime_budget", {})
            if result.trace is not None
            else {}
        )
        failure_kind = "" if ok else declared_failure_kind(result.result)
        # M2: a contract that declares exit_conditions bounds what this node
        # may emit. An undeclared kind degrades to the generic failed event
        # instead of routing an edge the compiler never validated.
        if (
            failure_kind
            and contract.exit_conditions
            and failure_kind not in contract.exit_conditions
        ):
            emit_event(
                "undeclared_failure_kind_dropped",
                "Specialist emitted a failure kind outside its contract",
                agent_id=spec.agent_id,
                emitted=failure_kind,
                declared=sorted(contract.exit_conditions),
            )
            failure_kind = ""
        return AuthoringNodeResult(
            node_id=spec.agent_id,
            status=NodeStatus.SUCCEEDED if ok else NodeStatus.FAILED,
            outputs=outputs,
            evidence=packet,
            trace=result.trace,
            verification=verification,
            cost=AuthoringCost(
                llm_calls=int(runtime_budget.get("model_calls", result.turns)),
                action_turns=result.turns,
            ),
            error=error,
            failure_kind=failure_kind,
        )


def _project_runtime_terminal_state(
    outputs: tuple[TypedArtifact, ...],
    safety_state: RuntimeSafetyState,
) -> tuple[TypedArtifact, ...]:
    """Stamp runtime-captured robot state onto execution evidence.

    The MotionLeaseGuard records post-motion proprioception on behalf of the
    executor (which is denied ``perception_read``). Only a capture from the
    current observation epoch is projected; anything older belongs to a
    previous motion and would be misleading evidence.
    """

    captured = safety_state.last_terminal_robot_state
    if not isinstance(captured, dict):
        return outputs
    if int(captured.get("observation_epoch", -1)) != safety_state.observation_epoch:
        return outputs
    return tuple(
        replace(
            artifact,
            payload={**artifact.payload, "terminal_robot_state": captured},
        )
        if artifact.schema == "robomex.execution_evidence.v1"
        else artifact
        for artifact in outputs
    )


def _outcome_status(
    *,
    terminal: bool,
    verification: VerificationStatus,
) -> AuthoringStatus:
    if not terminal:
        return AuthoringStatus.EXHAUSTED
    if verification == VerificationStatus.PASSED:
        return AuthoringStatus.SUCCEEDED
    if verification == VerificationStatus.FAILED:
        return AuthoringStatus.FAILED
    return AuthoringStatus.UNCERTAIN
