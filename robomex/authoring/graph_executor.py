"""Deterministic runtime for one contracted subgoal specialist graph."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from robomex.authoring.adapters import SubAgentFactory
from robomex.authoring.artifacts import ArtifactStore, StaleArtifactError
from robomex.authoring.graph import (
    EDGE_EVENT_SUCCESS,
    GraphEdge,
    KNOWN_EDGE_EVENTS,
    SubgoalGraphSpec,
    normalize_edge_event,
)
from robomex.authoring.result import (
    AuthoringCost,
    AuthoringNodeResult,
    AuthoringStatus,
    NodeStatus,
    SubgoalAuthoringContext,
    SubgoalOutcome,
    VerificationStatus,
)
from robomex.core.context import EvidencePacket
from robomex.core.events import emit_event
from robomex.core.sandbox import RuntimeSafetyState


class SubgoalGraphExecutor:
    """Run specialist nodes and route only along validated graph edges."""

    def __init__(
        self,
        *,
        factory: SubAgentFactory,
        safety_state: RuntimeSafetyState,
    ) -> None:
        self.factory = factory
        self.safety_state = safety_state

    def run(
        self,
        graph: SubgoalGraphSpec,
        context: SubgoalAuthoringContext,
    ) -> SubgoalOutcome:
        graph.validate()
        root = self._root(context)
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            (root / "subgoal_graph.json").write_text(
                json.dumps(graph.to_mapping(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        store = ArtifactStore(artifact_root=root, epoch_source=self.safety_state)
        entry_epoch = self.safety_state.observation_epoch
        results: list[AuthoringNodeResult] = []
        attempts: dict[str, int] = {}
        current = graph.entry
        recoveries = 0
        status = AuthoringStatus.FAILED
        verification = VerificationStatus.NOT_RUN
        terminal_evidence = EvidencePacket()
        note = ""

        max_visits = max(len(graph.nodes) + graph.max_recoveries * len(graph.nodes), 1)
        for visit in range(max_visits):
            node = graph.node_map[current]
            attempt = attempts.get(current, 0) + 1
            attempts[current] = attempt
            node_dir = (
                root / f"{visit:02d}_{current}_a{attempt}" if root is not None else None
            )
            if node_dir is not None:
                node_dir.mkdir(parents=True, exist_ok=True)
                (node_dir / "request.json").write_text(
                    json.dumps(node.to_mapping(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            emit_event(
                "subgoal_graph_node_start",
                "Specialist graph node started",
                node_id=current,
                specialist_skill=node.specialist.specialist_skill,
                attempt=attempt,
            )
            result: AuthoringNodeResult | None = None
            try:
                resolved = store.resolve(node.specialist.inputs, node.bindings)
                result = self.factory.run(
                    node.specialist,
                    context=context,
                    inputs=resolved,
                    artifact_dir=node_dir,
                )
                result = replace(result, attempt=attempt)
                if result.ok or (
                    node.specialist.verifier and bool(result.outputs)
                ):
                    store.publish(current, node.specialist.outputs, result.outputs)
            except Exception as exc:  # graph failures become routable node failures
                failure_kind = (
                    "stale_observation" if isinstance(exc, StaleArtifactError) else ""
                )
                result = (
                    AuthoringNodeResult(
                        node_id=current,
                        status=NodeStatus.FAILED,
                        error=str(exc),
                        attempt=attempt,
                        failure_kind=failure_kind,
                    )
                    if result is None
                    else replace(
                        result,
                        status=NodeStatus.FAILED,
                        error=str(exc),
                        attempt=attempt,
                        failure_kind=failure_kind or result.failure_kind,
                    )
                )
            results.append(result)
            terminal_evidence = (
                result.evidence if not result.evidence.is_empty else terminal_evidence
            )
            self._append_event(root, node, result, store, graph.edges)

            if current == graph.success_node:
                verification = result.verification
                if result.ok and verification == VerificationStatus.PASSED:
                    status = AuthoringStatus.SUCCEEDED
                    note = "Hard verifier passed on the current observation epoch."
                    break

            event = self._exit_event(result)
            edge = self._edge(graph.edges, current, event)
            if (
                edge is None
                and event not in (EDGE_EVENT_SUCCESS, "uncertain")
            ):
                # A specific failure kind falls back to the generic failed edge.
                # `uncertain` never does (M1.5 Fix D): an unverified state is not
                # a failure, and silently rerouting it onto the failure edge would
                # replay state-changing nodes against a world the previous attempt
                # already changed. Routing uncertainty is the Manager's declared
                # decision (an explicit `on: uncertain` edge); absent one, the
                # graph exits honestly and the Planner decides with the evidence.
                edge = self._edge(graph.edges, current, "failed")
            if edge is None:
                if event == "uncertain":
                    status = AuthoringStatus.UNCERTAIN
                    verification = VerificationStatus.UNCERTAIN
                    note = (
                        result.error
                        or f"Node {current!r} exited uncertain; no `on: uncertain` "
                        "edge was declared, so the outcome is escalated with its "
                        "partial evidence instead of being retried as a failure."
                    )
                    break
                if result.status == NodeStatus.EXHAUSTED:
                    status = AuthoringStatus.EXHAUSTED
                elif result.verification == VerificationStatus.UNCERTAIN:
                    status = AuthoringStatus.UNCERTAIN
                else:
                    status = AuthoringStatus.FAILED
                verification = result.verification
                note = result.error or f"Node {current!r} exited with {event!r}."
                break
            if edge.on != EDGE_EVENT_SUCCESS:
                recoveries += 1
                if recoveries > graph.max_recoveries:
                    status = (
                        AuthoringStatus.UNCERTAIN
                        if result.verification == VerificationStatus.UNCERTAIN
                        else AuthoringStatus.FAILED
                    )
                    verification = result.verification
                    note = "Subgoal graph exhausted its declared recovery budget."
                    break
            current = edge.target
        else:
            status = AuthoringStatus.EXHAUSTED
            note = "Subgoal graph exhausted its node-visit budget."

        total_cost = AuthoringCost()
        for result in results:
            total_cost += result.cost
        outcome = SubgoalOutcome(
            graph_name=f"subgoal_mas:{graph.task_skill}:{graph.variant}",
            status=status,
            verification=verification,
            node_results=tuple(results),
            artifacts=store.values(),
            cost=total_cost,
            terminal_node=current,
            note=note,
            strategy="dynamic_swarm",
            creator_status="completed",
            motion_attempted=self.safety_state.observation_epoch > entry_epoch,
            terminal_evidence=terminal_evidence,
        )
        if root is not None:
            (root / "artifact_store.json").write_text(
                json.dumps(store.snapshot(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (root / "swarm_run_result.json").write_text(
                json.dumps(outcome.to_json_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return outcome

    @staticmethod
    def _edge(
        edges: tuple[GraphEdge, ...],
        source: str,
        event: str,
    ) -> GraphEdge | None:
        return next(
            (edge for edge in edges if edge.source == source and edge.on == event),
            None,
        )

    @staticmethod
    def _exit_event(result: AuthoringNodeResult) -> str:
        """Route on declared, structured signals only — never free text.

        Priority: clean success, then a specialist/runtime-declared
        ``failure_kind`` from the closed vocabulary, then verification
        status, then node status. Free-text claims and error messages are
        intentionally never inspected (GaP-style "declare, don't infer").
        """

        clean_success = result.ok and result.verification not in {
            VerificationStatus.FAILED,
            VerificationStatus.UNCERTAIN,
        }
        if clean_success:
            return EDGE_EVENT_SUCCESS
        kind = SubgoalGraphExecutor._declared_failure_kind(result)
        if kind:
            return kind
        if result.verification == VerificationStatus.FAILED:
            return "failed"
        if result.verification == VerificationStatus.UNCERTAIN:
            return "uncertain"
        if result.status == NodeStatus.EXHAUSTED:
            return "exhausted"
        return "failed"

    @staticmethod
    def _declared_failure_kind(result: AuthoringNodeResult) -> str:
        """Read the structured failure kind attached to one node result."""

        candidates: list[Any] = [result.failure_kind]
        if isinstance(result.evidence.evidence, dict):
            candidates.append(result.evidence.evidence.get("failure_kind"))
        if result.evidence.verdict is not None:
            candidates.append(result.evidence.verdict.metadata.get("failure_kind"))
        for raw in candidates:
            if not raw:
                continue
            event = normalize_edge_event(raw)
            if event in KNOWN_EDGE_EVENTS and event != EDGE_EVENT_SUCCESS:
                return event
        return ""

    @staticmethod
    def _root(context: SubgoalAuthoringContext) -> Path | None:
        if context.artifact_dir is None:
            return None
        return context.artifact_dir / "authoring" / "swarm"

    @staticmethod
    def _append_event(
        root: Path | None,
        node: Any,
        result: AuthoringNodeResult,
        store: ArtifactStore,
        edges: tuple[GraphEdge, ...],
    ) -> None:
        if root is None:
            return
        exit_event = SubgoalGraphExecutor._exit_event(result)
        recovery = SubgoalGraphExecutor._edge(edges, node.node_id, exit_event)
        if recovery is None and exit_event not in (EDGE_EVENT_SUCCESS, "uncertain"):
            # Mirror the routing rule: `uncertain` never borrows the failed edge.
            recovery = SubgoalGraphExecutor._edge(edges, node.node_id, "failed")
        payload = {
            "schema": "robomex.subgoal_node_exit.v1",
            "node_id": node.node_id,
            "specialist_skill": node.specialist.specialist_skill,
            "role": node.specialist.role,
            "required_skills": list(node.specialist.required_skills),
            "status": result.status.value,
            "verification": result.verification.value,
            "failure_kind": result.failure_kind,
            "attempt": result.attempt,
            "error": result.error,
            "resolved_inputs": dict(node.bindings),
            "published_outputs": [
                artifact.key.to_mapping() for artifact in result.outputs
            ]
            if result.ok
            else [],
            "observation_epoch": store.observation_epoch,
            "cost": result.cost.to_json_dict(),
            "exit_event": exit_event,
            "recovery_decision": None
            if recovery is None or recovery.on == EDGE_EVENT_SUCCESS
            else {"event": recovery.on, "target": recovery.target},
        }
        with (root / "node_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
