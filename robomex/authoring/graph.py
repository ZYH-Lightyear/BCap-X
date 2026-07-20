"""Typed, skill-conditioned execution graph for one embodied subgoal."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from robomex.authoring.artifacts import ArtifactKey
from robomex.authoring.swarm_spec import SpecialistSpec
from robomex.dysc.contracts import SkillContract


# Re-exported here because the graph layer is where most callers touch the
# vocabulary; the definition lives in ``robomex.core.edge_events`` so that
# prompts can share it without import cycles.
from robomex.core.edge_events import (  # noqa: F401  (re-export)
    EDGE_EVENT_SUCCESS,
    EDGE_EVENTS,
    FAILURE_KINDS,
    KNOWN_EDGE_EVENTS,
    RUNTIME_EDGE_EVENTS,
    normalize_edge_event,
)


class GraphNodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    on: str = "success"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "GraphEdge":
        return cls(
            source=str(raw.get("from") or raw.get("source") or ""),
            target=str(raw.get("to") or raw.get("target") or ""),
            # PyYAML 1.1 may decode an unquoted ``on`` key as boolean True.
            on=normalize_edge_event(raw.get("on") or raw.get(True) or "success"),
        )

    def to_mapping(self) -> dict[str, str]:
        return {"from": self.source, "to": self.target, "on": self.on}


@dataclass(frozen=True)
class SubgoalNodeSpec:
    node_id: str
    specialist: SpecialistSpec
    bindings: dict[str, dict[str, str]] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    checkpoint: str = "none"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "specialist": self.specialist.to_mapping(),
            "bindings": dict(self.bindings),
            "params": dict(self.params),
            "checkpoint": self.checkpoint,
        }


@dataclass(frozen=True)
class SubgoalGraphSpec:
    task_skill: str
    variant: str
    nodes: tuple[SubgoalNodeSpec, ...]
    edges: tuple[GraphEdge, ...]
    entry: str
    success_node: str
    max_recoveries: int = 2

    @property
    def node_map(self) -> dict[str, SubgoalNodeSpec]:
        return {node.node_id: node for node in self.nodes}

    def validate(self) -> None:
        nodes = self.node_map
        if len(nodes) != len(self.nodes):
            raise ValueError("Subgoal graph contains duplicate node ids.")
        if self.entry not in nodes or self.success_node not in nodes:
            raise ValueError("Subgoal graph entry and success node must be declared.")
        edge_keys: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.source not in nodes or edge.target not in nodes:
                raise ValueError(
                    f"Graph edge {edge.source!r}->{edge.target!r} references an unknown node."
                )
            if edge.on not in KNOWN_EDGE_EVENTS:
                raise ValueError(
                    f"Graph edge {edge.source!r}->{edge.target!r} uses unknown event "
                    f"{edge.on!r}; declared edge events are: {', '.join(EDGE_EVENTS)}."
                )
            routable = self._routable_events(nodes[edge.source].specialist)
            if edge.on not in routable:
                raise ValueError(
                    f"Graph edge {edge.source!r}->{edge.target!r} listens for "
                    f"{edge.on!r}, but the {nodes[edge.source].specialist.specialist_skill!r} "
                    "contract never emits it — the edge would be permanently dead. "
                    f"Events routable from this node: {', '.join(sorted(routable))}."
                )
            key = (edge.source, edge.on)
            if key in edge_keys:
                raise ValueError(
                    f"Graph node {edge.source!r} has multiple edges for event {edge.on!r}."
                )
            edge_keys.add(key)
        success_targets = {
            edge.source: edge.target for edge in self.edges if edge.on == "success"
        }
        missing_success = [
            node.node_id
            for node in self.nodes
            if node.node_id != self.success_node and node.node_id not in success_targets
        ]
        if missing_success:
            raise ValueError(
                "Non-terminal graph nodes require one success edge: "
                f"{', '.join(missing_success)}."
            )
        for node in self.nodes:
            specialist = node.specialist
            has_world_capability = bool(
                specialist.requested_capabilities.intersection(
                    {"robot_motion", "gripper_control", "object_manipulation"}
                )
            )
            if specialist.changes_world or has_world_capability:
                if specialist.role != "action_executor":
                    raise ValueError(
                        f"Node {node.node_id!r} changes world state outside ActionExecutor."
                    )
                target = self.node_map.get(success_targets.get(node.node_id, ""))
                if target is None or not target.specialist.verifier:
                    raise ValueError(
                        f"World-changing node {node.node_id!r} must flow directly to a verifier."
                    )
            if specialist.verifier and (
                specialist.changes_world or has_world_capability
            ):
                raise ValueError("Verifier nodes must be strictly read-only.")
        output_schemas = {
            (node.node_id, port.name): port.schema
            for node in self.nodes
            for port in node.specialist.outputs
        }
        for node in self.nodes:
            declared_inputs = {port.name: port for port in node.specialist.inputs}
            for name, raw_ref in node.bindings.items():
                if name not in declared_inputs:
                    raise ValueError(f"Node {node.node_id!r} binds undeclared input {name!r}.")
                key = ArtifactKey.parse(raw_ref)
                schema = output_schemas.get((key.producer, key.port))
                if schema is None:
                    raise ValueError(
                        f"Node {node.node_id!r} consumes unpublished graph output {key.path!r}."
                    )
                if schema != declared_inputs[name].schema:
                    raise ValueError(
                        f"Node {node.node_id!r} input {name!r} expects "
                        f"{declared_inputs[name].schema!r}, got {schema!r}."
                    )
                if key.producer not in self._primary_ancestors(node.node_id):
                    raise ValueError(
                        f"Node {node.node_id!r} consumes {key.path!r} before its producer "
                        "on the primary success path."
                    )
            missing = [
                port.name
                for port in node.specialist.inputs
                if port.required and port.name not in node.bindings
            ]
            if missing:
                raise ValueError(
                    f"Node {node.node_id!r} lacks bindings for: {', '.join(missing)}."
                )
        terminal = nodes[self.success_node]
        if not terminal.specialist.verifier or terminal.checkpoint != "hard":
            raise ValueError("A successful subgoal graph must terminate at a hard verifier.")
        self._validate_primary_dag()

    def _validate_primary_dag(self) -> None:
        adjacency: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        for edge in self.edges:
            if edge.on == "success":
                adjacency[edge.source].append(edge.target)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("Primary subgoal graph contains a cycle.")
            if node_id in visited:
                return
            visiting.add(node_id)
            for target in adjacency[node_id]:
                visit(target)
            visiting.remove(node_id)
            visited.add(node_id)

        visit(self.entry)
        if set(adjacency) - visited:
            raise ValueError("Primary subgoal graph contains unreachable nodes.")

    @staticmethod
    def _routable_events(specialist: SpecialistSpec) -> frozenset[str]:
        """Events an edge from this node may listen for (M2 exit_conditions).

        A contract that declares ``exit_conditions`` constrains its edges to
        that subset plus the events the runtime itself can raise for any node
        (generic failure, exhaustion, uncertainty, stale inputs).  A contract
        without the declaration keeps the full closed vocabulary, so legacy
        skills keep composing unchanged.
        """

        if not specialist.exit_conditions:
            return KNOWN_EDGE_EVENTS
        return frozenset(specialist.exit_conditions) | frozenset(RUNTIME_EDGE_EVENTS)

    def _primary_ancestors(self, node_id: str) -> set[str]:
        reverse: dict[str, set[str]] = {node.node_id: set() for node in self.nodes}
        for edge in self.edges:
            if edge.on == "success":
                reverse[edge.target].add(edge.source)
        ancestors: set[str] = set()
        frontier = list(reverse[node_id])
        while frontier:
            current = frontier.pop()
            if current in ancestors:
                continue
            ancestors.add(current)
            frontier.extend(reverse[current] - ancestors)
        return ancestors

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "robomex.subgoal_graph.v1",
            "task_skill": self.task_skill,
            "variant": self.variant,
            "entry": self.entry,
            "success_node": self.success_node,
            "max_recoveries": self.max_recoveries,
            "nodes": [node.to_mapping() for node in self.nodes],
            "edges": [edge.to_mapping() for edge in self.edges],
        }


class SubgoalGraphCompiler:
    """Compile one Manager-authored graph using contracted specialist skills."""

    def __init__(self, contracts: dict[str, SkillContract]) -> None:
        self.contracts = contracts

    def compile(
        self,
        draft: dict[str, Any],
        *,
        task_skill: str,
        goal: str,
        postcondition: str,
        variant: str = "dynamic",
    ) -> SubgoalGraphSpec:
        """Resolve a dynamic graph draft against the authoritative contracts.

        The Manager owns topology, objectives and artifact bindings. Roles,
        ports, capabilities and budgets always come from the selected leaf
        contracts and cannot be authored by the model.
        """

        selected = dict(draft)
        raw_nodes = tuple(selected.get("nodes") or ())
        nodes: list[SubgoalNodeSpec] = []
        for raw in raw_nodes:
            raw = dict(raw)
            node_id = str(raw.get("id") or "")
            skill_id = str(raw.get("skill") or "")
            contract = self.contracts.get(skill_id)
            if contract is None:
                raise ValueError(
                    f"Dynamic graph references missing skill contract {skill_id!r}."
                )
            if not contract.role:
                raise ValueError(
                    f"Skill {skill_id!r} is reference-only and cannot own a graph node."
                )
            objective = _render(
                str(raw.get("objective") or f"Execute {skill_id} for {{goal}}."),
                goal,
                postcondition,
            )
            task = _render(str(raw.get("task") or objective), goal, postcondition)
            specialist = SpecialistSpec.from_contract(
                agent_id=node_id,
                contract=contract,
                objective=objective,
                task=task,
            )
            nodes.append(
                SubgoalNodeSpec(
                    node_id=node_id,
                    specialist=specialist,
                    bindings={
                        str(name): dict(value)
                        for name, value in dict(raw.get("inputs") or {}).items()
                    },
                    params=dict(raw.get("params") or {}),
                    checkpoint=str(raw.get("checkpoint") or "none"),
                )
            )
        graph = SubgoalGraphSpec(
            task_skill=task_skill,
            variant=variant,
            nodes=tuple(nodes),
            edges=tuple(GraphEdge.from_mapping(dict(edge)) for edge in selected.get("edges", ())),
            entry=str(selected.get("entry") or (nodes[0].node_id if nodes else "")),
            success_node=str(
                selected.get("success_node") or (nodes[-1].node_id if nodes else "")
            ),
            max_recoveries=max(0, int(selected.get("max_recoveries", 2))),
        )
        graph.validate()
        return graph


def _render(template: str, goal: str, postcondition: str) -> str:
    return template.replace("{goal}", goal).replace("{postcondition}", postcondition)
