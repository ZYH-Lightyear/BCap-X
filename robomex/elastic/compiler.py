"""Compiler for the fixed-topology subset of Elastic Graph v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from robomex.elastic.graph_spec import (
    ActivationLane,
    ArtifactBinding,
    ElasticGraphSpec,
)
from robomex.runtime.events import ControlOutcome


@dataclass(frozen=True)
class CompiledElasticGraph:
    spec: ElasticGraphSpec
    digest: str
    transition_table: dict[tuple[str, ControlOutcome], str]
    predecessors: dict[str, frozenset[str]]
    loop_by_activation: dict[str, str]
    loop_entries: dict[str, str]
    loop_limits: dict[str, int]

    def next_activation(self, source: str, outcome: ControlOutcome) -> str | None:
        return self.transition_table.get((source, outcome))


class ElasticGraphCompiler:
    """Validate a v2 graph without importing any v1 graph assumptions."""

    def compile(self, spec: ElasticGraphSpec | dict) -> CompiledElasticGraph:
        graph = spec if isinstance(spec, ElasticGraphSpec) else ElasticGraphSpec.model_validate(spec)
        node_map = {node.activation_id: node for node in graph.activations}
        if len(node_map) != len(graph.activations):
            raise ValueError("Elastic graph contains duplicate activation ids.")
        primary_ids = {
            node.activation_id
            for node in graph.activations
            if node.lane == ActivationLane.PRIMARY
        }
        service_ids = set(node_map) - primary_ids
        if graph.entry_activation not in primary_ids:
            raise ValueError("Graph entry must name a primary activation.")
        terminals = set(graph.terminal_activations)
        if len(terminals) != len(graph.terminal_activations):
            raise ValueError("Elastic graph contains duplicate terminal activations.")
        if not terminals.issubset(primary_ids):
            raise ValueError("Every terminal activation must be declared on the primary lane.")

        table: dict[tuple[str, ControlOutcome], str] = {}
        predecessors: dict[str, set[str]] = {node_id: set() for node_id in primary_ids}
        outgoing: dict[str, set[str]] = {node_id: set() for node_id in primary_ids}
        for transition in graph.transitions:
            if transition.source in service_ids or transition.target in service_ids:
                raise ValueError("Service activations use subscriptions, not primary transitions.")
            if transition.source not in primary_ids or transition.target not in primary_ids:
                raise ValueError(
                    f"Transition {transition.source!r}->{transition.target!r} references "
                    "an unknown primary activation."
                )
            key = (transition.source, transition.outcome)
            if key in table:
                raise ValueError(
                    f"Activation {transition.source!r} has multiple primary continuations "
                    f"for {transition.outcome.value!r}."
                )
            table[key] = transition.target
            predecessors[transition.target].add(transition.source)
            outgoing[transition.source].add(transition.target)
        terminal_outgoing = terminals.intersection(node_id for node_id, edges in outgoing.items() if edges)
        if terminal_outgoing:
            raise ValueError(
                "Terminal activations cannot have outgoing transitions: "
                f"{', '.join(sorted(terminal_outgoing))}."
            )
        missing_outgoing = primary_ids - terminals - {
            node_id for node_id, edges in outgoing.items() if edges
        }
        if missing_outgoing:
            raise ValueError(
                "Non-terminal primary activations require a continuation: "
                f"{', '.join(sorted(missing_outgoing))}."
            )
        reachable = self._reachable(graph.entry_activation, outgoing)
        if primary_ids - reachable:
            raise ValueError(
                "Elastic graph contains unreachable primary activations: "
                f"{', '.join(sorted(primary_ids - reachable))}."
            )
        if not terminals.intersection(reachable):
            raise ValueError("No terminal activation is reachable from the graph entry.")

        loop_by_activation = self._validate_loops(graph, outgoing, primary_ids)
        dominators = self._dominators(
            entry=graph.entry_activation,
            node_ids=primary_ids,
            predecessors=predecessors,
        )
        self._validate_bindings(graph, node_map, dominators)

        canonical = json.dumps(
            graph.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        return CompiledElasticGraph(
            spec=graph,
            digest=digest,
            transition_table=table,
            predecessors={key: frozenset(value) for key, value in predecessors.items()},
            loop_by_activation=loop_by_activation,
            loop_entries={loop.loop_id: loop.entry_activation for loop in graph.bounded_loops},
            loop_limits={loop.loop_id: loop.max_iterations for loop in graph.bounded_loops},
        )

    @staticmethod
    def _reachable(entry: str, outgoing: dict[str, set[str]]) -> set[str]:
        reached: set[str] = set()
        frontier = [entry]
        while frontier:
            current = frontier.pop()
            if current in reached:
                continue
            reached.add(current)
            frontier.extend(outgoing[current] - reached)
        return reached

    def _validate_loops(
        self,
        graph: ElasticGraphSpec,
        outgoing: dict[str, set[str]],
        primary_ids: set[str],
    ) -> dict[str, str]:
        declared: dict[frozenset[str], str] = {}
        loop_by_activation: dict[str, str] = {}
        loop_ids = [loop.loop_id for loop in graph.bounded_loops]
        if len(loop_ids) != len(set(loop_ids)):
            raise ValueError("Elastic graph contains duplicate bounded-loop ids.")
        for loop in graph.bounded_loops:
            members = frozenset(loop.activation_ids)
            unknown = members - primary_ids
            if unknown:
                raise ValueError(
                    f"Loop {loop.loop_id!r} references unknown primary activations: "
                    f"{', '.join(sorted(unknown))}."
                )
            if members in declared:
                raise ValueError("Two bounded-loop declarations cover the same activation set.")
            overlap = set(loop_by_activation).intersection(members)
            if overlap:
                raise ValueError(
                    "Bounded loops cannot overlap in the fixed-topology v2 profile: "
                    f"{', '.join(sorted(overlap))}."
                )
            declared[members] = loop.loop_id
            loop_by_activation.update(dict.fromkeys(members, loop.loop_id))

        cyclic_components = [
            component
            for component in self._strongly_connected_components(primary_ids, outgoing)
            if len(component) > 1 or any(node in outgoing[node] for node in component)
        ]
        for component in cyclic_components:
            frozen = frozenset(component)
            if frozen not in declared:
                raise ValueError(
                    "Every control-flow cycle must exactly match one declared bounded loop; "
                    f"undeclared component: {', '.join(sorted(component))}."
                )
            if not any(
                target not in component
                for source in component
                for target in outgoing[source]
            ):
                raise ValueError(
                    f"Bounded loop {declared[frozen]!r} has no continuation outside the loop."
                )
        unused = set(declared) - {frozenset(component) for component in cyclic_components}
        if unused:
            names = [declared[members] for members in unused]
            raise ValueError(
                "A bounded-loop declaration must describe an actual cyclic component: "
                f"{', '.join(sorted(names))}."
            )
        return loop_by_activation

    @staticmethod
    def _strongly_connected_components(
        node_ids: set[str], outgoing: dict[str, set[str]]
    ) -> list[set[str]]:
        index = 0
        indices: dict[str, int] = {}
        lowlinks: dict[str, int] = {}
        stack: list[str] = []
        on_stack: set[str] = set()
        components: list[set[str]] = []

        def visit(node_id: str) -> None:
            nonlocal index
            indices[node_id] = index
            lowlinks[node_id] = index
            index += 1
            stack.append(node_id)
            on_stack.add(node_id)
            for target in outgoing[node_id]:
                if target not in indices:
                    visit(target)
                    lowlinks[node_id] = min(lowlinks[node_id], lowlinks[target])
                elif target in on_stack:
                    lowlinks[node_id] = min(lowlinks[node_id], indices[target])
            if lowlinks[node_id] != indices[node_id]:
                return
            component: set[str] = set()
            while stack:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == node_id:
                    break
            components.append(component)

        for node_id in sorted(node_ids):
            if node_id not in indices:
                visit(node_id)
        return components

    @staticmethod
    def _dominators(
        *,
        entry: str,
        node_ids: set[str],
        predecessors: dict[str, set[str]],
    ) -> dict[str, set[str]]:
        dominators = {
            node_id: ({entry} if node_id == entry else set(node_ids))
            for node_id in node_ids
        }
        changed = True
        while changed:
            changed = False
            for node_id in sorted(node_ids - {entry}):
                preds = predecessors[node_id]
                common = set.intersection(*(dominators[pred] for pred in preds)) if preds else set()
                updated = {node_id} | common
                if updated != dominators[node_id]:
                    dominators[node_id] = updated
                    changed = True
        return dominators

    @staticmethod
    def _validate_bindings(graph, node_map, dominators) -> None:
        for node in graph.activations:
            input_map = {port.name: port for port in node.inputs}
            for binding in node.bindings:
                expected = input_map[binding.input_port]
                if not isinstance(binding, ArtifactBinding):
                    if binding.schema_id != expected.schema_id:
                        raise ValueError(
                            f"External binding {binding.ref!r} for {node.activation_id}."
                            f"{binding.input_port} declares {binding.schema_id!r}, expected "
                            f"{expected.schema_id!r}."
                        )
                    continue
                producer = node_map.get(binding.source_activation)
                if producer is None:
                    raise ValueError(
                        f"Activation {node.activation_id!r} binds output from unknown producer "
                        f"{binding.source_activation!r}."
                    )
                output_map = {port.name: port for port in producer.outputs}
                output = output_map.get(binding.source_port)
                if output is None:
                    raise ValueError(
                        f"Activation {node.activation_id!r} binds missing output "
                        f"{binding.source_activation}.{binding.source_port}."
                    )
                if output.schema_id != expected.schema_id:
                    raise ValueError(
                        f"Activation {node.activation_id!r} input {binding.input_port!r} "
                        f"expects {expected.schema_id!r}, got {output.schema_id!r}."
                    )
                if node.lane == ActivationLane.PRIMARY:
                    if producer.lane != ActivationLane.PRIMARY:
                        raise ValueError(
                            "Primary inputs from services must use a typed stream/external ref; "
                            "a service completion cannot dominate the primary control token."
                        )
                    if producer.activation_id not in dominators[node.activation_id]:
                        raise ValueError(
                            f"Producer {producer.activation_id!r} is not available on every path "
                            f"to consumer {node.activation_id!r}; insert an explicit selector/join."
                        )
