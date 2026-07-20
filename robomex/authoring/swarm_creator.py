"""Skill-conditioned manager for one subgoal specialist graph."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from robomex.authoring.adapters import SubAgentFactory
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
from robomex.core.coder import CompletionPolicy
from robomex.core.coder.action import (
    SkillEntry,
    build_skill_llm_content,
    render_available_skills,
)
from robomex.core.coder.turn_engine import TurnBudget, TurnEngine
from robomex.core.context import EvidencePacket
from robomex.core.events import emit_event
from robomex.core.sandbox import RuntimeSafetyState
from robomex.dysc.contracts import SkillContract, load_skill_contracts
from robomex.prompts.authoring import build_swarm_manager_prompt
from robomex.skills import SkillLibrary


class SubgoalSwarmManager:
    """Progressively load task guidance and dynamically compose one graph."""

    strategy = "dynamic_swarm"
    graph_name = "subgoal_mas_dynamic_v1"

    def __init__(
        self,
        *,
        policy: CompletionPolicy,
        library: SkillLibrary,
        factory: SubAgentFactory,
        capability_ceiling: frozenset[str],
        safety_state: RuntimeSafetyState,
        max_turns: int = 4,
        max_spawns: int = 0,
        max_protocol_errors: int = 3,
        max_finish_rejections: int = 0,
        environment: str = "",
        api_docs: str = "",
        contracts: dict[str, SkillContract] | None = None,
        graph_executor: SubgoalGraphExecutor | None = None,
    ) -> None:
        del max_spawns, max_finish_rejections
        self.policy = policy
        self.library = library
        self.factory = factory
        self.capability_ceiling = capability_ceiling
        self.safety_state = safety_state
        self.max_turns = max_turns
        self.max_protocol_errors = max_protocol_errors
        root = getattr(library, "root", None)
        self.contracts = (
            dict(contracts)
            if contracts is not None
            else (load_skill_contracts(root) if root is not None else {})
        )
        self.compiler = SubgoalGraphCompiler(self.contracts)
        self.executor = graph_executor or SubgoalGraphExecutor(
            factory=factory, safety_state=safety_state
        )
        self.system_prompt = build_swarm_manager_prompt(
            capability_ceiling=capability_ceiling,
            environment=environment,
            api_docs=api_docs,
        )

    def run(self, context: SubgoalAuthoringContext) -> SubgoalOutcome:
        task_skills = self._task_skills()
        specialist_catalog = self._specialist_catalog()
        root = self._root(context)
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        if not task_skills:
            return self._composition_failure(context, 0, task_skills)
        prompt: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._system_with_task_skills(task_skills),
            },
            {
                "role": "user",
                "content": self._initial_message(context, specialist_catalog),
            },
        ]
        engine = TurnEngine(
            self.policy,
            allowed_tools={"use_skill", "submit_graph"},
            budget=TurnBudget(
                max_model_calls=self.max_turns,
                max_protocol_errors=self.max_protocol_errors,
            ),
        )
        loaded_task_skills: list[str] = []
        while engine.can_call_model:
            request = list(prompt)
            step = engine.next(prompt)
            if step is None:
                break
            self._dump_llm(root, step.index, request, step.turn.raw)
            call = step.tool_call
            if call is None:
                engine.append_feedback(
                    prompt,
                    step.turn.error or "Use one task skill or submit one dynamic graph.",
                )
                continue
            engine.record_action(call.name)
            if call.name == "use_skill":
                skill_id = str(call.args.get("name") or "")
                record = task_skills.get(skill_id)
                if record is None:
                    engine.append_feedback(
                        prompt,
                        "Unknown task skill. Load one exact name from available_task_skills.",
                    )
                    continue
                if skill_id not in loaded_task_skills:
                    loaded_task_skills.append(skill_id)
                content = build_skill_llm_content(
                    getattr(record.skill, "root", None),
                    record.skill.body,
                )
                engine.append_feedback(
                    prompt,
                    "The following task skill is guidance, not a fixed graph. Dynamically "
                    "compose specialists for the current evidence and subgoal.\n\n"
                    f"{content}",
                )
                emit_event(
                    "task_skill_loaded",
                    "Manager progressively loaded task guidance",
                    task_skill=skill_id,
                    loaded_task_skills=list(loaded_task_skills),
                )
                continue
            task_skill = str(call.args.get("task_skill") or "")
            graph_draft = call.args.get("graph")
            if task_skill not in loaded_task_skills:
                engine.append_feedback(
                    prompt,
                    "submit_graph requires task_skill to name a task skill loaded earlier "
                    "with use_skill.",
                )
                continue
            if not isinstance(graph_draft, dict):
                engine.append_feedback(
                    prompt,
                    "submit_graph requires args.graph to be one JSON object.",
                )
                continue
            selected_specialists = {
                str(node.get("skill") or "")
                for node in graph_draft.get("nodes", ())
                if isinstance(node, dict)
            }
            unavailable = selected_specialists - set(specialist_catalog)
            if unavailable:
                engine.append_feedback(
                    prompt,
                    "Graph uses unavailable specialist skill(s): "
                    f"{', '.join(sorted(unavailable))}. Choose only from specialist_contracts.",
                )
                continue
            try:
                graph = self.compiler.compile(
                    graph_draft,
                    task_skill=task_skill,
                    goal=context.goal,
                    postcondition=context.postcondition,
                )
            except ValueError as exc:
                engine.append_feedback(
                    prompt,
                    "Dynamic graph validation failed. Repair the graph and resubmit the "
                    f"complete graph: {exc}",
                )
                continue
            self._dump_graph(root, graph_draft, graph)
            emit_event(
                "subgoal_graph_composed",
                "Manager composed and validated a dynamic specialist graph",
                task_skill=task_skill,
                node_count=len(graph.nodes),
                loaded_task_skills=list(loaded_task_skills),
            )
            outcome = self.executor.run(graph, context)
            manager_result = AuthoringNodeResult(
                node_id="swarm_manager",
                status=NodeStatus.SUCCEEDED,
                evidence=EvidencePacket.from_any(
                    {
                        "claim": f"composed dynamic graph from task skill {task_skill}",
                        "evidence": {
                            "task_skill": task_skill,
                            "loaded_task_skills": loaded_task_skills,
                            "graph_nodes": [node.node_id for node in graph.nodes],
                        },
                    },
                    default_source="swarm_manager",
                    default_turn=f"subgoal:{context.subgoal_index}",
                ),
                cost=AuthoringCost(llm_calls=engine.ledger.model_calls),
            )
            return replace(
                outcome,
                node_results=tuple((manager_result, *outcome.node_results)),
                cost=outcome.cost + manager_result.cost,
                creator_status="composed_graph",
            )
        return self._composition_failure(context, engine.ledger.model_calls, task_skills)

    def _task_skills(self) -> dict[str, Any]:
        task_skills = getattr(self.library, "task_skills", lambda: ())()
        return {record.skill_id: record for record in task_skills}

    def _specialist_catalog(self) -> dict[str, dict[str, Any]]:
        records = {
            record.skill_id: record
            for record in getattr(self.library, "all", lambda: ())()
        }
        catalog: dict[str, dict[str, Any]] = {}
        for skill_id, contract in sorted(self.contracts.items()):
            if not contract.role:
                continue
            if set(contract.capabilities) - set(self.capability_ceiling):
                continue
            record = records.get(skill_id)
            entry = {
                "name": record.skill.name if record is not None else skill_id,
                "description": record.skill.description if record is not None else "",
                "role": contract.role,
                "changes_world": contract.changes_world,
                "capabilities": list(contract.capabilities),
                "inputs": [port.to_mapping() for port in contract.input_ports],
                "outputs": [port.to_mapping() for port in contract.output_ports],
            }
            # Budget as declared data (M1.5 Fix E): the Manager sees what the
            # skill author recommends; it never re-authors the enforced budget.
            if contract.budget:
                entry["budget"] = dict(contract.budget)
            recommended = (
                record.skill.recommended_min_actions if record is not None else 0
            )
            if recommended:
                entry["recommended_min_actions"] = recommended
            # M2 contract face: declared exit events (with semantics) tell the
            # Manager which recovery edges are routable from this node, and
            # canonical function signatures document the primitives its leaf
            # agent will have bound in the sandbox.
            if contract.exit_conditions:
                entry["exit_conditions"] = dict(contract.exit_conditions)
            if contract.functions:
                skill_root = getattr(record.skill, "root", None) if record else None
                entry["functions"] = [
                    self._function_entry(function, skill_root)
                    for function in contract.functions
                ]
            catalog[skill_id] = entry
        return catalog

    @staticmethod
    def _function_entry(function: Any, skill_root: Any) -> dict[str, str]:
        from robomex.dysc.contract_checks import resolve_function_signature

        signature = ""
        if skill_root is not None:
            signature = resolve_function_signature(function, Path(skill_root))
        entry = {
            "name": function.name,
            "signature": signature or f"{function.name}(...)",
        }
        if function.description:
            entry["description"] = function.description
        return entry

    def _system_with_task_skills(self, task_skills: dict[str, Any]) -> str:
        entries = [
            SkillEntry(
                name=skill_id,
                description=record.skill.description,
                category="task",
            )
            for skill_id, record in task_skills.items()
        ]
        block = render_available_skills(entries)
        reminder = (
            "<system-reminder>\n"
            "Task skills use progressive disclosure. Names and descriptions are only a "
            "catalog; load relevant guidance with use_skill before composing a graph.\n"
            f"<available_task_skills>\n{block}\n</available_task_skills>\n"
            "</system-reminder>"
        )
        return f"{self.system_prompt}\n\n{reminder}"

    @staticmethod
    def _initial_message(
        context: SubgoalAuthoringContext,
        specialist_catalog: dict[str, dict[str, Any]],
    ) -> str | list[dict[str, Any]]:
        payload = {
            "task": context.task,
            "subgoal_index": context.subgoal_index,
            "goal": context.goal,
            "postcondition": context.postcondition,
            "observation_summary": context.observation_summary,
            "specialist_contracts": specialist_catalog,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if context.scene_image_path:
            from robomex.perception.render import image_content_part

            return [
                {"type": "text", "text": text},
                image_content_part(context.scene_image_path),
            ]
        return text

    def _composition_failure(
        self,
        context: SubgoalAuthoringContext,
        model_calls: int,
        task_skills: dict[str, Any],
    ) -> SubgoalOutcome:
        note = (
            "Manager could not compose a valid dynamic specialist graph."
            if task_skills
            else "No high-level task skill is available for progressive disclosure."
        )
        manager = AuthoringNodeResult(
            node_id="swarm_manager",
            status=NodeStatus.EXHAUSTED,
            cost=AuthoringCost(llm_calls=model_calls),
            error=note,
        )
        return SubgoalOutcome(
            graph_name=self.graph_name,
            status=AuthoringStatus.EXHAUSTED,
            verification=VerificationStatus.NOT_RUN,
            node_results=(manager,),
            artifacts=(),
            cost=manager.cost,
            terminal_node="swarm_manager",
            note=note,
            strategy=self.strategy,
            creator_status="exhausted",
        )

    @staticmethod
    def _root(context: SubgoalAuthoringContext) -> Path | None:
        if context.artifact_dir is None:
            return None
        return context.artifact_dir / "authoring" / "swarm" / "manager_trace"

    @staticmethod
    def _dump_llm(
        root: Path | None,
        turn: int,
        request: list[dict[str, Any]],
        response: str,
    ) -> None:
        if root is None:
            return
        io_dir = root / "llm_io"
        io_dir.mkdir(parents=True, exist_ok=True)
        (io_dir / f"turn_{turn:02d}_request.json").write_text(
            json.dumps({"messages": request}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (io_dir / f"turn_{turn:02d}_response.txt").write_text(response, encoding="utf-8")

    @staticmethod
    def _dump_graph(root: Path | None, draft: dict[str, Any], graph: Any) -> None:
        if root is None:
            return
        (root / "submitted_graph.json").write_text(
            json.dumps(draft, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (root / "compiled_graph.json").write_text(
            json.dumps(graph.to_mapping(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

def _embedded_input_bindings(raw_inputs: Any) -> dict[str, dict[str, str]]:
    """Compatibility helper retained for reading older traces, not runtime spawning."""

    if isinstance(raw_inputs, dict):
        return {
            str(name): dict(value)
            for name, value in raw_inputs.items()
            if isinstance(value, dict) and "$ref" in value
        }
    if not isinstance(raw_inputs, (list, tuple)):
        return {}
    return {
        str(item.get("name") or ""): dict(item)
        for item in raw_inputs
        if isinstance(item, dict) and item.get("name") and "$ref" in item
    }
