"""CodingAgent-backed SubAgents for focused RoboMEx inner-loop analyses."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from robomex.core.coder import CodingAgent, CompletionPolicy, SkillEntry, parse_action_payload
from robomex.core.context import (
    AttemptRecord,
    Diagnosis,
    EvidencePacket,
    LocalVerdict,
    PrimitiveTrace,
    StatePatch,
    WorkspaceArtifact,
    compact_json,
)
from robomex.core.coder.trace import AgentTrace, TurnRecord
from robomex.core.sandbox import (
    ActionBlockStatus,
    BlockExecutionResult,
    SemanticActionBlock,
)
from robomex.skills import (
    SkillLibrary,
)


@dataclass(frozen=True)
class SubAgentRequest:
    """One focused task delegated from Act to a SubAgent."""

    task: str
    inputs: dict[str, Any] = field(default_factory=dict)
    artifacts_dir: str | None = None
    task_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubAgentResult:
    """Small JSON-serializable result returned to Act."""

    name: str
    ok: bool
    result_type: str = ""
    claim: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    state_patch: StatePatch = field(default_factory=StatePatch)
    local_verdict: LocalVerdict | None = None
    artifact_refs: tuple[WorkspaceArtifact, ...] = ()
    trace_refs: tuple[str, ...] = ()
    primitive_traces: tuple[PrimitiveTrace, ...] = ()
    attempt_records: tuple[AttemptRecord, ...] = ()
    diagnoses: tuple[Diagnosis, ...] = ()
    evidence_packet: EvidencePacket = field(default_factory=EvidencePacket)
    uncertainty: tuple[str, ...] = ()
    recommended_next: str = ""
    loaded_skill_ids: tuple[str, ...] = ()
    turns: int = 0
    error: str = ""
    artifacts_dir: str | None = None
    task_id: str | None = None

    def resolved_evidence_packet(self) -> EvidencePacket:
        """Return the generic cross-agent evidence envelope for this result.

        Older callers may still fill ``result`` / ``local_verdict`` /
        ``state_patch`` directly.  Keep that path compatible by projecting those
        fields into an EvidencePacket when no explicit packet was supplied.
        """

        if not self.evidence_packet.is_empty:
            return self.evidence_packet
        payload: dict[str, Any] = {
            **self.result,
            "claim": self.claim or self.result.get("claim", ""),
            "confidence": self.result.get("confidence", 0.0),
            "state_patch": self.state_patch.to_json_dict(),
            "artifact_refs": [a.to_json_dict() for a in self.artifact_refs],
            "uncertainty": list(self.uncertainty),
            "recommended_next": self.recommended_next or str(self.result.get("recommended_next", "")),
        }
        if self.local_verdict is not None:
            payload["local_verdict"] = self.local_verdict.to_json_dict()
        return EvidencePacket.from_any(
            payload,
            default_source=self.name,
            default_turn=f"subagent:{self.task_id or ''}",
        )

    def to_json_dict(self) -> dict[str, Any]:
        packet = self.resolved_evidence_packet()
        return {
            "name": self.name,
            "ok": self.ok,
            "result_type": self.result_type,
            "claim": self.claim,
            "result": compact_json(self.result),
            "state_patch": self.state_patch.to_json_dict(),
            "local_verdict": None if self.local_verdict is None else self.local_verdict.to_json_dict(),
            "artifact_refs": [a.to_json_dict() for a in self.artifact_refs],
            "trace_refs": list(self.trace_refs),
            "primitive_traces": [t.to_json_dict() for t in self.primitive_traces],
            "attempt_records": [a.to_json_dict() for a in self.attempt_records],
            "diagnoses": [d.to_json_dict() for d in self.diagnoses],
            "evidence_packet": packet.to_json_dict(),
            "uncertainty": list(self.uncertainty),
            "recommended_next": self.recommended_next,
            "loaded_skill_ids": list(self.loaded_skill_ids),
            "turns": self.turns,
            "error": self.error,
            "artifacts_dir": self.artifacts_dir,
            "task_id": self.task_id,
        }


def write_subagent_result_artifact(request: SubAgentRequest, result: SubAgentResult) -> None:
    """Persist a stable per-SubAgent result manifest when an artifact dir is available."""

    if not request.artifacts_dir:
        return
    d = Path(request.artifacts_dir)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "robomex.subagent_result.v1",
        "request": {
            "task": request.task,
            "inputs": compact_json(request.inputs),
            "artifacts_dir": request.artifacts_dir,
            "task_id": request.task_id,
            "context": compact_json(request.context),
        },
        "result": result.to_json_dict(),
    }
    (d / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class SubAgent(Protocol):
    name: str

    def run(self, request: SubAgentRequest) -> SubAgentResult: ...


class SubAgentRegistry:
    """Runtime holder for the generic CodingAgentSubAgent."""

    def __init__(
        self,
        *,
        default_agent: SubAgent | None = None,
    ) -> None:
        self._default_agent = default_agent

    def set_default(self, agent: SubAgent) -> None:
        self._default_agent = agent

    def has_runtime(self) -> bool:
        return self._default_agent is not None

    def run(self, request: SubAgentRequest) -> SubAgentResult:
        if self._default_agent is not None:
            return self._default_agent.run(request)
        raise RuntimeError("No SubAgent runtime is configured.")


@dataclass(frozen=True)
class SubAgentExecutionPolicy:
    """Execution limits for CodingAgentSubAgent code blocks.

    SubAgents are task-first agents, not named workflows. This policy describes
    the runtime boundary they operate inside. The default policy keeps them
    read-only for robot control while still allowing live perception, geometry,
    IK checks, scoring, and artifact writes.
    """

    denied_calls: frozenset[str] = field(default_factory=lambda: frozenset({
        "close_gripper",
        "execute_joint_trajectory",
        "goto_home_joint_position",
        "goto_pose",
        "move_to_joints",
        "open_gripper",
        "reset",
        "step",
    }))

class PolicyBoundBlockExecutor:
    """Executor wrapper that enforces a SubAgentExecutionPolicy."""

    def __init__(self, inner: Any, policy: SubAgentExecutionPolicy | None = None) -> None:
        self.inner = inner
        self.policy = policy or SubAgentExecutionPolicy()

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        blocked = self._denied_names(block.code)
        if blocked:
            reason = (
                "SubAgent sandbox is read-only for robot execution. "
                f"Blocked motion/control call(s): {', '.join(blocked)}."
            )
            return BlockExecutionResult(
                block=block,
                ok=False,
                status=ActionBlockStatus.SKIPPED,
                stdout="",
                stderr=reason,
                info={"blocked_calls": blocked, "subagent_guard": "read_only"},
            )
        return self.inner.run_block(block)

    def _denied_names(self, code: str) -> tuple[str, ...]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ()
        names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                candidate = func.id
            elif isinstance(func, ast.Attribute):
                candidate = func.attr
            else:
                continue
            if candidate in self.policy.denied_calls:
                names.add(candidate)
        return tuple(sorted(names))

_SUBAGENT_SYSTEM_PROMPT = (
    "You are the RoboMEx Verifier SubAgent. You solve one focused read-only "
    "verification or diagnosis task for the Act Agent. Each turn, reply with exactly "
    "one JSON action object and no extra text, using use_skill, run_python, or finish. "
    "Stay in your lane: answer the delegated state/claim question, then finish. Do "
    "not decompose or solve the whole robot sub-goal, do not plan execution, do not "
    "generate grounding outputs, affordance candidates, placement points, or motion "
    "plans, and do not decide final robot action. You must not execute robot motion "
    "or gripper commands. You may inspect current observations, run read-only visual "
    "QA, segmentation for checking a claim, geometry measurement for diagnosis, and "
    "save overlays in ARTIFACTS_DIR. Treat request inputs such as bbox, object center, "
    "candidate pose, or prior labels as claims to verify, not ground truth. If a prior "
    "bbox or pose is stale, say so clearly instead of reusing it as the target. "
    "query_vlm is allowed for categorical visual judgment; never ask it for boxes, "
    "points, or coordinates. Use dedicated perception APIs only when they help verify "
    "the current state. Keep arrays and masks in EVIDENCE or artifacts, not in final "
    "JSON. If a verification task needs skill sidecar scripts, references, or assets, "
    "use only the entry points and usage patterns named in the loaded skill. Do not "
    "read or print a whole sidecar, and do not spend turns printing obs.keys(), image "
    "shapes, or inspect.signature unless a concrete error requires one short diagnostic. "
    "EVIDENCE is local to this Verifier and invisible to Act. "
    "Return a compact verdict only: pass/fail/uncertain, confidence, short evidence, "
    "artifact paths, uncertainty, and recommended_next. Do not return state_patch or "
    "persistent world facts. Finish with "
    "{\"tool\":\"finish\",\"args\":{\"claim\":\"...\",\"result\":{\"confidence\":0.0,"
    "\"evidence\":{},\"verdict\":{\"status\":\"pass|fail|uncertain\","
    "\"reason\":\"...\"},\"artifacts\":[],\"uncertainty\":[],"
    "\"recommended_next\":\"...\"}}}."
)


def render_subagent_system_prompt(api_docs: str = "") -> str:
    """Render the default SubAgent prompt with sandbox API docs appended."""

    docs = api_docs.strip()
    if not docs:
        return _SUBAGENT_SYSTEM_PROMPT
    return (
        f"{_SUBAGENT_SYSTEM_PROMPT}\n\n"
        "Available sandbox API functions (already imported):\n"
        f"{docs}"
    )


class CodingAgentSubAgent(CodingAgent):
    """Default SubAgent implementation: a focused, read-only verifier."""

    name = "verifier_subagent"
    description = "Read-only verifier/diagnoser for Act-owned robot execution."

    def __init__(
        self,
        executor: Any,
        policy: CompletionPolicy,
        library: SkillLibrary,
        *,
        max_turns: int = 8,
        system_prompt: str = _SUBAGENT_SYSTEM_PROMPT,
        execution_policy: SubAgentExecutionPolicy | None = None,
    ) -> None:
        super().__init__(
            executor=PolicyBoundBlockExecutor(executor, execution_policy),
            policy=policy,
            library=library,
            max_turns=max_turns,
            system_prompt=system_prompt,
            force_terminal_on_exhaust=True,
        )
        self._request = SubAgentRequest(task="")
        self._turn_records: list[TurnRecord] = []

    def run(self, request: SubAgentRequest) -> SubAgentResult:
        self._request = request
        self._turn_records = []
        return super().run()

    def _setup(self, prompt: list[dict]) -> None:
        art_dir = self._request.artifacts_dir or "/tmp"
        Path(art_dir).mkdir(parents=True, exist_ok=True)
        library_root = str(getattr(self.library, "root", ""))
        seed = (
            f"ARTIFACTS_DIR = {str(art_dir)!r}\n"
            f"SKILL_LIBRARY_ROOT = {library_root!r}\n"
            + "try:\n"
            "    EVIDENCE\n"
            "except NameError:\n"
            "    EVIDENCE = {}\n"
        )
        self.executor.run_block(
            SemanticActionBlock(
                name=f"{self.name}_seed",
                intent="seed subagent artifacts",
                code=seed,
                metadata={"subagent": self.name},
            )
        )

    def _skill_entries(self) -> list[SkillEntry]:
        return [
            SkillEntry(
                name=r.skill_id,
                description=r.skill.description or r.skill.name,
                category=r.skill.category.value,
            )
            for r in self.library.all()
        ]

    def _initial_user_message(self) -> str:
        payload = {
            "subagent": self.name,
            "task": self._request.task,
            "inputs": self._request.inputs,
            "artifacts_dir": self._request.artifacts_dir,
            "task_id": self._request.task_id,
            "context": self._request.context,
        }
        return (
            "Solve this focused Verifier request. Load a relevant skill only when it helps "
            "verify the claim or diagnose the current state. Do not move the robot. Do not "
            "take over Act's role: do not localize a new target for execution, do not "
            "generate grasp/placement candidates, and do not write recovery motion code. "
            "Treat request inputs as claims. Check them against the current observation and "
            "return pass/fail/uncertain with compact evidence and artifact paths. Finish as "
            "soon as the verification question is answered. Return finish.args.result as JSON.\n\n"
            "When the loaded skill provides sidecar scripts, references, or assets, use them "
            "only if they directly help verify the claim. Do not first probe observation keys, "
            "image shapes, or API signatures; the runtime already defines those contracts. "
            "Never return state_patch or persistent facts; Act owns all working memory and "
            "final decisions. EVIDENCE is only your local scratchpad and will not be visible "
            "to Act.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _agent_role(self) -> str:
        return "subagent"

    def _agent_label(self) -> str:
        label = _short_task_label(self._request.task) or self.name
        return f"SubAgent:{label}"

    def _llm_io_dir(self) -> Path | None:
        if not self._request.artifacts_dir:
            return None
        return Path(self._request.artifacts_dir) / "llm_io"

    def _block_metadata(self) -> dict:
        return {
            "subagent": self.name,
            "task": self._request.task,
        }

    def _on_python_turn(
        self,
        turn_idx: int,
        code: str,
        execution: BlockExecutionResult,
        prev_observation: dict | None,
        turns: list[Any],
    ) -> None:
        record = TurnRecord(turn_idx, code, execution)
        turns.append(record)
        self._turn_records.append(record)
        self._write_python_turn_artifacts(record)

    def _write_python_turn_artifacts(self, record: TurnRecord) -> None:
        if not self._request.artifacts_dir:
            return
        d = Path(self._request.artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"turn_{record.turn:02d}.py").write_text(record.code, encoding="utf-8")
        report = "\n".join(
            [
                f"# status    : {record.execution.status.value}",
                f"# ok        : {record.execution.ok}",
                f"# reward    : {record.execution.reward}",
                f"# terminated: {record.execution.terminated}",
                "",
                "## stdout",
                record.execution.stdout or "(empty)",
                "",
                "## stderr",
                record.execution.stderr or "(empty)",
            ]
        )
        (d / f"turn_{record.turn:02d}.out.txt").write_text(report, encoding="utf-8")

    def _feedback_message(self, execution: BlockExecutionResult) -> str:
        return (
            f"stdout:\n{execution.stdout}\n\nstderr:\n{execution.stderr}\n\n"
            "Continue the focused verification. Reply with exactly one JSON action: "
            "use_skill, run_python, or finish. If the delegated claim is answered, "
            "finish now with compact verdict JSON. Do not call robot motion APIs."
        )

    def _finalize(self, *, turns: list[Any], loaded: tuple[str, ...], terminal_raw: str | None) -> SubAgentResult:
        parsed = _parse_finish_result(terminal_raw)
        ok = bool(terminal_raw) and bool(parsed.get("ok", True))
        error = "" if terminal_raw else "SubAgent exhausted its budget before finish."
        if parsed.get("error"):
            error = str(parsed["error"])
        source_turn = f"subagent:{self._request.task_id or ''}"
        packet = EvidencePacket.from_any(parsed, default_source=self.name, default_turn=source_turn)
        patch = StatePatch()
        artifacts = _extract_artifact_refs(parsed, default_producer=self.name)
        if packet.artifact_refs:
            artifacts = _merge_artifacts(artifacts, packet.artifact_refs)
        primitive_traces = _extract_primitive_traces(parsed, default_producer=self.name)
        attempts = _extract_attempt_records(parsed)
        diagnoses = _extract_diagnoses(parsed)
        local_verdict = _extract_local_verdict(parsed) or packet.verdict
        result_type = str(parsed.get("result_type") or parsed.get("type") or "")
        trace_refs = tuple(str(v) for v in parsed.get("trace_refs", ()) or parsed.get("related_trace_ids", ()) or ())
        uncertainty = parsed.get("uncertainty", ())
        if isinstance(uncertainty, str):
            uncertainty = (uncertainty,)
        elif not isinstance(uncertainty, (list, tuple)):
            uncertainty = ()
        result = SubAgentResult(
            name=self.name,
            ok=ok,
            result_type=result_type,
            claim=str(parsed.get("claim", "")),
            result=parsed,
            state_patch=patch,
            local_verdict=local_verdict,
            artifact_refs=artifacts,
            trace_refs=trace_refs,
            primitive_traces=primitive_traces,
            attempt_records=attempts,
            diagnoses=diagnoses,
            evidence_packet=packet,
            uncertainty=tuple(str(v) for v in (uncertainty or packet.uncertainty)),
            recommended_next=str(parsed.get("recommended_next", "") or packet.recommended_next),
            loaded_skill_ids=loaded,
            turns=len(turns),
            error=error,
            artifacts_dir=self._request.artifacts_dir,
            task_id=self._request.task_id,
        )
        write_subagent_result_artifact(self._request, result)
        return result


def make_default_subagent_registry(
    *,
    executor: Any,
    policy: CompletionPolicy,
    library: SkillLibrary,
    max_turns: int = 8,
    execution_policy: SubAgentExecutionPolicy | None = None,
    system_prompt: str | None = None,
) -> SubAgentRegistry:
    return SubAgentRegistry(
        default_agent=CodingAgentSubAgent(
            executor=executor,
            policy=policy,
            library=library,
            max_turns=max_turns,
            execution_policy=execution_policy,
            system_prompt=system_prompt or _SUBAGENT_SYSTEM_PROMPT,
        )
    )


def _merge_artifacts(
    *groups: tuple[WorkspaceArtifact, ...],
) -> tuple[WorkspaceArtifact, ...]:
    seen: set[str] = set()
    out: list[WorkspaceArtifact] = []
    for group in groups:
        for artifact in group:
            key = artifact.artifact_id
            if key in seen:
                continue
            seen.add(key)
            out.append(artifact)
    return tuple(out)


def _result_payload(raw: dict[str, Any]) -> dict[str, Any]:
    result = raw.get("result")
    if isinstance(result, dict):
        return {**result, **{k: v for k, v in raw.items() if k not in {"result"}}}
    return raw


def _extract_artifact_refs(raw: dict[str, Any], *, default_producer: str = "") -> tuple[WorkspaceArtifact, ...]:
    payload = _result_payload(raw)
    items: list[Any] = []
    for key in ("artifact_refs", "artifacts"):
        value = payload.get(key)
        if isinstance(value, dict):
            items.extend({"artifact_id": k, "path": v, "kind": k} for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            items.extend(value)
    out = []
    for item in items:
        artifact = WorkspaceArtifact.from_any(item, default_producer=default_producer)
        if artifact is not None:
            out.append(artifact)
    return tuple(out)


def _extract_primitive_traces(raw: dict[str, Any], *, default_producer: str = "") -> tuple[PrimitiveTrace, ...]:
    payload = _result_payload(raw)
    traces = []
    for i, item in enumerate(payload.get("primitive_traces", ()) or payload.get("traces", ()) or ()):
        trace = PrimitiveTrace.from_any(item, default_id=f"{default_producer}:trace:{i}", default_producer=default_producer)
        if trace is not None:
            traces.append(trace)
    return tuple(traces)


def _extract_attempt_records(raw: dict[str, Any]) -> tuple[AttemptRecord, ...]:
    payload = _result_payload(raw)
    attempts = []
    for i, item in enumerate(payload.get("attempt_records", ()) or payload.get("attempts", ()) or ()):
        attempt = AttemptRecord.from_any(item, default_id=f"attempt:{i}")
        if attempt is not None:
            attempts.append(attempt)
    return tuple(attempts)


def _extract_diagnoses(raw: dict[str, Any]) -> tuple[Diagnosis, ...]:
    payload = _result_payload(raw)
    items = payload.get("diagnoses")
    if items is None and isinstance(payload.get("diagnosis"), dict):
        items = [payload["diagnosis"]]
    diagnoses = []
    for i, item in enumerate(items or ()):
        diagnosis = Diagnosis.from_any(item, default_id=f"diagnosis:{i}")
        if diagnosis is not None:
            diagnoses.append(diagnosis)
    return tuple(diagnoses)


def _extract_local_verdict(raw: dict[str, Any]) -> LocalVerdict | None:
    payload = _result_payload(raw)
    for key in ("local_verdict", "verdict"):
        verdict = LocalVerdict.from_any(payload.get(key))
        if verdict is not None:
            return verdict
    return LocalVerdict.from_any(payload)


def _short_task_label(task: str, *, max_len: int = 32) -> str:
    label = re.sub(r"[^0-9A-Za-z]+", "_", task.strip().lower()).strip("_")
    return label[:max_len].strip("_")


def _parse_finish_result(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    data = parse_action_payload(raw)
    if data is None:
        return {"claim": raw}
    if not isinstance(data, dict):
        return {"raw": data}
    args = data.get("args")
    if isinstance(args, dict):
        result = args.get("result")
        base = {
            key: value
            for key, value in args.items()
            if key in {"claim", "state_patch", "uncertainty", "recommended_next", "error", "ok"}
        }
        if isinstance(result, dict):
            return {**result, **base}
        return dict(args)
    return data
