"""CodingAgent-backed SubAgents for focused RoboMEx inner-loop analyses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robomex.core.coder import CodingAgent, CompletionPolicy, SkillEntry, parse_action_payload
from robomex.core.coder.agent import _compact_stream_for_prompt
from robomex.core.context import (
    AttemptRecord,
    Diagnosis,
    EvidencePacket,
    LocalVerdict,
    PrimitiveTrace,
    ArtifactRef,
    compact_json,
)
from robomex.core.coder.trace import AgentTrace, TurnRecord
from robomex.core.sandbox import (
    BlockExecutionResult,
    SemanticActionBlock,
)
from robomex.skills import (
    SkillLibrary,
)
from robomex.core.artifact_paths import finish_artifact_path_errors
from robomex.core.payload_specs import validate_payload
from robomex.prompts.authoring import build_agent_system_prompt, output_contract_for_ports


@dataclass(frozen=True)
class SubAgentRequest:
    """One structurally typed task delegated to a coding-agent node."""

    task: str
    task_kind: str = "verify"
    inputs: dict[str, Any] = field(default_factory=dict)
    artifacts_dir: str | None = None
    task_id: str | None = None
    scene_image_path: str | None = None
    observation_epoch: int = 0


@dataclass(frozen=True)
class SubAgentResult:
    """Small JSON-serializable result returned by a specialist node."""

    name: str
    ok: bool
    result_type: str = ""
    claim: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    local_verdict: LocalVerdict | None = None
    artifact_refs: tuple[ArtifactRef, ...] = ()
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
    trace: AgentTrace | None = None

    def resolved_evidence_packet(self) -> EvidencePacket:
        """Return the generic cross-agent evidence envelope for this result.

        Result payloads are evidence, not persistent world-state mutations.
        """

        if not self.evidence_packet.is_empty:
            return self.evidence_packet
        payload: dict[str, Any] = {
            **self.result,
            "claim": self.claim or self.result.get("claim", ""),
            "confidence": self.result.get("confidence", 0.0),
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
            "trace": None
            if self.trace is None
            else {
                "task": self.trace.task,
                "loaded_skill_ids": list(self.trace.loaded_skill_ids),
                "turns": len(self.trace.turns),
                "success": self.trace.success,
                "metadata": self.trace.metadata,
            },
        }


def write_subagent_result_artifact(request: SubAgentRequest, result: SubAgentResult) -> None:
    """Persist a stable per-SubAgent result manifest when an artifact dir is available."""

    if not request.artifacts_dir:
        return
    d = Path(request.artifacts_dir)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "robomex.subagent_result.v2",
        "request": {
            "task": request.task,
            "task_kind": request.task_kind,
            "inputs": compact_json(request.inputs),
            "artifacts_dir": request.artifacts_dir,
            "task_id": request.task_id,
            "scene_image_path": request.scene_image_path,
            "observation_epoch": request.observation_epoch,
        },
        "result": result.to_json_dict(),
    }
    (d / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


_SUBAGENT_SYSTEM_PROMPT = build_agent_system_prompt(
    role="Verifier",
    objective=(
        "Independently verify one concrete state or postcondition and diagnose uncertainty. "
        "Treat input artifacts as claims, not ground truth."
    ),
    capabilities=("perception_read", "artifact_write", "geometry_compute"),
    output_contract=output_contract_for_ports((("verifier_report", "robomex.verifier.v1"),)),
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
    """Configurable specialist backed by the shared CodingAgent tool loop."""

    def __init__(
        self,
        executor: Any,
        policy: CompletionPolicy,
        library: SkillLibrary,
        *,
        max_turns: int = 8,
        system_prompt: str | None = None,
        name: str = "verifier",
        description: str = "",
        objective: str = "",
        task_kind: str = "verify",
        output_ports: tuple[tuple[str, ...], ...] = (
            ("verifier_report", "robomex.verifier.v1"),
        ),
        preloaded_skills: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.description = description or f"RoboMEx {task_kind} specialist"
        self.objective = objective or (
            "Independently verify one concrete state or postcondition. Treat input "
            "artifacts as claims, not ground truth, and report uncertainty explicitly."
            if task_kind == "verify"
            else self.description
        )
        self.task_kind = task_kind
        self.output_ports = output_ports
        effective_prompt = system_prompt or build_agent_system_prompt(
            role=name,
            objective=self.objective,
            capabilities=("perception_read", "artifact_write", "geometry_compute"),
            output_contract=output_contract_for_ports(output_ports),
        )
        super().__init__(
            executor=executor,
            policy=policy,
            library=library,
            max_turns=max_turns,
            system_prompt=effective_prompt,
            force_terminal_on_exhaust=True,
            preloaded_skills=preloaded_skills,
        )
        self._request = SubAgentRequest(task="", task_kind=task_kind)
        self._turn_records: list[TurnRecord] = []
        self._materialized_result: dict[str, Any] | None = None

    def run(self, request: SubAgentRequest) -> SubAgentResult:
        self._request = request
        self._turn_records = []
        self._materialized_result = None
        return super().run()

    def _setup(self, prompt: list[dict]) -> None:
        del prompt
        art_dir = self._request.artifacts_dir or "/tmp"
        Path(art_dir).mkdir(parents=True, exist_ok=True)
        library_root = str(getattr(self.library, "root", ""))
        observation_epoch = self._request.observation_epoch
        # INPUTS is the mechanical data plane: resolved typed inputs live in the
        # sandbox so code can reference them without retyping numbers from the
        # prompt (M1.7 — weak-model agent-swarm communication). Depth is raised
        # above the prompt default so small nested arrays (e.g. a 3x3 OBB
        # rotation matrix under payload.obb.rotation_matrix) stay inline and
        # usable; large arrays never reach payloads — they travel as file refs.
        inputs_compact = compact_json(self._request.inputs, max_depth=10)
        inputs_literal = repr(inputs_compact)
        seed = (
            f"ARTIFACTS_DIR = {str(art_dir)!r}\n"
            f"SKILL_LIBRARY_ROOT = {library_root!r}\n"
            f"OBSERVATION_EPOCH = {observation_epoch}\n"
            f"INPUTS = {inputs_literal}\n"
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
                metadata={"subagent": self.name, "runtime_setup": True},
            )
        )
        # Test doubles may expose a direct namespace; keep it in sync with CapX.
        ns = getattr(self.executor, "sandbox_namespace", None)
        if isinstance(ns, dict):
            ns["ARTIFACTS_DIR"] = str(art_dir)
            ns["SKILL_LIBRARY_ROOT"] = library_root
            ns["OBSERVATION_EPOCH"] = observation_epoch
            ns["INPUTS"] = inputs_compact
            ns.setdefault("EVIDENCE", {})

    def _skill_entries(self) -> list[SkillEntry]:
        return [
            SkillEntry(
                name=r.skill_id,
                description=r.skill.description or r.skill.name,
                category=r.skill.category.value,
            )
            for r in self.library.all()
        ]

    def _initial_user_message(self) -> str | list:
        payload = {
            "subagent": self.name,
            "task": self._request.task,
            "task_kind": self._request.task_kind,
            "inputs": self._request.inputs,
            "artifacts_dir": self._request.artifacts_dir,
            "task_id": self._request.task_id,
            "observation_epoch": self._request.observation_epoch,
        }
        text = (
            f"Execute the structured {self._request.task_kind} node request. "
            "Resolved inputs are already in the sandbox variable "
            'INPUTS["<port_name>"]; reference them in code — do not retype numeric '
            "values from this message. Build a NODE_RESULT dict that matches the "
            "declared typed outputs, then finish with "
            '{"tool":"finish","args":{"claim":"...","result_var":"NODE_RESULT"}}. '
            "Do not return persistent state patches.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        scene_image_path = self._request.scene_image_path
        if isinstance(scene_image_path, str) and scene_image_path:
            from robomex.perception.render import image_content_part

            return [
                {"type": "text", "text": text},
                image_content_part(scene_image_path),
            ]
        return text

    def _agent_role(self) -> str:
        return self.task_kind

    def _agent_label(self) -> str:
        label = _short_task_label(self._request.task) or self.name
        return f"SubAgent:{label}"

    def _on_terminal_turn(
        self,
        turn_idx: int,
        raw: str,
        turns: list[Any],
        loaded: tuple[str, ...],
    ) -> tuple[bool, str]:
        del turn_idx, turns, loaded
        result_var = _finish_result_var(raw)
        if result_var is not None:
            materialized, materialize_error = self._materialize_result_var(result_var)
            if materialize_error:
                return (
                    False,
                    "typed_finish_rejected: "
                    + materialize_error
                    + ". Assign a JSON-safe dict to that variable (reference sidecar "
                    "returns / INPUTS / printed observations — no invented pose "
                    "literals), then finish again with the same result_var.",
                )
            assert materialized is not None
            self._materialized_result = materialized
            parsed = _promote_verifier_report(materialized)
        else:
            self._materialized_result = None
            parsed = _promote_verifier_report(_parse_finish_result(raw))
        outputs = parsed.get("outputs") if isinstance(parsed.get("outputs"), dict) else {}
        normalized = {
            str(name).split(":", 1)[0]: value for name, value in outputs.items()
        }
        required = tuple(item[0] for item in self.output_ports)
        # A finish that honestly declares failure (failure_kind, a fail/uncertain
        # verdict, or ok:false) is exempt from the required-ports check: the port
        # contract governs the success hand-off, and forcing placeholder outputs
        # here would only teach agents to fabricate payloads (M2). Ports the
        # agent does include are still validated below.
        declares_failure = _declares_failure(parsed)
        missing = (
            []
            if declares_failure
            else [name for name in required if name not in normalized]
        )
        invalid_payloads = [
            name
            for name in required
            if name in normalized
            and (
                not isinstance(normalized[name], dict)
                or not isinstance(normalized[name].get("payload"), dict)
            )
        ]
        undeclared = [name for name in normalized if name not in required]
        if missing or invalid_payloads or undeclared:
            details: list[str] = []
            if missing:
                details.append(f"missing exact output keys: {', '.join(missing)}")
            if undeclared:
                details.append(
                    f"undeclared output keys: {', '.join(undeclared)} — return "
                    f"exactly the declared ports ({', '.join(required)}) and move "
                    "extra evidence into a declared payload"
                )
            if invalid_payloads:
                details.append(
                    "payload must be a compact JSON object for: "
                    + ", ".join(invalid_payloads)
                )
            return (
                False,
                "typed_finish_rejected: "
                + "; ".join(details)
                + ". Do not inline arrays or use ellipsis placeholders; save arrays to "
                "artifact files and return their paths under `artifacts`.",
            )
        # Validate artifact file refs inside the producer's own budget: a
        # path-convention mistake is repairable here, but fatal after finish.
        artifacts_dir = (
            Path(self._request.artifacts_dir) if self._request.artifacts_dir else None
        )
        path_errors = finish_artifact_path_errors(normalized, artifacts_dir)
        if path_errors:
            return (
                False,
                "typed_finish_rejected: artifact path violation(s) — "
                + "; ".join(path_errors),
            )
        # Enforce canonical payload specs inside the producer's own budget so
        # a schema violation is repaired here, not discovered downstream.
        spec_errors: list[str] = []
        for item in self.output_ports:
            name, schema = item[0], item[1] if len(item) > 1 else ""
            value = normalized.get(name)
            if not isinstance(value, dict):
                continue
            payload = value.get("payload")
            if not isinstance(payload, dict):
                continue
            spec_error = validate_payload(schema, payload)
            if spec_error:
                spec_errors.append(f"`{name}` ({schema}): {spec_error}")
        if spec_errors:
            return (
                False,
                "typed_finish_rejected: payload spec violation(s) — "
                + "; ".join(spec_errors)
                + ". Repair the payload to match the canonical spec in your "
                "output contract, then finish again.",
            )
        if self.task_kind == "verifier":
            verdict = parsed.get("verdict")
            status = (
                str(verdict.get("status") or "").lower()
                if isinstance(verdict, dict)
                else str(verdict or "").lower()
            )
            if status not in {"pass", "fail", "uncertain"}:
                return (
                    False,
                    "typed_finish_rejected: verifier result requires "
                    '`verdict: {"status":"pass|fail|uncertain","reason":"..."}`.',
                )
        return True, ""

    def _materialize_result_var(
        self, var_name: str
    ) -> tuple[dict[str, Any] | None, str]:
        """Load a finish payload from a sandbox variable (mechanical data plane).

        Prefer a direct namespace read when the executor exposes
        ``sandbox_namespace`` (unit-test doubles). Otherwise run a
        ``runtime_setup`` block that JSON-serializes the variable to stdout
        behind a sentinel marker and recover it here.
        """

        if not var_name.isidentifier():
            return None, f"result_var {var_name!r} is not a valid Python identifier"

        ns = getattr(self.executor, "sandbox_namespace", None)
        if isinstance(ns, dict):
            if var_name not in ns:
                return None, f"NameError: {var_name!r} is not defined in the sandbox"
            return _coerce_finish_result(ns[var_name])

        sentinel = _FINISH_RESULT_SENTINEL
        code = (
            "import json\n"
            f"_name = {var_name!r}\n"
            f"_sentinel = {sentinel!r}\n"
            "try:\n"
            f"    _value = {var_name}\n"
            "except NameError as _exc:\n"
            "    print(_sentinel + json.dumps({'error': f'NameError: {_name!r} is not defined'}))\n"
            "else:\n"
            "    def _default(obj):\n"
            "        if hasattr(obj, 'tolist'):\n"
            "            return obj.tolist()\n"
            "        if isinstance(obj, (set, tuple)):\n"
            "            return list(obj)\n"
            "        raise TypeError(f'not JSON-serializable: {type(obj).__name__}')\n"
            "    try:\n"
            "        print(_sentinel + json.dumps(_value, default=_default, ensure_ascii=False))\n"
            "    except Exception as _exc:  # noqa: BLE001\n"
            "        print(_sentinel + json.dumps({'error': f'{type(_exc).__name__}: {_exc}'}))\n"
        )
        try:
            execution = self.executor.run_block(
                SemanticActionBlock(
                    name=f"{self.name}_materialize_{var_name}",
                    intent="materialize finish result_var from sandbox",
                    code=code,
                    metadata={
                        **self._block_metadata(),
                        "runtime_setup": True,
                        "result_var": var_name,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"failed to materialize result_var {var_name!r}: {exc}"
        if not execution.ok:
            detail = (execution.stderr or execution.stdout or "unknown error").strip()
            return None, f"failed to materialize result_var {var_name!r}: {detail}"
        payload_text = _extract_sentinel_payload(execution.stdout or "", sentinel)
        if payload_text is None:
            return None, (
                f"result_var {var_name!r} materialization produced no recoverable "
                "payload; ensure the variable is a JSON-safe dict"
            )
        try:
            loaded = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            return None, f"result_var {var_name!r} is not valid JSON: {exc}"
        if isinstance(loaded, dict) and "error" in loaded and len(loaded) == 1:
            return None, str(loaded["error"])
        return _coerce_finish_result(loaded)

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
        stdout = _compact_stream_for_prompt("stdout", execution.stdout)
        stderr = _compact_stream_for_prompt("stderr", execution.stderr)
        return (
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}\n\n"
            f"Continue the focused {self.task_kind} task. Reply with exactly one JSON action: "
            "use_skill, run_python, or finish. If the delegated claim is answered, "
            "finish now with typed output JSON. Keep stdout compact and write large "
            "debug data to artifacts. Respect the enforced capability policy."
        )

    def _finalize(self, *, turns: list[Any], loaded: tuple[str, ...], terminal_raw: str | None) -> SubAgentResult:
        if self._materialized_result is not None:
            parsed = _promote_verifier_report(dict(self._materialized_result))
            # Preserve claim / envelope fields from the finish action itself.
            overlay = _parse_finish_envelope(terminal_raw)
            parsed = {**parsed, **overlay}
        else:
            parsed = _promote_verifier_report(_parse_finish_result(terminal_raw))
        error = "" if terminal_raw else "SubAgent exhausted its budget before finish."
        if parsed.get("error"):
            error = str(parsed["error"])
        source_turn = f"subagent:{self._request.task_id or ''}"
        packet = EvidencePacket.from_any(parsed, default_source=self.name, default_turn=source_turn)
        artifacts = _extract_artifact_refs(parsed, default_producer=self.name)
        if packet.artifact_refs:
            artifacts = _merge_artifacts(artifacts, packet.artifact_refs)
        primitive_traces = _extract_primitive_traces(parsed, default_producer=self.name)
        attempts = _extract_attempt_records(parsed)
        diagnoses = _extract_diagnoses(parsed)
        local_verdict = _extract_local_verdict(parsed) or packet.verdict
        verdict_status = str(local_verdict.status if local_verdict is not None else "").lower()
        claim = str(parsed.get("claim", ""))
        exhausted_claim = claim.strip().lower() in {
            "out of steps",
            "budget exhausted",
            "exhausted",
        }
        ok = (
            bool(terminal_raw)
            and bool(parsed.get("ok", True))
            and not error
            and verdict_status not in {"fail", "failed", "error", "uncertain", "unknown"}
            and not exhausted_claim
        )
        if exhausted_claim and not error:
            error = "SubAgent exhausted its budget before producing a completed result."
        elif verdict_status in {"fail", "failed", "error", "uncertain", "unknown"} and not error:
            error = f"SubAgent returned non-success verdict {verdict_status!r}."
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
            claim=claim,
            result=parsed,
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
            trace=AgentTrace(
                task=self._request.task,
                loaded_skill_ids=loaded,
                turns=tuple(self._turn_records),
                success=ok,
                metadata={
                    "agent_role": self.task_kind,
                    "agent_name": self.name,
                    "terminal_result": parsed,
                },
            ),
        )
        write_subagent_result_artifact(self._request, result)
        return result


def _merge_artifacts(
    *groups: tuple[ArtifactRef, ...],
) -> tuple[ArtifactRef, ...]:
    seen: set[str] = set()
    out: list[ArtifactRef] = []
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


def _extract_artifact_refs(raw: dict[str, Any], *, default_producer: str = "") -> tuple[ArtifactRef, ...]:
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
        artifact = ArtifactRef.from_any(item, default_producer=default_producer)
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


_FINISH_RESULT_SENTINEL = "__ROBOMEX_FINISH_RESULT__"
_INLINE_ARRAY_LEN = 32


def _declares_failure(parsed: dict[str, Any]) -> bool:
    """True when a finish result structurally declares a failed attempt.

    Only explicit signals count: ``ok: false`` or a ``failure_kind`` string.
    A fail/uncertain verdict alone does not qualify — for a verifier that is
    the normal successful hand-off and its typed report stays required.
    """

    if parsed.get("ok") is False:
        return True
    kind = parsed.get("failure_kind")
    if isinstance(kind, str) and kind.strip():
        return True
    nested = parsed.get("result") if isinstance(parsed.get("result"), dict) else {}
    nested_kind = nested.get("failure_kind")
    return isinstance(nested_kind, str) and bool(nested_kind.strip())


def _finish_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    data = parse_action_payload(raw)
    if not isinstance(data, dict):
        return {}
    args = data.get("args")
    return args if isinstance(args, dict) else {}


def _finish_result_var(raw: str | None) -> str | None:
    name = _finish_args(raw).get("result_var")
    return name if isinstance(name, str) and name else None


def _parse_finish_envelope(raw: str | None) -> dict[str, Any]:
    """Claim / uncertainty / recommended_next from the finish action (not result_var)."""

    args = _finish_args(raw)
    return {
        key: value
        for key, value in args.items()
        if key in {"claim", "uncertainty", "recommended_next", "error", "ok"}
    }


def _extract_sentinel_payload(stdout: str, sentinel: str) -> str | None:
    for line in reversed(stdout.splitlines()):
        if sentinel in line:
            return line.split(sentinel, 1)[1]
    if sentinel in stdout:
        return stdout.split(sentinel, 1)[1].splitlines()[0]
    return None


def _inline_array_errors(value: Any, *, path: str = "result") -> list[str]:
    """Reject large numeric arrays that should have been persisted as files."""

    errors: list[str] = []
    if isinstance(value, list):
        if len(value) > _INLINE_ARRAY_LEN and all(
            isinstance(item, (int, float)) for item in value[: min(8, len(value))]
        ):
            errors.append(
                f"{path}: inline array of length {len(value)}; save it under "
                "ARTIFACTS_DIR and reference the path instead"
            )
            return errors
        for index, item in enumerate(value):
            errors.extend(_inline_array_errors(item, path=f"{path}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            errors.extend(_inline_array_errors(item, path=f"{path}.{key}"))
    return errors


def _coerce_finish_result(value: Any) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(value, dict):
        return None, (
            f"result_var must be a dict, got {type(value).__name__}; wrap your "
            'outputs as {"outputs": {...}, "recommended_next": "..."}'
        )
    array_errors = _inline_array_errors(value)
    if array_errors:
        return None, "; ".join(array_errors)
    return dict(value), ""


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
        # result_var is materialized separately; ignore the bare name here.
        if isinstance(args.get("result_var"), str) and args.get("result") is None:
            return _parse_finish_envelope(raw)
        result = args.get("result")
        base = _parse_finish_envelope(raw)
        if isinstance(result, dict):
            return {**result, **base}
        return {k: v for k, v in args.items() if k != "result_var"}
    return data


def _promote_verifier_report(parsed: dict[str, Any]) -> dict[str, Any]:
    """Project the typed VerifierReport verdict into the generic evidence envelope."""

    if isinstance(parsed.get("verdict"), (dict, str)):
        return parsed
    outputs = parsed.get("outputs")
    if not isinstance(outputs, dict):
        return parsed
    report = next(
        (
            value
            for name, value in outputs.items()
            if str(name).split(":", 1)[0] == "verifier_report"
            and isinstance(value, dict)
        ),
        None,
    )
    if not isinstance(report, dict):
        return parsed
    payload = report.get("payload") if isinstance(report.get("payload"), dict) else {}
    verdict = report.get("verdict") or payload.get("verdict")
    if not isinstance(verdict, (dict, str)):
        return parsed
    return {**parsed, "verdict": verdict}
