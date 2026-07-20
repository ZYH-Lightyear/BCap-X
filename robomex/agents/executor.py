"""CodeAsPolicy 技能 Agent:咨询技能、写代码的执行器。

它是共享 :class:`~robomex.core.coder.CodingAgent` 的轻量子类。执行器的特化在于:
上下文是任务(+观测);通过渐进披露感知*整库*(开头一份简短的 ``<available_skills>``
清单 + 用 ``use_skill`` 拉取正文);每个 python 轮都会打包证据;``finish`` 表示
Act 认为当前 sub-goal 尝试结束,控制权交回外层 Planner。

技能只是被*咨询*,绝不照搬执行;代码由策略自己生成。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from robomex.core.coder import CodingAgent, SkillEntry, parse_action_payload
from robomex.core.coder.agent import _compact_stream_for_prompt
from robomex.core.context import (
    AttemptRecord,
    Diagnosis,
    EvidencePacket,
    EvidenceTimelineItem,
    LocalVerdict,
    PrimitiveTrace,
    ArtifactRef,
    compact_json,
)
from robomex.core.coder.policy import CompletionPolicy
from robomex.core.coder.trace import AgentTrace, TurnRecord
from robomex.core.events import emit_event
from robomex.core.logging import get_logger
from robomex.core.sandbox import BlockExecutionResult, SemanticActionBlock
from robomex.perception import EvidenceCollector, save_video
from robomex.prompts import BASE_ACT_SYSTEM_PROMPT
from robomex.skills import (
    SkillLibrary,
)

_log = get_logger("executor")

_STATE_CHANGING_API_CALLS = frozenset({
    "close_gripper",
    "execute_joint_trajectory",
    "goto_home_joint_position",
    "goto_pose",
    "move_to_joints",
    "open_gripper",
    "reset",
    "step",
})

# 子目标开场注入沙箱:重置语义证据字典 + 抓子目标起始帧。沙箱 globals 跨 block 持久,
# 因此 EVIDENCE 与 OBS_BEFORE 会一直留到该子目标结束,作为调试和 planner 复盘材料。
def _evidence_seed(camera: str | None) -> str:
    return (
        "try:\n"
        "    EVIDENCE\n"
        "except NameError:\n"
        "    EVIDENCE = {}\n"
        "EVIDENCE.clear()\n"
        f"OBS_CAMERA = {camera!r}\n"
        "try:\n"
        "    _obs0 = get_observation()\n"
        "    if OBS_CAMERA:\n"
        "        OBS_BEFORE = _obs0[OBS_CAMERA]['images']['rgb'].copy()\n"
        "    else:\n"
        "        OBS_BEFORE = next(\n"
        "            cam['images']['rgb'].copy()\n"
        "            for cam in _obs0.values()\n"
        "            if isinstance(cam, dict)\n"
        "            and isinstance(cam.get('images'), dict)\n"
        "            and 'rgb' in cam['images']\n"
        "        )\n"
        "except Exception as _e:\n"
        "    OBS_BEFORE = None\n"
    )


def _extract_feedback_rgb(obs: dict[str, Any] | None, camera: str | None):
    if not obs:
        return None
    if camera:
        try:
            return obs[camera]["images"]["rgb"]
        except (KeyError, TypeError):
            return None
    for cam in obs.values():
        if isinstance(cam, dict) and isinstance(cam.get("images"), dict) and "rgb" in cam["images"]:
            return cam["images"]["rgb"]
    return None


class CodeAsPolicyAgent(CodingAgent):
    def __init__(
        self,
        executor: Any,
        policy: CompletionPolicy,
        library: SkillLibrary,
        collector: EvidenceCollector | None = None,
        max_turns: int = 6,
        system_prompt: str = BASE_ACT_SYSTEM_PROMPT,
        observation_camera: str | None = None,
    ) -> None:
        super().__init__(
            executor=executor,
            policy=policy,
            library=library,
            max_turns=max_turns,
            system_prompt=system_prompt,
        )
        self.collector = collector
        self.observation_camera = observation_camera
        self._task = ""
        self._expected_postcondition = ""
        self._observation_summary = ""
        self._video_dir: Path | None = None
        self._clips: list[dict] = []
        self._feedback = ""
        self._primitive_traces: list[PrimitiveTrace] = []
        self._attempt_records: list[AttemptRecord] = []
        self._diagnoses: list[Diagnosis] = []
        self._local_verdicts: list[LocalVerdict] = []
        self._artifact_refs: list[ArtifactRef] = []
        self._evidence_packets: list[dict[str, Any]] = []
        self._evidence_timeline: list[dict[str, Any]] = []
        self._local_context_notes: list[str] = []

    def run(
        self,
        task: str,
        observation_summary: str = "",
        expected_postcondition: str = "",
        video_dir: str | Path | None = None,
        feedback: str = "",
        scene_image_path: str | None = None,
    ) -> AgentTrace:
        self._task = task
        self._expected_postcondition = expected_postcondition
        self._observation_summary = observation_summary
        self._video_dir = Path(video_dir) if video_dir is not None else None
        self._scene_image_path = scene_image_path
        self._clips = []
        self._feedback = feedback or ""
        self._primitive_traces = []
        self._attempt_records = []
        self._diagnoses = []
        self._local_verdicts = []
        self._artifact_refs = []
        self._evidence_packets = []
        self._evidence_timeline = []
        self._local_context_notes = []
        return super().run()

    # ---- 钩子 --------------------------------------------------------------

    def _setup(self, prompt: list[dict]) -> None:
        """子目标开场:重置 EVIDENCE、抓起始帧 OBS_BEFORE + 注入 ARTIFACTS_DIR。"""

        art_dir = str(self._video_dir) if self._video_dir else "/tmp"
        seed = (
            f"ARTIFACTS_DIR = {art_dir!r}\n"
            + _evidence_seed(self.observation_camera)
        )
        try:
            self.executor.run_block(
                SemanticActionBlock(
                    name="evidence_seed", intent="seed subgoal evidence", code=seed
                )
            )
        except Exception as exc:  # noqa: BLE001 - 取证种子失败不该让子目标崩溃
            _log.warning("证据种子注入失败: %r", exc)

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
        from robomex.perception.render import image_content_part

        parts = [f"Task: {self._task}"]
        if self._expected_postcondition:
            parts.append(f"Expected postcondition: {self._expected_postcondition}")
        if self._observation_summary:
            parts.append(f"Observation: {self._observation_summary}")
        parts.append(
            "The current scene image is attached below. Inspect it before writing code. "
            "If the empty arm or gripper occludes the target or receptacle, you may call "
            "`goto_home_joint_position()` then `get_observation()` to obtain a clear view. "
            "If an object may already be held, do not open the gripper for observation; "
            "`goto_home_joint_position()` preserves the current gripper command."
        )
        if self._feedback:
            parts.append(
                "Feedback from a previous unresolved attempt/review. Use it to fix your approach:\n"
                f"{self._feedback}"
            )
        text = "\n\n".join(parts)

        if self._scene_image_path:
            return [
                {"type": "text", "text": text},
                image_content_part(self._scene_image_path),
            ]
        return text

    def _block_metadata(self) -> dict:
        return {"task": self._task}

    def _agent_role(self) -> str:
        return "act"

    def _agent_label(self) -> str:
        return "Act Agent"

    def _llm_io_dir(self) -> Path | None:
        if self._video_dir is None:
            return None
        return self._video_dir / "llm_io"

    def _allowed_action_kinds(self) -> set[str]:
        return super()._allowed_action_kinds()

    def _python_gate_message(self, code: str, loaded: tuple[str, ...]) -> str:
        if _opens_gripper_after_home(code):
            return (
                "Python execution is blocked because this code calls "
                "`goto_home_joint_position()` and then `open_gripper()` in the same block. "
                "Going home is an observation/repositioning action and must preserve any "
                "held object. Remove `open_gripper()` unless the current loaded motion skill "
                "is explicitly releasing at the target."
            )
        if loaded:
            return ""
        state_changing = _state_changing_calls(code)
        if not state_changing:
            return ""
        available = ", ".join(e.name for e in self._skill_entries()) or "(none)"
        return (
            "Python execution is blocked because this block would change robot/environment "
            f"state before any RoboMEx skill was loaded. State-changing call(s): "
            f"{', '.join(state_changing)}. Observation, evidence preparation, SubAgent "
            "delegation, VLM state checks, geometry, and IK checks may run before loading a "
            "skill; physical execution must first consult a relevant skill with use_skill. "
            f"Available skills: {available}."
        )

    def _record_evidence_packet(
        self,
        source: str,
        packet: EvidencePacket,
        *,
        call_index: int | None = None,
    ) -> None:
        if packet.is_empty:
            return
        entry: dict[str, Any] = {
            "source": source,
            "packet": packet.to_json_dict(),
        }
        if call_index is not None:
            entry["call_index"] = call_index
        self._evidence_packets.append(entry)
        self._evidence_timeline.append(
            EvidenceTimelineItem.from_packet(
                packet,
                source=source,
                subgoal_goal=self._task,
                subgoal_postcondition=self._expected_postcondition,
            ).to_json_dict()
        )

    def _append_local_context_note(self, note: str) -> None:
        text = str(note or "").strip().replace("\n", " ")
        if not text:
            return
        self._local_context_notes.append(text)
        if len(self._local_context_notes) > 8:
            self._local_context_notes = self._local_context_notes[-8:]

    def _local_context_for_prompt(self) -> str:
        if not self._local_context_notes:
            return ""
        lines = "\n".join(f"- {note}" for note in self._local_context_notes[-8:])
        return (
            "Local context updates from this Act sub-goal. Treat these as the freshest "
            "working memory and reconcile later actions/verifier checks against them:\n"
            f"{lines}\n\n"
        )

    def _feedback_message(self, execution: BlockExecutionResult) -> str | list:
        from robomex.perception.render import image_content_part, save_rgb

        stdout = _compact_stream_for_prompt("stdout", execution.stdout)
        stderr = _compact_stream_for_prompt("stderr", execution.stderr)
        text = (
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}\n\n"
            "Continue from the loaded skill guidance. Reply with exactly one JSON action: "
            "use_skill if the next step needs different guidance, run_python for the next "
            "code block, or finish. Keep stdout compact: print at most a short stage "
            "report with selected candidate, success/failure reason, and artifact paths; "
            "store large masks, arrays, candidate lists, and raw debug dumps in "
            "EVIDENCE or artifact files."
        )
        local_context = self._local_context_for_prompt()
        if local_context:
            text = f"{text}\n\n{local_context}"
        if self._video_dir:
            try:
                rgb = _extract_feedback_rgb(execution.observation, self.observation_camera)
                if rgb is None:
                    return text
                img_path = save_rgb(
                    self._video_dir / f"act_obs_latest.png", rgb
                )
                return [
                    {"type": "text", "text": text},
                    image_content_part(img_path),
                ]
            except (KeyError, TypeError, OSError):
                pass
        return text

    def _on_python_turn(
        self,
        turn_idx: int,
        code: str,
        execution: BlockExecutionResult,
        prev_observation: dict | None,
        turns: list[Any],
    ) -> None:
        if self.collector is not None:
            # 逐块存 before/after 帧,作为调试产物和 planner 复盘材料。
            self.collector.bundle_for_block(
                execution.block.name, prev_observation, execution.observation
            )
        self._save_block_clip(turn_idx, execution)
        _log.info(
            "turn %d: exec=%s reward=%s terminated=%s",
            turn_idx, execution.status.value, execution.reward, execution.terminated,
        )
        stderr = (execution.stderr or "").strip()
        if not execution.ok and stderr:
            _log.info("turn %d: 报错 -> %s", turn_idx, stderr.splitlines()[-1][:300])
        turns.append(TurnRecord(turn_idx, code, execution))
        self._primitive_traces.extend(_primitive_traces_from_execution(turn_idx, execution, producer="act"))
        calls = _state_changing_calls(code)
        if calls and execution.ok:
            self._append_local_context_note(
                f"act:t{turn_idx} executed state-changing call(s): {', '.join(calls)}. "
                "If the postcondition is not visually obvious, use the current observation "
                "or call the Verifier before declaring success."
            )
        attempt = _attempt_from_execution(
            turn_idx,
            execution,
            task=self._task,
            clips=self._clips,
        )
        if attempt is not None:
            self._attempt_records.append(attempt)
        diagnosis = _diagnosis_from_failed_execution(turn_idx, execution)
        if diagnosis is not None:
            self._diagnoses.append(diagnosis)

    def _save_block_clip(self, turn_idx: int, execution: BlockExecutionResult) -> None:
        """有动作的 block 才存视频:把这一块产生的帧区间写成 ``turn_NN.mp4``。

        帧区间由适配器记在 ``execution.info['video_range']`` 里(没动作 → 没区间 → 不存)。
        路径 + 区间一并记进 ``self._clips``,稍后随 ``trace.metadata['clips']`` 交给 Planner/日志。
        """

        if self._video_dir is None:
            return
        rng = (execution.info or {}).get("video_range")
        env = getattr(self.executor, "env", None)
        if not rng or env is None or not hasattr(env, "get_video_frames_range"):
            return
        start, end = int(rng[0]), int(rng[1])
        try:
            frames = env.get_video_frames_range(start, end)
        except Exception as exc:  # noqa: BLE001 - 取帧失败不该中断子目标
            _log.warning("turn %d: 取过程帧失败 -> %r", turn_idx, exc)
            return
        if not frames:
            return
        path = self._video_dir / f"turn_{turn_idx:02d}.mp4"
        try:
            saved = save_video(path, frames)
        except Exception as exc:  # noqa: BLE001 - 写视频失败不该中断子目标
            _log.warning("turn %d: 写过程视频失败 -> %r", turn_idx, exc)
            return
        if saved:
            self._clips.append({"turn": turn_idx, "path": saved, "start": start, "end": end})
            _log.info("turn %d: 已存过程视频 %s (%d 帧)", turn_idx, saved, len(frames))

    def _should_stop_after_python(self, execution: BlockExecutionResult) -> bool:
        return bool(execution.terminated)

    def _on_terminal_turn(
        self,
        turn_idx: int,
        raw: str,
        turns: list[Any],
        loaded: tuple[str, ...],
    ) -> tuple[bool, str]:
        return True, ""

    def _finalize(self, *, turns: list[Any], loaded: tuple[str, ...], terminal_raw: str | None) -> AgentTrace:
        success = terminal_raw is not None
        act_status = "finished" if success else "exhausted"
        terminal_result = _parse_terminal_result(terminal_raw)
        terminal_packet = EvidencePacket.from_any(
            terminal_result,
            default_source="act",
            default_turn=f"subgoal:{self._task}",
        )
        terminal_artifacts = _extract_artifact_refs(terminal_result, default_producer="act")
        if terminal_packet.artifact_refs:
            terminal_artifacts = _merge_artifacts(terminal_artifacts, terminal_packet.artifact_refs)
        terminal_traces = _extract_primitive_traces(terminal_result, default_producer="act")
        terminal_attempts = _extract_attempt_records(terminal_result)
        terminal_diagnoses = _extract_diagnoses(terminal_result)
        terminal_verdict = _extract_local_verdict(terminal_result) or terminal_packet.verdict
        if terminal_verdict is not None:
            self._local_verdicts.append(terminal_verdict)
        self._artifact_refs.extend(terminal_artifacts)
        self._primitive_traces.extend(terminal_traces)
        self._attempt_records.extend(terminal_attempts)
        self._diagnoses.extend(terminal_diagnoses)
        unresolved = None
        if not success:
            unresolved = {
                "subgoal": self._task,
                "status": "unresolved",
                "skills_used": list(loaded),
                "last_state_summary": "Act exhausted its turn budget before finish.",
                "suggested_recovery": "Re-plan from the current scene.",
            }
            self._diagnoses.append(
                _diagnosis_from_exhaustion(
                    task=self._task,
                    loaded=loaded,
                    turns=turns,
                )
            )
        terminal_evidence_entries = list(self._evidence_packets)
        terminal_timeline_entries = list(self._evidence_timeline)
        if not terminal_packet.is_empty:
            terminal_entry = {
                "source": "act:finish",
                "packet": terminal_packet.to_json_dict(),
            }
            terminal_evidence_entries.append(terminal_entry)
            terminal_timeline_entries.append(
                EvidenceTimelineItem.from_packet(
                    terminal_packet,
                    source="act:finish",
                    subgoal_goal=self._task,
                    subgoal_postcondition=self._expected_postcondition,
                    act_status=act_status,
                    loaded_skill_ids=loaded,
                ).to_json_dict()
            )
        return AgentTrace(
            task=self._task,
            loaded_skill_ids=loaded,
            turns=tuple(turns),
            success=success,
            metadata={
                "clips": tuple(self._clips),
                "unresolved": unresolved,
                "act_status": act_status,
                "terminal_raw": terminal_raw,
                "terminal_result": terminal_result,
                "evidence_packets": tuple(terminal_evidence_entries),
                "evidence_timeline": tuple(terminal_timeline_entries),
                "artifact_refs": tuple(a.to_json_dict() for a in self._artifact_refs),
                "primitive_traces": tuple(t.to_json_dict() for t in self._primitive_traces),
                "attempt_records": tuple(a.to_json_dict() for a in self._attempt_records),
                "diagnoses": tuple(d.to_json_dict() for d in self._diagnoses),
                "local_verdicts": tuple(v.to_json_dict() for v in self._local_verdicts),
            },
        )


def _primitive_traces_from_execution(turn_idx: int, execution: BlockExecutionResult, *, producer: str) -> list[PrimitiveTrace]:
    traces: list[PrimitiveTrace] = []
    block = execution.block
    for i, event in enumerate(execution.trace_events or ()):
        payload = dict(event.payload or {})
        for j, raw in enumerate(payload.get("primitive_traces", ()) or ()):
            parsed = PrimitiveTrace.from_any(
                raw,
                default_id=f"{producer}:t{turn_idx}:event{i}:primitive{j}",
                default_producer=producer,
            )
            if parsed is not None:
                traces.append(parsed)
        if payload.get("primitive_traces"):
            continue
        traces.append(
            PrimitiveTrace(
                trace_id=f"{producer}:t{turn_idx}:event{i}",
                primitive_name=event.event_type or "trace_event",
                status=execution.status.value,
                producer=producer,
                block_name=event.block_name or block.name,
                turn=turn_idx,
                outputs_summary=payload,
                error=execution.stderr if not execution.ok else "",
            )
        )
    traces.append(
        PrimitiveTrace(
            trace_id=f"{producer}:t{turn_idx}:block",
            primitive_name="run_python",
            status=execution.status.value,
            producer=producer,
            block_name=block.name,
            turn=turn_idx,
            inputs_summary={
                "intent": block.intent,
                "line_count": block.code.count("\n") + 1,
                "metadata": block.metadata,
            },
            outputs_summary={
                "ok": execution.ok,
                "reward": execution.reward,
                "terminated": execution.terminated,
                "truncated": execution.truncated,
                "info": execution.info,
            },
            error=execution.stderr if not execution.ok else "",
        )
    )
    return traces


def _attempt_from_execution(
    turn_idx: int,
    execution: BlockExecutionResult,
    *,
    task: str,
    clips: list[dict[str, Any]],
) -> AttemptRecord | None:
    calls = _state_changing_calls(execution.block.code)
    if not calls:
        return None
    if execution.terminated:
        outcome = "success"
        confidence = 0.8
    elif execution.ok:
        outcome = "executed"
        confidence = 0.5
    else:
        outcome = "failed"
        confidence = 0.9
    artifacts: list[ArtifactRef] = []
    for clip in clips:
        if int(clip.get("turn", -1)) != turn_idx:
            continue
        path = str(clip.get("path") or "")
        if path:
            artifacts.append(
                ArtifactRef(
                    artifact_id=f"act:t{turn_idx}:video",
                    kind="video",
                    path=path,
                    producer="act",
                    summary="Process video for this action block.",
                )
            )
    reason = ""
    if not execution.ok and execution.stderr:
        reason = execution.stderr.strip().splitlines()[-1][:240]
    return AttemptRecord(
        attempt_id=f"act:t{turn_idx}:attempt",
        object_key="",
        strategy=execution.block.intent or ",".join(calls),
        pose_or_target=compact_json(
            {
                "subgoal": task,
                "state_changing_calls": list(calls),
                "block": execution.block.name,
                "info": execution.info,
            },
            max_depth=3,
            max_items=8,
            max_string=240,
        ),
        related_trace_ids=(f"act:t{turn_idx}:block",),
        outcome=outcome,
        failure_reason=reason,
        recommended_repair=(
            "Inspect current observation and avoid repeating the same physical block "
            "unless evidence shows the scene or candidate changed."
            if outcome == "failed"
            else ""
        ),
        confidence=confidence,
        artifact_refs=tuple(artifacts),
    )


def _diagnosis_from_failed_execution(turn_idx: int, execution: BlockExecutionResult) -> Diagnosis | None:
    if execution.ok:
        return None
    reason = execution.stderr.strip().splitlines()[-1][:300] if execution.stderr else "execution failed"
    return Diagnosis(
        diagnosis_id=f"act:t{turn_idx}:diagnosis",
        failed_primitive="run_python",
        failure_type="execution_error",
        evidence_trace_ids=(f"act:t{turn_idx}:block",),
        next_route="Repair the code or choose a different candidate before retrying.",
        confidence=0.9,
        reason=reason,
    )


def _diagnosis_from_exhaustion(
    *,
    task: str,
    loaded: tuple[str, ...],
    turns: list[Any],
) -> Diagnosis:
    last_turn = turns[-1] if turns else None
    evidence_trace_ids = (f"act:t{last_turn.turn}:block",) if last_turn is not None else ()
    reason = "Act exhausted its action budget before finish."
    if last_turn is not None:
        stderr = (last_turn.execution.stderr or "").strip()
        stdout = (last_turn.execution.stdout or "").strip()
        if stderr:
            reason = stderr.splitlines()[-1][:300]
        elif stdout:
            reason = stdout[-300:]
    route = "Re-plan from the current scene."
    if loaded:
        route += f" Previously loaded skills: {', '.join(loaded)}."
    last_turn_id = last_turn.turn if last_turn is not None else "none"
    return Diagnosis(
        diagnosis_id=f"act:subgoal_exhausted:t{last_turn_id}",
        failed_primitive="subgoal",
        failure_type="action_budget_exhausted",
        evidence_trace_ids=evidence_trace_ids,
        next_route=route,
        confidence=0.7,
        reason=f"{task}: {reason}",
    )




def _result_payload(raw: dict[str, Any]) -> dict[str, Any]:
    result = raw.get("result")
    if isinstance(result, dict):
        return {**result, **{k: v for k, v in raw.items() if k != "result"}}
    return raw


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


def _extract_artifact_refs(raw: dict[str, Any], *, default_producer: str = "") -> tuple[ArtifactRef, ...]:
    payload = _result_payload(raw)
    items: list[Any] = []
    for key in ("artifact_refs", "artifacts"):
        value = payload.get(key)
        if isinstance(value, dict):
            items.extend({"artifact_id": k, "path": v, "kind": k} for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            items.extend(value)
    artifacts: list[ArtifactRef] = []
    for item in items:
        artifact = ArtifactRef.from_any(item, default_producer=default_producer)
        if artifact is not None:
            artifacts.append(artifact)
    return tuple(artifacts)


def _extract_primitive_traces(raw: dict[str, Any], *, default_producer: str = "") -> tuple[PrimitiveTrace, ...]:
    payload = _result_payload(raw)
    traces: list[PrimitiveTrace] = []
    for i, item in enumerate(payload.get("primitive_traces", ()) or payload.get("traces", ()) or ()):
        trace = PrimitiveTrace.from_any(item, default_id=f"{default_producer}:trace:{i}", default_producer=default_producer)
        if trace is not None:
            traces.append(trace)
    return tuple(traces)


def _extract_attempt_records(raw: dict[str, Any]) -> tuple[AttemptRecord, ...]:
    payload = _result_payload(raw)
    attempts: list[AttemptRecord] = []
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
    diagnoses: list[Diagnosis] = []
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


def _opens_gripper_after_home(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    calls.sort(key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))
    saw_home = False
    for node in calls:
        name = _call_name(node.func)
        if name == "goto_home_joint_position":
            saw_home = True
        elif name == "open_gripper" and saw_home:
            return True
    return False


def _state_changing_calls(code: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()
    names = {
        _call_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    return tuple(sorted(names & _STATE_CHANGING_API_CALLS))


def _parse_terminal_result(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    data = parse_action_payload(raw)
    if data is None:
        return {"claim": raw}
    args = data.get("args")
    if isinstance(args, dict):
        result = args.get("result")
        base = {
            key: value
            for key, value in args.items()
            if key in {"claim", "uncertainty", "recommended_next", "error", "ok"}
        }
        if isinstance(result, dict):
            return {**result, **base}
        return dict(args)
    return data


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""
