"""Trace format for policy-visible revision-local Context turns."""

from __future__ import annotations

import json
import pathlib
import shutil
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

from vaw.context_runtime.packet import ContextPacket
from vaw.context_runtime.workspace import ContextStepResult


class ContextTraceLogger:
    def __init__(self, trace_dir: str | pathlib.Path) -> None:
        self.dir = pathlib.Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for artifact in self.dir.glob("context_*.png"):
            artifact.unlink()
        for artifact in self.dir.glob("video_*.mp4"):
            artifact.unlink()
        self.steps_path = self.dir / "steps.jsonl"
        self.events_path = self.dir / "runtime_events.jsonl"
        self.meta_path = self.dir / "meta.json"
        self.contexts_dir = self.dir / "contexts"
        self.actions_dir = self.dir / "actions"
        shutil.rmtree(self.contexts_dir, ignore_errors=True)
        shutil.rmtree(self.actions_dir, ignore_errors=True)
        self.contexts_dir.mkdir(exist_ok=True)
        self.actions_dir.mkdir(exist_ok=True)
        self.steps_path.unlink(missing_ok=True)
        self.events_path.unlink(missing_ok=True)
        self.meta_path.unlink(missing_ok=True)
        self.index = 0
        self._event_seq = 0
        self._event_lock = threading.Lock()
        self._started_monotonic = time.monotonic()

    def freeze_turn_context(
        self,
        *,
        turn: int,
        image: np.ndarray,
        snapshot: dict[str, Any],
    ) -> str:
        """Persist the exact model-visible Context before querying the provider."""

        turn_dir = self.contexts_dir / f"turn_{int(turn):04d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        canvas_path = turn_dir / "canvas.png"
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(canvas_path)

        # Keep the historical flat Canvas name while the run is in flight.  It
        # is an artifact alias, not an independent source of Context truth.
        flat_path = self.dir / f"context_{int(turn) - 1:04d}.png"
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(flat_path)

        payload = {
            **snapshot,
            "turn": int(turn),
            "canvas": str(canvas_path.relative_to(self.dir)),
        }
        context_path = turn_dir / "context.json"
        _write_json(context_path, payload)
        relative = str(context_path.relative_to(self.dir))
        self.log_event(
            "turn_context_ready",
            {
                "turn": int(turn),
                "revision": payload.get("revision"),
                "context": relative,
                "canvas": payload["canvas"],
            },
        )
        return relative

    def log_model_request(
        self,
        *,
        turn: int,
        attempt: int,
        owner: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> str:
        """保存模型实际可见的请求，并移除体积巨大的内联图片数据。"""

        directory = _model_io_directory(self.contexts_dir, turn, attempt)
        path = directory / "request.json"
        _write_json(
            path,
            {
                "schema": "vaw-model-request-v1",
                "owner": owner,
                "turn": int(turn),
                "attempt": int(attempt),
                "messages": _sanitize_model_payload(messages),
                "tools": _sanitize_model_payload(tools),
            },
        )
        relative = str(path.relative_to(self.dir))
        self.log_event(
            "model_request_saved",
            {"turn": int(turn), "attempt": int(attempt), "request": relative},
        )
        return relative

    def log_model_response(
        self,
        *,
        turn: int,
        attempt: int,
        owner: str,
        response: Any | None = None,
        error: str | None = None,
    ) -> str:
        """保存未经清理的模型文本、推理文本、结构化调用与用量。"""

        directory = _model_io_directory(self.contexts_dir, turn, attempt)
        path = directory / "response.json"
        payload: dict[str, Any] = {
            "schema": "vaw-model-response-v1",
            "owner": owner,
            "turn": int(turn),
            "attempt": int(attempt),
        }
        if response is not None:
            payload.update(_model_response_payload(response))
        if error is not None:
            payload["error"] = str(error)
        _write_json(path, payload)
        relative = str(path.relative_to(self.dir))
        self.log_event(
            "model_response_saved",
            {
                "turn": int(turn),
                "attempt": int(attempt),
                "response": relative,
                "error": error,
            },
        )
        return relative

    def log_turn(
        self,
        *,
        turn: int,
        agent_owner: str,
        image: np.ndarray,
        packet: ContextPacket,
        function_call: dict[str, Any] | None,
        step: ContextStepResult | None,
        thought: str,
        env_success: bool | None,
        done: bool,
        raw_response_text: str = "",
        provider_reasoning: str = "",
        state_summary: dict[str, Any] | None = None,
        embodied_state_card: dict[str, Any] | None = None,
        interaction_memory_before: list[dict[str, Any]] | None = None,
        interaction_event: dict[str, Any] | None = None,
        interaction_memory_after: list[dict[str, Any]] | None = None,
        function_effect_kind: str | None = None,
        context_snapshot_ref: str | None = None,
        action_segment: dict[str, Any] | None = None,
    ) -> pathlib.Path:
        name = f"context_{self.index:04d}.png"
        path = self.dir / name
        if not path.is_file():
            Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
        record: dict[str, Any] = {
            "index": self.index,
            "turn": turn,
            "agent_owner": agent_owner,
            "context_schema_version": packet.schema,
            "context_image": name,
            "context_snapshot": context_snapshot_ref,
            "context_shape": list(np.asarray(image).shape),
            "context_packet": packet.summary(),
            "decision_mode": packet.decision.mode,
            # This manifest describes the image and text that were actually
            # visible when the call was chosen.  A physical call can advance
            # the revision, so its post-call manifest is logged separately.
            "context_manifest": _packet_manifest(packet),
            "result_manifest": step.manifest if step is not None else None,
            "function_call": function_call,
            "function_result": step.result if step is not None else None,
            "runtime_diagnostics": (
                step.trace_diagnostics if step is not None else None
            ),
            "revision_before": step.revision_before if step is not None else packet.revision,
            "revision_after": step.revision_after if step is not None else packet.revision,
            "decision_basis": thought,
            # Backward-compatible trace key used by existing M1.3 readers.
            "thought": thought,
            "raw_response_text": raw_response_text,
            "provider_reasoning": provider_reasoning,
            "state_summary": state_summary,
            "embodied_state_card": embodied_state_card,
            "interaction_memory_before": interaction_memory_before or [],
            "interaction_event": interaction_event,
            "interaction_memory_after": interaction_memory_after or [],
            "function_effect_kind": function_effect_kind,
            "action_segment": action_segment,
            "env_reward": 1.0 if env_success else 0.0,
            "env_success": env_success,
            "done": done,
        }
        _append_jsonl(self.steps_path, record)
        self.index += 1
        self.log_event(
            "turn_closed",
            {
                "turn": int(turn),
                "status": "error" if step is not None and not step.ok else "complete",
                "function": (function_call or {}).get("name"),
                "revision_before": record["revision_before"],
                "revision_after": record["revision_after"],
            },
        )
        return path

    def log_meta(self, values: dict[str, Any]) -> None:
        current: dict[str, Any] = {}
        if self.meta_path.exists():
            current = json.loads(self.meta_path.read_text(encoding="utf-8"))
        current.update(values)
        self.meta_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def log_event(self, event_type: str, values: dict[str, Any]) -> None:
        """Write runtime-only control provenance that is never policy-visible."""

        with self._event_lock:
            self._event_seq += 1
            record = {
                **values,
                "event_type": str(event_type),
                "event_seq": self._event_seq,
                "time_unix_s": time.time(),
                "elapsed_s": round(time.monotonic() - self._started_monotonic, 6),
            }
            _append_jsonl(self.events_path, record)

    def new_subagent_trace(
        self,
        index: int,
        instruction: str,
    ) -> SubagentTraceLogger:
        """Create an isolated trace for one synchronous Imagination call."""

        return SubagentTraceLogger(
            self.dir / "subagents" / f"imagination_{index:04d}",
            instruction=instruction,
        )


class SubagentTraceLogger:
    """Nested trace that never becomes part of Main's policy history."""

    def __init__(self, trace_dir: str | pathlib.Path, *, instruction: str) -> None:
        self.dir = pathlib.Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for artifact in self.dir.glob("context_*.png"):
            artifact.unlink()
        self.steps_path = self.dir / "steps.jsonl"
        self.meta_path = self.dir / "meta.json"
        self.steps_path.unlink(missing_ok=True)
        self.meta_path.write_text(
            json.dumps(
                {"agent": "imagination", "instruction": instruction},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.index = 0

    def log_meta(self, values: dict[str, Any]) -> None:
        current: dict[str, Any] = {}
        if self.meta_path.exists():
            current = json.loads(self.meta_path.read_text(encoding="utf-8"))
        current.update(values)
        self.meta_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def log_turn(
        self,
        *,
        turn: int,
        image: np.ndarray,
        packet: ContextPacket,
        function_call: dict[str, Any] | None,
        step: ContextStepResult | None,
        thought: str,
        result: dict[str, Any],
    ) -> pathlib.Path:
        name = f"context_{self.index:04d}.png"
        path = self.dir / name
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
        record = {
            "index": self.index,
            "turn": turn,
            "context_image": name,
            "context_shape": list(np.asarray(image).shape),
            "context_packet": packet.summary(),
            "function_call": function_call,
            "function_result": result,
            "runtime_diagnostics": (
                step.trace_diagnostics if step is not None else None
            ),
            "decision_basis": thought,
        }
        with self.steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.index += 1
        return path

    def log_model_request(
        self,
        *,
        turn: int,
        attempt: int,
        owner: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> str:
        directory = _model_io_directory(self.dir / "contexts", turn, attempt)
        path = directory / "request.json"
        _write_json(
            path,
            {
                "schema": "vaw-model-request-v1",
                "owner": owner,
                "turn": int(turn),
                "attempt": int(attempt),
                "messages": _sanitize_model_payload(messages),
                "tools": _sanitize_model_payload(tools),
            },
        )
        return str(path.relative_to(self.dir))

    def log_model_response(
        self,
        *,
        turn: int,
        attempt: int,
        owner: str,
        response: Any | None = None,
        error: str | None = None,
    ) -> str:
        directory = _model_io_directory(self.dir / "contexts", turn, attempt)
        path = directory / "response.json"
        payload: dict[str, Any] = {
            "schema": "vaw-model-response-v1",
            "owner": owner,
            "turn": int(turn),
            "attempt": int(attempt),
        }
        if response is not None:
            payload.update(_model_response_payload(response))
        if error is not None:
            payload["error"] = str(error)
        _write_json(path, payload)
        return str(path.relative_to(self.dir))


def _packet_manifest(packet: ContextPacket) -> dict[str, Any]:
    return packet.manifest()


def _model_io_directory(root: pathlib.Path, turn: int, attempt: int) -> pathlib.Path:
    directory = root / f"turn_{int(turn):04d}" / "model_io" / f"attempt_{int(attempt):02d}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _sanitize_model_payload(value: Any) -> Any:
    """保留请求结构，但不复制 base64 图片或潜在鉴权字段。"""

    if isinstance(value, list):
        return [_sanitize_model_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_model_payload(item) for item in value]
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).casefold()
            if lowered in {"authorization", "api_key", "apikey", "headers"}:
                clean[str(key)] = "[已隐藏]"
                continue
            if key == "url" and isinstance(item, str) and item.startswith("data:"):
                mime = item[5:].split(";", 1)[0] or "application/octet-stream"
                clean[str(key)] = f"[内联 {mime} 已省略；请查看该轮画布或知识参考图]"
                continue
            clean[str(key)] = _sanitize_model_payload(item)
        return clean
    if isinstance(value, str) and value.startswith("data:"):
        mime = value[5:].split(";", 1)[0] or "application/octet-stream"
        return f"[内联 {mime} 已省略；请查看该轮画布或知识参考图]"
    return value


def _model_response_payload(response: Any) -> dict[str, Any]:
    calls = []
    for call in getattr(response, "tool_calls", ()) or ():
        calls.append(
            {
                "id": getattr(call, "id", None),
                "name": getattr(call, "name", None),
                "arguments": _sanitize_model_payload(getattr(call, "args", {})),
                "parse_error": getattr(call, "parse_error", None),
            }
        )
    return {
        "raw_response_text": str(getattr(response, "raw_response_text", "") or ""),
        "parsed_text": str(getattr(response, "text", "") or ""),
        "provider_reasoning": str(getattr(response, "provider_reasoning", "") or ""),
        "tool_calls": calls,
        "finish_reason": getattr(response, "finish_reason", None),
        "usage": _sanitize_model_payload(getattr(response, "usage", None)),
    }


def _append_jsonl(path: pathlib.Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


__all__ = ["ContextTraceLogger", "SubagentTraceLogger"]
