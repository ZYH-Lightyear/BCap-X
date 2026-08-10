"""Trace format for policy-visible revision-local Context turns."""

from __future__ import annotations

import json
import pathlib
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
        self.steps_path.unlink(missing_ok=True)
        self.events_path.unlink(missing_ok=True)
        self.meta_path.unlink(missing_ok=True)
        self.index = 0

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
    ) -> pathlib.Path:
        name = f"context_{self.index:04d}.png"
        path = self.dir / name
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
        record: dict[str, Any] = {
            "index": self.index,
            "turn": turn,
            "agent_owner": agent_owner,
            "context_schema_version": packet.schema,
            "context_image": name,
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
            "env_reward": 1.0 if env_success else 0.0,
            "env_success": env_success,
            "done": done,
        }
        with self.steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.index += 1
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

        record = {"event_type": str(event_type), **values}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _packet_manifest(packet: ContextPacket) -> dict[str, Any]:
    return packet.manifest()


__all__ = ["ContextTraceLogger"]
