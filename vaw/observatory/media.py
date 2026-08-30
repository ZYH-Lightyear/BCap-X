"""Per-Function physical video segments for the Agent OS Observatory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from vaw.context_runtime.trace import ContextTraceLogger


@dataclass(frozen=True)
class ActionCaptureToken:
    segment_id: str
    turn: int
    function: str
    arguments: dict[str, Any]
    revision_before: int
    frame_start: int | None


class ActionMediaRecorder:
    """Cut the continuously captured simulator stream at physical calls."""

    def __init__(
        self,
        env: Any,
        trace: ContextTraceLogger,
        *,
        fps: int = 30,
    ) -> None:
        if fps <= 0:
            raise ValueError("action video fps must be positive")
        self.env = env
        self.trace = trace
        self.fps = int(fps)
        self._sequence = 0

    def start(
        self,
        *,
        turn: int,
        function: str,
        arguments: dict[str, Any],
        revision_before: int,
    ) -> ActionCaptureToken:
        self._sequence += 1
        segment_id = f"action_{self._sequence:04d}"
        frame_start = self._frame_count()
        token = ActionCaptureToken(
            segment_id=segment_id,
            turn=int(turn),
            function=str(function),
            arguments=dict(arguments),
            revision_before=int(revision_before),
            frame_start=frame_start,
        )
        directory = self.trace.actions_dir / segment_id
        directory.mkdir(parents=True, exist_ok=True)
        self.trace.log_event(
            "action_segment_started",
            {
                "turn": token.turn,
                "segment_id": segment_id,
                "function": token.function,
                "revision_before": token.revision_before,
                "frame_start": frame_start,
            },
        )
        return token

    def finish(
        self,
        token: ActionCaptureToken,
        *,
        outcome: str,
        revision_after: int,
    ) -> dict[str, Any]:
        directory = self.trace.actions_dir / token.segment_id
        directory.mkdir(parents=True, exist_ok=True)
        frame_end = self._frame_count()
        manifest: dict[str, Any] = {
            "schema": "vaw-action-segment-v1",
            "segment_id": token.segment_id,
            "turn": token.turn,
            "function": token.function,
            "arguments": token.arguments,
            "outcome": str(outcome),
            "revision_before": token.revision_before,
            "revision_after": int(revision_after),
            "frame_start": token.frame_start,
            "frame_end": frame_end,
            "fps": self.fps,
            "status": "empty",
            "streams": {},
        }
        try:
            if token.frame_start is None or frame_end is None:
                manifest["status"] = "unavailable"
            else:
                streams: dict[str, dict[str, Any]] = {}
                agentview = self._frames_range(
                    "get_video_frames_range", token.frame_start, frame_end
                )
                wrist = self._frames_range(
                    "get_wrist_video_frames_range", token.frame_start, frame_end
                )
                for name, frames in (("agentview", agentview), ("wrist", wrist)):
                    if not frames:
                        continue
                    path = directory / f"{name}.mp4"
                    count = _write_mp4(path, frames, fps=self.fps)
                    streams[name] = {
                        "path": str(path.relative_to(self.trace.dir)),
                        "frames": count,
                        "fps": self.fps,
                        "duration_s": round(count / self.fps, 4),
                    }
                if agentview:
                    poster = directory / "poster.jpg"
                    Image.fromarray(np.asarray(agentview[-1], dtype=np.uint8)).save(
                        poster,
                        quality=88,
                    )
                    manifest["poster"] = str(poster.relative_to(self.trace.dir))
                manifest["streams"] = streams
                manifest["status"] = "ready" if streams else "empty"
        except Exception as exc:
            manifest["status"] = "error"
            manifest["error"] = f"{type(exc).__name__}: {exc}"

        manifest_path = directory / "manifest.json"
        _write_json(manifest_path, manifest)
        relative = str(manifest_path.relative_to(self.trace.dir))
        summary = {
            "segment_id": token.segment_id,
            "manifest": relative,
            "status": manifest["status"],
            "streams": manifest["streams"],
            "poster": manifest.get("poster"),
        }
        self.trace.log_event(
            "action_segment_ready",
            {
                "turn": token.turn,
                "segment_id": token.segment_id,
                "manifest": relative,
                "status": manifest["status"],
                "revision_after": int(revision_after),
            },
        )
        return summary

    def _frame_count(self) -> int | None:
        function = getattr(self.env, "get_video_frame_count", None)
        if not callable(function):
            return None
        try:
            return int(function())
        except Exception:
            return None

    def _frames_range(self, name: str, start: int, end: int) -> list[np.ndarray]:
        function = getattr(self.env, name, None)
        if not callable(function):
            return []
        frames = function(start, end)
        return [np.asarray(frame, dtype=np.uint8) for frame in frames]


def _write_mp4(path: Path, frames: list[np.ndarray], *, fps: int) -> int:
    with imageio.get_writer(
        path,
        fps=fps,
        format="FFMPEG",
        codec="libx264",
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            rgb = np.asarray(frame, dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"video frame must be HxWx3, got {rgb.shape}")
            writer.append_data(np.ascontiguousarray(rgb))
    return len(frames)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = ["ActionCaptureToken", "ActionMediaRecorder"]
