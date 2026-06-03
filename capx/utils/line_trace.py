"""Line-level tracing for generated Python code execution.

The tracer is intentionally scoped to the synthetic filename used for a single
``exec`` call. It records user-code line events without stepping into libraries.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any


FrameCountFn = Callable[[], int]
EmitFn = Callable[[dict[str, Any]], None]


def _now() -> float:
    return time.time()


@dataclass
class LineTraceRecorder:
    """Collect and optionally emit line-level execution events."""

    block_index: int
    code: str
    get_frame_count: FrameCountFn
    emit_callback: EmitFn | None = None
    filename: str = field(init=False)
    events: list[dict[str, Any]] = field(default_factory=list)
    _source_lines: list[str] = field(init=False)
    _active: dict[str, Any] | None = None
    _next_line_index: int = 0

    def __post_init__(self) -> None:
        self.filename = f"<capx-code-block-{self.block_index}>"
        self._source_lines = self.code.splitlines()

    def compile(self) -> Any:
        return compile(self.code, self.filename, "exec")

    def record_stdout(self, text: str) -> None:
        self._record_stream("stdout_delta", text)

    def record_stderr(self, text: str) -> None:
        self._record_stream("stderr_delta", text)

    def trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        if frame.f_code.co_filename != self.filename:
            return None

        if event == "line":
            self._complete_active()
            self._start_line(frame.f_lineno)
        elif event == "exception":
            self._mark_exception(arg)
        elif event == "return":
            self._complete_active()
        return self.trace

    def finish(self) -> list[dict[str, Any]]:
        self._complete_active()
        return list(self.events)

    def _start_line(self, lineno: int) -> None:
        source = ""
        if 1 <= lineno <= len(self._source_lines):
            source = self._source_lines[lineno - 1]
        self._active = {
            "phase": "start",
            "block_index": self.block_index,
            "line_index": self._next_line_index,
            "lineno": lineno,
            "source": source,
            "frame_start": self.get_frame_count(),
            "frame_end": self.get_frame_count(),
            "stdout_delta": "",
            "stderr_delta": "",
            "trace_time": _now(),
        }
        self._next_line_index += 1
        self._emit(self._active)

    def _complete_active(self) -> None:
        if self._active is None:
            return
        event = dict(self._active)
        event["phase"] = "complete"
        event["frame_end"] = self.get_frame_count()
        event["trace_time"] = _now()
        self._emit(event)
        self._active = None

    def _mark_exception(self, arg: Any) -> None:
        if self._active is None:
            return
        exc_type, exc_value, _tb = arg
        self._active["exception_type"] = getattr(exc_type, "__name__", str(exc_type))
        self._active["exception_message"] = str(exc_value)
        event = dict(self._active)
        event["phase"] = "exception"
        event["frame_end"] = self.get_frame_count()
        event["trace_time"] = _now()
        self._emit(event)

    def _record_stream(self, field_name: str, text: str) -> None:
        if self._active is None or not text:
            return
        self._active[field_name] += text
        event = dict(self._active)
        event["phase"] = "update"
        event["frame_end"] = self.get_frame_count()
        event["trace_time"] = _now()
        self._emit(event)

    def _emit(self, event: dict[str, Any]) -> None:
        event_copy = dict(event)
        self.events.append(event_copy)
        if self.emit_callback is not None:
            self.emit_callback(event_copy)


@contextmanager
def tracing(recorder: LineTraceRecorder):
    previous_trace = sys.gettrace()
    sys.settrace(recorder.trace)
    try:
        yield recorder
    finally:
        sys.settrace(previous_trace)
        recorder.finish()


def save_events_jsonl(events: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
