"""Structured event logging for RoboMEx runs.

The normal ``run.log`` is optimized for humans tailing a run.  This module writes a
parallel JSONL stream that is stable enough for a debug UI and post-run analysis.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

_EVENT_LOG: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "robomex_event_log", default=None
)
_EVENT_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "robomex_event_context", default={}
)


def event_log_path() -> Path | None:
    """Return the active JSONL event path, if structured logging is enabled."""

    return _EVENT_LOG.get()


@contextlib.contextmanager
def event_log(path: str | Path | None) -> Iterator[None]:
    """Enable structured events for the current context."""

    if path is None:
        yield
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")
    token = _EVENT_LOG.set(p)
    try:
        yield
    finally:
        _EVENT_LOG.reset(token)


@contextlib.contextmanager
def event_scope(**fields: Any) -> Iterator[None]:
    """Temporarily add fields to every event emitted in this context."""

    base = dict(_EVENT_CONTEXT.get())
    base.update({k: v for k, v in fields.items() if v is not None})
    token = _EVENT_CONTEXT.set(base)
    try:
        yield
    finally:
        _EVENT_CONTEXT.reset(token)


def emit_event(event: str, message: str = "", **fields: Any) -> None:
    """Append one structured event if an event log is active."""

    path = _EVENT_LOG.get()
    record = {
        "id": uuid.uuid4().hex,
        "ts": datetime.now(timezone.utc).isoformat(),
        "t_rel": time.monotonic(),
        "event": event,
        "message": message,
        **_EVENT_CONTEXT.get(),
        **{k: v for k, v in fields.items() if v is not None},
    }
    _emit_human_event(record)
    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    except Exception:
        # Event logging is diagnostic only; never break robot execution.
        return


def preview(text: Any, limit: int = 500) -> str:
    """Return a compact single-field preview for large strings."""

    s = "" if text is None else str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + f"... <truncated {len(s) - limit} chars>"


def _emit_human_event(record: dict[str, Any]) -> None:
    line = _human_event_line(record)
    if not line:
        return
    logging.getLogger("robomex.trace").info(line)


def _human_event_line(record: dict[str, Any]) -> str:
    """Return a compact console line for high-signal events."""

    event = str(record.get("event", ""))
    role = _role_tag(record)
    turn = _turn_tag(record)
    budget = _budget_tag(record)
    prefix = f"{role}{turn}{budget}".strip()

    if event == "episode_start":
        return f"=== EPISODE START === {preview(record.get('task'), 110)}"
    if event == "subgoal_start":
        num = record.get("subgoal_number", "?")
        return f"=== SUBGOAL {num} START === {preview(record.get('goal'), 130)}"
    if event == "subgoal_end":
        num = record.get("subgoal_number", "?")
        status = "success" if record.get("success") else "unresolved"
        return f"=== SUBGOAL {num} END === {status}"
    if event == "agent_start":
        return f"{prefix} start max_actions={record.get('max_action_turns', '?')}"
    if event == "agent_end":
        status = "stopped" if record.get("stopped") else "exhausted"
        return (
            f"{prefix} end {status} actions={record.get('action_turns', 0)}/"
            f"{record.get('max_action_turns', '?')} decisions={record.get('decision_turns', '?')}"
        )
    if event == "agent_action":
        action = record.get("action", "?")
        payload = preview(record.get("payload_preview", ""), 120)
        return f"{prefix} action={action}: {payload}"
    if event == "skill_loaded":
        return f"{prefix} skill loaded: {record.get('skill')}"
    if event == "python_blocked":
        return f"{prefix} python BLOCKED: {preview(record.get('reason'), 180)}"
    if event == "repeat_warning":
        return f"{prefix} repeat warning x{record.get('repeats')}"
    if event == "code_execution_start":
        return f"{prefix} python start: {record.get('line_count', '?')} lines"
    if event == "code_execution_result":
        status = record.get("status", "?")
        ok = "ok" if record.get("ok") else "failed"
        stdout = _last_nonempty_line(record.get("stdout", ""))
        stderr = _last_nonempty_line(record.get("stderr", ""))
        detail = f" stdout={preview(stdout, 100)}" if stdout else ""
        if stderr:
            detail += f" stderr={preview(stderr, 120)}"
        return (
            f"{prefix} python result: {status}/{ok} reward={record.get('reward')} "
            f"terminated={record.get('terminated')} duration={record.get('duration_s', '?')}s{detail}"
        )
    if event == "terminal_review":
        if record.get("should_stop"):
            return f"{prefix} terminal accepted"
        return f"{prefix} terminal rejected: {preview(record.get('feedback'), 180)}"
    if event == "action_budget_exhausted":
        return f"{prefix} action budget exhausted"
    if event == "decision_budget_exhausted":
        return f"{prefix} decision budget exhausted"
    if event == "force_terminal":
        return f"{prefix} force terminal after action budget"
    return ""


def _role_tag(record: dict[str, Any]) -> str:
    role = str(record.get("agent_role") or "").lower()
    if role:
        return f"[{role.upper()}]"
    if str(record.get("event", "")).startswith("subgoal"):
        return "[SUBGOAL]"
    return "[RUN]"


def _turn_tag(record: dict[str, Any]) -> str:
    turn = record.get("turn")
    return "" if turn is None else f" t{int(turn):02d}"


def _budget_tag(record: dict[str, Any]) -> str:
    action_turns = record.get("action_turns")
    max_action_turns = record.get("max_action_turns")
    if action_turns is None or max_action_turns is None:
        return ""
    return f" a{action_turns}/{max_action_turns}"


def _last_nonempty_line(text: Any) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return repr(obj)
