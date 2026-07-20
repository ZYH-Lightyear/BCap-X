"""SSE live tail for events.jsonl."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

_LARGE_KEYS = {
    "raw",
    "code",
    "stdout",
    "stderr",
    "prompt_preview",
    "payload_preview",
    "terminal_raw",
    "initial_user_message",
}


def summarize_event(record: dict[str, Any], *, limit: int = 240) -> dict[str, Any]:
    """Keep live payloads small for SSE fan-out."""

    out: dict[str, Any] = {}
    for key in (
        "id",
        "ts",
        "t_rel",
        "event",
        "message",
        "task",
        "subgoal_index",
        "subgoal_number",
        "agent_role",
        "agent_label",
        "agent_id",
        "node_id",
        "turn",
        "action",
        "status",
        "ok",
        "success",
        "duration_s",
        "goal",
        "llm_request_path",
        "llm_response_path",
    ):
        if key in record and record[key] is not None:
            out[key] = record[key]
    for key in _LARGE_KEYS:
        if key in record and record[key] is not None:
            text = str(record[key])
            if len(text) > limit:
                text = text[:limit] + f"... <truncated {len(text) - limit} chars>"
            out[key] = text
    return out


async def tail_events_jsonl(
    path: Path,
    *,
    from_start: bool = False,
    poll_s: float = 0.5,
    heartbeat_s: float = 15.0,
) -> AsyncIterator[str]:
    """Yield SSE frames for new JSONL lines.

    Each data frame is a JSON object (summarized event). Heartbeats are SSE comments.
    """

    offset = 0 if from_start else (path.stat().st_size if path.exists() else 0)
    last_heartbeat = time.monotonic()
    partial = ""

    while True:
        if not path.exists():
            await asyncio.sleep(poll_s)
            continue
        size = path.stat().st_size
        if size < offset:
            # file truncated / rewritten
            offset = 0
            partial = ""
        if size > offset:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            partial += chunk
            while "\n" in partial:
                line, partial = partial.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = {"event": "malformed_event", "message": line[:500]}
                if not isinstance(record, dict):
                    continue
                payload = json.dumps(summarize_event(record), ensure_ascii=False)
                yield f"data: {payload}\n\n"
                last_heartbeat = time.monotonic()
        now = time.monotonic()
        if now - last_heartbeat >= heartbeat_s:
            yield ": heartbeat\n\n"
            last_heartbeat = now
        await asyncio.sleep(poll_s)
