"""Project append-only VAW traces into read-only Observatory views."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL records while another process may append a line."""

    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                break
            if isinstance(value, dict):
                records.append(value)
    return records


def discover_run_dirs(root: Path) -> list[Path]:
    """Discover trace runs below an Observatory workspace.

    A workspace may contain independent experiment collections such as
    ``context_runs`` and ``sweeps``.  Once a run is found, its artifact tree is
    pruned so Canvas, video, and nested Imagination traces are not mistaken for
    episodes.
    """

    if _is_run_dir(root):
        return [root]
    if not root.is_dir():
        return []
    runs: list[Path] = []
    for directory, child_dirs, file_names in os.walk(root):
        path = Path(directory)
        if _is_run_dir_from_names(path, set(file_names), set(child_dirs)):
            runs.append(path)
            child_dirs[:] = []
            continue
        child_dirs[:] = [
            name
            for name in child_dirs
            if name not in {"actions", "contexts", "subagents", "artifacts", "media"}
        ]
    runs.sort(key=_run_mtime, reverse=True)
    return runs


def summarize_run(run_dir: Path, *, workspace_root: Path | None = None) -> dict[str, Any]:
    meta = _read_json(run_dir / "meta.json")
    launcher = _read_json(run_dir / "launcher.json")
    steps = read_jsonl(run_dir / "steps.jsonl")
    events = read_jsonl(run_dir / "runtime_events.jsonl")
    closed = any(event.get("event_type") == "episode_closed" for event in events)
    latest = events[-1] if events else {}
    root = (workspace_root or run_dir.parent).resolve()
    run_id = encode_run_id(root, run_dir)
    relative = _relative_label(root, run_dir)
    collection = relative.split("/", 1)[0] if "/" in relative else "."
    launch_status = launcher.get("status")
    launch_pid = launcher.get("pid")
    launch_start_ticks = launcher.get("process_start_ticks")
    if (
        launch_status in {"starting", "running", "stopping"}
        and isinstance(launch_pid, int)
        and (
            not isinstance(launch_start_ticks, int)
            or _proc_start_ticks(launch_pid) != launch_start_ticks
        )
    ):
        launch_status = "exited"
    live = launch_status in {"starting", "running", "stopping"} or (
        not closed and bool(events)
    )
    return {
        "run_id": run_id,
        "run_name": run_dir.name,
        "display_name": (
            _nested(launcher, "spec", "run_name")
            or meta.get("run_name")
            or run_dir.name
        ),
        "relative_path": relative,
        "collection": collection,
        "task": meta.get("task_prompt"),
        "suite": meta.get("suite") or _nested(launcher, "spec", "suite"),
        "task_id": meta.get("task_id", _nested(launcher, "spec", "task_id")),
        "seed": meta.get("seed", _nested(launcher, "spec", "seed")),
        "model": meta.get("model") or _nested(launcher, "spec", "model"),
        "turns": len(steps),
        "revision": _latest_revision(steps),
        "status": (
            meta.get("terminate_mode")
            or launch_status
            or ("live" if not closed else "closed")
        ),
        "live": live,
        "env_success": meta.get("env_success"),
        "job_id": launcher.get("job_id"),
        "launcher_status": launch_status,
        "launcher_log": (
            "artifacts/launcher.log" if (run_dir / "launcher.log").is_file() else None
        ),
        "latest_event_seq": int(latest.get("event_seq", 0) or 0),
        "elapsed_s": latest.get("elapsed_s"),
    }


def compile_snapshot(
    run_dir: Path,
    *,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    meta = _read_json(run_dir / "meta.json")
    steps = read_jsonl(run_dir / "steps.jsonl")
    events = read_jsonl(run_dir / "runtime_events.jsonl")
    steps_by_turn = {
        int(step["turn"]): step
        for step in steps
        if isinstance(step.get("turn"), int)
    }
    events_by_turn: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        turn = event.get("turn")
        if isinstance(turn, int):
            events_by_turn.setdefault(turn, []).append(event)
    contexts = _load_contexts(run_dir)
    segments = _load_segments(run_dir)
    segments_by_turn = {
        int(segment["turn"]): segment
        for segment in segments
        if isinstance(segment.get("turn"), int)
    }
    turn_ids = sorted(set(contexts) | set(steps_by_turn) | set(events_by_turn))
    turns = [
        _compile_turn(
            run_dir,
            turn,
            contexts.get(turn),
            steps_by_turn.get(turn),
            events_by_turn.get(turn, []),
            segments_by_turn.get(turn),
        )
        for turn in turn_ids
    ]
    latest = events[-1] if events else {}
    closed = any(event.get("event_type") == "episode_closed" for event in events)
    return {
        "schema": "vaw-observatory-snapshot-v1",
        "run": {
            **summarize_run(run_dir, workspace_root=workspace_root),
            "profile": {
                "motion_backend": meta.get("motion_backend"),
                "imagination_model": meta.get("imagination_model"),
            },
        },
        "turns": turns,
        "action_segments": segments,
        "episode_videos": _load_episode_videos(run_dir, meta),
        "latest_event_seq": int(latest.get("event_seq", 0) or 0),
        "closed": closed,
    }


def compile_turn(
    run_dir: Path,
    turn: int,
    *,
    workspace_root: Path | None = None,
) -> dict[str, Any] | None:
    snapshot = compile_snapshot(run_dir, workspace_root=workspace_root)
    return next((item for item in snapshot["turns"] if item["turn"] == turn), None)


def compile_model_io(run_dir: Path, turn: int) -> dict[str, Any]:
    """按需读取一轮的完整模型请求/响应，避免让 snapshot 变得臃肿。"""

    root = run_dir / "contexts" / f"turn_{int(turn):04d}" / "model_io"
    attempts: list[dict[str, Any]] = []
    if root.is_dir():
        for directory in sorted(root.glob("attempt_*")):
            request = _read_json(directory / "request.json")
            response = _read_json(directory / "response.json")
            if request or response:
                attempts.append(
                    {
                        "attempt": request.get("attempt", response.get("attempt")),
                        "request": request or None,
                        "response": response or None,
                    }
                )
    if attempts:
        return {
            "schema": "vaw-observatory-model-io-v1",
            "turn": int(turn),
            "legacy": False,
            "attempts": attempts,
        }

    # 旧 trace 没有逐次请求文件；仍提供当时冻结的文本 Context 和原始回复，
    # 但明确标注为 legacy，绝不拿当前 System Prompt 冒充历史请求。
    context = _read_json(
        run_dir / "contexts" / f"turn_{int(turn):04d}" / "context.json"
    )
    step = next(
        (
            item
            for item in read_jsonl(run_dir / "steps.jsonl")
            if item.get("turn") == int(turn)
        ),
        {},
    )
    request = None
    if context.get("prompt_text"):
        request = {
            "schema": "vaw-legacy-context-request-v1",
            "note": "旧 trace 只保存了动态上下文，未保存完整 System Prompt 与工具结构。",
            "prompt_text": context.get("prompt_text"),
            "canvas": _artifact_url(context.get("canvas")),
        }
    response = None
    if step:
        response = {
            "schema": "vaw-legacy-model-response-v1",
            "raw_response_text": step.get("raw_response_text"),
            "parsed_text": step.get("decision_basis"),
            "provider_reasoning": step.get("provider_reasoning"),
            "tool_calls": [step.get("function_call")] if step.get("function_call") else [],
        }
    return {
        "schema": "vaw-observatory-model-io-v1",
        "turn": int(turn),
        "legacy": True,
        "attempts": (
            [{"attempt": 1, "request": request, "response": response}]
            if request or response
            else []
        ),
    }


def compile_imagination(
    run_dir: Path,
    session_id: str,
) -> dict[str, Any]:
    """Project one nested Imagination trace into an inspectable turn sequence."""

    session_dir = _resolve_imagination_dir(run_dir, session_id)
    meta = _read_json(session_dir / "meta.json")
    turns = []
    for step in read_jsonl(session_dir / "steps.jsonl"):
        turn = step.get("turn")
        if not isinstance(turn, int):
            continue
        image = step.get("context_image")
        turns.append(
            {
                "turn": turn,
                "canvas": _artifact_url(
                    str((session_dir / str(image)).relative_to(run_dir.resolve()))
                ) if image else None,
                "decision_basis": step.get("decision_basis"),
                "function_call": step.get("function_call"),
                "function_result": step.get("function_result"),
                "runtime_diagnostics": step.get("runtime_diagnostics"),
                "model_io_available": _nested_model_io_available(
                    session_dir, turn, step
                ),
            }
        )
    return {
        "schema": "vaw-observatory-imagination-v1",
        "session_id": session_id,
        "instruction": meta.get("instruction"),
        "status": meta.get("status") or ("running" if turns else "pending"),
        "action_id": meta.get("action_id"),
        "reason": meta.get("reason"),
        "turns": turns,
    }


def compile_imagination_model_io(
    run_dir: Path,
    session_id: str,
    turn: int,
) -> dict[str, Any]:
    """Read raw model I/O for one internal Imagination turn."""

    session_dir = _resolve_imagination_dir(run_dir, session_id)
    root = session_dir / "contexts" / f"turn_{int(turn):04d}" / "model_io"
    attempts = _read_model_io_attempts(root)
    if attempts:
        return {
            "schema": "vaw-observatory-model-io-v1",
            "turn": int(turn),
            "legacy": False,
            "attempts": attempts,
        }
    step = next(
        (
            item
            for item in read_jsonl(session_dir / "steps.jsonl")
            if item.get("turn") == int(turn)
        ),
        {},
    )
    response = None
    if step:
        response = {
            "schema": "vaw-legacy-imagination-response-v1",
            "parsed_text": step.get("decision_basis"),
            "tool_calls": [step.get("function_call")]
            if step.get("function_call")
            else [],
        }
    return {
        "schema": "vaw-observatory-model-io-v1",
        "turn": int(turn),
        "legacy": True,
        "attempts": (
            [{"attempt": 1, "request": None, "response": response}]
            if response
            else []
        ),
    }


def _compile_turn(
    run_dir: Path,
    turn: int,
    context: dict[str, Any] | None,
    step: dict[str, Any] | None,
    events: list[dict[str, Any]],
    segment: dict[str, Any] | None,
) -> dict[str, Any]:
    call = (step or {}).get("function_call") or _latest_call(events)
    result = (step or {}).get("function_result")
    error = result.get("error") if isinstance(result, dict) else None
    advisory = result.get("advisory") if isinstance(result, dict) else None
    return {
        "turn": turn,
        "status": _turn_status(step, events),
        "context_available": context is not None,
        "context": _public_context(context),
        "canvas": _artifact_url(context.get("canvas")) if context else None,
        "decision": {
            "basis": (step or {}).get("decision_basis"),
            "function": call.get("name") if isinstance(call, dict) else None,
            "arguments": call.get("arguments") if isinstance(call, dict) else None,
            "effect_kind": (step or {}).get("function_effect_kind"),
            "result": result,
            "error": error,
            "advisory": advisory,
            "revision_before": (step or {}).get("revision_before"),
            "revision_after": (step or {}).get("revision_after"),
        },
        "interaction_event": (step or {}).get("interaction_event"),
        "action_segment": segment,
        "model_io_available": _model_io_available(run_dir, turn, step),
        "imagination": _imagination_summary(run_dir, step, events),
        "event_seq": max((int(event.get("event_seq", 0) or 0) for event in events), default=0),
    }


def _public_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    if context is None:
        return None
    return {
        key: value
        for key, value in context.items()
        if key != "prompt_text"
    }


def _turn_status(step: dict[str, Any] | None, events: list[dict[str, Any]]) -> str:
    if step is not None:
        result = step.get("function_result")
        return "error" if isinstance(result, dict) and result.get("error") else "complete"
    kinds = [event.get("event_type") for event in events]
    for event_type, status in (
        ("action_segment_started", "action_running"),
        ("function_started", "function_selected"),
        ("model_decision_ready", "function_selected"),
        ("model_started", "thinking"),
        ("turn_context_ready", "context_ready"),
    ):
        if event_type in kinds:
            return status
    return "pending"


def _latest_call(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("function"):
            return {
                "name": event.get("function"),
                "arguments": event.get("arguments") or {},
            }
    return {}


def _load_contexts(run_dir: Path) -> dict[int, dict[str, Any]]:
    contexts: dict[int, dict[str, Any]] = {}
    root = run_dir / "contexts"
    if not root.is_dir():
        return contexts
    for path in sorted(root.glob("turn_*/context.json")):
        value = _read_json(path)
        turn = value.get("turn")
        if isinstance(turn, int):
            contexts[turn] = value
    return contexts


def _load_segments(run_dir: Path) -> list[dict[str, Any]]:
    segments = []
    root = run_dir / "actions"
    if not root.is_dir():
        return segments
    for path in sorted(root.glob("action_*/manifest.json")):
        value = _read_json(path)
        if value:
            value["manifest"] = _artifact_url(str(path.relative_to(run_dir)))
            if value.get("poster"):
                value["poster"] = _artifact_url(str(value["poster"]))
            streams = value.get("streams")
            if isinstance(streams, dict):
                for stream in streams.values():
                    if isinstance(stream, dict) and stream.get("path"):
                        stream["path"] = _artifact_url(str(stream["path"]))
            segments.append(value)
    return segments


def _load_episode_videos(
    run_dir: Path,
    meta: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    values = meta.get("videos")
    videos: dict[str, dict[str, Any]] = {}
    if isinstance(values, dict):
        for name, raw in values.items():
            if not isinstance(raw, dict) or not raw.get("path"):
                continue
            path = run_dir / str(raw["path"])
            if not path.is_file():
                continue
            videos[str(name)] = {
                **raw,
                "path": _artifact_url(str(raw["path"])),
            }
    if videos:
        return videos
    for path in sorted(run_dir.glob("video_*.mp4")):
        name = path.stem.removeprefix("video_")
        videos[name] = {"path": _artifact_url(path.name)}
    return videos


def _model_io_available(
    run_dir: Path,
    turn: int,
    step: dict[str, Any] | None,
) -> bool:
    root = run_dir / "contexts" / f"turn_{int(turn):04d}" / "model_io"
    return root.is_dir() or bool(
        step
        and (
            step.get("raw_response_text")
            or step.get("decision_basis")
            or step.get("provider_reasoning")
        )
    )


def _imagination_summary(
    run_dir: Path,
    step: dict[str, Any] | None,
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    is_closed_call = bool(
        step and (step.get("function_call") or {}).get("name") == "imagine_action"
    )
    start_event = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") == "imagination_started"
            and isinstance(event.get("trace"), str)
        ),
        None,
    )
    if not is_closed_call and start_event is None:
        return None
    trace = (
        _nested(step or {}, "runtime_diagnostics", "subagent", "trace")
        or (start_event or {}).get("trace")
    )
    if not isinstance(trace, str):
        return None
    session_id = Path(trace).name
    try:
        session_dir = _resolve_imagination_dir(run_dir, session_id)
    except ValueError:
        return None
    meta = _read_json(session_dir / "meta.json")
    turns = read_jsonl(session_dir / "steps.jsonl")
    return {
        "session_id": session_id,
        "instruction": meta.get("instruction")
        or ((step or {}).get("function_call") or {}).get("arguments", {}).get("instruction")
        or (start_event or {}).get("instruction"),
        "status": meta.get("status")
        or _nested(step or {}, "runtime_diagnostics", "subagent", "status")
        or "running",
        "action_id": meta.get("action_id")
        or ((step or {}).get("function_result") or {}).get("action_id")
        or (start_event or {}).get("action_id"),
        "reason": meta.get("reason")
        or ((step or {}).get("function_result") or {}).get("reason"),
        "turn_count": len(turns),
    }


def _resolve_imagination_dir(run_dir: Path, session_id: str) -> Path:
    if re.fullmatch(r"imagination_\d{4}", session_id) is None:
        raise ValueError("invalid imagination session id")
    root = (run_dir / "subagents").resolve()
    target = (root / session_id).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("imagination trace escapes run") from exc
    if not target.is_dir():
        raise ValueError("imagination trace not found")
    return target


def _read_model_io_attempts(root: Path) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    if not root.is_dir():
        return attempts
    for directory in sorted(root.glob("attempt_*")):
        request = _read_json(directory / "request.json")
        response = _read_json(directory / "response.json")
        if request or response:
            attempts.append(
                {
                    "attempt": request.get("attempt", response.get("attempt")),
                    "request": request or None,
                    "response": response or None,
                }
            )
    return attempts


def _nested_model_io_available(
    session_dir: Path,
    turn: int,
    step: dict[str, Any],
) -> bool:
    root = session_dir / "contexts" / f"turn_{int(turn):04d}" / "model_io"
    return root.is_dir() or bool(step.get("decision_basis"))


def _artifact_url(relative: str | None) -> str | None:
    return f"artifacts/{relative}" if relative else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and any(
        (path / name).exists()
        for name in (
            "launcher.json",
            "meta.json",
            "steps.jsonl",
            "runtime_events.jsonl",
            "contexts",
        )
    )


def _is_run_dir_from_names(
    path: Path,
    files: set[str],
    directories: set[str],
) -> bool:
    if not path.is_dir():
        return False
    return bool(
        files
        & {"launcher.json", "meta.json", "steps.jsonl", "runtime_events.jsonl"}
        or "contexts" in directories
    )


def encode_run_id(workspace_root: Path, run_dir: Path) -> str:
    root = workspace_root.resolve()
    target = run_dir.resolve()
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("run is outside Observatory workspace") from exc
    encoded = base64.urlsafe_b64encode(relative.encode("utf-8")).decode("ascii")
    return f"run-{encoded.rstrip('=')}"


def decode_run_id(workspace_root: Path, run_id: str) -> Path:
    if not run_id.startswith("run-"):
        raise ValueError("invalid run id")
    encoded = run_id[4:]
    padding = "=" * (-len(encoded) % 4)
    try:
        relative = base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid run id") from exc
    root = workspace_root.resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("run is outside Observatory workspace") from exc
    return target


def _relative_label(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _proc_start_ticks(pid: int) -> int | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields_after_name = value[value.rfind(")") + 2 :].split()
        return int(fields_after_name[19])
    except (OSError, ValueError, IndexError):
        return None


def _run_mtime(path: Path) -> float:
    candidates = [
        path / "runtime_events.jsonl",
        path / "steps.jsonl",
        path / "meta.json",
        path / "launcher.json",
    ]
    return max((item.stat().st_mtime for item in candidates if item.exists()), default=0.0)


def _latest_revision(steps: list[dict[str, Any]]) -> int | None:
    for step in reversed(steps):
        value = step.get("revision_after")
        if isinstance(value, int):
            return value
    return None


__all__ = [
    "compile_snapshot",
    "compile_turn",
    "compile_model_io",
    "decode_run_id",
    "discover_run_dirs",
    "encode_run_id",
    "read_jsonl",
    "summarize_run",
]
