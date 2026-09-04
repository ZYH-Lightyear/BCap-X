"""Read VAW episode traces for the viewer and for fitness extraction."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

# A move that achieves less than this fraction of its commanded length did not
# fail loudly; it silently did nothing, which reads as success to the policy.
_BLOCKED_ACHIEVED_FRACTION = 0.25
_MIN_COMMANDED_M = 0.002
# A run without terminate_mode whose steps file is still being appended is live.
_LIVE_WINDOW_S = 30.0
_TASK_IN_NAME = re.compile(r"task(\d+)", re.IGNORECASE)
_SEED_IN_NAME = re.compile(r"_s(\d+)$")
_SUITE_IN_NAME = re.compile(r"(libero_[a-z0-9_]+)")

# Historical traces remain valuable RSI evidence after the public Function API
# is renamed. These aliases are deliberately confined to the read path: the
# live Runtime and the model-facing schemas accept only the current names.
_MAIN_TRACE_ALIASES = {
    "detection_and_sam": "detect_region",
    "propose_pose": "preview_pose",
    "select": "preview_grasp",
    "call_imagination": "imagine_action",
    "delta_move": "move_tcp_delta",
    "reject_action": "discard_action",
    "commit": "execute_action",
    "done": "finish_task",
}
_IMAGINATION_TRACE_ALIASES = {
    "delta_move": "shift_preview",
    "rotate": "rotate_preview",
    "done": "finish_imagination",
}


def canonical_trace_function(name: Any, *, owner: str = "main") -> Any:
    """Return the current display/fitness name for a historical Function.

    This is trace compatibility, not a live API alias. Unknown values are
    preserved so diagnostic tools never erase malformed-run evidence.
    """

    if not isinstance(name, str):
        return name
    aliases = (
        _IMAGINATION_TRACE_ALIASES if owner == "imagination" else _MAIN_TRACE_ALIASES
    )
    return aliases.get(name, name)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file that another process may be appending to."""

    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # The writer is mid-line; the next poll picks the record up.
                break
    return records


def _tcp_xyz(record: dict[str, Any]) -> list[float] | None:
    robot = ((record.get("context_packet") or {}).get("world") or {}).get("robot")
    if not isinstance(robot, dict):
        return None
    pose = robot.get("tcp_pose") or robot.get("ee_pose")
    if not isinstance(pose, dict):
        return None
    position = pose.get("position_xyz")
    if not isinstance(position, list) or len(position) != 3:
        return None
    try:
        return [float(value) for value in position]
    except (TypeError, ValueError):
        return None


def _gripper(record: dict[str, Any]) -> float | None:
    robot = ((record.get("context_packet") or {}).get("world") or {}).get("robot")
    if not isinstance(robot, dict):
        return None
    try:
        return float(robot["gripper_opening"])
    except (KeyError, TypeError, ValueError):
        return None


def _norm(vector: list[float]) -> float:
    return sum(value * value for value in vector) ** 0.5


def _commanded_move_m(call: dict[str, Any], *, owner: str) -> float | None:
    """Length of a translation command, or None for non-motion functions."""

    name = canonical_trace_function(call.get("name"), owner=owner)
    if name not in {"move_tcp_delta", "shift_preview"}:
        return None
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return None
    delta = arguments.get("delta_xyz_m")
    if not isinstance(delta, list) or len(delta) != 3:
        return None
    try:
        return _norm([float(value) for value in delta])
    except (TypeError, ValueError):
        return None


def _motion(
    record: dict[str, Any],
    following: dict[str, Any] | None,
    *,
    physical: bool,
    owner: str,
) -> dict[str, Any] | None:
    """Compare a commanded translation against the displacement it produced.

    ``context_packet`` holds the state the agent saw *before* choosing, so the
    achieved displacement is only observable in the following turn.
    Imagination edits the Preview only; the real TCP is supposed to stay put,
    so a zero displacement there is not a blocked motion.
    """

    call = record.get("function_call") or {}
    commanded = _commanded_move_m(call, owner=owner)
    if commanded is None:
        return None
    if not physical:
        return {
            "commandedM": commanded,
            "achievedM": None,
            "blocked": False,
            "previewOnly": True,
        }
    before = _tcp_xyz(record)
    after = _tcp_xyz(following) if following is not None else None
    if before is None or after is None:
        return {"commandedM": commanded, "achievedM": None, "blocked": False}
    achieved = _norm([after[index] - before[index] for index in range(3)])
    blocked = (
        commanded >= _MIN_COMMANDED_M
        and achieved < _BLOCKED_ACHIEVED_FRACTION * commanded
    )
    return {
        "commandedM": commanded,
        "achievedM": achieved,
        "blocked": blocked,
        "deltaXyz": [after[index] - before[index] for index in range(3)],
    }


def _result_fields(record: dict[str, Any]) -> tuple[Any, str | None, str | None]:
    """Split a function result into payload, advisory and error text."""

    result = record.get("function_result")
    error = record.get("error")
    advisory = None
    payload: Any = result
    if isinstance(result, dict):
        advisory = result.get("advisory")
        error = error or result.get("error")
        payload = {
            key: value
            for key, value in result.items()
            if key not in {"advisory", "error"}
        }
        if not payload:
            payload = None
    return payload, advisory, (str(error) if error else None)


def _compile_turn(
    record: dict[str, Any],
    following: dict[str, Any] | None,
    *,
    owner: str,
    image_prefix: str,
) -> dict[str, Any]:
    call = record.get("function_call") or {}
    turn_owner = record.get("agent_owner") or owner
    payload, advisory, error = _result_fields(record)
    image = record.get("context_image")
    return {
        "index": record.get("index"),
        "turn": record.get("turn"),
        "owner": turn_owner,
        "function": canonical_trace_function(call.get("name"), owner=turn_owner),
        "arguments": call.get("arguments"),
        "result": payload,
        "advisory": advisory,
        "error": error,
        "basis": record.get("decision_basis"),
        "thought": record.get("thought"),
        "image": f"{image_prefix}{image}" if image else None,
        "tcp": _tcp_xyz(record),
        "grip": _gripper(record),
        "motion": _motion(
            record,
            following,
            physical=turn_owner != "imagination",
            owner=turn_owner,
        ),
        "envSuccess": record.get("env_success"),
        "done": record.get("done"),
    }


def _compile_series(
    records: list[dict[str, Any]],
    *,
    owner: str,
    image_prefix: str,
) -> list[dict[str, Any]]:
    return [
        _compile_turn(
            record,
            records[index + 1] if index + 1 < len(records) else None,
            owner=owner,
            image_prefix=image_prefix,
        )
        for index, record in enumerate(records)
    ]


def _subagent_dirs(run_dir: Path) -> list[Path]:
    root = run_dir / "subagents"
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if (path / "steps.jsonl").is_file())


def _load_meta(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "meta.json"
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _identity_from_name(name: str) -> dict[str, Any]:
    """Recover suite / task / seed from historical directory names."""

    task = _TASK_IN_NAME.search(name)
    seed = _SEED_IN_NAME.search(name)
    suite = _SUITE_IN_NAME.search(name)
    return {
        "taskId": int(task.group(1)) if task else None,
        "seed": int(seed.group(1)) if seed else None,
        "suite": suite.group(1) if suite else None,
    }


def _is_live(meta: dict[str, Any], steps_path: Path) -> bool:
    if meta.get("terminate_mode"):
        return False
    if not steps_path.is_file():
        return False
    return (time.time() - steps_path.stat().st_mtime) <= _LIVE_WINDOW_S


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """Board row: histogram and last call, no canvas rasters."""

    meta = _load_meta(run_dir)
    identity = _identity_from_name(run_dir.name)
    records = _read_jsonl(run_dir / "steps.jsonl")
    turns = _compile_series(records, owner="main", image_prefix="")
    tools: dict[str, int] = {}
    for turn in turns:
        name = turn.get("function")
        if isinstance(name, str) and name:
            tools[name] = tools.get(name, 0) + 1
    last = turns[-1] if turns else {}
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    task_id = meta.get("task_id")
    seed = meta.get("seed")
    return {
        "name": run_dir.name,
        "live": _is_live(meta, run_dir / "steps.jsonl"),
        "taskPrompt": meta.get("task_prompt"),
        "suite": meta.get("suite") or identity["suite"],
        "taskId": task_id if task_id is not None else identity["taskId"],
        "seed": seed if seed is not None else identity["seed"],
        "model": meta.get("model"),
        "envSuccess": meta.get("env_success"),
        "claimedSuccess": meta.get("claimed_success"),
        "terminateMode": meta.get("terminate_mode"),
        "turns": int(meta["turns"]) if meta.get("turns") is not None else len(turns),
        "tools": tools,
        "imaginationCalls": int(
            tools.get("imagine_action", 0) or len(_subagent_dirs(run_dir))
        ),
        "blockedMoves": sum(
            1 for turn in turns if (turn.get("motion") or {}).get("blocked")
        ),
        "lastFunction": last.get("function"),
        "lastBasis": last.get("basis"),
        "tokens": usage.get("total_tokens"),
    }


def compile_board(root: Path, *, single_run: bool) -> dict[str, Any]:
    directories = (
        [root]
        if single_run
        else [root / name for name in discover_runs(root)]
    )
    return {
        "singleRun": single_run,
        "runs": [
            summarize_run(directory)
            for directory in directories
            if (directory / "steps.jsonl").is_file()
        ],
    }


def compile_run(run_dir: Path, *, after: int | None = None) -> dict[str, Any]:
    """Build the viewer payload for one run directory.

    ``after`` is the last main-turn index the client already has.  The
    boundary turn is resent so its ``motion.achievedM`` can be filled in
    once the following observation exists.
    """

    meta = _load_meta(run_dir)
    all_turns = _compile_series(
        _read_jsonl(run_dir / "steps.jsonl"),
        owner="main",
        image_prefix="",
    )
    subagents = []
    for directory in _subagent_dirs(run_dir):
        relative = f"subagents/{directory.name}/"
        subagents.append(
            {
                "name": directory.name,
                "turns": _compile_series(
                    _read_jsonl(directory / "steps.jsonl"),
                    owner="imagination",
                    image_prefix=relative,
                ),
            }
        )
    blocked = sum(
        1 for turn in all_turns if (turn.get("motion") or {}).get("blocked")
    )
    turns = all_turns
    if after is not None:
        turns = [
            turn
            for turn in all_turns
            if int(turn.get("index") or 0) >= int(after)
        ]
    last_index = max((int(turn.get("index") or 0) for turn in all_turns), default=-1)
    return {
        "run": run_dir.name,
        "meta": meta,
        "turns": turns,
        "subagents": subagents,
        "after": after,
        "cursor": last_index,
        "summary": {
            "turnCount": len(all_turns),
            "blockedMoves": blocked,
            "errors": sum(1 for turn in all_turns if turn.get("error")),
            "envSuccess": meta.get("env_success", next(
                (
                    turn["envSuccess"]
                    for turn in reversed(all_turns)
                    if turn.get("envSuccess") is not None
                ),
                None,
            )),
            "claimedSuccess": meta.get("claimed_success"),
            "terminateMode": meta.get("terminate_mode"),
            "live": _is_live(meta, run_dir / "steps.jsonl"),
            "taskPrompt": meta.get("task_prompt"),
        },
    }


def discover_runs(root: Path) -> list[str]:
    """Names of run directories under ``root``, newest first."""

    if (root / "steps.jsonl").is_file():
        return []
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "steps.jsonl").is_file()
    ]
    candidates.sort(
        key=lambda path: (path / "steps.jsonl").stat().st_mtime,
        reverse=True,
    )
    return [path.name for path in candidates]



def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load meta, raw main steps, compiled turns, and subagent traces."""

    directory = Path(run_dir)
    meta = _load_meta(directory)
    records = _read_jsonl(directory / "steps.jsonl")
    turns = _compile_series(records, owner="main", image_prefix="")
    subagents = []
    for path in _subagent_dirs(directory):
        sub_records = _read_jsonl(path / "steps.jsonl")
        subagents.append(
            {
                "name": path.name,
                "records": sub_records,
                "turns": _compile_series(
                    sub_records, owner="imagination", image_prefix=""
                ),
                "meta": _load_meta(path),
            }
        )
    return {
        "dir": directory,
        "meta": meta,
        "records": records,
        "turns": turns,
        "subagents": subagents,
    }


__all__ = [
    "canonical_trace_function",
    "compile_board",
    "compile_run",
    "discover_runs",
    "load_run",
    "summarize_run",
]
