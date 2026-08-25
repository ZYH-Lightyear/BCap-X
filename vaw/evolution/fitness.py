"""Deterministic phase fitness and failure labels from a VAW episode trace."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from vaw.context_runtime.trace_read import load_run

PHASES = ("reach", "grasp", "transport", "place")
F4_SELECT_STREAK = 3
F3_DELTA_FRACTION = 0.40
_CONTAINER_TOKENS = ("basket", "bin", "bowl", "pot", "drawer", "shelf", "plate")


def evaluate_run(run_dir: str | pathlib.Path) -> dict[str, Any]:
    loaded = load_run(run_dir)
    records = loaded["records"]
    turns = loaded["turns"]
    meta = loaded["meta"]
    calls = [_calls(record) for record in records]
    env_success = bool(meta.get("env_success"))
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}

    reach = _reach(calls)
    grasp = _grasp(calls, records)
    transport = _transport(calls, grasp)
    place = bool(env_success and _has_function(calls, "open_gripper"))
    labels = _labels(calls, records, turns, grasp=grasp, reach=reach)

    fitness = {
        "run": pathlib.Path(run_dir).name,
        "task_id": meta.get("task_id"),
        "seed": meta.get("seed"),
        "task_prompt": meta.get("task_prompt"),
        "terminate_mode": meta.get("terminate_mode"),
        "env_success": env_success,
        "claimed_success": bool(meta.get("claimed_success")),
        "phases": {
            "reach": reach,
            "grasp": grasp,
            "transport": transport,
            "place": place,
        },
        "cost": {
            "turns": int(meta.get("turns") or len(turns)),
            "physical_ops": _physical_ops(records),
            "main_tokens": usage.get("main_total_tokens") or usage.get("total_tokens"),
            "imagination_tokens": usage.get("imagination_total_tokens") or 0,
        },
        "tools": _tool_histogram(calls),
        "labels": labels,
        "blocked_moves": sum(
            1 for turn in turns if (turn.get("motion") or {}).get("blocked")
        ),
    }
    return fitness


def write_episode_fitness(run_dir: str | pathlib.Path) -> pathlib.Path:
    directory = pathlib.Path(run_dir)
    payload = evaluate_run(directory)
    path = directory / "episode_fitness.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _calls(record: dict[str, Any]) -> dict[str, Any]:
    call = record.get("function_call") or {}
    result = record.get("function_result") or {}
    return {
        "name": call.get("name"),
        "arguments": call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
        "result": result if isinstance(result, dict) else {},
        "revision_before": record.get("revision_before"),
        "revision_after": record.get("revision_after"),
        "world_changes": (result or {}).get("world_changes")
        if isinstance(result, dict)
        else None,
    }


def _has_function(calls: list[dict[str, Any]], name: str) -> bool:
    return any(item["name"] == name for item in calls)


def _tool_histogram(calls: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in calls:
        name = item["name"]
        if isinstance(name, str) and name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _physical_ops(records: list[dict[str, Any]]) -> int:
    total = 0
    for record in records:
        before = record.get("revision_before")
        after = record.get("revision_after")
        if isinstance(before, int) and isinstance(after, int) and after > before:
            total += 1
    return total


def _reach(calls: list[dict[str, Any]]) -> bool:
    for item in calls:
        if item["name"] != "commit":
            continue
        result = item["result"]
        if result.get("error"):
            continue
        if "position_error_m" in result:
            return True
    return False


def _first_index(calls: list[dict[str, Any]], name: str) -> int | None:
    for index, item in enumerate(calls):
        if item["name"] == name:
            return index
    return None


def _grasp(calls: list[dict[str, Any]], records: list[dict[str, Any]]) -> bool:
    close_at = _first_index(calls, "close_gripper")
    if close_at is None:
        return False
    lifted = False
    source_removed = False
    for item in calls[close_at:]:
        if item["name"] == "delta_move":
            delta = item["arguments"].get("delta_xyz_m") or [0, 0, 0]
            if isinstance(delta, list) and len(delta) == 3:
                try:
                    if float(delta[2]) > 0:
                        lifted = True
                except (TypeError, ValueError):
                    pass
        removed = (item.get("world_changes") or {}).get("removed") if item.get("world_changes") else None
        if isinstance(removed, list) and removed:
            source_removed = True
    if source_removed and lifted:
        return True
    # Successful grasp-and-carry often removes the source region on the
    # approach commit *before* close.  A later +Z after close is then enough.
    earlier_removed = False
    for item in calls[: close_at + 1]:
        removed = (item.get("world_changes") or {}).get("removed") if item.get("world_changes") else None
        if isinstance(removed, list) and removed:
            earlier_removed = True
    return bool(earlier_removed and lifted)


def _transport(calls: list[dict[str, Any]], grasp: bool) -> bool:
    if not grasp:
        return False
    close_at = _first_index(calls, "close_gripper")
    assert close_at is not None
    open_at = None
    for index, item in enumerate(calls[close_at + 1 :], start=close_at + 1):
        if item["name"] == "open_gripper":
            open_at = index
            break
    window = calls[close_at + 1 : open_at]
    for item in window:
        if item["name"] in {"locate_point", "propose_pose", "commit"}:
            blob = " ".join(
                [
                    str(item["arguments"].get("query") or ""),
                    str(item["arguments"].get("point_id") or ""),
                    str((item["result"] or {}).get("action_id") or ""),
                ]
            ).lower()
            if item["name"] in {"propose_pose", "commit"} or any(
                token in blob for token in _CONTAINER_TOKENS
            ):
                return True
            if item["name"] == "locate_point":
                return True
    return False


def _labels(
    calls: list[dict[str, Any]],
    records: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    *,
    grasp: bool,
    reach: bool,
) -> list[str]:
    labels: list[str] = []
    select_streak = _select_error_streak(calls)
    if select_streak >= F4_SELECT_STREAK:
        labels.append("F4")
    if _aligned_never_releases(calls, grasp=grasp):
        labels.append("F1")
    if _phantom_empty(calls):
        labels.append("F2")
    delta_count = _tool_histogram(calls).get("delta_move", 0)
    turns_n = max(1, len(calls))
    if delta_count / turns_n >= F3_DELTA_FRACTION:
        labels.append("F3")
    if _imagination_discarded(calls):
        labels.append("F5")
    if not reach and "F4" not in labels and _has_function(calls, "select"):
        labels.append("F4")
    return labels


def _select_error_streak(calls: list[dict[str, Any]]) -> int:
    longest = current = 0
    for item in calls:
        failed_select = item["name"] == "select" and (
            item["result"].get("solve_ik") == "error"
            or item["result"].get("preview") == "unchanged"
        )
        if failed_select:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _aligned_never_releases(calls: list[dict[str, Any]], *, grasp: bool) -> bool:
    if not grasp or _has_function(calls, "open_gripper"):
        return False
    close_at = _first_index(calls, "close_gripper")
    if close_at is None:
        return False
    descend = 0
    for item in calls[close_at + 1 :]:
        if item["name"] != "delta_move":
            continue
        delta = item["arguments"].get("delta_xyz_m") or [0, 0, 0]
        if isinstance(delta, list) and len(delta) == 3:
            try:
                if float(delta[2]) < 0:
                    descend += 1
            except (TypeError, ValueError):
                pass
    return descend >= 3


def _phantom_empty(calls: list[dict[str, Any]]) -> bool:
    close_at = _first_index(calls, "close_gripper")
    if close_at is None:
        return False
    opened = False
    for item in calls[close_at + 1 :]:
        if item["name"] == "open_gripper":
            opened = True
            continue
        if item["name"] != "detection_and_sam" or opened:
            continue
        query = str(item["arguments"].get("query") or "").lower()
        if any(token in query for token in _CONTAINER_TOKENS):
            continue
        return True
    return False


def _imagination_discarded(calls: list[dict[str, Any]]) -> bool:
    for index, item in enumerate(calls):
        if item["name"] != "call_imagination":
            continue
        status = item["result"].get("status")
        if status not in {"ready", "partial"}:
            continue
        action_id = item["result"].get("action_id")
        if index + 1 >= len(calls):
            return True
        nxt = calls[index + 1]
        if nxt["name"] == "commit" and nxt["arguments"].get("action_id") == action_id:
            continue
        return True
    return False


__all__ = ["PHASES", "evaluate_run", "write_episode_fitness"]
