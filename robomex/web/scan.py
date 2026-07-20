"""Discover RoboMEx run directories that contain events.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from robomex.web.models import RunListResponse, RunSummary
from robomex.web.paths import resolve_under_workspace


def list_runs(root: str = "outputs/robomex_planner_live", *, limit: int = 100) -> RunListResponse:
    scan_root = resolve_under_workspace(root)
    if not scan_root.exists():
        return RunListResponse(root=str(scan_root), runs=[])

    runs: list[RunSummary] = []
    for events_file in scan_root.rglob("events.jsonl"):
        run_dir = events_file.parent
        summary = _read_json(run_dir / "summary.json")
        subgoals = summary.get("subgoals") if isinstance(summary.get("subgoals"), list) else []
        runs.append(
            RunSummary(
                path=str(run_dir),
                name=run_dir.name,
                mtime=events_file.stat().st_mtime,
                task=str(summary.get("task") or ""),
                planner_status=str(summary.get("planner_status") or ""),
                env_success=summary.get("env_success") if "env_success" in summary else None,
                n_subgoals=int(summary.get("n_subgoals") or len(subgoals) or 0),
                authoring_strategy=str(summary.get("authoring_strategy") or ""),
            )
        )
    runs.sort(key=lambda item: item.mtime, reverse=True)
    return RunListResponse(root=str(scan_root), runs=runs[:limit])


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}
