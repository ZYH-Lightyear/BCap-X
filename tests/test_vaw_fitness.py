"""Phase fitness on frozen m16y / m16z / m17a traces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaw.evolution.fitness import evaluate_run

_EVAL_ROOT = Path(__file__).resolve().parents[1] / "vaw" / "out" / "libero_pro_object_eval"
_PREFIXES = (
    "m16y_principles_opus5_",
    "m16z_jaw_flip_opus5_",
    "m17a_seed_plan_cache_opus5_",
)


def _completed_taxonomy_runs() -> list[Path]:
    if not _EVAL_ROOT.is_dir():
        return []
    runs: list[Path] = []
    for path in _EVAL_ROOT.iterdir():
        if not path.is_dir() or not path.name.startswith(_PREFIXES):
            continue
        if not (path / "steps.jsonl").is_file() or not (path / "meta.json").is_file():
            continue
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta.get("terminate_mode") is None:
            continue
        runs.append(path)
    return sorted(runs)


def test_place_matches_env_success_on_taxonomy_sweeps() -> None:
    runs = _completed_taxonomy_runs()
    if len(runs) < 20:
        pytest.skip(f"need frozen taxonomy traces, found {len(runs)}")
    mismatches = []
    for run in runs:
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        fit = evaluate_run(run)
        expected = bool(meta.get("env_success"))
        if fit["phases"]["place"] != expected:
            mismatches.append((run.name, expected, fit["phases"]))
    assert mismatches == []


def test_f4_flags_m16z_task8_select_loop() -> None:
    run = _EVAL_ROOT / "m16z_jaw_flip_opus5_task8_t0_s1"
    if not (run / "steps.jsonl").is_file():
        pytest.skip("m16z task8 trace missing")
    fit = evaluate_run(run)
    assert fit["phases"]["reach"] is False
    assert fit["phases"]["place"] is False
    assert "F4" in fit["labels"]
    assert fit["tools"].get("select", 0) >= 20


def test_f1_flags_m16y_task6_aligned_never_releases() -> None:
    run = _EVAL_ROOT / "m16y_principles_opus5_task6_t0_s1"
    if not (run / "steps.jsonl").is_file():
        pytest.skip("m16y task6 trace missing")
    fit = evaluate_run(run)
    assert fit["phases"]["grasp"] is True
    assert fit["phases"]["place"] is False
    assert "F1" in fit["labels"]
    assert "open_gripper" not in fit["tools"]
