"""Board summaries and incremental trace reads for the live viewer."""

from __future__ import annotations

import json
from pathlib import Path

from vaw.scripts.serve_trace import (
    _identity_from_name,
    compile_board,
    compile_run,
    summarize_run,
)


def _write_run(
    root: Path,
    name: str,
    *,
    meta: dict,
    steps: list[dict],
) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with (run / "steps.jsonl").open("w", encoding="utf-8") as handle:
        for record in steps:
            handle.write(json.dumps(record) + "\n")
    return run


def _tcp(x: float, y: float, z: float) -> dict:
    return {
        "world": {
            "robot": {
                "tcp_pose": {"position_xyz": [x, y, z]},
                "gripper_opening": 0.99,
            }
        }
    }


def test_identity_from_historical_directory_name() -> None:
    parsed = _identity_from_name("m16w_v46_oblique_contact_opus5_task4_t0_s1")
    assert parsed == {"taskId": 4, "seed": 1, "suite": None}


def test_summarize_run_uses_meta_and_falls_back_to_the_directory_name(
    tmp_path: Path,
) -> None:
    run = _write_run(
        tmp_path,
        "legacy_task7_t0_s3",
        meta={
            "task_prompt": "pick the milk",
            "terminate_mode": "max_turns",
            "env_success": False,
            "claimed_success": True,
            "usage": {"total_tokens": 12000},
        },
        steps=[
            {
                "index": 0,
                "turn": 1,
                "function_call": {"name": "detection_and_sam", "arguments": {}},
                "function_result": {},
                "decision_basis": "find milk",
                "context_packet": _tcp(0.4, 0.0, 0.2),
            },
            {
                "index": 1,
                "turn": 2,
                "function_call": {
                    "name": "delta_move",
                    "arguments": {"delta_xyz_m": [0.0, 0.0, -0.03]},
                },
                "function_result": {"advisory": "descending"},
                "decision_basis": "drop a little",
                "context_packet": _tcp(0.4, 0.0, 0.2),
            },
            {
                "index": 2,
                "turn": 3,
                "function_call": {"name": "call_imagination", "arguments": {}},
                "function_result": {},
                "decision_basis": "refine",
                "context_packet": _tcp(0.4, 0.0, 0.199),
            },
        ],
    )

    row = summarize_run(run)

    assert row["name"] == "legacy_task7_t0_s3"
    assert row["live"] is False
    assert row["taskId"] == 7
    assert row["seed"] == 3
    assert row["taskPrompt"] == "pick the milk"
    assert row["envSuccess"] is False
    assert row["claimedSuccess"] is True
    assert row["tools"]["delta_move"] == 1
    assert row["imaginationCalls"] == 1
    assert row["blockedMoves"] == 1
    assert row["lastFunction"] == "call_imagination"
    assert row["lastBasis"] == "refine"
    assert row["tokens"] == 12000


def test_compile_board_lists_newest_runs_first(tmp_path: Path) -> None:
    import os

    older = _write_run(
        tmp_path,
        "older_task1_t0_s1",
        meta={"terminate_mode": "goal", "env_success": True, "task_id": 1, "seed": 1},
        steps=[{"index": 0, "turn": 1, "function_call": {"name": "done"}}],
    )
    newer = _write_run(
        tmp_path,
        "newer_task2_t0_s1",
        meta={"suite": "libero_object_swap", "task_id": 2, "seed": 1, "model": "opus"},
        steps=[{"index": 0, "turn": 1, "function_call": {"name": "detection_and_sam"}}],
    )
    os.utime(older / "steps.jsonl", (1_700_000_000, 1_700_000_000))
    os.utime(newer / "steps.jsonl", (1_800_000_000, 1_800_000_000))

    board = compile_board(tmp_path, single_run=False)

    assert [row["name"] for row in board["runs"]] == [
        "newer_task2_t0_s1",
        "older_task1_t0_s1",
    ]
    assert board["runs"][0]["suite"] == "libero_object_swap"
    assert board["runs"][0]["model"] == "opus"
    assert board["runs"][0]["live"] is True
    assert board["runs"][1]["live"] is False


def test_compile_run_after_resends_the_boundary_turn(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "one",
        meta={"task_prompt": "place ketchup"},
        steps=[
            {
                "index": 0,
                "turn": 1,
                "function_call": {
                    "name": "delta_move",
                    "arguments": {"delta_xyz_m": [0.0, 0.0, -0.03]},
                },
                "context_packet": _tcp(0.5, 0.0, 0.20),
            },
            {
                "index": 1,
                "turn": 2,
                "function_call": {"name": "close_gripper", "arguments": {}},
                "context_packet": _tcp(0.5, 0.0, 0.17),
            },
            {
                "index": 2,
                "turn": 3,
                "function_call": {"name": "delta_move", "arguments": {"delta_xyz_m": [0.0, 0.0, 0.03]}},
                "context_packet": _tcp(0.5, 0.0, 0.17),
            },
        ],
    )

    full = compile_run(run)
    assert [turn["index"] for turn in full["turns"]] == [0, 1, 2]
    assert full["cursor"] == 2
    assert full["turns"][0]["motion"]["blocked"] is False

    tail = compile_run(run, after=1)
    assert [turn["index"] for turn in tail["turns"]] == [1, 2]
    assert tail["after"] == 1
    assert tail["summary"]["turnCount"] == 3


def test_imagination_preview_move_is_never_blocked(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "preview_only",
        meta={"task_prompt": "place ketchup"},
        steps=[
            {
                "index": 0,
                "turn": 1,
                "function_call": {"name": "call_imagination", "arguments": {}},
                "context_packet": _tcp(0.5, 0.0, 0.28),
            },
            {
                "index": 1,
                "turn": 2,
                "function_call": {"name": "close_gripper", "arguments": {}},
                "context_packet": _tcp(0.5, 0.0, 0.28),
            },
        ],
    )
    sub = run / "subagents" / "imagination_0001"
    sub.mkdir(parents=True)
    with (sub / "steps.jsonl").open("w", encoding="utf-8") as handle:
        for record in (
            {
                "index": 0,
                "turn": 1,
                "function_call": {
                    "name": "delta_move",
                    "arguments": {"delta_xyz_m": [0.0, 0.0, -0.03]},
                },
                "context_packet": _tcp(0.5, 0.0, 0.28),
            },
            {
                "index": 1,
                "turn": 2,
                "function_call": {
                    "name": "delta_move",
                    "arguments": {"delta_xyz_m": [0.0, 0.0, -0.03]},
                },
                "context_packet": _tcp(0.5, 0.0, 0.28),
            },
        ):
            handle.write(json.dumps(record) + "\n")

    compiled = compile_run(run)
    imagination = compiled["subagents"][0]["turns"][0]["motion"]

    assert imagination["blocked"] is False
    assert imagination["previewOnly"] is True
    assert compiled["summary"]["blockedMoves"] == 0
    assert summarize_run(run)["blockedMoves"] == 0
