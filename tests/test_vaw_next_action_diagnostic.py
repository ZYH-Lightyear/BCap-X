from __future__ import annotations

from pathlib import Path

from vaw.diagnostics.run_next_action import load_cases, recover_unclosed_intent, score_call


def test_next_action_cases_are_frozen() -> None:
    cases = load_cases(Path("vaw/diagnostics/next_action_cases.json").resolve())
    assert len(cases) == 11
    assert cases[0]["_step"]["context_image"] == "context_0000.png"
    assert cases[-1]["_step"]["context_manifest"]["active_action_id"] == "a7"


def test_lift_probe_scores_valid_delta_and_rejects_oversized_delta() -> None:
    case = load_cases(
        Path("vaw/diagnostics/next_action_cases.json").resolve(),
        {"next_07_closed_needs_lift_probe"},
    )[0]
    preferred, _ = score_call(
        case,
        {
            "name": "delta_move",
            "arguments": {
                "delta_xyz_m": [0, 0, 0.03],
                "frame": "base",
                "refinement_goal": "小幅抬升并检查物体是否随动",
            },
        },
        None,
    )
    invalid, reason = score_call(
        case,
        {
            "name": "delta_move",
            "arguments": {
                "delta_xyz_m": [0, 0, 0.15],
                "frame": "base",
                "refinement_goal": "小幅抬升并检查物体是否随动",
            },
        },
        None,
    )
    assert preferred == "preferred"
    assert invalid == "invalid_call"
    assert "above 0.03" in reason


def test_bad_preview_marks_direct_commit_unsafe() -> None:
    case = load_cases(
        Path("vaw/diagnostics/next_action_cases.json").resolve(),
        {"next_05_bad_closed_preview"},
    )[0]
    grade, _ = score_call(
        case,
        {"name": "commit", "arguments": {"action_id": "a1"}},
        None,
    )
    assert grade == "unsafe"


def test_unclosed_intent_is_recoverable_for_analysis_but_not_runtime() -> None:
    recovered = recover_unclosed_intent(
        '先检查目标。<tool_call>{"name":"detection_and_sam","arguments":{"query":"alphabet soup"}}'
    )
    assert recovered == {
        "name": "detection_and_sam",
        "arguments": {"query": "alphabet soup"},
    }
    assert recover_unclosed_intent(
        '<tool_call>{"name":"detection_and_sam","arguments":{}}</tool_call>'
    ) is None
