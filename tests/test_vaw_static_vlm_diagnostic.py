from __future__ import annotations

from pathlib import Path

from vaw.diagnostics.run_static_vlm import extract_answer, load_cases


def test_static_vlm_cases_are_frozen_and_valid() -> None:
    cases = load_cases(Path("vaw/diagnostics/static_vlm_cases.json").resolve())
    assert len(cases) == 10
    assert {case["category"] for case in cases} == {
        "close_readiness",
        "epistemic_calibration",
        "grasp_state",
        "pose_refinement",
        "preview_semantics",
    }


def test_extract_answer_accepts_plain_and_fenced_json() -> None:
    plain, plain_error = extract_answer(
        '{"choice":"b","visual_evidence":"罐体仍在桌面","confidence":"high"}'
    )
    fenced, fenced_error = extract_answer("```json\n{\"choice\":\"C\"}\n```")

    assert plain_error is None
    assert plain == {
        "choice": "B",
        "visual_evidence": "罐体仍在桌面",
        "confidence": "high",
    }
    assert fenced_error is None
    assert fenced == {"choice": "C"}


def test_extract_answer_reports_non_json_response() -> None:
    parsed, error = extract_answer("我选择 B，因为罐体没有随动。")
    assert parsed is None
    assert error == "response does not contain a JSON object"
