from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from vaw.agents.contracts import ModelResponse, ToolCall
from vaw.evolution.atlas import compile_atlas, compile_to_dir, render_atlas
from vaw.evolution.investigator import INSPECT_SEGMENT_TOOL, TraceInvestigator
from vaw.evolution.m3 import run_m3
from vaw.evolution.reviewer import EvidenceReviewer
from vaw.evolution.segments import TraceAccessor


class FakeProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.requests.append((messages, tools))
        return self.responses.pop(0)


def _png(path: Path, color: str, label: str) -> None:
    image = Image.new("RGB", (640, 400), color)
    ImageDraw.Draw(image).text((20, 20), label, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _trace(tmp_path: Path) -> tuple[Path, Path]:
    trace = tmp_path / "trace"
    trace.mkdir()
    _png(trace / "context_0000.png", "#334155", "initial")
    _png(trace / "context_0001.png", "#2563eb", "after A1")
    _png(trace / "context_0002.png", "#15803d", "after A2")
    (trace / "video_agentview.mp4").write_bytes(b"video")
    (trace / "meta.json").write_text(
        json.dumps(
            {
                "suite": "libero_spatial_task",
                "task_id": 0,
                "seed": 1,
                "task_prompt": "put the bowl on the plate",
                "env_success": False,
                "terminate_mode": "max_turns",
                "planner_error": "private",
            }
        ),
        encoding="utf-8",
    )
    steps = [
        {
            "turn": 1,
            "context_image": "context_0000.png",
            "function_call": {"name": "select", "arguments": {"seed_id": "s1"}},
            "function_result": {"action_id": "a1", "solve_ik": "returned"},
            "decision_basis": "private rationale",
        },
        {
            "turn": 2,
            "context_image": "context_0000.png",
            "function_call": {"name": "commit", "arguments": {"action_id": "a1"}},
            "function_result": {"position_error_m": 0.3},
            "runtime_diagnostics": {"solver": "IK_FAIL"},
        },
        {
            "turn": 3,
            "context_image": "context_0001.png",
            "function_call": {
                "name": "detection_and_sam",
                "arguments": {"query": "bowl"},
            },
            "function_result": {"region_id": "region2"},
        },
        {
            "turn": 4,
            "context_image": "context_0001.png",
            "function_call": {
                "name": "delta_move",
                "arguments": {"delta_xyz_m": [0, 0, 0.03], "frame": "base"},
            },
            "function_result": {"status": "completed"},
        },
        {"turn": 5, "context_image": "context_0002.png"},
    ]
    (trace / "steps.jsonl").write_text(
        "".join(json.dumps(step) + "\n" for step in steps),
        encoding="utf-8",
    )
    for index, (turn, function, arguments, start, end) in enumerate(
        (
            (2, "commit", {"action_id": "a1"}, 0, 10),
            (4, "delta_move", {"delta_xyz_m": [0, 0, 0.03], "frame": "base"}, 10, 20),
        ),
        start=1,
    ):
        action_dir = trace / "actions" / f"action_{index:04d}"
        action_dir.mkdir(parents=True)
        (action_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "segment_id": f"action_{index:04d}",
                    "turn": turn,
                    "function": function,
                    "arguments": arguments,
                    "outcome": "completed (solver=private)",
                    "frame_start": start,
                    "frame_end": end,
                    "fps": 30,
                    "streams": {},
                }
            ),
            encoding="utf-8",
        )
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "schema": "vaw-topreward-progress-v1",
                "trace": str(trace.resolve()),
                "model": "Qwen/Qwen3-VL-8B-Instruct",
                "states": [
                    {
                        "segment_id": "initial",
                        "turn": 0,
                        "function": "initial_state",
                        "raw_reward": -20.0,
                        "raw_delta": 0.0,
                    },
                    {
                        "segment_id": "action_0001",
                        "turn": 2,
                        "function": "commit",
                        "raw_reward": -17.0,
                        "raw_delta": 3.0,
                    },
                    {
                        "segment_id": "action_0002",
                        "turn": 4,
                        "function": "delta_move",
                        "raw_reward": -18.0,
                        "raw_delta": -1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return trace, progress


def _inspect_response(start: int = 1, end: int = 2) -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                id="call_1",
                name="inspect_segment",
                args={
                    "start_action": f"A{start}",
                    "end_action": f"A{end}",
                    "question": "动作后是否出现可见进展，随后是否回退？",
                },
            ),
        )
    )


def _finding_response(span: tuple[int, int] = (1, 2)) -> ModelResponse:
    return ModelResponse(
        text=json.dumps(
            {
                "findings": [
                    {
                        "span": list(span),
                        "observation": "A1 后状态改善，但 A2 后目标关系减弱。",
                        "insight": "局部动作后应依据新视觉结果判断是否继续同方向调整。",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def _labeled_finding_response() -> ModelResponse:
    return ModelResponse(
        text=json.dumps(
            {
                "findings": [
                    {
                        "span": ["A1", "A2"],
                        "observation": "A1 后状态改善，但 A2 后目标关系减弱。",
                        "insight": "局部动作后应依据新视觉结果判断是否继续同方向调整。",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )


def test_atlas_is_compact_aligned_and_deterministic(tmp_path: Path) -> None:
    trace, progress = _trace(tmp_path)
    payload = compile_atlas(trace, progress)

    assert payload["episode"]["env_success"] is False
    assert [action["function"] for action in payload["actions"]] == [
        "commit",
        "delta_move",
    ]
    assert payload["actions"][0]["arguments"] == {}
    assert payload["actions"][1]["progress"] == {"reward": -18.0, "delta": -1.0}
    serialized = json.dumps(payload)
    for forbidden in (
        "relative_progress",
        "position_error_m",
        "runtime_diagnostics",
        "planner_error",
        "IK_FAIL",
        "private rationale",
        "solver=private",
    ):
        assert forbidden not in serialized

    first = render_atlas(payload, tmp_path / "first.png")
    second = render_atlas(payload, tmp_path / "second.png")
    assert first.read_bytes() == second.read_bytes()


def test_inspect_segment_returns_policy_visible_sequence(tmp_path: Path) -> None:
    trace, progress = _trace(tmp_path)
    atlas = compile_atlas(trace, progress)
    evidence = TraceAccessor(atlas).inspect(1, 2, "发生了什么？", tmp_path / "segment.png")

    assert evidence.key == (1, 2)
    assert evidence.image.is_file()
    assert any("detection_and_sam" in event for event in evidence.actions)
    assert all("IK_FAIL" not in event for event in evidence.actions)
    assert Image.open(evidence.image).width == 2400


def test_investigator_uses_one_real_tool_and_bounded_visual_memory(tmp_path: Path) -> None:
    trace, progress = _trace(tmp_path)
    atlas_json, atlas_png = compile_to_dir(trace, progress, tmp_path / "atlas")
    provider = FakeProvider([_inspect_response(), _finding_response()])

    result = TraceInvestigator(provider, max_inspections=2).run(
        atlas_json,
        atlas_png,
        tmp_path / "investigation",
    )

    assert provider.requests[0][1] == [INSPECT_SEGMENT_TOOL]
    assert len(provider.requests) == 2
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["findings"][0]["span"] == [1, 2]
    assert payload["findings"][0]["evidence"] == "segments/segment_01.png"
    request_text = json.dumps(provider.requests[1][0], ensure_ascii=False)
    for forbidden in ("IK_FAIL", "position_error_m", "private rationale", str(trace)):
        assert forbidden not in request_text


def test_investigator_accepts_action_labels_in_final_finding(tmp_path: Path) -> None:
    trace, progress = _trace(tmp_path)
    atlas_json, atlas_png = compile_to_dir(trace, progress, tmp_path / "atlas")
    provider = FakeProvider([_inspect_response(), _labeled_finding_response()])

    result = TraceInvestigator(provider).run(
        atlas_json,
        atlas_png,
        tmp_path / "investigation",
    )

    assert json.loads(result.read_text(encoding="utf-8"))["findings"][0]["span"] == [1, 2]


def test_investigator_reuses_immutable_segment_without_duplicate_raster(
    tmp_path: Path,
) -> None:
    trace, progress = _trace(tmp_path)
    atlas_json, atlas_png = compile_to_dir(trace, progress, tmp_path / "atlas")
    provider = FakeProvider(
        [
            _inspect_response(1, 1),
            _inspect_response(1, 1),
            _finding_response((1, 1)),
        ]
    )

    TraceInvestigator(provider, max_inspections=3).run(
        atlas_json,
        atlas_png,
        tmp_path / "investigation",
    )

    segments = list((tmp_path / "investigation" / "segments").glob("*.png"))
    assert len(segments) == 1
    final_request = json.dumps(provider.requests[2][0], ensure_ascii=False)
    assert "未重复读取" in final_request


def test_reviewer_checks_each_finding_in_fresh_context(tmp_path: Path) -> None:
    trace, progress = _trace(tmp_path)
    atlas_json, atlas_png = compile_to_dir(trace, progress, tmp_path / "atlas")
    investigator = FakeProvider([_inspect_response(), _finding_response()])
    findings = TraceInvestigator(investigator).run(
        atlas_json,
        atlas_png,
        tmp_path / "investigation",
    )
    reviewer = FakeProvider(
        [ModelResponse(text='{"accepted":true,"reason":"前后视觉直接支持该结论。"}')]
    )

    result = EvidenceReviewer(reviewer).run(findings, tmp_path / "review.json")

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["findings"][0]["review"]["accepted"] is True
    assert reviewer.requests[0][1] is None
    assert len(reviewer.requests[0][0]) == 2


def test_complete_m3_writes_reviewed_artifacts(tmp_path: Path) -> None:
    trace, progress = _trace(tmp_path)
    investigator = FakeProvider([_inspect_response(), _finding_response()])
    reviewer = FakeProvider(
        [ModelResponse(text='{"accepted":false,"reason":"证据不足以支持通用结论。"}')]
    )

    result = run_m3(
        trace_dir=trace,
        progress_path=progress,
        output_dir=tmp_path / "m3",
        investigator_provider=investigator,
        reviewer_provider=reviewer,
        run_metadata={"schema": "vaw-m3-run-v1", "investigator_model": "fake"},
    )

    summary = json.loads(result.read_text(encoding="utf-8"))
    assert summary["findings"] == 1
    assert summary["accepted"] == 0
    assert summary["run"] == "run.json"
    assert json.loads((tmp_path / "m3" / "run.json").read_text())["investigator_model"] == "fake"
    assert (tmp_path / "m3" / "atlas" / "atlas.png").is_file()
    assert (tmp_path / "m3" / "investigation" / "segments" / "segment_01.png").is_file()
    assert not list(tmp_path.glob(".m3.staging-*"))
