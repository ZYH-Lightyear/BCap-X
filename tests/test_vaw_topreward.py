from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from vaw.diagnostics.run_topreward import score_trace
from vaw.diagnostics.topreward import (
    TopRewardScore,
    calibrate_progress,
    minmax_normalize,
    uniform_prefix_indices,
)


class FakeScorer:
    def __init__(self, rewards: list[float]) -> None:
        self._rewards = iter(rewards)
        self.received_frame_counts: list[int] = []

    @property
    def model_name(self) -> str:
        return "fake/topreward"

    def score(
        self, *, frames: list[np.ndarray], instruction: str
    ) -> TopRewardScore:
        assert instruction == "move the bowl onto the plate"
        self.received_frame_counts.append(len(frames))
        return TopRewardScore(
            raw_log_prob=next(self._rewards),
            token_count=1,
            scored_token_ids=(42,),
            scored_tokens=("True",),
        )


def _image(path: Path, value: int) -> None:
    Image.new("RGB", (32, 24), color=(value, value, value)).save(path)


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24)
    )
    assert writer.isOpened()
    try:
        for value in range(6):
            writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    finally:
        writer.release()


def _trace(tmp_path: Path) -> Path:
    trace = tmp_path / "trace"
    trace.mkdir()
    _video(trace / "video_agentview.mp4")
    for index, value in enumerate((10, 20, 30)):
        _image(trace / f"context_{index:04d}.png", value)
    (trace / "meta.json").write_text(
        json.dumps({"task_prompt": "move the bowl onto the plate"}),
        encoding="utf-8",
    )
    records = [
        {
            "turn": 1,
            "context_image": "context_0000.png",
            "function_call": {"name": "commit", "arguments": {"action_id": "a1"}},
            "function_result": {},
        },
        {
            "turn": 2,
            "context_image": "context_0001.png",
            "function_call": {"name": "open_gripper", "arguments": {}},
            "function_result": {},
        },
        {
            "turn": 3,
            "context_image": "context_0002.png",
            "function_call": {"name": "done", "arguments": {"success": False}},
            "function_result": {},
        },
    ]
    (trace / "steps.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    for index, (turn, function, start, end) in enumerate(
        ((1, "commit", 1, 3), (2, "open_gripper", 3, 6)), start=1
    ):
        action = trace / "actions" / f"action_{index:04d}"
        action.mkdir(parents=True)
        (action / "manifest.json").write_text(
            json.dumps(
                {
                    "segment_id": f"action_{index:04d}",
                    "turn": turn,
                    "function": function,
                    "arguments": {},
                    "outcome": "completed",
                    "revision_before": index,
                    "revision_after": index + 1,
                    "frame_start": start,
                    "frame_end": end,
                    "fps": 30,
                    "streams": {},
                }
            ),
            encoding="utf-8",
        )
    return trace


def test_uniform_prefix_indices_preserve_boundaries() -> None:
    assert uniform_prefix_indices(0, 16) == (0,)
    assert uniform_prefix_indices(4, 3) == (0, 2, 4)
    assert uniform_prefix_indices(2, 16) == (0, 1, 2)


def test_minmax_normalization_matches_topreward_behavior() -> None:
    assert minmax_normalize([-4.0, -2.0, -3.0]) == pytest.approx([0.0, 1.0, 0.5])
    assert minmax_normalize([-2.0, -2.0]) == [1.0, 1.0]


def test_failed_episode_uses_smaller_penalty_at_higher_progress() -> None:
    assert calibrate_progress(
        [0.0, 0.5, 0.8, 1.0], env_success=False
    ) == pytest.approx([0.0, 0.45, 0.744, 0.95])
    assert calibrate_progress([0.0, 0.5, 1.0], env_success=True) == [
        0.0,
        0.5,
        1.0,
    ]
    assert calibrate_progress([0.0, 0.5, 1.0], env_success=None) == [
        0.0,
        0.5,
        1.0,
    ]


def test_trace_scoring_aligns_only_physical_action_boundaries(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    scorer = FakeScorer([-4.0, -2.0, -3.0])
    progress_path = score_trace(
        trace_dir=trace,
        output_dir=tmp_path / "result",
        scorer=scorer,
        max_prefix_frames=4,
    )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["schema"] == "vaw-topreward-progress-v1"
    assert progress["preview_frames_scored"] is False
    assert progress["physical_boundaries_only"] is True
    assert scorer.received_frame_counts == [1, 3, 4]
    assert [state["function"] for state in progress["states"]] == [
        "initial_state",
        "commit",
        "open_gripper",
    ]
    assert [state["progress"] for state in progress["states"]] == pytest.approx(
        [0.0, 1.0, 0.5]
    )
    assert progress["states"][2]["raw_delta"] == pytest.approx(-1.0)


def test_failed_trace_calibrates_display_progress_but_preserves_raw_scores(
    tmp_path: Path,
) -> None:
    trace = _trace(tmp_path)
    meta_path = trace / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["env_success"] = False
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    progress_path = score_trace(
        trace_dir=trace,
        output_dir=tmp_path / "failed-result",
        scorer=FakeScorer([-4.0, -2.0, -3.0]),
        max_prefix_frames=4,
    )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["env_success"] is False
    assert progress["failure_penalty"] == {
        "kind": "linear_multiplicative",
        "low_progress_rate": 0.15,
        "high_progress_rate": 0.05,
    }
    assert [state["raw_reward"] for state in progress["states"]] == [
        -4.0,
        -2.0,
        -3.0,
    ]
    assert [state["relative_progress"] for state in progress["states"]] == pytest.approx(
        [0.0, 1.0, 0.5]
    )
    assert [state["progress"] for state in progress["states"]] == pytest.approx(
        [0.0, 0.95, 0.45]
    )
