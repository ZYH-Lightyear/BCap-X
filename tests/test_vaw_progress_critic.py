from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from vaw.diagnostics.progress_critic import (
    ActionEvidence,
    add_progress_dynamics,
    build_action_evidence,
    build_action_messages,
    label_distribution,
    progress_from_distribution,
    write_action_strip,
)
from vaw.diagnostics.progress_video import build_progress_video_project
from vaw.diagnostics.progress_observations import build_multiview_transitions
from vaw.diagnostics.run_pairwise_progress_demo import (
    ComparisonEdge,
    edge_target,
    fit_latent_progress,
)
from vaw.diagnostics.run_progress_sweep import discover_traces


def _image(path: Path, value: int) -> None:
    Image.new("RGB", (32, 24), color=(value, value, value)).save(path)


def _video(path: Path, values: list[int]) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24)
    )
    assert writer.isOpened()
    try:
        for value in values:
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    finally:
        writer.release()


def _trace(tmp_path: Path) -> Path:
    trace = tmp_path / "task8_s1"
    action_dir = trace / "actions/action_0001"
    action_dir.mkdir(parents=True)
    _image(trace / "context_0000.png", 10)
    _image(trace / "context_0001.png", 30)
    (action_dir / "agentview.mp4").write_bytes(b"video")
    (action_dir / "poster.jpg").write_bytes(b"poster")
    (trace / "video_agentview.mp4").write_bytes(b"episode-video")
    (trace / "meta.json").write_text(
        json.dumps({"task_prompt": "put the bowl on the plate"}), encoding="utf-8"
    )
    records = [
        {
            "turn": 1,
            "context_image": "context_0000.png",
            "function_call": {"name": "execute_action", "arguments": {"action_id": "a1"}},
            "function_result": {"status": "completed"},
        },
        {
            "turn": 2,
            "context_image": "context_0001.png",
            "function_call": {"name": "finish_task", "arguments": {"success": False}},
        },
    ]
    (trace / "steps.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (action_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "vaw-action-segment-v1",
                "segment_id": "action_0001",
                "turn": 1,
                "function": "execute_action",
                "arguments": {"action_id": "a1"},
                "outcome": "completed",
                "revision_before": 1,
                "revision_after": 2,
                "frame_start": 1,
                "frame_end": 31,
                "fps": 30,
                "streams": {"agentview": {"path": "actions/action_0001/agentview.mp4"}},
                "poster": "actions/action_0001/poster.jpg",
            }
        ),
        encoding="utf-8",
    )
    return trace


def test_action_evidence_uses_manifest_time_and_following_canvas(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    actions = build_action_evidence(trace)

    assert len(actions) == 1
    action = actions[0]
    assert action.before_canvas == (trace / "context_0000.png").resolve()
    assert action.after_canvas == (trace / "context_0001.png").resolve()
    assert action.time_start_s == 1 / 30
    assert action.time_end_s == 1.0


def test_missing_action_video_uses_visual_state_diff_strip(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    (trace / "actions/action_0001/agentview.mp4").unlink()
    manifest_path = trace / "actions/action_0001/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_end"] = manifest["frame_start"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    action = build_action_evidence(trace)[0]

    assert action.action_video is None
    assert action.time_end_s == action.time_start_s
    strip = write_action_strip(
        None,
        tmp_path / "fallback.jpg",
        fallback_before=action.before_canvas,
        fallback_after=action.after_canvas,
    )
    assert strip.is_file()


def test_corrupt_action_video_uses_visual_state_diff_strip(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    video = trace / "actions/action_0001/agentview.mp4"
    video.write_bytes(b"")
    action = build_action_evidence(trace)[0]

    strip = write_action_strip(
        action.action_video,
        tmp_path / "corrupt-fallback.jpg",
        fallback_before=action.before_canvas,
        fallback_after=action.after_canvas,
    )

    assert strip.is_file()


def test_terminal_action_uses_manifest_poster_as_post_action_state(
    tmp_path: Path,
) -> None:
    trace = _trace(tmp_path)
    first_step = (trace / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0]
    (trace / "steps.jsonl").write_text(first_step + "\n", encoding="utf-8")
    (trace / "context_0001.png").unlink()

    action = build_action_evidence(trace)[0]

    assert action.after_canvas == (trace / "actions/action_0001/poster.jpg").resolve()


def test_raw_multiview_evidence_uses_synchronized_manifest_boundaries(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    _video(trace / "video_agentview.mp4", [10, 20, 30, 40, 50])
    _video(trace / "video_wrist.mp4", [60, 70, 80, 90, 100])
    placeholder = trace / "placeholder.png"
    _image(placeholder, 0)
    action = ActionEvidence(
        index=1,
        segment_id="action_0001",
        turn=1,
        function="execute_action",
        arguments={},
        outcome="completed",
        revision_before=1,
        revision_after=2,
        frame_start=1,
        frame_end=4,
        fps=30,
        time_start_s=1 / 30,
        time_end_s=3 / 30,
        before_canvas=placeholder,
        after_canvas=placeholder,
        action_video=None,
        poster=None,
        function_result=None,
        runtime_diagnostics=None,
    )

    initial, transitions = build_multiview_transitions(
        trace_dir=trace,
        actions=[action],
        output_dir=tmp_path / "evidence",
    )

    assert initial.frame_index == 0
    assert transitions[0].before.frame_index == 0
    assert transitions[0].after.frame_index == 3
    before_agent = cv2.imread(str(transitions[0].before.agentview))
    after_agent = cv2.imread(str(transitions[0].after.agentview))
    before_wrist = cv2.imread(str(transitions[0].before.wrist))
    after_wrist = cv2.imread(str(transitions[0].after.wrist))
    assert float(before_agent.mean()) < float(after_agent.mean())
    assert float(before_wrist.mean()) < float(after_wrist.mean())


def test_absolute_progress_dynamics_preserve_drop_magnitude() -> None:
    states = [
        {"progress": 0.05},
        {"progress": 0.55},
        {"progress": 0.82},
        {"progress": 0.12},
    ]
    add_progress_dynamics(states)

    assert states[1]["delta"] == pytest.approx(0.5)
    assert states[2]["delta"] == pytest.approx(0.27)
    assert states[3]["delta"] == pytest.approx(-0.7)
    assert states[3]["acceleration"] < -0.9


def test_pairwise_log_odds_is_continuous_and_signed() -> None:
    advance, advance_weight = edge_target({"A": 0.73, "B": 0.22, "C": 0.05})
    unchanged, unchanged_weight = edge_target({"A": 0.05, "B": 0.9, "C": 0.05})
    regress, regress_weight = edge_target({"A": 0.04, "B": 0.16, "C": 0.8})

    assert advance > 0
    assert unchanged == pytest.approx(0.0)
    assert regress < 0
    assert all(weight > 0 for weight in (advance_weight, unchanged_weight, regress_weight))


def test_pairwise_graph_fit_preserves_large_regression() -> None:
    def edge(left: int, right: int, target: float) -> ComparisonEdge:
        return ComparisonEdge(
            kind="test",
            left_state=left,
            right_state=right,
            preference_signal=target,
            weight=1.0,
            probabilities={"A": 0.5, "B": 0.5, "C": 0.0},
            predicted_label="A",
            response_path="unused.json",
        )

    scores = fit_latent_progress(
        4,
        [
            edge(0, 1, 1.0),
            edge(1, 2, 1.0),
            edge(0, 2, 2.0),
            edge(2, 3, -2.5),
            edge(0, 3, -0.5),
        ],
    )

    assert scores[0] == pytest.approx(0.0)
    assert scores[2] > scores[1] > scores[0]
    assert scores[3] < scores[0]


def test_logprob_distribution_yields_absolute_expected_progress() -> None:
    choice = {
        "logprobs": {
            "content": [
                {
                    "token": "D",
                    "logprob": -0.2,
                    "top_logprobs": [
                        {"token": "D", "logprob": -0.2},
                        {"token": "C", "logprob": -1.7},
                        {"token": "E", "logprob": -2.5},
                    ],
                }
            ]
        }
    }
    distribution = label_distribution(choice)
    progress, probabilities = progress_from_distribution(
        distribution, predicted_label="D"
    )

    assert distribution["available"] is True
    assert probabilities["D"] > probabilities["C"] > probabilities["E"]
    assert progress is not None and 0.65 < progress < 0.8


def test_critic_request_strips_terminal_reward_fields(tmp_path: Path) -> None:
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    strip = tmp_path / "strip.jpg"
    for path, value in ((before, 10), (after, 20), (strip, 30)):
        _image(path, value)
    action = ActionEvidence(
        index=1,
        segment_id="action_0001",
        turn=1,
        function="execute_action",
        arguments={"action_id": "a1"},
        outcome="completed",
        revision_before=1,
        revision_after=2,
        frame_start=0,
        frame_end=3,
        fps=30,
        time_start_s=0,
        time_end_s=0.1,
        before_canvas=before,
        after_canvas=after,
        action_video=tmp_path / "action.mp4",
        poster=None,
        function_result={"env_success": True, "reward": 1.0, "status": "completed"},
        runtime_diagnostics=None,
    )
    messages = build_action_messages(
        task="put the bowl on the plate", initial=before, action=action, strip=strip
    )
    trace_text = messages[1]["content"][-1]["text"]

    assert "env_success" not in trace_text
    assert '"reward"' not in trace_text
    assert '"status": "completed"' in trace_text


def test_progress_video_project_embeds_scored_states(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    progress_path = tmp_path / "progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "task": "put the bowl on the plate",
                "states": [
                    {
                        "state_index": 0,
                        "turn": 0,
                        "function": "initial_state",
                        "time_s": 0.0,
                        "level": "A",
                        "progress": 0.0,
                        "delta": None,
                        "acceleration": None,
                    },
                    {
                        "state_index": 1,
                        "turn": 1,
                        "function": "execute_action",
                        "time_s": 1.0,
                        "level": "C",
                        "progress": 0.5,
                        "delta": 0.5,
                        "acceleration": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = build_progress_video_project(
        trace_dir=trace,
        progress_path=progress_path,
        output_dir=tmp_path / "video_project",
    )

    html = (output / "index.html").read_text(encoding="utf-8")
    assert "__DURATION__" not in html
    assert "__PROGRESS_DATA__" not in html
    assert "put the bowl on the plate" in html
    assert '"progress":0.5' in html
    assert (output / "agentview.mp4").read_bytes() == b"episode-video"


def test_sweep_discovery_only_returns_complete_direct_episode_dirs(
    tmp_path: Path,
) -> None:
    sweep = tmp_path / "sweep"
    complete = sweep / "libero_object" / "task0_s1"
    incomplete = sweep / "libero_object" / "task1_s1"
    nested = complete / "progress_critic" / "task2_s1"
    for trace in (complete, incomplete, nested):
        trace.mkdir(parents=True)
    for filename in ("meta.json", "steps.jsonl", "video_agentview.mp4"):
        (complete / filename).write_bytes(b"{}")
        (nested / filename).write_bytes(b"{}")
    (incomplete / "meta.json").write_text("{}", encoding="utf-8")

    assert discover_traces(sweep) == [complete.resolve()]
