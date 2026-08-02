from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vaw.cloud import build_scene_cloud
from vaw.renderers import ComparisonRenderer
from vaw.scripts.smoke_render import build_state, synthetic_obs
from vaw.web_presenter import build_web_snapshot
from vaw.workspace import Workspace
from vaw.types import Candidate, Pose, TOP_DOWN_QUAT_WXYZ


def _fixture():
    obs, mask = synthetic_obs()
    state = build_state(obs, mask)
    state.arm_joint_positions_rad = np.linspace(-0.4, 0.4, 7)
    cloud = build_scene_cloud(
        obs,
        ("agentview", "robot0_eye_in_hand"),
        revision=state.obs_revision,
    )
    return state, obs, cloud


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_snapshot_exposes_only_visual_evidence() -> None:
    state, obs, cloud = _fixture()

    snapshot = build_web_snapshot(
        state,
        obs,
        camera_name="agentview",
        wrist_camera_name="robot0_eye_in_hand",
        cloud=cloud,
        render_id="test",
    )

    assert snapshot["viewport"] == {"width": 1024, "height": 576}
    assert snapshot["focus"] is None
    assert snapshot["header"]["selectedId"] == "g1"
    assert snapshot["intent"]["ik"] == "pass"
    assert snapshot["intent"]["trajectory"] == "not_checked"
    assert snapshot["intent"]["collision"] == "not_checked"
    assert snapshot["scene"]["image"].startswith("data:image/png;base64,")
    assert [candidate["id"] for candidate in snapshot["candidates"]] == [
        "g1",
        "g2",
        "g3",
        "p1",
    ]
    assert all(candidate["image"] for candidate in snapshot["candidates"])

    forbidden = {
        "depth",
        "intrinsics",
        "pose_mat",
        "mask",
        "rawmask",
        "cloud",
        "points_world",
        "env_success",
        "privileged",
    }
    normalized = {key.replace("_", "").lower() for key in _walk_keys(snapshot)}
    assert forbidden.isdisjoint(normalized)
    json.dumps(snapshot)


def test_focus_is_explicit_and_becomes_stale_after_revision() -> None:
    state, obs, cloud = _fixture()
    state.focus_id = "obj1"

    focused = build_web_snapshot(
        state,
        obs,
        camera_name="agentview",
        wrist_camera_name="robot0_eye_in_hand",
        cloud=cloud,
    )
    assert focused["focus"]["objectId"] == "obj1"
    assert focused["focus"]["hasMask"] is True
    assert focused["focus"]["hasObb"] is True
    assert focused["focus"]["stale"] is False

    state.bump_revision()
    stale = build_web_snapshot(
        state,
        obs,
        camera_name="agentview",
        wrist_camera_name="robot0_eye_in_hand",
        cloud=cloud,
    )
    assert stale["focus"]["stale"] is True
    assert all(candidate["stale"] for candidate in stale["candidates"])


def test_robot_summary_carries_exact_proprioception() -> None:
    state, _, _ = _fixture()

    robot = state.summary()["robot"]

    assert len(robot["joint_positions_rad"]) == 7
    assert len(robot["ee_quat_wxyz"]) == 4
    assert len(robot["ee_position"]) == 3


def test_selector_prefers_current_revision_over_old_history() -> None:
    state, obs, cloud = _fixture()
    state.bump_revision()
    for index in range(6, 11):
        state.add_candidate(
            Candidate(
                candidate_id=f"g{index}",
                kind="grasp",
                pose=Pose(
                    np.array([0.70 + 0.005 * (index - 6), 0.02, 0.08]),
                    TOP_DOWN_QUAT_WXYZ.copy(),
                ),
                object_id="obj2",
                obs_revision=state.obs_revision,
            )
        )

    snapshot = build_web_snapshot(
        state,
        obs,
        camera_name="agentview",
        wrist_camera_name="robot0_eye_in_hand",
        cloud=cloud,
    )

    assert [candidate["id"] for candidate in snapshot["candidates"]] == [
        "g6",
        "g7",
        "g8",
        "g9",
        "g10",
    ]
    # The stale selection is still explicit in Intent; it simply cannot evict
    # a current candidate from the selector rail.
    assert snapshot["intent"]["selectedId"] == "g1"


class _SolidRenderer:
    width = 1024
    height = 576

    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value
        self.closed = False

    def render(self, state, obs, **kwargs):
        return np.full((576, 1024, 3), self.value, dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def test_comparison_artifacts_do_not_collide_with_trace_canvases(
    tmp_path: Path,
) -> None:
    obs, _ = synthetic_obs()

    class Api:
        camera_name = "agentview"
        wrist_camera_name = "robot0_eye_in_hand"

        def get_observation(self):
            return obs

    primary = _SolidRenderer("primary", 17)
    comparison = _SolidRenderer("comparison", 29)
    compare_dir = tmp_path / "_render_compare"
    compare_dir.mkdir()
    stale_artifact = compare_dir / "primary_0099.png"
    stale_artifact.touch()
    paired = ComparisonRenderer(primary, comparison, compare_dir)
    with Workspace(
        Api(),
        "test",
        trace_dir=tmp_path / "trace",
        renderer=paired,
    ) as workspace:
        result = workspace.step("observe")

    assert int(result.canvas[0, 0, 0]) == 17
    assert (tmp_path / "trace" / "canvas_0000.png").exists()
    assert (tmp_path / "_render_compare" / "primary_0000.png").exists()
    assert (tmp_path / "_render_compare" / "comparison_0000.png").exists()
    assert not stale_artifact.exists()
    assert not list((tmp_path / "trace").glob("*comparison*"))
    record = json.loads((tmp_path / "trace" / "steps.jsonl").read_text())
    assert record["renderer"] == "primary+compare:comparison"
    assert record["render_ms"] >= 0
    assert primary.closed and comparison.closed


def test_browser_renderer_is_fixed_size_and_deterministic() -> None:
    pytest.importorskip("playwright")
    from vaw.web_renderer import WebRenderer

    state, obs, cloud = _fixture()
    state.focus_id = "obj1"
    try:
        renderer = WebRenderer()
    except (PermissionError, RuntimeError) as exc:
        pytest.skip(f"browser runtime unavailable: {exc}")
    with renderer:
        first = renderer.render(
            state,
            obs,
            camera_name="agentview",
            wrist_camera_name="robot0_eye_in_hand",
            cloud=cloud,
        )
        second = renderer.render(
            state,
            obs,
            camera_name="agentview",
            wrist_camera_name="robot0_eye_in_hand",
            cloud=cloud,
        )
        dimensions = renderer._page.evaluate(
            """() => ({
                width: document.documentElement.scrollWidth,
                height: document.documentElement.scrollHeight,
                bodyWidth: document.body.scrollWidth,
                bodyHeight: document.body.scrollHeight
            })"""
        )

    assert first.shape == (576, 1024, 3)
    assert np.array_equal(first, second)
    assert dimensions == {
        "width": 1024,
        "height": 576,
        "bodyWidth": 1024,
        "bodyHeight": 576,
    }
