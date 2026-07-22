from __future__ import annotations

import numpy as np
import pytest

from robomex.orchestration.arena import (
    ArenaHypothesisConfig,
    ArenaPreviewConfig,
    ArenaRuntimeContextV1,
)
from robomex.orchestration.motion_preview import (
    MotionPreviewError,
    PointCloudMotionPreviewRenderer,
)
from robomex.test.test_action_protocol_v2 import _motion, _snapshot


class _Geometry:
    provider_id = "test-fk-pointcloud-v1"

    def __init__(self, *, invalid: bool = False) -> None:
        self.invalid = invalid

    def scene_points(self, *, context):
        del context
        points = np.linspace(-0.1, 0.1, 15_000).reshape(5_000, 3)
        if self.invalid:
            points[0, 0] = np.nan
        return points

    def tcp_positions(self, *, plan, context):
        del context
        return np.array(
            [[index * 0.01, index * 0.005, 0.2] for index, _ in enumerate(plan.motion.positions_rad)]
        )


def _context(plan):
    return ArenaRuntimeContextV1(
        episode_id="episode",
        workflow_id="workflow",
        graph_id="graph",
        graph_revision=1,
        slot_id="motion_arena",
        arena_run_id="arena-run",
        candidate_id="candidate-0",
        strategy="direct",
        snapshot_ref={
            "artifact_id": "snapshot-artifact",
            "content_digest": "sha256:" + "a" * 64,
        },
        expected_frame="world",
        world_id=plan.world_id,
        resource_id=plan.resource_id,
        robot_model_digest=plan.robot_model_digest,
        config_digest=plan.expected_snapshot.config_digest,
        hypothesis_config=ArenaHypothesisConfig(),
        preview_config=ArenaPreviewConfig(
            enabled=True,
            renderer_id="robomex.pointcloud_motion_preview.v1",
        ),
    )


def test_pointcloud_renderer_emits_content_addressed_views(tmp_path) -> None:
    plan = _motion(_snapshot())
    renderer = PointCloudMotionPreviewRenderer(
        geometry=_Geometry(),
        output_root=tmp_path / "previews",
    )

    first = renderer.render(plan=plan, context=_context(plan))
    second = renderer.render(plan=plan, context=_context(plan))

    assert tuple(frame.view_id for frame in first.frames) == ("perspective", "top_down")
    assert first == second
    assert first.terminal_position_m is not None
    for frame in first.frames:
        assert frame.media_type == "image/png"
        assert frame.media_digest.startswith("sha256:")
        assert frame.payload["rendered_point_count"] == 4_000
        assert (tmp_path / "previews" / frame.payload["path"].rsplit("/", 1)[-1]).is_file()


def test_pointcloud_renderer_rejects_nonfinite_geometry(tmp_path) -> None:
    plan = _motion(_snapshot())
    renderer = PointCloudMotionPreviewRenderer(
        geometry=_Geometry(invalid=True),
        output_root=tmp_path / "previews",
    )

    with pytest.raises(MotionPreviewError, match="NaN"):
        renderer.render(plan=plan, context=_context(plan))
