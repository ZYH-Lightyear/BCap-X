"""Plumb line: deterministic vertical drop, footprint and offset overlays."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_vaw_context_runtime import FakeContextApi
from vaw.context_runtime.attached_object import (
    ObjectProxyCandidate,
    bind_proxy_to_tcp,
    fit_gravity_stable_proxy,
    volume_triangles_base,
)
from vaw.context_runtime.model import Pose
from vaw.context_runtime.packet import ContextCompiler, _compute_plumb_lines
from vaw.context_runtime.plumb_line import (
    compute_plumb_line,
    draw_plumb_overlays,
    payload_bottom_from_triangles,
    surface_height_below,
)
from vaw.context_runtime.workspace import ContextWorkspace


def _attachment_triangles(
    center: np.ndarray,
    extent: np.ndarray,
    tcp: Pose,
) -> np.ndarray:
    grid = np.stack(
        np.meshgrid(
            np.linspace(-0.5, 0.5, 9),
            np.linspace(-0.5, 0.5, 7),
            np.linspace(-0.5, 0.5, 5),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    proxy = fit_gravity_stable_proxy("can", 1, grid * extent + center, padding_m=0.0)
    assert proxy is not None
    attachment = bind_proxy_to_tcp(ObjectProxyCandidate("a1", proxy), tcp)
    return volume_triangles_base(attachment, tcp)


def _overhead_camera(depth_m: float = 0.7, height: int = 120, width: int = 160) -> dict:
    """A synthetic camera 1 m up, looking straight down world -Z."""

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[2, 3] = 1.0
    return {
        "images": {
            "rgb": np.zeros((height, width, 3), dtype=np.uint8),
            "depth": np.full((height, width), depth_m, dtype=np.float64),
        },
        "intrinsics": np.array(
            [[120.0, 0.0, width / 2], [0.0, 120.0, height / 2], [0.0, 0.0, 1.0]]
        ),
        "pose_mat": pose,
    }


def test_payload_bottom_is_the_obb_bottom_face_centre() -> None:
    center = np.array([0.4, -0.05, 0.20])
    extent = np.array([0.06, 0.04, 0.10])
    tcp = Pose((0.4, -0.05, 0.30), (0.0, 0.0, 0.0, 1.0))
    triangles = _attachment_triangles(center, extent, tcp)

    anchor, corners = payload_bottom_from_triangles(triangles)

    assert np.allclose(anchor[:2], center[:2], atol=0.01)
    assert np.isclose(anchor[2], center[2] - extent[2] / 2, atol=0.01)
    assert corners.shape == (4, 3)
    assert np.allclose(corners[:, 2], anchor[2], atol=1e-6)


def test_surface_height_below_reads_the_observed_plane() -> None:
    camera = _overhead_camera(depth_m=0.7)  # plane at world z = 0.3

    surface = surface_height_below(camera, np.array([0.0, 0.0, 0.5]))

    assert surface is not None
    assert np.isclose(surface, 0.3, atol=0.01)


def test_surface_height_requires_points_below_the_anchor() -> None:
    camera = _overhead_camera(depth_m=0.7)

    # The anchor sits on the plane itself: nothing hangs below it.
    assert surface_height_below(camera, np.array([0.0, 0.0, 0.3])) is None


def test_compute_plumb_line_offsets_and_footprint() -> None:
    anchor = np.array([0.4, 0.0, 0.5])
    footprint = np.array(
        [
            [0.37, -0.02, 0.45],
            [0.43, -0.02, 0.45],
            [0.43, 0.02, 0.45],
            [0.37, 0.02, 0.45],
        ]
    )

    plumb = compute_plumb_line(
        "preview",
        anchor,
        0.3,
        footprint_base=footprint,
        target_center_base_xyz=(0.44, 0.03, 0.32),
    )

    assert plumb is not None
    assert np.isclose(plumb.height_m, 0.2)
    assert plumb.landing_base_xyz == (0.4, 0.0, 0.3)
    assert np.allclose(plumb.footprint_base[:, 2], 0.3)
    assert np.allclose(plumb.offset_xy_m, (0.04, 0.03))

    # Anchors at (or below) the surface produce no plumb at all.
    assert compute_plumb_line("current", anchor, 0.499) is None
    assert compute_plumb_line("current", anchor, None) is None


def test_draw_plumb_overlays_changes_pixels_only_when_plumbs_exist() -> None:
    camera = _overhead_camera()
    image = np.asarray(camera["images"]["rgb"]).copy()
    untouched = image.copy()

    draw_plumb_overlays(image, camera, [])
    assert np.array_equal(image, untouched)

    plumb = compute_plumb_line(
        "current",
        np.array([0.0, 0.0, 0.5]),
        0.3,
        footprint_base=np.array(
            [
                [-0.03, -0.02, 0.45],
                [0.03, -0.02, 0.45],
                [0.03, 0.02, 0.45],
                [-0.03, 0.02, 0.45],
            ]
        ),
        target_center_base_xyz=(0.05, 0.04, 0.32),
    )
    assert plumb is not None
    draw_plumb_overlays(image, camera, [plumb])
    assert not np.array_equal(image, untouched)


def test_compile_computes_a_current_plumb_for_an_attached_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fake robot's FK silhouette degenerates to a full-frame mask under
    # the synthetic camera; disable it as the evidence-lifecycle tests do.
    from vaw.context_runtime import evidence as evidence_module

    monkeypatch.setattr(evidence_module, "robot_silhouette", lambda camera, robot: None)
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region_id = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed_id = workspace.execute("propose_grasps", region_id=region_id).result["seed_ids"][0]
    action_id = workspace.execute("select", seed_id=seed_id).result["action_id"]
    workspace.begin_imagination("accept coarse fixture", action_id)
    workspace.execute_imagination("finish_imagination", status="ready")
    assert workspace.execute("commit", action_id=action_id).ok
    assert workspace.execute("close_gripper").ok
    attachment = workspace._private.attachment_hypothesis
    assert attachment is not None

    # The fake agentview cannot see under the payload; a synthetic overhead
    # camera provides the observed support plane at world z = 0.3.
    robot = workspace.state.robot
    assert robot is not None and robot.tcp_pose is not None
    payload_bottom_z = payload_bottom_from_triangles(
        volume_triangles_base(attachment, robot.tcp_pose)
    )[0][2]
    camera = _overhead_camera(depth_m=1.0 - (payload_bottom_z - 0.1))

    plumbs = _compute_plumb_lines(
        workspace.state,
        workspace._private,
        camera,
        None,
        None,
        None,
    )

    assert [plumb.kind for plumb in plumbs] == ["current"]
    assert plumbs[0].footprint_base is not None
    assert np.isclose(plumbs[0].height_m, 0.1, atol=0.02)


def test_dxy_offset_targets_the_semantic_anchor_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vaw.context_runtime import evidence as evidence_module

    monkeypatch.setattr(evidence_module, "robot_silhouette", lambda camera, robot: None)
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    point_id = workspace.execute(
        "locate_point", query="basket opening center"
    ).result["point_id"]
    point = workspace.state.points[point_id]
    assert workspace.execute(
        "propose_pose", point_id=point_id, offset_xyz=[0.03, -0.02, 0.15]
    ).ok
    proposal = workspace.state.action_proposal
    assert proposal is not None
    target_z = proposal.target.pose.position_xyz[2]
    camera = _overhead_camera(depth_m=1.0 - (target_z - 0.1))

    plumbs = _compute_plumb_lines(
        workspace.state,
        workspace._private,
        camera,
        proposal.target,
        None,
        workspace._private.action_artifacts,
    )

    assert [plumb.kind for plumb in plumbs] == ["preview"]
    # The offset arrow points at the measured locate_point anchor, not at a
    # whole-region centroid: for a basket that centroid sits in the body, and
    # a biased arrow sends refinement chasing a skewed target.
    assert plumbs[0].target_center_base_xyz == tuple(
        float(value) for value in point.position_xyz
    )
    assert np.allclose(plumbs[0].offset_xy_m, (-0.03, 0.02), atol=1e-9)


def test_dxy_offset_is_absent_without_a_semantic_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vaw.context_runtime import evidence as evidence_module

    monkeypatch.setattr(evidence_module, "robot_silhouette", lambda camera, robot: None)
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region_id = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed_id = workspace.execute("propose_grasps", region_id=region_id).result["seed_ids"][0]
    assert workspace.execute("select", seed_id=seed_id).ok
    proposal = workspace.state.action_proposal
    assert proposal is not None
    target_z = proposal.target.pose.position_xyz[2]
    camera = _overhead_camera(depth_m=1.0 - (target_z - 0.1))

    plumbs = _compute_plumb_lines(
        workspace.state,
        workspace._private,
        camera,
        proposal.target,
        None,
        workspace._private.action_artifacts,
    )

    assert [plumb.kind for plumb in plumbs] == ["preview"]
    assert plumbs[0].target_center_base_xyz is None
    assert plumbs[0].offset_xy_m is None


def test_short_dxy_draws_a_target_circle_instead_of_an_arrow() -> None:
    camera = _overhead_camera()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    plumb = compute_plumb_line(
        "current",
        np.array([0.0, 0.0, 0.42]),
        0.30,
        target_center_base_xyz=(0.006, 0.0, 0.30),
    )
    assert plumb is not None
    draw_plumb_overlays(image, camera, [plumb])
    # A 6 mm offset is < 12 px on this camera, so the marker is a compact
    # circle around the target rather than a long shaft.
    red = image[:, :, 0] > 180
    assert int(np.count_nonzero(red)) > 20
    rows, cols = np.nonzero(red)
    assert int(cols.max() - cols.min()) < 90
    assert int(rows.max() - rows.min()) < 60


def test_main_compile_with_attachment_still_renders() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region_id = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed_id = workspace.execute("propose_grasps", region_id=region_id).result["seed_ids"][0]
    action_id = workspace.execute("select", seed_id=seed_id).result["action_id"]
    workspace.begin_imagination("accept coarse fixture", action_id)
    workspace.execute_imagination("finish_imagination", status="ready")
    assert workspace.execute("commit", action_id=action_id).ok
    assert workspace.execute("close_gripper").ok

    packet = ContextCompiler().compile(workspace)

    assert packet.rasters["agentview"].shape == (120, 160, 3)
    assert packet.rasters["imagination_scene"].shape == (120, 160, 3)
    assert packet.schema == "vaw-context-v46-oblique-contact"
