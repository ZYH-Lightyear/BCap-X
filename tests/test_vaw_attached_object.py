from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from tests.test_vaw_context_runtime import FakeContextApi
from vaw.context_runtime.attached_object import (
    ObjectProxyCandidate,
    bind_proxy_to_tcp,
    fit_gravity_stable_proxy,
    volume_triangles_base,
)
from vaw.context_runtime.model import Pose
from vaw.context_runtime.packet import ContextCompiler
from vaw.context_runtime.workspace import ContextWorkspace


def _box_points(
    center: np.ndarray,
    extent: np.ndarray,
    yaw_deg: float,
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
    rotation = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    return grid * extent @ rotation.T + center


def test_gravity_stable_proxy_rejects_outliers_and_keeps_world_z() -> None:
    center = np.array([0.42, -0.08, 0.09], dtype=np.float64)
    extent = np.array([0.10, 0.05, 0.14], dtype=np.float64)
    points = _box_points(center, extent, 27.0)
    points = np.vstack((points, np.array([[4.0, -3.0, 2.0]])))

    proxy = fit_gravity_stable_proxy("can", 3, points, padding_m=0.0)

    assert proxy is not None
    rotation = np.asarray(proxy.rotation_base_from_obb)
    assert np.allclose(rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-8)
    assert np.allclose(proxy.center_base_xyz, center, atol=0.01)
    assert np.allclose(sorted(proxy.extent_xyz_m[:2]), sorted(extent[:2]), atol=0.015)
    assert np.isclose(proxy.extent_xyz_m[2], extent[2], atol=0.015)


def test_attachment_rigidly_follows_preview_tcp() -> None:
    points = _box_points(
        np.array([0.4, 0.0, 0.08]),
        np.array([0.06, 0.04, 0.10]),
        0.0,
    )
    proxy = fit_gravity_stable_proxy("can", 1, points)
    assert proxy is not None
    current = Pose((0.4, 0.0, 0.16), (0.0, 0.0, 0.0, 1.0))
    attachment = bind_proxy_to_tcp(ObjectProxyCandidate("a1", proxy), current)
    target = Pose((0.5, -0.02, 0.22), (0.0, 0.0, 0.0, 1.0))

    current_center = volume_triangles_base(attachment, current).reshape(-1, 3).mean(axis=0)
    target_center = volume_triangles_base(attachment, target).reshape(-1, 3).mean(axis=0)

    assert np.allclose(target_center - current_center, [0.1, -0.02, 0.06], atol=1e-8)


def test_grasp_close_preview_lifecycle_and_no_numeric_packet_leak() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region_id = workspace.execute("detection_and_sam", query="can").result["region_id"]
    geometry = workspace._private.region_geometry[region_id]
    assert geometry.volume_proxy is not None
    seed_id = workspace.execute("propose_grasps", region_id=region_id).result["seed_ids"][0]
    action_id = workspace.execute("select", seed_id=seed_id).result["action_id"]
    workspace.begin_refinement("accept coarse fixture", action_id)
    workspace.execute_imagination("finish_imagination", status="ready")

    assert workspace.execute("commit", action_id=action_id).ok
    assert workspace._private.object_proxy_candidate is not None
    assert workspace._private.attachment_hypothesis is None

    assert workspace.execute("close_gripper").ok
    assert workspace._private.object_proxy_candidate is None
    assert workspace._private.attachment_hypothesis is not None
    attachment = workspace._private.attachment_hypothesis

    assert workspace.execute(
        "delta_move", delta_xyz_m=[0.0, 0.0, 0.01], frame="base"
    ).ok
    assert workspace._private.attachment_hypothesis == attachment

    point_id = workspace.execute("locate_point", query="basket center").result["point_id"]
    transport_action = workspace.execute(
        "propose_pose",
        point_id=point_id,
        offset_xyz=[0.0, 0.0, 0.05],
    ).result["action_id"]
    workspace.begin_refinement("transport preview", transport_action)
    packet = ContextCompiler().compile_imagination(workspace)
    serialized = json.dumps(packet.summary(), ensure_ascii=False).lower()
    assert "extent_xyz_m" not in serialized
    assert "tcp_from_obb" not in serialized
    assert packet.rasters["imagination_scene"].shape == (720, 1200, 3)
    assert packet.rasters["contact_focus"].shape == (720, 832, 3)
    workspace._private.attachment_hypothesis = None
    without_proxy = ContextCompiler().compile_imagination(workspace)
    workspace._private.attachment_hypothesis = attachment
    assert not np.array_equal(
        packet.rasters["imagination_scene"],
        without_proxy.rasters["imagination_scene"],
    )
    assert not np.array_equal(
        packet.rasters["contact_focus"],
        without_proxy.rasters["contact_focus"],
    )

    workspace.execute_imagination("finish_imagination", status="failed")
    assert workspace.execute("open_gripper").ok
    assert workspace._private.attachment_hypothesis is None


def test_point_anchored_object_approach_can_create_attachment_proxy() -> None:
    """Attachment geometry must not depend on selecting a GraspNet seed."""

    workspace = ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki")
    region_id = workspace.execute("detection_and_sam", query="can").result["region_id"]
    point_id = workspace.execute(
        "locate_point",
        query="center of can",
        within_region_id=region_id,
    ).result["point_id"]
    action_id = workspace.execute(
        "propose_pose",
        point_id=point_id,
        offset_xyz=[0.0, 0.0, 0.05],
    ).result["action_id"]
    workspace.begin_refinement("keep point-anchored approach", action_id)
    workspace.execute_imagination("finish_imagination", status="ready")

    assert workspace.execute("commit", action_id=action_id).ok
    candidate = workspace._private.object_proxy_candidate
    assert candidate is not None
    assert candidate.proxy.query == "can"

    assert workspace.execute("close_gripper").ok
    attachment = workspace._private.attachment_hypothesis
    assert attachment is not None
    assert attachment.query == "can"
