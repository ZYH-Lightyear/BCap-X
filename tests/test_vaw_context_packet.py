from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from tests.test_vaw_context_runtime import FailedMotionBackend, FakeContextApi
from vaw.context_runtime import packet as packet_module
from vaw.context_runtime.contact_camera import ContactCameraPair, ContactCameraSelection
from vaw.context_runtime.packet import (
    CONTEXT_HEIGHT,
    CONTEXT_SCHEMA,
    CONTEXT_WEB_SCHEMA_VERSION,
    CONTEXT_WIDTH,
    IMAGINATION_CONTEXT_HEIGHT,
    IMAGINATION_CONTEXT_WIDTH,
    ContextCompiler,
    _contact_camera_center,
    _current_gripper_contact_points,
    _observed_closed_gripper_z_axis,
    _preview_contact_points,
    _presentation_region_ref,
    _required_contact_points,
)
from vaw.context_runtime.private import LastPhysicalArtifacts
from vaw.context_runtime.model import Pose
from vaw.context_runtime.workspace import ContextWorkspace


def _workspace() -> ContextWorkspace:
    api = FakeContextApi()
    api.rgb[:, :, 0] = np.arange(api.rgb.shape[1], dtype=np.uint8)
    return ContextWorkspace(api, "place the can in the basket", motion_backend="pyroki")


def _selected_action(workspace: ContextWorkspace) -> str:
    region = workspace.execute("detect_region", query="can").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    return workspace.execute("preview_grasp", seed_id=seed).result["action_id"]


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).replace("_", "").lower()
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_main_packet_uses_clean_current_rgb_and_fixed_viewport() -> None:
    workspace = _workspace()
    packet = ContextCompiler().compile(workspace)

    assert packet.schema == CONTEXT_SCHEMA
    assert packet.projection == "main"
    assert np.array_equal(
        packet.rasters["agentview"],
        workspace._private.camera("agentview")["images"]["rgb"],
    )
    assert packet.summary()["viewport"] == {
        "width": CONTEXT_WIDTH,
        "height": CONTEXT_HEIGHT,
    }


def test_observed_grasp_sweep_is_persistent_and_contact_only(monkeypatch) -> None:
    workspace = _workspace()
    segment = np.array(
        [[0.35, -0.03, 0.25], [0.35, 0.03, 0.25]], dtype=np.float64
    )
    routed_segments: list[np.ndarray] = []

    monkeypatch.setattr(
        packet_module,
        "_observed_grasp_sweep_segment",
        lambda robot: segment.copy(),
    )

    def capture_contact_render(*args, **kwargs):
        del args
        routed_segments.append(kwargs["grasp_sweep_segment_base"].copy())
        return np.zeros(
            (
                2 * packet_module.MAIN_CONTACT_HEIGHT
                + packet_module.CONTACT_PANEL_GAP,
                packet_module.MAIN_CONTACT_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

    monkeypatch.setattr(packet_module, "render_contact_focus", capture_contact_render)
    raw_rgb = workspace._private.camera("agentview")["images"]["rgb"].copy()
    compiler = ContextCompiler()

    before_close = compiler.compile(workspace)
    workspace.execute("close_gripper")
    after_close = compiler.compile(workspace)

    assert len(routed_segments) == 2
    assert all(np.array_equal(value, segment) for value in routed_segments)
    assert np.array_equal(before_close.rasters["agentview"], raw_rgb)
    assert np.array_equal(after_close.rasters["agentview"], raw_rgb)
    assert before_close.world.contact_front_raster_id == "contact_front"
    assert after_close.world.contact_front_raster_id == "contact_front"


def test_closed_gripper_z_axis_survives_revisions_until_open() -> None:
    workspace = _workspace()

    assert _observed_closed_gripper_z_axis(
        workspace.state.robot, workspace._private
    ) is None

    workspace.execute("close_gripper")
    first = _observed_closed_gripper_z_axis(
        workspace.state.robot, workspace._private
    )
    assert first is not None and first.shape == (3, 3)
    assert np.allclose(first[1], workspace.state.robot.tcp_pose.position_xyz)
    assert first[0, 2] < first[1, 2] < first[2, 2]

    workspace.refresh_observation()
    repeated = _observed_closed_gripper_z_axis(
        workspace.state.robot, workspace._private
    )
    assert repeated is not None

    workspace.execute("open_gripper")
    assert _observed_closed_gripper_z_axis(
        workspace.state.robot, workspace._private
    ) is None


def test_preview_hides_current_grasp_sweep_and_z_axis(monkeypatch) -> None:
    workspace = _workspace()
    workspace.execute("close_gripper")
    _selected_action(workspace)
    routed: list[tuple[np.ndarray | None, np.ndarray | None]] = []

    def capture_contact_render(*args, **kwargs):
        del args
        routed.append(
            (
                kwargs["grasp_sweep_segment_base"],
                kwargs["gripper_z_axis_base"],
            )
        )
        return np.zeros(
            (
                2 * packet_module.MAIN_CONTACT_HEIGHT
                + packet_module.CONTACT_PANEL_GAP,
                packet_module.MAIN_CONTACT_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

    monkeypatch.setattr(packet_module, "render_contact_focus", capture_contact_render)

    ContextCompiler().compile(workspace)

    assert routed == [(None, None)]


def test_verified_detect_region_persists_in_contact_overlay_until_occluded(
    monkeypatch,
) -> None:
    workspace = _workspace()
    region_id = workspace.execute("detect_region", query="can").result["region_id"]
    routed_evidence: list[tuple[np.ndarray | None, np.ndarray | None]] = []

    def capture_contact_render(*args, **kwargs):
        del args
        mask = kwargs["source_mask"]
        points = kwargs["source_points_base"]
        routed_evidence.append(
            (
                None if mask is None else np.asarray(mask).copy(),
                None if points is None else np.asarray(points).copy(),
            )
        )
        return np.zeros(
            (
                2 * packet_module.MAIN_CONTACT_HEIGHT
                + packet_module.CONTACT_PANEL_GAP,
                packet_module.MAIN_CONTACT_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

    monkeypatch.setattr(packet_module, "render_contact_focus", capture_contact_render)
    compiler = ContextCompiler()

    compiler.compile(workspace)
    region = workspace.state.regions[region_id]
    workspace.state.regions[region_id] = replace(region, status="occluded")
    compiler.compile(workspace)

    verified_mask, verified_points = routed_evidence[0]
    assert verified_mask is not None and verified_mask.any()
    assert verified_points is not None and len(verified_points) >= 3
    assert routed_evidence[1] == (None, None)


def test_verified_detect_region_reframes_cached_contact_camera() -> None:
    calls: list[object] = []

    def provider(request):
        calls.append(request)
        camera = {
            "images": {"rgb": np.zeros((358, 832, 3), dtype=np.uint8)},
            "intrinsics": np.array(
                [[360.0, 0.0, 416.0], [0.0, 360.0, 179.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            "pose_mat": np.eye(4, dtype=np.float64),
            "view_name": "front",
        }
        return ContactCameraPair(
            front=camera,
            side={**camera, "view_name": "side"},
            selection=ContactCameraSelection(1, 1, 1.0, 1.0, 1.0),
        )

    workspace = ContextWorkspace(
        FakeContextApi(),
        "pick the can",
        motion_backend="pyroki",
        contact_camera_provider=provider,
    )
    compiler = ContextCompiler()

    workspace.execute("open_gripper")
    compiler.compile(workspace)
    assert len(calls) == 1
    assert calls[0].subject_points_base is None

    workspace.execute("detect_region", query="can")
    compiler.compile(workspace)
    compiler.compile(workspace)

    assert len(calls) == 2
    assert calls[1].subject_points_base is not None
    assert len(calls[1].subject_points_base) >= 3


def test_main_packet_never_projects_previous_action_visual() -> None:
    api = FakeContextApi()
    api.rgb.fill(31)
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")
    compiler = ContextCompiler()

    initial = compiler.compile(workspace)
    assert "previous_action" not in initial.rasters
    assert "previousActionRasterId" not in initial.world.summary()
    assert "physicalTransition" not in initial.world.summary()

    before_revision = workspace.state.observation_revision
    api.rgb.fill(79)
    result = workspace.execute("close_gripper")
    after = compiler.compile(workspace)

    assert result.revision_before == before_revision
    assert result.revision_after == before_revision + 1
    # Runtime still retains the causal pre-action observation for trace,
    # replay, and Progress Critic; only the policy Canvas drops it.
    assert np.all(
        workspace._private.camera("agentview", previous=True)["images"]["rgb"]
        == 31
    )
    assert np.all(after.rasters["agentview"] == 79)
    assert "previous_action" not in after.rasters
    assert "previousActionRasterId" not in after.world.summary()
    assert "physicalTransition" not in after.world.summary()


def test_action_proposal_is_main_owned_until_explicit_refinement() -> None:
    workspace = _workspace()
    compiler = ContextCompiler()
    region = workspace.execute("detect_region", query="can").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    selected = workspace.execute("preview_grasp", seed_id=seed)

    main_packet = compiler.compile(workspace)
    assert main_packet.projection == "main"
    assert main_packet.decision.mode == "proposal"
    assert main_packet.manifest()["action_proposal"] == {
        "action_id": selected.result["action_id"],
        "intent": "approach can for grasp",
        "state": "planned",
        "executable": True,
    }
    assert main_packet.manifest()["regions"] == [
        {"id": region, "query": "can"}
    ]
    assert main_packet.manifest()["seed_ids"]
    assert main_packet.world.action["status"] == "planned"

    workspace.begin_imagination("下移直到罐体处于两指扫掠区域", selected.result["action_id"])
    focused = compiler.compile_imagination(workspace)
    assert focused.projection == "imagination"
    assert focused.summary()["viewport"] == {
        "width": IMAGINATION_CONTEXT_WIDTH,
        "height": IMAGINATION_CONTEXT_HEIGHT,
    }
    assert focused.decision.mode == "editing"
    assert focused.world.action["status"] == "refining"
    assert focused.world.refinement_goal == "下移直到罐体处于两指扫掠区域"

    workspace.execute_imagination("finish_imagination", status="ready")
    reviewed = compiler.compile(workspace)
    assert reviewed.decision.mode == "proposal"
    assert reviewed.world.action["status"] == "refined"


def test_manifest_marks_unplanned_action_as_not_executable() -> None:
    workspace = ContextWorkspace(
        FakeContextApi(),
        "task",
        motion_backend=FailedMotionBackend(),
    )
    point_id = workspace.execute("locate_point", query="center").result["point_id"]
    workspace.execute(
        "preview_pose",
        point_id=point_id,
        offset_xyz_m=[0.0, 0.0, 0.0],
    )

    proposal = ContextCompiler().compile(workspace).manifest()["action_proposal"]

    assert proposal is not None
    assert proposal["executable"] is False


def test_focused_projection_uses_same_revision_and_native_contact_rasters() -> None:
    workspace = _workspace()
    action_id = _selected_action(workspace)
    workspace.begin_imagination("向上 1cm", action_id)
    compiler = ContextCompiler()

    main = compiler.compile(workspace)
    focused = compiler.compile_imagination(workspace)

    assert main.revision == focused.revision
    assert main.manifest() == focused.manifest()
    assert set(main.rasters) == set(focused.rasters)
    for raster_id in main.rasters:
        if raster_id.startswith("contact_"):
            assert main.rasters[raster_id].shape != focused.rasters[raster_id].shape
        else:
            assert np.array_equal(main.rasters[raster_id], focused.rasters[raster_id])


def test_contact_camera_pair_is_locked_per_refinement_and_reselected_on_reentry() -> None:
    calls: list[object] = []

    def provider(request):
        calls.append(request)
        camera = {
            "images": {
                "rgb": np.zeros((358, 832, 3), dtype=np.uint8),
            },
            "intrinsics": np.array(
                [[360.0, 0.0, 456.0], [0.0, 360.0, 146.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            "pose_mat": np.eye(4, dtype=np.float64),
            "view_name": "front",
        }
        return ContactCameraPair(
            front=camera,
            side={**camera, "view_name": "side"},
            selection=ContactCameraSelection(1, 1, 1.0, 1.0, 1.0),
        )

    api = FakeContextApi()
    workspace = ContextWorkspace(
        api,
        "place the can in the basket",
        motion_backend="pyroki",
        contact_camera_provider=provider,
    )
    compiler = ContextCompiler()
    action_id = _selected_action(workspace)
    workspace.begin_imagination("向下微调", action_id)

    compiler.compile_imagination(workspace)
    compiler.compile_imagination(workspace)
    assert len(calls) == 1
    assert calls[0].required_points_base is not None
    assert len(calls[0].required_points_base) > 0

    workspace.execute_imagination(
        "move_tcp_delta", delta_xyz_m=[0.0, 0.0, 0.01], frame="base"
    )
    workspace.execute_imagination("finish_imagination", status="ready")
    compiler.compile(workspace)
    assert len(calls) == 2
    assert calls[1].preferred_signs == (1, 1)
    assert calls[1].width != calls[0].width

    workspace.begin_imagination("再次检查", action_id)
    compiler.compile_imagination(workspace)
    assert len(calls) == 3


def test_carrying_keeps_contact_pair_level_and_adds_steep_auxiliary() -> None:
    calls: list[object] = []

    def provider(request):
        calls.append(request)
        camera = {
            "images": {"rgb": np.zeros((358, 832, 3), dtype=np.uint8)},
            "intrinsics": np.array(
                [[360.0, 0.0, 456.0], [0.0, 360.0, 146.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            "pose_mat": np.eye(4, dtype=np.float64),
            "view_name": "front",
        }
        return ContactCameraPair(
            front=camera,
            side={**camera, "view_name": "side"},
            selection=ContactCameraSelection(
                1,
                1,
                1.0,
                1.0,
                1.0,
                side_elevation_deg=float(request.side_elevation_deg),
            ),
            auxiliary={**camera, "view_name": "auxiliary"},
        )

    workspace = ContextWorkspace(
        FakeContextApi(),
        "place the can in the basket",
        motion_backend="pyroki",
        contact_camera_provider=provider,
    )
    compiler = ContextCompiler()
    action_id = _selected_action(workspace)

    # Approaching a grasp keeps both panels level: finger clearance against the
    # support surface is only measurable from the horizon.
    assert compiler.compile(workspace).world.contact_side_elevation_deg == 0.0
    assert calls[-1].side_elevation_deg == 0.0
    assert calls[-1].auxiliary_elevation_deg == 68.0

    assert workspace.execute("execute_action", action_id=action_id).ok
    assert workspace.execute("close_gripper").ok
    assert workspace._private.attachment_hypothesis is not None

    packet = compiler.compile(workspace)

    assert packet.world.contact_side_elevation_deg == 0.0
    assert packet.world.summary()["contactSideElevationDeg"] == 0.0
    assert calls[-1].side_elevation_deg == 0.0
    assert calls[-1].auxiliary_elevation_deg == 68.0
    assert packet.world.contact_auxiliary_raster_id == "contact_auxiliary"
    assert "contact_auxiliary" in packet.rasters


def test_point_pose_contact_framing_uses_parent_destination_region() -> None:
    workspace = _workspace()
    region_id = workspace.execute(
        "detect_region", query="basket opening"
    ).result["region_id"]
    point_id = workspace.execute(
        "locate_point",
        query="center of basket opening",
        within_region_id=region_id,
    ).result["point_id"]
    workspace.execute(
        "preview_pose",
        point_id=point_id,
        offset_xyz_m=[0.0, 0.0, 0.18],
    )

    artifacts = workspace._private.action_artifacts
    assert _presentation_region_ref(workspace.state, artifacts) == region_id

    geometry = workspace._private.region_geometry[region_id]
    target = workspace.state.action_proposal.target.pose.position_xyz
    center = np.asarray(
        _contact_camera_center(target, geometry.filtered_object_points_base)
    )
    assert np.isfinite(center).all()
    assert not np.allclose(center, np.asarray(target))


def test_point_pose_commit_replaces_stale_grasp_points_with_destination_geometry() -> None:
    workspace = _workspace()
    stale_points = np.array([[0.1, -0.2, 0.04], [0.12, -0.18, 0.08]])
    workspace._private.last_physical_artifacts = LastPhysicalArtifacts(
        focus_pose=Pose((0.1, -0.2, 0.1), (0.0, 0.0, 0.0, 1.0)),
        subject_query="can",
        subject_points_base=stale_points,
    )
    region_id = workspace.execute(
        "detect_region", query="basket"
    ).result["region_id"]
    destination_points = workspace._private.region_geometry[
        region_id
    ].filtered_object_points_base.copy()
    point_id = workspace.execute(
        "locate_point",
        query="basket opening center",
        within_region_id=region_id,
    ).result["point_id"]
    action_id = workspace.execute(
        "preview_pose",
        point_id=point_id,
        offset_xyz_m=[0.0, 0.0, 0.18],
    ).result["action_id"]
    workspace.begin_imagination("align with the visible opening", action_id)
    workspace.execute_imagination("finish_imagination", status="ready")

    result = workspace.execute("execute_action", action_id=action_id)

    assert result.ok
    physical = workspace._private.last_physical_artifacts
    assert physical is not None
    assert physical.subject_query == "basket"
    assert np.array_equal(physical.subject_points_base, destination_points)


def test_current_only_contact_preview_does_not_count_as_planned_gripper() -> None:
    workspace = _workspace()
    region = workspace.execute("detect_region", query="can").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    action_id = workspace.execute("preview_grasp", seed_id=seed).result["action_id"]
    workspace.execute("execute_action", action_id=action_id)

    from vaw.context_runtime.packet import _current_physical_contact_preview

    preview = _current_physical_contact_preview(workspace)
    assert preview is not None
    assert preview.gripper_opening is None
    assert _preview_contact_points(workspace.state.robot, preview) is None
    live = _current_gripper_contact_points(workspace.state.robot)
    assert live is not None and len(live) > 0
    required = _required_contact_points(workspace.state.robot, preview)
    assert required is not None
    assert required[:, 2].max() >= live[:, 2].max() - 1e-9


def test_post_commit_contact_framing_requires_the_live_gripper_above_the_basket() -> None:
    calls: list[object] = []

    def provider(request):
        calls.append(request)
        camera = {
            "images": {"rgb": np.zeros((358, 832, 3), dtype=np.uint8)},
            "intrinsics": np.array(
                [[360.0, 0.0, 456.0], [0.0, 360.0, 146.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            "pose_mat": np.eye(4, dtype=np.float64),
            "view_name": "front",
        }
        return ContactCameraPair(
            front=camera,
            side={**camera, "view_name": "side"},
            selection=ContactCameraSelection(1, 1, 1.0, 1.0, 1.0),
        )

    workspace = ContextWorkspace(
        FakeContextApi(),
        "place the can in the basket",
        motion_backend="pyroki",
        contact_camera_provider=provider,
    )
    region_id = workspace.execute(
        "detect_region", query="basket"
    ).result["region_id"]
    basket = workspace._private.region_geometry[region_id].filtered_object_points_base
    point_id = workspace.execute(
        "locate_point",
        query="basket opening center",
        within_region_id=region_id,
    ).result["point_id"]
    action_id = workspace.execute(
        "preview_pose",
        point_id=point_id,
        offset_xyz_m=[0.0, 0.0, 0.18],
    ).result["action_id"]
    workspace.execute("execute_action", action_id=action_id)

    ContextCompiler().compile(workspace)

    assert calls
    required = np.asarray(calls[-1].required_points_base, dtype=np.float64)
    assert required.ndim == 2 and required.shape[1] == 3
    assert required[:, 2].max() > float(np.quantile(basket[:, 2], 0.98)) + 0.04
    span = float(np.linalg.norm(required.max(axis=0) - required.min(axis=0)))
    assert span <= 0.16 + 1e-6


def test_packet_modes_are_result_driven_without_owner_state() -> None:
    workspace = _workspace()
    compiler = ContextCompiler()
    assert compiler.compile(workspace).decision.mode == "idle"

    region = workspace.execute("detect_region", query="can").result["region_id"]
    assert compiler.compile(workspace).decision.mode == "grounding"
    seeds = workspace.execute("propose_grasps", region_id=region).result["seed_ids"]
    packet = compiler.compile(workspace)
    assert packet.decision.mode == "seeds"
    assert set(seeds).issubset(packet.decision.seed_ids)

    workspace.execute("preview_grasp", seed_id=seeds[0])
    assert compiler.compile(workspace).decision.mode == "proposal"
    assert not hasattr(workspace.state, "owner")
    assert not hasattr(workspace.state, "action_review")


def test_packet_is_deterministic_and_does_not_leak_private_state() -> None:
    workspace = _workspace()
    workspace.execute("detect_region", query="can")
    compiler = ContextCompiler()
    first = compiler.compile(workspace)
    second = compiler.compile(workspace)

    assert first.summary() == second.summary()
    for raster_id in first.rasters:
        assert np.array_equal(first.rasters[raster_id], second.rasters[raster_id])
    snapshot = first.web_snapshot(render_id="fixture")
    assert snapshot["schemaVersion"] == CONTEXT_WEB_SCHEMA_VERSION
    assert snapshot["projection"] == "main"
    serialized = json.dumps(snapshot, ensure_ascii=False).lower()
    forbidden = {
        "depth",
        "intrinsics",
        "posemat",
        "rawmask",
        "pointcloud",
        "privileged",
        "reward",
        "success",
    }
    assert forbidden.isdisjoint(set(_walk_keys(snapshot)))
    assert all(word not in serialized for word in ("receipt_id", "context_schema"))


def test_context_compiler_preview_gripper_style_is_explicit_and_validated() -> None:
    assert ContextCompiler().preview_gripper_style == "fk-mesh"
    assert (
        ContextCompiler(preview_gripper_style="semantic-wireframe").preview_gripper_style
        == "semantic-wireframe"
    )
    with pytest.raises(ValueError, match="unsupported preview gripper style"):
        ContextCompiler(preview_gripper_style="unknown")


def test_commit_refreshes_revision_and_clears_action_proposal() -> None:
    workspace = _workspace()
    compiler = ContextCompiler()
    action_id = _selected_action(workspace)
    workspace.begin_imagination("向上 1cm", action_id)
    workspace.execute_imagination(
        "move_tcp_delta", delta_xyz_m=[0.0, 0.0, 0.01], frame="base"
    )
    action_id = workspace.execute_imagination(
        "finish_imagination", status="ready"
    ).result["action_id"]
    before = workspace.state.observation_revision

    result = workspace.execute("execute_action", action_id=action_id)
    packet = compiler.compile(workspace)

    assert result.ok
    assert workspace.state.observation_revision == before + 1
    assert workspace.state.action_proposal is None
    assert packet.manifest()["action_proposal"] is None
    assert packet.decision.mode == "contact"
    assert workspace.state.last_physical_action is not None
    assert workspace._private.last_physical_artifacts is not None
    assert packet.world.contact_front_raster_id == "contact_front"
    assert packet.world.contact_side_raster_id == "contact_side"

    # Perception may add a small evidence inset, but it must not erase current
    # post-physical contact continuity.
    workspace.execute("detect_region", query="can")
    grounded = compiler.compile(workspace)
    assert grounded.decision.mode == "contact"
    assert grounded.decision.primary_raster_id is not None
    assert grounded.decision.primary_raster_id in grounded.rasters

    point_id = workspace.execute("locate_point", query="basket opening").result["point_id"]
    located = compiler.compile(workspace)
    assert located.decision.mode == "contact"
    assert located.decision.primary_raster_id == f"point:{point_id}"
    assert point_id in located.decision.point_ids
    assert located.decision.primary_raster_id in located.rasters


def test_locate_point_keeps_evidence_when_a_proposal_is_already_open() -> None:
    workspace = _workspace()
    compiler = ContextCompiler()
    _selected_action(workspace)
    point_id = workspace.execute("locate_point", query="basket opening").result["point_id"]
    packet = compiler.compile(workspace)

    assert packet.decision.mode == "proposal"
    assert packet.decision.primary_raster_id == f"point:{point_id}"
    assert point_id in packet.decision.point_ids
    assert packet.decision.primary_raster_id in packet.rasters
