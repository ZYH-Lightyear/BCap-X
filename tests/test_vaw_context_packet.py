from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import vaw.context_runtime.packet as packet_module
from tests.test_vaw_context_runtime import FakeContextApi
from vaw.context_runtime.model import (
    ActionAdjustment,
    ActionPrediction,
    ActionProposal,
    FunctionRecord,
    Pose,
    SpatialTargetSummary,
)
from vaw.context_runtime.packet import ContextCompiler
from vaw.context_runtime.web_renderer import ContextWebRenderer
from vaw.context_runtime.workspace import ContextWorkspace


def _workspace_with_evidence(*, select: bool = True) -> ContextWorkspace:
    api = FakeContextApi()
    api.rgb[:, :, 0] = np.arange(api.rgb.shape[1], dtype=np.uint8)
    api.bbox_outputs = [[2.0, 2.0, 10.0, 8.0]]
    api.point_outputs = [[4.0, 3.0]]
    workspace = ContextWorkspace(
        api, "place the mug in the basket", motion_backend="pyroki"
    )
    region_id = workspace.execute("inspect", query="mug").result["region_id"]
    workspace.execute("locate_point", query="mug center", within_region_id=region_id)
    candidates = workspace.execute("propose_grasps", region_id=region_id).result
    if select:
        workspace.execute("select", candidate_id=candidates["candidate_ids"][0])
    return workspace


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).replace("_", "").lower()
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_context_packet_is_deterministic_and_normalized() -> None:
    workspace = _workspace_with_evidence()
    compiler = ContextCompiler()

    first = compiler.compile(workspace)
    second = compiler.compile(workspace)

    assert first.summary() == second.summary()
    assert list(first.rasters) == list(second.rasters)
    for raster_id in first.rasters:
        assert np.array_equal(first.rasters[raster_id], second.rasters[raster_id])
        assert first.rasters[raster_id].dtype == np.uint8
        assert first.rasters[raster_id].ndim == 3
    assert first.world.active_action["action_id"] == "a1"
    assert first.world.active_action["kind"] == "grasp"
    assert first.world.active_action["source_ref"] == "g1"
    assert first.decision.mode == "proposal"
    assert first.decision.action_id == "a1"
    first_candidate = first.catalog.candidates[0]
    assert first_candidate.kind == "grasp"
    assert first_candidate.source_ref == "region1"
    # The fake solver returns arbitrary joints unrelated to its requested TCP.
    # The packet must not present that FK as an exact candidate preview.
    assert first_candidate.solve_ik == "mismatch"
    assert first_candidate.delta_from_anchor_xyz_m is not None
    assert np.isclose(np.linalg.norm(first_candidate.approach_vector_base), 1.0)
    candidate_raster = first.rasters[first_candidate.raster_id]
    assert np.any(np.all(candidate_raster == np.array([124, 58, 237]), axis=-1))


def test_persistent_world_replaces_raw_wrist_with_near_field_geometry() -> None:
    workspace = _workspace_with_evidence()

    packet = ContextCompiler().compile(workspace)

    assert np.array_equal(packet.rasters["agentview"], workspace.api.rgb)
    assert "wrist" not in packet.rasters
    assert packet.world.near_field_raster_id == "near_field"
    assert packet.rasters["near_field"].shape == (540, 480, 3)


def test_imagined_gripper_is_more_prominent_than_the_arm() -> None:
    rgb = np.full((12, 16, 3), 180, dtype=np.uint8)
    arm = np.zeros((12, 16), dtype=bool)
    arm[2:10, 2:14] = True
    gripper = np.zeros_like(arm)
    gripper[5:8, 7:10] = True

    raster = packet_module._overlay_robot_imagination(
        rgb,
        robot_mask=arm,
        gripper_mask=gripper,
    )

    violet = np.array(packet_module.VIOLET, dtype=np.float64)
    arm_distance = np.linalg.norm(raster[4, 4].astype(float) - violet)
    gripper_distance = np.linalg.norm(raster[6, 8].astype(float) - violet)
    assert gripper_distance < arm_distance


def test_candidate_focus_ignores_full_arm_extent() -> None:
    rgb = np.full((100, 200, 3), 180, dtype=np.uint8)
    robot_mask = np.zeros((100, 200), dtype=bool)
    robot_mask[:, :8] = True
    gripper_mask = np.zeros_like(robot_mask)
    gripper_mask[38:63, 82:111] = True
    camera = {
        "intrinsics": np.array(
            [[100.0, 0.0, 100.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        ),
        "pose_mat": np.eye(4),
    }

    raster = packet_module._candidate_crop(
        rgb,
        (78.0, 35.0, 108.0, 66.0),
        Pose((0.0, 0.0, 2.0), (0.0, 0.0, 0.0, 1.0)),
        camera,
        robot_mask=robot_mask,
        gripper_mask=gripper_mask,
    )

    assert raster.shape[1] < rgb.shape[1] // 2


def test_candidate_fk_mismatch_is_checked_against_exact_target(monkeypatch) -> None:
    target = Pose(
        (0.4, -0.1, 0.2),
        tuple(
            Rotation.from_euler("y", 25.0, degrees=True).as_quat().tolist()
        ),
    )
    offset = np.array([0.0, 0.0, -0.1168], dtype=np.float64)
    rotation = Rotation.from_quat(target.quaternion_xyzw).as_matrix()
    expected = np.eye(4, dtype=np.float64)
    expected[:3, :3] = rotation
    expected[:3, 3] = np.asarray(target.position_xyz) + rotation @ offset

    class FakeFK:
        def __init__(self, frame):
            self.value = frame

        def frame(self, joints, frame_name):
            del joints, frame_name
            return self.value

    monkeypatch.setattr(
        packet_module, "load_panda_urdf_fk", lambda: FakeFK(expected)
    )
    assert packet_module._candidate_fk_matches_target(
        tuple(np.zeros(7)), target, offset
    )

    shifted = expected.copy()
    shifted[0, 3] += 0.05
    monkeypatch.setattr(
        packet_module, "load_panda_urdf_fk", lambda: FakeFK(shifted)
    )
    assert not packet_module._candidate_fk_matches_target(
        tuple(np.zeros(7)), target, offset
    )


def test_candidate_arrow_points_along_public_approach_vector() -> None:
    camera = {
        "intrinsics": np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        ),
        "pose_mat": np.eye(4),
    }
    pose = Pose(
        (0.0, 0.0, 2.0),
        tuple(Rotation.from_euler("y", 90.0, degrees=True).as_quat().tolist()),
    )

    contact_x, contact_y, hand_x, hand_y = packet_module._project_pose(
        pose, camera
    )

    assert hand_x < contact_x
    assert np.isclose(hand_y, contact_y)


def test_context_packet_snapshot_does_not_leak_private_sensor_state() -> None:
    packet = ContextCompiler().compile(_workspace_with_evidence())
    snapshot = packet.web_snapshot(render_id="privacy")
    forbidden = {
        "depth",
        "intrinsics",
        "posemat",
        "rawmask",
        "cloud",
        "envsuccess",
        "reward",
        "privileged",
        "score",
        "obb",
    }

    assert forbidden.isdisjoint(set(_walk_keys(snapshot)))
    assert snapshot["schemaVersion"] == 4
    assert snapshot["schema"] == "vaw-context-v3"
    assert snapshot["viewport"] == {"width": 1440, "height": 1080}
    assert all(
        value.startswith("data:image/png;base64,")
        for value in snapshot["rasters"].values()
    )
    json.dumps(snapshot)


def test_result_driven_decision_modes_cover_the_public_outcomes() -> None:
    compiler = ContextCompiler()
    idle = ContextWorkspace(FakeContextApi(), "task")
    assert compiler.compile(idle).decision.mode == "idle"

    grounding = ContextWorkspace(FakeContextApi(), "task")
    region_id = grounding.execute("inspect", query="mug").result["region_id"]
    assert compiler.compile(grounding).decision.mode == "grounding"

    grounding.execute("propose_grasps", region_id=region_id)
    assert compiler.compile(grounding).decision.mode == "candidates"
    grounding.execute("select", candidate_id="g1")
    assert compiler.compile(grounding).decision.mode == "proposal"

    receipt = ContextWorkspace(FakeContextApi(), "task")
    receipt.execute("open_gripper")
    assert compiler.compile(receipt).decision.mode == "receipt"

    error = ContextWorkspace(FakeContextApi(), "task")
    error.execute("select", candidate_id="g999")
    assert compiler.compile(error).decision.mode == "error"

    terminal = ContextWorkspace(FakeContextApi(), "task")
    terminal.execute("done", success=False)
    assert compiler.compile(terminal).decision.mode == "terminal"


def test_future_action_kind_uses_proposal_mode_without_function_name_branch() -> None:
    workspace = ContextWorkspace(FakeContextApi(), "move right")
    revision = workspace.state.observation_revision
    workspace.state.active_action = ActionProposal(
        action_id="a42",
        kind="delta_move",
        source_ref=None,
        source_revision=revision,
        target_pose=Pose((0.43, 0.0, 0.4), (0.0, 0.0, 0.0, 1.0)),
        prediction=ActionPrediction(
            solve_ik="returned", joint_positions_rad=tuple(np.zeros(7))
        ),
        adjustment=ActionAdjustment(
            kind="delta_move",
            frame="base",
            reference_pose=Pose((0.4, 0.0, 0.4), (0.0, 0.0, 0.0, 1.0)),
            delta_xyz_m=(0.03, 0.0, 0.0),
        ),
    )
    workspace.state.add_record(
        FunctionRecord(
            function_name="future_propose_move",
            arguments={"delta_xyz": [0.0, 0.1, 0.0]},
            result={"action_id": "a42", "solve_ik": "returned"},
            revision_before=revision,
            revision_after=revision,
            action_id="a42",
        )
    )

    packet = ContextCompiler().compile(workspace)

    assert packet.decision.mode == "proposal"
    assert packet.decision.action_id == "a42"
    assert packet.world.active_action["kind"] == "delta_move"
    assert packet.world.active_action["adjustment"]["delta_xyz_m"] == [0.03, 0.0, 0.0]
    assert "source_ref" not in packet.world.active_action
    assert packet.decision.primary_raster_id == "decision:proposal"
    assert packet.rasters["decision:proposal"].shape == (480, 960, 3)


def test_physical_revision_removes_old_evidence_and_shows_receipt() -> None:
    workspace = _workspace_with_evidence()
    old_ids = workspace.state.manifest()

    workspace.execute("open_gripper")
    packet = ContextCompiler().compile(workspace)

    assert old_ids["valid_region_ids"]
    assert packet.catalog.regions == ()
    assert packet.catalog.points == ()
    assert packet.catalog.candidates == ()
    assert packet.world.active_action is None
    assert packet.decision.mode == "receipt"
    assert packet.decision.primary_raster_id == "decision:post_action"
    assert np.array_equal(packet.rasters["decision:post_action"], workspace.api.rgb)
    assert "receipt_id" not in packet.world.last_receipt
    assert "revision_before" not in packet.world.last_receipt
    assert "revision_after" not in packet.world.last_receipt


def test_commit_target_drives_current_rgb_review_and_survives_gripper_action() -> None:
    api = FakeContextApi()
    api.rgb[:, :, 0] = np.arange(api.rgb.shape[1], dtype=np.uint8)
    api.point_outputs = [[4.0, 3.0]]
    workspace = ContextWorkspace(
        api, "move then release", motion_backend="pyroki"
    )
    point_id = workspace.execute("locate_point", query="target").result["point_id"]
    proposal = workspace.execute(
        "propose_pose", point_id=point_id, offset_xyz=[0.0, 0.0, 0.1]
    )

    workspace.execute("commit", action_id=proposal.result["action_id"])
    committed = ContextCompiler().compile(workspace)

    assert workspace.state.last_spatial_target is not None
    assert workspace.state.last_spatial_target.action_id == "a1"
    assert committed.decision.primary_raster_id == "decision:post_action"
    review = committed.rasters["decision:post_action"]
    assert review.shape[1] == api.rgb.shape[1]
    assert review.shape[0] <= api.rgb.shape[0]

    workspace.execute("open_gripper")
    released = ContextCompiler().compile(workspace)

    assert workspace.state.last_spatial_target.action_id == "a1"
    assert released.revision == 3
    assert released.decision.primary_raster_id == "decision:post_action"
    assert released.rasters["decision:post_action"].shape[1] == api.rgb.shape[1]


def test_post_action_review_falls_back_to_current_rgb_for_unprojectable_target() -> None:
    api = FakeContextApi()
    api.rgb[:, :, 1] = np.arange(api.rgb.shape[0], dtype=np.uint8)[:, None]
    workspace = ContextWorkspace(api, "review unreachable target")
    workspace.state.last_spatial_target = SpatialTargetSummary(
        action_id="a9",
        target_pose=Pose((1.0, 2.0, 2.0), (0.0, 0.0, 0.0, 1.0)),
        revision_after=workspace.state.observation_revision,
    )

    workspace.execute("open_gripper")
    packet = ContextCompiler().compile(workspace)

    assert packet.decision.mode == "receipt"
    assert packet.decision.primary_raster_id == "decision:post_action"
    assert np.array_equal(packet.rasters["decision:post_action"], api.rgb)


def test_non_proposal_function_switches_canvas_but_keeps_action_executable() -> None:
    workspace = _workspace_with_evidence()
    compiler = ContextCompiler()
    action_id = workspace.state.active_action.action_id

    proposal_packet = compiler.compile(workspace)
    workspace.execute("inspect", query="table")
    grounding_packet = compiler.compile(workspace)

    assert proposal_packet.decision.mode == "proposal"
    assert "decision:proposal" in proposal_packet.rasters
    assert grounding_packet.decision.mode == "grounding"
    assert "decision:proposal" not in grounding_packet.rasters
    assert grounding_packet.world.active_action["action_id"] == action_id
    assert grounding_packet.manifest()["active_action_id"] == action_id
    assert workspace.state.manifest()["active_action_id"] == action_id


def test_context_browser_renderer_is_fixed_and_deterministic() -> None:
    pytest.importorskip("playwright")
    packet = ContextCompiler().compile(_workspace_with_evidence())
    delta_workspace = ContextWorkspace(
        FakeContextApi(), "move right", motion_backend="pyroki"
    )
    delta_workspace.execute("delta_move", delta_xyz_m=[0.02, 0.0, 0.0])
    delta_packet = ContextCompiler().compile(delta_workspace)
    candidates_packet = ContextCompiler().compile(_workspace_with_evidence(select=False))
    receipt_workspace = _workspace_with_evidence()
    receipt_workspace.execute(
        "commit", action_id=receipt_workspace.state.active_action.action_id
    )
    receipt_packet = ContextCompiler().compile(receipt_workspace)
    try:
        renderer = ContextWebRenderer()
    except (PermissionError, RuntimeError) as exc:
        pytest.skip(f"browser runtime unavailable: {exc}")
    with renderer:
        first = renderer.render(packet)
        second = renderer.render(packet)
        dimensions = renderer._page.evaluate(
            """() => ({
                width: document.documentElement.scrollWidth,
                height: document.documentElement.scrollHeight,
                overflow: getComputedStyle(document.documentElement).overflow,
                dpr: window.devicePixelRatio,
            })"""
        )
        decision_text = renderer._page.locator(".ctx-decision").inner_text()
        compact_candidates = renderer._page.locator(".ctx-candidate--compact").count()
        selected = renderer._page.locator(".ctx-candidate--selected").count()
        renderer.render(delta_packet)
        delta_text = renderer._page.locator(".ctx-proposal-facts").inner_text()
        renderer.render(candidates_packet)
        main_box = renderer._page.locator(".ctx-main-view").bounding_box()
        near_field_box = renderer._page.locator(
            ".ctx-near-field-view"
        ).bounding_box()
        candidate_facts = renderer._page.locator(".ctx-candidate-facts").all_inner_texts()
        removed_world_cards = renderer._page.locator(
            ".ctx-task-block, .ctx-event-block"
        ).count()
        removed_chrome = renderer._page.locator(
            ".ctx-topbar, .ctx-footer, .ctx-section-head, .ctx-id-strip"
        ).count()
        proprio_font = renderer._page.locator(".ctx-robot-grid code").nth(1).evaluate(
            "element => getComputedStyle(element).fontSize"
        )
        proprio_text = renderer._page.locator(".ctx-robot-compact").inner_text()
        renderer.render(receipt_packet)
        receipt_text = renderer._page.locator(".ctx-receipt-strip").inner_text()
        receipt_page_text = renderer._page.locator("main").inner_text()
        receipt_world_box = renderer._page.locator(".ctx-world").bounding_box()
        receipt_raster = renderer._page.locator(".ctx-receipt-raster img").count()

    assert first.shape == (1080, 1440, 3)
    assert np.array_equal(first, second)
    assert dimensions == {"width": 1440, "height": 1080, "overflow": "hidden", "dpr": 1}
    assert "score hidden" not in decision_text
    assert compact_candidates == 2
    assert selected == 1
    assert "IMAGINED · NOT EXECUTED" in delta_text
    assert "LOCAL REFINEMENT · BASE FRAME" in delta_text
    assert "delta_xyz_m" in delta_text
    assert main_box is not None and main_box["height"] >= 500
    assert near_field_box is not None and near_field_box["width"] >= 450
    assert near_field_box["height"] >= 500
    assert candidate_facts and all("source" not in text.lower() for text in candidate_facts)
    assert removed_world_cards == 0
    assert removed_chrome == 0
    assert float(proprio_font.removesuffix("px")) >= 12
    assert "OPEN" not in proprio_text
    assert "CLOSED" not in proprio_text
    assert "TASK EFFECT UNVERIFIED" in receipt_text
    assert "receipt" not in receipt_text.lower()
    assert "CURRENT RGB · R" not in receipt_page_text
    assert "R1→R2" not in receipt_page_text
    assert receipt_world_box is not None and receipt_world_box["height"] >= 785
    assert receipt_raster == 1
    assert "TARGET TCP" not in delta_text
    assert renderer.name == "context-web-v3-near-field"
