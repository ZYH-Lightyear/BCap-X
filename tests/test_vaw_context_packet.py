from __future__ import annotations

import json

import numpy as np

from tests.test_vaw_context_runtime import FakeContextApi
from vaw.context_runtime.packet import (
    CONTEXT_HEIGHT,
    CONTEXT_WIDTH,
    ContextCompiler,
    _observed_source_ref,
)
from vaw.context_runtime.near_field import gravity_stable_contact_frame_quaternion
from vaw.context_runtime.presentation import compile_near_field_preview
from vaw.context_runtime.model import LastPhysicalAction
from vaw.context_runtime.contact_camera import ContactCameraPair
from vaw.context_runtime.workspace import ContextWorkspace


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).replace("_", "").lower()
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _workspace() -> ContextWorkspace:
    api = FakeContextApi()
    api.rgb[:, :, 0] = np.arange(api.rgb.shape[1], dtype=np.uint8)
    return ContextWorkspace(api, "place the can in the basket", motion_backend="pyroki")


def _direct_camera(color: tuple[int, int, int]) -> dict:
    rgb = np.full((358, 832, 3), color, dtype=np.uint8)
    return {
        "images": {
            "rgb": rgb,
            "depth": np.full((358, 832), 0.3, dtype=np.float64),
        },
        "intrinsics": np.array(
            [[500.0, 0.0, 416.0], [0.0, 500.0, 179.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "pose_mat": np.eye(4, dtype=np.float64),
    }


def test_agentview_is_the_clean_current_rgb() -> None:
    workspace = _workspace()
    packet = ContextCompiler().compile(workspace)

    assert np.array_equal(packet.rasters["agentview"], workspace._private.camera("agentview")["images"]["rgb"])


def test_packet_modes_follow_owner_and_evidence_not_history() -> None:
    workspace = _workspace()
    compiler = ContextCompiler()
    assert compiler.compile(workspace).decision.mode == "idle"

    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    assert compiler.compile(workspace).decision.mode == "grounding"

    seed_id = workspace.execute("propose_grasps", region_id=region).result["seed_ids"][0]
    packet = compiler.compile(workspace)
    assert packet.decision.mode == "seeds"
    assert packet.decision.seed_ids == (seed_id,)

    workspace.set_refinement_goal("check grasp geometry")
    workspace.execute("select", seed_id=seed_id)
    packet = compiler.compile(workspace)
    assert packet.decision.mode == "editing"
    assert packet.world.owner == "imagination"
    assert packet.world.action["status"] == "editing"
    assert packet.world.action["target_role"] == "grasp_contact"
    assert "source_surface_delta_base_m" not in packet.world.action
    assert packet.world.imagination_scene_raster_id == "imagination_scene"

    action_id = workspace.execute("finish_imagination", status="ready").result[
        "action_id"
    ]
    packet = compiler.compile(workspace)
    assert packet.decision.mode == "reviewed"
    assert packet.world.action["action_id"] == action_id
    assert packet.world.action["intent"] == "check grasp geometry"
    assert packet.manifest()["review_action_id"] == action_id
    assert packet.world.action["status"] == "review"
    assert "handoff_reason" not in packet.world.action


def test_action_review_preserves_cumulative_imagination_edit() -> None:
    workspace = _workspace()
    compiler = ContextCompiler()
    workspace.set_refinement_goal("沿 base +Z 抬升 TCP")
    workspace.execute(
        "delta_move", delta_xyz_m=[0.0, 0.0, 0.03], frame="base"
    )
    workspace.execute(
        "delta_move", delta_xyz_m=[0.0, 0.0, -0.01], frame="base"
    )

    action_id = workspace.execute("finish_imagination", status="ready").result[
        "action_id"
    ]
    packet = compiler.compile(workspace)

    assert packet.world.action["action_id"] == action_id
    summary = packet.world.action["edit_summary"]
    assert summary["total_translation_base_m"] == [0.0, 0.0, 0.02]
    assert summary["previous_edit"]["delta_xyz_m"] == [0.0, 0.0, 0.03]
    assert summary["last_edit"]["delta_xyz_m"] == [0.0, 0.0, -0.01]
    assert "latest_edit" not in packet.world.action


def test_contact_camera_is_locked_to_imagination_start_pose() -> None:
    workspace = _workspace()
    compiler = ContextCompiler()
    workspace.set_refinement_goal("rotate without rotating the observed world")
    workspace.execute("start_imagination")

    before = compiler.compile(workspace).rasters["contact_focus"].copy()
    workspace.execute("rotate", axis="z", angle_deg=35.0, frame="base")
    preview = workspace._private.imagination_artifacts
    assert preview is not None and preview.initial_target is not None
    after = compiler.compile(workspace).rasters["contact_focus"]

    # The full raster changes because the purple target and rotate annotation
    # change, but the presenter derives the camera from the immutable initial
    # target rather than the edited endpoint.
    assert not np.array_equal(before, after)
    initial_pose = preview.initial_target.pose
    current_pose = workspace.state.imagination.target.pose
    assert initial_pose is not None and current_pose is not None
    initial_frame = gravity_stable_contact_frame_quaternion(initial_pose)
    current_frame = gravity_stable_contact_frame_quaternion(current_pose)
    assert not np.allclose(initial_frame, current_frame)
    compiled = compile_near_field_preview(
        workspace.state,
        workspace.state.imagination.target,
        preview,
    )
    assert compiled is not None
    assert np.allclose(compiled.contact_frame_quaternion_xyzw, initial_frame)


def test_compiler_uses_private_direct_contact_camera_only_for_active_target() -> None:
    requests = []

    def provider(request):
        requests.append(request)
        return ContactCameraPair(
            front=_direct_camera((90, 100, 110)),
            side=_direct_camera((130, 140, 150)),
        )

    api = FakeContextApi()
    workspace = ContextWorkspace(
        api,
        "task",
        motion_backend="pyroki",
        contact_camera_provider=provider,
    )
    compiler = ContextCompiler()
    compiler.compile(workspace)
    assert requests == []

    workspace.set_refinement_goal("check local geometry")
    workspace.execute("start_imagination")
    packet = compiler.compile(workspace)

    assert len(requests) == 1
    assert requests[0].width == 832
    assert requests[0].panel_height == 358
    assert packet.rasters["contact_focus"].shape == (720, 832, 3)
    assert "depth" not in json.dumps(packet.summary()).lower()


def test_main_gripper_review_is_pose_free_and_visualized_as_unexecuted() -> None:
    workspace = _workspace()
    action_id = workspace.execute("close_gripper").result["action_id"]

    packet = ContextCompiler().compile(workspace)

    assert packet.world.owner == "main"
    assert packet.decision.mode == "reviewed"
    assert packet.manifest()["review_action_id"] == action_id
    assert packet.world.action == {
        "status": "review",
        "target": {"gripper": "closed"},
        "target_role": "gripper_only",
        "action_id": action_id,
        "intent": "set gripper closed",
        "edit_summary": {"target_gripper": "closed"},
    }
    assert packet.world.contact_focus_raster_id == "contact_focus"
    assert packet.rasters["contact_focus"].shape == (720, 832, 3)


def test_failed_imagination_returns_to_visible_seed_catalog() -> None:
    workspace = _workspace()
    compiler = ContextCompiler()
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed_id = workspace.execute("propose_grasps", region_id=region).result[
        "seed_ids"
    ][0]
    workspace.set_refinement_goal("reject an unsuitable seed")
    workspace.execute("select", seed_id=seed_id)

    result = workspace.execute("finish_imagination", status="failed")
    packet = compiler.compile(workspace)

    assert result.result == {"status": "failed", "source_ref": seed_id}
    assert workspace.state.last_handoff is not None
    assert workspace.state.last_handoff.source_ref == seed_id
    assert packet.world.owner == "main"
    assert packet.world.action is None
    assert packet.decision.mode == "seeds"
    assert packet.decision.seed_ids == (seed_id,)


def test_non_executable_limit_reports_rejected_source_to_main() -> None:
    workspace = _workspace()
    region = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed_id = workspace.execute("propose_grasps", region_id=region).result[
        "seed_ids"
    ][0]
    workspace.set_refinement_goal("review this seed")
    workspace.execute("select", seed_id=seed_id)
    artifacts = workspace._private.imagination_artifacts
    assert artifacts is not None
    workspace._private.imagination_artifacts = type(artifacts)(
        planning_context=artifacts.planning_context,
        preview_plan=None,
        initial_target=artifacts.initial_target,
    )

    result = workspace.limit_imagination()

    assert result.result == {"status": "failed", "source_ref": seed_id}


def test_turn_limit_is_presented_as_failed_handoff_without_action() -> None:
    workspace = _workspace()
    workspace.execute("start_imagination")
    result = workspace.limit_imagination()

    packet = ContextCompiler().compile(workspace)

    assert result.result == {"status": "failed"}
    assert packet.world.owner == "main"
    assert packet.world.action is None
    assert "turn_limit" not in json.dumps(packet.summary())
    assert packet.decision.action_id is None
    assert packet.decision.mode != "reviewed"


def test_new_grounding_evidence_supersedes_but_preserves_action_review() -> None:
    workspace = _workspace()
    compiler = ContextCompiler()
    workspace.execute("delta_move", delta_xyz_m=[0.01, 0.0, 0.0], frame="base")
    action_id = workspace.execute("finish_imagination", status="ready").result[
        "action_id"
    ]
    assert compiler.compile(workspace).decision.mode == "reviewed"

    region_id = workspace.execute("detection_and_sam", query="basket").result["region_id"]
    packet = compiler.compile(workspace)

    assert packet.decision.mode == "grounding"
    assert packet.decision.region_ids == (region_id,)
    assert packet.manifest()["review_action_id"] == action_id
    assert packet.world.action["action_id"] == action_id


def test_grasp_preview_focuses_observed_region_instead_of_virtual_seed() -> None:
    workspace = _workspace()
    region_id = workspace.execute("detection_and_sam", query="can").result["region_id"]
    seed_id = workspace.execute("propose_grasps", region_id=region_id).result[
        "seed_ids"
    ][0]
    workspace.set_refinement_goal("review source alignment")
    workspace.execute("select", seed_id=seed_id)

    assert _observed_source_ref(workspace._private.imagination_artifacts) == region_id


def test_observed_views_stay_current_but_imagination_scene_gains_preview() -> None:
    workspace = _workspace()
    compiler = ContextCompiler()
    observed = compiler.compile(workspace)
    workspace.execute("delta_move", delta_xyz_m=[0.0, 0.0, -0.02], frame="base")
    editing = compiler.compile(workspace)

    assert np.array_equal(observed.rasters["agentview"], editing.rasters["agentview"])
    assert np.array_equal(
        observed.rasters["observed_scene"], editing.rasters["observed_scene"]
    )
    assert not np.array_equal(
        observed.rasters["imagination_scene"],
        editing.rasters["imagination_scene"],
    )
    assert observed.rasters["observed_scene"].shape == (570, 960, 3)
    assert editing.rasters["imagination_scene"].shape == (720, 1200, 3)
    assert "contact_focus" not in observed.rasters
    assert editing.world.contact_focus_raster_id == "contact_focus"
    assert editing.rasters["contact_focus"].shape == (720, 832, 3)


def test_packet_is_deterministic_and_does_not_leak_private_state() -> None:
    workspace = _workspace()
    workspace.execute("delta_move", delta_xyz_m=[0.01, 0.0, 0.0], frame="base")
    first = ContextCompiler().compile(workspace)
    second = ContextCompiler().compile(workspace)
    assert first.summary() == second.summary()
    assert all(np.array_equal(first.rasters[key], second.rasters[key]) for key in first.rasters)

    snapshot = first.web_snapshot(render_id="privacy")
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
        "trajectoryrad",
        "planningcontext",
        "history",
        "receipt",
    }
    assert forbidden.isdisjoint(set(_walk_keys(snapshot)))
    encoded = json.dumps(snapshot).lower()
    assert "functionrecord" not in encoded and "waypointdraft" not in encoded
    assert snapshot["schemaVersion"] == 32
    assert snapshot["schema"] == "vaw-context-v31-separated-control-guides"
    assert snapshot["viewport"] == {"width": CONTEXT_WIDTH, "height": CONTEXT_HEIGHT}


def test_commit_comparison_persists_across_nonphysical_grounding() -> None:
    workspace = _workspace()
    action_id = workspace.execute("open_gripper").result["action_id"]
    workspace.execute("commit", action_id=action_id)
    packet = ContextCompiler().compile(workspace)
    assert packet.decision.mode == "post_commit"
    assert packet.world.action is None
    assert packet.world.owner == "main"
    assert packet.world.last_physical_action is not None
    assert packet.world.last_physical_action.executed_stages == "gripper"
    assert packet.world.last_physical_action.outcome == "completed"
    assert packet.world.physical_verification is not None
    assert packet.world.physical_verification.kind == "release"
    assert packet.world.post_commit_before_raster_id == "post_commit:before"
    assert packet.world.post_commit_current_raster_id == "post_commit:current"
    assert packet.rasters["post_commit:before"].shape == (390, 760, 3)
    assert packet.rasters["post_commit:current"].shape == (390, 760, 3)

    workspace.consume_main_context()
    consumed = ContextCompiler().compile(workspace)
    assert consumed.decision.mode == "post_commit"
    assert consumed.world.last_physical_action is not None
    assert consumed.world.last_physical_action.executed_stages == "gripper"
    assert "post_commit:before" in consumed.rasters
    assert "post_commit:current" in consumed.rasters

    region_id = workspace.execute("detection_and_sam", query="basket").result[
        "region_id"
    ]
    workspace.consume_main_context()
    grounded = ContextCompiler().compile(workspace)
    assert grounded.decision.mode == "grounding"
    assert grounded.decision.region_ids == (region_id,)
    assert grounded.world.post_commit_before_raster_id == "post_commit:before"
    assert grounded.world.post_commit_current_raster_id == "post_commit:current"
    assert "post_commit:before" in grounded.rasters
    assert "post_commit:current" in grounded.rasters

    workspace.execute("delta_move", delta_xyz_m=[0.0, 0.0, 0.01], frame="base")
    editing = ContextCompiler().compile(workspace)
    assert workspace.state.last_physical_action is not None
    assert editing.world.last_physical_action is not None
    assert editing.world.last_physical_action.target_gripper == "open"
    action_id = workspace.execute("finish_imagination", status="ready").result[
        "action_id"
    ]
    assert action_id
    reviewed = ContextCompiler().compile(workspace)
    assert workspace.state.last_physical_action is not None
    assert reviewed.world.last_physical_action is not None
    assert reviewed.world.last_physical_action.target_gripper == "open"
    assert reviewed.decision.mode == "reviewed"
    assert reviewed.world.action is not None
    assert reviewed.world.action["action_id"] == action_id

    workspace.state.last_physical_action = LastPhysicalAction(
        intent="old failed move",
        executed_stages="arm",
        outcome="arm_failed",
    )
    assert ContextCompiler().compile(workspace).world.last_physical_action is None


def test_grasp_source_location_persists_for_lift_causal_verification() -> None:
    workspace = _workspace()
    region_id = workspace.execute("detection_and_sam", query="can").result[
        "region_id"
    ]
    seed_id = workspace.execute("propose_grasps", region_id=region_id).result[
        "seed_ids"
    ][0]
    workspace.set_refinement_goal("contact the can")
    workspace.execute("select", seed_id=seed_id)
    action_id = workspace.execute("finish_imagination", status="ready").result[
        "action_id"
    ]
    workspace.execute("commit", action_id=action_id)

    close_id = workspace.execute("close_gripper").result["action_id"]
    workspace.execute("commit", action_id=close_id)

    first = ContextCompiler().compile(workspace)
    assert first.world.causal_source_label == "can"
    assert first.world.causal_source_before_raster_id == "causal_source:before"
    assert first.world.causal_source_current_raster_id == "causal_source:current"
    assert first.world.physical_verification is not None
    assert first.world.physical_verification.kind == "closure"
    assert "PARTIAL GRIP OPENING" in first.world.physical_verification.ambiguity

    workspace.set_refinement_goal("lift to verify following")
    workspace.execute(
        "delta_move", delta_xyz_m=[0.0, 0.0, 0.03], frame="base"
    )
    lift_id = workspace.execute("finish_imagination", status="ready").result[
        "action_id"
    ]
    reviewed = ContextCompiler().compile(workspace)
    assert reviewed.decision.mode == "reviewed"
    workspace.execute("commit", action_id=lift_id)

    lifted = ContextCompiler().compile(workspace)
    assert lifted.decision.mode == "post_commit"
    assert lifted.world.causal_source_label == "can"
    assert lifted.world.physical_verification is not None
    assert lifted.world.physical_verification.kind == "arm_motion"
    assert "OCCLUSION" in lifted.world.physical_verification.ambiguity
    for raster_id in ("causal_source:before", "causal_source:current"):
        assert raster_id in lifted.rasters
        assert lifted.rasters[raster_id].shape == (390, 520, 3)
