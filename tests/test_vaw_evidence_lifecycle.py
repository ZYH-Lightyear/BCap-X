"""Cross-action evidence lifecycle: revalidation, reuse, ledger visibility."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_vaw_context_runtime import FakeContextApi
from vaw.context_runtime import evidence as evidence_module
from vaw.context_runtime.memory import interaction_outcome
from vaw.context_runtime.packet import ContextCompiler
from vaw.context_runtime.workspace import ContextWorkspace

# Captured at import time, before the autouse fixture patches the module, so
# the silhouette unit test can still exercise the real implementation.
_REAL_ROBOT_SILHOUETTE = evidence_module.robot_silhouette


@pytest.fixture(autouse=True)
def _no_robot_silhouette(monkeypatch: pytest.MonkeyPatch):
    """Keep verdicts deterministic: the fake camera has no meaningful FK view."""

    monkeypatch.setattr(evidence_module, "robot_silhouette", lambda camera, robot: None)


def _workspace(api: FakeContextApi | None = None) -> ContextWorkspace:
    return ContextWorkspace(api or FakeContextApi(), "task", motion_backend="pyroki")


def test_evidence_survives_a_physical_action_when_the_scene_is_unchanged() -> None:
    api = FakeContextApi()
    workspace = _workspace(api)
    region_id = workspace.execute("detect_region", query="basket").result["region_id"]
    point_id = workspace.execute(
        "locate_point", query="basket opening", within_region_id=region_id
    ).result["point_id"]
    revision = workspace.state.observation_revision

    result = workspace.execute("close_gripper")

    assert result.ok
    assert workspace.state.observation_revision == revision + 1
    region = workspace.state.regions[region_id]
    point = workspace.state.points[point_id]
    assert region.status == "verified"
    assert point.status == "verified"
    assert region.source_revision == workspace.state.observation_revision
    changes = result.result["world_changes"]
    assert set(changes["verified"]) == {region_id, point_id}
    assert "removed" not in changes
    # The surviving references stay silent in Live References: no status tag.
    manifest = workspace.state.manifest()
    assert {"id": region_id, "query": "basket"} in manifest["regions"]


def test_verified_grounding_is_reused_instead_of_regrounded() -> None:
    api = FakeContextApi()
    workspace = _workspace(api)
    region_id = workspace.execute("detect_region", query="basket").result["region_id"]
    point_id = workspace.execute(
        "locate_point", query="basket opening", within_region_id=region_id
    ).result["point_id"]
    workspace.execute("close_gripper")
    grounding_calls = len(api.bbox_queries)

    located = workspace.execute(
        "locate_point", query="basket opening", within_region_id=region_id
    )
    detected = workspace.execute("detect_region", query="basket")

    assert located.result["point_id"] == point_id
    assert located.result["reused"] is True
    assert detected.result["region_id"] == region_id
    assert len(api.bbox_queries) == grounding_calls


def test_moved_object_is_dropped_and_its_points_cascade() -> None:
    api = FakeContextApi()
    workspace = _workspace(api)
    region_id = workspace.execute("detect_region", query="can").result["region_id"]
    point_id = workspace.execute(
        "locate_point", query="can top", within_region_id=region_id
    ).result["point_id"]

    # The archived surface disappears: current depth now sees the background
    # far behind it, which is positive evidence the object left.
    api.depth = api.depth + 0.3
    result = workspace.execute("close_gripper")

    assert result.ok
    assert region_id not in workspace.state.regions
    assert point_id not in workspace.state.points
    assert region_id not in workspace._private.region_masks
    assert region_id not in workspace._private.evidence_archives
    removed = result.result["world_changes"]["removed"]
    assert f"{region_id}(can)" in removed
    report = workspace.last_evidence_report
    reasons = {item["id"]: item["reason"] for item in report["removed"]}
    assert reasons[point_id] == "parent_region_changed"


def test_occluded_evidence_is_kept_flagged_and_not_reused() -> None:
    api = FakeContextApi()
    workspace = _workspace(api)
    region_id = workspace.execute("detect_region", query="basket").result["region_id"]
    point_id = workspace.execute("locate_point", query="basket opening").result["point_id"]

    # A closer surface moved in front of the archived one: presence is
    # unprovable either way, so the evidence is kept but flagged.
    api.depth = api.depth - 0.3
    result = workspace.execute("close_gripper")

    assert result.ok
    region = workspace.state.regions[region_id]
    point = workspace.state.points[point_id]
    assert region.status == "occluded"
    assert point.status == "occluded"
    changes = result.result["world_changes"]
    assert set(changes["occluded"]) == {region_id, point_id}
    manifest = workspace.state.manifest()
    assert {"id": region_id, "query": "basket", "status": "occluded"} in manifest["regions"]
    packet_manifest = ContextCompiler().compile(workspace).manifest()
    assert {"id": region_id, "query": "basket", "status": "occluded"} in packet_manifest[
        "regions"
    ]

    # An occluded point is not silently reused; the caller gets a fresh fix.
    located = workspace.execute("locate_point", query="basket opening")
    assert located.result["point_id"] != point_id
    assert "reused" not in located.result


def test_occluded_evidence_recovers_once_the_view_clears() -> None:
    api = FakeContextApi()
    workspace = _workspace(api)
    region_id = workspace.execute("detect_region", query="basket").result["region_id"]

    original_depth = api.depth.copy()
    api.depth = api.depth - 0.3
    workspace.execute("close_gripper")
    assert workspace.state.regions[region_id].status == "occluded"

    # The occluder moves away; the original archive matches again.
    api.depth = original_depth
    result = workspace.execute("open_gripper")

    assert workspace.state.regions[region_id].status == "verified"
    assert region_id in result.result["world_changes"]["verified"]


def test_commit_revalidates_grasp_source_instead_of_force_dropping_it() -> None:
    api = FakeContextApi()
    workspace = _workspace(api)
    grasp_region = workspace.execute("detect_region", query="can").result["region_id"]
    bystander = workspace.execute("detect_region", query="basket").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=grasp_region).result["seed_ids"][0]
    action_id = workspace.execute("preview_grasp", seed_id=seed).result["action_id"]

    result = workspace.execute("execute_action", action_id=action_id)

    assert result.ok
    # Approach is not contact: an unchanged scene keeps both the grasp
    # source and the bystander after the pixel vote.
    assert workspace.state.regions[grasp_region].status == "verified"
    assert workspace.state.regions[bystander].status == "verified"
    changes = result.result["world_changes"]
    assert "removed" not in changes
    assert set(changes["verified"]) == {grasp_region, bystander}


def test_commit_drops_grasp_source_only_when_the_surface_actually_changed() -> None:
    api = FakeContextApi()
    workspace = _workspace(api)
    grasp_region = workspace.execute("detect_region", query="can").result["region_id"]
    seed = workspace.execute("propose_grasps", region_id=grasp_region).result["seed_ids"][0]
    action_id = workspace.execute("preview_grasp", seed_id=seed).result["action_id"]

    api.depth = api.depth + 0.3
    result = workspace.execute("execute_action", action_id=action_id)

    assert result.ok
    assert grasp_region not in workspace.state.regions
    assert f"{grasp_region}(can)" in result.result["world_changes"]["removed"]
    report = workspace.last_evidence_report
    reasons = {item["id"]: item["reason"] for item in report["removed"]}
    assert reasons[grasp_region] == "changed"


def test_repeated_gripper_closures_surface_an_advisory() -> None:
    workspace = _workspace()

    first = workspace.execute("close_gripper")
    workspace.execute("open_gripper")
    second = workspace.execute("close_gripper")

    assert "advisory" not in first.result
    assert workspace.state.gripper_close_count == 2
    assert "close_gripper attempt #2" in second.result["advisory"]

    outcome = interaction_outcome(second.result, physical_outcome="completed")
    assert outcome.startswith("completed")
    assert "close_gripper attempt #2" in outcome


def test_repeated_descent_does_not_inject_a_strategy_advisory() -> None:
    workspace = _workspace()

    first = workspace.execute("move_tcp_delta", delta_xyz_m=[0.0, 0.0, -0.01], frame="base")
    second = workspace.execute("move_tcp_delta", delta_xyz_m=[0.0, 0.0, -0.01], frame="base")
    third = workspace.execute("move_tcp_delta", delta_xyz_m=[0.0, 0.0, -0.01], frame="base")

    assert first.ok and second.ok and third.ok
    assert "advisory" not in first.result
    assert "advisory" not in second.result
    assert "advisory" not in third.result
    assert not hasattr(workspace.state, "descend_streak")

    outcome = interaction_outcome(third.result, physical_outcome="completed")
    assert outcome == "completed"
    assert "descending delta_move" not in outcome


def test_force_refresh_bypasses_verified_point_reuse() -> None:
    api = FakeContextApi()
    workspace = _workspace(api)
    point_id = workspace.execute("locate_point", query="basket opening").result["point_id"]
    workspace.execute("close_gripper")
    assert workspace.state.points[point_id].status == "verified"

    reused = workspace.execute("locate_point", query="basket opening")
    assert reused.result["point_id"] == point_id
    assert reused.result["reused"] is True

    refreshed = workspace.execute(
        "locate_point", query="basket opening", force_refresh=True
    )
    assert refreshed.ok
    assert refreshed.result["point_id"] != point_id
    assert "reused" not in refreshed.result


def test_interaction_outcome_does_not_promote_evidence_diffs_to_memory() -> None:
    outcome = interaction_outcome(
        {
            "position_error_m": 0.002,
            "world_changes": {
                "removed": ["region1(can)"],
                "occluded": ["region2"],
                "verified": ["p1"],
            },
        },
        physical_outcome="completed",
    )

    assert outcome == "completed"
    assert "region1" not in outcome
    assert interaction_outcome({"point_id": "p1", "reused": True}) == "ok"


def test_persisted_point_remains_a_valid_proposal_anchor() -> None:
    workspace = _workspace()
    point_id = workspace.execute("locate_point", query="basket opening").result["point_id"]
    workspace.execute("close_gripper")

    proposal = workspace.execute(
        "preview_pose", point_id=point_id, offset_xyz_m=[0.0, 0.0, 0.1]
    )

    assert proposal.ok
    assert workspace.state.action_proposal is not None


def test_silhouette_helper_declines_gracefully_without_fk_view() -> None:
    api = FakeContextApi()
    workspace = ContextWorkspace(api, "task", motion_backend="pyroki")
    camera = workspace._private.camera("agentview")

    mask = _REAL_ROBOT_SILHOUETTE(camera, workspace.state.robot)

    assert mask is None or (
        mask.dtype == bool and mask.shape == api.rgb.shape[:2]
    )
