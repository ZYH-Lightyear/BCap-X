"""Canonical payload specs: validation, publish-time enforcement, prompt rendering."""

from __future__ import annotations

import pytest

from robomex.authoring.artifacts import ArtifactStore, PortSpec, TypedArtifact
from robomex.core.payload_specs import validate_payload
from robomex.prompts.authoring import output_contract_for_ports


def _trajectory_payload() -> dict:
    return {
        "feasible": True,
        "waypoints": [
            {
                "name": "pregrasp",
                "phase": "pregrasp",
                "position_xyz": [0.77, 0.03, 0.14],
                "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
                "gripper": "open",
            },
            {
                "name": "close",
                "phase": "close",
                "position_xyz": [0.77, 0.03, 0.07],
                "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
                "gripper": "close",
            },
        ],
    }


def test_valid_trajectory_passes() -> None:
    assert validate_payload("robomex.trajectory.v1", _trajectory_payload()) is None


def test_trajectory_flat_fields_without_waypoints_are_rejected() -> None:
    error = validate_payload(
        "robomex.trajectory.v1",
        {"feasible": True, "grasp_position": [0.77, 0.03, 0.07]},
    )
    assert error is not None
    assert "waypoints" in error


def test_infeasible_trajectory_is_redirected_to_failure_kind() -> None:
    payload = _trajectory_payload()
    payload["feasible"] = False
    error = validate_payload("robomex.trajectory.v1", payload)
    assert error is not None
    assert "infeasible" in error


def test_trajectory_rejects_non_finite_and_unknown_phase() -> None:
    payload = _trajectory_payload()
    payload["waypoints"][0]["position_xyz"] = [float("nan"), 0.0, 0.1]
    assert "position_xyz" in validate_payload("robomex.trajectory.v1", payload)

    payload = _trajectory_payload()
    payload["waypoints"][1]["phase"] = "swoop"
    assert "phase" in validate_payload("robomex.trajectory.v1", payload)


def test_trajectory_waypoint_requires_exactly_one_target_kind() -> None:
    payload = _trajectory_payload()
    payload["waypoints"][0]["joints"] = [0.0] * 7
    error = validate_payload("robomex.trajectory.v1", payload)
    assert error is not None
    assert "exactly one" in error


def test_joint_space_waypoint_is_accepted() -> None:
    payload = {
        "feasible": True,
        "waypoints": [{"name": "home", "phase": "home", "joints": [0.0] * 7}],
    }
    assert validate_payload("robomex.trajectory.v1", payload) is None


def test_affordance_requires_finite_pose() -> None:
    valid = {
        "position": [0.7, 0.0, 0.07],
        "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
    }
    assert validate_payload("robomex.affordance.v1", valid) is None
    assert "position" in validate_payload("robomex.affordance.v1", {"position": [0.7]})


def test_execution_evidence_requires_primitive_report() -> None:
    valid = {
        "primitives": [{"name": "goto_pose", "status": "succeeded"}],
        "all_primitives_ok": True,
    }
    assert validate_payload("robomex.execution_evidence.v1", valid) is None
    assert "primitives" in validate_payload("robomex.execution_evidence.v1", {})

    bad_status = {
        "primitives": [{"name": "goto_pose", "status": "meh"}],
        "all_primitives_ok": True,
    }
    assert "status" in validate_payload("robomex.execution_evidence.v1", bad_status)


def test_unknown_schema_is_not_constrained() -> None:
    assert validate_payload("robomex.mask.v1", {"anything": 1}) is None


def test_store_publish_rejects_noncanonical_trajectory() -> None:
    store = ArtifactStore()
    artifact = TypedArtifact(
        port="trajectory",
        schema="robomex.trajectory.v1",
        producer="planner",
        payload={"feasible": True, "grasp_position": [0.7, 0.0, 0.07]},
    )
    with pytest.raises(ValueError, match="waypoints"):
        store.publish(
            "planner",
            (PortSpec("trajectory", "robomex.trajectory.v1"),),
            (artifact,),
        )
    assert store.values() == ()


def test_output_contract_renders_payload_spec() -> None:
    contract = output_contract_for_ports(
        (("trajectory", "robomex.trajectory.v1", "world"),)
    )
    assert "Canonical `robomex.trajectory.v1` payload" in contract
    assert "enforced at publish time" in contract


def test_output_contract_states_artifacts_dir_convention() -> None:
    contract = output_contract_for_ports((("plan", "test.value.v1"),))

    assert "ARTIFACTS_DIR" in contract
    assert "CWD" in contract
    assert "result_var" in contract
    assert "NODE_RESULT" in contract


def test_output_contract_mentions_verdict_only_for_verifier_ports() -> None:
    """The verifier_report boilerplate shown to every role taught a small
    model to emit an undeclared port (20260719_173232 subgoal_01)."""

    non_verifier = output_contract_for_ports((("object_geometry", "test.value.v1"),))
    verifier = output_contract_for_ports((("verifier_report", "robomex.verifier.v1"),))

    assert "verifier_report" not in non_verifier
    assert "outputs.verifier_report.payload.verdict" in verifier


def test_action_contract_prefers_result_var_and_inputs() -> None:
    from robomex.prompts.authoring import build_agent_system_prompt

    prompt = build_agent_system_prompt(
        role="plan",
        objective="plan a trajectory",
        capabilities=("geometry_compute",),
        output_contract=output_contract_for_ports((("trajectory", "robomex.trajectory.v1"),)),
    )
    assert "result_var" in prompt
    assert "INPUTS" in prompt
    assert "ungrounded pose literals" in prompt or "Never invent" in prompt
