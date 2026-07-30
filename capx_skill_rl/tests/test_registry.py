from __future__ import annotations

import pytest

from capx_skill_rl.tools import TOOL_NAMES, default_registry
from capx_skill_rl.tools.registry import ToolInputError

EXPECTED_TOOLS = (
    "vlm_bbox_detection",
    "vlm_point_detection",
    "sam3",
    "get_obb",
    "plan_grasp",
    "solve_ik",
    "move_to_joints",
    "open_gripper",
    "close_gripper",
    "go_home",
)


def test_registry_exposes_exactly_the_frozen_action_space() -> None:
    assert TOOL_NAMES == EXPECTED_TOOLS
    definitions = default_registry().definitions()
    assert tuple(item["function"]["name"] for item in definitions) == EXPECTED_TOOLS
    for item in definitions:
        parameters = item["function"]["parameters"]
        if "oneOf" in parameters:
            assert all(branch["additionalProperties"] is False for branch in parameters["oneOf"])
        else:
            assert parameters["additionalProperties"] is False


def test_schema_contains_only_model_supplied_arguments() -> None:
    definitions = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in default_registry().definitions()
    }
    assert set(definitions["vlm_bbox_detection"]["properties"]) == {"query"}
    assert set(definitions["vlm_point_detection"]["properties"]) == {"query"}
    assert [set(branch["properties"]) for branch in definitions["sam3"]["oneOf"]] == [
        {"text"},
        {"bbox"},
        {"point"},
    ]
    assert set(definitions["get_obb"]["properties"]) == {"mask_id"}
    assert set(definitions["plan_grasp"]["properties"]) == {"mask_id"}
    assert set(definitions["solve_ik"]["properties"]) == {
        "position",
        "quaternion",
    }
    assert set(definitions["move_to_joints"]["properties"]) == {"joints"}
    for name in ("open_gripper", "close_gripper", "go_home"):
        assert definitions[name]["properties"] == {}


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"text": "cup", "bbox": [0, 0, 2, 2]},
        {"point": [1]},
        {"bbox": [0, 0, 0, 2]},
        {"text": "cup", "prompt_type": "text"},
    ],
)
def test_sam3_requires_exactly_one_valid_prompt(arguments: dict) -> None:
    registry = default_registry()
    with pytest.raises(ToolInputError):
        registry.prepare("sam3", arguments)


def test_fixed_length_numeric_arguments_are_validated() -> None:
    registry = default_registry()
    with pytest.raises(ToolInputError):
        registry.prepare(
            "solve_ik",
            {"position": [0, 0], "quaternion": [0, 0, 0, 1]},
        )
    with pytest.raises(ToolInputError):
        registry.prepare("move_to_joints", {"joints": [0] * 8})
    with pytest.raises(ToolInputError):
        registry.prepare("open_gripper", {"force": 1})
