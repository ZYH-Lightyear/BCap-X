"""Public tool schemas and handlers."""

from __future__ import annotations

from capx_skill_rl.tools import control, perception
from capx_skill_rl.tools.registry import (
    ToolRegistry,
    ToolSpec,
    object_schema,
    validate_empty,
    validate_joints,
    validate_mask_id,
    validate_pose,
    validate_query,
    validate_sam3,
)


def _number_array(length: int) -> dict[str, object]:
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": length,
        "maxItems": length,
    }


def _sam3_schema() -> dict[str, object]:
    return {
        "type": "object",
        "oneOf": [
            object_schema(
                {"text": {"type": "string"}},
                required=("text",),
            ),
            object_schema(
                {"bbox": _number_array(4)},
                required=("bbox",),
            ),
            object_schema(
                {"point": _number_array(2)},
                required=("point",),
            ),
        ],
    }


def default_registry() -> ToolRegistry:
    specs = [
        ToolSpec(
            name="vlm_bbox_detection",
            description="Locate one queried target in the current RGB image.",
            parameters=object_schema(
                {"query": {"type": "string"}},
                required=("query",),
            ),
            validate=validate_query,
            handler=perception.vlm_bbox_detection,
        ),
        ToolSpec(
            name="vlm_point_detection",
            description="Locate one queried target point in the current RGB image.",
            parameters=object_schema(
                {"query": {"type": "string"}},
                required=("query",),
            ),
            validate=validate_query,
            handler=perception.vlm_point_detection,
        ),
        ToolSpec(
            name="sam3",
            description="Segment the current RGB using exactly one text, bbox, or point prompt.",
            parameters=_sam3_schema(),
            validate=validate_sam3,
            handler=perception.sam3,
        ),
        ToolSpec(
            name="get_obb",
            description="Estimate a mask's oriented bounding box in robot-base coordinates.",
            parameters=object_schema(
                {"mask_id": {"type": "string"}},
                required=("mask_id",),
            ),
            validate=validate_mask_id,
            handler=perception.get_obb,
        ),
        ToolSpec(
            name="plan_grasp",
            description="Return the best grasp pose for a mask in robot-base coordinates.",
            parameters=object_schema(
                {"mask_id": {"type": "string"}},
                required=("mask_id",),
            ),
            validate=validate_mask_id,
            handler=perception.plan_grasp,
        ),
        ToolSpec(
            name="solve_ik",
            description="Solve IK for a robot-base pose with an XYZW quaternion.",
            parameters=object_schema(
                {
                    "position": _number_array(3),
                    "quaternion": _number_array(4),
                },
                required=("position", "quaternion"),
            ),
            validate=validate_pose,
            handler=control.solve_ik,
        ),
        ToolSpec(
            name="move_to_joints",
            description="Move the seven arm joints to the requested configuration.",
            parameters=object_schema(
                {"joints": _number_array(7)},
                required=("joints",),
            ),
            validate=validate_joints,
            handler=control.move_to_joints,
            physical=True,
        ),
        ToolSpec(
            name="open_gripper",
            description="Open the gripper fully.",
            parameters=object_schema({}),
            validate=validate_empty,
            handler=control.open_gripper,
            physical=True,
        ),
        ToolSpec(
            name="close_gripper",
            description="Close the gripper fully.",
            parameters=object_schema({}),
            validate=validate_empty,
            handler=control.close_gripper,
            physical=True,
        ),
        ToolSpec(
            name="go_home",
            description="Return to the episode's home joint configuration.",
            parameters=object_schema({}),
            validate=validate_empty,
            handler=control.go_home,
            physical=True,
        ),
    ]
    return ToolRegistry(specs)


TOOL_NAMES = default_registry().names

__all__ = ["TOOL_NAMES", "ToolRegistry", "default_registry"]
