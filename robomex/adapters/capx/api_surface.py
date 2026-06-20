"""The CapX API surface RoboMEx skills are written against.

Execution target is the L4 API (``FrankaLiberoApiReducedSkillLibrary``), the
same level the CaP-Agent0 baseline uses. The static set below mirrors its
``functions()`` so skill ``apis:`` contracts can be validated without
importing CapX (which pulls in perception model dependencies).
"""

from __future__ import annotations

from typing import Any

# FrankaLiberoApiReduced.functions() (L3)
_L3_FUNCTIONS = frozenset({
    "get_observation",
    "segment_sam3_text_prompt",
    "segment_sam3_point_prompt",
    "point_prompt_molmo",
    "plan_grasp",
    "plan_grasp_from_point_clouds",
    "get_oriented_bounding_box_from_3d_points",
    "solve_ik",
    "move_to_joints",
    "open_gripper",
    "close_gripper",
    "goto_pose",
    "goto_home_joint_position",
    "subsample_point_cloud",
    "filter_noise",
})

# FrankaLiberoApiReducedSkillLibrary.functions() additions (L4)
_L4_EXTRAS = frozenset({
    "rotation_matrix_to_quaternion",
    "decompose_transform",
    "depth_to_point_cloud",
    "mask_to_world_points",
    "pixel_to_world_point",
    "transform_points",
    "interpolate_segment",
    "normalize_vector",
    "select_top_down_grasp",
})

L4_API_FUNCTIONS: frozenset[str] = _L3_FUNCTIONS | _L4_EXTRAS


def live_api_functions(api: Any) -> set[str]:
    """Function names from a live CapX api object (``api.functions()`` keys)."""

    return set(api.functions().keys())
