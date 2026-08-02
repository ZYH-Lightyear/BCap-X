from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a Franka Emika robot with the API described below.
Goal: Pick up the red cube and gently stack it on top of the green cube, then release it.

Key rules:
- The extent from get_object_pose(..., return_bbox_extent=True) is the FULL side length. Use extent[2]/2 for half-height.
- For placement orientation, reuse the grasp quaternion from sample_grasp_pose. Do NOT use the quaternion from get_object_pose (it is unreliable for orientation).
- Always use z_approach=0.1 when approaching an object for grasping or placing.
- After grasping, lift the cube to a safe height (at least +0.2m in Z) before moving laterally to the placement location.
- The stacking height formula is: place_z = green_center_z + green_extent[2]/2 + red_extent[2]/2
- Nothing should be dropped from a height. Always approach with z_approach for controlled descent.

Write ONLY executable Python code (no code fences). Import numpy if needed.
"""
ORACLE_CODE = """
import numpy as np
from scipy.spatial.transform import Rotation as Rt

# --------------------------------- pick ---------------------------------
# perceive(3D) 
_, _, green_ext = get_object_pose("green cube", return_bbox_extent=True)
red_pos_meas, _, red_ext = get_object_pose("red cube", return_bbox_extent=True)

# propose
grasp_pos, grasp_quat = sample_grasp_pose("red cube")
xy_err = float(np.linalg.norm(grasp_pos[:2] - red_pos_meas[:2]))
if xy_err > 0.02:
    # OBB 朝向对对称物体（方块等）往往不稳定，任务注释里一般建议放置时别用这个 quat
    # yaw 来自 CGN 的 grasp_quat：先把原抓取姿态拆成欧拉角，只抽出 Z 分量。这样即使 CGN 整体倾斜不可用，仍尽量保留它建议的“从哪个方向夹”，只丢掉坏的 pitch/roll.
    yaw = Rt.from_quat([grasp_quat[1], grasp_quat[2], grasp_quat[3], grasp_quat[0]]).as_euler("xyz")[2]
    q_xyzw = Rt.from_euler("xyz", [np.pi, 0.0, yaw]).as_quat()
    pick_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    pick_pos = red_pos_meas.copy()
else:
    pick_pos, pick_quat = grasp_pos, grasp_quat

# approach
goto_pose(pick_pos, pick_quat, z_approach=0.1)

# commit
close_gripper()

# manipulate(post)
post_pick_pos = pick_pos.copy()
post_pick_pos[2] += 0.2
goto_pose(post_pick_pos, pick_quat)

# --------------------------------- place ---------------------------------
# perceive(3D)
green_pos, _, _ = get_object_pose("green cube", return_bbox_extent=False)

# propose
place_pos = green_pos.copy()
place_pos[2] = green_pos[2] + green_ext[2]/2 + red_ext[2]/2

# approach
goto_pose(place_pos, pick_quat, z_approach=0.1)  # reuse pick quat

# commit
open_gripper()

# manipulate(post)
post_place_pos = place_pos.copy()
post_place_pos[2] += 0.1
goto_pose(post_place_pos, pick_quat)
"""


# ---------------------------- High-level Env -----------------------------
class FrankaPickPlaceCodeEnv(CodeExecutionEnvBase):
    """High-level code environment for Franka pick-and-place using SimpleExecutor."""

    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = [
    "FrankaPickPlaceCodeEnv",
]
