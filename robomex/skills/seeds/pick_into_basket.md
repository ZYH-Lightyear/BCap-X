---
kind: action
skill_id: pick_into_basket
name: Pick Object Into Basket
description: Grasp a table-top object with a top-down grasp and release it inside a basket.
requires: [object_mask, object_points]
produces: [object_grasped, object_in_container]
keywords: [pick, place, grasp, basket, container, top_down, release, transport]
verify:
  - "After grasp: the object stays between the fingers after a small lift (wrist view)."
  - "After release: the object center lies within the basket footprint and the gripper is open."
recovery:
  - "Grasp failed or object not held after lift: reset gripper, re-segment the object, re-rank grasp candidates by IK feasibility, retry once."
  - "Object released outside the basket: re-grasp from the current pose and re-plan a higher approach over the basket center."
version: "0.1"
---

Precondition: a verified mask and filtered 3D points of the target object (see the segmentation skill). Adapt the sketches to the current observation, do not copy them blindly.

Grasp (top-down, collision-checked):

```python
obb = get_oriented_bounding_box_from_3d_points(points)
grasp_pos, grasp_quat = get_top_down_grasp_from_obb(obb)
ok, traj, _ = plan_grasp_trajectory(object_name, object_mask=mask,
                                    grasp_poses=[(grasp_pos, grasp_quat)], use_world_collision=True)
assert ok and traj is not None, "no collision-free grasp path"
execute_joint_trajectory(traj, subsample=2)
close_gripper()
```

Place into the basket (carry high, release above the opening):

```python
held = get_object_3d_points_and_masks_from_language(object_name, use_multiview=False)
held_obb = get_oriented_bounding_box_from_3d_points(held["points_3d"])
basket = get_object_3d_points_and_masks_from_language("basket")
basket_pos = get_oriented_bounding_box_from_3d_points(filter_noise(basket["points_3d"])[0])["center"]
target = (np.array(basket_pos) + np.array([0, 0, 0.3]), np.array([0.0, 1.0, 0.0, 0.0]))
ok, traj = plan_with_grasped_object(target, object_name,
                                    object_pose=(held_obb["center"], so3_to_wxyz(held_obb["R"])),
                                    object_mask=held["agentview_mask"])
assert ok and traj is not None, "could not plan to basket"
execute_joint_trajectory(traj, subsample=2)
open_gripper()
```

Rules:

- Always check the planner's `ok` flag; a failed plan means re-perceive, not retry the same pose.
- Approach height above the basket must clear the held object's height (use its OBB extent, do not guess a constant).
- Release only when positioned over the basket opening; verify afterwards before declaring success.
