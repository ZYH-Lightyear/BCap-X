"""Per-API-call diagnostic harness for the Robosuite cube-stack task.

Re-runs a cube-stack program (the original trial_03 oracle code, or an
improved variant) against the SAME low-level env + FrankaControlApi used in
the failing run, and after EVERY api call captures the current end-frame RGB
and overlays:

  * the pose the API returned (perception / grasp result),
  * the privileged ground-truth (GT) cube pose from the simulator,

both projected into the robot0_robotview camera. This makes the perception vs
GT gap visible step by step.

Reuses existing infra:
  - FrankaControlApi (get_object_pose / sample_grasp_pose / goto_pose /
    open_gripper / close_gripper)  -> perception + IK + grasp already exist.
  - FrankaRobosuiteCubesLowLevel.get_observation()["cube_poses"] -> GT poses.
  - camera intrinsics / extrinsics (pose_mat) already exposed on the obs.

Run inside the `sci` conda env with pyroki(:8116) + graspnet(:8115) servers up
and SAM3_SERVICE_URL pointing at the remote SAM3 service.
"""

from __future__ import annotations

import argparse
import pathlib
import numpy as np
import cv2

from capx.envs.simulators.robosuite_cubes import FrankaRobosuiteCubesLowLevel
from capx.integrations.franka.control import FrankaControlApi

CAM = "robot0_robotview"

# cubeA = primary = RED (the cube to pick), cubeB = secondary = GREEN (base).
GT_KEYS = {"red cube": "primary", "green cube": "secondary"}
GT_COLOR = {"red cube": (255, 60, 60), "green cube": (60, 220, 60)}


# ----------------------------- geometry helpers ---------------------------- #
def world_to_pixel(p_base, pose_mat, K):
    """Project a point in the robot-base frame to pixel coords.

    pose_mat is camera->base (4x4); invert to get base->camera.
    """
    world_to_cam = np.linalg.inv(pose_mat)
    pc = (world_to_cam @ np.append(np.asarray(p_base, float), 1.0))[:3]
    if pc[2] <= 1e-6:
        return None, pc[2]
    uv = K @ pc
    return (float(uv[0] / uv[2]), float(uv[1] / uv[2])), float(pc[2])


def quat_wxyz_to_R(q):
    from scipy.spatial.transform import Rotation as Rt
    q = np.asarray(q, float)
    return Rt.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def draw_axes(img, center_base, quat_wxyz, pose_mat, K, length=0.06, thickness=2):
    """Draw projected XYZ axes (R=x, G=y, B=z) of a pose."""
    R = quat_wxyz_to_R(quat_wxyz)
    o_uv, oz = world_to_pixel(center_base, pose_mat, K)
    if o_uv is None:
        return
    cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # x,y,z in BGR
    for i in range(3):
        tip = np.asarray(center_base, float) + R[:, i] * length
        t_uv, tz = world_to_pixel(tip, pose_mat, K)
        if t_uv is None:
            continue
        cv2.line(img, (int(o_uv[0]), int(o_uv[1])),
                 (int(t_uv[0]), int(t_uv[1])), cols[i], thickness, cv2.LINE_AA)


def marker(img, uv, color, label, r=7):
    if uv is None:
        return
    x, y = int(uv[0]), int(uv[1])
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 2 * r, 2)
    cv2.circle(img, (x, y), r, color, 2)
    if label:
        cv2.putText(img, label, (x + 9, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)


def banner(img, lines, org=(8, 20)):
    y = org[1]
    for ln, col in lines:
        cv2.putText(img, ln, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, ln, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 1, cv2.LINE_AA)
        y += 20


# ---------------------------- tracing API wrapper -------------------------- #
class TracedApi:
    """Wraps FrankaControlApi so every call snapshots the end-frame RGB with
    both the returned pose and the GT cube poses overlaid.
    """

    def __init__(self, env, api, outdir: pathlib.Path):
        self.env = env
        self.api = api
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.records = []

    # --- GT + camera access (always available on this env) ---
    def _cam(self):
        obs = self.env.get_observation()
        rgb = obs[CAM]["images"]["rgb"]
        K = obs[CAM]["intrinsics"]
        pose_mat = obs[CAM]["pose_mat"]
        return obs, rgb, K, pose_mat

    def _gt(self, obs, name):
        arr = obs["cube_poses"][GT_KEYS[name]]
        return np.asarray(arr[:3], float), np.asarray(arr[3:7], float)

    def _snapshot(self, call, ret_pos=None, ret_quat=None, focus=None, extra=None):
        obs, rgb, K, pose_mat = self._cam()
        img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

        # GT overlays for both cubes
        gt_txt = []
        for name in ("red cube", "green cube"):
            gpos, gquat = self._gt(obs, name)
            uv, _ = world_to_pixel(gpos, pose_mat, K)
            marker(img, uv, GT_COLOR[name][::-1], f"GT {name.split()[0]}")
            draw_axes(img, gpos, gquat, pose_mat, K, length=0.04)
            gt_txt.append((name, gpos))

        # API-returned pose overlay (perception / grasp)
        err_txt = []
        if ret_pos is not None:
            uv, _ = world_to_pixel(ret_pos, pose_mat, K)
            marker(img, uv, (0, 200, 255), "API", r=9)  # cyan-ish (BGR)
            if ret_quat is not None:
                draw_axes(img, ret_pos, ret_quat, pose_mat, K, length=0.06, thickness=2)
            if focus in GT_KEYS:
                gpos, _ = self._gt(obs, focus)
                d = float(np.linalg.norm(np.asarray(ret_pos, float) - gpos))
                err_txt.append((f"|API - GT {focus.split()[0]}| = {d*1000:.1f} mm",
                                (0, 200, 255)))

        reward = float(self.env.compute_reward())
        done = bool(self.env.task_completed())
        lines = [(f"step {self.step:02d}: {call}", (255, 255, 255))]
        if ret_pos is not None:
            lines.append((f"API pos = [{ret_pos[0]:.3f} {ret_pos[1]:.3f} {ret_pos[2]:.3f}]",
                          (0, 200, 255)))
        lines += err_txt
        for name, gpos in gt_txt:
            lines.append((f"GT {name.split()[0]:5s}= [{gpos[0]:.3f} {gpos[1]:.3f} {gpos[2]:.3f}]",
                          GT_COLOR[name][::-1]))
        lines.append((f"reward={reward:.3f}  done={done}", (200, 255, 200)))
        if extra:
            lines.append((extra, (255, 255, 0)))
        banner(img, lines)

        path = self.outdir / f"step_{self.step:02d}_{call}.png"
        cv2.imwrite(str(path), img)
        self.records.append({
            "step": self.step, "call": call, "reward": reward, "done": done,
            "api_pos": None if ret_pos is None else np.asarray(ret_pos).tolist(),
            "img": str(path),
        })
        self.step += 1
        return path

    # --- traced API calls ---
    def get_object_pose(self, name, return_bbox_extent=False):
        out = self.api.get_object_pose(name, return_bbox_extent=return_bbox_extent)
        pos, quat = out[0], out[1]
        self._snapshot(f"get_object_pose[{name.split()[0]}]", pos, quat, focus=name)
        return out

    def sample_grasp_pose(self, name):
        pos, quat = self.api.sample_grasp_pose(name)
        self._snapshot(f"sample_grasp_pose[{name.split()[0]}]", pos, quat, focus=name)
        return pos, quat

    def goto_pose(self, pos, quat, z_approach=0.0, tag=""):
        self.api.goto_pose(pos, quat, z_approach=z_approach)
        self._snapshot(f"goto_pose{('_'+tag) if tag else ''}", pos, quat)

    def open_gripper(self):
        self.api.open_gripper()
        self._snapshot("open_gripper")

    def close_gripper(self):
        self.api.close_gripper()
        self._snapshot("close_gripper")


# --------------------------------- programs -------------------------------- #
def program_original(T: "TracedApi"):
    """The exact trial_03 oracle code, instrumented per call."""
    _, _, green_ext = T.get_object_pose("green cube", return_bbox_extent=True)
    _, _, red_ext = T.get_object_pose("red cube", return_bbox_extent=True)

    pick_pos, pick_quat = T.sample_grasp_pose("red cube")
    T.goto_pose(pick_pos, pick_quat, z_approach=0.1, tag="pick")
    T.close_gripper()
    post_pick_pos = pick_pos.copy(); post_pick_pos[2] += 0.2
    T.goto_pose(post_pick_pos, pick_quat, tag="lift")

    green_pos, _, _ = T.get_object_pose("green cube", return_bbox_extent=False)
    place_pos = green_pos.copy()
    place_pos[2] = green_pos[2] + green_ext[2] / 2 + red_ext[2] / 2
    place_quat = np.array([0.0, 0.0, 1.0, 0.0])
    T.goto_pose(place_pos, pick_quat, z_approach=0.1, tag="place")
    T.open_gripper()
    post_place_pos = place_pos.copy(); post_place_pos[2] += 0.1
    T.goto_pose(post_place_pos, place_quat, tag="retract")


def program_improved(T: "TracedApi"):
    """Improved cube-stack program. Reuses the SAME existing APIs.

    Lesson from the GT-overlay diagnosis (same seed, local SAM3):
      * When Contact-GraspNet is already close to the cube center, forcing a
        top-down re-orientation / Z rewrite produces a WEAK grasp (cube slips
        ~10cm below TCP during lift) and the cube falls on the place move.
      * When CGN is far off (trial_03-style corner pinch), fall back to a
        top-down grasp at the measured cube center.

    So: keep the CGN pose when |grasp_xy - red_xy| is small; otherwise fall
    back. Always place with the SAME quat used for the successful pick.
    """
    from scipy.spatial.transform import Rotation as Rt

    _, _, green_ext = T.get_object_pose("green cube", return_bbox_extent=True)
    red_pos_meas, _, red_ext = T.get_object_pose("red cube", return_bbox_extent=True)

    grasp_pos, grasp_quat = T.sample_grasp_pose("red cube")

    xy_err = float(np.linalg.norm(grasp_pos[:2] - red_pos_meas[:2]))
    # ~2cm: within a cube half-side; beyond this CGN is pinching a corner.
    if xy_err > 0.02:
        yaw = Rt.from_quat([grasp_quat[1], grasp_quat[2], grasp_quat[3], grasp_quat[0]]).as_euler("xyz")[2]
        q_xyzw = Rt.from_euler("xyz", [np.pi, 0.0, yaw]).as_quat()
        pick_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        pick_pos = red_pos_meas.copy()
        print(f"[improved] CGN xy_err={xy_err*1000:.1f}mm -> fallback top-down at cube center")
    else:
        pick_pos, pick_quat = grasp_pos, grasp_quat
        print(f"[improved] CGN xy_err={xy_err*1000:.1f}mm -> keep CGN pose")

    T.goto_pose(pick_pos, pick_quat, z_approach=0.1, tag="pick")
    T.close_gripper()

    post_pick_pos = pick_pos.copy(); post_pick_pos[2] += 0.20
    T.goto_pose(post_pick_pos, pick_quat, tag="lift")

    green_pos, _, _ = T.get_object_pose("green cube", return_bbox_extent=False)
    place_pos = green_pos.copy()
    place_pos[2] = green_pos[2] + green_ext[2] / 2 + red_ext[2] / 2

    T.goto_pose(place_pos, pick_quat, z_approach=0.1, tag="place")
    T.open_gripper()
    post_place_pos = place_pos.copy(); post_place_pos[2] += 0.1
    T.goto_pose(post_place_pos, pick_quat, tag="retract")


PROGRAMS = {"original": program_original, "improved": program_improved}


# ---------------------------------- runner --------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", choices=list(PROGRAMS), default="original")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    # Seed EVERYTHING before env construction so layout is reproducible across
    # original vs improved runs.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = FrankaRobosuiteCubesLowLevel(privileged=False, enable_render=True, seed=args.seed)
    env.robosuite_env.rng = np.random.default_rng(args.seed)
    # Force a deterministic placement draw under our rng.
    if hasattr(env.robosuite_env, "placement_initializer"):
        env.robosuite_env.placement_initializer.rng = env.robosuite_env.rng
    env.reset(seed=args.seed)
    env.enable_video_capture(True)

    api = FrankaControlApi(env, use_sam3=True)
    T = TracedApi(env, api, pathlib.Path(args.outdir))

    # save an initial frame
    T._snapshot("reset")

    err = None
    try:
        PROGRAMS[args.program](T)
    except Exception as e:  # noqa: BLE001
        import traceback
        err = traceback.format_exc()
        T._snapshot("EXCEPTION", extra=repr(e)[:60])

    # final summary frame
    fp = T._snapshot("final")
    final = T.records[-1]
    print(f"[{args.program}] final reward={final['reward']:.4f} done={final['done']} "
          f"steps={T.step} out={args.outdir}")
    if err:
        print("ERROR during program:\n", err)

    import json
    (pathlib.Path(args.outdir) / "records.json").write_text(
        json.dumps({"program": args.program, "seed": args.seed,
                    "final_reward": final["reward"], "done": final["done"],
                    "records": T.records}, indent=2))


if __name__ == "__main__":
    main()
