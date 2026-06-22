"""Staged pick-and-place with VLM checkpoints between sub-actions.

Hard-coded pipeline for LIBERO spatial task 0 that demonstrates true
verify-before-commit: every sub-action runs alone, saves annotated evidence,
and a VLM checkpoint must approve (or correct) it before the next sub-action
is allowed to run.

Checkpoints:
  step1 SAM3 candidates -> numbered thin boxes (no fill), blue = preselect
  step2 VLM: "is the blue box the bowl between plate and ramekin? else which #"
  step3 grasp planned ON the confirmed mask -> dot+arrow overlay
  step4 VLM: "is the grasp point on the selected bowl? APPROVE/REJECT"
  step5 execute grasp + lift, re-segment, measure overhang
  step6 transit corridor check (text report)
  step7 place metrics (text report) -> release height computed
  step8 execute place, final verification

Each step writes outputs/staged_verified_demo/step_XX_<name>/
  evidence.png / vlm_qa.txt / decision.json

Run:
    cd /mnt/data/xuyingjie/BCap-X
    MUJOCO_GL=egl GRASPNET_SERVICE_URL=http://127.0.0.1:8125 \
      .venv-libero/bin/python scripts/staged_verified_pickplace.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUT = REPO_ROOT / "outputs" / "staged_verified_demo"
TASK = "Pick the akita black bowl between the plate and the ramekin and place it on the plate"

_step_idx = 0


def step_dir(name: str) -> Path:
    global _step_idx
    _step_idx += 1
    d = OUT / f"step_{_step_idx:02d}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_png(d: Path, rgb: np.ndarray, name: str = "evidence.png") -> Path:
    p = d / name
    cv2.imwrite(str(p), cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    return p


def save_decision(d: Path, decision: dict) -> None:
    (d / "decision.json").write_text(json.dumps(decision, indent=2, default=str))


# ----------------------------- VLM client ----------------------------- #
_cfg = yaml.safe_load((REPO_ROOT / "env_configs/libero/franka_libero_spatial_0.yaml").read_text())


def ask_vlm(d: Path, question: str, image: np.ndarray | None = None) -> str:
    """One VLM checkpoint call. Saves Q and A to vlm_qa.txt in the step dir."""
    import requests

    content: list[dict] = [{"type": "text", "text": question}]
    if image is not None:
        save_png(d, image, "vlm_input.png")  # exact image sent to the VLM
        buf = io.BytesIO()
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    payload = {"model": _cfg.get("model", ""),
               "messages": [{"role": "user", "content": content}],
               "temperature": 0.0, "max_tokens": 4000}
    resp = requests.post(
        _cfg["server_url"],
        headers={"Authorization": _cfg["api_key"], "Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    answer = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    (d / "vlm_qa.txt").write_text(f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n")
    # full request record (image replaced by a reference to keep it readable)
    record = {
        "model": payload["model"],
        "temperature": payload["temperature"],
        "max_tokens": payload["max_tokens"],
        "prompt_text": question,
        "image_attached": image is not None,
        "image_file": "evidence.png" if image is not None else None,
        "answer": answer,
    }
    (d / "vlm_request.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"[vlm] {answer[:200]}")
    return answer


# --------------------------- annotation helpers --------------------------- #
def draw_numbered_boxes(
    rgb: np.ndarray, results: list[dict], selected: int
) -> np.ndarray:
    """Thin numbered boxes; the number sits at the BOX CENTER so the
    number-to-box association is unambiguous even when boxes interleave.
    Selected box is blue, others yellow."""
    out = np.ascontiguousarray(np.asarray(rgb).copy())
    for i, r in enumerate(results):
        box = r.get("box")
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (40, 120, 255) if i == selected else (255, 220, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"{i + 1}"
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(out, tag, (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, tag, (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        if i == selected:
            cv2.putText(out, "SELECTED", (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, "SELECTED", (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def parse_select(answer: str, n: int, default: int) -> int:
    """Parse 'SELECT: k' (1-based) from the VLM answer."""
    import re

    m = re.search(r"SELECT\s*[:#]?\s*(\d+)", answer, re.IGNORECASE)
    if m:
        k = int(m.group(1)) - 1
        if 0 <= k < n:
            return k
    return default


# ------------------------------- pipeline ------------------------------- #
def main() -> None:
    from capx.integrations.franka.libero_evidence import FrankaLiberoApiEvidence
    from capx.envs.simulators.libero import FrankaLiberoEnv
    from capx.utils.pointcloud_render import draw_grasp_point_on_image, pose_from_position_wxyz

    print("loading env ...")
    env = FrankaLiberoEnv(suite_name="libero_spatial", task_id=0, seed=0)
    env.enable_video_capture(True, clear=True)  # record the whole episode
    api = FrankaLiberoApiEvidence(env)
    obs = api.get_observation()
    rgb = obs["agentview"]["images"]["rgb"]
    K = obs["agentview"]["intrinsics"]
    cam_pose = obs["agentview"]["pose_mat"]
    depth = np.asarray(obs["agentview"]["images"]["depth"])

    def mask_points(mask):
        m = np.asarray(mask).astype(bool)
        d = depth[:, :, 0] if depth.ndim == 3 else depth
        ys, xs = np.where(m)
        z = d[ys, xs]
        ok = (z > 0.01) & (z < 3.0)
        x_cam = (xs[ok] - K[0, 2]) * z[ok] / K[0, 0]
        y_cam = (ys[ok] - K[1, 2]) * z[ok] / K[1, 1]
        p = np.stack([x_cam, y_cam, z[ok]], axis=-1)
        return p @ cam_pose[:3, :3].T + cam_pose[:3, 3]

    # ---------------- step 1: SAM3 candidates (numbered) ----------------
    d = step_dir("sam3_candidates")
    all_results = sorted(api.segment_sam3_text_prompt(rgb, "black bowl"),
                         key=lambda r: -r["score"])
    # drop low-score noise (ramekin/plate misdetections); keep at least two
    results = [r for r in all_results if r["score"] >= 0.3][:4] or all_results[:2]
    assert results, "no bowl candidates"
    preselect = 0  # naive top-1, to be verified
    save_png(d, rgb, "raw.png")  # clean image before any annotation
    annotated = draw_numbered_boxes(rgb, results, preselect)
    save_png(d, annotated)
    save_decision(d, {"candidates": [round(r["score"], 3) for r in results],
                      "preselect": preselect + 1})
    print(f"[step1] {len(results)} candidates, preselect #1 "
          f"(scores {[round(r['score'], 2) for r in results]})")

    # ---------------- step 2: VLM checkpoint on target selection ----------------
    d = step_dir("verify_target")
    q = (
        f"Task: {TASK}\n"
        f"The image shows numbered candidate boxes for 'black bowl'. The BLUE box "
        f"(#{preselect + 1}) is currently selected.\n"
        "Question: is the blue box the bowl BETWEEN the plate and the ramekin? "
        "The plate is the large flat striped dish; the ramekin is the small white cup.\n"
        "Answer EXACTLY: 'SELECT: <number>' for the correct bowl, then one sentence why."
    )
    ans = ask_vlm(d, q, annotated)
    chosen = parse_select(ans, len(results), preselect)
    confirmed = draw_numbered_boxes(rgb, results, chosen)
    save_png(d, confirmed, "evidence.png")
    save_decision(d, {"vlm_choice": chosen + 1, "changed": chosen != preselect})
    print(f"[step2] VLM chose #{chosen + 1} (preselect was #{preselect + 1})")

    target_mask = results[chosen]["mask"]
    target_pts = mask_points(target_mask)
    target_center = np.median(target_pts, axis=0)
    pre_top = float(np.percentile(target_pts[:, 2], 98))
    pre_bottom = float(np.percentile(target_pts[:, 2], 2))

    # ---------------- step 3: plan grasp on the CONFIRMED mask ----------------
    d = step_dir("plan_grasp")
    import viser.transforms as vtf
    from capx.utils.depth_utils import depth_to_pointcloud

    pc_full_parts = []
    for cam in ["agentview", "robot0_eye_in_hand"]:
        dd = obs[cam]["images"]["depth"]
        pts_c = depth_to_pointcloud(dd, obs[cam]["intrinsics"], subsample_factor=1)
        pts_h = np.concatenate([pts_c, np.ones((len(pts_c), 1))], axis=1)
        pc_full_parts.append((obs[cam]["pose_mat"] @ pts_h.T).T[:, :3])
    pc_full = np.concatenate(pc_full_parts)

    grasp_tf, grasp_scores = api.plan_grasp_from_point_clouds(
        pc_full.astype(np.float32), target_pts.astype(np.float32))
    grasp_tf = np.asarray(grasp_tf).reshape(-1, 4, 4)
    grasp_scores = np.asarray(grasp_scores).reshape(-1)
    order = list(np.argsort(-grasp_scores))
    # geometric pre-filter: a bowl grasp must be at rim height, not the bottom
    rim_z = pre_top - 0.015
    order = [i for i in order if grasp_tf[i][2, 3] >= rim_z] + \
            [i for i in order if grasp_tf[i][2, 3] < rim_z]
    save_decision(d, {"n_candidates": len(order),
                      "scores": [round(float(grasp_scores[i]), 3) for i in order[:5]],
                      "rim_z_threshold": rim_z})
    print(f"[step3] {len(order)} candidates planned (rim_z>={rim_z:.3f} preferred)")

    # ---------------- step 4: VLM checkpoint with repair loop ----------------
    g_pos = g_quat = None
    for attempt, idx in enumerate(order[:3]):
        d = step_dir(f"verify_grasp_try{attempt + 1}")
        pose_se3 = vtf.SE3.from_matrix(grasp_tf[idx]) @ vtf.SE3.from_rotation(
            vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2))
        cand_pos = pose_se3.wxyz_xyz[-3:]
        cand_quat = pose_se3.wxyz_xyz[:4]
        overlay = draw_grasp_point_on_image(
            confirmed, pose_from_position_wxyz(cand_pos, cand_quat), K, cam_pose,
            color=(0, 220, 0), label="planned grasp")
        save_png(d, overlay)
        q = (
            f"Task: {TASK}\n"
            "The green dot is the planned grasp point (arrow = approach direction); "
            "the blue box (number 2 style marking) is the confirmed target bowl.\n"
            "A good bowl grasp pinches the RIM of the target bowl. Question: is this "
            "grasp point on the confirmed target bowl's rim or body (not another "
            "object, not the bowl's inside bottom)?\n"
            "Answer EXACTLY 'APPROVE' or 'REJECT', then one sentence why."
        )
        ans = ask_vlm(d, q, overlay)
        approved = ans.upper().strip().startswith("APPROVE") or (
            "APPROVE" in ans.upper() and "REJECT" not in ans.upper())
        save_decision(d, {"attempt": attempt + 1, "grasp_pos": cand_pos.tolist(),
                          "grasp_z": float(cand_pos[2]), "score": float(grasp_scores[idx]),
                          "approved": bool(approved)})
        print(f"[step4] try{attempt + 1}: grasp {np.round(cand_pos, 3)} "
              f"-> {'APPROVED' if approved else 'REJECTED'}")
        if approved:
            g_pos, g_quat = cand_pos, cand_quat
            break
    if g_pos is None:
        print("[step4] all grasp candidates rejected -- stopping")
        return

    # ---------------- step 5: execute grasp + lift, measure overhang ----------------
    d = step_dir("execute_grasp")
    api.open_gripper()
    api.goto_pose(g_pos, g_quat, z_approach=0.10)
    api.goto_pose(g_pos, g_quat, z_approach=0.0)
    api.close_gripper()
    lift_pos = np.array([g_pos[0], g_pos[1], g_pos[2] + 0.15])
    api.goto_pose(lift_pos, g_quat, z_approach=0.0)

    obs2 = api.get_observation()
    tcp_z = float(obs2["robot_cartesian_pos"][2])
    overhang = tcp_z - pre_bottom - 0.15  # fallback from pre-grasp geometry
    held_pts = target_pts
    try:
        res = api.get_object_3d_points_and_masks_from_language("black bowl", use_multiview=False)
        pts_now = np.asarray(res["points_3d"])
        if len(pts_now) > 100 and np.percentile(pts_now[:, 2], 98) > pre_top + 0.05:
            held_pts = pts_now
            overhang = tcp_z - float(np.percentile(pts_now[:, 2], 2))
            print(f"[step5] post-lift re-segmentation ok, measured overhang={overhang:.3f}m")
        else:
            overhang = tcp_z - (pre_bottom + 0.15)
            print(f"[step5] re-segmentation not lifted/unreliable, fallback overhang={overhang:.3f}m")
    except Exception as e:
        overhang = tcp_z - (pre_bottom + 0.15)
        print(f"[step5] re-segmentation failed ({e}), fallback overhang={overhang:.3f}m")
    overhang = float(np.clip(overhang, 0.02, 0.30))
    rgb5 = api.get_observation()["agentview"]["images"]["rgb"]
    save_png(d, rgb5)
    save_decision(d, {"tcp_z": tcp_z, "overhang": overhang})

    # ---------------- step 6: transit corridor check ----------------
    d = step_dir("check_transit")
    plate_res = api.get_object_3d_points_and_masks_from_language("plate", use_multiview=False)
    plate_pts = np.asarray(plate_res["points_3d"])
    plate_center = np.median(plate_pts, axis=0)
    obstacles = {}
    for i, r in enumerate(results):
        if i != chosen:
            obstacles[f"bowl_{i + 1}"] = mask_points(r["mask"])
    transit_tcp_z = tcp_z
    rep = api.check_transit_clearance(
        held_pts, overhang, target_center[:2], plate_center[:2], transit_tcp_z, obstacles)
    if not rep["ok"]:
        transit_tcp_z = rep["min_safe_tcp_z"] + 0.01
        print(f"[step6] raised transit height to {transit_tcp_z:.3f}m")
    save_decision(d, {"transit_tcp_z": transit_tcp_z, "report": rep["rows"],
                      "min_safe_tcp_z": rep["min_safe_tcp_z"]})

    # ---------------- step 7: place metrics ----------------
    d = step_dir("check_place")
    rep2 = api.check_place_metrics(plate_pts, overhang, plate_center[:2], clearance=0.02)
    save_decision(d, {"release_tcp_z": rep2["release_tcp_z"], "ok": rep2["ok"]})
    if not rep2["ok"]:
        print("[step7] place metrics NOT ok -- stopping")
        return

    # ---------------- step 8: execute place + final verification ----------------
    d = step_dir("execute_place")
    down_quat = np.array([0.0, 1.0, 0.0, 0.0])
    api.goto_pose(np.array([plate_center[0], plate_center[1], transit_tcp_z]),
                  down_quat, z_approach=0.0)
    api.goto_pose(np.array([plate_center[0], plate_center[1], rep2["release_tcp_z"]]),
                  down_quat, z_approach=0.0)
    api.open_gripper()
    api.goto_pose(np.array([plate_center[0], plate_center[1], transit_tcp_z]),
                  down_quat, z_approach=0.0)

    # let the bowl settle on the plate before reading the success predicate:
    # On() relies on physical contact, which is only established after the
    # object falls and stabilizes. open_gripper(steps=30) advances the sim
    # while holding the arm; repeat to settle without moving the gripper.
    settle_completed = []
    for k in range(6):
        api.open_gripper()
        sc = bool(env.task_completed()) if hasattr(env, "task_completed") else False
        settle_completed.append(sc)
        if sc:
            break
    reward = settle_completed[-1] if settle_completed else None
    print(f"[step8] settle success readings: {settle_completed}")
    obs3 = api.get_observation()
    rgb8 = obs3["agentview"]["images"]["rgb"]
    save_png(d, rgb8)
    final = {}
    try:
        res = api.get_object_3d_points_and_masks_from_language("black bowl", use_multiview=False)
        c = np.median(np.asarray(res["points_3d"]), axis=0)
        final = {"bowl_center": c.tolist(),
                 "dist_to_plate_center": float(np.linalg.norm(c[:2] - plate_center[:2]))}
        print(f"[step8] final bowl center {np.round(c, 3)}, "
              f"dist to plate center {final['dist_to_plate_center']:.3f}m")
    except Exception as e:
        print(f"[step8] final re-segmentation failed: {e}")
    final["task_completed"] = bool(reward)
    save_decision(d, final)
    print(f"[done] task_completed={reward}")

    # ---------------- save the full-episode video ----------------
    try:
        import imageio

        frames = env.get_video_frames(clear=False)
        if frames:
            vid = OUT / "episode.mp4"
            imageio.mimsave(str(vid), [np.asarray(f, dtype=np.uint8) for f in frames], fps=30)
            print(f"[video] saved {vid} ({len(frames)} frames)")
        else:
            print("[video] no frames captured")
    except Exception as e:
        print(f"[video] save failed: {e}")


if __name__ == "__main__":
    main()
