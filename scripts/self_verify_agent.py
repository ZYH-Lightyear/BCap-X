"""Block-structured self-verifying VLM agent driver.

The VLM writes ONE Python program split into blocks separated by verify
markers:

    <block code>
    # @verify: <check_type> key=var ...
    <next block code>
    ...

The driver:
  1. asks the VLM for the full program (system prompt lists every check),
  2. splits it into (code, verify_marker) blocks,
  3. executes each block in a persistent namespace,
  4. dispatches the declared check; on PASS continues, on FAIL hands the
     block + failure message back to the VLM to rewrite, re-executes, retries,
  5. saves per-block code/evidence/decision and the full episode video.

This keeps the engine generic: the agent decides WHAT to verify (via markers)
and HOW to fix it (by rewriting the block); the registry only provides the
checking machinery.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from capx.verifier.checks import REGISTRY, VERIFY_CATALOG  # noqa: E402

OUT = REPO_ROOT / "outputs" / "self_verify_demo"
MAX_BLOCK_RETRIES = 3
# Only these check types are actually run; others are parsed but SKIPPED
# (their blocks still execute, just without verification). Set via env var
# CAPX_ACTIVE_CHECKS as a comma list. Default: only the bowl-selection check.
ACTIVE_CHECKS = set(
    c for c in os.environ.get("CAPX_ACTIVE_CHECKS", "sam3_target,grasp_select").split(",") if c
)
_cfg = yaml.safe_load((REPO_ROOT / "env_configs/libero/franka_libero_spatial_0.yaml").read_text())


def build_system_prompt(task: str) -> str:
    """Build the system prompt for an arbitrary LIBERO task.

    `task` is the natural-language goal (e.g. taken from
    ``env.handle.task_language``). The grasp/place pattern below is GENERIC --
    it refers to the target object and the receptacle by role, not by a
    hard-coded name, so it transfers across tasks (bowl->plate, soup->basket...).
    """
    return f"""\
You control a Franka robot by writing ONE Python program for this task:
  {task}

Read the task above to identify TWO things:
  - TARGET  = the object to pick up (the noun after "pick up").
  - RECEPTACLE = where to put it (the noun after "place it in/on"). If the task
    has a spatial constraint that disambiguates the target among look-alikes
    (e.g. a "between ... and ..." / "left/right" phrase), use it as the
    sam3_target `relation`; otherwise leave relation empty.

The program MUST be organized in BLOCKS. After a block that produces something
checkable, add a verification marker comment on its own line:

    # @verify: <check_type> param=var ...

The driver runs the program block by block on the simulator, then runs each
declared check. If a check fails, you will be shown the FULL program and the
failure, and asked to rewrite the ENTIRE program (keeping all blocks mutually
consistent). Put perception/planning BEFORE the action it guards.

{VERIFY_CATALOG}

API functions (already in scope; `import numpy as np` yourself). Use EXACTLY
these signatures — do not invent methods, do not call .save() on anything:

  obs = get_observation()
     obs["agentview"]["images"]["rgb"]     -> (H,W,3) uint8
     obs["agentview"]["images"]["depth"]   -> (H,W) float32
     obs["agentview"]["intrinsics"]        -> (3,3)
     obs["agentview"]["pose_mat"]          -> (4,4) cam->world
     obs["robot_cartesian_pos"]            -> (8,) [x,y,z, qw,qx,qy,qz, grip]
  results = segment_sam3_text_prompt(rgb, "<TARGET>")   # short noun phrase
     -> list of {{"mask":(H,W) bool, "box":[x1,y1,x2,y2], "score":float}}
  mask_to_world_points(mask, depth, intrinsics, pose_mat) -> (N,3) world points
  cands = sample_grasp_candidates("<TARGET>", k=3, target_points=tp)
     -> list of {{"position":(3,), "quaternion":(4,) wxyz, "score":float}}, best first
     MUST pass target_points=<the SELECTED instance's points> so the grasp is on
     THAT instance. Without it, it re-segments and may grasp a DIFFERENT object.
  goto_pose(position, quaternion, z_approach=0.1)   # z_approach raises target z (approach)
  open_gripper(); close_gripper(); goto_home_joint_position()
  recep_pos, _ = get_object_pose("<RECEPTACLE>", use_multiview=True)
     -> recep_pos (3,) = OBB GEOMETRIC center of the receptacle (robust to
     occlusion). Use THIS for the place xy/height. Do NOT use np.median of the
     point cloud -- the receptacle is partly occluded and the median is biased.
  get_object_3d_points_and_masks_from_language("<RECEPTACLE>", use_multiview=False)["points_3d"]
  check_transit_clearance(held_points, overhang, start_xy, goal_xy, transit_tcp_z, obstacles)
  check_place_metrics(surface_points, overhang, place_xy, clearance=0.02)

CORRECT pattern (replace <TARGET>/<RECEPTACLE> with this task's nouns; keep
structure & ordering):

  import numpy as np
  obs = get_observation()
  rgb = obs["agentview"]["images"]["rgb"]
  depth = obs["agentview"]["images"]["depth"]
  K = obs["agentview"]["intrinsics"]; cam = obs["agentview"]["pose_mat"]
  cands_seg = segment_sam3_text_prompt(rgb, "<TARGET>")
  cands_seg = sorted([c for c in cands_seg if c["score"] >= 0.3], key=lambda c: -c["score"])
  sel = 0
  # @verify: sam3_target masks=cands_seg selected=sel object="<TARGET>" relation="<spatial constraint or empty>"
  # the SELECTED instance drives the grasp -- get ITS points, not a fresh text query:
  tp = mask_to_world_points(cands_seg[sel]["mask"], depth, K, cam)
  # Get SEVERAL grasp candidates ON THE SELECTED instance's points (tp). Passing
  # target_points=tp is REQUIRED -- otherwise sample_grasp_candidates re-segments
  # by text and may plan grasps on a DIFFERENT object than the one just verified.
  cands = sample_grasp_candidates("<TARGET>", k=3, target_points=tp)
  gsel = 0
  # @verify: grasp_select candidates=cands selected=gsel object="<TARGET>"
  # grasp_select keeps the highest-confidence candidate (sets gsel) and saves a
  # visualization of all candidates. Use the CHOSEN candidate's POSITION only;
  # grasp TOP-DOWN (its quaternion can be tilted/sideways and would tip the object over):
  position = cands[gsel]["position"]
  down_quat = np.array([0.0, 0.0, 1.0, 0.0])
  # grasp timing: approach ABOVE, descend to grasp point, THEN close, THEN lift
  goto_pose(position, down_quat, z_approach=0.10)    # straight above the object
  goto_pose(position, down_quat, z_approach=0.0)     # straight down at grasp point
  close_gripper()                                    # close ONLY when in place
  goto_pose(position + np.array([0,0,0.15]), down_quat, z_approach=0.0)  # lift
  # PLACE into/onto the receptacle. Use its OBB geometric CENTER (not a
  # point-cloud median) so the object lands centered; release just above it.
  recep_pos, _ = get_object_pose("<RECEPTACLE>", use_multiview=True)
  place_xy = recep_pos[:2]
  goto_pose(np.array([place_xy[0], place_xy[1], position[2] + 0.15]), down_quat, z_approach=0.0)  # above receptacle center
  goto_pose(np.array([place_xy[0], place_xy[1], recep_pos[2] + 0.08]), down_quat, z_approach=0.0)  # down to just above the receptacle
  open_gripper()                                     # release
  goto_pose(np.array([place_xy[0], place_xy[1], recep_pos[2] + 0.20]), down_quat, z_approach=0.0)  # retreat up

Rules:
- Output ONLY executable Python (no fences, no prose, no .save()).
- The grasp MUST target the instance chosen by `sel` (use cands_seg[sel]); never
  let a later block silently re-pick a different instance.
- Close the gripper only AFTER descending to the grasp point, never above it.
- GRASP CHOICE: always get MULTIPLE candidates with sample_grasp_candidates
  (k>=3) and gate them with a `# @verify: grasp_select` marker. grasp_select
  keeps the highest-confidence candidate (sets gsel) and saves a visualization.
  Then grasp the CHOSEN candidate by index (cands[gsel]). Do NOT use
  sample_grasp_pose (it hides the candidates and saves no visualization).
- GRASP ON THE SELECTED INSTANCE: you MUST pass target_points=tp (the points of
  the instance chosen+verified by sam3_target) to sample_grasp_candidates. Never
  let it re-segment by text -- that can plan grasps on a different object. Compute
  tp from cands_seg[sel]["mask"] right after sam3_target.
- GRIPPER ORIENTATION: grasp TOP-DOWN. Use only the POSITION from the chosen
  candidate; for orientation always use the gripper-down quaternion (0,0,1,0)
  wxyz for BOTH grasp and place. The candidate's own quaternion is often tilted
  and will tip the object over -- do NOT feed it to goto_pose.
- Carry and place with the SAME (0,0,1,0) down orientation (never re-tilt mid-carry).
- PLACE CENTER: get the receptacle center from get_object_pose("<RECEPTACLE>",
  use_multiview=True) (OBB geometric center), and place the object there. NEVER
  use np.median(...) for the place xy -- the receptacle is partly occluded so the
  median is biased off-center, which makes the success check fail.
- PLACE HEIGHT: descend to just above the receptacle before releasing -- do not
  drop from high up (the object bounces and rolls off) and do not rely on a
  hand-picked overhang.
- Variable names in @verify markers MUST exist at that point; variables persist.
"""


def query_llm(messages: list[dict], image: np.ndarray | None = None) -> str:
    import requests

    msgs = [dict(m) for m in messages]
    if image is not None:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        msgs[-1] = {"role": "user", "content": [
            {"type": "text", "text": msgs[-1]["content"]},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}
    r = requests.post(_cfg["server_url"],
                      headers={"Authorization": _cfg["api_key"], "Content-Type": "application/json"},
                      json={"model": _cfg.get("model", ""), "messages": msgs,
                            "temperature": 0.0, "max_tokens": 4000}, timeout=300)
    r.raise_for_status()
    m = r.json()["choices"][0]["message"]
    return (m.get("content") or m.get("reasoning_content") or "").strip()


def strip_fences(code: str) -> str:
    code = re.sub(r"^```[a-zA-Z]*\n", "", code.strip())
    return re.sub(r"\n```$", "", code).strip()


def _save_video(env, out_dir: Path) -> None:
    try:
        import imageio

        frames = env.get_video_frames()
        if frames:
            imageio.mimsave(str(out_dir / "episode.mp4"),
                            [np.asarray(f, np.uint8) for f in frames], fps=30)
            print(f"[video] {len(frames)} frames")
        else:
            print("[video] no frames")
    except Exception as e:
        print(f"[video] failed: {e}")


def split_blocks(program: str) -> list[dict]:
    """Split into blocks ending at @verify markers. The trailing code (after the
    last marker) becomes a final unverified block."""
    blocks, cur = [], []
    for line in program.splitlines():
        m = re.match(r"\s*#\s*@verify:\s*(\w+)\s*(.*)", line)
        if m:
            params = dict(re.findall(r"(\w+)\s*=\s*([^\s]+)", m.group(2)))
            for kv in re.findall(r'(\w+)\s*=\s*"([^"]*)"', m.group(2)):
                params[kv[0]] = kv[1]
            blocks.append({"code": "\n".join(cur), "check": m.group(1), "params": params})
            cur = []
        else:
            cur.append(line)
    if any(s.strip() and not s.strip().startswith("#") for s in cur):
        blocks.append({"code": "\n".join(cur), "check": None, "params": {}})
    return blocks


def main(suite_name: str = "libero_object", task_id: int = 0, seed: int = 0) -> None:
    """Run the block-structured self-verifying agent on a LIBERO task.

    Args:
        suite_name: LIBERO suite, e.g. libero_object / libero_spatial /
            libero_goal / libero_10 / libero_90.
        task_id: index of the task within the suite.
        seed: env seed (selects the init state).
    The task description is read from the environment (handle.task_language),
    so nothing about the task is hard-coded here.
    """
    import imageio

    from capx.integrations.franka.libero_evidence import FrankaLiberoApiEvidence
    from capx.envs.simulators.libero import FrankaLiberoEnv

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"loading env: suite={suite_name} task_id={task_id} seed={seed} ...")
    env = FrankaLiberoEnv(suite_name=suite_name, task_id=task_id, seed=seed)
    env.enable_video_capture(True, clear=True)
    api = FrankaLiberoApiEvidence(env)

    task = env.handle.task_language
    system_prompt = build_system_prompt(task)
    print(f"TASK: {task}")
    (OUT / "task.txt").write_text(f"suite={suite_name} task_id={task_id}\n{task}\n")

    ns: dict = {"api": api, "np": np}
    for name, fn in api.functions().items():
        ns[name] = fn

    def ask_vlm(d: Path, question: str, image=None) -> str:
        if image is not None:
            cv2.imwrite(str(d / "vlm_input.png"),
                        cv2.cvtColor(np.asarray(image, np.uint8), cv2.COLOR_RGB2BGR))
        ans = query_llm([{"role": "system", "content": "You are a strict robot-action verifier."},
                         {"role": "user", "content": question}], image)
        (d / "vlm_qa.txt").write_text(f"Q:\n{question}\n\nA:\n{ans}\n")
        return ans

    # initial full program
    program = strip_fences(query_llm(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": "Write the full block-structured program now."}]))
    (OUT / "program_initial.py").write_text(program)

    MAX_PROGRAM_ATTEMPTS = 4
    final_ok = False
    for prog_attempt in range(MAX_PROGRAM_ATTEMPTS):
        adir = OUT / f"program_attempt_{prog_attempt}"
        adir.mkdir(parents=True, exist_ok=True)
        (adir / "program.py").write_text(program)

        blocks = split_blocks(program)
        for b in blocks:  # only active checks run; others execute unverified
            if b["check"] and b["check"] not in ACTIVE_CHECKS:
                b["check"], b["params"] = None, {}
        print(f"[attempt {prog_attempt}] {len(blocks)} blocks, "
              f"{sum(1 for b in blocks if b['check'])} verified")

        # fresh world for every full-program attempt (verify runs on real sim)
        env.reset()
        env.enable_video_capture(True, clear=True)
        ns = {"api": api, "np": np}
        for name, fn in api.functions().items():
            ns[name] = fn

        failure = None  # (block_index, message)
        for i, blk in enumerate(blocks):
            d = adir / f"block_{i:02d}_{blk['check'] or 'plain'}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "code.py").write_text(blk["code"])
            try:
                exec(compile(blk["code"], f"<block_{i}>", "exec"), ns, ns)
            except Exception as e:
                failure = (i, f"execution error: {type(e).__name__}: {e}")
                print(f"[attempt {prog_attempt}] block {i} exec error: {e}")
                break
            if blk["check"]:
                result = REGISTRY[blk["check"]](ns, blk["params"], d, ask_vlm)
                (d / "decision.json").write_text(json.dumps(
                    {"check": blk["check"], "ok": result.ok,
                     "message": result.message}, indent=2, default=str))
                print(f"[attempt {prog_attempt}] block {i} @{blk['check']}: "
                      f"{'PASS' if result.ok else 'FAIL'} - {result.message}")
                if result.ok:
                    # A PASS may still carry a chosen index (e.g. grasp_select
                    # falling back to the top-scored candidate, or sam3_target
                    # confirming the selection). Write it back so the index
                    # variable the program uses downstream is consistent.
                    sel_var = blk["params"].get("selected")
                    chosen = (result.data or {}).get("selected")
                    if sel_var and chosen is not None and ns.get(sel_var) != chosen:
                        print(f"[attempt {prog_attempt}] block {i}: set "
                              f"{sel_var}={chosen} (from PASS)")
                        ns[sel_var] = chosen
                if not result.ok:
                    # IN-PLACE CORRECTION: if the check returned a corrected
                    # selection index (and it's not an "uncertain"/out-of-range
                    # case), write it back into the SAME namespace and re-verify,
                    # then continue this very execution -- no reset, no rewrite,
                    # so the corrected index stays valid against THIS segmentation.
                    sel_var = blk["params"].get("selected")
                    corrected = (result.data or {}).get("selected")
                    uncertain = bool((result.data or {}).get("uncertain"))
                    if sel_var and corrected is not None and not uncertain:
                        print(f"[attempt {prog_attempt}] block {i}: in-place fix "
                              f"{sel_var}={ns.get(sel_var)} -> {corrected}, re-verifying")
                        ns[sel_var] = corrected
                        result = REGISTRY[blk["check"]](ns, blk["params"], d, ask_vlm)
                        (d / "decision_recheck.json").write_text(json.dumps(
                            {"check": blk["check"], "ok": result.ok,
                             "message": result.message, "corrected_to": corrected},
                            indent=2, default=str))
                        print(f"[attempt {prog_attempt}] block {i} recheck: "
                              f"{'PASS' if result.ok else 'FAIL'} - {result.message}")
                    if not result.ok:
                        # genuinely unrecoverable here (uncertain target, or the
                        # correction still failed) -> fall back to full rewrite.
                        failure = (i, f"@{blk['check']} failed: {result.message}")
                        break

        if failure is None:
            final_ok = True
            break

        # rewrite the WHOLE program with full context + the failure
        fail_i, fail_msg = failure
        rewrite_prompt = (
            "Your program was executed block by block on the real simulator and "
            f"failed at block {fail_i}.\n\nFULL PROGRAM:\n{program}\n\n"
            f"FAILURE at block {fail_i}: {fail_msg}\n\n"
            "Rewrite the ENTIRE program to fix this, keeping all blocks "
            "consistent with each other (later blocks must use the variables and "
            "decisions from earlier blocks). Keep the # @verify markers. "
            "Output only executable Python, no fences, no prose.")
        program = strip_fences(query_llm(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": rewrite_prompt}]))
        print(f"[attempt {prog_attempt}] program rewritten after block {fail_i} failure")

    if not final_ok:
        print("[verify] program never fully passed")
        (OUT / "result.json").write_text(json.dumps(
            {"verified": False, "attempts": MAX_PROGRAM_ATTEMPTS}, indent=2))
        _save_video(env, OUT)
        return

    # ---- the verified run IS the executed run (no reset, no re-segmentation) ----
    # The last attempt executed every block to completion on the real sim while
    # verifying (and applying in-place corrections) along the way, so the world
    # is already in its final state and `ns` holds the exact bowls/sel/grasp that
    # were verified. Resetting and re-running would re-segment and let the chosen
    # index drift, which is the bug we are removing. We just record the result of
    # THIS run -- what was verified is what was executed.
    final_program = program
    (OUT / "final_program.py").write_text(final_program)
    print("[verify] full program passed and executed in-place; wrote final_program.py")

    completed = False
    for _ in range(6):
        api.open_gripper()
        if env.task_completed():
            completed = True
            break
    print(f"[done] task_completed={completed}")
    (OUT / "result.json").write_text(json.dumps(
        {"verified": True, "executed": True, "exec_error": None,
         "task_completed": completed}, indent=2))
    _save_video(env, OUT)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
