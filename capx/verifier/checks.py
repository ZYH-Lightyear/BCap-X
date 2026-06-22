"""Verify-point registry for the block-structured VLM agent.

The VLM writes one Python program organized in blocks. A block that produces
something checkable ends with a marker comment:

    # @verify: <check_type> key=var_name key2=var_name2 ...

When the driver reaches such a marker, it dispatches to the matching check
here. Each check reads the named variables from the execution namespace,
produces evidence, decides pass/fail (visual checks consult the VLM; metric
checks compute numbers + rules), and returns a VerifyResult. On failure the
driver hands the block + message back to the VLM to rewrite.

All check types and their expected params are documented in
``VERIFY_CATALOG`` and surfaced to the VLM in the system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from capx.utils.pointcloud_render import draw_grasp_point_on_image, pose_from_position_wxyz


@dataclass
class VerifyResult:
    ok: bool
    message: str
    evidence_png: Path | None = None
    data: dict | None = None


# Human-readable catalog injected into the system prompt.
VERIFY_CATALOG = """\
Available verification points (declare with a comment AFTER the block):

  # @verify: sam3_target masks=<var> selected=<var> object="<text>" relation="<text>"
      Checks relation-aware target selection. <masks> is the SAM3 results list,
      <selected> is your chosen index (int). A VLM judge confirms the selected
      candidate is the one matching the relation, or returns the correct index.
      On change, your block must re-read the corrected index from the message.

  # @verify: grasp_select candidates=<var> selected=<var> object="<text>"
      Picks the HIGHEST-CONFIDENCE grasp candidate (index 0 of the score-sorted
      <candidates> from sample_grasp_candidates) and writes it back to your
      <selected> index. Also saves a numbered, color-coded visualization of all
      candidate grasp points (chosen one highlighted) for inspection. No VLM call.

  # @verify: grasp position=<var> quaternion=<var> target_points=<var>
      Checks a single planned grasp. Renders the grasp point + approach arrow on
      the target. A VLM judge approves or rejects (e.g. point off the target).

  # @verify: holding object="<text>" tcp_z=<var> pre_grasp_top_z=<var>
      After lifting, re-segments the object and checks it actually rose with
      the gripper. Fails if the object was not picked up.

  # @verify: transit held_points=<var> overhang=<var> start_xy=<var> goal_xy=<var> transit_tcp_z=<var> obstacles=<var>
      Geometric corridor-clearance check for the carry motion. <obstacles> is a
      dict name->(N,3) points. Reports min_safe_tcp_z if too low.

  # @verify: place surface_points=<var> overhang=<var> place_xy=<var>
      Geometric place check. Computes release_tcp_z = surface_z + overhang +
      clearance and whether place_xy is within 3cm of the receptacle centre
      (the task success predicate). Returns release_tcp_z.
"""


def _api(ns: dict) -> Any:
    api = ns.get("api")
    if api is None:
        raise RuntimeError("namespace has no 'api'")
    return api


def _mask_points(api, mask) -> np.ndarray:
    obs = api.get_observation()
    cam = obs[api.camera_name]
    depth = np.asarray(cam["images"]["depth"])
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    K, pose = cam["intrinsics"], cam["pose_mat"]
    ys, xs = np.where(np.asarray(mask).astype(bool))
    z = depth[ys, xs]
    ok = (z > 0.01) & (z < 3.0)
    x = (xs[ok] - K[0, 2]) * z[ok] / K[0, 0]
    y = (ys[ok] - K[1, 2]) * z[ok] / K[1, 1]
    p = np.stack([x, y, z[ok]], axis=-1)
    return p @ pose[:3, :3].T + pose[:3, 3]


def _draw_numbered_boxes(rgb, results, selected):
    import cv2

    out = np.ascontiguousarray(np.asarray(rgb).copy())
    for i, r in enumerate(results):
        box = r.get("box")
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        # all boxes use the same color -- do NOT reveal the pre-selected one
        color = (255, 220, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # Put the number OUTSIDE the box (just above its top-left corner) so it
        # never overlaps the object. If there is no room above, drop it just
        # below the bottom edge instead.
        label = str(i + 1)
        h, w = out.shape[:2]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        tx = x1
        ty = y1 - 6                      # baseline just above the top edge
        if ty - th < 0:                  # no room above -> place below the box
            ty = min(y2 + th + 6, h - 2)
        tx = max(0, min(tx, w - tw - 1))
        # filled label chip in the box color for contrast, dark text on top
        cv2.rectangle(out, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), color, -1)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return out


# ------------------------------- checks ------------------------------- #
def check_sam3_target(ns, params, d: Path, ask_vlm) -> VerifyResult:
    import re

    api = _api(ns)
    masks = ns[params["masks"]]
    selected = int(ns[params["selected"]])
    obj = params.get("object", "target")
    relation = params.get("relation", "")
    # SAM3 open-vocab returns many low-score masks; keep only confident,
    # reasonably-sized candidates so the numbered overlay stays readable.
    ranked = sorted(enumerate(masks), key=lambda kv: -kv[1].get("score", 0))
    keep = [(i, m) for i, m in ranked if m.get("score", 0) >= 0.3][:5]
    if not keep:
        keep = ranked[:3]
    idx_map = [i for i, _ in keep]          # display position -> original index
    disp_results = [m for _, m in keep]
    try:
        disp_selected = idx_map.index(selected)
    except ValueError:
        disp_selected = 0
    rgb = api.get_observation()[api.camera_name]["images"]["rgb"]
    img = _draw_numbered_boxes(rgb, disp_results, disp_selected)
    q = (
        f"Target object: '{obj}'. Spatial constraint: '{relation}'.\n"
        f"The image shows {len(disp_results)} candidate boxes; each box's number is "
        "printed just OUTSIDE its top-left corner.\n"
        "Decide which numbered box is the correct instance under the spatial "
        "constraint.\n"
        "FIRST write one short sentence of reasoning. THEN, on the LAST line, write "
        "EXACTLY 'SELECT: <number>'. The SELECT line MUST be last."
    )
    ans = ask_vlm(d, q, img)
    # Reasoning comes first, so take the LAST 'SELECT: <n>' in the answer (the
    # required final line), not the first match which could appear in prose.
    matches = re.findall(r"SELECT\s*[:#]?\s*(\d+)", ans, re.IGNORECASE)
    if not matches:
        return VerifyResult(
            False,
            "UNCERTAIN: the verifier did not return a parseable 'SELECT: <n>' "
            "decision. Re-run target selection (e.g. tighten the SAM3 prompt or "
            "reduce candidates) and make the choice explicit.",
            d / "vlm_input.png", {"uncertain": True})
    disp_choice = int(matches[-1]) - 1
    if not (0 <= disp_choice < len(idx_map)):
        return VerifyResult(False, f"verifier returned out-of-range box {disp_choice + 1}",
                            d / "vlm_input.png", {"uncertain": True})
    corrected = idx_map[disp_choice]
    if corrected == selected:
        return VerifyResult(True, f"target (orig index {selected}) confirmed",
                            d / "vlm_input.png", {"selected": selected})
    return VerifyResult(
        False,
        f"WRONG TARGET: you selected original index {selected} but the correct "
        f"instance is original index {corrected}. Set your selected index to "
        f"{corrected} and continue.",
        d / "vlm_input.png", {"selected": corrected})


def _draw_numbered_grasps(rgb, cam, candidates, selected):
    """Draw each grasp candidate as a colored dot + approach arrow + number on
    the agentview image. Selected candidate is highlighted; numbers are 1-based.

    candidates: list of dicts with "position" (3,) and "quaternion" (4,) wxyz.
    """
    import cv2

    from capx.utils.pointcloud_render import HIGHLIGHT_PALETTE

    out = np.ascontiguousarray(np.asarray(rgb).copy())
    K, pose = cam["intrinsics"], cam["pose_mat"]
    for i, c in enumerate(candidates):
        color = tuple(HIGHLIGHT_PALETTE[i % len(HIGHLIGHT_PALETTE)])
        pos = np.asarray(c["position"], dtype=float)
        quat = np.asarray(c["quaternion"], dtype=float)
        out = draw_grasp_point_on_image(
            out, pose_from_position_wxyz(pos, quat), K, pose,
            color=color, label=f"#{i + 1}" + (" SEL" if i == selected else ""))
    return out


def check_grasp_select(ns, params, d: Path, ask_vlm) -> VerifyResult:
    """Pick the highest-confidence grasp candidate (no VLM judgement).

    The candidates from sample_grasp_candidates are already sorted by descending
    grasp score, so the best one is index 0. This check just selects it and
    saves a visualization of all candidates (numbered, color-coded, with the
    chosen one highlighted) for inspection -- it never calls the VLM.

    Returns the chosen index in data["selected"] so the driver writes it back to
    the program's grasp-index variable.

    Params:
      candidates=<var>  list of {"position","quaternion","score"} dicts (sorted)
      selected=<var>    int, the grasp-index variable the program will use
      object="<text>"   optional, only used in the saved image's filename/log
    """
    api = _api(ns)
    candidates = ns[params["candidates"]]
    if not candidates:
        return VerifyResult(False, "no grasp candidates to select from",
                            None, {"uncertain": True})

    best = 0  # candidates are score-sorted; highest confidence is index 0
    # save the candidate visualization (highlight the chosen one) for evidence
    cam = api.get_observation()[api.camera_name]
    img = _draw_numbered_grasps(cam["images"]["rgb"], cam, candidates, best)
    try:
        import cv2

        cv2.imwrite(str(d / "grasp_candidates.png"),
                    cv2.cvtColor(np.asarray(img, np.uint8), cv2.COLOR_RGB2BGR))
    except Exception:
        pass

    score = candidates[best].get("score")
    score_str = f"{score:.3f}" if score is not None else "n/a"
    return VerifyResult(
        True,
        f"selected highest-confidence grasp #1 (score {score_str}) of "
        f"{len(candidates)} candidates",
        d / "grasp_candidates.png", {"selected": best})


def check_grasp(ns, params, d: Path, ask_vlm) -> VerifyResult:
    api = _api(ns)
    pos = np.asarray(ns[params["position"]], dtype=float)
    quat = np.asarray(ns[params["quaternion"]], dtype=float)
    cam = api.get_observation()[api.camera_name]
    img = draw_grasp_point_on_image(
        np.asarray(cam["images"]["rgb"]), pose_from_position_wxyz(pos, quat),
        cam["intrinsics"], cam["pose_mat"], color=(0, 220, 0), label="planned grasp")
    q = (
        "Green dot = planned grasp point, arrow = approach direction.\n"
        "Is the grasp point on the intended target object (on its rim/body, not "
        "another object, not empty space)?\n"
        "Reply EXACTLY 'APPROVE' or 'REJECT', then one sentence."
    )
    ans = ask_vlm(d, q, img)
    ok = ans.strip().upper().startswith("APPROVE") or (
        "APPROVE" in ans.upper() and "REJECT" not in ans.upper())
    msg = "grasp approved" if ok else (
        "GRASP REJECTED: the grasp point is not on the target. Re-plan the grasp "
        "(pick a different candidate whose point lies on the target) and continue.")
    return VerifyResult(ok, msg, d / "vlm_input.png")


def check_holding(ns, params, d: Path, ask_vlm) -> VerifyResult:
    api = _api(ns)
    obj = params.get("object", "object")
    tcp_z = float(ns[params["tcp_z"]])
    pre_top = float(ns[params["pre_grasp_top_z"]])
    try:
        res = api.get_object_3d_points_and_masks_from_language(obj, use_multiview=False)
        pts = np.asarray(res["points_3d"])
        lifted = len(pts) > 100 and float(np.percentile(pts[:, 2], 98)) > pre_top + 0.05
    except Exception:
        lifted = False
    if lifted:
        return VerifyResult(True, "object is held (rose with the gripper)", None)
    return VerifyResult(
        False,
        "NOT HOLDING: after lifting, the object did not rise with the gripper, so "
        "the grasp failed. Re-plan and re-execute the grasp (different candidate).",
        None)


def check_transit(ns, params, d: Path, ask_vlm) -> VerifyResult:
    api = _api(ns)
    rep = api.check_transit_clearance(
        ns[params["held_points"]], float(ns[params["overhang"]]),
        ns[params["start_xy"]], ns[params["goal_xy"]],
        float(ns[params["transit_tcp_z"]]), ns[params["obstacles"]])
    if rep["ok"]:
        return VerifyResult(True, "transit height clears all obstacles", None, rep)
    return VerifyResult(
        False,
        f"TRANSIT UNSAFE: raise the transit TCP height to at least "
        f"{rep['min_safe_tcp_z'] * 100:.1f}cm and continue.",
        None, rep)


def check_place(ns, params, d: Path, ask_vlm) -> VerifyResult:
    api = _api(ns)
    rep = api.check_place_metrics(
        ns[params["surface_points"]], float(ns[params["overhang"]]),
        ns[params["place_xy"]], clearance=float(params.get("clearance", 0.02)))
    if rep["ok"]:
        return VerifyResult(
            True,
            f"place ok; use release_tcp_z={rep['release_tcp_z']:.3f} m",
            None, rep)
    return VerifyResult(
        False,
        "PLACE INVALID: place_xy must be within 3cm of the receptacle centre "
        "(task success requires bowl-centre within 3cm of plate-centre). Set "
        "place_xy to the receptacle centre and continue.",
        None, rep)


REGISTRY: dict[str, Callable[..., VerifyResult]] = {
    "sam3_target": check_sam3_target,
    "grasp_select": check_grasp_select,
    "grasp": check_grasp,
    "holding": check_holding,
    "transit": check_transit,
    "place": check_place,
}
