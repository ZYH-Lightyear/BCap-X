"""Verifier-as-code for *Estimate Object Geometry*.

Run by the Verifier INSIDE the CapX sandbox: the orchestrator ``exec``s this
source into the executor's persistent global namespace, so the L4 API functions
injected there (``get_observation``, ``segment_sam3_text_prompt``,
``mask_to_world_points``, ``filter_noise``,
``get_oriented_bounding_box_from_3d_points``) are reachable as globals.

Verification target (docs §5.6, "A" handoff): it verifies the **executor's real
artifacts**, read from the manifest the executor leaves in ``RESULT`` (scalars +
inline OBB + on-disk ``.npy`` paths for the heavy ``points``/``mask``). It loads
those arrays, projects the measured OBB + top/bottom points back onto the
agentview RGB (the *provenance* of the height), saves the overlay, and asks a VLM
to judge it against ``ref/verify.md``.

If no manifest is present it falls back to an INDEPENDENT re-measurement (segment
+ geometry from the object name), flagged ``evidence_source="reproduced"`` so a
foreign/stale result never silently stands in for the executor's.

Entry point: ``verify(object_name, out_dir, rubric_path=..., ...)``.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_DEFAULT_RUBRIC = (
    "PASS only if the green oriented bounding box tightly encloses the named "
    "object and nothing else, and the measured height spans the object from its "
    "true top down to where it rests on the table (not the table surface itself)."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _api(name: str):
    """Fetch an L4 API function from the sandbox's persistent global namespace."""
    fn = globals().get(name)
    if fn is None:
        raise RuntimeError(
            f"API function '{name}' not in sandbox namespace; verify.py must be "
            "exec'd inside the CapX executor globals, not imported standalone."
        )
    return fn


def measure(object_name: str, camera: str = "agentview"):
    """Independently segment the object and recover its world points + OBB."""
    get_observation = _api("get_observation")
    segment = _api("segment_sam3_text_prompt")
    mask_to_world_points = _api("mask_to_world_points")
    filter_noise = _api("filter_noise")
    get_obb = _api("get_oriented_bounding_box_from_3d_points")

    obs = get_observation()
    cam = obs[camera]
    rgb = np.asarray(cam["images"]["rgb"]).astype(np.uint8)
    depth = np.asarray(cam["images"]["depth"])

    results = segment(rgb, text_prompt=object_name)
    if not results:
        raise RuntimeError(f"segmentation returned no mask for '{object_name}'")
    mask = max(results, key=lambda r: r["score"])["mask"]

    points = mask_to_world_points(mask, depth, cam["intrinsics"], cam["pose_mat"])
    points, _ = filter_noise(points)
    points = np.asarray(points)
    if len(points) == 0:
        raise RuntimeError(f"no 3D points survived filtering for '{object_name}'")
    obb = get_obb(points)

    top_z = float(points[:, 2].max())
    bottom_z = float(points[:, 2].min())
    stats = {
        "object": object_name,
        "n_points": int(len(points)),
        "height": top_z - bottom_z,
        "top_z": top_z,
        "bottom_z": bottom_z,
        "extent": [float(x) for x in np.asarray(obb["extent"]).reshape(-1)],
    }
    return cam, rgb, np.asarray(mask), points, obb, stats


def load_claim(camera: str = "agentview", manifest_path: str | None = None):
    """Load the executor's actual artifacts from its manifest (A-plan handoff).

    Reads the manifest from a JSON file if ``manifest_path`` is given, else from
    the sandbox global ``RESULT``. Heavy arrays come back via ``np.load`` of the
    recorded paths; the OBB is reconstructed from its inline floats. Returns the
    same 6-tuple as ``measure``, or ``None`` if no usable manifest exists.
    """
    if manifest_path and Path(manifest_path).is_file():
        claim = json.loads(Path(manifest_path).read_text())
    else:
        claim = globals().get("RESULT")
    if not isinstance(claim, dict) or "points_path" not in claim:
        return None

    points = np.asarray(np.load(claim["points_path"]))
    mask_path = claim.get("mask_path")
    if mask_path and Path(mask_path).is_file():
        mask = np.asarray(np.load(mask_path))
    else:
        mask = None

    obb_d = claim.get("obb") or {}
    obb = {
        "center": np.asarray(obb_d.get("center", points.mean(axis=0))),
        "extent": np.asarray(obb_d.get("extent", points.max(axis=0) - points.min(axis=0))),
        "R": np.asarray(obb_d.get("R", np.eye(3))),
    }

    obs = _api("get_observation")()
    cam = obs[camera]
    rgb = np.asarray(cam["images"]["rgb"]).astype(np.uint8)
    if mask is None:
        mask = np.zeros(rgb.shape[:2], dtype=bool)

    top_z = float(claim.get("top_z", points[:, 2].max()))
    bottom_z = float(claim.get("bottom_z", points[:, 2].min()))
    stats = {
        "object": claim.get("object", "?"),
        "n_points": int(claim.get("n_points", len(points))),
        "height": float(claim.get("height", top_z - bottom_z)),
        "top_z": top_z,
        "bottom_z": bottom_z,
        "extent": [float(x) for x in np.asarray(obb["extent"]).reshape(-1)],
    }
    return cam, rgb, np.asarray(mask), points, obb, stats


def numeric_guards(stats: dict) -> tuple[bool, list[str]]:
    """Cheap deterministic checks; catch absurd values before spending a VLM call."""
    flags: list[str] = []
    if stats["n_points"] < 50:
        flags.append(f"too few points ({stats['n_points']})")
    if not (stats["top_z"] > stats["bottom_z"]):
        flags.append("inverted height (top_z <= bottom_z)")
    ext = stats["extent"]
    if any(e < 0.01 for e in ext) or any(e > 0.5 for e in ext):
        flags.append(f"implausible extents {ext}")
    return (len(flags) == 0), flags


def _project_points(points: np.ndarray, world_to_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """World points -> integer pixel coords (only those in front of the camera)."""
    pts = np.asarray(points)
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    cam = (world_to_cam @ hom.T).T[:, :3]
    z = cam[:, 2].copy()
    in_front = z > 1e-6
    uv = (K @ cam.T).T
    px = uv[:, :2] / np.where(z == 0, 1e-6, z)[:, None]
    return px[in_front].astype(int)


def _mark_point(draw, point: np.ndarray, world_to_cam: np.ndarray, K: np.ndarray,
                color: tuple, text: str, wh: tuple) -> None:
    """Draw a ringed marker + text label at a single world point's projection."""
    w, h = wh
    cam = world_to_cam @ np.append(np.asarray(point, dtype=float), 1.0)
    if cam[2] <= 1e-6:
        return  # behind the camera
    uv = K @ cam[:3]
    x, y = int(uv[0] / cam[2]), int(uv[1] / cam[2])
    if not (0 <= x < w and 0 <= y < h):
        return
    r = 5
    draw.ellipse([x - r, y - r, x + r, y + r], outline=(0, 0, 0), width=3)
    draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
    tx = min(x + 8, w - 70)  # keep the label inside the frame
    ty = max(y - 6, 18)
    draw.text((tx + 1, ty + 1), text, fill=(0, 0, 0))
    draw.text((tx, ty), text, fill=color)


def render_evidence(cam: dict, rgb: np.ndarray, mask: np.ndarray, points: np.ndarray,
                    obb: dict, stats: dict, out_path) -> str:
    """Render the full provenance chain onto the agentview RGB.

    Layers (so a wrong-object grab is unmistakable):
      1. the SAM mask, translucent yellow (what segmentation actually grabbed);
      2. the filtered 3D points reprojected as red dots (what fed the geometry /
         the height's z-range);
      3. the measured oriented bounding box in green;
      4. the estimated highest (cyan) and lowest (magenta) points, each labelled
         with its world-frame z, so the height's endpoints are visible;
      5. a height/extent text label.
    """
    from capx.utils.visualization_utils import (
        draw_oriented_bounding_box,
        overlay_segmentation_masks,
    )

    K = np.asarray(cam["intrinsics"])
    world_to_cam = np.linalg.inv(np.asarray(cam["pose_mat"]))  # pose_mat is cam->world

    base = overlay_segmentation_masks(np.asarray(rgb), [mask.astype(bool)], opacity=0.5)
    img = draw_oriented_bounding_box(base, obb, world_to_cam, K)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    h, w = img.shape[:2]

    px = _project_points(points, world_to_cam, K)
    step = max(1, len(px) // 400)  # cap dot count so the overlay stays readable
    for x, y in px[::step]:
        if 0 <= x < w and 0 <= y < h:
            draw.point((int(x), int(y)), fill=(255, 0, 0))

    # Mark the actual top/bottom points that define the measured height.
    pts = np.asarray(points)
    top_pt = pts[int(pts[:, 2].argmax())]
    bottom_pt = pts[int(pts[:, 2].argmin())]
    _mark_point(draw, top_pt, world_to_cam, K, (0, 255, 255),
                f"top z={stats['top_z']:.3f}", (w, h))
    _mark_point(draw, bottom_pt, world_to_cam, K, (255, 0, 255),
                f"bot z={stats['bottom_z']:.3f}", (w, h))

    label = (
        f"{stats['object']}  h={stats['height']:.3f}m  "
        f"top={stats['top_z']:.3f} bot={stats['bottom_z']:.3f}  n={stats['n_points']}"
    )
    draw.rectangle([0, 0, pil.width, 16], fill=(0, 0, 0))
    draw.text((3, 3), label, fill=(255, 255, 0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)
    return str(out_path)


def judge(image_path: str, rubric: str, stats: dict, model: str, server_url: str,
          api_key: str | None = None) -> dict:
    """VLM-as-code: read the overlay and return a JSON verdict against the rubric."""
    from capx.llm.client import ModelQueryArgs, query_model

    data = base64.b64encode(Path(image_path).read_bytes()).decode()
    system = (
        "You are a robot perception verifier. You see one image: an agentview RGB "
        "with a GREEN oriented bounding box and a measured-height label. Judge ONLY "
        "what is visible. Reply with one JSON object: "
        '{"verdict":"passed"|"failed"|"uncertain","confidence":0.0-1.0,"reason":"..."}'
    )
    text = (
        f"Object: {stats['object']}\n"
        f"Measured: height={stats['height']:.3f} m, top_z={stats['top_z']:.3f}, "
        f"bottom_z={stats['bottom_z']:.3f}, extent={stats['extent']}\n\n"
        f"Success rubric:\n{rubric}\n\n"
        "Does the geometry estimate look trustworthy? JSON verdict only."
    )
    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
        ]},
    ]
    args = ModelQueryArgs(
        model=model, server_url=server_url, api_key=api_key, temperature=0.0, max_tokens=512,
    )
    content = query_model(args, prompt)["content"]
    match = _JSON_RE.search(content)
    if not match:
        return {"verdict": "uncertain", "confidence": 0.0, "reason": "unparseable judge reply"}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": "uncertain", "confidence": 0.0, "reason": "malformed judge JSON"}
    return {
        "verdict": str(payload.get("verdict", "uncertain")).lower(),
        "confidence": float(payload.get("confidence", 0.0)),
        "reason": str(payload.get("reason", "")),
    }


def verify(
    object_name: str,
    out_dir: str,
    rubric_path: str | None = None,
    camera: str = "agentview",
    model: str = "openrouter/qwen/qwen3.6-plus",
    server_url: str = "http://localhost:8110/chat/completions",
    api_key: str | None = None,
    use_vlm: bool = True,
    manifest_path: str | None = None,
    allow_reproduce: bool = True,
) -> dict:
    """End-to-end: load executor artifacts -> numeric guards -> render -> VLM judge.

    Primary path verifies the executor's REAL products via its manifest
    (``RESULT`` / ``manifest_path``). Falls back to an independent re-measurement
    only if no manifest exists (``evidence_source="reproduced"``). Prints a single
    ``VERIFY_RESULT <json>`` line and returns the same dict.
    """
    result: dict = {"object": object_name}

    artifacts = None
    try:
        artifacts = load_claim(camera=camera, manifest_path=manifest_path)
    except Exception as exc:  # corrupt manifest -> note and fall back
        result["claim_load_error"] = repr(exc)

    if artifacts is not None:
        result["evidence_source"] = "agent"
    elif allow_reproduce:
        try:
            artifacts = measure(object_name, camera=camera)
            result["evidence_source"] = "reproduced"
        except Exception as exc:  # re-measurement failed -> hard FAIL
            result.update(verdict="failed", confidence=1.0,
                          reason=f"measure error: {exc!r}", evidence_source="reproduced")
            print("VERIFY_RESULT " + json.dumps(result))
            return result
    else:
        result.update(verdict="uncertain", confidence=0.0, evidence_source="missing",
                      reason="no executor manifest (RESULT) found and reproduce disabled")
        print("VERIFY_RESULT " + json.dumps(result))
        return result

    cam, rgb, mask, points, obb, stats = artifacts
    result["object"] = stats["object"]
    result.update(stats)
    ok, flags = numeric_guards(stats)
    result["numeric_ok"] = ok
    result["numeric_flags"] = flags

    result["overlay"] = render_evidence(
        cam, rgb, mask, points, obb, stats, Path(out_dir) / "geometry_overlay.png",
    )

    rubric = _DEFAULT_RUBRIC
    if rubric_path and Path(rubric_path).is_file():
        rubric = Path(rubric_path).read_text()

    if not ok:
        result.update(verdict="failed", confidence=1.0,
                      reason="numeric guards: " + "; ".join(flags))
    elif use_vlm:
        try:
            result.update(judge(result["overlay"], rubric, stats, model, server_url, api_key))
        except Exception as exc:
            result.update(verdict="uncertain", confidence=0.0, reason=f"vlm judge error: {exc!r}")
    else:
        result.update(verdict="uncertain", confidence=0.0, reason="vlm disabled")

    print("VERIFY_RESULT " + json.dumps(result))
    return result
