"""Build the policy-visible snapshot consumed by the read-only Web renderer.

Robot geometry and camera projection stay in Python.  The browser receives
already-rendered RGB evidence plus small semantic labels; it never receives
depth, camera matrices, masks, point clouds, or privileged task state.
"""

from __future__ import annotations

import base64
import contextlib
import io
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vaw.camera import DEFAULT_FOV_DEG, fit_physical_camera, orbit_camera
from vaw.cloud import SceneCloud, splat_cloud
from vaw.gripper_mesh import mask_outline
from vaw.render import (
    Painter,
    _crop_around,
    _draw_gripper_fk,
    _obb_corners,
    _object_colors,
    _orbit_target,
    _quat_to_matrix,
    _sync_view_base,
    _tint_indices,
)
from vaw.state import ActionState
from vaw.types import Candidate, ObjectEntry, Pose

SCENE_SIZE = (686, 382)
FOCUS_SIZE = (286, 142)
CANDIDATE_SIZE = (132, 66)
SELF_SIZE = (132, 62)

_SURFACE = (236, 240, 247)
_INK = (23, 32, 51)
_BLUE = (37, 99, 235)
_GREEN = (22, 163, 74)
_VIOLET = (124, 58, 237)
_AMBER = (217, 119, 6)
_RED = (220, 38, 38)
_WHITE = (255, 255, 255)


def _font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont:
    names = (
        ("DejaVuSansMono.ttf", "DejaVuSans.ttf")
        if mono
        else ("DejaVuSans.ttf", "DejaVuSansMono.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


F_LABEL = _font(12)
F_SMALL = _font(10, mono=True)


def _data_url(image: Image.Image | np.ndarray) -> str:
    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    buffer = io.BytesIO()
    pil.convert("RGB").save(buffer, format="PNG", optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    *,
    font: ImageFont.FreeTypeFont = F_LABEL,
) -> tuple[int, int, int, int]:
    x, y = int(round(xy[0])), int(round(xy[1]))
    box = draw.textbbox((x, y), text, font=font)
    padded = (box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2)
    draw.rounded_rectangle(padded, radius=3, fill=(*_WHITE, 226), outline=(*color, 190))
    draw.text((x, y), text, fill=color, font=font)
    return padded


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _scene_base(
    state: ActionState,
    obs: dict[str, Any] | None,
    cloud: SceneCloud,
    camera_name: str,
) -> tuple[Image.Image, Painter]:
    width, height = SCENE_SIZE
    colors = _object_colors(state)
    image = Image.new("RGBA", SCENE_SIZE, (*_SURFACE, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    cam_obs = obs.get(camera_name) if obs else None

    if state.view.is_physical and cam_obs is not None:
        camera, (off_x, off_y, paste_w, paste_h) = fit_physical_camera(
            cam_obs, width, height
        )
        frame = Image.fromarray(np.asarray(cam_obs["images"]["rgb"], dtype=np.uint8))
        image.paste(frame.resize((paste_w, paste_h)), (off_x, off_y))
        for object_id, entry in state.objects.items():
            if entry.mask is None:
                continue
            mask = Image.fromarray(np.asarray(entry.mask, dtype=np.uint8) * 255)
            mask = mask.resize((paste_w, paste_h), Image.Resampling.NEAREST)
            layer = np.zeros((paste_h, paste_w, 4), dtype=np.uint8)
            selected = np.asarray(mask) > 0
            layer[selected] = (*colors[object_id], 48)
            image.alpha_composite(Image.fromarray(layer), (off_x, off_y))
    else:
        target = _orbit_target(state, cloud)
        camera = orbit_camera(
            target,
            state.view.azimuth_deg,
            state.view.elevation_deg,
            state.view.base_distance_m,
            width,
            height,
            fov_deg=DEFAULT_FOV_DEG / max(state.view.zoom, 0.1),
        )
        rendered = splat_cloud(
            cloud,
            camera,
            background=_SURFACE,
            tints=_tint_indices(cloud, state, colors),
        )
        image.paste(Image.fromarray(rendered), (0, 0))

    return image, Painter(draw, camera, _region(width, height), margin=10)


def _region(width: int, height: int):
    # Importing Region here avoids making the Web snapshot part of render.py's
    # fixed legacy layout while still reusing its projection-aware Painter.
    from vaw.render import Region

    return Region(0, 0, width, height)


def _draw_object_boxes(
    draw: ImageDraw.ImageDraw,
    painter: Painter,
    state: ActionState,
) -> list[tuple[int, int, int, int]]:
    labels: list[tuple[int, int, int, int]] = []
    colors = _object_colors(state)
    for object_id, entry in state.objects.items():
        corners = _obb_corners(entry.obb) if entry.obb is not None else None
        if corners is None:
            continue
        projected = painter.project(corners)
        visible = projected[projected[:, 2] > 0.05]
        if len(visible) < 2:
            continue
        x0 = int(np.clip(visible[:, 0].min(), 1, SCENE_SIZE[0] - 2))
        y0 = int(np.clip(visible[:, 1].min(), 1, SCENE_SIZE[1] - 2))
        x1 = int(np.clip(visible[:, 0].max(), 1, SCENE_SIZE[0] - 2))
        y1 = int(np.clip(visible[:, 1].max(), 1, SCENE_SIZE[1] - 2))
        stale = entry.obs_revision < state.obs_revision
        color = _RED if stale else colors[object_id]
        draw.rectangle((x0, y0, x1, y1), outline=(*color, 230), width=2)
        suffix = " STALE" if stale else ""
        labels.append(
            _label(draw, (x0 + 3, max(y0 - 17, 3)), f"{object_id}{suffix}", color)
        )
    return labels


def _draw_candidate_anchors(
    draw: ImageDraw.ImageDraw,
    painter: Painter,
    state: ActionState,
    occupied: list[tuple[int, int, int, int]] | None = None,
) -> None:
    occupied = list(occupied or [])
    candidates = _visible_candidates(state)
    for index, candidate in enumerate(candidates):
        point = painter.point(candidate.pose.position)
        if point is None:
            continue
        x, y = point
        stale = candidate.obs_revision < state.obs_revision
        selected = candidate.candidate_id == state.selected_id
        color = _RED if stale else (_GREEN if selected else _VIOLET)
        radius = 9 if selected else 7
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(*_WHITE, 150),
            outline=(*color, 255),
            width=3 if selected else 2,
        )
        draw.line((x - 4, y, x + 4, y), fill=color, width=2)
        draw.line((x, y - 4, x, y + 4), fill=color, width=2)

        text = f"{candidate.candidate_id} TCP" if selected else candidate.candidate_id
        label_x = min(x + radius + 5, SCENE_SIZE[0] - 74)
        label_y = max(4, min(y - radius - 4, SCENE_SIZE[1] - 22))
        # Candidate IDs must not merge into one visual token. Move only labels;
        # the ring remains on the true projected target.
        for _ in range(12):
            estimate = (
                int(label_x - 3),
                int(label_y - 3),
                int(label_x + 68),
                int(label_y + 15),
            )
            if not any(_overlaps(estimate, other) for other in occupied):
                break
            label_y = 5 + ((int(label_y) + 19 + index * 3) % (SCENE_SIZE[1] - 25))
        box = _label(draw, (label_x, label_y), text, color, font=F_SMALL)
        occupied.append(box)


def _render_scene(
    state: ActionState,
    obs: dict[str, Any] | None,
    cloud: SceneCloud,
    camera_name: str,
) -> str:
    image, painter = _scene_base(state, obs, cloud, camera_name)
    draw = ImageDraw.Draw(image, "RGBA")
    painter.draw = draw
    object_labels = _draw_object_boxes(draw, painter, state)
    _draw_candidate_anchors(draw, painter, state, object_labels)

    if state.arm_joint_positions_rad is not None:
        opening = state.gripper_opening if state.gripper_opening is not None else 1.0
        _draw_gripper_fk(
            image,
            painter,
            state.arm_joint_positions_rad,
            opening,
            _BLUE,
            fill_alpha=24,
        )
    return _data_url(image)


def _crop_camera(
    cam_obs: dict[str, Any],
    center_world: np.ndarray,
    *,
    span_m: float,
    size: tuple[int, int],
) -> tuple[Image.Image, Painter, tuple[float, float, float, float]]:
    crop = _crop_around(cam_obs, center_world, span_m)
    camera, (off_x, off_y, paste_w, paste_h) = fit_physical_camera(
        cam_obs, size[0], size[1], crop=crop
    )
    source = Image.fromarray(np.asarray(cam_obs["images"]["rgb"], dtype=np.uint8))
    patch = source.crop(tuple(int(round(v)) for v in crop))
    image = Image.new("RGBA", size, (*_SURFACE, 255))
    image.paste(patch.resize((paste_w, paste_h)), (off_x, off_y))
    painter = Painter(ImageDraw.Draw(image, "RGBA"), camera, _region(*size), margin=0)
    return image, painter, crop


def _focus_snapshot(
    state: ActionState,
    obs: dict[str, Any] | None,
    camera_name: str,
) -> dict[str, Any] | None:
    object_id = state.focus_id
    entry = state.objects.get(object_id) if object_id else None
    cam_obs = obs.get(camera_name) if obs else None
    if entry is None or cam_obs is None or entry.centroid_world is None:
        return None

    span = 0.14
    if entry.obb is not None:
        with contextlib.suppress(KeyError, TypeError, ValueError):
            span = float(np.max(np.asarray(entry.obb["extent"], dtype=np.float64))) * 2.0
    span = float(np.clip(span, 0.10, 0.45))
    image, painter, crop = _crop_camera(
        cam_obs,
        entry.centroid_world,
        span_m=span,
        size=FOCUS_SIZE,
    )
    draw = ImageDraw.Draw(image, "RGBA")
    painter.draw = draw

    if entry.mask is not None:
        source_mask = Image.fromarray(np.asarray(entry.mask, dtype=np.uint8) * 255)
        patch = source_mask.crop(tuple(int(round(v)) for v in crop))
        patch = patch.resize(FOCUS_SIZE, Image.Resampling.NEAREST)
        selected = np.asarray(patch) > 0
        if selected.any():
            layer = np.zeros((*selected.shape, 4), dtype=np.uint8)
            layer[selected] = (*_BLUE, 40)
            layer[mask_outline(selected)] = (*_BLUE, 255)
            image.alpha_composite(Image.fromarray(layer))

    corners = _obb_corners(entry.obb) if entry.obb is not None else None
    if corners is not None:
        projected = painter.project(corners)
        visible = projected[projected[:, 2] > 0.05]
        if len(visible) >= 2:
            x0 = int(np.clip(visible[:, 0].min(), 1, FOCUS_SIZE[0] - 2))
            y0 = int(np.clip(visible[:, 1].min(), 1, FOCUS_SIZE[1] - 2))
            x1 = int(np.clip(visible[:, 0].max(), 1, FOCUS_SIZE[0] - 2))
            y1 = int(np.clip(visible[:, 1].max(), 1, FOCUS_SIZE[1] - 2))
            draw.rectangle((x0, y0, x1, y1), outline=(*_BLUE, 220), width=2)

    stale = entry.obs_revision < state.obs_revision
    return {
        "objectId": entry.object_id,
        "name": entry.name,
        "image": _data_url(image),
        "stale": stale,
        "hasMask": entry.mask is not None,
        "hasObb": entry.obb is not None,
        "revision": entry.obs_revision,
    }


def _candidate_image(
    candidate: Candidate,
    state: ActionState,
    obs: dict[str, Any] | None,
    camera_name: str,
) -> str | None:
    cam_obs = obs.get(camera_name) if obs else None
    if cam_obs is None:
        return None
    image, painter, _ = _crop_camera(
        cam_obs,
        candidate.pose.position,
        span_m=0.16,
        size=CANDIDATE_SIZE,
    )
    draw = ImageDraw.Draw(image, "RGBA")
    painter.draw = draw
    preview = state.previews.get(candidate.candidate_id)
    color = _GREEN if candidate.candidate_id == state.selected_id else _VIOLET

    if preview is not None and preview.ik_ok and preview.joint_positions_rad is not None:
        _draw_gripper_fk(
            image,
            painter,
            preview.joint_positions_rad,
            1.0 if state.gripper_open else 0.0,
            _VIOLET,
            fill_alpha=58,
        )
    else:
        rotation = _quat_to_matrix(candidate.pose.quat_wxyz)
        approach = rotation @ np.array([0.0, 0.0, 1.0])
        tail = candidate.pose.position - 0.065 * approach
        painter.segment(tail, candidate.pose.position, _VIOLET, width=3)

    point = painter.point(candidate.pose.position)
    if point is not None:
        x, y = point
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(*_WHITE, 190), outline=color, width=3)
        draw.line((x - 4, y, x + 4, y), fill=color, width=2)
        draw.line((x, y - 4, x, y + 4), fill=color, width=2)
    return _data_url(image)


def _current_gripper_image(
    state: ActionState,
    obs: dict[str, Any] | None,
    camera_name: str,
) -> str | None:
    if state.ee_pose is None:
        return None
    cam_obs = obs.get(camera_name) if obs else None
    if cam_obs is None:
        return None
    image, painter, _ = _crop_camera(
        cam_obs,
        state.ee_pose.position,
        span_m=0.18,
        size=SELF_SIZE,
    )
    if state.arm_joint_positions_rad is not None:
        opening = state.gripper_opening if state.gripper_opening is not None else 1.0
        _draw_gripper_fk(
            image,
            painter,
            state.arm_joint_positions_rad,
            opening,
            _BLUE,
            fill_alpha=48,
        )
    return _data_url(image)


def _wrist_image(
    obs: dict[str, Any] | None,
    wrist_camera_name: str,
) -> str | None:
    cam = obs.get(wrist_camera_name) if obs else None
    if cam is None or "rgb" not in cam.get("images", {}):
        return None
    return _data_url(np.asarray(cam["images"]["rgb"], dtype=np.uint8))


def _opening_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 0.65:
        return "open"
    if value <= 0.15:
        return "closed"
    return "partial"


def _candidate_status(candidate: Candidate, state: ActionState) -> dict[str, Any]:
    preview = state.previews.get(candidate.candidate_id)
    ik = "unchecked" if preview is None else "pass" if preview.ik_ok else "fail"
    return {
        "id": candidate.candidate_id,
        "kind": candidate.kind,
        "objectId": candidate.object_id,
        "selected": candidate.candidate_id == state.selected_id,
        "stale": candidate.obs_revision < state.obs_revision,
        "ik": ik,
        "image": None,
    }


def _visible_candidates(state: ActionState) -> list[Candidate]:
    """Five selector slots ordered by current evidence, not episode age.

    A physical observation makes old candidates stale.  If another grounding
    then proposes ``g6..g10``, showing the episode's first five would hide every
    actionable current candidate.  The stale selection remains visible in the
    dedicated Intent panel; the selector belongs to current evidence.
    """

    values = list(state.candidates.values())
    fresh = [candidate for candidate in values if candidate.obs_revision == state.obs_revision]
    if fresh:
        # When several tools add candidates in one revision, the newest group
        # is the one the agent just asked to compare. Preserve its generation
        # order after taking the tail.
        return fresh[-5:]

    ordered: list[Candidate] = []
    selected = state.selected
    if selected is not None:
        ordered.append(selected)
    if state.focus_id is not None:
        ordered.extend(state.candidates_of(state.focus_id))
    ordered.extend(reversed(values))
    unique: dict[str, Candidate] = {}
    for candidate in ordered:
        unique.setdefault(candidate.candidate_id, candidate)
    return list(unique.values())[:5]


def _receipt_snapshot(state: ActionState) -> dict[str, Any] | None:
    if not state.receipts:
        return None
    receipt = state.receipts[-1]
    return {
        "id": receipt.receipt_id,
        "op": receipt.op,
        "candidateId": receipt.candidate_id,
        "unpredictedFailure": receipt.unpredicted_failure,
        "hasDiscrepancy": bool(receipt.discrepancy),
    }


def build_web_snapshot(
    state: ActionState,
    obs: dict[str, Any] | None,
    *,
    camera_name: str,
    wrist_camera_name: str,
    cloud: SceneCloud,
    render_id: str = "",
) -> dict[str, Any]:
    """Return the minimal JSON-able visual snapshot for ``vaw-ui``."""

    _sync_view_base(state, obs, cloud, camera_name)
    candidates = []
    for candidate in _visible_candidates(state):
        item = _candidate_status(candidate, state)
        item["image"] = _candidate_image(candidate, state, obs, camera_name)
        candidates.append(item)

    selected = state.selected
    selected_preview = (
        state.previews.get(selected.candidate_id) if selected is not None else None
    )
    focus = _focus_snapshot(state, obs, camera_name)
    opening = (
        float(np.clip(state.gripper_opening, 0.0, 1.0))
        if state.gripper_opening is not None
        else None
    )
    return {
        "schemaVersion": 1,
        "renderId": render_id,
        "viewport": {"width": 1024, "height": 576},
        "header": {
            "task": state.instruction,
            "revision": state.obs_revision,
            "view": state.view.preset,
            "source": (
                "physical RGB"
                if state.view.is_physical
                else "reconstructed point cloud"
            ),
            "selectedId": state.selected_id,
        },
        "scene": {
            "image": _render_scene(state, obs, cloud, camera_name),
            "objectCount": len(state.objects),
            "candidateCount": len(state.candidates),
        },
        "focus": focus,
        "self": {
            "wristImage": _wrist_image(obs, wrist_camera_name),
            "currentImage": _current_gripper_image(state, obs, camera_name),
            "gripperState": _opening_band(state.gripper_opening),
            "opening": opening,
            "revision": state.obs_revision,
            "jointsObserved": state.arm_joint_positions_rad is not None,
        },
        "intent": {
            "selectedId": state.selected_id,
            "kind": selected.kind if selected is not None else None,
            "nextImage": (
                _candidate_image(selected, state, obs, camera_name)
                if selected is not None
                else None
            ),
            "ik": (
                "unchecked"
                if selected_preview is None
                else ("pass" if selected_preview.ik_ok else "fail")
            ),
            "trajectory": "not_checked",
            "collision": "not_checked",
        },
        "candidates": candidates,
        "receipt": _receipt_snapshot(state),
    }


__all__ = ["build_web_snapshot"]
