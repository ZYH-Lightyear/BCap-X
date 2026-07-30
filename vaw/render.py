"""Deterministic canvas rendering (numpy + PIL only; no browser, no GPU).

The canvas is the visual half of the action state, laid out as one frame:

    +--------------------------------------------+--------------+
    | header: rev | gripper | sel | view | instr  |  DataPanel   |
    |                                            +--------------+
    |  main view: physical RGB, or the scene      |    Focus     |
    |  point cloud from a virtual viewpoint,      +--------------+
    |  with sparse annotations + axis gizmo       |    wrist     |
    +--------------------------------------------+--------------+

Two rules decide what goes where (see ``docs/vaw_implementation_plan.md`` §1.4):

1. **The canvas carries spatial relations; numbers live in the state summary.**
   A VLM reads prompt text far more reliably than small text rendered into
   pixels, so nothing here prints a coordinate.
2. **Annotation density is graded by zoom.** The main view stays sparse (the
   selected candidate plus a few alternatives) to stay legible; the focus inset
   is a single object's viewport and can afford every candidate with its
   approach axis. Density is spent on what the agent acts on: the focus inset
   drops the object box precisely because it is already cropped to that object.

Rendering is a pure function of (state, observation, cloud): the same trace
always produces the same pixels, which is what makes canvases usable as SFT
inputs and RL observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vaw.camera import (
    DEFAULT_FOV_DEG,
    VirtualCamera,
    fit_physical_camera,
    orbit_camera,
    spherical_of,
)
from vaw.cloud import SceneCloud, build_scene_cloud, splat_cloud
from vaw.geometry import project_world_to_pixel
from vaw.gripper_mesh import (
    load_panda_urdf_fk,
    mask_outline,
    rasterize_silhouette,
)
from vaw.state import ActionState

CANVAS_W, CANVAS_H = 1024, 576


@dataclass(frozen=True)
class Region:
    """A rectangle of the canvas that one camera or panel owns."""

    x0: int
    y0: int
    w: int
    h: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x0 + self.w - 1, self.y0 + self.h - 1)


MAIN = Region(0, 0, 768, CANVAS_H)
PANEL = Region(772, 4, 248, 206)
FOCUS = Region(772, 214, 248, 186)
WRIST = Region(772, 404, 248, 168)
HEADER_H = 22

# Object colours are assigned by insertion order and never reused within an
# episode, so obj1 is the same colour in every canvas of a trace.
_PALETTE = [
    (86, 156, 255), (255, 106, 96), (250, 196, 62), (72, 199, 130),
    (191, 134, 240), (74, 208, 214), (255, 148, 88), (154, 205, 90),
]
_KIND_COLOR = {"grasp": (250, 196, 62), "place": (74, 208, 214), "waypoint": (168, 176, 192)}
_SELECTED = (74, 240, 130)
_WARN = (255, 140, 70)
_BAD = (255, 92, 92)
_TEXT = (232, 234, 240)
_MUTED = (150, 156, 170)
_BG = (22, 23, 28)
_PANEL_BG = (30, 32, 38)

#: Main view keeps at most this many unselected candidates. The rest are listed
#: in the data panel — clutter costs more than completeness on a 768px view.
MAIN_VIEW_CANDIDATES = 4


def _font(size: int, mono: bool = True) -> ImageFont.FreeTypeFont:
    names = (
        ["DejaVuSansMono.ttf", "DejaVuSans.ttf"] if mono else ["DejaVuSans.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


F_HEADER = _font(14)
F_BODY = _font(13)
F_SMALL = _font(11)


class Painter:
    """A camera bound to a canvas region: projects world points to canvas pixels.

    ``margin`` is how far outside its region the painter may draw. The main view
    allows a little overhang so a marker at the edge is still hinted at, but the
    insets use zero: an annotation escaping the focus box would land on top of a
    neighbouring panel and read as belonging to it.
    """

    def __init__(
        self,
        draw: ImageDraw.ImageDraw,
        cam: VirtualCamera,
        region: Region,
        *,
        margin: int = 40,
    ) -> None:
        self.draw = draw
        self.cam = cam
        self.region = region
        self.margin = margin

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """(N, 3) world -> (N, 3) [u, v, z_cam] in canvas coordinates."""
        uvz = project_world_to_pixel(
            np.asarray(points_world).reshape(-1, 3), self.cam.intrinsics, self.cam.pose_mat
        )
        uvz[:, 0] += self.region.x0
        uvz[:, 1] += self.region.y0
        return uvz

    def point(self, world: np.ndarray) -> tuple[float, float] | None:
        u, v, z = self.project(np.asarray(world).reshape(1, 3))[0]
        if z <= 0.05 or not self.inside(u, v):
            return None
        return float(u), float(v)

    def inside(self, u: float, v: float, margin: int | None = None) -> bool:
        m = self.margin if margin is None else margin
        r = self.region
        return r.x0 - m <= u <= r.x0 + r.w + m and r.y0 - m <= v <= r.y0 + r.h + m

    def label(self, xy: tuple[float, float], text: str, color, font=None) -> None:
        """Draw a label, nudged so its box cannot leave the painter's region."""
        font = font or F_SMALL
        width = self.draw.textlength(text, font=font)
        x = min(xy[0], self.region.x0 + self.region.w - width - 4)
        y = min(max(xy[1], self.region.y0 + 2), self.region.y0 + self.region.h - font.size - 4)
        _label(self.draw, (max(x, self.region.x0 + 2), y), text, color, font=font)

    def polyline(self, points_world: np.ndarray, color, width: int = 2, dash: int = 0) -> None:
        uvz = self.project(points_world)
        pts = [(float(u), float(v)) for u, v, z in uvz if z > 0.05]
        if len(pts) < 2:
            return
        if dash:
            for i in range(0, len(pts) - 1, 2):
                self.draw.line([pts[i], pts[i + 1]], fill=color, width=width)
        else:
            self.draw.line(pts, fill=color, width=width)

    def segment(self, a: np.ndarray, b: np.ndarray, color, width: int = 2) -> None:
        pa, pb = self.point(a), self.point(b)
        if pa and pb:
            self.draw.line([pa, pb], fill=color, width=width)


# ===================================================================== #
# Small drawing helpers                                                  #
# ===================================================================== #

def _label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color,
    font: ImageFont.FreeTypeFont = F_SMALL,
    anchor_box: bool = True,
) -> None:
    x, y = xy
    if anchor_box:
        w = draw.textlength(text, font=font)
        draw.rectangle((x - 2, y - 1, x + w + 3, y + font.size + 3), fill=(*_BG, 205))
    draw.text((x, y), text, fill=color, font=font)


def _obb_corners(obb: dict[str, Any]) -> np.ndarray | None:
    """8 corners of an oriented box, or None if the dict is not usable."""
    try:
        center = np.asarray(obb["center"], dtype=np.float64).reshape(3)
        extent = np.asarray(obb["extent"], dtype=np.float64).reshape(3)
        rot = np.asarray(obb["R"], dtype=np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError):
        return None
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64
    )
    return center[None, :] + (rot @ (signs * extent[None, :] / 2).T).T


def _draw_bbox(painter: Painter, obb: dict[str, Any], color, width: int = 1) -> None:
    """Screen rectangle spanning the projected oriented box.

    The box is only ever an identity cue — "this region is obj1". Its twelve 3D
    edges also state an orientation, but at the size objects occupy here those
    edges run across the object and hide the very texture the box points at, and
    the orientation they encode is the fitter's frame rather than anything the
    agent acts on. Four edges around the same projected extent keep the claim and
    give the pixels back.
    """
    corners = _obb_corners(obb)
    if corners is None:
        return
    uvz = painter.project(corners)
    visible = uvz[uvz[:, 2] > 0.05]
    if len(visible) < 2:
        return
    u0, v0 = visible[:, 0].min(), visible[:, 1].min()
    u1, v1 = visible[:, 0].max(), visible[:, 1].max()
    if not (painter.inside(u0, v0) or painter.inside(u1, v1)):
        return
    r = painter.region
    u0, u1 = np.clip([u0, u1], r.x0, r.x0 + r.w - 1)
    v0, v1 = np.clip([v0, v1], r.y0, r.y0 + r.h - 1)
    painter.draw.rectangle((u0, v0, u1, v1), outline=color, width=width)


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _draw_gripper_fk(
    img: Image.Image,
    painter: Painter,
    joint_positions_rad: np.ndarray,
    opening: float,
    color,
    *,
    fill_alpha: int,
) -> bool:
    """Fill the exact joint→URDF-FK hand silhouette. False if unavailable.

    A stick figure places the gripper but says nothing about whether the fingers
    clear what is next to them, or how wide they are relative to the object —
    which is most of what the glyph exists to support. The outline is drawn
    opaque over a translucent fill so the evidence underneath stays readable:
    this marks an intended pose, not an object that is there.
    """
    try:
        fk = load_panda_urdf_fk()
        if fk is None:
            return False
        tris = fk.triangles(joint_positions_rad, opening)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False
    uvz = painter.project(tris.reshape(-1, 3))
    mask = rasterize_silhouette(
        uvz[:, :2].reshape(-1, 3, 2), uvz[:, 2].reshape(-1, 3), img.width, img.height
    )
    # Confine it to the painter's own region, or a gripper near the edge of the
    # main view would spill over the side panels.
    region = np.zeros_like(mask)
    r = painter.region
    region[r.y0 : r.y0 + r.h, r.x0 : r.x0 + r.w] = True
    mask &= region
    if not mask.any():
        return False

    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub = mask[y0:y1, x0:x1]
    layer = np.zeros((*sub.shape, 4), dtype=np.uint8)
    layer[sub] = (*color, fill_alpha)
    layer[mask_outline(sub)] = (*color, 255)
    img.alpha_composite(Image.fromarray(layer, "RGBA"), (x0, y0))
    return True


def _draw_gripper(painter: Painter, pose, open_width: float, color, width: int = 3) -> None:
    """A 3D gripper glyph: stem along the approach axis + two fingers.

    The fallback for when the baked hand mesh is missing. Drawn in 3D rather than
    as a 2D crosshair because the whole point of the virtual viewpoint is that
    orientation is legible from more than one angle.
    """
    rot = _quat_to_matrix(pose.quat_wxyz)
    approach = rot @ np.array([0.0, 0.0, 1.0])
    across = rot @ np.array([0.0, 1.0, 0.0])
    p = np.asarray(pose.position, dtype=np.float64)
    half = max(open_width, 0.012)
    base = p - 0.055 * approach
    bar_a, bar_b = base + half * across, base - half * across
    painter.segment(p - 0.095 * approach, base, color, width=width)
    painter.segment(bar_a, bar_b, color, width=width)
    painter.segment(bar_a, bar_a + 0.05 * approach, color, width=width)
    painter.segment(bar_b, bar_b + 0.05 * approach, color, width=width)


def _draw_candidate(
    painter: Painter,
    cand,
    color,
    *,
    selected: bool,
    preview_flag: str = "",
    approach_axis: bool = False,
) -> None:
    pt = painter.point(cand.pose.position)
    if pt is None:
        return
    u, v = pt
    r = 9 if selected else 6
    painter.draw.ellipse((u - r, v - r, u + r, v + r), outline=color, width=3 if selected else 2)
    if approach_axis:
        rot = _quat_to_matrix(cand.pose.quat_wxyz)
        tail = cand.pose.position - 0.06 * (rot @ np.array([0.0, 0.0, 1.0]))
        painter.segment(tail, cand.pose.position, color, width=1)
    painter.label((u + r + 3, v - r - 2), cand.candidate_id + preview_flag, color)


# ===================================================================== #
# Main view                                                              #
# ===================================================================== #

def _object_colors(state: ActionState) -> dict[str, tuple[int, int, int]]:
    return {
        oid: _PALETTE[i % len(_PALETTE)] for i, oid in enumerate(state.objects)
    }


def _orbit_target(state: ActionState, cloud: SceneCloud) -> np.ndarray:
    """What the virtual camera orbits: the focused object, else the scene.

    Focus drives framing as well as the inset, so ``inspect`` then ``view`` does
    what it looks like it should: orbit the thing you are working on. The scene
    centre used to anchor the presets stays separate, so those angles do not
    shift under the agent when the focus changes.
    """
    focus_id, _ = state.focus_target()
    if focus_id:
        entry = state.objects.get(focus_id)
        centroid = entry.centroid_world if entry is not None else None
        if centroid is not None:
            return centroid
    return state.view.target if state.view.target is not None else cloud.center


def _tint_indices(
    cloud: SceneCloud, state: ActionState, colors: dict[str, tuple[int, int, int]]
) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """Which cloud points to tint for each grounded object.

    Membership is a geometric containment test against the object's oriented box
    rather than a re-run of segmentation: an approximate tint is enough to say
    "this blob is obj1". The box matters over a simple radius — a sphere sized
    to hold a tall object also swallows the table around it, and a tint bleeding
    onto the table is a claim about the scene that is simply false.
    """
    tints: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    if len(cloud) == 0:
        return tints
    for oid, entry in state.objects.items():
        pts = entry.points_world
        corners = _obb_corners(entry.obb) if entry.obb is not None else None
        if corners is not None:
            center = np.asarray(entry.obb["center"], dtype=np.float64).reshape(3)
            extent = np.asarray(entry.obb["extent"], dtype=np.float64).reshape(3)
            rot = np.asarray(entry.obb["R"], dtype=np.float64).reshape(3, 3)
            # The box is fitted to the surface one camera could see, so it stops
            # short of the object's far side. The proportional slack lets points
            # another camera contributed still count as the same object.
            margin = 0.008 + 0.1 * extent
            local = (cloud.points - center[None, :]) @ rot
            inside = np.all(np.abs(local) <= (extent / 2 + margin)[None, :], axis=1)
            near = np.flatnonzero(inside)
        elif pts is not None and len(pts):
            centroid = np.median(pts, axis=0)
            radius = float(np.percentile(np.linalg.norm(pts - centroid, axis=1), 90)) + 0.008
            near = np.flatnonzero(np.linalg.norm(cloud.points - centroid, axis=1) <= radius)
        else:
            continue
        if len(near):
            tints.append((near, colors[oid]))
    return tints


def _render_main(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    state: ActionState,
    obs: dict[str, Any] | None,
    cloud: SceneCloud,
    camera_name: str,
    colors: dict[str, tuple[int, int, int]],
) -> Painter:
    """Fill the main region and return its painter (annotations drawn by caller)."""
    view = state.view
    cam_obs = obs.get(camera_name) if obs else None

    if view.is_physical and cam_obs is not None:
        cam, (ox, oy, pw, ph) = fit_physical_camera(cam_obs, MAIN.w, MAIN.h)
        frame = Image.fromarray(np.asarray(cam_obs["images"]["rgb"], dtype=np.uint8))
        img.paste(frame.resize((pw, ph)), (MAIN.x0 + ox, MAIN.y0 + oy))
        painter = Painter(draw, cam, MAIN)
        for oid, entry in state.objects.items():
            if entry.mask is None:
                continue
            _overlay_mask(img, entry.mask, colors[oid], (ox, oy, pw, ph), MAIN)
        return painter

    # Orbit around whatever is being looked at, and zoom by narrowing the field
    # of view rather than by moving the camera in: pulling the eye closer walks
    # it into the table and pushes the target out of frame, while a longer lens
    # magnifies with the viewing geometry unchanged.
    target = _orbit_target(state, cloud)
    cam = orbit_camera(
        target,
        view.azimuth_deg,
        view.elevation_deg,
        view.base_distance_m,
        MAIN.w,
        MAIN.h,
        fov_deg=DEFAULT_FOV_DEG / max(view.zoom, 0.1),
    )
    rendered = splat_cloud(cloud, cam, background=_BG, tints=_tint_indices(cloud, state, colors))
    img.paste(Image.fromarray(rendered), (MAIN.x0, MAIN.y0))
    return Painter(draw, cam, MAIN)


def _overlay_mask(
    img: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    paste: tuple[int, int, int, int],
    region: Region,
) -> None:
    """Tint a source-resolution mask onto the letterboxed frame."""
    ox, oy, pw, ph = paste
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return
    layer = np.zeros((*m.shape, 4), dtype=np.uint8)
    layer[m] = (*color, 96)
    tinted = Image.fromarray(layer, "RGBA").resize((pw, ph), Image.NEAREST)
    img.alpha_composite(tinted, (region.x0 + ox, region.y0 + oy))


def _annotate_main(
    img: Image.Image,
    painter: Painter,
    state: ActionState,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    # Object boxes and labels: identity cues that survive any viewpoint.
    for oid, entry in state.objects.items():
        color = colors[oid]
        stale = entry.obs_revision < state.obs_revision
        if entry.obb is not None:
            _draw_bbox(painter, entry.obb, (*color, 150) if stale else color)
        centroid = entry.centroid_world
        if centroid is None:
            continue
        pt = painter.point(centroid)
        if pt is None:
            continue
        u, v = pt
        painter.draw.line([(u, v), (u + 22, v + 26)], fill=_MUTED, width=1)
        text = f"{oid} {entry.name}" + (" (stale)" if stale else "")
        # Labels lean below the object; candidate markers cluster above a grasp
        # target, and two overlapping labels are worse than a longer leader.
        painter.label((u + 24, v + 22), text, color, font=F_BODY)

    sel = state.selected

    # Candidates: selected always, plus the best few others.
    others = [c for c in state.candidates.values() if c.candidate_id != state.selected_id]
    others.sort(key=lambda c: -c.score)
    shown = others[:MAIN_VIEW_CANDIDATES]
    for cand in shown:
        _draw_candidate(
            painter, cand, _KIND_COLOR.get(cand.kind, _TEXT), selected=False,
            preview_flag=_preview_flag(state, cand.candidate_id),
        )
    if sel is not None:
        _draw_candidate(
            painter, sel, _SELECTED, selected=True,
            preview_flag=_preview_flag(state, sel.candidate_id), approach_axis=True,
        )

    # Virtual gripper (the pending action) and, when they differ, the real arm.
    # Absent until the agent aims somewhere: a placeholder pose would put the
    # brightest glyph on the canvas at the world origin, standing for a decision
    # nobody made, and would drag the real-arm glyph in with it (a phantom at the
    # origin always differs from the arm, so the "when they differ" test passes).
    virtual = state.virtual_gripper
    if virtual is None:
        return
    opening = state.gripper_opening if state.gripper_opening is not None else 1.0
    preview = state.previews.get(state.selected_id) if state.selected_id else None
    target_joints = preview.joint_positions_rad if preview is not None else None
    if target_joints is None or not _draw_gripper_fk(
        img,
        painter,
        target_joints,
        1.0 if state.gripper_open else 0.0,
        (255, 255, 255),
        fill_alpha=90,
    ):
        # Before preview there is no IK configuration to render exactly.  Keep a
        # deliberately schematic glyph rather than inventing a panda_hand pose
        # from the candidate with another hand-written TCP transform.
        _draw_gripper(painter, virtual, 0.04 if state.gripper_open else 0.014, (255, 255, 255))
    show_current = state.ee_pose is not None and (
        np.linalg.norm(state.ee_pose.position - virtual.position) > 0.02
    )
    if show_current and (
        state.arm_joint_positions_rad is None
        or not _draw_gripper_fk(
            img,
            painter,
            state.arm_joint_positions_rad,
            opening,
            _MUTED,
            fill_alpha=48,
        )
    ):
        _draw_gripper(painter, state.ee_pose, max(0.04 * opening, 0.012), _MUTED, width=2)


def _preview_flag(state: ActionState, candidate_id: str) -> str:
    preview = state.previews.get(candidate_id)
    if preview is None:
        return ""
    return " IK" if preview.ik_ok else " IK!"


def _draw_gizmo(painter: Painter, state: ActionState, cloud: SceneCloud) -> None:
    """World axis compass, bottom-left of the main view.

    Without it a rotated view makes "move it left" ambiguous — the agent's
    nudge/propose_pose arguments are world-frame, so it needs to see where the
    world axes point in the picture it is looking at.
    """
    cam = painter.cam
    # Large enough to remain legible after the full 1024x576 canvas is resized
    # for a VLM.  The previous 28 px / 2 px compass nearly disappeared in the
    # physical agentview, especially for a foreshortened axis.
    ox, oy = MAIN.x0 + 64, MAIN.y0 + MAIN.h - 64
    length = 42
    painter.draw.ellipse(
        (ox - 3, oy - 3, ox + 3, oy + 3),
        fill=(220, 224, 234),
        outline=(22, 23, 28),
        width=1,
    )
    for axis, color, name in (
        (np.array([1.0, 0, 0]), (255, 106, 96), "X"),
        (np.array([0, 1.0, 0]), (72, 199, 130), "Y"),
        (np.array([0, 0, 1.0]), (86, 156, 255), "Z"),
    ):
        du = float(np.dot(axis, cam.right)) * length
        dv = float(np.dot(axis, cam.down)) * length
        painter.draw.line([(ox, oy), (ox + du, oy + dv)], fill=color, width=3)
        painter.draw.text(
            (ox + du * 1.2 - 4, oy + dv * 1.2 - 7), name, fill=color, font=F_BODY
        )


# ===================================================================== #
# Right strip: data panel, focus inset, wrist inset                      #
# ===================================================================== #

def _panel_frame(draw: ImageDraw.ImageDraw, region: Region, title: str) -> int:
    draw.rectangle(region.box, fill=_PANEL_BG, outline=(64, 68, 78))
    draw.text((region.x0 + 6, region.y0 + 4), title, fill=_MUTED, font=F_SMALL)
    return region.y0 + 20


def _render_panel(
    draw: ImageDraw.ImageDraw,
    state: ActionState,
    colors: dict[str, tuple[int, int, int]],
    focus_id: str | None,
) -> None:
    """The legend: which marker on the canvas is which id, and its status.

    Deliberately not a data table — a pose printed at 11px is a number the model
    will misread, and it is already in the state summary. What the panel adds is
    the marker-to-id binding the canvas cannot state in words.
    """
    y = _panel_frame(draw, PANEL, "DataPanel  id -> marker")
    x = PANEL.x0 + 8
    line_h = 16
    y_limit = PANEL.y0 + PANEL.h - 6

    rows: list[tuple[str, tuple[int, int, int], str]] = []
    for oid, entry in state.objects.items():
        flags = []
        if oid == focus_id:
            flags.append("focus")
        if entry.obs_revision < state.obs_revision:
            flags.append("stale")
        suffix = f"  [{' '.join(flags)}]" if flags else ""
        rows.append(("box", colors[oid], f"{oid} {entry.name[:16]}{suffix}"))

    cands = sorted(state.candidates.values(), key=lambda c: (c.kind, -c.score))
    for cand in cands:
        selected = cand.candidate_id == state.selected_id
        color = _SELECTED if selected else _KIND_COLOR.get(cand.kind, _TEXT)
        bits = [cand.candidate_id, cand.kind[:5]]
        if cand.score:
            bits.append(f"{cand.score:.2f}")
        if selected:
            bits.append("SEL")
        preview = state.previews.get(cand.candidate_id)
        if preview is not None:
            bits.append("IK" if preview.ik_ok else "IK-fail")
        rows.append(("ring", color, " ".join(bits)))

    for glyph, color, text in rows:
        if y + line_h > y_limit:
            draw.text((x, y), f"... +{len(rows) - rows.index((glyph, color, text))} more", fill=_MUTED, font=F_SMALL)
            break
        cy = y + 6
        if glyph == "box":
            draw.rectangle((x, cy - 5, x + 9, cy + 4), fill=(*color, 120), outline=color)
        else:
            draw.ellipse((x, cy - 5, x + 9, cy + 4), outline=color, width=2)
        draw.text((x + 15, y), text, fill=_TEXT, font=F_SMALL)
        y += line_h

    # Robot row, pinned to the bottom: entity zero's own state.
    opening = state.gripper_opening
    robot = f"robot  gripper {'open' if state.gripper_open else 'closed'}"
    if opening is not None:
        robot += f" ({opening:.2f})"
    draw.line(
        [(PANEL.x0 + 4, y_limit - 16), (PANEL.x0 + PANEL.w - 5, y_limit - 16)],
        fill=(64, 68, 78),
        width=1,
    )
    draw.text((x, y_limit - 13), robot, fill=_MUTED, font=F_SMALL)


def _render_focus(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    state: ActionState,
    obs: dict[str, Any] | None,
    cloud: SceneCloud,
    camera_name: str,
    colors: dict[str, tuple[int, int, int]],
    focus_id: str | None,
    explicit: bool,
) -> None:
    """Zoomed viewport on one object, with the dense annotations the main view
    cannot afford: every candidate of that object, each with its approach axis."""
    draw.rectangle(FOCUS.box, fill=_PANEL_BG, outline=(64, 68, 78))
    entry = state.objects.get(focus_id) if focus_id else None
    centroid = entry.centroid_world if entry is not None else None
    if entry is None or centroid is None:
        draw.text(
            (FOCUS.x0 + 6, FOCUS.y0 + 4), "Focus  (call inspect to zoom)", fill=_MUTED, font=F_SMALL
        )
        return

    inner = Region(FOCUS.x0 + 1, FOCUS.y0 + 1, FOCUS.w - 2, FOCUS.h - 2)
    # Frame the object with room around it: a crop tight to the object shows its
    # texture but not its relation to the gripper or the surface it sits on,
    # which is usually the reason to look closely in the first place.
    span = 0.14
    if entry.obb is not None:
        try:
            span = float(np.max(np.asarray(entry.obb["extent"], dtype=np.float64))) * 2.2
        except (KeyError, TypeError, ValueError):
            span = 0.14
    span = float(np.clip(span, 0.10, 0.45))

    cam_obs = obs.get(camera_name) if obs else None
    if state.view.is_physical and cam_obs is not None:
        crop = _crop_around(cam_obs, centroid, span)
        cam, (ox, oy, pw, ph) = fit_physical_camera(cam_obs, inner.w, inner.h, crop=crop)
        frame = Image.fromarray(np.asarray(cam_obs["images"]["rgb"], dtype=np.uint8))
        patch = frame.crop((int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])))
        img.paste(patch.resize((pw, ph)), (inner.x0 + ox, inner.y0 + oy))
        painter = Painter(draw, cam, inner, margin=0)
    else:
        # Magnify with a narrow lens from a safe standoff instead of putting the
        # eye centimetres from the object, where it ends up inside neighbouring
        # geometry and renders the inside of the scene.
        standoff = max(0.45, span * 2.5)
        fov = max(np.degrees(2 * np.arctan((span / 2) / standoff)), 6.0)
        cam = orbit_camera(
            centroid,
            state.view.azimuth_deg,
            state.view.elevation_deg,
            standoff,
            inner.w,
            inner.h,
            fov_deg=float(fov),
        )
        rendered = splat_cloud(
            cloud, cam, background=_PANEL_BG, tints=_tint_indices(cloud, state, colors)
        )
        img.paste(Image.fromarray(rendered), (inner.x0, inner.y0))
        painter = Painter(draw, cam, inner, margin=0)

    color = colors.get(focus_id, _TEXT)
    # No box in here. The inset is already cropped to this one object, using that
    # same box to set the crop, so drawing it outlines roughly the whole viewport
    # while covering the detail that was the reason to zoom in.
    for cand in state.candidates_of(focus_id):
        selected = cand.candidate_id == state.selected_id
        _draw_candidate(
            painter,
            cand,
            _SELECTED if selected else _KIND_COLOR.get(cand.kind, _TEXT),
            selected=selected,
            preview_flag=_preview_flag(state, cand.candidate_id),
            approach_axis=True,
        )
    draw.rectangle(FOCUS.box, outline=(64, 68, 78))
    tag = f"Focus {focus_id}" + ("" if explicit else " (auto)")
    _label(draw, (FOCUS.x0 + 5, FOCUS.y0 + 3), tag, color)


def _crop_around(
    cam_obs: dict[str, Any], center_world: np.ndarray, span_m: float
) -> tuple[float, float, float, float]:
    """Pixel crop box spanning ``span_m`` across a world point, clipped to frame.

    Spans the same width as the virtual focus camera's field of view, so the
    inset means the same magnification whichever source it draws from.
    """
    h, w = np.asarray(cam_obs["images"]["rgb"]).shape[:2]
    K = np.asarray(cam_obs["intrinsics"], dtype=np.float64)
    uvz = project_world_to_pixel(
        np.asarray(center_world).reshape(1, 3), K, cam_obs["pose_mat"]
    )[0]
    u, v, z = uvz
    if not np.isfinite(u) or z <= 0.05:
        return (0.0, 0.0, float(w), float(h))
    half = max(K[0, 0] * (span_m / 2) / z, 24.0)
    x1, y1 = float(np.clip(u - half, 0, w - 2)), float(np.clip(v - half, 0, h - 2))
    x2, y2 = float(np.clip(u + half, x1 + 2, w)), float(np.clip(v + half, y1 + 2, h))
    return (x1, y1, x2, y2)


def _render_wrist(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    obs: dict[str, Any] | None,
    wrist_camera_name: str,
) -> None:
    draw.rectangle(WRIST.box, fill=_PANEL_BG, outline=(64, 68, 78))
    cam = obs.get(wrist_camera_name) if obs else None
    if cam is None or "rgb" not in cam.get("images", {}):
        draw.text((WRIST.x0 + 6, WRIST.y0 + 4), "wrist (unavailable)", fill=_MUTED, font=F_SMALL)
        return
    frame = Image.fromarray(np.asarray(cam["images"]["rgb"], dtype=np.uint8))
    avail_h = WRIST.h - 18
    scale = min(WRIST.w / frame.width, avail_h / frame.height)
    pw, ph = max(int(frame.width * scale), 1), max(int(frame.height * scale), 1)
    img.paste(
        frame.resize((pw, ph)), (WRIST.x0 + (WRIST.w - pw) // 2, WRIST.y0 + 16 + (avail_h - ph) // 2)
    )
    draw.rectangle(WRIST.box, outline=(64, 68, 78))
    _label(draw, (WRIST.x0 + 5, WRIST.y0 + 3), "wrist", _MUTED)


# ===================================================================== #
# Entry point                                                            #
# ===================================================================== #

def _sync_view_base(state: ActionState, obs: dict[str, Any] | None, cloud: SceneCloud, camera_name: str) -> None:
    """Anchor the view state to the physical mount for this observation.

    Presets and the azimuth clamp are defined relative to the real camera, so
    they have to be recomputed whenever the observation changes rather than
    hard-coded for one scene.
    """
    target = cloud.center
    cam_obs = obs.get(camera_name) if obs else None
    if cam_obs is not None and cam_obs.get("pose_mat") is not None:
        eye = np.asarray(cam_obs["pose_mat"], dtype=np.float64)[:3, 3]
        az, el, dist = spherical_of(eye, target)
        state.view.sync_base(az, el, float(np.clip(dist, 0.4, 3.0)))
    state.view.target = target


def render_canvas(
    state: ActionState,
    obs: dict[str, Any] | None,
    *,
    camera_name: str = "agentview",
    wrist_camera_name: str = "robot0_eye_in_hand",
    cloud: SceneCloud | None = None,
) -> np.ndarray:
    """Render the workspace canvas. Returns (CANVAS_H, CANVAS_W, 3) uint8.

    Works with a partial or absent observation (renders what it can), so the
    same function serves the no-env smoke test and the live environment.

    :param cloud: fused scene cloud for this observation. Built here when not
        supplied; the workspace caches one per observation revision instead.
    """
    if cloud is None:
        cloud = build_scene_cloud(
            obs, (camera_name, wrist_camera_name), revision=state.obs_revision
        )
    _sync_view_base(state, obs, cloud, camera_name)
    focus_id, focus_explicit = state.focus_target()
    colors = _object_colors(state)

    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), (*_BG, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    painter = _render_main(img, draw, state, obs, cloud, camera_name, colors)
    draw = ImageDraw.Draw(img, "RGBA")  # masks were alpha-composited under it
    painter.draw = draw
    _annotate_main(img, painter, state, colors)
    _draw_gizmo(painter, state, cloud)

    _render_panel(draw, state, colors, focus_id)
    _render_focus(
        img, draw, state, obs, cloud, camera_name, colors, focus_id, focus_explicit
    )
    draw = ImageDraw.Draw(img, "RGBA")
    _render_wrist(img, draw, obs, wrist_camera_name)
    draw = ImageDraw.Draw(img, "RGBA")

    # Header last so nothing overdraws it.
    view = state.view
    header = (
        f"rev={state.obs_revision}  gripper={'open' if state.gripper_open else 'closed'}"
        f"  sel={state.selected_id or '-'}  view={view.preset}"
        f" az={view.azimuth_deg:.0f} el={view.elevation_deg:.0f} zoom={view.zoom:.1f}"
    )
    draw.rectangle((0, 0, MAIN.w - 1, HEADER_H - 1), fill=(*_BG, 225))
    draw.text((7, 4), header, fill=_TEXT, font=F_HEADER)
    instr = f"task: {state.instruction}"
    draw.rectangle((0, HEADER_H, MAIN.w - 1, HEADER_H + 19), fill=(*_BG, 190))
    draw.text((7, HEADER_H + 3), instr[:96], fill=_MUTED, font=F_BODY)
    if not view.is_physical:
        note = "reconstructed point cloud (view != agentview)"
        _label(draw, (7, MAIN.h - 20), note, _WARN)

    return np.asarray(img.convert("RGB"))
