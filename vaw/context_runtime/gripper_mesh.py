"""Franka geometry used by the current VAW Context renderer.

The virtual gripper is the one thing on the canvas no sensor image can show: it
is where the agent *intends* to put the hand. A stick figure conveys position and
little else — whether the fingers actually clear the neighbouring object, and how
wide they are relative to it, are exactly the judgements the glyph has to support,
and they need the hand's real outline.

Geometry is evaluated by forward kinematics from the seven observed arm joints
against an isolated Panda URDF, the same joint->FK route the old RoboMEx renderer
used. Never reconstruct ``panda_hand`` from a reported Cartesian pose and a
hand-written TCP offset instead: how the reported frame relates to the URDF link
is a convention one can only guess at, and a mesh drawn from a wrong guess is a
confident lie about where the hand is. The joints admit no such guess.

It follows that whoever has no joint solution has no exact pose either, and
should fall back to the schematic glyph rather than invent one.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


class PandaUrdfGripperFK:
    """Exact Panda gripper visuals evaluated from seven arm joints.

    The URDF instance is private to the renderer and never touches MuJoCo.  A
    lock protects ``yourdfpy.update_cfg`` because it mutates the scene graph.
    This is the same joint→URDF-FK route used by the old RoboMEx AgentWorld,
    with one correction: the finger joint follows the observed normalized
    opening instead of being hard-coded fully open.
    """

    _ARM_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))
    _GRIPPER_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
    _ROBOT_LINKS = (
        *(f"panda_link{index}" for index in range(9)),
        *_GRIPPER_LINKS,
    )

    def __init__(self, urdf: Any | None = None) -> None:
        if urdf is None:
            from robot_descriptions.loaders.yourdfpy import load_robot_description

            urdf = load_robot_description("panda_description")
        if not callable(getattr(urdf, "update_cfg", None)):
            raise TypeError("urdf must expose update_cfg()")
        self.urdf = urdf
        finger = urdf.joint_map["panda_finger_joint1"]
        self.finger_max_q = float(finger.limit.upper)
        self._lock = threading.RLock()

    @property
    def model_id(self) -> str:
        return "robot_descriptions.panda_description:yourdfpy"

    def triangles(
        self,
        joint_positions_rad: np.ndarray,
        gripper_opening: float,
    ) -> np.ndarray:
        """Return world/robot-base gripper triangles for one exact joint state."""

        return self._triangles_for_links(
            joint_positions_rad,
            gripper_opening,
            self._GRIPPER_LINKS,
        )

    def hand_triangles(
        self,
        joint_positions_rad: np.ndarray,
        gripper_opening: float,
    ) -> np.ndarray:
        """Return only the rigid palm/hand collision body.

        The policy renderer uses this subset for a faint occupancy fill while
        keeping the fingers and the rest of the robot as transparent line art.
        It prevents the hand crossbar from visually disappearing without
        hiding the object inside the grasp aperture.
        """

        return self._triangles_for_links(
            joint_positions_rad,
            gripper_opening,
            ("panda_hand",),
        )

    def robot_triangles(
        self,
        joint_positions_rad: np.ndarray,
        gripper_opening: float,
    ) -> np.ndarray:
        """Return the complete Franka visual mesh for one exact joint state."""

        return self._triangles_for_links(
            joint_positions_rad,
            gripper_opening,
            self._ROBOT_LINKS,
        )

    def _triangles_for_links(
        self,
        joint_positions_rad: np.ndarray,
        gripper_opening: float,
        links: tuple[str, ...],
    ) -> np.ndarray:
        """Evaluate selected URDF visual links under one shared FK update."""

        joints = np.asarray(joint_positions_rad, dtype=np.float64).reshape(-1)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("Panda FK requires exactly seven finite arm joints")
        opening = float(gripper_opening)
        if not np.isfinite(opening):
            raise ValueError("gripper opening must be finite")

        with self._lock:
            config = dict(zip(self._ARM_JOINTS, joints, strict=True))
            config["panda_finger_joint1"] = (
                self.finger_max_q * float(np.clip(opening, 0.0, 1.0))
            )
            self.urdf.update_cfg(config)
            parts = self._visual_meshes(links)
        if not parts:
            raise RuntimeError("Panda URDF has no requested visual meshes")
        return np.concatenate([vertices[faces] for vertices, faces in parts], axis=0)

    def frame(
        self,
        joint_positions_rad: np.ndarray,
        frame_name: str = "panda_hand",
        *,
        gripper_opening: float = 1.0,
    ) -> np.ndarray:
        """Expose an FK frame for calibration tests and diagnostics."""

        joints = np.asarray(joint_positions_rad, dtype=np.float64).reshape(-1)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("Panda FK requires exactly seven finite arm joints")
        with self._lock:
            config = dict(zip(self._ARM_JOINTS, joints, strict=True))
            config["panda_finger_joint1"] = (
                self.finger_max_q * float(np.clip(gripper_opening, 0.0, 1.0))
            )
            self.urdf.update_cfg(config)
            return self._frame(frame_name)

    def _visual_meshes(
        self,
        links: tuple[str, ...],
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        children = self.urdf.scene.graph.transforms.children
        names = tuple(
            dict.fromkeys(
                name
                for link in links
                for name in children.get(link, ())
                if name in self.urdf.scene.geometry
            )
        )
        meshes: list[tuple[np.ndarray, np.ndarray]] = []
        for name in names:
            geometry = self.urdf.scene.geometry[name]
            vertices = np.asarray(geometry.vertices, dtype=np.float64)
            faces = np.asarray(geometry.faces, dtype=np.int64)
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or faces.ndim != 2
                or faces.shape[1] != 3
            ):
                continue
            transform = self._frame(name)
            homogeneous = np.column_stack(
                (vertices, np.ones(len(vertices), dtype=np.float64))
            )
            world_vertices = np.ascontiguousarray(
                (homogeneous @ transform.T)[:, :3]
            )
            meshes.append((world_vertices, np.ascontiguousarray(faces)))
        return tuple(meshes)

    def _frame(self, frame_name: str) -> np.ndarray:
        try:
            value = self.urdf.scene.graph.get(frame_to=frame_name)[0]
        except Exception as exc:
            raise RuntimeError(f"Panda URDF has no frame {frame_name!r}") from exc
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise RuntimeError(f"Panda URDF frame {frame_name!r} is invalid")
        return np.ascontiguousarray(matrix)


@lru_cache(maxsize=1)
def load_panda_urdf_fk() -> PandaUrdfGripperFK | None:
    """Load the exact FK provider once; return ``None`` if deps are unavailable."""

    try:
        return PandaUrdfGripperFK()
    except (ImportError, FileNotFoundError):
        return None


def semantic_parallel_jaw_triangles(
    position_xyz: tuple[float, float, float] | np.ndarray,
    quaternion_xyzw: tuple[float, float, float, float] | np.ndarray,
    gripper_opening: float,
) -> np.ndarray:
    """Return a deliberately simple, metric parallel-jaw target glyph.

    The glyph is expressed in the public fingertip/contact TCP frame: local
    ``+Z`` points from the palm toward the contact plane and local ``Y`` is the
    finger-closing axis.  It has two equal fingers and one thin palm crossbar;
    callers render only its 3-D edges, never a filled mask.  This is a visual
    comparison mode, not a replacement for FK or collision geometry.
    """

    position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    opening = float(gripper_opening)
    if not (
        np.isfinite(position).all()
        and np.isfinite(quaternion).all()
        and np.isfinite(opening)
        and np.linalg.norm(quaternion) > 1e-12
    ):
        raise ValueError("semantic gripper pose and opening must be finite")

    # The Panda's maximum inner finger gap is approximately 8 cm.  Keep the
    # visual proportions metric so contact-camera scale bars remain meaningful.
    inner_gap = 0.08 * float(np.clip(opening, 0.0, 1.0))
    finger_thickness_y = 0.014
    finger_depth_x = 0.022
    finger_length_z = 0.082
    finger_center_z = -0.5 * finger_length_z
    finger_center_y = 0.5 * (inner_gap + finger_thickness_y)
    palm_span_y = max(0.115, inner_gap + 2.0 * finger_thickness_y + 0.012)

    local = np.concatenate(
        (
            _cuboid_triangles(
                center=(0.0, -finger_center_y, finger_center_z),
                size=(finger_depth_x, finger_thickness_y, finger_length_z),
            ),
            _cuboid_triangles(
                center=(0.0, finger_center_y, finger_center_z),
                size=(finger_depth_x, finger_thickness_y, finger_length_z),
            ),
            _cuboid_triangles(
                center=(0.0, 0.0, -0.092),
                size=(0.026, palm_span_y, 0.009),
            ),
        ),
        axis=0,
    )
    rotation = Rotation.from_quat(quaternion / np.linalg.norm(quaternion)).as_matrix()
    return np.ascontiguousarray(local @ rotation.T + position)


def _cuboid_triangles(
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> np.ndarray:
    half = 0.5 * np.asarray(size, dtype=np.float64)
    origin = np.asarray(center, dtype=np.float64)
    signs = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    vertices = origin + signs * half
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return vertices[faces]


def rasterize_silhouette(
    tris_px: np.ndarray,
    depths: np.ndarray,
    width: int,
    height: int,
    *,
    max_pixels_per_chunk: int = 8_000_000,
) -> np.ndarray:
    """Union of the triangles' pixel coverage: a (height, width) bool mask.

    A silhouette needs no depth buffer — overlapping triangles of one solid body
    contribute the same pixels — so this is a coverage test per triangle and
    nothing more.

    Triangles are processed in chunks sized by their bounding boxes, because the
    vectorised inside-test allocates ``chunk x bbox_h x bbox_w`` booleans and the
    projected size of a triangle varies with how close the camera is.
    """
    mask = np.zeros((height, width), dtype=bool)
    if len(tris_px) == 0:
        return mask

    tris = np.asarray(tris_px, dtype=np.float64)
    keep = np.asarray(depths, dtype=np.float64).min(axis=1) > 1e-6
    keep &= np.isfinite(tris).all(axis=(1, 2))
    tris = tris[keep]
    if len(tris) == 0:
        return mask

    x0 = np.floor(tris[:, :, 0].min(axis=1)).astype(np.int64)
    x1 = np.ceil(tris[:, :, 0].max(axis=1)).astype(np.int64)
    y0 = np.floor(tris[:, :, 1].min(axis=1)).astype(np.int64)
    y1 = np.ceil(tris[:, :, 1].max(axis=1)).astype(np.int64)
    # Wholly off-frame triangles cost nothing to drop and would otherwise widen
    # every chunk's grid to the distance they sit outside the view.
    on_frame = (x1 >= 0) & (y1 >= 0) & (x0 < width) & (y0 < height)
    tris, x0, x1, y0, y1 = tris[on_frame], x0[on_frame], x1[on_frame], y0[on_frame], y1[on_frame]
    if len(tris) == 0:
        return mask
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    x1 = np.clip(x1, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)

    # Group by bounding-box size so one huge triangle does not set the grid for
    # thousands of small ones. Order is by span, then chunked to a pixel budget.
    span = np.maximum(x1 - x0, y1 - y0)
    order = np.argsort(span, kind="stable")
    start = 0
    while start < len(order):
        # Grow the chunk while the grid it forces stays inside the budget.
        grid = int(span[order[start]]) + 1
        end = start + max(1, max_pixels_per_chunk // (grid * grid))
        end = min(end, len(order))
        grid = int(span[order[start:end]].max()) + 1
        while (end - start) * grid * grid > max_pixels_per_chunk and end - start > 1:
            end -= 1
            grid = int(span[order[start:end]].max()) + 1

        idx = order[start:end]
        _fill_chunk(mask, tris[idx], x0[idx], y0[idx], grid, width, height)
        start = end
    return mask


def _fill_chunk(
    mask: np.ndarray,
    tris: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    grid: int,
    width: int,
    height: int,
) -> None:
    """Inside-test a grid of candidate pixels per triangle, OR the hits into mask."""
    off = np.arange(grid, dtype=np.float64)
    # Pixel centres: a triangle covering a pixel means covering its centre, and
    # sampling at corners instead drops thin triangles that pass between them.
    px = x0[:, None, None] + off[None, None, :] + 0.5
    py = y0[:, None, None] + off[None, :, None] + 0.5

    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]

    def edge(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Signed area of (p, q, pixel): the standard edge function.
        return (q[:, 0, None, None] - p[:, 0, None, None]) * (py - p[:, 1, None, None]) - (
            q[:, 1, None, None] - p[:, 1, None, None]
        ) * (px - p[:, 0, None, None])

    w0, w1, w2 = edge(a, b), edge(b, c), edge(c, a)
    # Accept either winding: the meshes are not consistently oriented after
    # merging several visuals, and a silhouette does not care which way a face
    # points.
    inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
    if not inside.any():
        return

    tri_i, row, col = np.nonzero(inside)
    vs = y0[tri_i] + row
    us = x0[tri_i] + col
    ok = (vs >= 0) & (vs < height) & (us >= 0) & (us < width)
    mask[vs[ok], us[ok]] = True


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of ``mask``: itself minus its 4-neighbour erosion."""
    if not mask.any():
        return mask
    eroded = mask.copy()
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]
    return mask & ~eroded


def thick_mask_outline(mask: np.ndarray, *, radius: int = 2) -> np.ndarray:
    """Return a visible silhouette outline without filling the enclosed mask.

    ``mask_outline`` is intentionally one pixel wide for geometry operations.
    Policy-visible robot contours need to survive raster scaling, so this helper
    expands that boundary symmetrically while leaving the silhouette interior
    untouched.
    """

    if radius < 0:
        raise ValueError("radius must be non-negative")
    outline = mask_outline(np.asarray(mask, dtype=bool))
    for _ in range(radius):
        padded = np.pad(outline, 1, mode="constant")
        outline = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    return outline


def overlay_projected_mesh_outline(
    image: np.ndarray,
    triangles_base: np.ndarray,
    triangles_px: np.ndarray,
    depths: np.ndarray,
    *,
    color: tuple[int, int, int] = (216, 203, 255),
    camera_position_base: np.ndarray | None = None,
    view_forward_base: np.ndarray | None = None,
    crease_angle_deg: float = 32.0,
) -> None:
    """Draw a transparent 3-D line rendering over an RGB raster.

    The silhouette supplies the outer boundary.  Shared edges between visible
    faces are retained only at real geometric creases, which conveys palm and
    finger thickness without filling or hiding any observed RGB pixel.

    Exactly one viewing model is required: ``camera_position_base`` for a
    perspective camera, or ``view_forward_base`` for an orthographic view.
    The input ``triangles_px`` must already use the final raster coordinates.
    """

    values = np.asarray(triangles_base, dtype=np.float64).reshape(-1, 3, 3)
    projected = np.asarray(triangles_px, dtype=np.float64).reshape(-1, 3, 2)
    triangle_depths = np.asarray(depths, dtype=np.float64).reshape(-1, 3)
    if not (len(values) == len(projected) == len(triangle_depths)):
        raise ValueError("3-D triangles, projected triangles and depths must align")
    perspective = camera_position_base is not None
    orthographic = view_forward_base is not None
    if perspective == orthographic:
        raise ValueError("provide exactly one camera position or view-forward vector")

    mask = rasterize_silhouette(
        projected,
        triangle_depths,
        image.shape[1],
        image.shape[0],
    )
    if np.any(mask):
        image[mask_outline(mask)] = np.asarray(color, dtype=np.uint8)

    segments = _visible_crease_segments(
        values,
        projected,
        triangle_depths,
        camera_position_base=camera_position_base,
        view_forward_base=view_forward_base,
        crease_angle_deg=crease_angle_deg,
    )
    if not segments:
        return
    rendered = Image.fromarray(np.asarray(image, dtype=np.uint8))
    draw = ImageDraw.Draw(rendered, "RGBA")
    rgba = (*color, 245)
    for start, end in segments:
        draw.line((*start, *end), fill=rgba, width=1)
    image[:] = np.asarray(rendered, dtype=np.uint8)


def _visible_crease_segments(
    triangles_base: np.ndarray,
    triangles_px: np.ndarray,
    depths: np.ndarray,
    *,
    camera_position_base: np.ndarray | None,
    view_forward_base: np.ndarray | None,
    crease_angle_deg: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return projected shared edges that express visible 3-D structure."""

    triangles = np.asarray(triangles_base, dtype=np.float64).reshape(-1, 3, 3)
    projected = np.asarray(triangles_px, dtype=np.float64).reshape(-1, 3, 2)
    triangle_depths = np.asarray(depths, dtype=np.float64).reshape(-1, 3)
    flat = triangles.reshape(-1, 3)
    flat_px = projected.reshape(-1, 2)
    if len(flat) == 0:
        return []

    # URDF visuals repeat vertices per triangle and per sub-mesh.  Weld only
    # for line extraction; the actual rendering geometry remains untouched.
    quantized = np.rint(flat / 1e-5).astype(np.int64)
    _keys, inverse = np.unique(quantized, axis=0, return_inverse=True)
    face_indices = inverse.reshape(-1, 3)
    vertex_count = int(inverse.max()) + 1
    vertices_px = np.zeros((vertex_count, 2), dtype=np.float64)
    counts = np.zeros(vertex_count, dtype=np.int64)
    np.add.at(vertices_px, inverse, flat_px)
    np.add.at(counts, inverse, 1)
    vertices_px /= counts[:, None]

    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-10
    normals[valid] /= lengths[valid, None]
    centers = triangles.mean(axis=1)
    if camera_position_base is not None:
        camera = np.asarray(camera_position_base, dtype=np.float64).reshape(3)
        front_facing = valid & (
            np.einsum("ij,ij->i", normals, centers - camera) < 0.0
        )
    else:
        forward = np.asarray(view_forward_base, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-10:
            raise ValueError("view-forward vector must be non-zero")
        front_facing = valid & ((normals @ (forward / norm)) < 0.0)
    front_facing &= np.isfinite(projected).all(axis=(1, 2))
    front_facing &= np.isfinite(triangle_depths).all(axis=1)
    front_facing &= triangle_depths.min(axis=1) > 1e-6

    adjacency: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(face_indices):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (int(min(start, end)), int(max(start, end)))
            adjacency.setdefault(key, []).append(face_index)

    crease_limit = float(np.cos(np.deg2rad(float(crease_angle_deg))))
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for (start_index, end_index), faces in adjacency.items():
        # The silhouette already supplies open/sub-mesh boundaries.  Drawing
        # every such edge turns a close contact view into a dense wire ball.
        if len(faces) < 2:
            continue
        first, second = faces[:2]
        if not (front_facing[first] and front_facing[second]):
            continue
        if abs(float(np.dot(normals[first], normals[second]))) > crease_limit:
            continue
        start = vertices_px[start_index]
        end = vertices_px[end_index]
        if not np.isfinite((start, end)).all():
            continue
        segments.append(
            (
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
            )
        )
    return segments
