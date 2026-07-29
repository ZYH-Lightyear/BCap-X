"""Virtual cameras for the canvas: the workspace's viewpoint is a state, not a
fixed physical mount.

The one design choice everything else follows from: a virtual camera is
expressed as the same ``(intrinsics, pose_mat)`` pair the physical cameras
report, so :func:`vaw.geometry.project_world_to_pixel` projects masks,
candidates, paths and point clouds into a virtual view with no special cases.

Camera convention (matching the Cap-X observation and OpenCV): camera frame has
+z forward, +x right, +y down, and ``pose_mat`` is camera-to-world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])

# Orbit envelope. Azimuth is clamped relative to the physical camera because a
# single-view depth map only carries geometry for surfaces that camera can see:
# orbiting behind the scene would render the *back* of a shell of points, which
# looks like a plausible scene but is empty of evidence. Elevation stops short
# of 90 deg because a camera looking straight down has no well-defined "right".
MAX_AZIMUTH_OFFSET_DEG = 75.0
ELEVATION_RANGE_DEG = (10.0, 85.0)
ZOOM_RANGE = (0.5, 2.5)

DEFAULT_FOV_DEG = 45.0


@dataclass
class VirtualCamera:
    """A camera we can project with: intrinsics + camera-to-world pose."""

    intrinsics: np.ndarray
    pose_mat: np.ndarray
    width: int
    height: int

    @property
    def eye(self) -> np.ndarray:
        return np.asarray(self.pose_mat, dtype=np.float64)[:3, 3]

    @property
    def right(self) -> np.ndarray:
        return np.asarray(self.pose_mat, dtype=np.float64)[:3, 0]

    @property
    def down(self) -> np.ndarray:
        return np.asarray(self.pose_mat, dtype=np.float64)[:3, 1]

    @property
    def forward(self) -> np.ndarray:
        return np.asarray(self.pose_mat, dtype=np.float64)[:3, 2]


def orbit_camera(
    target: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    distance_m: float,
    width: int,
    height: int,
    fov_deg: float = DEFAULT_FOV_DEG,
) -> VirtualCamera:
    """Build a camera orbiting ``target`` at the given spherical angles."""
    target = np.asarray(target, dtype=np.float64).reshape(3)
    az = math.radians(azimuth_deg)
    el = math.radians(min(elevation_deg, ELEVATION_RANGE_DEG[1]))
    eye = target + distance_m * np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )

    forward = target - eye
    forward /= np.linalg.norm(forward) or 1.0
    right = np.cross(forward, WORLD_UP)
    norm = np.linalg.norm(right)
    if norm < 1e-8:  # looking straight down: pick an arbitrary stable right
        right = np.array([0.0, 1.0, 0.0])
    else:
        right = right / norm
    down = np.cross(forward, right)

    pose_mat = np.eye(4)
    pose_mat[:3, 0] = right
    pose_mat[:3, 1] = down
    pose_mat[:3, 2] = forward
    pose_mat[:3, 3] = eye

    focal = (width / 2) / math.tan(math.radians(fov_deg) / 2)
    intrinsics = np.array(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]]
    )
    return VirtualCamera(intrinsics, pose_mat, width, height)


def fit_physical_camera(
    cam_obs: dict[str, Any],
    width: int,
    height: int,
    *,
    crop: tuple[float, float, float, float] | None = None,
) -> tuple[VirtualCamera, tuple[int, int, int, int]]:
    """Adapt a physical camera to a viewport, optionally cropped (a zoom).

    Returns the camera whose intrinsics project world points straight into
    viewport-local pixels, plus the ``(x, y, w, h)`` box the (cropped, scaled)
    RGB should be pasted into. Aspect ratio is preserved — letterboxing a real
    camera frame beats stretching it, both for the model's object recognition
    and for keeping one scale factor in the intrinsics.
    """
    src_h, src_w = np.asarray(cam_obs["images"]["rgb"]).shape[:2]
    x1, y1, x2, y2 = (0.0, 0.0, float(src_w), float(src_h)) if crop is None else crop
    crop_w, crop_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)

    scale = min(width / crop_w, height / crop_h)
    paste_w, paste_h = max(int(round(crop_w * scale)), 1), max(int(round(crop_h * scale)), 1)
    off_x, off_y = (width - paste_w) // 2, (height - paste_h) // 2

    K = np.asarray(cam_obs["intrinsics"], dtype=np.float64).copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] = (K[0, 2] - x1) * scale + off_x
    K[1, 2] = (K[1, 2] - y1) * scale + off_y
    cam = VirtualCamera(
        K, np.asarray(cam_obs["pose_mat"], dtype=np.float64), width, height
    )
    return cam, (off_x, off_y, paste_w, paste_h)


def spherical_of(eye: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`orbit_camera`: (azimuth_deg, elevation_deg, distance)."""
    d = np.asarray(eye, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    distance = float(np.linalg.norm(d)) or 1e-6
    azimuth = math.degrees(math.atan2(d[1], d[0]))
    elevation = math.degrees(math.asin(np.clip(d[2] / distance, -1.0, 1.0)))
    return azimuth, elevation, distance


@dataclass
class ViewState:
    """The agent-controlled viewpoint, part of the persistent action state.

    ``preset == "agentview"`` is special: the main view then renders the
    physical camera's own RGB frame instead of splatted geometry. It is the
    highest-fidelity view available (real texture, no reconstruction holes), so
    it stays the default; ``view`` exists to leave it when occlusion or depth
    ambiguity makes geometry the more informative picture.
    """

    azimuth_deg: float = 0.0
    elevation_deg: float = 35.0
    zoom: float = 1.0
    preset: str = "agentview"
    #: Filled in from the live observation each time the canvas renders, so
    #: relative clamping and presets track the actual camera mount.
    base_azimuth_deg: float = 0.0
    base_elevation_deg: float = 35.0
    base_distance_m: float = 1.2
    target: np.ndarray | None = None

    @property
    def is_physical(self) -> bool:
        return self.preset == "agentview"

    def sync_base(self, azimuth: float, elevation: float, distance: float) -> None:
        self.base_azimuth_deg = azimuth
        self.base_elevation_deg = elevation
        self.base_distance_m = distance
        if self.preset == "agentview":
            self.azimuth_deg, self.elevation_deg = azimuth, elevation

    def summary(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "azimuth_deg": round(self.azimuth_deg, 1),
            "elevation_deg": round(self.elevation_deg, 1),
            "zoom": round(self.zoom, 2),
            "source": "physical camera" if self.is_physical else "reconstructed point cloud",
        }


#: Preset viewpoints, as offsets from the physical camera's own spherical pose.
#: Keeping them relative means the same name means the same thing on any mount.
PRESETS: dict[str, dict[str, float]] = {
    "agentview": {"d_azimuth": 0.0, "elevation": -1.0, "zoom": 1.0},
    "top": {"d_azimuth": 0.0, "elevation": 80.0, "zoom": 1.0},
    "left": {"d_azimuth": -60.0, "elevation": 35.0, "zoom": 1.0},
    "right": {"d_azimuth": 60.0, "elevation": 35.0, "zoom": 1.0},
    "low": {"d_azimuth": 0.0, "elevation": 12.0, "zoom": 1.0},
    "close": {"d_azimuth": 0.0, "elevation": -1.0, "zoom": 1.9},
}


def resolve_view(
    view: ViewState,
    *,
    preset: str | None = None,
    azimuth_deg: float | None = None,
    elevation_deg: float | None = None,
    zoom: float | None = None,
) -> list[str]:
    """Apply a ``view`` op to ``view`` in place. Returns clamp notes.

    Angles are absolute world azimuth / elevation in degrees; a preset resolves
    to angles relative to the physical camera. Out-of-envelope requests are
    clamped rather than rejected, and every clamp is reported so the agent can
    learn where the envelope is instead of guessing from silent failures.
    """
    notes: list[str] = []
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset '{preset}'; available: {sorted(PRESETS)}")
        spec = PRESETS[preset]
        view.preset = preset
        azimuth_deg = view.base_azimuth_deg + spec["d_azimuth"]
        # elevation -1 is the sentinel for "keep the physical camera's own".
        elevation_deg = (
            view.base_elevation_deg if spec["elevation"] < 0 else spec["elevation"]
        )
        zoom = spec["zoom"] if zoom is None else zoom

    if azimuth_deg is not None:
        offset = _wrap_deg(float(azimuth_deg) - view.base_azimuth_deg)
        clamped = float(np.clip(offset, -MAX_AZIMUTH_OFFSET_DEG, MAX_AZIMUTH_OFFSET_DEG))
        if abs(clamped - offset) > 1e-6:
            notes.append(
                f"azimuth clamped to {MAX_AZIMUTH_OFFSET_DEG:.0f} deg from the camera "
                "(no depth data beyond that)"
            )
        view.azimuth_deg = view.base_azimuth_deg + clamped

    if elevation_deg is not None:
        clamped = float(np.clip(float(elevation_deg), *ELEVATION_RANGE_DEG))
        if abs(clamped - float(elevation_deg)) > 1e-6:
            notes.append(f"elevation clamped to {ELEVATION_RANGE_DEG}")
        view.elevation_deg = clamped

    if zoom is not None:
        clamped = float(np.clip(float(zoom), *ZOOM_RANGE))
        if abs(clamped - float(zoom)) > 1e-6:
            notes.append(f"zoom clamped to {ZOOM_RANGE}")
        view.zoom = clamped

    # Any explicit angle takes the view off the physical camera: from here on
    # the main view is reconstructed geometry, and the header says so.
    if preset is None and (azimuth_deg is not None or elevation_deg is not None):
        view.preset = "custom"
    return notes


def _wrap_deg(angle: float) -> float:
    """Wrap to (-180, 180] so azimuth offsets never take the long way round."""
    return (angle + 180.0) % 360.0 - 180.0
