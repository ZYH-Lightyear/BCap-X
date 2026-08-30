"""Simulation-only, session-locked Contact Camera rendering for LIBERO-PRO.

The policy-visible Contact views are rendered directly by MuJoCo instead of
reprojecting surfaces observed by agentview / wrist RGB-D.  This deliberately
adds two simulation cameras and must not be described as sensor re-layout in
experiments.  Camera mutations are restored immediately after each render and
never enter the public Function or manifest contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation

# How far a Contact camera may back off to fit tall geometry, as a multiple of
# its close-up distance.  Beyond this the panel stops being a contact view.
_MAX_FRAMING_DISTANCE_RATIO = 4.6


@dataclass(frozen=True)
class ContactCameraRequest:
    """One immutable Contact Camera frame in robot-base coordinates."""

    center_base_xyz: tuple[float, float, float]
    frame_quaternion_xyzw: tuple[float, float, float, float]
    width: int
    panel_height: int
    # Private, sensor-derived geometry used only to choose an unobstructed
    # camera pair.  It is never copied into a ContextPacket.
    subject_points_base: np.ndarray | None = None
    # Virtual target geometry that must remain inside both panels. Unlike the
    # observed subject it is not scored for visibility because it is rendered
    # later as a line-art overlay, but it participates in camera centering and
    # zoom so a missing carried-object proxy cannot clip the target gripper.
    required_points_base: np.ndarray | None = None
    # Keep left/right semantics stable across physical revisions.  When set,
    # visibility scoring is bypassed and this previously selected end of each
    # orthogonal axis is rendered again around the new current geometry.
    preferred_signs: tuple[int, int] | None = None
    # Tilt the SIDE camera above the horizon while keeping it on the same
    # gripper-locked azimuth.  Two horizontal panels resolve height but leave
    # lateral placement unobservable, so carrying a payload trades the
    # redundant second elevation readout for an oblique XY readout.
    side_elevation_deg: float = 0.0
    # Optional third view used outside the orthogonal Contact pair.  It looks
    # diagonally across FRONT/SIDE and steeply downward, so the upper Canvas
    # panel can reveal fingers and carried objects that a pure side view hides
    # behind the Panda hand.  It never changes the two metric Contact views.
    auxiliary_elevation_deg: float | None = None


@dataclass(frozen=True)
class ContactCameraSelection:
    """Trace-only explanation of one session-locked camera choice."""

    front_sign: int
    side_sign: int
    score: float
    front_visibility: float
    side_visibility: float
    azimuth_offset_deg: float = 0.0
    side_elevation_deg: float = 0.0

    def summary(self) -> dict[str, float | int]:
        return {
            "front_sign": int(self.front_sign),
            "side_sign": int(self.side_sign),
            "score": round(float(self.score), 6),
            "front_visibility": round(float(self.front_visibility), 6),
            "side_visibility": round(float(self.side_visibility), 6),
            "azimuth_offset_deg": round(float(self.azimuth_offset_deg), 3),
            "side_elevation_deg": round(float(self.side_elevation_deg), 3),
        }


@dataclass(frozen=True)
class ContactCameraPair:
    """Direct MuJoCo RGB cameras plus private calibration for overlays."""

    front: dict[str, Any]
    side: dict[str, Any]
    selection: ContactCameraSelection = ContactCameraSelection(1, 1, 1.0, 1.0, 1.0)
    auxiliary: dict[str, Any] | None = None


@dataclass(frozen=True)
class OppositeSceneCameraRequest:
    """One fixed, complementary world-camera request in robot-base coordinates."""

    center_base_xyz: tuple[float, float, float]
    agentview_forward_base_xyz: tuple[float, float, float]
    width: int
    height: int


class LiberoOppositeSceneCameraProvider:
    """Render a gravity-stable view from the far side of the workspace.

    The view direction is derived once from the calibrated agentview direction,
    then rotated around world Z.  The compiler locks the requested centre for
    the episode, so physical revisions change pixels without moving the camera.
    """

    def __init__(
        self,
        env: Any,
        *,
        camera_name: str = "frontview",
        distance_m: float = 0.82,
        elevation_m: float = 0.38,
        fovy_deg: float = 52.0,
        yaw_offset_deg: float = 150.0,
    ) -> None:
        self._env = env
        self._camera_name = str(camera_name)
        self._distance_m = float(distance_m)
        self._elevation_m = float(elevation_m)
        self._fovy_deg = float(fovy_deg)
        self._yaw_offset_deg = float(yaw_offset_deg)
        if not np.isfinite(self._distance_m) or self._distance_m <= 0.0:
            raise ValueError("distance_m must be positive and finite")
        if not np.isfinite(self._elevation_m) or self._elevation_m < 0.0:
            raise ValueError("elevation_m must be finite and non-negative")
        if not np.isfinite(self._fovy_deg) or not 5.0 <= self._fovy_deg <= 120.0:
            raise ValueError("fovy_deg must be in [5, 120]")
        if not np.isfinite(self._yaw_offset_deg):
            raise ValueError("yaw_offset_deg must be finite")

    def __call__(self, request: OppositeSceneCameraRequest) -> dict[str, Any]:
        sim = self._env.handle.env.sim
        center_base = _vector3(request.center_base_xyz, "center_base_xyz")
        agentview_forward_base = _vector3(
            request.agentview_forward_base_xyz,
            "agentview_forward_base_xyz",
        )
        width = int(request.width)
        height = int(request.height)
        if width <= 0 or height <= 0:
            raise ValueError("opposite scene camera dimensions must be positive")

        base_position_world, base_rotation_world = _base_pose_world(self._env, sim)
        center_world = base_position_world + base_rotation_world @ center_base
        up_world = _unit(base_rotation_world[:, 2], "camera gravity up")
        agentview_forward_world = _horizontal_axis(
            base_rotation_world @ agentview_forward_base,
            up_world,
            "agentview horizontal forward",
        )
        yaw = Rotation.from_rotvec(
            up_world * np.deg2rad(self._yaw_offset_deg)
        ).as_matrix()
        opposite_forward_world = _unit(
            yaw @ agentview_forward_world,
            "opposite camera forward",
        )

        camera_id = int(sim.model.camera_name2id(self._camera_name))
        saved_position = np.asarray(
            sim.model.cam_pos[camera_id], dtype=np.float64
        ).copy()
        saved_quaternion = np.asarray(
            sim.model.cam_quat[camera_id], dtype=np.float64
        ).copy()
        saved_fovy = float(sim.model.cam_fovy[camera_id])
        try:
            camera, _depth_metric = _render_rgbd_camera(
                sim,
                camera_name=self._camera_name,
                view_name="opposite",
                center_world=center_world,
                horizontal_forward_world=opposite_forward_world,
                up_world=up_world,
                base_position_world=base_position_world,
                base_rotation_world=base_rotation_world,
                width=width,
                height=height,
                distance_m=self._distance_m,
                elevation_m=self._elevation_m,
                fovy_deg=self._fovy_deg,
            )
            return camera
        finally:
            sim.model.cam_pos[camera_id] = saved_position
            sim.model.cam_quat[camera_id] = saved_quaternion
            sim.model.cam_fovy[camera_id] = saved_fovy
            sim.forward()


class LiberoContactCameraProvider:
    """Render gravity-stable orthogonal cameras without stepping LIBERO."""

    _CAMERAS: tuple[tuple[Literal["front", "side"], str, int], ...] = (
        ("front", "frontview", 0),
        ("side", "sideview", 1),
    )

    def __init__(
        self,
        env: Any,
        *,
        # Sit in front of neighbouring table objects (a tomato can is ~12 cm
        # off the cream-cheese box).  16 cm still covers fingers + the local
        # object at 34°; 26–38 cm puts the can between the lens and the hand.
        distance_m: float = 0.16,
        fovy_deg: float = 34.0,
        framing_padding_m: float = 0.035,
    ) -> None:
        self._env = env
        self._distance_m = float(distance_m)
        self._fovy_deg = float(fovy_deg)
        self._framing_padding_m = float(framing_padding_m)
        if not np.isfinite(self._distance_m) or self._distance_m <= 0.0:
            raise ValueError("distance_m must be positive and finite")
        if not np.isfinite(self._fovy_deg) or not 5.0 <= self._fovy_deg <= 120.0:
            raise ValueError("fovy_deg must be in [5, 120]")
        if (
            not np.isfinite(self._framing_padding_m)
            or self._framing_padding_m < 0.0
        ):
            raise ValueError("framing_padding_m must be finite and non-negative")

    def __call__(self, request: ContactCameraRequest) -> ContactCameraPair:
        sim = self._env.handle.env.sim
        center_base = _vector3(request.center_base_xyz, "center_base_xyz")
        frame_quaternion = _quaternion_xyzw(
            request.frame_quaternion_xyzw,
            "frame_quaternion_xyzw",
        )
        width = int(request.width)
        height = int(request.panel_height)
        if width <= 0 or height <= 0:
            raise ValueError("contact camera dimensions must be positive")
        side_elevation_deg = float(request.side_elevation_deg)
        if not np.isfinite(side_elevation_deg) or not 0.0 <= side_elevation_deg <= 80.0:
            raise ValueError("side_elevation_deg must be in [0, 80]")
        auxiliary_elevation_deg = request.auxiliary_elevation_deg
        if auxiliary_elevation_deg is not None:
            auxiliary_elevation_deg = float(auxiliary_elevation_deg)
            if (
                not np.isfinite(auxiliary_elevation_deg)
                or not 0.0 <= auxiliary_elevation_deg <= 80.0
            ):
                raise ValueError("auxiliary_elevation_deg must be in [0, 80]")

        base_position_world, base_rotation_world = _base_pose_world(self._env, sim)
        center_world = base_position_world + base_rotation_world @ center_base
        frame_rotation_world = (
            base_rotation_world @ Rotation.from_quat(frame_quaternion).as_matrix()
        )

        subject_points = _points3(request.subject_points_base)
        required_points = _points3(request.required_points_base)
        saved: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        candidates: list[
            tuple[
                float,
                int,
                int,
                float,
                float,
                float,
                dict[str, dict[str, Any]],
            ]
        ] = []
        try:
            for _view_name, camera_name, _forward_index in self._CAMERAS:
                camera_id = int(sim.model.camera_name2id(camera_name))
                saved.append(
                    (
                        camera_id,
                        np.asarray(sim.model.cam_pos[camera_id], dtype=np.float64).copy(),
                        np.asarray(sim.model.cam_quat[camera_id], dtype=np.float64).copy(),
                        float(sim.model.cam_fovy[camera_id]),
                    )
                )

            up_world = _unit(frame_rotation_world[:, 2], "camera gravity up")
            front_zero = _horizontal_axis(
                frame_rotation_world[:, 0], up_world, "contact front"
            )
            side_zero = _unit(np.cross(up_world, front_zero), "contact side")
            # FRONT and SIDE azimuths are geometric invariants. FRONT looks
            # along the session-locked gripper-normal axis and SIDE along its
            # orthogonal closing axis. Visibility may choose only which end of
            # either axis to observe from; it must never rotate the azimuths
            # away from the gripper. Elevation is the one degree of freedom the
            # caller may add, and only on SIDE, so the pair keeps a level panel
            # for height and gains an oblique panel for lateral placement.
            front_axis = front_zero
            side_axis = side_zero
            sign_pairs = (
                (request.preferred_signs,)
                if request.preferred_signs is not None
                else ((1, 1), (1, -1), (-1, 1), (-1, -1))
            )
            for front_sign, side_sign in sign_pairs:
                if front_sign not in {-1, 1} or side_sign not in {-1, 1}:
                    raise ValueError("preferred_signs must contain only -1 or 1")
                front_world = float(front_sign) * front_axis
                side_world = float(side_sign) * side_axis
                rendered: dict[str, dict[str, Any]] = {}
                scores: dict[str, float] = {}
                for (
                    (view_name, camera_name, _forward_index),
                    horizontal_forward,
                    view_elevation_deg,
                ) in zip(
                    self._CAMERAS,
                    (front_world, side_world),
                    (0.0, side_elevation_deg),
                    strict=True,
                ):
                    tilt = np.radians(view_elevation_deg)
                    # Frame against the tilted optical axis so an oblique panel
                    # crops the same geometry a level panel would.
                    forward_tilted = (
                        np.cos(tilt) * horizontal_forward - np.sin(tilt) * up_world
                    )
                    up_tilted = (
                        np.sin(tilt) * horizontal_forward + np.cos(tilt) * up_world
                    )
                    distance_m = _framing_distance(
                        subject_points,
                        required_points,
                        center_base,
                        horizontal_forward_base=(
                            base_rotation_world.T @ forward_tilted
                        ),
                        up_base=(base_rotation_world.T @ up_tilted),
                        width=width,
                        height=height,
                        fovy_deg=self._fovy_deg,
                        minimum_m=self._distance_m,
                        padding_m=self._framing_padding_m,
                    )
                    camera, depth_metric = self._render_view(
                        sim,
                        camera_name=camera_name,
                        view_name=view_name,
                        center_world=center_world,
                        horizontal_forward_world=horizontal_forward,
                        up_world=up_world,
                        base_position_world=base_position_world,
                        base_rotation_world=base_rotation_world,
                        width=width,
                        height=height,
                        distance_m=distance_m,
                        elevation_deg=view_elevation_deg,
                    )
                    rendered[view_name] = camera
                    scores[view_name] = _visibility_score(
                        subject_points,
                        camera,
                        depth_metric,
                    )
                if auxiliary_elevation_deg is not None:
                    # A diagonal azimuth exposes both fingers; the steeper
                    # elevation clears the hand body that occludes a 55° pure
                    # side view during transport and placement.
                    auxiliary_horizontal = _unit(
                        front_world + side_world,
                        "contact auxiliary diagonal",
                    )
                    auxiliary_tilt = np.radians(auxiliary_elevation_deg)
                    auxiliary_forward = (
                        np.cos(auxiliary_tilt) * auxiliary_horizontal
                        - np.sin(auxiliary_tilt) * up_world
                    )
                    auxiliary_up = (
                        np.sin(auxiliary_tilt) * auxiliary_horizontal
                        + np.cos(auxiliary_tilt) * up_world
                    )
                    auxiliary_distance = _framing_distance(
                        subject_points,
                        required_points,
                        center_base,
                        horizontal_forward_base=(
                            base_rotation_world.T @ auxiliary_forward
                        ),
                        up_base=base_rotation_world.T @ auxiliary_up,
                        width=width,
                        height=height,
                        fovy_deg=self._fovy_deg,
                        minimum_m=self._distance_m,
                        padding_m=self._framing_padding_m,
                    )
                    auxiliary, _ = self._render_view(
                        sim,
                        camera_name="sideview",
                        view_name="auxiliary",
                        center_world=center_world,
                        horizontal_forward_world=auxiliary_horizontal,
                        up_world=up_world,
                        base_position_world=base_position_world,
                        base_rotation_world=base_rotation_world,
                        width=width,
                        height=height,
                        distance_m=auxiliary_distance,
                        elevation_deg=auxiliary_elevation_deg,
                    )
                    rendered["auxiliary"] = auxiliary
                front_score = scores["front"]
                side_score = scores["side"]
                # Both views must be useful. Maximising the weaker view avoids
                # selecting one excellent image paired with one fully occluded
                # image while preserving the exact camera axes.
                pair_score = 0.7 * min(front_score, side_score) + 0.3 * (
                    0.5 * (front_score + side_score)
                )
                candidates.append(
                    (
                        pair_score,
                        front_sign,
                        side_sign,
                        0.0,
                        front_score,
                        side_score,
                        rendered,
                    )
                )
        finally:
            for camera_id, position, quaternion, fovy in saved:
                sim.model.cam_pos[camera_id] = position
                sim.model.cam_quat[camera_id] = quaternion
                sim.model.cam_fovy[camera_id] = fovy
            if saved:
                sim.forward()

        if not candidates:
            raise RuntimeError("LIBERO contact cameras did not render both views")
        # ``max`` is stable, so equal scores retain the first (canonical)
        # candidate and keep deterministic fixtures byte-identical.
        (
            score,
            front_sign,
            side_sign,
            azimuth_offset_deg,
            front_score,
            side_score,
            rendered,
        ) = max(
            candidates,
            key=lambda item: item[0],
        )
        _attach_current_gripper_masks(
            sim,
            rendered,
            saved_cameras=saved,
            width=width,
            height=height,
        )
        return ContactCameraPair(
            front=rendered["front"],
            side=rendered["side"],
            selection=ContactCameraSelection(
                front_sign=front_sign,
                side_sign=side_sign,
                score=score,
                front_visibility=front_score,
                side_visibility=side_score,
                azimuth_offset_deg=azimuth_offset_deg,
                side_elevation_deg=side_elevation_deg,
            ),
            auxiliary=rendered.get("auxiliary"),
        )

    def _render_view(
        self,
        sim: Any,
        *,
        camera_name: str,
        view_name: Literal["front", "side", "auxiliary"],
        center_world: np.ndarray,
        horizontal_forward_world: np.ndarray,
        up_world: np.ndarray,
        base_position_world: np.ndarray,
        base_rotation_world: np.ndarray,
        width: int,
        height: int,
        distance_m: float,
        elevation_deg: float,
    ) -> tuple[dict[str, Any], np.ndarray]:
        # Tilt around the framed centre so the panel keeps its zoom: the
        # camera slides along a sphere of radius ``distance_m`` instead of
        # simply rising and drifting away from the subject.
        tilt = np.radians(float(elevation_deg))
        return _render_rgbd_camera(
            sim,
            camera_name=camera_name,
            view_name=view_name,
            center_world=center_world,
            horizontal_forward_world=horizontal_forward_world,
            up_world=up_world,
            base_position_world=base_position_world,
            base_rotation_world=base_rotation_world,
            width=width,
            height=height,
            distance_m=float(distance_m) * float(np.cos(tilt)),
            elevation_m=float(distance_m) * float(np.sin(tilt)),
            fovy_deg=self._fovy_deg,
        )


def _render_rgbd_camera(
    sim: Any,
    *,
    camera_name: str,
    view_name: str,
    center_world: np.ndarray,
    horizontal_forward_world: np.ndarray,
    up_world: np.ndarray,
    base_position_world: np.ndarray,
    base_rotation_world: np.ndarray,
    width: int,
    height: int,
    distance_m: float,
    elevation_m: float,
    fovy_deg: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Render one temporary MuJoCo camera and return RGB plus private calibration."""

    camera_id = int(sim.model.camera_name2id(camera_name))
    camera_position_world = (
        center_world
        - float(distance_m) * _unit(horizontal_forward_world, "camera horizontal")
        + float(elevation_m) * up_world
    )
    forward_world = center_world - camera_position_world
    camera_rotation_world = _look_at_rotation(forward_world, up_world)
    camera_quaternion_xyzw = Rotation.from_matrix(camera_rotation_world).as_quat()
    sim.model.cam_pos[camera_id] = camera_position_world
    sim.model.cam_quat[camera_id] = np.roll(camera_quaternion_xyzw, 1)
    sim.model.cam_fovy[camera_id] = float(fovy_deg)
    sim.forward()
    rendered = sim.render(
        camera_name=camera_name,
        width=width,
        height=height,
        depth=True,
    )
    if not isinstance(rendered, (tuple, list)) or len(rendered) != 2:
        raise RuntimeError("MuJoCo camera did not return RGB-D")
    rgb_raw, depth_raw = rendered
    rgb = np.ascontiguousarray(np.asarray(rgb_raw, dtype=np.uint8)[::-1])
    normalized_depth = np.asarray(depth_raw, dtype=np.float64)[::-1]
    if (
        normalized_depth.shape != (height, width)
        or not np.isfinite(normalized_depth).all()
        or normalized_depth.min() < -1e-6
        or normalized_depth.max() > 1.0 + 1e-6
    ):
        raise RuntimeError("MuJoCo camera returned invalid normalized depth")
    depth_metric = _metric_depth(sim, np.clip(normalized_depth, 0.0, 1.0))
    intrinsics = _intrinsics(width, height, float(fovy_deg))
    base_from_camera = _base_from_image_camera(
        base_position_world,
        base_rotation_world,
        camera_position_world,
        camera_rotation_world,
    )
    camera = {
        "images": {"rgb": rgb},
        "intrinsics": intrinsics,
        "pose_mat": base_from_camera,
        "view_name": view_name,
        # Kept only until the provider renders the matching MuJoCo
        # segmentation.  These values are removed before ContactCameraPair is
        # returned and never enter a ContextPacket.
        "_mujoco_camera_name": camera_name,
        "_mujoco_position_world": camera_position_world.copy(),
        "_mujoco_quaternion_wxyz": np.roll(camera_quaternion_xyzw, 1),
        "_mujoco_fovy_deg": float(fovy_deg),
    }
    return camera, depth_metric


def _attach_current_gripper_masks(
    sim: Any,
    rendered: dict[str, dict[str, Any]],
    *,
    saved_cameras: list[tuple[int, np.ndarray, np.ndarray, float]],
    width: int,
    height: int,
) -> None:
    """Attach pixel-exact current-gripper masks from the same MuJoCo camera.

    Contact RGB comes from robosuite's MuJoCo Panda model.  Projecting a
    separate Panda URDF onto that image is not exact: the two descriptions use
    different finger origins and visual meshes.  Element segmentation instead
    shares the RGB camera, geometry, FK and occlusion buffer, so its visible
    pixels align by construction.

    Two masks are produced.  ``finger_mask`` covers the visible fingers.
    ``palm_floor_mask`` is a thin band on the palm's lower silhouette edge:
    during a top-down approach the palm underside, not the fingertips, is what
    first touches the object, and leaving it unmarked draws attention to the
    highlighted fingers instead.  Only the band is marked, because the rest of
    the palm is not a contact surface and a full mask would bury the fingers.

    The raw segmentation is private presenter state.  Only the composited
    raster is policy-visible.
    """

    finger_geom_ids = _visual_geom_ids(sim.model, ("finger1_visual", "finger2_visual"))
    palm_geom_ids = _visual_geom_ids(sim.model, ("hand_visual",))
    cameras = tuple(rendered.values())
    if not finger_geom_ids:
        for camera in cameras:
            camera.pop("_mujoco_camera_name", None)
            camera.pop("_mujoco_position_world", None)
            camera.pop("_mujoco_quaternion_wxyz", None)
            camera.pop("_mujoco_fovy_deg", None)
        return
    try:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - LIBERO requires MuJoCo
            raise RuntimeError("MuJoCo segmentation is unavailable") from exc
        geom_object_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        for camera in cameras:
            camera_name = str(camera["_mujoco_camera_name"])
            camera_id = int(sim.model.camera_name2id(camera_name))
            sim.model.cam_pos[camera_id] = np.asarray(
                camera["_mujoco_position_world"],
                dtype=np.float64,
            )
            sim.model.cam_quat[camera_id] = np.asarray(
                camera["_mujoco_quaternion_wxyz"],
                dtype=np.float64,
            )
            sim.model.cam_fovy[camera_id] = float(camera["_mujoco_fovy_deg"])
            sim.forward()
            raw = sim.render(
                camera_name=camera_name,
                width=width,
                height=height,
                depth=False,
                segmentation=True,
            )
            segmentation = np.asarray(raw, dtype=np.int32)
            if segmentation.shape != (height, width, 2):
                raise RuntimeError(
                    "MuJoCo camera returned invalid element segmentation"
                )
            segmentation = segmentation[::-1]
            is_geom = segmentation[..., 0] == geom_object_type
            camera["finger_mask"] = np.ascontiguousarray(
                is_geom & np.isin(segmentation[..., 1], finger_geom_ids)
            )
            if palm_geom_ids:
                palm = is_geom & np.isin(segmentation[..., 1], palm_geom_ids)
                camera["palm_floor_mask"] = _lower_edge_band(palm, height)
    finally:
        for camera in cameras:
            camera.pop("_mujoco_camera_name", None)
            camera.pop("_mujoco_position_world", None)
            camera.pop("_mujoco_quaternion_wxyz", None)
            camera.pop("_mujoco_fovy_deg", None)
        # The RGB candidate search already restored these cameras once.  The
        # selected-pair segmentation temporarily reused them, so restore them
        # again without advancing physics.
        for camera_id, position, quaternion, fovy in saved_cameras:
            sim.model.cam_pos[camera_id] = position
            sim.model.cam_quat[camera_id] = quaternion
            sim.model.cam_fovy[camera_id] = fovy
        if saved_cameras:
            sim.forward()


def _visual_geom_ids(model: Any, suffixes: tuple[str, ...]) -> tuple[int, ...]:
    """Return robosuite Panda visual geom IDs, independent of model prefix."""

    names = getattr(model, "geom_names", ())
    ids = [
        int(model.geom_name2id(str(name)))
        for name in names
        if name is not None and str(name).endswith(suffixes)
    ]
    return tuple(sorted(set(ids)))


def _lower_edge_band(mask: np.ndarray, height: int) -> np.ndarray:
    """Keep a thin band along the bottom-most visible run of every column.

    On a gravity-stable panel this traces the palm's lower boundary, which is
    the surface that ends a descent.  A band rather than an outline keeps it
    legible after the panel is rescaled for the canvas.
    """

    thickness = max(5, int(round(0.028 * float(height))))
    rows = np.arange(mask.shape[0], dtype=np.int64)[:, None]
    occupied = mask.any(axis=0)
    # ``argmax`` on the row-reversed mask finds the first True from the bottom.
    bottom = (mask.shape[0] - 1) - np.argmax(mask[::-1], axis=0)
    bottom = np.where(occupied, bottom, -1)
    band = (rows <= bottom) & (rows > bottom - thickness)
    return np.ascontiguousarray(band)


def _base_pose_world(env: Any, sim: Any) -> tuple[np.ndarray, np.ndarray]:
    body_id = int(env.base_link_idx)
    position = np.asarray(sim.data.xpos[body_id], dtype=np.float64).reshape(3)
    quaternion_wxyz = np.asarray(sim.data.xquat[body_id], dtype=np.float64).reshape(4)
    rotation = Rotation.from_quat(np.roll(quaternion_wxyz, -1)).as_matrix()
    return position.copy(), rotation


def _look_at_rotation(forward_world: np.ndarray, up_hint_world: np.ndarray) -> np.ndarray:
    """Return MuJoCo camera axes; cameras look along local -Z."""

    forward = _unit(forward_world, "camera forward")
    up_hint = _unit(up_hint_world, "camera up")
    right = _unit(np.cross(forward, up_hint), "camera right")
    up = _unit(np.cross(right, forward), "camera corrected up")
    rotation = np.column_stack((right, up, -forward))
    if np.linalg.det(rotation) < 0.999:
        raise ValueError("contact camera rotation is not right-handed")
    return rotation


def _base_from_image_camera(
    base_position_world: np.ndarray,
    base_rotation_world: np.ndarray,
    camera_position_world: np.ndarray,
    camera_rotation_world: np.ndarray,
) -> np.ndarray:
    """Match the image-camera convention already used by FrankaLiberoTask."""

    base_from_mujoco_rotation = base_rotation_world.T @ camera_rotation_world
    # MuJoCo cameras look along local -Z with +Y up.  After vertically
    # flipping ``sim.render`` into normal image row order, our pinhole helpers
    # expect +X right, +Y down and +Z forward.  This is the same
    # ``Ry(pi) @ Rz(pi)`` correction used by FrankaLiberoTask, i.e.
    # diag(+1, -1, -1).  Using only ``Ry(pi)`` mirrors both overlay axes while
    # leaving the directly rendered RGB unchanged.
    base_from_image_rotation = base_from_mujoco_rotation @ np.diag((1.0, -1.0, -1.0))
    base_from_image_translation = base_rotation_world.T @ (
        camera_position_world - base_position_world
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = base_from_image_rotation
    transform[:3, 3] = base_from_image_translation
    return transform


def _intrinsics(width: int, height: int, fovy_deg: float) -> np.ndarray:
    focal = 0.5 * height / np.tan(np.deg2rad(fovy_deg) * 0.5)
    return np.array(
        [
            [focal, 0.0, 0.5 * width],
            [0.0, focal, 0.5 * height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _metric_depth(sim: Any, normalized_depth: np.ndarray) -> np.ndarray:
    extent = float(sim.model.stat.extent)
    near = float(sim.model.vis.map.znear) * extent
    far = float(sim.model.vis.map.zfar) * extent
    if not np.isfinite((near, far)).all() or near <= 0.0 or far <= near:
        raise RuntimeError("MuJoCo camera near/far planes are invalid")
    return near / (1.0 - normalized_depth * (1.0 - near / far))


def _visibility_score(
    subject_points_base: np.ndarray | None,
    camera: dict[str, Any],
    depth_metric: np.ndarray,
) -> float:
    """Score in-frame subject coverage that is not hidden by nearer geometry."""

    if subject_points_base is None or len(subject_points_base) == 0:
        return 1.0
    points = subject_points_base
    if len(points) > 2048:
        indices = np.linspace(0, len(points) - 1, 2048, dtype=np.int64)
        points = points[indices]
    from vaw.context_runtime.geometry import project_world_to_pixel

    projected = project_world_to_pixel(
        points,
        camera["intrinsics"],
        camera["pose_mat"],
    )
    height, width = depth_metric.shape
    u = projected[:, 0]
    v = projected[:, 1]
    z = projected[:, 2]
    valid = (
        np.isfinite(projected).all(axis=1)
        & (z > 0.01)
        & (u >= 1.0)
        & (u < width - 1.0)
        & (v >= 1.0)
        & (v < height - 1.0)
    )
    if not np.any(valid):
        return 0.0
    valid_indices = np.flatnonzero(valid)
    px = np.rint(u[valid]).astype(np.int64)
    py = np.rint(v[valid]).astype(np.int64)
    observed = depth_metric[py, px]
    # Sensor-derived points are not pixel-perfect in a newly rendered view.
    # Only a clearly nearer surface counts as an occluder.
    visible = observed >= z[valid] - 0.025
    visible_ratio = float(np.count_nonzero(visible)) / float(len(points))
    if not np.any(visible):
        return 0.0
    visible_projected = projected[valid_indices[visible], :2]
    span = np.ptp(visible_projected, axis=0)
    coverage = float(np.prod(np.maximum(span, 0.0))) / float(width * height)
    coverage_factor = 0.8 + 0.2 * min(coverage / 0.06, 1.0)
    return visible_ratio * coverage_factor


def _framing_distance(
    subject_points_base: np.ndarray | None,
    required_points_base: np.ndarray | None,
    center_base: np.ndarray,
    *,
    horizontal_forward_base: np.ndarray,
    up_base: np.ndarray,
    width: int,
    height: int,
    fovy_deg: float,
    minimum_m: float,
    padding_m: float,
) -> float:
    """Choose a zoom that contains the task-relevant geometry.

    Contact panels are very wide and vertically shallow.  A fixed distance
    clips a container below a high placement waypoint even when both are part
    of the same local decision.  Fit robust subject extents with gripper-sized
    padding while preserving the configured close-up distance for grasps.
    """

    points = _points3(subject_points_base)
    required_points = _points3(required_points_base)
    if points is None and required_points is None:
        return float(minimum_m)
    center = np.asarray(center_base, dtype=np.float64).reshape(3)
    forward = _unit(horizontal_forward_base, "camera horizontal")
    up = _unit(up_base, "camera up")
    right = _unit(np.cross(forward, up), "camera right")
    vertical_limits: list[float] = []
    horizontal_limits: list[float] = []
    depth_limits: list[float] = []
    if points is not None:
        relative = points - center
        projected_up = relative @ up
        projected_right = relative @ right
        projected_forward = relative @ forward
        vertical_limits.append(
            0.5 * float(np.quantile(projected_up, 0.98) - np.quantile(projected_up, 0.02))
        )
        horizontal_limits.append(
            0.5
            * float(
                np.quantile(projected_right, 0.98)
                - np.quantile(projected_right, 0.02)
            )
        )
        depth_limits.append(-float(np.quantile(projected_forward, 0.02)) + 0.08)
    if required_points is not None:
        required_relative = required_points - center
        # Required virtual geometry is exact and usually much smaller than the
        # RGB-D subject cloud. Do not let robust subject quantiles discard it.
        vertical_limits.append(float(np.max(np.abs(required_relative @ up))))
        horizontal_limits.append(float(np.max(np.abs(required_relative @ right))))
        depth_limits.append(-float(np.min(required_relative @ forward)) + 0.08)
    vertical_half = max(vertical_limits, default=0.0)
    horizontal_half = max(horizontal_limits, default=0.0)
    # Required target vertices already include the target gripper whenever
    # available.  Keep only a narrow safety margin so the contact geometry is
    # genuinely close-up instead of being surrounded by empty floor.
    vertical_half += float(padding_m)
    horizontal_half += float(padding_m)
    tangent = float(np.tan(np.deg2rad(fovy_deg) * 0.5))
    aspect = float(width) / float(height)
    fit_vertical = vertical_half / max(tangent, 1e-6)
    fit_horizontal = horizontal_half / max(tangent * aspect, 1e-6)
    fit_depth = max(depth_limits, default=0.0)
    required = 1.08 * max(fit_vertical, fit_horizontal, fit_depth)
    # The ceiling is expressed as a multiple of the close-up distance so that
    # changing the FOV rescales the whole framing envelope with it.
    maximum_m = _MAX_FRAMING_DISTANCE_RATIO * float(minimum_m)
    return float(np.clip(max(float(minimum_m), required), minimum_m, maximum_m))


def _horizontal_axis(value: Any, up: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64).reshape(3)
    horizontal = axis - float(np.dot(axis, up)) * up
    return _unit(horizontal, name)


def _points3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    points = np.asarray(value, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    return np.ascontiguousarray(points) if len(points) else None


def _vector3(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    return result


def _quaternion_xyzw(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(result))
    if result.shape != (4,) or not np.isfinite(result).all() or norm <= 1e-12:
        raise ValueError(f"{name} must be a finite xyzw quaternion")
    return result / norm


def _unit(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(result).all() or norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero and finite")
    return result / norm


__all__ = [
    "ContactCameraPair",
    "ContactCameraRequest",
    "ContactCameraSelection",
    "LiberoContactCameraProvider",
    "LiberoOppositeSceneCameraProvider",
    "OppositeSceneCameraRequest",
]
