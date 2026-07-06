"""Agent-facing unified API facade for CaP SkillBench.

This module deliberately does not replace CapX's backend APIs.  It is installed
inside a CodeExecutionEnvBase execution namespace and aliases a small, stable
manipulation contract onto whatever CapX functions the selected environment
already injected.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def install_unified_api(namespace: dict[str, Any], *, profile: str = "auto") -> None:
    """Install unified helper functions into an execution namespace.

    Args:
        namespace: Usually ``globals()`` from code executed by
            ``CodeExecutionEnvBase.step``. It already contains the raw CapX helper
            functions installed by the selected backend.
        profile: Reserved for explicit future routing. ``"auto"`` infers camera
            names and motion wrappers from the injected functions/observation.
    """

    raw = dict(namespace.get("_CAPX_RAW_FUNCTIONS", {}))
    for name, value in list(namespace.items()):
        if callable(value) and not name.startswith("_"):
            raw.setdefault(name, value)
    namespace["_CAPX_RAW_FUNCTIONS"] = raw
    namespace["_CAPX_UNIFIED_PROFILE"] = profile

    def _raw_fn(name: str) -> Callable[..., Any] | None:
        fn = raw.get(name)
        return fn if callable(fn) else None

    def _raw_observation() -> dict[str, Any]:
        fn = _raw_fn("get_observation") or _raw_fn("get_env_observation")
        if fn is None:
            env = namespace.get("env")
            if env is not None and hasattr(env, "get_observation"):
                return env.get_observation()
            raise RuntimeError("No raw observation function is available.")
        return fn()

    def get_observation(*, raw_obs: bool = False) -> dict[str, Any]:
        """Return a normalized observation dict.

        The normalized schema is:

        ``obs["cameras"][name]`` with ``rgb``, ``depth``, ``intrinsics``,
        ``pose_mat``, and ``raw`` fields; ``obs["robot"]["arms"]["default"]``
        with end-effector pose/joints when available; and ``obs["raw"]`` for
        escape-hatch debugging.
        """

        obs = _raw_observation()
        if raw_obs:
            return obs
        return _normalize_observation(obs, raw)

    def get_raw_observation() -> dict[str, Any]:
        """Return the backend CapX observation without normalization."""

        return _raw_observation()

    def capabilities() -> dict[str, Any]:
        """Describe which optional unified operations are backed by this env."""

        return {
            "profile": namespace.get("_CAPX_UNIFIED_PROFILE", "auto"),
            "mobile_base": _raw_fn("navigate_to_pose") is not None,
            "arms": _available_arms(raw),
            "wrist_camera": bool(_camera_from_obs(get_observation()["raw"], "wrist")),
            "native_goto_pose": _raw_fn("goto_pose") is not None,
            "native_query_vlm": _raw_fn("query_vlm") is not None,
            "native_plan_grasp": _raw_fn("plan_grasp") is not None,
        }

    def get_camera(name: str = "main", obs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a normalized camera view by semantic name."""

        normalized = obs if obs is not None else get_observation()
        if "cameras" in normalized:
            cameras = normalized["cameras"]
            if name in cameras:
                return cameras[name]
            if name == "main" and cameras:
                return next(iter(cameras.values()))
        raw_obs = normalized.get("raw", normalized)
        cam = _camera_from_obs(raw_obs, name)
        if cam is None:
            raise KeyError(f"No camera named {name!r}; available={list(_camera_map(raw_obs))}")
        return _normalize_camera(name, cam)

    def query_vlm(prompt: str, images: Any = None, **kwargs: Any) -> str:
        """Ask the configured VLM through the backend when available."""

        fn = _raw_fn("query_vlm")
        if fn is None:
            raise RuntimeError("query_vlm is not available in this backend API.")
        return fn(prompt, images=images, **kwargs)

    def ground_point(text: str, *, camera: str = "main", image: np.ndarray | None = None) -> tuple[int | None, int | None]:
        """Ground text to one image point using the backend pointing API."""

        fn = _raw_fn("point_prompt_molmo")
        if fn is None:
            raise RuntimeError("ground_point is not available in this backend profile.")
        rgb = image if image is not None else get_camera(camera)["rgb"]
        result = fn(rgb, text)
        if isinstance(result, dict):
            value = result.get(text)
            if value is None and result:
                value = next(iter(result.values()))
            if value is not None:
                return tuple(value)  # type: ignore[return-value]
        return (None, None)

    def segment_object(target: str | tuple[float, float] | list[float], *, camera: str = "main", image: np.ndarray | None = None) -> list[dict[str, Any]]:
        """Segment an object by text or point prompt on a semantic camera."""

        rgb = image if image is not None else get_camera(camera)["rgb"]
        if isinstance(target, str):
            fn = _raw_fn("segment_sam3_text_prompt")
            if fn is None:
                raise RuntimeError("text segment_object is not available in this backend profile.")
            return fn(rgb, target)
        fn = _raw_fn("segment_sam3_point_prompt")
        if fn is None:
            raise RuntimeError("point segment_object is not available in this backend profile.")
        return fn(rgb, (float(target[0]), float(target[1])))

    def mask_to_world_points(mask: np.ndarray, *, camera: str = "main", obs: dict[str, Any] | None = None) -> np.ndarray:
        """Convert a mask on a semantic camera to world-frame points."""

        cam = get_camera(camera, obs=obs)
        fn = _raw_fn("mask_to_world_points")
        if fn is not None:
            return fn(mask, cam["depth"], cam["intrinsics"], cam["pose_mat"])
        return _mask_to_world_points(mask, cam["depth"], cam["intrinsics"], cam["pose_mat"])

    def pixel_to_world_point(u: int, v: int, z: float | None = None, *, camera: str = "main", obs: dict[str, Any] | None = None) -> np.ndarray:
        """Deproject one pixel on a semantic camera into the world frame."""

        cam = get_camera(camera, obs=obs)
        if z is None:
            depth = np.asarray(cam["depth"])
            z = float(depth[int(v), int(u)])
        fn = _raw_fn("pixel_to_world_point")
        if fn is not None:
            return fn(int(u), int(v), float(z), cam["intrinsics"], cam["pose_mat"])
        return _pixel_to_world_point(int(u), int(v), float(z), cam["intrinsics"], cam["pose_mat"])

    def depth_to_point_cloud(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        fn = _raw_fn("depth_to_point_cloud")
        return fn(depth, intrinsics) if fn is not None else _depth_to_point_cloud(depth, intrinsics)

    def get_oriented_bounding_box(points: np.ndarray) -> dict[str, Any]:
        fn = _raw_fn("get_oriented_bounding_box_from_3d_points")
        if fn is None:
            raise RuntimeError("get_oriented_bounding_box_from_3d_points is not available.")
        return fn(points)

    def plan_grasps(mask: np.ndarray | None = None, *, camera: str = "main", obs: dict[str, Any] | None = None, segmentation: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Plan grasp candidates from a mask/segmentation on a semantic camera."""

        fn = _raw_fn("plan_grasp")
        if fn is None:
            raise RuntimeError("plan_grasp is not available in this backend API.")
        cam = get_camera(camera, obs=obs)
        seg = segmentation if segmentation is not None else mask
        if seg is None:
            raise ValueError("plan_grasps requires mask or segmentation.")
        return fn(cam["depth"], cam["intrinsics"], np.asarray(seg).astype(np.int32))

    def goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, *, arm: str = "default", approach_z: float = 0.0) -> Any:
        """Move an arm to a world-frame pose using the backend route."""

        native = _raw_fn("goto_pose")
        if native is not None and arm in {"default", "arm0", "left", "right"}:
            try:
                return native(position, quaternion_wxyz, z_approach=approach_z)
            except TypeError:
                return native(position, quaternion_wxyz)

        suffix = _arm_suffix(arm)
        solve = _raw_fn(f"solve_ik_{suffix}") or _raw_fn("solve_ik")
        move = _raw_fn(f"move_to_joints_{suffix}") or _raw_fn("move_to_joints") or _raw_fn("move_to_joint_positions")
        if solve is None or move is None:
            move_hand = _raw_fn("move_hand")
            if move_hand is not None:
                return move_hand((np.asarray(position), np.asarray(quaternion_wxyz)), arm=0 if suffix == "arm0" else 1)
            raise RuntimeError("No route for goto_pose in this backend API.")
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        if approach_z > 0:
            approach = pos.copy()
            approach[2] += approach_z
            _move_with_ik(solve, move, approach, quat, arm)
        return _move_with_ik(solve, move, pos, quat, arm)

    def solve_ik(position: np.ndarray, quaternion_wxyz: np.ndarray, *, arm: str = "default") -> np.ndarray:
        suffix = _arm_suffix(arm)
        fn = _raw_fn(f"solve_ik_{suffix}") or _raw_fn("solve_ik")
        if fn is None:
            raise RuntimeError("solve_ik is not available.")
        try:
            return fn(position, quaternion_wxyz, arm=0 if suffix == "arm0" else 1)
        except TypeError:
            return fn(position, quaternion_wxyz)

    def move_to_joints(joints: np.ndarray, *, arm: str = "default") -> Any:
        suffix = _arm_suffix(arm)
        fn = _raw_fn(f"move_to_joints_{suffix}") or _raw_fn("move_to_joints") or _raw_fn("move_to_joint_positions")
        if fn is None:
            raise RuntimeError("move_to_joints is not available.")
        return fn(joints)

    def open_gripper(*, arm: str = "default") -> Any:
        suffix = _arm_suffix(arm)
        fn = _raw_fn(f"open_gripper_{suffix}") or _raw_fn("open_gripper")
        if fn is None:
            raise RuntimeError("open_gripper is not available.")
        try:
            return fn(arm=0 if suffix == "arm0" else 1)
        except TypeError:
            return fn()

    def close_gripper(*, arm: str = "default") -> Any:
        suffix = _arm_suffix(arm)
        fn = _raw_fn(f"close_gripper_{suffix}") or _raw_fn("close_gripper")
        if fn is None:
            raise RuntimeError("close_gripper is not available.")
        try:
            return fn(arm=0 if suffix == "arm0" else 1)
        except TypeError:
            return fn()

    def home(*, arm: str = "default") -> Any:
        fn = _raw_fn("goto_home_joint_position") or _raw_fn("reset_robot_joints")
        if fn is None:
            raise RuntimeError("home is not available in this backend API.")
        return fn()

    def navigate_to_pose(xy_yaw: Any) -> Any:
        fn = _raw_fn("navigate_to_pose")
        if fn is None:
            raise RuntimeError("navigate_to_pose is not available in this backend API.")
        return fn(xy_yaw)

    exports = {
        "capabilities": capabilities,
        "get_observation": get_observation,
        "get_raw_observation": get_raw_observation,
        "get_camera": get_camera,
        "query_vlm": query_vlm,
        "ground_point": ground_point,
        "segment_object": segment_object,
        "mask_to_world_points": mask_to_world_points,
        "pixel_to_world_point": pixel_to_world_point,
        "depth_to_point_cloud": depth_to_point_cloud,
        "get_oriented_bounding_box": get_oriented_bounding_box,
        "plan_grasps": plan_grasps,
        "goto_pose": goto_pose,
        "solve_ik": solve_ik,
        "move_to_joints": move_to_joints,
        "open_gripper": open_gripper,
        "close_gripper": close_gripper,
        "home": home,
        "navigate_to_pose": navigate_to_pose,
        "rotation_matrix_to_quaternion": raw.get("rotation_matrix_to_quaternion") or _rotation_matrix_to_quaternion,
        "decompose_transform": raw.get("decompose_transform") or _decompose_transform,
        "transform_points": raw.get("transform_points") or _transform_points,
        "interpolate_segment": raw.get("interpolate_segment") or _interpolate_segment,
        "normalize_vector": raw.get("normalize_vector") or _normalize_vector,
        "select_top_down_grasp": raw.get("select_top_down_grasp"),
        "filter_noise": raw.get("filter_noise") or _identity_filter_noise,
        "subsample_point_cloud": raw.get("subsample_point_cloud") or _subsample_point_cloud,
    }
    namespace.update({k: v for k, v in exports.items() if v is not None})


def _available_arms(raw: dict[str, Any]) -> list[str]:
    if "solve_ik_arm0" in raw or "open_gripper_arm0" in raw:
        return ["arm0", "arm1"]
    return ["default"]


def _arm_suffix(arm: str) -> str:
    if arm in {"default", "arm0", "left"}:
        return "arm0"
    if arm in {"arm1", "right"}:
        return "arm1"
    return str(arm)


def _move_with_ik(solve: Callable[..., Any], move: Callable[..., Any], pos: np.ndarray, quat: np.ndarray, arm: str) -> Any:
    try:
        joints = solve(pos, quat, arm=0 if _arm_suffix(arm) == "arm0" else 1)
    except TypeError:
        joints = solve(pos, quat)
    return move(joints)


def _normalize_observation(obs: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    cameras = {name: _normalize_camera(name, cam) for name, cam in _camera_map(obs).items()}
    robot = {"arms": {}}
    cart = obs.get("robot_cartesian_pos")
    joints = obs.get("robot_joint_pos")
    if cart is not None:
        arr = np.asarray(cart)
        robot["arms"]["default"] = {
            "ee_pos": arr[:3],
            "ee_quat_wxyz": arr[3:7] if arr.shape[0] >= 7 else None,
            "gripper": float(arr[7]) if arr.shape[0] >= 8 else None,
            "joints": joints,
        }
    elif callable(raw.get("get_current_eef_pose")):
        try:
            pos, quat = raw["get_current_eef_pose"]()
            robot["arms"]["default"] = {"ee_pos": pos, "ee_quat_wxyz": quat, "gripper": None, "joints": None}
        except Exception:
            pass
    normalized = dict(obs)
    normalized.update({"cameras": cameras, "robot": robot, "raw": obs})
    return normalized


def _camera_map(obs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    named: dict[str, dict[str, Any]] = {}
    seen_ids: set[int] = set()
    if "agentview" in obs:
        named["main"] = obs["agentview"]
        seen_ids.add(id(obs["agentview"]))
    if "robot0_robotview" in obs:
        named["main"] = obs["robot0_robotview"]
        seen_ids.add(id(obs["robot0_robotview"]))
    if "robot0_eye_in_hand" in obs:
        named["wrist"] = obs["robot0_eye_in_hand"]
        seen_ids.add(id(obs["robot0_eye_in_hand"]))

    for key, value in _iter_camera_like(obs):
        if id(value) in seen_ids:
            continue
        semantic = _semantic_camera_name(key, named)
        named.setdefault(semantic, value)
        seen_ids.add(id(value))
    return named


def _camera_from_obs(obs: dict[str, Any], name: str) -> dict[str, Any] | None:
    return _camera_map(obs).get(name)


def _iter_camera_like(obj: Any, prefix: str = ""):
    if not isinstance(obj, dict):
        return
    if "images" in obj and isinstance(obj["images"], dict) and "rgb" in obj["images"]:
        yield prefix, obj
        return
    elif "rgb" in obj and "depth" in obj:
        yield prefix, {"images": {"rgb": obj["rgb"], "depth": obj["depth"]}}
        return
    for key, value in obj.items():
        if isinstance(value, dict):
            yield from _iter_camera_like(value, f"{prefix}/{key}" if prefix else str(key))


def _semantic_camera_name(raw_name: str, existing: dict[str, Any]) -> str:
    low = raw_name.lower()
    if "wrist" in low or "eye_in_hand" in low:
        base = "wrist"
    elif "zed" in low or "agentview" in low or "robotview" in low:
        base = "main"
    elif "left" in low:
        base = "wrist_left"
    elif "right" in low:
        base = "wrist_right"
    else:
        base = "main" if "main" not in existing else raw_name.strip("/") or "camera"
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def _normalize_camera(name: str, cam: dict[str, Any]) -> dict[str, Any]:
    images = cam.get("images", cam)
    depth = images.get("depth")
    if isinstance(depth, np.ndarray) and depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]
    return {
        "name": name,
        "rgb": images.get("rgb"),
        "depth": depth,
        "intrinsics": cam.get("intrinsics"),
        "pose_mat": cam.get("pose_mat"),
        "raw": cam,
    }


def _depth_to_point_cloud(depth_img: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_img)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    h, w = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    z = depth
    x = (x_grid - cx) * z / fx
    y = (y_grid - cy) * z / fy
    return np.dstack((x, y, z))


def _mask_to_world_points(mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(ys) == 0:
        return np.empty((0, 3))
    depth_arr = np.asarray(depth)
    if depth_arr.ndim == 3:
        depth_arr = depth_arr[:, :, 0]
    z_vals = depth_arr[ys, xs]
    valid = z_vals > 0
    xs, ys, z = xs[valid], ys[valid], z_vals[valid]
    if len(xs) == 0:
        return np.empty((0, 3))
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    points_cam = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=-1)
    hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    return (extrinsics @ hom.T).T[:, :3]


def _pixel_to_world_point(u: int, v: int, z: float, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    p_cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z, 1.0])
    return (extrinsics @ p_cam)[:3]


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    import viser.transforms as vtf

    return vtf.SO3.from_matrix(np.asarray(R)).wxyz


def _decompose_transform(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(T)[:3, 3], _rotation_matrix_to_quaternion(np.asarray(T)[:3, :3])


def _transform_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    original_shape = points.shape
    pts = np.asarray(points).reshape(-1, 3)
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    return (transform_matrix @ hom.T).T[:, :3].reshape(original_shape)


def _interpolate_segment(p1: np.ndarray, p2: np.ndarray, step: float = 0.03) -> list[np.ndarray]:
    p1, p2 = np.asarray(p1), np.asarray(p2)
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-6:
        return [p1]
    n = int(np.ceil(dist / step))
    return [p1 + (p2 - p1) * t for t in np.linspace(0, 1, n + 1)]


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v)
    norm = np.linalg.norm(arr)
    return arr if norm < 1e-6 else arr / norm


def _identity_filter_noise(points: np.ndarray, colors: np.ndarray | None = None):
    return points, colors


def _subsample_point_cloud(pc: np.ndarray, max_points: int = 10000) -> np.ndarray:
    if len(pc) > max_points:
        return pc[np.random.choice(len(pc), max_points, replace=False)]
    return pc
