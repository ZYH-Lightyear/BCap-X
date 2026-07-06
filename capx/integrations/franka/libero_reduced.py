import json
import logging
import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

logger = logging.getLogger(__name__)

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase
from capx.integrations.franka.common import (
    apply_tcp_offset,
    close_gripper as _close_gripper,
    extract_arm_joints,
    get_oriented_bounding_box_from_3d_points as _get_obb,
    open_gripper as _open_gripper,
    solve_ik_with_convergence,
)
from capx.integrations.vision.graspnet import init_contact_graspnet, init_contact_graspnet_point_clouds
from capx.integrations.vision.molmo import init_molmo  # noqa: F401  # kept for backward compatibility
from capx.integrations.vision.point_backend import init_point_backend
from capx.integrations.vision.sam3 import init_sam3, init_sam3_box_prompt, init_sam3_point_prompt
from capx.integrations.motion.pyroki import init_pyroki

from capx.utils.camera_utils import obs_get_rgb
from capx.utils.depth_utils import (
    deproject_pixel_to_camera,
    depth_color_to_pointcloud,
    depth_to_pointcloud,
    depth_to_rgb,
)
from capx.utils.visualization_utils import draw_molmo_point, overlay_segmentation_masks

_curobo_api = None

def _get_curobo_api():
    """Lazy import of cuRobo API to avoid warp init before Isaac Sim."""
    global _curobo_api
    if _curobo_api is None:
        from capx.integrations.motion import curobo_api as _mod
        _curobo_api = _mod
    return _curobo_api

from sklearn.cluster import DBSCAN

# ------------------------------- Control API ------------------------------
class FrankaLiberoApiReduced(ApiBase):
    """Robot control helpers for Franka.
    """
    _TCP_OFFSET = np.array([0.0, 0.0, -0.1], dtype=np.float64)

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        # Initialize perception models for non-privileged API
        # if self.use_sam3:
        self.sam3_seg_fn = init_sam3()
        self.sam3_point_prompt_fn = init_sam3_point_prompt()
        self.sam3_box_prompt_fn = init_sam3_box_prompt()

        # else:
        # Pointing backend is selected at runtime via CAPX_POINT_BACKEND
        # (defaults to "molmo"; generic VLM pointing is still available for legacy
        # experiments, but RoboMEx grounding should use the dedicated bbox/point
        # detection helpers rather than raw query_vlm coordinate prompts).
        self.molmo_point_fn = init_point_backend()
            # self.sam2_point_prompt_fn = init_sam2_point_prompt()
        self.grasp_net_plan_fn = init_contact_graspnet()
        self.grasp_net_plan_point_clouds_fn = init_contact_graspnet_point_clouds()
        self.ik_solve_fn = init_pyroki()
        self.camera_name = "agentview"
        self.wrist_camera_name = "robot0_eye_in_hand"
        self.cfg = None
        self._vlm_backend: dict[str, str] = {}
        self._debug_output_dir: pathlib.Path | None = None
        self._debug_block_idx = 0
        self._debug_counter = 0

        # JIT warmup: trigger JAX compilation with a dummy IK call
        try:
            self.ik_solve_fn(
                target_pose_wxyz_xyz=np.array([1, 0, 0, 0, 0.3, 0, 0.5]),
                prev_cfg=None,
            )
        except Exception:
            pass

    def set_debug_context(self, output_dir: str, block_idx: int) -> None:
        self._debug_output_dir = pathlib.Path(output_dir)
        self._debug_block_idx = block_idx
        self._debug_counter = 0

    def configure_vlm_backend(
        self,
        *,
        model: str | None = None,
        server_url: str | None = None,
        api_key: str | None = None,
        coord_space: str | None = None,
    ) -> None:
        """Configure VLM APIs for this API instance without relying on process env."""

        cfg = {
            "model": model,
            "server_url": server_url,
            "api_key": api_key,
            "coord_space": coord_space,
        }
        self._vlm_backend = {k: str(v) for k, v in cfg.items() if v not in (None, "")}

    def _vlm_backend_value(self, key: str, env_name: str, default: str | None = None) -> str | None:
        import os

        return getattr(self, "_vlm_backend", {}).get(key) or os.getenv(env_name) or default

    def _save_debug_overlay(self, name: str, image: np.ndarray) -> None:
        if self._debug_output_dir is None:
            return
        self._debug_output_dir.mkdir(parents=True, exist_ok=True)
        path = self._debug_output_dir / (
            f"block_{self._debug_block_idx:02d}_{self._debug_counter:03d}_{name}.png"
        )
        self._debug_counter += 1
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)

    @staticmethod
    def _draw_label(image: np.ndarray, text: str) -> np.ndarray:
        out = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(out, "RGBA")
        draw.rectangle((8, 8, min(out.width - 8, 520), 34), fill=(0, 0, 0, 165))
        draw.text((16, 14), text[:80], fill=(255, 255, 255, 255))
        return np.asarray(out)

    @staticmethod
    def _draw_boxes(image: np.ndarray, results: list[dict[str, Any]]) -> np.ndarray:
        out = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(out)
        for result in results[:5]:
            box = result.get("box")
            if box is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                draw.rectangle((x1, y1, x2, y2), outline=(255, 220, 0), width=3)
        return np.asarray(out)

    @staticmethod
    def _draw_grasp_points(
        image: np.ndarray,
        grasp_poses: np.ndarray,
        grasp_scores: np.ndarray,
        intrinsics: np.ndarray,
        *,
        top_k: int = 8,
    ) -> np.ndarray:
        out = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(out)
        order = np.argsort(grasp_scores)[::-1][:top_k]
        for rank, idx in enumerate(order):
            pose = grasp_poses[idx]
            xyz = pose[:3, 3]
            if xyz[2] <= 0:
                continue
            uvw = intrinsics @ xyz
            x, y = int(uvw[0] / uvw[2]), int(uvw[1] / uvw[2])
            color = (0, 255, 80) if rank == 0 else (255, 180, 0)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=3)
            draw.text((x + 8, y - 8), f"{rank}:{grasp_scores[idx]:.2f}", fill=color)
        return np.asarray(out)


    def functions(self) -> dict[str, Any]:
        fns = {}
        fns["get_observation"] = self.get_observation
        fns["segment_sam3_text_prompt"] = self.segment_sam3_text_prompt
        fns["segment_sam3_point_prompt"] = self.segment_sam3_point_prompt
        fns["segment_sam3_box_prompt"] = self.segment_sam3_box_prompt
        fns["point_prompt_molmo"] = self.point_prompt_molmo
        fns["query_vlm"] = self.query_vlm
        fns["vlm_bbox_detection"] = self.vlm_bbox_detection
        fns["vlm_point_detection"] = self.vlm_point_detection
        fns["parse_vlm_detections"] = self.parse_vlm_detections
        fns["plan_grasp"] = self.plan_grasp
        fns["plan_grasp_from_point_clouds"] = self.plan_grasp_from_point_clouds
        fns["get_oriented_bounding_box_from_3d_points"] = (
            self.get_oriented_bounding_box_from_3d_points
        )
        fns["solve_ik"] = self.solve_ik
        fns["move_to_joints"] = self.move_to_joints
        fns["open_gripper"] = self.open_gripper
        fns["close_gripper"] = self.close_gripper
        fns["goto_pose"] = self.goto_pose

        fns["goto_home_joint_position"] = self.goto_home_joint_position
        fns["subsample_point_cloud"] = self.subsample_point_cloud
        fns["filter_noise"] = self.filter_noise

        # # CuRobo, uncomment these for the coding agent to use them!
        # fns["parse_grasp_poses_for_curobo"] = self.parse_grasp_poses_for_curobo
        # fns["plan_grasp_trajectory"] = self.plan_grasp_trajectory
        # fns["plan_with_grasped_object"] = self.plan_with_grasped_object
        # fns["execute_joint_trajectory"] = self.execute_joint_trajectory
        return fns


    def get_observation(self) -> dict[str, Any]:
        """Get the observation of the environment.
        Returns:
            observation:
                A dictionary containing the observation of the environment.
                The dictionary contains the following keys:
                - ["agentview"]["images"]["rgb"]: Current color camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["agentview"]["images"]["depth"]: Current depth camera image as a numpy array of shape (H, W), dtype float32.
                - ["agentview"]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["agentview"]["pose_mat"]: Camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot0_eye_in_hand"]["images"]["rgb"]: Current wrist camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["robot0_eye_in_hand"]["images"]["depth"]: Current wrist camera depth image as a numpy array of shape (H, W), dtype float32.
                - ["robot0_eye_in_hand"]["intrinsics"]: Wrist camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["robot0_eye_in_hand"]["pose_mat"]: Wrist camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot_cartesian_pos"]: Current end-effector (panda_hand) pose in the robot/world frame as a numpy array of shape (8,), dtype float64. The first 3 elements are the robot's end-effector XYZ, the next 4 elements are the quaternion wxyz, and the last element is the gripper position normalized, 0 (closed) to 1 (open).
                - ["robot_joint_pos"]: Current joint positions as a numpy array of shape (7,), dtype float64. The last element is the gripper position normalized, 0 (closed) to 1 (open).
        """
        obs = self._env.get_observation()
        obs[self.camera_name]["images"]["depth"] = obs[self.camera_name]["images"]["depth"].squeeze(-1)
        obs[self.wrist_camera_name]["images"]["depth"] = obs[self.wrist_camera_name]["images"]["depth"].squeeze(-1)
        return obs


    def segment_sam3_point_prompt(
        self,
        rgb: np.ndarray,
        point_coords: tuple[float, float],
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image, optionally conditioned on an image coordinate point prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            point_coords:
                (x, y) pixel coordinates of the point prompt.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool,
                              where True means the pixel belongs to the instance.
                  - "score": float confidence score.

        Example:
            >>> rgb = obs["agentview"]["images"]["rgb"]
            >>> masks = segment_sam3_point_prompt(rgb, (100, 100))
        """
        results = self.sam3_point_prompt_fn(Image.fromarray(rgb), point_coords)
        masks = [r["mask"] for r in results if "mask" in r]
        overlay = overlay_segmentation_masks(rgb, masks) if masks else rgb.copy()
        point_overlay = draw_molmo_point(
            overlay,
            {"point_prompt": (int(point_coords[0]), int(point_coords[1]))},
        )
        self._save_debug_overlay(
            "sam3_point_prompt",
            self._draw_label(point_overlay, f"SAM3 point prompt: {point_coords}"),
        )
        return results

    def segment_sam3_box_prompt(
        self,
        rgb: np.ndarray,
        box: list[float] | tuple[float, float, float, float],
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation conditioned on an image-coordinate box prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            box:
                [x1, y1, x2, y2] pixel coordinates.
        """
        results = self.sam3_box_prompt_fn(Image.fromarray(rgb), box)
        masks = [r["mask"] for r in results if "mask" in r]
        overlay = overlay_segmentation_masks(rgb, masks) if masks else rgb.copy()
        overlay = self._draw_boxes(overlay, [{"box": box}])
        self._save_debug_overlay(
            "sam3_box_prompt",
            self._draw_label(overlay, f"SAM3 box prompt: {[round(float(v), 1) for v in box]}"),
        )
        return results

    def segment_sam3_text_prompt(
        self,
        rgb: np.ndarray,
        text_prompt: str,
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image conditioned on a text prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            text_prompt:
                Text prompt for SAM3 segmentation.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool,
                              where True means the pixel belongs to the instance.
                  - "box": list [x1, y1, x2, y2] in pixel coordinates.
                  - "score": float confidence score.
        Note:
            Returns an empty list if no results are found.

        Example:
            >>> rgb = obs["agentview"]["images"]["rgb"]
            >>> masks = segment_sam3(rgb, text_prompt="red mug")
        """
        results = self.sam3_seg_fn(rgb, text_prompt=text_prompt)
        if len(results) == 0:
            print(f"[segment_sam3_text_prompt] SAM3 returned no results for prompt: '{text_prompt}'")
            self._save_debug_overlay(
                "sam3_text_prompt_empty",
                self._draw_label(rgb, f"SAM3 no result: {text_prompt}"),
            )
            return []
        masks = [r["mask"] for r in results if "mask" in r]
        overlay = overlay_segmentation_masks(rgb, masks) if masks else rgb.copy()
        overlay = self._draw_boxes(overlay, results)
        self._save_debug_overlay(
            "sam3_text_prompt",
            self._draw_label(overlay, f"SAM3 text prompt: {text_prompt}"),
        )
        return results

    # --------------------------------------------------------------------- #
    # Molmo point prompt
    # --------------------------------------------------------------------- #
    def point_prompt_molmo(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> dict[str, tuple[int | None, int | None]]:
        """Use Molmo to point to a coordinate in the image based on a text prompt.

        Args:
            image: np.ndarray: The RGB image to process. Shape: (H, W, 3), dtype uint8.
            text_prompt: str: The text prompt to point to.

        Returns:
            dict[str, tuple[int | None, int | None]]: Pixel coordinates for each
            object query; (None, None) if parsing failed.
        """
        result = self.molmo_point_fn(Image.fromarray(image), objects=[text_prompt])
        overlay = draw_molmo_point(image, result)
        self._save_debug_overlay(
            "point_prompt",
            self._draw_label(overlay, f"Point prompt: {text_prompt}"),
        )
        return result

    # --------------------------------------------------------------------- #
    # 通用 VLM 查询(把 capx.llm.client.query_model 暴露成 code-block API)
    # --------------------------------------------------------------------- #
    def query_vlm(
        self,
        prompt: Any,
        images: Any = None,
        *,
        image: Any = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Ask a vision-language model a question about image(s) and get its raw text reply.

        A thin, general-purpose bridge to the same LLM proxy the planner uses, so code
        blocks can do their own visual reasoning (e.g. read a label, classify scene
        state, sanity-check an annotated image). You parse the returned string yourself.
        Do not use this API for spatial grounding. Use ``vlm_bbox_detection`` for boxes
        and ``vlm_point_detection`` for points.

        Model/endpoint come from the API instance configuration when RoboMEx launches
        the env. Other CapX launch paths may still use environment variables:
          - CAPX_VLM_MODEL       (default "vapi/gpt-5.5", or the
            RoboMEx run model when launched through RoboMEx)
          - CAPX_VLM_SERVER_URL  (default "http://localhost:8110/chat/completions")
          - CAPX_VLM_API_KEY     (optional)

        Preferred call:
            ``query_vlm(prompt, images=rgb_or_list)``

        Compatibility calls accepted to avoid wasting robot-agent turns:
            ``query_vlm(rgb_or_path, prompt)``
            ``query_vlm(prompt=prompt, image=rgb_or_path)``

        Args:
            prompt: Instruction/question for the VLM. Be explicit about the output
                format you want (e.g. ask for a JSON object) so it is easy to parse.
            images: Optional RGB image/path/PIL image, or a list of them, to attach.
            temperature: Decoding temperature (default 0.0 for deterministic parsing).
            max_tokens: Maximum response tokens. VLM grounding usually needs only a
                short JSON answer; pass a larger value only for open-ended visual QA.

        Returns:
            str: The model's text reply, verbatim.

        Example:
            >>> rgb = get_observation()["agentview"]["images"]["rgb"]
            >>> reply = query_vlm(
            ...     "Reply ONLY JSON {\"state\":\"open|closed|unknown\"}: is the drawer open?",
            ...     images=rgb,
            ... )
        """
        import base64
        import io
        import os
        from os import PathLike

        from capx.llm.client import ModelQueryArgs, query_model

        def _is_image_like(value: Any) -> bool:
            if isinstance(value, (np.ndarray, Image.Image, PathLike)):
                return True
            if isinstance(value, str):
                lower = value.lower()
                if lower.startswith("data:image/"):
                    return True
                if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                    return True
                try:
                    return pathlib.Path(value).is_file()
                except OSError:
                    return False
            return False

        if image is not None:
            if images is not None:
                raise TypeError("query_vlm received both images= and image=; pass only one")
            images = image

        # Backward-compatible robot-agent pattern: query_vlm(image, prompt).
        if _is_image_like(prompt) and isinstance(images, str):
            prompt, images = images, prompt

        if not isinstance(prompt, str):
            raise TypeError(
                "query_vlm prompt must be text. Use query_vlm(prompt, images=rgb) "
                "or the compatibility form query_vlm(rgb_or_path, prompt)."
            )

        def _image_to_pil(value: Any) -> Image.Image:
            if isinstance(value, Image.Image):
                return value.convert("RGB")
            if isinstance(value, (str, PathLike)):
                text = str(value)
                if text.startswith("data:image/"):
                    raise TypeError("query_vlm images= does not accept pre-encoded data URLs")
                return Image.open(text).convert("RGB")
            arr = np.asarray(value)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
                raise ValueError(
                    f"query_vlm image must have shape (H,W,3/4), got {arr.shape!r}"
                )
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            return Image.fromarray(arr).convert("RGB")

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if images is not None:
            img_list = images if isinstance(images, list) else [images]
            for im in img_list:
                buf = io.BytesIO()
                _image_to_pil(im).save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                )

        args = ModelQueryArgs(
            model=model or self._vlm_backend_value("model", "CAPX_VLM_MODEL", "vapi/gpt-5.5"),
            server_url=self._vlm_backend_value(
                "server_url",
                "CAPX_VLM_SERVER_URL",
                "http://localhost:8110/chat/completions",
            ),
            api_key=self._vlm_backend_value("api_key", "CAPX_VLM_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = query_model(args, [{"role": "user", "content": content}])
        return result.get("content", "") if isinstance(result, dict) else str(result)

    def _resolve_vlm_grounding_model(self, model: str | None) -> str:
        return model or self._vlm_backend_value("model", "CAPX_VLM_MODEL", "vapi/gpt-5.5") or "vapi/gpt-5.5"

    def _vlm_grounding_coord_space(self, model: str | None, coord_space: str | None) -> str:
        override = coord_space or self._vlm_backend_value("coord_space", "CAPX_VLM_COORD_SPACE")
        if override in {"auto", "pixel", "norm1000", "fraction"}:
            return override
        resolved_model = self._resolve_vlm_grounding_model(model).lower()
        if "qwen" in resolved_model:
            return "norm1000"
        return "pixel"

    def vlm_bbox_detection(
        self,
        rgb: np.ndarray,
        target_name: str,
        *,
        model: str | None = None,
        coord_space: str | None = None,
    ) -> list[float]:
        """Locate a target with the VLM and return [x1, y1, x2, y2] pixel coordinates."""
        H, W = rgb.shape[:2]
        resolved_model = self._resolve_vlm_grounding_model(model)
        resolved_coord_space = self._vlm_grounding_coord_space(resolved_model, coord_space)
        coord_instruction = (
            "using 0-1000 normalized coordinates"
            if resolved_coord_space == "norm1000"
            else "using REAL PIXEL coordinates"
        )
        prompt = (
            "You are given a robot scene image. "
            f"Find the single object or target that best matches: '{target_name}'. "
            f"The image size is width={W}, height={H}. "
            f"Reply ONLY JSON {{\"box\": [x1, y1, x2, y2]}} {coord_instruction}. "
            "The box must tightly cover the target itself. No prose."
        )
        reply = self.query_vlm(
            prompt, images=rgb, model=resolved_model, temperature=0.0, max_tokens=160
        )
        det = self.parse_vlm_detections(reply, image=rgb, coord_space=resolved_coord_space)
        assert det["boxes"], f"VLM bbox detection failed for '{target_name}': {reply!r}"
        box = [float(v) for v in det["boxes"][0]]
        self._save_debug_overlay(
            "vlm_bbox_detection",
            self._draw_label(self._draw_boxes(rgb, [{"box": box}]), f"VLM bbox: {target_name}"),
        )
        return box

    def vlm_point_detection(
        self,
        rgb: np.ndarray,
        target_name: str,
        *,
        model: str | None = None,
        coord_space: str | None = None,
    ) -> list[float]:
        """Locate a target with the VLM and return [x, y] pixel coordinates."""
        H, W = rgb.shape[:2]
        resolved_model = self._resolve_vlm_grounding_model(model)
        resolved_coord_space = self._vlm_grounding_coord_space(resolved_model, coord_space)
        coord_instruction = (
            "using 0-1000 normalized coordinates"
            if resolved_coord_space == "norm1000"
            else "using REAL PIXEL coordinates"
        )
        prompt = (
            "You are given a robot scene image. "
            f"Find the single point at the center of the object or target: '{target_name}'. "
            f"The image size is width={W}, height={H}. "
            f"Reply ONLY JSON {{\"point\": [x, y]}} {coord_instruction}. No prose."
        )
        reply = self.query_vlm(
            prompt, images=rgb, model=resolved_model, temperature=0.0, max_tokens=160
        )
        det = self.parse_vlm_detections(reply, image=rgb, coord_space=resolved_coord_space)
        assert det["points"], f"VLM point detection failed for '{target_name}': {reply!r}"
        point = [float(v) for v in det["points"][0]]
        overlay = draw_molmo_point(rgb, {"point_prompt": (int(point[0]), int(point[1]))})
        self._save_debug_overlay(
            "vlm_point_detection",
            self._draw_label(overlay, f"VLM point: {target_name}"),
        )
        return point

    def parse_vlm_detections(
        self,
        reply: str,
        image: np.ndarray | tuple[int, int] | None = None,
        *,
        coord_space: str = "auto",
    ) -> dict[str, Any]:
        """Robustly parse object boxes/points out of a free-form VLM grounding reply.

        This is a low-level parser used by ``vlm_bbox_detection`` and
        ``vlm_point_detection``. Different models wrap their grounding answer
        differently: ``{"box": [x1,y1,x2,y2]}``, a markdown-fenced JSON array of
        ``{"bbox_2d": [...], "label": ...}``, or ``{"point_2d": [x, y]}``.
        This helper tolerates all of those plus surrounding
        prose, ```` ```json ```` fences and trailing junk, so callers never have to
        hand-roll a brittle regex (a single missed key like ``box`` vs ``bbox_2d``
        otherwise silently yields no detection).

        Args:
            reply: Raw VLM grounding string.
            image: Optional RGB image ``(H, W, 3)`` or an explicit ``(H, W)`` shape.
                When provided, coordinates are rescaled to absolute pixels and clipped
                to the image; when ``None`` the raw numbers are returned unchanged.
            coord_space: How to interpret the raw coordinates.

                - ``"auto"`` (default): treat coordinates as real pixels, except
                  values in ``[0, 1]`` are treated as fractions. RoboMEx no longer
                  guesses a 0-1000 normalized convention because that corrupts GPT
                  pixel boxes in 800x512 LIBERO images.
                - ``"norm1000"``: always treat as ``0-1000`` normalized.
                - ``"fraction"``: always treat as ``0-1`` fractions.
                - ``"pixel"``: already absolute pixels, no rescaling.

        Returns:
            dict with keys:
                - ``"boxes"``: list of ``[x1, y1, x2, y2]`` (pixels if an image/shape
                  was given), each normalized so ``x1<=x2`` and ``y1<=y2``.
                - ``"points"``: list of ``[x, y]`` = the center of every parsed box,
                  followed by any explicit point detections. ``points[0]`` is the most
                  convenient single location for the first detection.
                - ``"labels"``: list of label strings (or ``None``) aligned to the
                  boxes-then-points detection order.
                - ``"n"``: total number of detections parsed.
            All lists are empty when nothing parseable is found; this never raises on
            malformed input.

        Example:
            >>> rgb = get_observation()["agentview"]["images"]["rgb"]
            >>> reply = '{"box": [120, 80, 210, 190]}'
            >>> det = parse_vlm_detections(reply, image=rgb)
            >>> if det["boxes"]:
            ...     x1, y1, x2, y2 = det["boxes"][0]
            ...     cx, cy = det["points"][0]
        """
        empty = {"boxes": [], "points": [], "labels": [], "n": 0}

        def _extract_json_text_local(text: str) -> str:
            text = (text or "").strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
            if not starts:
                return text
            start = min(starts)
            open_ch = text[start]
            close_ch = "}" if open_ch == "{" else "]"
            end = text.rfind(close_ch)
            return text[start : end + 1] if end >= start else text[start:]

        def _try_load_json_local(text: str) -> Any:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None

        payload = _try_load_json_local(_extract_json_text_local(reply or ""))
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return empty

        # Resolve image extent (if any) for rescaling/clipping.
        W = H = None
        if image is not None:
            if isinstance(image, np.ndarray):
                H, W = int(image.shape[0]), int(image.shape[1])
            else:
                H, W = int(image[0]), int(image[1])

        def _nums(v: Any) -> list[float] | None:
            if isinstance(v, (list, tuple)):
                try:
                    return [float(x) for x in v]
                except (TypeError, ValueError):
                    return None
            return None

        box_keys = ("box", "bbox_2d", "bbox", "box_2d", "boundingbox")
        pt_keys = ("point_2d", "point", "coordinate", "coord", "xy")

        raw_boxes: list[list[float]] = []
        box_labels: list[Any] = []
        raw_points: list[list[float]] = []
        pt_labels: list[Any] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            for k in box_keys:
                if k in item:
                    bb = _nums(item[k])
                    if bb and len(bb) >= 4:
                        raw_boxes.append(bb[:4])
                        box_labels.append(label)
                    break
            for k in pt_keys:
                if k in item:
                    pt = _nums(item[k])
                    if pt and len(pt) >= 2:
                        raw_points.append(pt[:2])
                        pt_labels.append(label)
                    break

        if not raw_boxes and not raw_points:
            return empty

        # Decide coordinate space once. RoboMEx grounding is pixel-first.
        all_vals = [abs(v) for b in raw_boxes for v in b] + [abs(v) for p in raw_points for v in p]
        space = coord_space
        if space == "auto":
            mx = max(all_vals) if all_vals else 0.0
            space = "fraction" if mx <= 1.0 else "pixel"

        def _scale(x: float, y: float) -> tuple[float, float]:
            if W is None or H is None:
                return x, y  # cannot rescale without image size
            if space == "fraction":
                x, y = x * W, y * H
            elif space == "norm1000":
                x, y = x / 1000.0 * W, y / 1000.0 * H
            x = float(np.clip(x, 0, W - 1))
            y = float(np.clip(y, 0, H - 1))
            return x, y

        boxes: list[list[float]] = []
        centers: list[list[float]] = []
        for bb in raw_boxes:
            x1, y1 = _scale(bb[0], bb[1])
            x2, y2 = _scale(bb[2], bb[3])
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            boxes.append([x1, y1, x2, y2])
            centers.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

        points: list[list[float]] = list(centers)
        for pt in raw_points:
            px, py = _scale(pt[0], pt[1])
            points.append([px, py])

        labels = box_labels + pt_labels
        return {"boxes": boxes, "points": points, "labels": labels, "n": len(labels)}

    def get_oriented_bounding_box_from_3d_points(self, points: np.ndarray) -> dict[str, Any]:
        """Get the oriented bounding box from 3D points.

        Args:
            points: np.ndarray: The 3D points to get the oriented bounding box from.
                Shape: (N, 3), dtype float64.

        Returns:
            dict[str, Any]: The oriented bounding box. The dictionary contains the following keys:
                - "center": np.ndarray: The center of the oriented bounding box in point cloud frame.
                - "extent": np.ndarray: The extent of the oriented bounding box.
                - "R": np.ndarray: The rotation matrix of the oriented bounding box in point cloud frame.

        Example:
            >>> points = np.random.randn((100, 3))
            >>> obb = get_oriented_bounding_box_from_3d_points(points)
        """
        return _get_obb(points)

    # --------------------------------------------------------------------- #
    # Grasp planner (Contact-GraspNet)
    # --------------------------------------------------------------------- #
    def plan_grasp(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmentation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plan grasp candidates using Contact-GraspNet for a single instance.

        This is a thin wrapper around the Contact-GraspNet planner. It does not
        apply any camera/world transforms: the returned candidate poses are in
        the camera frame and callers must transform them before robot motion.
        The reduced API applies a small local grasp offset to the candidates
        before returning them.

        Args:
            depth:
                Depth image in meters.
                Shape: (H, W) or (H, W, 1), dtype float32/float64.
            intrinsics:
                Camera intrinsic matrix.
                Shape: (3, 3), dtype float64.
            segmentation:
                Instance segmentation map where each integer > 0 corresponds to a
                unique object instance ID.
                Shape: (H, W) or (H, W, 1), dtype int32/int64.

        Returns:
            grasp_poses:
                np.ndarray of shape (K, 4, 4), dtype float64.
                Homogeneous transforms for each candidate grasp IN THE CAMERA FRAME.
            grasp_scores:
                np.ndarray of shape (K,), dtype float64.
                Confidence score for each candidate grasp.

        Example:
            >>> cam = obs["agentview"]
            >>> rgb = cam["images"]["rgb"]
            >>> depth = cam["images"]["depth"][:, :, 0]
            >>> sam3_results = sam3_seg_fn(rgb, text_prompt="red mug")
            >>> best = max(sam3_results, key=lambda d: d["score"])
            >>> mask = best["mask"]
            >>> K = cam["intrinsics"]
            >>> grasp_sample_tf, grasp_scores = plan_grasp(
            ...     depth=depth,
            ...     intrinsics=K,
            ...     segmentation=mask,
            ... )
            >>> best_idx = grasp_scores.argmax()
            >>> best_T = grasp_sample_tf[best_idx]  # (4, 4), camera frame
            >>> camera_extrinsics = cam["pose_mat"]
            >>> grasp_sample_world_frame = camera_extrinsics @ best_T
        """
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[:, :, 0]

        grasp_sample, grasp_scores, _ = self.grasp_net_plan_fn(
            depth,
            intrinsics,
            segmentation,
            1,
        )

        assert len(grasp_sample) > 0, "No grasp candidates found"

        grasp_sample_tf = (
            vtf.SE3.from_matrix(grasp_sample) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))
        ).as_matrix()
        depth_vis = depth_to_rgb(depth)
        mask_overlay = overlay_segmentation_masks(depth_vis, [segmentation > 0], opacity=0.35)
        grasp_overlay = self._draw_grasp_points(
            mask_overlay,
            grasp_sample_tf,
            grasp_scores,
            intrinsics,
        )
        self._save_debug_overlay("grasp_plan", self._draw_label(grasp_overlay, "GraspNet candidates"))
        return grasp_sample_tf, grasp_scores

    def plan_grasp_from_point_clouds(
        self,
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plan grasp candidates using Contact-GraspNet given a full point cloud and a segmented point cloud of the object wanting to be grasped. These point clouds can be composed from multiple viewpoints.

        Args:
            pc_full:
                Point cloud of the full environment, including the object to be grasped.
                Shape: (N, 3), dtype float32.
            pc_segment:
                Point cloud of the segmented object to be grasped.
                Shape: (N, 3), dtype float32.
        Returns:
            grasp_sample_tf: (4, 4) homogeneous transform for the grasp pose in THE POINT CLOUD FRAME.
            grasp_scores: (N,) array of grasp scores.
        """
        grasp_sample, grasp_scores, _ = self.grasp_net_plan_point_clouds_fn(pc_full, pc_segment, segmap_id=1)
        
        assert len(grasp_sample) > 0, "No grasp candidates found"

        grasp_sample_tf = (
            vtf.SE3.from_matrix(grasp_sample) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))
        ).as_matrix()
        return grasp_sample_tf, grasp_scores

    # --------------------------------------------------------------------- #
    # IK / motion primitives
    # --------------------------------------------------------------------- #
    def solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> np.ndarray:
        """Solve inverse kinematics for the panda_hand link.

        Args:
            position:
                Target position in world frame.
                Shape: (3,), dtype float64.
            quaternion_wxyz:
                Target orientation as a unit quaternion in world frame.
                Shape: (4,), [w, x, y, z], dtype float64.

        Returns:
            joints:
                np.ndarray of shape (7,), dtype float64.
                Joint angles for the 7 DoF Franka arm.

        Example:
            >>> target_pos = np.array([0.5, 0.0, 0.3])
            >>> target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity, wxyz
            >>> joints = solve_ik(target_pos, target_quat)
            >>> move_to_joints(joints)
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        # Clamp to reachable workspace to avoid IK timeouts
        pos = np.clip(pos, [-0.1, -0.5, 0.005], [0.75, 0.5, 0.9])
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)

        # Try requested orientation first, then fallbacks
        orientations = [
            ("requested", quat_wxyz),
            ("top-down", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)),
            ("45-tilt", np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float64)),
            ("side-approach", np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float64)),
        ]

        for label, quat in orientations:
            try:
                off_pos = apply_tcp_offset(pos, quat, self._TCP_OFFSET) if label != "requested" else offset_pos
                self.cfg = solve_ik_with_convergence(
                    self.ik_solve_fn, quat, off_pos, self.cfg
                )
                if label != "requested":
                    logger.info("IK solved with fallback orientation: %s", label)
                return extract_arm_joints(self.cfg)
            except Exception:
                logger.warning("IK failed with orientation '%s', trying next fallback", label)
                continue

        raise RuntimeError(
            f"IK failed for position {pos} with all orientation fallbacks"
        )

    # Single arm control APIs

    def move_to_joints(self, joints: np.ndarray) -> None:
        """Move the robot to a given joint configuration in a blocking manner.
        Interpolation is slightly rudimentary so it is recommended to keep pre-grasp cartesian offsets smaller e.g. 0.075m.

        Args:
            joints:
                Target joint angles for the 7-DoF Franka arm.
                Shape: (7,), dtype float64.

        Returns:
            None

        Example:
            >>> joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]) # this is an example home joint configuration for the Franka arm
            >>> move_to_joints(joints)
        """
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

        # self._env.move_to_joints_non_blocking(joints)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        _open_gripper(self._env, steps=30)

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        _close_gripper(self._env, steps=30)

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> None:
        """Go to pose using Inverse Kinematics with optional approach motion.

        Combines solve_ik + move_to_joints into a single call. If z_approach > 0,
        first moves to position offset by z_approach in world +Z, then moves to
        the final position.

        Args:
            position: (3,) XYZ target in meters (world frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion (world frame).
            z_approach: World +Z offset for approach motion (meters). Default 0.0.

        Example:
            >>> goto_pose(np.array([0.5, 0.0, 0.3]), np.array([0.0, 1.0, 0.0, 0.0]), z_approach=0.075)
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)

        if z_approach > 0.0:
            approach_pos = pos.copy()
            approach_pos[2] += z_approach
            joints = self.solve_ik(approach_pos, quat)
            self.move_to_joints(joints)

        joints = self.solve_ik(pos, quat)
        self.move_to_joints(joints)
    
    def goto_home_joint_position(
        self,
        tolerance: float = 0.008,
        max_steps: int = 360,
        retries: int = 1,
    ) -> None:
        """Return the arm to its reset joint configuration with high manipulability.

        This preserves the current gripper command. It must not be paired with
        ``open_gripper()`` merely to get a clearer observation, because the robot may
        already be holding an object.
        """
        home = getattr(self._env, "home_joint_position", None)
        if home is None:
            raise RuntimeError("Home joint position is unavailable in the current environment.")
        joints = np.asarray(home, dtype=np.float64).reshape(7)
        attempts = max(1, int(retries) + 1)
        last_status = None
        for _ in range(attempts):
            last_status = self._env.move_to_joints_blocking(
                joints,
                tolerance=float(tolerance),
                max_steps=int(max_steps),
                settle_steps=8,
                strict=False,
            )
            if isinstance(last_status, dict) and last_status.get("converged"):
                return
            if not isinstance(last_status, dict):
                return
        final_error = (
            float(last_status.get("final_error", float("inf")))
            if isinstance(last_status, dict)
            else float("inf")
        )
        raise RuntimeError(
            "goto_home_joint_position did not reach home: "
            f"final_error={final_error:.6f}, tolerance={float(tolerance):.6f}, "
            f"attempts={attempts}, max_steps={int(max_steps)}"
        )
    
    def subsample_point_cloud(self, pc: np.ndarray, max_points: int = 10000) -> np.ndarray:
        """Randomly subsample a point cloud to a maximum number of points.
        
        Args:
            pc: (N, 3) array of points.
            max_points: The maximum number of points to subsample to. Default is 10000.
        Returns:
            subsampled_pc: (M, 3) array of subsampled points where M <= max_points.
        """
        if len(pc) > max_points:
            return pc[np.random.choice(len(pc), max_points, replace=False)]
        return pc
    
    def filter_noise(self, points: np.ndarray, colors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Filter noise from the point cloud.
        Args:
            points: (N, 3) array of points.
            colors: (N, 3) array of colors. Optional, default is None.
        Returns:
            (M, 3) array of points.
            (M, 3) array of colors. Optional, default is None.
        """
        eps = 0.005  # Maximum distance between samples to be neighbors
        min_samples = 10
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)
        filtered_pointcloud = points[labels != -1]
        if colors is not None:
            filtered_colors = colors[labels != -1]
        else:
            filtered_colors = None
        return filtered_pointcloud, filtered_colors

    def parse_grasp_poses_for_curobo(
        self, grasp_poses_world: np.ndarray, grasp_scores: np.ndarray, top_k: int = 15
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parses grasp poses from the world frame into a format compatible with CuRobo.

        Args:
            grasp_poses_world: (N, 4, 4) array of grasp poses in world frame
            grasp_scores: (N,) array of grasp scores
            top_k: number of top grasp poses to return. Default is 15.
        Returns:
            positions: (N, 3) array of grasp positions
            quaternions: (N, 4) array of grasp quaternions
            scores: (N,) array of grasp scores sorted by score
        
        Example:
            >>> grasp_poses_cam, grasp_scores = plan_grasp_from_point_clouds(...)
            >>> positions, quaternions, scores = parse_grasp_poses_for_curobo(grasp_poses_cam, grasp_scores, top_k=15)
        """
        grasp_poses_world = np.asarray(grasp_poses_world)
        positions = grasp_poses_world[..., 3][:, :3]
        rotations = grasp_poses_world[..., :3][:, :3, :3]
        quaternions = vtf.SO3.from_matrix(rotations).wxyz
        scores = grasp_scores
        k = min(top_k, len(positions))
        order = np.argsort(-scores)[:k]
        return positions[order], quaternions[order], scores[order]
    
    ### CuRobo-related functions ###
    def create_curobo_world_from_depth(
        self,
        depth_image: np.ndarray,
        object_mask: np.ndarray,
        intrinsics: np.ndarray,
        camera_pose: np.ndarray | None = None,
        **kwargs: Any,
    ):
        """Create a CuRobo WorldConfig from a depth image and object mask. Stores the result for use by plan_grasp_trajectory."""
        world = _get_curobo_api().create_curobo_world_from_depth(
            depth_image, object_mask, intrinsics, camera_pose=camera_pose, **kwargs
        )
        self._curobo_world_config = world
        return world

    def create_curobo_world_from_pointcloud(
        self, point_cloud: np.ndarray, object_mask: np.ndarray, **kwargs: Any
    ):
        """Create a CuRobo WorldConfig from a point cloud and per-point object mask. Stores the result for use by plan_grasp_trajectory."""
        world = _get_curobo_api().create_curobo_world_from_pointcloud(point_cloud, object_mask, **kwargs)
        self._curobo_world_config = world
        return world

    def create_curobo_world_from_observation(
        self,
        object_mask: np.ndarray,
        *,
        camera_name: str | None = None,
        object_name: str = "object",
        scene_name: str = "scene",
        **kwargs: Any,
    ):
        """Create a CuRobo WorldConfig from the current observation: uses this camera's depth, intrinsics, and pose to build the world, split by object_mask. Stores the result for use by plan_grasp_trajectory.
        
        """
        obs = self._env.get_observation()
        cam = camera_name or self.camera_name
        depth = obs[cam]["images"]["depth"]
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        intrinsics = obs[cam]["intrinsics"]
        pose = obs[cam]["pose"]
        camera_pose_4x4 = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=pose[3:]),
            translation=pose[:3],
        ).as_matrix()
        world = _get_curobo_api().create_curobo_world_from_depth(
            depth,
            object_mask,
            intrinsics,
            camera_pose=camera_pose_4x4,
            object_name=object_name,
            scene_name=scene_name,
            **kwargs,
        )
        self._curobo_world_config = world
        return world

    def update_curobo_world(
        self,
        *,
        camera_name: str | None = None,
        robot_distance_threshold: float = 0.15,
        robot_file: str = "franka.yml",
        **kwargs: Any,
    ) -> Any:
        """Build CuRobo world from current observation (full depth, robot excluded) and store it.

        Uses create_curobo_world_from_depth_full: single scene mesh with points near the
        robot removed (robot_distance_threshold) so start/IK configs are not in collision.
        Same logic as curobo_test script. Use before plan_grasp_trajectory when using
        collision checking.

        Args:
            camera_name: Camera to use; default is self.camera_name.
            robot_distance_threshold: Distance (m) to exclude points near the robot. Default is 0.15.
            robot_file: CuRobo robot config for segmenter. Default is 'franka.yml'.
            **kwargs: Passed to create_curobo_world_from_depth_full.

        Returns:
            The WorldConfig that was stored in self._curobo_world_config.
        """
        obs = self._env.get_observation()
        cam = camera_name or self.camera_name
        depth = obs[cam]["images"]["depth"]
        if depth.ndim == 3:
            depth = np.asarray(depth[:, :, 0], dtype=np.float64)
        else:
            depth = np.asarray(depth, dtype=np.float64)
        intrinsics = np.asarray(obs[cam]["intrinsics"], dtype=np.float64)
        pose = obs[cam]["pose"]
        camera_pose_4x4 = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=pose[3:]),
            translation=np.asarray(pose[:3], dtype=np.float64),
        ).as_matrix()
        robot_joint_pos = np.asarray(obs["robot_joint_pos"], dtype=np.float64)
        world = _get_curobo_api().create_curobo_world_from_depth_full(
            depth,
            intrinsics,
            camera_pose=camera_pose_4x4,
            robot_joint_position=robot_joint_pos,
            robot_file=robot_file,
            robot_distance_threshold=robot_distance_threshold,
            **kwargs,
        )
        self._curobo_world_config = world
        return world

    def plan_grasp_trajectory(
        self,
        object_name: str,
        *,
        object_mask: np.ndarray, #| None = None,
        grasp_poses: list[tuple[np.ndarray, np.ndarray]], #| None = None,
        top_k_grasps: int = 15,
        use_world_collision: bool = True,
        robot_distance_threshold: float = 0.15,
        robot_collision_sphere_buffer: float | None = -0.01,
        collision_activation_distance: float | None = 0.001,
        world_config: Any = None,
        # use_multiview: bool = False,
        **kwargs: Any,
    ) -> tuple[bool, np.ndarray | None, int | None]:
        """Plan a collision-free trajectory to one of the top-k grasp poses.

        Builds a CuRobo world with object mesh + scene mesh (robot excluded via segmenter).
        Collision between the object and the rest of the world is disabled so the robot
        can approach the object. If grasp_poses is None, samples grasps for object_name.

        Args:
            object_name: Name of the object (used for mask and grasps if not provided).
            object_mask: precomputed mask (H, W) of the object.
            grasp_poses: list of (position (3,), quaternion_wxyz (4,)) in world frame.
            top_k_grasps: Number of top grasps to try (by score). Default 15.
            use_world_collision: If True, use world for collision (scene only; object ignored for grasp). Default is True.
            robot_distance_threshold: Passed when building world. Default is 0.15.
            robot_collision_sphere_buffer: Passed to plan_to_grasp_poses. Default is -0.01.
            collision_activation_distance: Passed to plan_to_grasp_poses. Default is 0.001.
            world_config: Optional WorldConfig. If None, builds via update_curobo_world_with_object. Default is None.
            **kwargs: Passed through to plan_to_grasp_poses.

        Returns:
            (success, joint_trajectory, goalset_index): joint_trajectory is (T, 7) or None.
        """
        if grasp_poses is None:
            # positions, quaternions, scores = self._sample_grasp_poses_for_object(object_name, use_multiview=use_multiview)
            # if len(positions) == 0:
            #     return False, None, None
            # k = min(top_k_grasps, len(positions))
            # order = np.argsort(-scores)[:k] if scores is not None and len(scores) else np.arange(k)
            # grasps_to_try = [(positions[i], quaternions[i]) for i in order]
            assert grasp_poses is not None, "grasp_poses must be provided"
        else:
            grasps_to_try = [(np.asarray(p), np.asarray(q)) for p, q in grasp_poses]

        world = world_config
        if world is None:
            self.update_curobo_world_with_object(
                object_name,
                object_mask=object_mask,
                robot_distance_threshold=robot_distance_threshold,
            )
            world = self._curobo_world_config
            # Disable collision with the object so the robot can reach into it
            ignore_obstacle_names = [getattr(self, "_curobo_world_object_name", object_name.replace(" ", "_"))]
        else:
            ignore_obstacle_names = []

        obs = self._env.get_observation()
        start_joint_position = obs["robot_joint_pos"]

        success, joint_traj, goalset_idx = _get_curobo_api().plan_to_grasp_poses(
            world,
            start_joint_position,
            grasps_to_try,
            use_world_collision=use_world_collision,
            robot_collision_sphere_buffer=robot_collision_sphere_buffer,
            collision_activation_distance=collision_activation_distance,
            ignore_obstacle_names=ignore_obstacle_names if use_world_collision else None,
            **kwargs,
        )
        return success, joint_traj, goalset_idx

    def execute_joint_trajectory(
        self,
        joint_trajectory: np.ndarray,
        *,
        subsample: int = 1,
        tolerance: float = 0.01,
        max_steps: int = 120,
    ) -> None:
        """Execute a joint-space trajectory (T, 7) by moving to each waypoint with move_to_joints_blocking.

        Args:
            joint_trajectory: (T, 7) joint positions in radians.
            subsample: Use every Nth waypoint (1 = all). Larger values speed up execution. Default is 1.
            tolerance: Passed to move_to_joints_blocking. Default is 0.01.
            max_steps: Passed to move_to_joints_blocking. Default is 120.
        """
        traj = np.asarray(joint_trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] < 7:
            raise ValueError(f"joint_trajectory must be (T, 7), got shape {traj.shape}")
        indices = list(range(0, len(traj), subsample))
        if indices and indices[-1] != len(traj) - 1:
            indices.append(len(traj) - 1)
        for i in indices:
            joints = traj[i, :7]
            self._env.move_to_joints_blocking(
                joints, tolerance=tolerance, max_steps=max_steps
            )
    
    def update_curobo_world_with_object(
        self,
        object_name: str,
        *,
        object_mask: np.ndarray | None = None,
        camera_name: str | None = None,
        robot_distance_threshold: float = 0.15,
        robot_file: str = "franka.yml",
        object_name_in_world: str | None = None,
        scene_name: str = "scene",
        **kwargs: Any,
    ) -> Any:
        """Build CuRobo world from current observation with object/scene split (robot excluded) and store it.

        Uses create_curobo_world_from_depth_with_object: creates two separate meshes:
        - Object mesh (from object_mask)
        - Scene mesh (everything else)

        Robot points are excluded so the start configuration is not in collision. Use this
        before plan_with_grasped_object to attach the object and plan motion.

        Args:
            object_name: Name of the object (used to get object_mask if object_mask is None).
            object_mask: Optional precomputed mask (H, W). If None, get_object_mask(object_name) is used.
            camera_name: Camera to use; default is self.camera_name.
            robot_distance_threshold: Distance (m) to exclude points near the robot. Default is 0.15.
            robot_file: CuRobo robot config for segmenter. Default is 'franka.yml'.
            object_name_in_world: Name for the object in WorldConfig (default: object_name, spaces → underscores).
            scene_name: Name for the scene mesh in WorldConfig. Default is 'scene'.
            **kwargs: Passed to create_curobo_world_from_depth_with_object.

        Returns:
            The WorldConfig that was stored in self._curobo_world_config.
        """
        obs = self._env.get_observation()
        cam = camera_name or self.camera_name
        depth = obs[cam]["images"]["depth"]
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        elif depth.ndim == 2:
            pass
        else:
            raise ValueError(f"Depth image has invalid shape: {depth.shape}")
        depth = np.asarray(depth, dtype=np.float64)
        intrinsics = np.asarray(obs[cam]["intrinsics"], dtype=np.float64)
        pose = obs[cam]["pose"]
        camera_pose_4x4 = vtf.SE3.from_rotation_and_translation(
            rotation=vtf.SO3(wxyz=pose[3:]),
            translation=np.asarray(pose[:3], dtype=np.float64),
        ).as_matrix()
        robot_joint_pos = np.asarray(obs["robot_joint_pos"], dtype=np.float64)

        if object_mask is None:
            object_mask = self.get_object_mask(object_name)
        if object_mask is None:
            raise ValueError(f"Could not get object mask for '{object_name}' to create world with object/scene split")
        mask_bool = np.asarray(object_mask, dtype=bool)
        if getattr(self._env, "output_dir", None):
            rgb = np.asarray(obs[cam]["images"]["rgb"])
            self._save_rgb_with_object_mask(rgb, mask_bool, object_name, suffix="world")

        # World-safe name (no spaces) for CuRobo lookup
        world_object_name = object_name_in_world if object_name_in_world is not None else object_name
        mesh_name = world_object_name.replace(" ", "_")
        # Place object mesh at current EE pose so the world matches the grasped location (fixes wrong position in planning/debug)
        ee_pos, ee_quat_wxyz = self.get_ee_pose()
        world = _get_curobo_api().create_curobo_world_from_depth_with_object(
            depth,
            object_mask,
            intrinsics,
            camera_pose=camera_pose_4x4,
            robot_joint_position=robot_joint_pos,
            robot_file=robot_file,
            robot_distance_threshold=robot_distance_threshold,
            object_name=mesh_name,
            scene_name=scene_name,
            object_pose_override=(ee_pos, ee_quat_wxyz),
            **kwargs,
        )
        self._curobo_world_config = world
        self._curobo_world_object_name = mesh_name
        return world

    def plan_with_grasped_object(
        self,
        target_pose: tuple[np.ndarray, np.ndarray],
        object_name: str,
        *,
        object_pose: tuple[np.ndarray, np.ndarray] | None = None,
        object_mask: np.ndarray | None = None,
        world_config: Any = None,
        robot_collision_sphere_buffer: float | None = -0.01,
        collision_activation_distance: float | None = 0.01,
        **kwargs: Any,
    ) -> tuple[bool, np.ndarray | None]:
        """Plan a collision-free trajectory to move a grasped object to a target pose.

        NOTE: the grasped object must not be in collision with the scene before calling this function. First lift the object then plan the trajectory to the target pose.

        If object_mask is provided, the CuRobo world is rebuilt from the current observation
        so the object mesh is at its current (e.g. post-lift) position; collision between
        object and scene is enabled. The object is then attached to the robot and motion
        is planned to the target pose.

        Args:
            target_pose: (position (3,), quaternion_wxyz (4,)) target pose in world frame (e.g. basket).
            object_name: Name of the object to attach (used for world mesh name: spaces → underscores).
            object_pose: Optional (position, quat_wxyz) of object; unused for now, for API consistency. Default is None.
            object_mask: If provided, rebuild world from current observation with this mask before planning. Default is None.
            world_config: Optional WorldConfig. If None, uses stored or rebuilds when object_mask given. Default is None.
            robot_collision_sphere_buffer: Override robot collision_sphere_buffer (m). Default is -0.01m.
            collision_activation_distance: Distance (m) to activate collision cost. Default is 0.01m.
            **kwargs: Passed through to plan_with_grasped_object.

        Returns:
            (success, joint_trajectory): joint_trajectory is (T, 7) or None.
        """
        if object_mask is not None:
            self.update_curobo_world_with_object(
                object_name,
                object_mask=object_mask,
                robot_distance_threshold=0.15,
            )
        world = world_config if world_config is not None else self._curobo_world_config
        if world is None:
            raise ValueError(
                "No world_config available. Call update_curobo_world_with_object (or pass object_mask) first."
            )

        obs = self._env.get_observation()
        start_joint_position = obs["robot_joint_pos"]
        object_name_for_attach = getattr(
            self, "_curobo_world_object_name", None
        ) or object_name.replace(" ", "_")

        debug_out_dir = getattr(self._env, "output_dir", None)
        success, joint_traj = _get_curobo_api().plan_with_grasped_object(
            world,
            start_joint_position,
            target_pose,
            object_name_for_attach,
            robot_collision_sphere_buffer=robot_collision_sphere_buffer,
            collision_activation_distance=collision_activation_distance,
            debug_out_dir=debug_out_dir,
            **kwargs,
        )
        return success, joint_traj
