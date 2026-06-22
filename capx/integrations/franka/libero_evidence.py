"""Evidence-instrumented baseline LIBERO API for the staged agent.

Subclasses ``FrankaLiberoApi`` (the exact API set of the baseline config
``franka_libero_spatial_0.yaml``). NOTHING about the baseline functions is
changed -- same signatures, same returns, same internal grasp-orientation
handling. The only additions are:

1. Side-channel evidence (V1.6): SAM3 calls save a bbox-only overlay (no mask
   fill, so object colors stay judgeable); language grounding and grasp
   sampling print compact numeric summaries to stdout, which flows back to
   the LLM through the existing multi-turn loop.
2. Two agent-callable prospective checks (verify-before-commit):
   ``check_transit_clearance`` and ``check_place_metrics``. Pure computation,
   printed as text reports (evidence-typing principle).
"""

from __future__ import annotations

import pathlib
from typing import Any

import cv2
import numpy as np
from PIL import Image

from capx.envs.base import BaseEnv
from capx.integrations.franka.libero import FrankaLiberoApi
from capx.utils.pointcloud_render import draw_grasp_point_on_image, pose_from_position_wxyz
from capx.verifier.boxes import ObjectBox, corridor_clearance


class FrankaLiberoApiEvidence(FrankaLiberoApi):
    """Baseline API + side-channel evidence + prospective checks."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env, use_sam3=True)
        self._debug_output_dir: pathlib.Path | None = None
        self._debug_block_idx = 0
        self._debug_counter = 0

    # ----- optional debug hooks used by the trial runner ----- #
    def set_debug_context(self, output_dir: str, block_idx: int) -> None:
        self._debug_output_dir = pathlib.Path(output_dir)
        self._debug_block_idx = block_idx
        self._debug_counter = 0

    def _save_debug_overlay(self, name: str, image: np.ndarray) -> None:
        if self._debug_output_dir is None:
            return
        self._debug_output_dir.mkdir(parents=True, exist_ok=True)
        path = self._debug_output_dir / (
            f"block_{self._debug_block_idx:02d}_{self._debug_counter:03d}_{name}.png"
        )
        self._debug_counter += 1
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)

    def functions(self) -> dict[str, Any]:
        fns = super().functions()
        # expose SAM3 candidate list so the agent can do relation-aware target
        # selection + verification (the baseline high-level grounding hides
        # this behind an internal top-1 choice, which is the wrong-object source)
        fns["segment_sam3_text_prompt"] = self.segment_sam3_text_prompt
        fns["mask_to_world_points"] = self.mask_to_world_points
        # multi-candidate grasp sampling (top-k) for obstruction-aware selection
        fns["sample_grasp_candidates"] = self.sample_grasp_candidates
        fns["check_transit_clearance"] = self.check_transit_clearance
        fns["check_place_metrics"] = self.check_place_metrics
        return fns

    @staticmethod
    def mask_to_world_points(mask, depth, intrinsics, pose_mat):
        """Lift a binary mask to (N,3) world-frame points via depth + camera pose."""
        m = np.asarray(mask).astype(bool)
        d = np.asarray(depth)
        if d.ndim == 3:
            d = d[:, :, 0]
        ys, xs = np.where(m)
        z = d[ys, xs]
        ok = (z > 0.01) & (z < 3.0)
        ys, xs, z = ys[ok], xs[ok], z[ok]
        K = np.asarray(intrinsics)
        x = (xs - K[0, 2]) * z / K[0, 0]
        y = (ys - K[1, 2]) * z / K[1, 1]
        pts = np.stack([x, y, z], axis=-1)
        return pts @ np.asarray(pose_mat)[:3, :3].T + np.asarray(pose_mat)[:3, 3]

    # ------------------------------------------------------------------ #
    # side-channel evidence on baseline calls (no behavior change)
    # ------------------------------------------------------------------ #
    def segment_sam3_text_prompt(
        self, rgb: np.ndarray, text_prompt: str
    ) -> list[dict[str, Any]]:
        results = super().segment_sam3_text_prompt(rgb, text_prompt)
        try:
            if results:
                overlay = np.ascontiguousarray(np.asarray(rgb).copy())
                for r in sorted(results, key=lambda x: -x["score"])[:5]:
                    box = r.get("box")
                    if box is None:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 220, 0), 2)
                    cv2.putText(overlay, f"{r['score']:.2f}", (x1, max(y1 - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 2, cv2.LINE_AA)
                self._save_debug_overlay(f"sam3_{text_prompt.replace(' ', '_')}", overlay)
        except Exception as e:
            print(f"[evidence] sam3 overlay failed: {e}")
        return results

    def get_object_3d_points_and_masks_from_language(
        self, text_prompt: str, use_multiview: bool = True
    ) -> dict[str, Any]:
        result = super().get_object_3d_points_and_masks_from_language(
            text_prompt, use_multiview=use_multiview
        )
        try:
            pts = result.get("points_3d")
            if pts is not None and len(pts) > 0:
                center = np.median(pts, axis=0)
                top_z = float(np.percentile(pts[:, 2], 98))
                print(
                    f"[evidence] '{text_prompt}': score={result.get('agentview_score', 0):.2f} "
                    f"n_points={len(pts)} center=[{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}] "
                    f"top_z={top_z:.3f}"
                )
        except Exception as e:
            print(f"[evidence] grounding summary failed: {e}")
        return result

    def sample_grasp_pose(
        self, object_name: str, use_multiview: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        position, quaternion = super().sample_grasp_pose(object_name, use_multiview=use_multiview)
        try:
            obs = self.get_observation()
            cam = obs[self.camera_name]
            pose = pose_from_position_wxyz(position, quaternion)
            overlay = draw_grasp_point_on_image(
                np.asarray(cam["images"]["rgb"]), pose, cam["intrinsics"], cam["pose_mat"],
                color=(0, 220, 0), label=f"grasp: {object_name}",
            )
            self._save_debug_overlay(f"grasp_{object_name.replace(' ', '_')}", overlay)
        except Exception as e:
            print(f"[evidence] grasp overlay failed: {e}")
        return position, quaternion

    # ------------------------------------------------------------------ #
    # prospective checks (text evidence reports)
    # ------------------------------------------------------------------ #
    def check_transit_clearance(
        self,
        held_points: np.ndarray,
        overhang: float,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        transit_tcp_z: float,
        obstacles: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Prospective check for a CARRY move: will the held object clear every obstacle along the straight-line path? Call this BEFORE moving a grasped object; if it fails, use the returned min_safe_tcp_z.

        Args:
            held_points: (N, 3) world points of the held object (points_3d from segmentation).
            overhang: TCP-to-object-bottom distance in meters (after lifting:
                tcp_z - min z of the object's points).
            start_xy: (2,) current object xy.
            goal_xy: (2,) destination xy.
            transit_tcp_z: planned TCP height in meters during the move.
            obstacles: dict name -> (M, 3) world points of every other nearby object.

        Returns:
            dict: ok (bool), min_safe_tcp_z (float, meters), rows (per-obstacle details).
        """
        held_box = ObjectBox.from_points("held_object", np.asarray(held_points))
        obstacle_boxes = [
            ObjectBox.from_points(str(name), np.asarray(pts))
            for name, pts in obstacles.items()
            if len(np.asarray(pts)) >= 20
        ]
        rows, min_safe = corridor_clearance(
            held_box=held_box,
            overhang=float(overhang),
            start_xy=np.asarray(start_xy, dtype=np.float64).reshape(2),
            goal_xy=np.asarray(goal_xy, dtype=np.float64).reshape(2),
            transit_tcp_z=float(transit_tcp_z),
            obstacles=obstacle_boxes,
        )
        ok = float(transit_tcp_z) >= min_safe
        print("[evidence] transit corridor clearance "
              f"(planned tcp_z={transit_tcp_z * 100:.1f}cm, overhang={overhang * 100:.1f}cm):")
        for row in rows:
            print(
                f"  - {row['obstacle']}: top_z={row['obstacle_top_z_cm']}cm "
                f"intrudes={row['intrudes_corridor']} "
                f"z_clearance={row['z_clearance_cm']}cm -> "
                f"{'PASS' if row['pass'] else 'FAIL'}"
            )
        print(f"[evidence] min_safe_tcp_z={min_safe * 100:.1f}cm -> "
              f"{'OK to move' if ok else 'TOO LOW, raise transit height'}")
        return {"ok": ok, "min_safe_tcp_z": float(min_safe), "rows": rows}

    def check_place_metrics(
        self,
        surface_points: np.ndarray,
        overhang: float,
        place_xy: np.ndarray,
        clearance: float = 0.02,
    ) -> dict[str, Any]:
        """Prospective check for a PLACE: computes the release TCP height from geometry instead of guessing. Call BEFORE descending to release.

        Args:
            surface_points: (N, 3) world points of the target receptacle.
            overhang: TCP-to-held-object-bottom distance in meters.
            place_xy: (2,) intended release xy.
            clearance: drop clearance in meters above the surface (default 2cm).

        Returns:
            dict: ok (bool), release_tcp_z (float, meters), surface_z,
            region_radius, dist_to_center (floats, meters).
        """
        pts = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
        surface_z = float(np.percentile(pts[:, 2], 80))
        center = np.median(pts, axis=0)
        region_radius = float(np.percentile(
            np.linalg.norm(pts[:, :2] - center[:2], axis=1), 90))
        place_xy = np.asarray(place_xy, dtype=np.float64).reshape(2)
        dist = float(np.linalg.norm(place_xy - center[:2]))
        release_tcp_z = surface_z + float(overhang) + float(clearance)
        in_region = dist < 0.6 * region_radius
        sane = 0.005 <= clearance <= 0.05 and 0.01 <= overhang <= 0.30
        ok = in_region and sane
        print("[evidence] place metrics: "
              f"surface_z={surface_z * 100:.1f}cm region_r={region_radius * 100:.1f}cm "
              f"dist_to_center={dist * 100:.1f}cm overhang={overhang * 100:.1f}cm")
        print(f"[evidence] computed release_tcp_z={release_tcp_z * 100:.1f}cm "
              f"(= surface_z + overhang + clearance) -> "
              f"{'OK to place' if ok else 'NOT OK: ' + ('outside region' if not in_region else 'parameter out of range')}")
        return {
            "ok": ok,
            "release_tcp_z": release_tcp_z,
            "surface_z": surface_z,
            "region_radius": region_radius,
            "dist_to_center": dist,
        }
