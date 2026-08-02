"""M0/M1.2 smoke test: exercise state/camera/cloud/render/protocol, no env.

    python -m vaw.scripts.smoke_render

Builds a synthetic table scene *in world space*, rasterises it into RGB-D from
two camera mounts, and then runs the real rendering path on it. That detour
matters: rendering the canvas from a virtual viewpoint only means anything if
the depth it re-projects came from actual geometry, so a flat fake depth map
would test nothing. Writes one canvas per viewpoint to vaw/out/smoke/.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from PIL import Image

from vaw.camera import VirtualCamera, orbit_camera
from vaw.cloud import build_scene_cloud
from vaw.geometry import project_world_to_pixel
from vaw.protocol import parse_action, tool_definitions
from vaw.render import CANVAS_H, CANVAS_W, render_canvas
from vaw.state import ActionState
from vaw.types import TOP_DOWN_QUAT_WXYZ, Candidate, ObjectEntry, Pose, PreviewResult, Receipt

OUT = pathlib.Path(__file__).resolve().parent.parent / "out" / "smoke"

TABLE_Z = 0.0
CAN_CENTER = np.array([0.77, 0.03, 0.05])
CAN_SIZE = np.array([0.07, 0.07, 0.10])
INSTRUCTION = "pick the alphabet soup and place it in the basket"


def _plane(x_range, y_range, z, step, color):
    xs = np.arange(*x_range, step)
    ys = np.arange(*y_range, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)
    return pts, np.tile(np.asarray(color, dtype=np.uint8), (len(pts), 1))


def _box_surface(center, size, step, color):
    """Surface points of an axis-aligned box (all six faces)."""
    c, s = np.asarray(center, dtype=np.float64), np.asarray(size, dtype=np.float64)
    lo, hi = c - s / 2, c + s / 2
    faces = []
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        ga = np.arange(lo[a], hi[a] + step, step)
        gb = np.arange(lo[b], hi[b] + step, step)
        ma, mb = np.meshgrid(ga, gb)
        for value in (lo[axis], hi[axis]):
            pts = np.zeros((ma.size, 3))
            pts[:, a] = ma.ravel()
            pts[:, b] = mb.ravel()
            pts[:, axis] = value
            faces.append(pts)
    pts = np.concatenate(faces, axis=0)
    return pts, np.tile(np.asarray(color, dtype=np.uint8), (len(pts), 1))


def synthetic_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points + colors + an integer object tag (1 = the soup can)."""
    parts = [
        _plane((0.30, 1.05), (-0.38, 0.38), TABLE_Z, 0.0025, (150, 132, 110)),
        _box_surface(CAN_CENTER, CAN_SIZE, 0.0025, (208, 76, 66)),          # tag 1
        _box_surface([0.55, -0.24, 0.075], [0.07, 0.07, 0.15], 0.003, (228, 228, 232)),
        _box_surface([0.62, -0.10, 0.055], [0.06, 0.06, 0.11], 0.003, (96, 132, 200)),
    ]
    # Basket: four thin walls, open top.
    for dx, dy, sx, sy in ((-0.09, 0, 0.014, 0.18), (0.09, 0, 0.014, 0.18),
                           (0, -0.09, 0.18, 0.014), (0, 0.09, 0.18, 0.014)):
        parts.append(
            _box_surface([0.62 + dx, 0.24 + dy, 0.055], [sx, sy, 0.11], 0.003, (206, 176, 86))
        )
    points = np.concatenate([p for p, _ in parts], axis=0)
    colors = np.concatenate([c for _, c in parts], axis=0)
    tags = np.zeros(len(points), dtype=np.int32)
    tags[len(parts[0][0]) : len(parts[0][0]) + len(parts[1][0])] = 1
    return points, colors, tags


def rasterize(
    points: np.ndarray, colors: np.ndarray, tags: np.ndarray, cam: VirtualCamera
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render world points to (rgb, depth, tag_map) by painter's-order z-buffer."""
    h, w = cam.height, cam.width
    rgb = np.zeros((h * w, 3), dtype=np.uint8)
    depth = np.zeros(h * w, dtype=np.float32)
    tag_map = np.zeros(h * w, dtype=np.int32)

    uvz = project_world_to_pixel(points, cam.intrinsics, cam.pose_mat)
    u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]
    ok = (z > 0.05) & np.isfinite(u) & np.isfinite(v)
    ui, vi = np.rint(u[ok]).astype(np.int64), np.rint(v[ok]).astype(np.int64)
    ok2 = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi = ui[ok2], vi[ok2]
    zz = z[ok][ok2]
    cc = colors[ok][ok2]
    tt = tags[ok][ok2]

    # 3x3 blocks: the sampled surface points land ~2-3 px apart, so single
    # pixels would leave a dotted depth map and the fused cloud would come out
    # an order of magnitude sparser than a real sensor's.
    idx, dep, col, tag = [], [], [], []
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            pu, pv = ui + du, vi + dv
            keep = (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
            idx.append(pv[keep] * w + pu[keep])
            dep.append(zz[keep])
            col.append(cc[keep])
            tag.append(tt[keep])
    flat = np.concatenate(idx)
    zz, cc, tt = np.concatenate(dep), np.concatenate(col, axis=0), np.concatenate(tag)
    order = np.argsort(-zz, kind="stable")  # nearest written last
    rgb[flat[order]] = cc[order]
    depth[flat[order]] = zz[order]
    tag_map[flat[order]] = tt[order]
    return rgb.reshape(h, w, 3), depth.reshape(h, w), tag_map.reshape(h, w)


def synthetic_obs() -> tuple[dict, np.ndarray]:
    """Two-camera observation of the synthetic scene + the can's true mask."""
    points, colors, tags = synthetic_scene()
    agent_cam = orbit_camera([0.65, 0.0, 0.06], 168.0, 38.0, 1.05, 640, 480)
    wrist_cam = orbit_camera(CAN_CENTER, 150.0, 62.0, 0.38, 256, 256)

    rgb, depth, tag_map = rasterize(points, colors, tags, agent_cam)
    wrist_rgb, wrist_depth, _ = rasterize(points, colors, tags, wrist_cam)

    obs = {
        "agentview": {
            "images": {"rgb": rgb, "depth": depth},
            "intrinsics": agent_cam.intrinsics,
            "pose_mat": agent_cam.pose_mat,
        },
        "robot0_eye_in_hand": {
            "images": {"rgb": wrist_rgb, "depth": wrist_depth},
            "intrinsics": wrist_cam.intrinsics,
            "pose_mat": wrist_cam.pose_mat,
        },
        "robot_cartesian_pos": np.array([0.60, -0.05, 0.30, 0.0, 1.0, 0.0, 0.0, 1.0]),
        "robot_joint_pos": np.zeros(7),
    }
    return obs, tag_map == 1


def build_state(obs: dict, can_mask: np.ndarray) -> ActionState:
    state = ActionState(instruction=INSTRUCTION)
    state.bump_revision()
    cart = obs["robot_cartesian_pos"]
    state.ee_pose = Pose(cart[:3], cart[3:7])
    state.gripper_opening = float(cart[7])

    cam = obs["agentview"]
    from vaw.geometry import mask_to_world_points

    points = mask_to_world_points(
        cam["images"]["depth"], can_mask, cam["intrinsics"], cam["pose_mat"]
    )
    state.add_object(
        ObjectEntry(
            object_id=state.next_id("obj"),
            name="alphabet soup",
            score=0.94,
            obs_revision=state.obs_revision,
            mask=can_mask,
            box=[0.0, 0.0, 1.0, 1.0],
            points_world=points,
            obb={
                "center": CAN_CENTER.tolist(),
                "extent": CAN_SIZE.tolist(),
                "R": np.eye(3).tolist(),
            },
        )
    )

    top = CAN_CENTER[2] + CAN_SIZE[2] / 2
    for dx, dy, score in ((0.0, 0.0, 0.88), (-0.02, 0.015, 0.74), (0.018, -0.02, 0.63)):
        state.add_candidate(
            Candidate(
                candidate_id=state.next_id("g"),
                kind="grasp",
                pose=Pose(
                    np.array([CAN_CENTER[0] + dx, CAN_CENTER[1] + dy, top + 0.005]),
                    TOP_DOWN_QUAT_WXYZ.copy(),
                ),
                score=score,
                source="plan_grasp",
                object_id="obj1",
                obs_revision=state.obs_revision,
            )
        )
    state.add_candidate(
        Candidate(
            candidate_id=state.next_id("p"),
            kind="place",
            pose=Pose(np.array([0.62, 0.24, 0.22]), TOP_DOWN_QUAT_WXYZ.copy()),
            source="propose_pose",
            obs_revision=state.obs_revision,
        )
    )

    state.select("g1")
    state.add_preview(
        PreviewResult(
            candidate_id="g1",
            ik_ok=True,
            predicted_ee=state.candidates["g1"].pose.copy(),
            notes="endpoint IK solved; trajectory not planned",
        )
    )
    state.add_receipt(
        Receipt(
            receipt_id=state.next_id("r"),
            op="commit_gripper",
            gripper_opening=1.0,
            discrepancy={},
        )
    )
    return state


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obs, can_mask = synthetic_obs()
    state = build_state(obs, can_mask)
    cloud = build_scene_cloud(obs, ("agentview", "robot0_eye_in_hand"), revision=state.obs_revision)
    print(f"[smoke] fused cloud: {len(cloud)} points, center={np.round(cloud.center, 3).tolist()}")

    def shot(name: str) -> np.ndarray:
        canvas = render_canvas(state, obs, cloud=cloud)
        assert canvas.shape == (CANVAS_H, CANVAS_W, 3), canvas.shape
        Image.fromarray(canvas).save(OUT / f"canvas_{name}.png")
        return canvas

    from vaw.camera import resolve_view

    agentview = shot("agentview")
    # Determinism is a hard requirement (canvases become SFT inputs and RL
    # observations), so assert it rather than trusting the code to be pure.
    assert np.array_equal(agentview, render_canvas(state, obs, cloud=cloud)), "render not deterministic"

    for preset in ("top", "left", "right", "low", "close"):
        resolve_view(state.view, preset=preset)
        shot(preset)

    # Focus is explicit: without inspect the detailed crop stays unavailable.
    resolve_view(state.view, preset="agentview")
    state.focus_id = "obj1"
    shot("focus_explicit")
    resolve_view(state.view, preset="left")
    shot("focus_left")

    resolve_view(state.view, preset="agentview")
    state.focus_id = None
    Image.fromarray(render_canvas(state, None)).save(OUT / "canvas_no_obs.png")

    summary = state.summary()
    (OUT / "state_summary.json").write_text(json.dumps(summary, indent=2))
    tools = tool_definitions()
    (OUT / "tool_definitions.json").write_text(json.dumps(tools, indent=2))

    op, args = parse_action({"name": "view", "arguments": {"preset": "top"}})
    assert op == "view" and args["preset"] == "top"
    op, args = parse_action({"name": "inspect", "arguments": {"object_id": "obj1"}})
    assert op == "inspect"

    # Detail follows explicit focus. A selection alone must not reveal the crop
    # or expanded numeric evidence.
    assert "focus" not in summary, summary
    obj1 = next(o for o in summary["objects"] if o["id"] == "obj1")
    assert "obb_yaw_deg" not in obj1 and "n_points" not in obj1, obj1
    state.focus_id = "obj1"
    focused = state.summary()
    assert focused["focus"] == {"object_id": "obj1", "requested": True}
    focused_obj1 = next(o for o in focused["objects"] if o["id"] == "obj1")
    assert "obb_yaw_deg" in focused_obj1 and "n_points" in focused_obj1, focused_obj1
    state.add_object(
        ObjectEntry(
            object_id=state.next_id("obj"),
            name="milk carton",
            score=0.7,
            obs_revision=state.obs_revision,
            obb={"center": [0.55, -0.24, 0.075], "extent": [0.07, 0.07, 0.15], "R": np.eye(3).tolist()},
        )
    )
    obj2 = next(o for o in state.summary()["objects"] if o["id"] == "obj2")
    assert "obb_yaw_deg" not in obj2 and "n_points" not in obj2, obj2

    print(f"[smoke] ops: {sorted(t['function']['name'] for t in tools)}")
    print(f"[smoke] summary keys: {sorted(summary)}")
    print(f"[smoke] view state: {summary['view']}")
    print(f"[smoke] wrote {len(list(OUT.glob('canvas_*.png')))} canvases to {OUT}")
    print("smoke OK")


if __name__ == "__main__":
    main()
