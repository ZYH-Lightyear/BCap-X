#!/usr/bin/env python3
"""手动跑一遍 pick 流水线,不接 agent —— 用来验证技能正文的代码能不能直接跑通。

流程固定:grounding -> OBB 短轴抓取位姿 -> 可行性预检 -> 预抓取 -> 下降 -> 合爪
-> 抬起 -> 抓取验证。每一步都把「请求了什么」和「实际达成了什么」并排打出来,
不做任何重试和兜底 —— 兜底会把偏差藏起来,而这个脚本存在的意义就是让偏差露出来。

为什么直接调 API 而不是走 ``code_env.step(code_string)``:后者把代码当字符串执行,
traceback 被 Tee 走、失败点的变量没法检查。这里从 ``_exec_globals`` 里取出同一批
函数对象直接调,副作用完全一致 —— 同一个仿真、同样的录像、reward 照样能读。

三处已知的静默失真,脚本会显式检出(它们不会抛异常,只会让结果悄悄变错):

1. ``solve_ik`` 把目标位置 clip 到固定工作空间,请求 x=0.76 会被压成 0.75。
2. ``solve_ik`` 在请求姿态解不出来时会静默换成 top-down / 45-tilt / side-approach,
   而 ``goto_pose`` 没有把这个信息透出来。短轴对齐抓取一旦被换成通用 top-down,
   整个 affordance 计算就白做了。
3. **请求的"TCP"不在指尖,而在指尖下方约 8.9cm。** ``_TCP_OFFSET`` 是 ``(0,0,-0.1)``,
   ``apply_tcp_offset`` 算的是 ``pos + R·offset``,自上而下时把 eef 往**上**推 10cm;
   而实测指尖只在 eef 前方 1.1cm。两者不抵消,净效果是手指在请求位置的上方 8.9cm
   处合拢。把物体表面的抓取点直接传给 ``goto_pose``,爪子必然在物体上方合空 ——
   前面所有 grounding 和 affordance 算得再准都没用。这一项这个脚本只报不改:
   要不要在 ``libero_reduced`` 里改偏移量是上游的契约问题,不该由调用方各补各的。

用法::

    python scripts/manual_pick_pipeline.py
    python scripts/manual_pick_pipeline.py --object "alphabet soup can" --finger-axis x
    python scripts/manual_pick_pipeline.py --suite libero_object --task-id 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# LIBERO 必须在 mujoco 导入前定好渲染后端,否则无显示环境下直接崩。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

DEFAULT_CONFIG = "env_configs/libero/franka_libero_cap_agent0.yaml"

#: 与 ``libero_reduced.FrankaLiberoApiReduced.solve_ik`` 里的 ``np.clip`` 边界一致。
#: 那里是静默裁剪,改了那边这里要跟着改 —— 没有更好的办法,常量没有暴露出来。
IK_POS_LOW = np.array([-0.1, -0.5, 0.005])
IK_POS_HIGH = np.array([0.75, 0.5, 0.9])

#: Franka panda 双指夹爪的最大开口(米)。用来判断物体短边塞不塞得进去。
MAX_GRIPPER_WIDTH = 0.08

#: 位置到位判据(米)。超过这个值说明 IK 裁剪、IK 误差或碰撞把手挡住了。
POSITION_TOLERANCE = 0.01

#: 归一化夹爪读数高出「空爪合拢」基线多少才算真的夹住了东西。基线本身不写死,
#: 每次开跑现场标定 —— 空爪合拢并不读 0,写死一个阈值必然误判。
GRIPPER_HOLD_MARGIN = 0.05

#: 重新定位时,新中心离预期位置多远就不再认为是同一个物体(米)。
#: 超出即报歧义,绝不静默当成「物体移动了」。
ASSOCIATION_GATE = 0.06


# ---------------------------------------------------------------------------
#  输出
# ---------------------------------------------------------------------------


class Report:
    """把每一步的检查结果同时打到终端和一份 JSON 里。"""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.failed = False

    def stage(self, title: str) -> None:
        print(f"\n{'─' * 72}\n{title}\n{'─' * 72}", flush=True)

    def info(self, label: str, value: Any) -> None:
        print(f"    {label:<28} {_fmt(value)}", flush=True)

    def check(self, ok: bool, label: str, detail: Any = "") -> bool:
        """硬判据:不通过就把整轮标记为失败,预检阶段会据此中断。"""
        mark = "✓" if ok else "✗"
        print(f"  {mark} {label:<30} {_fmt(detail)}", flush=True)
        self.checks.append({"label": label, "ok": bool(ok), "detail": _fmt(detail)})
        if not ok:
            self.failed = True
        return ok

    def note(self, ok: bool, label: str, detail: Any = "") -> bool:
        """诊断项:值得看见,但不构成中断理由。

        例如回转体没有有意义的短轴 —— 这是事实,不是错误,任意偏航都能抓。
        """
        mark = "✓" if ok else "⚠"
        print(f"  {mark} {label:<30} {_fmt(detail)}", flush=True)
        self.checks.append(
            {"label": label, "ok": bool(ok), "detail": _fmt(detail), "severity": "note"}
        )
        return ok

    def fatal(self, message: str) -> None:
        print(f"\n  !! 中断: {message}", flush=True)
        self.checks.append({"label": "ABORT", "ok": False, "detail": message})
        self.failed = True


def _fmt(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=4, suppress_small=True)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


# ---------------------------------------------------------------------------
#  感知:和 grounding-objects 技能里那条链完全一致
# ---------------------------------------------------------------------------


def ground_object(fns: dict[str, Callable], name: str, camera: str) -> dict[str, Any]:
    """定位一个具名物体。失败抛 RuntimeError。"""
    cam = fns["get_observation"]()[camera]
    rgb = cam["images"]["rgb"]
    box = fns["vlm_bbox_detection"](rgb, name)
    results = fns["segment_sam3_box_prompt"](rgb, box)
    if not results:
        raise RuntimeError(f"SAM3 在框 {box} 内没分割出 {name!r}")
    mask = max(results, key=lambda r: r.get("score", 0.0))["mask"]
    pts = fns["mask_to_world_points"](
        mask, cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]
    )
    pts = pts[np.isfinite(pts).all(axis=1)]  # 无穷远点必须剔掉
    if len(pts) < 10:
        raise RuntimeError(f"{name!r} 只有 {len(pts)} 个有效三维点,深度不可用")
    return {
        "mask": mask,
        "points": pts,
        "center": pts.mean(axis=0),
        "obb": fns["get_oriented_bounding_box_from_3d_points"](pts),
        "box": box,
        "n_points": len(pts),
        # 留着给叠加图用:再取一次观测会重新渲染两个相机,而且画面可能已经变了。
        "rgb": rgb,
        "intrinsics": cam["intrinsics"],
        "pose_mat": cam["pose_mat"],
    }


# ---------------------------------------------------------------------------
#  抓取位姿叠加图
# ---------------------------------------------------------------------------


def project_world_to_pixel(
    point: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    shape: tuple[int, int],
) -> tuple[float, float] | None:
    """世界点投影到像素。

    LIBERO 的 ``pose_mat`` 按相机到世界处理,但不同后端的相机 z 符号相反,所以两个
    符号都算一遍,取落在画面内的那个。这一段照搬 ``robomex_old/perception/render.py``
    里踩出来的写法,不要"简化"掉 —— 符号猜错时投影会安静地落在画面外。
    """
    height, width = shape
    k = np.asarray(intrinsics, dtype=np.float64)
    p_cam = np.linalg.inv(np.asarray(camera_pose, dtype=np.float64)) @ np.append(
        np.asarray(point, dtype=np.float64).reshape(3), 1.0
    )

    fallback: tuple[float, float] | None = None
    for z in (float(p_cam[2]), float(-p_cam[2])):
        if abs(z) < 1e-6:
            continue
        u = float(k[0, 0] * (p_cam[0] / z) + k[0, 2])
        v = float(k[1, 1] * (p_cam[1] / z) + k[1, 2])
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        if 0 <= u < width and 0 <= v < height:
            return u, v
        fallback = fallback or (u, v)
    return fallback


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        return np.eye(3)
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


#: 夹爪各部件在 **eef 系**里的位置(米),+z 是接近方向。原点取 ``gripper0_eef``,
#: 因为那正是 ``solve_ik`` 控制、``robot_cartesian_pos`` 上报的那个连杆 —— 以它为锚
#: 画出来的线框才和机器人真实所在的位置一致。
#:
#: 这些数是**量出来的**,不是查手册估的。复现方法:把机械臂送到任意自上而下位姿,
#: 读 ``sim.data.xpos`` 里 ``gripper0_eef`` / ``gripper0_leftfinger`` /
#: ``gripper0_finger_joint1_tip`` / ``robot0_right_hand`` 四个 body,转到 eef 局部系。
#: 它们只用于画图,不参与任何控制。
GRIPPER_TIP_Z = 0.0114  # 指尖:在 eef 前方仅 1.1cm,不是 10cm
GRIPPER_FINGER_BACK_Z = -0.0446
GRIPPER_FLANGE_Z = -0.0970
GRIPPER_OPEN_HALF_SEP = 0.0485  # 全开时单侧指尖偏移
GRIPPER_PALM_HALF_PERP = 0.020
GRIPPER_PALM_HALF_OPEN = 0.045
GRIPPER_FINGER_HALF_PERP = 0.0125
GRIPPER_FINGER_HALF_OPEN = 0.010

#: 立方体 8 个角的符号,以及连接它们的 12 条棱(只差一位的两个角相邻)。
_CUBE_SIGNS = np.array(
    [[1.0 - 2 * ((i >> 2) & 1), 1.0 - 2 * ((i >> 1) & 1), 1.0 - 2 * (i & 1)] for i in range(8)]
)
_CUBE_EDGES = [
    (i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1
]


def _draw_wire_box(
    draw: Any,
    project: Callable[[np.ndarray], tuple[float, float] | None],
    origin: np.ndarray,
    rot: np.ndarray,
    center_local: np.ndarray,
    half_local: np.ndarray,
    *,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    """把一个定义在夹爪局部系里的长方体投影成线框画出来。"""
    corners = origin + (center_local + _CUBE_SIGNS * half_local) @ rot.T
    pixels = [project(corner) for corner in corners]
    for i, j in _CUBE_EDGES:
        if pixels[i] is not None and pixels[j] is not None:
            draw.line((*pixels[i], *pixels[j]), fill=fill, width=width)


def save_grasp_overlay(
    path: Path,
    grounded: dict[str, Any],
    grasp: dict[str, Any],
    *,
    request_pos: np.ndarray,
    finger_axis: str,
    tcp_offset: np.ndarray,
    jaw_half_sep: float = GRIPPER_OPEN_HALF_SEP,
    approach_len: float = 0.08,
) -> str:
    """把夹爪按抓取位姿以三维线框画到 grounding 用的那张图上。

    画整个夹爪而不是几条示意线:掌部一个盒子、两根手指各一个盒子,按实测比例摆在
    张开状态。这样"手指会不会跨在物体两侧"、"掌部会不会先撞到别的东西"、"指尖到底
    扎多深"都能直接看出来,而这三件事光看三个浮点数看不出来。

    线框锚在 ``request_pos`` 经 ``apply_tcp_offset`` 算出的 **eef 位姿**上,不是锚在
    ``request_pos`` 本身。这两者差了将近 10cm:``_TCP_OFFSET`` 是 ``(0,0,-0.1)``,
    自上而下时把 eef 往**上**推 10cm,而实测指尖只在 eef 前方 1.1cm —— 所以传进
    ``solve_ik`` 的那个"TCP"实际落在指尖下方约 8.9cm 处,根本不是手指合拢的地方。
    锚错了这张图会画得很漂亮而且完全是假的。

    Args:
        grasp: affordance 算出来的抓取位姿,``position`` 是**希望指尖到达**的点,
            画成黄点。
        request_pos: 实际要传给 ``solve_ik`` 的 position,决定线框画在哪。开了 TCP
            补偿时它和 ``grasp["position"]`` 不同。
    """
    from PIL import Image, ImageDraw

    rgb = np.asarray(grounded["rgb"]).astype(np.uint8)
    height, width = rgb.shape[:2]
    image = Image.fromarray(rgb[..., :3]).convert("RGBA")

    mask = np.asarray(grounded["mask"]).astype(bool)
    if mask.shape == (height, width):
        tint = np.zeros((height, width, 4), dtype=np.uint8)
        tint[mask] = (0, 255, 0, 50)
        image.alpha_composite(Image.fromarray(tint, mode="RGBA"))

    draw = ImageDraw.Draw(image)
    draw.rectangle([float(v) for v in grounded["box"]], outline=(255, 0, 0, 200), width=1)

    def project(point: np.ndarray) -> tuple[float, float] | None:
        return project_world_to_pixel(
            point, grounded["intrinsics"], grounded["pose_mat"], (height, width)
        )

    target = np.asarray(grasp["position"], dtype=np.float64)
    quat = np.asarray(grasp["quaternion_wxyz"], dtype=np.float64)
    rot = quat_wxyz_to_matrix(quat)
    eef = hand_target_for(np.asarray(request_pos, dtype=np.float64), quat, tcp_offset)
    tip = eef + rot[:, 2] * GRIPPER_TIP_Z

    # 开合方向占局部哪一根轴,另一根就是手指的厚度方向。
    open_idx = 0 if finger_axis == "x" else 1
    perp_idx = 1 - open_idx

    def local(z: float, along_open: float = 0.0) -> np.ndarray:
        vec = np.zeros(3)
        vec[2] = z
        vec[open_idx] = along_open
        return vec

    def half(z: float, open_h: float, perp_h: float) -> np.ndarray:
        vec = np.zeros(3)
        vec[2] = z
        vec[open_idx] = open_h
        vec[perp_idx] = perp_h
        return vec

    # 接近箭头画在法兰**后面**,画到指尖会整根埋在线框内部看不见。
    tail = project(eef + rot[:, 2] * (GRIPPER_FLANGE_Z - approach_len))
    head = project(eef + rot[:, 2] * (GRIPPER_FLANGE_Z - 0.005))
    if tail is not None and head is not None:
        _draw_arrow(draw, tail, head, fill=(255, 140, 0, 230), width=3)

    _draw_wire_box(
        draw, project, eef, rot,
        local((GRIPPER_FLANGE_Z + GRIPPER_FINGER_BACK_Z) / 2.0),
        half((GRIPPER_FINGER_BACK_Z - GRIPPER_FLANGE_Z) / 2.0,
             GRIPPER_PALM_HALF_OPEN, GRIPPER_PALM_HALF_PERP),
        fill=(120, 170, 255, 220), width=2,
    )
    for side in (-1.0, 1.0):
        _draw_wire_box(
            draw, project, eef, rot,
            local((GRIPPER_TIP_Z + GRIPPER_FINGER_BACK_Z) / 2.0, side * jaw_half_sep),
            half((GRIPPER_TIP_Z - GRIPPER_FINGER_BACK_Z) / 2.0,
                 GRIPPER_FINGER_HALF_OPEN, GRIPPER_FINGER_HALF_PERP),
            fill=(0, 230, 255, 255), width=2,
        )

    # 想抓的地方 vs 手指实际合拢的地方。两者不重合就是抓空,画一根线把差距标出来。
    gap = float(np.linalg.norm(tip - target))
    target_px = project(target)
    tip_px = project(tip)
    if target_px is not None:
        if tip_px is not None and gap > 0.01:
            draw.line((*tip_px, *target_px), fill=(255, 60, 60, 255), width=2)
            mid = ((tip_px[0] + target_px[0]) / 2 + 6, (tip_px[1] + target_px[1]) / 2)
            draw.text(mid, f"{gap * 100:.1f}cm", fill=(255, 90, 90, 255))
        u, v = target_px
        draw.ellipse((u - 4, v - 4, u + 4, v + 4), fill=(255, 230, 0, 255), outline=(0, 0, 0, 255))

    # 场景底色是木纹,浅黄字直接压上去读不出来,先铺一条深底。
    lines = [
        f"target z={target[2]:.3f}  yaw={np.degrees(grasp['yaw_cmd']):.0f}deg  {finger_axis}-jaw",
        f"fingertips z={tip[2]:.3f}   gap to target = {gap * 100:.1f}cm",
    ]
    box = draw.multiline_textbbox((6, 5), "\n".join(lines))
    draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0, 190))
    draw.multiline_text(
        (6, 5), "\n".join(lines),
        fill=(255, 90, 90, 255) if gap > 0.01 else (255, 235, 60, 255),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return str(path)


def _draw_arrow(draw: Any, start: tuple[float, float], end: tuple[float, float],
                *, fill: tuple[int, int, int, int], width: int) -> None:
    draw.line((*start, *end), fill=fill, width=width)
    vec = np.array([end[0] - start[0], end[1] - start[1]], dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return
    direction = vec / norm
    normal = np.array([-direction[1], direction[0]])
    head = 8.0
    tip = np.array(end, dtype=np.float64)
    draw.polygon(
        [tuple(tip), tuple(tip - direction * head + normal * head * 0.45),
         tuple(tip - direction * head - normal * head * 0.45)],
        fill=fill,
    )


# ---------------------------------------------------------------------------
#  affordance:OBB 短轴抓取
# ---------------------------------------------------------------------------


def obb_short_axis_grasp(
    points: np.ndarray,
    obb: dict[str, Any],
    *,
    finger_axis: str,
    depth_from_top: float,
    to_quat: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """由 OBB 算一个自上而下、手指开合方向对齐物体短边的抓取位姿。

    顶面取 **OBB 八个角点的 z 上界**,不要用 ``points[:, 2].max()``。后者是所有统计量
    里对离群点最敏感的一个:一个坏深度像素就能把顶面顶高好几厘米,于是抓取点悬在物体
    上方合了个空。而 ``get_oriented_bounding_box_from_3d_points`` 内部做过统计离群点
    剔除,角点上界既避开了离群点,又不依赖「OBB 竖直轴真的竖直」这个假设。

    Args:
        points: (N, 3) 世界系点云,只用来交叉核对顶面估计。
        obb: ``{"center", "extent", "R"}``,R 的列向量是三条轴。
        finger_axis: 夹爪张合方向对应的末端轴,``"x"`` 或 ``"y"``。
            Franka panda hand 的约定不是自明的,所以做成开关由录像来判。
        depth_from_top: 抓取点在物体顶面以下多少米。
        to_quat: 旋转矩阵转 wxyz 四元数,传 API 里那个而不是自己写。
    """
    center = np.asarray(obb["center"], dtype=np.float64)
    extent = np.asarray(obb["extent"], dtype=np.float64)
    rot = np.asarray(obb["R"], dtype=np.float64)

    # R 的第 i 列是第 i 条轴,它的世界 z 分量就是 rot[2, i]。
    vertical = int(np.argmax(np.abs(rot[2, :])))
    horizontals = [i for i in range(3) if i != vertical]
    short = min(horizontals, key=lambda i: extent[i])
    long_ = max(horizontals, key=lambda i: extent[i])

    axis = rot[:, short].copy()
    axis[2] = 0.0  # 投影到水平面:自上而下抓取只关心偏航
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        raise RuntimeError(
            f"短轴 {rot[:, short]} 几乎垂直,水平投影退化,OBB 不可用于自上而下抓取"
        )
    axis /= norm

    yaw = float(np.arctan2(axis[1], axis[0]))
    # 自上而下基准姿态:夹爪 z 轴朝下。列向量 x=[1,0,0] y=[0,-1,0] z=[0,0,-1],
    # 对应四元数 [0, 1, 0, 0](绕 x 转 π),就是 solve_ik docstring 里的 canonical。
    base = np.diag([1.0, -1.0, -1.0])
    # 绕世界 z 转 yaw 之后,夹爪 x 轴落在 axis 上;要让 y 轴落上去就再转 90°。
    yaw_cmd = yaw + (np.pi / 2 if finger_axis == "y" else 0.0)
    cos_y, sin_y = np.cos(yaw_cmd), np.sin(yaw_cmd)
    rot_z = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
    quat = np.asarray(to_quat(rot_z @ base), dtype=np.float64)

    # OBB 的 8 个角点:局部 (±a/2, ±b/2, ±c/2) 转到世界系。
    signs = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
    )
    corners = center + (signs * (extent / 2.0)) @ rot.T
    z_top = float(corners[:, 2].max())
    z_bottom = float(corners[:, 2].min())
    grasp_z = max(z_top - depth_from_top, z_bottom + 0.005)

    # 水平两边差不多长时,「短轴」是噪声而不是形状。回转体(罐、瓶、碗)必然如此,
    # 此时任意偏航都能抓,但要说出来,而不是给一个假装有意义的角度。
    short_e, long_e = float(extent[short]), float(extent[long_])
    anisotropy = (long_e - short_e) / long_e if long_e > 1e-9 else 0.0

    return {
        "position": np.array([center[0], center[1], grasp_z]),
        "quaternion_wxyz": quat,
        "yaw": yaw,
        "yaw_cmd": yaw_cmd,
        "short_axis": axis,
        "short_extent": short_e,
        "long_extent": long_e,
        "vertical_extent": float(extent[vertical]),
        "anisotropy": anisotropy,
        "z_top": z_top,
        "z_bottom": z_bottom,
        # 交叉核对:点云极值和 OBB 角点差多少就是离群污染有多重。
        "z_top_raw": float(points[:, 2].max()),
        "z_bottom_raw": float(points[:, 2].min()),
    }


# ---------------------------------------------------------------------------
#  运动:每一次都对比请求与达成
# ---------------------------------------------------------------------------


def hand_target_for(position: np.ndarray, quat: np.ndarray, tcp_offset: np.ndarray) -> np.ndarray:
    """请求的 TCP 位置对应的 ``panda_hand`` 连杆位置。

    ``solve_ik`` 把入参当作 TCP(指尖)并加一个末端系偏移去解连杆位姿,而
    ``robot_cartesian_pos`` 报的是连杆。两者直接相减会得到一个恒定的假误差 ——
    自上而下姿态下正好是 10cm,和真实控制精度无关。
    """
    from capx.integrations.franka.common import apply_tcp_offset

    return np.asarray(apply_tcp_offset(position, quat, tcp_offset), dtype=np.float64)


def request_for_fingertips(
    tip_target: np.ndarray, quat: np.ndarray, tcp_offset: np.ndarray
) -> np.ndarray:
    """把「想让指尖到达哪」反解成必须传给 ``solve_ik`` 的那个 ``position``。

    正向链路是 ``tip = pos + R·tcp_offset + R·(0,0,GRIPPER_TIP_Z)``,反过来就是
    ``pos = tip - R·(tcp_offset + (0,0,GRIPPER_TIP_Z))``。

    这是**验证假设用的**,不是修复。真要修,该改的是 ``libero_reduced._TCP_OFFSET``
    —— 每个调用方各补一份偏移,只会让下一个人更难发现契约本身是错的。
    """
    rot = quat_wxyz_to_matrix(quat)
    correction = np.asarray(tcp_offset, dtype=np.float64) + np.array([0.0, 0.0, GRIPPER_TIP_Z])
    return np.asarray(tip_target, dtype=np.float64) - rot @ correction


def move_and_verify(
    fns: dict[str, Callable],
    report: Report,
    label: str,
    position: np.ndarray,
    quat: np.ndarray,
    tcp_offset: np.ndarray,
) -> bool:
    """解 IK 并移动,把静默裁剪、静默换姿态、位置误差三件事都检出来。

    这里刻意不用 ``goto_pose``:它内部调 ``solve_ik`` 时没传 ``return_info``,
    姿态被换掉了调用方一无所知。
    """
    report.info(f"{label} 请求 TCP", position)

    clipped = np.clip(position, IK_POS_LOW, IK_POS_HIGH)
    clip_delta = float(np.linalg.norm(clipped - position))
    report.check(
        clip_delta < 1e-9,
        f"{label} 未被工作空间裁剪",
        "" if clip_delta < 1e-9 else f"被压到 {_fmt(clipped)},差 {clip_delta * 100:.1f}cm",
    )

    try:
        joints, ik_info = fns["solve_ik"](position, quat, return_info=True)
    except Exception as exc:  # noqa: BLE001 - IK 彻底失败是有效结论
        report.check(False, f"{label} IK 可解", f"{type(exc).__name__}: {exc}")
        return False

    used = ik_info.get("orientation_used")
    report.check(
        used == "requested",
        f"{label} 姿态未被替换",
        "" if used == "requested" else f"IK 静默改用了 {used!r},短轴对齐已失效",
    )

    fns["move_to_joints"](joints)

    achieved_pos, achieved_quat, _ = robot_state(fns)
    expected = hand_target_for(position, quat, tcp_offset)
    error = float(np.linalg.norm(achieved_pos - expected))
    report.info(f"{label} 期望连杆位置", expected)
    report.info(f"{label} 实际连杆位置", achieved_pos)
    report.check(
        error <= POSITION_TOLERANCE,
        f"{label} 到位",
        f"误差 {error * 100:.2f}cm",
    )
    # 四元数有正负号双重表示,比较时取绝对内积。
    quat_align = abs(float(np.dot(achieved_quat, quat)))
    report.check(quat_align > 0.99, f"{label} 姿态到位", f"|cos| = {quat_align:.4f}")
    return error <= POSITION_TOLERANCE


def robot_state(fns: dict[str, Callable]) -> tuple[np.ndarray, np.ndarray, float]:
    """末端位置、wxyz 四元数、归一化夹爪开口(0 全闭 1 全开)。

    ``get_ee_pose`` 虽然在 API 类里定义了,却没有注册进 ``functions()``,沙箱里根本
    取不到 —— agent 也一样取不到。唯一的本体状态入口是 ``get_observation()`` 里那个
    ``robot_cartesian_pos``:xyz(3)+ wxyz(4)+ 夹爪(1),信息一样,只是埋得深。

    一次 ``get_observation()`` 会把两个相机都渲染一遍,所以三个量一次读回来,
    不要分开调。
    """
    state = np.asarray(
        fns["get_observation"]()["robot_cartesian_pos"], dtype=np.float64
    ).reshape(-1)
    return state[:3].copy(), state[3:7].copy(), float(state[7])


# ---------------------------------------------------------------------------
#  环境
# ---------------------------------------------------------------------------


def build_env(config: str, overrides: dict[str, Any]) -> Any:
    import capx.integrations  # noqa: F401 - 导入包才会触发 API 注册
    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader

    cfg = DictLoader.load(config)
    for dotted, value in overrides.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            if key not in node:
                raise KeyError(f"配置里没有路径 {dotted!r}(卡在 {key!r})")
            node = node[key]
        node[leaf] = value
    return instantiate(cfg["env"])


def read_tcp_offset(code_env: Any) -> np.ndarray:
    """从 API 实例上读 TCP 偏移,不在这里另抄一份常量。

    抄常量的问题不是麻烦,是它会在上游改了之后继续给出看起来合理的错误结论。
    """
    for api in getattr(code_env, "_apis", {}).values():
        offset = getattr(api, "_TCP_OFFSET", None)
        if offset is not None:
            return np.asarray(offset, dtype=np.float64)
    raise RuntimeError("在任何 API 实例上都找不到 _TCP_OFFSET,位置校验会失真")


def save_png(rgb: Any, path: Path) -> None:
    from PIL import Image

    array = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else rgb
    Image.fromarray(np.asarray(array).astype("uint8")).save(path)


def save_video(code_env: Any, path: Path) -> str | None:
    frames = code_env.get_video_frames(clear=False)
    if not frames:
        return "(没有采到任何帧)"
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=30)
    return f"{path}({len(frames)} 帧)"


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()

    overrides: dict[str, Any] = {}
    if args.suite:
        overrides["env.cfg.low_level.suite_name"] = args.suite
    if args.task_id is not None:
        overrides["env.cfg.low_level.task_id"] = args.task_id

    run_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    report = Report()
    report.stage("0. 建环境")
    report.info("配置", args.config)
    for key, value in overrides.items():
        report.info(f"覆盖 {key}", value)

    code_env = build_env(args.config, overrides)
    if not args.no_video:
        code_env.enable_video_capture(True, clear=True)
    code_env.reset(seed=args.seed)

    # 直接取沙箱命名空间里的函数对象:和 agent 通过 run_python 拿到的是同一批。
    fns = code_env._exec_globals
    handle = getattr(code_env.low_level_env, "handle", None)
    task_language = getattr(handle, "task_language", None)
    report.info("任务", task_language)
    report.info("目标物体", args.object)
    report.info("输出目录", run_dir)

    obs = fns["get_observation"]()
    save_png(obs[args.camera]["images"]["rgb"], frames_dir / "00_initial.png")

    home_pos, _, home_width = robot_state(fns)
    report.info("初始连杆位置", home_pos)
    report.info("初始夹爪开口", home_width)

    tcp_offset = read_tcp_offset(code_env)
    report.info("TCP 偏移(末端系)", tcp_offset)

    # -- 0.5 夹爪标定 -------------------------------------------------------
    # 空爪合拢并不读 0。不现场量一下,后面「指间有没有东西」就只能靠猜一个阈值,
    # 而猜错的方向恰好是「永远判定抓住了」。
    report.stage("0.5 夹爪标定(空爪合拢读数是多少)")
    fns["close_gripper"]()
    _, _, empty_closed = robot_state(fns)
    fns["open_gripper"]()
    _, _, full_open = robot_state(fns)
    report.info("空爪合拢 / 全开", f"{empty_closed:.4f} / {full_open:.4f}")
    hold_threshold = empty_closed + GRIPPER_HOLD_MARGIN
    report.info("判定夹住的阈值", f"> {hold_threshold:.4f}")

    # -- 1. grounding ------------------------------------------------------
    report.stage("1. Grounding(VLM 框 -> SAM3 掩码 -> 世界点云)")
    try:
        grounded = ground_object(fns, args.object, args.camera)
    except Exception as exc:  # noqa: BLE001
        report.fatal(f"grounding 失败: {type(exc).__name__}: {exc}")
        return _finish(report, code_env, run_dir, args)

    report.info("像素框", np.asarray(grounded["box"]))
    report.info("有效三维点", grounded["n_points"])
    report.info("点云质心", grounded["center"])
    report.info("OBB extent", np.asarray(grounded["obb"]["extent"]))
    report.check(grounded["n_points"] >= 200, "点数足够拟合 OBB", grounded["n_points"])

    # -- 2. affordance -----------------------------------------------------
    report.stage("2. Affordance(OBB 短轴 -> 抓取位姿)")
    try:
        grasp = obb_short_axis_grasp(
            grounded["points"],
            grounded["obb"],
            finger_axis=args.finger_axis,
            depth_from_top=args.depth_from_top,
            to_quat=fns["rotation_matrix_to_quaternion"],
        )
    except Exception as exc:  # noqa: BLE001
        report.fatal(f"affordance 计算失败: {type(exc).__name__}: {exc}")
        return _finish(report, code_env, run_dir, args)

    report.info(
        "短边 / 长边 / 高",
        f"{grasp['short_extent']:.4f} / {grasp['long_extent']:.4f} / {grasp['vertical_extent']:.4f}",
    )
    report.info("顶面 / 底面 z(OBB 角点)", f"{grasp['z_top']:.4f} / {grasp['z_bottom']:.4f}")
    report.info("顶面 / 底面 z(点云极值)", f"{grasp['z_top_raw']:.4f} / {grasp['z_bottom_raw']:.4f}")
    contamination = grasp["z_top_raw"] - grasp["z_top"]
    report.note(
        contamination < 0.01,
        "点云顶面未被离群点抬高",
        f"比 OBB 角点高 {contamination * 100:.1f}cm",
    )
    report.info("短轴(水平投影)", grasp["short_axis"])
    report.note(
        grasp["anisotropy"] > 0.15,
        "水平两边差异足够大,短轴有意义",
        f"各向异性 {grasp['anisotropy'] * 100:.0f}%"
        + ("" if grasp["anisotropy"] > 0.15 else " —— 近似回转体,偏航角是噪声(任意偏航都能抓)"),
    )
    report.info(f"偏航 yaw / 指令 yaw({args.finger_axis} 轴张合)",
                f"{np.degrees(grasp['yaw']):.1f}° / {np.degrees(grasp['yaw_cmd']):.1f}°")
    report.info("抓取 TCP 位置", grasp["position"])
    report.info("抓取四元数 wxyz", grasp["quaternion_wxyz"])

    # -- 3. 可行性预检 ------------------------------------------------------
    report.stage("3. 可行性预检(动之前该拒绝的就拒绝)")
    report.check(
        grasp["short_extent"] < MAX_GRIPPER_WIDTH,
        "短边塞得进夹爪",
        f"{grasp['short_extent'] * 100:.1f}cm < {MAX_GRIPPER_WIDTH * 100:.0f}cm",
    )
    quat = grasp["quaternion_wxyz"]

    def to_request(tip_target: np.ndarray) -> np.ndarray:
        """把「想让指尖到达哪」换算成实际要传给 solve_ik 的 position。"""
        if not args.compensate_tcp:
            return np.asarray(tip_target, dtype=np.float64)
        return request_for_fingertips(tip_target, quat, tcp_offset)

    pregrasp = grasp["position"] + np.array([0.0, 0.0, args.approach_height])
    lift = grasp["position"] + np.array([0.0, 0.0, args.lift_height])

    # 裁剪检查要查**实际发出去的那个 position**,不是我们心里想的目标点。开了补偿
    # 之后目标会往下推 8.9cm,这时候撞上 z>=0.005 的下界才是真问题。
    for label, tip_target in (("抓取点", grasp["position"]), ("预抓取点", pregrasp), ("抬起点", lift)):
        pos = to_request(tip_target)
        delta = float(np.linalg.norm(np.clip(pos, IK_POS_LOW, IK_POS_HIGH) - pos))
        report.check(
            delta < 1e-9,
            f"{label}在工作空间内",
            "" if delta < 1e-9 else f"超出 {delta * 100:.1f}cm,solve_ik 会静默裁剪",
        )

    # 手指真正合拢的地方,和我们想抓的点差多远。这是整条流水线里最容易被忽略、
    # 后果又最彻底的一项:``_TCP_OFFSET`` 把 eef 往接近方向的反方向推 0.1m,而指尖
    # 只在 eef 前方 GRIPPER_TIP_Z,两者不抵消。差 8.9cm 意味着前面所有 affordance
    # 计算都白做 —— 爪子会在物体上方合空。
    tip_at = hand_target_for(to_request(grasp["position"]), quat, tcp_offset) + (
        quat_wxyz_to_matrix(quat)[:, 2] * GRIPPER_TIP_Z
    )
    tip_gap = float(np.linalg.norm(tip_at - grasp["position"]))
    report.info("TCP 补偿", "已开启" if args.compensate_tcp else "未开启")
    report.info("指尖实际所在", tip_at)
    report.check(
        tip_gap <= 0.01,
        "指尖落在想抓的点上",
        f"差 {tip_gap * 100:.1f}cm —— 想抓 z={grasp['position'][2]:.3f} 但指尖在 z={tip_at[2]:.3f}",
    )

    # 位姿定下来之后先落一张叠加图。动之前就能看出钳口有没有跨在物体上、
    # 指尖是不是悬在半空 —— 这些光看三个浮点数是看不出来的。
    report.info("抓取位姿叠加图", save_grasp_overlay(
        frames_dir / "grasp_overlay.png",
        grounded,
        grasp,
        request_pos=to_request(grasp["position"]),
        finger_axis=args.finger_axis,
        tcp_offset=tcp_offset,
    ))

    if report.failed and not args.force:
        report.fatal("预检未通过。要照样执行看看效果,加 --force")
        return _finish(report, code_env, run_dir, args)

    # -- 4~7. 执行 ---------------------------------------------------------
    report.stage("4. 移动到预抓取点")
    fns["open_gripper"]()
    move_and_verify(fns, report, "预抓取", to_request(pregrasp), quat, tcp_offset)
    save_png(fns["get_observation"]()[args.camera]["images"]["rgb"], frames_dir / "01_pregrasp.png")

    report.stage("5. 下降到抓取点")
    move_and_verify(fns, report, "抓取", to_request(grasp["position"]), quat, tcp_offset)
    save_png(fns["get_observation"]()[args.camera]["images"]["rgb"], frames_dir / "02_at_grasp.png")

    report.stage("6. 合爪")
    ee_before, _, before = robot_state(fns)
    fns["close_gripper"]()
    _, _, after = robot_state(fns)
    report.info("夹爪开口 合前 -> 合后", f"{before:.4f} -> {after:.4f}")
    report.info("空爪基线", f"{empty_closed:.4f}")
    report.check(
        after > hold_threshold,
        "指间有东西",
        f"{after:.4f} vs 阈值 {hold_threshold:.4f}"
        + ("" if after > hold_threshold else " —— 合到了空爪基线,什么都没夹到"),
    )
    save_png(fns["get_observation"]()[args.camera]["images"]["rgb"], frames_dir / "03_closed.png")

    report.stage("7. 抬起")
    move_and_verify(fns, report, "抬起", to_request(lift), quat, tcp_offset)
    save_png(fns["get_observation"]()[args.camera]["images"]["rgb"], frames_dir / "04_lifted.png")

    # -- 8. 抓取验证 --------------------------------------------------------
    report.stage("8. 抓取验证")
    ee_after, _, width_after = robot_state(fns)
    ee_delta = ee_after - ee_before
    report.info("末端位移", ee_delta)
    report.check(width_after > hold_threshold, "抬起后指间仍有东西", width_after)
    verify_object_moved_with_hand(fns, report, args, grounded["center"], ee_delta)

    return _finish(report, code_env, run_dir, args)


def verify_object_moved_with_hand(
    fns: dict[str, Callable],
    report: Report,
    args: argparse.Namespace,
    before_center: np.ndarray,
    ee_delta: np.ndarray,
) -> None:
    """重新定位目标,判断它是跟着手走了、留在原地、还是根本认不出来。

    重新定位是按名字问 VLM 的,而场景里往往有多个同类物体(好几个罐头),画面一变
    它完全可能给出**另一个实例**。所以不能拿「新中心减旧中心」直接当物体位移 ——
    那样会把「换了个罐子」误报成「罐子飞了 36 厘米」,而 agent 会照着这个假状态继续决策。

    做法是拿两个假设去套:抓住了(应在旧位置 + 末端位移)、没抓住(应在旧位置)。
    哪个都套不上就报歧义,不猜。
    """
    try:
        after = ground_object(fns, args.object, args.camera)
    except Exception as exc:  # noqa: BLE001
        report.check(False, "抬起后仍能定位到目标", f"{type(exc).__name__}: {exc}")
        return

    center = after["center"]
    report.info("重新定位到的中心", center)
    d_held = float(np.linalg.norm(center - (before_center + ee_delta)))
    d_stayed = float(np.linalg.norm(center - before_center))
    report.info("距「被夹住」假设", f"{d_held * 100:.1f}cm")
    report.info("距「留在原地」假设", f"{d_stayed * 100:.1f}cm")

    if min(d_held, d_stayed) > ASSOCIATION_GATE:
        report.check(
            False,
            "重新定位到的是同一个物体",
            f"两个假设都对不上(门限 {ASSOCIATION_GATE * 100:.0f}cm),"
            "很可能 VLM 这次指到了另一个同类物体 —— 状态不可信,不做判断",
        )
        return

    report.check(
        d_held < d_stayed,
        "物体跟着手在动",
        f"跟手 {d_held * 100:.1f}cm vs 原地 {d_stayed * 100:.1f}cm"
        + ("" if d_held < d_stayed else " —— 物体留在原地,没抓起来"),
    )


def _finish(report: Report, code_env: Any, run_dir: Path, args: argparse.Namespace) -> int:
    report.stage("结果")
    try:
        reward = float(code_env.compute_reward())
    except Exception:  # noqa: BLE001
        reward = float("nan")
    report.info("reward", reward)

    if not args.no_video:
        report.info("录像", save_video(code_env, run_dir / "rollout.mp4"))
    report.info("画面", run_dir / "frames")

    hard = [c for c in report.checks if c.get("severity") != "note"]
    report.info("硬判据通过", f"{sum(1 for c in hard if c['ok'])}/{len(hard)}")
    for item in report.checks:
        if not item["ok"]:
            mark = "⚠" if item.get("severity") == "note" else "✗"
            print(f"    {mark} {item['label']}  {item['detail']}", flush=True)

    (run_dir / "report.json").write_text(
        json.dumps(
            {"object": args.object, "finger_axis": args.finger_axis,
             "reward": reward, "checks": report.checks},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n报告 {run_dir / 'report.json'}\n", flush=True)
    return 0 if not report.failed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="CapX 环境配置 YAML")
    parser.add_argument("--suite", default=None, help="覆盖 LIBERO suite")
    parser.add_argument("--task-id", type=int, default=None, help="覆盖 task_id")
    parser.add_argument("--object", default="alphabet soup can", help="要抓的物体名")
    parser.add_argument("--camera", default="agentview", help="用哪个相机做 grounding")
    parser.add_argument(
        "--finger-axis", choices=("x", "y"), default="y",
        help="夹爪张合方向对应的末端轴。哪个对由录像判,默认 y",
    )
    parser.add_argument("--depth-from-top", type=float, default=0.02,
                        help="抓取点在物体顶面以下多少米,默认 0.02")
    parser.add_argument("--approach-height", type=float, default=0.075,
                        help="预抓取点比抓取点高多少米,默认 0.075(move_to_joints 的插值建议值)")
    parser.add_argument("--lift-height", type=float, default=0.15,
                        help="抬起点比抓取点高多少米,默认 0.15")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--compensate-tcp", action="store_true",
        help="把所有目标沿接近方向反推,使**指尖**落在算出来的抓取点上。"
             "用来验证「TCP 契约错位约 8.9cm」这个诊断,不是长期修法",
    )
    parser.add_argument("--force", action="store_true", help="预检不过也照样执行")
    parser.add_argument("--no-video", action="store_true", help="不录像")
    parser.add_argument("--out", default="outputs/manual_pick")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
