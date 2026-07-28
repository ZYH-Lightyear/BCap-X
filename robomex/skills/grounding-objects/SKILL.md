---
name: grounding-objects
description: >
  定位场景中的一个物体
produces_outputs:
  mask: "(H, W) bool 掩码"
  obb: "有向包围盒 {center, extent, R}"
  corners: "(8, 3) OBB 八个角点的世界坐标"
  top_z: "顶面世界高度(米),由角点取上界"
  center_xy: "(2,) 水平中心。没有 z —— 见「为什么不给 center」"
  box: "[x1, y1, x2, y2] 像素框"
  n_points: "有效三维点数"
---

# grounding-objects

拿到一个具名物体的 mask 和三维位置。一次调用走完 VLM 检测 → SAM3 分割 → 深度反投影。

## 何时使用

需要某个物体的**空间信息**时:抓取前定位目标、确定容器的放置点、确认物体还在不在场景里。

**不要用在**不涉及空间定位的视觉问题上——判断夹爪是否已经夹住、读包装上的文字、
判断任务是否完成。那些直接 `query_vlm(prompt, images=rgb)` 更快,而且
`query_vlm` 本身也不适合做空间定位。

## 做法

把下面这个函数原样贴进 `run_python` 执行一次。执行命名空间跨轮保持,所以定义一次之后
后续每一轮都能直接调用,不用重复粘贴。

```python
def ground_object(name, camera="agentview"):
    """定位一个具名物体。失败抛 RuntimeError。"""
    import numpy as np
    cam = get_observation()[camera]
    rgb = cam["images"]["rgb"]
    box = vlm_bbox_detection(rgb, name)          # 找不到会抛 AssertionError
    results = segment_sam3_box_prompt(rgb, box)
    if not results:
        raise RuntimeError(f"SAM3 在框 {box} 内没分割出 {name!r}")
    mask = max(results, key=lambda r: r.get("score", 0.0))["mask"]
    pts = mask_to_world_points(
        mask, cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]
    )
    pts = pts[np.isfinite(pts).all(axis=1)]      # 无穷远点必须剔掉
    if len(pts) < 10:
        raise RuntimeError(f"{name!r} 只有 {len(pts)} 个有效三维点,深度不可用")
    obb = get_oriented_bounding_box_from_3d_points(pts)
    # OBB 八个角:局部 (±a/2, ±b/2, ±c/2) 转到世界系。R 的列向量是三条轴。
    c, e, R = (np.asarray(obb[k], float) for k in ("center", "extent", "R"))
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    corners = c + (signs * (e / 2)) @ R.T
    return {
        "mask": mask,
        "obb": obb,
        "corners": corners,
        "top_z": float(corners[:, 2].max()),
        "center_xy": c[:2].copy(),
        "box": box,
        "n_points": len(pts),
    }
```

## 用法

```python
g = ground_object("alphabet soup can")
print(g["box"], g["center_xy"], g["top_z"], g["n_points"])
# 实测: [342.0, 201.0, 376.0, 253.0] [0.4145 -0.0808] 0.1079 1459
```

输出都是世界坐标,交给抓取/放置技能去算位姿。`mask` 可以直接作为
`plan_grasp(depth, intrinsics, segmentation)` 的 `segmentation` 参数。

物体名用自然、具体的说法。`"alphabet soup can"`、`"basket"` 都能命中;
`"the object"`、`"it"` 这种指代不明的会让 VLM 给出乱框。

## 为什么不给 center

单视角点云只覆盖**看得见的那层壳**——顶面和朝向相机的一侧,背面和底面一个点都没有。
在这么一团点上取平均,得到的是"可见表面的重心",它系统性地偏向相机、偏向顶面。
偏多少取决于物体多高、相机多斜、被挡了多少,不是一个能标定掉的常数。

这个技能早先返回 `center = pts.mean(axis=0)` 并且说"可以直接喂给 `goto_pose`"。
那是错的,而且错在最坏的那一类:它不报错,只是让爪子稳定地合在物体上方某处,
而前面 grounding 和 affordance 算得再准都救不回来。

三个分量的可观测性其实完全不同,不该打包成一个向量:

- **x、y 可信。** 顶面的水平轮廓和物体的水平轮廓基本重合,所以 `obb["center"][:2]` 能用。
- **z 不可信,而且你不需要它。** 自上而下抓取要的从来不是"中心高度",而是**顶面高度**
  ——那是唯一真正被观测到的高度。`top_z` 给的就是它,想往下扎多深自己减。

`top_z` 取的是 **OBB 八个角点的 z 上界**,不是 `pts[:, 2].max()`。后者是所有统计量里
对离群点最敏感的一个:一个坏深度像素就能把顶面抬高好几厘米,于是抓取点悬在半空。
实测两者能差 3~4.5cm。`get_oriented_bounding_box_from_3d_points` 内部做过离群点剔除,
角点上界既避开了离群点,又不依赖"OBB 竖直轴真的竖直"这个假设。

## 为什么不直接用 segment_sam3_text_prompt

`segment_sam3_text_prompt(rgb, text)` 一步就能出 mask,但它在目标含糊或不在画面里时
直接返回**空 list**,不给任何中间信息——你无法区分"物体不在场景里"和"物体在但没分割出来"。

先 VLM 出框把这两种失败分开了:`vlm_bbox_detection` 抛异常说明 VLM 就没找到目标,
该换个说法或换个相机;它给了框而 `segment_sam3_box_prompt` 返回空,说明目标位置知道了
但分割失败,该检查框对不对。多一次调用,换来可诊断的失败。

## 失败处理

三种失败各有各的信号,不要笼统地重试:

- `AssertionError: VLM bbox detection failed for '<name>'` —— VLM 没找到。换一个更具体的
  名字重试一次;仍失败就换相机(`camera="robot0_eye_in_hand"`)。你每轮都会收到当前画面,
  先看一眼那张图确认物体是不是真的可见,再决定要不要继续试。
- `RuntimeError: SAM3 ... 没分割出` —— 框拿到了但分割为空。少见,通常意味着框偏了。
- `RuntimeError: ... 只有 N 个有效三维点` —— 深度不可用。多半是物体被遮挡,或框套到了
  背景。换个视角再试。

还有一种**不抛异常**的失败:`vlm_bbox_detection` 只返回一个最佳框,场景里有多个同类物体
(两个罐子、两个碗)时,它每次可能指到不同的那一个。实测同一句 `"alphabet soup can"`
两次调用分别命中了 x=0.41 和 x=0.76 的两个罐子。

后果是:你以为在复查刚才那个物体,其实换了一个。所以**重新定位之后要和上次的位置对一对**
——离得远就是换了目标,不是物体动了。判不出来就如实说"分不清",别静默挑一个:
针对错误对象正确地执行一个动作,是这里最难发现的失败。要指定具体哪一个,把空间关系写进
名字里(`"the can closer to the basket"`)。

`mask_to_world_points` 只保留深度为正的像素,但无穷远点会漏过去,所以
`np.isfinite` 那一行不能省。
