# Franka API 解读（High-level / Low-level）

本文档基于 `capx/integrations/franka/control.py`（`FrankaControlApi`）与 `control_reduced.py`（`FrankaControlApiReduced`），统一格式说明各函数的签名、流程与关键步骤。

**约定**

- 坐标系：世界系（除非注明相机系）
- 姿态：四元数 **WXYZ**
- 位置 / 尺寸：米
- `bbox_extent`：OBB **完整边长**（非半长）
- High-level 内部已封装感知 + IK + 执行；Low-level 把流水线拆成原语，由调用方组合

**层级对照**

| High-level | 大致等价的 Low-level 组合 |
| --- | --- |
| `get_object_pose` | `get_observation` → `segment_sam3_*` → 深度反投影 → `get_oriented_bounding_box_from_3d_points` → 相机→世界变换 |
| `sample_grasp_pose` | `get_observation` → `segment_sam3_*` → `plan_grasp` → 相机→世界 + TCP 平移 |
| `goto_pose` | （可选接近点）`solve_ik` → `move_to_joints` →（终点）`solve_ik` → `move_to_joints` |
| `open_gripper` / `close_gripper` | 同名 Low-level（直接 `_set_gripper` + step） |

---

## 1. High-level API（`FrankaControlApi`）

源码：`capx/integrations/franka/control.py`  
配置中常见：`apis: [FrankaControlApi]`

---

### 1.1 get_object_pose

```text
position, quaternion_wxyz, bbox_extent = get_object_pose(object_name, return_bbox_extent=False)
```

| | |
| --- | --- |
| **输入** | `object_name: str` — 自然语言物体名（如 `"red cube"`） |
| **其他参数** | `return_bbox_extent: bool = False` |
| **输出** | `position: (3,)`；`quaternion_wxyz: (4,)`；`bbox_extent: (3,) \| None`（完整边长；关闭时为 `None`） |

**流程**

```
object_name
  → RGB-D 观测
  → 语言分割得 mask（SAM3 或 OwlViT+SAM2）
  → 深度点云（相机系）筛选物体点
  → 统计滤波 + Open3D OBB
  → 相机外参变换到世界系
  → position / quaternion / extent
```

**关键步骤**

1. `obs = env.get_observation()`，取 `robot0_robotview` 的 RGB、Depth、内参、外参。
2. **SAM3 路径**（默认 `use_sam3=True`）：`sam3_seg_fn(rgb, text_prompt=object_name)`，取最高分 `mask`；**OwlViT 路径**：检测 bbox → SAM2 分割 → 框内像素最多的实例 ID。
3. 有效深度 mask：`~isnan(depth)`；全图点云后用实例索引筛点。
4. Open3D：`remove_statistical_outlier` → `get_oriented_bounding_box()`。
5. `obb_tf_world = cam_extrinsic @ obb_tf`；返回世界系中心与朝向。
6. **注意**：对称物体（方块）上 OBB 朝向常不稳定；放置建议用 `sample_grasp_pose` 的 quat 或固定朝下 `(0,0,1,0)`，不要依赖本函数 quat。

---

### 1.2 sample_grasp_pose

```text
position, quaternion_wxyz = sample_grasp_pose(object_name)
```

| | |
| --- | --- |
| **输入** | `object_name: str` |
| **其他参数** | 无 |
| **输出** | `position: (3,)`；`quaternion_wxyz: (4,)` — 建议用于抓取/放置朝向 |

**流程**

```
object_name
  → RGB-D + 分割 mask（同 get_object_pose）
  → Contact-GraspNet 生成候选抓取（相机系）
  → 取最高分候选 + 沿抓取轴平移 0.12m（TCP 对齐）
  → 相机→世界
  → grasp position / quaternion
```

**关键步骤**

1. 与 `get_object_pose` 相同方式得到 `segmentation` 与实例 ID。
2. `grasp_net_plan_fn(depth, intrinsics, segmentation, instance_idx)` → 多个 `(4,4)` 抓取与分数。
3. 最高分位姿再右乘平移 `[0,0,0.12]`（抓取系沿 Z），对齐夹爪 TCP。
4. `grasp_world = cam_extrinsic @ grasp_cam`；返回世界系 `wxyz_xyz` 的位置与四元数。
5. **注意**：应使用本函数返回的 quat；不要用 `get_object_pose` 的 quat 做抓取朝向。

---

### 1.3 goto_pose

```text
None = goto_pose(position, quaternion_wxyz, z_approach=0.0)
```

| | |
| --- | --- |
| **输入** | `position: (3,)`；`quaternion_wxyz: (4,)` |
| **其他参数** | `z_approach: float = 0.0` — 末端坐标系沿 **-Z** 的接近距离（米） |
| **输出** | `None` |

**流程**

```
目标 TCP 位姿
  → 末端系 TCP offset 补偿 → 法兰/手掌目标
  → [若 z_approach≠0] IK → 阻塞运动到接近点
  → IK → 阻塞运动到目标点
```

**关键步骤**

1. `offset_pos = pos + R(quat) @ TCP_OFFSET`（默认 `[0,0,-0.107]`）。
2. `z_approach != 0` 时：先到 `offset_pos + R @ [0,0,-z_approach]`，再落到 `offset_pos`。带 `z_approach` 的一次调用已包含最终到达，无需再对同一目标调一次 `goto_pose`。
3. `ik_solve_fn`（PyRoKi HTTP `:8116`）求 7-DoF；仿真可带 `prev_cfg` 保关节连续；真机有额外 yaw 偏置。
4. `env.move_to_joints_blocking(joints)` 直到误差收敛或步数耗尽。夹爪状态不变。

---

### 1.4 open_gripper / close_gripper

```text
None = open_gripper()
None = close_gripper()
```

| | |
| --- | --- |
| **输入** | 无 |
| **其他参数** | 无（内部固定 `steps=30`） |
| **输出** | `None` |

**流程**

```
设置夹爪开度目标（开=1.0 / 关=0.0）
  → 连续 step 仿真/控制若干步
  → 完成
```

**关键步骤**

1. 调用 `common.open_gripper` / `close_gripper`：`env._set_gripper(1.0|0.0)`。
2. 循环 `env._step_once()` 共 30 步，让夹爪物理到位。
3. 不移动手臂；抓取后抬升 / 放置前下降需另调 `goto_pose`。

---

### 1.5 home_pose（仿真额外暴露）

```text
None = home_pose()
```

仅非真机时挂入 `functions()`。固定 7 关节角 → `move_to_joints_blocking`。

---

## 2. Low-level API（`FrankaControlApiReduced`）

源码：`capx/integrations/franka/control_reduced.py`  
配置中常见：`apis: [FrankaControlApiReduced]`（或 LIBERO 的 `FrankaLiberoApiReduced` 同类原语）

模型需自行编排：观测 → 分割 → 点云/抓取 → IK → 关节运动 → 夹爪。

---

### 2.1 get_observation

```text
obs = get_observation()
```

| | |
| --- | --- |
| **输入** | 无 |
| **输出** | `obs: dict`，核心键见下 |

**关键字段**

- `obs["robot0_robotview"]["images"]["rgb"]` — `(H,W,3)` uint8  
- `obs["robot0_robotview"]["images"]["depth"]` — `(H,W)` float32（API 内已 squeeze）  
- `obs["robot0_robotview"]["intrinsics"]` — `(3,3)`  
- `obs["robot0_robotview"]["pose_mat"]` — `(4,4)` 相机外参  

**流程**：`env.get_observation()` → 规范化 depth 维度 → 返回。

---

### 2.2 segment_sam3_text_prompt

```text
results = segment_sam3_text_prompt(rgb, text_prompt)
```

| | |
| --- | --- |
| **输入** | `rgb: (H,W,3)`；`text_prompt: str` |
| **输出** | `list[dict]`，每项含 `mask (H,W) bool`、`box [x1,y1,x2,y2]`、`score` |

**流程**：RGB + 文本 → SAM3 → 候选 mask 列表（调用方按 score 取最优）。

**关键步骤**：`sam3_seg_fn(rgb, text_prompt=...)`；可选可视化叠加。

---

### 2.3 segment_sam3_point_prompt

```text
results = segment_sam3_point_prompt(rgb, point_coords)
```

| | |
| --- | --- |
| **输入** | `rgb`；`point_coords: (x, y)` 像素坐标 |
| **输出** | 同 SAM3，含 `mask` / `score` |

**流程**：点提示 → SAM3 → masks。常与 `point_prompt_molmo` 串联：语言定点 → 点分割。

---

### 2.4 point_prompt_molmo

```text
points_dict = point_prompt_molmo(image, text_prompt)
```

| | |
| --- | --- |
| **输入** | `image: (H,W,3)`；`text_prompt: str` |
| **输出** | `dict[str, (x,y) \| (None,None)]` — 查询名 → 像素坐标 |

**流程**：Molmo VLM 根据文本在图上打点；失败则为 `(None, None)`。

---

### 2.5 detect_object_owlvit / segment_sam2（`use_sam3=False`）

```text
detections = detect_object_owlvit(rgb, text)
masks = segment_sam2(rgb, box=None)
```

| | |
| --- | --- |
| **OwlViT 输出** | `list[{box, label, score}]` |
| **SAM2 输出** | `list[{mask, score?}]`，可选用检测框作 `box` |

**流程**：开放词汇检测 →（可选）框条件分割。等价于 High-level 非 SAM3 分支的拆分版。

---

### 2.6 get_oriented_bounding_box_from_3d_points

```text
obb = get_oriented_bounding_box_from_3d_points(points)
```

| | |
| --- | --- |
| **输入** | `points: (N,3)` — 通常为**相机系或世界系**点云（与输入一致） |
| **输出** | `dict`：`center (3,)`、`extent (3,)`、`R (3,3)` |

**流程**：点云 → Open3D OBB（统计滤波等在 helper 内）。  
**对应 High-level**：`get_object_pose` 的几何收尾；调用方需自行做相机→世界变换。

---

### 2.7 plan_grasp

```text
grasp_poses, grasp_scores = plan_grasp(depth, intrinsics, segmentation)
```

| | |
| --- | --- |
| **输入** | `depth: (H,W)`；`intrinsics: (3,3)`；`segmentation: (H,W)` 实例图（或二值 mask，内部按 instance=1） |
| **输出** | `grasp_poses: (K,4,4)` **相机系**；`grasp_scores: (K,)` |

**流程**

```
depth + mask
  → Contact-GraspNet
  → 候选齐次变换（已含沿抓取 Z +0.12m）
  → 返回相机系位姿与分数
```

**关键步骤**

1. 规范化 depth / segmentation 维度。
2. `grasp_net_plan_fn(...)`；结果再统一平移 `+0.12` 沿抓取 Z。
3. **不**做世界系变换、**不**再做 TCP offset；调用方：`T_world = pose_mat @ T_cam`，再 `solve_ik`。
4. **对应 High-level**：`sample_grasp_pose` 的核心，但 High-level 会自动变换到世界系。

---

### 2.8 solve_ik

```text
joints = solve_ik(position, quaternion_wxyz)
```

| | |
| --- | --- |
| **输入** | `position: (3,)`；`quaternion_wxyz: (4,)` — 世界系 TCP 目标 |
| **输出** | `joints: (7,)` |

**流程**

```
TCP 位姿 → apply_tcp_offset → PyRoKi IK（可迭代收敛）→ 7 关节角
```

**关键步骤**

1. `offset_pos = apply_tcp_offset(pos, quat, TCP_OFFSET)`。
2. 仿真：`solve_ik_with_convergence`（warm-start `prev_cfg`）；真机：单次求解 + yaw 偏置。
3. **不执行运动**；需再调 `move_to_joints`。接近运动需调用方自己算 `z_approach` 点并求解两次。

---

### 2.9 move_to_joints

```text
None = move_to_joints(joints)
```

| | |
| --- | --- |
| **输入** | `joints: (7,)` |
| **输出** | `None` |

**流程**：`env.move_to_joints_blocking(joints)` — 关节空间阻塞跟踪至收敛。  
**对应 High-level**：`goto_pose` 的执行段。

---

### 2.10 open_gripper / close_gripper（Low-level）

```text
None = open_gripper()
None = close_gripper()
```

与 High-level **同语义、同实现路径**（`common.open/close_gripper`，`steps=30`）。

双臂模式改为：`open/close_gripper_arm0`、`open/close_gripper_arm1`，运动为 `move_to_joints_arm0/arm1/both`，IK 为 `solve_ik_arm0/arm1`。

---

## 3. High-level 伪代码 ↔ Low-level 拼装示例

### 3.1 `get_object_pose` ≈

```python
obs = get_observation()
cam = obs["robot0_robotview"]
rgb, depth, K, T_wc = cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]

results = segment_sam3_text_prompt(rgb, object_name)
best = max(results, key=lambda r: r["score"])
# 深度反投影得到物体点云 points_cam（调用方实现）
obb = get_oriented_bounding_box_from_3d_points(points_cam)
# T_world = T_wc @ SE3(R=obb["R"], t=obb["center"])
# return position, quaternion_wxyz, obb["extent"]
```

### 3.2 `sample_grasp_pose` ≈

```python
obs = get_observation()
cam = obs["robot0_robotview"]
results = segment_sam3_text_prompt(cam["images"]["rgb"], object_name)
mask = max(results, key=lambda r: r["score"])["mask"]
poses, scores = plan_grasp(cam["images"]["depth"], cam["intrinsics"], mask)
T_cam = poses[scores.argmax()]          # (4,4), camera frame
T_world = cam["pose_mat"] @ T_cam
# position, quat_wxyz = decompose(T_world)
```

### 3.3 `goto_pose(..., z_approach=0.1)` ≈

```python
# approach
joints_a = solve_ik(approach_pos, quat_wxyz)
move_to_joints(joints_a)
# final
joints_f = solve_ik(position, quat_wxyz)
move_to_joints(joints_f)
```

其中 `approach_pos` 在末端系沿 -Z 抬高 `z_approach`（与 High-level 一致）；`solve_ik` 内已含 TCP offset。

---

## 4. 常用注意事项

1. **放置朝向**：用 `sample_grasp_pose` 的 quat；`get_object_pose` 的 quat 仅作参考，方块上常不可靠。  
2. **extent**：`return_bbox_extent=True` 时为全边长；半高用 `extent[2]/2`。  
3. **`z_approach`**：末端 -Z，不是无强制世界 +Z；顶视抓取时二者近似。  
4. **Low-level `plan_grasp`**：输出在**相机系**且已含 +0.12m；勿重复平移。  
5. **Low-level `solve_ik`**：只求解不运动；High-level `goto_pose` = IK + 执行（+ 可选两段接近）。  
6. 依赖服务：SAM3 / Contact-GraspNet / PyRoKi（默认 `http://127.0.0.1:8116`）需已启动。
