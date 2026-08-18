# VAW M1.5.1 — Attached Object OBB Preview

> 实现状态（2026-08-15）：核心几何、私有生命周期、三视图 raster、Prompt 边界与离线测试已接入；
> 真实 LIBERO-PRO place trace 验收仍待运行。

## 1. 目标

本增量解决 Place Imagination 的一个结构性缺口：当前 Preview 只移动紫色目标机器人，真实物体
点云保持在当前 observation 中不动，因此 Imagination 无法判断“若执行目标动作，被夹持物体将位于
容器哪里”。Main 如果把“物体进入篮口”作为 refinement 停止条件，SubAgent 会等待一个 Canvas
无法呈现的结果并持续累积平移。

目标是在不引入动力学预测、object tracking 或新的 Agent Function 的前提下，增加一个明确标注的
刚性携带体积假设：

```text
agentview mask + RGB-D ─┐
                       ├─ fused object cloud ─ gravity-stable OBB
wrist mask + RGB-D ────┘                         │
                                                 ▼
                                       bind to grasp TCP
                                                 │
                        target TCP ───────────────┤
                                                 ▼
                              carried-object OBB Preview mask
```

该 OBB 不是公共世界状态、通用物体表示或环境真值。它只是一项 episode-private presenter
artifact，用于回答以下反事实问题：

> 如果物体保持当前相对夹爪的刚性关系，执行目标 Waypoint 后，它大约会占据哪里？

## 2. 设计边界

### 2.1 本轮实现

- 复用当前 `detection_and_sam` 已有的 agentview/wrist mask、RGB-D lift、base-frame 多视角融合和
  `filter_noise`；
- 从融合物体点云计算重力稳定 OBB；
- 在成功的 grasp approach 与后续 `close_gripper` 之间建立私有 attachment hypothesis；
- 在 Imagination 的全局视图及 Contact Front/Side 中，把 OBB 随目标 TCP 刚性变换并渲染成封闭
  半透明 mask；
- 修正 Main/Imagination Prompt，使 placement refinement 判断预测占据体积与容器的几何关系，
  不再要求静止的真实点云随紫色夹爪移动；
- `open_gripper` 后清除 attachment hypothesis，物理释放结果仍由新 observation 验证。

### 2.2 明确不实现

- 不预测抓取成功、滑移、旋转漂移、掉落、反弹或容器接触动力学；
- 不把 OBB 加入 Function list、manifest、Function result 或公共 `ContextState`；
- 不加入永久 object ID、跨任务 Scene Memory、PDDL、物体 tracker 或 privileged simulator pose；
- 不把 carried OBB 加入 CuRobo collision world；本轮只改视觉反事实表达；
- 不用 OBB 替代 region、point、mask 或一般 affordance 表达；它只表示当前抓取来源的携带体积。

## 3. 当前可复用能力

当前 VAW 已在 region 创建时私下构建：

```text
RegionGeometryArtifact
├── agentview_mask
├── wrist_mask | None
├── object_points_base
├── scene_points_base
└── filtered_object_points_base
```

实现已经完成：

1. agentview mask 对应 depth 点提升到 base frame；
2. wrist 通过相同 query 运行 SAM3，并把 mask 对应点提升到 base frame；
3. 两个 segmented cloud 空间一致时合并，否则选择更可信的单视角，避免错误融合；
4. 复用 CaP-X `filter_noise`，失败时安全回退未过滤点云；
5. Direct Contact Camera 已提供 RGB、intrinsics 和 `base_from_camera`，可直接投影三维 cuboid。

因此无需新建点云服务，也无需改变 CaP-X API。需要新增的是 OBB canonicalization、attachment
生命周期和 rasterization。

## 4. 私有数据模型

建议在 episode-private context 增加两个不可序列化 artifact：

```python
@dataclass(frozen=True)
class ObjectVolumeProxy:
    query: str
    source_revision: int
    center_base_xyz: tuple[float, float, float]
    rotation_base_from_obb: tuple[tuple[float, float, float], ...]
    extent_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class AttachmentHypothesis:
    query: str
    origin_action_id: str
    tcp_from_obb: tuple[tuple[float, float, float, float], ...]
    extent_xyz_m: tuple[float, float, float]
```

同时需要一个短暂的 `ObjectProxyCandidate`，把 revision-local region proxy 带过一次成功的对象
接近物理刷新，直到 `close_gripper`。它不依赖接近位姿来自 grasp seed 还是 region 内 point：

```text
ObjectProxyCandidate
├── originating_action_id
├── source_query
├── object OBB in base frame
└── grasp approach outcome
```

这些对象：

- 不出现在 `ContextPacket.summary()`；
- 不向 provider 暴露 center、extent、rotation 或点云；
- 只允许 compiler 读取并生成 policy-visible raster；
- 可在 trace diagnostics 中记录数值，便于离线检查坐标和生命周期。

## 5. OBB 计算

### 5.1 输入

优先使用 `filtered_object_points_base`；若过滤结果为空，回退 `object_points_base`。输入点必须满足：

- shape 为 `N×3`；
- 坐标有限；
- 至少包含足以形成三维体积的有效点；
- agentview/wrist 融合继续沿用当前空间一致性策略，不能无条件拼接两个语义 mask。

### 5.2 重力稳定 OBB

不直接使用自由 6-DoF PCA OBB。部分可见点云、圆柱和近似对称物体会使 PCA 轴随机翻转，导致
相邻运行中的 Preview 方向跳变。

采用 gravity-aligned yaw OBB：

1. base `+Z` 固定为重力方向；
2. 在 XY 平面计算鲁棒二维最小面积矩形；
3. Z 上下界使用去除极端离群点后的范围；
4. 对 XY 轴顺序和符号做确定性 canonicalization；
5. extent 可增加一个小的 presenter-only 保守 margin，补偿深度量化和不可见背面。

对圆柱等旋转对称物体，yaw 本身不具有语义，但投影出的占据体积仍应稳定。对倾斜携带姿态，OBB
随后整体跟随 TCP 旋转，不重新从目标画面估计。

### 5.3 失败回退

- 多视角不可用：使用 agentview 单视角点云；
- OBB 数值退化但有足够点：回退 base-axis-aligned box；
- 点云不足或尺寸异常：不创建 proxy，Canvas 不伪造 carried volume；
- 失败只写 trace diagnostics，不使 `detection_and_sam` 本身失败。

## 6. Attachment 生命周期

### 6.1 创建 proxy candidate

`detection_and_sam` 创建 region 时计算 `ObjectVolumeProxy`。`propose_grasps/select` 保持当前公共
返回值不变；`locate_point(within_region_id) / propose_pose` 通过 point 的父 region 访问同一 proxy。
两条路径都只在成功执行对象接近 Action 后晋升私有候选。

粗 Action 在 Imagination 中被平移或旋转时，物体仍在真实世界中不动，因此此阶段不能把 OBB
附着在紫色夹爪上。抓取 Preview 仍按当前方式展示真实目标物体与虚拟夹爪。

### 6.2 跨 grasp approach 保存

成功执行对象接近的 `commit` 会刷新 revision 并清空 region。此时只保留与该成功 action
关联的一份 `ObjectProxyCandidate`；失败 commit 不晋升候选。已有 attachment 时，目的地 point
Action 不得用容器 proxy 覆盖它。

该候选不是永久 object memory，只允许存活到：

- 随后的 `close_gripper`；
- 新的 grasp action 覆盖它；
- `open_gripper` 或 episode reset 清除它。

### 6.3 绑定到真实 TCP

`close_gripper` 时，以闭合位置的真实 TCP 和候选 OBB 建立相对变换：

```text
T_tcp_obb = inverse(T_base_tcp_close) @ T_base_obb
```

生成 `AttachmentHypothesis`。闭合命令本身不证明抓取成功，因此名称和 Canvas 标签必须始终使用
`HYPOTHESIS/ASSUMPTION`，不能使用 `attached=true`、`grasp verified` 等真值措辞。

### 6.4 后续物理动作

- `delta_move`、arm `commit` 和 perception 不清除 hypothesis；
- Main 仍须从真实视觉或小幅抬升判断物体是否随动；
- 新 grasp approach 成功后覆盖旧候选/hypothesis；
- `open_gripper` 成功后清除 attachment；若 backend 明确返回失败，则保留 hypothesis 并把失败
  交还 Main，避免在真实夹爪仍可能闭合时错误丢失携带体积；
- episode reset 清除全部 proxy。

## 7. Imagination 变换

对于当前目标 TCP：

```text
T_base_obb_preview = T_base_tcp_target @ T_tcp_obb
```

由 `T_base_obb_preview` 和 extent 生成八个 cuboid corners。`delta_move` 与 `rotate` 每次编辑目标
TCP 后都重新计算，OBB 自动累计相同的平移和旋转。

必须测试以下不变量：

- target 等于当前 TCP 时，Preview OBB 与 attachment reference 重合；
- base/tool translation 使用现有动作语义，不在 presenter 中重复转换；
- base/tool rotation 只通过最终 target TCP pose 影响 OBB；
- 公共四元数继续使用 `xyzw`，矩阵变换不经过新的 wxyz 边界；
- TCP offset 只在现有 hand/TCP adapter 中处理一次。

## 8. Canvas 设计

### 8.1 颜色和语义

```text
RGB     current real robot appearance
PURPLE  virtual target robot
AMBER   carried object volume under rigid-attachment assumption
```

OBB 不使用传统稀疏线框。compiler 将八个顶点投影到相机，计算 cuboid silhouette，绘制：

- 约 25–35% 透明度的封闭 amber fill；
- 高对比、不透明的 amber contour；
- 必要时使用 depth-aware edge clipping，但不能让 mask 产生点云式破洞；
- 不显示 center、extent、yaw、pose 或 score 数值。

标签固定为：

```text
CARRIED VOLUME · RIGID ASSUMPTION
SLIP / CONTACT / RELEASE DYNAMICS NOT PREDICTED
```

不得标记为 `OBSERVED OBJECT`、`GRASPED` 或 `WILL LAND HERE`。

### 8.2 Main Canvas

上层 `OBSERVED NOW` 保持纯真实世界，不叠加 OBB hypothesis。

下层 `IMAGINATION` 在以下位置显示目标 carried volume：

- 全局目标机器人视图；
- Contact Front；
- Contact Side。

容器继续来自当前真实 RGB/Contact Camera，不移动。这样 Main 能看到目标占据体积与开口的预计
相对关系，同时仍能区分 observed 和 hypothetical。

### 8.3 Focused Imagination Canvas

Imagination 看到与 Main 相同的目标 OBB mask，但局部视图更大。Contact Camera 的 session lock、
gravity up 和 visibility selection 保持不变。相机 focus 可使用目标 OBB 与目标夹爪联合包围盒，避免
长物体或偏心抓取被裁切。

## 9. Prompt 与委派契约

### 9.1 Main Prompt

增加：

> 琥珀色 CARRIED VOLUME 是刚性附着假设，不是真实 observation。它可以用于判断若物体保持当前
> 相对夹爪关系，目标动作是否会把其占据体积带到容器开口上方；不能证明抓持、滑移、释放或落入。

释放边界改为：

> 释放前不要求物体已经位于容器内部。若当前真实全局视觉支持物体仍被夹持，并且位于容器开口
> 上方、水平投影在安全开口内且下方存在无阻挡下落通道，可以 `open_gripper`；释放后再从新
> observation 验证实际结果。

### 9.2 `refine_action` 委派

无 attachment hypothesis 时，Main 只能要求 Imagination 对齐目标夹爪/TCP 与容器，不能要求它判断
物体随动。

有 hypothesis 时，可以委派：

> 在 RIGID ASSUMPTION 下，使琥珀色 carried volume 的水平占据落在篮口安全范围内、底部高于篮沿，
> 保持夹爪闭合与当前姿态；不要预测释放或落点。

### 9.3 Imagination Prompt

增加：

> 真实物体点云保持固定；琥珀色 volume 才是随目标 TCP 变换的刚性携带假设。只能用它判断目标
> 占据体积与静态场景的几何关系。`ready` 不代表物体确实被抓住，也不代表释放后会进入容器。

## 10. Function 与 Packet 兼容性

Main/Imagination Function 名称、参数和返回值全部不变。尤其不增加：

```text
get_obb
attach_object
preview_drop
verify_grasp
```

`ContextPacket` 不新增数值 OBB 字段。建议直接把 OBB 烘焙进现有 `imagination_scene` 与
`contact_focus` raster；若 Web 层只消费这些 raster，TypeScript schema 无需增加机器人学数据。

由于 policy-visible 图像语义发生变化，trace 版本建议更新为：

```text
context schema: vaw-context-v34-attached-volume
renderer: context-web-v34-attached-volume
```

## 11. 实现步骤

### Stage A — Geometry

- 增加 `ObjectVolumeProxy`、`ObjectProxyCandidate` 和 `AttachmentHypothesis`；
- 实现 deterministic gravity-aligned OBB；
- 在 region geometry 构建后生成 proxy；
- 增加点云不足、退化和单视角回退的 trace diagnostics。

### Stage B — Lifecycle

- 在 grasp seed 或 region 内 point 的成功 `commit` 边界保存候选 proxy；
- 在 `close_gripper` 时绑定真实 TCP；
- 确保 ordinary revision refresh 不误删 hypothesis；
- 在 open、新 grasp、reset 和相关失败路径清理；
- 不改变 evidence/action 的 revision-local 规则。

### Stage C — Presenter

- 实现 OBB corner 生成、相机投影、silhouette rasterization；
- 接入 global Imagination、Contact Front/Side；
- Contact Camera focus 覆盖目标夹爪与 carried volume；
- 更新 renderer/schema 名称，保持固定 `2048×1280` 和 DPR=1。

### Stage D — Prompt

- 修正释放前提；
- 收紧 Main 的 transport/place delegation；
- 在两类 Prompt 中明确 rigid assumption 与未预测边界；
- Function schema 保持不变。

### Stage E — Real validation

- scripted 流程验证 grasp proxy → close → lift → place preview → open 清理；
- 在 `libero_object_swap:0` 跑一次 Main/Imagination episode；
- 人工检查 carried mask 是否随 delta/rotate 正确变换、是否改善篮口对齐判断。

## 12. 测试计划

### Geometry

- agentview-only、wrist-only、空间一致融合和不一致回退；
- 点云乱序下 OBB 输出确定性；
- 圆柱、长盒、近方形和退化平面 fixture；
- base `+Z` 恒为重力轴，yaw canonicalization 不随机翻转；
- OBB extent 覆盖过滤后点云的目标比例且无异常膨胀。

### Lifecycle

- grasp seed 绑定正确 region proxy；
- failed/rejected grasp action 不产生 attachment；
- successful approach 后 candidate 跨 revision 存活；
- close 生成 hypothesis，但不生成 `grasp_verified` 真值；
- direct/committed arm motion 保留 hypothesis；
- open、新 grasp 和 reset 清除或覆盖；
- unrelated region/point 不会错误替换 carried source。

### Transform

- `inverse(T_tcp) @ T_obb` round-trip；
- base/tool delta 与 rotate 后 OBB 和 TCP 使用同一刚体变换；
- `xyzw`、hand/TCP offset 与现有 FK overlay 对齐；
- 当前 pose 与 target pose 相同时无视觉漂移。

### Canvas

- 三个 Preview 视图使用同一个目标 OBB；
- mask 封闭、无点云破洞、轮廓清楚且不过度遮挡；
- `OBSERVED NOW` 不出现 amber hypothesis；
- 无 hypothesis 时完全不显示 carried volume；
- snapshot 仍为 `2048×1280×3`、DPR=1、确定性；
- Packet/message 不泄漏 OBB 数值、raw mask/cloud、depth、calibration 或 privileged pose。

### Agent contract

- Main 不再把“静止真实物体点云进入篮口”作为 Imagination 停止条件；
- Imagination 能根据 amber volume 在有限 edits 内返回 ready/failed；
- ready → commit 后，Main 只根据新真实 observation 决定是否释放；
- `open_gripper` 后必须从真实 observation 验证 placement，不能依据旧 Preview done。

## 13. 难度评估

总体难度：**中等，约 6/10**。它不是一个新的 perception 或 physics 项目，现有实现已覆盖约一半
基础能力。

| 部分 | 难度 | 说明 |
|---|---:|---|
| 多视角 mask/点云融合 | 低 | 已实现，只需复用与补测试 |
| 重力稳定 OBB | 中 | 算法短，但要处理退化、对称与确定性 |
| Attachment 生命周期 | 中高 | 最容易出现跨 revision 误留、错绑和假真值 |
| 刚体变换 | 低到中 | 数学简单，必须严格复用 TCP/xyzw 约定 |
| 三视图 mask 投影 | 中 | Contact Camera 已有标定，主要是 silhouette 与遮挡表现 |
| Prompt/Function 边界 | 中 | 需要防止模型把 hypothesis 当 observation |
| 真实任务验收 | 中高 | 要同时检查假抓、滑移、容器遮挡和清理路径 |

最大风险不是 OBB 计算，而是 attachment hypothesis 被模型误读为“已经抓住”。因此实现优先级必须
是：先建立明确生命周期和视觉语义，再追求 OBB 外观精细度。

## 14. 完成标准

本增量完成需要同时满足：

1. 不改变 Agent Function surface；
2. place Preview 中 carried volume 随目标 TCP 的 delta/rotate 稳定变换；
3. Main/Imagination 都能区分 observed object 与 rigid hypothesis；
4. 无 hypothesis 时不伪造物体运动；
5. open 后不再显示 carried volume；
6. 至少一条真实 LIBERO-PRO trace 中，Imagination 使用目标占据体积对齐篮口并主动停止；
7. Main commit 后依据真实视觉释放，而不是依据 Preview 直接宣称 placement 成功。
