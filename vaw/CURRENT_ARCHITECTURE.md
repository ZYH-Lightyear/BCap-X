# VAW 当前架构契约

> **当前实现基线**：Web schema 32 / `vaw-context-v31-separated-control-guides` /
> renderer `context-web-v31-separated-control-guides` / 固定 `2048×1280`。
>
> 本文只描述当前代码，不记录历史方案。运行方法见 [`README.md`](README.md)，研究目标、
> 逐版本假设与真实任务验收见
> [`M1_5_AGENTIC_SYSTEM_COMPLETION.md`](M1_5_AGENTIC_SYSTEM_COMPLETION.md)。

## 1. 系统边界

VAW 是一个 history-free、双 Agent 的 Visual Context Runtime，不是 GUI Agent，也不是
pick/place 状态机。模型不点击 Canvas；每轮输出一个 structured Function call。

```text
LIBERO-PRO current observation
             │
             ▼
     episode-private context
 RGB-D · calibration · masks · plans
             │ trusted compiler
             ▼
 ContextPacket + one 2048×1280 Canvas
             │
       ┌─────┴──────────────┐
       ▼                    ▼
 Main Agent          Imagination Agent
 task semantics       local pose geometry
 perception/seeds     delta/rotate/gizmo
 gripper review       ready/failed
       └───── ActionReview ─┘
                    │
                    ▼
            Main commit/reject
                    │
            commit is only physics
                    │
                    ▼
          fresh real observation
```

系统不回放 Function transcript、旧 Canvas 或 reasoning history。Main 只有一条 overwrite-only
`Main Working Focus`；它保存上一轮 Main 自己的短 belief，不是环境真值，也不进入
Imagination。一次 commit 后只保留一条 revision-local `LastPhysicalAction` 及必要的真实
before/current 视觉对照。

## 2. Function ownership

代码注册 15 个 Function，但任一请求只暴露当前所有权允许的一个工具面，不会把 15 个工具同时
交给模型。

### 2.1 普通 Main

```text
detection_and_sam
propose_grasps
locate_point
propose_pose
select
start_imagination
open_gripper
close_gripper
done
```

- 感知 Function 只增加当前 revision 的 evidence，不改变真实世界。
- `select/propose_pose/start_imagination` 创建空间 Imagination，并把控制权交给
  Imagination Agent。
- `open_gripper/close_gripper` 不直接控制夹爪；它们创建 pose-free、gripper-only
  `ActionReview`。

### 2.2 Imagination

```text
delta_move
rotate
show_rotation_gizmo
finish_imagination
```

- `delta_move/rotate` 只编辑当前虚拟 `ActionTarget` 并更新规划与 Canvas。
- `show_rotation_gizmo` 只显示所选 frame/axis 的 `−10°/+10°` 真实夹爪姿态对照，不编辑
  target。
- `finish_imagination(ready)` 创建 `ActionReview`；`failed` 或 turn limit 不创建可提交动作。

### 2.3 Main Review

```text
commit
reject_action
select
propose_pose
done
```

Review 是一次性 offer。Main 下一次成功调用若不是 `commit`，旧 review 会被销毁。
`commit(action_id)` 是唯一物理 Function；空间 review 只执行 arm，gripper-only review 只执行
夹爪。成功返回只证明 controller 调用完成，不证明抓取、释放或任务效果成立。

## 3. Context Builder

### 3.1 Private episode context

以下信息只能存在于 backend、compiler 或 trace：

- RGB-D、相机内外参和 raw mask/cloud；
- grasp/planner source、trajectory、returned joints 的私有缓存；
- Contact Camera 配置和 MuJoCo 临时相机状态；
- environment reward、success 和 privileged object pose。

物理 commit 后统一采集新 observation、提升 revision，并清除旧 region、point、seed、
Imagination、Review 和相关规划缓存。

### 3.2 Agent-visible input

Main 每次请求重新构造为：

```text
Main System Prompt
User Task
Current Policy State (minimal manifest)
optional Main Working Focus
optional latest handoff / review edit summary / LastPhysicalAction
current Context PNG
current ownership-specific Function definitions
```

Imagination 每次请求重新构造为：

```text
Imagination System Prompt
Refinement Goal
current cumulative Edit Summary
current Context PNG
four Imagination Functions
```

两者都没有 transcript history。Manifest 只携带 owner、当前 review action ID 和仍有效的
region/point/seed ID，不含 schema 名、revision、receipt ID 或实现元数据。

## 4. Canvas contract

### 4.1 上层：`OBSERVED NOW · REAL WORLD`

- 干净的当前 agentview；
- 与 agentview 标定一致的 dense RGB-D world surface；
- 当前真实 gripper opening、TCP 和 joints；
- 不叠加 region、seed、紫色 target 或 planner 结论。

### 4.2 下层：当前决策面

- `ACTION SEEDS`：最多五个统一尺度候选；
- `IMAGINATION / ACTION REVIEW`：camera-aligned 全局 Preview + 两张 Direct Contact Camera；
- `GROUNDING`：当前 region/point evidence；
- `POST-COMMIT VERIFY`：真实 before/current 因果对照；
- idle/error/terminal：只显示当前仍有决策价值的真实证据。

紫色几何始终表示未执行目标；当前机器人在 Preview 中只使用白色轮廓。Canvas 不预测物体会被
抓住、随动、释放或进入容器。

### 4.3 Direct Contact Camera 与控制提示

`CONTACT FRONT` 和 `CONTACT SIDE` 是 episode-private MuJoCo 相机的直接 RGB raster：

- 相机在一次 Imagination session 内锁定；
- WORLD +Z 保持竖直，target rotate 不会反向旋转地面；
- 当前场景、白色当前轮廓、青色 previous Preview 和紫色 current Preview 共享同一真实视角；
- 它们是 simulation-only active sensor，不能描述为仅重排原 observation 的普通 Canvas。

每张 Contact View 的控制图例彼此分离：

- 左上 `ROTATE BASE`：固定斜视的三维右手正向控制图例，不是 scene-projected overlay；
- 右上 `MOVE BASE`：根据该 Contact Camera 标定，显示两个最具屏幕可见性的 BASE 正轴；
- `DEPTH` 子卡：单独显示最接近视线方向的第三个 BASE 轴，并用 `IN/OUT`、叉/点表达深度
  正方向；
- `− MOVE = REVERSE`：负向始终与所示正轴相反；
- `show_rotation_gizmo`：在独立侧栏显示一个所选轴的 `−10°/+10°` 结果，不遮挡接触目标。

这一分离是 v31 的策略可见语义变化，因此不能继续沿用 v30 renderer 元数据。

## 5. 不变量

1. 每轮必须且只能调用一个 Function。
2. 只有 `commit` 计为物理动作并刷新真实 observation。
3. evidence/seed/action 引用只在当前 observation revision 有效。
4. Main 不执行毫米级局部编辑；Imagination 不做任务级 commit。
5. Planner returned/checked、GRIP 数值和命令完成都不是 task-effect truth。
6. renderer 失败终止实验，不静默回退其他 Canvas。
7. 新旧策略可见 Canvas 语义必须使用不同 schema/renderer 标识。

## 6. 代码对应关系

```text
vaw/context_runtime/model.py          public semantic state
vaw/context_runtime/private.py        episode-private artifacts
vaw/context_runtime/functions.py      Function semantics
vaw/context_runtime/workspace.py      revision lifecycle + dispatch
vaw/context_runtime/protocol.py       prompts + ownership tool surfaces
vaw/context_runtime/runtime.py        history-free dual-agent loop
vaw/context_runtime/presentation.py   private state → presentation view
vaw/context_runtime/packet.py         ContextPacket compiler
vaw/context_runtime/contact_camera.py direct MuJoCo Contact Cameras
vaw/context_runtime/near_field.py      Contact compositing + control guides
vaw/context_runtime/web_renderer.py    fixed Playwright renderer
vaw/context_runtime/trace.py           audit-only trace
vaw-ui/src/context/                    deterministic Canvas layout
```

## 7. 文档优先级

发生冲突时按以下顺序判断：

1. 当前代码与测试；
2. 本文的当前架构契约；
3. `README.md` 的运行接口；
4. `M1_5_AGENTIC_SYSTEM_COMPLETION.md` 的研究目标和版本记录；
5. 带 `ARCHIVED` 标记的 M1.3/M1.4 文档仅用于历史复现。
