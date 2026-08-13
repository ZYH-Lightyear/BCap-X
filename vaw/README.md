# VAW — Visual Action Workspace

`vaw` 是面向 LIBERO-PRO 的双 Agent Visual Context Runtime。它把当前真实观测与动作想象
编译为固定 `2048×1280` Canvas，但模型不点击页面：动作通道始终是 structured Function
call。

当前架构不再回放 Function history，也不维护 Persistent Waypoint。Main Agent 负责语义决策
和物理提交；Imagination Agent 在独立、无物理副作用的会话中连续检查和微调一个
`ActionTarget`。

当前实现契约见 [`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md)；权威目标、完成定义和
逐版本验收路线见 [`M1_5_AGENTIC_SYSTEM_COMPLETION.md`](M1_5_AGENTIC_SYSTEM_COMPLETION.md)。
双 Agent 基线见
[`M1_4_2_DUAL_AGENT_RUNTIME.md`](M1_4_2_DUAL_AGENT_RUNTIME.md)；早期 20-turn 抓取审计见
[`M1_4_3_GRASP_CONTEXT_OPTIMIZATION.md`](M1_4_3_GRASP_CONTEXT_OPTIMIZATION.md)，二者只作为
历史失败证据，不再定义当前完成标准。

## Runtime

```text
CURRENT OBSERVED
      │
      ▼
Main Agent ── perception / ActionSeed ──► Imagination Agent
   │                 │                    │
   │                 └─ open / close ──► gripper-only ActionReview
   ▲                                      │
   │                         delta / rotate
   │                                      │
   └──────── ActionReview / failed ────────────────┘
   │
   └── Main reviews ── commit(ActionReview) ──► physical world ──► fresh observation
```

每次 provider 请求都从零构造。Main 只看到 task、当前 Canvas、minimal policy state、最近一次
handoff，以及当前 observation revision 的一条 overwrite-only `LastPhysicalAction`；大幅
previous/current 对照只在 commit 后出现一次。Imagination 只看到 task、
refinement goal、当前 `ActionTarget` 和当前 Canvas。两边都看不到对方或自己的历史
call/result/rationale。

感知与运动有明确的信息边界。同一 observation revision 中重复相同的
`detection_and_sam` query 会复用已有 region，不会制造新的世界状态；已有准确 region 时，
`locate_point` 应显式传 `within_region_id`。`delta_move` 是每轴最多 3cm 的当前/想象 TCP
局部修正；远处语义目标应先定位 region 和 point，再通过 `propose_pose` 进入局部 Preview。

## Function space

Main Agent（普通决策面）：

```text
detection_and_sam(query, within_region_id?)
locate_point(query, within_region_id?)
propose_grasps(region_id) -> seed_ids
propose_pose(point_id, offset_xyz, refinement_goal, quaternion_xyzw?)
select(seed_id, refinement_goal)
start_imagination(refinement_goal)
open_gripper() -> action_id
close_gripper() -> action_id
done(success)
```

Action Review 决策面只在 Imagination 交回一个待审动作时出现：

```text
commit(action_id)
reject_action(action_id)
select / propose_pose
done(success)
```

普通 Main 看不到 `commit`；因此只有在最终 Preview 已进入 Action Review 后才可能执行。审查时
Main 不再二次裁决毫米级位置、方向、两指通道或接触间隙，也不能把同一 target 交回继续微调；
这些局部几何只由 Imagination Agent 判断。Main 只检查任务/动作意图、真实执行前提、planner
可执行性，以及它是否是合理的下一次物理动作。`reject_action` 显式销毁 offer，不刷新真实
observation；`select/propose_pose` 则从不同起点创建新的 Imagination。

Imagination Agent：

```text
delta_move(delta_xyz_m, frame)
rotate(axis, angle_deg, frame)
show_rotation_gizmo(frame, axis)
finish_imagination(status="ready" | "failed")
```

除 `commit` 外，所有 Function 都不会改变真实世界。`select/propose_pose/start_imagination`
创建空间 Imagination；Main 不直接拥有空间 editor。`open_gripper/close_gripper` 是 Main-only，
直接创建纯夹爪 `ActionReview`，不进入 Imagination，也不触发 controller。两类 Review 都必须由
Main 在下一轮显式 `commit(action_id)` 才会产生物理效果；空间与夹爪目标不在同一次 Review 中
隐式组合。只有 Imagination 主动调用 `finish_imagination(status="ready")` 才会产生空间
`ActionReview`。
Main 启动 Imagination 时必须显式提供一句短的 `refinement_goal`，不能把整段 rationale 当成
局部控制目标。Imagination 每次请求只收到当前 Canvas、目标几何和累计 `EditSummary`，不收到
Function transcript。普通 Main 只额外收到一条 overwrite-only `Main Working Focus`：上一轮
Main 自己的一句依据，用于在感知调用后保留“抓取失败，正在重试”这类短期任务关系；它不是
环境真值，也不会进入 Imagination；Action Review 不重新注入旧 Main Working Focus，避免旧的
局部几何判断形成自我强化。默认最多连续想象 6 轮；主动 ready 才产生 `review_required`，达到
上限直接产生 `failed`，不创建 action ID，`turn_limit` 只写 trace。只有 Main 审查宏观条件后
调用 `commit` 才构成批准。`ActionReview` 是一次决策的 offer：Main 的下一次成功调用若不是
`commit`，旧 review 会被明确丢弃，不能在后续回合被误提交。

## Canvas

Web schema 32 / `vaw-context-v31-separated-control-guides` / renderer
`context-web-v31-separated-control-guides`：

- 上层 `OBSERVED NOW · REAL WORLD`：干净 agentview、与 agentview 标定透视一致的稠密
  RGB-D surface 和四行本体状态；
- `ACTION SEEDS`：最多五个候选以固定五列占满下层，统一尺度并完整显示；每张卡直接标出
  精确 `APPROACH BASE [x,y,z]`，使 Main 不必从二维投影猜 side/top-down；
- active target 时下层为 `IMAGINATION · NOT EXECUTED`：左侧保留当前 RGB-D surface 的
  camera-aligned 全局 Preview；右侧同时显示重力稳定、session 锁定的 `CONTACT FRONT` 和
  `CONTACT SIDE`。两张图提供互补的闭合通道、前后和高度证据，但不再宣称是随 target 旋转的
  TOOL 平面；紫色 target 始终表示未执行；
- 真实 LIBERO-PRO 的 Contact View 不再从 agentview/wrist RGB-D 重投影 novel view，而是由
  两张 episode-private、重力稳定且 session 锁定的 MuJoCo Contact Camera 直接光栅化；因此
  背景、物体表面、机器人与遮挡边界具有与 agentview 相同的稠密图像质量，不再产生重投影孔洞。
  这是 simulation-only active sensor，实验中必须与仅重排原观测的 Canvas 版本区分；
- Contact View 左上 `ROTATE BASE` 是固定斜视的三维右手正向控制图例；右上 `MOVE BASE`
  根据真实相机标定只显示两个最具屏幕可见性的 BASE 正轴，最接近视线方向的第三轴独立放入
  `DEPTH` 子卡，并用 `IN/OUT`、叉/点表达深度正方向，避免三轴与标签堆叠；
- `rotate` 仍使用所选 +轴的右手定则；`show_rotation_gizmo(frame, axis)` 不再把三轴旋转环覆盖
  在物体中心，而是在每张 Contact View 的独立右侧栏显示该单轴 `−10° / +10°` 两张真实夹爪
  姿态对照。它只解释符号，不修改 target、不规划、不执行；
- 每次空间编辑后，Contact Focus 同时显示青色 `PREVIOUS PREVIEW` 和紫色 `CURRENT PREVIEW`，
  让 history-free Agent 在一张当前图里比较编辑前后；
- Canvas 不再绘制 Waypoint 文字卡；下层空间全部用于当前 scene raster、紫色 target 与
  Direct Contact Camera，
  避免把待验证的 intent、target 或 source metric 误读为已经成立的世界状态；
- 没有 active target 时，下层明确标成 `CURRENT EVIDENCE · OBSERVED` 或
  `CURRENT GEOMETRY · OBSERVED`；当前真实机器人仅以白色轮廓标记，不再使用蓝色实体 mask，
  也不会被误标成未执行想象；
- Main 审查 Imagination 交回的 ActionReview 时，当前紫色 Preview 始终优先于旧的
  post-commit 页面，确保 commit 审查的是将要执行的 target；最近物理命令与仍缺少的证据继续
  保留在文本 Context，不再占用 Preview 画面；
- commit 后下层切换为同一 Canvas 内的真实因果对照；若存在最近 grasp source，同时显示
  `SOURCE BEFORE → FIXED SOURCE CROP NOW` 与 `CURRENT ACTION AREA`。前两张图使用固定
  像素区域而非 tracking，明确标注机器人遮挡也可能造成变化；大图只显示一次，后续非物理
  grounding 在同一 observation 内保留一个紧凑的
  `LAST COMMIT · CURRENT OBSERVED` 当前画面锚点，并与新 evidence 并列；它在进入新
  Imagination/Review 时隐藏，在下一次 commit 时替换，不声明任务效果；
- close/open/arm commit 分别编译为 `closure/release/arm_motion · UNVERIFIED`。中间夹爪开度
  不再被当作 open/closed 真值；状态只说明仍需物体随动、当前目标关系或当前 RGB 变化证据，
  不注入抓取/放置成功结论；
- BASE/WORLD 坐标提示由 robot-base 几何投影产生，并固定在角落以避免遮挡 target；
- grounding、ActionSeed 与 refinement 信息只占用下层固定 overlay，不改变双层版式；
- 外层 padding、上下层 gap、视觉卡片 gap 与主要 border 统一压缩为 `1–4px`；新增分辨率完全
  分配给真实 RGB-D、Preview 和 Contact View，不增加装饰性留白；
- 紫色几何只存在于下层，并始终表示未执行的预测；
- Imagination 交回 Main 后仍保留精确的初始/最终 target 与累计 base-frame 位移/旋转，避免
  ActionReview 丢失局部编辑方向；
- 无 receipt 页面、旧 Function history、Task 重复文本或 privileged state。

Depth、相机参数、raw mask/cloud、planner trajectory 和环境 success 只存在于 private context
或 trace，不进入策略消息。

## 代码结构

```text
vaw/context_runtime/
  model.py          # evidence、ActionTarget/Seed、ImaginationState、ActionReview
  private.py        # sensor、planner、source provenance、visual edit artifacts
  functions.py      # Function 语义、Imagination-only editor 与唯一 commit
  presentation.py   # semantic/runtime state → policy-visible presentation
  workspace.py      # revision 生命周期与 dispatch
  protocol.py       # 两个独立 System Prompt 和工具视图
  runtime.py        # history-free Main/Imagination ownership loop
  packet.py         # private state → policy-visible ContextPacket
  scene_view.py     # camera-aligned dense RGB-D surface + 当前/目标实体 FK silhouette
  web_renderer.py   # fixed Playwright screenshot
  trace.py          # 完整审计记录；不回灌策略
```

## 构建与运行

```bash
cd /mnt/data/zyh/BCap-X/vaw-ui
npm run build

cd /mnt/data/zyh/BCap-X
source .venv-libero/bin/activate
python -m vaw.scripts.run_context_agent \
  --mode agent \
  --suite libero_object_swap \
  --task-id 0 \
  --seed 1 \
  --model vapi/gpt-5.5 \
  --imagination-model vapi/qwen3.5-plus \
  --protocol text \
  --max-turns 32 \
  --max-imagination-turns 6 \
  --motion-backend curobo \
  --record-video
```

省略 `--imagination-model` 时两个角色复用 `--model`，但 provider 请求和 Context 仍完全
隔离。

## 测试

```bash
source /mnt/data/zyh/BCap-X/.venv-libero/bin/activate
python -m pytest -q \
  tests/test_vaw_context_runtime.py \
  tests/test_vaw_context_agent.py \
  tests/test_vaw_context_packet.py \
  tests/test_vaw_near_field.py \
  tests/test_vaw_gripper_fk.py \
  tests/test_vaw_contact_camera.py \
  tests/test_vaw_semantic_grounding.py
```

当前实现不修改 CaP-X、RoboMEx、`capx_skill_rl` 或已有输出 trace。
