# VAW — Visual Action Workspace

论文计划见 `docs/gui_as_policy_v2_cvpr_plan.md`；**架构图、Agent Runtime 设计与
Milestone 以 `docs/vaw_implementation_plan.md` 为准**。本包实现视觉动作工作区：
Agent 通过 **14 个离散界面操作**（ground / propose / select / nudge / preview /
commit / move_xyz …）操作机器人，工具输出（mask、grasp、waypoint）实例化为工作区里
可引用、可视化的候选对象，只有 `commit` / `move_xyz` / `commit_gripper` 改变物理世界。

## 架构

```text
vaw/
  types.py        # Pose / ObjectEntry / Candidate / PreviewResult / Receipt
  state.py        # ActionState：对象、候选、virtual gripper、视角、focus、回执、事件
  geometry.py     # 投影 / 反投影 / 位姿插值等纯几何工具
  camera.py       # 虚拟相机（az/el/zoom → intrinsics+pose_mat）、预设、视角包络
  cloud.py        # RGB-D → 世界系彩色点云融合 + z-buffer splatting（纯 numpy）
  render.py       # 确定性画布渲染（PIL，无浏览器）：主视图 / DataPanel / Focus / wrist
  preview.py      # 几何 rollout：IK 可行性 + 直线路径点云碰撞（M2 接 cuRobo）
  executor.py     # commit 执行 + 回执 + preview–execution discrepancy
  ops.py          # 操作注册表 = 动作空间 = agent 工具协议（@op 装饰器）
  protocol.py     # 操作集导出为 function-calling 工具定义 / 解析 agent 动作
  workspace.py    # Workspace 门面：绑定 Cap-X ApiBase，step(op) → (canvas, receipt)
                  #   内置 TraceLogger：每步 JSONL + PNG，日志格式即训练数据格式
  agents/         # 由仓库内 agentx/ fork 而来（详见下方"Agent Runtime"）
    contracts.py  #   纯数据：ToolCall / ModelResponse / StepRecord / EpisodeResult
    providers/    #   OpenAI 兼容端点 + <tool_call> 文本协议
    chat.py       #   history：孤儿 tool call 修复 + 画布窗口（prune_images）
    runtime.py    #   VAWRuntime：每轮一个 op、显式 done、预算耗尽强制终止
    teacher.py    #   teacher_provider()：capx proxy :8110，默认 native 协议
    student.py    #   student_provider()：本地 vLLM :8120，默认 text 协议
  train/          # M4/M5：接口已定义，实现留空（SFT 走 LLaMA-Factory，RL 走 verl）
    collect.py    #   教师 rollout 收集 + 成功过滤 → SFT 样本
    rewards.py    #   R_task / R_progress(TOPReward) / P_viol / cost
    rl.py         #   verl 多轮 AgentLoop 接入点
  scripts/
    smoke_render.py   # M0/M1.2 验证：合成 RGB-D 场景跑通 camera/cloud/render/protocol
    smoke_runtime.py  # M0 验证：假 provider + 假机器人跑通 runtime，无需环境与模型
    scripted_pick.py  # M1 验证：LIBERO 上脚本化 ops 序列跑通完整闭环
```

关键约定：

- **一切可引用**：对象 `obj1`、候选 `g1/p2`、回执 `r1` 都有短 id，`ActionState.summary()`
  产出进 prompt 的紧凑 JSON；重数组（mask/点云）只留在内存、只画进画布。
- **物理边界**：只有 `commit` / `move_xyz` / `commit_gripper` 改变世界，其余操作只改
  belief 与画布。注意 `nudge` 只编辑画布上的候选，`move_xyz` 才真的移动机器人。
- **日志即数据**：`TraceLogger` 每步落 `steps.jsonl`（op、args、receipt、state summary）
  + `canvas_XXXX.png`，教师 trace 与学生 rollout 同一格式，SFT/RL 直接消费。
- **对 Cap-X 只有运行时依赖**：`Workspace` 接收任意实现了所需方法的 api 对象
  （`FrankaLiberoApiReduced` 即可），vaw 包本身不 import capx。
- **渲染确定性**：画布是 `(state, obs, cloud)` 的纯函数（`smoke_render` 有逐字节断言）。
  这是画布能当 SFT 输入与 RL 观测的前提，也是不用浏览器的原因。

## Canvas（1024×576，设计依据见实现计划 §1.4）

```text
+--------------------------------------+------------+
| header: rev/gripper/sel/view · task  | DataPanel  |  id ↔ marker 图例，不放数值
|                                      +------------+
|  主视图 768px：物理相机 RGB，或点云    |  Focus     |  焦点物体放大 + 全候选 + 接近轴
|  虚拟视角；稀疏标注 + 左下 gizmo      +------------+
|                                      |  wrist     |  腕相机，爪内有物的直接证据
+--------------------------------------+------------+
```

- **视角是状态**：`view(preset|azimuth_deg|elevation_deg|zoom)` 改 `ActionState.view`。
  默认 `agentview` 用物理相机 RGB（外观最强）；其他角度渲染融合点云（几何准但稀疏），
  header 与角标标明当前来源。方位角限物理机位 ±75°（单视角深度没有背面证据），
  越界裁剪并在回执里说明。缩放是收窄视场而非拉近相机。
- **焦点是状态**：`inspect(object_id)` 一次同时做三件事——focus 视口切到该物体、
  summary 里该物体与其候选展开为全字段（其余压缩，context 有界）、返回几何详情。
  未 `inspect` 时 focus 隐式落在 selected 候选所属物体上并标 `(auto)`。
- **数值不进画布**：位姿/分数/间隙/宽度全在 state summary 文本里，画布只承担空间关系。

## Agent Runtime

`vaw/agents/` 是仓库内 `agentx/`（Qwen-Code headless `AgentCore` 的 Python 移植）
的 fork。保留 provider / chat 两层与循环骨架、幻觉守卫；丢掉 `ToolScheduler`（并行
批次与"一步一个 op"冲突）、`tools/`、`skills.py`、`cli.py`、`trace.py`（Workspace
自带 TraceLogger）。选 fork 而非依赖：要改的四处都在循环内部，加开关会让 coding
agent 与 VAW 互相拖累。

与上游循环的四处差异（理由见 `runtime.py` 模块 docstring 与实现计划 §2）：

1. **一轮一个 op**——多余的调用只回拒绝、不执行（但必须回，否则 call id 成孤儿，
   端点会拒掉整个下一次请求）。
2. **观测是结构性的**——`step` 永远返回新渲染画布，不需要 observe 钩子。
3. **`done` 是显式 op**——纯文本不终止 episode；预算耗尽由 runtime 补
   `done(success=False)`，保证每条 trace 都有终止步。
4. **上下文确定性**——不压缩：最近 K 张画布为图片，其余留文本回执，state 每轮全量
   重发。训练与推理必须看到同一份上下文。

```python
from vaw.agents.runtime import run_episode
from vaw.agents.teacher import teacher_provider

result = run_episode(teacher_provider(), api, "put the red mug on the plate",
                     trace_dir="runs/vaw/ep0")
```

## Milestones

见 `docs/vaw_implementation_plan.md` §3（唯一维护处）。当前进度：M0、M0.5（runtime）、
M1.1（环境接线，`scripted_pick.py` 在 libero_object task 0 上 pick 成功）、
M1.2（Canvas v2 首版：虚拟视角 + `view` / `inspect` + 四区布局）已完成；
下一步 M1.4 真模型首跑，再用失败归因数据迭代布局并冻结界面。

## 运行冒烟测试

```bash
python -m vaw.scripts.smoke_render    # state/render/protocol → vaw/out/smoke/
python -m vaw.scripts.smoke_runtime   # agent 循环         → vaw/out/smoke_runtime/
```
