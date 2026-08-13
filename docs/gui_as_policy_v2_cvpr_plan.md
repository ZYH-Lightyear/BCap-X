# Visual Action Workspace：训练 VLM 通过视觉动作工作区操作机器人（v2，CVPR 计划）

> 取代 v1 笔记（`gui_as_policy_research_note.md`）中模糊的性质清单。本版回答三件事：
> 定位是什么、训什么/优化目标是什么、实现怎么做到 paper-level 最小。

> **实现快照说明（2026-08-13）**：本文是论文研究计划，包含尚未完成的训练与奖励设计；它不
> 充当运行时接口文档。当前 M1.5 实现采用 history-free Main/Imagination ownership、互斥工具面、
> commit-only physics、`2048×1280` Canvas 和 simulation-only Direct Contact Camera。精确契约见
> [`vaw/CURRENT_ARCHITECTURE.md`](../vaw/CURRENT_ARCHITECTURE.md)，运行方法见
> [`vaw/README.md`](../vaw/README.md)。论文 Method 更新时应明确区分通用 VAW 表示与
> Direct Contact Camera 这一仿真实验性观测条件。

## 0. 论文标题、一句话总结与 Main Contributions

**标题（主选）**：

> **VAW: Learning Robot Manipulation through a Visual Action Workspace**

备选：
- *Interface-as-Policy: Training Vision-Language Models to Manipulate Robots through a Visual Action Workspace*
- *Act on the Canvas: Physics-Verified Reinforcement Learning for VLM Robot Agents*

**一句话总结**：

> 我们把机器人操作重铸为对「视觉动作工作区」的界面操作——工具产出的候选动作
> 在工作区中被实例化、预演（preview）、提交（commit）——从而将 VL-Action 从
> 连续控制问题转化为离散、可验证、可自举数据的界面决策问题；奖励无需人工标注：
> 冻结 VLM 的 token 概率给出稠密任务进展（TOPReward），preview–execution 差异
> 惩罚未经验证的提交，用 SFT + 多轮 RL 把 frontier agent 才具备的操作能力压进
> 一个 8B 开源 VLM。

**Main Contributions（paper level）**：

1. **Visual Action Workspace（表示）**：一种介于 VLA 与 Code-as-Policy 之间的
   VL-Action 因子化——持久视觉动作状态 + 冻结工具空间 + 低层控制器。它把
   机器人动作参数化为约 15 个离散、效果可查的界面操作（ground / propose /
   select / nudge / preview / commit），并把本体状态、世界状态、动作候选与预演结果
   整合成 VLM 原生可读的视觉上下文，使模型能通过视觉推理理解“我在哪里、世界现在
   怎样、下一步可能发生什么”，同时保留开放工具组合能力。这里的操作是语义
   workspace op，不是像素点击或网页控件操作。
2. **Verified-Commit 学习信号（优化目标）**：一套不依赖人工标注的奖励设计——
   (a) 稠密任务进展来自冻结 VLM 的 token 概率（TOPReward, 2602.19313，已在
   Qwen3-VL 上验证 VOC 0.87–0.94）；(b) preview–execution 差异作为**不对称
   惩罚**：只惩罚 *unpredicted failure*（preview 判可行、执行却碰撞/跌落/大偏差），
   不奖励"可预测"本身（谱系：MOPO 式模型不确定性惩罚、经典 expectation-based
   execution monitoring）。二者均为真机可算的过程信号。
3. **自举训练配方（训练）**：frontier agent 在同一工作区协议下自动生成教师
   traces（无需遥操数据）→ 成功过滤 SFT 冷启动 → 多轮 RL（GRPO/PPO），
   将操作能力蒸馏并强化进 Qwen3-VL-8B。
4. **结果与分析**：在精度/接触类操作任务上达到或超过 training-free frontier
   agent（VIA 式），每 episode 成本低一个数量级；提出机器人版 Effective
   Operation Usage 与 preview 校准度指标，量化界面、工具与训练各自的贡献。

## 1. 定位：三方格局中的空点

| | 感知/操作工具 | 持久视觉动作状态 | 可训练 |
|---|---|---|---|
| Code-as-Policy (CaP-X, Fu et al.) | 有 | 无 | 否（prompt 工程） |
| VIA (2607.11119) | 无（刻意去掉） | 有（3D 界面 + virtual gripper） | 否（仅 frontier 闭源模型） |
| VLA (SaPaVe / OptimusVLA / π0 …) | 无 | 无（在权重里） | 是（需遥操数据，丢通用性） |
| **本文** | 有 | 有 | **是（8B 开源 VLM，SFT+RL）** |

CVPR 2026 已验证的三条相邻线，交点无人占据：

- **VLA 训练线**（SaPaVe、OptimusVLA、ActiveVLA、ACoT-VLA）：端到端连续动作，
  依赖遥操数据。World-model 分支（DynBridge、AHA-WAM）的公认痛点：
  *imagination 与 control 脱节，视觉连贯但物理不一致*。
- **GUI Agent RL 线**（UI-TARS-2、ARPO、EMPO）：界面操作策略可被多轮 RL 训练
  （UI-TARS-2 结论：长程任务 PPO 稳于 GRPO）。但动作无物理后果。
- **Agentic tool-use 后训练线**（Skill-3D、ReGRPO@ECCV26、GROW、Z-1）：
  SFT+GRPO 可把工具调用策略压进 4–8B 开源 VLM。但任务是 QA/软件，非具身控制。
- **VIA**：training-free，只有 frontier 模型可用（$4–36/episode、40–160 次调用），
  其 future work 明确写了"用界面 Agent 生成 demonstration 训练快策略"——本文直接做这件事。

## 2. 核心 Claim

> VLA 训练 (V,L)→连续动作；本文训练 (V,L)→界面操作。
> 视觉动作工作区（Visual Action Workspace）把机器人操作**离散化、外显化、可验证化**：
> 工具产出候选（mask、grasp、waypoint、轨迹）写入工作区，Agent 在工作区上
> ground / propose / select / nudge / **preview / commit**。这一动作参数化使得：
>
> 1. **数据可自举**：frontier agent（教师）在工作区上自动生成成功 traces，无需遥操；
> 2. **过程奖励免费**：每个界面操作的效果规则可查（grounding IoU、候选可行性、commit 成败）；
> 3. **提交可被验证**：preview（几何 rollout）给出每次 commit 的预期结果，执行回执
>    暴露偏差；未经验证的提交可被惩罚、失败可被归因，且信号真机可算。
>
> 由此，一个 8B 开源 VLM 经 SFT + 多轮 RL 后，在精度/接触任务上达到或超过
> training-free frontier agent，且每 episode 成本低一个数量级。

对三个邻居的一句话回应：

- **vs VLA**：不训低层控制，动作知识因子化为「界面状态 + 工具 + 控制器」，保留 VLM
  通用性；数据来自教师 agent traces，规模化成本远低于遥操。
- **vs VIA**：VIA 的 minimalism 把所有感知压给 frontier 模型（贵、慢、小模型不可用）；
  工具在这里不是 CaP 式的盲抽象，而是**在工作区上被视觉验证的证据来源**。
- **vs GUI-agent RL**：新域（物理后果、3D、preview–commit）；软件 GUI 里不存在
  "预演物理后果再提交"的动作结构，也就不存在 unpredicted-failure 这类惩罚信号。
  本文借用的是视觉状态外显与可引用性，不训练 click/type 等 computer-use 行为：
  workspace 不是待操作的软件环境，而是 VLM 理解具身状态的视觉 context。

## 3. 方法

### 3.1 工作区 = 状态 + 视觉 Context + 语义操作集

- **ActionState**（单一 JSON 可序列化结构）：
  - `objects[]`：id、mask 引用、3D 点/OBB、置信度、observation revision；
  - `candidates[]`：id、类型（grasp / place / waypoint / trajectory）、位姿、可行性标注；
  - `virtual_gripper`：当前目标位姿 + 开合；
  - `receipts[]`：每次 commit 的执行回执（实际位姿、夹爪宽度、接触/成败、前后 mask diff）。
- **画布（Canvas）**：服务端确定性渲染。RGB 主视图 + 一个渲染 3D 视图（点云 + 候选
  gripper mesh + preview 轨迹扫过体积），叠加对象 id / 候选 id 标签。
  VIA 证明截图足够；确定性渲染对训练是优点。Canvas 的作用不是模拟软件 GUI，而是把
  原本分散在 RGB-D、proprio、工具返回值与历史动作中的 context，统一投影到一个
  视觉坐标系中。
- **WebUI（仅作为视觉渲染尝试）**：可在不改变 ActionState、操作协议、冻结工具、
  preview–commit 物理边界和底层控制器的前提下，用 Web 技术把同一视觉 context
  渲染得更明亮、清晰和层次化，例如显式 `inspect` 后的大幅 object crop、独立 OBB /
  affordance dossier，以及 current/proposed 的 preview 对照。它的产物仍是一张供
  VLM 阅读的 observation；Agent 仍输出同一组 structured workspace op，**不点击页面、
  不操作控件，WebUI 不是新的 action space**。
- **可视化尝试的判据**：M1 只比较 `当前 Canvas renderer + structured op` 与
  `Web-rendered visual workspace + structured op`。若 Web 渲染只是更 fancy，却没有
  改善本体状态理解、世界状态维护、Focus 信息增益、selector 绑定或 preview
  反事实判断，就不进入主线。两组必须保持同一 ActionState、文本 summary、模型、
  任务、工具和动作协议，从而只测视觉 context 表示本身。
- **操作集（动作空间，约 12–15 个）**：
  - 认知操作（只改状态与画布）：`ground(text)`、`inspect(id)`、`propose_grasps(obj_id)`、
    `propose_place(obj_id, target)`、`select(cand_id)`、`nudge(cand_id, Δpose)`、
    `rotate(cand_id, axis, deg)`、`view(camera_op)`；
  - 想象操作：`preview(cand_id)` → 几何 rollout（IK 可行性 + cuRobo 轨迹 + 扫过体积
    对点云碰撞 + 预测末端位姿），渲染进画布；
  - 物理操作：`commit()`（唯一改变世界的操作）、`gripper(open|close)`、`done()`。
  - 每个操作返回：更新后的画布图 + 简短结构化回执文本。

### 3.2 快慢闭环（world model 的落地方式）

- **慢系统**：VLM 在工作区上构造候选、preview 想象后果、决定 commit。
  preview 即显式的、物理接地的 world model（几何 rollout，不训练视频模型）。
- **快系统**：commit 后控制器执行；执行回执与 preview 预测比对，偏差写回
  ActionState（discrepancy 事件），既作为下一步决策的输入，也作为奖励信号。

### 3.3 训什么、优化目标是什么

**训**：策略 π_θ(a_t | visual_workspace_t, state_summary_t)，基座 Qwen3-VL-8B
（4B 做 scaling 点）。输出 = 语义 workspace op（离散操作 + 少量连续参数，文本化为
结构化 action）；输入视觉表示可以由当前 Canvas renderer 或 Web renderer 产生，但
不训练像素点击、键盘输入或网页导航。
**不训**：低层控制（控制器）、感知工具（SAM3/GraspNet 冻结）、world model（几何 rollout）。

**Stage 1 — 教师 SFT**：frontier agent（Claude/GPT，经同一操作集）在 LIBERO/robosuite
任务上 rollout，按任务成功 + 回执一致性过滤，得到 traces；SFT 学格式、工具路由、
grounding 引用、commit 时机。（= 直接实现 VIA 的 future work，故事钩子。）

**Stage 2 — 多轮 RL**（GRPO 起步，UI-TARS-2 表明长程下 PPO 更稳，作为备选）：

```text
R(τ) = R_task                       # 任务成功（二值）
     + λ1 · R_progress              # 稠密任务进展：冻结 VLM（Qwen3-VL-32B，非策略本身）
                                    #   按 TOPReward 计算 log p(True|"任务已完成", 轨迹前缀)
                                    #   的增量；零训练、指令敏感、真机可算
     − λ2 · Σ_commit P_viol         # 不对称验证惩罚：仅罚 unpredicted failure ——
                                    #   preview 判可行但执行碰撞/掉落/末端偏差超容差；
                                    #   不奖励"可预测"（防保守 hacking）；接触偏差设容差
     + λ3 · R_state                 # 过程奖励（仅训练期，仿真真值）：grounding IoU、
                                    #   被 commit 候选的有效性、无效操作惩罚
     − λ4 · C(τ)                    # 步数 / 工具调用成本
     + R_format                     # 结构化输出合规
```

奖励设计的三条依据：
- **R_progress**：TOPReward（2602.19313）已验证冻结 VLM token 概率是可靠的
  零训练进展估计（Qwen3-VL-8B 上 VOC 0.87–0.94，指令敏感）；打分模型与被训
  策略分离（用 32B 或独立实例），避免自我奖励退化。
- **P_viol 为什么是惩罚而非正奖励**：把"预测准"当正奖励会被保守策略 hack
  （自由空间移动永远可预测），且惩罚接触动力学的固有偏差会把策略推离
  接触丰富任务。惩罚形态有谱系：MOPO 的模型不确定性惩罚（offline MBRL）、
  经典 expectation-based execution monitoring、SV-VLA 的执行 verifier。
- **preview–commit 的主要价值在机制而非奖励**：discrepancy 写回 ActionState
  触发重观察与失败归因（推理时收益），奖励项只是其副产品。

### 3.4 指标（除成功率外）

- **ETU**（Effective Tool/Operation Usage，移植自 Skill-3D）：被采纳进 committed action
  的操作占比；
- **Preview 校准度**：预测–实际偏差分布随训练的收敛；
- **成本曲线**：成功率 vs（操作数 / token / 美元），对比 training-free frontier（VIA 式）。
- **Focus 信息增益**：同一对象在 Scene Canvas 与显式 `inspect` crop 下，对象属性、
  affordance、OBB/空间关系和下一步操作判断的准确率变化；
- **Selector grounding accuracy**：对象 / candidate / preview badge 是否绑定到正确
  可视实体；
- **Visual-context efficiency**：在相同 structured op 下，比较 renderer 的 wall time、
  视觉 token、状态读取准确率与下一步决策准确率。Web 渲染若只改善观感，却增加视觉
  噪声或让模型把注意力浪费在界面装饰上，不视为净收益。

## 4. 实现计划（paper-level 最小）

复用 Cap-X 已有（勘察确认）：`FrankaLiberoApiReduced`（grounding/SAM3/GraspNet/
点云/IK/夹爪）、cuRobo 规划（已实现未默认暴露）、LIBERO 仿真内 Viser 渲染、
AgentX 循环、`agentx_capx_eval.py` 桥接。

代码在仓库根的 `vaw/` 包（框架已搭建，详见 `vaw/README.md` 的架构与 milestone 表）：

```text
vaw/
  types.py / state.py   # ActionState：objects/candidates/previews/receipts，summary() 进 prompt
  geometry.py           # 投影/反投影/路径插值（纯 numpy，无 capx 依赖）
  render.py             # 确定性画布渲染（PIL）：mask 叠加、候选、preview 路径、腕部小窗
  preview.py            # 几何 rollout：IK 可行性 + 路径点云碰撞（M2 接 cuRobo）
  executor.py           # commit → 回执 + discrepancy + unpredicted_failure 标记
  ops.py / protocol.py  # 12 个操作 = 动作空间 = function-calling 工具定义
  workspace.py          # 门面 + TraceLogger（steps.jsonl + canvas PNG = 训练数据格式）
  agents/teacher.py     # M3：frontier 模型驱动（占位）
  train/                # M4/M5：collect / rewards / rl 接口已定义，实现留空（verl）
  scripts/              # smoke_render（M0 已通过）、scripted_pick（M1 目标）
```

原则：单一 ActionState 数据结构贯穿渲染、操作、奖励；操作即动作空间即日志格式，
教师 trace 与学生 rollout 同一协议，无转换层。确定性 Canvas 是主线；WebUI 只作为
隔离的 visualization ablation，共用 ActionState、state summary 与 structured
workspace op，不引入第二套 GUI action protocol。不训 world model、不引入 skill
library（留第二篇）。

里程碑：
1. workspace + 教师 agent 在 LIBERO 3–5 个任务上跑通（系统正确性）；在此阶段完成
   Canvas 与 Web renderer 可视化尝试的受控比较，重点测本体/世界状态理解、Focus
   信息增益、selector 错绑和 preview 反事实判断，再决定采用哪一种 visual context
   renderer；两者始终使用同一 structured action space；
2. 收集教师 traces（数百条成功轨迹量级，对齐 Skill-3D 的 500 SFT / 1k RL 规模）；
3. SFT → zero-shot 学生基线；
4. 多轮 RL（先 GRPO，监控 reward 方差，不稳则切 PPO）；
5. 对照：CaP-X 裸工具 ReAct / VIA 式无工具界面 / 本方法（training-free 与 trained 各一列），
   消融：去 R_progress、去 P_viol、去过程奖励、去持久状态。

## 5. 风险

- **教师成功率不足** → SFT 数据难产：先在教师侧允许 R_state 提示（训练期特权信息），
  或用 rejection sampling 扩采样；任务选型避开 VIA 已饱和的简单任务，聚焦精度/接触类。
- **多轮 RL 不稳**（ARPO/EMPO 均报告）：任务过滤（只保留 16 采样内至少 1 成功的任务，
  ARPO 做法）、SFT 冷启动必开（Skill-3D 消融证明）、GRPO→PPO 备选。
- **奖励设计的已知陷阱（已规避）**：早期方案曾把 preview–execution 一致性设为
  正向过程奖励，会被保守策略 hack（可预测 ≠ 正确），且惩罚接触动力学固有偏差
  会伤害接触任务——现改为不对称 P_viol 惩罚 + TOPReward 进展奖励。P_viol 的
  容差阈值需按任务类别标定，是一个真实的调参负担。
- **R_progress 依赖打分 VLM 的感知上限**：TOPReward 自述对细粒度空间/小物体
  任务噪声较大——恰是我们的主打任务域；对策：R_progress 只作 shaping
  （权重小于 R_task 与 R_state），并在消融中单独报告其贡献。
- **可视化反客为主**：Web renderer 的卡片、状态区和装饰可能比原始环境证据更显眼，
  让模型把能力花在“读界面”而不是理解机器人身体、场景关系与物理后果上。对策是
  WebUI 只渲染 visual context、不提供可点击控件，并与当前 Canvas 在同一 structured
  action space 下比较；重点报告本体状态读取、世界状态维护、Focus 信息增益和
  preview 反事实判断。无明确净收益则不进入 M3 教师采集。
- **审稿人问"这不就是 VIA+工具+训练"**：回答在 §2 的三条一句话回应 +
  preview–commit 动作结构带来的可验证提交与失败归因（软件 GUI 域不存在）。
- **命名**：坚持 Visual Action Workspace / Visual Context，避免 "GUI-as-Policy"
  或 "computer-use for robotics" 的表述；后者会让审稿人误以为贡献是用 GUI Agent
  点击网页来控制机器人，而不是为 VLM 外显具身 context 与 counterfactual preview。
