# RoboMEx × LIBERO-PRO 评测计划

> 状态：当前 AgentWorld action-chunk 主线的评测 source of truth  
> 架构定义：[RoboMEx 高层设计](robomex_high_level_design.md)

## 1. 评测目标

LIBERO-PRO 用于检验 RoboMEx 的三个核心主张：

1. 最新 RGB-D、robot state、动作回执和候选想象组成的 AgentWorld，是否比原始历史
   更适合 Coding Agent 消费；
2. 多个独立 Coding Agents 生成并渲染 motion candidates，是否比单 Agent更可靠；
3. Planner 每次只给出一个细粒度 ActionIntent，并在真实动作后重新观察，是否能处理
   遮挡、抓空、滑落和抓持姿态变化。

这不是为每个任务编写固定 workflow。任务差异只能进入 LIBERO 环境、自然语言和
Skills，不能进入 AgentWorld、Swarm catalog、Graph compiler 或 physical runtime。

## 2. 非特权边界

Policy 可见：

- 主相机与 wrist RGB-D；
- robot joint/gripper state；
- 由当前 observation revision 产生的 tracks、geometry 和 artifacts；
- 已提交的 action receipts 与短动作历史；
- 候选动作的 2D gripper overlay 和 3D point-cloud trajectory render。

Policy 不可见：

- BDDL predicate；
- simulator object ground-truth pose；
- privileged segmentation/identity；
- task reward 或成功标志。

Reward 仅在 episode 结束后进入 evaluator result，不进入 Planner、Manager、Coding Agent
或 AgentWorld。

## 3. 正式范围

第一阶段评测六个 10-task suites：

- `libero_object_swap`
- `libero_object_task`
- `libero_spatial_swap`
- `libero_spatial_task`
- `libero_goal_swap`
- `libero_goal_task`

六套任务共用同一 profile、Planner–Swarm–Graph 主循环和 physical runtime。

## 4. 任务族与所需能力

### 4.1 Rigid object pick-and-place

覆盖食品、容器、碗、瓶子、盘子、basket、stove 和 rack 等刚体取放。

需要：

- semantic/instance grounding；
- geometry、grasp 与 support/container affordance；
- collision-aware exact joint motion；
- 抓持后的逐动作视觉闭环；
- gripper-only action chunk。

主要观察失败：错误实例、深度/OBB 偏差、抓取点偏前、object-to-TCP 不确定、固定
world-axis placement offset 和执行后自报成功。

### 4.2 Relational grounding

覆盖 `between`、`next to`、`on`、`in`、table center 等关系，尤其是场景中存在多个
外观相同黑碗的任务。

需要：

- 为每个可见实例建立独立证据；
- 用当前关系而不是 detector 顺序选择目标；
- 对冲突 grounding hypotheses 获取新证据；
- 单独报告 target identity accuracy。

### 4.3 Articulated manipulation

覆盖抽屉开启与抽屉内取放。它需要 articulation axis、handle/contact affordance 和
动作后重新观察，不能靠增加 motion candidate 数量自动获得。

该能力在通用 articulation Skills 与 proposal-safe tools 接通前单独标记为 unsupported，
不能在 core 中加入 drawer task 分支。

### 4.4 Contact/control interaction

覆盖 push plate 与 stove/control 操作。它需要短接触动作、接触后重新定位和控制状态
视觉判断。

在 contact/control Skills 完成前使用独立 capability manifest 评测，不让缺失能力拖低
刚体主实验的可解释性。

## 5. 当前通用 Swarm catalog

| Recipe | 用途 | Graph |
|---|---|---|
| `motion_single` | 低风险、低歧义 arm motion | snapshot → one candidate → admission → execute |
| `motion_swarm` | 中高风险或有歧义的 arm motion | snapshot → candidate Arena → admission → execute |
| `gripper` | 单一开/合夹爪效果 | snapshot → Coding Agent → admission → execute |
| `wait` | 有界机械稳定 | snapshot → Coding Agent → admission → execute |

Motion candidates 使用独立 workspace、代码与 lineage，并产生：

1. 主相机/wrist 图像上的实际 IK gripper overlay；
2. 点云中的 grasp frame、robot pose 和完整 trajectory render。

只有通过 revision、frame/TCP、joint limits、workspace 和 CuRobo exact-path collision
admission 的一个 sealed action 才能作用于环境。

## 6. 代表性闭环

### 6.1 Alphabet soup → basket

1. Planner 根据最新图像输出一个细粒度 ActionIntent，例如“移动到罐头上方”。
2. Manager 选择 `motion_single` 或 `motion_swarm`。
3. Coding Agents 按需读取 grounding/grasp/motion Skills，生成代码化候选。
4. Arena 检查候选证据与几何可行性，Runtime admission 唯一动作。
5. 执行动作并强制采集新 RGB-D、robot state、receipt 和动作证据。
6. Planner 再决定下降、闭合夹爪或恢复，不执行预先写好的 pick suffix。

### 6.2 Relational black bowl → plate

Grounding Agents 必须分别描述多个黑碗及其与 plate/ramekin 的关系。歧义未解除时不得
进入 grasp；可通过不同 grounding strategy 或主动观察补证据。

### 6.3 抓持姿态不确定的碗精细放置

每轮只执行一个小幅 alignment ActionIntent。候选 render 同时显示夹爪、碗底和盘子；
动作后重新估计相对误差，再决定继续对齐、下降、释放或撤离。禁止使用跨抓取姿态不变的
固定 world-axis offset。

## 7. Baselines 与消融

Baselines：

- Cap-X/CaP-Agent0；
- single universal Coding Agent；
- 单 Agent + AgentWorld；
- RoboMEx motion swarm；
- oracle control upper bound，仅用于诊断 capability/backend 上限。

核心消融：

- raw observation/history vs AgentWorld role view；
- single Coding Agent vs motion swarm；
- heuristic/static recipe vs VLM Manager；
- eager Skill injection vs on-demand Skill；
- single motion vs multi-candidate imagination；
- small vs large Coding Agent model。

所有比较固定 task seeds、模型版本、tool/model token budget、最大动作数和 wall-time。

## 8. 指标

任务指标：

- environment success；
- Spatial target identity accuracy；
- grasp/placement/articulation/contact capability success；
- perturbation recovery rate。

系统指标：

- physical actions per episode；
- model calls、tokens、wall-time；
- candidate rejection/all-invalid rate；
- stale/frame/TCP/collision admission rejection；
- duplicate physical write、unsafe continuation 和 post-termination command；
- failure attribution 到 perception、candidate synthesis、selection、admission 或 execution。

开发目标：

- Object：task success ≥70%；
- Spatial：target identity ≥90%，task success ≥60%；
- Goal rigid placement ≥60%；
- articulation/contact/control capability 各 ≥50%；
- duplicate writer、shadow physical command、terminated 后命令为零。

正式实验每个 task 至少 20 seeds，报告 bootstrap 95% confidence interval。

## 9. 运行入口

检查 profile：

```bash
python -m robomex inspect-profile --profile libero-pro-local
```

只完成环境、AgentWorld、Skills、catalog 和 backend 装配：

```bash
python -m robomex run \
  --profile libero-pro-local \
  --task libero_object_swap:0 \
  --prepare-only
```

运行完整 episode：

```bash
python -m robomex run \
  --profile libero-pro-local \
  --task libero_object_swap:0
```

每次运行输出到：

```text
outputs/robomex_libero_live/YYYYMMDD_HHMMSS/
```

正式批量评测器尚未完成前，不把单次 live episode 当作成功率证据。

## 10. 完成条件

LIBERO-PRO 阶段完成需要同时满足：

- 六套 suite 使用同一自然语言入口和 action-chunk loop；
- core 中没有 task name、object ID 或固定动作序列分支；
- 每个 motion candidate 有独立代码、证据和双视图 imagination lineage；
- 动作后 observation revision 严格递增；
- privileged evaluator 信息不进入 policy trace；
- collision、stale evidence 和 invalid frame fail closed；
- 所有正式失败均能回放并归因；
- 固定预算、多 seed 和置信区间实验完成。
