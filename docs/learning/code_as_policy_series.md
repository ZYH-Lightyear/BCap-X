# Code-as-Policy 系列论文梳理：CaP-X、Playful、GaP、ASPIRE

> 整理日期：2026-07-23。本文按当日 arXiv 最新版本整理；这些工作均为 2026 年预印本，后续版本的数字和实验协议可能调整。

## 0. 先说结论

这四篇不是四个相互替代的算法，而是在回答 Code-as-Policy 系统的四个不同问题：


| 工作                 | 它最核心的问题                                  | 一句话答案                                                                             |
| ------------------ | ---------------------------------------- | --------------------------------------------------------------------------------- |
| **CaP-X**          | 怎样公平地衡量并增强“写代码控制机器人”的 Agent？             | 用分层 Benchmark 拆开 API 抽象、交互轮次和视觉 grounding，再以 VDM、自动技能合成、并行推理和 RL 提升 coding agent。 |
| **Playful / RATs** | 没有下游指令时，机器人能否先通过“玩”积累以后有用的技能？            | 用好奇心选择“新颖但可学”的自提任务，经多 Agent 执行、验证和诊断，将成功程序蒸馏成持久代码技能。                              |
| **GaP**            | 面向需要长期重复运行的工业任务，free-form Python 是否足够可靠？ | 把 policy 变成可静态检查的有向计算图，调用模块化技能，并在参数化仿真中反复 rehearsal、定位节点失败和改图。                    |
| **ASPIRE**         | 如何把每次执行失败变成可复用、可跨任务迁移的修复经验？              | 记录 primitive 级多模态 trace，闭环诊断并验证修复，将修复提炼成 skill，再用 evolutionary search 跳出局部修补。     |


可以把它们放到一条系统生命周期上理解：

```text
CaP-X：建立环境、评价坐标系和基础 Agent
    ↓
Playful：任务到来之前，用 self-directed play 预先积累技能
    ↓
ASPIRE：任务执行之中/之后，从失败、修复、验证中持续积累技能
    ↓
GaP：把成熟策略固化为可检查、可重复部署的计算图并优化吞吐
```

这个顺序是便于理解的系统视角，不代表四篇论文声明了严格的线性继承关系。

## 1. Paper 与资源链接


| 简称             | 论文全名                                                                                        | 版本                   | Paper                                                                                 | Project                                                             |
| -------------- | ------------------------------------------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| CaP-X          | *CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation*    | arXiv v2, 2026-07-02 | [arXiv](https://arxiv.org/abs/2603.22435) · [HTML](https://arxiv.org/html/2603.22435) | [CaP-Gym](https://capgym.github.io/)                                |
| Playful / RATs | *Playful Agentic Robot Learning*                                                            | arXiv v1, 2026-06-17 | [arXiv](https://arxiv.org/abs/2606.19419) · [HTML](https://arxiv.org/html/2606.19419) | [Playful RATs](https://playful-rats.github.io/)                     |
| GaP            | *GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness for Variational Automation Tasks* | arXiv v1, 2026-07-06 | [arXiv](https://arxiv.org/abs/2607.05369) · [HTML](https://arxiv.org/html/2607.05369) | [Graph-as-Policy](https://graph-robots.github.io/gap/)              |
| ASPIRE         | *ASPIRE: Agentic /Skills Discovery for Robotics*                                            | arXiv v1, 2026-06-30 | [arXiv](https://arxiv.org/abs/2607.00272) · [HTML](https://arxiv.org/html/2607.00272) | [NVIDIA GEAR ASPIRE](https://research.nvidia.com/labs/gear/aspire/) |


仓库中已有 CaP-X、Playful 和 ASPIRE 的本地 PDF，位于 `docs/papers/code as policy/`；GaP 的整理以 arXiv v1 和官方项目页为准。

## 2. 横向总表


| 维度          | CaP-X                                                                          | Playful / RATs                                    | GaP                                                            | ASPIRE                                                              |
| ----------- | ------------------------------------------------------------------------------ | ------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------- |
| 核心定位        | Benchmark + agent harness + RL 平台                                              | 自主 play-time 技能预学习                                | 工业/商业 Variational Automation 的结构化 policy                       | 由执行失败驱动的持续技能发现                                                      |
| Policy 基本单位 | 可执行 Python 程序，组合 perception/control primitives                                 | Python policy + 可调用代码技能                           | 具有 typed I/O 的有向计算图节点与 data/control edge                       | Python robot program + 可检索的 validated repair skill                  |
| 反馈粒度        | stdout/stderr、环境状态、RGB 或 VDM 文本差分                                              | goal verdict、step verdict、failure diagnosis       | node 前后状态、接触和仿真执行结果                                            | 每个 primitive 的 API I/O、状态、关键帧、overlay、grasp/plan 结果                 |
| 主要学习/搜索机制   | Test-time 多轮自纠；自动 skill synthesis；GRPO/RLVR                                    | 新颖性 + competence frontier 的好奇心选任务；play 后蒸馏        | 参数化仿真并行 rehearsal；修改图拓扑、节点和参数                                  | trace-guided repair + skill admission + program evolutionary search |
| 长期记忆        | 9 个自动归纳的 task-agnostic helper skills                                           | skill library + failure memory + reliability tier | MORSL（初始 51 skills）及优化后的部署图                                    | 经验证的失败签名、适用条件、修复策略/代码草图                                             |
| 是否更新模型权重    | Agent0 否；CaP-RL 是                                                              | 否                                                 | 否                                                              | 否                                                                   |
| 主要 Bench    | CaP-Bench 7-task core；CaP-Gym 187 tasks；LIBERO-PRO、BEHAVIOR、real robot         | LIBERO-PRO、MolmoSpaces、RoboSuite、real robot       | 新建 8 个 VA tasks（4 sim + 4 real）                                | LIBERO-Pro、Robosuite、BEHAVIOR-1K、LIBERO-Pro Long、real robot         |
| 主要外部评价指标    | Zero-shot Pass@1、task success、dense reward、code compilation、navigation success | 下游 task success rate / percentage-point gain      | success rate、completion/cycle time、throughput、sequence success | held-out task success、navigation success、tokens-to-first-success    |
| 最能代表论文的结果   | Agent0 在 7 个核心任务中 4 个达到/超过 human；CaP-RL 仿真均值约 20%→72%                          | LIBERO-PRO 23.2%→43.8%；MolmoSpaces 21.0%→38.0%    | 大幅扰动下 SR 0.93–0.99；爆米花 33%→94% sim、90% real                    | LIBERO-Pro 总体 18%→72%；execution engine + search 消融 14%→62%→72%      |


## 3. CaP-X

### 3.1 核心方法论

CaP-X 把“LLM/VLM 写机器人代码”变成一个可系统实验的研究对象，由四层组成：

1. **CaP-Gym**：以 Gymnasium/REPL 形式连接代码执行器和底层机器人环境。Agent 接收观察、生成 Python，程序可多次调用 perception、geometry、motion/control primitives。
2. **CaP-Bench**：沿三个轴做受控评价：
  - primitive abstraction：人写的 high-level macro vs 原子化 low-level API；
  - temporal interaction：single-turn vs 带执行反馈的 multi-turn；
  - perceptual grounding：状态、原始 RGB、或 VLM 生成的结构化视觉差分文本。
3. **CaP-Agent0**：把 Benchmark 中有效的机制组合起来：multi-turn、Visual Differencing Module（VDM）、自动合成的 task-agnostic skill library、并行/多模型候选代码集成。
4. **CaP-RL**：把生成的程序当作 action，在物理仿真给出的可验证 reward 上用 GRPO 直接 post-train coding LLM。

CaP-Bench 的八个 tier 可概括为：


| Tier | 交互          | Primitive / perception                  | 主要用途                             |
| ---- | ----------- | --------------------------------------- | -------------------------------- |
| S1   | single-turn | high-level + privileged noiseless state | 隔离纯规划能力，作为 reasoning upper bound |
| S2   | single-turn | high-level + noisy perception           | 接近以往高层 CaP 设置                    |
| S3   | single-turn | low-level + API usage examples          | 测低层组合能力及 in-context 示例作用         |
| S4   | single-turn | low-level，仅 signature/docstring         | 最少人类 scaffolding 的严格设置           |
| M1   | multi-turn  | stdout/stderr 与执行 trace                 | 测试调试、自省和恢复                       |
| M2   | multi-turn  | 直接回传 RGB                                | 测原始多模态 grounding                 |
| M3   | multi-turn  | VDM 将视觉变化转成结构化文本                        | 测显式文本 grounding                  |
| M4   | multi-turn  | low-level API + VDM                     | 在低层表达力下用 test-time compute 补可靠性  |


### 3.2 Novelty

- **第一次把 CaP 的“能力”和人写 API scaffolding 系统拆开测**。论文的关键发现不是“LLM 能控制机器人”，而是高成功率有多少来自 high-level primitives。
- **把 abstraction、interaction、grounding 三个变量放在同一 Benchmark 中做 controlled ablation**，而不只在若干演示任务上展示成功案例。
- **用 Agent 自动合成中层技能替代固定人类宏**：从 12 个模型在 7 个任务上的成功 S3 rollout 中抽取重复逻辑，归纳出 9 个 task-agnostic helper。
- **把 RLVR 用在代码生成机器人 Agent 本身**：不是让 LLM 写 reward、再训练另一个 policy，而是用真实物理执行结果更新 coding model。

### 3.3 Benchmark 与实验设置


| 层次               | 内容                                                                                        | 规模/协议                                                          |
| ---------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| CaP-Gym 全量发布     | Robosuite、LIBERO-PRO、BEHAVIOR                                                             | 共 187 tasks：7 + 130 + 50                                       |
| CaP-Bench 核心受控实验 | Cube Lift、Cube Stack、Spill Wipe、Peg Insertion、Cube Re-stack、Two-Arm Lift、Two-Arm Handover | 7 tasks；12 个开源/闭源 LM/VLM；每 task/tier 100 trials                |
| LIBERO-PRO 泛化    | Object、Goal、Spatial；Pos 与 Task perturbation                                               | 与 OpenVLA、π 系列 VLA 比较                                          |
| BEHAVIOR 长程移动操作  | Pick up Radio、Pick up Soda Can                                                            | 每 task 25 trials；分别报 navigation/task success                   |
| CaP-RL           | Cube Lift、Cube Stack、Spill Wipe                                                           | S1 privileged API 上每 task 训练 50 iterations；S2 和 real Franka 评价 |


### 3.4 评价指标

- **Zero-shot Pass@1 task success rate**：核心指标；一条样本生成/执行的程序是否完成任务。
- **Code compilation / execution success rate**：区分语法/API 使用失败与物理任务失败。
- **Average dense environment reward**：观察“未达到最终成功但有多少任务进展”。
- **Navigation success rate**：BEHAVIOR 中是否到达目标 1 m 内。
- **Task completion success rate**：最终是否取到目标物。

### 3.5 主要结果


| 结果                 | 大致表现                                                                                                  |
| ------------------ | ----------------------------------------------------------------------------------------------------- |
| 抽象层级               | API 越高层，12 个模型整体越好；低层 S3/S4 暴露明显的代码正确性和机器人几何推理缺口。                                                     |
| 多轮与视觉              | stdout/stderr 多轮反馈普遍有益；直接塞 RGB（M2）反而比 text-only M1 差；VDM 文本差分（M3/M4）最稳定。                              |
| CaP-Agent0         | 即使只使用 low-level primitives，也在 7 个核心任务中的 4 个达到或超过人类专家程序的成功率。                                           |
| LIBERO-PRO         | CaP-Agent0 六个 split 为 22/18/26/17/12/14%，总体约 **18%**；最好 VLA 总体约 **13%**，OpenVLA 等多项为 0。               |
| BEHAVIOR           | Radio：Nav 80%、Task 56%；Soda Can：Nav 84%、Task 72%。Radio task success 高于 human 36%，Soda 与 human 72% 持平。 |
| CaP-RL 仿真          | Qwen2.5-Coder-7B：Lift 25→80%、Stack 4→44%、Wipe 30→93%，三任务均值约 **20→72%**。                               |
| CaP-RL sim-to-real | Franka 上 Lift 24→84%、Stack 12→76%，接近 human 的 92%/84%。                                                 |


### 3.6 怎么评价

CaP-X 最重要的贡献是**建立研究坐标系**。它让后续论文能够明确地说自己提升了哪一种能力：少人类抽象、多轮恢复、视觉 grounding、技能积累，还是模型训练。

需要注意：全量环境是 187 tasks，但论文最完整、最受控的模型对比集中在 7-task core；human program 是专家经过迭代调试后的参考程序，与 model Zero-shot Pass@1 的计算过程并不完全对称；Agent0 的并行多模型集成也使用了显著 test-time compute。CaP-RL 的强结果目前主要覆盖三个相对基础任务，而且在 privileged S1 上训练。

## 4. Playful Agentic Robot Learning / RATs

### 4.1 核心方法论

Playful 将学习时点从“接到任务以后”前移到“任务到来以前”。RATs（Robotics Agent Teams）在没有外部任务 reward 的 play phase 中反复执行：

```text
观察场景
  → LLM 生成候选练习任务
  → Goldilocks curiosity 选“新颖但可学”的任务
  → Planner 检索技能并分解步骤
  → Policy Writer 写 Code-as-Policy
  → Quality / Plan / Step / Goal Verifier 检查
  → Failure Diagnoser 指导 retry，必要时 SubAgent 单练瓶颈
  → 成功代码抽取为 parameterized helper skill
  → 失败压缩进 failure memory
  → 更新 skill reliability，周期性去重和清理
```

任务选择的内在目标由两部分组成：

- **Object-Skill Novelty**：偏好历史上少尝试的物体 × 技能组合；
- **Competence Frontier**：用技能经验成功率的 Wilson lower bound 估计可学性，避开太容易或几乎不可能的任务。

技能有 `experimental → verified → deprecated` 生命周期；测试时既可作为纯 plug-in context 注入 CaP-Agent0，也可由完整 RATs execution team 使用。

### 4.2 Novelty

- 将 developmental robotics 的 **self-directed play / intrinsic motivation** 带入 Code-as-Policy：机器人主动决定“现在练什么”，而不只是被动解用户给的任务。
- 把开放语言任务生成、可执行代码、step-level verification、失败诊断和持久代码技能连接成闭环。
- 证明 learned skill library 可以作为**可移植的 inference-time 模块**插入其他 CaP Agent，不更新底座模型权重，也能跨 simulator、跨 real setting 改善结果。
- 不只保留成功轨迹，还维护 failure memory 与技能可靠性，减少反复踩相同的坑。

### 4.3 Benchmark 与实验设置


| Benchmark   | 用途                                    | 协议                                                                        |
| ----------- | ------------------------------------- | ------------------------------------------------------------------------- |
| LIBERO-PRO  | play + in-domain downstream           | Object/Goal/Spatial × Pos/Task 共 60 held-out tasks；每 task 10 trials，共 600 |
| MolmoSpaces | play + in-domain downstream           | Open/Close/Pick/Pick-and-Place 各 10 tasks × 10 trials，共 400               |
| RoboSuite   | 不参与 play 的 cross-environment transfer | 7 tasks × 50 trials，共 350                                                 |
| Real robot  | preliminary sim-to-real transfer      | Pick up red cube、Place cube in bowl，各 40 trials                           |


LIBERO-PRO 和 MolmoSpaces 各运行 **50 次 play iteration**，底座使用 `gemini-3.1pro-preview`。

### 4.4 评价指标

- 对外评价几乎统一为 **task success rate**，并报告相对 baseline 的 percentage-point gain。
- 对内任务选择用 **object-skill novelty** 与基于 Wilson bound 的 **competence frontier**；这是 play curriculum 的选择分数，不应与最终 benchmark 指标混为一谈。
- 同时比较两种 evaluation mode：skill plug-in 到 CaP-Agent0，或完整 `RATs Exec.`。

### 4.5 主要结果


| 实验                    | Baseline         | RATs / 加技能 | 变化           |
| --------------------- | ---------------- | ---------- | ------------ |
| LIBERO-PRO in-domain  | CaP-Agent0 23.2% | RATs 43.8% | **+20.6 pp** |
| MolmoSpaces in-domain | CaP-Agent0 21.0% | RATs 38.0% | **+17.0 pp** |
| RoboSuite cross-env   | 40.3%            | 49.1%      | **+8.9 pp**  |
| Real robot            | 30.0%            | 38.8%      | **+8.8 pp**  |


LIBERO-PRO 消融更能说明增益来源：


| Test-time system | Play skills  | 平均成功率     |
| ---------------- | ------------ | --------- |
| CaP-Agent0       | No Play      | 23.2%     |
| CaP-Agent0       | Random Play  | 24.7%     |
| CaP-Agent0       | Curious Play | 32.3%     |
| RATs Exec.       | No Play      | 36.3%     |
| RATs Exec.       | Curious Play | **44.3%** |


因此，主结果不是“只要多采 rollout 就行”：random play 几乎没有帮助；好奇心 curriculum 和更强 test-time execution 各自有效，组合最好。

主表的 43.8% 与消融表的 44.3% 不是计算错误：主评测对每个任务跑 10 次，`RATs Exec.` 消融为控制成本只跑 5 次，因此采样协议不同。

### 4.6 怎么评价

这篇最有价值的点是把 skill acquisition 变成一个**任务到来前的主动过程**，并用消融证明“玩什么”很重要。对希望构建持续运行 Agent 的系统，这比单次 prompt engineering 更具长期意义。

局限也明显：50 次 play iteration 和多 Agent 验证/重试成本不低；最终 task success 同时受 play skill 和 RATs execution harness 影响，必须看消融而不能只看 43.8%；跨环境中也有负迁移，如 Two-Arm Handover 24%→20%，而 Nut Assembly 仍为 0，说明代码技能并非天然可组合、可迁移。

## 5. GaP：Graph-as-Policy

### 5.1 核心方法论

GaP 针对 **Variational Automation（VA）**：工作站、机器人与大致任务类别已知，但物体几何、位置、姿态和指令实例持续变化；目标是长期、重复、可靠执行，而非任意开放世界 generalist robotics。

其流程为：

1. Orchestrator 根据自然语言任务把目标切成 semantic segments。
2. 多个 specialized skill agent 从 **MORSL**（Modular Open Robot Skill Library，初始 51 skills）合成局部 subgraph。
3. Orchestrator 将 subgraph 连接成具有 typed data edge、control branch、retry/recovery path 的完整计算图，并做静态结构检查。
4. 在 NVIDIA Isaac 参数化环境中并行采样不同 task instance 做 rehearsal。
5. 记录每个 node 执行前后状态与 contact/physics 结果，将失败定位到具体 node。
6. Agent 修改 graph topology、替换等价 node、调整 edge 或参数，直至性能 plateau。
7. 优化图交给 edge interpreter，无需 LLM 持续在线即可反复执行。

MORSL 混合了 model-based 与 model-free 模块：SAM/Grounding DINO/Molmo 等 perception，Contact GraspNet/GraspGen/M2T2，cuRobo motion planning，2D/3D 几何工具、ROS translator、verification/control primitives 等。

### 5.2 Novelty

- **把 policy 的输出表示从 free-form script 改成 typed directed execution graph**，把数据依赖、控制分支、并行、恢复和接口约束显式化。
- **多 Agent 分区生成 + 静态验证**：每个 Agent 只处理局部 subgraph，减少一个 LLM 同时维护全局 dataflow、局部逻辑和接口细节的负担。
- **node-localized self-learning**：仿真失败不是只反馈“任务失败”，而是借助节点前后状态定位失败源，再修改图结构/参数。
- 提出并开放 **8 个 Variational Automation Benchmark tasks**，把成功率、周期和吞吐放到同一评价框架，更贴近工业部署。

### 5.3 Benchmark 与实验设置

论文所谓 8 tasks 是按 sim/real 实例分别计数：


| VA family                 | Sim | Real | 变化/任务内容                                                        |
| ------------------------- | --- | ---- | -------------------------------------------------------------- |
| I. Fulfill Grocery Orders | ✓   | ✓    | 单目标取放；XY 20×20 cm、basket swap、item permutation、mixed variation |
| II. Pack Grocery Items    | ✓   | ✓    | 6 次尝试将 6 个物品装入篮子                                               |
| III. Make Popcorn         | ✓   | ✓    | 开炉、抓锅柄、放锅、移锅、关炉的长程流程                                           |
| IV. Insert USB-C Cables   |     | ✓    | UR5 + wrist camera + force feedback；端口位置/角度和插入顺序变化             |
| V. Wash Crates            | ✓   |      | 双 Franka 协作抓、翻转、清洗和放置；姿态扰动、持续吞吐                                |


前两类仿真主表共 **5,500 trials，每个 cell 100 instances**。Baseline 包括 CaP-X、π0.5、MolmoAct2、TipTop，以及用 GaP 先调整 wrist camera 后再交给 VLA 的组合版本。

### 5.4 评价指标

- **Success rate (SR)**：任务/instance 是否完成。
- Pack Grocery 特殊定义：6 次 grasp attempt 后成功放入 basket 的物品数 / 6。
- Cable：既报 **per-insertion SR**，也报整个顺序任务 **sequence success**。
- **Completion / execution / cycle time**：秒。
- **Sustained throughput**：连续运行时 successes per hour。
- 论文形式化目标同时考虑 success 与 `success rate / cycle time`，而不只追求一次成功。

### 5.5 主要结果


| 实验                         | 主要结果                                                                                |
| -------------------------- | ----------------------------------------------------------------------------------- |
| Grocery/Pack simulation    | GaP 在各种 positional/geometry variation 下 SR 约 **0.93–0.99**；VLA 在强扰动下常降至约 0.10–0.26。 |
| GaP + VLA                  | 先用 GaP 的交互感知/视角调整把 VLA 输入拉回训练分布，部分条件下带来 2× 以上提升。                                    |
| Real Grocery Fulfillment   | TipTop 8/25，GaP **25/25**。                                                          |
| Real Grocery Packing       | TipTop 10/30，GaP **28/30**。                                                         |
| Make Popcorn self-learning | 初始约 33%；10 轮 rehearsal 后 **94% sim**，真实 **18/20 = 90%**。                            |
| Cable insertion            | 总体 per-insertion **121/130 = 93.1%**；并报告 ascending/descending/odd/even 整序列成功与耗时。    |
| Crate Washing              | GaP **143/150 = 95.3%**，专家手写图 **148/150 = 98.7%**；吞吐 18.33 vs 19.33 success/hour。   |
| Ablation                   | graphless raw Python 和把 specialized agents 压成单 Agent 的版本均降到 0，主要死于接口/结构验证失败。        |


### 5.6 怎么评价

GaP 的核心不是“再加几个 Agent”，而是**约束 Agent 的产物形态**。可检查图把传统 ROS/TAMP 的工程可靠性与 foundation-model 的组合能力连接起来，这对需要重复部署和故障定位的系统很实用。

但它不是通用机器人已经被解决：VA 假设已知工作站、robot/sensor configuration、对象集合或几何模型和有界 variation；Bench 主要仍是 quasi-static manipulation，仅 cable 强依赖力反馈。Grocery I/II 的初始图已经很强，未运行 self-learning，因而“self-learning”最直接的证据主要来自 Make Popcorn。Crate Washing 虽接近专家成功率，但约 179 s 的 cycle time 和 18 success/hour 仍显示真实工业吞吐有明显优化空间。

## 6. ASPIRE

### 6.1 核心方法论

ASPIRE 的核心循环是：

```text
程序执行
  → 每个 primitive 生成细粒度 multimodal trace
  → Actor 定位失败 primitive 和根因
  → 写 repair code
  → 在 debug configurations 上重执行验证
  → Coordinator 审计可迁移性与 API 合规性
  → validated repair 进入共享 skill library
  → 后续任务检索复用
  → 难题用 evolutionary search 扩展候选策略
```

系统有三个组件：

1. **Robot Execution Engine**：记录 perception/planning/control 调用的 API、输入输出、返回状态、RGB 前后关键帧、overlay、grasp candidate、object pose、motion-planning result。它保留 primitive 周围的关键证据，而非把整段视频全部塞给 Agent。
2. **Continually Expanding Skill Library**：存储的不是完整 task script，而是 `failure signature + when-to-apply guard + validated repair + optional code sketch + provenance`。知识类型可涵盖定位、感知 prompt、grasp constraint、导航恢复、运动 primitive、scene reasoning 和 debug workflow。
3. **Evolutionary Search over Programs**：以当前最好程序和失败 trace 为 parent，生成多样化候选、执行评价、保留强候选继续迭代，避免单轨 debug 困在局部修补循环。

架构采用 coordinator–actor：coordinator 管共享 library 和 skill admission，actor 独立写、跑、诊断和修复；actor 不共享完整对话/rollout，只通过压缩后的 validated skills 传递经验。

### 6.2 Novelty

- 将多模态反馈从 scene-level summary 下沉到 **primitive-level causal trace**，让 Agent 能区分“感知成功但导航规划失败”这类具体因果链。
- 将 skill 重新定义为**经过执行验证的 repair knowledge**，而非预写 workflow 或整条成功轨迹。
- 用 skill admission 将“某次 patch 有效”和“可跨 task 复用”分开：Coordinator 只接纳通过 debug validation、API policy 与可迁移性审计的修复。
- 展示 repair knowledge 的三种迁移：短任务→长任务、sim→real、Franka→不同 embodiment/API；迁移的是 in-context 技能知识，不是同一套低层控制权重。

### 6.3 Benchmark 与实验设置


| Benchmark        | 学习/评价 split                          | 任务与协议                                                                            |
| ---------------- | ------------------------------------ | -------------------------------------------------------------------------------- |
| LIBERO-Pro       | learn seeds 51–65；eval seeds 1–50    | Object/Goal/Spatial × Pos/Task；每 suite/split 10 tasks，每 task 50 held-out seeds   |
| Robosuite        | learn seeds 101–125；eval seeds 1–100 | 7 个单臂/双臂 contact-rich tasks；每 task 100 trials                                    |
| BEHAVIOR-1K      | learn seeds 26–35；eval seeds 1–25    | Soda Can 与 Radio 长程移动操作；incremental block execution                              |
| LIBERO-Pro Long  | skill 来自 LIBERO-90；测试不再 debug        | Pos/Task 各 10 个 held-out long-horizon tasks，zero-shot transfer                   |
| Real YAM station | sim skill 来自 Franka 环境               | Bowl-on-plate、lift soda can、drawer；比较有/无 skill 的调试 token 与 20 次 held-out success |


仿真主实验使用 Claude Code + Claude Opus 4.6、1M context；真实跨 embodiment 实验使用 Codex GPT-5.5 reasoning-xhigh。ASPIRE 每个 LIBERO-Pro/Robosuite task 学出一个程序，再跨 held-out seeds 评价；CaP-Agent0 则可为每个 seed 重新生成并 retry。

### 6.4 评价指标

- **Held-out task success rate**：一个生成程序跨未见 seeds 的稳健性。
- **Navigation success / task completion success**：BEHAVIOR-1K 分别评价到达与最终操作。
- **Zero-shot transfer success**：冻结 LIBERO-90 skill library，在 LIBERO-Pro Long 不额外 debug/retry。
- **Output tokens / total tokens to first success**：真实机器人调试成本。
- **Held-out real success rate**：达到首次成功的程序再做 20 次评价，而不是只报调试中某一次成功。

### 6.5 主要结果


| 实验                        | Baseline                   | ASPIRE                             | 说明                                                  |
| ------------------------- | -------------------------- | ---------------------------------- | --------------------------------------------------- |
| LIBERO-Pro overall        | CaP-Agent0 18%             | **72%**                            | Object 96.5%、Goal 63%、Spatial 55.5%（各自 Pos/Task 平均） |
| Robosuite 7-task mean     | CaP-Agent0 68%             | **81%**                            | Two-Arm Handover **20→92%**；Two-Arm Lift 略降 74→71%  |
| BEHAVIOR Soda Can         | CaP-Agent0 Nav/Task 84/72% | **92/88%**                         | task success +16 pp                                 |
| BEHAVIOR Radio            | CaP-Agent0 Nav/Task 80/56% | **100/88%**                        | task success +32 pp                                 |
| LIBERO-Pro Long zero-shot | CaP-Agent0 3.8%            | **30.5%**                          | Pos 22.6%、Task 38.3%；library 越大总体越好                 |
| Real: bowl on plate       | 20/20                      | 20/20                              | 总 token 8.65M→5.11M                                 |
| Real: lift soda can       | 13/20                      | **19/20**                          | 总 token 61.94M→6.58M                                |
| Real: drawer              | 无有效程序 / 0/20               | **11/20**                          | 总 token budget 334.9M→81.7M                         |
| 组件消融                      | base 14%                   | engine 62% → engine+search **72%** | Execution engine 是最大增益，search 继续解决剩余 hard tasks     |


### 6.6 怎么评价

ASPIRE 对系统设计最有启发的地方是：**长期记忆应该保存“被执行证据验证过的修复”，而不是保存 Agent 的自然语言自信或整段 chat history**。这使 skill library 成为不同 actor、任务和 embodiment 之间的窄接口。

需要谨慎理解其结果：real transfer 不是 simulation policy 直接落地，而是把 simulation-discovered skill 作为 in-context guidance，真实 Agent 仍需调试；搜索循环依赖 frontier LLM、大上下文、很多 rollout 和 token，成本很高；系统仍受预定义 primitive API 限制。官方也明确承认尚缺完全自主真实学习所需的 success detection、安全 reset/monitoring、calibration，以及长期 skill 去重、过时和冲突管理。

## 7. 四篇工作真正的差异

### 7.1 “Skill”在四篇里不是同一个概念


| 工作      | Skill 的来源                                   | Skill 的形态                                          | 主要作用                                 |
| ------- | ------------------------------------------- | -------------------------------------------------- | ------------------------------------ |
| CaP-X   | 从成功 S3 rollout 中归纳重复 helper                 | task-agnostic Python function                      | 补回低层 API 上缺失的中层抽象                    |
| Playful | 自提 play task 的成功执行，或针对瓶颈的 isolated practice | parameterized callable code + reliability metadata | 在下游任务到来前积累可组合能力                      |
| GaP     | 初始人工/开源 MORSL + agent 配置与组合                 | typed graph node/skill declaration                 | 可靠连接 perception、planning、control 并部署 |
| ASPIRE  | 失败诊断→patch→跨 debug config 验证                | failure-triggered repair guidance/code sketch      | 遇到类似失败时少走弯路、跨任务迁移                    |


### 7.2 它们优化的是不同时间尺度


| 时间尺度         | 对应工作       | 典型优化                                        |
| ------------ | ---------- | ------------------------------------------- |
| 单次请求内        | CaP-Agent0 | 多轮观察、debug、候选集成                             |
| 下游任务到来前      | Playful    | 自主 curriculum 与 proactive skill acquisition |
| 多任务持续运行中     | ASPIRE     | 从每次失败积累 validated repair                    |
| 稳定任务类长期部署前/中 | GaP        | sim rehearsal、graph optimization、吞吐与可靠性     |
| 模型生命周期       | CaP-RL     | 用环境 reward 更新 coding model 权重               |


### 7.3 不能直接拿一个成功率排榜

- CaP-X 的核心 7 tasks、Playful 的 60/40 held-out tasks、GaP 的有界工业 workcell、ASPIRE 的 debug/eval split 并非同一难度。
- Playful 报的是 play 前后下游增益；ASPIRE 报一个程序跨 seeds 的成功；CaP-Agent0 通常允许每个 trial 多轮生成；GaP 的图部署后可不依赖在线 LLM。
- GaP 还优化 cycle time/throughput，ASPIRE 还衡量 token-to-first-success，这些都不是单一 task success 能覆盖的。

## 8. 对做系统的直接启发

如果要把四篇的强项合成一个工程系统，比较合理的分层是：

1. **CaP-X 式环境与协议**：底层 primitive 统一 trace schema，严格区分 low/high-level API，保留 task predicate、dense progress 与 held-out seeds。
2. **Playful 式主动技能获取**：空闲期选择 novelty × learnability 的练习目标，但技能必须有 reliability lifecycle。
3. **ASPIRE 式 trace 与 skill admission**：每个 primitive 记录可定位因果的证据；只有通过重执行验证的修复才能进入长期库。
4. **GaP 式结构化部署**：成熟 task program 编译成 typed graph，做静态验证、显式 recovery route 和并行执行；LLM 不必留在每次生产执行的关键路径上。
5. **CaP-RL 式模型改进**：当环境 predicate 足够可靠、数据量足够时，再考虑用 verifiable physical reward 更新 coding model；在此之前优先做好执行闭环与记忆质量。

一句话总结：**CaP-X 告诉我们怎么测，Playful 告诉我们提前学什么，ASPIRE 告诉我们如何从失败中积累，GaP 告诉我们怎样把策略变成可部署的工程结构。**

## 9. 主要来源

- Fu et al., [CaP-X paper](https://arxiv.org/abs/2603.22435), [project page](https://capgym.github.io/).
- Zhang et al., [Playful Agentic Robot Learning paper](https://arxiv.org/abs/2606.19419), [project page](https://playful-rats.github.io/).
- Chen et al., [GaP paper](https://arxiv.org/abs/2607.05369), [project page](https://graph-robots.github.io/gap/).
- Lu et al., [ASPIRE paper](https://arxiv.org/abs/2607.00272), [project page](https://research.nvidia.com/labs/gear/aspire/).

