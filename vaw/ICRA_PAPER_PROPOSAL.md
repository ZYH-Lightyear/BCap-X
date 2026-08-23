# VAW ICRA Paper Proposal：Verified Imagination + RSI

> 状态：proposal v1（2026-08-18）  
> 目标：ICRA 2027，假设截稿 2026-09-15（以官网为准，剩余约 4 周）  
> 页数预算：6 页正文 + 2 页付费页（预留给实验表和 appendix 指引）  
> 依赖文档：[`AGENTIC_CONTEXT_OS.md`](AGENTIC_CONTEXT_OS.md)、
> [`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md)、
> [`M1_5_2_IMAGINATION_AGENT_CALL.md`](M1_5_2_IMAGINATION_AGENT_CALL.md)（§16–§18）

## 0. 一句话

VLM 的操作空间能力是**接口受限**而非能力缺失：给它一个带物理验证的空间工作区
（harness），这个能力不仅能被**释放**（zero-shot 操作、在 VLA 崩溃的扰动下保持鲁棒），
还能被**测量**（诊断性 probe 集），进而被**自我改进**（同一套验证信号驱动的 RSI 循环，
进化 agent 的推理组件而非固定 offset）。

## 1. Thesis 与三幕故事结构

全文只立一个 thesis。benchmark 和 RSI 不并列为独立 contribution，而是 thesis 的
证据链环节，三幕复用同一套物理验证层级：

```text
第一幕 测量（§3 诊断）
    诊断性 probe 集量化 VLM 在操作相关空间判断上的缺陷
    （公尺度 grounding / 相对位姿 / 接触几何 / 净空 / 反事实位姿比较）
    → 结论：缺陷画像明确且与任务失败相关 → 差距是接口问题

第二幕 干预（§4 Harness）
    Context OS + Verified Imagination 逐项补偿第一幕的缺陷
    → 未训练 VLM 在 LIBERO-PRO 上 zero-shot 操作，
      在 VLA 成功率崩溃的扰动设置下保持稳定

第三幕 改进（§5 RSI 方法）
    harness 的验证信号（planner 可行性 → commit TCP 误差 → env_success）
    构成 fitness，驱动 agent 推理组件的进化
    → 在 held-out 任务 + 冻结 probe 上验证提升是泛化的推理改进，
      不是对 eval 集的过拟合

同一套 verified signals 的三种用途：诊断 → commit 门控 → 进化 fitness
（这是 teaser 图的主线）
```

## 2. 标题候选（工作用）

1. *Verified Imagination: A Self-Improving Spatial Reasoning Harness for
   Zero-Shot Robotic Manipulation*
2. *Interface, Not Capability: Unlocking and Evolving VLM Spatial Reasoning
   for Manipulation*
3. *From Diagnosis to Self-Improvement: A Physically-Verified Workspace for
   VLM Manipulation Agents*

倾向 1：把 method 名词（Verified Imagination）和 RSI 卖点（Self-Improving）都放进
标题，"zero-shot" 交代 setting。

## 3. Contribution 声明（Intro 用，严格三条）

1. **Embodied agent harness（Context OS + Verified Imagination）**：把视觉 context
   从单张 RGB 扩展为带公尺度标注、接触视图、反事实预览和物理验证门控的空间工作区，
   使未训练 VLM 在 LIBERO-PRO 上完成 zero-shot 操作，并在 VLA 崩溃的扰动设置下
   保持成功率稳定。
2. **RSI 循环作为方法**：harness 的物理验证层级构成无需人工标注的 fitness 信号，
   驱动 agent 推理组件（Imagination instruction 模板、Main 路由策略、prompt 片段）
   的进化；在 held-out 任务与冻结 probe 集上验证改进的泛化性。
3. **诊断性 spatial probe 集**：自动生成、带 ground truth 的操作空间判断测试，
   跨主流 VLM 给出缺陷画像，且 probe 分数对端到端成功率有预测效度——为
   "interface-limited" 论断提供测量证据。

注意措辞纪律：

- 不 claim「成功率高于 VLA」，claim「扰动下的鲁棒性差距」；
- 不 claim「测量 VLM 与 VLA 的距离」，claim「测量 VLM 空间判断与操作需求的差距」；
- 不 claim「recursive self-improvement 已实现闭环」，claim「验证信号驱动的
  self-improvement，评估器冻结、自改面受限」（对齐 RSI 文献的分类而非其最强形态）。

## 4. 方法设计

### 4.1 第二幕：Harness（已有，论文中压缩为 1.5 页）

复用现有实现，论文只讲四层：

- **Context Canvas**：当前真实世界 + 公尺度标注 + Action Preview；
- **Verified Imagination**：SubAgent 在聚焦视图中做反事实位姿编辑，每步 edit 必须
  通过 planner 验证（原子事务，失败回滚）；
- **Task Memory / Live References**：跨 revision 的物理 primitive ledger 与
  revision-local 引用（含 M1.5.2 §7.5 的失败可见性）；
- **验证层级**：plan 可行性（每 edit）→ commit TCP 误差（每执行）→ env_success
  （每 episode）。这一小节要作为独立 subsection 写，因为三幕都引用它。

方法叙述纪律（对齐写作 skill）：每个组件的动机必须回指第一幕对应的缺陷类别，
避免写成「对 naive baseline 的增量修补」。

### 4.2 第三幕：RSI 循环（本文新方法核心，2 页）

**进化什么（自改面，刻意受限）**：

```text
可进化 artifacts（全部是外置、可审计、可回滚的文本/配置）
├── Imagination instruction 模板库（按任务相位/几何类别索引）
├── Main 的路由策略描述（何时直接 commit / 何时调用 Imagination / 失败后策略）
└── System Prompt 中的空间推理指引片段

不进化：模型权重、planner、renderer、验证器本身、Function schema
```

**Fitness（无人工标注）**：

```text
分级信号，直接来自 M1_5_2 §18.1 的机器可读标签
├── session 级：refine 成功率、失败类别分布（§7.4 四类枚举）
├── 执行级：commit TCP 误差、每 episode 的 imagination token 开销
└── episode 级：env_success（主信号）
```

**循环结构（generation-based，非在线）**：

```text
1. 用当前 artifact 集在训练任务集上跑 N episodes，落盘标签
2. Mutation proposer（LLM）读取失败 session 的 (instruction, reason, 结果) 三元组，
   提出 artifact 变体（重写模板 / 调整路由规则）
3. 变体在训练任务集上评估，按 fitness 选择
4. 每代结束在【冻结验收集】上测一次：held-out 任务 + 冻结 probe 集
   ——验收集分数只记录、绝不参与选择
```

**防过拟合设计（RSI claim 的生命线）**：

- 训练/held-out 任务划分在第 0 天冻结（见 §5.2），进化过程绝不接触 held-out；
- 冻结 probe 集同时监测「进化是否损害通用空间判断」（防止模板退化成任务特例）；
- 报告进化代数曲线：训练集 fitness 与 held-out 成功率同时上升 → 泛化的推理改进；
  只有训练集上升 → 过拟合，claim 降级（见 §7 cut line）。

**与 automatic prompt optimization（APE/OPRO/DSPy 类）的区分**（Related Work 必写）：

1. fitness 是物理验证信号（planner/TCP/env），不是文本任务的 LLM 评分；
2. 自改面是 embodied agent 的推理组件而非单条 prompt，且受 Context OS 不变量约束；
3. 评估含物理泛化（held-out 物体/布局）而非同分布测试集。

### 4.3 第一幕：诊断 probe 集（测量仪器，1 页）

- 用确定性渲染器自动生成，ground truth 免费：五个 probe 族——公尺度距离估计、
  相对位姿判断、接触几何（夹指-物体关系）、净空/碰撞预判、反事实位姿比较
  （两个候选 pose 哪个可行）；
- 规模「不大」但每族 ≥100 题、双视图（全局 + contact view），难度分层；
- 评测对象是**模型**不是 agent：GPT-5 系、Claude 系、Gemini、Qwen-VL、InternVL
  等 5–6 个 VLM；
- 关键实验：probe 分数与该模型作为 Main 时端到端成功率的跨模型相关性
  （§5.1 KE-1）；
- 论文定位为 diagnostic instrument，不叫 "benchmark contribution"，规避
  「自己出题自己考」的循环论证指控。

## 5. 实验计划

### 5.1 三个 killer experiments（每幕一个）

```text
KE-1（第一幕）probe → 成功率预测效度
    x 轴：各 VLM probe 综合分；y 轴：该 VLM 驱动 VAW 的端到端成功率
    期望：显著正相关 → "缺的就是这些能力" 成立

KE-2（第二幕）扰动鲁棒性差距
    LIBERO(原版) vs LIBERO-PRO(扰动) 上：
    VLA（OpenVLA、π0 级 checkpoint）成功率崩塌曲线 vs VAW 保持平稳
    期望：VLA Δ ≥ 60pp 下降，VAW Δ ≤ 10pp

KE-3（第三幕）进化泛化曲线
    x 轴：进化代数；y 轴：训练任务 fitness + held-out 成功率 + 冻结 probe 分
    期望：三条线同向上升或持平，held-out 提升显著
```

### 5.2 主表与任务划分

- 任务源：LIBERO-PRO 的 object / spatial / goal 三个 suite 的扰动设置；
- 训练/held-out 划分（第 0 天冻结）：进化训练集用 object suite 的一半物体，
  held-out = 另一半物体 + spatial/goal 各抽若干任务（跨 suite 泛化）；
- 每任务 ≥10 seeds；主表报告 mean ± std；
- 行：VLA baselines ×2、CaP-X、VAW(no-RSI)、VAW(RSI 第 k 代)；
- 列：各 suite 原版/扰动成功率 + 平均 token/episode + 平均 wall-clock。

### 5.3 Ablations（利用架构天然可拆的优势）

```text
- Imagination（去掉 → Main 直接 commit planned）
- Contact View / 公尺度标注（退化为裸 RGB canvas）
- 验证门控（Imagination edit 不经 planner 验证）
- Task Memory / 失败可见性（M1.5.2 §7.5 开关）
- RSI（第 0 代 vs 第 k 代，即主表已含）
```

### 5.4 成本诚实披露

- token/episode、VLM 调用次数、wall-clock vs VLA 单次前向；
- harness 使用深度/标定/渲染器而 VLA 只用 RGB —— 主动声明为 system-level
  comparison，并在 limitation 中讨论。

## 6. Claim–Evidence Map

| Claim | Evidence | 状态 |
|---|---|---|
| 空间能力 interface-limited | KE-1 相关性 + KE-2 干预效果 | 需实验 |
| zero-shot 扰动鲁棒性 | KE-2 主表 | 需实验（当前 demo 1/4，先过 M1.5.2） |
| RSI 改进泛化推理 | KE-3 held-out + 冻结 probe | 需实验（风险最高） |
| probe 有预测效度 | KE-1 | 需实验 |
| 各组件必要性 | §5.3 ablations | 需实验 |
| 验证信号无需人工标注 | 结构性事实（M1_5_2 §18.1 标签） | 已支持 |

## 7. 四周时间表（含 cut line）

写作与实验并行，draft 从第 2 周开始，不留到最后一周。

```text
W1（8/18–8/24）双线
    A线：M1.5.2 第一步（失败分类 + 连续失败可见性）→ 复测 t1 → 刷成功率
         目标：object suite 扰动设置 zero-shot ≥ 60%，否则 KE-2 没数字
    B线：probe 生成器（复用渲染器）+ 五族题目定义 + 任务划分冻结

W2（8/25–8/31）
    A线：RSI 循环最小实现（generation 评估脚本 + mutation proposer + 标签聚合）
         起跑第 1–2 代
    B线：probe 跑 5–6 个 VLM；VLA baseline 复现开跑（OpenVLA/π0 on LIBERO-PRO）
    写作：Intro + 方法 §4.1/§4.3 初稿

W3（9/1–9/7）
    进化跑到 3–5 代，KE-3 数据成型；KE-1/KE-2 出图；ablations 排队跑
    写作：方法 §4.2 + 实验节初稿；teaser/pipeline 图定稿

W4（9/8–9/15）
    补测 + 全文打磨 + adversarial self-review（five-dimension checklist）
    冻结实验，只修文字

Cut line（W3 周中检查点，硬性决定）：
    - KE-3 held-out 无提升 → RSI 从 contribution 降为 analysis section，
      标题去掉 Self-Improving，故事回退为 1+2 瘦身版（KE-1 + KE-2 仍完整成立）
    - VLA baseline 复现不出 → 引用 LIBERO-PRO 原文数字 + 自测其中一个
    - probe 只来得及 3 族 → 砍反事实比较和净空，保留与 harness 组件对应的三族
```

## 8. 评审攻击预案

| 攻击 | 预案 |
|---|---|
| "elaborate prompt engineering" | KE-1 缺陷画像 → 组件动机逐一对应；ablation 显示每层贡献 |
| "benchmark 为方法定制" | probe 测模型不测 agent；跨 6 个 VLM 报告；定位为 instrument |
| "RSI = prompt 优化换皮" | 物理 fitness + held-out 物理泛化 + 冻结 probe 三重区分（§4.2） |
| "与 VLA 比较不公平" | 主动框成 system-level comparison + 成本表 + limitation 讨论 |
| "只在仿真" | limitation 明写；harness 输入（RGB-D+标定）真实机器人可得，不依赖仿真特权态 |
| "样本量小" | 每任务 ≥10 seeds、mean±std、任务划分第 0 天冻结并在 appendix 公示 |

## 9. 非目标（本文不做，写进 conclusion 作为延伸）

- 7B 蒸馏 / RL 训练（future work：RSI 进化数据即蒸馏数据源）；
- 模型权重层面的自改进；
- 真实机器人实验（如时间富余可加 demo 图，不作为 claim）；
- 完整 RSI 闭环（评估器自身进化）——本文评估器冻结。

## 10. 与现有 milestone 的关系

- W1-A 线 = M1.5.2 §17 第一步 + 第二步，原计划不变、优先级提前；
- §18.1 机器可读标签是 RSI fitness 的直接前置，随 W1-A 一并落地；
- M1.5.2 §11.1 重命名继续推迟——论文文本用新名词（Imagination Agent Call /
  ActionProposal），代码重命名等冻结期，两者解耦；
- M3/M4（冻结 + 蒸馏）顺延为本文 future work 与下一篇的主体。
