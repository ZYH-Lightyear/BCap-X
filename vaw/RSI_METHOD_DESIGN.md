# RSI Method 高层设计：Verified Evolution（工作名）

> 状态：方法论定形稿（2026-08-20）
> 作用：回答"RSI 方法论定了没有"，并给出论文级 formulation
> 上游：`ICRA_PAPER_PROPOSAL.md` §4.2（RSI 循环）、`M1_6_RSI_READINESS_PLAN.md` §4（工程落地）
> 方法论工具：`docs/paper/SKILL.md`（thesis spine / 8 问质量过滤 / archetype 选择）

---

## 1. 方法论现状：哪些已定、哪一块没定

**已定**（M1.6 §4 + ICRA §4.2，两文档一致）：

- 三层分界：Contract 冻结 / Knowledge 唯一进化面 / Verifier 冻结；
- 进化面：playbook 文本 + 路由规则（外置、可审计、git 可回滚）；
- fitness：分级物理信号（planner 可行性 → TCP 误差 → 相位 fitness → env_success）；
- 循环：RHI 式轨迹局部自比较（与上一代成对比，不搞种群）+ AIDE² 采纳纪律；
- 防过拟合：held-out 冻结、预算对等 TTS 对照、公私分割、评估器冻结。

**此前未定、本文档定下**：

1. 论文级 formulation——概念枢纽（hinge）、方法命名、claim 强度；
2. **模型层（7B / 权重更新）在方法中的位置**——这正是"Harness 和 Model 一起改"的问题。

结论先行：**操作层方法论已定，本文档补齐的是它的理论外壳与两阶段扩展。**
7B 路线方向正确，但它是方法的第二阶段而非本篇的联合优化，理由见 §4。

---

## 2. 概念枢纽（one-sentence surprise）

按 formulation playbook，先过第一问：不提方法名，观察本身能否惊到人。

> 文本域的 harness 自进化正卡在评估器上——LLM-judge 可腐蚀、与生成器同源、
> 预算对等后跑不赢简单采样（Ai2 2607.12227）。而具身操作这个通常被认为
> **更难**做自我改进的领域，恰好免费提供了文本域缺失的那块拼图：
> 一个比生成器更便宜、更客观、且结构上不可腐蚀的分级评估器
> （运动学可行性 → 执行误差 → 环境结果）。

即：**具身不是自我改进的困难场景，而是它的天然温床**。这是 archetype C
（结构张力：进化需要可信评估器 ↔ 可扩展的评估器都可被 hack）叠加 B
（把 harness-evolution 社区的困境与具身验证层级桥接）。

对应 NOTES §6.2 的表：所有转得起来的自进化系统都有一个"比生成器更便宜更客观的
评估器"；我们的差异化是这个评估器来自物理而非另一个 LLM。

### Thesis spine（论文可直接用的 8 句）

1. **Setting**: Agent harnesses increasingly determine what VLMs can do in
   embodied manipulation.
2. **Bottleneck**: However, making the harness improve itself has stalled:
   under matched budgets, harness evolution driven by LLM feedback fails to
   beat simple test-time scaling, and its gains do not transfer.
3. **Diagnosis**: We find this failure stems from the evaluator, not the
   optimizer: text-domain evolution scores candidates with signals that are
   as corruptible as the generator itself.
4. **Hinge**: A physically-verified manipulation harness already emits a
   hierarchy of cheap, objective verdicts (plan feasibility, execution error,
   environment outcome) at every step, which can be reused as an
   incorruptible fitness signal.
5. **Question**: Can a manipulation agent improve its own reasoning
   components using only the verification signals its harness already
   produces, with gains that survive held-out tasks and budget-matched
   search baselines?
6. **Answer**: We propose Verified Evolution, which mutates a phase-indexed
   playbook (delegation routing, anchoring procedures, stopping criteria)
   and accepts a mutation only when frozen physical verifiers report strict
   improvement.
7. **Evidence**: Across held-out objects and layouts, evolved playbooks
   improve success beyond budget-matched parallel sampling, and frozen
   spatial probes confirm the gains are not task memorization.
8. **Implication**: These findings suggest embodied verification is a
   sufficient substrate for harness-level self-improvement, and its verified
   traces form a natural curriculum for the next stage: improving the
   models inside the harness.

第 8 句是给你的 7B 路线留的接口，见 §4。

---

## 3. 方法主干：一个评估器，两个旋钮

统一形式化。Agent 系统是五元组：

```text
A = (M_main, M_imag, K, C, V)
  M_main / M_imag : Main 与 Imagination 的基模
  K (Knowledge)   : 相位索引 playbook + 路由规则（文本）
  C (Contract)    : Function API、Canvas 渲染与图例、事件语义 —— 冻结
  V (Verifier)    : planner / TCP 误差 / evidence 复验 / env_success —— 冻结
```

分级 fitness 对任意配置可测：`F(A) = (相位成功率, env_success, 预算成本)`，
全部来自 V，零人工标注。**方法 = 冻结 V，轮流转动两个旋钮**：

### Stage 1（本篇论文）：text knob——进化 K

```text
K_{i+1} = accept(mutate(K_i | 失败三元组)) iff F 严格提升（train），val 复核
```

即 M1.6 §4.4 的循环。M_main、M_imag 都冻结（同一大模型档）。
产出物有两个：更好的 K，以及**副产品——大量 planner-verified 的
imagination session 与相位标注轨迹**。

**K 的内容空间（禁 offset 不变量，2026-08-20 定形）**：
可进化知识必须是"条件 → 程序"形式，且只属于三个族——

```text
1. 感知策略：在什么条件下自己的空间判断不可靠、该用哪个视图/工具补证据
   （例：斜视角下 XY 深度歧义 → 换顶视图或两点求心）
2. 操作程序：动作的顺序性与验证时机知识
   （例：先对准再下降；接触后复验载荷-目标关系）
3. 路由策略：何时委派 Imagination、何时可自行估计
   （例：载荷遮挡 contact 视图 → 委派；开口完整可见 → 自行求心估 XY）
```

程序只写工具调用与证据间关系（对角/中心/同框/可见性）；
**不得存储绝对量值、物体名、任务 ID**——一切数值在运行时由感知获取。
该规则作为内容契约写进 mutation/著述指令；held-out 评测天然兜底。
进化学到的因此是可泛化的空间推理能力，而非对训练环境的 offset 记忆。

### Stage 2（下一篇 / conclusion roadmap）：weight knob——蒸馏 M_imag 到 7B

```text
M_imag(7B) ← 拒绝采样蒸馏(Stage 1 的 verified imagination sessions)
           planner 验证 = 免费的样本过滤器：只学"通过验证的 edit 序列"
之后可再转回 text knob：针对 7B Imagination 重新进化 K（SIA 式交替）
```

关键设计判断：**两个旋钮共享同一个 F 与同一套防过拟合协议**。
方法的本体不是某个循环实现，而是"物理验证层级作为可复用评估器"这一框架；
text knob 与 weight knob 是它的两次实例化。这让 Stage 2 不是换题目，
而是同一方法的第二次转轮——飞轮叙事（NOTES §6.10）落在结构上而非口号上。

---

## 4. 对"7B Imagination + Harness/权重联合进化"的评估

### 方向为什么是对的（四条结构性理由）

1. **Imagination 是 harness 切出来的"可蒸馏切面"**。它的子问题接口极窄：
   聚焦 canvas 内做厘米级/10° 以内的位姿编辑，每步有 planner 即时判定，
   上下文有界（不背任务全局）。窄接口 + 密集验证反馈 + 有界上下文，
   正是小模型可胜任、也最容易训的问题形态。
2. **稀疏 contract 保证可替换性**。Main↔Imagination 只传
   `ready/failed + reason`，换掉 M_imag 不扰动 Main 的上下文分布——
   RHI 所谓 task-specific contract 的直接红利。
3. **SIA 实证**：联合更新（W+H）全面优于只更新 harness（H-only），
   权重旋钮不是摆设（NOTES §5.1）。
4. **训练数据免费且质量有下界**：每条 verified session 都是
   "带物理验证标签的 edit 序列"，planner 过滤天然构成拒绝采样，
   不需要教师打分。

### 为什么本篇不做联合优化（四条硬约束）

1. **数据还不存在**。至今 0 条自发 imagination trace（M1.6 §1）。
   必须先让 text knob 进化出委派行为，7B 的训练分布才存在。
   这不是取舍而是**依赖关系**：Stage 1 的输出是 Stage 2 的输入。
2. **评审防线会破**。"自改面 = 外置可审计文本"是对 Ai2 式攻击的核心防御；
   加入梯度更新后，预算对等要跨"episode 数 × 梯度步数"两个轴对齐，
   四周窗口内做不干净，反而把已收敛的 claim 拖下水。
3. **弱模型风险未测**。Continual Harness 的教训：弱模型 + 厚 harness
   可能更差（Flash-Lite 3–13% < baseline 20%）。7B 能否用好聚焦 canvas
   是经验问题，必须先测后训。
4. **ICRA 提案 §9 已把 7B 蒸馏列为非目标**，cut line 纪律不宜破。

### 本篇窗口内可以为 Stage 2 做的最小动作（便宜、值得）

- **W2 支线：imagination-replay 探测**。把已有（及 Stage 1 新产生的）
  imagination session 的决策点做成离线重放题：给 7B 级 VLM
  （如 Qwen2.5-VL-7B）看同样的聚焦 canvas，比较它与大模型的 edit 选择
  及 planner 判定结果。零训练、纯推理，产出一个数字：
  7B 在该切面上的 zero-shot harness-benefit。
  不为零 → Stage 2 可行性有脚注级证据，写进 conclusion；
  接近零 → 说明必须蒸馏而非直接换，也是有效信息。
- 所有 Stage 1 的 sweep 落盘时保留完整 imagination session
  （已是现状），显式标注为未来蒸馏语料。

---

## 5. 命名与 claim 强度

**方法名：Verified Evolution**（与第二幕 Verified Imagination 对称，
名字暴露核心操作——每次进化选择都过物理验证；可作名词复用）。
备选：Physically-Verified Harness Evolution（描述性、无对称美感）。

Claim 纪律（对齐 ICRA §3 措辞纪律 + NOTES §0）：

- claim "harness-level self-improvement driven by physical verification"；
- 不 claim 完整 RSI（评估器冻结、改进器本身未被改进）；
- 仅当 held-out 稳定超过预算对等 TTS 时，才在 discussion 里说
  "接近 RSI Level 1 的采纳纪律"（AIDE² 的用语）；
- Stage 2 只出现在 conclusion，一句话："verified traces from Stage 1
  form a distillation curriculum for the models inside the harness."

---

## 6. Evidence 设计镜像 formulation（tension paper 的义务）

张力型论文必须报告张力的**两端**，不只报平均分：

1. **KE-3 扩展为四条线**：训练 fitness / held-out 成功率 / 冻结 probe /
   **预算对等 parallel sampling**。进化线必须在 held-out 上压过采样线，
   否则按 cut line 降级（ICRA §7）。
2. **评估器不可腐蚀性的边界实验**（可选但威力大）：把 fitness 换成
   LLM-judge 跑同样代数，展示进化退化或 hack——直接演示
   "评估器决定上限"，把 Ai2 的批评变成我们的论据。预算不够则降为
   discussion 引用。
3. **模型档位分层**（NOTES §6.7）：至少两档 M_main 报告进化收益，
   预期非单调，如实报。
4. **失败边界**：进化学不会的失败族（如需要新视图才能解的任务）
   如实列出——它们正是"可观察性闭包"的边界，反过来支撑
   "Canvas 冻结为契约"的设计选择。

---

## 7. Idea quality filter 自检（playbook 8 问）

| # | 问题 | 判定 |
|---|---|---|
| 1 | 一句话惊喜（不带方法名） | §2：具身是自我进化的温床而非难点 —— 成立 |
| 2 | 结构重要性 | 改变 harness-evolution 社区对"评估器从哪来"的看法 —— 成立 |
| 3 | 可操作检验 | fitness 全可测；TTS 对照可跑；probe 可视化 —— 成立 |
| 4 | 方法必然性 | 评估器免费 → 复用作 fitness → 只进化文本 —— 链条紧 |
| 5 | 紧凑性 | 一个机制（验证复用）+ 两个支撑（playbook 因子化、采纳纪律） —— 成立 |
| 6 | 验证广度 | 任务族 × 模型档 × 预算轴 × held-out —— 成立 |
| 7 | 边界清晰 | §6.4 失败族 + 弱模型非单调 + 评估器冻结的局限 —— 成立 |
| 8 | 更广相关性 | 任何有确定性 verifier 的 agent 域（编译器、物理仿真、形式验证） —— 成立 |
| — | 最弱一环 | KE-3 实验风险（held-out 无提升），cut line 已备 |

---

## 8. 一页总结

```text
方法名   Verified Evolution
枢纽     具身操作免费提供文本域自进化缺失的不可腐蚀评估器
形式化   A=(M_main, M_imag, K, C, V)；冻结 C/V，轮转两个旋钮
Stage 1  text knob：进化 K（本篇；M1.6 §4 即其工程落地）
Stage 2  weight knob：7B M_imag 拒绝采样蒸馏（下一篇；数据来自 Stage 1）
你的提议 方向对，定位为 Stage 2；本篇做 imagination-replay 探测铺路
防线     held-out + 预算对等 TTS + 公私分割 + 评估器冻结 + 分层报告
```
