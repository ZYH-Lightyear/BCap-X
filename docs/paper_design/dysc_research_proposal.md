# DySC: 面向具身 Coding Agent 的动态技能组合

## 研究动机

近期 RATs 和 ASPIRE 等工作表明，coding agent 可以通过交互、试错、自我调试或程序进化获得可复用的机器人技能。这类方法很有启发性，但它们大多沿用了一个以技能为中心的假设：只要 agent 学会了更好的 grounding、affordance、grasp、place 或 recovery skill，就能在新任务中组合这些技能并实现泛化。

我们认为，这个假设对具身操作任务是不完整的。真实或高保真仿真中的机器人任务并不是静态程序执行问题。即使一个 agent 正确定位了目标，估计出了合理的抓取姿态，执行了 IK 可行的动作，它仍然可能失败：点云可能不完整，物体可能滑动，夹爪接触点可能和估计点不一致，抓住物体后物体相对夹爪的姿态可能改变，释放时物体可能弹开、偏移或仍然卡在夹爪中。

因此，具身 coding agent 的瓶颈不只是“有没有可复用技能”，而是“能否在物理状态不断变化时，逐 turn 地灵活编排技能”。换句话说，泛化不应只发生在单个 skill 层面，也应发生在 skill 的组合策略层面。

我们提出 **DySC: Dynamic Skill Composition**。DySC 的核心观点是：

> 对具身 coding agent 来说，可复用的基本单元不是孤立 skill，而是在物理反馈下逐 turn 编排 skills 的闭环组合策略。

## 核心问题

现有 skill library 常把技能理解为代码片段、自然语言经验、工具函数或可调用模块。例如，一个抓取技能可能包含：

```text
segment object -> estimate point cloud -> compute grasp pose -> move -> close gripper
```

一个放置技能可能包含：

```text
ground target -> compute target center -> compensate grasp offset -> move -> open gripper
```

这些技能对单步 reasoning 很有价值，但在具身场景中，很多失败并不是因为这些步骤完全错误，而是因为它们之间的闭环编排不够灵活。以“抓碗并放到盘子上”为例：抓碗可以通过点云拟合碗沿、用 GraspNet 估计抓取姿态，或者采用接近垂直的抓取姿态；放置时也可以估计抓取点到碗中心或碗底的 offset，再补偿 TCP 目标位置。但实际执行时，offset 往往不够准，碗可能在夹爪中轻微旋转，释放前的空间关系也会和计划时不同。此时真正重要的不是再写一个更复杂的静态 offset 公式，而是能否通过视觉和局部动作逐步修正 object-target relation。

这说明我们需要研究的不是单个 skill 如何写得更好，而是：

```text
observe -> choose skill -> execute bounded action -> verify relation -> update composition
```

这个逐 turn 闭环本身是否可以被学习、复用和迁移。

## 方法概览

DySC 将一个 embodied coding agent 建模为四层结构：

1. **Composable Skill Interface**：将技能改造成可组合、证据感知、不确定性感知的结构化单元。
2. **Agent Role Layer**：每个 agent 是一个 role-conditioned controller，拥有动态 skill access policy、evidence I/O contract 和执行边界。
3. **Multi-Agent Skill Society Graph**：MAS 拓扑本身表示 skill composition policy，节点是角色，边上传 typed evidence、uncertainty、predicate 和 failure。
4. **Reflexive Execution Loop**：把技能执行视为 relation-error reduction 的闭环过程，而不是一次性代码生成。

这四层共同把 flat skill library 转化为可搜索、可进化、可分析的动态技能组合系统。

## 1. Composable Skill Interface

DySC 不把 skill 仅仅表示为一段代码或一段自然语言说明，而是将每个 skill 表示为可被组合策略调度的结构化对象：

```text
Skill = {
  preconditions,
  inputs,
  outputs,
  evidence,
  uncertainty,
  postconditions,
  failure_modes,
  next_skill_affordances
}
```

例如，grounding skill 不应只输出一个 bbox 或 mask。它还应输出候选目标、关系证据、歧义来源、置信度，以及是否需要进一步验证。

affordance skill 不应只输出一个 grasp pose 或 place pose。它还应输出多个假设、对象坐标系假设、接触不确定性、预期执行后证据，以及对后续 motion/place/verify skill 的约束。

motion skill 不应只负责执行某个 pose。它需要声明该动作是否改变物理状态、是否可逆、执行后应检查什么 observation、哪些失败需要 local correction。

verification skill 不应只是最后问 VLM “是否成功”。它需要检查具身任务谓词，例如 object 是否被 gripper 稳定携带，object 是否在 support region 上，drawer 是否打开到足够程度，object 是否已经从 gripper 释放。

这个接口的作用是让每个 skill 都成为 **orchestration-aware skill**。skill 的输出不是终点，而是下一轮组合决策的证据。

## 2. Agent Role Layer

DySC 中的 agent role 不是固定名字的 SubAgent，也不是人工写死的一组工具权限。一个 role-conditioned controller 由五部分定义：

```text
Role = {
  objective,
  skill_access_policy,
  consumes_evidence,
  emits_evidence,
  execution_boundary,
  cost_model
}
```

其中最关键的是 **skill access policy**。它不是静态注册表，而是可初始化、可统计、可进化的绑定关系：

```text
Binding(role, skill) = allowed | preferred | forbidden | conditional
```

例如，PerceptionScout 初始可能偏好 relational grounding 和 pose-alias fallback；AffordanceGeometer 初始可能偏好 OBB geometry、open-bowl rim grasp 和 support-surface placement；PredicateVerifier 初始偏好 held-state check 和 object-target relation check。但这些绑定不应在代码里硬编码。它们应该存放在 role spec / society graph 中，并允许后续 evolution 修改。

这样做有两个目的。第一，它避免所有 agent 共享完整 skill library 导致职责混乱。第二，它使 MAS 可以像 EvoMAS 那样进化：某个 role 可以获得新 skill，失去低效 skill，拆分成两个更专门的 role，或与另一个冗余 role 合并。

需要强调的是，coding agent 的优势仍然被保留。role 不会把控制策略写死成固定模板。一个 MotionExecutor 可以在受 skill guidance 和 evidence contract 约束的前提下，自己写 while loop、局部视觉伺服、重试逻辑和 bounded correction。DySC 约束的是“该 agent 应该在什么证据空间里思考、能访问哪些 skill、需要产出什么 typed evidence”，而不是剥夺 coding agent 自己编写控制逻辑的能力。

因此，第一版实现也不应把 role-skill 绑定固定写入 SubAgent 构造函数。正确形式是：

```text
SocietySpec
  roles
  skill_access_policies
  evidence_contracts
  communication_topology
  budget_policy
```

运行时根据 `SocietySpec` 生成 agent 的 library view 和 prompt view；进化时直接修改 `SocietySpec`，而不是改代码。

## 3. Multi-Agent Skill Society Graph

DySC 的核心不是“有多个 agent 帮忙执行 skill”，而是：

> MAS 本身就是 dynamic skill composition policy 的表示形式。

一个 Skill Society Graph 可以表示为：

```text
S = (A, K, E, B, T)

A: agent roles
K: skills and motifs
E: typed evidence, predicates, uncertainty, failures
B: role-skill bindings
T: communication and control topology
```

执行时，每个 agent 接收 typed evidence，根据自己的 role objective 和 skill access policy 选择或编写下一步代码，随后发出新的 evidence、predicate verdict、failure diagnosis 或 action result。通信拓扑决定这些信息流向谁，也就决定了 grounding、affordance、motion、verification 和 recovery 如何被逐 turn 编排。

例如：

```text
PerceptionScout -> AffordanceGeometer -> MotionExecutor -> PredicateVerifier
```

不是普通的多 agent 聊天，而是一种 composition policy：先建立目标证据，再生成物理 affordance，再执行动作，再验证谓词。失败后拓扑可以切到：

```text
PredicateVerifier -> RecoveryPlanner -> PerceptionScout
```

这表示：验证失败触发 recovery，recovery 再要求重新 grounding 或切换 affordance family。

相比单 agent 的隐式 prompt reasoning，Skill Society Graph 让 composition policy 变成显式对象。它可以被记录、比较、搜索、变异和剪枝。这也是 DySC 与普通 multi-agent engineering 的区别。

## 4. Dynamic Skill Composition Policy

DySC 的核心模块是一个 skill-level policy：

```text
pi(next_skill | observation, task, evidence, uncertainty, history)
```

它不执行固定 pipeline，而是在每一 turn 决定下一步应该调用哪个技能、是否需要局部修正、是否应该验证、是否应该 recovery，或是否可以终止。

例如：

- 如果目标身份存在歧义，先调用 relational grounding，而不是直接进入 grasp。
- 如果 grasp pose IK 可行但接触区域不确定，先调用 contact/affordance verifier，而不是直接 close gripper。
- 如果物体被边缘抓住，保留 object-frame evidence，并在 place 阶段消费该证据。
- 如果目标 support surface 很小，release 前进入 local alignment loop，而不是一次性释放。
- 如果任务后置条件不确定，先验证 task predicate，而不是 self-claim success。
- 如果同一 motion 失败多次，切换 affordance family 或 re-ground，而不是重复执行。

DySC 要学习的不是“某个任务该用哪个固定 workflow”，而是可迁移的 **composition motifs**。例如：

```text
relational grounding -> affordance proposal -> motion execution -> held-state verification
```

```text
object-frame preservation -> target grounding -> local alignment -> controlled release -> predicate verification
```

```text
failed execution -> diagnose uncertainty source -> re-ground or switch affordance
```

这些 motif 比单个 skill 更可迁移，因为它们编码的是具身交互结构，而不是某个对象或某个场景的解法。

在 DySC 中，这个 policy 不一定是一个独立神经网络或规则器。更自然的表示是 Skill Society Graph 本身：role、skill binding、evidence edge 和 topology 共同决定下一步组合行为。

## 5. Reflexive Execution Loop

DySC 将技能执行视为闭环反馈过程。每一轮执行遵循：

```text
1. Observe current state
2. Update evidence memory
3. Estimate task-relevant relation error
4. Select next skill or local correction
5. Execute bounded action
6. Verify whether the intended relation improved
7. Continue, switch, recover, or finish
```

这里最重要的抽象是 **relation error**。机器人操作任务的成功通常不是到达某个绝对 pose，而是满足某个关系谓词：

- object is held by gripper；
- object is inside container；
- bowl is supported by plate；
- object center is aligned with target support region；
- drawer is open enough；
- object is released and no longer coupled to gripper。

因此，DySC 不把 manipulation 看成一次性求解目标 pose，而是看成在多轮观测和动作中逐步减少 task-relevant relation error。

例如，在 bowl-on-plate 任务中，静态方法可能在抓取时估计 bowl center 和 grasp point 的 offset，然后在放置时用该 offset 修正 TCP release position。但这个 offset 在真实执行中是不稳定的：碗可能在夹爪里倾斜，抓取点可能和点云估计不一致，视觉观测可能受到遮挡。DySC 会把这个 offset 当成 uncertain evidence，而不是确定事实。系统可以先移动到 plate 上方，重新观测 bowl-plate-gripper 的关系，通过小步 xy local move 减少 bowl center 与 plate support region 的误差，再下降、释放、后退并验证最终谓词。

这使得方法的鲁棒性不依赖于每个单独 skill 都极其准确，而依赖于组合策略能否在物理反馈中修正偏差。

## 学习与适应

DySC 可以从交互轨迹中学习。一次 episode 结束后，系统不只提取成功代码片段或失败修复规则，还提取成功的组合决策：

- 哪种 observation 触发了哪个 skill？
- 哪种 uncertainty 需要 verifier？
- 哪些 evidence 需要跨 turn 保存？
- 哪个 local correction 降低了 relation error？
- 哪条 recovery path 在失败后有效？
- 哪些 skill 调用是冗余或高成本的？

这些信息被蒸馏成可复用的 composition motifs。例如：

```text
When placing a rim-grasped container onto a small support surface:
  preserve object-center evidence after grasp;
  ground target support surface from current observation;
  move above target conservatively;
  observe held-object / target relation;
  perform local xy correction before release;
  release only when relation error is below threshold;
  verify object-target predicate after retreat.
```

这个 motif 不绑定具体对象。它可以从 bowl-on-plate 迁移到 cup-on-tray、object-in-basket、bottle-on-rack 等任务，只要底层 grounding、affordance 和 motion skill 可用。

与 RATs 或 ASPIRE 不同，DySC 学到的主要 artifact 不是一个新的静态 skill，也不是一个特定程序修复，而是一个闭环组合模式。

## MAS 进化

DySC 将 dynamic skill composition 表述为 evolving multi-agent skill society。进化对象不是单个 prompt，也不是单个 skill，而是整个 society spec：

```text
Genotype = {
  roles,
  role objectives,
  skill access policies,
  evidence contracts,
  communication topology,
  composition motifs,
  budget policy
}
```

进化算子包括：

- **Role Birth**：当某类 uncertainty 或 failure 反复出现时，新建角色。例如 self-claim success 但 predicate false 时，产生 PredicateVerifier。
- **Role Split**：当一个 role 同时处理过多异质不确定性时拆分。例如 ActAgent 拆成 AffordanceGeometer 和 MotionExecutor。
- **Role Merge**：当两个 role 的 evidence 和 skill 使用高度冗余时合并，降低成本。
- **Skill Binding Mutation**：给某个 role 增加、移除、禁用或条件启用某个 skill。
- **Topology Mutation**：改变 evidence 路由和控制流，例如在 release 前插入 verifier，或在 failure 后回到 PerceptionScout。
- **Motif Mutation**：在 composition motif 中插入、删除、替换 phase，例如 `place -> verify` 变成 `hover -> reobserve -> local_align -> release -> verify`。
- **Motif Crossover**：把不同任务中有效的子图组合，例如将 open-container target grounding 与 offset-aware placement 组合。
- **Budget Pruning**：如果某个 role/edge/verification 在某类任务中长期无收益，则降低调用频率或剪枝。

fitness 不只看任务成功率，也看 predicate consistency、relation error reduction、跨 perturbation transfer、token cost、大模型调用次数、turn 数和 society complexity。

这使 DySC 与 EvoMAS 形成清晰关系：EvoMAS 进化 general reasoning workflows；DySC 进化 embodied skill composition societies。前者的 message 多是自然语言推理，后者的 message 是 typed evidence、physical predicates、uncertainty 和 action effects。

## 成本效率

DySC 的低成本来自适应对象的转移：

```text
from expensive program-level self-repair
to reusable composition-level adaptation
```

现有自进化方法通常需要大模型反复阅读长轨迹、重写代码、诊断错误并生成修复。DySC 则将常见执行过程压缩成可复用 motifs，使小模型或规则化 policy 可以处理多数 turn。大模型只在以下情况介入：

- 当前 motif 不适用；
- 出现新的 failure type；
- 多个 verifier 给出冲突判断；
- 需要总结新的 composition motif；
- 需要重组 skill/role 结构。

这不是简单地“用小模型替代大模型”，而是通过结构化组合策略减少每次任务都从头 reasoning 的需求。

## 预期贡献

DySC 的核心贡献是重新定义 embodied coding agent 中的技能复用问题。

已有工作通常问：

> How can agents acquire reusable robot skills?

DySC 问的是：

> How can agents acquire reusable ways of composing skills under physical feedback?

这一转变很重要，因为机器人操作中的很多失败并不是缺少某个技能，而是 perception、affordance、motion、verification 与 recovery 之间的编排过于静态和脆弱。

具体来说，DySC 的方法贡献包括：

1. **提出 Dynamic Skill Composition 作为新的问题表述。**  
   具身 coding agent 的泛化对象不只是 individual skills，而是物理反馈下逐 turn 的 skill composition policy。

2. **提出 Skill Society Graph 作为 composition policy 的显式表示。**  
   MAS 不是辅助执行框架，而是 role、skill binding、evidence contract 和 topology 组成的动态组合策略。

3. **提出 evidence-aware 的 composable skill interface。**  
   每个 skill 不只输出 action 或代码，还输出不确定性、证据、后置条件和后续组合约束。

4. **提出 relation-error-driven 的 reflexive execution loop。**  
   将操作任务视为多轮反馈中逐步满足关系谓词，而不是一次性计算目标 pose。

5. **提出从轨迹中蒸馏 reusable composition motifs 的机制。**  
   学习可跨任务迁移的闭环编排模式，而不是只扩张 flat skill library。

6. **提出 embodied MAS evolution 机制。**  
   通过 role birth/split/merge、skill binding mutation、topology mutation、motif crossover 和 budget pruning，进化 success/generalization/cost 的 Pareto frontier。

7. **提出 cost-aware 的组合级适应机制。**  
   将大模型调用集中在结构性失败和 motif 更新上，让 routine execution 由小模型、工具和已学组合模式完成。

## 一句话总结

DySC 的主张是：

> 具身 coding agent 的下一步不是学习更多孤立技能，而是学习如何在物理反馈中动态组合技能。真正可迁移的不是某段抓取或放置代码，而是使 grounding、affordance、motion、verification 和 recovery 在逐 turn 闭环中协同工作的组合策略。
