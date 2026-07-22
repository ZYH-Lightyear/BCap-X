# RoboMEx Agent Swarm 架构摘要

RoboMEx 是面向 Code-as-Policy 的 embodied Agent Swarm framework。它不替代 GaP 的 task-level typed robot graph，而是研究多个受约束、可按风险展开的 Coding Agent，如何在持续变化的物理世界中比一个 universal coder 更稳定、可控并易于失败归因。

下面关于 `ReactivePlanner → Session → SubgoalSwarmManager → SubgoalGraphExecutor` 的描述是
**截至 M3 的 v1 实现基线**。v1 继续作为迁移起点、failure corpus 和论文 baseline 独立运行，但
不是 M3.5 目标架构必须保留的 class graph。目标系统只继承 Planner/Manager/Skill-augmented Coding
Worker、typed contract 和确定性物理 authority 的语义分工；同步外层循环、一次性 Manager、单
cursor Executor、per-subgoal overwrite store 与共享 persistent globals 均允许替换。

M3.5 已平行建立 event-driven v2：`EpisodeOrchestrator` 管 active `SubgoalIntent`、事件、预算和
跨 node/subgoal 生命周期；versioned `SwarmManagerSession` 以 bounded invocation 管 Agent roster 与受限结构修订；
`ActivationScheduler` 驱动 fixed-topology loop、typed outcome 和 elastic slot；episode-scoped
append-only data plane 保存 artifacts/history 并由唯一 Reducer 提交控制 belief；只有
`ActionSupervisor + SealedActionRunner` 能调用 world-changing API。v1/v2 只在外部
`IntentOutcome`、manifest 和评测指标处对齐，不要求内部 node 顺序或 artifact 布局兼容。

截至 M3，外层 `ReactivePlanner` 根据任务、当前场景图、高层 task skill 菜单及历史执行结果，每次生成一个自然语言 sub-goal 和可观察 postcondition。`Session` 负责循环调用 Planner、运行 sub-goal、刷新场景并汇总 episode；目前支持动态重规划，但最终是否成功仍以环境 reward、terminated 和 task-completed 信号为准。

截至 M3，每个 sub-goal 由 `dynamic_swarm` 或 `universal` 两种策略之一执行。`dynamic_swarm` 先向 `SubgoalSwarmManager` 渐进披露 high-level task skill：Manager 最初只看名称和描述，通过 `use_skill` 按需加载完整 `SKILL.md`，再结合当前图像与 specialist contract 动态生成节点、typed bindings、条件边和 recovery。生成图通过结构验证后才交给确定性的 `SubgoalGraphExecutor`。Manager 不能发明 role、capability、端口或预算；这些均来自 leaf skill `contract.yaml`。`universal` 保留为独立单 Coding Agent baseline。

Leaf Coding Agent 仍采用多轮 Code-as-Policy 循环，但只在一个 specialist stage 内编写 `SemanticActionBlock`。Required skills 自动预加载，额外 Skill 由 role-conditioned library view 限定。**截至 M3**，Grounding/Affordance/MotionPlanner/Verifier 不能改变物理状态，只有 ActionExecutor 拥有受界 robot API；M3.5 将进一步收紧为 Agent 只能构造/选择 sealed action spec 并提交 admission layer，raw world-changing API 只对 deterministic runtime adapter 可见。

Skill 仍是以 `SKILL.md` 为主体的 prose-first 自包含目录包，并可携带 references、scripts 与轻量 `contract.yaml`。task skill 正文只提供高层组合知识，不承载 nodes/edges；leaf contract 只声明机器需要的 role、typed ports、capability boundary 和 budget。**截至 M3**，Graph 是每个 subgoal 的独立动态产物。

**截至 M3**，`ArtifactStore` 是 Swarm 唯一机器数据面。GroundingArtifact、AffordanceArtifact、TrajectoryArtifact、ExecutionEvidence 与 VerifierReport 使用 `producer.port` 全限定地址，下游只能通过 `$ref` 绑定。Store 检查 required port、schema、frame、observation epoch 与文件引用，并在整批通过后原子发布；自然语言 evidence 不能替代 graph edge。

截至 M3，`RuntimeSafetyState` 是 observation epoch 的唯一来源。ActionExecutor 之后必须进入当前 epoch 的 hard Verifier；executor 自检只是 soft evidence。Verifier failure 只能走已验证动态图声明的 recovery edge。运行时增量保存 Manager 提交图、编译图、`subgoal_graph.json`、node exit events、ArtifactStore 和统一 outcome，从而让失败可以定位到 grounding、affordance、motion planning、action coding 或 verification 阶段。

## M3.5 v2 架构（完整 baseline 已实现，live/paper 验收进行中）

> 2026-07-22 revision：完整代码基线、Elastic Graph/Swarm/Monitor 生命周期、迁移顺序、
> 测试和论文消融，以
> [`robomex_m35_elastic_swarm_implementation_plan.md`](robomex_m35_elastic_swarm_implementation_plan.md)
> 为 source of truth。代码完成不等于真实机器人统计或论文假设已经验收。

20260720 live run 证明“显式图 + nominal typed ports”仍不足以保护**声明的表示语义**：
OBB 可以在 producer/consumer 字段边界消失，抓持 offset 无法跨 subgoal，统一 epoch
会同时错误处理 snapshot 与持久 belief，recovery 会回到已 stale 的输入，planner 检查
的 IK 也可能不是 executor 实际执行的 IK。

M3.5 已把当前数据面升级为 episode-scoped `EpisodeDataPlane`，并拆成两类逻辑存储：
append-only artifact store 保存不可变
`ObservationSnapshot`、由 snapshot/revision 定址的 `CollisionWorld`、geometry、affordance、
单段 `MotionPlan`、runtime-owned `ActionAttempt` 与执行回执；episode-level
`EmbodiedStateLedger` 保存 held object、attachment、gripper 与 object relation 的
**runtime 权威 belief**，而不是未经条件的物理真值。Agent 只能发布
measurement/estimate/candidate/proposed effect；deterministic gate 负责 schema、revision、
plan identity 与安全不变量，learned `EvidenceVerifier` 只根据证据提出
`passed/failed/uncertain` transition proposal，封闭的 deterministic `StateReducer` 是
ledger 的唯一 commit 入口。

Ledger 将 object belief 分成三条封闭正交轴：attachment
`{not_held, attempted, verified_held, unknown}`、localization
`{localized, unlocalized, ambiguous}`，以及按 `(subject, family, target)` 定址的
containment/support relation enum；`lost_object` 只是一种 failure kind，不是 state value。

单一 observation epoch 已升级为 dependency-aware revision vector，artifact 按
snapshot/derived/plan/action receipt/episode state/session static 选择性失效。skill
contract 增加 physical precondition/effect/invalidation，recovery edge 增加状态 guard；
graph compiler 对所有 success/recovery path 做 freshness、state 与 verifier closure
分析。Motion Planner 每次只输出一个带 plan ID、start/collision-world guards 的不可变
joint segment；runtime 建立 `ActionAttempt`，deterministic runtime adapter 原样执行且禁止二次 IK，
回执由 runtime 根据实际调用/trace 生成并绑定同一 attempt/plan，而不是由 Agent 自报。
open/close 使用独立 sealed `GripperCommand`，但走同一 write-ahead attempt/receipt 协议。
这里的 ActionExecutor Agent 只能提交或在 admission 前修复 sealed spec；真正调用 raw motion/
gripper primitive 的是 runtime adapter，不存在“Agent 兜底直接动作”旁路。
WAL 在 `admitted` 后、首个 primitive 前 durable flush；重启时 orphan attempt 一律标为
`indeterminate_after_crash`，保守失效相关 belief/plan，待 controller quiescence 与 fresh
observation 通过后才恢复 admission，不假设 physical exactly-once。
若 terminal receipt 已 durable、但同 `effect_id` 的 Reducer commit 尚未落盘，restart
按 `action_id/effect_id` 幂等补交 conservative revision/state transition；已有 commit 则 no-op。

该设计的静态部分保证每个 Action input 有唯一兼容 producer，且所有到达路径都包含所需
admission checks；运行时只有在当下 schema/validity/state/plan guards 全部通过时才允许
`Execute(n,t)`。它不保证 perception、learned verifier 或 effect model
必然描述真实世界，也不证明连续几何、碰撞与接触正确。未监测的世界变化仍可破坏 belief，
因此 M3.5 的目标是 *execution-gated by construction*，不是“物理真值由构造保证”。

完整 schema 与研究边界见
[`robomex_m35_embodied_data_plane.md`](robomex_m35_embodied_data_plane.md)。当前 baseline 已实现
action-facing 的窄状态、append-only artifacts 与 sealed plan/receipt，不先建设一个全量通用
ledger。Graph 也不再按“整图运行期永久冻结”或单一 `executed prefix` 定义：并行 service、重试和
bounded loop 下未必存在一条线性前缀；不可变的是已经 commit 的 activation events、artifacts、
attempts 和 receipts，以及当前 admitted action 的 plan/monitor/input identity。

Manager 在已声明 Slot/role/budget 内 spawn、suspend、resume、terminate Agent 属于 **roster
update**，只改变 Agent lifecycle，不改变 graph revision。只有填充或替换声明过的 inactive future
control/dataflow region 才是 **GraphPatch**，必须以 base revision 重新编译并原子 commit。runtime
同时提供 global-quiescence profile，并默认支持 **affected-scope barrier**：
被改区域及 causal dependents 无 running activation、未决 artifact reservation 或 admitted action
即可，独立 Tracker/Monitor service 不必停止。Manager 提交初始 scaffold 后 sleep，仅在
slot/monitor/verifier 的结构化事件上按预算唤醒；普通 node success、alignment iteration 和 frame
tick 不调用 Manager。未声明 slot、已 commit history、运行中 action 和安全 contract 始终不可修改；
memory promotion、mutation/crossover 和不受限拓扑演化继续留给 M5。
