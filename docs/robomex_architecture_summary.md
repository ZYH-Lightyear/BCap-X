# RoboMEx Subgoal-Level MAS 架构摘要

RoboMEx 是面向 Code-as-Policy 的 subgoal-level MAS optimization framework。它不替代 GaP 的 task-level typed robot graph，而是在外层 Reactive Planner 给出的单个 embodied subgoal 内，研究多个受约束的小型 Coding Agent 是否比一个 universal subgraph coder 更稳定、可控并易于失败归因。

外层 `ReactivePlanner` 根据任务、当前场景图、高层 task skill 菜单及历史执行结果，每次生成一个自然语言 sub-goal 和可观察 postcondition。`Session` 负责循环调用 Planner、运行 sub-goal、刷新场景并汇总 episode；目前支持动态重规划，但最终是否成功仍以环境 reward、terminated 和 task-completed 信号为准。

每个 sub-goal 由 `dynamic_swarm` 或 `universal` 两种策略之一执行。`dynamic_swarm` 先向 `SubgoalSwarmManager` 渐进披露 high-level task skill：Manager 最初只看名称和描述，通过 `use_skill` 按需加载完整 `SKILL.md`，再结合当前图像与 specialist contract 动态生成节点、typed bindings、条件边和 recovery。生成图通过结构验证后才交给确定性的 `SubgoalGraphExecutor`。Manager 不能发明 role、capability、端口或预算；这些均来自 leaf skill `contract.yaml`。`universal` 保留为独立单 Coding Agent baseline。

Leaf Coding Agent 仍采用多轮 Code-as-Policy 循环，但只在一个 specialist stage 内编写 `SemanticActionBlock`。Required skills 自动预加载，额外 Skill 由 role-conditioned library view 限定。Grounding/Affordance/MotionPlanner/Verifier 不能改变物理状态；只有 ActionExecutor 可以执行受界机器人 API code。

Skill 仍是以 `SKILL.md` 为主体的 prose-first 自包含目录包，并可携带 references、scripts 与轻量 `contract.yaml`。task skill 正文只提供高层组合知识，不承载 nodes/edges；leaf contract 只声明机器需要的 role、typed ports、capability boundary 和 budget。Graph 是每个 subgoal 的独立动态产物。

`ArtifactStore` 是 Swarm 唯一机器数据面。GroundingArtifact、AffordanceArtifact、TrajectoryArtifact、ExecutionEvidence 与 VerifierReport 使用 `producer.port` 全限定地址，下游只能通过 `$ref` 绑定。Store 检查 required port、schema、frame、observation epoch 与文件引用，并在整批通过后原子发布；自然语言 evidence 不能替代 graph edge。

`RuntimeSafetyState` 是 observation epoch 的唯一来源。ActionExecutor 之后必须进入当前 epoch 的 hard Verifier；executor 自检只是 soft evidence。Verifier failure 只能走已验证动态图声明的 recovery edge。运行时增量保存 Manager 提交图、编译图、`subgoal_graph.json`、node exit events、ArtifactStore 和统一 outcome，从而让失败可以定位到 grounding、affordance、motion planning、action coding 或 verification 阶段。
