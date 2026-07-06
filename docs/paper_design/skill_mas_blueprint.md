# Skill-MAS: 面向低成本机器人 Coding Agent 的结构蓝图

本文规划一个以 `Skill + Trace + Multi-Agent Diagnosis` 为核心的 RoboMex 下一代架构。目标不是简单堆叠更多 SubAgent，而是把机器人执行过程变成可诊断、可修复、可沉淀的闭环：Act Coding Agent 负责编写和执行动作代码；专门智能体负责局部分析；每个物理模块拥有自己的局部 verifier；全局 Diagnostician 负责失败归因和修复路由；Skill Library 承担长期知识记忆。

## 1. 核心判断

Grounding、Affordance、Motion Planning 本身都应该有各自的验证方式。比如 grounding 的验证不是“任务成功了吗”，而是“mask/bbox 是否指向目标物体”；affordance 的验证不是“抓起来了吗”，而是“候选抓取点是否在物体可接触区域、是否有 IK、是否避开已失败姿态”；motion 的验证不是“物体是否在盘子上”，而是“机器人是否真的到达目标 pose、是否发生大偏移、IK 解是否导致 wrist flip”。因此，单独存在的全局 `Verifier SubAgent` 如果只做成功/失败判断，会过于粗糙。

更合理的命名是 **Diagnostician Agent**。它不是替代各模块 verifier，而是读取执行 trace、局部 verifier 输出、world state 和 attempt history，然后回答：这次失败属于哪类？哪个 primitive 或 skill 需要被 invalidated？下一步应该调用哪个 specialist 或 repair skill？

## 2. 总体结构图

```text
                  ┌────────────────────────────┐
                  │        Planner Agent        │
                  │  task -> subgoal/postcond   │
                  └──────────────┬─────────────┘
                                 │
                                 v
┌────────────────────────────────────────────────────────────┐
│                    Act Coding Agent                         │
│  main actor: reads skills, writes code, executes actions     │
│  owns final robot API calls: goto_pose / grasp / release     │
└──────────────┬───────────────────────────────┬─────────────┘
               │                               │
               v                               v
┌───────────────────────────┐       ┌───────────────────────────┐
│ Primitive Execution Layer │       │      Skill Library         │
│ segment / grasp / motion  │       │ base skills + repair skills│
│ release / query state     │       │ when-to-apply + wrappers   │
└──────────────┬────────────┘       └─────────────┬─────────────┘
               │                                  │
               v                                  │
┌────────────────────────────────────────────────────────────┐
│                    Primitive Trace Store                    │
│ before/after images, bbox, mask, points, candidates, IK,    │
│ reached pose, gripper width, object relation, artifacts     │
└──────────────┬─────────────────────────────────────────────┘
               │
               v
┌────────────────────────────────────────────────────────────┐
│                 Local Specialist Agents                     │
│ Grounding Agent     + Grounding Verifier                    │
│ Affordance Agent    + Affordance Verifier                   │
│ Motion Agent        + Motion Verifier                       │
│ Placement Agent     + Placement Verifier                    │
└──────────────┬─────────────────────────────────────────────┘
               │ structured local diagnoses
               v
┌────────────────────────────────────────────────────────────┐
│                  Diagnostician Agent                        │
│ failure attribution, invalidated facts, repair routing,      │
│ skill admission proposal, escalation decision                │
└──────────────┬─────────────────────────────────────────────┘
               │
               v
┌────────────────────────────────────────────────────────────┐
│ WorldState + AttemptHistory + Skill Curator                 │
│ current facts, failed attempts, stale facts, learned repairs │
└────────────────────────────────────────────────────────────┘
```

## 3. 各组件职责

**Act Coding Agent** 是系统中心。它不是纯 manager，也不是只会调 SubAgent 的 dispatcher。它负责读取 Planner 的 subgoal、检索 skill、调用必要的 specialist、写最终执行代码，并决定何时 finish。Act 可以先执行简单动作，也可以在不确定时请求局部诊断，但最终机器人动作必须由 Act 统一发起。

**Primitive Trace Store** 是整个系统的证据层。每次调用 `segment_object`、`grasp_open_bowl`、`goto_pose`、`close_gripper`、`release_at` 都要写 trace。trace 里不只保存日志文本，而要保存结构化字段和 artifacts，比如 bbox、mask overlay、grasp candidates、IK status、commanded pose、reached pose、before/after image。没有 trace，Agent 只能凭图像和聊天历史猜；有 trace，诊断才能落到具体 primitive。

**Specialist Agents** 是专门知识智能体，但它们只做局部问题。Grounding Agent 负责“这是不是目标物体”；Affordance Agent 负责“哪个接触点和姿态更合理”；Motion Agent 负责“机器人能不能稳定到达”；Placement Agent 负责“放置目标中心和释放姿态如何定义”。这些 Agent 可以使用小模型，因为它们的输入被 trace 和 skill 约束过，问题范围较窄。

**Local Verifier** 是每个 specialist 内部的检查机制。例如 Grounding Verifier 检查 crop、mask、相对关系；Affordance Verifier 检查抓取点是否在点云/rim/side wall 上、是否重复失败 pose；Motion Verifier 检查 reached pose error 和 joint jump；Placement Verifier 检查 final object center 是否满足 relation。它们输出局部 verdict，不直接宣布整任务成功。

**Diagnostician Agent** 是全局诊断者。它读取所有局部 verdict 和 trace，做 failure attribution。例如一次失败可能不是 affordance 错，而是 motion 没到位；也可能 motion 到位了，但 grasp point 在碗外侧；也可能抓取成功，但 release 没补偿 rim grasp offset。Diagnostician 的输出应该结构化：`failed_primitive`、`failure_reason`、`invalidated_facts`、`recommended_repair_skill`、`need_large_model`、`confidence`。

**Skill Library** 分为 base skill 和 repair skill。Base skill 是人写的 wrapper 使用说明；repair skill 是从失败和验证中沉淀的经验，包含 `failure_signature`、`when_to_apply`、`repair_code_or_policy`、`validation_status` 和 `cost_profile`。长期知识不应该埋在 SubAgent prompt 里，而应该沉淀为 skill。

## 4. Case: 抓碗并放到盘子上

任务：`pick the akita black bowl next to the ramekin and place it on the plate`。

第一步，Act 读取任务，调用 Grounding Agent。Grounding Agent 使用 `segment_object` wrapper 得到碗的 bbox、mask、center_xyz，并由 Grounding Verifier 检查“目标是 ramekin 右侧的碗，而不是前方另一个碗”。通过后写入 WorldState：`akita_bowl.grounding`。

第二步，Act 调用 Affordance Agent。Affordance Agent 检索 `grasp_open_bowl` skill，生成 rim grasp candidates。Affordance Verifier 检查候选是否位于 rim/side wall，是否 IK 可行，是否与 AttemptHistory 中失败 pose 过近。输出 `selected_grasp` 和 `object_center_offset_from_grasp`。

第三步，Act 执行抓取。Primitive Trace 记录 commanded pregrasp/grasp pose、实际 reached pose、gripper width、before/after image。若碗没有被抓起，Motion Verifier 先检查是否真的到达 grasp pose。如果 reached pose 偏差很大，Diagnostician 判定失败为 `motion_tracking_failure`，而不是错误地否定 Affordance。如果到位但没抓住，则判定 `bad_contact_point`，invalidate 当前 grasp pose，并请求 Affordance Agent 生成不同候选。

第四步，抓取成功后，Act 调用 Placement Agent。Placement Agent 估计盘子中心和 desired object center。由于碗是 rim grasp，TCP 不等于碗中心，Placement Verifier 要求使用 `object_center_offset_from_grasp` 计算 release pose：`tcp_release_pos = desired_object_center - offset`。释放后，Placement Verifier 检查碗中心是否在盘子 support region 内，而不是检查 gripper release pose。

第五步，如果放偏，Diagnostician 不应简单说“place failed”。它要判断是 plate grounding 错、offset 错、release pose 没到位，还是物体滑动。若发现 offset 没用或 offset 过期，则写入 AttemptHistory，并触发 repair skill：`off_center_rim_grasp_release_compensation`。

## 5. 研究方法叙事

这套结构的 paper story 可以这样讲：ASPIRE 证明 trace-guided repair 能发现机器人技能，但它依赖强模型和昂贵搜索。我们提出 Skill-MAS，把昂贵的全局 debug 拆成 trace-routed local diagnosis。每个 specialist 用局部 verifier 限制问题范围，使小模型也能可靠工作；全局 Diagnostician 只做失败归因和路由；验证成功的 repair 才进入 skill library。这样系统不是靠更多 Agent 暴力推理，而是靠 trace、skill 和结构化诊断降低搜索空间。

核心贡献可以表述为：

1. 一个面向机器人 coding agent 的 trace-routed MAS 架构。
2. 一种局部 verifier + 全局 diagnostician 的失败归因机制。
3. 一种把 validated repair 沉淀为 skill memory 的低成本学习闭环。

最终目标是让 RoboMex 从“Agent 聊天式协作”转向“证据驱动的多智能体诊断与技能修复系统”。
