# RoboMEx 后续实现点与当前隐患

本文记录 2026-07-05 重构后的 RoboMEx 状态、后续应继续实现的工程点，以及当前设计仍存在的隐患。当前方法定调是：**Trace-centered context envelope + task-first SubAgents + clean workflow skills**。目标不是把系统做成固定 phase/profile 流水线，也不是把成功轨迹包装成 callable API skill，而是让小模型 Act Coding Agent 在在线执行时通过技能工作流、局部证据和高频观测更稳地完成任务。

## 当前状态

已经完成的关键基建：

- Act 仍是唯一执行机器人动作的 Coding Agent。
- SubAgent 是 task-first 的只读 CodingAgent，Act 通过自然语言任务委托，不通过固定 `name/profile/phase` 路由。
- 新增通用 `EvidencePacket`，用于承载 `claim / confidence / evidence / artifacts / facts / verdict / uncertainty / recommended_next`。
- `EVIDENCE` 保持为单 Agent / 单 subgoal 的局部 scratchpad；跨 Agent 信息必须通过 evidence packet、artifact ref、state patch 或 trace 传递。
- Episode 结束会生成 `skill_evolution_candidates.json`，只作为离线 clean skill evolution 的候选输入，不自动修改技能库。
- Skills 文档开始转向 workflow memory：候选生成、局部检查、失败模式、clean reusable rules、weak priors、prohibited shortcuts。

这使 RoboMEx 初步区别于 RATs 和 ASPIRE：

- 不把 skill 主要定义为成功代码函数库。
- 不把 benchmark-specific pixel/coordinate heuristic 提升为正式技能规则。
- 不靠固定多 Agent profile 流水线驱动系统。

## 后续必须实现的点

### 1. EvidencePacket 的调试 UI 与可读摘要

当前 `EvidencePacket` 已经进入 trace metadata，但它还主要是 JSON 记录。下一步需要让它真正变成 Agent 间通信和人类调试的中心。

需要实现：

- 在 episode summary 中生成 compact evidence timeline。
- 在 Web UI / debug panel 中展示每次 SubAgent call 的 claim、verdict、artifact、recommended_next。
- Planner history 只读取 evidence 摘要，不读取完整 SubAgent JSON。
- 为 `evidence` 内部的自由字段做安全截断，避免大数组或长文本进入 prompt。

验收标准：

- 打开一次 live run 目录，可以清楚看到每个 subgoal 内 Act 收到了哪些 SubAgent evidence。
- Planner prompt 中不会出现大段原始 SubAgent payload。

### 2. Primitive Trace 自动化仍不够

当前 trace 主要来自 `run_python` block、execution events 和 Agent 主动返回的结构字段。真正成熟的系统应该自动记录关键 primitive 的输入输出摘要，而不是依赖模型在 `finish` 里写。

需要实现：

- 为常用 API 调用建立轻量 event wrapper，例如 segmentation、bbox detection、point detection、GraspNet、IK、goto_pose、open/close gripper。
- 每个 wrapper 记录输入摘要、输出摘要、artifact refs、错误、耗时。
- 不记录大数组本体，只记录 shape、统计量、保存路径。
- 将 wrapper events 转成 `PrimitiveTrace`，并自动进入 `TraceStore`。

验收标准：

- 即使 Act 没有在 `finish` 中写 `primitive_traces`，运行后仍能看到感知、抓取、运动相关 trace。
- 失败诊断可以定位到具体 primitive，而不是只看到“某个 Python block 失败”。

### 3. SubAgent 的有效使用还没有闭环保证

当前 prompt 鼓励 Act 在非平凡分析时调用 SubAgent，但这仍是软约束。模型可能继续自己写大量观察代码。

需要实现：

- 统计每个 subgoal 中 Act 自己写的 read-only analysis block 数量。
- 当连续出现无动作、低价值 observation probing 时，在下一轮 prompt 中给出 nudge：应加载 skill 或委托 SubAgent，而不是继续探查框架 schema。
- 不做硬性 phase gate，但做行为级反馈，例如“你已经两次只打印 observation 信息，下一步应产生 evidence 或执行动作”。
- 将 SubAgent 调用成本、耗时、产物质量写入 summary。

验收标准：

- Act 不再反复写 `get_observation()` / `obs.keys()` / shape probing。
- SubAgent 调用是否发生、是否有用，可以被 trace 明确审计。

### 4. Skill 文档还需要系统性重写

目前只有部分核心 skills 加入了 clean rules / weak priors / prohibited shortcuts。后续需要把整个 skill library 统一到同一方法口径。

需要实现：

- 为每个 `SKILL.md` 使用统一结构：
  - Purpose
  - When to use
  - Workflow
  - Candidate generation
  - Local checks
  - Failure modes
  - Clean reusable rules
  - Weak priors
  - Prohibited shortcuts
  - Artifacts to save
- 将长案例、实验失败记录、对象 prompt registry 放入 `references/`，不要塞进主 `SKILL.md`。
- 对明显 benchmark-specific 的内容加标签：`weak_prior` 或 `shortcut_reject`，不能作为默认执行规则。
- 检查所有 skill 是否仍暗示固定 SubAgent role 或固定 pipeline。

验收标准：

- 每个 skill 都能告诉 Agent 如何生成候选、如何局部检查、何时不信某个信号。
- `SKILL.md` 不变成巨大 benchmark 答案表。

### 5. Clean Skill Evolution 仍只是候选文件

当前 `skill_evolution_candidates.json` 只是 trace digest，还没有 curator。后续应实现离线 skill evolution，但不能自动把经验写入正式技能。

需要实现：

- 离线 `SkillEvolutionCurator`，输入 episode trace 和 candidates。
- 输出 candidate patches，而不是直接改 `SKILL.md`。
- 每条 patch 标注：
  - `clean_rule`
  - `weak_prior`
  - `failure_case`
  - `shortcut_reject`
- Anti-shortcut filter 检查：
  - 是否依赖固定 pixel / world coordinate？
  - 是否只对单个 seed / task / object instance 有效？
  - 是否能改写成类别、几何、关系或候选验证规则？
- 人工 review 后再 admit 到 skill。

验收标准：

- 成功/失败 episode 可以自动产出 skill patch 候选。
- 没有任何自动流程直接把 benchmark shortcut 写进正式 skill。

### 6. Planner 仍需要更强的状态一致性

之前出现过“物体已经抓起，但 Planner/Act 仍继续找原物体”的问题。当前 world state 机制有所改善，但事实过期、冲突和置信度策略还不完善。

需要实现：

- 将 `robot.held_object`、`object.state`、`object.location` 等事实形成建议命名规范，但不要变成强 schema。
- 对高风险事实要求 provenance，例如来自 lift check、VLM state check、env reward、SubAgent verdict。
- 引入事实失效机制：
  - 物体被 release 后，held-object 事实应自动失效。
  - 重新观察强烈矛盾时，低置信事实应被 invalidated。
- Planner prompt 中突出“已提升事实是 fallible belief，但不能无视高置信事实”。

验收标准：

- 如果上一 subgoal 确认物体已被抓起，下一 subgoal 不应再次定位桌面上的同一个物体，除非当前图像或诊断明确否定。

## 当前实现后的主要隐患

### 隐患 1: EvidencePacket 可能变成新的弱 schema

虽然 `EvidencePacket` 是开放外壳，但如果 prompt 或测试逐渐要求某些 `evidence` 内部字段，就会重新退化成固定 phase schema。

控制方式：

- Runtime 只理解外壳字段，不依赖 `evidence.target`、`evidence.candidates` 等内部键。
- 内部字段只给 Agent 和人类调试使用。
- 新功能不得通过判断 `result_type == grounding` 或类似字段路由。

### 隐患 2: Skill 文档可能再次膨胀成 benchmark 经验表

ASPIRE 风格 skill 很强，但容易把 benchmark layout、pixel threshold、对象坐标写成“技能”。这会提高短期准确率，但破坏方法泛化性。

控制方式：

- 主 `SKILL.md` 只放可泛化 workflow。
- 具体 task/seed 经验进入 `references/failure_cases.md`。
- 任何固定坐标、固定像素、固定对象位置默认进入 `shortcut_reject`，除非能改写成泛化几何/关系规则。

### 隐患 3: SubAgent 可能被 Act 当成昂贵的普通函数

如果 Act 每个细节都 call SubAgent，会引入 RATs/ASPIRE 类似的高 token 成本；如果完全不用 SubAgent，又退回单 Agent 随机写代码。

控制方式：

- 只在不确定性实际影响动作安全/成功率时调用 SubAgent。
- summary 中统计 SubAgent 调用次数、耗时、是否产出有用 evidence。
- 后续可加入“低价值 SubAgent 调用”诊断，而不是硬性禁止。

### 隐患 4: 当前执行粒度仍可能两极化

模型可能一次写很长脚本，也可能写很多无动作观察代码。RoboMEx 需要的是 phase-level code block，但不能通过固定 phase enum 强制。

控制方式：

- Prompt 强调“one bounded code block that advances the subgoal”。
- 对连续无动作探查给 nudge。
- 对超长脚本只在高风险时提示拆分，例如包含多个物理阶段、多个 release/regrasp。

### 隐患 5: 当前 tests 主要验证结构，不验证 live 行为

`pytest robomex/test` 能证明 contract 没坏，但不能证明 live LIBERO 中 Act 真会更好地使用 SubAgent 和 skills。

控制方式：

- 增加 mock integration replay：给定模型输出序列，检查 evidence packet 如何流入下一 prompt。
- 增加 1-2 个真实 LIBERO smoke run 的人工验收 checklist。
- 每次 live run 后检查：
  - 是否产生 evidence packets？
  - 是否产生有用 artifact？
  - 是否避免重复无意义 observation probing？
  - Planner 是否尊重高置信 world facts？

### 隐患 6: 当前离线 skill evolution 还没有质量门

已经有 `skill_evolution_candidates.json`，但没有 curator 和 anti-shortcut filter。此时如果人工直接把 candidates 复制进 skill，仍可能污染 skill library。

控制方式：

- 在实现 curator 前，不自动 admit skill patch。
- 文档和 prompt 明确 candidates 是候选，不是事实。
- 每条候选必须经过 clean-rule / weak-prior / shortcut-reject 分类。

## 推荐实现顺序

1. Evidence timeline 和 debug UI 展示。
2. Primitive API trace wrapper。
3. Act 连续低价值 observation probing 的 nudge。
4. 全量 skill 文档按 workflow memory 模板重写。
5. Planner world-state consistency 改进。
6. 离线 SkillEvolutionCurator + anti-shortcut filter。
7. 小规模 live benchmark，对比：
   - no SubAgent
   - current SubAgent
   - evidence packet + trace wrapper + improved skills

## 方法边界

RoboMEx 后续实现必须坚持以下边界：

- 不恢复固定 Verifier 外环硬门控。
- 不引入固定 SubAgent profiles。
- 不把 skill 主要做成 callable helper function library。
- 不把 benchmark-specific shortcut 提升为默认技能规则。
- 不让完整 chat history 成为 Agent 间 context。
- 不让大数组、长视频、长 JSON 直接进入 prompt。

系统应该继续沿着这条线推进：

```text
Act owns execution.
SubAgents produce compact evidence.
Skills store clean workflow memory.
Trace is the shared context.
Skill evolution is offline, reviewed, and anti-shortcut filtered.
```
