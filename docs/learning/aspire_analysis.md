# ASPIRE 对 RoboMex Agentic 设计的启发

本文分析 NVIDIA GEAR 的 ASPIRE: Agentic Skills Discovery for Robotics，并结合 RoboMex 当前的 Skills + SubAgent 设计问题，整理可借鉴的系统思路。ASPIRE 的项目页见 <https://research.nvidia.com/labs/gear/aspire/>，论文见 <https://arxiv.org/abs/2607.00272>。

## 1. ASPIRE 的核心思想

ASPIRE 不是一个“预先注册很多专家 SubAgent，然后让每个 SubAgent 固定负责一个模块”的系统。它的核心更接近：

```text
任务程序执行失败
  -> 记录细粒度 primitive trace
  -> Coding Agent 定位失败原因
  -> 写代码修复
  -> 重新执行验证
  -> 把被验证过、可迁移的修复经验沉淀成 skill
  -> 后续任务检索这些 skill 作为上下文指导
```

论文把 ASPIRE 概括为三个组件：

1. Closed-loop robot execution engine：每个 perception、planning、grasping、control primitive 都记录输入、输出、状态、图像、overlay、候选抓取、运动规划结果等细粒度 trace。
2. Continually expanding skill library：skill 不是预先手写完整任务程序，而是从“失败 -> 修复 -> 验证成功”的经验中提炼出的可复用知识。
3. Evolutionary search over programs：当单一路径 debug 容易陷入局部循环时，系统生成多个候选程序并行探索，保留表现最好的程序和失败 trace。

这点和 RoboMex 目前的问题高度相关。RoboMex 当前更像是：

```text
Planner -> Act -> SubAgent/Skill -> Evidence -> Act 自行解释 -> Planner 继续
```

但 ASPIRE 强调的是：

```text
Actor writes program
  -> engine records primitive-level traces
  -> Actor diagnoses exact failed primitive
  -> repair is validated
  -> coordinator admits reusable repair into skill library
```

也就是说，ASPIRE 的“skill”不是给 Agent 一段泛泛的操作说明，而是经过执行验证的故障修复知识。

## 2. ASPIRE 不是固定专家 SubAgent 系统

你现在担心“每个 SubAgent 负责一个专门任务这块没有完整做起来”，这个判断是合理的。ASPIRE 给出的反例是：系统并不需要一开始就把 Grounding Agent、Affordance Agent、Verifier Agent、Motion Agent 都定义成固定专家，并强制 Act 调度它们。

ASPIRE 的 coordinator-actor 架构里，中心 coordinator 管理共享 skill library，并把任务分发给 actor coding agents。每个 actor 自己写、执行、诊断、修复程序。不同 actor 之间不共享完整聊天历史，也不共享原始 rollout，而是把可迁移经验压缩成 skill library。

这对 RoboMex 的启发是：

- SubAgent 不应该是主要知识载体。
- Skill library 才应该是长期知识载体。
- SubAgent 可以作为临时 compute worker、debug worker、verification worker，但它的产出必须被结构化审计后才能进入长期记忆。
- 如果 SubAgent 只是返回一大段自然语言 claim，而 Act 自己决定是否信任，那么系统会非常不稳定。

换句话说，RoboMex 不应该把“专门化”绑定在 SubAgent 名字上，而应该绑定在 skill / trace / validated repair 上。

## 3. ASPIRE 的 Skill 观和 RoboMex 当前 Skill 观的差别

RoboMex 当前的 skill 更像人写的 workflow guide：

```text
什么时候用
怎么调用 wrapper
常见失败模式
输出哪些 evidence
```

这仍然有价值，但还不是 ASPIRE 意义上的 skill。ASPIRE 的 skill 更偏向：

```text
failure signature:
  什么失败现象触发了这个经验

when-to-apply guard:
  在什么场景下可以检索并应用

validated repair:
  已经被调试和验证过的修复策略

origin:
  从哪些任务/失败 trace 中归纳出来
```

论文附录明确说，skill-library entry 会包含触发失败 trace 中抽取的问题、when-to-apply guard、validated repair snippet、origin task。这个格式比“给 Agent 看的一段说明”更工程化。

RoboMex 当前最大的问题是 Evidence、Skill、WorldState、SubAgentResult 的边界混在一起。比如这次 trial 里，SubAgent 能输出 `grasp_affordance.akita_black_bowl`，Act 也能执行，但失败后没有稳定形成：

```text
这个 grasp 已尝试
失败原因是什么
不要重复使用哪个 pose
下一次应该换哪类策略
```

所以系统会反复相信 `ik_ok=True`、`score=1.0` 的旧 affordance。ASPIRE 的做法会把这种失败转成 trace-grounded repair skill，而不是简单保留一个高置信 grasp candidate。

## 4. 对 RoboMex 的直接批判

结合 `outputs/robomex_opus4-8/20260704_134917`，RoboMex 当前主要问题不是“SubAgent 数量不够”，而是缺少 ASPIRE 式的闭环。

### 4.1 Evidence 不是长期记忆

Evidence 适合做单次执行代码块内的局部变量，例如 mask、points、candidate list。它不适合直接承担跨 subgoal、跨 SubAgent 的状态通信。

应该分层：

```text
Local Evidence:
  当前 Act/SubAgent 的临时数据，如 point cloud、mask、candidate list。

Artifact:
  overlay、视频、trace json、candidate visualization。

WorldState:
  当前世界事实，如 object center、held state、object on plate。

AttemptHistory:
  已尝试的 grasp/place/motion，成功或失败，失败原因。

SkillLibrary:
  从多个已验证 repair 中提炼出的长期可复用经验。
```

当前 RoboMex 最大的问题是 Evidence 和 WorldState 之间的升级规则不清楚。SubAgent 说了很多，但不一定被稳定吸收；被吸收的 affordance 又不一定在失败后被 invalidate。

### 4.2 Verifier 不应该只是另一个“判断 SubAgent”

如果 Verifier 只是看图后说“成功/失败”，它会变成另一个不稳定的自然语言模块。ASPIRE 更强调 execution engine 的 trace：每个 primitive 的输入输出、状态码、视觉证据都要被记录，Verifier 应该基于 trace 做状态更新。

RoboMex 中更合理的 Verifier 形态是：

```text
ExecutionTraceVerifier:
  输入 primitive trace + before/after observation + task postcondition
  输出 structured state patch + failed primitive attribution
```

它要回答的不只是“成功了吗”，还要回答：

- 哪个 primitive 失败了？
- 是 grounding 错、grasp pose 错、IK 不可达、运动没到位、还是 release offset 错？
- 哪个 old affordance 应该被降级或废弃？
- 这次失败是否产生可复用 repair pattern？

### 4.3 SubAgent 不应该承担长期专业身份

固定 `Grounding SubAgent`、`Affordance SubAgent` 的设计看起来清晰，但实际容易出现两个问题：

1. Act 把任务扔给 SubAgent 后，SubAgent 生成一堆 claim，Act 仍然不知道该不该信。
2. 专家边界会变得僵硬。例如“碗放偏”既是 placement affordance 问题，也是 held-object frame 问题，也是 verifier/state update 问题。

ASPIRE 给出的更好方向是：让临时 agent 围绕一个 failure trace 或 repair hypothesis 工作，而不是围绕一个永久专家标签工作。

## 5. RoboMex 应该借鉴的架构

我建议 RoboMex 从“SubAgent swarm”转向“Trace-guided Actor + Skill Admission”。

### 5.1 Act 仍然是主 Actor

Act 不是单纯执行器，也不是只会调 SubAgent 的 manager。它应该是主 coding actor：

```text
Act:
  读取 task + current world state + retrieved skills
  写执行代码
  执行 primitive
  读取 trace
  判断是否需要修复
  调用临时 SubAgent 做局部分析
  产出 final state patch / repair report
```

SubAgent 可以存在，但它是临时 worker，不是长期记忆。

### 5.2 新增 Primitive Trace 层

RoboMex 目前的 log 偏“事件流”，但还不够像 ASPIRE 的 primitive trace。应该给每个关键 API 包装 trace：

```text
segment_object:
  input target_name, image, camera
  output bbox, mask, center, confidence, overlays

grasp proposal:
  input points, strategy
  output candidates, selected pose, IK status, overlay

motion:
  input target pose/joints
  output reached pose, error, collision/IK status, before/after image

gripper:
  input open/close command
  output width, contact/lift check, before/after image

release:
  input desired object center, held-object offset
  output final object center, target relation, success/failure
```

然后 Act 和 Verifier 不应该凭自然语言回忆，而应该读这些 trace。

### 5.3 Skill 从“workflow guide”升级为“validated repair”

现有 `perception/affordance/motion/task` taxonomy 可以保留，但 skill 文件应该分成两类：

```text
Human-authored base skill:
  wrapper API、使用约束、基本经验。

Learned repair skill:
  failure signature、when-to-apply、validated repair、origin trace、revalidation status。
```

例如碗放盘子的 skill 不应该只是“用 release_at 补偿 offset”，而应该记录成：

```text
failure:
  bowl is grasped by rim, TCP pose is not object center, direct release at plate center places bowl beside plate.

guard:
  held object is open bowl / plate-like / rim grasp; object_center_offset_from_grasp exists.

repair:
  compute tcp_release_pos = desired_object_center - object_center_offset_from_grasp;
  verify final bowl center relative to plate center, not gripper release pose.

validation:
  passed on LIBERO bowl-on-plate seeds ...
```

这才接近 ASPIRE 的 skill。

### 5.4 增加 Skill Admission

每次 episode 结束后，不应该只保存 summary。应该有一个 post-run reviewer：

```text
RunReviewer:
  读取 trace + world state history + attempt history
  提取 failure signatures
  标记 stale/misleading skill
  生成 candidate repair skill

SkillAuditor:
  检查是否真的被验证
  检查是否过拟合单个物体/seed
  检查 API 是否合规
  决定是否写入 learned skill library
```

这是 ASPIRE 中 coordinator 的核心作用。它不是运行时再多叫几个 SubAgent，而是负责把经验沉淀成可复用知识。

## 6. 不建议照搬的部分

ASPIRE 的 evolutionary search 很强，但对 RoboMex 当前阶段可能太重。它需要大量 rollouts、LLM calls、debug seeds、validation seeds。项目页也承认 search loop 计算成本高，真实机器人还需要 robust success detection、安全 reset、安全监控和标定维护。

所以 RoboMex 不应该马上实现完整 evolutionary search。更合理的路线是：

1. 先做 primitive trace。
2. 再做 structured verifier。
3. 再做 attempt history。
4. 然后做 skill admission。
5. 最后再考虑多候选 program search。

## 7. 对当前 RoboMex 的下一步建议

短期内，我建议停止继续扩充固定 SubAgent 类型，先把以下四件事做扎实：

### Step 1: Trace 化所有关键 wrapper

`segment_object`、`grasp_open_bowl`、`grasp_graspnet`、`find_placement`、`release_at`、`goto_pose`、`close_gripper` 都要输出统一 trace record。

### Step 2: 引入 AttemptHistory

每次抓取/放置都记录：

```text
attempt_id
object
strategy
pose
precondition
postcondition check
success/failure
failure_reason
invalidated_facts
artifacts
```

这可以直接解决“重复用失败 grasp”的问题。

### Step 3: Verifier 产出 state patch，而不是自然语言

Verifier 可以是 SubAgent，但必须输出：

```text
held_object
object_relation
failed_primitive
invalidated_facts
recommended_repair_class
confidence
artifact_refs
```

### Step 4: Skill Library 分层

保留现有 base skills，同时新增：

```text
robomex/skills/learned/
  bowl_rim_grasp_failure_recovery/
  off_center_bowl_release_compensation/
  ambiguous_soup_can_grounding/
```

这些 learned skills 必须来自成功验证，而不是手写猜测。

## 8. 结论

你现在的直觉是对的：RoboMex 当前的问题不是“还缺几个专家 SubAgent”，而是整个系统还没有形成 ASPIRE 式的经验闭环。

更准确地说，RoboMex 应该从：

```text
Skills + fixed SubAgents Swarm
```

转向：

```text
Trace-guided Coding Actor
  + temporary analysis workers
  + structured verifier
  + attempt history
  + validated repair skill library
```

SubAgent 仍然可以存在，但它不应该是方法的核心卖点。真正的核心应该是：机器人执行失败后，系统能否定位失败 primitive，提出修复，验证修复，并把修复沉淀为后续可检索、可迁移、不过拟合的 skill。

这也意味着 RoboMex 的 method 可以更清晰地表述为：

> RoboMex is an embodied coding-agent framework that converts robot execution traces into reusable physical repair skills. Instead of relying on fixed expert agents, it uses temporary coding workers and structured verification to transform failed rollouts into validated, retrievable skills for future manipulation programs.

这个方向比“Agent Swarm + 专家 SubAgent”更稳，也更接近 ASPIRE 给出的证据。
