# VAW DAY–NIGHT MMSkill Evolution：当前实现

> 状态：已对齐代码（2026-09-03）
> 详细门控与防退化设计见 `SKILL_EVOLVER_ANTI_REGRESSION_PLAN.md`

## 1. 目标与边界

VAW 的递归改进对象是 Agent Harness 中的 MMSkill Library，而不是底层控制器、
极简 System Prompt 或 VLM 权重：

```text
LIBERO-90 DAY rollout
→ TOPReward 导航曲线
→ M3 Trace Investigator + Evidence Reviewer
→ M4 Skill Evolver + Mutation Reviewer
→ inactive candidate generation
→ paired Gate
→ human approval / promotion
→ post-promotion audit / rollback
```

系统不恢复固化任务阶段 evaluator，不建立 PDDL、object tracker 或任务 phase machine。
LIBERO-PRO object/spatial/goal 是 sealed held-out，不进入技能生成和开发 Gate。

## 2. Skill 是唯一知识正文

每个技能的稳定契约仍是短 `SKILL.md`：

```markdown
---
name: 技能名称
description: 当……时使用。
---

# 技能名称

简洁、通用、能直接帮助当前视觉决策的知识。
```

Skill 正文不保存 task ID、seed、绝对场景坐标、成功率、planner telemetry 或环境真值。
Main 每轮只看到完整的轻量技能索引；显式调用 `consult_mmskill(skill_id)` 后才读取正文。

技能可以携带 `0..N` 张历史视觉参考：

```text
skill-id/
├── SKILL.md
└── references/
    ├── index.json
    ├── provenance.json
    └── r001.png ... rNNN.png
```

Reference 不是固定 positive/negative 二元组。每张图描述自己的 `state`、`view`、
`when_to_use` 和 `visual_cue`，且只能复制 M3 已审核的 policy-visible raster，或在 REVISE
时显式保留父代 reference。Runtime 只在技能被 consult 后显示参考板，并始终标注
`REFERENCE · NOT CURRENT`。

## 3. DAY：冻结策略下采集事实

`EvolutionSpec` 冻结：

- LIBERO-90 evolve/gate/reserve split；
- seed；
- Main/Imagination 模型；
- Runtime、Prompt、Canvas、motion backend 与预算；
- GatePolicy。

`python -m vaw.evolution.day` 支持在命名 split 内选择 task 子集和 resume。每个 run 只新增：

```text
run.json
day_index.json
logs/
traces/
```

`day_index.json` 只保存 episode 终局、cost 和 trace 指针。完整 Canvas、Function trace 与
private telemetry 不复制进索引；private 数据也不会进入操作 Agent 或 Skill Evolver Prompt。

## 4. M3：自主发现可学习证据

M3 不按 Function 名称硬切 Decision Window。Episode Atlas 提供 Task、terminal outcome、
TOPReward 原始曲线、物理动作 marker 和稀疏 storyboard。Trace Investigator 通过唯一工具

```text
inspect_segment(start_action, end_action, question)
```

自主缩放并检查局部高分辨率 Canvas。它的上下文是 Atlas、overwrite-only notebook 和最近
视觉证据，而不是不断增长的完整 transcript。最终可输出零条或多条
`span + observation + insight`。

每条 finding 再交给独立 Evidence Reviewer。Reviewer 只判断 observation 是否真正可见、
insight 是否有证据且可迁移；它不写技能、不修复 finding。没有 accepted finding 时，M4
直接输出 `NO_CHANGE`，不会为了跑通流程伪造技能。

## 5. M4：统一 Skill Evolver Agent

M4 不再拆成互不连贯的规则 Writer/Curator。一个 library-aware、read-only Evolver 初始只看：

- 当前轻量 skill index；
- M3 accepted finding 摘要；
- 单 mutation 约束；
- 剩余 inspection budget。

它按需调用：

```text
inspect_evidence(evidence_id)
inspect_skill(skill_id)
inspect_reference(skill_id, reference_id)
```

上下文每轮重建；notebook 只保存已经检查过的短文本，图片只保留最近一张。最终只允许：

```text
NO_CHANGE
ADD one skill
REVISE one skill
RETIRE one skill
```

ADD/REVISE 同时输出最终 `SKILL.md` 和完整 reference 选择；RETIRE 不携带正文。
解析失败、无效引用或模型超时显式失败，不换模型、不生成兜底技能。

Mutation Reviewer 使用全新上下文，只读候选 diff、相关技能与被引用证据，最终只输出
`accept/reject`。只有 accept 后才原子密封 CandidatePackage；Evolver 草稿和 reject 不能进入
generation。

## 6. Candidate 与不可变 Generation

CandidatePackage 保存单项 mutation、M3 provenance、Evolver/Reviewer 原始 request/response、
候选 Skill 和可选 references。manifest digest 覆盖除自身外的全部内容，密封后改写会被拒绝。

GenerationStore 基于父代 snapshot 执行一次 ADD/REVISE/RETIRE：

- ADD：加入新技能目录；
- REVISE：整体替换技能目录，包括 reference 集合；
- RETIRE：只从新 generation 移除技能；
- 父代永远不原地修改。

新 generation 默认 inactive。创建 generation 不会改变在线 Runtime；只有完成 Gate、人工批准并
显式 promote 后，才原子切换 `active_generation` 指针。

## 7. Paired Gate 与成本

Baseline 与 Candidate 必须具有相同 task、seed、init state、模型、Runtime、Prompt、Canvas、
预算和 motion backend，唯一变量是 generation。Gate 首先确认两份 DAY run 的 generation 身份
和冻结配置，再计算：

```text
recovery   = baseline failure → candidate success
regression = baseline success → candidate failure
```

默认正式条件：至少 2 个 recovery、至少 1 个实际 consult 目标技能的归因 recovery、recovery
严格多于 regression，candidate 成功数不低于 baseline。Primary regression 必须使用冻结
confirmation seed 复核；确认退化直接拒绝。基础设施失败产生 `INCONCLUSIVE`，不冒充任务失败。

turn/token/wall-clock 成本只在“双方都成功”的相同 episode 上比较。这样不会把 baseline 根本
没完成的 episode 与 candidate 的成功成本硬比；但能阻止技能让原本都会成功的任务显著变慢
或变贵。turn/token 使用冻结护栏，wall-clock 因共享服务负载敏感而只报告。

## 8. Promotion 与回滚

晋升顺序固定为：

```text
Reviewer ACCEPT
→ inactive generation
→ Gate PASSED
→ human approve
→ atomic promote
```

Promotion 会再次核对 parent 仍是当前 active generation、candidate digest、Gate report hash 和
approval ledger event。后续 DAY 中发现父代成功而当前代失败，只标记 suspect；相同 task 在
confirmation seed 再现才是 confirmed degradation，并按冻结 policy 自动回滚。历史 generation、
candidate 和 audit 记录都不会删除。

## 9. 入口与产物

主要入口：

```bash
python -m vaw.evolution.day ...
python -m vaw.evolution.m3 ...
python -m vaw.evolution.evolve init ...
python -m vaw.evolution.evolve propose ...
python -m vaw.evolution.evolve materialize ...
python -m vaw.evolution.evolve gate ...
python -m vaw.evolution.evolve approve ...
python -m vaw.evolution.evolve promote ...
python -m vaw.evolution.evolve audit ...
python -m vaw.evolution.evolve rollback ...
python -m vaw.evolution.evolve status ...
```

没有自动 `cycle → approve → promote`；真实模型调用、Gate 与晋升保持可单独审查。

## 10. 当前验证状态

- deterministic engineering smoke 已跑通完整
  `M3 fixture → Evolver → Reviewer → Candidate → g001 → Gate → Approval → Promotion → Audit`；
- 真实 LIBERO-90 task 0 / seed 1 trace 已完成 TOPReward、M3 与 M4；Gemini Investigator 检查四段
  证据后没有提出 finding，M4 正确 `NO_CHANGE`；
- 已用另一条经 M3 Reviewer 接受的真实 spatial evidence 跑通 M4 的按需调查、独立 Reviewer
  和 Visual Reference 候选密封；候选保持 inactive，未被当作真实收益；
- 工程 smoke 的脚本化 task outcome 只验证连通性，不代表技能已经产生真实收益；
- 下一项实证工作是积累多个 accepted LIBERO-90 finding，并运行真正的 baseline/candidate
  paired Gate。未经 Gate 与人工 approval，不会把候选设为 active。
