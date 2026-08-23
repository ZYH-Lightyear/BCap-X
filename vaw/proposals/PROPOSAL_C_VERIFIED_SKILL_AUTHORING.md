# Proposal C：Verified Skill Authoring——部署期"经验→技能"的物理公证自著述

> 定位：独立提案（ICRA/IROS/CoRL 均可；与 A 互补——A 是离线外循环，C 是在线内循环）
> 风险档位：中（无训练依赖；主要风险在技能复用率的实证）
> 谱系：Hermes Agent + Curator（经验自动写成 SKILL.md + 生命周期）+
> Voyager（技能库先例）+ Apodex（验证从推理中剥离）

## 0. 人话版

让 Agent 平时干活时自己记"错题本"：每完成一个任务，如果发现了新解法，
就把它写成一条可复用的经验（如"往容器里放东西前，先定位开口的对角两个点、
算出中心当锚点"）。但行业实测表明 AI 自己写的经验普遍不可信
（SkillsBench：人写的技能 +16.2 分，AI 写的 ≈0），所以本方法的核心是**收录前先做实验**：
每条候选经验拿 N 个同类型但布局不同的新任务做对照重跑，用了确实更好才收进本子；
长期没用或失效的经验定期降级归档。和 A 的区别：A 是寒假集中改总攻略（离线），
C 是平时边干边攒经验（在线），两者互补。

## 1. 一句话

让操作 agent 在部署流中把自己的成功经验自动写成相位索引的技能条目，
但技能的**准入不由自评或使用计数决定，而由物理复验公证**：候选技能在
扰动重放中重新执行、通过验证层级才入库——把 Voyager/Hermes 式
"自著述技能库"从"写了就信"升级为"验证后才信"。

## 2. 概念枢纽

文章里 Hermes 的两个设计（触发式自著述 + Curator 生命周期）是 harness
自进化最轻量的落地形态，但它和 Voyager 共享同一个未解决的问题：
**技能的采信依据是生成它的那个模型的自我判断**（任务完成了、代码跑通了），
这与 Ai2 指出的"评估器与生成器同源"是同一个病灶，只是发生在在线场景。
J-space 的发现（模型能识别自己在被评估）说明自评信号在结构上不可靠。

具身操作再次提供解药：一条候选技能（如"容器放置前，locate 开口左下与
右上两点求心"）可以在**扰动重放**中被重新执行——同任务不同布局/光照/初始位姿
下重跑，用冻结的验证层级（planner 可行性、TCP 误差、env_success）判定其
增益是否稳健。技能准入从"模型觉得有用"变成"物理复验证明有用"。

一句话惊喜（不带方法名）：**自著述技能库的熵增问题，根源不在維护而在准入；
具身域可以把准入做成一个物理实验。**

## 3. 方法概要

### 触发（Hermes 式，确定性）

episode 结束后满足任一条件即触发著述：
(i) env_success 且该任务族历史成功率低于阈值（新解法值得记）；
(ii) 失败但某相位 fitness 显著优于历史（局部进步值得记）；
(iii) advisory ledger 显示重复行为被某个策略打破（如第一次成功委派 Imagination）。

### 著述（LLM，产出受 schema 与内容语法双重约束）

作者模型读该 episode 的事件流 + canvas 截图，写出候选技能条目。
条目分两部分：**agent 运行时可见的只有条件与程序**；出处与验证记录是
curator 专用元数据，对 agent 不可见，且必须是可重跑复核的机器指针，
不是文字自我声明：

```yaml
skill:
  # ---- agent 可见部分：只有条件 → 程序，禁止一切存储量值 ----
  phase: align          # 相位索引，运行时确定性注入（同 playbook 机制）
  precondition: 目标为开口容器且开口可见
  procedure: locate 开口对角两点 → 事件回传坐标求心 → 以求心点为 XY 锚
  # ---- curator 专用元数据：agent 不可见 ----
  provenance:
    source_trace: out/.../m16r_prime/episode_042/   # 原始 trace 指针
    admission_sweep: sweeps/vsa_admit_0007/          # 准入对照实验落盘记录，可重跑复核
  status: candidate     # candidate → verified → active → stale → archived
```

**内容规则（禁 offset）**：技能程序只写工具调用与证据间关系
（对角、中心、同框、可见性），**不得存储绝对量值（cm/度/像素坐标）、
物体名与任务 ID**。一切数值在运行时由感知获取（如 locate 回传坐标）——
技能存储的是"获得数字的方法"，不是数字。该规则写进作者模型的著述指令即可；
扰动准入实验天然淘汰记 offset 的技能（offset 恰恰不随布局变化）。

### 公证（本提案核心，全确定性判据）

候选技能进入**扰动重放队列**：同任务族抽 N 个扰动配置（布局/初始位姿/干扰物），
分别以"注入该技能"与"不注入"跑对照 episode。
准入判据：相位 fitness 均值严格提升且无回归（成功的相位不得变差）。
通过 → `verified`，进入运行时注入池；不通过 → 归档（保留供离线分析，即
Proposal A 的 mutation proposer 语料）。

### 生命周期（Hermes Curator 式）

使用计数、命中率、最近使用代龄；`active → stale → archived` 流转；
定期由小模型合并近重复条目。**Curator 的判据同样物理化**：
stale 技能降级前跑一次小规模重放确认其增益已消失（环境或上游 playbook
变化可能使技能过时）。

## 4. 与 Proposal A/B 的关系

- A 是离线外循环（generation-based，训练集上全局优化 playbook）；
  C 是在线内循环（部署流中逐 episode 积累，无外循环、无训练集依赖）。
  两者共享 playbook schema 与注入机制，代码复用率高。
- C 的归档库（含被拒技能）是 A 的 mutation proposer 的天然语料；
  A 进化出的路由规则决定 C 的技能在什么相位被消费。
- 合并风险：若与 A 同投一个会议，评审会问"两个都是改文本，区别何在"——
  回答是优化时机与信号来源（离线 fitness 搜索 vs 在线扰动重放公证），
  但更稳妥的是错开档期或合并为一篇的两个组件。

## 5. 实验设计

- **主曲线（continual 设定）**：任务流（LIBERO-PRO 任务随机序列）上的
  累计成功率曲线：VSA vs 无技能库 vs "写了就信"（Hermes/Voyager 式准入）。
  预期：无公证的技能库先升后被污染拖平（熵增），VSA 持续上升。
- **迁移**：技能库冻结后在 held-out 任务族上测零著述复用率与成功率增量。
- **准入消融**：公证阈值 N（重放次数）扫描——展示"验证成本 vs 库质量"的
  帕累托面；使用自评准入的技能污染率统计。
- **生命周期消融**：关掉 Curator 跑长任务流，展示库熵增对成功率的侵蚀。
- **定性表**：进化出的技能条目全文展示（可解释性是本方法的展示面）。

## 6. 风险与 cut line

- **复用率风险**：LIBERO-PRO 任务族内多样性不足，技能命中率低 →
  扩展到 spatial/goal suite 混流；仍不足则降级为"技能公证机制 + 污染对照"
  的机制型论文（主曲线换成污染率）。
- **重放成本风险**：每条候选 N 次重放太贵 → 分级公证（先 planner-only
  快筛，通过者再跑全 episode）。
- **与 A 抢故事**：见 §4，档期错开或合并。

## 7. 参考文献

1. 周星星. 自进化（Self-evolving／RSI），一篇就够了. 知乎专栏"AI 煎饼摊", 2026-07-30.
   https://zhuanlan.zhihu.com/p/2065227313973825752
2. Nous Research. Hermes Agent. 开源项目（MIT），2026-02.
   （≥5 次工具调用触发自动写 SKILL.md；Curator 生命周期：活跃→陈旧→归档 + 去重）
3. Wang, G., et al. Voyager: An Open-Ended Embodied Agent with Large Language
   Models. arXiv:2305.16291, 2023.（自著述技能库先例；自验证准入的原型）
4. Apodex. Apodex-1.0: A Verification-Centric Agent Team for Discoverative
   Intelligence. 2026-06-08.（验证从推理中结构性剥离的设计原则）
5. Allen Institute for AI. Rethinking the Evaluation of Harness Evolution for
   Agents. arXiv:2607.12227, 2026.（评估器与生成器同源的病灶；本提案在线场景的
   同型批判与解法）
6. Anthropic. Verbalizable Representations Form a Global Workspace in Language
   Models (J-space). 2026-07-06.（模型自评信号结构不可靠的证据）
7. Zhang et al. Self-Harness. 2026-06.（harness 自著述方向，转引自 1）
8. Shinn, N., et al. Reflexion: Language Agents with Verbal Reinforcement
   Learning. NeurIPS 2023.（经验文本化的先例，无验证准入，Related Work）
9. Zhao, A., et al. ExpeL: LLM Agents Are Experiential Learners. AAAI 2024.
   （跨任务经验抽取，Related Work）
10. Liu, B., et al. LIBERO. NeurIPS 2023.（任务流来源）
11. Lee, H., et al. Recursive Harness Self-Improvement. arXiv:2607.15524, 2026.
    （harness 文本化表示与相位注入机制的共享基础）
