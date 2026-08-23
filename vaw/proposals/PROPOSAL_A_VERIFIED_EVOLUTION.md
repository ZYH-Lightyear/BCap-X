# Proposal A：Verified Evolution——物理验证信号驱动的 Harness 自进化

> 定位：**已并入 Proposal B 的合并叙事（2026-08-20 定案）**——本提案的循环
> 整体成为 B 的"文本旋钮"章节，先行实施；若 B 训练侧无产出则自动降级回本篇独立成文
> 风险档位：低（四周窗口内可完成；cut line 已备）
> 谱系：RHI（进化循环骨架）+ Ai2（评测协议）+ AIDE²（采纳纪律）+ Autoresearch（自改面纪律）

## 0. 人话版

给机器人 Agent 一本"操作攻略"（什么时候该怎么做的文字手册）。每晚让一个编辑模型
看当天的失败录像，对攻略提出一处修改；第二天用新旧两版攻略各跑一遍同一批任务，
用机器人的**物理成绩单**（抓没抓起来、放没放对、花了多少步）判断新版是否真的更好——
更好才保留，否则扔掉。攻略是纯文本、用 git 管理，每次改动可回滚、可阅读。
防作弊：期末考题（held-out 任务）平时锁起来不给看；还要和"不改攻略、单纯多试几次"
的笨办法在同样预算下比，赢了才算数。

## 1. 一句话

具身操作免费提供了文本域 harness 自进化一直缺失的东西——一个比生成器更便宜、
更客观、结构上不可腐蚀的分级评估器（planner 可行性 → 执行误差 → 环境结果）；
用它做 fitness，只进化外置 playbook 文本，就能得到在 held-out 任务上超过
预算对等 test-time scaling 的 harness 自改进。

## 2. 概念枢纽（为什么现在值得写）

文章（周星星，2026）梳理的现状恰好构成一个张力：

- Harness 层是自进化最现实的爆发点（翁荔，2026；RHI 实测低推理档超最高推理档、省 60%）；
- 但 Ai2（2026）证明：预算对等后，harness evolution（67.4）连初始基线（68.2）都跑不过，
  更输给 parallel sampling（72.3），held-out 上仅 +0.6——因为**评估信号（LLM-judge /
  同分布 benchmark 分数）与生成器同源、可被搜索预算冒充**。

张力的解法不在优化器，在评估器。具身操作是唯一一个评估器天然免费且不可腐蚀的
agent 域：运动学 planner 每步给可行性判定、commit 给 TCP 误差、环境给 env_success，
三级信号全部来自物理而非另一个 LLM。**具身不是自我进化更难的场景，而是它的温床。**

## 3. 方法概要

（详细工程落地见 `../M1_6_RSI_READINESS_PLAN.md` §4，理论外壳见 `../RSI_METHOD_DESIGN.md`）

- **自改面（刻意受限，Autoresearch 纪律）**：相位索引 playbook 文本 +
  Imagination 委派路由规则。Contract（API/Canvas/事件语义）、Verifier、模型权重全部冻结。
- **知识内容约束（禁 offset）**：可进化文本只允许"条件 → 程序"形式的三类知识——
  感知策略（何时/如何补证据）、操作程序（动作顺序与验证时机）、
  路由策略（何时委派 Imagination / 何时自行估计）。
  **不得存储绝对量值（cm/度/坐标）、物体名、任务 ID**——数值必须在运行时
  从感知中获取（如 locate 回传坐标后求心）。该规则作为内容契约写进
  mutation proposer 的指令；held-out 评测天然兜底。
  这把 ICRA 提案 §0"进化推理组件而非固定 offset"落成明确的进化空间定义。
- **循环（RHI 轻量化选择）**：轨迹局部自比较——candidate 只和上一代成对比较，
  不搞种群搜索；mutation proposer 读失败 trace 的（相位 fitness、事件流、advisory、
  canvas 截图）提出单点变异。
- **采纳（AIDE² 纪律）**：训练集 sweep 严格提升才采纳，预期低采纳率（AIDE² 为 10%）；
  每代一个 git commit，天然可回滚可审计。
- **防过拟合协议（直接回应 Ai2 的两个漏洞）**：
  1. train/val/held-out 任务划分第 0 天冻结（参照 Ai2 的 45/10/34 比例折算）；
  2. **预算对等 TTS 对照**：同 episode/token 预算的 parallel sampling 与
     sequential refinement；
  3. 公私分割：proposer 不可见 held-out 分数与任务 ID（AIDE² 公私分数隔离的等价物）；
  4. 冻结 probe 集监测"进化是否损害通用空间判断"。

## 4. 天然的 showcase（已有实证支撑）

当前 VAW 的 opus-5（T=0）在 Imagination 可选化后**从未主动委派过一次**（m16q、m16r），
且放置相位因锚点自信错配而失败。第 0 代路由策略的这个缺陷给进化留了一个可解释的
第一步：若进化学会"容器/插入类放置委派 Imagination"，产物是一条人类可读的路由规则——
比任何抽象分数都有说服力的定性结果。

## 5. 实验设计（镜像张力型 formulation：报告张力两端）

- **KE-3 四线图**：进化代数 × {训练 fitness、held-out 成功率、冻结 probe 分、
  预算对等 parallel sampling 水平线}。主 claim = 进化线在 held-out 上压过采样线。
- **主表**：VLA baselines（OpenVLA、π0 级）/ VAW gen-0 / VAW gen-k / TTS 对照，
  在 LIBERO-PRO 扰动设置上，每任务 ≥10 seeds。
- **边界实验**：两档 Main 模型分层报告（Lin et al. 非单调性预期）；进化学不会的
  失败族如实列出（＝可观察性闭包的边界）。
- **可选杀器**：fitness 换成 LLM-judge 跑同代数，展示退化/hack——把 Ai2 的批评
  变成本文论据。

## 6. 风险与 cut line

- 最大风险：KE-3 held-out 无提升 → RSI 降级为 analysis section，
  故事回退为"测量+干预"两幕（KE-1/KE-2 独立成立）。
- 次风险：进化代数不够（sweep 太慢）→ 缩小训练任务集、并行 sweep 基建优先。

## 7. 参考文献

1. 周星星. 自进化（Self-evolving／RSI），一篇就够了. 知乎专栏"AI 煎饼摊", 2026-07-30.
   https://zhuanlan.zhihu.com/p/2065227313973825752
2. Lee, H., Xu, J., Seely, J., Lee, D., Zaharia, M., Tang, Y.
   Recursive Harness Self-Improvement. arXiv:2607.15524, Sakana AI / UC Berkeley, 2026.
3. Allen Institute for AI. Rethinking the Evaluation of Harness Evolution for Agents.
   arXiv:2607.12227, 2026.（Terminal-Bench 2.1 预算对等对照与 45/10/34 held-out 协议）
4. Weco AI. AIDE²: The First Evidence of Recursive Self-Improvement. 2026-07-14.
   （双层优化、10% 采纳率、公私分数隔离、三外部基准二阶验证）
5. Karpathy, A. autoresearch. https://github.com/karpathy/autoresearch, 2026-03.
   （单文件自改面 + 客观数值评估器的纪律）
6. Weng, L. Harness Engineering for Self-Improvement. 博客, 2026-07-04.
7. Lin et al. harness-updating 与 harness-benefit 的双轴测量. 2026.（转引自 6）
8. lsl.zone. A Taxonomy of Self-Evolving Agents. 2026-07-08.
   （Artifacts / Harness / Model 三层定义）
9. Yang, C., et al. Large Language Models as Optimizers (OPRO). arXiv:2309.03409, 2023.
   （Related Work 区分：文本域 prompt 优化）
10. Khattab, O., et al. DSPy: Compiling Declarative Language Model Calls.
    arXiv:2310.03714, 2023.（同上）
11. Liu, B., et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot
    Learning. NeurIPS 2023.（任务源；LIBERO-PRO 为其扰动扩展）
12. Kim, M., et al. OpenVLA: An Open-Source Vision-Language-Action Model.
    arXiv:2406.09246, 2024.（VLA baseline）
13. Black, K., et al. π0: A Vision-Language-Action Flow Model for General Robot
    Control. arXiv:2410.24164, 2024.（VLA baseline）
