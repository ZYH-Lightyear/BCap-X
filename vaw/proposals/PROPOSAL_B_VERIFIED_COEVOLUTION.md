# Proposal B：Verified Co-Evolution——物理归因驱动的 Harness 与权重联合自进化

> 定位：**主叙事提案（2026-08-20 定案：先 A 后 B、日夜联进化，讲 B 的故事）**
> ——A 的循环整体成为本文的"文本旋钮"章节，不是两篇拼接；见 §4.5
> 风险档位：中高（训练侧有清晨门控兜底，优雅降级到 A 是结构自带的）
> 谱系：SIA（双旋钮联合优化）+ Continual Harness（内外层分工）+
> 翁荔/Lin et al.（中档模型红利）+ On-Policy Distillation（蒸馏效率）

## 0. 人话版

Proposal A 只改"攻略"（文字），这一篇把"改攻略"和"训练小模型"轮流做。
关键问题是每一轮该干哪个——不靠拍脑袋，靠物理证据归因：失败记录显示
"指令合理但动作总被运动规划器拒绝" = 员工手笨 → 这轮训练小模型；
"好动作明明存在但攻略没引导到" = 手册不好 → 这轮改攻略。
训练数据零人工标注：大模型做精细调整时，每一步都被规划器判过对错，
把"判对"的序列攒起来教 7B 小模型（错的自动被过滤掉），再用"规划器每步判对错"
当奖励做强化学习。最终卖点：部署时精细活只需 7B 小模型，大模型只出现在培养流程里。

## 1. 一句话

在带物理验证的操作 harness 里，"改知识（playbook 文本）"和"改能力（子代理权重）"
可以被同一套验证信号联合驱动：验证层级不仅提供 fitness，还免费提供
**"这次失败该归因于知识还是能力"的调度信号**——这正是 SIA 用一个 LLM
Feedback-Agent 去猜、而具身域可以直接测出来的东西。

## 2. 概念枢纽

SIA（Hebbar et al., 2026）证明了联合更新（W+H）在三个异质领域全面优于只改
harness（H-only）——权重旋钮不是摆设。但它的核心调度器 Feedback-Agent
是一个读轨迹的 LLM：**决定"该拧哪个旋钮"这件事本身没有可信依据**。
Continual Harness（Karten et al., 2026）用固定分工回避了调度问题
（内层高频改流程、外层低频蒸馏补课），代价是弱模型档上机制反而有害
（Flash-Lite 完成度 20% → 3–13%）。

VAW 的验证层级恰好把调度问题变成可测量的归因问题。Imagination 子代理的
每一步位姿编辑都有 planner 即时判定，session 有四类失败枚举，episode 有
env_success。由此可确定性区分两类失败：

```text
知识失败：instruction/路由本身错了
    —— 好的编辑序列存在（教师模型或事后搜索可找到），
       但 playbook 没引导 agent 走到那里          → 拧文本旋钮
能力失败：instruction 合理、模型做不出合规编辑
    —— 同样的 instruction 下，编辑反复被 planner 拒绝
       或 TCP 误差系统性超标                       → 拧权重旋钮
```

**归因信号来自物理验证器而非 LLM 判断**，这是本提案与 SIA 的本质区别，
也是"联合训练"在具身域比在文本域更早可行的原因。

## 3. 方法概要

系统五元组 `A = (M_main, M_imag, K, C, V)`，C（契约）与 V（验证器）冻结：

- **M_main**：大模型，冻结。负责全局任务推理与委派决策。
- **M_imag**：7B 级 VLM（如 Qwen2.5-VL-7B），**可训练**。子问题接口极窄——
  聚焦 canvas 内做厘米级/10° 以内位姿编辑，每步有 planner 即时判定，
  上下文有界。这是 harness 从大问题里切出来的"可蒸馏切面"。
- **K**：相位索引 playbook + 路由规则（文本，可进化，即 Proposal A 的循环）。

### 交替循环（generation-based）

```text
1. 当前 (K_i, θ_i) 在训练任务集上 sweep → 分级 fitness + 失败归因统计
2. 归因调度（确定性规则，非 LLM）：
   知识失败占优 → 文本代：Proposal A 的 mutation-采纳循环更新 K
   能力失败占优 → 权重代：用累积的 verified 数据更新 θ（见下）
3. 每代结束在冻结验收集（held-out 任务 + 冻结 probe）上记录，不参与选择
```

### 权重更新的两条腿（全部零人工标注）

1. **拒绝采样蒸馏（冷启动）**：大模型跑 Imagination 产生的 session 中，
   planner 通过且 session 达成 ready 的编辑序列 = 免费过滤的教师数据。
   On-Policy Distillation 的经验：蒸馏比 RL 省约 10 倍算力，先蒸馏后 RL。
2. **RLVR（精调）**：Imagination 编辑循环天然是 dense verifiable reward 场景——
   每步 planner 判定即 reward，session ready/failed 即 outcome reward，
   与 DeepSeek-R1 的可验证奖励范式同构，但奖励来自运动学而非单元测试。

### 为什么这个组合在经济上成立（翁荔/Lin et al. 论据）

harness-benefit 红利落在中档模型上；7B 子代理 + 厚 harness 恰好是
生产部署形态。若联合训练成立，结论是"大模型只出现在飞轮里
（教师/进化 proposer），部署时不需要它"——这是比成功率更值钱的 claim。

## 4. 与 Proposal A 的关系：依赖而非竞争

- A 的循环就是 B 的文本代；A 的 sweep 落盘的 verified imagination session
  就是 B 的蒸馏语料。**做 A 的每一步都在为 B 积累燃料，零浪费。**
- 当前硬约束：自发 imagination trace 为 0（路由从未触发），
  所以 B 必须排在 A 的路由进化之后——不是取舍，是依赖关系。

## 4.5 合并叙事定案：先 A 后 B，日夜联进化（2026-08-20）

论文只讲一个方法（Verified Co-Evolution），A 作为其文本旋钮章节内嵌。
工程上用**日夜分班**让两个旋钮互不阻塞：

```text
白天（进化班）：文本旋钮跑 A 的 mutation-采纳循环；
              每个 episode 的 verified imagination session 自动落盘进语料库
夜里（训练班）：GPU 空闲时段，在累积语料上拒绝采样蒸馏 7B（LoRA）
清晨（门控）：新 checkpoint 先过 imagination-replay 离线快测（分钟级）——
              赢过现役版本才上岗参加白天 sweep，否则继续用旧版本
归因统计：每代 sweep 输出知识失败/能力失败占比，决定次日资源侧重
          （v1 可固定日夜交替、归因仅作分析；v2 归因驱动调度）
```

**清晨门控是核心 de-risk**：训练侧无论多失败都不可能拖垮进化主线，
最坏情况自动退化为纯 A——优雅降级是结构自带的，不需要 cut line 决策。

**冷启动**：第一晚之前跑一批脚本强制委派的采数据 episode
（只进语料库，不进任何对比表）。

**Claim 阶梯**（"稍微有点效果就可以"的严格化）：

```text
T1（必须）：文本旋钮 held-out 赢过预算对等 TTS      —— A 的主结果，论文地基
T2（头条）：W+H 端到端优于 H-only（SIA 式消融）      —— 成了就是完整故事
T3（保底）：蒸馏 7B 在 Imagination 切面恢复教师 X% 表现且成本降一个量级
          + 归因曲线显示两旋钮各自贡献             —— T1+T3 已足以主张
                                                    "物理验证的联合进化可行"
```

**预算记账**：预算对等表增加 GPU-hours 列，训练开销明示
（联合训练下 Ai2 式攻击必盯此处，主动披露）。

## 5. 实验设计

- **主消融（对齐 SIA）**：W+H vs H-only（=Proposal A）vs W-only vs gen-0，
  LIBERO-PRO 扰动设置 + held-out。预期 W+H > H-only 才成立。
- **归因有效性**：把确定性归因调度换成随机调度/LLM 调度的消融——
  证明"物理归因"这个 hinge 有增量。
- **模型档位分层**（Continual Harness 教训）：M_imag 用 7B 与 2B 两档，
  如实报告弱档可能的负增益。
- **成本表**：部署态（大 Main + 7B Imagination）vs 全大模型，
  token/episode 与 wall-clock。
- **go/no-go 探针（可立即做，零训练）**：imagination-replay——把已有
  session 决策点做成离线重放题给冻结 7B，测 zero-shot 编辑合规率。
  接近零 → 必须先蒸馏；显著非零 → 可先换后训。

## 6. 风险与 cut line

- **数据风险**：verified session 累积量不足以蒸馏 → 用教师模型批量跑
  Imagination-only 采集（把委派改为脚本强制，仅用于采数据，不进对比表）。
- **能力地板风险**：7B 蒸馏后仍过不了 planner 合规率下限 → 降级为
  "验证过滤蒸馏数据集 + 负结果分析"，A 的故事不受影响。
- **档期风险**：训练基建 + 交替代数在 4 周内完不成 → ICRA 只写 A，
  B 的 replay 探针结果进 A 的 conclusion 作为 roadmap 证据。

## 7. 参考文献

1. 周星星. 自进化（Self-evolving／RSI），一篇就够了. 知乎专栏"AI 煎饼摊", 2026-07-30.
   https://zhuanlan.zhihu.com/p/2065227313973825752
2. Hebbar, P., Manawat, Y., et al.（Hexo Labs）. SIA: Self Improving AI with
   Harness & Weight Updates. arXiv:2605.27276, 2026.（已核对 PDF：
   `papers/rsi/2605.27276_SIA_Harness_and_Weight_Updates.pdf`；
   双旋钮联合优化；W+H 全面优于 H-only；Feedback-Agent 调度）
3. Karten, S., Zhang, J., et al.（Princeton）. Continual Harness: Online
   Adaptation for Self-Improving Foundation Agents. arXiv:2605.09998, 2026.
   （已核对 PDF：`papers/rsi/2605.09998_Continual_Harness_Online_Adaptation.pdf`；
   内层高频改 harness、外层 PRM+软 SFT 蒸馏；Gemini Pro 档成本降 40%；
   Flash-Lite 档负增益）
4. Weng, L. Harness Engineering for Self-Improvement. 博客, 2026-07-04.
   （中档模型 harness-benefit 红利）
5. Lin et al. harness-updating / harness-benefit 双轴测量. 2026.（转引自 4）
6. Thinking Machines. On-Policy Distillation. 2025-10-27.
   （forking token；蒸馏 vs RL 约 10 倍算力差；自输出 SFT 的退化警告）
7. DeepSeek-AI. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
   Reinforcement Learning. arXiv:2501.12948, 2025.（可验证奖励 RL 范式）
8. Chen, Z., et al. Self-Play Fine-Tuning (SPIN). arXiv:2401.01335, 2024.
   （自对弈式模型层自进化，Related Work）
9. Lee, H., et al. Recursive Harness Self-Improvement. arXiv:2607.15524, 2026.
   （"harness 作为数据生成组件 / harness-in-the-loop learning"的动机框架）
10. Allen Institute for AI. Rethinking the Evaluation of Harness Evolution for
    Agents. arXiv:2607.12227, 2026.（预算对等协议，B 同样必须遵守，
    且梯度步数需纳入预算轴）
11. Bai, Y., et al.（Qwen Team）. Qwen2.5-VL Technical Report. arXiv:2502.13923,
    2025.（候选 7B 子代理基模）
12. Liu, B., et al. LIBERO. NeurIPS 2023.（任务源）
13. Kim, M., et al. OpenVLA. arXiv:2406.09246, 2024.（部署态成本对照）
