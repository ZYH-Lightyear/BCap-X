# Proposal B：Verified Co-Evolution——案卷诊断驱动的 Harness 与权重联合自进化

> 定位：**主叙事提案**（与 `../MASTER_PLAN_VERIFIED_COEVOLUTION.md` 一一对应）
> 2026-08-20：先 A 后 B、日夜联进化、讲 B
> 2026-08-24 **v2**：诊断综合 / 选择不综合；分析由 VLM 自主；硬锚只守终局配对
> 工程细节以 Master Plan 为准；本文只定故事与 claim
> 谱系：SIA（双旋钮）+ Continual Harness + 翁荔/Lin + On-Policy Distillation + Ai2（预算对等）

## 0. 人话版

Proposal A 只改「攻略」文字；这一篇把「改攻略」和「训练精细活小模型」排成日夜两班。
失败不靠人复盘，也不靠手写相位检测器：VLM 自己读一条案卷（录像 + 轨迹 + 成本），
写出走到哪、哪步错，再改手册的一处。手册有没有真变好，只看同一道题同一随机种子
终局是否多做成——诊断可以很软，过不过必须硬。夜里用「逐步规划通过」的精细动作
序列蒸馏 7B；清晨考不过就不上岗，白天不受影响。卖点：大模型出现在培养流程里，
部署时精细活可以只留 7B。

## 1. 一句话

同一套 episode 案卷驱动两个旋钮：VLM 自主分析并改知识（文本）；
逐步几何合规只用来过滤蒸馏语料（权重）。选择不看软分，只看终局配对——
评估和执行一样是接口问题：harness 把失败编成可审案卷，而不是再造一套
仿真特权「分级物理 fitness」。

## 2. 概念枢纽

SIA（Hebbar et al., 2026）证明 W+H 优于 H-only，但用另一个 LLM 当
Feedback-Agent 决定拧哪颗旋钮——裁判和选手同源。Ai2（2026）证明：
用可优化的软信号做采纳，预算对齐后赢不了多采样。Continual Harness
用固定日夜分工躲开调度，但弱模型档上机制可能有害。

本方法的差异不在「具身免费送不可腐蚀的三级物理分」，而在**分岗**：

```text
诊断（软，VLM 自主）     读 Trace + Video + Cost → 进展与失败叙事
提案（软，VLM 自主）     叙事 → Knowledge 单点修改
选择（硬，公式）         同 (task, seed) 终局成功净增才采纳
训练过滤（局部硬）       planner 逐步通过 → 可入 7B 语料
```

诊断综合，选择不综合。VLM 不给自己打采纳分。planner 不进选择公式。
手写任务相位 / 失败标签不进 proposer。这套案卷在真机上仍然成立
（终局可换成脚本或抽检）；仿真特权里程碑不是方法本体。

知识 vs 能力、该拧哪颗旋钮：v1 **不调度**，固定日夜交替；v2 再考虑
用诊断叙事做资源侧重。与 SIA 的区别首先是「选择不可被 VLM 投票」，
不是「物理器代替 Feedback-Agent 做调度」。

## 3. 方法概要

系统五元组 `A = (M_main, M_imag, K, C, V)`，C（契约）与 V（验证器）冻结：

- **M_main**：大模型，冻结。负责全局任务推理与委派决策。
- **M_imag**：7B 级 VLM（如 Qwen2.5-VL-7B），**可训练**。子问题接口极窄——
  聚焦 canvas 内做厘米级/10° 以内位姿编辑，每步有 planner 即时判定，
  上下文有界。这是 harness 从大问题里切出来的"可蒸馏切面"。
- **K**：相位索引 playbook + 路由规则（文本，可进化，即 Proposal A 的循环）。

### 交替循环（generation-based）

```text
v1（固定日夜，不调度）
  白天：sweep → VLM 诊断案卷 → VLM 改 K → 配对终局净增则采纳
  夜里：累积语料上更新 θ；清晨门控，失败则不上岗
  代末：held-out / probe 只记录，不参与选择

v2（以后）：才用诊断叙事决定次日资源侧重（文本代 vs 权重代）
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
- 夜班依赖白班落盘的 Imagination session；语料不够时用强制委派采数
  （只进语料库，不进对比表）。不是「必须先进化出委派才能开夜班」。

## 4.5 合并叙事定案：先 A 后 B，日夜联进化（2026-08-20）

论文只讲一个方法（Verified Co-Evolution），A 作为其文本旋钮章节内嵌。
工程上用**日夜分班**让两个旋钮互不阻塞：

```text
白天（进化班）：VLM 诊断案卷 → 单点改 K → 配对终局净增才采纳；
              Imagination session 无条件落盘进语料库
夜里（训练班）：累积语料上蒸馏 7B（LoRA）
清晨（门控）：重放快测，赢过现役才上岗，否则白天继续用旧版
v1：固定日夜交替。v2 以后才用诊断叙事做资源侧重。
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
          —— T1+T3 已足以主张「案卷驱动的联合进化可行」
```

**预算记账**：预算对等表增加 GPU-hours 列，训练开销明示
（联合训练下 Ai2 式攻击必盯此处，主动披露）。

## 5. 实验设计

- **主消融（对齐 SIA）**：W+H vs H-only（=Proposal A）vs W-only vs gen-0，
  LIBERO-PRO 扰动设置 + held-out。预期 W+H > H-only 才成立。
- **选择纪律（对齐 Ai2）**：同一套提案，若改用 VLM 偏好当采纳信号，
  预期 held-out / 相对 TTS 变差或持平——用来证明「诊断可以软、选择必须硬」。
  v1 若档期不够，降为 discussion；v2 再跑调度消融。
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
