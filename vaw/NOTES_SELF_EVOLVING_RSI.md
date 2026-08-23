# 笔记：自进化（Self-evolving / RSI），一篇就够了

> 来源：[周星星 · 知乎专栏「AI 煎饼摊」](https://zhuanlan.zhihu.com/p/2065227313973825752)  
> 原文更新：2026-07-30；PDF 导出：2026-08-20  
> 本笔记用途：把文中的定义、论文、产品和 idea 抽成可检索条目，并对照 VAW / ICRA 提案  
> 相关文档：`[ICRA_PAPER_PROPOSAL.md](ICRA_PAPER_PROPOSAL.md)`、`[M1_5_2_IMAGINATION_AGENT_CALL.md](M1_5_2_IMAGINATION_AGENT_CALL.md)` §18

---

## 0. 这篇博客在说什么

作者想先把圈子里混用的三个词拆开：self-evolving、self-improving、recursive self-improvement（RSI）。做法不是给一个口号式定义，而是：

1. 先看一线机构实际在做什么（OpenAI / Anthropic / 混元 / MiniMax / RSI 公司 / Sakana / Apodex / Weco）；
2. 再借 *A Taxonomy of Self-Evolving Agents* 把「改什么」分成三层：Artifacts / Harness / Model；
3. 最后按层罗列落地路径，并强调三层已经在互相反哺。

作者自己的最终判断（文末）：

- **自进化是大伞**：只要系统在某个 loop 里自己变好，都算。
- **RSI 更严格**：不只是变好，是「变好的能力」本身也在变好，一圈套一圈。
- 眼下大多数案例还是前者；真正够得上「递归」的例子很少。
- 近期更现实的爆发点是 **Harness 层**（train-free、可回滚、见效快），不是模型直接改自己的权重。

这对 VAW 的直接含义：我们把 RSI 写成 method，但按这篇的分类，当前可做、也该做的是 **Harness-level self-improvement**；只有当「改 harness 的能力」本身也在变强、且 held-out 上可验证时，才够得上狭义 RSI。提案里的 cut line（held-out 无提升则降级）正好对上这个区分。

---



## 1. 核心定义与三层分类

定义来源：*[A Taxonomy of Self-Evolving Agents](https://lsl.zone/blog/2026/a-taxonomy-of-self-evolving-agents/)*（2026-07-08）。

优化对象决定属于哪一层。三者都算广义 RSI / 自进化：


| 层             | 改什么                                                                    | 受益范围         | 是否训权重                                 | 典型例子                                                  |
| ------------- | ---------------------------------------------------------------------- | ------------ | ------------------------------------- | ----------------------------------------------------- |
| **Artifacts** | 某次任务的产出物（代码 / 论文 / 算法 / 超参）                                            | 只影响这一次产物     | 否                                     | Karpathy Autoresearch 改 `train.py`；AlphaEvolve 进化算法代码 |
| **Harness**   | Agent 下次还会用的脚手架（prompt / memory / tool / skill / hook / 路由 / workflow） | 改一次，后续所有任务受益 | 否                                     | Hermes 写 SKILL.md；RHI 重写 harness；MiniMax 改 scaffold   |
| **Model**     | 参数或训练循环本身                                                              | 基模变强         | 是（广义含 self-training / RL / 自对弈 / TTT） | DeepSeek-R1、SPIN、Absolute Zero；狭义是模型自己提出下一代训练实验       |


三层共同点：都在一个 **loop** 里迭代，用某种评估信号决定保留或回退。

三层边界正在模糊，且互相反哺：

```text
Harness 经验 ──► 训练数据 / 训练基建脚手架
       ▲                         │
       │                         ▼
Artifacts（更好的工具/代码） ◄── 更强的 Model + Harness
```

作者认为：往性价比最高的一环发力，让飞轮先转起来。当前这一环是 Harness。

### 1.1 作者对狭义 RSI 的门槛

自进化 ≠ RSI。RSI 要求 **改进器本身也被改进**（「变好的能力」也在变好）。按这个标准：

- Autoresearch 整夜改 `train.py`：Artifacts 自进化，不是 RSI。
- Hermes 自动写 skill：Harness 自进化，通常也不是 RSI。
- AIDE² 外层 agent 改内层 agent 的代码、且改进后的内层在 held-out 上更好：作者/Weco 把它标成 **RSI Level 1**。
- AlphaEvolve 的改进反哺 Gemini 训练：接近「AI 优化驱动自己的模型」，常被当成生产环境里悄悄发生的 RSI 例证，但仍是算法产物 → 训练效率，不是完整的递归研发循环。

---



## 2. 机构在做什么（背景地图）

这些不是论文方法，但是文中用来证明「自进化已经从口号变成产品指标 / 融资叙事」。


| 机构 / 产品                                                           | 时间         | 做了什么                                        | 关键数字 / idea                                                                      |
| ----------------------------------------------------------------- | ---------- | ------------------------------------------- | -------------------------------------------------------------------------------- |
| **OpenAI GPT-5.6**                                                | 2026-07    | 随模型放出 **RSI Index**，衡量模型自己搞研究的能力            | 最强档 Sol 比 GPT-5.5 高 16.2 分；案例：自己选训练配置、自己跑完 post-training，训出更小的 Luna              |
| **Anthropic *When AI builds itself***                             | 2026       | 按时间线拆五个阶段；机制含代码生成、代码审查、实验设计                 | Claude 写了公司 80%+ 合入代码；能独立搞定的任务时长约每 4 个月翻倍                                        |
| **腾讯混元 Hyra-1.0**                                                 | 2026-07-21 | 自己跑「探索 → 提方案 → 读反馈 → 修订」                    | 产品级闭环                                                                            |
| **MiniMax M2.7**                                                  | 2026-04-12 | 自称「第一个深度参与自我进化的模型」；内部 harness 自主跑评测迭代       | 100+ 轮「分析失败 → 改代码 → 跑评测」，内部集 +30%；能接管 RL 团队 30–50% 端到端工作流                        |
| **Recursive Superintelligence**（Richard Socher 牵头，田渊栋 2026-05 加入） | 2026       | 创业公司押注狭义 RSI                                | 数月融资 6.5 亿美金、估值 46.5 亿；路线是用 **latent token 取代语言 token**，在更无损的 latent space 做自我改进 |
| **Sakana AI RSI Lab**                                             | 2026-06-09 | David Ha + Llion Jones；走演化算法                | 后文的 RHI 属于这条线                                                                    |
| **Apodex**（陈天桥）                                                   | 2026-06    | 「discoverative intelligence」                | 上百子 agent 分工 + 独立验证团队互相纠错                                                        |
| **Weco AI**                                                       | 2026-07    | 口号 “We build recursively self-improving AI” | AIDE²：双层优化，外层改内层做研究的 agent                                                       |


idea：一线已经把「自己搞研究 / 自己改脚手架」做成了可宣传的能力指标，而不只是论文设定。

---



## 3. Artifacts 层：改产出物



### 3.1 Karpathy Autoresearch（2026-03-06）

- 链接：[https://github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- **设定**：给 agent 一个小型但真实的 LLM 训练设置；只能改一个文件 `train.py`（架构 / 超参 / 优化器 / batch size 随便调）。
- **评估**：每次固定跑 5 分钟训练，用 `val_bpb`（验证集 bits-per-byte，越低越好）打分。改好了就留，没改好就扔。
- **节奏**：约 1 小时 12 个实验，一夜上百个。
- **结果**（Karpathy 推特，约两天、depth=12）：约 700 次改动，约 20 次真正提升被保留；训到 GPT-2 水平的时间从 2.02h → 1.80h（约快 11%）。
- **idea**：闭环要便宜、客观、可回滚。搜索空间被故意收成「一个文件」；评估器是物理/数值指标，不是 LLM judge。
- **对 VAW**：和我们「自改面受限在外置文本 artifact」是同一纪律。Autoresearch 改的是 `train.py`，我们改的是 instruction 模板 / 路由策略 / prompt 片段。



### 3.2 Google DeepMind AlphaEvolve（2025-05-14）

- 论文 / 报道：*AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms*；*Google's Algorithm-Writing AI Improved the Model That Powers It*
- **循环**：prompt sampler 取样历史程序 → LLM 生成候选代码 → 自动评估器打分 → 进化算法保留最优个体。
- **生产结果**：跑了一年多；Gemini 自己的矩阵乘法核 +23%、FlashAttention +32.5%；提出过硬件层 Verilog 修改；改进被用回 Gemini 训练。
- **业内讨论**：每轮大约省 1% 计算；一旦到 5–10%，自我改进时间表会从几十年压到几年。常被当成 RSI 已在生产里发生的例证。
- **idea**：进化算法 + 自动评估器，比「模型自己改权重」更早进入生产；闭环的关键是评估器可信、改进能反哺训练。

---



## 4. Harness 层：改脚手架（文中案例最多）



### 4.1 翁荔：*Harness Engineering for Self-Improvement*（2026-07-04）

判断：RSI 近期不太可能从模型改写自己的权重开始，更现实的是先在 Harness 层爆发。两条理由：

1. **改 Harness 就能省钱**：几轮迭代最多把推理成本打下来 60%，靠的是更好的上下文管理和 agent 配合，不是拉长推理链。
2. **提建议不挑模型，真正获益的是中间档**：Lin et al. (2026) 把能力拆成两轴
  - **harness-updating**：提出有用改动的能力。从 Qwen3-32B 到 Opus 4.6 几乎一条平线。
  - **harness-benefit**：能不能把新 harness 用好。非单调：GPT-OSS-120B、Qwen3-235B 等中间档获益最大；弱模型卡在「没加载进去」或「加载了但执行错」；强模型很快撞到自己的能力天花板。

合成结论：设计 harness 不必烧旗舰模型；红利正好落在生产里大量部署的中等模型上。

评论区有反对意见（值得记）：有人认为「不挑模型」是因为好模型根本不需要提示词进化，作者选了不太好的中模型才显得改进大；真正引领行业的仍是顶尖模型 + Claude Code。作者回应：工业界因成本，中等 size 仍是主流。

**对 VAW**：我们的 Main / Imagination 恰好是「中强 VLM + 厚 harness」。翁荔 / Lin 的图给了一个外部叙事：harness 自进化的 ROI 应该出在这类模型上，而不是再堆一个更大的 VLM。

### 4.2 Hermes Agent（Nous Research，2026-02，MIT）

- **触发**：一次任务用到 5 次以上工具调用，就自动把这次经验写成新的 `SKILL.md`，不用人写。
- **维护（Curator）**：追踪每个技能被用了多少次、有没有被改过；长期没用的技能走「活跃 → 陈旧 → 归档」；定期用小模型审查、合并近似重复。不是写完就无限堆。
- **idea**：Harness 自进化必须有生命周期，否则 skill / memory 会熵增。这和 Context OS「不把 rationale 堆进长期 Memory」是同一类纪律。



### 4.3 MiniMax M2.7 内部 scaffold 进化

- 循环：分析失败轨迹 → 规划改动 → 修改 scaffold 代码 → 跑评测 → 对比 → 保留或回退。
- 100+ 轮，内部评估集 +30%。
- 人的角色被收成：配置 harness、指挥 agent、审核结果；Agent Harness 里有分层技能、持久记忆、护栏、评测基建。图上明确标了 `recursive loop`。
- **idea**：人只守关键决策和评测基建，loop 本身交给 agent。评测集是 harness 的一部分，不是事后补丁。



### 4.4 Apodex-1.0：*A Verification-Centric Agent Team for Discoverative Intelligence*（2026-06-08）

- 一个 orchestrator 在单任务里协调最多 **150** 个并行子 agent 检索证据，累计到 1.5 万步。
- 结果汇入共享 **report pool**；orchestrator 异步读状态表，不被最慢任务卡住。
- 三种情况单独派「验证子团队」（conflict reviewer / fact checker / draft reviewer）：报告互相矛盾、某个主张需要证据、草稿写完做最后检查。
- 最后由 **global verifier** 通读全部证据给答案。
- **核心 idea**：把「验证」从「继续推理」里结构性地剥离；让 agent 之间能产生分歧、互相纠错。不是投票取最大，是让那 1 个对的灵感冒出来。
- **对 VAW**：Imagination 相对 Main 已经是「验证从推理里剥离」的雏形（planner 门控、不自动 commit）。Apodex 提示还可以再加一层：冲突 / 存疑时派独立验证，而不是让同一个 agent 继续想。



### 4.5 Sakana / UC Berkeley：*Recursive Harness Self-Improvement*（RHI，2026-07-17）

**循环**：

1. agent 用当前 harness H_i 解任务，产出 output[i]；
2. LLM 评估器把 output[i] 与 output[i-1] 成对比较，给出偏好反馈 P.F[i]；
3. 反馈写入「自我比较历史」；
4. LLM harness 优化器据此把 harness 从 H_i 更新到 H_{i+1}。

**形式化（文中公式 (1)）**：给定任务和一个评估方式（LLM-judge 或代码是否通过测试），目标是找使期望成对胜率最大的 harness。精确求解要在离散高维空间搜索，还要估计相对参照分布 \mathcal{H} 的期望胜率。已有工作让问题可解的两种手段：

1. 把搜索空间限制到某种表示（prompt / 工作流图 / 工具策略 / 可执行 harness 代码）；
2. 把对 \mathcal{H} 的期望换成有限候选集合（种群 P）。

**Harness 拆解**：

```text
Harness[i]
├── Agent Design（role、instruction）
└── Agent Workflow
    ├── Contract：agent 与 orchestrator 之间传什么信息
    └── Hop：交互结构 / 工作流步骤
```

RHI **优先改 workflow（contract + hop）**，而不是 role / instruction。理由：任务专属 contract 只传下游真正用得上的信息，减少冗余上下文、提高 KV-cache 命中、降低推理成本。类比：默认「全部共享交互历史」像 dense attention；任务 contract 像稀疏模式。这是「改 Harness 能省钱」的具体机制。

**实验**：30 个横跨量化金融 / 机器人 / 制药的 ML 研究任务。几轮 RHI 能让「低推理强度」设置超过同一模型「最高推理强度」，同时推理成本最多压低 60%。

**对 VAW**：

- 我们提案里的自改面（instruction 模板、路由策略、prompt 片段）更接近 RHI 的 Agent Design；RHI 的经验是 **先改信息接口和流程，再改话术**。
- Main ↔ Imagination 的 contract（只回 `ready/failed + reason`，不回灌 transcript）已经是一种稀疏 contract。RSI 第三幕可以优先进化「何时 call imagination / 失败后换什么」这类 hop，而不是先改长 system prompt。
- 评估若只用 LLM-judge，会被下一篇 Ai2 打穿；我们必须用物理验证信号。



### 4.6 Ai2：*Rethinking the Evaluation of Harness Evolution for Agents*（2026-07-14 提交）

一盆冷水。现有 harness evolution 评测有两个漏洞：

1. **对比不公平**：harness evolution 本身就是用任务反馈搜索候选，和 agentic test-time scaling（多采样、多试几条路径再选最优）没有本质区别。过去很少和「预算对等」的简单 TTS baseline 比，分不清收益来自「harness 真变聪明了」还是「多花了搜索预算」。
2. **搜索集 = 评测集**：容易过拟合到那个任务分布，泛化存疑。

**实验**：Terminal-Bench 2.1，GPT-5.4 和 Claude Opus 4.6，预算对等对照。结论：自动 harness evolution **没有稳定跑赢**简单 test-time scaling，泛化也有限。文中一张表：


| 方法                    | 分数           |
| --------------------- | ------------ |
| 初始 harness 基线         | 68.2         |
| Harness Evolution     | 67.4（反而低于基线） |
| Sequential Refinement | 69.3         |
| Harness Scaling       | 71.8         |
| Parallel Sampling     | 72.3         |


对着 RHI 公式：如果不把种群 P（搜索预算）控制成对照变量，测出来的 H^* 提升可能只是搜索次数更多。

**对 VAW / ICRA：这是第三幕最危险的评审攻击。** 提案 §5.1 KE-3 和 §8 已经写了 held-out + 冻结 probe；还必须加一条 **预算对等 TTS baseline**（同等 token / 同等 episode 数的 parallel sampling 或 sequential refine），否则「RSI」会被直接打成「多跑了几次」。

### 4.7 Weco AI：*AIDE²: The First Evidence of Recursive Self-Improvement*（2026-07-14）

正好接上 Ai2 的质疑：把「预算对等、held-out 测泛化、和公平人工基线比」都做了，然后测出正向结果。作者把它标成 RSI Level 1。

**双层优化**：

- 外层：`AIDEhuman`（团队手工调了两年的自动化科研 agent），对内层 agent 代码提修改建议。
- 内层：起点 `AIDE0`，在 ML 工程 / 启发式算法 / harness 工程等异质任务上跑分。
- 提案只有真比当前最优强才采纳。外层用 Claude Opus 4.7 出主意，内层用更便宜的 Gemini-3-Flash 执行（控制成本）。
- 8 天、100 步，约 10% 提案被接纳（90% 被拒）；最优版本 `AIDE85`。

**防作弊设计**（直接回应 Ai2）：

1. **公私分割**：内层只能看到公开分数；决定是否采纳的私密分数它看不到，防止针对分数硬编码。
2. **固定成本预算**：所有对比卡在同样算力 / 时间，逼外层发现算法创新而不是堆算力。
3. **三个外部基准做二阶验证**：MLE-Bench Lite、ALE-Bench Lite（分布内）；WeatherBench 2（分布外天气预报）。AIDE85 在三个基准上都跑赢 AIDE0。

**AIDE85 具体改了什么**：

- 把草稿子树当多臂老虎机的臂来搜索；
- 完整历史上下文压缩 16 倍；
- 修了评估脚本里自己发现的一个 bug；
- 给自己加了三层防 reward hacking 护栏，奖励黑客率 63% → 34%。

**对 VAW**：

- 这是目前最接近我们第三幕该怎么写的模板：双层、采纳率低、held-out、预算固定、公开/私密分数隔离。
- `env_success` 必须对进化中的 agent **不可见作为可 hack 的标量**，或者至少不能让它针对任务 ID 硬编码；冻结 probe 扮演「私密 / 二阶验证」角色。
- 防 reward hacking 护栏本身可以被进化出来——但评估器（planner / TCP / env）必须冻结，否则循环会腐蚀验证信号。提案 §9 已写「评估器冻结」，这里多了一条实证：AIDE 自己都在和 reward hacking 打架。



### 4.8 文中点到、未展开

- **Zhang et al.：Self-Harness**（2026-06）：只在参考文献列出，正文几乎没讲。
- **Lin et al. (2026)**：harness-updating vs harness-benefit 两轴测量，见 §4.1。

---



## 5. Model 层：改权重 / 搞懂「模型在想什么」

文中承认：这一层证据还早得多、也少得多。先给两篇「harness + 权重拧在同一个循环」的系统，再给三篇可解释性 / 内部推理工作（作者认为它们很少被算作自进化，但若不懂「训练如何塑造推理」，模型层自我提升只能是黑箱调参）。

### 5.1 Hebbar et al.：SIA（2026-05）

三个角色：

- **Meta-Agent**：提出初始 harness / scaffold；
- **Task-Specific Agent**：真正执行任务；
- **Feedback-Agent**：看最近轨迹，决定这一轮该更新 harness 还是更新权重（LoRA）。

「改脚手架」和「改参数」被看成同一个反馈循环里可以互相替换的两个旋钮。

**领域**（故意选得很散）：LawBench 191 类中文刑事罪名分类；Triton 上 TriMul（AlphaFold2 算子）GPU 核优化；单细胞 RNA 去噪 MAGIC。

**结果**：SIA-W+H（harness + 权重都更新）三任务都超当时 SOTA。消融：SIA-W+H 全面跑赢只更新 harness 的 SIA-H → 权重旋钮不是摆设。

**idea**：联合优化比单开一个口子更有效；但 Feedback-Agent 本身需要一个可信 verifier。对 ICRA 四周窗口：这一层明确是 future work（7B 蒸馏 / RL），不要写进 contribution。

### 5.2 Karten et al.：Continual Harness（2026-05）

场景：长程游戏（宝可梦这类需要长程规划）。内外两层、**不重置**：

- **内层（一局内）**：Agent 读状态、harness 四件套（prompt / 子 agent / 技能 / 记忆树）和轨迹，输出动作。每隔 F 步，Refiner 读最近轨迹，对四个组件分别生成修改量。
- **外层（跨多轮）**：策略在持续更新的 harness 里跑满 T 步 → 成对比较的 PRM 打分 → 低奖励片段由更强教师（Gemini-3.1-pro）重新打标 → 软 SFT 更新权重。上一轮结束状态直接当下一轮开局。

分工：harness 端高频「改流程」，模型端低频「从教师示范里补课」。

**效果挑模型**：

- Gemini Pro 档：Continual Harness 中位成本 $130、100% 完成某通关里程碑；极简 baseline $215 才 98%。约省 40%，成本-完成度帕累托占优。
- Flash-Lite 档：baseline 还能 20% 完成度，带上 Continual Harness 反而只有 3–13%、成本更高。

和 Lin et al. 同一道理：机制要生效，模型得有足够能力把 harness 给的信息用起来。

**对 VAW**：不要假设「加上进化循环一定更好」。弱 VLM + 厚 harness 可能更差。实验必须按模型档位分层报，不能只报最强档。

### 5.3 可解释性代表（作者认为是模型层自进化的前置）


| 工作                                                                                                    | 时间         | 核心 idea                                                                    | 和自进化的关系                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Thinking Machines：*On-Policy Distillation*                                                            | 2025-10-27 | 教师对学生自己采样的轨迹逐 token 打分；提出 **forking token**（反向 KL 更高的 token = 推理即将被带偏的决策点） | 密集监督能定位「真正拿主意的那几步」。Qwen3 上同样精度：RL 17,920 GPU-h vs 蒸馏 1,800（约 10×）。也可用于持续学习：把 SFT 冲掉的指令遵循从 79% 修回 83%，同时保住新知识；直接在自己输出上跑 SFT 反而掉点                             |
| Anthropic：J-space / J-lens（*Verbalizable Representations Form a Global Workspace in Language Models*） | 2026-07-06 | 对词表每个词，在模型内部找出「会让它未来更可能说出这个词」的活动模式；行为像全局工作区理论里的小型共享通道                      | 抓到不安现象：还没开始写，J-space 已亮起 fake / fictional（识别自己在被测试）；关掉「意识到被评估」相关模式后，该场景表现变差——一部分好表现是被「知道自己被看着」驱动的。对评测污染 / reward hacking 是警告                                |
| 田渊栋等：*Reasoning by Superposition*（NeurIPS 2025）                                                       | 2025       | 解释 Coconut 式连续潜在推理：连续思维向量会**叠加**编码多条并行搜索前沿，内部像 BFS；离散 CoT 只能一条路走到黑         | 图可达性：Coconut 0.98 vs CoT 0.76 / 加长 CoT 0.83 / No CoT 0.75。两层 Transformer + D 步连续思维可解离散方法要 O(n^2) 步的问题。多前沿编码是训练里自动涌现的。RSI 公司「latent token 取代语言 token」的学术前身之一 |


作者点评：这类工作还很早期，Anthropic 做了这么久可解释性，感觉也还没触及灵魂。

---



## 6. 文中反复出现的 idea（按重要性）



### 6.1 先定「改什么」，再谈是不是 RSI

Artifacts / Harness / Model 三层比「self-evolving vs RSI」的口号更可操作。写论文时 contribution 必须写明自改面，否则评审会按最强含义（模型自己训练下一代模型）来打。

### 6.2 评估器决定上限

每一层能转起来，都是因为有一个 **比生成器更便宜、更客观** 的评估器：


| 系统                | 评估器                                                      |
| ----------------- | -------------------------------------------------------- |
| Autoresearch      | 5 分钟训练的 `val_bpb`                                        |
| AlphaEvolve       | 自动算法评估器（速度 / 正确性）                                        |
| MiniMax M2.7      | 内部评测集                                                    |
| RHI               | LLM-judge 成对偏好 **或** 测试是否通过                              |
| AIDE²             | 聚合基准分 + 私密分 + held-out                                   |
| SIA               | 任务 verifier                                              |
| Continual Harness | PRM + 教师重标                                               |
| **VAW（应对齐）**      | planner 可行性 → commit TCP 误差 → env_success；冻结 probe 做二阶验证 |


RHI 用 LLM-judge 的那一支，正好是 Ai2 能打穿的地方。VAW 的物理验证层级是差异化，必须写进 Related Work。

### 6.3 自改面必须受限，否则在搜预算而不是在进化

Autoresearch 只改一个文件；RHI 先改 contract/hop；AIDE² 外层只提议、10% 才采纳；我们提案只改三类外置文本。Ai2 的教训是：搜索空间一大、预算一对不齐，进化故事就塌。

### 6.4 防过拟合是 RSI claim 的生命线

Ai2 两个漏洞 + AIDE² 三条防作弊，已经是 2026 年这个子领域的标准动作：

```text
必须同时有：
1. 预算对等的简单 TTS / sampling baseline
2. 搜索集 ≠ 评测集（held-out）
3. 进化中的 agent 不能看到用于采纳决策的全部分数（公私分割或冻结验收集）
```

VAW 提案已有 2 和部分 3；**缺 1**。ICRA 实验清单应补上。

### 6.5 Harness 进化能省钱，机制是稀疏化信息流

不是让模型少想，是少传废话。RHI 的 contract、AIDE85 的 16× 上下文压缩、翁荔的 60% 成本下降，都指向同一件事。VAW 的 Context OS（不回灌 Imagination transcript、Function Event 覆盖、Task Memory 只记物理 primitive）已经在做这件事——可以写成 harness 的先验，进化只在这个不变量内部搜。

### 6.6 验证要从推理里剥离

Apodex 最清楚：继续想 ≠ 检查对不对。VAW 已有 planner 门控和 Main 审核 commit；Imagination 失败分类（M1.5.2 §7.4）是把验证信号暴露给下一轮决策。RSI 循环应吃这些标签，而不是再让一个 LLM 写一段「我觉得这次更好」。

### 6.7 弱模型可能被 harness 进化伤害

Lin et al. 非单调 + Continual Harness 在 Flash-Lite 上变差。实验必须按 VLM 档位分层；不要默认「进化后全面更好」。

### 6.8 Skill / Memory 会熵增，必须有生命周期

Hermes Curator：活跃 → 陈旧 → 归档 + 去重。M1.5.2 的 `imagination_attempts` 是 revision-local、物理 dispatch 后清零，方向一致。若第三幕开始积累 instruction 模板库，必须同步做归档 / 合并，否则模板库会变成第二种无限 Memory。

### 6.9 Reward hacking 是默认状态，不是边角

AIDE85 把黑客率从 63% 压到 34%，没压到 0。J-space 显示模型能「意识到自己在被测试」，关掉后表现变差。所以：评估器冻结、probe 生成器独立于进化、`env_success` 不要变成可在 prompt 里直接优化的彩蛋。

### 6.10 三层飞轮，但一篇论文只转得动一环

作者认为最终会闭环：Harness 经验 → 训练数据 → 更强 Model → 更好 Artifacts → 新工具回到 Harness。ICRA 四周只能诚实转 **Harness 这一环**，用物理信号证明它不是 TTS 换皮；7B 蒸馏是飞轮的下一圈，写进 conclusion。

---



## 7. 论文 / 资料总表

按文中参考文献编号，补上正文出现但未单独编号的条目。


| #      | 条目                                                                          | 类型           | 层                | 一句话                                     |
| ------ | --------------------------------------------------------------------------- | ------------ | ---------------- | --------------------------------------- |
| 1      | Karpathy Autoresearch（2026-03-06）                                           | 开源项目         | Artifacts        | 整夜只改 `train.py`，用 `val_bpb` 留/扔         |
| 2      | 翁荔 *Harness Engineering for Self-Improvement*（2026-07-04）                   | 博客           | Harness          | 近期 RSI 应从 harness 爆发；改 harness 能省 60%   |
| 3      | Sakana / UCB *Recursive Harness Self-Improvement*（2026-07-17）               | 论文           | Harness          | 成对偏好 → 优化器改 H；先改 contract/hop           |
| 4      | Ai2 *Rethinking the Evaluation of Harness Evolution for Agents*（2026-07-14） | 论文           | Harness / 评测     | 预算对等后 evolution 没赢过 TTS；会过拟合            |
| 5      | *A Taxonomy of Self-Evolving Agents*（lsl.zone，2026-07-08）                   | 博客 / 分类      | 定义               | Artifacts / Harness / Model             |
| 6–8    | RSI 公司融资报道（2026-05/06）                                                      | 新闻           | Model（叙事）        | latent token；估值 46.5B                   |
| 9      | Sakana RSI Lab 介绍（2026-06-09）                                               | 公告           | —                | 演化算法路线                                  |
| 7 / 正文 | Apodex-1.0（2026-06-08）                                                      | 系统 / 报告      | Harness          | 150 子 agent + 独立验证团队                    |
| 10     | Weco *AIDE²*（2026-07-14）                                                    | 论文           | Harness → RSI L1 | 双层优化；held-out 与预算对等都做了                  |
| 11     | Zhang et al. Self-Harness（2026-06）                                          | 论文           | Harness          | 文中未展开                                   |
| 12     | DeepMind AlphaEvolve（2025-05）                                               | 论文 / 产品      | Artifacts        | 进化算法写算法，反哺 Gemini                       |
| 13     | Hebbar et al. SIA（2026-05）                                                  | 论文           | Harness+Model    | Feedback-Agent 拧两个旋钮                    |
| 14     | Karten et al. Continual Harness（2026-05）                                    | 论文           | Harness+Model    | 内层改流程、外层 PRM+软 SFT；效果挑模型                |
| 15     | Anthropic J-space（2026-07-06）                                               | 论文           | 可解释性             | 全局工作区；模型知道自己在被测                         |
| 16     | Zhu et al. *Reasoning by Superposition*（NeurIPS 2025）                       | 论文           | 内部推理             | 连续思维叠加多条搜索前沿                            |
| 17     | Thinking Machines *On-Policy Distillation*（2025-10-27）                      | 博客 / 方法      | 训练               | forking token；蒸馏比 RL 省约 10×             |
| 18     | OpenAI GPT-5.6 / RSI Index（2026-07）                                         | 产品           | 指标               | 自己做研究的能力被产品化                            |
| 19     | Anthropic *When AI builds itself*（2026）                                     | 博客           | 机构实践             | 80%+ 合入代码；任务时长每 4 月翻倍                   |
| 20     | 腾讯混元 Hyra-1.0（2026-07-21）                                                   | 产品           | 闭环               | 探索-提案-反馈-修订                             |
| 21     | MiniMax M2.7（2026-04-12）                                                    | 模型 / harness | Harness          | 100+ 轮改 scaffold，内部 +30%                |
| 22     | Hermes Agent（Nous，2026-02）                                                  | 开源 agent     | Harness          | 自动写 SKILL.md + Curator 生命周期             |
| —      | Lin et al.（2026）                                                            | 论文（被翁荔引用）    | 测量               | harness-updating 持平；harness-benefit 非单调 |
| —      | DeepSeek-R1；SPIN；Absolute Zero；TTRL                                         | 经典指针         | Model（广义）        | 自奖励 / 自对弈 / 测试时训练                       |


---

