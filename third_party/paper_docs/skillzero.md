## 论文主题定义：SkillZero——面向 Skill Internalization 的 In-Context Agentic Reinforcement Learning
本文的核心主题是 **SkillZero / SKILL0：一种通过强化学习将外部 Agent Skills 内化到模型参数中的训练框架**。它研究的问题是：当智能体可以通过外部技能库获得任务指导时，如何避免模型在推理阶段长期依赖技能检索和技能注入，而是在训练过程中逐步把技能中的程序性知识转化为模型自身的行为能力。

更明确地说，SkillZero 不是为了构建一个更强的运行时技能检索系统，而是提出一种 **skills at training, zero at inference** 的范式：训练早期向智能体提供结构化技能上下文，帮助其探索和完成多步任务；训练过程中通过动态课程逐步撤除技能；最终使模型在无技能上下文的情况下仍能执行相应的任务策略。

---

## 1. 核心定义
**SkillZero 是一种以技能内化为目标的 in-context agentic RL 框架**。它把 agent 任务建模为多轮交互过程：模型根据任务指令、历史观察和当前上下文生成动作，环境返回新观察与奖励，直到任务完成或达到最大步数。

SkillZero 的训练目标不是让模型在每一步都“读懂并执行外部 skill”，而是让模型经历一个从依赖技能到摆脱技能的过程：

1. **技能辅助阶段**：模型在 prompt 中看到 general skill 与 task-specific skill，用这些指导完成环境交互；
2. **策略强化阶段**：通过 GRPO 等 agentic RL 目标，把成功轨迹中的行为模式更新到模型参数中；
3. **动态撤除阶段**：定期比较有技能与无技能验证表现，只保留当前策略仍然真正受益的技能；
4. **零技能阶段**：当 skill budget 降为 0 后，训练和验证都在无技能上下文下进行，检验模型是否已经完成内化。

因此，SkillZero 的核心不是“技能调用”，而是 **把技能作为训练时脚手架，用强化学习把脚手架中的程序性知识转移进模型策略**。

---

## 2. 方法的核心方面
### 2.1 SkillBank：离线分组的技能库
SkillZero 首先假设存在一个离线构建的 SkillBank。SkillBank 中的技能不是逐条运行时检索的短片段，而是按任务类别组织的 Markdown 技能文件。

技能库通常包含两层：

- **General skills**：跨任务通用的策略原则，例如系统化探索、证据确认、避免重复操作等；
- **Task-specific skills**：针对特定任务类别的程序性知识，例如 ALFWorld 中的 clean、heat、cool，或 Search-QA 中的 direct retrieval、multi-hop reasoning、compare 等。

在代码实现中，技能文件位于 `third_party/SkillZero/skills/` 下，任务到技能文件的映射由 `skill_mapping.json` 描述。技能文件内部通过 `### SECTION ###` 分段，运行时被解析为可注入 prompt 的文本片段。

这个设计的重点是：SkillZero 不把技能看作一次性检索结果，而是看作可被课程机制选择、保留、撤除的训练资源。

---

### 2.2 In-Context Skill Injection：训练时把技能注入 agent 上下文
在训练 rollout 中，SkillZero 会把当前任务相关的技能内容拼入 prompt。prompt 模板中保留 `{skill_context}` 占位符，环境管理器根据任务类型选择应该注入的技能。

对 ALFWorld 而言，任务类型由 `gamefile` 路径推断，例如 pick、clean、heat、cool 等。对 Search-QA 而言，任务类型来自样本中的 `skill_type`，例如 direct retrieval、multi-hop reasoning 或 compare。

注入后的上下文通常包含：

- general skill；
- 当前任务类别对应的 task-specific skill；
- 当前 observation；
- admissible actions 或可用工具格式；
- 历史交互信息；
- 输出格式约束，例如 `<think>`、`<action>`、`<search>`、`<answer>` 和 `<compression>`。

这一部分对应论文中的 **In-Context Reinforcement Learning (ICRL)**：技能不是推理阶段永久依赖的外部模块，而是在训练阶段作为上下文条件参与策略采样，让 RL 能从更高质量的探索和成功轨迹中学习。

---

### 2.3 Context Rendering：把长历史与技能上下文压缩为视觉输入
多轮 agent 任务会产生很长的交互历史。如果直接把完整文本历史和技能都放进上下文，token 成本会快速增长。SkillZero 因此继承并使用 AgentOCR 的思路：把历史交互渲染成图像，再交给视觉语言模型读取。

在 OCR 模式下，prompt 中包含 `<image>`，图像中呈现过去若干步的 observation、action、search query 或 search result。不同信息类型会用颜色区分，例如：

- ALFWorld 中 observation 与 action 使用不同颜色高亮；
- Search-QA 中 `<search>` 与 `<information>` 使用不同颜色高亮。

同时，模型还需要输出一个 `<compression>` 因子，决定下一步图像压缩程度。压缩越强，视觉 token 成本越低，但图像质量可能下降。SkillZero 用奖励项鼓励模型在任务成功的前提下选择更高效的压缩。

这一部分解决的是 **长上下文效率问题**：让 agent 可以保留多步历史，又不让文本 token 成本成为主要瓶颈。

---

### 2.4 Composite Reward：任务成功奖励与压缩效率奖励
SkillZero 的 reward 由两部分组成：

1. **任务奖励**：环境判断任务是否完成。ALFWorld 主要依据 `won`，Search-QA 主要依据最终答案的 exact match；
2. **压缩奖励**：当轨迹成功时，根据模型选择的压缩因子给予额外奖励。

论文中的压缩奖励形式可以概括为：

\[
\tilde{r}_t = r_t + \lambda \log(c_t)
\]

其中 \(r_t\) 是任务奖励，\(c_t\) 是压缩因子，\(\lambda\) 控制压缩奖励权重。只有成功轨迹才鼓励压缩，避免模型为了降低 token 成本而牺牲任务完成率。

这个 reward 设计体现了 SkillZero 的两个目标：

- 学会完成 agent 任务；
- 学会以更低上下文成本完成任务。

---

### 2.5 Dynamic Curriculum：根据 helpfulness 动态撤除技能
Dynamic Curriculum 是 SkillZero 最关键的方法组件。它解决的问题是：如果训练全程都给完整技能，模型可能只学会“照着 prompt 做”，而不会真正内化；如果一开始不给技能，探索又可能太困难。

SkillZero 的做法是在训练过程中定期评估每类技能对当前策略的帮助程度。每次 validation 会运行两组评估：

1. **with skill**：验证时注入当前技能；
2. **without skill**：验证时关闭技能。

然后对每个任务类别计算：

\[
\Delta_k = Acc_k^{with\ skill} - Acc_k^{without\ skill}
\]

\(\Delta_k\) 可以理解为当前策略对技能 \(k\) 的依赖程度或帮助程度：

- 若 \(\Delta_k > 0\)，说明当前模型在该任务上仍然受益于这个技能；
- 若 \(\Delta_k \le 0\)，说明这个技能已经不再提供明显帮助，或可能成为噪声；
- 随着训练推进，成功内化的技能应当让 \(\Delta_k\) 逐渐趋近于 0。

随后，课程管理器会执行三步：

1. **Filter**：过滤掉非正向帮助的技能；
2. **Rank**：按 \(\Delta_k\) 从高到低排序；
3. **Select**：在当前 skill budget 内保留 top-k 技能。

skill budget 本身按训练阶段逐步下降。例如：

- ALFWorld 使用 `[6, 3, 0]`；
- Search-QA 使用 `[5, 3, 0]`。

当 budget 为 0 时，所有技能都会被移除，模型必须在无技能上下文中继续完成任务。

---

### 2.6 GRPO Agentic RL：把技能辅助下的行为更新进策略
SkillZero 使用基于 veRL 的 agentic RL 训练流程，核心优化器是 GRPO。每个训练 batch 会对同一任务采样多条 rollout，并基于组内回报归一化计算 advantage。

从方法角度看，GRPO 在 SkillZero 中承担的是“内化通道”的作用：

- 技能上下文提高早期探索质量；
- 环境奖励筛选出成功行为；
- 组内 advantage 强化高质量轨迹；
- 动态课程逐步降低外部技能依赖；
- 最终策略在无技能输入下仍保留任务执行能力。

因此，SkillZero 中的 RL 不是简单地在有技能 prompt 上微调模型，而是在不断变化的上下文条件下优化策略，使模型适应从有技能到无技能的分布迁移。

---

## 3. 方法流程
SkillZero 的完整训练流程可以概括为：

1. **构建 SkillBank**  
   离线准备 general skills 与 task-specific skills，并建立任务类别到技能文件的映射。

2. **初始化 agent 环境**  
   根据任务类型创建 ALFWorld 或 Search-QA 环境，加载 prompt 模板、OCR 渲染器和技能映射。

3. **带技能 rollout**  
   在训练早期，agent prompt 中包含技能上下文；模型根据当前 observation、历史图像和可用动作生成下一步。

4. **环境交互与轨迹收集**  
   agent 执行动作，环境返回新 observation、reward、done 和额外信息；系统累计 episode reward、episode length、success rate 等指标。

5. **计算 RL reward**  
   reward manager 把 episode reward 与压缩奖励合并，并将最终分数写入 response 的最后一个有效 token。

6. **GRPO 更新模型**  
   使用组内多轨迹回报计算 advantage，更新 actor 策略。

7. **周期性双验证**  
   每隔固定步数分别运行 with-skill 和 without-skill validation，计算各任务的 helpfulness delta。

8. **更新 active skills**  
   根据 delta 过滤、排序并裁剪技能集合，同时受当前 skill budget 约束。

9. **进入无技能阶段**  
   当 budget 降为 0 后，训练与验证不再注入技能，用于强化和检验真正的 skill internalization。

---

## 4. 主题边界
### 4.1 SkillZero 不是什么？
1. **不是推理时技能增强系统**  
   SkillZero 的目标不是在 inference 阶段继续检索和注入技能，而是训练后尽量去除技能依赖。

2. **不是技能自动生成框架本身**  
   代码中包含技能生成 prompt，但主训练流程默认使用已经准备好的 SkillBank。论文重点在 skill internalization，而不是技能从零生成。

3. **不是单纯的 prompt engineering**  
   技能 prompt 只是训练脚手架，真正目标是通过 RL 把技能行为写入模型策略。

4. **不是纯文本 RL agent**  
   SkillZero 强调视觉上下文压缩，通过 AgentOCR 将历史轨迹渲染成图像，降低长上下文 token 成本。

5. **不是固定课程学习**  
   虽然有 `[6,3,0]` 或 `[5,3,0]` 这样的 budget schedule，但具体保留哪些技能由 with/without skill 的 on-policy validation delta 决定。

---

## 5. ALFWorld 与 Search-QA 中的方法差异
### 5.1 ALFWorld
ALFWorld 是具身文本环境，智能体需要在房间中寻找、拿取、清洁、加热、冷却或放置物体。其方法特点是：

- 任务类型从 `gamefile` 路径中推断；
- action 必须来自当前 admissible actions；
- episode 较长，最大步数通常为 50；
- history length 也较长，因此 OCR 历史压缩更重要；
- skill budget 通常为 `[6,3,0]`；
- success rate 会按 pick、clean、heat、cool 等子任务统计。

### 5.2 Search-QA
Search-QA 是搜索增强问答环境，智能体需要决定继续搜索还是给出最终答案。其方法特点是：

- action 格式为 `<search> query </search>` 或 `<answer> answer </answer>`；
- 任务类型由数据中的 `skill_type` 提供；
- 需要外部 retriever 服务；
- 最大步数较短，通常为 4；
- skill budget 通常为 `[5,3,0]`；
- reward 主要来自最终答案与标准答案的匹配。

二者共享 SkillZero 的核心机制：训练时技能注入、OCR 视觉历史、GRPO 更新、with/without skill 双验证、动态技能裁剪和最终无技能推理。

---

## 6. 可从哪些视角理解该主题？
### 6.1 从技能增强到技能内化
传统 skill-augmented agent 把能力放在外部技能库中，模型在推理时读取技能并执行。SkillZero 改变了能力的归属：技能库只是训练时辅助，最终能力应当转移到模型参数中。

### 6.2 从课程学习视角
SkillZero 的课程不是简单地按时间移除技能，而是结合 on-policy helpfulness。模型还依赖哪些技能，就保留哪些；已经不再帮助模型的技能，就优先撤除。

### 6.3 从探索效率视角
无技能 RL 可能探索困难，完整技能 RL 又可能造成 prompt 依赖。SkillZero 用“先给技能，再逐步撤掉”的方式兼顾早期探索效率和后期自主能力。

### 6.4 从训练-推理一致性视角
如果训练时一直有技能，而推理时没有技能，会产生明显分布差异。SkillZero 通过后期无技能训练显式缩小这个差异。

### 6.5 从上下文效率视角
SkillZero 不只追求更高成功率，也追求更低上下文成本。OCR 历史压缩和压缩奖励共同服务于 token-efficient agent behavior。

---

## 7. 关键术语与概念
### 7.1 Skill Internalization
指模型不再依赖运行时技能文本，而是通过训练把技能中的程序性知识吸收到参数中，并在无技能上下文下表现出相同行为能力。

### 7.2 In-Context Reinforcement Learning
指训练 rollout 时把技能作为上下文提供给策略，让模型在技能辅助下探索和学习，再通过 RL 更新把行为模式固化进模型。

### 7.3 SkillBank
离线组织的技能库，由 general skills 和 task-specific skills 构成。SkillZero 使用 SkillBank 作为训练脚手架，而不是最终推理依赖。

### 7.4 Helpfulness Delta
有技能验证成功率与无技能验证成功率的差值：

\[
\Delta_k = Acc_k^{with\ skill} - Acc_k^{without\ skill}
\]

它用于衡量当前策略是否仍然从某类技能中获益。

### 7.5 Dynamic Curriculum
根据 helpfulness delta 动态选择技能集合，并在训练阶段逐步降低 skill budget 的课程机制。其核心动作是 filter、rank 和 select。

### 7.6 Skill Budget
每个阶段允许保留的最大技能数量。随着训练推进，budget 从完整技能集逐步降到 0，强制模型进入无技能状态。

### 7.7 Context Rendering
将历史 observation、action、search result 等文本交互渲染为图像，使视觉语言模型用较低 token 成本读取长历史。

### 7.8 Compression Reward
当任务成功时，对更高图像压缩因子给予额外奖励，鼓励模型在保证任务完成的情况下减少视觉上下文成本。

---

## 8. 总结性定义
综上，SkillZero 可以定义为：

**SkillZero 是一种面向智能体技能内化的强化学习框架。它将离线技能库作为训练时上下文脚手架，通过 in-context skill injection 提升早期探索和任务完成质量，再利用 GRPO 将成功行为更新进模型参数；同时通过 with/without skill 双验证估计每类技能的 on-policy helpfulness，并在逐步下降的 skill budget 下动态过滤、排序和撤除技能，最终使模型在无技能上下文下仍能完成多轮 agent 任务。**

它的研究边界不在于构建更复杂的运行时技能检索器，而在于提出一种从“外部技能增强”走向“模型自主执行”的训练范式：技能在训练中出现，在推理中消失，但技能所代表的程序性行为被保留在模型策略中。
