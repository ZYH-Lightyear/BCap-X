## 论文主题定义：Skill1——通过强化学习统一演化 Skill-Augmented Agents
本文的核心主题是 **Skill1：一种用单一强化学习目标统一训练技能选择、技能利用与技能蒸馏的 Skill-Augmented Agent 框架**。它研究的问题是：当一个 LLM agent 拥有持久化 skill library 时，如何让 agent 不只是被动读取技能，也不只是事后写入经验，而是让“找技能、用技能、写技能”三个能力在同一个策略模型中共同进化。

更明确地说，Skill1 关注的是 **skill lifecycle 的统一演化**：

1. **Skill Selection**：模型生成查询，从技能库中检索候选技能，并重排候选技能；
2. **Skill Utilization**：模型在选中技能的条件下执行多步环境交互；
3. **Skill Distillation**：模型基于当前轨迹反思并写入新的可复用技能。

Skill1 的关键主张是：这三个环节不应该由不同模块、不同 reward 或不同 teacher 分别优化。它们共同决定最终任务是否成功，因此应由同一个 policy、同一个任务结果信号驱动。论文用 **task-outcome reward 的低频趋势与高频变化** 来解决三阶段的信用分配问题。

---

## 1. 核心定义
**Skill1 是一种统一优化 skill-augmented agent 生命周期的 agentic RL 框架**。给定任务 \(x\)、环境状态 \(e\)、以及持久技能库 \(\mathcal{B}\)，agent 的一次完整 rollout 不只是普通的 action trajectory，而是包含四类由同一 policy 生成的内容：

\[
\tau=(q,z,a_1,o_1,\ldots,a_T,o_T,s_{\text{new}})
\]

其中：

- \(q\)：模型生成的自然语言检索 query；
- \(z\)：从技能库中选出的技能；
- \(a_{1:T}\)：模型在环境中的多步动作；
- \(s_{\text{new}}\)：模型从当前轨迹中蒸馏出的新技能；
- \(r(\tau)\)：环境返回的最终任务结果，通常是 success/failure。

Skill1 把 skill 定义为两部分：

- **strategy / strat**：具体策略，告诉 agent 未来如何行动；
- **scenario description / desc**：适用场景，告诉检索器什么时候应该找到这条技能。

这个定义很重要，因为 Skill1 的技能库不是简单的 trajectory buffer。它存储的是经过模型反思压缩后的程序性经验，并且每个技能带有可检索的场景描述和可更新的 utility score。

---

## 2. 方法的核心方面
### 2.1 统一的三阶段 Agent Workflow
Skill1 的完整工作流是：

```text
Task
  -> Query Generation
  -> Skill Retrieval
  -> Skill Re-ranking
  -> Skill-conditioned Multi-turn Execution
  -> Trajectory Reflection / Skill Distillation
  -> Skill Library Update
```

与很多先前 skill-agent 方法相比，Skill1 的核心差异在于：**query、rerank、environment action、distillation output 都由同一个 policy 生成**。这意味着训练时梯度可以作用到 skill lifecycle 的多个生成片段，而不是只更新执行动作部分。

在代码中，这个流程主要体现在 `third_party/Skill1/agent_system/multi_turn_rollout/rollout_loop.py`。该文件把一次 rollout 拆成若干 phase：

- `query`：生成用于检索技能库的查询；
- `rerank`：对检索出来的候选技能排序；
- `play`：在技能条件下执行环境动作；
- `distill`：根据轨迹输出新的技能/lesson。

因此，Skill1 不是“先离线构建技能库，再训练一个使用者”，而是让训练中的同一个 agent 持续扮演 selector、executor 和 distiller。

---

### 2.2 Skill Selection：可训练的 Query Generation 与 Re-ranking
Skill1 的技能选择不是固定检索器单独完成的。它分成两步：

1. **Query generation**  
   policy 根据任务生成自然语言 query：

   \[
   q \sim \pi_\theta(\cdot|x)
   \]

   然后使用冻结 encoder \(\mathcal{E}\) 在 skill library 中按语义相似度检索 top-\(K\) 候选：

   \[
   \mathcal{B}_K=\operatorname{topK}_{s\in\mathcal{B}}\operatorname{sim}(\mathcal{E}(q),\mathcal{E}(s.\text{desc}))
   \]

2. **Re-ranking**  
   policy 看到候选技能后输出一个排序 \(\sigma\)，选择排在最前的技能作为当前 rollout 的指导技能。

这个设计的关键是：冻结 encoder 只负责提供基础候选集合，真正决定“应该怎么问”和“哪个技能最有用”的，是可训练的 policy。

在代码中：

- `build_query_generation_obs()` 构建 query generation prompt；
- `apply_generated_queries()` 解析 `<query>...</query>` 并重新检索技能；
- `build_rerank_obs()` 构建候选技能排序 prompt；
- `apply_rerank_results()` 解析 `<rank>...</rank>` 并重排技能；
- `compute_rerank_rewards()` 用 NDCG 给 rerank 输出打分。

ALFWorld 和 WebShop 中的 prompt 模板分别位于：

- `third_party/Skill1/agent_system/environments/prompts/alfworld.py`
- `third_party/Skill1/agent_system/environments/prompts/webshop.py`

例如 ALFWorld 的 query prompt 要求模型输出一句 `<query>`，而 rerank prompt 要求模型输出 `<rank>3,1,2</rank>` 这样的候选排序。

---

### 2.3 Skill Utilization：在选中技能条件下执行多步任务
选出技能 \(z\) 后，Skill1 把技能文本注入到任务 prompt 中，让 policy 执行多步环境交互：

\[
a_t \sim \pi_\theta(\cdot|x,z.\text{strat},o_{\leq t})
\]

这一步与普通 agentic RL 类似，但区别是 prompt 中包含从 skill library 取出的可复用经验。技能文本通常以如下形式加入上下文：

```text
Relevant skills from the skill library:
...
Warning: These lessons may be outdated. Use them only if they align with your current observation.
```

这条 warning 体现了 Skill1 对技能使用的态度：技能是指导而非硬规则。agent 需要根据当前 observation 判断技能是否适用。

Skill utilization 的 reward 是最直接的任务结果：

\[
R_i^{\text{util}} = r(\tau_i)
\]

也就是说，执行阶段不用额外设计复杂的中间奖励。只要任务完成，当前动作序列就获得正向强化。

---

### 2.4 Skill Distillation：从轨迹中写入可复用技能
每次 rollout 结束后，policy 会进入 distill phase。它看到：

- 当前任务描述；
- 当前轨迹；
- 任务是否成功；
- 可选的成功/失败参考轨迹；
- 结构化 JSON 输出格式。

然后模型需要输出一个新的经验总结，例如：

- 完成了哪些 subtask；
- 哪些 subtask 成功或失败；
- 哪个动作决策导致当前结果；
- 对未来 agent 最有价值的 lesson；
- 这个 lesson 适用于什么一般场景，即 `description_head`。

在代码中，ALFWorld 的 distill prompt 会要求输出：

```json
{
  "subtasks": [
    {"name": "pick_up_object", "description": "...", "status": "completed"}
  ],
  "task_success": true,
  "action_lesson": "...",
  "navigation_lesson": "...",
  "description_head": "..."
}
```

`step_distill()` 会解析 JSON，将有效 lesson 写入 `SkillLibrary.admit()`。在 `first_order_diff` 模式下，代码只在实际成功且 JSON 可解析时写入新技能；论文方法中也强调，新技能只在 \(r(\tau)=1\) 时被 admit 到技能库。

这一步的关键价值是：Skill1 不把完整 trajectory 直接塞进 memory，而是让 policy 学会写短、可检索、可迁移的 strategy。

---

### 2.5 Skill Library：带 Utility 的持久技能库
Skill1 的技能库 \(\mathcal{B}\) 不是静态文档集合，而是随训练动态增长和淘汰的经验库。每条记录大致包含：

- `skill_id`：技能唯一 ID；
- `scenario_desc`：原始任务/场景描述；
- `description_head`：更抽象的适用场景描述；
- `strategy`：可注入 prompt 的技能策略；
- `trajectory`：生成该技能的轨迹；
- `utility_score`：该技能的历史效用；
- `count`：被更新或命中的次数；
- `attempt_type`：来自 success、failure 或 unknown；
- `created_at_step` / `created_at_progress`：技能产生时间。

代码实现位于 `third_party/Skill1/agent_system/memory/memory.py` 的 `SkillLibrary`。其核心操作包括：

- `retrieve()`：按相似度与 utility 组合分数检索 top-\(K\)；
- `admit()`：写入新技能，若 strategy 和场景高度相似则合并；
- `update_utility()`：用 EMA 更新技能效用；
- `_retire()`：当超过容量时淘汰低效且低频技能。

Skill1 默认使用 `sentence-transformers/all-MiniLM-L6-v2` 作为冻结检索 encoder，也支持 TF-IDF fallback。检索评分可以是纯相关性，也可以是 relevance 与 utility/UCB 的组合：

\[
\text{score}(s)=w\cdot\text{relevance}(s)+(1-w)\cdot\text{utility}(s)
\]

当使用 UCB 时，低访问次数但可能有价值的技能会得到探索 bonus，避免技能库过早坍缩到少数热门技能。

---

## 3. 统一信用分配：从同一个任务结果中拆出三类学习信号
Skill1 最核心的方法论是：**所有训练信号都来自同一个 task-outcome reward \(r(\tau)\)，但通过低频趋势和高频变化分配给不同能力**。

### 3.1 Utilization Credit：当前结果直接训练执行能力
执行能力最容易分配信用，因为 action sequence 直接导致任务结果：

\[
R_i^{\text{util}}=r(\tau_i)
\]

这部分用 GRPO 优化，类似普通多轮 agent RL。

---

### 3.2 Selection Credit：用低频趋势训练技能排序
技能选择关心的是“一个技能长期是否有用”，不能只看单次 rollout 的成功或失败。因此 Skill1 为每个技能维护 utility trend：

\[
U(s)\leftarrow(1-\alpha)U(s)+\alpha r(\tau_i)
\]

这相当于用 EMA 估计技能在相关任务上的长期效用。对于一次检索得到的候选集合 \(\mathcal{B}_K\)，Skill1 用各技能的 \(U(s)\) 构造理想排序，再用 NDCG 奖励 policy 的 rerank 输出：

\[
R_i^{\text{rerank}}=\mathrm{NDCG}(\sigma_i,\operatorname{argsort}(-U(\mathcal{B}_K^i)))
\]

低频趋势的作用是：它把 noisy 的单次任务结果平滑成技能质量估计，从而指导 policy 学会选择稳定有效的技能。

---

### 3.3 Distillation Credit：用高频变化训练新技能生成
技能蒸馏关心的是“当前轨迹是否提供了超出现有技能库的新信息”。Skill1 用当前结果减去已检索技能的最高 utility 作为 distillation reward：

\[
R_i^{\text{distill}}=r(\tau_i)-\hat{U}_i
\]

其中：

\[
\hat{U}_i=\max_{s\in\mathcal{B}_K^i}U(s)
\]

直觉是：

- 如果当前 rollout 成功，而检索到的技能 utility 不高，说明当前经验可能填补了技能库的空白，应奖励 distillation；
- 如果已有技能 utility 已经很高，当前成功只是复用了已有知识，蒸馏新技能的边际价值较低；
- 如果当前失败，则不应把这条经验作为成功技能写入库。

这就是论文所说的 **high-frequency variation**：它衡量当前结果相对于技能库趋势的偏差，用来判断新经验是否值得固化为技能。

代码中 `step_distill()` 的 `first_order_diff` 模式对应这一思想：

\[
\text{current\_reward}=r_\tau-u_{\hat{}}
\]

并将该 intrinsic reward 按 `lambda_distill` 加入训练样本。

---

### 3.4 Joint Objective：同一策略的一次联合更新
最终 Skill1 把三部分目标合并：

\[
\mathcal{J}(\theta)=
\mathcal{J}^{\text{util}}(\theta)
+\lambda_1\mathcal{J}^{\text{rerank}}(\theta)
+\lambda_2\mathcal{J}^{\text{distill}}(\theta)
\]

其中：

- utilization 和 distillation 使用 GRPO 风格的组内 advantage；
- rerank 因为每个 rollout 检索到的候选集合不同，论文采用 REINFORCE-style objective；
- query generation 作为 rollout prefix，会通过后续任务成功和 query contrastive reward 获得梯度。

训练脚本中，`algorithm.adv_estimator=grpo`，并通过 `env.rollout.n` 设置每个任务的 group rollout 数。ALFWorld 与 WebShop 的脚本分别位于：

- `third_party/Skill1/launch_scripts/alfworld/train_alfworld.sh`
- `third_party/Skill1/launch_scripts/webshop/train_webshop.sh`

脚本中还启用了：

- `enable_memory=True`
- `enable_query_generation=True`
- `enable_description_head=True`
- `enable_rerank=True`
- `distill_reward_type=first_order_diff`
- `lambda_distill`
- `lambda_rerank`
- `train_retrieve_type=ucb`
- `eval_retrieve_type=greedy`

这些配置对应论文中的完整 Skill1 pipeline。

---

## 4. 与已有 Skill Agent 方法的区别
### 4.1 相比 SkillZero
SkillZero 的目标是 **skill internalization**：训练阶段提供技能，逐步撤除技能，让模型最终在推理时不依赖外部 skill。

Skill1 的目标则是 **skill lifecycle co-evolution**：它保留并持续更新外部 skill library，让模型学会更好地检索、使用和扩展技能库。

可以概括为：

- SkillZero：skills at training, zero at inference；
- Skill1：skills during training and inference, but selector/user/distiller are jointly optimized。

---

### 4.2 相比 Trace2Skill
Trace2Skill 固定 agent 参数，通过离线分析大量成功/失败轨迹来编辑 skill directory。它的核心是 many-to-one 的 skill consolidation。

Skill1 则在线训练 policy 参数，让模型在 rollout 中自己执行 skill selection、utilization 和 distillation。它的核心是通过 RL 信号让三种能力同步变强。

可以概括为：

- Trace2Skill：固定模型，改技能文件；
- Skill1：训练模型，同时演化技能库。

---

### 4.3 相比传统 Memory / Reflexion
传统 Reflexion 或 memory 方法通常把失败教训或成功经验写入记忆，再在后续任务中拼接到 prompt 中。这类方法的问题是：

- 选择机制常常固定或启发式；
- 蒸馏质量依赖 prompt，不直接受 RL 优化；
- 选择、执行、反思可能使用不同 reward 或没有统一 reward；
- memory 容易变长、重复、噪声高。

Skill1 的改进是：

- query 和 rerank 都由 policy 生成并可训练；
- distillation output 是 RL 目标的一部分；
- skill utility 由真实 task outcome 更新；
- 技能库有容量控制、utility 更新和淘汰机制；
- 三个阶段共享最终任务成功信号。

---

## 5. 实验与结论
Skill1 在论文中主要评估两个文本 agent 环境：

- **ALFWorld**：多步家庭任务，需要导航、找物体、拿取、加热、清洗、放置等；
- **WebShop**：网页购物任务，需要搜索、筛选、比较、选择和购买。

论文报告的主结果是：

- ALFWorld 平均成功率达到 **97.5%**；
- WebShop 也超过 prior skill-based 与 RL baselines；
- 相比 GiGPO 这类无显式技能库的 RL-only 方法，Skill1 显示出显式技能复用的增益；
- 相比 RetroAgent / SkillRL 等方法，Skill1 的优势来自选择、利用、蒸馏三个阶段都参与统一优化。

Ablation 的主要结论：

- 去掉 skill library，性能下降最大，说明显式技能库是基础；
- 去掉 selection，说明错误路由会成为下游执行瓶颈；
- 去掉 distillation，技能库质量下降并带来更高上下文成本；
- 设置 \(\lambda_1=0\) 或 \(\lambda_2=0\) 都会降低性能，说明 selection credit 与 distillation credit 互补；
- 同时移除两者下降更明显，证明三阶段能力具有相互依赖关系。

---

## 6. 代码实现中的 Method 对应关系
### 6.1 Workflow 与 rollout
- `agent_system/multi_turn_rollout/rollout_loop.py`  
  负责 query、rerank、play、distill 四个 phase 的 rollout 组织，并把不同 phase 的 reward 写入训练样本。

- `agent_system/reward_manager/episode.py`  
  将每条样本的 episode reward 放在 response 最后一个 token 上，供 PPO/GRPO 更新。

- `verl/trainer/main_ppo.py`  
  创建环境、reward manager、trajectory collector，并使用 `RayPPOTrainer` 执行训练。

### 6.2 Skill library
- `agent_system/memory/memory.py`  
  `SkillLibrary` 实现技能持久化、embedding 检索、utility 更新、admit、retire。

核心字段包括：

- `scenario_desc`
- `description_head`
- `strategy`
- `trajectory`
- `utility_score`
- `count`
- `attempt_type`

### 6.3 环境管理器
- `agent_system/environments/env_manager.py`  
  针对 Search、ALFWorld、WebShop 等环境实现技能检索、技能注入、query 解析、rerank 奖励、distill 解析和技能库更新。

关键方法包括：

- `reset()`：初始化任务并检索技能；
- `build_query_generation_obs()`：构建 query 生成 prompt；
- `apply_generated_queries()`：用模型生成的 query 重新检索；
- `build_rerank_obs()`：构建候选技能排序 prompt；
- `compute_rerank_rewards()`：计算 rerank NDCG；
- `distill()`：构建轨迹反思 prompt，并更新已有技能 utility；
- `step_distill()`：解析模型输出的新技能，计算 distill reward 并写入技能库。

### 6.4 Prompt 模板
- `agent_system/environments/prompts/alfworld.py`
- `agent_system/environments/prompts/webshop.py`
- `agent_system/environments/prompts/search.py`

这些文件定义了：

- 带技能注入的 action prompt；
- trajectory reflection / distillation prompt；
- query generation prompt；
- rerank prompt。

---

## 7. 方法论总结
Skill1 的核心方法论可以压缩为一句话：

> Skill1 将 skill-augmented agent 的选择、执行、蒸馏视为同一条可训练生成链，并通过单一任务结果 reward 的趋势-偏差分解，为三个阶段分配不同信用，从而实现 policy 与 skill library 的共同演化。

更具体地说，它的贡献包括：

1. **统一生命周期建模**  
   把 query、rerank、action、distill 都纳入同一 policy 的 rollout，而不是让技能检索器、执行器和反思器彼此割裂。

2. **单一任务结果信号**  
   不引入额外 teacher 或人工 reward，而是从 \(r(\tau)\) 中派生 selection、utilization、distillation 的训练信号。

3. **低频趋势训练选择**  
   用 skill utility 的 EMA 估计长期有效性，并用 NDCG 奖励 rerank。

4. **高频变化训练蒸馏**  
   用 \(r(\tau)-\hat{U}\) 判断当前经验是否超出现有技能库能力边界，从而奖励有增量价值的新技能。

5. **动态技能库维护**  
   技能库不是静态 prompt collection，而是有检索、效用更新、合并、淘汰和容量控制的持久 memory。

6. **RL 与显式技能复用互补**  
   Skill1 证明参数内化的 RL-only agent 与外部技能库不是二选一。显式技能库能提供跨任务复用，而 RL 让模型学会更好地读、选、写这些技能。

---

## 8. 对 CapX 的启发
对当前 CapX / Code-as-Policy agent 来说，Skill1 最值得借鉴的不是具体的 ALFWorld/WebShop prompt，而是它对 skill lifecycle 的统一训练方式。

CapX 当前已经有若干可类比模块：

- 可执行代码轨迹；
- 成功/失败评测；
- skill library compilation；
- 多轮任务执行；
- perception 与 robot API；
- RL 训练框架。

Skill1 启发的升级方向是：不要只把成功代码片段离线编译成 skill library，而是把 **技能选择、代码执行、轨迹反思/技能生成** 都纳入一个统一 agent loop。

一个面向 CapX 的 Skill1-style 版本可以是：

1. **Skill Selection for Code-as-Policy**  
   模型先生成自然语言或结构化 query，例如“需要先定位透明容器，再估计可抓取姿态，再执行放置校验”，检索已有 manipulation skill。

2. **Skill-conditioned Code Generation**  
   模型在检索到的 semantic action skill 或 code skill 条件下生成 Python policy code。

3. **Trajectory-grounded Skill Distillation**  
   执行后，根据代码 trace、视觉前后状态、API 返回、异常和成功标签，生成新的技能：
   - 何时使用；
   - 关键 perception call；
   - 几何约束；
   - motion primitive 顺序；
   - 常见失败修正。

4. **Outcome-derived Credit Assignment**  
   用任务成功率更新技能 utility，用当前成功是否超过已检索技能的历史 utility 来奖励新技能蒸馏。

这会把 CapX 从“执行代码并收集成功函数”推进到“统一演化 Code-as-Policy skill lifecycle”：模型不只是会写代码，还会学会 **选择哪段经验、如何基于经验写策略、如何把执行轨迹压缩成下一次可复用的技能**。

---

## 9. 局限性
Skill1 本身仍有几个明显限制：

1. **环境主要是文本型**  
   论文只在 ALFWorld 和 WebShop 上验证，还没有证明能直接扩展到视觉、机器人或连续控制环境。

2. **技能库规模固定**  
   论文中技能库容量上限为 5000。更大任务空间可能需要层次化技能库、聚类、版本控制或冲突消解。

3. **技能质量依赖模型反思格式**  
   如果 distill JSON 解析失败，或 lesson 过于具体/过于泛化，技能库质量会受影响。

4. **检索器仍是冻结模块**  
   Skill1 训练 query 和 rerank，但底层 embedding encoder 默认冻结。复杂多模态任务可能需要训练更强的 task-skill retriever。

5. **统一 reward 仍然需要合理的 stage credit 设计**  
   虽然所有信号来自任务结果，但低频趋势和高频变化如何定义，在不同任务域中可能需要调整。

这些限制也说明，如果把 Skill1 思路迁移到 CapX，应重点补足 multimodal retrieval、code/trajectory structured distillation、机器人任务中的 action-level credit assignment。
