## 论文主题定义：Trace2Skill——从执行轨迹中蒸馏可迁移 Agent Skills
本文的核心主题是 **Trace2Skill：一种将 agent 执行轨迹中的局部经验归纳、合并为可迁移技能目录的框架**。它研究的问题是：当智能体在真实任务中产生大量成功和失败轨迹时，如何把这些轨迹中暴露出的失败模式、成功路径、工具使用习惯和领域操作细节，转化为一个紧凑、可复用、无冲突的 skill，而不是把每条轨迹作为零散 memory 存起来，或按时间顺序反复编辑 skill。

更明确地说，Trace2Skill 不是一个运行时检索记忆系统，也不是简单的“从单条失败轨迹写一条反思”。它的核心范式是：**先收集大量 execution traces，再由多个 analyst 并行提出 trajectory-local patches，最后通过层次化合并把局部 lesson 归纳成一个统一的 skill directory**。

---

## 1. 核心定义
**Trace2Skill 是一种 many-to-one 的技能演化框架**。它把一个 skill 定义为一个人类可读的目录：

\[
\mathcal{S}=(M,\mathcal{R})
\]

其中：

- **\(M\)** 是根文档，通常是 `SKILL.md`，存放高频、通用、直接可执行的程序性知识；
- **\(\mathcal{R}\)** 是辅助资源集合，例如 `references/*.md`、脚本、模板或低频细节说明。

给定一个固定 agent \(\pi_\theta\)、一个初始技能 \(\mathcal{S}_0\)、一组用于演化的任务 \(\mathcal{D}_{evolve}\)，Trace2Skill 的目标是在不更新模型参数的情况下，利用 agent 在这些任务上的执行轨迹构造新技能：

\[
\mathcal{S}^*=\mathcal{E}(\mathcal{S}_0,\mathcal{D}_{evolve};\pi_\theta)
\]

并希望新技能在不相交的测试集 \(\mathcal{D}_{test}\) 上提升 agent 的 pass rate。

这个定义有两个关键点：

1. **模型参数固定**：改进来自 skill，而不是微调模型；
2. **技能是可迁移文件**：最终产物是一个普通 skill directory，可被其他模型、其他规模或相关 OOD 任务复用。

---

## 2. 方法的核心方面
### 2.1 Trajectory Generation：冻结 agent 生成带标签轨迹
Trace2Skill 的第一阶段是生成 execution traces。固定 agent 使用初始技能 \(\mathcal{S}_0\) 在演化任务集上运行，产生多条 ReAct 风格轨迹。每条轨迹通常包含：

- 任务输入；
- agent 的推理和工具调用过程；
- 中间 observation；
- 最终输出；
- 成功或失败标签；
- 在 spreadsheet 场景中，还包括工作目录、输入表格、输出表格和评测结果。

这些轨迹被拆成两类：

- **失败轨迹 \(\mathcal{T}^{-}\)**：用于发现系统性错误、缺失规范和操作陷阱；
- **成功轨迹 \(\mathcal{T}^{+}\)**：用于提取稳定有效的 solution path 和 winning patterns。

这一阶段的重点不是立刻修改技能，而是先构建足够宽的经验池，让后续归纳不被单条轨迹或顺序偏差主导。

---

### 2.2 Error Analyst：从失败轨迹中定位可修复的因果失败
失败轨迹由 error analyst 处理。与单次 LLM 总结不同，Trace2Skill 的 error analyst 是一个 agentic analyst：它可以读取日志、检查工作目录、运行脚本、比较输出和 ground truth，并尝试构造最小修复来验证自己的诊断。

在 spreadsheet 场景中，error analyst 的工作流包括：

1. **理解任务和失败表面**  
   判断 `output.xlsx` 中哪些 cell、range、sheet、format、formula 或 datatype 出错。

2. **追溯失败到 agent 行为**  
   阅读 agent log，定位导致错误的决策、代码、工具调用或错误假设。

3. **最小修复验证**  
   通过修改输出或重做关键转换，生成 `output_fixed.xlsx`。

4. **重新评测**  
   使用 evaluation tool 确认修复是否足以使输出通过。

5. **输出 failure memory items**  
   将具体失败上升为可复用 lesson，例如范围偏移、公式未重算、覆盖原格式、错误排序删除行等。

这个阶段强调 **causal、validated、generalizable**。如果失败无法被可靠解释，相关轨迹不应被转化为 patch，以避免把猜测写进技能。

---

### 2.3 Success Analyst：从成功轨迹中提取精简成功路径
成功轨迹由 success analyst 处理。它通常不需要像 error analyst 那样进行多轮文件检查，而是通过单次或轻量分析提取成功经验。

Success analyst 的目标有两个：

1. **Lean Solution Path**  
   从完整成功日志中剔除失败尝试、绕路、自我纠错和无关探索，只保留最终导致正确答案的最小动作序列。

2. **Success Memory Items**  
   从成功路径中抽象出可复用策略，例如先读取列头再定位范围、写入公式后重算并读回、先验证目标 cell 再提交等。

成功轨迹提供的是“应该强化什么”，失败轨迹提供的是“应该避免什么”。Trace2Skill 可以只用失败、只用成功，也可以组合两者。

---

### 2.4 Parallel Patch Proposal：并行提出 trajectory-local patches
第二阶段的核心是并行 patch proposal。每个 analyst 读取当前冻结的原始 skill snapshot 和一批分析记录，然后提出一个局部 patch。这里的 patch 不是完整重写技能，而是最小、局部、可合并的编辑建议。

Patch 通常包含：

- **reasoning**：解释这个 patch 针对哪些失败或成功模式；
- **edits**：对 `SKILL.md` 或 `references/*.md` 的结构化编辑；
- **changelog_entries**：简短变更说明。

代码中支持的 patch 操作包括：

- `insert_after`
- `insert_before`
- `append_to_section`
- `replace_in_section`
- `add_section`
- `delete_section`
- `create`
- `delete_file`

这个设计有两个重要约束：

1. **patch 是局部的**  
   每个 analyst 只负责从自己的轨迹或小批量轨迹中提出 concise edit，避免一次性大改技能。

2. **patch 面向原始 skill snapshot**  
   并行 analyst 互不依赖，不会因为前一个 patch 改了 skill 而影响后一个 patch 的分析，这避免了顺序编辑中的 order dependence。

---

### 2.5 Hierarchical Patch Consolidation：层次化合并局部补丁
第三阶段是 Trace2Skill 的关键：把大量 trajectory-local patches 合并成一个统一、无冲突、非冗余的 skill update。

合并过程类似 map-reduce：

1. **Map**：每个 analyst 或每个 batch 生成一个 patch；
2. **Reduce Level 1**：每次合并若干 patch，去重、消冲突、保留独特 insight；
3. **Reduce Level 2...L**：重复合并，直到只剩一个 merged patch；
4. **Apply**：把最终 patch 转换成完整文件编辑，应用到 skill directory；
5. **Validate**：检查文件格式和 guardrails，避免破坏 skill。

论文中将层次合并形式化为：

\[
p^{(\ell+1)}
=\mathcal{M}(\pi_\theta,\mathcal{S}_0,\{p_1^{(\ell)},...,p_B^{(\ell)}\})
\]

其中 \(\mathcal{M}\) 是 merge operator。它需要完成：

- **Deduplicate**：合并重复或近似重复的修改；
- **Resolve conflicts**：对同一位置或相互矛盾的建议做取舍或综合；
- **Preserve unique insights**：不同 patch 反映不同失败或成功模式时应尽量保留；
- **Maintain conciseness**：最终技能不能膨胀为轨迹集合；
- **Ensure independence**：最终 edits 不能在同一 passage 上互相冲突；
- **Atomic create/link pairs**：新建 `references/*.md` 时必须同步在 `SKILL.md` 中添加入口，删除文件也要删除对应链接。

这一阶段不仅是机械合并，更是 **inductive generalization**：多条独立轨迹反复提出的类似 patch，说明它很可能是系统性任务规律；只在单条轨迹中出现的细节，则更可能是 idiosyncratic，应被降权或舍弃。

---

### 2.6 Apply and Guardrails：把语义补丁落成真实技能文件
合并后的 patch 还需要被转换为真实文件内容。Trace2Skill 支持 JSON patch 和 markdown semantic patch 两类管线，但最终都会落到 skill directory 的文件编辑上。

这一阶段的约束包括：

- 不允许破坏受保护文件；
- 不允许保留断裂的 reference link；
- 对缺失 section 或无法定位的 target text 要拒绝或跳过；
- 对 overlapping edits 要避免并行冲突；
- 修改后的 `SKILL.md` 和 references 必须保持可读、可用、结构清晰；
- 如果创建辅助文件，必须说明何时读取它，而不是让它成为不可达材料。

因此，Trace2Skill 的最终产物不是一堆独立 memory，而是一个可以直接被 agent 加载的技能目录。

---

## 3. 两种技能演化模式
### 3.1 Skill Deepening：深化已有人工技能
Skill Deepening 从一个已有的 human-written skill 开始。这个 skill 通常已经有不错的通用原则，但可能存在：

- 对某些模型不适配；
- 缺少真实任务中的操作陷阱；
- 对特定工具链细节覆盖不足；
- 对格式、公式、重算、验证等实际 workflow 约束不够具体。

Trace2Skill 通过执行轨迹发现这些缺口，并把它们补充为更具体的 SOP。Deepening 的目标是：**让一个已有技能更稳、更具体、更可迁移**。

---

### 3.2 Skill Creation from Scratch：从弱初稿创建技能
Skill Creation 从一个弱初始技能开始，这个初稿可能来自 LLM 的 parametric knowledge，或者在某些实验中近似为 no skill。它通常缺少真实 execution traces 中才会暴露的操作细节。

Trace2Skill 在这个设置下不是“凭空写技能”，而是先让 agent 在任务中运行，再从成功和失败经验中归纳技能。Creation 的目标是：**把真实轨迹中的经验压缩成一个有用的初始技能，使它达到或接近人工技能的效果**。

---

### 3.3 Error / Success / Combined 三种信号
Trace2Skill 在实验中区分三类演化信号：

- **+Error**：只用失败轨迹，重点补齐会导致任务失败的规则和防错检查；
- **+Success**：只用成功轨迹，重点强化可复用的成功 workflow；
- **+Combined**：同时使用失败和成功轨迹，既学习避免错误，也学习重复有效路径。

在 spreadsheet、math reasoning 和 DocVQA 实验中，Combined 往往能产生更稳健的技能，因为它同时覆盖负面约束和正面过程。

---

## 4. 与其他范式的区别
### 4.1 不同于 Sequential Skill Editing
顺序编辑是在每条轨迹之后立即修改技能。它的问题是：

- 后续修改依赖前面修改的顺序；
- 早期错误编辑可能污染后续分析；
- 多条轨迹中的重复 lesson 难以全局去重；
- 运行耗时长，且不容易并行。

Trace2Skill 的并行 many-to-one consolidation 则先冻结原始 skill，让所有 analyst 并行提出 patch，再统一合并。这更接近人类专家“先看一批案例，再写操作规范”的方式。

---

### 4.2 不同于 Retrieval Memory
Retrieval memory 系统通常把每条轨迹或每条反思存入记忆库，推理时根据 query 检索相关 memory。Trace2Skill 不这样做。

它的核心差异是：

- retrieval memory 在推理时仍依赖检索；
- 检索可能受到 query mismatch、噪声和 OOD 迁移限制；
- memory 是分散的，难以形成统一规范；
- Trace2Skill 将多条轨迹归纳进一个 skill，推理时直接加载 skill，不需要 test-time retrieval。

因此，Trace2Skill 的产物更像 **portable skill file**，而不是 episodic memory database。

---

### 4.3 不同于单次 LLM 失败总结
单次 LLM 失败总结通常只读日志，然后直接写 lesson。Trace2Skill 的 error analyst 更强：它可以检查文件、运行工具、比较输出、构造修复并验证。

这使得 patch 更 grounded：

- 不只是“看起来像错因”；
- 而是被 artifact inspection 和 minimal fix 验证过；
- 最后提炼成对未来 agent 有帮助的 failure memory。

---

## 5. Spreadsheet 场景中的具体机制
Trace2Skill 的开源仓库重点实现了 spreadsheet setting。整体流程如下：

1. **运行 SpreadsheetBench agent**  
   `run_spreadsheetbench.py` 让 agent 使用 `spreadsheet_agent/skills/` 中的技能执行表格任务。

2. **评测输出**  
   `evaluate_with_official.py` 计算任务是否通过。

3. **匹配结果与日志**  
   `analyze_results.py` 把失败或成功结果与 agent logs 对齐。

4. **失败分析**  
   `analysis/run_error_analysis.py` 用 agentic analyst 分析失败轨迹，输出 failure cause 和 failure memory。

5. **成功分析**  
   `analysis/run_success_analysis_llm.py` 从成功日志中提取 lean solution path 和 success memory。

6. **并行技能演化**  
   `skill_evolver/run_parallel_skill_evolution.py` 根据错误记录演化技能；`skill_evolver/run_parallel_combined_skill_evolution.py` 根据错误和成功记录组合演化技能。

7. **新技能评估**  
   将演化后的 skill directory 用于 held-out split，检验它是否真的提升泛化表现。

Spreadsheet agent 使用技能的方式也很直接：`cli_skill_preloaded_agent.py` 会发现 skills 目录中的 `SKILL.md`，读取并去掉 frontmatter，然后把完整技能内容预加载进 system prompt。也就是说，演化后的技能不是在运行时被检索，而是作为 agent 的显式操作指南进入上下文。

---

## 6. 可从哪些视角理解该主题？
### 6.1 从经验归纳视角
Trace2Skill 的本质是从大量具体经验中做归纳。每条 trajectory-local patch 是局部观察，层次化 consolidation 则把这些观察提炼为跨任务可复用的规则。

### 6.2 从技能工程视角
它把 skill 写作从人工经验驱动变成 trace-grounded workflow：agent 先在任务中犯错或成功，然后系统把这些行为证据转成 skill patch。

### 6.3 从知识压缩视角
大量日志、工具调用和表格 artifact 被压缩成少量 SOP、checklist 和 reference files。最终 skill 不保存全部轨迹，只保存对未来有用的程序性知识。

### 6.4 从可迁移性视角
Trace2Skill 强调生成的 skill 应跨模型规模、模型家族和 OOD 任务迁移。技能不绑定到单个 episode，也不要求使用同一个模型参数。

### 6.5 从系统安全性视角
因为 LLM 编辑技能可能引入幻觉或破坏已有规则，Trace2Skill 在 patch 格式、合并、应用、reference link 和验证上加入 guardrails，降低技能被单个坏 patch 污染的风险。

---

## 7. 关键术语与概念
### 7.1 Execution Trace
agent 在任务中的完整执行记录，包括任务输入、推理、工具调用、观察、输出和成功/失败标签。

### 7.2 Trajectory-Local Lesson
从单条或小批量轨迹中抽取出的局部经验。它可能指出一个失败原因，也可能总结一个成功路径。

### 7.3 Patch Proposal
analyst 根据轨迹和当前技能提出的结构化编辑建议。它通常不是完整文件，而是对 `SKILL.md` 或 references 的最小修改。

### 7.4 Parallel Patch Proposal
多个 analyst 并行读取冻结的原始 skill 和不同轨迹记录，独立提出 patches。它避免顺序编辑带来的依赖和污染。

### 7.5 Patch Consolidation
把多个局部 patches 层次化合并为一个 coherent patch 的过程。它负责去重、消冲突、保留独特 insight 并保持技能简洁。

### 7.6 Skill Deepening
从已有人工技能出发，用轨迹经验补齐缺陷、增加约束、提高稳定性和迁移性。

### 7.7 Skill Creation
从弱初稿或 parametric draft 出发，用真实执行经验创建有用技能。

### 7.8 Failure Memory Item
从失败轨迹中抽取的可复用防错经验，要求具有因果性、可验证性和泛化性。

### 7.9 Success Memory Item
从成功轨迹中抽取的可复用正向策略，通常来自精简后的 lean solution path。

### 7.10 Standard Operating Procedure
多条轨迹反复支持的稳定操作规范。Trace2Skill 的最终 skill 往往表现为一组 SOP，而不是零散案例。

---

## 8. 总结性定义
综上，Trace2Skill 可以定义为：

**Trace2Skill 是一种从 agent 执行轨迹中自动演化技能的 many-to-one 框架。它先用固定 agent 和初始 skill 生成成功/失败轨迹，再通过 error analyst 与 success analyst 将单条轨迹中的局部经验转化为结构化 patch，随后用并行 map-reduce 式层次合并将大量 patches 去重、消冲突并归纳为一个统一 skill directory。最终产物不是检索式记忆库，也不是顺序追加的反思列表，而是一个可直接加载、可迁移、无须模型参数更新的程序性技能文件。**

它的研究边界不在于训练更强的基础模型，也不在于保存所有历史轨迹，而在于构建一条从真实执行经验到便携技能的归纳管线：让 agent 的成功和失败不只是日志，而成为未来 agent 可复用的操作规范。
