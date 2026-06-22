# 多模态Skill -> Better Code as Policy
当前的Capx实际是一个“非常不Agent”的方法, Cap0-Agent的方法实际上一个非常粗糙的, 仅仅是code block执行完的两张图做diff, 来判断下一步coding怎么优化, 这其实是非常不合理的, 两张图能够带来的信息量实际上是非常少的.

我的直观的感受是, Code Block的生成应该进一步细化, 细化到一个 “一句话能够描述清楚的语义动作”, 每个子“语义动作”都会对应一个sub code block, 然后, 可以用一个能力强的VLM, 判断这个语义动作是成功还是失败, 这样, 成功的语义动作就可以被总结为skill library, 反哺code的生成.

此外, Trace2Skill, Skill0是以text驱动的, MMskills是多模态的, 显然, Cap-X的Libero也是多模态的, 那么感知Skill的使用就是这个Agent必须有的能力.

比如, 在一些之前实验中, 我观察到, 代码写的看似没问题, capx的抓取姿态使用graspnet生成的, 有时它的quat就是不对, 导致抓不起来, 那么, 怎么写skill, 运用合理的感知工具, 让这个抓取的dof更合理? 是不是就是skill可以提升的点? 运用好可视化 + VLM的分析能力, 内化成skill是不是好些呢? 这里我想的可能比较理想, 缺乏细节考虑, 但运用点云, 深度, sam, vlm, 在一段coding前提供足够的视觉先验, 我觉得这更加的合理呢? 就比如在“真的动起来前”, 在一个点云内“提前动一遍”, 像MMSkills的branch那样? (这部分我们再一起头脑风暴一下)

此外, 就比如在place时, 通常code都会写移动到那个要place的位置上方 X cm, 再松开夹爪, 但是抬高多少, 只是靠猜, agent自己写code时, 不知道它抓的物体到底多高, 没有高度的概念, 通常抬高的少了, 这就是一个标准的缺乏感知的case.

我想让你帮我, 从/mnt/data/zyh/BCap-X/third_party 中挖掘, 总结, 思考一下, 我该以一个怎样的思路, 将Cap-X升级为一篇全新的高质量Code as Policy Agent Paper.

# v0 Design

当前 Cap-X 的核心问题是：它虽然是 Code-as-Policy 框架，但当前的多轮 agent 形态仍然偏粗粒度。Cap-Agent0 主要是在一个 code block 执行完之后，通过执行前后的图像或视频差分来判断是否需要重新生成代码。这种反馈粒度太粗，因为它只能描述“整段代码执行后环境发生了什么”，却很难回答“是哪一个语义动作失败了、为什么失败、应该把什么经验写入 skill”。

一个更合理的方向是：不要把模型生成的程序视为一个整体 code block，而是进一步拆成一组 **一句话能够描述清楚的 semantic action blocks**。每个 semantic action block 都有明确意图、对应的小段代码、使用的感知工具、执行前条件和执行后验证条件。这样，VLM 和几何工具判断的对象就不再是整段程序，而是“这个语义动作是否成功”。

例如，一个 pick-and-place 任务不应该只是生成一段完整 Python，而应该拆成：

1. 定位目标物体；
2. 分割目标物体；
3. 从 mask 和 depth 构建点云；
4. 估计物体几何尺寸和放置高度；
5. 生成并筛选 grasp pose；
6. 验证 IK / collision / grasp orientation；
7. 执行 grasp；
8. 验证是否抓起；
9. 移动到 place pose；
10. 松爪并验证目标状态。

每个子动作都可以单独执行、单独记录、单独判断成功或失败。成功的子动作可以被总结为 skill；失败的子动作可以被分析为 failure memory 或 patch，反过来改进后续 code generation。

## 核心 Thesis

现有 Code-as-Policy agent 的主要瓶颈不只是“不会写 Python”，而是 **代码生成缺少感知条件、几何约束和动作级验证**。整段 code block 的视觉差分反馈太粗，无法把失败归因到抓取姿态、mask 错误、place 高度、IK 不可达、轨迹碰撞或目标状态误判。

因此，一个新的高质量 Cap-X paper 可以围绕以下主张展开：

**Code-as-Policy should be decomposed into perception-grounded semantic actions, where each action is planned, pre-validated, executed, and post-verified with multimodal evidence. Action-level successes and failures can then be distilled into reusable perceptual skills that improve future code generation.**

中文概括就是：

**把整段代码执行反馈，升级为感知 grounded 的动作级代码生成与动作级 skill 学习。**

## 与现有三类 Skill 工作的关系

### MMSkills 的启发

MMSkills 的关键不是“技能里放图片”，而是把技能表示成状态条件化的多模态程序性知识。它强调：

- 什么视觉状态下该使用这个 skill；
- 什么状态下不该使用；
- 哪些视觉线索说明当前步骤有效；
- 哪些验证线索说明任务完成；
- 视觉证据不是固定坐标，而是状态判断依据。

迁移到 Cap-X / LIBERO 中，skill 不应该只是代码函数，而应该包含感知状态和几何验证。例如 “place object” skill 应该说明：如何用 SAM mask + depth 估计物体高度，如何确定 receptacle surface z，如何设置 approach height，如何在松爪后验证物体是否在目标区域。

### Trace2Skill 的启发

Trace2Skill 的核心是 trajectory-local lesson -> patch proposal -> hierarchical consolidation。它不是顺序地把每条轨迹写进 memory，而是并行分析大量轨迹，再合并成一个统一 skill。

Cap-X 可以借鉴这一点，把每个 semantic action block 的成功/失败记录作为 action-level trace：

- 成功动作产生 success memory；
- 失败动作产生 failure cause 和 failure memory；
- 多个 action-level patches 被合并成 compact skill；
- 最终得到的是 perceptual skill library，而不是散乱的 trial log 或成功代码函数集合。

### SkillZero 的启发

SkillZero 的核心是 skills at training, zero at inference。它说明 skill 可以先作为训练或探索脚手架，再逐步撤掉，让模型内化能力。

对 Cap-X 来说，第一阶段可以先做 training-free 的 skill-augmented agent；后续如果接 CaP-RL，可以考虑：

- 训练早期提供完整 perceptual skill；
- 后期逐渐减少 skill text；
- 验证模型是否在无 skill 或少 skill 条件下仍然执行正确感知检查；
- 将感知 skill 进一步内化到 code policy 中。

但 v0 不建议一开始就把 skill internalization 作为主目标，否则工程量会过大。v0 更适合先证明 action-level multimodal skill 本身能显著提升 Code-as-Policy。

## 方法设计：Semantic Action Block

新系统的基本单元应该是 semantic action block。每个 block 至少包含：

- **Action name**：一句话描述动作，例如 `Estimate object geometry`；
- **Intent**：为什么要执行这个动作；
- **Precondition**：执行前必须满足什么感知/几何条件；
- **Perception tools**：使用哪些工具，例如 SAM3、Molmo/Qwen point prompt、depth、point cloud、GraspNet、IK、CuRobo；
- **Code**：对应的子 code block；
- **Expected output**：这个 block 应该产出什么，例如 mask、OBB、grasp pose、place pose；
- **Validator**：如何判断动作是否成功；
- **Failure modes**：常见失败类型；
- **Skill update hook**：如果失败或成功，应该如何总结成 skill。

示例：

```text
Action: Estimate object geometry
Intent: estimate target object height and center before placing
Precondition: target object is visible in agentview or wrist camera
Tools: SAM3 text prompt, depth, mask_to_world_points, OBB
Expected output: object center, extent, top_z, bottom_z
Validator: mask has sufficient depth coverage; OBB height is within plausible range
Code: ...
```

这一步的重点是让 VLM 和几何规则有明确的判断对象。判断不再是“任务是不是完成”，而是“这个语义动作是不是完成了它承诺的中间目标”。

## 方法设计：Perception Branch / Dry-Run Before Motion

Cap-X 的一个重要升级点是：在机器人真正动起来之前，先运行一个 perception branch。这个 branch 不执行物理动作，只做感知、几何估计和可行性验证。

Perception branch 可以包括：

1. 获取 RGB-D observation；
2. 用 VLM/point prompt 找到目标；
3. 用 SAM3 分割目标；
4. 从 depth + mask 得到目标点云；
5. 估计 OBB、中心、高度、表面法向；
6. 采样 grasp candidates；
7. 检查 grasp orientation、IK feasibility、碰撞风险；
8. 生成 debug overlay / point cloud visualization；
9. 让 VLM 或 rule-based validator 审核是否足够可靠；
10. 只有通过验证后才进入 motion block。

这相当于 robotics 版本的 MMSkills branch loading：主 agent 不直接盲动，而是在分支里加载视觉和几何证据，得到紧凑指导后再执行。

## 方法设计：Action-Level Skill Distillation

当前 Cap-X 的 skill library compilation 主要从成功 trial 中抽取函数。这更像 code utility library，而不是 skill library。它能总结出 `rotation_matrix_to_quaternion`、`mask_to_world_points` 这类有用函数，但很难得到“何时用、如何验证、失败时怎么修”的程序性知识。

新的 skill distillation 应该以 semantic action block 为单位，而不是以完整 trial 或函数为单位。

一个 action-level skill 可以包含：

- **When to use**：什么任务/状态下使用；
- **When not to use**：哪些状态说明不要用；
- **Perception prerequisites**：需要哪些 mask、depth、point cloud、VLM 判断；
- **Procedure**：推荐代码流程；
- **Geometry checks**：硬约束检查；
- **VLM checks**：视觉状态检查；
- **Postcondition**：执行后应满足的状态；
- **Common failures**：从失败 action traces 中总结出的常见问题；
- **Fallbacks**：失败后的替代方案。

这会比“成功代码里的函数集合”更像真正的 robot skill。

## 方法设计：Geometry-Aware Validation

VLM 很重要，但不应该让 VLM 单独判断所有事情。机器人任务里很多错误应该用几何规则判定。

典型 validator 包括：

- quaternion 是否单位化、是否使用正确 wxyz 顺序；
- grasp approach axis 是否合理；
- gripper opening direction 是否与目标 OBB 主轴兼容；
- top-k GraspNet grasp 是否可 IK；
- grasp pose 是否和点云目标对齐；
- place height 是否大于目标表面高度加物体半高和 clearance；
- 轨迹是否碰撞；
- grasp 后目标点云是否随 gripper 移动；
- 松爪后目标中心是否落在 receptacle 区域内。

VLM 更适合判断：

- 分割结果是否对应正确物体；
- 目标是否被遮挡；
- 抓取后物体是否看起来离桌；
- 放置后物体是否在目标容器/区域附近；
- debug overlay 是否显著错误。

最终应该是 **VLM + geometry validators** 的组合，而不是纯视觉差分。

## 关键 Case 1：GraspNet Quaternion / Grasp DOF 失败

你观察到的 GraspNet quat 不对，是一个非常好的 motivating example。当前 agent 往往会盲选最高分 grasp pose，然后直接执行。但最高分并不一定可达、姿态正确或适合当前夹爪。

这可以形成一个 skill：

**Grasp Pose Selection Skill**

核心规则：

- 不要盲选 GraspNet top-1；
- 将 top-k grasp pose 投影回 RGB-D 或点云；
- 检查 grasp approach axis 是否朝向物体；
- 检查夹爪开口方向是否和物体 OBB 主轴兼容；
- 检查 quaternion 顺序和单位化；
- 对 top-k grasp 逐个跑 IK feasibility；
- 若 top-1 不可达或姿态异常，选择满足几何约束的最高分候选；
- 执行前保存 grasp overlay 或点云可视化，供 VLM/validator 审核。

这个 case 的价值在于：代码本身可能没有语法错误，API 调用也看似正确，但缺少感知和几何验证导致执行失败。这正是新方法要解决的问题。

## 关键 Case 2：Place Height 靠猜

Place 高度是另一个非常强的 motivating example。当前 code 常写成“移动到目标位置上方 X cm，再松开夹爪”，但这个 X 往往是猜的。agent 不知道抓取物体有多高，也不知道 receptacle/table surface z，导致放置过低、碰撞、拖拽或提前松爪失败。

这可以形成一个 skill：

**Geometry-Aware Placement Skill**

核心流程：

1. 用 SAM mask + depth 得到被抓物体或目标放置区域的点云；
2. 估计物体 OBB extent，得到物体高度；
3. 估计目标 surface z；
4. 计算 `place_z = surface_z + object_height / 2 + clearance`；
5. 计算 `approach_z = place_z + max(0.05, object_height * ratio)`；
6. 移动到 approach pose；
7. 下探到 place pose；
8. 松爪；
9. 通过目标中心、mask 位置、高度和视觉状态验证放置是否成功。

这个 skill 能体现“感知先验”对 code generation 的直接帮助：不是让模型凭常识猜高度，而是要求它先用点云和 OBB 算高度。

## 系统模块草案

v0 系统可以包含四个模块：

1. **Semantic Code Planner**  
   把任务分解成 semantic action blocks，而不是一次生成整段代码。

2. **Perception Branch Executor**  
   对每个高风险动作先执行感知和几何 dry-run，生成 mask、点云、OBB、grasp candidates、IK feasibility、debug overlays。

3. **Action-Level Verifier**  
   使用 VLM + geometry rules 判断每个 semantic action 是否成功，并标注 failure type。

4. **Multimodal Skill Distiller**  
   借鉴 Trace2Skill，把 action-level success/failure traces 转化为 skill patches，再合并为 perceptual skill library。

## 最小可落地实验

v0 可以先聚焦 LIBERO pick/place 任务，不要一开始覆盖所有 Cap-X 任务。

建议实验设置：

- 选 20-30 个 LIBERO / LIBERO-PRO pick-and-place 类任务；
- 使用 `FrankaLiberoApiReduced` 或 `FrankaLiberoApiReducedSkillLibrary`；
- 打开 video recording、wrist camera、debug overlays 和 line trace；
- 要求模型输出 semantic action blocks；
- 每个 block 保存 intent、code、line trace、video segment、debug overlay、stdout/stderr、reward delta；
- 用 VLM + geometry rules 给 block 标注 success/failure；
- 从成功/失败 action traces 中蒸馏 perceptual skills；
- 在 held-out LIBERO-PRO perturbations 上评测泛化。

## Baselines

可以对比：

1. 原始 Cap-X single-turn；
2. Cap-X multi-turn visual differencing；
3. Cap-X + 当前 function-level skill library；
4. Cap-X + semantic action decomposition；
5. Cap-X + semantic action decomposition + perception branch；
6. Cap-X + semantic action decomposition + perception branch + action-level skill library。

关键 ablation：

- 去掉 VLM verifier；
- 去掉 geometry validator；
- 去掉 perception branch；
- 只用成功 skill；
- 只用失败 skill；
- success + failure combined skill；
- 不用 wrist camera；
- 不用 point cloud / depth，只用 RGB。

## 预期贡献

这篇 paper 可以主打以下贡献：

1. **A new action-level view of Code-as-Policy**  
   将代码生成从 monolithic program 升级为 semantic action blocks。

2. **Perception-grounded code execution**  
   在真实 motion 前引入 perception branch，用多模态感知和几何验证为代码执行提供先验。

3. **Action-level multimodal skill distillation**  
   从动作级成功/失败轨迹中蒸馏可复用 skill，而不是只从完整成功代码中抽函数。

4. **A practical robot manipulation agent improvement on Cap-X / LIBERO**  
   在不训练或少训练模型的情况下，提高 Code-as-Policy 在多模态机器人任务中的成功率、可解释性和泛化能力。

## 一句话定位

这条路线可以被定位为：

**From monolithic Code-as-Policy to perception-grounded semantic action skills.**

或者：

**A multimodal skill framework that makes Code-as-Policy agents plan, verify, and learn at the level of semantic robot actions rather than whole programs.**

# v1 Design：Code 作为桥梁的自进化机器人技能学习

在进一步读完 Skill1、SkillZero、Trace2Skill、MMSkills 之后，我觉得 Cap-X 更有潜力的升级方向不是简单做“VLM + Code-as-Policy + Skill Library”，而是把 **code** 定义为机器人自我演化的核心桥梁：

> Robot uses code as the executable interface between multimodal perception, grasp/control APIs, trajectory understanding, and reusable skill evolution.

换句话说，机器人不是直接从像素到动作学习，也不是只把语言经验拼到 prompt 里，而是通过代码把视觉感知、几何推理、grasp、motion primitive、验证逻辑组织成可执行策略。然后再从专家轨迹、VLA 视频轨迹、以及自身试错轨迹中持续蒸馏新的 multimodal executable skills。

这个方向可以被理解为一个 **self-evolving robot code agent**：机器人通过 code policy 执行任务，通过视觉和环境反馈判断成败，通过轨迹反思更新技能库，并在之后的任务中更好地选择、组合、改写这些技能。

## 1. 更强的一句话定位

原来的 v0 定位是：

> From monolithic Code-as-Policy to perception-grounded semantic action skills.

进一步 sharpen 之后，可以升级为：

> **Code as the interface for self-evolving robot skills.**

或者更完整地说：

> **A VLM Code-as-Policy agent that learns to select, ground, execute, verify, and distill multimodal robot skills through executable code and RL-based lifecycle training.**

这个定位比“多模态 Skill + RL Training”更具体，因为它明确说明：

- code 是中介，不只是输出格式；
- skill 是多模态、可执行、可验证的，不只是文本经验；
- RL 训练的对象是 skill lifecycle，不是低层控制器；
- 轨迹来源可以包括专家演示、VLA 视频、自身试错；
- 自进化发生在 skill selection、code utilization、skill distillation 三个层面。

## 2. 论文核心问题

当前机器人 foundation model / VLA / Code-as-Policy 方法各有短板：

1. **VLA 方法**通常直接学习 observation-to-action 或 observation-to-token policy，泛化强但可解释性和可编辑性弱，失败后很难把经验抽象成可复用程序；
2. **传统 Code-as-Policy**可以生成可读代码，但通常是单次生成，缺少动作级反馈、技能复用和持续学习机制；
3. **文本 skill agent**如 Skill1 / Trace2Skill 已经能选择和蒸馏技能，但主要在文本环境中，skill 缺少真实视觉 grounding、几何约束和机器人执行验证；
4. **当前 Cap-X**有代码执行、感知 API、trace 和 robot task 环境，但还没有形成“选择技能 -> 执行代码 -> 验证动作 -> 蒸馏技能 -> 更新技能库”的闭环。

因此，新的 paper 可以聚焦于：

> How can a robot agent self-evolve executable multimodal skills by using code as the bridge between perception, action, verification, and skill distillation?

这个问题的关键不是“让机器人会更多动作”，而是让机器人学会 **如何组织、复用、验证和更新自己的可执行技能**。

## 3. Multimodal Executable Skill 的定义

为了区别于 Skill1 中的文本 skill，也区别于普通函数库，Cap-X 的 skill 可以定义为 **Multimodal Executable Skill**。

一个 skill 不只是自然语言 lesson，而应该包含：

1. **Scenario / Applicability**  
   这个技能适用于什么任务、物体、视觉条件和几何关系。

2. **Perceptual Preconditions**  
   执行前需要确认什么，例如目标物体是否可见、mask 是否稳定、depth 是否有效、目标是否被遮挡、grasp region 是否可达。

3. **Perception API Plan**  
   应该调用哪些感知工具，例如 SAM、VLM point prompt、depth-to-pointcloud、GraspNet、pose estimator、object height estimator。

4. **Executable Code Sketch**  
   可复用的小段 code template 或 semantic action block，而不是完整 monolithic program。

5. **Geometric Constraints**  
   例如 grasp approach direction、place height、clearance、collision margin、TCP offset、object size estimate。

6. **Success / Failure Verifier**  
   如何判断动作成功，例如物体是否离开桌面、是否在 gripper 中、是否进入目标容器、目标区域视觉状态是否改变。

7. **Recovery Policy**  
   如果失败，应该重新 segment、换 grasp candidate、增加 place height、换 viewpoint，还是重新规划。

因此，一个 skill 的抽象形式可以是：

\[
s = (\text{desc}, \text{visual preconditions}, \text{api plan}, \text{code}, \text{constraints}, \text{verifier}, \text{recovery})
\]

这个定义会是论文的新意之一：它把 skill 从纯文本策略升级为机器人环境中可执行、可感知、可验证的程序性知识。

## 4. 为什么 Code 是关键桥梁

Cap-X 最独特的资产不是“有 VLM”，而是 **有 code execution trace**。这使得它相比纯 VLA 或纯语言 agent 有一个非常重要的优势：

> 机器人执行经验可以被结构化为代码、API 调用、参数、返回值、异常、视觉前后状态和任务结果。

这比直接从视频中学习低层动作更容易蒸馏成 skill。

例如一次失败 grasp 轨迹中，系统可以记录：

- 目标物体文本描述；
- SAM mask；
- GraspNet 返回的候选 grasp poses；
- 被选中的 grasp quaternion；
- IK 是否可达；
- gripper close 后深度图是否显示物体移动；
- wrist camera 中物体是否进入夹爪；
- 最终是否成功拿起；
- 失败发生在哪一行 code / 哪个 semantic action block。

这些信息天然适合被总结成 skill：

> When GraspNet returns a top-down grasp with unstable quaternion for thin or tilted objects, validate approach vector against object principal axis and table normal before executing. If the approach is near-horizontal or collision-prone, re-rank candidates using point-cloud clearance and IK feasibility.

这类 skill 不是普通 VLA 视频轨迹能直接得到的，也不是纯文本 Reflexion 能可靠得到的。它依赖 Cap-X 的 code trace 和感知 API。

## 5. 轨迹来源：专家、VLA 视频与自我试错

这个方向可以把 skill acquisition 分成三类来源。

### 5.1 Expert Code Trajectories

专家写的 Code-as-Policy 轨迹可以作为高质量 seed skills。它们提供：

- 稳定 API 调用顺序；
- 合理的几何参数；
- 成功验证逻辑；
- 常见任务的 code template。

这些轨迹适合作为 skill library 的初始化数据。

### 5.2 VLA / Human Video Trajectories

VLA 或人类视频轨迹可以提供更丰富的视觉先验，尤其是：

- 任务分解顺序；
- 接触前后视觉变化；
- grasp / place 的语义意图；
- 物体状态变化；
- failure recovery 行为。

但视频本身不能直接成为可执行 skill，因此需要 VLM 将视频理解成 semantic action blocks，再映射到 Cap-X 的 API/code skeleton。

这里可以形成一个方法模块：

> Video-to-Code-Skill Distillation：从视频中识别语义动作、视觉前后条件和成功判据，再转化为可执行 code skill。

### 5.3 Self-Trial Trajectories

机器人自身试错轨迹最适合 RL，因为它有明确的任务结果和执行日志。每次尝试后可以产生：

- 成功 skill；
- failure memory；
- verifier patch；
- recovery strategy；
- skill utility update。

自我试错不是为了让机器人盲目探索低层动作，而是为了优化高层 code skill lifecycle。

## 6. 哪些点真正值得 Train

这个方向里最重要的是：不要为了 RL 而 RL。应该只训练那些 **可归因、可验证、对任务成功有长期收益** 的点。

### 6.1 Train Skill Selection

模型需要学会当前任务该检索和调用哪些 skill。

例如“把碗从柜子里拿出来放到水槽”不应该只检索 generic pick-place，而应该检索：

- open cabinet；
- locate bowl in container；
- grasp concave object；
- place into sink basin；
- verify object inside receptacle。

训练信号可以来自：

- 选中 skill 后的任务成功率；
- selected skill utility；
- semantic action block 成功率；
- 是否减少 retry 次数。

### 6.2 Train Skill Grounding

同一个 skill 在不同视觉场景中需要绑定到不同 mask、point cloud、grasp pose 和 coordinate frame。这里的训练目标是让模型学会：

- 什么时候该调用 SAM；
- 什么时候需要 VLM point prompt；
- 什么时候应该用 depth 建点云；
- 什么时候需要检查 grasp pose；
- 什么时候不应该相信 top-1 detector / grasp candidate。

这部分是 Cap-X 的核心机会，因为当前很多失败并不是“不会写 Python”，而是没有把视觉和几何条件纳入代码决策。

### 6.3 Train Code Utilization

Skill 被选中后，模型还需要把 skill 改写成当前任务的可执行代码。

值得训练的不是底层控制，而是：

- API 调用顺序；
- 参数选择；
- coordinate transform；
- 验证逻辑；
- exception handling；
- recovery branch；
- semantic action block 的边界。

例如 place 动作不应该只写“在目标位置上方 10cm 松手”，而应该根据 object height、container depth、gripper clearance 和 target surface normal 估计 release height。

### 6.4 Train Skill Distillation

模型需要学会从轨迹中写出下一次真的有用的 skill，而不是泛泛总结。

好的 distillation 应该回答：

- 这个技能适用于什么视觉场景；
- 关键 API 调用是什么；
- 哪些几何量必须检查；
- 哪个参数范围可靠；
- 成功/失败如何判断；
- 失败后应该如何恢复。

这个点可以借鉴 Skill1 的 \(r(\tau)-\hat{U}\)：如果当前轨迹成功，并且超过已有 skill 的历史 utility，说明它有增量价值，应该奖励 distillation。

## 7. 哪些点不应该作为主训练目标

为了让 paper 聚焦，以下部分不建议作为主要 train 对象：

1. **底层 grasp pose predictor**  
   GraspNet / ContactGraspNet 可以作为工具，不必让 VLM 重新学习 grasp。

2. **底层 motion planning / IK / collision checking**  
   这些最好作为确定性或近似确定性 API，训练重点应是何时调用、如何验证、如何处理失败。

3. **低层连续控制**  
   如果 paper 主线是 Code-as-Policy agent，就不要变成 imitation learning / VLA control paper。

4. **基础 segmentation model**  
   SAM 或其他分割模型可以作为感知工具。除非有明确数据和指标，否则不应把训练 segmentation 作为贡献。

5. **泛泛的语言反思质量**  
   反思只有在能提升后续执行时才有意义。论文应该评估 skill 被检索后的实际任务收益，而不是只评估总结文本看起来好不好。

## 8. RL Training 的合理定位

RL 在这里的作用应该是：

> Optimize lifecycle decisions of executable multimodal skills, not low-level robot control.

具体可以训练四类决策：

1. **Selection decision**  
   当前任务和视觉场景下，选哪个 skill 或 skill composition。

2. **Grounding decision**  
   当前 skill 应绑定哪个 object、mask、pose、frame、grasp candidate。

3. **Execution / Verification decision**  
   是否执行、是否先 dry-run、是否需要额外 verifier、失败后是否 retry。

4. **Distillation decision**  
   当前轨迹是否值得写成新 skill，应该写成什么形式，应该覆盖哪个已有 skill 或新增一条。

RL reward 可以来自：

- task success；
- semantic action block success；
- VLM/geometry verifier；
- code execution exception rate；
- retry count；
- skill utility improvement；
- \(r(\tau)-\hat{U}\) 形式的增量技能价值；
- successful generalization on held-out task variants。

这会让 RL 部分不再是“我们也用了 GRPO”，而是明确回答：**哪些机器人技能生命周期决策无法用监督标签直接标注，但可以通过执行反馈优化**。

## 9. 一个可能的系统闭环

完整系统可以设计为：

1. **Task + Observation Input**  
   输入任务语言、RGB-D、robot state、可用 API 文档。

2. **Skill Query Generation**  
   VLM/LLM 生成 skill retrieval query，包括任务目标、关键物体、视觉条件和预期动作。

3. **Multimodal Skill Retrieval**  
   从 skill library 检索相关技能，检索 key 包括文本 desc、视觉 precondition、object category、action type、API signature。

4. **Skill Re-ranking / Composition**  
   模型选择一个或多个 skill，并决定组合顺序。

5. **Semantic Action Code Generation**  
   生成一组 semantic action blocks，每个 block 包含 intent、code、precondition、postcondition。

6. **Perception-Grounded Dry Run**  
   在真实 motion 前先运行 perception branch：segmentation、depth、point cloud、grasp candidates、IK feasibility、collision check。

7. **Robot Execution**  
   执行通过验证的 code block。

8. **Action-Level Verification**  
   用 VLM、几何规则、环境 reward 判断每个 block 成功或失败。

9. **Trajectory-to-Skill Distillation**  
   将成功/失败轨迹蒸馏成 multimodal executable skill 或 failure patch。

10. **Skill Utility Update**  
   用任务结果和 block-level 结果更新 skill utility，淘汰低效技能。

这个闭环比 v0 的 semantic action skill 更进一步，因为它把 skill selection、execution、verification、distillation 都放到了可训练生命周期中。

## 10. 论文贡献可以如何写

这篇 paper 可以主打以下贡献：

1. **Multimodal Executable Skill**  
   提出一种面向机器人 Code-as-Policy agent 的技能表示，融合视觉前置条件、感知 API、可执行代码、几何约束、验证器和恢复策略。

2. **Code-Trace-Based Skill Distillation**  
   利用 code execution trace、视觉前后状态、API 返回和环境反馈，将专家/VLA/自我试错轨迹蒸馏为可复用机器人技能。

3. **RL for Skill Lifecycle Decisions**  
   将 RL 用于优化 skill selection、grounding、code utilization 和 distillation，而不是低层控制，并提出 outcome-derived credit assignment。

4. **Self-Evolving Code-as-Policy Robot Agent**  
   构建一个可持续更新技能库的机器人 agent，在 LIBERO/BEHAVIOR/Franka manipulation 上展示跨任务泛化和失败恢复能力。

## 11. ICRA / ICLR 角度的判断

从 ICRA 角度，这个方向是比较自然且有吸引力的，因为它解决的是机器人系统里真实存在的问题：VLM 会看但不会稳定执行，Code-as-Policy 可解释但缺少学习闭环，传统 skill library 可复用但缺少视觉 grounding。只要实验能证明对 manipulation success、generalization、recovery 有显著提升，ICRA 是有希望的。

从 ICLR 角度，要求会更高。ICLR 不会满足于“把 Skill1 搬到机器人”。需要把方法抽象讲清楚：

- 为什么 code trace 是一种新的 skill learning substrate；
- multimodal executable skill 与文本 skill / trajectory memory 的本质区别；
- 哪些 lifecycle decisions 是可训练的；
- 如何从单一任务结果分配 selection、grounding、execution、distillation 的 credit；
- 这个框架是否能泛化到多个环境，而不是只服务于一个 demo。

因此，这个方向如果只做成“Cap-X + skill library + GRPO”，新意不够。但如果做成 **code-mediated self-evolving multimodal robot skills**，并且把 skill 表示、轨迹蒸馏和 RL 信用分配讲成一个统一方法，就有顶会潜力。

## 12. 当前最应该推进的最小可行版本

为了尽快验证方向，我建议先做一个 MVP：

1. 在 LIBERO / LIBERO-PRO 中选 20-30 个具有明确 pick、place、open、close、container、height、occlusion 的任务；
2. 将生成代码拆成 semantic action blocks；
3. 为每个 block 记录 code trace、API outputs、RGB-D before/after、wrist camera、success/failure；
4. 设计第一版 Multimodal Executable Skill schema；
5. 用专家成功代码和当前 Cap-X 成功轨迹初始化 skill library；
6. 加入 skill retrieval + code generation prompt；
7. 用 VLM + geometry validator 标注 block success；
8. 从成功和失败 block 中蒸馏 skill / failure patch；
9. 做离线 skill library ablation；
10. 再接 RL，训练 selection / grounding / distillation 的关键生成段。

这个顺序比较重要：先证明 skill 表示和 code trace distillation 有用，再上 RL。否则很容易变成一个复杂但不稳定的系统。

## 13. 新的一句话总结

最终这条路线可以总结为：

> **Cap-X can become a self-evolving robot code agent by using executable code traces as the substrate for learning multimodal, perception-grounded, and verifiable robot skills.**
