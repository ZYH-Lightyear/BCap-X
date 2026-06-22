# Watch-Action Verified Code-as-Policy Proposal

## 0. 一句话核心

我们想做的不是给 CaP-X 打一个工程补丁，而是提出一个新的具身代码执行范式：

> **不要把 LLM 生成的机器人代码直接执行，而是先把它拆成 WATCH 和 ACTION，并在执行前验证它到底 ground 到了什么，以及即将怎样改变世界。**

核心 slogan：

> **WATCH 确认对象，ACTION 确认干预。**

英文表述：

> **Watch grounds the referent; Action grounds the intervention.**

更方法论的表述：

> **From generate-then-execute to verify-before-commit.**

---

## 1. 为什么从 CaP-X 出发会自然得到这个问题

CaP-X 的基本思路是让大模型根据任务、API 文档和视觉反馈生成可执行机器人代码。这个方向本身是合理的，因为代码可以显式调用 perception、motion、grasp、robot APIs，比纯自然语言 action 更可解释。

但我们在 LIBERO/CaP-X 上看到一个非常典型的问题：

```text
任务：Pick the akita black bowl between the plate and the ramekin and place it on the plate.
结果：代码没有报错，模型回答 FINISH，但 reward=0，task_completed=False。
```

这个失败说明：

> **代码能运行，不代表具身任务被正确完成。**

当前 CaP-X 更像是：

```text
生成整段程序 -> 执行整段程序 -> 事后问模型 FINISH / REGENERATE
```

这个流程的问题是，它把所有代码都当成同一种东西。但在机器人任务里，代码其实天然分成两类：

1. **WATCH**：看、分割、定位、估计、判断关系，不改变世界。
2. **ACTION**：抓、移动、放置、开关夹爪，会改变世界，而且经常不可逆。

所以我们的核心观点是：

> **具身代码的关键不是“能不能执行”，而是“哪些部分可以先安全观察，哪些部分必须在 commit 前验证”。**

---

## 2. 当前观察到的两个关键 failure mode

### 2.1 WATCH failure：关系指代 grounding 错误

第一个问题来自视觉 grounding。

SAM3 或类似 segmentation foundation model 对简单类别通常可以工作，例如 `cup`、`bowl`、`plate`。但当任务包含方位、关系、多个相似实例时，它可能分不清真正目标。

典型例子：

```text
the left cup
the black bowl between the plate and the ramekin
the mug next to the plate
the object behind the bowl
```

如果场景里有两个一模一样的杯子，任务说“左边的杯子”，SAM3 可能只是识别出 cup，但未必真的理解 left。也就是说，SAM3 可能解决了 category grounding，却没有解决 relation-aware instance grounding。

因此第一个核心问题是：

> **Foundation segmentation models can segment object categories, but they may fail at relation-aware referent grounding.**

中文说法：

> **模型可能看到了正确类别，却选错了具体实例。**

这类错误在机器人中很严重，因为后续 action 即使执行得很好，也是在操作错误对象。

---

### 2.2 ACTION failure：抓取和放置缺少 metric grounding

第二个问题来自 action。

即使 WATCH 阶段找到了正确对象，后续抓取和放置也可能失败。CaP-X/LLM 生成的代码经常包含一些经验参数，例如：

```text
从物体上方 10cm 接近
抓起后抬高 15cm
在目标表面上方 3cm 释放
使用第一个 GraspNet pose
使用固定朝下的姿态
```

这些数值看起来合理，但本质上是 magic numbers。它们没有严格来自当前场景的几何、物体尺寸、夹爪宽度、目标表面高度、障碍物距离或 IK 可达性。

GraspNet 也不是绝对可靠的。它可能给出：

```text
抓在物体边缘的 grasp
与目标 mask 不一致的 grasp
几何上看似可抓但机器人不可达的 grasp
会碰撞桌面或邻近物体的 grasp
放置时不满足目标关系的 pose
```

因此第二个核心问题是：

> **Generated manipulation code often lacks metric and physical grounding.**

中文说法：

> **模型不只是可能选错对象，也可能用错误的几何参数去操作正确对象。**

---

## 3. 核心定义：WATCH 与 ACTION

我们把 LLM 生成的具身代码视为一种 typed embodied program，其中每个片段属于 WATCH 或 ACTION。

### 3.1 WATCH

WATCH 是不改变世界状态的代码。

它的目标不是完成任务，而是形成一个可验证的 scene belief。

WATCH 可以包括：

```text
观察图像
调用 SAM3/VLM 做 grounding
获取 depth/point cloud
估计物体中心和尺寸
判断 left/right/between/on/inside 等关系
检查目标是否唯一
判断当前状态是否满足某个条件
```

WATCH 的关键问题是：

> **我准备操作的对象，到底是不是任务说的那个对象？**

WATCH 的输出不应该只是一个 mask，而应该是一个带证据的 grounding hypothesis：

```text
target category
candidate instances
anchor objects
relation evidence
selected target
confidence / ambiguity
```

---

### 3.2 ACTION

ACTION 是会改变世界状态的代码。

它的目标是对环境施加干预，例如抓取、移动、放置、开夹爪、关夹爪、推、拉等。

ACTION 的关键问题是：

> **这个动作在当前几何和机器人约束下，是否足够可靠，可以 commit 到环境？**

ACTION 不应该被直接执行，而应该先生成候选，再进行 prospective verification。

ACTION 的输出不只是一个 pose 或一段 motion code，而应该是一个 commitment proposal：

```text
action type
referenced verified target
candidate grasp/place/motion parameters
supporting evidence
risk / uncertainty
approve / reject / repair
```

---

## 4. 方法总览：Watch-Action Verified Code-as-Policy

整体 pipeline 可以写成：

```text
Language Task
  -> Typed Code Proposal
  -> WATCH Verification
  -> ACTION Commitment Verification
  -> Execute Approved Action
  -> Post-action State Update
  -> Verified Skill Evolution
```

更具体地说：

1. **Typed Code Proposal**
   - LLM 不再只生成一整段直接执行的程序。
   - 它生成带有 WATCH/ACTION 语义的候选程序，或者系统自动把程序中的 API 调用切分成 WATCH 与 ACTION。

2. **WATCH Verification**
   - 对任务中的 referring expression 做 grounding。
   - 不把完整短语直接丢给 SAM3，而是拆成 target、anchor、relation。
   - 例如把 “black bowl between the plate and the ramekin” 拆成：
     - target: black bowl
     - anchors: plate, ramekin
     - relation: between
   - 然后用 mask、depth、point cloud、VLM、多视角一致性验证选中的实例是否正确。

3. **ACTION Commitment Verification**
   - 对每个高风险动作，在真正执行前先验证候选 action。
   - 对抓取，验证 grasp 是否落在目标 mask/point cloud 上、宽度是否匹配、IK 是否可达、是否可能碰撞。
   - 对放置，验证目标区域是否正确、release pose 是否位于目标 surface 内、放置高度是否来自几何而非经验常数。

4. **Execute Approved Action**
   - 只有通过 verifier 的 action 才真正 commit 到环境。
   - 如果 verifier 拒绝，则返回 repair signal，而不是盲目执行。

5. **Post-action State Update**
   - 执行后重新 WATCH，更新 scene belief。
   - 不是只问模型一句 FINISH，而是检查目标对象、关系和机器人状态是否满足任务条件。

6. **Verified Skill Evolution**
   - 成功轨迹不是简单存代码，而是存 grounding-action-verification procedure。
   - 系统逐渐学会哪些 WATCH schema 和 ACTION schema 在相似任务中更可靠。

---

## 5. WATCH 模块应该解决什么

WATCH 的重点不是“再加一个 VLM 判断”，而是建立 relation-aware grounding。

### 5.1 为什么不能直接用 SAM3 完整短语

对于：

```text
black bowl between the plate and the ramekin
```

如果直接调用：

```text
SAM3("black bowl between the plate and the ramekin")
```

模型可能只抓住 `black bowl`，忽略 `between plate and ramekin`。因此更合理的是：

```text
先找所有 black bowl
再找 plate 和 ramekin
再判断哪个 bowl 满足 between 关系
最后让 VLM/geometry 做一致性确认
```

这不是硬编码某一个任务，而是把 referring expression 分成三个可泛化部分：

```text
category grounding
anchor grounding
relation verification
```

### 5.2 可以支持的关系算子

关系算子可以先从常见空间关系开始：

```text
left_of / right_of
front_of / behind
between
next_to / nearest_to
on / inside
above / below
```

这些关系不需要完全依赖语言模型，可以由多种证据共同判断：

```text
2D mask center
3D point cloud center
object extent
depth ordering
support surface
multi-view consistency
VLM confirmation
```

### 5.3 WATCH 的研究价值

WATCH 的 paper-level 贡献是：

> **把开放词汇分割问题转化为关系感知的实例确认问题。**

也就是说，系统不再问“SAM3 能不能分出这个短语”，而是问：

> **在所有候选实例里，哪个实例最符合任务描述中的关系约束？如果不确定，是否应该拒绝进入 ACTION？**

---

## 6. ACTION 模块应该解决什么

ACTION 的重点是 verify-before-commit。

机器人 action 和普通程序不同。普通程序错了可以重新运行，但机器人动作错了可能已经改变了物体位置，甚至让后续任务不可恢复。因此 ACTION 不能只靠事后 FINISH 判断。

### 6.1 抓取验证

对于 grasp，ACTION verifier 应该检查：

```text
目标是否来自 WATCH verified target
grasp contact 是否落在目标 mask/point cloud 上
grasp width 是否匹配物体尺寸
grasp approach 是否避开桌面和邻近物体
IK / PyRoKi 是否可达
多个 grasp candidates 中哪个风险最低
```

关键不是“用不用 GraspNet”，而是：

> **GraspNet 输出的是候选，不是可以直接 commit 的动作。**

### 6.2 放置验证

对于 place，ACTION verifier 应该检查：

```text
target receptacle 是否来自 WATCH verified target
placement point 是否落在目标 support surface 内
release height 是否根据表面高度和物体几何计算
释放前机器人是否真的持有目标物体
释放后是否可能满足任务关系
motion path 是否可达且低碰撞风险
```

这直接对应我们现在看到的问题：CaP-X/LLM 很容易用固定高度和固定 offset 去放置，而这些参数未必适配当前物体和场景。

### 6.3 ACTION 的研究价值

ACTION 的 paper-level 贡献是：

> **把抓取、放置、移动从“执行代码”改写成“候选干预的承诺验证”。**

也就是说，ACTION verifier 不只是判断代码有没有错，而是判断：

> **这个具体动作是否足够 grounded、feasible、safe，可以被 commit 到真实或仿真环境。**

---

## 7. 从 Semantic Grounding 到 Metric Grounding

这个点可以成为论文的一个核心 insight。

过去很多 Code-as-Policy 工作更强调 semantic grounding：

```text
任务说的是哪个物体？
应该调用哪个 API？
代码逻辑是否符合语言目标？
```

但机器人操作还需要 metric grounding：

```text
抓取点在哪里？
接近高度是多少？
释放高度是多少？
夹爪宽度是否合适？
目标 surface 的真实边界在哪里？
动作路径是否可达？
```

我们的观点是：

> **Reliable embodied code requires both semantic grounding and metric grounding.**

对应中文：

> **可靠的具身代码不仅要知道“操作谁”，还要知道“以什么几何方式操作”。**

这能把我们的工作和单纯 prompt engineering、单纯事后反思、单纯 symbolic verification 区分开。

---

## 8. 与 NESYRO 的关系和区别

NESYRO 是一个很重要的竞争相关工作，因为它也在讲 reliable code-as-policy 和 neuro-symbolic verification。

但我们和 NESYRO 的问题设定不同。

NESYRO 更像是在问：

```text
当前 symbolic facts 是否足够？
skill 的 symbolic preconditions 是否满足？
是否需要通过 safe probe 获取缺失信息？
```

我们更关注：

```text
视觉模型是否选对了具体实例？
GraspNet/LLM 给出的具体 grasp/place candidate 是否可执行？
连续几何参数是否被当前场景约束？
```

可以这样定位：

> **NESYRO verifies symbolic task preconditions; our method verifies multimodal grounding and metric action commitment.**

潜在区别：

```text
NESYRO 依赖较强的 symbolic domain abstraction。
我们面向 CaP-X 这种开放 Python API 和多模态工具调用环境。

NESYRO 擅长检查 symbolic predicates。
我们关注 SAM3 选错实例、GraspNet pose 不稳定、placement height 不合理等连续操作问题。

NESYRO 的 safe probe 主要解决 partial observability。
我们的 WATCH/ACTION verifier 解决 wrong grounding 和 unsafe commitment。
```

所以不能简单说我们“比 NESYRO 加了 VLM”。更准确的说法是：

> **我们把 verification 的对象从 symbolic plan correctness 推到 embodied code 的 referent grounding 与 intervention grounding。**

---

## 9. Skill Evolution：不是存代码，而是存验证过的操作程序

如果每次都从零生成、验证、修复，系统效率会很低。因此我们需要 skill evolution。

但这里的 skill 不应该只是传统意义上的 code snippet。我们真正想内化的是：

```text
什么样的 WATCH schema 能稳定选对对象
什么样的 relation verifier 能处理 left/between/next-to
什么样的 grasp candidate 在某类物体上更可靠
什么样的 placement metric 在某类 receptacle 上更稳定
哪些 verifier rejection 最终被证明是正确的
哪些 repair strategy 后续成功率更高
```

也就是说，我们存的是：

> **verified grounding-action procedure**

而不是单纯存：

> **一段曾经成功过的代码。**

这可以形成一种自进化机制：

```text
成功 verified trace -> 抽象成 schema -> 下次相似任务优先复用 -> 失败时更新 verifier/repair memory
```

---

## 10. 方法图的文字版

可以在 proposal 或 paper 里画成下面这个结构：

```text
Instruction
   |
   v
LLM Typed Program Proposal
   |
   v
+----------------------+       reject / ask repair
| WATCH Verifier       | --------------------------+
| - target candidates  |                           |
| - anchors            |                           |
| - relation evidence  |                           |
+----------------------+                           |
   | approved grounding                            |
   v                                               |
+----------------------+       reject / repair     |
| ACTION Verifier      | --------------------------+
| - grasp evidence     |
| - geometry evidence  |
| - IK/motion evidence |
| - VLM check          |
+----------------------+
   | approved commitment
   v
Execute Action
   |
   v
Post-action WATCH
   |
   v
Skill Evolution / Next Step
```

最重要的是，这张图要让审稿人看到：

> **我们不是多加几个工具，而是改变 embodied code 的执行语义。**

---

## 11. 预期实验设计

### 11.1 Motivation Case Study

从当前失败 case 开始：

```text
Pick the akita black bowl between the plate and the ramekin and place it on the plate.
```

展示 CaP-X 原始流程的问题：

```text
模型生成代码
代码执行无报错
模型判断 FINISH
但 reward=0，task_completed=False
```

然后分析失败可能来自两处：

```text
WATCH: 是否选对 between plate and ramekin 的 black bowl？
ACTION: grasp/place candidate 是否几何上可靠？
```

### 11.2 WATCH Benchmark

构造或筛选带关系指代的任务：

```text
left/right
between
next to
behind/in front of
多个相似实例
```

对比：

```text
SAM3 full phrase
SAM3 category-only + naive selection
VLM-only grounding
ours: relation-aware WATCH verifier
```

指标：

```text
referent accuracy
wrong-object rate
ambiguity detection rate
relation satisfaction rate
```

### 11.3 ACTION Benchmark

测试 grasp 和 place：

```text
first GraspNet candidate
LLM heuristic parameters
CaP-X original action code
ours: ACTION commitment verifier
```

指标：

```text
grasp success rate
placement success rate
bad candidate rejection rate
approved candidate success rate
collision / drop / wrong-place rate
```

### 11.4 End-to-End Evaluation

在 CaP-X/LIBERO 上做端到端对比：

```text
LIBERO spatial
LIBERO object
LIBERO goal
LIBERO 10
selected LIBERO 90
```

指标：

```text
task success rate
false finish rate
wrong-object manipulation rate
action failure rate
average verifier calls
average runtime
```

### 11.5 Skill Evolution Evaluation

对比有无 verified skill memory：

```text
without skill evolution
with verified grounding-action skill evolution
```

指标：

```text
相似任务成功率提升
重复错误减少
repair 次数减少
跨任务迁移能力
```

---

## 12. 可以主张的贡献

这篇 paper 可以把贡献压成四点，而不是写得太散：

1. **Failure diagnosis**
   - 我们指出 CaP-X 风格 embodied coding agent 的两个核心 grounding gap：relation-aware referent grounding gap 和 metric action grounding gap。

2. **Watch-Action typed execution paradigm**
   - 我们提出把生成代码分成 WATCH 和 ACTION，并引入 verify-before-commit 的执行语义。

3. **Multimodal grounding and commitment verification**
   - WATCH 用关系、anchor、几何和 VLM 验证目标对象；ACTION 用 GraspNet、几何、IK/motion 和视觉证据验证动作候选。

4. **Verified skill evolution**
   - 我们把成功轨迹内化为 grounding-action-verification schema，而不是简单存储代码片段。

---

## 13. ICRA / ICLR 定位

### 更像 ICRA / CoRL 的版本

强调机器人可靠操作：

```text
错误对象 grounding
抓取和放置前验证
不可逆动作的执行前检查
LIBERO + 真实机器人潜力
```

这个版本会更自然，因为问题非常 robotics。

### 更像 ICLR 的版本

强调一般 agent 范式：

```text
typed embodied programs
verify-before-commit execution semantics
multimodal tool-augmented verification
self-evolving grounded skills
```

如果投 ICLR，需要避免看起来只是机器人系统工程，而要强调：

> **这是具身智能体执行 LLM-generated programs 的一种新语义。**

---

## 14. Abstract 草稿

Current Code-as-Policy systems generate executable robot programs from language and visual observations, but they often commit generated actions to the environment once the code is syntactically valid or executable. Our CaP-X experiments reveal two grounding failures that are not captured by code execution alone: visual foundation models may select the wrong object under relational referring expressions, and generated manipulation code may use grasp or placement parameters that are not grounded in the current scene geometry. We propose Watch-Action Verified Code-as-Policy, a verify-before-commit paradigm for embodied code agents. The key idea is to type generated robot programs into WATCH segments, which observe and ground task referents without changing the world, and ACTION segments, which intervene in the environment and therefore require prospective commitment verification. WATCH verification decomposes referring expressions into target categories, anchors, and spatial relations, while ACTION verification evaluates grasp, placement, and motion candidates using multimodal evidence from visual grounding, segmentation, geometry, grasp proposals, and reachability checks. Successful verified traces are further distilled into reusable grounding-action skills. Experiments on CaP-X/LIBERO evaluate whether this paradigm reduces wrong-object manipulation, failed grasps and placements, and false finishes compared with whole-program Code-as-Policy baselines.

---

## 15. 当前最清晰的 paper thesis

最终可以把 thesis 写得非常简单：

> **CaP-X 的失败不是因为模型只会写错代码，而是因为生成的具身代码在执行前没有验证两个东西：它看的是不是正确对象，它做的是不是可靠干预。**

我们的回答是：

```text
WATCH 验证对象和关系。
ACTION 验证抓取、放置和运动候选。
只有通过验证的 action 才能 commit。
成功的验证轨迹被内化成技能。
```

一句话版本：

> **WATCH before ACTION, verify before commit.**
