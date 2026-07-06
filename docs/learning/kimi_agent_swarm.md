# Kimi Agent Swarm：从“模型即 Agent”到 RoboMEx 的专家群体

这份文档把 Agent Swarm 的讨论重新聚焦到 Kimi / Moonshot AI 的思路上。相比 AutoGen、CrewAI、LangGraph 这类通用多 Agent 编排框架，Kimi 对 RoboMEx 更有启发的地方不在于“很多 Agent 互相聊天”，而在于它把 Agentic 能力放到了模型训练、长程工具调用、轨迹数据和产品形态里。RoboMEx 现在要解决的也不是单纯搭一个多 Agent demo，而是让机器人在视觉不确定、动作不可逆、任务长程且失败代价高的环境中，稳定地观察、决策、执行和复盘。

先说明一个边界：我没有找到公开、权威、完整披露的“Kimi Agent Swarm 内部架构文档”。也就是说，公开资料并没有直接说明 Kimi 内部有哪些命名 Agent、它们如何注册、如何调度、如何共享状态、如何做失败恢复。因此这里的“Kimi Agent Swarm”不是对 Kimi 内部系统的逆向复刻，而是基于公开材料中可以确认的方向，抽象出一种对 RoboMEx 有用的工程思想：

- Kimi 强调“模型即 Agent”，不是只把普通聊天模型包一层 prompt。
- Kimi K2 把 Agentic 行为作为训练和后训练目标，包括工具使用、环境交互、数据合成和强化学习。
- Kimi K2 Thinking 展现了长程、多步、连续工具调用能力。
- Kimi Researcher、OK Computer 等产品形态更像是一个可见主 Agent 背后隐藏多个专业工作流。
- 长上下文、联网搜索、代码执行、文件处理、多模态理解，是 Agent Swarm 能成立的基础设施。

因此，对 RoboMEx 来说，重点不是照搬某个“多 Agent 框架”的类名，而是吸收 Kimi 的核心范式：一个强主 Agent 负责持续推理和工具使用，必要时把子问题派发给专业能力单元；所有交互沉淀为轨迹；轨迹再反过来改进技能、提示词、工具和策略。

## 1. Kimi 的核心启发：模型即 Agent

传统 Agent 系统常见的做法是：

```text
聊天模型 + 规划 prompt + 工具列表 + 循环执行 = Agent
```

这种方式能快速工作，但很脆弱。模型本身未必真正理解工具调用的长期结构，也未必能稳定处理“先观察、再判断、再行动、再验证”的闭环。它可能会把工具调用当成普通文本写出来，可能在中途忘记约束，也可能为了完成格式而牺牲真实任务。

Kimi 的方向更接近：

```text
模型在训练和后训练阶段就接触真实或合成的 Agent 轨迹：
目标 -> 推理 -> 工具调用 -> 环境反馈 -> 修正 -> 最终结果
```

这意味着 Agentic 能力不是纯 prompt 包装，而是模型能力的一部分。Kimi K2 的公开技术报告强调 agentic intelligence，指出模型需要在复杂动态环境中自主感知、规划、推理和行动。它的后训练也包括大规模 agentic 数据合成，以及与真实和合成环境交互的强化学习。

这对 RoboMEx 很关键。RoboMEx 现在的 Act Coding Agent、SubAgent、skills、JSON action、轨迹日志、可视化 overlay，本质上都在生成机器人版的 Agent 轨迹。每一次任务运行都包含：

```text
用户任务
  -> Planner 子目标
  -> Act 选择技能或调用 SubAgent
  -> Grounding / Affordance 产出证据
  -> run_python 执行动作
  -> 环境图像、视频、reward、错误信息
  -> 成功、失败或恢复策略
```

如果从 Kimi 的角度看，这些不只是日志，而是未来的训练数据、评测数据、技能进化数据和工程调试数据。

## 2. Kimi Agent Swarm 不等于“越多 Agent 越好”

很多多 Agent 系统容易陷入一个误区：把任何能力都拆成一个 Agent，然后让它们对话。这样看起来很智能，但在机器人里很危险。原因有三个。

第一，延迟会爆炸。一个 LIBERO 动作任务本来就需要 VLM、点云、仿真和模型推理。如果每一步都调用三四个 Agent，成本和时间会很快失控。

第二，责任边界会模糊。如果 Grounding Agent、Affordance Agent、Act Agent 都可以直接决定动作，失败时很难知道是目标识别错、抓取点错、运动代码错，还是任务规划错。

第三，物理动作不可逆。聊天任务中一个 Agent 说错话可以修正，机器人打开夹爪把物体掉了，状态就已经变了。

所以 Kimi 风格给 RoboMEx 的启发应当是“强主 Agent + 任务委托式 SubAgent”，而不是“群聊式自治 Agent”。主 Agent 仍然是 Act Coding Agent。它负责理解当前子目标，决定是否需要观察、定位、抓取候选、路径计划或验证。SubAgent 更像隐藏的专家工作台：它可以独立推理、调用技能、产出结构化结果和证据文件，但默认不直接控制机器人执行不可逆动作。

一个更合理的结构是：

```text
Planner
  只负责把用户任务拆成自然语言子目标和成功条件。

Act Coding Agent
  是唯一默认拥有机器人动作执行权的 Agent。
  它可以调用技能、写代码、调用 SubAgent、根据反馈继续执行。

Generic CodingAgentSubAgent
  接收 Act 给出的 focused natural-language task。
  根据任务自行选择 perception / affordance / motion / task skills。
  输出 compact evidence、候选、状态判断和 artifact 路径，不直接执行机器人动作。
```

这才是适合 RoboMEx 的 Swarm：不是 Agent 数量很多，也不是提前注册一堆固定 profile，而是 Act 能把物理任务生命周期中的不确定问题委托出去，并拿回明确证据。

## 3. Kimi K2 对 RoboMEx 的真正价值：轨迹中心主义

Kimi K2 公开材料中最值得借鉴的是“轨迹”思想。Agent 能力不是靠一条 prompt 长出来的，而是通过大量任务轨迹学习出来的。对 RoboMEx 来说，也应该把一次运行看成可复用资产，而不是一次性输出。

一个机器人轨迹应该至少包含：

- 任务文本和 benchmark 名称。
- Planner 生成的每个 subgoal 和 postcondition。
- Act 每一轮看到的观察、输出的 JSON action、执行的代码。
- SubAgent 的输入任务、调用过的技能、产出的 bbox、mask、点、候选姿态或状态判断。
- 每个候选 affordance 的可视化图、3D 坐标、姿态、评分和选择理由。
- 执行动作前后的图像、视频、reward、环境终止信号。
- 失败原因：识别错、抓取点错、碰撞、代码错误、策略错误、模型格式错误。
- 最终是否成功，以及成功路径能否沉淀成 skill。

这和 Kimi 的 agentic data synthesis 非常接近。区别是 Kimi 的环境可以是网页、代码、搜索、文件，而 RoboMEx 的环境是仿真或真实机器人。RoboMEx 轨迹更贵、更慢、更需要结构化。因此我们更应该把每个 SubAgent 的输出做成 artifact，而不是只存在聊天上下文里。

例如，Act 委托“为开口朝上的碗提出可执行抓取候选”时，SubAgent 不应该只返回一句“抓碗边缘”。它应该返回：

```json
{
  "object": "bowl",
  "strategy": "open_bowl_rim_topdown",
  "candidates": [
    {
      "position": [x, y, z],
      "quaternion": [qx, qy, qz, qw],
      "jaw_axis": [tx, ty, tz],
      "approach_axis": [0, 0, -1],
      "score": 0.82,
      "risk": "near rim, avoid inner empty region"
    }
  ],
  "artifacts": {
    "overlay": "affordance_candidates.png",
    "metadata": "affordance_candidates.json"
  }
}
```

这种结果对 Act 有用，对 debug 有用，对后续自动生成 skill 也有用。

## 4. Kimi Researcher / OK Computer 的产品启发

公开介绍中，Kimi Researcher 更像一个研究型 Agent：它能围绕问题进行搜索、阅读、整理和生成研究报告。OK Computer 则更偏向通用计算机操作和浏览器任务。这些产品没有公开内部 Agent 划分，但从用户体验看，它们很可能不是简单的一次模型回复，而是一个可见 Agent 背后调度搜索、阅读、写作、代码、文件、浏览器等工具或工作流。

这给 RoboMEx 一个很重要的产品级启发：用户不应该直接面对一堆 SubAgent。用户应该面对一个清晰的 RoboMEx Agent。内部可以有定位、affordance、motion、状态检查、skill writer 等不同任务形态，但对外只表现为：

```text
我理解任务 -> 我观察目标 -> 我生成抓取候选 -> 我执行 -> 我检查结果 -> 我继续
```

这也解释了为什么 RoboMEx 不应该急着把 SubAgent 类型绑定死。Kimi 风格不是要求所有专家都是同一种类。当前默认可以先用通用 CodingAgentSubAgent：它能读 skill、写代码、调用工具，并被只读执行边界约束。以后某些委托也可以落到固定 workflow，比如一个稳定的 bowl_grasp.py；还有一些可以是检索器、日志分析器或 VLM judge。关键是它们对 Act 暴露统一接口：

```text
call_subagent(task, inputs)
```

Act 不需要指定内部到底是“Grounding Agent”还是“Affordance Agent”。它只需要把问题说清楚，并根据返回的证据决定下一步。

## 5. 对 RoboMEx 的 Skill 定位

在 Kimi 风格下，skill 不是简单的“动作宏”。它更像压缩后的经验包。一个 skill 应该包含：

- 适用场景：什么时候应该用。
- 输入需求：需要 bbox、mask、point cloud、目标名称还是历史观察。
- 物理假设：物体形状、可抓取部位、接触风险、夹爪方向。
- 失败模式：VLM 容易混淆什么、点云哪里不可靠、哪些姿态危险。
- 可执行脚本：必要时提供稳定代码，不让模型每次手写。
- 可视化资产：overlay、候选点图、调试图。
- 成功判据：如何判断这个 skill 的输出可信。

因此 RoboMEx 当前把 skills 分成四类是合理的：

- `perception`：看清楚、找准、消歧。
- `affordance`：决定哪里可操作、怎么接触。
- `motion`：路径、姿态、执行控制。
- `task`：把上层任务组织成多个物理步骤。

通用 SubAgent 会根据委托任务自行消费相应 skills：定位/消歧任务通常加载 perception skills，抓取/放置候选任务通常加载 affordance skills，状态检查任务可组合 perception 和 task rubrics。Act Coding Agent 则根据任务需要加载 task 和 motion skills，并把这些 evidence 组合成动作代码。

这也能解释“抓碗”为什么不应该是普通 action skill。抓碗边缘的核心不是立刻执行动作，而是提出一个物理可行的接触方案：从上往下接近，夹爪 jaw axis 与直径正交，抓取点在碗边缘略下方，避开碗心空洞。它更适合是 affordance skill，由 SubAgent 在相关委托中调用，输出候选姿态，然后 Act 决定是否执行。

## 6. RoboMEx 应该如何实现 Kimi 风格 Swarm

第一阶段，不要追求复杂 Swarm。先把 Act 做强。

Act Coding Agent 应该是唯一稳定的内层执行核心。它需要可靠的 JSON action 协议、raw request / raw response 日志、错误恢复、代码执行反馈、技能加载、SubAgent 调用和停止条件。当前你已经从 native tool call 回到 JSON 协议，这是务实选择。因为 VAPI / OpenRouter 这类中转 API 对 tool call 的支持不稳定，而 JSON 文本协议更容易训练、兼容和修复。

第二阶段，把定位、affordance、状态检查等任务形态都交给默认 CodingAgentSubAgent。

SubAgent 不需要在注册时声明“能看哪些 skill”，也不需要 profile。Act 只给任务和可选 inputs，SubAgent 自己根据任务加载 skill。这样更接近 Kimi 风格：专家不是死板函数，而是一个有上下文、有工具、有技能库的 agentic worker。但为了降低风险，它的权限应该默认限制在观察、分析、产出 artifact，不直接执行机器人动作。

第三阶段，强化状态检查委托，但不要恢复旧式硬循环。

旧的 Act <-> Verifier 之所以不舒服，是因为 verifier 被做成硬性规则，容易打断 Act 的自然执行。更好的方式是：状态检查只是 Act 可委托的一类任务。Act 在不确定时调用它，比如“我是否已经抓住碗？”、“这个 can 是否在 basket 里？”、“当前状态是否适合继续 place？”这样检查不再是外部裁判，而是 Act 的工具化感知能力。

第四阶段，把轨迹变成 skill 进化的数据源。

每次成功或失败都要能回答：

- 哪个 perception skill 有用？
- 哪个 affordance 候选被选中？
- 哪个 motion 片段导致成功或失败？
- 哪个 prompt 诱发了错误代码？
- 哪个物体类别需要专门 skill？

这就是 RoboMEx 版的 agentic data synthesis。短期内不一定训练模型，但可以先自动生成 skill 草稿、失败案例库、benchmark 报告和 prompt patch。

## 7. 与通用 Agent Swarm 框架的区别

AutoGen、CrewAI、LangGraph 这类框架主要解决软件层的编排问题：谁先说、谁调用工具、状态图怎么走、消息怎么传。它们对 RoboMEx 有参考价值，但不能直接解决机器人问题。机器人 Swarm 多了几类约束：

- 视觉证据必须可回放，不能只靠文本。
- 物理动作必须最小化不可逆风险。
- SubAgent 的建议必须落到坐标、姿态、路径、接触点。
- 成功率、延迟、token、API 成本都要被记录。
- “看错物体”和“抓错物体”比普通问答错误严重得多。

所以 RoboMEx 不应该把方法贡献写成“我们用了多 Agent”。更好的方法主张是：

```text
我们提出了一种面向机器人操作的 Kimi-style Specialist Swarm：
以 Act Coding Agent 为主执行体，
以 perception / affordance / motion / task skills 为知识载体，
以 task-first CodingAgentSubAgent 承接定位、affordance、状态检查等专家委托，
以可回放轨迹和 artifact 为持续改进数据。
```

这比泛泛讲 Agent Swarm 更有研究味道，也更贴合你的项目。

## 8. 对当前 RoboMEx 的具体建议

短期最应该做三件事。

第一，继续强化 affordance 委托能力。SubAgent 应该能产出多候选、多策略、多视角 overlay，而不是只给一个 grasp pose。对 bowl、plate、can、bottle、drawer handle 这类典型物体，分别建立 affordance skill。每个 skill 都输出结构化候选和可视化证据。

第二，强化定位/消歧委托能力。对 alphabet soup 这种 VLM 容易误判的物体，不要只让 VLM label。应该先用 SAM / bbox / point 找候选，再 crop，再让 VLM 做选择题式判断，比如“哪个候选更像瓶身有字母的罐头，而不是 tomato soup”。这比开放式 caption 稳定。

第三，建立 swarm 级别日志。每个 SubAgent 调用都应该保存：

- 输入任务。
- 加载的 skill。
- 原始模型请求和响应。
- 输出 JSON。
- 图像、crop、overlay、候选文件。
- Act 是否采纳了它的结果。
- 采纳后执行是否成功。

这样你才能真正比较：不用 SubAgent、只做定位委托、定位+affordance 委托、再加入状态检查委托时的成功率、延迟和成本。

## 9. 最终框架图

```text
User Task
  |
  v
Planner
  - natural-language subgoal
  - postcondition
  |
  v
Act Coding Agent
  - owns robot execution
  - loads task / motion skills
  - decides when to call specialists
  - writes and runs code
  |
  +--> Generic CodingAgentSubAgent(task, inputs)
         - chooses relevant skills itself
         - localization / affordance / placement / state-checking tasks
         - returns compact evidence and artifacts

All interactions
  -> trajectory logs
  -> artifact dataset
  -> skill evolution
  -> future training / evaluation
```

这个设计的核心不是“多个 Agent 显得复杂”，而是把机器人物理操作拆成可证据化、可复盘、可进化的专家能力。Kimi 给我们的启发是：真正强的 Agent 系统不是靠外层 prompt 硬控，而是靠模型的长程工具能力、结构化轨迹、专家工作流和持续数据闭环。RoboMEx 可以先用工程方式实现这个闭环，再逐步把成功轨迹变成更强的 skill、SubAgent 策略，甚至未来的训练数据。

## References

- Kimi K2: Open Agentic Intelligence: https://arxiv.org/abs/2507.20534
- Kimi chatbot overview: https://zh.wikipedia.org/wiki/Kimi_%28%E8%81%8A%E5%A4%A9%E6%A9%9F%E5%99%A8%E4%BA%BA%29
- Kimi chatbot English overview: https://en.wikipedia.org/wiki/Kimi_%28chatbot%29
- Kimi-VL Technical Report: https://arxiv.org/abs/2504.07491
- K^2-Agent: https://arxiv.org/abs/2603.00676
- O-Researcher: https://arxiv.org/abs/2601.03743
- RATs local reference: `third_party/RATs`
