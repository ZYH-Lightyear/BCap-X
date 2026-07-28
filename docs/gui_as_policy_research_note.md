# GUI-as-Policy：面向 Tool-Using VLM Agent 的显式机器人动作界面

> 本文整理当前调研与讨论结论。它是研究构想而非冻结方案，具体实现由最小系统的 failure mode 决定。

## 1. VLM Agent 与 VLA 之间的缺口

Code-as-Policy 让通用 VLM 通过代码或 function call 调用机器人 API，保留视觉理解、语言推理、代码生成和工具组合能力。但控制通常是离散的：Agent 观察、执行工具，再从新图像或日志中推断结果。系统缺少持续的动作表示，因此 Agent 容易忘记末端位置、是否抓住物体和上一动作的进度，也难以在执行前想象动作的三维效果。

VLA 把视觉、语言与动作的映射训练进权重，并在视觉、本体状态和控制反馈组成的高频闭环中运行，因而具有更强的动作先验与身体连续性。但其 VL-Action knowledge 与机器人本体、数据和动作空间高度绑定。

本研究不主张 GUI 替代 VLA。目标更具体：搭建一套结合 GUI 特性与 Code-as-Policy 的 Agentic Robot Control System，通过合理的 GUI Context 缓解通用 VLM 缺乏动作感知、动作想象和本体状态保持的问题；系统工作后，再针对真实瓶颈训练。

## 2. 文献带来的启发

GUI Agent 的能力并不只来自截图。[UI-TARS](https://arxiv.org/abs/2501.12326) 用 GUI action traces 学习统一动作空间和 grounding；[Aguvis](https://arxiv.org/abs/2412.04454)（ICML 2025）先训练 grounding，再训练 planning/reasoning；[Agent S2](https://arxiv.org/abs/2504.00906) 随新 observation 更新计划；[AgentProg](https://arxiv.org/abs/2512.10371) 把历史重构成带变量、控制流和 belief state 的程序。这说明稳定 Context 是系统维护的 execution state，而非截图与对话的堆叠。

Agentic-VLA 也在处理慢推理与快速控制的矛盾。[Hume](https://arxiv.org/abs/2505.21432) 让低频 System 2 选择动作、高频 System 1 执行；[Agentic Robot](https://arxiv.org/abs/2505.23450) 分离规划、执行和验证；[SV-VLA](https://arxiv.org/abs/2604.02965) 由低频 VLA 生成 action chunk、轻量 verifier 持续检查。这说明动作知识可以分布在不同模块与时间尺度中。

[VIA](https://arxiv.org/html/2607.11119v1) 是最直接的参照。通用 Agent 在浏览器三维界面中操作 virtual target gripper，检查 waypoint，再交给 PI controller。系统还提供 gripper pose、像素对应的三维点和精确位移/旋转工具，目前限于 quasi-static simulation tasks。VIA 验证了方向，却没有回答 GUI、Tool Space 与可训练动作知识如何形成统一方法。

## 3. 核心方法：重新因子化 VL-Action

三种控制方式可以概括为：

```text
VLA:             (V, L) → learned policy → Robot Action
Code-as-Policy:  (V, L) → Code / Tool Calls → Robot Action
GUI-as-Policy:   (V, L) → VLM Agent ↔ GUI Action State
                                      ↔ Tool Space → Controller → Robot Action
```

GUI-as-Policy 不是把完整 VL-Action knowledge “搬到图片里”，而是将其因子化：

- **VLM Agent**负责语义推理、代码生成和工具管理；
- **Tool Space**保存 segmentation、geometry、grasp、IK、trajectory planning 等机器人能力；
- **GUI Action State**表示身体状态、可交互区域、候选动作、目标 gripper 和执行结果；
- **Controller**承担高频轨迹跟踪、接触控制和安全约束。

GUI 是 Tool Space 之上的显式动作工作区。Agent 用感知、grasp、IK 等工具构造候选并写入 GUI，再选择、编辑、检查和提交。Code 是组织工具的语言，GUI 是让动作状态持续可见的表示介质。

## 4. 为什么它不只是前端

如果 GUI 只显示图像和 API 日志，它确实只是包装。要成为 Policy 的组成部分，至少需要三个性质：

1. **持久 Action State**：工具结果成为带 ID、置信度和 observation revision 的可引用界面对象。
2. **动作先实例化**：物理动作先表现为 virtual gripper、waypoint、trajectory 或 interaction axis；选中的 GUI candidate 就是 policy proposal。
3. **Commit Boundary**：工具调用只能改变 belief、候选和 GUI；只有 `commit` 能改变物理世界，执行回执再更新回 GUI。

因此 Agent 管理两类动作：调用工具获取信息、构造候选的 **cognitive/epistemic actions**，以及选择 GUI action 后提交的 **physical actions**。这保留了开放 Tool Space，同时给物理动作提供类似 VLA action latent、但外显可检查的表示。

## 5. Claim、Cap-X 基础与训练

主 Claim 可以限定为：

> 在相同 VLM、机器人观测和底层工具下，让 Agent 通过持久 GUI Action State 构造、检查并提交物理动作，相比直接 Code/Tool Calling，可以改善 action grounding、动作可行性、身体状态保持和失败恢复，同时保留开放工具组合能力。

次级 Claim 是：

> GUI interaction traces 可以用于针对性训练 action grounding、tool routing、candidate selection 和 commit decision，而不必从头训练完整低层 VLA。

当前 Cap-X 已具备 RGB-D、本体状态、grounding、SAM3、GraspNet、点云、IK、motion 和 gripper control，足够搭建第一版。缺口是 GUI-facing 协议：状态快照、可视 `ground/inspect`、virtual target、preview、`commit`、action receipt，以及接触任务所需的短距离 `follow_axis`。

已有 AgentX + Cap-X ReAct 开发运行显示，Agent 会把“夹爪闭合”误当成“抓住物体”，随后重复整段 pick-and-place。这不是正式证据，但说明图像、代码和工具不会自动形成稳定 Action State。

## 6. 实验闭环与待讨论问题

最直接的实验顺序是：

1. Raw Cap-X APIs + 最小 ReAct Agent；
2. 加入短小、结构化的 proprioceptive context；
3. 加入 GUI Action State、candidate visualization 与 preview–commit；
4. 根据前三组暴露的瓶颈进行针对性训练。

各组必须固定模型、工具、seed、动作预算和 evaluator。除成功率外，还应测量 action validity、持有状态判断、重复动作、扰动恢复、token 和 wall time。如果 Raw ReAct 在简单任务上饱和，GUI 必须在精细放置、articulated/contact manipulation、扰动与效率上体现价值；否则 claim 不成立。

仍需讨论：GUI Action State 的最小字段；哪些工具结果必须可视化；何时求解 IK，何时 `follow_axis` 并重新观察；以及优先训练哪个瓶颈。

当前最核心的方法判断是：**GUI-as-Policy 不是 GUI + ReAct，也不是给 API 增加前端。它是一种 Tool-managed Visual Action State：Agent 使用工具构造 GUI，通过 GUI 表达和检查 Policy，再由工具与控制器执行 Policy。**
