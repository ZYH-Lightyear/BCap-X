# RoboMEx 高层设计：Belief-Guided Agent Swarm

> 状态：方法级设计草案
>
> 本文只回答“RoboMEx 应该是什么、如何运行、智能来自哪里”。
> 暂不讨论 schema、class、文件结构、测试数量和代码迁移顺序。

## 1. 一句话定义

RoboMEx 是一个面向机器人操控的 belief-guided Coding Agent Swarm：

> 它持续维护一个 Agent 可用的物理世界理解；Reactive Planner 决定当前要完成的物理 Subgoal；Swarm Manager 根据当前的不确定性、风险和任务需求动态组织 Coding Agents；Agents 用 Skills 和代码提出感知、几何、motion、monitor 与 verification 方案；Graph 把本轮 Swarm 的组织结果编译成可执行、可监控的物理闭环；真实环境中最终只执行一条经过验证的 action。

## 2. RoboMEx 真正要解决的问题

Code-as-Policy 的瓶颈并不只是“模型会不会调用 API”，而是：

- 机器人关键数值会在图像、语言、代码变量、Agent 边界和动作 API 之间发生语义腐化；
- 物体在动作后会移动、旋转、被遮挡、滑落，旧的 grounding 和 motion 很快失效；
- 单个 Coding Agent 很难同时做好感知、几何、动作、监控和异常恢复；
- 静态 Graph 可以执行计划，却很难决定物理世界变化后“现在应该换谁、补什么证据、是否继续”；
- 简单堆叠更多 Agents 又会带来 token、延迟和错误传播。

因此，RoboMEx 的核心问题是：

> 如何让一组会写代码、会调用机器人 Skills 的 Agents，围绕持续变化的物理世界形成一个低成本、可重组、可验证的闭环 policy。

## 3. 总体架构

~~~text
                         Task Prompt
                              |
                              v
                    Reactive Task Planner
                              |
                    embodied Subgoal
                              |
                              v
Agent World --------> Swarm Manager <-------- semantic events
    |                         |
    | role-specific context   | organize / recruit / retire
    v                         v
             Coding Agent Swarm
  Grounding · Geometry · Affordance · Motion
       Monitor · Verifier · Active Perception
                         |
              hypotheses / code / candidates
                         |
                         v
                 Dynamic Graph Runtime
       lifecycle · dataflow · loops · monitoring
                         |
                 one admitted action
                         |
                         v
                   Robot / Environment
                         |
          observation · receipt · action video
                         |
                         v
                     Agent World
~~~

这套架构中：

- Agent World 提供对当前物理世界的共同理解；
- Planner 决定“下一步要达成什么”；
- Manager 决定“为此刻的问题组织谁”；
- Agents 决定“有哪些可行解释和方案”；
- Graph 决定“这些工作如何安全、有序地运行”；
- Runtime 决定“哪一条 action 被允许真正改变世界”。

## 4. 六个高层模块

### 4.1 Agent World：共享的物理 belief

Agent World 不是一张 RGB 图片，也不是全部聊天记录，更不是 simulator ground truth。

它是当前 episode 中对物理世界的共享 belief，至少能够表达：

- 机器人、夹爪、目标物体和环境当前大致处于什么状态；
- 哪些事实是确定的，哪些是不确定或互相冲突的；
- 某个物体是否仍被抓持、是否可见、是否被遮挡；
- 最近执行了什么动作，预期发生什么，实际观察到什么；
- 哪些策略已经尝试过，为什么失败；
- 当前最阻碍下一步动作的未知量是什么。

不同 Agents 不需要读取完整世界。Agent World 会针对角色提供不同视图：

- Grounding Agent 关注目标身份、可见性、历史 track 和候选区域；
- Motion Agent 关注物体 pose、碰撞、目标 affordance 和 robot state；
- Monitor Agent 关注动作预期、关键约束和连续观测；
- Manager 关注 Subgoal、主要不确定性、风险、已有 Agents 和成本。

Agent World 的价值不是“保存更多数据”，而是让物理状态在 Agents 之间保持同一个语义。

### 4.2 Reactive Planner：生成 embodied Subgoal

Planner 根据 Task Prompt、当前 Agent World 和上一阶段结果，生成下一个 embodied Subgoal。

例如“把碗放到盘子上”可以在不同状态下产生：

- 找到并抓起目标碗；
- 将已抓持的碗移动到盘子附近；
- 根据当前抓持姿态精细对齐并释放；
- 碗掉落后重新定位并恢复抓取。

Planner 管理任务语义，不管理 Agent roster，也不直接写 motion code。

### 4.3 Swarm Manager：围绕不确定性组织团队

Manager 是 Subgoal 级的高层组织者。它的核心不是“画 Graph”，而是判断：

- 当前动作被什么未知量或风险阻塞；
- 一个 Agent 是否已经足够；
- 是否需要多个不同方法产生候选；
- 是继续计算，还是先主动获取新证据；
- 哪些 Agents 应短暂运行，哪些需要持续 tracking 或 monitoring；
- 什么时候替换、暂停或结束某个 Agent；
- 当前 Swarm 已经完成、无法完成，还是需要重新组织。

Manager 应采用 risk-adaptive 策略：

- 简单、低风险、belief 清晰时，只启用最小团队；
- grounding 冲突、motion 多解或风险升高时，再展开多个 Agents；
- Agents 高度相关时，不把“多数一致”误认为独立证据；
- 若现有信息不足，优先组织能够获得新证据的 Agent，而不是继续烧 token 猜测。

这可以概括为 **Belief-Guided Swarm Allocation**：根据当前 belief 中真正阻塞动作的不确定性，分配最有价值且互补的 Agents。

### 4.4 Coding Agent Swarm：完成真实工作

Swarm 中的 Grounding、Geometry、Affordance、Motion、Monitor 和 Verifier 仍然是 Coding Agents。

它们：

1. 检索自己的 Skills；
2. 选择或组合 Skills；
3. 编写局部代码；
4. 在图像、点云、renderer、motion planner 或 replay 中运行；
5. 修复局部 code/API 错误；
6. 输出有类型的 hypothesis、monitor、verification 或 action candidate。

Swarm 不是简单让多个相同模型重复回答。不同成员应当：

- 使用不同信息、模型、Skills 或观察视角；
- 对同一个物理问题提出互补 hypotheses；
- 可以竞争，也可以串联或相互验证；
- 必要时在 shadow environment 中尝试 motion；
- 任务完成后及时退出，避免长期占用上下文和预算。

### 4.5 Dynamic Graph Runtime：Swarm 的执行形式

Graph 是 Agent Swarm 的 runtime representation，而不是论文中另一个独立“大脑”。

Manager 先表达本轮需要怎样的团队和协作关系，系统再把这些承诺编译成 Graph。Graph 负责：

- 哪些 Agents 同时或依次运行；
- 数据从谁流向谁；
- 哪些 Agents 持续 tracking/monitoring；
- 多个候选在哪里汇合并选择；
- 哪些局部闭环可以反复运行；
- 哪个异常事件会停止当前动作或唤醒 Manager；
- 物理动作必须按什么顺序执行。

Graph 可以随语义事件更新，但不需要每个控制帧都重新生成。

因此：

> Swarm 提供适应性，Graph 把适应性变成可执行的组织结构。

### 4.6 Physical Runtime：唯一真实执行与反馈

Agents 可以并行思考、写代码、渲染、replay 和模拟多个方案，但真实环境中一次只能有一条 action 获得物理执行权。

Physical Runtime 负责：

- 检查 action 是否仍基于当前有效的世界状态；
- 检查 frame、IK、collision、workspace 和安全约束；
- 保证 monitor 已经就位；
- 执行唯一被选中的 action；
- 收集 robot state、观测、action receipt 和视频；
- 在异常时停止或中断；
- 把结果送回 Agent World。

“single physical writer”只限制真实机器人命令，不限制多个 Motion Agents 在点云、CuRobo 或 renderer 中并行尝试。

## 5. 系统如何运行

一次完整闭环分为七步：

1. Planner 从任务和 Agent World 产生当前 Subgoal。
2. Manager 找出阻塞这个 Subgoal 的关键不确定性和风险。
3. Manager 组织一个最小 Swarm；必要时才扩展更多 Agents。
4. Agents 使用 Skills 和代码产生 hypotheses、evidence、monitors 和 action candidates。
5. 系统选择、融合、拒绝候选，或决定先获取新证据。
6. Graph Runtime 执行一条被允许的 action，同时持续 monitoring。
7. 新观测更新 Agent World：
   - 普通误差由当前局部闭环继续修正；
   - 歧义、遮挡、滑移等事件唤醒 Manager 重组 Swarm；
   - Subgoal 完成或语义前提失效时返回 Planner。

这里有三种不同时间尺度：

- 控制帧：由 deterministic runtime、tracker 和 monitor 处理；
- 语义事件：由 Manager 处理；
- Subgoal 边界：由 Planner 处理。

这样可以避免高层模型在每帧运行，也避免物理世界已经变化时仍盲目执行旧 Graph。

## 6. 案例：将碗放到盘子上

假设机器人已经抓住碗，但抓住的是碗边，碗相对夹爪的姿态不确定。

### 正常路径

1. Planner 产生“将已抓持的碗精细放到盘子上”的 Subgoal。
2. Agent World 表示：
   - 碗仍被抓住；
   - 碗相对夹爪的姿态存在不确定性；
   - 盘子中心与可放置区域已有候选；
   - 直接使用固定 world-axis offset 风险较高。
3. Manager 组织：
   - attached-object pose Agent；
   - plate grounding/placement Agent；
   - 一个或多个 Motion Agents；
   - 持续运行的 attachment/alignment Monitor；
   - 必要的 Verifier。
4. Agents 在点云或 2D 图上渲染夹爪与碗的候选状态，淘汰明显不合理的 motion。
5. Graph 每次只执行一个小的对齐动作。
6. 新观察更新碗底与盘子中心的相对误差。
7. 达到放置条件后，Graph 执行下降、释放、撤离和验证。

移动多少、向哪个方向、什么时候停止，来自当前 belief 下的闭环修正，而不是任务专用 offset。

### 碗在运输中掉落

1. Monitor code 发现 attachment 约束不再成立。
2. Physical Runtime 中断当前动作，并使后续 release/place 失效。
3. Agent World 更新为“attachment lost”，同时保存最后可信位置和新位置 hypotheses。
4. Manager 不再沿旧 place Graph 继续：
   - 若当前 Subgoal 允许恢复，它组织 locate + regrasp Swarm；
   - 若原 Subgoal 前提已失效，则返回 Planner，由 Planner 生成恢复抓取 Subgoal。
5. 新 Swarm 和新 Graph 从更新后的 Agent World 继续。

这不是为“掉碗”硬编码一个 if 分支，而是统一规则：

> 当动作结果破坏了当前 Subgoal 或 Graph 所依赖的物理 belief 时，旧执行承诺失效，系统根据新的 belief 重新组织。

## 7. 与现有工作的高层区别

### 相对 GaP

GaP 的核心是 Graph-as-Policy：多 Agents 负责生成、仿真和优化可部署 Graph。

RoboMEx 的核心应是：

- 部署期间仍存在的、围绕当前物理 belief 动态组织的 Coding Agent Swarm；
- Graph 只是本轮 Swarm 的编译后 runtime representation；
- 物理语义事件可以改变 Agents 的生命周期与后续执行结构。

因此不与 GaP 竞争“谁的 Graph 更复杂”，而是研究 Graph 之上的在线 Swarm policy。

### 相对 ASPIRE

ASPIRE 的核心是利用细粒度执行 trace 调试程序、演化可复用 Skills。

RoboMEx 的核心应是：

- 在一次正在进行的物理 episode 内维护世界 belief；
- 在真正执行下一动作前组织互补 Agents、比较方案或主动取证；
- 动作后根据物理变化立即重组，而不是主要依赖跨 rollout 的程序进化。

### 相对 CaP-X

CaP-X 证明 Coding Agent 可以通过 API 和 Skills 控制机器人。

RoboMEx 要进一步证明：

- code 可以成为多个 Agents 共享的、显式的机器人 action language；
- Agent World 让 code 与持续变化的物理世界保持对齐；
- Swarm 可以比单 Coding Agent 更可靠地处理不确定性和恢复。

## 8. 论文故事

RoboMEx 可以被理解成一种显式、agentic 的 VLA：

~~~text
Vision + Language + Agent World
              |
     Belief-Guided Agent Swarm
              |
        executable code
              |
    Graph + Physical Runtime
              |
             Action
~~~

其中：

- Agent World 提供显式、持续的 embodied context；
- Agent Swarm 承担可解释的 policy reasoning；
- code 是连接 VLM reasoning 与机器人能力的 action language；
- Graph/Runtime 把开放式 reasoning 收敛成安全、可执行的 action。

最值得验证的三个方法主张是：

1. **Shared embodied belief** 能减少跨 Agent、跨动作的物理语义腐化；
2. **Belief-guided elastic swarm** 能以受控成本处理感知歧义、motion 多解和异常恢复；
3. **Compiled graph execution** 能让动态 Swarm 在真实机器人上保持可监控、可恢复和单一物理权限。

这套设计有成为强工作的潜力，但是否超过 GaP、ASPIRE，最终取决于三个实验事实：

- Agent World 是否真的比 raw observation/history 更可靠；
- 动态 Swarm 是否在同等成本下优于单 Agent 或固定 ensemble；
- 遮挡、滑移、掉落和精细放置时，系统是否真的会改变策略，而不只是重新生成相似代码。

## 9. 架构冻结条件

在进入完整代码重构之前，只需要共同确认五件事：

1. Agent World 是共享 belief，而不是全量黑板或 learned dynamics model；
2. Planner 管 Subgoal，Manager 管 Swarm，Agents 做具体工作；
3. Swarm 的扩展由不确定性与风险驱动，不默认全员并行；
4. Graph 是编译后的 runtime structure，不是 Manager 的自由输出，也不是论文主体；
5. 真实动作始终单写入，所有并行尝试都发生在 shadow/replay/render 层。

确认这五点后，下一份文档才应该回答：

- 当前 RoboMEx 哪些模块保留、重构或删除；
- Agent World、Manager、Swarm 与 Graph 分别映射到哪些代码；
- 如何从 Bowl baseline 逐步迁移到通用 live 系统；
- 每个阶段如何测试与验收。
