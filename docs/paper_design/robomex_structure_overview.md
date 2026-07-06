# RoboMEx 重构后框架结构

> 本文描述当前 RoboMEx 的 agentic 框架设计。重构目标是先把 Act Coding Agent 做成
> Qwen-Code 风格的工程内核:结构化工具动作、渐进披露技能、单 agent 自主推进并主动停机。
> 独立审查器不再是主循环里的硬门控;Act 认为一个 sub-goal 尝试结束后,控制权直接交回 Planner。

---

## 1. 一句话概括

当前 RoboMEx 是一个两层循环:

- 外层 `ReactivePlanner` 负责看任务、当前场景图、历史执行结果和高层技能菜单,每次只提出下一个
  natural-language sub-goal,或输出 `DONE` 结束 episode。
- 内层 `CodeAsPolicyAgent` 负责完成这个 sub-goal。它像 Qwen-Code agent 一样,每轮只选择一个工具动作:
  `use_skill`、`run_python` 或 `finish`。
- `finish` 只表示 Act 认为当前 sub-goal 尝试可以交回 Planner,不是外部审查判定的成功证明。
- Planner 收到 Act trace、状态摘要和刷新后的场景后,决定继续规划、修正目标,还是结束。

这种设计先淡化硬规则,把核心能力压到一个高质量 Act Coding Agent 上:它要会读技能、写代码、看反馈、自我修正、适时停止。

---

## 2. 总体流程图

```mermaid
flowchart TD
    A["Entry: examples/run_planner_live.py<br/>构造 RoboMExConfig"] --> B["RoboMExAgent.run(task)"]

    B --> C["ReactivePlanner.next_subgoal<br/>输入: task + scene image + history + high-level skill menu"]
    C -->|DONE| Z["Episode end<br/>summary / videos / events 落盘"]
    C -->|sub-goal + postcondition| D["CodeAsPolicyAgent.run<br/>Act-only inner loop"]

    subgraph Inner["Qwen-Code-style Act Coding Agent"]
        D --> E["System prompt<br/>Tier-0 API docs + available_skills 短菜单"]
        E --> F["ModelTurn"]
        F --> G{"tool call?"}
        G -->|use_skill(name)| H["加载 SKILL.md 全文<br/>追加到上下文"]
        H --> F
        G -->|run_python(code,intent)| I["CapXExecutorAdapter.run_block<br/>在持久沙箱执行代码"]
        I --> J["返回 stdout/stderr/status/reward/observation<br/>保存 turn code/output/video"]
        J --> F
        G -->|finish(claim)| K["生成 AgentTrace<br/>act_status=finished"]
        G -->|invalid/empty| L["nudge/retry<br/>要求返回合法工具动作"]
        L --> F
        F -->|action budget exhausted| M["AgentTrace<br/>act_status=exhausted/unresolved"]
    end

    K --> N["SubGoalResult<br/>success = trace.success/env signal<br/>note = Act claim or unresolved reason"]
    M --> N
    N --> O["scene_refresh<br/>用最新 observation 保存场景图"]
    O --> P["history += result"]
    P --> C
```

---

## 3. 核心模块

| 模块 | 职责 |
|---|---|
| `robomex/core/session.py` | 框架入口。`RoboMExConfig` 注入技能库、planner policy、code policy、executor、产物目录等依赖;`RoboMExAgent.run()` 串起整个 episode。 |
| `robomex/agents/planner.py` | 外层反应式 planner。每步返回自然语言 `Goal` + `Postcondition`,或 `DONE`;历史里包含 Act 的结果和 note。 |
| `robomex/agents/executor.py` | 当前主力 Act agent。继承共享 `CodingAgent`,负责把 sub-goal 转成多轮技能读取和 Python 执行。 |
| `robomex/core/coder/agent.py` | Qwen-Code-style agent loop。维护 prompt、解析工具动作、加载技能、执行代码、处理无效回复、预算和终止。 |
| `robomex/core/coder/action.py` | 统一 `ModelTurn` / `ToolCall` / JSON action 解析。Provider 边界使用文本 JSON action,内部归一化为 `ToolCall`。 |
| `robomex/core/coder/policy.py` | 模型策略层。`LLMCodePolicy` 走文本 JSON action,用于在线执行、训练回放和跨代理兼容。 |
| `robomex/core/sandbox/capx.py` | CapX/LIBERO 执行适配器。负责把 `run_python` 代码块送进真实 env 沙箱,并回收 observation、reward、terminated、视频范围等。 |
| `robomex/skills/` | 技能库。系统提示只放短菜单,Act/SubAgent 通过 `use_skill` 按需拉取具体 `SKILL.md`。 |

---

## 4. Act Agent 的工作方式

Act 的动作空间被收敛为四个 JSON action:

1. `use_skill(name)`: 读取一个技能的完整操作说明。初始 prompt 只给技能名和短描述,避免把所有技能正文塞进上下文。
2. `call_subagent(task, inputs)`: 委托一个 focused natural-language task 给通用只读 CodingAgentSubAgent。Act 不选择固定 SubAgent 名称或 profile;SubAgent 根据任务和技能菜单自行选择 workflow,结果作为 evidence 回灌给 Act。
3. `run_python(code, intent)`: 在 CapX 持久沙箱中执行一个 Python block。沙箱里的变量、`EVIDENCE`、观测结果和中间计算可以跨 block 复用。
4. `finish(claim)`: Act 主动声明当前 sub-goal 尝试结束,把控制权交回 Planner。

每轮模型调用后,`CodingAgent` 会把工具结果作为新的 user message 回灌给模型。代码执行失败时,stderr 和状态会直接进入下一轮上下文;模型需要自己修正。`use_skill` 不消耗 Python action budget,`run_python` 才消耗预算。若模型反复输出空内容、非法 JSON、多个动作或重复动作,agent 会给 nudge,但不把流程写死成固定 O-A-V 管线。

---

## 5. Planner 和 Act 的边界

Planner 不写代码,也不直接选择底层 perception/action API。它只做任务分解和重规划:

- 输入:原始任务语言、当前场景图、高层技能菜单、已执行 sub-goal 的 `SubGoalResult`。
- 输出:下一个 sub-goal,例如“pick up the milk and place it near the basket”,附带建议技能和期望后置条件。
- 停止:当 Planner 判断任务已经完成或无法继续时输出 `DONE`。

Act 是当前 sub-goal 内的 swarm manager 和 coding executor。它可以直接读技能、写代码,也可以把定位、抓取候选、放置候选、状态检查等不确定环节委托给通用 SubAgent。SubAgent 的调用契约是任务文本,不是规则匹配字段;返回 JSON 只是为了把 compact evidence、置信度和 artifact 路径稳定回传给 Act。

Act 不负责全局任务策略。它只在一个 sub-goal 内自由组合技能、SubAgent 和 API,并在合适时 `finish`。这样 Planner 可以在每个 sub-goal 后基于刷新场景重新决策,避免把一条长计划一次性写死。

---

## 6. 审查能力的新位置

当前主路径已经改成 Act-only:

- session 不再自动运行独立审查 agent。
- session 不再保留 `subgoal_max_attempts` 这类硬重试门控。
- 状态检查可以由 Act 自行完成,也可以在需要时作为普通 `call_subagent(task=...)` 委托出去;它不是固定的外部裁判。

这不是否定审查,而是把当前工程优先级前移:先让 Act 本身具备足够好的工具使用、状态读取、错误恢复和停机判断。后续若强化 grounding、affordance、review 能力,也应作为“任务委托给通用 SubAgent + skill workflow”的形式加入,而不是恢复固定角色路由或硬循环。

---

## 7. 与 Qwen-Code 对齐的点

- Agent 通过“模型回合 -> 工具调用 -> 工具结果回灌 -> 下一回合”的循环工作。
- 技能采用渐进披露:先给短菜单,需要时再加载正文。
- 停止由 agent 主动调用 `finish`,不是外部解析 `FINISH` 文本或固定轮数强行推断。
- 动作被结构化为内部 `ToolCall`,但 provider 边界稳定采用文本 JSON action adapter。
- Runtime 和 policy 分离:在线执行、训练、回放都通过 `LLMCodePolicy` 产出 JSON action,再解析为内部 `ToolCall`。

当前不把 VAPI/OpenRouter 这类中转服务的 provider-native tool calling 作为 RoboMEx 主路径;它们在多层代理下容易丢失 `tools/tool_calls` 字段。JSON action adapter 是默认且唯一的在线执行协议。

---

## 8. 当前设计取舍

这个版本有意先减少硬性规则:

- 不再要求固定的 observe -> act -> verify 顺序。
- 不再让 `query_vlm` 做 bbox/point 检测;空间 grounding 应通过 `vlm_bbox_detection` / `vlm_point_detection` 相关技能/API。
- 不再让独立审查器决定每个 sub-goal 是否可以交回 Planner。
- 不把 Grounding、Affordance、Verification 做成强绑定流程。当前只有一个通用 CodingAgentSubAgent runtime;Act 用自然语言任务委托它,SubAgent 自行加载相关 skill 并产出 evidence。

短期目标是得到一个工程上干净、可调试、可逐步替换 policy 的 Act Coding Agent。长期目标是在这个内核上扩展 skill workflow 和 SubAgent 能力,但默认接口仍应保持 task-first:Act 提出具体问题,SubAgent 根据技能自行工作,最终只回传执行决策所需的 compact evidence。
