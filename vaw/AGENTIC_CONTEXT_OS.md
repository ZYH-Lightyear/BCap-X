# VAW Agentic Context OS

本文是 VAW 当前 Context 与 Memory 的唯一规范。历史 Milestone 文档只记录实验演进，不作为实现接口。

## 1. 设计目标

Main 每轮从最新物理状态重新构造上下文：

```text
Task
+ Current Canvas / Live References
+ Embodied State Card
+ Short-Term Interaction Memory
→ one Main ReAct decision
```

Canvas 提供当前视觉事实；Interaction Memory 只记录 Function 事务。系统不把调用意图推断成
“抓住、放入、打开或完成”等语义事实，也不维护任务 phase machine、自由文本 world summary 或旧图片历史。

## 2. Context 分层

### 2.1 Task

`Task` 在 episode 内保持不变，是 Main 唯一的任务语言输入。Runtime 不维护模型自写的 Goal、阶段或
语义摘要；Main 每轮根据 Task、最新 Canvas、本体状态和最近 Function 事务重新判断下一步。

### 2.2 Current Canvas 与 Live References

Canvas 只表达当前 observation：上排左侧是 Agentview，上排右侧是 68° 斜俯 Contact Auxiliary，
下排并列水平 Contact Front 与 Contact Side，并叠加当前有效的 grounding 或未执行 Preview。上一物理
动作前的图像不进入 Main policy Canvas，只保留在 trace/video。
Live References 列出当前仍可引用的 region、point、seed 和 action。seed/action 随物理 revision 失效；
region/point 由确定性证据复验标记为 `verified`、`occluded` 或删除。

Contact View 中贴合真实表面的琥珀点状区域表示当前 `verified` detect evidence，并跨非物理 turns
持续显示；复验为 `occluded` 或变化后才隐藏。浅紫机器人及 Preview 刚性载荷体积表示未执行几何，
不预测抓持、碰撞、释放或任务效果。当前 Canvas 优先于历史 Function 意图和工具自报状态。

### 2.3 Embodied State Card

State Card 每轮直接从最新 `RobotState` 和 Interaction Memory 投影：

```json
{
  "tcp_pose":{"position_xyz":[...],"quaternion_xyzw":[...]},
  "gripper_opening":0.031,
  "last_action":{"function":"execute_action","outcome":"completed","turns_ago":1,"revisions_ago":0},
  "last_gripper_action":{"function":"close_gripper","outcome":"completed","turns_ago":3,"revisions_ago":2},
  "active_action":{"action_id":"a4","intent":"move above target","state":"planned"}
}
```

未知值保持 `null`。State Card 不包含 attachment、carrying、inside-container 等未经当前视觉确认的语义。

### 2.4 Short-Term Interaction Memory

Runtime 在每次 Function transaction 完成后追加：

```text
t1 [call] detect_region({"query":"can"}) -> ok (r1->r1)
t2 [call] preview_pose({...}) -> ok (r1->r1)
t3 [action] execute_action({"action_id":"a1"}) -> completed (r1->r2)
t4 [action] close_gripper({}) -> dispatched_unsettled (...) (r2->r3)
```

- `[call]`：感知、几何、规划、Imagination 或终止声明；
- `[action]`：Function Registry 中 `world_effect=physical` 的操作；
- outcome 来自实际 Function result 和执行 receipt，不来自任务语义推断；
- revision 是否变化明确指出世界是否可能已改变；
- 完整事件永远写入 trace；Prompt 只显示最近五次已提交调用，并把其中连续、完全相同的事件显示为 `xN`。

Function 的物理属性和 arm/gripper channel 由内部 `FunctionRegistry` 声明。发送给模型的仍是标准 tool
schema，runtime 不通过函数名分支推断物理作用。

## 3. Main 与 Imagination

Main 每轮只调用一个 Main Function。完整 Function schemas 通过 provider tool channel 提供，不再在
用户文本中重复函数名列表。

Imagination 是 `imagine_action(instruction, action_id?)` 内同步运行的局部 SubAgent，只编辑空间
ActionProposal。它接收局部 instruction、Focused Canvas、edit summary 和预算；不接收 Main 的
Interaction Memory，也不能执行动作、控制夹爪或声明任务完成。`ready/partial/failed` 作为普通 Function
outcome 返回 Main，是否执行仍由 Main 决定。

## 4. Trace

每个 Main step 除原有 Canvas、Function、result 和 planner diagnostics 外，还记录：

```text
embodied_state_card
interaction_memory_before
interaction_event
interaction_memory_after
function_effect_kind
revision_before / revision_after
decision_basis / raw_response_text
```

`interaction_memory_before` 与 `embodied_state_card` 是 provider 调用前冻结的真实输入投影；
`interaction_event` 是本轮新增事务，`interaction_memory_after` 只用于 replay。完整 backend
diagnostics 只进入 trace，不回灌策略上下文。

## 5. 不变量

1. Main 每轮只调用一个 Main Function；Imagination 每轮只调用一个局部 Function。
2. Workspace 管物理状态，Runtime 只维护 InteractionMemory，不保存模型自写任务阶段。
3. Function 调用不等于物理 dispatch；物理 dispatch 不等于预期语义效果成立。
4. 当前 Canvas 和测量 RobotState 永远优先于历史调用。
5. 不生成 `[observed]` 语义历史，不建立固定任务阶段、恢复规则或抓取状态机。
6. 未知状态保持未知，不构造“合理默认值”。
7. renderer、协议和物理安全错误显式失败；不做静默动作 fallback。
8. 完整诊断保存在 trace，Agent Context 只保留继续决策所需的轻量事实。
