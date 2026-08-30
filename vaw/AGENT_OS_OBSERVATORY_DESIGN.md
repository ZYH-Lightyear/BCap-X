# VAW Agent OS Observatory

## 1. 目标

Agent OS Observatory 是 VAW 的实时观察与受控 episode 启动面板。它不展示原始
HTTP request/response，也不根据函数序列猜测任务阶段；它展示模型在某一轮真正
获得的结构化 Context、模型据此形成的 decision basis、实际调用的 Function，
以及该 Function 是否改变了物理世界。

Trace 数据面保持只读。控制面只允许启动或停止完整的 `run_context_agent`
episode，不允许浏览器提交 shell、任意文件路径或直接调用 Robot Function。

核心导航单位是 **Turn**，物理视频的导航单位是 **Action Segment**：

```text
Turn Context -> Model Decision -> Function Call -> Function Result
                                      |
                                      +-- physical --> Action Segment + new revision
                                      +-- read-only -> no video       + same revision
```

## 2. 页面结构

### Run Header

显示 task、model、suite/task/seed、当前 revision、运行状态和累计预算。它只显示
真实运行元数据，不计算成功概率或任务阶段。

### Turn Rail

左侧按时间排列每一轮：

- `Context ready`：Canvas 和结构化 Context 已经冻结，正在等待模型；
- `Thinking`：模型请求进行中；
- `Function selected`：已有 decision basis 和 Function；
- `Action running`：物理动作正在执行；
- `Observing`：等待动作后 observation；
- `Complete / Error`：Turn 已提交或失败。

每项只显示 turn、Function、`call/action`、revision 变化和结果。不得出现
`detect/grasp/carry/place` 等由 UI 猜测出的固定阶段。

### Context Stage

中央上方显示该轮实际输入模型的 Canvas。Canvas 旁边以经过排版的组件显示：

- Task；
- Embodied State Card；
- Short-Term Interaction Memory；
- Live References 和 observation revision。

这些组件来自调用模型前冻结的 `TurnContextSnapshot`，不是事后读取当前 Runtime
状态，因此历史 Turn 不会被新 observation 污染。

### Decision Stage

右侧显示：

- 清理 tool/context tags 后的原始 `decision_basis`，不再经过另一个 LLM 总结；
- Function 名称和参数，以 definition list 呈现而非 raw JSON；
- Function result、advisory/error、revision before/after；

### Physical Action Tape

底部是一条不断向右追加的物理 Action Tape。只有 Registry 中
`world_effect=physical` 的 Function 才创建 segment：

```text
A01 commit [r1->r2] | A02 close_gripper [r2->r3] | A03 delta_move [r3->r3, failed]
```

每个 segment 独立保存 agentview 和 wrist 视频、poster、帧区间、Function、参数、
outcome 和 revision。选择某一 Turn 时播放它对应的 segment；“Play through here”
按顺序播放截至该 Turn 的所有 segment。前端只顺序排队已有文件，不反复转码生成
越来越大的累计视频。

动作运行时先出现 `recording` segment；动作结束且视频编码完成后原位切换成可播放
状态。失败或 `dispatched_unsettled` 的动作仍保留完整视频，并使用真实 outcome 标记。

### Workspace 与 Task Launcher

Observatory 指向 `vaw/out` 这样的实验 workspace，而不是某一个固定 trace 文件夹。
后端递归发现 `context_runs`、`sweeps` 等 collection，为相对路径生成稳定 opaque
run ID；前端可以按 collection 筛选。Task Launcher 接收 typed episode 参数，使用
参数数组启动独立 `run_context_agent` 子进程，并立即写入 `launcher.json`。因此环境
初始化期间 UI 也能看到 run，随后同一 run 自动进入 Turn/SSE 实时视图。

## 3. 运行时数据边界

新增一个轻量 `ObservatoryJournal`，由 Runtime 单线程追加事件；UI 对 trace 永远
只读：

```text
run/
├── launcher.json
├── launcher.log
├── steps.jsonl
├── runtime_events.jsonl
├── contexts/
│   └── turn_0001/
│       ├── context.json
│       └── canvas.png
└── actions/
    └── action_0001/
        ├── manifest.json
        ├── agentview.mp4
        ├── wrist.mp4
        └── poster.jpg
```

`context.json` 保存 UI 所需的结构化 `TurnContextSnapshot`：task、
embodied_state、`interaction_memory_before`、live references 和 revision。
它必须在 provider 调用之前冻结，是模型可见 Context 的审计副本，不保存 API
key、base64 图片或 provider headers。

旧格式不能直接复用 `steps.jsonl.interaction_memory`：该字段是在 Function 完成、
本轮事件已经写入 Memory 后记录的。例如 Turn 5 的模型输入只包含 t1--t4，但旧 Turn 5
trace 行中的字段已经包含 t5。当前格式已经显式分开：

```text
TurnContextSnapshot.interaction_memory_before  # 模型真实看到的历史
TurnTransaction.interaction_event              # 本轮完成后新增的一条事件
TurnTransaction.interaction_memory_after       # 用于 replay/digest，可不在主 UI 展示
```

Observatory 的 Context 区只能显示 `interaction_memory_before`，并使用
`InteractionMemory.prompt_lines()` 的真实文本或其等价结构化事件渲染；不得让 UI
重新摘要、改写 outcome、丢弃早期条目或加入当前 Turn 的调用。

`runtime_events.jsonl` 增加单调 `event_seq` 和时间戳，事件仅包括：

```text
turn_context_ready
model_started
model_decision_ready
function_started
action_segment_started
action_segment_ready
turn_closed
episode_closed
```

事件只引用 artifact 路径，不内嵌 Canvas、视频或长文本。已有 `steps.jsonl` 仍是
Turn 完成后的事实记录；Journal 负责暴露正在进行的状态，两者不重复推理。

## 4. 动作视频切分

现有 LIBERO backend 已提供 frame count 和 range API。物理 Function 的统一执行边界
记录：

1. dispatch 前的 `agentview/wrist frame_start`；
2. Function 执行、settle 和 post-action observation；
3. `frame_end`；
4. 立即把两个 range 异步编码到本 Action 的目录；
5. 写入 `action_segment_ready`。

只读 Function 不创建空视频。episode 结束时可以额外生成完整视频，但 Observatory
不依赖它。这样 UI 在每次物理动作结束后即可看到对应片段，不必等待整个任务结束。

## 5. 服务与前端

后端使用 FastAPI：

```text
GET /api/runs
GET /api/runs/{run}/snapshot
GET /api/runs/{run}/turns/{turn}
GET /api/runs/{run}/artifacts/{path}
GET /api/runs/{run}/stream?after_seq=N
```

`stream` 使用 SSE 推送 Journal 事件并支持断线续传；snapshot 用于首次加载和历史
回放。所有 artifact 路径必须 resolve 在当前 run 内。

前端使用 React，但状态模型保持简单：`selectedTurn`、`followLive`、
`selectedCamera` 和 `actionPlaybackQueue`。Canvas、Context、Decision 和 Action Tape
都从同一个 Turn ID 投影，避免界面各区域显示不同 revision。

## 6. 实施顺序

1. 定义 `TurnContextSnapshot`、`ActionMediaSegment` 和 Journal event；
2. 在模型调用前持久化 Context，在物理 Function 边界切分视频；
3. 重写 trace projector，删除旧 viewer 的固定 phase 推断；
4. 实现 FastAPI snapshot、artifact 和 SSE；
5. 实现 Turn Rail、Context Stage、Decision Stage 和 Action Tape；
6. 用 live LIBERO run 验证 Context 冻结、实时更新和视频对齐；
7. 再将相同数据接口接入 RSI 的 progress/diagnosis overlay。

首版不加入 reward 曲线、TOPReward 或诊断结论。Observatory 先保证“模型看到了
什么、决定了什么、环境实际发生了什么”三者在同一 Turn/revision 上准确对齐。
