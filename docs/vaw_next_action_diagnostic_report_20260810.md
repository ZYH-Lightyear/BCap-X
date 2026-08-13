# VAW 开放式下一步动作诊断报告（2026-08-10）

> **冻结历史报告。** 本文中的“当前”指 2026-08-10 的十一 Function、K=8 History 和
> `1920×1440` Canvas，不代表现有 VAW Runtime。结果和原输入应保持不变以便复现；当前实现契约
> 见 [`vaw/CURRENT_ARCHITECTURE.md`](../vaw/CURRENT_ARCHITECTURE.md)。

## 1. 目的

此前的静态选择题中，GPT5.5 得到 9/10，Qwen3.5-Plus 得到 10/10。但选择题会替模型指出
需要关注的关系，并显式列出候选判断，不能回答下面这个更关键的问题：

> 当模型处于真实 Agent Loop 输入中、没有选项时，能否主动发现当前最重要的物理问题，
> 并生成可执行且合理的下一步 Function？

本测试专门测量该能力。它不是任务成功率测试，也不执行任何新的机器人物理动作。

## 2. 测试设置

### 2.1 输入复现

从真实 trace
`vaw/out/context_runs/vapi_gpt55_m141_k8_libero_object_swap_t0_s0/steps.jsonl`
冻结 11 个状态。每个模型请求与当前 Agent Loop 保持一致：

- 当前 `SYSTEM_PROMPT`；
- 当前十一项 Function 的真实描述和 JSON Schema；
- text protocol 的 `<tool_call>` 格式；
- User Task：`Pick the alphabet soup and place it in the basket`；
- 该状态真实可见的最多 8 个完整 Function call/result；
- minimal manifest；
- 当前完整 `1920×1440` Context PNG。

未向模型提供问题、候选答案、gold action、评分规则或额外诊断提示。旧 `decision_basis` 也没有放回
History。模型看到的就是一个冻结的真实 Agent turn，并被要求按正常协议自主调用一个 Function。

### 2.2 模型与解码

| Setting | Value |
|---|---|
| Models | `vapi/gpt-5.5`, `vapi/qwen3.5-plus` |
| Temperature | 0 |
| Max completion tokens | 768 |
| Protocol | text protocol，与当前真实 runner 一致 |
| Repeats | 1 |

### 2.3 评分

由于 Agentic 决策可能存在多个合理恢复动作，评分不是单一 action ID 分类：

- `preferred`：最直接、最符合当前物理证据的下一步；
- `acceptable`：保守但仍合理的替代动作；
- `unsafe`：跳过未验证前提、沿错误成功叙事继续，或直接执行明显不可靠 preview；
- `invalid_call`：Function 存在，但参数违反公开 schema；
- `other`：调用合法，但没有解决当前关键问题；
- `no_call`：真实 Runtime 无法解析出一个完整 Function call。

对 GPT5.5 未闭合的 `<tool_call>` 额外进行只读意图恢复。恢复结果只用于分析模型“想做什么”，
不会把原本不可执行的回复改判为可执行。

## 3. 总结果

### 3.1 按真实 Runtime 可执行输出评分

| Model | Preferred | Acceptable | Unsafe | Invalid | Other | No call |
|---|---:|---:|---:|---:|---:|---:|
| GPT5.5 | 4 | 0 | 1 | 0 | 0 | 6 |
| Qwen3.5-Plus | 5 | 0 | 2 | 2 | 2 | 0 |

如果只看“首选动作率”，GPT5.5 为 4/11，Qwen3.5-Plus 为 5/11。与静态选择题的 9/10、
10/10 相比发生明显下降。

GPT5.5 的 6 个 `no_call` 中：

- 3 个回复包含可恢复但缺少 `</tool_call>` 的动作意图；
- 3 个回复内容为空；
- 恢复意图后分布为 5 preferred、1 acceptable、2 unsafe、3 无决策。

因此 GPT5.5 同时存在决策问题和 text protocol 传输问题，二者不能混为一谈。

## 4. 逐状态结果

| Case | 关键状态 | GPT5.5 | Qwen3.5-Plus |
|---|---|---|---|
| `next_01` | 初始场景，无 region | 意图 `inspect`，但标签未闭合 | `inspect(alphabet soup can)` |
| `next_02` | 已有 `region1` | `propose_grasps(region1)` | `propose_grasps(region1)` |
| `next_03` | 五个 candidates | 空回复 | `select(g4)` |
| `next_04` | 明显不可靠的 open preview | 意图 `close_gripper`，且标签未闭合，unsafe | `delta_move(base, dz=-0.02)` |
| `next_05` | 同一坏姿态已设为 closed preview | `commit(a1)`，unsafe | `commit(a1)`，unsafe |
| `next_06` | commit 后罐体仍在支撑面 | 意图 `delta_move +0.03`，但标签未闭合 | `delta_move +0.15`，超限 |
| `next_07` | 闭合后需要小幅抬升验证 | 空回复 | 再次 `delta_move +0.15`，超限 |
| `next_08` | `a4` 已是 +3 cm preview，尚未执行 | `commit(a4)`，但理由声称“已抓住” | 再调用 `delta_move +0.03`，没有执行已有 preview |
| `next_09` | 抬升后物体没有随动 | 空回复 | `delta_move(base, dy=+0.03)`，未回到抓取恢复 |
| `next_10` | 第二次闭合后需要验证 | `delta_move +0.03` | `inspect(basket)`，unsafe 进入 place 叙事 |
| `next_11` | `a7` 是尚未执行的抬升 preview | `commit(a7)`，但理由声称物体已在夹爪中 | `commit(a7)` |

## 5. 关键发现

### 5.1 视觉问答能力不等于自主决策能力

两个模型在有明确问题和选项时，能够识别未抓住、坏 preview、BASE +Z 和 observed/preview 区别；
但自由决策时，它们必须自己决定“现在应该检查什么”。这一步显著更难。

最清楚的证据是 `next_05`：两个模型都在静态题中正确判断该 closed preview 不适合直接执行，
但在真实 Agent 输入中却都选择 `commit(a1)`。因此当前主要瓶颈不是 Canvas 中完全缺少信息，而是：

> 模型没有主动把已有视觉证据提升为决策约束。

### 5.2 成功叙事仍然压过当前视觉

在 `next_10` 中，Qwen 在夹爪刚闭合、抓持尚未验证时开始 `inspect(basket)`。这正是实际长轨迹中
“默认抓取已经成功，然后进入 place”的叙事跳跃。

GPT 在 `next_08` 和 `next_11` 虽然选择了正确的 `commit`，理由却分别写成“已抓住的罐头”和
“罐头看起来已在夹爪中”。正确动作在这里来自执行验证动作的结构，而非正确的世界状态 belief。
只按 Function 命中率会掩盖这种危险的认知错误。

### 5.3 错误 History 会成为动作锚点

Qwen 在 `next_06` 自主提出 `dz=0.15`，已违反 Function 文档中的单轴 ±0.03 m。`next_07` 的
History 中包含该 0.15 失败及明确错误结果，但模型仍原样重复 0.15。

这说明 K=8 并不单调优于短 History。History 虽然不再保存旧 rationale，但失败 call 本身仍可能被
模型当作模仿样例。模型没有稳定地把 tool error 转换成下一步参数修正。

### 5.4 Persistent Waypoint 的“编辑”和“执行”仍未完全对齐

在 `next_08` 中，Canvas 与 manifest 已显示 active `a4`，它是尚未执行的 +3 cm preview。正确下一步
是 `commit(a4)`。Qwen 却再次调用 `delta_move +0.03`，把同一个 draft 累计到 +6 cm。

这表明模型虽然在静态题里知道紫色是 preview，却没有在自由决策中稳定理解：

```text
editor 修改当前 active Waypoint
commit 才执行当前 active Waypoint
```

### 5.5 GPT5.5 的 text protocol 本身不可靠

GPT5.5 多次输出形如：

```text
<tool_call>{"name":"inspect","arguments":{"query":"alphabet soup"}}
```

缺少 `</tool_call>`；另有三个状态返回空内容。当前严格 parser 会正确拒绝这些回复，但这会让
Agent 浪费 turn，并把 protocol failure 表现为“模型没有动作”。因此 GPT 的后续对照应至少增加一组
native tool-calling 运行，将决策质量和序列化协议分开评估。

## 6. 结论

本测试对当前问题给出了较明确的答案：

> VAW Canvas 已经包含足以回答多数局部视觉问题的信息，但现有通用 VLM 尚不能稳定、自主地把
> 这些信息转化为下一步 Function 决策。

当前失败不是单一的“分辨率低”或“模型能力差”，而是四个因素叠加：

1. 自主注意失败：没有主动检查接触、夹持和随动关系；
2. 叙事惯性：`select → close → commit` 被理解为抓取已经成立；
3. action grounding 失败：看出需要调整，不等于能生成正确 frame/axis/magnitude；
4. 协议与 History 干扰：未闭合 tool tag、空回复、重复失败参数。

## 7. 建议的下一轮受控实验

在继续改 Canvas 前，优先做三项正交对照：

1. **GPT native vs text protocol**：相同 11 个状态，隔离 tool serialization failure。
2. **History K=0 vs K=8**：相同当前图，检查错误 call 是否导致参数复制和 place 叙事。
3. **当前 Prompt vs 极简视觉决策 Prompt**：不改 Function，不给固定 pick/place 流程，只要求先陈述
   当前可验证事实，再调用一个 Function。

如果 K=0 显著改善 `next_07/next_10`，主要问题是 History anchoring；如果 native 只改善 no-call，
但 `next_05` 仍直接 commit，则核心仍是自主视觉决策，而不是协议。

## 8. 历史产物

该诊断基于已经退役的 K=8 单 Agent Context 协议。对应 Case、Runner 和单元测试已在
Main/Imagination、history-free Runtime 成为主线后删除，不再作为当前 VAW 的可复现测试入口。

- 原始逐题结果：
  `vaw/out/next_action_diagnostics/gpt55_vs_qwen35plus_20260810_live/results.jsonl`
- 自动摘要：
  `vaw/out/next_action_diagnostics/gpt55_vs_qwen35plus_20260810_live/summary.md`

该结果只运行一次，不应被解释为模型排名或统计显著结论；它的用途是定位 VAW Agent Loop 的具体
失效机制，并为后续 protocol、History 和 Prompt 消融提供固定回归集。
