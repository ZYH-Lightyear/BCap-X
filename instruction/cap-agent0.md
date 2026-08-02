# CaP-Agent0：VDM Multiturn 完整流程

基于真实跑通数据：

`outputs/qwen3.5-397b-a17b/franka_robosuite_cube_stack_multiturn_vdm_qwen35_smoke`

| 项 | 值 |
| --- | --- |
| Config | `env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml` |
| 任务 | place red cube on green cube, then open gripper |
| Code model / VDM | `qwen3.5-397b-a17b`（同一模型兼任） |
| 模式 | `use_img_differencing=true`（图像差分；非 video VDM） |
| 结果 | 5 code blocks，4× REGENERATE，1× FINISH；reward=1.0，task completed |

---

## 一句话

Agent 先写代码并在仿真里执行；每执行完一个 code block，VDM 对比**执行前后两帧**给出文本差分，Agent 据此决定 `REGENERATE`（补代码）或 `FINISH`（结束）。

---

## 端到端数据流

```text
env.reset
  → 截取初始 RGB (visual_feedback_00)
  → VDM 描述初始场景，写入 initial prompt
  → Code LLM 生成初始 Python (code_init / block_0)
        │
        ▼
┌─── multiturn loop（每 block 一次）──────────────────────────┐
│  env.step(code_block_i)                                      │
│    · SAM3 / GraspNet / IK 等 API 在 sandbox 内执行             │
│    · 录 video_turn_i / line_trace/block_i.jsonl               │
│  截取当前 RGB → visual_feedback_{i+1}                        │
│  VDM(prev_img, curr_img, task) → 文本差分                      │
│  组装 decision prompt:                                       │
│    initial prompt + executed_code + stdout/stderr            │
│    + VDM 差分文本（本配置不把原图塞进决策 prompt）              │
│  Code LLM → REGENERATE <code> | FINISH                       │
│    REGENERATE: 截断未执行块，追加新 block，继续 loop           │
│    FINISH: 跳出 loop                                          │
└──────────────────────────────────────────────────────────────┘
  → 落盘 summary / all_responses / video_combined / …
```

核心实现：`capx/envs/trial.py`（`_run_single_trial` / `_handle_multi_turn_step`）+ `capx/utils/launch_utils.py`（decision prompt / 解析）。

---

## 阶段拆解（对照本 run 产物）

### 1. 初始化 + 初始 VDM

- 渲染初始主相机图 → `visual_feedback_00.png`（红/绿方块分开放桌上，夹爪打开）。
- VDM 被问：「结合任务目标，客观描述初始场景，不要写代码」。
- 真实写入 `initial_prompt.txt` 的描述大意：桌上有绿块与红块且未堆叠，夹爪打开，可开始操作。
- 该文本拼进 initial prompt（API 文档 + Goal），**不**再附原图（本 yaml 未开 `use_visual_feedback`）。

### 2. 初始代码生成

Code LLM 输出完整 pick-place 脚本（`code_init.txt` / block 0）：

`sample_grasp_pose("red cube") → goto_pose(+z_approach) → close → get_object_pose("green cube", extent) → 算放置高度 → goto → open_gripper`

决策记为 `decision: "initial"`（见 `all_responses.json`）。

### 3. 执行 → VDM 差分 → 多轮决策

每步固定三段输入交给决策 LLM：

1. **已执行代码** `{executed_code}`
2. **stdout / stderr**（编译/运行错误会进 stderr；成功时 stderr 为空）
3. **VDM 文本**：对比 `visual_feedback_{t}` vs `visual_feedback_{t+1}`，说明场景变化与任务是否完成

响应只能二选一：

- `REGENERATE` + \`\`\`python ... \`\`\`
- `FINISH`

本配置 `use_img_differencing=true` 且 `use_visual_feedback=false`：决策侧只吃 **VDM 文本**，不直接看当前帧（与 `_multiturn_vf` 不同）。

### 4. 本 smoke trial 的真实轮次

| Turn | Block | 执行结果（line_trace） | 视觉状态（vf） | Agent 决策 |
| --- | --- | --- | --- | --- |
| 0 | pick-place 全流程 | 跑完 open_gripper；无异常 | vf_00→vf_01：红块已被夹在绿块上方，**夹爪仍闭合** | REGENERATE → 再 `open_gripper()` |
| 1 | 仅 `open_gripper()` | 成功 | vf_01→vf_02：仍见闭合夹持堆叠 | REGENERATE → open + 抬升（但错误地用二元解包 `get_object_pose`） |
| 2 | open + 错误解包 | `ValueError: too many values to unpack`（API 实际返回 3 元组）→ 中间目录 `sandboxrc_1` | vf 几乎不变 | REGENERATE → 修正为 `pos, quat, _ = get_object_pose(...)` |
| 3 | open + 抬升 | 成功 | vf_03→vf_04：夹爪打开，红在绿上 | REGENERATE → 先抬高再 open，再 `home_pose()` |
| 4 | lift + open + home | 成功 | vf_04→vf_05：堆叠完成，臂回安全位 | FINISH |

汇总（`summaries.txt`）：`Average regenerations: 4`，`Average finishes: 1`，`Average code blocks: 5`，耗时约 669s。

### 5. 收尾落盘

最终目录：`trial_01_sandboxrc_0_reward_1.000_taskcompleted_1/`

| 产物 | 含义 |
| --- | --- |
| `all_responses.json` | initial / regenerate×4 / finish 决策链 |
| `code.py` / `raw_response.sh` | 最终与初始代码 |
| `visual_feedback_00..05.png` | 初始 + 每 turn 后快照（VDM 差分用） |
| `video_turn_00..04.mp4`、`video_combined.mp4` | 每 turn 与拼接视频 |
| `summary.txt` | 全程序 + stdout/stderr + reward |
| `trial_01_live/line_trace/block_*.jsonl` | 逐行执行与异常 |
| `trial_01_live/debug_overlays/` | SAM3 / depth / pose 调试图 |

（`sandboxrc_1` 是 block 2 报错时的中间快照；最终成功为 `sandboxrc_0`。）

---

## 与相近模式的差别

| 模式 | Agent 看到什么 |
| --- | --- |
| `_multiturn` | stdout/stderr，无视觉 |
| `_multiturn_vdm`（本 run） | stdout/stderr + **VDM 前后帧文本差分** |
| `_multiturn_vf` | stdout/stderr + **原始图像** |

本 run 另开了 `record_video`，故有 per-turn / combined 视频；差分本身走的是 **image VDM**（`use_img_differencing`），不是 `use_video_differencing`。

---

## 流程要点（从本数据直接可见）

1. **VDM 负责“发生了什么”**：把前后帧压成文本，供决策 LLM 判断堆叠/夹爪是否达标。
2. **决策 LLM 负责“还要不要改代码”**：闭环是 `REGENERATE` / `FINISH`，不是继续跑未执行的旧 block。
3. **感知 API 错误会被 stderr + VDM 一起纠正**：block 2 的三元组解包错误触发 regenerate 并修好。
4. **任务完成判据是视觉语义 + FINISH**：堆叠与松爪在 vf 上成立后，Agent 才输出 `FINISH`；环境侧最终 `task_completed=1`。
