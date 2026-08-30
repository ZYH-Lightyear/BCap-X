# VAW — Visual Action Workspace

VAW 是面向 LIBERO-PRO 的 Main-owned Visual ReAct Runtime。Main Agent 始终读取当前完整 Canvas
并负责语义感知、任务规划、动作批准与结束判断；需要局部 6-DoF 微调时，它通过
`imagine_action` 同步委派给一个只看 Focused Canvas 的 Imagination SubAgent。

当前契约见 [`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md)，Context/Memory 的唯一规范见
[`AGENTIC_CONTEXT_OS.md`](AGENTIC_CONTEXT_OS.md)。当前基线不包含外部技能或知识加载模块；后续 RSI
设计不会作为休眠兼容代码混入在线 Main Context。
历史里程碑已迁入 [`archive/`](archive/README.md)。评测失败分类见
[`EVAL_FAILURE_TAXONOMY.md`](EVAL_FAILURE_TAXONOMY.md)。

## Function space

Main：

```text
detect_region(query, within_region_id?)
propose_grasps(region_id)
locate_point(query, within_region_id?)
preview_pose(point_id, offset_xyz_m, quaternion_xyzw?)
preview_grasp(seed_id)
imagine_action(instruction, action_id?)
move_tcp_delta(delta_xyz_m, frame)
open_gripper()
close_gripper()
discard_action(action_id)
execute_action(action_id)
finish_task(success)
```

Imagination（只存在于 `imagine_action` 内部）：

```text
shift_preview(delta_xyz_m, frame)
rotate_preview(axis, angle_deg, frame)
inspect_rotation(frame, axis)
finish_imagination(status="ready" | "failed")
```

`preview_grasp/preview_pose` 创建 planned 空间 Preview 并缓存规划；当 Preview 与当前视觉证据足以判断时，
Main 可以直接 `execute_action`。`imagine_action` 是可选的局部空间推理工具，不是执行的前置条件；
没有 planned Action 时从当前真实 TCP 懒创建再进入。
failed 时回滚到进入前的 ActionProposal。Main 的 `move_tcp_delta/open_gripper/
close_gripper` 是立即执行的简单物理控制，不经过 Imagination。Imagination 内的
`shift_preview` 只编辑虚拟动作。

默认 `--motion-backend curobo` 使用自适应混合路由：CuRobo 负责粗 Action；Imagination 的完整 target
相对当前真实 TCP 在 `6 cm / 20°` 内时使用 PyRoki，超过该局部范围仍使用 CuRobo。Main 的立即
`move_tcp_delta` 固定使用 PyRoki。两者在 adapter 边界对齐同一 TCP，且 `execute_action` 精确使用 cached plan
的 backend，不会静默降级。Canvas 仅对 Preview 的掌部横梁/指根做淡紫半透明占用提示，其余机器人保持细线，
这个提示不等同于 collision checking。

虚拟夹爪支持可比对的纯视觉开关：`--preview-gripper fk-mesh`（默认，完整 returned-joints FK
mesh）或 `--preview-gripper semantic-wireframe`（对称等长双指 + 细掌梁，只有 3-D 线框）。后者仍由
returned joints 的实际 TCP 定位；planner 没有返回 joint solution 时不绘制理想化夹爪。该开关不改变
Function、Action、planner、execution 或 Context schema，trace meta 会记录所用模式。

Focused Contact View 使用与夹爪闭合方向平行/正交的两台水平 MuJoCo camera，并只在正反侧之间
动态选择可见度更高的一侧。场景内不画位移箭头，统一使用 `5 cm` 视觉标尺。

## Context、Memory 与 trace

- Main：Task、一张当前 `2048×1280` Canvas、Live References、最新
  Embodied State Card 和统一 Short-Term Interaction Memory。
- Imagination：一张放大的 Focused Imagination Canvas、一句场景语言的局部 instruction、累计 edit summary。
- 两者都不接收 transcript history、旧 rationale、solver telemetry 或旧图片。
- Memory 只记录真实 Function transaction，并由 Registry 区分 `[call]` 与 `[action]`；它不自动写入
  抓住、放入、打开或完成等语义判断。
- provider 调用前冻结 `contexts/turn_XXXX/context.json + canvas.png`；其中
  `interaction_memory_before` 是该轮模型真正看到的历史。本轮事务分别写入
  `interaction_event` 和 `interaction_memory_after`，不会产生一轮因果错位。
- Main turns 写入根 `steps.jsonl`；内部微调写入
  `subagents/imagination_XXXX/steps.jsonl`，不会污染 Main Context。
- 每个 Registry 标记为 physical 的 Function 独立保存 `actions/action_XXXX/` 下的
  agentview/wrist 视频片段；只读 Function 不产生空动作视频。

## 构建与真实运行

```bash
cd /mnt/data/zyh/BCap-X/vaw-ui
npm run build

cd /mnt/data/zyh/BCap-X
source .venv-libero/bin/activate
python -m vaw.scripts.run_context_agent \
  --mode agent \
  --suite libero_object_swap \
  --task-id 0 \
  --seed 1 \
  --model vapi/claude-opus-5 \
  --imagination-model vapi/claude-opus-5 \
  --server-url http://127.0.0.1:8110/chat/completions \
  --protocol native \
  --temperature 0 \
  --max-tokens 4096 \
  --max-turns 32 \
  --max-imagination-turns 6 \
  --max-time-s 3600 \
  --max-physical-ops 30 \
  --motion-backend curobo \
  --record-video \
  --trace-dir /mnt/data/zyh/BCap-X/vaw/out/context_runs/main_react_t0_s1
```

在另一个终端启动 Agent OS Observatory。`--workspace` 会递归发现其中的
多个 trace collection；UI 还可以通过受控 Launch API 启动完整 episode：

```bash
python -m vaw.scripts.serve_observatory \
  --workspace /mnt/data/zyh/BCap-X/vaw/out \
  --host 0.0.0.0 \
  --port 8301
```

浏览器只能提交 suite、task、seed、model、motion backend 和预算等 typed 参数，
不能提交 shell、任意文件路径或直接 Robot Function。新 run 默认写入
`context_runs`，也可以选择 workspace 下安全命名的其他 collection。

页面按 Turn 展示冻结 Canvas、Embodied State、真实
Short-Term Interaction Memory、Decision Basis 和 Function result；底部
Physical Action Tape 按 Function 分段播放动作视频。实时更新来自带单调 `event_seq`
的 SSE，不根据函数名推断固定任务阶段。

省略 `--imagination-model` 时两个角色复用同一模型配置，但 provider request、Canvas projection
和 trace 仍互相隔离。

批量评测与相位 fitness：

```bash
python -m vaw.evolution.sweep --tag baseline-v54 --seeds 1,2,3 --resume --workers 1
python -m vaw.evolution.compare \
  --baseline vaw/out/sweeps/gen0/sweep_summary.json \
  --candidate vaw/out/sweeps/candidate/sweep_summary.json
```

## 验证

```bash
source /mnt/data/zyh/BCap-X/.venv-libero/bin/activate
MPLCONFIGDIR=/tmp/vaw-mpl python -m pytest -q \
  tests/test_vaw_context_runtime.py \
  tests/test_vaw_evidence_lifecycle.py \
  tests/test_vaw_context_memory.py \
  tests/test_vaw_context_agent.py \
  tests/test_vaw_context_packet.py \
  tests/test_vaw_contact_camera.py \
  tests/test_vaw_observatory.py
```

当前实现不修改 CaP-X、RoboMEx、`capx_skill_rl` 或已有输出 trace。
