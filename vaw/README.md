# VAW — Visual Action Workspace

VAW 是面向 LIBERO-PRO 的 Main-owned Visual ReAct Runtime。Main Agent 始终读取当前完整 Canvas
并负责语义感知、任务规划、动作批准与结束判断；需要局部 6-DoF 微调时，它通过
`refine_action` 同步委派给一个只看 Focused Canvas 的 Imagination SubAgent。

当前契约见 [`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md)，Context/Memory 的唯一规范见
[`AGENTIC_CONTEXT_OS.md`](AGENTIC_CONTEXT_OS.md)。

## Function space

Main：

```text
detection_and_sam(query, within_region_id?)
propose_grasps(region_id)
locate_point(query, within_region_id?)
propose_pose(point_id, offset_xyz, quaternion_xyzw?)
select(seed_id)
refine_action(action_id, instruction)
delta_move(delta_xyz_m, frame)
open_gripper()
close_gripper()
reject_action(action_id)
commit(action_id)
done(success)
```

Imagination（只存在于 `refine_action` 内部）：

```text
delta_move(delta_xyz_m, frame)
rotate(axis, angle_deg, frame)
show_rotation_gizmo(frame, axis)
finish_imagination(status="ready" | "failed")
```

`select/propose_pose` 只创建粗空间 Preview，不能直接 commit。`refine_action` 可连续编辑该空间
动作；只有返回 `ready` 后，Main 才能选择 `commit`。Main 的 `delta_move/open_gripper/
close_gripper` 是立即执行的简单物理控制，不经过 Imagination，也不需要 commit。Imagination 内
同名 `delta_move` 仍只编辑虚拟动作。

默认 `--motion-backend curobo` 使用自适应混合路由：CuRobo 负责粗 Action；Imagination 的完整 target
相对当前真实 TCP 在 `6 cm / 20°` 内时使用 PyRoki，超过该局部范围仍使用 CuRobo。Main 的立即
`delta_move` 固定使用 PyRoki。两者在 adapter 边界对齐同一 TCP，且 commit 精确使用 cached plan
的 backend，不会静默降级。Canvas 仅对 Preview 的掌部横梁/指根做淡紫半透明占用提示，其余机器人保持细线，
这个提示不等同于 collision checking。

虚拟夹爪支持可比对的纯视觉开关：`--preview-gripper fk-mesh`（默认，完整 returned-joints FK
mesh）或 `--preview-gripper semantic-wireframe`（对称等长双指 + 细掌梁，只有 3-D 线框）。后者仍由
returned joints 的实际 TCP 定位；planner 没有返回 joint solution 时不绘制理想化夹爪。该开关不改变
Function、Action、planner、commit 或 Context schema，trace meta 会记录所用模式。

Focused Contact View 使用与夹爪闭合方向平行/正交的两台水平 MuJoCo camera，并只在正反侧之间
动态选择可见度更高的一侧。场景内不画位移箭头，统一使用 `5 cm` 视觉标尺。

## Context、Memory 与 trace

- Main：一张当前 `2048×1280` Canvas、最多六条物理 Task Memory、Live References 和一个
  Current Function Event。
- Imagination：一张放大的 Focused Imagination Canvas、一句局部 instruction、累计 edit summary。
- 两者都不接收 transcript history、旧 rationale、receipt、solver telemetry 或旧图片。
- 感知/规划失败只出现于一轮 Current Event；已 dispatch 的物理失败才以 `effect_unknown` 写入
  Task Memory。
- Main turns 写入根 `steps.jsonl`；内部微调写入
  `subagents/imagination_XXXX/steps.jsonl`，不会污染 Main Context。

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
  --protocol text \
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

省略 `--imagination-model` 时两个角色复用同一模型配置，但 provider request、Canvas projection
和 trace 仍互相隔离。

## 验证

```bash
source /mnt/data/zyh/BCap-X/.venv-libero/bin/activate
MPLCONFIGDIR=/tmp/vaw-mpl python -m pytest -q \
  tests/test_vaw_context_runtime.py \
  tests/test_vaw_context_memory.py \
  tests/test_vaw_context_agent.py \
  tests/test_vaw_context_packet.py \
  tests/test_vaw_near_field.py \
  tests/test_vaw_gripper_fk.py \
  tests/test_vaw_contact_camera.py \
  tests/test_vaw_semantic_grounding.py
```

当前实现不修改 CaP-X、RoboMEx、`capx_skill_rl` 或已有输出 trace。
