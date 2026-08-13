# ARCHIVED — VAW M1.4.2 Main / Imagination Dual-Agent Runtime

> **历史基线，不是当前接口。** 本文保留最初双 Agent 拆分时的状态、预算耗尽和
> `open/close` Imagination 语义，供复现设计演进。当前实现已改为 history-free Main /
> Imagination ownership、Main-only gripper ActionReview、turn-limit failed 和 `2048×1280`
> Direct Contact Camera Canvas。现行契约见 [`CURRENT_ARCHITECTURE.md`](CURRENT_ARCHITECTURE.md)，
> 当前 Milestone 见 [`M1_5_AGENTIC_SYSTEM_COMPLETION.md`](M1_5_AGENTIC_SYSTEM_COMPLETION.md)。

## 目标

把任务级决策与局部几何微调分离。Main Agent 不再被低价值 Function transcript 淹没，
Imagination Agent 也不需要理解整段任务历史；两者只通过一个最小 handoff 通信。

## 状态边界

公共语义状态只有：

```text
ActionTarget(pose?, gripper?)
ActionSeed(seed_id, target)
ImaginationState(target, refinement_goal)
ActionReview(action_id, target, handoff_reason)
ImaginationHandoff(status, action_id?)
```

pose 与 gripper 不能同时为空。运动轨迹、grasp source、latest visual edit、turn count 和
planner diagnostics 都是 private artifacts，不复制进公共命令状态。

## 所有权

`ContextState.imagination != None` 时 owner 为 Imagination，否则为 Main。它不是 pick/place
phase，也不决定任务流程，只防止两个模型同时编辑同一目标。

- `detection_and_sam`、`locate_point` 和 `propose_grasps` 保持 Main ownership；
- `select/propose_pose/delta_move/rotate/open/close` 创建 Imagination session；
- Imagination 连续原位修改 `ActionTarget`，不创建 action-id 链；
- `finish_imagination(ready)` 将最终 target 交给 Main 审查并分配 action ID；
- `failed` 放弃目标；预算耗尽也保留最终 target，但明确标记为 `budget_exhausted`；
- 两种非失败 handoff 都只是 `ActionReview`，不是执行批准。Main 查看最终 Preview 后自行决定
  commit、换 seed 或重新进入 Imagination。

## Context 策略

不再存在 K-history。每次请求仅包含当前事实：

```text
Main: system + task + policy state + optional handoff + current Canvas
      + optional previous observed before last commit

Imagination: system + task + refinement goal + current ActionTarget + current Canvas
```

Main 启动动作时的自然语言理由被规范化为 `refinement_goal`。它只活在本次 imagination
session，不进入世界状态，不跨 commit 保存。Imagination 的 rationale/calls/results 全部只写
trace。

## 物理边界

`commit(action_id)` 是 Main 对 `ActionReview` 的显式批准：若有 pose，精确执行 private cached plan；arm
成功后才执行可选 gripper target；无论成功失败只 refresh 一次 observation。新的 revision
清除全部 evidence、seeds、imagination 和 action review。

commit 后的第一个 Main 请求额外获得一张明确标注的 previous-observed RGB，用于比较动作前后；
它在该 Main turn 后立即删除，不形成持久历史。

## Canvas

固定 `1920×1440`：agentview 始终只显示真实当前世界；上层 gripper-local 在 Main 阶段显示
真实近场，在 Imagination 阶段使用同一 revision 的密集点云叠加紫色虚拟机器人，帮助进行局部
几何微调。下层按语义状态显示 grounding、seeds、editing、reviewed、error、terminal 或 idle。

## 验收

- Main 与 Imagination 工具集合严格分离；
- provider 请求不含 assistant/tool history；
- preview 不调用物理 backend、不更新 revision；
- continuous edits 修改同一个 target；
- completed/failed/budget-exhausted handoff 清晰，且 ActionReview 不冒充批准；
- 只有 commit 产生真实运动并刷新；
- packet/Canvas 不泄露 raw sensor、trajectory、reward 或 privileged success；
- Web build 与 `1920×1440×3` screenshot 通过。

## 基线状态

本版本作为 M1.4.3 的固定对照组。真实 trace `dual_agent_review_v7_t0_s1` 没有在 20 个总
Function turn 内抓起目标：arm-only commit 后缺少可见的动作因果信息，Main 重新启动 detection
与 grasp proposal。后续优化与逐版本结果记录在 `M1_4_3_GRASP_CONTEXT_OPTIMIZATION.md`。
