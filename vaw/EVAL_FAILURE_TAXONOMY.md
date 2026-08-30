# VAW 评测失败分类（m16y / m16z / m17a）

> 状态：C2 需求文档（2026-08-23）
> 取代早期“没有 trace 进入 Imagination”的过期诊断。该问题已解决：m16z 六个成功中四个
> 使用了 `call_imagination`，且下一轮即 `commit` 同一 `action_id`。
>
> 数据：`out/libero_pro_object_eval/`，suite `libero_object_swap`，
> 模型 `vapi/claude-opus-5`，T=0，seed=1，`max_turns=32`。

本文是相位 fitness 提取器的需求：每个标签必须能从 `steps.jsonl` + `meta.json`
确定性打出，不得依赖 LLM-judge。

## 0. 三次 sweep

| sweep | 配置 | env_success | 备注 |
| --- | --- | --- | --- |
| m16y | principles | 4/10（task 0/2/3/4） | 失败全是 `max_turns` |
| m16z | jaw_flip | 6/10（task 1/2/3/4/6/9） | 失败 task 0/5/7/8 |
| m17a | seed plan cache | 已完成 5 题中 2/5（task 3/4） | `plan_reused` 触发 0 次；task 1/2 回归是 VLM 抖动，不是缓存 |

同模型同任务单 seed 成绩差异约 ±20 点。采纳信号不能只看总成功率差。

## 1. 成功签名（m16z task 1/2/3/4/6/9）

典型流水线：

```text
detect → propose_grasps → select(1) → commit → close
→ delta 抬升/运输 → locate basket → propose_pose/commit
（或 call_imagination → 下一轮 commit 同一 action）
→ open_gripper
```

六个成功的共同点：`select` 恰好 1 次；`open_gripper` 恰好 1 次且发生在释放；
`delta_move` 用于运输而非预抓取对位；用了 Imagination 的 4 个都在下一轮
`commit` 同一 `action_id`。

| task | turns | select | delta_move | imagination | open | env |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 23 | 1 | 4 | 2 | 1 | True |
| 2 | 18 | 1 | 7 | 1 | 1 | True |
| 3 | 20 | 1 | 8 | 0 | 1 | True |
| 4 | 28 | 1 | 16 | 0 | 1 | True |
| 6 | 26 | 1 | 16 | 0 | 1 | True |
| 9 | 17 | 1 | 5 | 1 | 1 | True |

## 2. 失败模式

计数针对 m16y + m16z 的 10 个失败 episode（m17a 的失败作为对照附在条目里）。

### F4 seed select 死循环 — 4/10，约 93 轮

反复 `select`，`solve_ik=error` / `executable=false`，不 commit、不 close。
机器人几乎不动。

**典型：** `m16z_jaw_flip_opus5_task8_t0_s1`（chocolate pudding）

- turn 02 `propose_grasps` → `s1,s2,s3`
- turn 03–32 共 30 次 `select`，全部 `solve_ik=error`，0 次 `commit`
- 多次重选同一个 `s2`

```text
[basis 04] 当前 a1 的 executable=false，无法 commit；改用几何方向实质不同的候选 s2。
[basis 30] a27 的 executable=false，无法 commit…应换用几何方向实质不同的候选 seed。
```

同族：`m16y_principles_opus5_task7_t0_s1`（21 次 select）、
`m16y_principles_opus5_task8_t0_s1`（10 次）、
`m16z_jaw_flip_opus5_task5_t0_s1`（4 次 select + 从未 close）。

**检测：** 连续 `select` 且 `solve_ik=error` 或后续 Action `executable=false`
的最长游程 ≥ 3。`reach=0` 即本模式的相位位。

### F1 对齐了不放 — 1/10，约 15 轮

持有载荷悬在容器口上方，反复 −Z / XY `delta_move`，终止前从未 `open_gripper`。

**典型：** `m16y_principles_opus5_task6_t0_s1`（butter）

- turn 08 `close_gripper`（GRIP 0.493）
- turn 14 `commit` 运输到篮口
- turn 15–29 共约 14 次 `delta_move` 微调，无 `open_gripper`
- turn 30 改判空载并 +Z

```text
[basis 26] 夹爪携带黄油悬于篮口上方，H 10.4cm…需先向 −X/−Y 方向小步修正
[basis 30] …Contact View 中夹爪下方未见任何被携带的黄油载荷…先抬升恢复净空
```

对照：`m17a_seed_plan_cache_opus5_task1_t0_s1` 同样从未 `open_gripper`，
在篮口上方连降后误判空载（见 F2）。

**检测：** 存在成功 `close_gripper` 之后的 −Z `delta_move` 游程，且全 episode
在该 close 之后没有 `open_gripper`。

### F2 幻觉空夹爪 — 2/10，约 12 轮

GRIP 停在中间开度、后续画面仍夹着物体，模型却根据 Contact 指间「看空」或
AGENTVIEW 地上有盒，判定抓取失败，于是 +Z / 重新 detection。

**典型：** `m16y_principles_opus5_task1_t0_s1` turn 30–32；
`m16y_principles_opus5_task9_t0_s1` turn 25；
`m17a_seed_plan_cache_opus5_task1_t0_s1` turn 24。

```text
[basis 30] 夹爪已闭合（GRIP 0.273），Contact 两视图都显示奶酪盒被夹在两指之间…
[basis 31] 夹爪已完全闭合（GRIP 0.032）但 Contact Front 显示两指间为空…
```

**检测：** 源 region 在抬升后仍 `verified` 于原位（附着假设与复验矛盾），
或 close 后立即 open 且随后重新 `detection_and_sam` 同一物体。机器可判的
G3 矛盾优先于模型自述。

### F3 delta_move 伺服税 — 1 主 + 1 副，约 38 轮

长距离或预抓取对位用 3cm/turn 的 `delta_move` 链，吃掉预算。

**典型：** `m16y_principles_opus5_task9_t0_s1`（23 次 δ，其中 T5–T21 约 17 次
预抓取对位）；`m16y_principles_opus5_task6_t0_s1` 篮口段 14 次 δ。

**检测：** `delta_move` 次数 / 总轮次；预抓取段 = 第一次 `close_gripper` 之前的
`delta_move` 计数。

### F5 Imagination 成果被丢弃 — 2/10

`call_imagination` 返回 `ready`/`partial` 后，下一轮不是同一 `action_id` 的
`commit`。

**典型：** `m16z_jaw_flip_opus5_task0_t0_s1` turn 28 `ready, a4` → turn 29
`detection_and_sam`。`m16y_principles_opus5_task5_t0_s1` turn 32 `ready, a6`
即耗尽（更贴近 F8）。

反例（成功路径）：m16z task1/2/9、m16z task7 的 partial → 下一轮 commit。

**检测：** 对每次 `call_imagination` 且 status ∈ {ready, partial}，检查下一
Main turn 是否为 `commit` 且 `action_id` 相同。

### F6 重复 detection — 伴随模式

已 grounding 且仍 `verified` 的对象被再次 `detection_and_sam` / `locate_point`。
常见于 F2/F4/F8 之后，单独不作为主导死因。

### F7 抓取失败重试 — 1/10

反复 close → 判失败 → open / delta，从未进入运输。

**典型：** `m16y_principles_opus5_task1_t0_s1`（close@T17/T29，open@T18/T32）。

### F8 运输阶段预算耗尽 — 2/10

已经抓住并抬起，篮口定位或 Imagination 未完成即 `max_turns`。

**典型：** `m16y_principles_opus5_task5_t0_s1`（close@T25，imagination@T32 ready）；
`m16z_jaw_flip_opus5_task7_t0_s1`（close@T28，locate basket@T32）。

## 3. 相位位（C2 必须产出）

| 位 | 成立条件 | 失败对照 |
| --- | --- | --- |
| `reach` | 至少一次 `commit` 且该 Action 当时 executable | m16z task8 = 0 |
| `grasp` | `close_gripper` 后载荷随动（随后 +Z 且源 region 不再原位 verified） | m16y task1/7/8 |
| `transport` | 载荷足迹进入目标容器邻域（locate/propose_pose 到容器后仍持有） | m16z task7 刚开始即结束 |
| `place` | 执行过 `open_gripper` 且 `env_success` | 必须与 meta.env_success 一致 |

成本：`turns`、`physical_ops`、`main_tokens`、`imagination_tokens`。

## 4. 明确不是本分类的东西

- m17a 的 seed plan cache：5 个已完成 run 中 `plan_reused=true` 次数为 0，
  不能解释 task1/2 的回归。
- 「从不委派 Imagination」：已被 m16z 成功案例证伪。
- 契约层缺口（seed 目录不标 executable、H 在容器内量篮底、blocked 运动仍标
  completed）本期只测量，不在此修复。
