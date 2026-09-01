# VAW Skill Evolution

本目录实现分阶段 DAY–NIGHT Skill Evolution。当前完成到 Stage 3：不可变
generation、可注入 Skill Library，以及轻量 DAY rollout orchestrator。

## DAY 的任务选择

实验初始化时，`EvolutionSpec` 将 LIBERO-90 划分冻结在 `split.json`：

```json
{
  "evolve_suite": "libero_90",
  "evolve_tasks": [0, 1, 2, 3],
  "gate_tasks": [4, 7],
  "reserve_tasks": [5, 6],
  "heldout_suites": [
    "libero_object_task",
    "libero_spatial_task",
    "libero_goal_task"
  ]
}
```

任务集合不是 DAY runner 的代码常量。要改变长期 evolve 范围，应在创建实验前修改
`EvolutionSpec`；实验创建后配置不可原地修改。单次调试可以只运行所选 split 的子集：

```bash
python -m vaw.evolution.day \
  --experiment-root /path/to/experiment \
  --split evolve \
  --tasks 0,3 \
  --seeds 1 \
  --resume
```

`--tasks` 不能跨出命名 split，`--seeds` 不能跨出冻结的
`rollout_config.seeds`。这允许低成本 smoke，同时避免 Evolve、Gate 和 Reserve 数据混用。

## Rollout 配置

Runtime 参数保存在冻结的 `rollout_config.runner_args`：

```json
{
  "seeds": [1, 2],
  "workers": 1,
  "runner_args": {
    "model": "vapi/gemini-3.7-flash",
    "imagination_model": "vapi/gemini-3.7-flash",
    "server_url": "http://127.0.0.1:8110/chat/completions",
    "protocol": "native",
    "temperature": 0,
    "max_tokens": 4096,
    "max_turns": 64,
    "max_imagination_turns": 6,
    "max_time_s": 3600,
    "max_physical_ops": 30,
    "motion_backend": "curobo",
    "record_video": true
  }
}
```

DAY 自己注入 suite、task、seed、trace 路径和 skill generation，配置不能覆盖这些字段。
API key 不写入冻结配置或索引，应由本地服务环境提供。

## 输出边界

每个 DAY run 只新增：

```text
run.json                 # 本批次冻结选择
day_index.json           # episode 终局与 trace 指针
logs/                    # 子进程 stdout/stderr
traces/                  # Context Runtime 的原始 trace
```

`day_index.json` 不复制 Canvas、Function transaction、depth、mask、cloud、相机参数或
planner telemetry。Stage 4 直接从 `trace_dir` 指向的原始 trace 编译 Decision Window。
