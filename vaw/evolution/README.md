# VAW Skill Evolution

本目录实现分阶段 DAY–NIGHT Skill Evolution。当前包括不可变 generation、
可注入 Skill Library、轻量 DAY rollout，以及主动式 M3 轨迹调查闭环。

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
planner telemetry。M3 只在离线调查时按需读取 `trace_dir` 指向的原始 trace。

## M3：主动式 Trace Investigation

M3 不再把每次物理动作预切成固定 Decision Window，也不先用规则替模型判断“哪里值得学习”。
数据流固定为：

```text
EpisodeAtlas
→ TraceInvestigator + inspect_segment
→ EvidenceReviewer
```

### EpisodeAtlas

Atlas 将完整 episode 收敛为一张 TOPReward 原始曲线与逐物理动作 storyboard。曲线只是导航
线索，不是成功真值；Atlas 不包含 solver 日志、模型 rationale、receipt、revision 或 backend
error。它保留所有物理边界，不先选择局部窗口。

### TraceInvestigator

Investigator 初始只看到 Task、终局 outcome 和 Atlas。它只有一个只读工具：

```text
inspect_segment(start_action, end_action, question)
```

工具返回该区间内的高分辨率 policy-visible Canvas 与精简语义动作。Agent 可以缩放或移动
区间继续调查；上下文始终重建为 Atlas、overwrite-only 调查笔记和最近三段证据，不累计整段
图像 transcript。最终 finding 只有 `span`、`observation` 和 `insight`，证据路径由 Runtime
根据实际读取结果附加。

### EvidenceReviewer

Reviewer 对每条 finding 使用全新上下文，只判断视觉 observation 是否成立、insight 是否由
证据支持且可迁移。它不修复 finding，也不写技能；拒绝结果保留供人工审查。

完整命令：

```bash
python -m vaw.evolution.m3 \
  --trace /path/to/trace \
  --progress /path/to/topreward/progress.json \
  --output-dir /path/to/m3 \
  --model vapi/gemini-3.7-flash
```

默认最多读取四次局部证据，可用 `--max-inspections` 调整。对支持 Qwen thinking
开关的本地服务，可加 `--disable-thinking` 使 Function call 和最终 JSON 保持在正文。

输出包括 `atlas/atlas.{json,png}`、每次 `inspect_segment` 的图像与问题、模型实际可见证据的路径/摘要、
`investigation/findings.json`、逐条 Reviewer 请求、`review.json` 和汇总 `m3.json`。

M3 不会生成或修改 `SKILL.md`，不会自动批准候选，也不会改变 active generation。没有
accepted finding 时，M4 会明确 `NO_CHANGE`，不会为了形成候选而降低证据门槛。

## M4：Skill Evolver 与独立 Reviewer

M4 初始只读取 accepted finding 摘要与轻量技能索引，并按需调用三个只读工具：

```text
inspect_evidence
inspect_skill
inspect_reference
```

最终每轮最多形成一个 `ADD / REVISE / RETIRE`，也允许 `NO_CHANGE`。ADD/REVISE 可选择
`0..N` 张经审核的真实 policy-visible raster 作为 Visual Reference。Independent Reviewer
使用全新上下文，只做 accept/reject；只有 accept 后才生成密封 CandidatePackage。

```bash
python -m vaw.evolution.evolve propose \
  --experiment-root /path/to/experiment \
  --mutation-id m001 \
  --m3-output /path/to/m3 \
  --output-dir /path/to/m4 \
  --model vapi/gemini-3.7-flash
```

Visual Reference 存放在技能的 `references/` 目录；只有 Main 调用 `consult_mmskill` 后才进入
Canvas，并标记为 `REFERENCE · NOT CURRENT`。它不覆盖当前真实视觉。

## Gate、晋升与回滚

创建的新 generation 默认 inactive。配对 Gate 读取 baseline/candidate 的 DAY index，验证
generation 身份和冻结配置，并比较 recovery、regression、技能 consult 归因以及双方都成功
episode 的 turn/token/wall-clock 成本。turn/token 使用冻结护栏；wall-clock 只报告，不因共享
服务负载波动自动拒绝候选。Gate 不隐式启动 rollout。

```bash
python -m vaw.evolution.evolve materialize ...
python -m vaw.evolution.evolve gate ...
python -m vaw.evolution.evolve approve ...
python -m vaw.evolution.evolve promote ...
python -m vaw.evolution.evolve audit ...
```

`approve` 是显式人工边界；系统没有自动批准或自动晋升命令。Post-promotion audit 只有在
confirmation seed 再现父代成功、当前代失败时，才按冻结 policy 回滚 active pointer。

完整工程连通性 smoke：

```bash
python -m vaw.evolution.smoke \
  --output-dir /absolute/new/output/directory
```

该 smoke 使用脚本化模型响应和 episode outcome，只验证 Candidate、Generation、Gate、Approval、
Promotion 与 Audit 的工程边界；不能当作真实技能收益。

## Multi-Agent RSI Cycle

`cycle.py` 把真实 DAY、TOPReward 导航、M3 Investigator/Reviewer、M4 Skill
Evolver/Reviewer、候选代复测、Skill Effect Reviewer 和 Gate 串成一条可恢复链：

```text
baseline DAY → progress → M3 → M4 → inactive generation
             → candidate DAY → local effect review → paired Gate
```

Cycle 只编排已有模块，不修改 Agent 结论，也不会自动批准或晋升。每个阶段有独立目录，
`cycle.json` 冻结 task、seed、parent、candidate 和 mutation 身份；`--resume` 只能继续完全
相同的配置。

用于工程验证的五个 LIBERO-90 task 覆盖不同能力：

| task | 内容 | 主要覆盖 |
|---:|---|---|
| 0 | close the top drawer | articulation |
| 9 | put the black bowl on the plate | bowl placement |
| 24 | put the black bowl in the bottom drawer | constrained insertion |
| 46 | put alphabet soup in the basket | grasp + container placement |
| 69 | put chocolate pudding to the left of the plate | spatial relation |

完整命令示例：

```bash
python -m vaw.evolution.cycle \
  --experiment-root /path/to/experiment \
  --cycle-id smoke-5task-s1 \
  --tasks 0,9,24,46,69 \
  --seeds 1 \
  --parent-generation g000 \
  --candidate-generation g001 \
  --mutation-id m001 \
  --model vapi/gemini-3.7-flash \
  --until gate \
  --resume
```

`--until` 可在任一边界停止，便于查看真实产物后再继续。正式统计 Gate 仍使用冻结的
Gate-15；这个五任务 cycle 是工程 smoke，因此即使出现单个 recovery 也不能自动晋升。

`MultiAgentCycle.run_transfer()` 只允许 `heldout_suites`，输出标记为 `transfer`，且不会
进入 progress、M3 或 M4。当前迁移 smoke 使用 `libero_spatial_task:0,4`；这些结果只用于
最终迁移报告，不能再用于改写本代技能。
