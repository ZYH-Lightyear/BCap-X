# RoboMEx 实现现状(insights)

> **文档定位**:本文**只描述 `robomex/` 代码里真正落地的东西**——已接入主循环的部分,以及
> 代码已存在但尚未接线的**占位 / 休眠实现**(均显式标注)。所有"还没有代码"的设想已从本文删除,
> 以免干扰对当前系统的思考。需要愿景/路线讨论时另开文档。
>
> 基线:2026-06-22(最小化重构 + 注释中文化之后)。

---

## 0. 一句话

用**可执行代码**作为机器人策略的组合接口:一个薄的**反应式 Planner** 在高层技能菜单上逐步给出
下一个 sub-goal,一个**执行 Coding Agent** 渐进披露三类技能、在 CapX 沙箱里写代码并执行,每个代码
轮由 `TaskSignalVerifier` 用 env 信号校验。两层循环到 planner 说 DONE 或触顶为止。

---

## 1. 已实现的顶层闭环

```mermaid
flowchart TB
  Env["CapX 沙箱 (LIBERO)"] -->|RGB-D / state| RP
  RP["ReactivePlanner<br/>任务 + 当前场景图 + 历史 + 高层技能菜单<br/>→ 下一个 sub-goal(+高层 skill id) / DONE"]
  RP -->|sub-goal + primary_skill_id| EX["CodeAsPolicyAgent(执行)<br/>USE SKILL 渐进披露 → 写一个 python block → 沙箱执行<br/>每轮:TaskSignalVerifier(env 信号)"]
  EX -->|AgentTrace(并入历史)| RP
  EX -->|run_block| Env
  RP -.scene_refresh(真机).-> RP
```

- 循环体在 `core/session.py` 的 `RoboMExAgent.run`:`for _ in range(max_subgoals)` →
  `planner.next_subgoal(task, results, scene)` → `executor.run(goal, ..., primary_skill_id)` →
  把 `SubGoalResult` 记入 `results`(作为 planner 下一步的历史)→(真机)`scene_refresh`。
- 成功判定:episode 成功 = 所有 sub-goal 成功;sub-goal 成功 = 其 `AgentTrace.success`
  = 任一轮 `execution.terminated` 或最后一轮 `verification.passed`。

---

## 2. 代码结构(包树 + 每个模块职责)

```text
robomex/
├── __init__.py              顶层入口,暴露 RoboMExAgent / RoboMExConfig / EpisodeResult
├── core/                    框架内核:沙箱 + 共享 coder + 接线入口
│   ├── session.py           RoboMExConfig(依赖容器)+ RoboMExAgent + EpisodeResult ——唯一接线入口
│   ├── logging.py           configure_logging / get_logger:框架统一日志 + 可选落盘
│   ├── coder/               执行器与验证器共享的 Coding Agent 内核
│   │   ├── agent.py         CodingAgent:推理循环 + 渐进披露 + 循环安全护栏(模板,子类填钩子)
│   │   ├── action.py        动作解析:parse_action / SkillEntry / render_available_skills /
│   │   │                    build_skill_llm_content / BlockExecutor 协议
│   │   ├── policy.py        CompletionPolicy / LLMCodePolicy / ScriptedCodePolicy
│   │   └── trace.py         TurnRecord / AgentTrace
│   └── sandbox/             动作空间 + 运行后端
│       ├── action_block.py  SemanticActionBlock / BlockExecutionResult / 状态枚举 / trace 事件
│       └── capx.py          CapXExecutorAdapter(鸭子类型驱动 CapX 代码执行 env)
├── agents/                  四个角色 agent(都建在 core/coder 之上)
│   ├── planner.py           ReactivePlanner / TwoLevelAgent / SubGoal / *PlannerPolicy / parse_next_subgoal
│   ├── executor.py          CodeAsPolicyAgent(CodingAgent 子类;每轮调 verifier)
│   ├── verifier.py          VerifyCodeAgent              ← 占位/休眠(见 §6)
│   └── evolve.py            SkillDistiller               ← 占位/no-op(见 §6)
├── skills/                  技能载体 + 磁盘库 + 内置技能
│   ├── schema.py            Skill / SkillCategory / from_dir / from_markdown / to_markdown
│   ├── store.py             SkillLibrary(admit/get/all/compound_skills)+ SkillUtility/update_utility
│   └── builtin/             3 个内置技能包 + load_builtin_skills / render_inventory
├── verification/            验证工具箱(不是 agent)
│   ├── verifier.py          VerificationResult / Verifier / TaskSignalVerifier(已接)/ CompositeVerifier(休眠)
│   ├── vlm_judge.py         VLMJudgeVerifier             ← 休眠
│   ├── primitives.py        vlm_judge 原语               ← 休眠
│   └── context.py           VerifierContext / sanitize_code / build_op_trace / collect_verify_resources ← 休眠
├── perception/              多模态证据(默认不启用,见 §7)
│   ├── evidence.py          证据数据类型(EvidenceKind/Role/Artifact/Bundle)
│   ├── collector.py         EvidenceCollector            ← 实现但默认不接
│   └── render.py            save_rgb / render_before_after
├── examples/
│   ├── run_planner.py       离线两层 demo(脚本 + mock,零依赖,已验证可跑)
│   └── run_planner_live.py  真机两层 demo(LLM + CapX env)
└── test/
    └── test_coding_agent.py 离线测试(覆盖执行器 + VerifyCodeAgent 雏形)
```

---

## 3. 三类技能(已实现)

### 3.1 载体:目录包
- `SkillCategory ∈ {high_level, observation, action}`(单一枚举,`schema.py`)。
- 一个技能 = 一个目录包:
  ```text
  builtin/<category>/<skill_id>/
  ├── SKILL.md          # 执行读者:frontmatter(name/description/category)+ prose 正文(原样注入 prompt)
  └── ref/verify.md     # 验证读者的 rubric  ← 当前休眠:默认循环不读它(仅 verify-as-code 套件会用)
  ```
- 载体**支持但内置技能未带**的 sidecar(`schema.py` 已有访问器):
  - `ref/` 下的视觉参考(`reference_paths()`);
  - `scripts/verify.py`(`verifier_path()`)。
- **消费机制 = 渐进披露**:系统提示只放 `<available_skills>` 短清单(name + description),agent 用
  `USE SKILL: <id>` 拉全文(`core/coder/action.py`)。

### 3.2 磁盘库:`SkillLibrary`(`store.py`)
- **已用**:`admit()`(写回技能包 + 拷贝 ref/scripts sidecar + 写 `utility.json`)、`get()`、
  `all()`、`compound_skills()`(高层技能 = planner 菜单)。两个 demo 都用 `admit(load_builtin_skills())`
  把内置技能装进一个临时库。
- **占位(已实现但当前无调用方)**:`SkillUtility`(call/success 统计)与 `update_utility()`——
  写好了但主循环没有调用,留给将来的 Evolve。

### 3.3 内置技能(3 个,`skills/builtin/`)
| id | category | 作用 |
|---|---|---|
| `pick_object` | high_level | 复合技能:给 planner 当菜单 + 建议性编排,指向 segment/grasp 叶子技能 |
| `segment_object` | observation | 语言分割 → mask + 去噪世界系点云(内含示例代码) |
| `grasp_object` | action | 从点云推一个 top-down 抓取,执行并用夹爪开度确认 |

每个都带 `ref/verify.md`(当前**休眠**)。

---

## 4. 角色 Agent 的实现状态

| 角色 | 代码 | 状态 |
|---|---|---|
| ① 反应式 Planner | `agents/planner.py`:`ReactivePlanner` / `TwoLevelAgent` | **已实现并接主循环**。`LLMPlannerPolicy`(多模态,看场景图)+ `ScriptedPlannerPolicy`(离线) |
| ② 执行 Coding Agent | `agents/executor.py`:`CodeAsPolicyAgent` | **已实现并接主循环**。`CodingAgent` 子类;渐进披露整库;`primary_skill_id` 提示先读哪个高层技能;`FINISH`/`terminated` 终止 |
| ③ 验证(env 信号) | `verification/verifier.py`:`TaskSignalVerifier` | **已实现并接主循环**:执行器每个 python 轮默认调用,只看 env 的 `reward/terminated/task_completed` |
| ③ 验证(verify-as-code) | `agents/verifier.py`:`VerifyCodeAgent`(+ `verification/context.py`、`vlm_judge`、`VLMJudgeVerifier`、`CompositeVerifier`) | **占位/休眠**:代码完整、有离线测试,但**未接主循环**(见 §6) |
| ④ Evolve | `agents/evolve.py`:`SkillDistiller` | **占位/no-op**:`evolve()` 恒返回 `None`,且主循环**无调用方** |

---

## 5. 沙箱与执行(已实现)

- `core/sandbox/action_block.py`:`SemanticActionBlock`(自然语言意图 + python 代码)、
  `BlockExecutionResult`(ok/status/stdout/stderr/reward/terminated/observation/info…)、状态枚举、trace 事件。
- `core/sandbox/capx.py`:`CapXExecutorAdapter` 用鸭子类型驱动 CapX env(`step` / 可选 line-trace),
  把结果归一化成 `BlockExecutionResult`。
- `core/coder/action.py`:`BlockExecutor` 协议——内核只依赖 `run_block(block) -> result`,离线可注入 mock。

---

## 6. 验证现状:env 信号已接 / verify-as-code 休眠

- **已接**:`TaskSignalVerifier`——优先级"沙箱/运行时报错→失败;`terminated`(reward 1.0)或
  `task_completed`→通过;否则 uncertain"。它是执行器默认 verifier,逐轮产出 `VerificationResult`
  写入 `TurnRecord`。
- **休眠(代码在,仅测试引用,未接主循环)**:
  - `VerifyCodeAgent`(`agents/verifier.py`):与执行器同核的独立验证 Coding Agent,读
    `VerifierContext`(脱敏 op-trace + rubric)、可 seed `vlm_judge` 写验证代码。
  - `verification/context.py`:`VerifierContext` / `sanitize_code` / `build_op_trace` /
    `collect_verify_resources`(把技能的 `ref/verify.md` + `scripts/verify.py` 路由进来)。
  - `verification/primitives.py` 的 `vlm_judge`、`verification/vlm_judge.py` 的 `VLMJudgeVerifier`、
    `verifier.py` 的 `CompositeVerifier`:均定义完整但**默认循环不使用**。

---

## 7. 感知 / 证据(已实现,默认不启用)

- `perception/`:`EvidenceCollector`(按块存 before/after RGB + 合成对比图)、`render.py`、`evidence.py` 数据类型。
- `CodeAsPolicyAgent` 和 `RoboMExConfig` 都接受 `collector` 参数,但**默认 `None`**,两个 demo 也未传入——
  即证据采集链路已实现但**当前关闭**。

---

## 8. 入口与测试(已实现)

- **唯一接线入口**:`core/session.py` 的 `RoboMExConfig`(收集 library / 两个 policy / executor /
  可选 verifier·collector / 各上限 / 可选 inner system prompt / 可选 `artifacts_dir`)+ `RoboMExAgent`。
- **日志 + 产物落盘**:`core/logging.py` 的 `configure_logging`(控制台 + 可选文件)。运行时输出:
  planner 每步决策、内层每轮 `USE SKILL` / 代码行数 / 执行状态·reward·terminated / 裁决。给定
  `artifacts_dir` 时落盘:`planner.jsonl`(每步原始回复+决策)、`subgoal_NN/turn_*.py`(内层代码)
  + `turn_*.out.txt`(stdout/stderr/状态/裁决)、`summary.json`(episode 汇总)。真机脚本另存
  `run.log` 与 `video_episode*.mp4`。
- **离线**:`examples/run_planner.py`——`ScriptedPlannerPolicy` + `ScriptedCodePolicy` + `MockExecutor`,
  零依赖,已验证可跑。
- **真机**:`examples/run_planner_live.py`——`LLMPlannerPolicy` + `LLMCodePolicy` + `CapXExecutorAdapter(env)`
  + `TaskSignalVerifier`,经 `RoboMExConfig/RoboMExAgent` 装配。依赖:LLM 代理 `:8110`、感知服务
  sam3 `:8114` / contact-graspnet `:8115` / pyroki `:8116`、CapX env(`MUJOCO_GL` 默认 `egl`)。
- **测试**:`test/test_coding_agent.py`——离线驱动执行器与 `VerifyCodeAgent` 雏形,断言共享循环
  (渐进披露、python 轮 + 反馈、终止解析、重复动作 / 强制终止护栏)。

---

## 9. 明确**尚未实现**(代码里没有,勿假设存在)

- verify-as-code **未接主循环**(`VerifyCodeAgent` 只在测试里跑);默认验证只有 `TaskSignalVerifier`。
- Evolve **没有真实蒸馏**(`SkillDistiller` 是 no-op 且无调用方);`update_utility` 无人调用。
- **多模态证据回灌**到执行 agent 的 prompt 未实现(`collector` 默认关闭,且执行 policy 走纯文本)。
- 除 3 个内置技能外,**没有**其他技能(如关系指代消歧、几何高度估计、轨迹 replay 避障等都还没有代码)。
- 没有"提议新 API""技能退役"等机制;`references/`/`templates/` 等更丰富的技能目录结构尚未落地。
