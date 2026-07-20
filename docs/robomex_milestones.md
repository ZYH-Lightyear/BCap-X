# RoboMEx Harness Agentic 化里程碑

> 状态基线：2026-07-19。配套阅读：`docs/robomex_architecture_summary.md`（架构摘要）、
> `third_party/graph-as-policy/`（GaP 方法与运行时）、`third_party/open-robot-skills/`
> （GaP 的 skill/tool 库，作为 skill 设计参照物）、`third_party/qwen-code` 与
> `third_party/opencode`（coding-agent loop / 权限 / skill 注入的工程参照物）。

## 0. 愿景与定位

RoboMEx 是 subgoal-level 的 Code-as-Policy MAS 框架：外层 ReactivePlanner 逐个产出
embodied subgoal，内层 SubgoalSwarmManager 针对活场景动态编排受契约约束的小型
Coding Agent 图。与 GaP（task-level typed graph、运行时零 LLM、canonical script 人工手写）
的差异化定位：

1. **运行时灵活度**：叶子节点是活的 Coding Agent，可以在新场景下偏离既有实现；
2. **灵活度自动结晶（evolve）**：叶子自由编码的成功轨迹经蒸馏固化为 skill 函数，
   后续一轮调用——GaP 的 canonical scripts 靠人写，RoboMEx 让系统自己生产。

要让这个故事成立，前提是 harness 本身稳定、可归因、可验收。本文档定义达到该状态的
里程碑与验收标准。

## 1. 设计原则（对照 GaP 收敛）

1. **Declare, don't infer**：路由、出口、能力全部来自结构化声明，禁止从自由文本推断。
2. **封闭词表 + 编译期校验**：Manager 笔误在图编译时报错并回喂修复，而不是运行时死边。
3. **契约优先（contract-first）**：`contract.yaml` 是机器权威面（role/ports/capabilities/
   budget/exit_conditions/functions）；`SKILL.md` 只承载机器管不了的 why（适用边界、
   hard rules 的原因、失败杠杆、多模态证据约定）。
4. **实现面 = 注入沙箱的高级原语，不是强制路径**：skill 的类型化函数进入叶子 agent
   的沙箱命名空间，agent 保留每轮决策权；只有契约本身禁止偏离的角色
   （action_executor）走"脚本优先、agent 兜底"。
5. **为修复而写的错误**：失败消息包含"排除了什么"与"可用杠杆"（参照
   open-robot-skills `plan_grasp.py` 的异常文案），服务下游 LLM 修复回路。
6. **成功判定归 verifier/环境，不归执行者自述**：executor 自检永远是 soft evidence。

## 2. 现状基线

### 已完成（M0，2026-07-15）

- 封闭边事件词表 `robomex/core/edge_events.py`（success/failed/failed_grasp/
  wrong_grounding/stale_observation/infeasible/exhausted/uncertain + 别名归一化）；
- 图编译期拒绝词表外事件；`_exit_event` 删除子串匹配，只路由结构化
  `failure_kind`（finish 结果 / verdict metadata / 类型化异常 `StaleArtifactError`）；
- specialist finish 协议新增 `failure_kind` 契约；Manager prompt 披露完整词表；
- 修复 `authoring → agents → core → session → authoring` 循环导入
  （`robomex/core/__init__.py` 惰性解析 session）。

### 已完成（M1，2026-07-17 代码面；2026-07-18 live 复验通过）

- **B2**：`MotionLeaseGuard` 在世界改变 block 结束后自动采集
  `terminal_robot_state`（末端位姿/关节/夹爪开度 + epoch），写入
  `BlockExecutionResult.info` 并缓存到 `RuntimeSafetyState`；执行反馈消息附带
  该状态并明示"引用它，别调 get_observation"；adapters 把当前 epoch 的采集
  投影进 `execution_evidence` payload；`grasp_object`/`release_at` SKILL.md
  删除自行观测要求，verifier SKILL.md 增加交叉校验指引。
- **B3**：新增 `robomex/core/payload_specs.py`（单一权威源）——
  `trajectory.v1`（feasible 布尔 + 有序 waypoints，phase 枚举、pose/joints
  二选一、数值有限、≤16 个）、`affordance.v1`（position/quaternion 有限）、
  `execution_evidence.v1`（primitives 列表 + all_primitives_ok）。三处消费：
  ArtifactStore publish 时拒绝、specialist typed-finish 门禁在 producer 预算内
  回喂修复、output contract 渲染进 producer prompt。错误消息按"修复杠杆"写。
- **B7**：`RoboMExAgent._merge_swarm_node_results` 把每个
  `AuthoringNodeResult` 确定性投影进 episode memory：沙箱 primitive traces →
  TraceStore；每次节点尝试 → AttemptRecord（含 outcome/failure_kind/
  recommended_repair）；失败节点 → Diagnosis。Planner 由此获得非空历史。
- **B8**：session 主循环每个 subgoal 结束后检查 `task_completed`/`terminated`，
  命中即 `planner_status="env_terminated"` 提前收束。
- 测试基线：2026-07-17 全量 156 passed / 5 skipped（含 payload specs、
  终态采集/投影、episodic 投影、短路信号的专项测试；顺手修复了过期的
  planner prompt 断言）。B3 之后新增/收紧的测试要点：坏 trajectory payload
  在 publish 与 typed-finish 两层都被拒、`infeasible` 走 failure_kind 而非发布、
  runtime 终态只在 epoch 一致时投影。

### live 验收状态（2026-07-18 复跑，M1 修复已验证生效；暴露 M1.5 新病灶）

- 服务编排验证通过：`scripts/serve_up.sh` 一键起 tmux session（proxy :8110 /
  sam3 / graspnet / pyroki / webui / trace UI），proxy 端到端回包正常。
- 2026-07-17 首跑（`outputs/robomex_planner_live/20260717_215732`）在启动阶段
  崩溃于 `SkillLibrary.admit` 的 `FileExistsError`（B9）。事后确认根因是
  **并发竞态**：YAML 配了 `num_workers: 3` + parallel ensemble，三个进程抢写
  同一时间戳输出目录；单进程本地复现无此问题。
- 2026-07-18 复跑（`outputs/robomex_planner_live/20260718_233219`）对 subgoal_00
  审计确认：**B2 生效**（无 capability-blocked 的 `get_observation`，
  `terminal_robot_state` 由 runtime 采集并投影进 execution_evidence）、
  **B3 生效**（无 schema KeyError，trajectory/affordance payload 全部过验）、
  **B5/B6/B7 生效**（verifier 实际运行；TraceStore/AttemptHistory/Diagnoses
  非空；planner 基于非空历史决策，未逐字重复 subgoal）。B8 本轮未触发
  （环境未提前 terminated）。
- 同一轮审计暴露了新一层病灶（登记为 N1/N2/N3，见下表）：核心诊断是
  **subagent 写碎片化、无用代码不是模型能力差，是 harness 工程逼出来的**——
  verify 节点 4 个 action 的去向：1 撞 import 墙、1 手写 importlib 样板、
  1 被 `getattr` 误杀、1 内省不透明返回对象，有效工作量为零。

### 已完成（M1.5「把 subagent 升华成合格的 coding agent」，2026-07-19）

对照 qwen-code / opencode 的工程共识（declare-don't-infer；权限管真实副作用
边界而非语言内建；错误消息为 LLM 修复而写；环境语义显式进 prompt），一次性
修复 N1/N2/N3 + B9：

- **Fix A（N1 根修）**：artifact 的 `observation_epoch` 一律由 runtime 盖章
  （`outputs_from_finish` 忽略 agent 回显值），并从 output contract 删除教
  agent 回显 epoch 的文案。此前 executor 自己的运动推进了 epoch，回显开工值
  导致自己的证据被判 stale，物理成功的抓取被作废。
- **Fix B（N2）**：`DYNAMIC_OR_PROCESS_CALLS` 移除 `getattr`/`setattr`/`delattr`，
  新增 `SAFE_BUILTINS`（内省与数据整形内建在任何 policy 下都不拦）；真边界
  保留（eval/exec/`__import__`/subprocess/os.system）。denial 消息升级为修复
  指引：列出被拒原因、该节点的授权 capability 集与对应可用 env API 名。
- **Fix C（N2，M2 functions 注入的先导）**：skill 被 preload/`use_skill` 加载时，
  runtime 立即以内部 setup 块（`runtime_setup` 标记，不占 action 预算、跳过
  能力校验）把 `<skill_root>/scripts` 挂上沙箱 `sys.path`；loaded-skill 消息
  列出模块名并明示 "already on sys.path — import 直接用，禁写 importlib 样板"。
- **Fix D（N2 中最不 agentic 的一处）**：删除 graph executor 把 `uncertain`
  静默借用 `failed` 边的兜底。`uncertain` 是一等公民：Manager 可显式声明
  `on: uncertain` 边（有界补验），不声明则整图以 `AuthoringStatus.UNCERTAIN`
  + 部分证据诚实上抛，由 Planner 决策；planner 历史渲染把 checkpoint_uncertain
  与失败明确区分。此前 executor 在下游背叛了 planner prompt 在上游"uncertainty
  is not failure"的承诺，把整条已改变世界的链打回重跑。
- **Fix E（治碎片化的根）**：`_ACTION_CONTRACT` 明示沙箱是跨 turn 持久命名空间
  （禁 `globals()['x']=x` 样板与重复 import）、每块完成一个完整阶段的减轮指引；
  budget 成为声明数据——SKILL.md frontmatter 支持 `recommended_min_actions`，
  与 contract 的 `budget` 一起渲染进 Manager 的 specialist catalog（决策归
  Manager/contract，runtime 不设硬下限）。
- **Fix F**：`verify_grasp_state.py` 的 `build_grasp_vlm_questions` 改返回
  plain dict；SKILL.md Optional Sidecars 写清全部函数签名与返回结构（"不用
  花 turn 内省"）。巡检其余 6 个 builtin sidecar：均已是 plain-dict 返回。
- **Fix G（含 B9 根修）**：`SkillLibrary.admit` 的 sidecar 拷贝改
  `copytree(dirs_exist_ok=True, ignore=__pycache__)` 幂等化，消除并发竞态窗口；
  `compact_json` 对自身产出的 summary 幂等（不再把已压缩数据二次包成
  `{"type":"dict","repr":...}` 套娃，manifests / evidence packet / planner
  可见证据同时受益）。
- 测试基线：2026-07-19 全量 **168 passed / 5 skipped**（新增 Fix A–G 专项用例：
  epoch 盖章、内省内建放行 + denial 指引、sys.path 注入零预算、uncertain
  不借 failed 边 / 显式 uncertain 边路由、budget-as-data catalog、sidecar
  plain dict、admit 幂等、compact 幂等）。

### 已完成（M1.7「弱基模数据面：机械传递，不靠听写」，2026-07-19）

20260719_183117 复跑确认 M1.6 生效，但暴露更深一层：`compute_grasp` sidecar
算出正确 top-down quat，agent 在 finish 手写了编造的 `[1,0,0,0]`（identity=
夹爪朝天），IK 静默 fallback 掩护，执行期 4 个 goto 全不收敛。根因是 **typed
finish / 跨节点输入都要 LLM 听写数值**。按减法原则落地：

- **`result_var` finish**：`finish.args.result_var` 由 runtime 从沙箱命名空间
  物化（测试双暴露 `sandbox_namespace`；CapX 走 `runtime_setup` + stdout
  sentinel），再过既有端口/spec/路径门禁；缺失/大数组 → `typed_finish_rejected`
  可修复。模型只指认变量，不再复述位姿数字。
- **`INPUTS` 注入**：`_setup` 把 resolved inputs seed 进沙箱；初始消息要求代码
  引用 `INPUTS["port"]`，禁止从 prompt 抄数值。
- **契约收缩**：`_ACTION_CONTRACT` / output contract 首推
  `NODE_RESULT` + `result_var`；`_ARTIFACT_CONTRACT` 加出处规则（禁止无出处
  位姿字面量）。
- **本体常量下沉**：8+ skill Reference Code 改为可照抄闭环；
  `grasp_object` / `plan_bounded_motion` 写明 top-down=`[0,1,0,0]`、
  identity 永远不对；`solve_ik` docstring 示例改 top-down；fallback 时
  `print` 进 stdout + 可选 `return_info=True`（`orientation_used`）。
- 测试基线：全量 **198 passed / 5 skipped**（`test_result_var_data_plane.py`、
  `test_solve_ik_info.py`）。

### 已完成（M1.6「finish 收尾关卡前移 + skill 参考代码」，2026-07-19）

27B 小模型 live 复跑（20260719_173232）审计结论：**5 个 subgoal 的物理工作
（分割/几何/IK/执行）几乎全对，全部死在 typed finish/artifact 发布这道"收尾
关卡"，且关卡在 agent 循环结束后才检查，agent 无修复机会**。四类死因：
CWD 相对路径逃逸 artifact root（sg00/sg04）、多返回未声明端口
`verifier_report`（sg01，系 output contract 对所有角色渲染 verifier 文案所教）、
sidecar 无签名文档致 5 轮预算烧在 `dir()`/`inspect` 探索（sg02）、相对路径
按 CWD-优先解析产生双重路径（sg03）。修复不动架构，只做三件事：

- **finish 门前移**：`CodingAgentSubAgent._on_terminal_turn` 新增未声明端口
  拒绝（列出应返回的 exact keys）与 artifact 路径校验
  （`robomex/core/artifact_paths.py` 的 `finish_artifact_path_errors`），错误
  以可修复文案反馈给 agent，在其自身预算内重试；此前这些错误由
  `outputs_from_finish` 在 `agent.run()` 之后抛出，直接杀死节点。
- **路径规则唯一化**：相对 artifact 路径一律按节点 ARTIFACTS_DIR 解析
  （`resolve_artifact_ref_path`，gate 与 `_file_refs` 共用），删除"CWD 下存在
  则按 CWD 解析"的二义性；output contract 明示 ARTIFACTS_DIR 约定与
  "return exactly these output keys and no others"；verifier_report 文案仅在
  该端口确实声明时渲染。
- **SKILL.md 增加 `## Reference Code`**：8 个带 sidecar 的 builtin skill 全部
  写入精确签名 + 规范调用代码块 + 返回键清单（含 ARTIFACTS_DIR 存盘与
  finish artifacts 引用写法）；`plan_bounded_motion` 明示 pick 计划止于 lift
  （sg00 曾生成多余的 release/home 航点）。模板测试改为：允许 ```python，
  且带 scripts/ 的 skill 必须有 Reference Code 节——该节同时是未来
  subagent evolve 把"运行时探索成果"写回 skill 的容器。
- 测试基线：2026-07-19 全量 **190 passed / 5 skipped**（新增
  `test_artifact_paths.py`：路径规则、gate 拒绝→修复闭环、未声明端口拒绝；
  `test_payload_specs.py` 增补 contract 条件渲染用例）。

### 已确认的缺陷清单（来源：20260715_115453 / 20260717_215732 / 20260718_233219 审计）

| # | 缺陷 | 根因 | 状态 |
|---|---|---|---|
| B2 | ActionExecutor 被 capability 拦截烧光预算 | `grasp_object` prose 要求记录 terminal state，契约禁止 `perception_read`；观测与运动混在同一 block 导致整块 SKIPPED | 已修（M1，live 复验通过） |
| B3 | trajectory payload 键名漂移致 KeyError | `robomex.trajectory.v1` 等 schema 只校验"是 dict"（`artifacts.py` 的 `_mapping_payload`） | 已修（M1，live 复验通过） |
| B4 | 恢复边从不触发 | 事件词表不匹配 | 已修（M0） |
| B5 | verifier 从未运行 | B2+B4 的下游后果 | 已消除（live 复验通过） |
| B6 | Planner 逐字重复 subgoal | B7 导致反馈为空 | 已消除（live 复验通过） |
| B7 | TraceStore/AttemptHistory/Diagnoses 全空 | dynamic_swarm 模式下节点结果 → episodic memory 的投影未接线 | 已修（M1，live 复验通过） |
| B8 | 环境 terminated 不短路 episode | session 主循环只事后计算 env_success | 已修（M1，live 未触发场景，单测覆盖） |
| B9 | live 入口在 `library.admit` 崩溃（`FileExistsError: .../scripts`） | 并发竞态：多 worker 抢写同一输出目录，admit 的 copytree 不幂等 | 已修（M1.5 Fix G，待 live 复验） |
| N1 | execute 成功后自己的证据被判 stale（`StaleArtifactError`），成功抓取被作废 | prompt 教 agent 回显 `observation_epoch`，`outputs_from_finish` 让回显值覆盖 runtime 当前值；而 executor 的运动已推进 epoch | 已修（M1.5 Fix A，待 live 复验） |
| N2 | verify 节点 4 action 预算烧光仍无 verdict；产出碎片化样板代码 | 四重工程摩擦：sidecar import 无准备（Fix C）、`getattr` 被误杀（Fix B）、sidecar 返回不透明对象（Fix F）、`uncertain` 被 executor 静默改写成 `failed` 触发整链重跑（Fix D）；叠加 loop 语义不明（Fix E） | 已修（M1.5 Fix B/C/D/E/F，待 live 复验） |
| N3 | 落盘 manifests / evidence packet 出现 `{"type":"dict","repr":...}` 套娃 | `compact_json` 被多次应用且不幂等，二次压缩把已压缩 summary 再包一层 | 已修（M1.5 Fix G，待 live 复验） |
| N4 | finish / 跨节点输入靠 LLM 听写数值；弱模型编造 identity quat 当 top-down | 沙箱变量与 LLM 上下文只有 stdout；payload 无出处校验；`solve_ik` 静默 fallback | 已修（M1.7 result_var + INPUTS + 常量下沉 + return_info，待 live 复验） |

### skill 库审计结论（2026-07-16，对照 open-robot-skills）

- 分层 taxonomy（task/perception/affordance/motion/verification）与"task skill 指导
  Manager 编排、specialist skill 指导对应 agent"的双层结构是对的，与 GaP 同构，不推倒；
- 载体有缺口：契约缺**控制面**（无 exit_conditions）与**实现面**（无可注入函数）；
  prose 与契约可矛盾且无校验（B2 即实例）；schema 是名义类型（B3 即实例）；
- motif 类 skill（`grasp_with_bounded_offsets_and_verify` 等）在 swarm 模式下无消费者，悬空。

---

## 3. 里程碑

### M1 — Harness 正确性（不动格式，只修 bug）✅ 代码面完成，live 验收待跑

**目标**：一次 live episode 中，执行链路能走到 verifier，episodic memory 有真实内容。

任务（全部完成，实现摘要见 §2「已完成（M1 代码面）」）：
1. ✅ **B2**：终态采集由 runtime 代办——`MotionLeaseGuard` 在世界改变 block 结束后把
   robot state 写入 `BlockExecutionResult.info`，adapters 投影进 `execution_evidence`；
   `grasp_object`/`release_at` SKILL.md 删除 executor 自行观测的要求。
2. ✅ **B3**：`robomex/core/payload_specs.py` 为 `robomex.trajectory.v1` /
   `robomex.affordance.v1` / `robomex.execution_evidence.v1` 提供真验证器
   （waypoints 结构、phase 枚举、joints/pose 二选一、数值有限）；同一份规范
   渲染进 producer 的 output contract。坏 payload 在 typed-finish 门禁与
   publish 两层被拒，错误回喂给 producer 在自己预算内修复。
3. ✅ **B7**：`_merge_swarm_node_results` 把 `AuthoringNodeResult` 投影进
   TraceStore/AttemptHistory/Diagnoses。
4. ✅ **B8**：session 主循环监听 `terminated`/`task_completed` 并以
   `planner_status="env_terminated"` 提前收束。

**验收进度**：
- ✅ 单测：2026-07-19 全量 168 passed / 5 skipped；
- ✅ live（20260718_233219）：无 capability-blocked 的观测调用、无 schema
  KeyError、verifier 实际运行、`attempt_history.json` 非空。B8 短路未触发
  （环境未提前完成），由单测覆盖。M1 验收关闭；同轮暴露的 N1/N2/N3 已在
  M1.5 修复，待下一次 live 复验（人工执行）。

### M1.5 — 把 subagent 升华成合格的 coding agent ✅ 代码面完成，live 复验待跑

**目标**：消除逼 subagent 写碎片化代码的工程摩擦（详见 §2「已完成（M1.5）」）。
Fix A–G 全部落地并有专项测试。live 复验观察三个信号：
1. execute 成功后不再 stale 自失效（N1）；
2. verify 在预算内出 definitive verdict，或 `uncertain` 被 Planner 正确消化
   而非触发重抓（N2/Fix D）；
3. subagent 单 turn 代码块完成完整阶段，无 importlib/globals 样板（Fix C/E）。

### M1.7 — 弱基模数据面（机械传递） ✅ 代码面完成，live 复验待跑

**目标**：Agent Swarm 节点间数值通信不再依赖弱模型听写（详见 §2「已完成（M1.7）」）。
live 复验观察：
1. finish payload 中不再出现 stdout / sidecar 无出处的位姿字面量；
2. plan 节点用 `return_info` 在规划期拦下坏姿态；
3. subagent（qwen3.5-27b）优先走 `result_var` + `INPUTS` 引用范式。

### M2 — 契约载体升级（skill 重写的前置） ✅ 代码面完成，live 复验待跑

**目标**：`contract.yaml` 成为完整的机器权威面，加载期能拦截 prose/契约矛盾。
（M1.5 的 Fix C——runtime 替 agent 完成 scripts/ 的 import 准备——是本里程碑
`functions:` 注入的先导：沙箱侧的加载通道已打通，M2 只需把契约声明接上。）

落地内容（全部有专项测试，`robomex/test/test_skill_contract_upgrade.py`，21 项）：

1. **`exit_conditions:`**（`dysc/contracts.py`）——每 skill 声明自己的出口事件子集
   及语义（`{event: 含义}`）。编译期：`SubgoalGraphSpec.validate` 按源节点声明
   校验出边，监听未声明事件的边被拒绝为"永久死边"（错误消息列出该节点可路由
   事件集）。runtime 自身可对任意节点触发的事件
   （`failed/exhausted/uncertain/stale_observation`，`core/edge_events.py::RUNTIME_EDGE_EVENTS`）
   始终可路由，不受声明约束。运行期：specialist 抛出未声明的 `failure_kind` 时，
   adapter 降级为通用 `failed` 并 emit `undeclared_failure_kind_dropped` 事件
   （编译器从未为它校验过边，直接路由等于走未验证路径）。未声明 `exit_conditions`
   的旧契约保持全词表行为，零迁移成本。
2. **`functions:`**——`{name, entry: "scripts/x.py:fn", signature?, description?}`。
   skill 加载时（`use_skill`/preload），`CodingAgent._ensure_skill_contract_bindings`
   用 runtime_setup 块把函数直接绑定进沙箱命名空间（importlib，按 skill root 幂等
   缓存）；签名由 loader 从源码 AST 如实推导（声明的 `signature` 优先），随
   skill-load 消息告知 agent"已定义、直接调用、无需 import"。非强制路径：
   agent 可调用、改参、绕过。
3. **`prompts:`**——`{name, path: "prompts/x.md"}`。模板文本在 host 侧读出、
   以字面量注入沙箱 `PROMPTS[name]`；skill 包新增 `prompts/` 侧车目录
   （`SKILL_PACKAGE_EXTRA_DIRS` 已扩展，admit 会拷贝）。机制 + 单测已就绪；
   builtin 暂无打样——verify 技能用 sidecar 函数动态构造 VLM 问题，比静态模板
   更合适,不为打样而重复。
4. **loader 一致性校验**（`dysc/contract_checks.py`，`load_skill_contracts` 默认
   开启）：函数 entry 文件存在且函数在模块顶层可内省（AST）、prompt 文件存在、
   exit_conditions ⊆ 闭合词表且必须含 `success`、拼写必须是规范形（拒绝
   `failure` 等别名）；prose/契约矛盾尽力校验——SKILL.md 的 ```python 代码块经
   AST 调用扫描,若调用了 forbidden_capabilities 对应的 env API 即报错
   （M1-B2 的教训固化为编译期防线）。一次性报出全部问题。
5. **Manager 可见面**：`_specialist_catalog` 渲染 `exit_conditions`（含语义）与
   `functions` 签名；Manager prompt 的 Edge events 段说明按节点声明子集校验。

打样：`plan_bounded_motion`（`exit_conditions: success/infeasible` +
`build_place_trajectory` 函数）、`verify_grasp_and_lift_via_robot_state`
（`success/failed_grasp/wrong_grounding` + `verify_grasp_with_vlm` 函数）、
`grasp_object`（`exit_conditions: success`——物理成败归下游 verifier 判定）。
两个打样 skill 的 Reference Code 已改为直接调用预绑定函数（不再示范 import）。

顺带修正：诚实声明失败的 finish（`ok:false` 或带 `failure_kind`）不再被
required-ports 门拒绝——端口契约管的是成功交接,强行要求会教 agent 捏造
payload；agent 附带的输出仍照常走 spec 校验。

**验收**：单测全绿（219 passed）。live 复验观察：
1. Manager 给 verify 节点挂 `on: failed_grasp` 边能编译通过并真实路由；
2. plan/verify 叶子第一轮直接调用预绑定 canonical 函数（无 import/importlib 样板）；
3. 无 `undeclared_failure_kind_dropped` 事件刷屏（有则说明契约声明面偏窄，按需补）。

### M3 — Skill 库重写（代码完成，live 复验待跑）

**目标**：18 个 builtin skill 迁移到新模板，内容质量对齐 open-robot-skills。

模板要求（每个 skill）：
- contract：完整六面（role/capabilities/ports/budget/exit_conditions/functions）；
- SKILL.md：**When to use / When NOT to use**（现库普遍缺 NOT）、hard rules 带 why、
  每个 failure_kind 的修复杠杆、**多模态证据约定**（各阶段消费/产出哪些视觉证据、
  overlay 规范、observation epoch 新鲜度要求）；
- 长篇 rationale 移入 `references/` 懒加载；
- task skill（pick/place）正文对齐"推荐节点流 + 策略选择表"的写法，但保持
  "生成适配活场景的图"的原则（不固化 nodes/edges）。

顺带决断 motif 层：内容并入对应 task skill 的失败杠杆段，或改造成 Manager
可消费的图模式模板；不再悬空。

**验收**：`gap skills check` 式的自检脚本对全库绿灯（格式、引用、契约一致性）；
live episode 中 Manager 基于新 task skill 编排出的图结构合理（人工 review 一次）。

**完成状态（2026-07-20）**：
- 18 个原始 builtin 中，2 个悬空 motif 已分别并入 `pick_object` /
  `place_object` 的失败修复杠杆；运行库收敛为 16 个机器权威 skill package。
- 16/16 contract 均显式声明 role/capabilities/ports/budget/exit_conditions/functions，
  16/16 SKILL.md 均包含 When NOT to use 与 Multimodal Evidence Contract。
- 抓取 canonical builder 结构性保证 approach/lift 高于抓取 TCP；抓取/释放执行器
  关键 primitive 失败即停止并发出 `execution_fault`；verifier 发出可路由的
  `failed_grasp` / `failed_placement` / `wrong_grounding`。
- cuRobo 已作为可选双轨后端暴露，规划与执行 capability 仍严格分离；默认保持
  canonical 短段 Cartesian 路径。
- `python -m robomex.scripts.check_skills` 自检 16 package 绿灯。代码侧单测完成，
  live checklist（恢复边真实流量、轨迹无桌下航点、Manager 图人工 review）待复跑。

### M4 — 稳定性验收 + 效率基线

**目标**："完善的基建"有可量化的定义，为 evolve 和 paper 提供对照基线。

验收 checklist（一次 live episode 全绿）：
1. 每个 subgoal 到达 verifier 并产出 verdict（无 `verification: not_run`）；
2. 内部 authoring status 与 `env_success` 一致（不再"内部全败、环境成功"）；
3. planner 收到非空 AttemptHistory/TraceStore，不逐字重复 subgoal；
4. 零 capability-blocked、零 schema KeyError；
5. 恢复边至少真实触发一次且路由正确（failure_kind 链路有流量）。

效率基线（记录，供 M5 消融）：每 subgoal LLM calls / action turns / 时长，
函数注入命中率（叶子第一轮即调用 canonical 函数的比例）。

### M5 — Evolve 闭环 + Paper 实验

**目标**：把"灵活度结晶"做成机制，支撑论文主张。

任务：
1. 蒸馏管线：Tier 2 自由编码成功轨迹 → 参数化 → 生成 fixtures 单测（参照
   open-robot-skills `tests/` 的 FakeContext 模式）→ 经 gate 提升为 `functions:` 条目；
2. `SkillDistiller.evolve` 从 no-op 变为上述管线的入口；
3. 消融实验：有/无 functions 注入、有/无 failure_kind 路由、有/无蒸馏闭环，
   指标 = 成功率、LLM 成本、失败归因粒度；对照 = universal 单 agent baseline 与
   （可行时）GaP 静态图。

**验收**：至少一个技能函数由系统自动蒸馏产生并在后续 episode 中被 Tier 0/1 消费；
消融表格可直接进论文。

---

## 4. 非目标

- 不复刻 GaP 的 task-level 全局持久图与 rehearsal 机制（定位差异，见 §0）；
- 不追求 M1-M3 阶段的成功率数字，只追求链路正确与可归因；
- 不在 M5 之前引入学习类组件（VLA policy 节点等）。

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| 函数注入后叶子 agent 退化为"只会调函数"，灵活度名存实亡 | M4 记录函数命中率与偏离率；M5 消融中保留无函数对照组 |
| schema 收紧后 producer 修复回路预算不足 | 验证错误消息按"修复杠杆"格式写；预算数据驱动再调 |
| skill 重写工作量大 | M2 先打样 2 个，M3 按 task → 高频 specialist → 长尾的顺序分批 |
| live 验收受环境随机性干扰 | 验收跑 3 seeds；checklist 按"链路正确性"而非成功率判定 |
