# RoboMEx × LIBERO-PRO：High-Level Implementation and Evaluation Plan

> 日期：2026-07-23
> 定位：将 LIBERO-PRO 作为 Belief-Centered Elastic Coding Swarm 的首个系统性仿真验证场，而不是为每个任务手写 workflow。
> 上位架构：[RoboMEx Belief-Centered Swarm Master Plan](robomex_belief_centered_swarm_master_plan.md)

## 0. 结论

这套方案有能力覆盖当前 CaP-X 默认评测中的大部分 LIBERO-PRO 任务，但难度要分层讨论。

- **可以优先完成**：Object-Swap/Object-Task 的十类 object-to-basket；大多数 Goal pick/place；除 drawer case 外的大多数 Spatial relational pick/place。
- **可以完成，但需要新增通用能力**：open drawer、drawer 内取放、push plate、turn on stove。
- **不能靠增加 Agent 数量自动解决**：articulated manipulation、持续接触、开关/旋钮操作。如果没有对应的 belief、affordance、motion protocol 和 verifier，Swarm 只会并行地产生更多不可靠代码。

LIBERO-PRO 非常适合验证 RoboMEx 的三个核心论点：

1. **World/Context**：同一场景里存在两个外观相同的黑碗，任务通过 between、next to、on、in 等关系指定目标；这正好测试 role-specific World Panel 和 competing grounding hypotheses。
2. **Elastic Swarm**：Swap/Task perturbation 会改变布局或任务组合；Manager 应根据当前 uncertainty 选择一个 Grounder、多个 Grounders、active perception 或不同 motion candidates，而不是调用固定 Bowl Graph。
3. **Dynamic execution**：drawer、push、precise placement 都要求动作后重新观察；这正好测试 action-conditioned belief、bounded micro-actions、Monitor 和 pending-plan replacement。

当前代码和历史日志也提供了“可行但不稳定”的直接证据：

- 非特权配置已经暴露 RGB-D、SAM3/VLM grounding、ContactGraspNet、PyRoKi/CuRobo 接口；
- 旧 RoboMEx 在一组 30-trial Object-Swap batch 中取得 10 次环境成功，其中 alphabet soup 为 3/3、cream cheese 为 2/3；
- 同一批数据中 env success 与 Agent 自报 success 明显不一致，且后半任务出现 runtime errors。

因此正确结论不是“RoboMEx 已能完成 LIBERO-PRO”，而是：

> 基础 manipulation path 已经被验证为可达；新的 EBW、WorldPanel、SwarmPlan、Verifier 和 sealed runtime 有明确的失败数据与任务结构可以验证其增量价值。

## 1. 评测范围

第一阶段聚焦 CaP-X 默认的六个 10-task suites：

- libero_object_swap；
- libero_object_task；
- libero_spatial_swap；
- libero_spatial_task；
- libero_goal_swap；
- libero_goal_task。

Swap 与 Task 版本共享自然语言任务集合，但扰动设置不同。因此是 60 个 suite-task cells、30 个独特语言目标。

不在第一阶段直接扩到 LIBERO-10、LIBERO-90 或全部 Obj/Sem/Env perturbations。只有六套核心评测稳定后，才把它们作为 out-of-distribution 扩展。

## 2. 任务族可行性矩阵

### 2.1 Tier A：Rigid pick-and-place

包括：

- Object 0–9：把指定食品/容器放入 basket；
- Goal 1、2、4、6、8、9：把 bowl、wine bottle、cream cheese 放到 stove、cabinet、bowl、plate 或 rack；
- Spatial 中目标已经明确且可见的 bowl-to-plate。

所需能力：

- semantic/instance grounding；
- rigid-object geometry；
- grasp hypothesis；
- collision-aware transport；
- support/container placement；
- attachment 与 placement verification。

与当前 API 和已有 Skills 高度匹配，属于最高可行性任务。

主要失败不会来自 Task Planner，而来自：

- mask 选错实例；
- depth/OBB/affordance 偏差；
- 抓持后 object-to-TCP pose 不确定；
- placement 仍使用固定 world-axis offset；
- Agent 自报完成但 BDDL predicate 未满足；
- action 后目标被遮挡或对象滑移。

这些正是新架构首先要解决的问题。

### 2.2 Tier B：Relational grounding

包括 Spatial 任务：

- between plate and ramekin；
- next to ramekin/cookie box/plate；
- table center；
- on cookie box/ramekin/stove/cabinet；
- in top drawer。

这些场景通常有两个外观相同的黑碗。任务不是检测“black bowl”，而是选择满足空间关系的那个实例。

所需新增能力：

- 多实例 entity registry；
- plate、ramekin、cookie box、stove、cabinet 等 anchor grounding；
- On、In、NextTo、Between 的 relation hypotheses；
- relation confidence 与 freshness；
- relational Grounding Arbiter；
- ambiguity 时的 alternate view 或 wrist-camera evidence。

除 top-drawer case 外，motion 本身仍是 rigid pick/place；真正难点是目标身份。它是验证 Grounding Swarm 最合适的任务族。

### 2.3 Tier C：Articulated manipulation

包括：

- Goal 0：open middle drawer；
- Goal 3：把 bowl 放入 top drawer；
- Spatial 4：从 top drawer 取 bowl 再放到 plate。

需要新增：

- cabinet、drawer region、handle 的 part-whole belief；
- handle mask/pose；
- prismatic joint axis 和 open fraction hypotheses；
- pre-grasp、handle grasp、bounded pull、release；
- drawer-state Monitor；
- drawer motion 后 scene geometry/collision world 更新；
- 对 drawer 内目标的重新 grounding。

这里不能把 open drawer 简化为一次 goto_pose。正确执行是：

~~~text
ground handle -> estimate axis -> approach -> establish contact/grasp
-> pull a small segment -> observe drawer delta
-> update axis/open fraction -> continue or correct
-> verify region is accessible
~~~

### 2.4 Tier D：Contact-rich non-grasp manipulation

包括 Goal 5：push plate to the front of the stove。

需要新增：

- plate footprint、support plane、front-of-stove goal region；
- push affordance：contact point、direction、height、stability；
- bounded Cartesian push primitive；
- contact-preserving motion/velocity limits；
- plate tracking 和 relation error；
- lost-contact、rotation、overshoot detection；
- action-after-observation correction loop。

Motion Swarm 可以提出不同 contact sides 或 push distances，但每次只能 admission 一条短 micro-push。

### 2.5 Tier E：Control-state interaction

包括 Goal 7：turn on the stove。

需要先确认具体 LIBERO asset 是 button、switch 还是 rotary control，再注册对应 skill family。

通用能力：

- control-part grounding；
- current control state；
- interaction affordance；
- push/rotate trajectory；
- contact-state Monitor；
- independent TurnOn verifier。

这是当前六套评测中风险最高的一类。它不应阻塞 Object/Spatial/placement 论文实验，但必须作为“架构能否吸收新 manipulation primitive”的后期验证。

## 3. LIBERO-PRO 中的目标架构

~~~text
LIBERO task language
        |
        v
Reactive Task Planner -------------------------------+
        |                                            |
  SubgoalIntent                                      |
        |                                            |
        v                                            |
Swarm Manager <----------- Manager WorldPanel        |
        |                                            |
  SwarmPlan / Delta                                  |
        |                                            |
        v                                            |
Deterministic SwarmPlanCompiler                      |
        |                                            |
  short-lived Graph IR                               |
        |                                            |
  +-----+---------+----------+-----------+           |
  |               |          |           |           |
Grounding      Affordance   Motion     Monitor        |
Agents         Agents       Agents     Agent          |
  |               |          |           |           |
  +-------- hypotheses / candidates -------+         |
                         |                            |
                    Arbiter/Gates                     |
                         |                            |
                   Sealed Action                      |
                         |                            |
                LIBERO Action Backend                 |
                         |                            |
             RGB-D / robot state / receipt            |
                         |                            |
                         v                            |
             Embodied Belief Workspace --------------+

LIBERO BDDL truth / reward ---> hidden Evaluator only
~~~

关键原则：

- policy path 只接收 task language、RGB-D、robot state、已允许 API 结果；
- privileged object poses、BDDL internal state 和 reward 不进入 EBW 或 WorldPanel；
- BDDL predicate 只用于 episode 评测和 oracle upper bound；
- replay、normal sim、fault-injected sim 使用同一 orchestration path；
- Graph 由 SwarmPlanCompiler 生成，不按 30 个任务分别手写。

## 4. LIBERO Embodied Belief Workspace

### 4.1 Entity types

- movable rigid object；
- support surface/region；
- container/interior region；
- articulated fixture；
- articulated link/drawer；
- handle；
- control/button/knob；
- robot、gripper、attached object。

### 4.2 Required beliefs

每个 entity 至少维护：

- semantic label 与 instance identity；
- pose hypotheses；
- mask/depth/point-cloud evidence；
- visibility、occlusion、track status；
- supporting surface/container；
- relation hypotheses；
- grasp/placement/contact affordances；
- freshness 与 provenance。

episode state 还维护：

- attachment；
- object-to-TCP transform hypotheses；
- drawer open fraction/joint axis；
- stove control state；
- action ledger 与 expected effects；
- tried strategies/candidate failures；
- active roster；
- unresolved questions。

### 4.3 Relation semantics

第一批关系：

- On；
- In；
- NextTo；
- Between；
- Open；
- TurnedOn。

On/In 是 committed task relations；NextTo/Between 多用于 target selection。所有关系都必须有 observation revision、supporting evidence 和 uncertainty。

### 4.4 Hidden evaluator firewall

LIBERO simulator 可以提供 ground truth，但必须分成两个实例：

- **PolicyWorld**：非特权 RGB-D、robot state、allowed APIs；
- **EvaluatorWorld**：BDDL predicates、true object states、reward。

EvaluatorWorld 只输出 metrics，不可被 Manager、Agents、Arbiter 或 Verifier读取。Oracle experiments 使用独立配置并明确标注 upper bound。

## 5. World Panels

### 5.1 ManagerWorldPanel

包含：

- task/subgoal；
- relevant entities 和 unresolved identity；
- current attachment；
- recent action-conditioned changes；
- candidate/strategy history；
- risk、budget 和 available agent profiles；
- semantic events。

不包含原始完整 RGB-D history 或 evaluator truth。

### 5.2 GroundingWorldPanel

包含：

- 当前 agentview 与 wrist-camera refs；
- candidate instances；
- relation anchors；
- visibility/occlusion；
- prior masks/tracks；
- task language 中的 relational phrase；
-允许的 SAM3、VLM、geometry probes。

### 5.3 AffordanceWorldPanel

按角色选择：

- grasp：object geometry、free surfaces、collision context；
- placement：support/container geometry、held-object footprint；
- drawer：handle geometry、joint-axis hypotheses；
- push：contact sides、goal relation、support plane；
- stove：control-part geometry/state。

### 5.4 MotionWorldPanel

包含：

- robot joints/EE/TCP；
- attachment 与 object-to-TCP hypotheses；
- goal pose/region；
- collision world revision；
- action bounds；
- candidate affordance refs；
- prior IK/collision/path failures。

### 5.5 Monitor/VerifierWorldPanel

Monitor 关注：

- expected change；
- allowed deviation；
- track/attachment/contact；
- interrupt predicates。

Verifier 关注：

- independent post-action observations；
- target task relation；
- gripper/attachment；
- drawer/control state；
- evidence conflict。

Agent 若缺少字段，只能提交 PanelExpansionRequest 或 EvidenceRequest，不能读取隐藏全局状态。

## 6. Swarm Profiles

### 6.1 Relational Grounding Swarm

候选成员：

- SAM3 text Grounder；
- VLM bbox/point → SAM3 Grounder；
- anchor-first Relation Grounder；
- temporal/wrist-camera Tracker；
- Grounding Arbiter。

选择逻辑：

- 单实例、置信且稳定：只运行一个低成本 Grounder；
- 多个同类实例：并行至少两个独立 grounding strategies；
- relation anchors 不完整：先 ground anchors；
- hypotheses 冲突：请求新视角，不直接按 confidence 最大值执行。

### 6.2 Grasp Swarm

候选成员：

- ContactGraspNet；
- geometry/OBB side grasp；
- open-bowl/rim-aware grasp；
- reachability/collision critic；
- grasp preview renderer。

Arbiter 根据 collision、IK、antipodal/contact score、task-dependent post-grasp manipulability 和 preview 选择，而不是只选网络最高分。

### 6.3 Placement Swarm

候选成员：

- support-center placement；
- container-interior placement；
- object-footprint-aware placement；
- attached-object visual servo；
- motion preview/feasibility critic。

对于 bowl-on-plate 或 object-in-basket，粗运输后进入短 micro-motion loop，直到 relation verifier 的误差进入容差。

### 6.4 Articulation Swarm

候选成员：

- handle Grounder；
- joint-axis estimator；
- handle-grasp author；
- bounded-pull motion author；
- drawer-state Monitor；
- active-view Agent。

Manager 在 pull 方向不确定或 drawer 没有按预期移动时替换 axis hypothesis 或取新证据。

### 6.5 Push Swarm

候选成员：

- contact-point proposer；
- push-direction proposer；
- contact-preserving motion planner；
- plate tracker；
- goal-region verifier。

不同 Agents 可在 shadow 中比较 push sides 和 distances，但每轮只执行一次短 push。

### 6.6 Control Interaction Swarm

候选成员：

- control-part Grounder；
- button/switch/knob classifier；
- interaction affordance proposer；
- contact motion author；
- state-change verifier。

如果 asset-specific geometry 无法从 observation 可靠推断，应将 fixture adapter 作为显式 backend capability，而不是让 Agent 猜隐藏 simulator joint。

## 7. 四个代表性运行案例

### 7.1 Object：alphabet soup → basket

1. Planner 输出 pick target 的 SubgoalIntent。
2. Manager先启用一个 Grounder；若 mask/depth 稳定，不展开额外 Agents。
3. Grasp Agent 检索 rigid-container Skills，生成多个 grasp candidates。
4. Motion candidates 经过 IK/CuRobo、preview 和 sealed admission。
5. 执行后 Monitor/Verifier 更新 AttachmentAcquired。
6. Planner 输出 place-in-basket intent。
7. Placement Agent 估计 basket interior 和 held-object footprint。
8. 粗运输后按新 observation 微调。
9. release、retreat，独立验证 In(object, basket)。

这个任务主要验证风险自适应：简单 case 不应永远启动大型 Swarm。

### 7.2 Spatial：black bowl between plate and ramekin → plate

1. Grounding Panel 展示两个 black bowl candidates 及 plate/ramekin anchors。
2. 两个 Grounders 分别执行：
   - direct relational VLM grounding；
   - ground-all-instances + 3D relation calculation。
3. Arbiter检查目标是否位于 anchors 的几何 between region。
4. 若冲突，Manager 招募 wrist-camera/alternate-view Agent。
5. identity 一旦 committed，后续 grasp/motion 绑定 stable entity id，而不是自然语言“the bowl”。
6. place 阶段使用 held-object pose 与 plate support region 做闭环。

这是 H1/H2 的首要案例。

### 7.3 Goal：open top drawer and put bowl inside

1. Planner 先产生 make-top-drawer-accessible intent。
2. Articulation Swarm 定位 handle、估计 pull axis、执行 bounded pull。
3. 每一小段后更新 drawer open fraction 和 collision world。
4. Drawer accessible 后当前 Subgoal Completed。
5. Planner 产生 pick bowl，再产生 place-in-drawer。
6. Placement Agent 使用 drawer interior region，而不是 cabinet center。
7. release 后验证 In(bowl, top drawer region)。

如果 drawer 在拉动中没有移动，Monitor 返回 ActionDiverged；Manager更换 contact/axis strategy，而不是继续执行旧 suffix。

### 7.4 Goal：push plate to front of stove

1. Grounding Agent 定位 plate 与 stove front goal region。
2. Affordance Agents 提出 plate 后侧的多个 contact points。
3. Motion Agents 在 shadow 中比较 reachability、collision 和 predicted displacement。
4. 选择一条短 push；Monitor 跟踪 contact 与 plate motion。
5. action 后 EBW 更新 plate pose/relation。
6. 未达到 goal region则重新计算下一个 micro-push；overshoot 或 rotation 触发策略更换。
7. Verifier确认 On(plate, stove-front-region)。

## 8. 实施路线

### LP0 — Benchmark Freeze 与 Failure Corpus

交付：

- 固定六个 suites、task IDs、seeds、API profile；
- 解析现有 Object-Swap batch 和 live traces；
- 将 success、false finish、segmentation、IK、placement、runtime crash 分类；
- 保存 representative replay fixtures；
- 冻结 current RoboMEx/Cap-X baseline。

Gate：

- 每个历史 episode 可定位 task、seed、API、model、env reward 和 failure stage；
- Agent finish 与 env success 分开统计；
- 不把不完整旧 batch 当正式论文结果。

### LP1 — LIBERO Adapter 与 Evaluator Firewall

交付：

- LIBERO ObservationBackend；
- RobotState/Gripper bridge；
- Sealed LIBERO ActionBackend；
- task-language adapter；
- hidden BDDL evaluator；
- replay/sim config；
- per-action observation/receipt/video。

Gate：

- policy path 无 privileged pose/BDDL/reward；
- oracle upper-bound config 与 normal config 物理隔离；
- replay 与 live-sim 使用相同 runtime path；
- env terminated 后不再发送 action。

### LP2 — EBW + Panels on Object Tasks

交付：

- rigid object、basket/container、support regions；
- attachment、object-to-TCP、In/On；
- five WorldPanel profiles；
- evidence/freshness/provenance；
- Agent/env completion disagreement 修复。

课程：

1. cream cheese → basket；
2. alphabet soup → basket；
3. butter/tomato sauce；
4. 全 Object-Swap；
5. Object-Task perturbations。

Gate：

- deterministic replay；
- no privileged leakage；
- false-finish rate 明显下降；
- Object task 的失败能归因到 grounding、grasp、motion、placement 或 verification。

### LP3 — SwarmPlan Compiler + Rigid Manipulation Swarms

交付：

- Manager输出 SwarmPlan/Delta；
- Grounding、Grasp、Placement、Monitor assignments；
- candidate arbitration；
- risk-adaptive 1→K expansion；
- Graph runtime IR；
- sealed action admission。

Gate：

- 无手写 task-specific Graph；
- 简单 episode 默认不启动多余 Agents；
- disagreement/all-invalid 会重组 Swarm；
- same-budget single Agent vs Swarm 对照可运行。

### LP4 — Spatial Relations

交付：

- multi-instance entity beliefs；
- anchors；
- NextTo/Between/On/In hypotheses；
- relation Arbiter；
- active-view protocol；
- identity continuity。

课程：

1. next to plate；
2. on stove/cabinet/cookie box；
3. next to ramekin/cookie box；
4. between plate and ramekin；
5. top-drawer target 延后到 LP6。

Gate：

- relational target identity accuracy 单独评测；
- 两个 identical bowls 不靠 BDDL instance id；
- swap perturbation 后仍根据 observation 选择目标；
- committed entity id 在 grasp/place 间保持稳定。

### LP5 — Goal Placement Generalization

交付：

- stove cook region；
- cabinet top；
- bowl/plate/rack/basket placement profiles；
- support/container geometry；
- attached-object micro-servo；
- independent relation verifier。

课程：

- bowl on plate/stove/cabinet；
- cream cheese in bowl；
- wine bottle on cabinet/rack。

Gate：

- 同一 placement core 支持至少五种 support/container semantics；
- 不为每个 task 写固定 offset；
- grasp pose 变化时仍可根据 current held-object belief完成放置。

### LP6 — Articulated Drawer

交付：

- fixture/part/handle entity model；
- joint-axis/open-fraction belief；
- handle-grasp/pull Skills；
- bounded articulation loop；
- drawer-state verifier；
- collision-world refresh。

课程：

1. open middle drawer；
2. open top drawer；
3. bowl into top drawer；
4. bowl from top drawer → plate。

Gate：

- drawer motion 不符合 expected delta 时中止旧动作；
- open fraction 来自 observation/evidence，不来自 hidden joint；
- drawer open 后重新 grounding interior object；
- interaction failure 可区分 handle miss、lost grasp、wrong axis 和 blocked motion。

### LP7 — Push 与 Stove Control

交付：

- contact affordance；
- micro-push protocol；
- contact/pose monitor；
- stove control-part adapter；
- interaction state verifier。

Gate：

- push 任务具备 action-observe-correct loop；
- turn-on 任务不使用 evaluator state做 policy input；
- fixture-specific backend 明确声明，不藏在 prompt；
- contact失效时不会继续长 open-loop trajectory。

### LP8 — Perturbation、Recovery 与 Full Evaluation

故障与扰动：

- object position swap；
- identical-object ambiguity；
- partial occlusion；
- empty/wrong masks；
- stale depth；
- grasp offset/in-gripper rotation；
- IK/collision failure；
- slip/drop；
- drawer axis error；
- push lost contact/overshoot；
- Agent code error、Manager wake、runtime restart。

Gate：

- Ambiguous、Anomaly、Unable、Completed 真实改变 roster 或 pending execution；
- drop/occlusion 后不继续旧 place suffix；
- zero physical-writer violations；
-所有 failure 均有 typed reason 和 evidence refs。

## 9. 评测设计

### 9.1 Baselines

- CaP-X / current single coding agent；
- current RoboMEx v1 dynamic swarm；
- static v2 Bowl-style workflow where applicable；
- Single + Raw History + Static；
- Single + WorldPanel + Static；
- Swarm + WorldPanel + Static；
- Swarm + WorldPanel + Dynamic Manager；
- Oracle-state upper bound，单独报告。

### 9.2 核心消融

| 因素 | A | B |
|---|---|---|
| Context | raw observations/history | EBW role-specific Panels |
| Organization | single Agent | elastic Swarm |
| Adaptation | fixed workflow | event-driven Manager |
| Perception | single Grounder | hypothesis arbitration |
| Motion | single plan | shadow candidate Arena |
| Model | small | large |

### 9.3 指标

任务：

- BDDL environment success；
- intervention-free success；
- perturbation recovery；
- success under fixed budget。

Belief：

- target identity accuracy；
- pose/relation accuracy；
- track continuity；
- stale rejection；
- unsupported Panel fact；
- attachment accuracy；
- drawer/control state accuracy。

Swarm：

- candidate diversity；
- arbitration regret；
- all-invalid detection；
- reorganization success；
- Agent lifecycle/cost。

执行：

- IK/collision rejection；
- action count/path length；
- event-to-stop；
- false finish；
- action after termination；
- unsafe/duplicate write。

成本：

- tokens、calls、GPU seconds、wall time；
- perception calls；
- shadow candidates；
- success/token 和 success/minute。

### 9.4 预算公平

- 所有方法使用相同 observation 与 action backend；
- 主对照固定总 token、model call、wall time 和 action budget；
- 另报告每种方法的 best success-cost Pareto；
- Panel、render、CuRobo、Verifier 成本全部计入；
- env reward 只作为 evaluator signal。

### 9.5 Progressive acceptance targets

下面是工程晋级目标，不是提前承诺的论文结果：

- Oracle control upper bound：每个新 skill family 先达到 80% 以上，证明 action implementation 可行；
- Non-privileged Object：完整十任务达到至少 70% env success，再进入 full spatial；
- Spatial relational：target identity 至少 90%，task success 至少 60%；
- Goal rigid placement：至少 60%；
- drawer/push/control：每个 primitive family 至少 50%，且无 unsafe continuation；
- 每个 cell 的正式样本量在 pilot 后 power analysis；开发阶段至少 20 seeds/task，论文阶段报告 95% CI。

如果 oracle upper bound 都低，优先修 action/skill/backend；如果 oracle 高而 non-privileged 低，优先修 grounding/EBW；如果单 Agent与 Swarm相同，停止增加 roster，重新检查 arbitration 和 event semantics。

## 10. 论文价值

LIBERO-PRO 不应只作为“又跑了一个 benchmark”。它应该回答：

1. identical objects + relational language 时，typed belief 与 Grounding Swarm 是否更可靠；
2. layout/task perturbation 后，Manager 是否真的改变团队和策略；
3. object 被遮挡、掉落或动作偏离时，旧 Graph 是否会被及时失效；
4. role-specific Panels 是否让小模型接近大模型；
5. Swarm 的额外成本是否换来足够成功率和恢复能力；
6. 新 primitive family 能否只通过 Skills/protocols/predicates 接入，而不改 core runtime。

最有论文辨识度的主实验不是最简单的 object-to-basket，而是：

- relational black-bowl selection；
- grasp-pose-uncertain precise placement；
- drawer action 后 world geometry 更新；
- occlusion/drop 后 dynamic Swarm reorganization。

Object suite 是稳定底座和成本实验；Spatial suite 是核心 grounding 证据；Goal suite 是 manipulation breadth 与动态闭环证据。

## 11. Definition of Done

只有同时满足以下条件，才能说“新 RoboMEx 完整支持核心 LIBERO-PRO”：

- 六个目标 suites 共用同一 application/runtime path；
- 任务差异只存在于 task manifest、predicates、protocol/Skills 和 asset capabilities；
- Manager 不生成 raw Graph；
- policy path 完全非特权；
- EBW 支持 rigid、relation、attachment、articulation、contact/control state；
- World Panels 按角色投影并可审计；
- Grounding/Grasp/Placement/Articulation/Push/Control Swarms 可按风险动态创建；
- action 后会更新 belief，并可替换 pending execution；
- Agent finish 不覆盖 BDDL evaluator；
- env termination 后绝不继续发动作；
- replay、sim、fault-injected sim 使用同一控制路径；
- Object、Spatial、Goal 三族达到预先冻结的统计门槛；
- 与 single Agent/static workflow 做同预算对照；
- 每次失败可以定位到 perception、belief、arbitration、motion、action、monitor 或 verification。

## 12. 建议的立即开发顺序

不要先同时攻 60 个 cells。第一条主线应是：

1. cream cheese → basket：打通 LIBERO Adapter、Evaluator firewall、EBW 和 Panels；
2. alphabet soup → basket：打通 Grounding/Grasp/Placement SwarmPlan；
3. black bowl next to plate → plate：加入 identical-instance relation belief；
4. black bowl between plate and ramekin → plate：验证多 anchor arbitration；
5. bowl on plate：复用 precise placement；
6. open middle drawer：加入 articulation；
7. bowl into top drawer：验证 Planner/Manager/World 跨 Subgoal协作；
8. push plate：加入 contact loop；
9. turn on stove：最后接入 control interaction；
10. 再运行六套完整 benchmark 与扰动消融。

这条顺序使每个新任务只增加一种真正的新能力，同时持续复用前面的系统，而不是把 LIBERO-PRO 变成 30 个硬编码 demo。
