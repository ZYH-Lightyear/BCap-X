# DySC 工程蓝图：Evolving Multi-Agent Skill Societies

## 目标

DySC 的第一版工程实现不能把 MAS 写成一组固定 SubAgent，也不能把 role-skill 绑定硬编码在构造函数里。否则后续会出现两个问题：

1. role、skill、evidence、motif 各自一套冗余 schema，难以统一进化；
2. 一旦想做 role birth、skill binding mutation、topology mutation，就必须改代码而不是改 genotype。

因此第一版就应把 MAS 表示为可配置、可记录、可进化的 **SocietySpec**。运行时根据 SocietySpec 实例化 agent views；学习和进化时修改 SocietySpec，而不是修改 Python 代码。

核心原则：

> MAS 不是固定 SubAgent 集合，而是 dynamic skill composition policy 的可变图表示。

## 核心抽象

### SocietySpec

SocietySpec 是 DySC 的 genotype。它描述当前这代 multi-agent skill society 如何组织 skill composition。

```yaml
schema: robomex.dysc.society.v1
name: libero_seed_society

roles:
  perception_scout:
    objective: ground task-relevant objects and relations
    model_policy: small
    execution_boundary: read_only
    skill_access_policy: perception_scout_policy
    consumes:
      - observation
      - task_language
      - failure_context
    emits:
      - object_grounding
      - relation_evidence
      - grounding_uncertainty

  affordance_geometer:
    objective: convert grounded objects into physical affordance hypotheses
    model_policy: small
    execution_boundary: read_only_with_ik
    skill_access_policy: affordance_geometer_policy
    consumes:
      - object_grounding
      - grounding.points
      - relation_evidence
    emits:
      - grasp_affordance
      - placement_affordance
      - object_frame_offset
      - affordance_uncertainty

  motion_executor:
    objective: execute bounded motion code under skill and evidence guidance
    model_policy: main
    execution_boundary: motion_allowed
    skill_access_policy: motion_executor_policy
    consumes:
      - grasp_affordance
      - placement_affordance
      - local_correction_target
    emits:
      - action_result
      - primitive_trace
      - physical_state_change

  predicate_verifier:
    objective: verify physical task predicates and relation error
    model_policy: small
    execution_boundary: read_only
    skill_access_policy: predicate_verifier_policy
    consumes:
      - action_result
      - predicate_claim
      - observation
    emits:
      - predicate_verdict
      - relation_error
      - recommended_next

skill_access_policies:
  perception_scout_policy:
    allow_tags: [perception, geometry]
    prefer:
      - relational_grounding
      - pose_alias_segmentation_fallback
    forbid_tags: [motion_execution]

  affordance_geometer_policy:
    allow_tags: [affordance, geometry, verification_light]
    prefer:
      - open_bowl_rim_grasp
      - obb_top_down_grasp
      - support_surface_place
    forbid_tags: [motion_execution]

  motion_executor_policy:
    allow_tags: [motion, motif, verification_light]
    prefer:
      - bounded_grasp_execute
      - local_xy_servo
      - controlled_release

  predicate_verifier_policy:
    allow_tags: [verification, perception]
    prefer:
      - held_state_check
      - object_target_relation_check
    forbid_tags: [motion_execution]

topology:
  - from: perception_scout
    to: affordance_geometer
    when: object_grounding.available
  - from: affordance_geometer
    to: motion_executor
    when: affordance.available
  - from: motion_executor
    to: predicate_verifier
    when: action_result.world_changed
  - from: predicate_verifier
    to: perception_scout
    when: predicate_verdict.status in [fail, uncertain]
```

### RoleSpec

RoleSpec 不是 SubAgent 类名，而是生成 agent runtime 的配置。

```text
RoleSpec = {
  objective,
  prompt_profile,
  skill_access_policy,
  consumes_evidence,
  emits_evidence,
  execution_boundary,
  model_policy,
  budget,
}
```

同一个 `CodingAgentSubAgent` runtime 可以被不同 RoleSpec 包装成 PerceptionScout、AffordanceGeometer 或 PredicateVerifier。区别来自 prompt、library view、execution policy 和 evidence contract，而不是新写多个固定类。

### SkillAccessPolicy

SkillAccessPolicy 负责决定某个 role 在当前 state 下看见哪些 skills。

它不应只支持 `allowed_skill_ids`，还应支持 tag、condition 和 evolution metadata：

```yaml
allow_tags:
  - perception
prefer:
  - relational_grounding
conditional:
  - skill: grasp_graspnet
    when: affordance_uncertainty.high
forbid:
  - goto_pose
forbid_tags:
  - motion_execution
stats:
  call_count: 0
  success_count: 0
  token_cost: 0
```

第一版可以先实现 `allow_tags/prefer/forbid/forbid_tags`，但数据结构必须为 `conditional/stats` 留位置。

### Skill Contract

`SKILL.md` 继续保留 Codex/Claude-style 的人类可读说明。结构化信息放在可选 sidecar：

```text
contract.yaml
```

示例：

```yaml
schema: robomex.dysc.skill_contract.v1
skill_id: open_bowl_rim_grasp
tags: [affordance, geometry, open_container]
changes_world: false
inputs:
  - object_grounding
  - grounding.points
outputs:
  - grasp_affordance
  - object_frame_offset
uncertainty:
  - rim_visibility
  - contact_depth
postconditions:
  - grasp_candidate_selected
suggested_next:
  success:
    - bounded_grasp_execute
  uncertain:
    - affordance_contact_check
    - grasp_graspnet
  failure:
    - resegment_object
```

这样旧的 `use_skill` 仍然只读 SKILL.md；DySC runtime 额外读取 contract.yaml 来构建 graph。

### Evidence Message

DySC 需要 typed message，而不是只传自然语言。

```json
{
  "schema": "robomex.dysc.message.v1",
  "producer": "affordance_geometer",
  "message_type": "grasp_affordance",
  "payload": {
    "pos": [0.414, 0.329, 0.035],
    "quat": [0.0, -0.60, 0.79, 0.0],
    "object_center_offset_from_grasp": [0.025, -0.008, 0.0],
    "ik_ok": true
  },
  "supports": ["candidate_grasp_pose"],
  "uncertainty": ["rim_contact_depth"],
  "recommended_next": ["bounded_grasp_execute", "held_state_check"]
}
```

消息可以被 agent 读，也可以被 graph 和 evolution 使用。

## 第一版架构

### 1. Skill Library View

新增 `SkillLibraryView`，它不复制 skill，只按 SocietySpec 过滤现有 SkillLibrary。

```python
class SkillLibraryView:
    def __init__(self, library, access_policy, contracts):
        ...

    def all(self, category=None):
        ...

    def get(self, skill_id):
        ...
```

所有 role 都通过 view 看 skill。禁止在 role 构造函数里写：

```python
allowed=["segment_object", "held_state_check"]
```

正确方式是：

```python
RoleRuntime(role_spec, library_view=build_view(role_spec, society_spec))
```

### 2. Role Runtime Factory

新增 factory，根据 RoleSpec 生成 runtime：

```python
def build_role_runtime(role_spec, base_library, executor, policies):
    library_view = build_skill_library_view(role_spec.skill_access_policy)
    execution_policy = build_execution_policy(role_spec.execution_boundary)
    system_prompt = render_role_prompt(role_spec)
    return CodingAgentSubAgent(
        library=library_view,
        executor=executor_with_boundary,
        policy=select_model(role_spec.model_policy),
        system_prompt=system_prompt,
    )
```

MotionExecutor 可以暂时继续由现有 Act Agent 承担；read-only roles 走 SubAgent runtime。不要强行让所有 node 都是同一种 SubAgent。

### 3. Society Router

新增 router，根据 topology 和当前 evidence state 选择下一个 role：

```python
next_role = society_router.route(
    current_role,
    evidence_state,
    last_message,
    topology,
)
```

第一版可以先只做 recommendation，不强制接管 Act：

```text
DySC recommendation:
  current phase: post-grasp
  next role: predicate_verifier
  recommended skills: held_state_check
  finish gate: do not finish until held_state predicate is supported
```

这能减少对现有 loop 的侵入。

### 4. Composition Trace

新增 composition trace，不要把所有字段塞进旧 `AgentTrace`。

```text
CompositionTurn:
  role
  selected_skill
  consumed_evidence
  emitted_evidence
  action_type
  world_changed
  predicate_before
  predicate_after
  relation_error_before
  relation_error_after
  token_cost
  failure_type
  recommended_next
```

旧 trace 继续用于兼容；DySC trace 用于学习和进化。

## 进化接口

第一版进化可以离线跑，不必在线改变执行中的 society。

输入：

```text
SocietySpec + CompositionTraces + Episode Outcomes
```

输出：

```text
New SocietySpec + motif updates + skill binding updates
```

最小可实现的 operators：

1. `insert_verifier_edge`
2. `add_skill_to_role_policy`
3. `remove_low_value_skill_from_role_policy`
4. `split_role_by_skill_tags`
5. `prune_unused_edge`
6. `promote_successful_trace_to_motif`

后续再加 role birth、crossover、budget pruning。

## 与 Coding Agent 灵活性的关系

DySC 不应把控制策略全都模板化。coding agent 的优势是能临场写逻辑，例如：

```python
while relation_error > threshold and steps < max_steps:
    observe()
    compute_delta()
    goto_pose(current + clipped_delta)
```

因此 skill 和 society 的作用不是替代 coding，而是约束与引导 coding：

- 给出可复用 evidence keys；
- 限定 role 的关注范围；
- 推荐 next skill / verifier；
- 设置 finish gate；
- 记录 relation error；
- 在失败后修改 society graph。

也就是说，DySC 让 coding agent 在更好的认知结构里写代码，而不是把它降级成固定 planner。

## 工程落地顺序

1. 新增 `contract.yaml` loader 和 `SkillContract`。
2. 新增 `SocietySpec` / `RoleSpec` / `SkillAccessPolicy` schema。
3. 新增 `SkillLibraryView`，让 role 根据 policy 动态看到 skill subset。
4. 新增 `SocietyRouter`，先只产出 recommendation。
5. 在 Act prompt 中注入 DySC recommendation 和 finish gate。
6. 记录 `CompositionTrace`。
7. 从现有 outputs 离线生成第一批 motif / binding statistics。
8. 实现最小 offline evolution operators。

这个顺序能保证第一版就不写死 role-skill 绑定，同时又不需要立刻重写 RoboMEx 主循环。
