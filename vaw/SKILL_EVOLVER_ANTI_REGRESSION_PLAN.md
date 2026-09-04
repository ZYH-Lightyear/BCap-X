# VAW Skill Evolver Agent：防退化技能进化实施计划

## 1. 目标

本计划把现有 DAY rollout、M3 Trace Investigator、MMSkill Library 和不可变
Generation Store 连接为可审计的技能进化闭环：

```text
LIBERO-90 DAY
→ M3 调查与证据复核
→ Skill Evolver
→ Candidate Reviewer
→ inactive generation
→ paired development gate
→ human approval
→ promotion
→ post-promotion audit / rollback
```

每代最多提出一个 `ADD`、`REVISE` 或 `RETIRE`，也允许 `NO_CHANGE`。
技能正文只存在于 `SKILL.md`；Visual Reference 属于技能目录，不复制进 mutation。
系统不修改已经验证的极简 VAW System Prompt，也不把完整 trace、planner telemetry、
private sensor data 或环境真值送给操作 Agent。

## 2. 最小领域模型

Evolution schema 从 v2 开始，不对 v1 做静默迁移。

```text
SkillMutation
├── mutation_id
├── operation          ADD | REVISE | RETIRE
├── skill_id
├── evidence_ids
└── rationale
```

正式 Gate 的 seed 与阈值冻结在 `EvolutionSpec.gate_policy`：

```text
primary_seeds             1, 2
confirmation_seeds        3
min_recoveries            2
min_attributed_recoveries 1
max_turn_ratio            1.5
max_token_ratio           1.5
auto_rollback             true
```

Gate 只保存逐 episode 配对事实。recovery、regression 和成功率在读取时派生，
不重复写入 JSON。turn、token 与 wall-clock 已存在于 DAY index；Gate 只对双方都成功的
episode 比较 turn/token 比例，wall-clock 作为负载敏感的报告项，不单独决定晋升。

## 3. Candidate Package 与 Generation

候选目录是 Skill Evolver、Reviewer 和 Generation Store 的交接边界：

```text
runs/evolver/e001/             # Evolver/Reviewer 工作记录，允许 reject/no-change
└── ...

candidates/m001/               # 只有 Reviewer accept 后才密封
├── mutation.json
├── evidence.json
├── manifest.json
├── skill/                    # ADD/REVISE only
│   ├── SKILL.md
│   └── references/           # optional, Stage C
│       ├── index.json
│       └── r*.png
├── evolver/
│   ├── request.json
│   └── response.json
└── review/
    ├── request.json
    ├── response.json
    └── report.json
```

`evidence.json` 使用候选包内的局部 `e1...eN` 标识，记录 M3 finding、policy-visible
raster 与 SHA-256。Skill 正文和图片不复制到 JSON。

Generation Store 只接受 Candidate Package，并根据 mutation 在父 snapshot 上执行一次
明确变更：ADD 新增、REVISE 整体替换、RETIRE 删除。新 generation 使用 staging 目录
构建、复核 digest 后原子发布；创建过程不改变 active generation。

Candidate 不是 Evolver 草稿。Evolver 输出和 Reviewer reject/no-change 保存在 run 目录；
只有 Reviewer `accept` 后才把 mutation、证据、最终 Skill、Visual Reference 和两次模型审计
记录一次性密封。候选内容摘要覆盖除 manifest 自身以外的全部文件，后续改写会被拒绝。

## 4. M3 与 M4 的职责边界

M3 只回答“轨迹中发生了什么、证据支持什么可迁移经验”，不决定修改哪个 Skill。每条通过
Evidence Reviewer 的 finding 提供短 observation、insight、动作 span 和 policy-visible raster。

M4 只回答“当前 Skill Library 是否已经覆盖、应该怎样改变”。它不重新承担整段轨迹发现，
也不预测 mutation 的真实收益；收益与成本只能由后续 paired Gate 判断。

```text
DAY trace
→ M3 Trace Investigator + Evidence Reviewer
→ accepted finding catalog
→ M4 Skill Evolver + Independent Mutation Reviewer
→ sealed Candidate
→ empirical Gate
```

## 5. M4 Skill Evolver Agent

Skill Evolver 是有界、只读的 Agent Loop。初始 Context 只有当前轻量 skill index 和
M3 已接受 finding 摘要；它自主判断哪些 Skill 相关、现有能力是否欠缺。它可调用：

```text
inspect_evidence(evidence_id)
inspect_skill(skill_id)
inspect_reference(skill_id, reference_id)
```

`inspect_evidence` 返回一张真实 policy-visible raster、M3 observation/insight、精简动作和
provenance；`inspect_skill` 返回完整 `SKILL.md` 与 reference catalog；历史图片只通过
`inspect_reference` 按需读取。上下文不累计原始对话，只重建 skill/finding index、已检查 ID、
少量已读取正文和最近视觉证据。

最终只允许一个 `ADD / REVISE / RETIRE` 或 `NO_CHANGE`。ADD/REVISE 输出完整 Skill 正文，
并同时给出最终 Visual Reference 选择；RETIRE 不输出正文。解析失败、引用不存在或服务失败
均显式结束，不换模型、不拼接兜底技能。

Independent Mutation Reviewer 使用全新 Context，可按需读取同一批证据和现有 Skill，只检查
operation、diff、证据、跨任务通用性、冲突与 Visual Reference。它只输出 `accept` 或
`reject`，不自动重写技能。Reviewer accept 后才密封 CandidatePackage。

当前 Skill 数量较小时不为每个 Skill 启动子 Agent。一个 library-aware Evolver 足以完成相关
Skill 检索、能力缺口判断和单项 mutation；独立 Reviewer 只提供防止自证循环的第二视角。

## 6. 多状态 Visual Reference

一个 Skill 可以有 `0..N` 张 Visual Reference，不固定为 positive/negative：

```json
{
  "schema": "vaw-mmskill-references-v1",
  "references": [
    {
      "reference_id": "r001",
      "state": "contact_ready",
      "view": "contact_front",
      "when_to_use": "判断物体是否进入两指闭合扫掠区域。",
      "visual_cue": "物体位于两指之间，掌部仍有净空。",
      "file": "r001.png",
      "sha256": "..."
    }
  ]
}
```

Visual Reference 不再由独立 Curator 生成。Skill Evolver 在同一次 ADD/REVISE 决策中输出
技能的最终 reference 集合，也可以输出空集合。REVISE 采用 copy-on-write，因此能保留、
增加、替换或删除旧图片；RETIRE 随技能一起退出新 generation。

图片只能复制 mutation 引用的真实 policy-visible raster，禁止生成示意图或使用 private
数据。Runtime 平时只展示轻量 skill index；`consult_mmskill` 后将该技能已经审核的最终
reference 集合确定性编译成一张紧凑参考板，不再启动额外的 Skill Consultant。所有历史图片
明确标注 `REFERENCE · NOT CURRENT`，参考板不能替代当前真实 Canvas。

## 7. 防退化与成本感知 Paired Gate

baseline 与 candidate 使用相同 task、seed、init state、模型、Runtime、Prompt、Canvas、
预算和 motion backend，唯一变量是 generation。

```text
recovery   = baseline failure → candidate success
regression = baseline success → candidate failure
```

所有 regression 都计入，因为常驻的轻量 skill index 本身也可能改变行为。ADD/REVISE
要求至少一个 recovery 中 candidate 实际 consult 目标技能；RETIRE 要求对应 recovery 中
baseline 曾 consult 被移除技能。

正式通过条件：至少两个 recovery、至少一个归因 recovery、recovery 严格多于 regression，
且 candidate 总成功数不低于 baseline。primary regression 需要使用冻结 confirmation seed
配对确认；confirmed regression 直接拒绝。基础设施失败返回 `INCONCLUSIVE`。

效果条件通过后，再检查双方都成功的 episode：candidate 的平均 turns 与 tokens 不得超过
冻结比例；wall-clock 同时进入报告，但不用于自动拒绝。recovery 可以付出合理额外成本，因为
baseline 没有完成任务；共同成功却显著变慢说明 Skill 可能引入过度 consult 或冗余 SOP。

完全相同 generation digest、task、seed、init state、模型、Prompt、Canvas、Runtime 和预算的
baseline 允许复用。Candidate Gate 按冻结顺序执行，可以在 confirmed regression 或已不可能
满足 recovery 时提前拒绝；正式通过不能提前，必须完成规定的 pairs。

Promotion 必须经过 Reviewer ACCEPT、Gate PASSED 和人工批准。后续 DAY 只产生 suspect；
只有父代/当前代 confirmation pair 再次确认退化时，才允许自动回滚活动指针。历史
generation 永不删除或改写。

## 8. DAY–NIGHT 节奏

g000 首次可完整运行 Evolve-30 建立证据池。之后不因每个 mutation 重跑完整 Evolve-30；
DAY runner 从冻结 split 中选择可配置的 5–10 task 批次轮转。未被本代采用的 accepted finding
可以继续留在证据池；只有 active generation 改变后才重新判断旧证据是否仍适用。

每代仍只包含一个 mutation，用于保持归因清晰。一个 DAY evidence pool 可以先后支持多个
候选尝试，Reviewer reject 或 Gate reject 不强制重新采集整批 DAY。

## 9. 实施状态（2026-09-03）

1. **Stage A — Domain / Store / Candidate Package：已实现。** schema v2、精简模型、
   operation-aware materialization 与密封候选包均已落地。
2. **Stage B — M4 Skill Evolver / Reviewer：已实现。** 三工具短上下文 Agent Loop、
   `NO_CHANGE / ADD / REVISE / RETIRE`、独立审核和 accept 后密封均已落地。
3. **Stage C — Runtime Visual Reference：已实现。** 支持 `0..N` 多状态 reference；只有
   `consult_mmskill` 后才在当前 Canvas 旁显示，并标注 `REFERENCE · NOT CURRENT`。
4. **Stage D — Paired Gate：已实现。** 读取两份冻结 DAY index，检查 generation 身份、
   recovery/regression、consult 归因、confirmation 与共同成功 episode 的成本比例；wall-clock
   已纳入报告但不自动否决。
5. **Stage E — Approval / Promotion / Audit：已实现。** CLI、人工 approval、原子 active
   pointer、suspect/confirmed degradation 与按策略回滚均已落地。
6. **Stage F — Smoke：已实现工程全链路，并完成一条真实证据 smoke。** 工程 smoke 覆盖
   candidate、Gate、promotion、audit；真实 smoke 覆盖 LIBERO-90 trace、TOPReward、M3 和
   M4 的无证据安全终止；另用已审核的真实 spatial evidence 跑通 Evolver、独立 Reviewer 与
   带 Visual Reference 的密封候选。M3/M4 CLI 同时保存无密钥 provider run metadata。正式
   paired task 收益评测仍需更多真实 episode。

历史开发按阶段审查；本轮经用户授权连续完成余下模块。所有真实 promotion 仍必须显式人工批准。

## 10. 真实案例 Smoke

Smoke 使用 `libero_90:0`、seed 1、`vapi/gemini-3.7-flash`、temperature 0 和 CuRobo，
输出到：

```text
vaw/out/evolution/skill_evolver_real_smoke_libero90_t0_s1_20260903/
```

真实证据 smoke 复用了已完成的 LIBERO-90 task 0 / seed 1 DAY trace，重新生成官方
TOPReward 曲线，并运行 Gemini 3.7 Flash M3。M3 主动检查了四段证据但没有提出 finding，因此
M4 明确输出 `NO_CHANGE`，没有伪造 candidate。另有独立的 deterministic engineering smoke
跑通 Candidate → inactive generation → Gate → Approval → Promotion → Audit；脚本化 episode
结果只证明工程连通性，不代表技能真实收益。补充的真实 spatial evidence M4 run 产出一个
Reviewer 接受但尚未 materialize 的候选，用于验证多模态 Agent Loop；它同样不构成收益证明。
三类结果必须分开解释。
