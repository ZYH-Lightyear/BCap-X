# Verified Co-Evolution 总体实施方案（Master Plan）

> 状态：实施蓝图 v1（2026-08-20 定案）
> 决议：先 A 后 B、日夜联进化、讲 B 的故事（`proposals/PROPOSAL_B_VERIFIED_COEVOLUTION.md` §4.5）
> 上游：`M1_6_RSI_READINESS_PLAN.md`（bug 修复 + Canvas 冻结，本方案的 M1.6）、
> `RSI_METHOD_DESIGN.md`（方法论）、`ICRA_PAPER_PROPOSAL.md`（第一、二幕不变）
> 本文档覆盖：里程碑 M1.6 → M2.0、系统架构、工程实现、实验矩阵、时间表、风险登记

---

## 0. 信心评估与成立前提

**结论：有信心，但按阶梯陈述。**

| 层级 | 内容 | 判断 | 依据 |
| --- | --- | --- | --- |
| T1 | 文本旋钮 held-out 赢过预算对等 TTS | 高置信 | 每个环节都是已验证机制的延伸；路由缺陷是已知低垂果实 |
| T3 | 蒸馏 7B 恢复教师大部分 Imagination 表现、成本降一个量级 | 高置信 | 窄接口+密集验证反馈+有界上下文，是小模型最易学的题型；语料由 planner 天然过滤 |
| T2 | W+H 端到端优于 H-only | 真研究不确定性 | SIA 在三个领域成立，但具身域无先例；失败不毁论文（优雅降级结构自带） |

**成立前提（必须先满足，按序检查）：**

1. **GPU**：一台可训 Qwen2.5-VL-7B LoRA 的机器（A100/H100 单卡；QLoRA 则 48G 可行）。
   唯一外部硬依赖。
2. **gen-0 成功率下限**：M1.6 修复后 object suite 扰动设置 zero-shot ≥ 50%。
   低于此值说明 harness 本身仍有结构缺陷，进化没有健康的起点，先回去修。
3. **sweep 吞吐下限**：单 episode ≤ 15 分钟、可 4 路以上并行。
   否则每代评估周期过长，四周窗口跑不满 3 代。

---

## 1. 论文 claim ↔ 里程碑映射

```text
第一幕 诊断（probe 集）          ← ICRA 提案 B 线，独立推进，不在本文档展开
第二幕 干预（harness zero-shot） ← M1.6（bug + canvas 冻结 + gen-0 sweep）
第三幕 联合进化（本文核心）
    T1 文本旋钮 vs TTS           ← M1.7（基建）+ M1.8（循环）
    T3 蒸馏可行性 + 成本          ← M1.9（语料 + 训练 + 门控）
    T2 W+H vs H-only             ← M2.0（联合运行 + 消融矩阵）
```

---

## 2. 系统总体架构

### 2.1 分层视图

```text
┌─────────────────────────────────────────────────────────────────┐
│ 契约层（冻结）——现有代码                                          │
│   vaw/scripts/run_context_agent.py      单 episode 入口           │
│   vaw/context_runtime/*.py              Runtime/Canvas/验证器      │
│   vaw/agents/providers/*                模型后端                   │
│   tests/test_vaw_*.py                   144 项回归 + contract hash │
├─────────────────────────────────────────────────────────────────┤
│ 知识层（唯一进化面）——新增                                        │
│   vaw/playbooks/*.md                    相位索引 playbook（git 管理）│
│   vaw/context_runtime/playbook.py       加载 + 相位确定性注入        │
├─────────────────────────────────────────────────────────────────┤
│ 进化层（白班）——新增 vaw/evolution/                               │
│   fitness.py    trace → 相位 fitness（确定性提取）                 │
│   sweep.py      批量评测编排（并行、崩溃隔离、预算记账）             │
│   propose.py    mutation proposer（读失败三元组 → 单点变异）        │
│   loop.py       代管理：对比、采纳、git commit、代日志              │
│   attribution.py 知识失败/能力失败归因统计                          │
│   baselines.py  预算对等 TTS（parallel sampling / seq refinement） │
│   budget.py     episodes / tokens / GPU-hours 记账                │
├─────────────────────────────────────────────────────────────────┤
│ 训练层（夜班）——新增 vaw/train/                                   │
│   corpus.py         imagination session → SFT 语料（JSONL）        │
│   distill_lora.py   Qwen2.5-VL-7B LoRA 蒸馏                       │
│   replay_gate.py    清晨门控：离线重放快测                          │
│   serve_imagination.py  本地 vLLM 服务 7B checkpoint               │
├─────────────────────────────────────────────────────────────────┤
│ 产物存储                                                          │
│   vaw/playbooks/          当前代 playbook（git 历史 = 进化史）      │
│   vaw/out/sweeps/<gen>/   每代 sweep 产物 + sweep_summary.json     │
│   vaw/out/corpus/         verified session 语料（累积）            │
│   checkpoints/imagination_7b/<date>/    每晚 checkpoint + 门控记录  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 日夜数据流

```text
白天：playbooks/ ──注入──► run_context_agent ──trace──► fitness.py
        ▲                                                │
        │ 采纳(严格提升)                                   ▼
      loop.py ◄──候选 sweep──◄ propose.py ◄──失败三元组──sweep_summary
                                                          │
      （旁路，无条件）imagination sessions ──corpus.py──► out/corpus/
夜里：out/corpus/ ──distill_lora.py──► candidate checkpoint
清晨：replay_gate.py(candidate vs 现役) ──胜──► serve_imagination 切换
                                        ──负──► 弃，白天继续用现役
```

关键性质：两条线通过"语料落盘"单向耦合，训练侧任何失败都不影响进化主线。

---

## 3. 里程碑

### M1.6 契约收尾 + Canvas 冻结 + gen-0（≈1.5 周，细节见 M1_6_RSI_READINESS_PLAN.md）

内容不重复，此处只列出口判据：

- 阶段 A 四项修复落地（释放判据 prompt、盲降 advisory、routing 正向触发、locate 逃生口文档）；
- Canvas B1（plumb line + XY 偏差标注）、B2–B4 视图、B5 图例/schema v42，审计矩阵全绿后冻结；
- **gen-0 baseline sweep**：object suite 全任务 × 3 seeds，产出成功率与相位 fitness 基线。
- 出口判据：全量测试通过；gen-0 object suite ≥ 50%（§0 前提 2）。

### M1.7 进化基建（≈4 天，与 M1.6 后半并行）

**7.1 Prompt 因子化**

- `protocol.py` 的 SYSTEM_PROMPT 拆为：contract 段（工具/事件/图例语义）留在
  `protocol.py`；策略性内容迁出为 `vaw/playbooks/{grasp,transport,align,place,recover,routing}.md`。
- 新增 `context_runtime/playbook.py`：加载 playbook 目录、按相位拼装注入。
  相位由 ContextState 确定性推断（无载荷+无接触=grasp；attached+远离目标=transport；
  attached+目标邻域=align/place；连续失败恢复=recover）。
- 内容契约（禁 offset，三类知识）写为 `playbooks/CONTENT_CONTRACT.md`，
  同时作为 propose.py 指令的一部分。
- 测试：contract 段 hash 守护（改动即测试失败）；playbook 注入的快照测试；
  gen-0 playbook（现 prompt 策略内容的忠实迁移）跑单任务与迁移前行为一致。

**7.2 相位 fitness（`evolution/fitness.py`）**

- 输入：episode trace 目录（events + meta.json）。输出：
  `{pick: 0/1, transport: 0/1, place: 0/1, turns, physical_ops, main_tokens, imag_tokens}`。
- 判定全确定性：pick = attached 事件且载荷离开支撑面；transport = 载荷 OBB 进入
  目标 region 邻域（bbox 外扩）；place = env_success。
- 测试：对 m16l/m16q/m16r 三条已有 trace 断言已知标签。

**7.3 Sweep 编排（`evolution/sweep.py`）**

- 子进程调用 `run_context_agent.py`（每 episode 独立进程，MuJoCo 崩溃隔离），
  参数矩阵 = 任务 × seeds × playbook 版本 × imagination 模型；
- 并行度可配（默认 4）；输出 `out/sweeps/<tag>/sweep_summary.json`
  （成功率、相位 fitness 汇总、budget.py 记账、失败 episode 索引）；
- 任务划分在此冻结：`evolution/task_split.json`（train/val/held-out，第 0 天写死，git 锁定）。
- 出口判据：同一 playbook 两次 sweep（同 seeds）结果一致；吞吐满足 §0 前提 3。

### M1.8 文本旋钮循环（=Proposal A 主体，≈1 周运行期）

**8.1 Mutation proposer（`evolution/propose.py`）**

- 输入：上一代 sweep 的失败 episode 三元组（相位 fitness、事件流摘要、关键 canvas 截图）
  + 当前 playbook + 内容契约；
- 输出：对单个 playbook 文件的一处 diff + 一句改动理由（进代日志，可读的进化史）；
- proposer 不可见：held-out 任务列表、任务 ID、val/held-out 分数。

**8.2 代循环（`evolution/loop.py`）**

```text
while 代数 < N:
  candidate = propose(P_i, 失败三元组)
  F_cand = sweep(candidate, train)
  if F_cand 严格优于 F_i（先 place 相位，再总成功率，再成本）:
      val 复核通过 → git commit 采纳，P_{i+1} = candidate
  else: 归档拒绝记录
  attribution.py 输出本代知识/能力失败占比
```

**8.3 TTS 对照（`evolution/baselines.py`）**

- parallel sampling：gen-0 playbook，同任务多 seed 取最好；
- sequential refinement：gen-0 playbook，失败后同任务重试至预算耗尽；
- 预算对等以 episodes 与 total tokens 双轴对齐（budget.py 出表）。

出口判据：≥3 代完成；每代采纳/拒绝记录与理由完整；
**showcase 检查**：routing playbook 是否进化出委派规则（定性结果，论文用）。

### M1.9 权重旋钮基建（≈1 周，与 M1.8 白班并行推进）

**9.1 语料（`train/corpus.py` + 采数据模式）**

- `run_context_agent.py` 加 `--force-imagination` 旗标：commit 前强制委派
  （仅用于采数据 episode，产物只进语料库、不进任何对比表）；
- session → SFT 样本：输入 =（聚焦 canvas 图像、instruction、当前状态文本），
  目标 = 下一步 edit 函数调用 JSON；只保留 planner 全通过且 session ready 的序列；
- 冷启动目标量：≥ 200 条 verified session（约数百个采数据 episode）。

**9.2 蒸馏（`train/distill_lora.py`）**

- 基模 Qwen2.5-VL-7B，LoRA（r=16 起步），多模态 SFT 标准配方；
- 每晚在**累积**语料上重训/续训，产出 `checkpoints/imagination_7b/<date>/`。

**9.3 清晨门控（`train/replay_gate.py`）**

- 从语料库留出的重放集（不参与训练）上离线快测：
  edit 合规率（planner 通过率）、session ready 率、与教师选择一致率；
- candidate 全指标不劣于现役才切换 `serve_imagination.py` 指向；
- **第一次运行即 go/no-go 探针**：7B zero-shot 分数落盘——
  显著非零 → 蒸馏起点好；接近零 → 必须蒸馏（也是论文数据点）。

**9.4 Runtime 接入**

- `run_context_agent.py` 加 `--imagination-model`（默认与 --model 相同），
  `agents/providers` 增加本地 vLLM 后端；Main 与 Imagination 模型解耦。

出口判据：蒸馏 v0 checkpoint 走通"训练→门控→上岗/拒绝"全流程；
replay 曲线（zero-shot vs 蒸馏 v0 vs 教师）成图。

### M2.0 联合运行 + 论文实验（≈1.5 周）

- 日夜排班连续运行：白班 M1.8 循环继续，夜班蒸馏，清晨门控；
  归因统计每日出表（v1 固定日夜交替，归因作分析）；
- **主消融矩阵**（train 上进化、held-out 上报告，每格 ≥10 seeds）：

| 行 | Main | Imagination | Playbook |
| --- | --- | --- | --- |
| gen-0 | 大模型 | 大模型 | v0 |
| H-only | 大模型 | 大模型 | 进化后 |
| W-only | 大模型 | 蒸馏 7B | v0 |
| W+H | 大模型 | 蒸馏 7B | 进化后 |
| TTS ×2 | 大模型 | 大模型 | v0 + 预算对等采样/重试 |

- 成本表：每行 token/episode、GPU-hours、wall-clock；
- 冻结 probe（若 ICRA B 线产出）同步测进化前后通用空间判断；
- 论文写作与实验并行（W3 起 draft，写作规范走 `docs/paper/SKILL.md`）。

出口判据：claim 阶梯着地层级判定（T1/T3 必须，T2 如实报告）；
所有表格可由 out/sweeps/ 产物脚本再生。

---

## 4. 实验矩阵与 claim 对应

```text
KE-A（T1）：进化代数 × {train fitness, held-out 成功率, TTS 水平线}
            通过标准：held-out 上进化线显著高于 TTS 线
KE-B（T3）：replay 三点曲线（7B zero-shot / 蒸馏 v0..vk / 教师）
            + 部署成本对比（大模型 Imagination vs 7B Imagination）
            通过标准：蒸馏恢复教师 ≥70% ready 率，成本 ≤1/5
KE-C（T2）：主消融矩阵 W+H vs H-only vs W-only vs gen-0
            通过标准：W+H ≥ H-only（显著则头条，持平则如实报告并分析归因）
KE-D（定性）：进化史展示——被采纳 mutation 的全文 + 理由 + 前后行为对比
            （routing 从不委派 → 学会委派是首选素材）
```

---

## 5. 时间表（2026-08-20 → 09-15，假设 ICRA 截稿）

```text
W1  8/20–8/26   M1.6 阶段 A 修复 + Canvas B0/B1；M1.7.1 因子化并行起步
W2  8/27–9/02   Canvas B2–B5 + 冻结；gen-0 sweep；M1.7.2/7.3 完成；
                采数据模式 + 7B zero-shot replay 探针（go/no-go 落盘）
W3  9/03–9/09   白班：M1.8 循环跑 3–5 代 + TTS 对照
                夜班：蒸馏 v0/v1 + 清晨门控
                周中 checkpoint（9/6）：T2 有无初步信号 → 定标题与 claim 措辞
W4  9/10–9/15   M2.0 主消融矩阵冻结实验；写作冲刺；adversarial self-review
```

诚实预期：四周内 T1+T3 有完整证据，T2 大概率只有初步信号
（联合运行代数有限）。若 T2 未着地：论文仍讲 co-evolution 框架，
T2 作为 preliminary + 完整版投下一档期（CoRL/RSS 2027）——
决定点在 9/6，不拖到最后一周。

---

## 6. 风险登记表

| # | 风险 | 触发信号 | 对策 |
| --- | --- | --- | --- |
| R1 | GPU 训练环境不可用 | W2 前未确认 | T3 降级为 replay 探针 + 语料统计；故事回退 A+teaser |
| R2 | gen-0 成功率 < 50% | M1.6 出口 | 停止进化排期，回修 harness（进化救不了结构缺陷） |
| R3 | sweep 吞吐不足 | W2 实测 | 缩小训练任务集；提高并行；episode 提前终止规则 |
| R4 | 路由进化不出委派 | M1.8 前 2 代 | 检查 advisory 是否进入失败三元组；proposer 提示加强失败归因可见性 |
| R5 | 7B zero-shot ≈ 0 且蒸馏 v0 也差 | W3 门控 | 检查 SFT 样本格式/图像分辨率；换 2B→7B 对照排查；最坏 T3 改讲"语料贡献" |
| R6 | 进化在 train 涨 held-out 不涨 | KE-A | 按 cut line 降级；本身也是可发表的负结果分析 |
| R7 | MuJoCo/环境崩溃传染 | sweep 期 | 每 episode 独立进程 + 超时杀进程（sweep.py 内建） |
| R8 | 预算对等被质疑 | 写作期 | budget.py 全程记账，附录公开逐代账单 |

---

## 7. 非目标

- 进化 Canvas、验证器、Function schema、Main 权重（契约/验证层冻结）；
- 种群搜索（只做轨迹局部自比较）；
- LLM-judge 参与任何采纳决策；
- Proposal C（在线技能自著述）——本轮不做，档期错开独立推进；
- 真实机器人实验（limitation 说明输入可得性）。

---

## 8. 立即行动清单（本周）

1. 确认 GPU 训练环境（§0 前提 1）——唯一需要用户确认的外部依赖；
2. M1.6 阶段 A 四项修复动工（M1_6 文档 §2）；
3. `evolution/task_split.json` 任务划分第 0 天冻结（写死并 commit）；
4. Canvas B1 plumb line 实现（M1_6 文档 §3 B1）。
