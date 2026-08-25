# Verified Co-Evolution 总体实施方案（Master Plan）

> 状态：实施蓝图 **v2（2026-08-24）**
> 决议：先 A 后 B、日夜联进化、讲 B 的故事（`proposals/PROPOSAL_B_VERIFIED_COEVOLUTION.md`）
> 论文故事：`proposals/PROPOSAL_B_VERIFIED_COEVOLUTION.md`
> 工程本文：里程碑、模块、信号纪律、立即行动
> 已删除：`RSI_METHOD_DESIGN.md`（串行「先文本后权重」外壳）
>
> **v2 相对 v1 的方法修订（2026-08-24）**
> 1. 内核与领域插件分离；LIBERO-PRO 是第一个实例，不是方法本体。
> 2. 案卷 = Trace + Video + Cost + 进展叙述。**诊断综合，选择不综合。**
> 3. **分析 / 归因 / 改稿由 VLM 自主完成**；人只冻契约、案卷格式与采纳公式。
> 4. 选择只认同 `(task, seed)` 的 `env_success` 配对净增。Cost 入案卷，不作否决闸。
> 5. planner / TCP、手写相位位、F1–F5 **不进选择、不进 proposer**。
>    planner 只留给夜班语料过滤（训练岗）。

---

## 0. 信心评估与成立前提

**结论：有信心，但按阶梯陈述。**

| 层级 | 内容 | 判断 | 依据 |
| --- | --- | --- | --- |
| T1 | 文本旋钮 held-out 赢过预算对等 TTS | 中高 | 环本身是已验证机制；收益取决于诊断质量与任务划分，不再押「路由是低垂果实」 |
| T3 | 蒸馏 7B 恢复教师大部分 Imagination 表现、成本降一个量级 | 高 | 窄接口 + 有界上下文；语料用 planner 逐步过滤（训练岗，不是选择岗） |
| T2 | W+H 端到端优于 H-only | 不确定 | SIA 在其它域成立，具身无先例；失败则优雅降级为纯文本旋钮 |

**成立前提：**

1. **GPU（夜班）**：可训 Qwen2.5-VL-7B LoRA 的卡。白班不依赖此项。
2. **gen-0 噪声底已入库**：`vaw/out/sweeps/gen0/` 跑完即可转白班。
   不再设 50% 硬门槛；成功率低说明第一刀该打恢复/抓取，不说明不能进化。
3. **sweep 能按题并行 seed**：现默认单题 3 seed。吞吐不够时先缩小 train 集，不阻塞建环。

---

## 1. 论文 claim ↔ 里程碑

```text
第一幕 诊断（probe 集）          ← ICRA 提案，独立，本文不展开
第二幕 干预（harness zero-shot） ← M1.6 已基本落地（v46 + gen-0）
第三幕 联合进化（本文 + Proposal B）
    T1 文本旋钮 vs TTS           ← M1.7 剩余 + M1.8
    T3 蒸馏可行性 + 成本          ← M1.9
    T2 W+H vs H-only             ← M2.0
```

---

## 2. 模块化架构（内核 vs 插件）

知识内容可以域特化（那正是被进化的东西）。**机制不能焊死在 LIBERO 上。**
审计标准：换成任何「有终局判定 + 可录像/可审计轨迹」的 agent 域，内核五层一行不改。

```text
领域插件（可替换）
  Environment          任务源 + 观测          ← 现：LIBERO-PRO
  Observation Encoder  观测 → Agent 工作区    ← 现：Context OS / Canvas v46
  Terminal Check       终局成功与否           ← 现：env_success；真机=脚本/抽检
  （可选）Geometry      逐步运动学合规         ← 现：planner / TCP；只服务训练岗

Agent 系统（被进化对象所在）
  Executor             Main，权重冻结
  Specialist           Imagination，窄接口，可替换/可训
  Knowledge Store      playbooks/*.md，唯一文本进化面

评测基座（内核）
  Episode Runner       sweep.py：隔离进程 + 落盘
  Trace Store          steps.jsonl / video / meta
  Paired Comparator    compare.py：同 (task, seed)
  Ledger               成本记账（turns / tokens / 物理动作）；不作选择闸

进化引擎（内核 · 白班）
  Case Builder         失败 episode → 匿名案卷（抹任务 ID）
  Diagnose VLM         读案卷 → 进展 + 失败叙事（不打采纳分、不给 diff）
  Propose VLM          叙事 + 当前 K + 内容契约 → 单文件单处 diff
  Selector             配对 env_success 净增 → git commit / 归档
  Archive              代日志、被拒 diff、生命周期

训练引擎（内核 · 夜班）
  Corpus Builder       verified Imagination 序列 → SFT
  Trainer              7B LoRA
  Deployment Gate      重放不劣于现役才上岗

治理（横切）
  Task Split           train / val / held-out 冻结
  Content Contract     知识可写什么
  Budget + TTS         预算对等对照（报告用，不进选择公式）
```

### 2.1 代码对照（2026-08-24）

```text
已有
  run_context_agent.py / context_runtime/* / playbooks/*
  evolution/fitness.py     事后标签表（F1–F5、相位位）；非正式选择依据
  evolution/sweep.py       批量评测
  evolution/compare.py     配对比较（采纳应走这里的 env_success 位）
  --imagination-model      Main / Imagination 已解耦

未写
  evolution/task_split.json
  evolution/diagnose.py    Case Builder + 诊断 VLM
  evolution/propose.py
  evolution/loop.py
  evolution/baselines.py / budget.py
  train/ 整层
  --force-imagination
```

`attribution.py` **v1 不建**。知识 vs 能力调度是 v2 以后的事；v1 固定日夜交替。

### 2.2 验证信号（瘦身，方法核心）

执行 Agent 靠 harness 释放能力；进化 Agent 靠读 episode 案卷。两边同一哲学。

```text
案卷 = Trace + Video + Cost + 诊断 VLM 写的进展/失败叙事

诊断（综合，允许软）
  冻结指令的 VLM 读案卷
  → 走到哪、哪一步开始错、为什么
  → 只喂 proposer；不打分、不建议 diff

选择（不综合，防 Goodhart）
  同 (task, seed)：候选 env_success 净增 > 0 则采纳
  否则拒绝
  Cost / 叙事 / 相位标签 / planner 均不投票

训练过滤（夜班另岗）
  planner 逐步通过 且 session ready → 可入语料
```

诊断与提案必须拆成两次调用（可同一模型）：先叙事、再改稿，禁止边看边改。

### 2.3 日夜数据流

```text
白天：Knowledge ──注入──► Runner ──案卷──► diagnose.py
        ▲                                      │
        │ 配对 env_success 净增                  ▼
      loop.py ◄──候选 sweep──◄ propose.py ◄── 失败叙事
        │
        └──（旁路）Imagination session ──corpus──► out/corpus/

夜里：out/corpus/ ──distill──► candidate θ
清晨：replay_gate（胜）──► Specialist 上岗
                   （负）──► 白天继续用现役
```

两条线只通过语料单向耦合。夜班失败不影响白班。

---

## 3. 里程碑

### M1.6 契约收尾 + Canvas 冻结 + gen-0（大部分已完成）

已落地：阶段 A 修复、plumb line、schema **v46**、playbook 拆分（gen-0 为 `injection=all`）、
fitness / sweep / compare、gen-0 10×3  sweep。

不再要求 B2–B4 全绿；冻结点就是 v46。相位机注入仍是可选项，不是开环前置。

### M1.7 剩余基建

- **7.1 已完成**：playbook 文件 + `CONTENT_CONTRACT.md` + contract hash。
- **7.2 降级**：`fitness.py` 保留作出事后对照表，**从选择公式移除**。
- **7.3 未完成**：冻结 `evolution/task_split.json`；sweep 已可用。
- **7.4 新增**：`diagnose.py`（匿名案卷 + 诊断 VLM）。

### M1.8 文本旋钮循环

**8.1 诊断（`diagnose.py`）**

- 输入：失败 episode 的 `steps.jsonl`、`video_*.mp4`（或抽帧）、成本字段。
- 抹掉任务 ID / held-out 信息，编号为 `case_1..N`。
- 输出：进展一句话 + 失败叙事。禁止输出分数、禁止输出 playbook diff。

**8.2 提案（`propose.py`）**

- 输入：本代诊断叙事 + 当前全部 playbook + `CONTENT_CONTRACT.md`。
- 输出：恰好一个文件、一处 diff + 一句理由。
- 硬校验：只落在 `vaw:slot` 内；无绝对量值 / 物体名 / 任务 ID；contract hash 不变。
- 不可见：held-out 列表、任务 ID、任何选择分数。

**8.3 代循环（`loop.py`）**

```text
while 代数 < N:
  叙事 = diagnose(失败案卷)
  candidate = propose(K_i, 叙事)
  sweep(candidate, train)
  if 配对 env_success 净增 > 0:     # compare.py，只看终局位
      val 上 env_success 无净退步 → git commit，K_{i+1} = candidate
  else:
      归档拒绝（diff + 叙事 + 配对表）
```

预期采纳率低。McNemar 只记录，不作门槛。

**8.4 TTS（`baselines.py`）**

预算对等的平行采样 / 顺序重试，只上报告，不进采纳。

出口：≥3 代有完整代日志；held-out 只在代末记录、不参与选择。

### M1.9 权重旋钮（与白班并行，可晚一周）

- `--force-imagination`：只进语料，不进对比表。
- `corpus.py`：planner 全过且 `ready` 的 edit 序列。
- `distill_lora.py` / `replay_gate.py` / 本地服务 7B。
- `--imagination-model` 已存在。

### M2.0 联合运行 + 消融

| 行 | Main | Imagination | Playbook |
| --- | --- | --- | --- |
| gen-0 | 大模型 | 大模型 | v0 |
| H-only | 大模型 | 大模型 | 进化后 |
| W-only | 大模型 | 蒸馏 7B | v0 |
| W+H | 大模型 | 蒸馏 7B | 进化后 |
| TTS ×2 | 大模型 | 大模型 | v0 + 预算对等 |

v1 固定日夜交替。归因调度（知识失败 vs 能力失败决定次日侧重）标为 v2，不做本窗口交付。

---

## 4. 实验矩阵

```text
KE-A（T1）：代数 × {train 终局成功率, held-out 终局成功率, TTS 线}
            通过：held-out 进化线高于 TTS
KE-B（T3）：replay（7B zero-shot / 蒸馏 / 教师）+ 部署成本
KE-C（T2）：W+H vs H-only vs W-only vs gen-0
KE-D：被采纳 diff 全文 + 诊断叙事 + 前后行为（定性）
```

---

## 5. 时间表

v1 的 8/20–9/15 四周日历**作废**（M1.6 已滑期）。里程碑名保留，按「白班一周建环、夜班可并行」重排，不以截稿倒推。

诚实预期：先交付可转的文本旋钮环 + 案卷诊断；7B 上岗视 GPU 与语料量，允许只落到 T1。

---

## 6. 风险

| # | 风险 | 对策 |
| --- | --- | --- |
| R1 | 无训练 GPU | 夜班降级；白班故事不受影响 |
| R2 | gen-0 终局偏低 | 照开白班；诊断会自然偏向恢复/抓取 |
| R3 | sweep 慢 | 缩小 train；保持单题多 seed |
| R4 | 诊断编造原因 | 选择仍只看终局；浪费的是一代提案 |
| R5 | 提案违反契约 | 硬校验拒绝，本代跳过或重试 ≤2 |
| R6 | train 涨 held-out 不涨 | cut line：降级为 harness analysis |
| R7 | 进程崩溃 | sweep 已按 episode 隔离 |
| R8 | 「这就是 LLM-judge 进化」 | 选择公式不含 VLM 分；附录公开代账单 |

---

## 7. 非目标

- 进化 Canvas、终局判定器、Function schema、Main 权重。
- 种群搜索。
- **VLM 参与采纳投票**（诊断与提案是 VLM 的；过不过不是）。
- 手写相位 / F 标签作为选择或 proposer 输入。
- 用 planner / TCP 当选择 fitness。
- Proposal C（在线技能自著述）。
- 本窗口真实机器人实验（案卷四件在真机上仍成立，作为 limitation 说明）。

---

## 8. 立即行动

1. 冻结 `evolution/task_split.json`（10 题切 train / val / held-out）。
2. `diagnose.py`：案卷打包 + 诊断 VLM（只描述）。
3. `propose.py` + `loop.py`：接 `compare.py` 的配对 `env_success`。
4. 夜班：`--force-imagination` + `corpus.py`（有 GPU 再蒸）。
