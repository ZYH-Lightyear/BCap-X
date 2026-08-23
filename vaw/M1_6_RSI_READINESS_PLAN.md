# M1.6 计划：修 bug → Canvas 完备化 → RSI-ready 版本

> 状态：**实施中（2026-08-20 起）**
> 已落地（schema `vaw-context-v42-routing-advisories`，renderer `context-web-v42-plumb-line`，
> 154 项 VAW 测试通过）：
> - 阶段 A 全部四项：A1 释放判据改写（矛盾时禁降/禁释放而非禁修正，配 force_refresh）、
>   A2 盲降 advisory（`descend_streak` ledger，连续 3 次主导下降且无新鲜 grounding → 提示，
>   新证据/夹爪命令/commit 重置）、A3 正向路由触发清单（prompt + call_imagination 工具描述）、
>   A4 `locate_point(force_refresh=true)` 逃生口（跳过 verified 复用，多视角矛盾时用）
> - B1 plumb line：`plumb_line.py`（底面中心垂线 + 落点足迹 + H/dXY cm 标注，
>   current 青 / preview 紫 / 偏差红），画进 main 投影三个场景 raster 与直连 Contact 面板；
>   图例已写入两个系统提示；`tests/test_vaw_plumb_line.py` 7 项
> - B0 审计矩阵骨架：`vaw/diagnostics/CANVAS_OBSERVABILITY_AUDIT.md`（已知红格 + 填格流程）
> - v43（2026-08-20，m16s seed1–3 复跑后的两项修正，schema
>   `vaw-context-v43-partial-imagination` / renderer `context-web-v43-anchor-dxy`，
>   157 项 VAW 测试通过）：
>   ① Imagination turn_limit 不再回滚——最后一次已通过规划校验的编辑作为 `status=partial`
>   交回 Main 审查（可 commit / 继续委派 / reject）；回滚只保留给语义失败与内部错误
>   （详见 `M1_5_2_IMAGINATION_AGENT_CALL.md` 头部语义修订）；
>   ② dXY 目标中心从 region 点云中位数改为 Action 的语义 anchor point（locate_point 测得点），
>   无 point 来源则不画箭头——seed2 中筐体中位数把 dXY 拉偏 ~13cm，子代理追偏目标耗尽预算
> - v44（2026-08-21，m16t task4 复盘驱动，schema `vaw-context-v44-landing-topdown` /
>   renderer `context-web-v44-landing-topdown`，164 项 VAW 测试通过）：**B3 落点俯视面板**——
>   携带载荷时右上 OPPOSITE VIEW 自动切换为 LANDING TOP-DOWN：落点正上方的 live MuJoCo
>   俯视实拍（`LiberoTopdownSceneCameraProvider`，相机高度自适应载荷顶面），叠加足迹多边形、
>   dXY 箭头与 base ±X/±Y 投影罗盘（角落 glyph，方向即 delta_move 符号）；主提示明确
>   "沿口上 vs 开口内以俯视图为准，近水平 Contact View 不可判"。动机：task4 turn 25–30
>   两个 Contact 视角把压在篮沿上的足迹误读为"开口内"，agent 连续三次只调 z 不调 XY。
>   Imagination 投影暂不显示该面板（画布布局未扩展，已在代码中留 gate）。
> - **v44 失效 / v45 替换（2026-08-21）**：schema `vaw-context-v45-virtual-top`。
>   m16u task4 `context_0010`/`context_0016` 证明物理俯视相机被腕部+载荷完全自遮挡，
>   MuJoCo 直渲染俯视这条路结构性失败。v45 改为 RVT-2 式虚拟视图引擎
>   （`virtual_views.py`：RGB-D 反投影 + URDF FK 剔除机器人 + 准正交针孔点渲染）。
> - **v45 失效 / v46 替换（2026-08-21）**：schema `vaw-context-v46-oblique-contact` /
>   renderer `context-web-v46-oblique-contact`，182 项 VAW 测试通过。m16v task4
>   `context_0001` 证明只有 agentview + wrist 两台物理相机时，融合点云对桌面后方、
>   筐内与物体顶面根本没有采样，虚拟俯视退化为带空洞的碎点图，coverage 长期不过闸而
>   回退 OPPOSITE VIEW。**结论：RVT-2 的视点自由建立在多相机稠密点云上，两相机复现
>   不了它，硬渲反而产生 VLM 无法解析的画面。** v46 删除 `virtual_views.py`，右上退回
>   OPPOSITE VIEW，把"视点灵活性"放回本来就有相机位姿自由度和可见性搜索的 Contact 排：
>   `ContactCameraRequest.side_elevation_deg` 让 SIDE 在锁定方位轴上抬升，携带载荷时
>   取 55° 斜俯视，方位正负号在新仰角下重新搜索以避开手臂；面板标题标出
>   `OBLIQUE 55° DOWN`，MOVE BASE 罗盘按实际相机位姿投影，因此仍与 delta_move 符号一致。
>   动机与 v44/v45 相同——两个近水平面板只能就高度互相印证，横向对齐不可观测，于是
>   agent 只会调 Z；区别是 v46 用真实渲染的斜视角取得该自由度，而非合成一张俯视图。
> 未做：A 验收的 m16r 同任务复跑；B2/B4；B5 的 UI 图例面板；Imagination 投影的俯视面板；
> G1（seed 可执行性预检）、G4（受阻运动上浮）；阶段 C 全部；**v46 Contact 排评估**
> 前置：M1.5.2（Imagination 可选化 + 事务回滚）、v41 evidence lifecycle 已落地，144 项 VAW 测试通过
> 相关：`ICRA_PAPER_PROPOSAL.md`（论文主线）、`NOTES_SELF_EVOLVING_RSI.md`（外部方法论）、
> `M1_5_2_IMAGINATION_AGENT_CALL.md`（Imagination 契约）
> 外部依据（已核对正文）：RHI（arXiv:2607.15524，轨迹局部自比较、先改 contract/hop）、
> Ai2（arXiv:2607.12227，预算对等 TTS 对照 + 45/10/34 held-out 划分，进化在 held-out 上仅 +0.6）

---

## 0. 总原则：三层分界与两个闭包条件

RSI 开始之前，系统必须完成分层冻结：

| 层 | 内容 | RSI 期间 |
| --- | --- | --- |
| **Contract（契约层）** | Function API、Canvas 渲染与图例、事件语义、复验 verifier、基模 | **冻结** |
| **Knowledge（知识层）** | playbook（相位策略）、路由规则（何时委派 Imagination）、锚定程序 | **唯一进化面** |
| **Verifier（验证层）** | planner 可行性、commit TCP 误差、env_success、evidence 复验 | **冻结**（防 reward hacking，AIDE² 教训） |

冻结 Contract 需满足两个闭包条件，这决定了阶段 B 必须先于阶段 C：

1. **可观察性闭包**：知识层可能表达的任何策略，其判定所需的信息都能在 Canvas 上分辨。
   否则进化会撞上"策略对了但看不见"的墙——书上架任务左右视角被架子挡住就是反例。
2. **可表达性闭包**：观察到的失败模式，知识层都存在能修复它的表达。
   例如"容器开口两点求心"必须能用现有 locate_point + 事件文本写成一条 playbook 程序。

用户方向确认：**Canvas 本身不作为进化面**（evolve canvas 太难且会污染对照）；进化的是
Agent *使用* Canvas 的策略。这与 RHI 的实证一致——收益最大的进化面是信息接口与流程
（contract/hop），不是渲染器代码。

---

## 1. 诊断：为什么至今没有一条 trace 进入 Imagination

事实链（已核对 trace meta 与 protocol.py）：

- **m16l/m16m（v39）进过 Imagination**——因为当时 commit 有硬门控，refine 是 commit 的前置条件，进入是被结构强制的。
- **M1.5.2（v40）去掉硬门控后，m16q、m16r（opus-5，T=0）一次都没进过。**

三个原因叠加：

1. **Prompt 框架全是"去强调"**。现 prompt 中 call_imagination 的所有出现位置都在说
   "它是可选工具，不是 commit 的前置条件""判断清晰时可直接 commit"（protocol.py:72-73、262、272、278）。
   只有负向限定，没有一条正向触发条件（什么情形*应该*委派）。
2. **T=0 下的自信错配**。m16r 显示 agent 在锚点系统性偏差时依然自评"清晰"——
   "感到不确定时委派"这一路由前提在错得自信的场景下永远不触发。
3. **没有负反馈信号**。连续盲降、反复微调都不会产生任何提示委派的事件。

**对策定位（重要）**：不恢复硬门控。路由触发条件属于知识层——
阶段 A 先人工写一版 routing playbook-v0（如"容器/插入类放置默认委派"），
阶段 C 让它成为**第一个进化对象**。这恰好是论文最好讲的 RSI showcase：
第 0 代从不委派、放置阶段失败；进化后学会在特定相位委派——产物是一条可读的路由规则，
可解释、可回滚、可 ablate。

---

## 2. 阶段 A：契约层收尾与 bug 修复（预计 1–2 天）

来自 m16r 分析的已确认问题，全部是确定性修复：

- **A1 释放判据 prompt 缺陷**。现规则在"XY 证据不一致"时禁止修正，导致 agent 盲降到底。
  改为："证据不一致时，先补证据（换视角 locate / 复验），不得在未补证据的情况下继续下降或释放"。
  该内容属于未来 playbook 的雏形，按 playbook-v0 的行文风格写。
- **A2 盲降 advisory**。advisory ledger 增加一条：连续 N 次（默认 3）-z 方向 delta_move、
  且期间无新增载荷-目标同框证据 → 在 Function Event 中提示
  "连续下降但缺少对齐证据；建议补证据或委派 Imagination"。复用 v41 的 ledger 机制，不加门控。
- **A3 routing playbook-v0**。在 prompt 中给 call_imagination 补正向触发清单：
  容器放置、插入/上架、载荷-目标间隙小于载荷尺度、连续两次物理动作未改善对齐。
- **A4 locate 强制重测逃生口文档化**。verified 短路已有 invalidate 通道，
  在工具描述中写明何时该用（agent 有理由怀疑复验误判时）。

**A 验收**：全量 VAW 测试通过；单条 m16r 同任务复跑，观察 A2/A3 是否在放置相位触发。

---

## 3. 阶段 B：Canvas 完备化（固定契约基座，预计 1–1.5 周）

### B0 可观察性审计（先做，产出决定 B1–B4 的优先级）

建三维审计矩阵：**任务族 ×决策时刻 × 被控自由度**。

- 任务族：LIBERO-PRO 全任务按目标形态聚类（开口容器 / 平面放置 / 插入-上架 / 抽屉-铰链）。
- 决策时刻：pre-grasp 选点、grasp 后确认、carry 途中、align（XY）、descend（Z）、release 前。
- 判据：该时刻要判定的自由度，在至少一个视图中位移 1cm 是否产生可分辨的像素变化。
- 数据源：已有 trace（m16q/m16r/m16l）逐帧标注 + 每个任务族挑一个任务空跑采样。

产出：`vaw/diagnostics/CANVAS_OBSERVABILITY_AUDIT.md`，红格（不可分辨）驱动 B1–B4 排序。
已知红格：携带态 contact 视图不同框载荷与目标沿口；agentview 下 XY 深度歧义；书上架任务
左右视角被架体遮挡。

### B1 垂直辅助线（plumb line）——用户提议，采纳并展开

**几何定义**（全确定性，无学习组件）：

- 起点：载荷 OBB 底面中心（无载荷时为 TCP）；沿世界 -Z 投射垂线；
  与场景表面求交（privileged depth 渲染，v41 复验通道已在用同一数据源）。
- 渲染：垂线本体为虚线，画进 agentview / opposite / contact 所有场景视图；
  交点处画**落点足迹**（载荷 OBB 底面轮廓在交点高度的投影）；标注离面高度（cm）。
- **横向偏差向量**：若存在目标 region，同时渲染目标开口中心标记，
  以及落点→目标中心的 XY 偏差箭头 + cm 标注。
  这一条把"XY 对齐"从 VLM 深度猜测变成读一个渲染出来的数字，直接命中 m16r 的失败根因。
- **preview 同步**：对 planned Action 的 preview 位姿也画一条 plumb line（颜色区分 current/preview），
  commit 前即可回答"提交后载荷悬在哪上方"。
- **图例纪律**：图例必须写明"几何垂线与表面交点，非物理预测（不含倾倒/弹跳/滑动）"。
  它和 preview gripper 一样是确定性几何假设的可视化，符合 VAW 验证哲学。

实现落点：payload OBB 来自 M1.5.1 attached-object 几何；深度来自 privileged depth；
投影复用 preview 的相机投影工具；渲染在 web_renderer 加 overlay 层。

### B2 携带态双主体取景（carry-view）

载荷附着期间，自动生成一个同框视锥：包含载荷 OBB 与目标 region 的最小视锥，
从两个确定性选择的正交方位渲染（选择准则：最大化当前被控轴在像素上的分离度）。
替代/补充现有 contact 视图在携带态的取景。

### B3 放置相位近顶视图

align/descend 时刻自动加一个以目标 region 为中心的俯视裁剪视图；
机械臂自遮挡部分用 FK 剪影灰显（v41 已有剪影通道），让"被挡"成为显式信息而非误导。

### B4 任务轴视图（书上架类）

目标 region OBB 主轴确定插入轴，渲染沿插入轴方向的视图（视轴 ≈ 开口法线）。
解决"架子有洞、左右视角比不出位置"的问题。触发条件确定性：目标 region 的开口法线
与世界 Z 夹角超过阈值（侧开口容器）时启用。

### B5 契约固化

- 所有新 overlay 进图例；schema 升 `v42-canvas-complete`；vaw-ui types/renderer 同步。
- 每个新视图/辅助线加渲染回归测试（合成场景断言 overlay 存在且几何正确）。
- **Canvas 冻结判据**：审计矩阵全绿 + 开口容器与上架两类任务各 ≥2 条人工检查 trace
  中不存在"所需自由度不可分辨"的决策时刻。达标后 canvas 进入 Contract，RSI 期间不再改。

**B 收尾动作**：跑一轮 LIBERO-PRO baseline sweep（全任务 × 3 seeds），得到第 0 代成功率，
作为 RSI 的 generation-0 冻结基线。

---

## 4. 阶段 C：RSI v1（预计 1.5–2 周）

### C1 Prompt 因子化

- `protocol.py` 拆为：**contract 段**（工具语义、事件语义、canvas 图例——冻结，加 hash 守护测试）
  与 **playbook 库**（相位索引：grasp / transport / align / place / recover / **routing**）。
- Runtime 按当前相位确定性注入对应 playbook（不由模型自选，区别于 SKILL.md 的自加载）。
- playbook-v0 = 现 prompt 策略性内容的人工整理 + 阶段 A 的 A1/A3。

### C2 相位级 fitness

Trace 后处理器，从 events + evidence 确定性判定各相位成败与成本：

- `pick`: 载荷附着且离开支撑面；`transport`: 载荷到达目标邻域；
  `place`: env_success 或落点在目标 region 内；
- 每相位附轮次数、token 数、物理动作数。写入 meta，供 proposer 与采纳判据消费。

### C3 Sweep 基建

批量评测器：任务子集 × seeds 并行、单 episode 崩溃隔离、输出 `sweep_summary.json`
（总/分相位成功率、预算统计、advisory 触发统计）。所有进化对比都走它。

### C4 进化循环 v1（RHI 轨迹局部自比较 + AIDE² 采纳纪律）

```text
loop:
  1. 当前 playbook P_i 在训练任务集上 sweep → fitness F_i + 失败 trace 集
  2. proposer（强模型）读失败三元组（相位 fitness、事件流+advisory、canvas 截图）
     → 提出对 P_i 的一个局部 mutation → P_candidate
  3. P_candidate 同预算 sweep → F_candidate
  4. F_candidate 严格优于 F_i（先比 place 相位，再比总成功率，再比成本）→ 采纳，否则弃
  5. 每代改动 = 一个 git commit，天然可回滚可审计
```

- 改动面**只有 playbook 文本与路由规则**（对应 Autoresearch"只改一个文件"的纪律）。
- 不搞种群搜索：与上一代成对比较即可（RHI 的轻量化选择）。
- 预期采纳率低是常态（AIDE² 为 10%），拒绝也是数据。
- proposer 用 Opus 档；执行 agent 至少两档模型分层报（Lin et al. 非单调性，弱模型可能受害）。

### C5 防过拟合协议（直接回应 Ai2，论文防线）

1. **任务划分**：LIBERO-PRO 任务分 train / val / held-out（比例参照 Ai2 的 45/10/34 折算），
   进化只在 train 上搜索，采纳看 val，论文主结果报 held-out。
2. **预算对等 TTS 对照**（当前提案唯一缺失项，见 NOTES §6.4）：
   同 episode/token 预算的 parallel sampling（多 seed 取最好）与 sequential refinement 各一条 baseline。
   Ai2 的教训：不设这条，"RSI"会被打成"多跑了几次"（其实验中 HE 67.4 < 平采样 72.3）。
3. **公私分割**：proposer 可见相位 fitness 与失败描述，**不可见 held-out 分数与任务 ID**，
   防止对任务硬编码（AIDE² 公私分数隔离的等价物）。
4. **评估器冻结**：planner / TCP 误差 / env_success / evidence 复验全部不在进化面内。
5. **叙事对齐**：按 NOTES §0，这是 Harness-level self-improvement；
   只有当进化出的 playbook 在 held-out 上稳定优于预算对等 TTS 时，才在论文里主张接近 RSI。

### C6 Playbook 生命周期（Hermes Curator 模式）

每条 playbook 记录使用计数与采纳代数；被拒 mutation 归档不删除；
定期合并近重复条目。防止模板库成为第二种无限 Memory。

---

## 5. 时间表（对齐 ICRA 四周窗口）

| 周 | 内容 | 出口判据 |
| --- | --- | --- |
| W1 | 阶段 A 全部 + B0 审计 + B1 plumb line | m16r 同任务复跑放置改善；审计矩阵成文 |
| W2 | B2–B5 + baseline sweep | Canvas 冻结判据达标；generation-0 成功率入库 |
| W3 | C1–C3 + 循环干跑 1 代 | contract hash 测试上线；sweep_summary 可复现 |
| W4 | C4–C5 正式跑 + TTS 对照 | held-out 对比表（进化 vs TTS vs gen-0）成型 |

---

## 6. 非目标

- 进化 Canvas 渲染代码或任何 verifier（契约/验证层冻结）。
- 恢复 commit 硬门控（用 advisory + routing playbook 替代）。
- 种群级搜索、模型权重更新（SFT/RL 是飞轮下一圈，写 conclusion）。
- LLM-judge 作为采纳信号（只用物理验证层级）。
