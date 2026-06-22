# Verifier 任务清单（长期规划）

## 使用说明

```text
1. 状态有四种：未开始 / 进行中 / 已完成 / 已取消（保留行不删，方便追溯决策）。完成一项改一项，同时更新总览表计数。
2. 主体：我 = verifier 侧负责人；合作者 = skill 侧负责人；双方 = 需要一起做。
3. 编号规则：V<里程碑>.<序号>，例如 V4.3 = M4 里程碑第 3 个任务。引用任务时直接说编号。
4. 重大变更（加任务、砍任务、改验收标准）在文末 Changelog 记一行。
```

## 核心范式（2026-06-15 修正，统领全表）

```text
不是"把校验旁路挂进多轮执行循环"，而是"自验证分块代码 + 校验/执行两阶段分离"：

1. VLM 写一份完整 Python，按块组织，每个需要校验的块在末尾用注释声明验证点：
       # @verify: <check_type> param=var ...
   所有可用校验手段在 system prompt 开头一次性告知 VLM。
2. 校验阶段（逐块、可反复改、不算最终执行）：
   逐块执行 → 派发到对应校验（sam3=关系校验, grasp=模拟器实试抓+holding,
   transit/place=几何文本报告）→ 不过则把"该块代码+失败信息"交回 VLM
   只重写这一块 → 重试至通过。每个块的校验基于前面已通过块的结果。
   关键：校验阶段对机器人/物体的改动必须可回滚（sim 状态存档/回档），
   否则破坏"最终从初始态一次性执行"的前提。
3. 校验全过 → 得到一份"每块都验证过的完整代码" → env.reset() 回初始态 →
   合并执行一次，一步到位。

承载这个范式的是 M9；M7（旧"接入多轮 trial 循环"）相应调整。
```

---

## 0. 总览

| 里程碑 | 内容 | 任务数 | 已完成 | 整体状态 |
|--------|------|--------|--------|----------|
| M0 | 已有资产盘点（不是任务，是底子） | - | - | 已确认 |
| M1 | 证据可视化基建 | 6（V1.5 取消） | 3 | 进行中 |
| M2 | ActionTrace 接口契约 | 5 | 0 | 未开始 |
| M3 | WATCH verifier（关系指代） | 6 | 0 | 未开始 |
| M4 | ACTION verifier（grasp） | 8 | 0 | 未开始 |
| M5 | ACTION verifier（place + transit） | 8 | 1 | 进行中 |
| M6 | 后置验证与校准 | 5 | 0 | 未开始 |
| M7 | 校验/执行两阶段驱动接入 | 6 | 0 | 未开始 |
| M8 | 评测与联调 | 5 | 0 | 未开始 |
| M9 | 自验证分块 agent（核心范式） | 7 | 2 | 进行中 |

**推进顺序**：M1/M2 先行 → M9 搭框架（块解析+校验派发+回滚+两阶段）与各 check 并行做（M3/M4/M5 是 M9 各 check 的实现细节）→ M6 → M7 接入 → M8。
**最小可演示闭环**：M9 框架 + sam3_target check + grasp check（模拟器实试）→ between bowl 任务校验全过后一次性执行成功。

---

## M0. 已有资产盘点（直接复用，不用造）

| 资产 | 位置 | 状态 |
|------|------|------|
| SAM3 / ContactGraspNet / PyRoKi sidecar 服务 | `capx/serving/launch_servers.py` | 已存在 |
| VLM 调用通道（OpenAI 兼容 proxy） | `capx/serving/openrouter_server.py` | 已存在 |
| 2D 可视化：mask overlay / OBB / point 绘制 | `capx/utils/visualization_utils.py` | 已存在 |
| block 级 debug overlay 保存钩子 | `capx/integrations/franka/libero_reduced.py` (`set_debug_context`) | 已存在 |
| line-level 执行 trace（按 block 分文件） | `capx/utils/line_trace.py` | 已存在 |
| 几何工具：`mask_to_world_points` / `depth_to_point_cloud` / OBB | `capx/integrations/franka/libero_reduced_skill_library.py` | 已存在 |
| IK 求解（PyRoKi）+ fallback | `libero_reduced.py::solve_ik` | 已存在 |
| 特权真值 API（oracle 标签来源） | `capx/integrations/franka/libero_privileged.py` | 已存在 |
| cuRobo 碰撞规划（默认被注释） | `capx/integrations/franka/libero.py` L88-91 | 已实现未启用 |
| 点云 o3d 可视化参考代码 | `capx/third_party/contact_graspnet_pytorch/.../visualization_utils_o3d.py` | 可借鉴 |

---

## M1. 证据可视化基建（verifier 的眼睛）

> 说明：这里的"多视角"指虚拟相机渲染点云，不移动机器人，不违反固定视角原则。

| 编号 | 任务 | 主体 | 状态 | 产出与验收标准 | 依赖 |
|------|------|------|------|----------------|------|
| V1.1 | 离屏点云渲染器：场景点云 + 高亮目标 + 可选 mesh，2-3 个固定虚拟相机位 | 我 | 已完成 | `capx/utils/pointcloud_render.py`；给定点云输出 PNG，肉眼可分辨目标 | 无 |
| V1.2 | gripper pose overlay：夹爪线框按 grasp pose 投影到 RGB 和点云渲染图，带 approach 箭头和开口宽度 | 我 | 已完成 | 输入 grasp pose 输出叠加图；人工核对 5 个 pose 投影位置正确 | V1.1 |
| V1.3 | top-k grasp 候选总览图：编号 + 得分 + IK 可达性着色 | 我 | 已完成 | 单图呈现 k 个候选；VLM 能按编号指认候选 | V1.2 |
| V1.4 | place 度量证据文本报告（metric evidence report）：measurements 数字 + derived_checks 规则核验 + context，JSON 与 VLM prompt 模板两种形态；top 视角投影图仅用于语义 in-region/障碍判断；度量量一律不渲染成图给 VLM 目测（proposal §7.5） | 我 | 进行中 | 真实场景生成报告，数字与特权真值一致（待 oracle 校验）；VLM 基于纯文本报告给出 approve/reject/uncertain（已验证） | V1.1 |
| V1.5 | ~~EE 轨迹 preview 投影到 RGB~~ | 我 | 已取消 | 取消理由：路径安全是度量问题，斜视角 2D 投影读不出间隙（证据分型原则）；该职能已被 V5.8 走廊间隙表用数字接管且自带修复值；人工查看用平台已有执行视频即可 | - |
| V1.6 | API 调用时自动产证据：在 `sample_grasp_pose`/`goto_pose`/分割等 API 调用点挂钩子（沿用 `_save_debug_overlay` 模式），调用现有具名渲染函数与报告生成器自动落盘证据；不做注册抽象，5 类证据直接函数调用 | 我 | 未开始 | agent 跑一个 trial，每次关键 API 调用自动产出对应证据图/报告，零手工 | V1.1-V1.4 |
| V1.7 | evidence artifact 目录规范：每个语义动作一个目录 + `meta.json`，命名 `{action_idx}_{action_type}/{evidence_name}.png`；与 M2 的 ActionTrace `evidence_paths` 字段同一套规范（和 V2.4 一起定，单一事实来源） | 我 | 未开始 | 命名规范文档 + 落盘实现；VLM 输入、debug、论文配图三用 | V1.6, V2.1 |

## M2. ActionTrace 接口契约（与 skill 侧解耦的关键，最高优先级）

| 编号 | 任务 | 主体 | 状态 | 产出与验收标准 | 依赖 |
|------|------|------|------|----------------|------|
| V2.1 | 定义 `ActionTrace` JSON schema（intent / code / evidence / 决策 / postcondition / imagined-realized / failure_type / outcome） | 我 | 未开始 | schema 文档 + 示例 JSON；skill 侧确认字段够用 | 无 |
| V2.2 | 定义 `VerifierReport` schema（决策 + 各检查项明细） | 我 | 未开始 | schema 文档 + 示例 JSON | V2.1 |
| V2.3 | block → 语义动作的切分与标注记录（先人工/规则切分） | 我 | 未开始 | 切分记录格式 + 在 1 个 trial 上人工标注示例 | V2.1 |
| V2.4 | 落盘实现：挂在现有 trial 输出目录下，与 `line_trace` 并列 | 我 | 未开始 | 跑一个 trial 产出合法的 ActionTrace 文件 | V2.1-V2.3 |
| V2.5 | 与合作者评审并冻结 v1 接口（之后改动走版本号） | 双方 | 未开始 | 双方签字确认的 v1 schema；写进文档 | V2.1-V2.4 |

## M3. WATCH verifier（关系指代 grounding）

| 编号 | 任务 | 主体 | 状态 | 产出与验收标准 | 依赖 |
|------|------|------|------|----------------|------|
| V3.1 | referring expression 分解：target / anchors / relation（LLM 结构化输出） | 我 | 未开始 | 对 LIBERO spatial 全部任务指令解析正确（人工核对） | 无 |
| V3.2 | relation 几何判定库：left/right/between/next_to/on/inside/behind/front_of | 我 | 未开始 | `capx/verifier/relations.py` + 单元测试 | 无 |
| V3.3 | 候选实例打分 + ambiguity report（得分接近 → uncertain） | 我 | 未开始 | 输出 selected target + confidence + ambiguity 标记 | V3.1, V3.2 |
| V3.4 | VLM 仲裁通路：候选 overlay 图 → VLM 选择/否决 | 我 | 未开始 | uncertain 案例经 VLM 仲裁后给出最终决策 | V3.3, V1.6 |
| V3.5 | WATCH 决策输出接入：uncertain 时拒绝进入 ACTION | 我 | 未开始 | 决策写入 VerifierReport | V3.3, V2.2 |
| V3.6 | oracle 自动标注：特权 API 真值判定 referent 对错，生成 benchmark 标签 | 我 | 未开始 | WATCH benchmark 标签集，零人工标注 | V3.3 |

## M4. ACTION verifier — grasp

| 编号 | 任务 | 主体 | 状态 | 产出与验收标准 | 依赖 |
|------|------|------|------|----------------|------|
| V4.1 | grasp 点落在目标 mask/点云上的 contact 一致性检查 | 我 | 未开始 | 检查函数 + 阈值可配 | 无 |
| V4.2 | gripper 开口宽度 vs 目标 OBB extent 匹配检查 | 我 | 未开始 | 检查函数 + 单元测试 | 无 |
| V4.3 | approach 方向 vs 桌面法向/物体主轴合理性检查 | 我 | 未开始 | 检查函数 + 单元测试 | 无 |
| V4.4 | quaternion 单位化与 wxyz 顺序检查 | 我 | 未开始 | 检查函数；能抓出历史 bug 案例 | 无 |
| V4.5 | pregrasp + grasp 双位姿 IK 可达性检查（复用 PyRoKi） | 我 | 未开始 | 不可达候选被正确标记 | 无 |
| V4.6 | clearance/碰撞代理：优先启用 cuRobo（取消注释+验证），否则点云近邻距离代理 | 我 | 未开始 | 碰撞候选被正确标记；记录用的哪条路线 | V4.5 |
| V4.7 | top-k 候选综合重排（得分 x 可达性 x clearance x contact） | 我 | 未开始 | 重排后 top-1 通过全部检查 | V4.1-V4.6 |
| V4.8 | grasp VerifierReport 输出 + evidence 图 | 我 | 未开始 | 每次 grasp 验证产出报告 + 图 | V4.7, V1.3, V2.2 |

## M5. ACTION verifier — place

| 编号 | 任务 | 主体 | 状态 | 产出与验收标准 | 依赖 |
|------|------|------|------|----------------|------|
| V5.1 | receptacle support surface z 估计（mask + depth） | 我 | 未开始 | 估计值与特权真值误差 < 1cm | 无 |
| V5.2 | 抬升后测量 TCP-物体最低点悬垂距离：抬升后的观测里分割被抓物体（SAM3/点云差分），物体点云 min_z + 本体感知 TCP z → 悬垂距离（替代"object_height/2"假设，修正碗沿抓取等非质心抓取的高度计算） | 我 | 未开始 | 悬垂距离与特权真值误差 < 1cm | 无 |
| V5.3 | place point 在目标区域内检查（mask 内 + 距中心距离） | 我 | 未开始 | 检查函数 + 单元测试 | V5.1 |
| V5.4 | release height 计算与检查：release_tcp_z = surface_z + 悬垂距离 + clearance（悬垂距离来自 V5.2 实测），拒绝固定常数 | 我 | 未开始 | 检查函数；magic number 案例被拒绝 | V5.1, V5.2 |
| V5.5 | 释放前 holding 检查（gripper 闭合宽度 + 腕部相机） | 我 | 未开始 | 空抓案例被正确拒绝 | 无 |
| V5.6 | place pose IK 可达性 + 下探路径检查 | 我 | 未开始 | 不可达/危险路径被标记 | V5.4 |
| V5.7 | place VerifierReport 输出 + 虚拟放置投影图 | 我 | 未开始 | 每次 place 验证产出报告 + 图 | V5.3-V5.6, V1.4, V2.2 |
| V5.8 | transit（运送段）走廊间隙检查：场景 box 化（`capx/verifier/boxes.py`）+ 被抓物体扫掠走廊 + 逐障碍物间隙表 + min_safe_tcp_z 修复值，证据以文本报告交付 | 我 | 已完成 | 真实场景逐障碍输出间隙表；低高度方案被 hard reject 且报告自带修复高度（已验证，悬垂暂用 pre-grasp 代理，V5.2 完成后替换） | V5.2(代理可先行) |

## M6. 后置验证与 imagined-vs-realized 校准

| 编号 | 任务 | 主体 | 状态 | 产出与验收标准 | 依赖 |
|------|------|------|------|----------------|------|
| V6.1 | grasp postcondition：物体离开支撑面、随 gripper 移动（前后点云/mask 对比） | 我 | 未开始 | 判定与特权真值一致率 > 90% | M4 |
| V6.2 | place postcondition：释放、中心在区域内、高度贴近 surface | 我 | 未开始 | 判定与特权真值一致率 > 90% | M5 |
| V6.3 | imagined vs realized evidence 结构化对比，写入 ActionTrace | 我 | 未开始 | 每个动作的 trace 含对比字段 | V6.1, V6.2, V2.4 |
| V6.4 | 用 oracle 统计 false approve / false reject（混淆矩阵） | 我 | 未开始 | 混淆矩阵脚本 + 首批统计结果 | V6.3 |
| V6.5 | 阈值校准流程：量化调整 verifier 松紧，产出校准曲线 | 我 | 未开始 | 校准前后混淆矩阵对比（论文素材） | V6.4 |

## M7. 校验/执行两阶段驱动接入

> 把 M9 的自验证分块驱动接入正式 eval 流程（替代旧的"旁路挂进多轮循环"设想）。

| 编号 | 任务 | 主体 | 状态 | 产出与验收标准 | 依赖 |
|------|------|------|------|----------------|------|
| V7.1 | 把 M9 驱动包装成可被 `capx/envs/launch.py` 调用的 eval 入口 | 我 | 未开始 | between bowl 任务经统一入口端到端跑通 | M9 |
| V7.2 | 块重写预算与失败兜底（单块 N 次、整程序回退策略） | 我 | 未开始 | 超预算时优雅失败并记录 | V7.1 |
| V7.3 | uncertain 分支：校验不确定时当前视角补证 → VLM 仲裁 → 仍不确定则拒绝 | 我 | 未开始 | uncertain 分支完整可走通 | V7.1, V9.2 |
| V7.4 | YAML 开关：verifier 开/关、各 check 开关、预算；关掉后等同原始 CaP-X | 我 | 未开始 | 关掉开关行为与 baseline 一致 | V7.1 |
| V7.5 | runtime / token / 校验调用次数统计落盘 | 我 | 未开始 | 每 trial 输出调用次数与耗时 | V7.1 |
| V7.6 | 等算力 baseline 配置：给原始 CaP-X 同等 VLM 预算（如 best-of-N） | 我 | 未开始 | baseline 配置文件 + 跑通记录 | V7.5 |

## M8. 评测与联调

| 编号 | 任务 | 主体 | 状态 | 产出与验收标准 | 依赖 |
|------|------|------|------|----------------|------|
| V8.1 | WATCH benchmark：LIBERO spatial 关系任务子集 + oracle 标签 | 我 | 未开始 | referent accuracy / wrong-object rate / ambiguity detection rate 数据表 | M3, V3.6 |
| V8.2 | ACTION benchmark：grasp/place 验证效果 | 我 | 未开始 | rejection rate / approved success rate 数据表 | M4, M5, M6 |
| V8.3 | end-to-end LIBERO spatial 对比（原始 / +verifier / +等算力 baseline，>=3 seeds） | 我 | 未开始 | 主结果表 | M7 |
| V8.4 | 与 skill 侧联调：skill 的 verification schema 注入 verifier 检查清单 | 双方 | 未开始 | ActionTrace 接口闭环验证通过 | V2.5, M7 |
| V8.5 | 失败案例集整理（motivation case + 典型 reject/approve 案例） | 我 | 未开始 | 论文素材库 | V8.3 |

## M9. 自验证分块 agent（核心范式）

> VLM 写带 `# @verify:` 标记的分块代码；驱动器逐块执行+派发校验+失败回灌让 VLM 重写该块；校验阶段可回滚；全过后从初始态一次性执行。框架在 `scripts/self_verify_agent.py` + 校验注册表 `capx/verifier/checks.py`。

| 编号 | 任务 | 主体 | 状态 | 产出与验收标准 | 依赖 |
|------|------|------|------|----------------|------|
| V9.1 | 块解析器 + 校验派发注册表：解析 `# @verify: type k=v`，按 type 派发到 check，失败把"块代码+信息"交回 VLM 重写并重跑 | 我 | 已完成 | `split_blocks` 解析正确；`REGISTRY` 可派发（已冒烟测试通过） | 无 |
| V9.2 | sam3_target check：关系指代校验（编号框图→VLM 选择/纠正→回灌正确 index 让块改） | 我 | 已完成 | between bowl 场景，naive top-1 被纠正为正确实例并改代码继续 | V9.1, M3 |
| V9.3 | grasp check 改为"模拟器实试抓"：校验阶段真抓+抬升+holding 检测，失败回灌重规划；试抓后回滚到初始态 | 我 | 未开始 | 抓空案例被检出并触发重写；回滚后状态与试抓前一致 | V9.1, M4 |
| V9.4 | sim 状态存档/回档：校验阶段开始前存档，每次涉及动作的校验后回档，保证"最终执行从初始态" | 我 | 未开始 | 存档→试抓→回档后物体/机器人位姿误差 < 1mm | 无 |
| V9.5 | transit / place check 接入分块校验（复用 V5.8 / V1.4 的文本报告，失败回灌修复值/约束） | 我 | 未开始 | 低运送高度、place 偏心案例被检出并改代码 | V9.1, V5.8, V1.4 |
| V9.6 | 校验/执行两阶段分离：全块校验通过 → env.reset() → 合并校验后代码一次性执行 → settle 后读成功判定 | 我 | 未开始 | between bowl 校验全过后一次性执行，task_completed=True | V9.2-V9.5 |
| V9.7 | 每块产物落盘：每块的代码（逐次 try）、证据图、VLM 问答、决策 JSON + 全程视频 | 我 | 进行中 | 目录结构齐全，可复盘每块改了什么（已部分实现，video+vlm_input 已加） | V9.1 |

---

## Changelog

- 2026-06-11: 初版（清单格式），M0 资产盘点确认，M1-M8 共 49 个任务全部"未开始"。
- 2026-06-11: V1.1 完成（`capx/utils/pointcloud_render.py`，纯 numpy z-buffer，无 GL 依赖）；V1.2 进行中（RGB/点云线框投影已实现，LIBERO spatial task 0 真实场景验证：腕部相机线框与真实指尖对齐；待接 GraspNet 候选后补多 pose 核对）。demo：`scripts/m1_render_demo_libero.py`，产物在 `outputs/m1_render_demo/`。
- 2026-06-12: 设计修正（讨论产出）：确立"证据分型原则"——语义判断给 VLM 看图，度量判断由几何算数字，VLM 只以文本接收数字做 sanity check，不让 VLM 从像素目测厘米级量。V1.4 重定义为放置证据三件套；V5.2 改为抬升后实测 TCP-物体最低点悬垂距离（修正 object_height/2 的质心抓取假设，碗沿抓取场景下该假设错误）；V5.4 公式同步更新。
- 2026-06-12: M1 收尾审视：V1.5 取消（路径间隙是度量问题，斜视角投影读不出，职能已被 V5.8 数字化接管）；V1.6 保留但去抽象化——改为"API 调用点挂钩子自动产证据"，不做注册机制；V1.7 保留，规范与 M2 的 evidence_paths 统一（和 V2.4 一起定）。
- 2026-06-12: 新增并完成 V5.8 transit 走廊检查（`capx/verifier/boxes.py` + `build_transit_report`，demo `scripts/m1_transit_report_demo.py`）：场景 box 化后，故意给低运送高度（TCP 6.4cm），间隙表抓出真实碰撞——被抓碗底部 2.0cm 低于路径上另一只碗的顶部 3.9cm（间隙 -1.9cm），hard reject 并自带修复值 min_safe_tcp_z=11.3cm；LLM 纯文本判决 reject 且引用了正确的障碍物。proposal §7.2 新增 transit future 描述。设计原则补充：物体统一 box 化表示（最适合计算）。
- 2026-06-12: V1.4 主体完成（待 oracle 校验后关闭）：新增 `capx/verifier/metric_report.py`（报告构建/硬规则短路/`format_for_vlm`），proposal 新增 §7.5"度量证据以结构化文本交付 VLM"。真实场景端到端验证（`scripts/m1_place_report_demo.py`）：SAM3 ground plate+bowl → 几何计算 surface_z=0.3cm、悬垂 4.4cm、release_tcp=6.6cm → 6 条规则全 PASS → qwen 纯文本判决 "DECISION: approve" 并主动指出 2cm clearance 的落差风险。注意：reasoning 模型需 max_tokens>=4000，否则思考耗尽 token 导致 content=None。
- 2026-06-12: V1.2、V1.3 完成（`scripts/m1_grasp_sam3_demo.py`，产物在 `outputs/m1_grasp_demo/`）：SAM3 → mask 升 3D → Contact-GraspNet top-5 → PyRoKi IK 标注 → RGB 总览图（角落图例）+ 点云三视角全链路打通。两个 WATCH 级发现：(1) SAM3 top-1 "black bowl"(0.79) 不是任务要的 between bowl(0.76)，grasp 全部落在错误的碗上——motivation case 的 wrong-object failure 在证据图中直接可见；(2) SAM3 不认识 "ramekin"（score 0.002）。运维记录：8115 GraspNet 服务僵死（accept 不响应），已在 8125 重启（`GRASPNET_SERVICE_URL` 指向），demo 脚本已对 pc_full 降采样至 4 万点。
- 2026-06-15: 范式修正（重要）：确立"自验证分块代码 + 校验/执行两阶段分离"为核心范式（见顶部"核心范式"段）。新增 M9 承载之，已完成 V9.1（块解析器+校验注册表 `capx/verifier/checks.py` + 驱动 `scripts/self_verify_agent.py`，块解析冒烟通过），V9.2/V9.7 进行中。M7 由"旁路挂进多轮循环"改为"把 M9 两阶段驱动接入 eval 入口"。关键设计约束：校验阶段对机器人/物体的改动必须可回滚（V9.4 sim 存档/回档），grasp 校验须在模拟器实试抓而非仅 VLM 看图（V9.3）。
- 2026-06-15: 确认任务成功判据（BDDL）：`On(bowl,plate)` = 碗心与盘心水平距离 < 3cm 且接触且碗不高于盘心——比"放到盘上"严格，place 校验须瞄准盘心、容差对齐 3cm。修复 demo 成功判定时机：`compute_reward()` 是 step 缓存值，须用 `task_completed()`（实时 check_success）并在松爪后让仿真沉降数十步再读。
- 2026-06-15: V9.2 完成（`scripts/test_v92_sam3.py` 聚焦验证）：sam3_target 关系校验链路跑通——naive top-1(0.785) 经 VLM 关系判断纠正为正确的 between bowl(index1, 0.754)，失败信息回灌后 VLM 重写该块改 sel=1，重跑 PASS 收敛。修复两个 bug：(1) `check_sam3_target` 漏了 SAM3 分数过滤，把 200 个 mask 全画上去糊死画面——加回 `score>=0.3` top-5 过滤（与 `m1_grasp_sam3_demo` 一致）；(2) 解析不到 `SELECT:` 时不再默认 PASS，改判 uncertain 重试（避免静默误判通过）。新增能力：`FrankaLiberoApiEvidence` 暴露 `segment_sam3_text_prompt` 给 agent 拿候选列表（baseline 高层 grounding 把选实例藏在内部 top-1，是 wrong-object 源头）。
