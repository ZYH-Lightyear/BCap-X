# self_verify 选对了碗却抓不到 —— 根因定位、prompt 修复与测试

对比对象：

- **CaP-X 原生（成功）**：`outputs/qwen3_5_397b_distill/franka_libero_spatial_0/trial_01_..._reward_1.000_taskcompleted_1/code.py`
- **self_verify（失败）**：`outputs/self_verify_demo/final_program.py`

任务：`Pick the akita black bowl between the plate and the ramekin and place it on the plate`。
失败 run：`verified=true, executed=true, task_completed=false`，且**选对象（sam3_target）verify 是 PASS**（`block_00 ... "ok": true, "target (orig index 1) confirmed"`，VLM 也答 `SELECT: 2` 并给出正确空间理由）。

---

## 1. 现象：不是选错碗，是抓取姿态“歪了”，连碗都没夹住

逐帧看 `outputs/self_verify_demo/episode.mp4`（244 帧）与成功 trial 的 `video_turn_01.mp4`：

| | self_verify（失败） | CaP-X（成功） |
|---|---|---|
| 接近姿态 | 夹爪**几乎水平、侧着横怼**进碗群（帧 58/74/82） | 夹爪**接近竖直、从正上方往下罩**碗沿（帧 52/91） |
| 闭合瞬间 | 指尖戳在碗壁外侧/碗缝，**把碗推开** | 夹爪开口朝下对准碗沿，**夹住** |
| 结果 | 3 个碗原封不动留在桌上，盘子空（帧 183） | 碗被夹起并放上盘子，reward=1 |

→ 正如判断：**选对象没问题，是物理抓取时姿态歪了、没夹住。**

---

## 2. 根因：grasp 的“朝向”来自 GraspNet，两份 code 都原样信任它，但只有 CaP-X 被 prompt 引导成了“竖直朝下”

两份 code 都调同一个 `sample_grasp_pose`（`libero.py:347`）：内部 `plan_grasp_from_point_clouds` → `grasp_scores.argmax()` → 返回 GraspNet 的原始 `(position, quaternion)`，**不做任何朝向约束**。GraspNet 对开口朝上的薄壁碗，不同 run/点云会给出朝向差异很大的候选，argmax 偶尔就是一个“横向抓”。

谁来兜底“竖直朝下”？——只有 **system prompt**。这正是两边的分水岭。

### CaP-X 原生 prompt（成功）有两处“朝下”引导

来自实际发给模型的 `prompts_and_responses/initial_prompt.txt`：

- `get_object_pose` docstring：
  > “use the grasp pose quaternion **OR `(0, 0, 1, 0)` wxyz as the gripper down orientation**”
- `sample_grasp_pose` docstring：
  > “**Do use the grasp sample quaternion** from sample_grasp_pose.”

模型因此脑子里有“**gripper down orientation = (0,0,1,0)**”这个锚点。成功 code 的放置段直接用了它：`place_quaternion = np.array([0.0, 0.0, 1.0, 0.0])`。
（注：CaP-X 的抓取段仍用了 GraspNet quaternion，但那次 argmax 恰好是个能从上方罩住碗的候选，所以成功——它本质上是“运气 + 有 down-orientation 先验”。）

### self_verify 的 prompt（失败）在抓取朝向上几乎是空的

原 `self_verify_agent.py` 的 SYSTEM_PROMPT：

- grasp 示例：`position, quaternion = sample_grasp_pose("akita black bowl")` 然后 `goto_pose(position, quaternion, ...)` —— **直接喂 GraspNet 的 quaternion，没有任何“应朝下”的引导**；
- 注释只讲 timing（approach→descend→close→lift），**完全不提朝向**；
- 没有 `(0,0,1,0)` 这个 down-orientation 锚点；
- `grasp` 这个 verify check 只问“点在不在目标上”，也不管朝向（且默认还没激活）。

**结论**：self_verify 的 system prompt 缺了 CaP-X prompt 里那条“gripper-down = (0,0,1,0)、抓取自上而下”的引导，于是模型原样吃下 GraspNet 偶发的横向 quaternion → 横着怼 → 抓空。
**这是 prompt 层面的问题，不是高层 API、不是底层 IK、不是端口。**

---

## 3. 修复（只改 self_verify 的 SYSTEM_PROMPT，对齐 CaP-X）

文件：`scripts/self_verify_agent.py`，两处编辑，不动 API / 底层 / 端口 / 默认 check。

1. **CORRECT pattern 的 grasp 段**：只从 `sample_grasp_pose` 取 **position**，姿态强制用 `down_quat = (0,0,1,0)`，grasp/抬起全程用它：
   ```python
   position, _ = sample_grasp_pose("akita black bowl")  # use the POSITION only
   down_quat = np.array([0.0, 0.0, 1.0, 0.0])
   quaternion = down_quat
   # @verify: grasp position=position quaternion=quaternion target_points=tp
   goto_pose(position, down_quat, z_approach=0.10)
   goto_pose(position, down_quat, z_approach=0.0)
   close_gripper()
   goto_pose(position + np.array([0,0,0.15]), down_quat, z_approach=0.0)
   ```
2. **Rules 段**新增明确规则：
   > GRIPPER ORIENTATION: a bowl is grasped TOP-DOWN. Always use the gripper-down
   > orientation quaternion (0,0,1,0) wxyz for BOTH grasp and place; take only the
   > POSITION from sample_grasp_pose. Its returned quaternion is often tilted and
   > will knock the bowl over -- do NOT feed it to goto_pose.
   > Carry and place with the SAME (0,0,1,0) down orientation.

这把 CaP-X prompt 里“down orientation = (0,0,1,0)”的先验，以更强、更明确的形式（直接禁用 GraspNet 的歪 quaternion）注入 self_verify。

---

## 4. 测试

环境与服务（端口不动，默认 8114/8115/8116）：

```bash
cd /mnt/data/xuyingjie/BCap-X
bash scripts/start_libero_services.sh            # 起 SAM3/GraspNet/PyRoKi
MUJOCO_GL=egl .venv-libero/bin/python scripts/self_verify_agent.py
```

判定：`outputs/self_verify_demo/result.json` 的 `task_completed`，以及 `outputs/self_verify_demo/episode.mp4` 里夹爪是否**自上而下竖直**抓取、碗是否被夹起放上盘子。

> 测试结果待补：______

---

## 5. 后续修复（driver 架构 + 多候选抓取选择）

定位过程中又发现两个比"姿态"更本质的结构性问题，一并修了。

### 5.1 verify 结果在 commit 阶段失效（episode 抓错的真正原因）

**现象**：`program_attempt_1` 里 sam3_target 明明 verify 选对了中间那个碗，但 `episode.mp4` 还是抓错。

**根因**：旧流程是“两段独立执行”——verify 全 PASS 后 `env.reset()` 重置环境、用一个**全新空 namespace** 重跑 `final_program.py` 录 `episode.mp4`。重跑时 `bowls = segment_sam3_text_prompt(...)` 是**第二次独立分割**，`sorted` 后的 `bowls[sel]`（`sel` 是写死的整数索引）可能已经指向**另一个碗**——索引漂移。而且重跑是裸 `exec`，`# @verify` 只是注释、不再触发任何检查，错了也没人拦。即“**验过的那次 ≠ 录下来的那次**”。

**修复（方案 2，`self_verify_agent.py`）**：删掉末尾的 reset + 重跑。verify 全 PASS 的那一轮**本身就已逐块执行到底**（同一个 sim、同一个 `ns`），世界已是终态、变量就是验过的那批。直接读 `task_completed` + 录 `episode.mp4`。→ **验的那次 = 跑的那次 = 录的那次**，索引不再漂移。

### 5.2 选错了如何当场救回 → 一次成功（就地纠正）

旧流程：任何 verify FAIL 都 `break` → 把整段程序丢回 VLM 重写 → 下一轮 `env.reset()` 重新分割，上一轮算出的正确索引在新分割里又失效。

**修复（`self_verify_agent.py`）**：verify FAIL 时若 check 返回了纠正索引（`result.data["selected"]`，非 uncertain），driver **就地把它写回同一个 namespace**（`ns[sel_var]=corrected`）→ 复验 → **继续往下执行**，不重写、不重置。因为没有二次分割，纠正索引在整段里恒有效。只有“候选里压根没有对的”（uncertain）才退回整段重写。
检查器本来就会算出 `corrected`（见 `checks.py` 的 `check_sam3_target` / `check_grasp_select` 返回 `data["selected"]`），过去被“重写+重置”浪费掉了，现在直接消费它。

### 5.3 多候选抓取 + 抗干扰选择（可视化 + VLM 打分）

**动机**：`sample_grasp_pose` 只返回 argmax 一个 grasp，没有“哪个抓取点最不容易被旁边的碗/盘子/ramekin 干扰”的判断。

**新增**：
- API `sample_grasp_candidates(object, k=3)`（`libero.py`，evidence 子类 `functions()` 暴露）：返回 top-k 候选 `[{"position","quaternion","score"}]`，复用 `sample_grasp_pose` 的同源点云+GraspNet 路径，只是不取 argmax。
- verify check `grasp_select`（`checks.py`，已注册进 REGISTRY + VERIFY_CATALOG）：把 ≥3 个候选用**点+接近箭头+编号(①②③，颜色区分，选中标 SEL)** 画在 agentview 图上（复用 `draw_grasp_point_on_image` + `HIGHLIGHT_PALETTE`），问 VLM **“哪个抓取点在下降/闭合时最不容易被邻近物体挡住/碰到”**，VLM 答 `SELECT: <编号>`。返回 `data["selected"]` → 走 5.2 的就地纠正写回 `gsel`。
- SYSTEM_PROMPT 改为引导 VLM 写“`sample_grasp_candidates(k=3)` → `# @verify: grasp_select` → 用 `cands[gsel]` 抓”，并默认激活：`CAPX_ACTIVE_CHECKS` 默认 `sam3_target,grasp_select`。

**可视化落盘**（你要求的）：每次 grasp_select 的候选标号图存为
`program_attempt_*/block_xx_grasp_select/vlm_input.png`，VLM 的选择/理由存 `vlm_qa.txt` + `decision.json`（发生纠正时还有 `decision_recheck.json` 记 `corrected_to`）。

**离线烟雾测试**（无 GPU 本机）：`_draw_numbered_grasps` 正确画出 3 个编号候选点；`check_grasp_select` 正确把 `SELECT: 2` 解析为 0-based 索引 1 并返回 `data["selected"]=1`。✅

### 5.4 现在的端到端语义

```
reset → 分割一次 → sam3_target verify（选错→就地改 sel→复验→继续）
      → sample_grasp_candidates(k=3) → grasp_select verify
             （画①②③→VLM 选抗干扰最强→选错→就地改 gsel→复验→继续）
      → 用 cands[gsel].position + 竖直朝下(0,0,1,0) 抓 → 抬 → 搬 → 放
      → 全块 PASS → 直接录 episode.mp4 + 读 task_completed   （不再 reset 重跑）
```

### 5.5 运行（命令不变，默认就跑多候选抓取选择）

```bash
cd /mnt/data/xuyingjie/BCap-X
bash scripts/start_libero_services.sh
MUJOCO_GL=egl .venv-libero/bin/python scripts/self_verify_agent.py
```

跑完看：
- `outputs/self_verify_demo/program_attempt_*/block_*_grasp_select/vlm_input.png` —— 3 个候选抓取点的标号可视化图
- 同目录 `vlm_qa.txt` / `decision.json`（含 `corrected_to`）—— VLM 选了几号、为什么
- `episode.mp4` / `result.json` —— 抓取是否竖直、是否抓住、`task_completed`

> 端到端测试结果待补（需在有服务的机器上跑）：______

