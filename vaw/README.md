# VAW — Visual Action Workspace

`vaw` 当前只保留 revision-local Context Runtime。它把 LIBERO-PRO 的当前双相机观测、
工具产生的局部证据、机器人本体状态和动作想象编译为一张固定 `1440×1080` Context
图，供单个 VLM 逐轮调用 Function 控制机器人。

VAW 不是 GUI Agent：模型不点击页面，也不操作 DOM。Web 页面只是确定性的只读视觉
renderer；动作通道始终是 structured Function call。

设计与里程碑见：

- [`CONTEXT_RUNTIME_MILESTONES.md`](CONTEXT_RUNTIME_MILESTONES.md)
- [`M1_3_2_VISUAL_DENSITY.md`](M1_3_2_VISUAL_DENSITY.md)

## Agent 每轮看到什么

每次 provider 请求都重新构造，仅包含：

1. 中文 System Prompt 与 LIBERO-PRO task prompt；
2. 当前一张 `1440×1080` Context PNG；
3. 当前 minimal manifest；
4. 最近最多三个完整的 Function call/result transaction。

旧图片、旧 decision basis、receipt ID、renderer/schema 元数据、reward 和环境成功真值不会
进入下一轮策略上下文。Depth、相机参数、raw mask/cloud 和 backend 也只存在于 episode-local
private context。

## Function space

当前 Agent-visible Function 固定为 11 个：

```text
inspect(query, within_region_id?)
locate_point(query, within_region_id?)
propose_grasps(region_id)
propose_pose(point_id, offset_xyz, quaternion_xyzw?)
select(candidate_id)
delta_move(delta_xyz_m, frame?, action_id?)
rotate(axis, angle_deg, frame?, action_id?)
commit(action_id)
open_gripper()
close_gripper()
done(success)
```

`inspect`、`locate_point`、`propose_grasps` 产生当前 revision 的证据；`select`、
`propose_pose`、`delta_move`、`rotate` 只创建或编辑 Action Proposal；`commit` 与两个
gripper Function 才改变物理世界。物理动作刷新 observation，并使旧 revision 的
region/point/candidate/action 引用失效。

## 当前代码结构

```text
vaw/
  context_runtime/
    model.py              # evidence、candidate、proposal、receipt 与 ContextState
    workspace.py          # 私有 episode context、revision 生命周期与 Function dispatch
    functions.py          # 11 个 Function handler
    protocol.py           # Function schema、解析与中文 System Prompt
    history.py            # protocol-safe K=3 transaction window
    packet.py             # trusted state → policy-visible ContextPacket
    web_renderer.py       # ContextPacket → 固定 Web screenshot
    browser_renderer.py   # 持久 Playwright/Chromium bridge
    runtime.py            # 单 VLM loop
    trace.py / video.py   # trace 与视频产物
    motion.py             # Pyroki/CuRobo proposal prediction 与执行边界
    geometry.py           # frame、四元数、投影与 TCP 几何
    gripper_mesh.py       # URDF/FK robot imagination raster
    near_field.py         # 双相机 RGB-D 的 TCP 近场几何 raster
  agents/
    contracts.py          # provider/runtime 公共数据结构
    providers/            # OpenAI-compatible native/text tool-call transport
    teacher.py/student.py # provider 配置
  scripts/
    run_context_agent.py  # 真实 LIBERO-PRO agent/scripted runner
    check_gripper_overlay.py
```

`vaw-ui/` 只保留 schemaVersion 4 的 Context 页面。Persistent World 中 raw wrist RGB
已替换为当前双相机 RGB-D 融合的 gripper-local 双视图；raw wrist 仍只进入 trace 视频。
旧 `Workspace`、PIL renderer、
schema-v1 Web 页面、14-op protocol 和旧 runner 已从当前代码删除；删除前状态保存在 Git
checkpoint `fd8d89a`，不会与当前运行路径并存。

## 构建 Web renderer

```bash
cd /mnt/data/zyh/BCap-X/vaw-ui
npm ci
npm run build
```

一次性安装浏览器（若环境尚未安装）：

```bash
source /mnt/data/zyh/BCap-X/.venv-libero/bin/activate
python -m pip install "playwright>=1.50,<2"
python -m playwright install chromium
```

## 真实 LIBERO-PRO 运行

服务启动后：

```bash
cd /mnt/data/zyh/BCap-X
source .venv-libero/bin/activate

python -m vaw.scripts.run_context_agent \
  --mode agent \
  --suite libero_object_swap \
  --task-id 0 \
  --model vapi/qwen3.5-plus \
  --protocol text \
  --motion-backend curobo \
  --record-video
```

真实接线 smoke（脚本不属于 Agent policy）：

```bash
python -m vaw.scripts.run_context_agent \
  --mode scripted \
  --suite libero_object_swap \
  --task-id 0 \
  --motion-backend pyroki
```

默认 trace 写入 `vaw/out/context_runs/`，包含当前 Context PNG、结构化 trace、meta 和
可用时导出的 agentview/wrist/context 视频。

## 回归测试

```bash
cd /mnt/data/zyh/BCap-X
source .venv-libero/bin/activate

python -m pytest -q \
  tests/test_vaw_context_agent.py \
  tests/test_vaw_context_packet.py \
  tests/test_vaw_context_runtime.py \
  tests/test_vaw_gripper_fk.py
```

当前代码不会修改 CaP-X、RoboMEx、`capx_skill_rl` 或其他项目路径。
