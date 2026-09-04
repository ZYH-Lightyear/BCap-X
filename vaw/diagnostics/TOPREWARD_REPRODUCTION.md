# TOPReward × VAW 复现记录

## 复现边界

本实现对齐官方仓库 `TOPReward/TOPReward@4877a0e` 的 Qwen 路径：

- 使用冻结的 `Qwen/Qwen3-VL-8B-Instruct`；
- 输入为真实 AgentView 视频前缀和原始 Task；
- 每个前缀均匀采样 15 帧；
- 使用官方真值判断提示；
- 只读取提示末尾 `True` token 的 log-probability；
- episode 内 min-max 仅用于单条视频显示；失败轨迹再使用 15%→5% 的线性惩罚，
  不作为训练或跨任务比较的 reward。

设归一化进度为 `p∈[0,1]`。仅当 `env_success=false` 时，显示曲线采用：

```text
penalty(p) = 0.15 - 0.10p
display(p) = p × (1 - penalty(p))
```

因此低进度处惩罚率为 15%，进度越高惩罚越低，失败 episode 的显示上限为 95%。
`env_success=true` 或缺少终局标签时不做该校准；`raw_reward` 始终保持不变。

VAW 适配只新增一条因果边界：曲线节点只落在初始状态和真实物理 Action 的结束帧。
Detection、proposal、Imagination 和其他非物理 Preview 不产生进度节点，也不进入模型视频。

## Spatial 轨迹

测试轨迹：

```text
suite: libero_spatial_task
task: 0
seed: 1
instruction: Pick the akita black bowl not between the plate and the ramekin
             and place it on the plate
agent: vapi/gemini-3.7-flash
env_success: false
```

结果：

| 状态 | Turn | 物理 Action | Raw log P(True) | Raw Δ | 显示进度 |
|---:|---:|---|---:|---:|---:|
| 0 | 0 | initial | -20.625 | 0.000 | 0.0% |
| 1 | 4 | commit | -16.625 | +4.000 | 51.8% |
| 2 | 6 | commit (dispatched/unsettled) | -15.750 | +0.875 | 64.0% |
| 3 | 7 | close_gripper | -14.875 | +0.875 | 76.6% |
| 4 | 8 | delta_move +Z 3 cm | -14.250 | +0.625 | 85.7% |
| 5 | 13 | commit | -14.375 | -0.125 | 83.9% |
| 6 | 15 | commit | -13.625 | +0.750 | 95.0% |
| 7 | 16 | open_gripper | -14.000 | -0.375 | 89.4% |
| 8 | 17 | delta_move +Z 3 cm | -14.625 | -0.625 | 80.2% |

该曲线能识别若干局部变化：抓取接近、闭合和抬升阶段总体上升；运输中的一次动作轻微回落；
释放和末次抬升后连续下降。与此同时，轨迹最终环境判定失败，但归一化进度仍处于高位。这说明
TOPReward 可作为 Trace Investigator 的弱导航信号，不能单独充当成功真值、技能写入条件
或 generation promotion gate。

## 产物

真实输出位于：

```text
vaw/out/topreward_eval/spatial_task0_s1_qwen3vl8b_official15/
├── progress.json
├── progress.jsonl
├── states/*/response.json
└── topreward_action_curve.mp4
```

视频为 `1920×1080 @ 30 FPS`，时长 `8.7 s`。左侧保持真实 AgentView，右侧显示当前 Action、
原始 token log-probability 与显示进度，底部曲线只在对应物理动作结束后揭示新节点。

## 运行命令

```bash
cd /mnt/data/zyh/BCap-X
MPLCONFIGDIR=/tmp/mplconfig \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_VERBOSITY=error \
.venv-topreward/bin/python -m vaw.diagnostics.run_topreward \
  --trace vaw/out/sweeps/full_libero_pro_gemini37flash_seed1_20260829_005724/libero_spatial_task/task0_s1 \
  --output-dir vaw/out/topreward_eval/spatial_task0_s1_qwen3vl8b_official15 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --device cuda:0 \
  --local-files-only \
  --render-video
```

## 当前结论

8B 是最合适的首个忠实复现基线：它与官方默认模型一致，单卡可运行，也便于先验证方法语义。
后续可以把相同接口切换到更大的 Qwen3-VL，但必须保留原始 8B 曲线作为 controlled baseline，
并分别报告模型、采样帧数和 raw log-probability，不能混合归一化结果。
