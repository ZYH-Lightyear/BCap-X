# VAW Static VLM Diagnostic

> **冻结诊断资产。** 题目与图片来自旧 Canvas/schema，用于复现当时的视觉理解结果，不代表当前
> `vaw-context-v31-separated-control-guides` 输入。不要在更新 Runtime 时重写题目或覆盖图片；
> 当前架构见 [`../CURRENT_ARCHITECTURE.md`](../CURRENT_ARCHITECTURE.md)。

这是一个与 Agent Loop 隔离的小型静态诊断集，用于回答一个更窄的问题：模型能否仅凭
当前 VAW Canvas 正确理解抓持状态、闭合条件、preview/observed 归因和局部位姿微调。

诊断请求中只有一张当前图片和一道固定选择题，不包含 Agent System Prompt、Task History、
manifest、Function List 或环境真值。模型必须返回：

```json
{"choice":"B","visual_evidence":"蓝黄罐仍留在桌面，而夹爪已经上移","confidence":"high"}
```

## 数据集

`static_vlm_cases.json` 当前冻结 10 题：

- 4 题抓持状态与随动证据；
- 2 题是否适合直接闭合；
- 1 题是否应 `rotate + delta_move`；
- 2 题 observed/preview 与 BASE +Z 语义；
- 1 题命令执行、规划成功与物理效果的证据边界。

每张图片都记录 SHA-256。若原 trace 被重新运行覆盖，runner 会拒绝执行，避免同一个 case ID
悄悄变成另一道题。图片仍留在原 trace 目录，不复制 raw depth、点云或其他私有状态。

## 运行

先验证题目、图片和模型矩阵，不发送图片：

```bash
python -m vaw.diagnostics.run_static_vlm --dry-run
```

通过本地 VAPI proxy 同时运行 GPT5.5 和 Qwen3.5-Plus：

```bash
python -m vaw.diagnostics.run_static_vlm \
  --models vapi/gpt-5.5 vapi/qwen3.5-plus \
  --server-url http://127.0.0.1:8110/chat/completions \
  --temperature 0 \
  --max-tokens 512 \
  --allow-image-egress
```

如需测试稳定性，可增加 `--repeats 3`；如需测试 VIA 更接近的高推理设置，可单独运行支持
该参数的模型并增加 `--reasoning-effort xhigh`，不要把不同 reasoning setting 混进同一基线。

默认结果写入 `vaw/out/static_vlm_diagnostics/<timestamp>/`：

- `run_config.json`：精确模型、prompt 和数据集 hash；
- `results.jsonl`：每题原始回答、provider reasoning、解析结果、延迟和 token usage；
- `summary.json`：按模型和题型聚合；
- `summary.md`：便于快速阅读的对照表。

`--allow-image-egress` 是显式确认：这些 trace PNG 会被发送到 `--server-url` 指定的服务。

## Absolute Progress Critic Demo

`run_progress_demo` 对每个真实物理动作结束后的状态独立估计任务完成进度，再由连续状态计算：

```text
delta_t = progress_t - progress_(t-1)
accel_t = delta_t - delta_(t-1)
```

Critic 只看到 Task、初始 Canvas、动作前后 Canvas、动作起始/中间/结束帧以及当前函数结果；
不会收到未来步骤、`env_success`、reward 或 Agent 的成功声明。下面的命令使用本机 VLM，保存
逐状态请求、响应、token logprobs、曲线数据和同步 H.264 视频：

```bash
python -m vaw.diagnostics.run_progress_demo \
  --trace /absolute/path/to/task8_s1 \
  --server-url http://127.0.0.1:5088/v1/chat/completions \
  --model /mnt/nas/maqi/Gemma_4_31B \
  --render-video
```

默认输出到 `<trace>/progress_critic/absolute_progress_<timestamp>/`。中断后可用原输出目录加
`--resume`，已经完成的 Critic 状态不会重新请求。`progress_overlay.mp4` 使用 action manifest
中的精确全局帧号同步 AgentView；新的进度点只在对应 AFTER 证据出现后显示。

## TOPReward Reproduction

`run_topreward` 复现官方 Qwen3-VL 打分路径：冻结模型，将“视频前缀 + Task”组织成真值判断，
只读取提示末尾 `True` token 的 log-probability。默认模型为官方使用的
`Qwen/Qwen3-VL-8B-Instruct`，每个前缀均匀采样 15 帧；模型不会看到 VAW Preview、
Imagination Canvas、reward 或环境成功真值。

先安装独立依赖。推荐单独环境，避免改变 LIBERO-PRO Runtime 的 Transformers 版本：

```bash
python -m venv --system-site-packages .venv-topreward
.venv-topreward/bin/python -m pip install \
  "transformers==4.57.1" "accelerate>=1.0" "qwen-vl-utils==0.0.14"
```

对已有 VAW trace 运行并生成 Action 对齐视频：

```bash
MPLCONFIGDIR=/tmp/mplconfig \
.venv-topreward/bin/python -m vaw.diagnostics.run_topreward \
  --trace /absolute/path/to/libero_spatial_task/task0_s1 \
  --output-dir /absolute/path/to/topreward_result \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --device cuda:0 \
  --local-files-only \
  --render-video
```

输出包括逐状态 `states/*/response.json`、`progress.json[l]` 和
`topreward_action_curve.mp4`。曲线节点严格为初始状态与真实物理动作结束边界；非物理
Function call 不产生伪进度节点。`raw_reward` 是比较与学习应使用的原始 log-probability；
`relative_progress` 是官方 episode 内 min-max 形式。`progress` 在终局失败时再做线性显示校准：
惩罚率从低进度的 15% 下降到高进度的 5%，所以失败轨迹最多显示 95%，但仍保留局部趋势。
二者都只用于单条视频诊断，不能跨 episode 比较；原始学习证据始终是 `raw_reward`。
