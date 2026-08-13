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
