## 自定义快速启动：LIBERO 测试（本地 OpenRouter + Qwen 打点）

这份文档把“最短路径跑起一个 LIBERO trial”整理成可复制的步骤，适用于你当前的 CaP-X repo。

### 前置假设

- **工作目录**：`/mnt/users/zhangyh/cap-x`
- **Python 环境**：已经创建并可用 `./.venv-libero/`（若你用的是别的 venv，自行替换）
- **GPU**：SAM3 / Contact-GraspNet / PyRoKi 通常需要 GPU 才会跑得动
- **LLM**：通过本地 OpenRouter proxy（OpenAI-compatible）转发到 `openrouter/qwen/qwen3.6-plus`

---

## 1) 激活环境

```bash
cd /mnt/users/zhangyh/cap-x
source .venv-libero/bin/activate
```

---

## 2) 启动本地 OpenRouter proxy（8110）

项目里使用的 proxy 是 OpenAI-compatible `/chat/completions`，评测命令会用它作为 `--server-url`。

```bash
cd /mnt/users/zhangyh/cap-x
source .venv-libero/bin/activate

# 按你当前项目实际的启动方式运行（以下是常见方式）
uv run --no-sync --active capx/serving/openrouter_server.py \
  --host 127.0.0.1 \
  --port 8110
```

### 自检：端口是否可访问

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8110/docs
```

返回 `200` 说明服务起来了。

---

## 3) 启动 3 个必要 API：SAM3 / Contact-GraspNet / PyRoKi

对应配置文件 `env_configs/libero/franka_libero_cap_agent0.yaml` 里的 `api_servers`：

- **PyRoKi**：`8116`
- **Contact-GraspNet**：`8115`
- **SAM3**：`8114`

推荐一次性起全套（如果你之前已经有统一 launcher，也可以用你自己的 profile 启动）。

```bash
cd /mnt/users/zhangyh/cap-x
source .venv-libero/bin/activate

uv run --no-sync --active capx/serving/launch_pyroki_server.py \
  --host 127.0.0.1 --port 8116 --robot panda_description --target-link panda_hand

uv run --no-sync --active capx/serving/launch_contact_graspnet_server.py \
  --host 127.0.0.1 --port 8115

uv run --no-sync --active capx/serving/launch_sam3_server.py \
  --host 127.0.0.1 --port 8114 --device cuda
```

### 自检：三个服务是否都在线

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8114/docs
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8115/docs
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8116/docs
```

都返回 `200` 即正常。

---

## 4) 用 Qwen 打点后端启动 LIBERO CaP-Agent0 测试

你要跑的 YAML：

- `env_configs/libero/franka_libero_cap_agent0.yaml`

关键点：

- **CAPX_POINT_BACKEND=qwen**：切到 Qwen VLM 打点适配（不影响 Molmo 原实现）
- **CAPX_TRIAL_TIMEOUT_SECONDS**：建议调大（multimodel ensemble 很慢）
- **output-dir**：建议单独目录，便于区分不同实验

示例（跑 10 个 trial）：

```bash
cd /mnt/users/zhangyh/cap-x
source .venv-libero/bin/activate

CAPX_POINT_BACKEND=qwen \
CAPX_TRIAL_TIMEOUT_SECONDS=2400 \
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/libero/franka_libero_cap_agent0.yaml \
  --model "openrouter/qwen/qwen3.6-plus" \
  --server-url "http://127.0.0.1:8110/chat/completions" \
  --visual-differencing-model "openrouter/qwen/qwen3.6-plus" \
  --visual-differencing-model-server-url "http://127.0.0.1:8110/chat/completions" \
  --total-trials 10 \
  --output-dir ./outputs/franka_libero_cap_agent0_qwen_pointer_v2
```

---

## 5) 输出物在哪里看？

输出目录结构（示例）：

- `outputs/franka_libero_cap_agent0_qwen_pointer_v2/`
  - `trial_01_sandboxrc_.../`
    - `summary.txt`：模型生成的多轮代码（按 code block 拼接）
    - `code.py`：最终代码聚合
    - `all_responses.json`：每轮响应内容
    - `visual_feedback_XX.png`：每轮 multi-turn 的 render 快照（不等价于视频帧）
    - `video_*.mp4` / `video_combined.mp4`：若该 trial 成功结束（或最终 timeout flush）会生成

---

## 6) 常见问题与快速判断

### (A) “一直 IK fallback / 机器人不动 / visual_feedback 看起来重复”

- 这通常表示 **代码在 `goto_pose()` 之前就异常退出**（IK 失败 / grasp 失败 / 分割失败），仿真状态没推进。
- `visual_feedback_XX.png` 是每轮的快照，若状态不变会高度相似。

建议优先看：
- `trial_XX/.../summary.txt` 里最后一个 code block 的 traceback
- 终端日志里是否出现大量 `IK failed with orientation ...`

### (B) “没有视频生成”

在当前实现中，视频通常在以下时机写盘：
- trial 正常结束时（写 `video_combined.mp4` 等）
- 或最后一次 timeout summary 尝试 flush

如果 trial 反复 timeout 并重试、或根本没产生 frame buffer（机器人没动），视频可能缺失或只有极短片段。

### (C) 验证 Qwen 打点是否工作

你可以用已有脚本把点画到 `visual_feedback_*.png` 上：

```bash
cd /mnt/users/zhangyh/cap-x
source .venv-libero/bin/activate

uv run --no-sync --active tests/visualize_point_backends.py \
  --image-path outputs/franka_libero_cap_agent0_qwen_pointer_v2/trial_01_sandboxrc_1_reward_0.000_taskcompleted_0/visual_feedback_00.png \
  --objects "alphabet soup,basket" \
  --out-dir outputs/point_debug/quick_check
```

会生成 overlay PNG 和 JSON 报告，确认 Qwen 输出的 `point_2d`（0–1000）是否正确映射到像素。

---

## 7) 一键清单（最短路径）

- **终端 1**：OpenRouter proxy（8110）
- **终端 2**：SAM3（8114）
- **终端 3**：Contact-GraspNet（8115）
- **终端 4**：PyRoKi（8116）
- **终端 5**：launch 评测（CAPX_POINT_BACKEND=qwen + 长 timeout + output-dir）

