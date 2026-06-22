# CaP-X 架构总览：API 分层 / Sandbox 设计 / CaP-Agent0 设计

> 本文把 CaP-X 框架里三块核心设计整理到一处，并在最后给出运行一个 LIBERO 任务所需的**最小文件目录**。
>
> 阅读顺序建议：先看「整体数据流」建立全局直觉，再分别读 API 分层、Sandbox、CaP-Agent0 三节，最后对照最小目录定位代码。

---

## 0. 整体数据流（一图看懂）

```
YAML config (env_configs/libero/*.yaml)
        │  _load_config / instantiate
        ▼
launch.py ──► runner.py ──► trial.py(_run_single_trial)   ← CaP-Agent0 主循环
                                  │
        ┌─────────────────────────┼──────────────────────────┐
        │ 1. 生成代码 (LLM/ensemble) │ 2. 多轮反馈 (visual diff)  │
        ▼                         ▼                          ▼
   capx/llm/client.py        VLM 描述/差分              FINISH / REGENERATE 决策
        │                                                     
        ▼  action = Python 代码字符串
CodeExecutionEnvBase.step()  ← Sandbox 层 (capx/envs/tasks/base.py)
        │  exec(code, globals)   把 API 函数注入命名空间
        ▼
APIs (capx/integrations/franka/libero*.py)  ← API 分层
        │  调用感知/运动服务
        ▼
FrankaLiberoEnv (capx/envs/simulators/libero.py)  ← 低层仿真
        │
        ▼
LIBERO-PRO 仿真器 (MuJoCo) + 感知/IK 服务 (SAM3 / GraspNet / PyRoKi)
```

三层职责清晰分离：
- **API 层**：定义 LLM 能调用的"工具"（函数 + docstring）。
- **Sandbox 层**：把 LLM 生成的 Python 代码安全地 `exec`，并捕获 stdout/stderr/reward。
- **CaP-Agent0 层**：编排"生成代码 → 执行 → 看反馈 → 再生成"的多轮闭环。

---

## 1. API 分层设计

API 是"模型生成的代码能调用的工具集"。每个 API 是一组带 docstring 的 Python 函数，**docstring 本身就是模型看到的接口文档**。

### 1.1 基类与注册机制 — `capx/integrations/base_api.py`

所有 API 继承 `ApiBase`：

```13:43:capx/integrations/base_api.py
class ApiBase(ABC):
    """Base class for tool APIs.

    Guidelines:
    - Expose public functions via `functions()`.
    - Each function should have a Google-style docstring with Args/Returns.
    - `combined_doc()` returns a standardized aggregate doc for prompts.
    - If your API needs access to the environment, implement `set_env(env)`.
    """
```

关键约定：

| 成员 | 作用 |
| --- | --- |
| `functions() -> dict[str, Callable]` | 抽象方法，返回"函数名 → 可调用对象"。**只有这里列出的函数才会暴露给 LLM。** |
| `combined_doc()` | 把所有函数的签名 + docstring 拼成标准化文本，注入到任务 prompt。 |
| `_env` | 低层仿真环境句柄（构造时注入）。 |
| `_log_step(...)` | Web UI 执行日志（无 UI 时 no-op）。 |
| `enable_webui()` | 打开/关闭执行步骤可视化。 |

注册采用工厂表 + `lru_cache`，同一 worker 内可复用 API 实例：

```127:139:capx/integrations/base_api.py
def register_api(name: str, factory: Callable[[], ApiBase]) -> None:
    _API_FACTORIES[name] = factory


@lru_cache(maxsize=256)
def get_api(name: str) -> Callable[[BaseEnv], ApiBase]:
    if name not in _API_FACTORIES:
        raise KeyError(f"API '{name}' not registered")
    return _API_FACTORIES[name]
```

注册集中在 `capx/integrations/__init__.py`，YAML 里通过名字引用：

```yaml
apis:
  - FrankaLiberoApiReducedSkillLibrary
```

### 1.2 抽象层级：从"特权"到"原语 + 技能库"

CaP-X 的核心研究维度之一就是 **abstraction level（抽象层级）**。同一个 LIBERO 任务可以挂不同抽象层级的 API，从而衡量模型在"高层指令 vs 低层原语"下的编码能力。LIBERO 提供 4 层 API：

| 抽象层级 | API 类（`capx/integrations/franka/`） | 感知来源 | 暴露的关键函数 |
| --- | --- | --- | --- |
| **L0 特权（Privileged）** | `FrankaLiberoPrivilegedApi`（`libero_privileged.py`） | 仿真 ground-truth | `get_object_pose` / `get_all_object_poses` / `sample_grasp_pose` / `goto_pose` / `open_gripper` / `close_gripper` / `goto_pose_interactive_cartesian` |
| **L1 完整感知（High-level）** | `FrankaLiberoApi`（`libero.py`） | SAM3 + GraspNet | `get_object_pose`（感知估计）/ `get_object_3d_points_and_masks_from_language` / `sample_grasp_pose` / `goto_pose` / `get_oriented_bounding_box_from_3d_points` / `goto_home_joint_position` |
| **L2 低层原语（Reduced）** | `FrankaLiberoApiReduced`（`libero_reduced.py`） | SAM3 + Molmo/Qwen 打点 + GraspNet + PyRoKi IK | `segment_sam3_text_prompt` / `segment_sam3_point_prompt` / `point_prompt_molmo` / `plan_grasp` / `plan_grasp_from_point_clouds` / `solve_ik` / `move_to_joints` / `goto_pose` / `subsample_point_cloud` / `filter_noise` |
| **L3 原语 + 技能库（Skill Library）** | `FrankaLiberoApiReducedSkillLibrary`（`libero_reduced_skill_library.py`） | 同 L2 | L2 全部 + 自动合成的复用技能：`rotation_matrix_to_quaternion` / `decompose_transform` / `depth_to_point_cloud` / `mask_to_world_points` / `pixel_to_world_point` / `transform_points` / `interpolate_segment` / `normalize_vector` / `select_top_down_grasp` |

设计要点：

- **层级越低，模型要写的代码越多**。L0 只需 `goto_pose(get_object_pose(...))`；L2/L3 需要模型自己跑分割 → 点云 → 抓取规划 → IK → 关节控制的完整流水线。
- **L3 的技能库是"自动合成"的**：`libero_reduced_skill_library.py` 顶部注释说明这些函数是从 `reduced_api` / `reduced_api_exampleless` 实验里 LLM 生成代码中挖掘出来的高频可复用函数（182 个候选 → 过滤后 73 个），再筛成稳定的坐标变换 / 感知 / 几何工具。
- `functions()` 里用注释预留了 **CuRobo 运动规划**（`plan_grasp_trajectory` / `plan_with_grasped_object` / `execute_joint_trajectory`），取消注释即可让模型使用 GPU 加速的碰撞规划。

### 1.3 docstring 即文档

`combined_doc()` 会把每个函数渲染成：

```
name(signature)
  Doc:
    <完整 docstring，含 Args/Returns/Example>
```

然后在 `CodeExecutionEnvBase._get_complete_prompt()` 拼接到任务 prompt 后面（见下一节）。因此 **写 API 时 docstring 必须完整准确**——这是模型唯一的接口说明。

---

## 2. Sandbox 设计（代码执行环境）

Sandbox 负责把"模型吐出的一段 Python 代码"安全执行，并以 Gym 接口暴露 `reset()` / `step()`。核心在 `capx/envs/tasks/base.py`。

### 2.1 两种执行器

| 执行器 | 用途 | 特点 |
| --- | --- | --- |
| `SimpleExecutor` | 极简一次性执行 | 每次 `run()` 用全新 globals，返回 `{ok, result}` 或 `{ok, error}`。 |
| `CodeExecutionEnvBase` | 正式评测用（Gym Env） | **持久化命名空间**，跨多轮代码块保留变量；捕获 stdout/stderr；计算 reward；支持行级 trace。 |

`SimpleExecutor` 注入的全局变量：

```75:87:capx/envs/tasks/base.py
    def run(self, code: str, *, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        g: dict[str, Any] = {
            "__name__": "__main__",
            "env": self._env,
            "APIS": self._apis,
            "INPUTS": inputs or {},
            "RESULT": None,
        }
        try:
            exec(code, g, g)
            return {"ok": True, "result": g.get("RESULT")}
        except BaseException as exc:  # defensive; propagate minimal info
            return {"ok": False, "error": repr(exc)}
```

### 2.2 API 函数直接注入命名空间

`CodeExecutionEnvBase` 把所有 API 的函数**按名字平铺**进执行命名空间，所以模型可以直接写 `goto_pose(...)` 而不必 `APIS["..."].goto_pose(...)`：

```233:251:capx/envs/tasks/base.py
    def _init_exec_globals(self) -> None:
        g: dict[str, Any] = {
            "__name__": "__main__",
            "env": self.low_level_env,
            "APIS": self._apis,
            "INPUTS": {},
            "RESULT": None,
        }
        # Bind helper functions from APIs into the global namespace for convenience
        for api in self._apis.values():
            for fn_name, fn in api.functions().items():
                g[fn_name] = fn
        self._exec_globals = g
```

注意：命名空间是 `self._exec_globals`，**跨 step 持久化**（`reset()` 才重建），因此模型可以在 block N 定义变量、在 block N+1 继续用。

### 2.3 stdout/stderr 捕获 + 行级 trace

执行时用 `Tee` 把输出同时写到控制台和缓冲区（既能实时看，又能存档）：

```161:217:capx/envs/tasks/base.py
    def _exec_user_code(self, code: str) -> dict[str, Any]:
        ...
        stdout_buffer = io.StringIO()
        tee_out = Tee(sys.stdout, stdout_buffer, on_write=...)
        stderr_buffer = io.StringIO()
        tee_err = Tee(sys.stderr, stderr_buffer, on_write=...)
        ok = True
        try:
            with (contextlib.redirect_stdout(tee_out),
                  contextlib.redirect_stderr(tee_err)):
                exec(compiled, self._exec_globals, self._exec_globals)
        except BaseException:
            ok = False
            traceback.print_exc(file=tee_err)   # 完整 traceback 写回 stderr
        ...
        return {"ok": ok, "stdout": ..., "stderr": ..., "result": ...}
```

- 可选的 `LineTraceRecorder`（`capx/utils/line_trace.py`）能把每行代码执行与视频帧号对应起来，供 Web UI 做"代码行 ↔ 画面"联动。

### 2.4 Gym 接口与 sandbox 返回码

- `step(action: str)`：执行代码 → `compute_reward()`（委托给低层 env）→ 返回 `(obs, reward, terminated, truncated, info)`。
- `info` 里携带 `sandbox_rc`（0=成功 / 1=异常）、`stdout`、`stderr`、`task_completed`、`task_prompt`。
- `terminated = (reward == 1.0)`；`truncated` 取决于低层仿真步数是否超 `max_steps`。

```337:344:capx/envs/tasks/base.py
        info = {
            "sandbox_rc": 0 if exec_result["ok"] else 1,
            "stdout": exec_result["stdout"],
            "stderr": exec_result["stderr"],
            "task_prompt": self._task_prompt,
            "task_completed": task_completed,
        }
        return obs, reward, bool(terminated), bool(truncated), info
```

### 2.5 超时、重试与隔离（在 runner 层）

Sandbox 本身是进程内 `exec`（不是容器隔离），健壮性靠 **runner 层**保障（`capx/envs/runner.py`）：

- `SIGALRM` 墙钟超时：`CAPX_TRIAL_TIMEOUT_SECONDS`（默认 1000s）。
- 自动重试：`CAPX_TRIAL_MAX_RETRIES`（默认 3 次），仅对超时类错误重试。
- 超时后 best-effort flush 已录制的视频帧，保留可检查的产物。
- 多 worker 并行（`num_workers`）：每个 worker 各自 `instantiate` 一份 env。

> 说明：CaP-X 的 sandbox 是"信任执行"模型——代码可 `import` 任意已装包、可直接操作 `env`。隔离粒度是 **进程级 + 超时**，不是安全沙箱。

---

## 3. CaP-Agent0 设计

CaP-Agent0 是**免训练（training-free）**的智能体框架，核心是把 LLM 编码能力包进一个"多轮视觉差分 + 并行集成推理 + 技能库"的闭环。逻辑主体在 `capx/envs/trial.py::_run_single_trial`，配置开关在 `capx/envs/launch.py`。

### 3.1 单 trial 主循环（5 步）

```829:845:capx/envs/trial.py
def _run_single_trial(...):
    """Execute a single trial end-to-end.

    Steps:
        1. Reset the environment.
        2. Capture initial visual feedback (if configured).
        3. Query the model for initial code generation.
        4. Execute code blocks one-by-one, with optional multi-turn regeneration.
        5. Save artifacts (code, logs, per-turn videos, combined video) and return a TrialSummary.
    """
```

执行循环：把模型生成的代码切成多个 code block，逐块 `env.step()`，每块后做一次"多轮决策"（最多 `MULTITURN_LIMIT = 10` 块）。

### 3.2 三大能力

#### (A) 多轮再生成 / 完成判定（multi-turn）

每个 code block 执行后，把 `executed_code` + `console_stdout` + `console_stderr` 填进 `multi_turn_prompt`，让模型决策：
- `REGENERATE` + 新代码 → 替换掉后续未执行的 block，`num_regenerations += 1`。
- `FINISH` → 认为任务完成，结束循环，`num_finishes += 1`。

决策解析见 `_parse_multi_turn_decision`，决策步骤见 `_handle_multi_turn_step`。

#### (B) 视觉差分（Visual Differencing）

让 VLM 把"画面变化"翻译成文字反馈喂回给编码模型，三种粒度：

| 配置开关 | 机制 | 实现函数 |
| --- | --- | --- |
| `use_visual_feedback` | 直接把当前帧图像贴进 prompt | `_capture_initial_visual_feedback` / `_get_visual_feedback` |
| `use_img_differencing` | VLM 比较"前一帧 vs 当前帧"，输出差异描述 | `_get_visual_differencing_feedback` |
| `use_video_differencing` | 把"这一轮执行的整段视频"喂给 VDM，描述发生了什么 | `_get_video_differencing_feedback` |

均支持 `use_wrist_camera` 多视角（主相机 + 腕部相机）。差分模型独立可配（`visual_differencing_model` + 其 server url）。

#### (C) 并行集成推理（Parallel Ensemble）— `capx/llm/client.py`

`use_parallel_ensemble` 打开后，先并行采样多个候选解，再用一个 synthesis 模型综合成最优解：

```75:77:capx/llm/client.py
ENSEMBLE_CONFIGS = [
    ("openrouter/qwen/qwen3.6-plus", [0.1, 0.5, 0.9]),
]
```

- `use_multimodel=False` → `query_single_model_ensemble`：同一模型不同温度多次采样后综合。
- `use_multimodel=True` → `query_model_ensemble`：多个模型（`ENSEMBLE_CONFIGS`）各自采样后综合。
- 初始代码生成与每一轮多轮决策都可走 ensemble，过程数据存进 `ensemble_data` / `multiturn_ensemble_data` 供分析。

#### (D)（可选）演进式技能库

trial 成功（`task_completed`）且开启 `evolve_skill_library` 时，从最终代码里抽取可复用技能写回技能库：

```1147:1158:capx/envs/trial.py
    if config.get("evolve_skill_library", False) and info_step.get("task_completed", False):
        try:
            from capx.skills import SkillLibrary
            skill_lib = SkillLibrary(path=config.get("skill_library_path", None))
            new_skills = skill_lib.extract_from_code(final_code, task_name=...)
            skill_lib.save()
```

> 这与 §1.2 的 L3 技能库 API 形成闭环：离线挖掘高频函数 → 固化进 `FrankaLiberoApiReducedSkillLibrary`；在线运行时还能继续演进。

### 3.3 CaP-Agent0 的典型配置

`env_configs/libero/franka_libero_cap_agent0.yaml` 是开箱即用的 Agent0 配置：

```17:18:env_configs/libero/franka_libero_cap_agent0.yaml
    apis:
      - FrankaLiberoApiReducedSkillLibrary
```
```63:71:env_configs/libero/franka_libero_cap_agent0.yaml
record_video: true
output_dir: ./outputs/franka_libero_cap_agent0
use_img_differencing: true
use_parallel_ensemble: true
use_multimodel: true

trials: 50
num_workers: 3
```

即：**L3 技能库 API + 图像差分 + 多模型并行集成 + 多轮再生成**，是 CaP-Agent0 完整形态。

### 3.4 关键配置开关速查（`LaunchArgs`）

| 开关 | 含义 |
| --- | --- |
| `model` / `server_url` | 主编码模型及其 OpenAI 兼容端点 |
| `use_visual_feedback` | 贴原始图像 |
| `use_img_differencing` / `use_video_differencing` | 图像/视频差分 |
| `use_wrist_camera` | 加入腕部相机多视角 |
| `use_parallel_ensemble` / `use_multimodel` | 并行集成 / 多模型集成 |
| `visual_differencing_model(_server_url)` | 差分用的 VLM 配置 |
| `use_oracle_code` | 跳过模型，直接跑预置 oracle 代码（调试/上界） |
| `record_video` / `output_dir` | 录像与产物目录 |
| `total_trials` / `num_workers` | 试验数与并行度（覆盖 YAML） |

---

## 4. LIBERO 任务最小目录（文件文字目录）

下面是"跑通一个 LIBERO 任务"所需的**最小文件集合**（去掉与 LIBERO 无关的 robosuite / r1pro / behavior / RL / web-ui 等）。每个文件标注了在三层架构中的职责。

```text
BCap-X/
├── env_configs/
│   └── libero/
│       ├── franka_libero_spatial_0.yaml          # [配置] 单任务示例 (L1 感知 API)
│       ├── franka_libero_spatial_0_privileged.yaml  # [配置] L0 特权版
│       └── franka_libero_cap_agent0.yaml         # [配置] CaP-Agent0 完整形态 (L3 + 差分 + 集成)
│
├── capx/
│   ├── envs/
│   │   ├── launch.py                  # [入口] CLI 解析 + 分发 (headless / web-ui)
│   │   ├── runner.py                  # [编排] 批量/并行/超时/重试
│   │   ├── trial.py                   # [CaP-Agent0] 单 trial 主循环 + 多轮 + 视觉差分
│   │   ├── base.py                    # 低层 env 抽象 BaseEnv + get_env 注册
│   │   ├── tasks/
│   │   │   ├── base.py                # [Sandbox] CodeExecutionEnvBase / SimpleExecutor / CodeExecEnvConfig
│   │   │   └── franka/
│   │   │       └── franka_libero_env.py   # [任务] FrankaLiberoCodeEnv (高层代码环境)
│   │   ├── simulators/
│   │   │   └── libero.py              # [仿真] FrankaLiberoEnv (低层 MuJoCo 控制)
│   │   ├── adapters/
│   │   │   └── libero_wrapper.py      # LIBERO-PRO benchmark 封装 (suite/task → 场景)
│   │   └── configs/
│   │       ├── loader.py              # YAML 加载 (DictLoader)
│   │       └── instantiate.py         # _target_ 递归实例化 (类 hydra)
│   │
│   ├── integrations/
│   │   ├── __init__.py                # [API 注册] register_api(...) 集中注册
│   │   ├── base_api.py                # [API 基类] ApiBase + get_api / 注册表
│   │   └── franka/
│   │       ├── common.py              # 共享运动/视觉工具 (TCP 偏移、IK 收敛、夹爪)
│   │       ├── libero_privileged.py            # [API L0] 特权 (ground-truth)
│   │       ├── libero.py                       # [API L1] 完整感知 (SAM3+GraspNet)
│   │       ├── libero_reduced.py               # [API L2] 低层原语
│   │       └── libero_reduced_skill_library.py # [API L3] 原语 + 自动合成技能库
│   │
│   ├── integrations/vision/           # [感知服务客户端]
│   │   ├── sam3.py                    # SAM3 分割 (文本/点提示)
│   │   ├── graspnet.py                # Contact-GraspNet 抓取规划
│   │   ├── point_backend.py           # 打点后端选择 (molmo / qwen)
│   │   ├── molmo.py                   # Molmo 打点
│   │   └── qwen_vlm_point.py          # Qwen VLM 打点 (CAPX_POINT_BACKEND=qwen)
│   │
│   ├── integrations/motion/           # [运动服务客户端]
│   │   ├── pyroki.py                  # PyRoKi IK 客户端
│   │   ├── pyroki_context.py          # PyRoKi 机器人上下文 (panda)
│   │   ├── pyroki_snippets/           # IK 求解片段 (solve_ik / solve_ik_vel_cost)
│   │   └── curobo_api.py              # (可选) CuRobo GPU 运动规划
│   │
│   ├── llm/
│   │   └── client.py                  # [CaP-Agent0] query_model / 并行集成 / synthesis
│   │
│   ├── serving/                       # [服务进程] YAML api_servers 自动拉起
│   │   ├── openrouter_server.py       # LLM 代理 (OpenAI 兼容, :8110)
│   │   ├── launch_sam3_server.py      # SAM3 服务 (:8114)
│   │   ├── launch_contact_graspnet_server.py  # GraspNet 服务 (:8115)
│   │   ├── launch_pyroki_server.py    # PyRoKi IK 服务 (:8116)
│   │   └── launch_servers.py          # 一键按 profile 拉起感知服务
│   │
│   ├── utils/
│   │   ├── launch_utils.py            # 配置加载 / 产物保存 / prompt 构建 / 视觉反馈
│   │   ├── line_trace.py              # [Sandbox] 行级 trace ↔ 视频帧
│   │   ├── camera_utils.py            # 相机/反投影工具
│   │   ├── depth_utils.py             # 深度图 → 点云
│   │   ├── parallel_eval.py           # 多 worker 并行
│   │   └── video_utils.py             # 视频编码/写盘
│   │
│   └── third_party/
│       └── LIBERO-PRO/                # [外部] LIBERO-PRO 基准 (assets / bddl / 任务套件)
│
└── ~/.libero/config.yaml              # [运行时] LIBERO 资产路径 (headless 需手建)
```

### 4.1 各文件在三层架构中的归属

| 层 | 关键文件 |
| --- | --- |
| **CaP-Agent0（编排）** | `envs/launch.py`、`envs/runner.py`、`envs/trial.py`、`llm/client.py` |
| **Sandbox（执行）** | `envs/tasks/base.py`、`envs/tasks/franka/franka_libero_env.py`、`utils/line_trace.py` |
| **API（工具）** | `integrations/base_api.py`、`integrations/__init__.py`、`integrations/franka/libero*.py`、`integrations/vision/*`、`integrations/motion/*` |
| **低层仿真** | `envs/simulators/libero.py`、`envs/adapters/libero_wrapper.py`、`third_party/LIBERO-PRO/` |
| **服务进程** | `serving/*` + `~/.libero/config.yaml` |

### 4.2 最小运行序（对应文件）

1. 启动 `serving/openrouter_server.py`（:8110）—— LLM 代理。
2. 启动 `serving/launch_{sam3,contact_graspnet,pyroki}_server.py`（:8114/8115/8116）—— 或由 YAML `api_servers` 自动拉起。
3. `python capx/envs/launch.py --config-path env_configs/libero/franka_libero_spatial_0.yaml --total-trials 10`
   - `launch.py` → `runner.py` → `trial.py` 主循环。
   - `trial.py` 调 `llm/client.py` 生成代码 → `tasks/base.py` sandbox 执行 → `integrations/franka/libero.py` 调感知/IK → `simulators/libero.py` 推进仿真。

> 详细命令与排错见 `docs/libero-tasks.md` 与 `docs/custom_quick_start_libero_test.md`。

---

## 5. 小结

- **API 分层**：用 `ApiBase` + 注册表把"工具"与"环境"解耦；LIBERO 提供 L0→L3 四个抽象层级（特权 / 完整感知 / 低层原语 / 原语+技能库），docstring 即模型文档。
- **Sandbox**：`CodeExecutionEnvBase` 以 Gym 接口把 LLM 代码 `exec` 到持久化命名空间，平铺注入 API 函数，捕获 stdout/stderr/reward，超时与重试由 runner 兜底。
- **CaP-Agent0**：免训练闭环 = 多轮再生成 + 视觉/视频差分 + 并行（多模型）集成 + 可选演进式技能库；一份 `franka_libero_cap_agent0.yaml` 即可启用完整形态。
