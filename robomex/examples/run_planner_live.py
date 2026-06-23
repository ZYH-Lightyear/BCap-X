"""在真实 CapX/LIBERO(-PRO) 任务上跑两层 RoboMEx agent(真机)。

外层 ReactivePlanner 每步给出**下一个** sub-goal(任务 + 当前场景图 + 高层技能菜单 +
历史);内层 CodeAsPolicyAgent 在真实 env 上执行它(并被告知先咨询哪个高层技能),
随后刷新场景、再次询问 planner,直到它说 DONE。每个 sub-goal 结束后由独立的
VerifyCodeAgent(复用同一沙箱 + query_vlm)做子目标级视觉裁决。目标是在真实
LIBERO-PRO 任务上端到端验证反应式两层接线。

前置依赖(与 baseline 用的是同一批服务):
    - LLM 代理                          :8110
    - sam3 / contact-graspnet / pyroki :8114 / :8115 / :8116

用法::

    uv run --no-sync --active robomex/examples/run_planner_live.py \\
        --config-path env_configs/libero/franka_libero_cap_agent0.yaml \\
        --model openrouter/qwen/qwen3.6-plus
"""

from __future__ import annotations

import os

# MuJoCo 必须在 import 仿真之前选好 GL 后端。这里照搬 launch.py;在没有 EGL 的无头
# 机器上想用 CPU 渲染,可用 MUJOCO_GL=osmesa 覆盖。
os.environ.setdefault("MUJOCO_GL", "egl")

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

_CODE_AS_POLICY_PROMPT = (
    "You are a robot Code-as-Policy agent controlling a Franka arm in LIBERO. "
    "You are given ONE sub-goal of a larger task. Each turn, write ONE block of executable "
    "Python that advances the sub-goal, grounding every decision in the current observation "
    "via get_observation(). Skill guidance below is advisory: adapt it, do not copy it blindly. "
    "All API functions listed below are already imported into the namespace. "
    "Reply with a single ```python``` code block, or the word FINISH when the sub-goal is complete."
)


@dataclass
class LiveArgs:
    """单次真机两层试验的 CLI 参数。"""

    config_path: str = "env_configs/libero/franka_libero_cap_agent0.yaml"
    """YAML env 配置(与 CaP-Agent0 baseline 用的同一个)。"""

    model: str = "openrouter/qwen/qwen3.6-plus"
    """planner 和内层 code agent 共用的模型(经代理转发)。"""

    server_url: str = "http://localhost:8110/chat/completions"
    """本地 LLM 代理端点。"""

    api_key: str | None = None
    """可选 API key(通常由代理注入)。"""

    max_turns: int = 6
    """每个 sub-goal 内层最多的代码生成轮数。"""

    output_dir: str = "./outputs/robomex_planner_live"
    """本次运行产物的根目录;会在其下创建一个时间戳子目录。"""

    seed: int | None = None
    """可选的 env reset 种子。"""


def _build_env(config_path: str) -> Any:
    """像 trial worker 那样实例化高层 CapX env。"""

    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader

    configs_dict = DictLoader.load([os.path.expanduser(config_path)])
    if "env" not in configs_dict:
        raise ValueError(f"config {config_path} has no 'env' key")
    return instantiate(configs_dict["env"])


def _task_language(env: Any) -> str:
    """LIBERO 任务指令字符串,用作 planner 的任务目标。"""

    handle = getattr(env.low_level_env, "handle", None)
    lang = getattr(handle, "task_language", None) if handle is not None else None
    return lang or "complete the manipulation task"


def _api_docs(env: Any) -> str:
    """拼接内层 code agent 需要的 API 文档。"""

    apis = getattr(env, "_apis", {})
    return "\n".join(api.combined_doc() for api in apis.values())


def _save_scene_image(obs: dict, path: str) -> str | None:
    """保存 agentview RGB 让 planner 能看到场景;取不到则返回 None。"""

    from robomex.perception.render import save_rgb

    try:
        rgb = obs["agentview"]["images"]["rgb"]
    except (KeyError, TypeError):
        return None
    try:
        return save_rgb(path, rgb)
    except Exception:
        return None


def _flush_video(env: Any, target_dir: Path, suffix: str) -> None:
    """把**自上次取帧以来**累计的帧取走(clear=True)并立即写成 mp4。

    `enable_video_capture` 后 env 持续往帧缓冲里录;每个 sub-goal 结束时调用一次,
    取到的就正好是这段 sub-goal 的帧,写完缓冲清空、下段重新累计。多相机则各写一路。
    """

    from robomex.core.logging import get_logger

    log = get_logger("live")
    if not hasattr(env, "get_video_frames"):
        return
    try:
        frames = env.get_video_frames(clear=True)
    except Exception as exc:  # noqa: BLE001 - 录像失败不该让整个 run 挂掉
        log.warning("取视频帧失败: %r", exc)
        return
    if not frames:
        return

    from capx.utils.video_utils import _write_video

    if isinstance(frames, dict):
        for cam, cam_frames in frames.items():
            if cam_frames:
                _write_video(cam_frames, str(target_dir), suffix=f"{suffix}_{cam}")
    else:
        _write_video(frames, str(target_dir), suffix=suffix)


def main(args: LiveArgs) -> None:
    from robomex import RoboMExAgent, RoboMExConfig
    from robomex.agents import LLMPlannerPolicy
    from robomex.core.coder import LLMCodePolicy
    from robomex.core.logging import configure_logging
    from robomex.core.sandbox import CapXExecutorAdapter
    from robomex.skills import SkillLibrary, load_builtin_skills

    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 控制台 + run.log 双写;之后所有 robomex.* 的日志都会进这里。
    log = configure_logging(log_file=out_dir / "run.log")
    log.info("构建 env: %s (MUJOCO_GL=%s)", args.config_path, os.environ.get("MUJOCO_GL"))

    env = _build_env(args.config_path)
    obs, _info = env.reset(seed=args.seed)

    # 开启整段 episode 的视频录制(env 支持时);run 结束后写盘。
    if hasattr(env, "enable_video_capture"):
        try:
            env.enable_video_capture(True, clear=True)
            log.info("已开启视频录制")
        except Exception as exc:  # noqa: BLE001 - 录像不可用不该阻断 run
            log.warning("开启视频录制失败: %r", exc)

    task = _task_language(env)
    log.info("任务: %s", task)

    scene_path = _save_scene_image(obs, str(out_dir / "scene.png"))
    log.info("初始场景图: %s", scene_path or "(取不到 -> planner 仅凭文本规划)")

    library = SkillLibrary(str(out_dir / "library"))
    for skill in load_builtin_skills():
        library.admit(skill, source="builtin")

    system_prompt = (
        f"{_CODE_AS_POLICY_PROMPT}\n\nAvailable API functions (already imported):\n{_api_docs(env)}"
    )

    # 框架入口:把所有依赖收进一个 RoboMExConfig,再交给 RoboMExAgent 装配 + 运行
    # 反应式两层循环。每个 sub-goal 结束后跑独立的 VerifyCodeAgent 做视觉裁决。
    config = RoboMExConfig(
        library=library,
        planner_policy=LLMPlannerPolicy(model=args.model, server_url=args.server_url, api_key=args.api_key),
        code_policy=LLMCodePolicy(model=args.model, server_url=args.server_url, api_key=args.api_key),
        executor=CapXExecutorAdapter(env),
        enable_verification=True,
        max_turns=args.max_turns,
        max_subgoals=8,
        inner_system_prompt=system_prompt,
        observation_summary=(
            "A LIBERO tabletop scene. Call get_observation() for agentview/wrist RGB, "
            "depth, intrinsics, and camera pose."
        ),
        artifacts_dir=str(out_dir),
    )
    agent = RoboMExAgent(config)

    # 真机场景刷新:每个 sub-goal 之后,用最新观测重渲染 planner 看到的场景
    #(与具体 env 相关;离线时保持 None)。
    step = {"n": 0}

    def scene_refresh(observation: dict) -> str | None:
        step["n"] += 1
        return _save_scene_image(observation, str(out_dir / f"scene_step{step['n']}.png"))

    # 每个 sub-goal 跑完的当下,就把这段视频写进它自己的 subgoal_NN/(不等整段结束)。
    def on_subgoal_end(index: int, result: Any, sg_dir: Path | None) -> None:
        if sg_dir is not None:
            _flush_video(env, sg_dir, suffix="subgoal")

    agent.run(
        task,
        scene_image_path=scene_path,
        scene_refresh=scene_refresh,
        on_subgoal_end=on_subgoal_end,
    )

    # 收尾:把最后一段 sub-goal 之后的残余帧(若有)落到 episode 根目录。
    _flush_video(env, out_dir, suffix="tail")
    log.info("本次运行产物目录: %s", out_dir)
    log.info("  ├─ run.log          完整日志")
    log.info("  ├─ planner.jsonl    每步 planner 原始回复 + 决策")
    log.info("  ├─ summary.json     episode 汇总")
    log.info("  ├─ subgoal_NN/      内层代码 turn_*.py + 输出 turn_*.out.txt + 逐块过程视频 turn_*.mp4 + 整段 video_subgoal*.mp4")
    log.info("  └─ scene*.png       每步 planner 看到的场景图")


if __name__ == "__main__":
    main(tyro.cli(LiveArgs))
