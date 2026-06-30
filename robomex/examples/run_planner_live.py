"""在真实 CapX/LIBERO(-PRO) 任务上跑两层 RoboMEx agent(真机)。

外层 ReactivePlanner 每步参考高层技能指导给出**下一个自然语言** sub-goal(任务 +
当前场景图 + 历史);内层 CodeAsPolicyAgent 在真实 env 上自主选择并组合技能执行它。
Act 调用 finish 后刷新场景、再次询问 planner,直到它说 DONE。目标是在真实
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
    "You are given ONE sub-goal of a larger task. Each turn, reply with exactly one JSON "
    "action object. Use run_python to execute one block of Python that advances the sub-goal, "
    "grounding every decision in the current observation "
    "via get_observation(). Skill guidance below is advisory: adapt it, do not copy it blindly. "
    "The CORE API primitives listed below (sensing, motion, geometry, query_vlm) are always "
    "imported and available. Higher-level capabilities -- segmentation, grasp planning, "
    "placement -- are NOT listed here; they are provided through skills: consult the skill menu "
    "with `{\"tool\":\"use_skill\",\"args\":{\"name\":\"<skill>\"}}` to load each skill's recipe "
    "and the exact APIs it uses. "
    "`query_vlm` is for visual QA, state classification, and sanity checks only; do not use it "
    "to produce object coordinates, boxes, or points. Use the perception skills that call "
    "`vlm_bbox_detection` / `vlm_point_detection` for spatial grounding. "
    "Use `{\"tool\":\"run_python\",\"args\":{\"code\":\"...\", \"intent\":\"...\"}}` for code and "
    "`{\"tool\":\"finish\",\"args\":{\"claim\":\"...\"}}` when this sub-goal attempt is complete "
    "and control should return to the "
    "Planner."
)

# Tier-0 API:写任何 manipulation 代码都绕不开的“词汇”——传感 + 运动原语 + 通用 VLM 问答工具 +
# 纯几何/坐标变换。它们与具体技能策略无关、数量小且稳定,故**常驻**内层 system prompt。
# 其余注册的 API 一律视作 Tier-1“能力”(分割 / 抓取规划 / 点云处理 / molmo 等):函数仍被
    # CapX import 进沙箱、随时可调,但**文档不进基座 prompt**,改由相关 Skill 在被 use_skill 加载
# 时给出示例与签名。好处:技能成为这些能力的唯一策划入口,既省上下文,又避免 agent 绕过技能
# 直接乱用底层能力(例如绕开 segment_object 直接 segment_sam3_text_prompt)。新 API 默认 Tier-1
# (隐藏),要常驻就显式加进下面这张名单。
_TIER0_APIS: frozenset[str] = frozenset({
    # 传感
    "get_observation",
    # 运动原语
    "goto_pose", "open_gripper", "close_gripper", "move_to_joints",
    "goto_home_joint_position", "solve_ik",
    # 通用视觉问答 / 状态判断工具；空间 grounding 应通过专用 detection API 的技能完成。
    "query_vlm",
    # 纯几何 / 坐标变换(无模型、无策略,处处要用)
    "decompose_transform", "rotation_matrix_to_quaternion", "transform_points",
    "pixel_to_world_point", "mask_to_world_points", "depth_to_point_cloud",
    "normalize_vector",
})


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
    """每个 sub-goal 内层最多执行多少个 python action block; use_skill 不计入。"""

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


def _api_docs(env: Any, allow: frozenset[str] = _TIER0_APIS) -> str:
    """只把 **Tier-0** API 的文档拼进内层 system prompt。

    复刻 :meth:`capx.integrations.base_api.ApiBase.combined_doc` 的格式
    (``name(signature)`` + 缩进的 docstring),但按 ``allow`` 名单逐函数过滤:Tier-1
    能力级 API 不在此出现(它们仍被 CapX import 进沙箱、可调用,文档由对应 Skill 给出)。
    遍历 ``env._apis`` 的所有 API 组,按名字判定 tier。
    """

    import inspect

    lines: list[str] = []
    for api in getattr(env, "_apis", {}).values():
        try:
            fns = api.functions()
        except Exception:  # noqa: BLE001 - 某个 API 组取函数失败不该毁掉整段文档
            continue
        for name, fn in fns.items():
            if name not in allow:
                continue
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = "(…)"
            doc = inspect.getdoc(fn) or ""
            lines.append(f"{name}{sig}")
            if doc:
                lines.append("  Doc:")
                lines.extend(f"    {ln}" for ln in doc.splitlines())
            lines.append("")
    return "\n".join(lines).strip()


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


def _concat_episode_video(out_dir: Path, log: Any) -> None:
    """把各 sub-goal 的 ``video_subgoal*.mp4``(+ 收尾 ``video_tail*.mp4``)按时序拼成一段
    完整 episode 视频 ``video_full*.mp4``,落在 episode 根目录。

    逐 sub-goal 的分段视频仍各自保留;这里只是额外读回它们的帧、按 subgoal 序号(tail 最后)
    拼接重写。多相机各拼一路(``video_full_<cam>.mp4``)。读不到/无片段则静默跳过。
    """

    import re

    import numpy as np

    from robomex.perception.render import save_video

    seg_re = re.compile(r"video_(?:subgoal|tail)(?:_(?P<cam>.+))?\.mp4$")

    # 收集 (排序键, 相机, 路径):sub-goal 段按 NN 排,tail 段排最后。
    segments: list[tuple[int, str, Path]] = []
    for sg in out_dir.glob("subgoal_*"):
        if not sg.is_dir():
            continue
        try:
            order = int(sg.name.split("_")[1])
        except (IndexError, ValueError):
            order = 9998
        for mp4 in sg.glob("video_subgoal*.mp4"):
            m = seg_re.match(mp4.name)
            segments.append((order, (m.group("cam") or "") if m else "", mp4))
    for mp4 in out_dir.glob("video_tail*.mp4"):
        m = seg_re.match(mp4.name)
        segments.append((9999, (m.group("cam") or "") if m else "", mp4))

    if not segments:
        return

    import imageio

    by_cam: dict[str, list[tuple[int, Path]]] = {}
    for order, cam, path in segments:
        by_cam.setdefault(cam, []).append((order, path))

    for cam, items in by_cam.items():
        items.sort(key=lambda x: (x[0], str(x[1])))
        frames: list[np.ndarray] = []
        for _, path in items:
            try:
                reader = imageio.get_reader(str(path), format="FFMPEG")
                for fr in reader.iter_data():
                    frames.append(np.asarray(fr))
                reader.close()
            except Exception as exc:  # noqa: BLE001 - 单段读失败不该毁掉整段拼接
                log.warning("拼接读取失败 %s: %r", path, exc)
        if not frames:
            continue
        name = "video_full.mp4" if not cam else f"video_full_{cam}.mp4"
        save_video(out_dir / name, frames, fps=30)
        log.info("完整 episode 视频: %s(%d 帧 / %d 段)", out_dir / name, len(frames), len(items))


def run_episode(env: Any, obs: dict, task: str, out_dir: Path, args: LiveArgs, log: Any) -> Any:
    """在**已 reset 的** ``env`` 上跑一整段 RoboMEx episode,产物落到 ``out_dir``。

    单跑入口(``main``)和批量入口(``run_planner_batch``)共用这段逻辑:装配
    ``RoboMExConfig`` + ``RoboMExAgent``,接好真机场景刷新与逐 sub-goal 视频落盘,
    跑完返回 :class:`EpisodeResult`(含 ``execution``,可由此算 env 客观判据)。
    """

    from robomex import RoboMExAgent, RoboMExConfig
    from robomex.agents import LLMPlannerPolicy
    from robomex.core.coder import LLMCodePolicy
    from robomex.core.sandbox import CapXExecutorAdapter
    from robomex.skills import SkillLibrary, load_builtin_skills

    out_dir.mkdir(parents=True, exist_ok=True)

    # Code-block APIs such as query_vlm read their VLM backend from environment
    # variables. By default, keep them on the same proxy route as the planner/act
    # model so ``--model vapi/...`` does not silently fall back to OpenRouter.
    os.environ.setdefault("CAPX_VLM_MODEL", args.model)
    os.environ.setdefault("CAPX_VLM_SERVER_URL", args.server_url)
    if args.api_key:
        os.environ.setdefault("CAPX_VLM_API_KEY", args.api_key)

    # 开启整段 episode 的视频录制(env 支持时);逐 sub-goal / 收尾时写盘。
    if hasattr(env, "enable_video_capture"):
        try:
            env.enable_video_capture(True, clear=True)
            log.info("已开启视频录制")
        except Exception as exc:  # noqa: BLE001 - 录像不可用不该阻断 run
            log.warning("开启视频录制失败: %r", exc)

    scene_path = _save_scene_image(obs, str(out_dir / "scene.png"))
    log.info("初始场景图: %s", scene_path or "(取不到 -> planner 仅凭文本规划)")

    library = SkillLibrary(str(out_dir / "library"))
    for skill in load_builtin_skills():
        library.admit(skill, source="builtin")

    system_prompt = (
        f"{_CODE_AS_POLICY_PROMPT}\n\nAvailable API functions (already imported):\n{_api_docs(env)}"
    )

    # 框架入口:把所有依赖收进一个 RoboMExConfig,再交给 RoboMExAgent 装配 + 运行
    # 反应式两层循环。执行器在 inner loop 内 use_skill/run_python;finish 交回 Planner。
    config = RoboMExConfig(
        library=library,
        planner_policy=LLMPlannerPolicy(model=args.model, server_url=args.server_url, api_key=args.api_key),
        code_policy=LLMCodePolicy(model=args.model, server_url=args.server_url, api_key=args.api_key),
        executor=CapXExecutorAdapter(env),
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

    result = agent.run(
        task,
        scene_image_path=scene_path,
        scene_refresh=scene_refresh,
        on_subgoal_end=on_subgoal_end,
    )

    # 收尾:把最后一段 sub-goal 之后的残余帧(若有)落到 episode 根目录。
    _flush_video(env, out_dir, suffix="tail")
    # 再把所有分段拼成一段完整 episode 视频(分段视频仍保留)。
    _concat_episode_video(out_dir, log)
    return result


def main(args: LiveArgs) -> None:
    from robomex.core.logging import configure_logging

    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 控制台 + run.log 双写;之后所有 robomex.* 的日志都会进这里。
    log = configure_logging(log_file=out_dir / "run.log")
    log.info("构建 env: %s (MUJOCO_GL=%s)", args.config_path, os.environ.get("MUJOCO_GL"))

    env = _build_env(args.config_path)
    obs, _info = env.reset(seed=args.seed)

    task = _task_language(env)
    log.info("任务: %s", task)

    run_episode(env, obs, task, out_dir, args, log)

    log.info("本次运行产物目录: %s", out_dir)
    log.info("  ├─ run.log          完整日志")
    log.info("  ├─ planner.jsonl    每步 planner 原始回复 + 决策")
    log.info("  ├─ summary.json     episode 汇总")
    log.info("  ├─ subgoal_NN/      内层代码 turn_*.py + 输出 turn_*.out.txt + 逐块过程视频 turn_*.mp4 + 整段 video_subgoal*.mp4")
    log.info("  └─ scene*.png       每步 planner 看到的场景图")


if __name__ == "__main__":
    main(tyro.cli(LiveArgs))
