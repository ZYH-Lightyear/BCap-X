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

DEFAULT_MAX_SUBGOALS = 8
DEFAULT_SUBAGENT_MAX_TURNS = 8
DEFAULT_SCENE_CAMERA = ""

# Tier-0 API:写任何 manipulation 代码都绕不开的“词汇”——传感 + 运动原语 + 通用 VLM 问答工具 +
# 纯几何/坐标变换。它们与具体技能策略无关、数量小且稳定,故常驻 Act/SubAgent system prompt。
CORE_PROMPT_APIS: frozenset[str] = frozenset({
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

# Tier-1 capability API:分割、VLM grounding、抓取规划、点云处理等能力函数。它们也要对
# Agent 可见,避免模型靠 inspect/open 探测签名;但 prompt 会明确要求优先通过相关 Skill 的
# workflow 使用这些 API,不要把它们当作绕过 Skill 的底层捷径。
CAPABILITY_PROMPT_APIS: frozenset[str] = frozenset({
    "vlm_bbox_detection", "vlm_point_detection",
    "segment_sam3_text_prompt", "segment_sam3_point_prompt", "segment_sam3_box_prompt",
    "plan_grasp", "plan_grasp_from_point_clouds",
    "get_oriented_bounding_box_from_3d_points",
    "subsample_point_cloud", "filter_noise",
})

# Tier-2/advanced API:可注入 sandbox,但默认不进 prompt。它们通常是内部解析、低频 helper
# 或容易误用的组合函数;只有当某个 skill 明确需要时才通过 runtime.prompt_api_names 打开。
ADVANCED_PROMPT_APIS: frozenset[str] = frozenset({
    "point_prompt_molmo",
    "parse_vlm_detections",
    "interpolate_segment",
    "select_top_down_grasp",
})

DEFAULT_PROMPT_APIS: frozenset[str] = CORE_PROMPT_APIS | CAPABILITY_PROMPT_APIS


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

    max_subgoals: int = DEFAULT_MAX_SUBGOALS
    """外层 planner 最多拆分多少个 sub-goal;可由 YAML 的 robomex.runtime.max_subgoals 配置。"""

    subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS
    """每次 SubAgent 委托最多多少个 run_python action block。"""

    scene_camera: str = DEFAULT_SCENE_CAMERA
    """保存给 planner 看的场景图相机名;空值表示从 observation 自动选择。"""

    act_observation_camera: str = ""
    """Act 取 OBS_BEFORE/反馈图的相机名;空值表示跟随 scene_camera。"""


@dataclass(frozen=True)
class RuntimeSettings:
    """RoboMEx live runner settings that are not environment-construction fields."""

    max_subgoals: int = DEFAULT_MAX_SUBGOALS
    subagent_max_turns: int = DEFAULT_SUBAGENT_MAX_TURNS
    scene_camera: str = DEFAULT_SCENE_CAMERA
    act_observation_camera: str = DEFAULT_SCENE_CAMERA
    prompt_api_names: frozenset[str] = DEFAULT_PROMPT_APIS
    subagent_denied_calls: frozenset[str] | None = None


def _load_config_dict(config_path: str) -> dict[str, Any]:
    from capx.envs.configs.loader import DictLoader

    return DictLoader.load([os.path.expanduser(config_path)])


def _build_env(config_path: str) -> Any:
    """像 trial worker 那样实例化高层 CapX env。"""

    from capx.envs.configs.instantiate import instantiate

    configs_dict = _load_config_dict(config_path)
    if "env" not in configs_dict:
        raise ValueError(f"config {config_path} has no 'env' key")
    return instantiate(configs_dict["env"])


def _robomex_vlm_config(config_path: str) -> dict[str, Any]:
    """Read optional RoboMEx VLM settings from the env YAML.

    Preferred shape:

    robomex:
      vlm:
        model: vapi/gpt-5.5
        server_url: http://localhost:8110/chat/completions
        coord_space: pixel

    """

    cfg = _load_config_dict(config_path)
    robomex_cfg = cfg.get("robomex") if isinstance(cfg.get("robomex"), dict) else {}
    vlm = robomex_cfg.get("vlm") if isinstance(robomex_cfg.get("vlm"), dict) else None
    if not isinstance(vlm, dict):
        return {}
    return {str(k): v for k, v in vlm.items() if v not in (None, "")}


def _robomex_runtime_config(config_path: str) -> dict[str, Any]:
    """Read optional RoboMEx runtime settings from ``robomex.runtime``."""

    cfg = _load_config_dict(config_path)
    robomex_cfg = cfg.get("robomex") if isinstance(cfg.get("robomex"), dict) else {}
    runtime = robomex_cfg.get("runtime") if isinstance(robomex_cfg.get("runtime"), dict) else {}
    if not isinstance(runtime, dict):
        return {}
    return {str(k): v for k, v in runtime.items() if v is not None}


def _coerce_str_frozenset(value: Any) -> frozenset[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = [str(part).strip() for part in value]
    else:
        raise TypeError(f"expected string/list setting, got {type(value).__name__}")
    return frozenset(item for item in items if item)


def _runtime_settings(args: LiveArgs) -> RuntimeSettings:
    """Merge YAML runtime settings with explicit CLI overrides."""

    cfg = _robomex_runtime_config(args.config_path)
    prompt_api_names = _coerce_str_frozenset(cfg.get("prompt_api_names")) or DEFAULT_PROMPT_APIS
    denied_calls = _coerce_str_frozenset(cfg.get("subagent_denied_calls"))

    def effective_int(name: str, cli_value: int, default: int) -> int:
        if cli_value != default:
            return int(cli_value)
        if name in cfg:
            return int(cfg[name])
        return default

    def effective_str(name: str, cli_value: str, default: str) -> str:
        if cli_value != default:
            return cli_value
        if name in cfg:
            return str(cfg[name])
        return default

    scene_camera_cli_override = args.scene_camera != DEFAULT_SCENE_CAMERA
    scene_camera = effective_str("scene_camera", args.scene_camera, DEFAULT_SCENE_CAMERA)
    act_observation_camera = (
        args.act_observation_camera
        or (scene_camera if scene_camera_cli_override else str(cfg.get("act_observation_camera") or scene_camera))
    )

    return RuntimeSettings(
        max_subgoals=effective_int("max_subgoals", args.max_subgoals, DEFAULT_MAX_SUBGOALS),
        subagent_max_turns=effective_int("subagent_max_turns", args.subagent_max_turns, DEFAULT_SUBAGENT_MAX_TURNS),
        scene_camera=scene_camera,
        act_observation_camera=act_observation_camera,
        prompt_api_names=prompt_api_names,
        subagent_denied_calls=denied_calls,
    )


def _configure_vlm_backend(args: LiveArgs, log: Any | None = None) -> dict[str, str]:
    """Resolve code-block VLM API config from YAML, falling back to the run model."""

    cfg = _robomex_vlm_config(args.config_path)
    applied: dict[str, str] = {}

    model = cfg.get("model")
    server_url = cfg.get("server_url")
    api_key = cfg.get("api_key")
    coord_space = cfg.get("coord_space")

    applied["model"] = str(model or args.model)
    applied["server_url"] = str(server_url or args.server_url)
    if api_key or args.api_key:
        applied["api_key"] = str(api_key or args.api_key)
    if coord_space:
        applied["coord_space"] = str(coord_space)

    if log is not None:
        source = "config" if cfg else "run model fallback"
        log.info(
            "VLM grounding backend: model=%s server_url=%s coord_space=%s (%s)",
            applied.get("model"),
            applied.get("server_url"),
            applied.get("coord_space", "auto"),
            source,
        )
    return applied


def _task_language(env: Any) -> str:
    """LIBERO 任务指令字符串,用作 planner 的任务目标。"""

    handle = getattr(env.low_level_env, "handle", None)
    lang = getattr(handle, "task_language", None) if handle is not None else None
    return lang or "complete the manipulation task"


def _api_docs(env: Any, allow: frozenset[str] = DEFAULT_PROMPT_APIS) -> str:
    """Render selected sandbox API docs for Act/SubAgent prompts.

    The docs are grouped by intended use instead of hidden by tier. Capability APIs
    are visible so agents do not need schema probing, but the prompt still directs
    them to use Skills as the workflow layer around these APIs.
    """

    import inspect

    groups: dict[str, list[str]] = {
        "core": [],
        "capability": [],
        "advanced": [],
        "other": [],
    }

    def group_for(name: str) -> str:
        if name in CORE_PROMPT_APIS:
            return "core"
        if name in CAPABILITY_PROMPT_APIS:
            return "capability"
        if name in ADVANCED_PROMPT_APIS:
            return "advanced"
        return "other"

    signature_overrides = {
        "query_vlm": "(prompt, images=None, *, image=None, model=None, temperature=0.0, max_tokens=1024)",
    }
    doc_overrides = {
        "query_vlm": (
            "Ask visual QA or categorical state questions. Canonical form: "
            "query_vlm('question', images=rgb_or_crop). Compatibility forms "
            "query_vlm(rgb_or_path, 'question') and query_vlm(image=rgb_or_path, "
            "prompt='question') are accepted. Do not ask it for bbox, point, "
            "coordinates, masks, or grasp poses."
        ),
    }

    for api in getattr(env, "_apis", {}).values():
        try:
            fns = api.functions()
        except Exception:  # noqa: BLE001 - 某个 API 组取函数失败不该毁掉整段文档
            continue
        for name, fn in fns.items():
            if name not in allow:
                continue
            if name in signature_overrides:
                sig = signature_overrides[name]
            else:
                try:
                    sig = str(inspect.signature(fn))
                except (TypeError, ValueError):
                    sig = "(…)"
            doc = doc_overrides.get(name) or inspect.getdoc(fn) or ""
            lines = groups[group_for(name)]
            lines.append(f"{name}{sig}")
            if doc:
                lines.append("  Doc:")
                lines.extend(f"    {ln}" for ln in doc.splitlines())
            lines.append("")

    sections: list[str] = []
    section_specs = [
        (
            "core",
            "Core APIs: stable sensing, motion, VLM QA, and geometry primitives. "
            "These may be used directly when they advance the current sub-goal.",
        ),
        (
            "capability",
            "Capability APIs: grounding, segmentation, grasp planning, and point-cloud "
            "utilities. Prefer loading the relevant RoboMEx skill first so these APIs "
            "are used with the right workflow, validation, and artifacts. Do not inspect "
            "signatures or source just to discover how to call them; use the docs below.",
        ),
        (
            "advanced",
            "Advanced/internal APIs: use only when a loaded skill or a prior error makes "
            "the need explicit.",
        ),
        (
            "other",
            "Additional configured APIs.",
        ),
    ]
    for key, title in section_specs:
        body = "\n".join(groups[key]).strip()
        if body:
            sections.append(f"{title}\n{body}")
    return "\n\n".join(sections).strip()


def _select_rgb_camera(obs: dict, camera: str = DEFAULT_SCENE_CAMERA) -> Any | None:
    if camera:
        try:
            return obs[camera]["images"]["rgb"]
        except (KeyError, TypeError):
            return None
    if not isinstance(obs, dict):
        return None
    for cam in obs.values():
        if not isinstance(cam, dict):
            continue
        try:
            return cam["images"]["rgb"]
        except (KeyError, TypeError):
            continue
    return None


def _save_scene_image(obs: dict, path: str, *, camera: str = DEFAULT_SCENE_CAMERA) -> str | None:
    """保存相机 RGB 让 planner 能看到场景;camera 为空时自动选第一个 RGB。"""

    from robomex.perception.render import save_rgb

    rgb = _select_rgb_camera(obs, camera)
    if rgb is None:
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
    from robomex.agents.subagents import SubAgentExecutionPolicy, render_subagent_system_prompt
    from robomex.core.coder import LLMCodePolicy
    from robomex.core.sandbox import CapXExecutorAdapter
    from robomex.prompts import render_libero_act_system_prompt
    from robomex.skills import SkillLibrary, load_builtin_skills

    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = _runtime_settings(args)

    # Code-block APIs such as query_vlm / vlm_bbox_detection receive their VLM backend
    # through the CapX API instance, configured from YAML's `robomex.vlm` section.
    vlm_backend = _configure_vlm_backend(args, log)

    # 开启整段 episode 的视频录制(env 支持时);逐 sub-goal / 收尾时写盘。
    if hasattr(env, "enable_video_capture"):
        try:
            env.enable_video_capture(True, clear=True)
            log.info("已开启视频录制")
        except Exception as exc:  # noqa: BLE001 - 录像不可用不该阻断 run
            log.warning("开启视频录制失败: %r", exc)

    scene_path = _save_scene_image(obs, str(out_dir / "scene.png"), camera=runtime.scene_camera)
    log.info("初始场景图: %s", scene_path or "(取不到 -> planner 仅凭文本规划)")

    library = SkillLibrary(str(out_dir / "library"))
    for skill in load_builtin_skills():
        library.admit(skill, source="builtin")

    api_docs = _api_docs(env, runtime.prompt_api_names)
    system_prompt = render_libero_act_system_prompt(api_docs)
    subagent_system_prompt = render_subagent_system_prompt(api_docs)
    code_policy = LLMCodePolicy(model=args.model, server_url=args.server_url, api_key=args.api_key)
    scene_camera_label = runtime.scene_camera or "auto"
    act_camera_label = runtime.act_observation_camera or "auto"
    log.info("Act/SubAgent code policy: JSON action adapter")
    log.info(
        "RoboMEx runtime: max_subgoals=%d subagent_max_turns=%d scene_camera=%s act_observation_camera=%s",
        runtime.max_subgoals,
        runtime.subagent_max_turns,
        scene_camera_label,
        act_camera_label,
    )

    # 框架入口:把所有依赖收进一个 RoboMExConfig,再交给 RoboMExAgent 装配 + 运行
    # 反应式两层循环。执行器在 inner loop 内 use_skill/run_python;finish 交回 Planner。
    config = RoboMExConfig(
        library=library,
        planner_policy=LLMPlannerPolicy(model=args.model, server_url=args.server_url, api_key=args.api_key),
        code_policy=code_policy,
        executor=CapXExecutorAdapter(env, vlm_backend=vlm_backend),
        max_turns=args.max_turns,
        max_subgoals=runtime.max_subgoals,
        inner_system_prompt=system_prompt,
        subagent_system_prompt=subagent_system_prompt,
        code_policy_kind="json_action_adapter",
        observation_camera=runtime.act_observation_camera,
        subagent_max_turns=runtime.subagent_max_turns,
        subagent_execution_policy=(
            SubAgentExecutionPolicy(denied_calls=runtime.subagent_denied_calls)
            if runtime.subagent_denied_calls is not None
            else None
        ),
        observation_summary=(
            f"A LIBERO tabletop scene. Planner snapshots use the {scene_camera_label!r} camera. "
            "Call get_observation() for available RGB/depth, intrinsics, and camera poses."
        ),
        artifacts_dir=str(out_dir),
    )
    agent = RoboMExAgent(config)

    # 真机场景刷新:每个 sub-goal 之后,用最新观测重渲染 planner 看到的场景
    #(与具体 env 相关;离线时保持 None)。
    step = {"n": 0}

    def scene_refresh(observation: dict) -> str | None:
        step["n"] += 1
        return _save_scene_image(
            observation,
            str(out_dir / f"scene_step{step['n']}.png"),
            camera=runtime.scene_camera,
        )

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
    log.info("  ├─ subagents.json   SubAgent runtime 配置与执行边界")
    log.info("  ├─ subgoal_NN/      Act turn_*.py/out + SubAgent request/result/code artifacts + 过程视频")
    log.info("  └─ scene*.png       每步 planner 看到的场景图")


if __name__ == "__main__":
    main(tyro.cli(LiveArgs))
