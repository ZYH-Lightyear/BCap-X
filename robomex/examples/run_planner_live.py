"""Live two-level RoboMEx agent on a real CapX/LIBERO(-PRO) task.

Outer ReactivePlanner (task + initial scene image + high-level skill menu) emits a
JSON To-Do list of sub-goals; the inner CodeAsPolicyAgent runs each sub-goal on the
real env in order. NO re-planning and NO VLMJudge yet -- the inner loop verifies
with the env task signal only (TaskSignalVerifier). The goal is to validate the
two-level plumbing on a real LIBERO-PRO task end to end.

Prerequisites (same servers the baseline uses):
    - LLM proxy                        :8110
    - sam3 / contact-graspnet / pyroki :8114 / :8115 / :8116

Usage::

    uv run --no-sync --active robomex/examples/run_planner_live.py \\
        --config-path env_configs/libero/franka_libero_cap_agent0.yaml \\
        --model openrouter/qwen/qwen3.6-plus
"""

from __future__ import annotations

import os

# MuJoCo needs a GL backend chosen before the sim is imported. Mirror launch.py;
# override with MUJOCO_GL=osmesa for CPU rendering on headless boxes without EGL.
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
    """CLI arguments for a single live two-level trial."""

    config_path: str = "env_configs/libero/franka_libero_cap_agent0.yaml"
    """YAML env config (same one the CaP-Agent0 baseline uses)."""

    model: str = "openrouter/qwen/qwen3.6-plus"
    """Model for BOTH the planner and the inner code agent (routed through the proxy)."""

    server_url: str = "http://localhost:8110/chat/completions"
    """Local LLM proxy endpoint."""

    api_key: str | None = None
    """Optional API key (proxy usually injects it)."""

    max_turns: int = 6
    """Max inner code-generation turns per sub-goal."""

    output_dir: str = "./outputs/robomex_planner_live"
    """Root for this run's artifacts; a timestamp dir is created under it."""

    seed: int | None = None
    """Optional env reset seed."""


def _build_env(config_path: str) -> Any:
    """Instantiate the high-level CapX env exactly like the trial workers do."""

    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader

    configs_dict = DictLoader.load([os.path.expanduser(config_path)])
    if "env" not in configs_dict:
        raise ValueError(f"config {config_path} has no 'env' key")
    return instantiate(configs_dict["env"])


def _task_language(env: Any) -> str:
    """LIBERO task instruction string, used as the planner's task goal."""

    handle = getattr(env.low_level_env, "handle", None)
    lang = getattr(handle, "task_language", None) if handle is not None else None
    return lang or "complete the manipulation task"


def _api_docs(env: Any) -> str:
    """Concatenated API documentation the inner code agent needs."""

    apis = getattr(env, "_apis", {})
    return "\n".join(api.combined_doc() for api in apis.values())


def _save_scene_image(obs: dict, path: str) -> str | None:
    """Save the initial agentview RGB so the planner can see the scene; None on miss."""

    from robomex.perception.render import save_rgb

    try:
        rgb = obs["agentview"]["images"]["rgb"]
    except (KeyError, TypeError):
        return None
    try:
        return save_rgb(path, rgb)
    except Exception:
        return None


def main(args: LiveArgs) -> None:
    from robomex.adapters.capx.executor import CapXExecutorAdapter
    from robomex.agent import CodeAsPolicyAgent, LLMCodePolicy
    from robomex.library import SkillLibrary
    from robomex.planner import (
        LLMPlannerPolicy,
        ReactivePlanner,
        SubGoalResult,
    )
    from robomex.skills.skills_library import load_skills_library
    from robomex.verification import TaskSignalVerifier

    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[live] building env from {args.config_path} (MUJOCO_GL={os.environ.get('MUJOCO_GL')}) ...")
    env = _build_env(args.config_path)
    obs, _info = env.reset(seed=args.seed)
    task = _task_language(env)
    print(f"[live] task: {task}")

    scene_path = _save_scene_image(obs, str(out_dir / "scene.png"))
    print(f"[live] initial scene image: {scene_path or '(unavailable -> planning from text only)'}")

    library = SkillLibrary(str(out_dir / "library"))
    for skill in load_skills_library():
        library.admit(skill, source="builtin")
    menu = [r.skill_id for r in library.compound_skills()]
    print(f"[live] high-level skills (planner menu): {menu}")

    planner = ReactivePlanner(
        library,
        LLMPlannerPolicy(model=args.model, server_url=args.server_url, api_key=args.api_key),
    )

    system_prompt = (
        f"{_CODE_AS_POLICY_PROMPT}\n\nAvailable API functions (already imported):\n{_api_docs(env)}"
    )
    inner = CodeAsPolicyAgent(
        executor=CapXExecutorAdapter(env),
        policy=LLMCodePolicy(model=args.model, server_url=args.server_url, api_key=args.api_key),
        library=library,
        verifier=TaskSignalVerifier(),  # env signal only; VLMJudge intentionally skipped
        max_turns=args.max_turns,
        system_prompt=system_prompt,
    )

    # Plan first so we can print the To-Do list before executing.
    subgoals = planner.plan(task, scene_image_path=scene_path)
    print(f"\n[live] To-Do list ({len(subgoals)} sub-goals):")
    for i, sg in enumerate(subgoals, 1):
        print(f"  {i}. {sg.goal}")
        print(f"     skill={sg.skill} | done-when: {sg.postcondition}")
    if not subgoals:
        print("[live] planner returned no sub-goals; aborting.")
        return

    # Run each sub-goal on the same (persistent) env, in order. No re-planning.
    observation_summary = (
        "A LIBERO tabletop scene. Call get_observation() for agentview/wrist RGB, "
        "depth, intrinsics, and camera pose."
    )
    results: list[SubGoalResult] = []
    for sg in subgoals:
        print(f"\n[live] >>> sub-goal: {sg.goal}")
        trace = inner.run(sg.goal, observation_summary)
        results.append(SubGoalResult(subgoal=sg, trace=trace, success=trace.success))
        print(f"[live]     turns={len(trace.turns)} success={trace.success} "
              f"loaded={list(trace.loaded_skill_ids)}")
        for t in trace.turns:
            print(f"[live]       turn {t.turn}: exec={t.execution.status.value} "
                  f"verdict={t.verification.status.value}")

    overall = bool(results) and all(r.success for r in results)
    print(f"\n[live] plan success={overall}  ({sum(r.success for r in results)}/{len(results)} sub-goals)")
    print(f"[live] artifacts under: {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(LiveArgs))
