"""Live RoboMEx Skill Agent on a real CapX/LIBERO environment.

This is roadmap step 1 ("真机最小验证"): take the *existing* v1.1 loop and swap
the three offline mock components in ``run_skill_agent.py`` for real ones --
nothing about the architecture changes, we only validate the plumbing on real
observations and gauge VLM-judge quality/cost.

    MockExecutor        -> CapXExecutorAdapter(env)   # env from the YAML config
    ScriptedCodePolicy  -> LLMCodePolicy(:8110)
    default verifier    -> CompositeVerifier(TaskSignalVerifier, VLMJudgeVerifier)

Prerequisites (must already be running; the same servers the baseline uses):
    - LLM proxy                       :8110
    - sam3 / contact-graspnet / pyroki :8114 / :8115 / :8116

Usage::

    uv run --no-sync --active robomex/examples/run_skill_agent_live.py \\
        --config-path env_configs/libero/franka_libero_cap_agent0.yaml \\
        --model openrouter/qwen/qwen3.6-plus

Outputs (evidence overlays, seeded/learned skill library) land under
``<output_dir>/<timestamp>/``.
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
    "Consult a relevant skill first with `USE SKILL: <name>` (skills are listed in the "
    "system reminder); its guidance is advisory -- adapt it, do not copy it blindly. "
    "Each turn, write ONE block of executable Python that advances the task, grounding "
    "every decision in the current observation via get_observation(). "
    "All API functions listed below are already imported into the namespace. "
    "Reply with a ```python``` code block, a `USE SKILL: <name>` line, or the word FINISH "
    "when the task is complete."
)


@dataclass
class LiveArgs:
    """CLI arguments for a single live RoboMEx trial."""

    config_path: str = "env_configs/libero/franka_libero_cap_agent0.yaml"
    """YAML env config (same one the CaP-Agent0 baseline uses)."""

    model: str = "openrouter/qwen/qwen3.6-plus"
    """Code-generation model, routed through the local proxy."""

    judge_model: str = "openrouter/qwen/qwen3.6-plus"
    """VLM model used by the gate-3 effect verifier."""

    server_url: str = "http://localhost:8110/chat/completions"
    """Local LLM proxy endpoint shared by policy and judge."""

    api_key: str | None = None
    """Optional API key (proxy usually injects it)."""

    max_turns: int = 6
    """Max code-generation turns before giving up."""

    output_dir: str = "./outputs/robomex_live"
    """Root for this run's evidence + skill library; a timestamp dir is created under it."""

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
    """LIBERO task instruction string, used as the agent's task goal."""

    handle = getattr(env.low_level_env, "handle", None)
    lang = getattr(handle, "task_language", None) if handle is not None else None
    return lang or "complete the manipulation task"


def _api_docs(env: Any) -> str:
    """Concatenated API documentation the LLM needs to write valid calls."""

    apis = getattr(env, "_apis", {})
    return "\n".join(api.combined_doc() for api in apis.values())


def main(args: LiveArgs) -> None:
    from robomex.adapters.capx.executor import CapXExecutorAdapter
    from robomex.agent import CodeAsPolicyAgent, LLMCodePolicy
    from robomex.distill import SkillDistiller
    from robomex.library import SkillLibrary
    from robomex.perception import EvidenceCollector
    from robomex.skills.skills_library import load_skills_library
    from robomex.verification import (
        CompositeVerifier,
        TaskSignalVerifier,
        VLMJudgeVerifier,
    )

    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[live] building env from {args.config_path} (MUJOCO_GL={os.environ.get('MUJOCO_GL')}) ...")
    env = _build_env(args.config_path)
    env.reset(seed=args.seed)
    task = _task_language(env)
    print(f"[live] task: {task}")

    library = SkillLibrary(str(out_dir / "library"))
    for skill in load_skills_library():
        library.admit(skill, source="builtin")
    print(f"[live] library: {[(r.skill.category.value, r.skill_id) for r in library.all()]}")

    system_prompt = (
        f"{_CODE_AS_POLICY_PROMPT}\n\nAvailable API functions (already imported):\n{_api_docs(env)}"
    )

    agent = CodeAsPolicyAgent(
        executor=CapXExecutorAdapter(env),
        policy=LLMCodePolicy(model=args.model, server_url=args.server_url, api_key=args.api_key),
        library=library,
        verifier=CompositeVerifier(
            TaskSignalVerifier(),
            VLMJudgeVerifier(model=args.judge_model, server_url=args.server_url, api_key=args.api_key),
        ),
        collector=EvidenceCollector(str(out_dir / "evidence")),
        max_turns=args.max_turns,
        system_prompt=system_prompt,
    )

    trace = agent.run(
        task=task,
        observation_summary=(
            "A LIBERO tabletop scene. Call get_observation() for agentview/wrist RGB, "
            "depth, intrinsics, and camera pose."
        ),
    )

    print(f"\n[live] loaded skills: {list(trace.loaded_skill_ids)}")
    print(f"[live] turns={len(trace.turns)} success={trace.success}")
    for t in trace.turns:
        print(
            f"  turn {t.turn}: exec={t.execution.status.value} "
            f"verdict={t.verification.status.value} :: {t.verification.summary[:160]}"
        )

    learned = SkillDistiller(library).evolve(trace)
    print(f"\n[live] distilled skill: {learned.skill_id if learned else None}")
    print(f"[live] evidence + library under: {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(LiveArgs))
