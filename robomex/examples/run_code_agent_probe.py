"""Probe a single Code Agent on a perception sub-goal in real LIBERO.

This is a focused unit-test harness for the *inner* Code Agent only -- no planner,
no skill isolation. The library is loaded normally; the agent is handed one small
perception task (by default: estimate an object's height) and must ground + compose
the code itself (segment -> recover points -> estimate geometry -> print result).

Because a measurement task does not trigger an env success signal, the meaningful
output is the agent's printed result captured in each turn's stdout -- this script
dumps the generated code + stdout per turn so you can read what the skills produced.

After the agent finishes, the script runs the skill's verifier-as-code
(``observation/estimate_geometry/scripts/verify.py``) inside the SAME sandbox.
The verifier checks the executor's REAL artifacts via the manifest the agent
leaves in ``RESULT`` (A-plan handoff: scalars + on-disk ``.npy`` paths), projects
the measured OBB + top/bottom points back onto the agentview RGB, saves
``geometry_overlay.png``, and VLM-judges it against ``ref/verify.md``. It also
prints the routed verify resources + a sanitized op-trace (the fact-only verifier
context, docs §5.6). If the agent left no manifest, the verifier falls back to an
independent re-measurement, flagged ``evidence_source="reproduced"``.

Prerequisites (servers the perception skills need):
    - LLM proxy :8110   (code generation AND the VLM judge)
    - sam3 :8114        (segmentation; height needs segment + geometry)

Usage::

    uv run --no-sync --active robomex/examples/run_code_agent_probe.py \\
        --object "the black bowl" \\
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
    "You are given ONE perception task. Consult a relevant skill first with "
    "`USE SKILL: <name>` (skills are listed in the system reminder); its guidance is "
    "advisory -- adapt it, do not copy it blindly. Each turn, write ONE block of executable "
    "Python that advances the task, grounding every decision in the current observation via "
    "get_observation(). All API functions listed below are already imported into the namespace. "
    "PRINT the measured result you are asked for so it is visible in stdout. "
    "If a consulted skill has a '## Report' section, follow it on your final step "
    "to record your result into RESULT (saving heavy arrays under EVIDENCE_DIR) so "
    "the verifier can check your actual artifacts. "
    "Reply with a ```python``` code block, a `USE SKILL: <name>` line, or the word FINISH "
    "when the task is complete."
)


@dataclass
class ProbeArgs:
    """CLI arguments for a single Code Agent perception probe."""

    object: str = "the black bowl"
    """The object whose height the agent should estimate."""

    task: str | None = None
    """Override the full task string; if unset it is derived from --object."""

    config_path: str = "env_configs/libero/franka_libero_cap_agent0.yaml"
    """YAML env config (same one the CaP-Agent0 baseline uses)."""

    model: str = "openrouter/qwen/qwen3.6-plus"
    """Code-generation model, routed through the local proxy."""

    server_url: str = "http://localhost:8110/chat/completions"
    """Local LLM proxy endpoint."""

    api_key: str | None = None
    """Optional API key (proxy usually injects it)."""

    max_turns: int = 6
    """Max code-generation turns before giving up."""

    seed: int | None = None
    """Optional env reset seed."""

    verify: bool = True
    """After the agent finishes, run the skill's verifier-as-code in the same sandbox."""

    verify_vlm: bool = True
    """Let the verifier call the VLM judge; if False it only renders + numeric guards."""

    output_dir: str = "./outputs/code_agent_probe"
    """Where the library copy and verifier overlay images are written."""


def _build_env(config_path: str) -> Any:
    """Instantiate the high-level CapX env exactly like the trial workers do."""

    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader

    configs_dict = DictLoader.load([os.path.expanduser(config_path)])
    if "env" not in configs_dict:
        raise ValueError(f"config {config_path} has no 'env' key")
    return instantiate(configs_dict["env"])


def _api_docs(env: Any) -> str:
    """Concatenated API documentation the LLM needs to write valid calls."""

    apis = getattr(env, "_apis", {})
    return "\n".join(api.combined_doc() for api in apis.values())


def main(args: ProbeArgs) -> None:
    from robomex.adapters.capx.executor import CapXExecutorAdapter
    from robomex.agent import CodeAsPolicyAgent, LLMCodePolicy
    from robomex.library import SkillLibrary
    from robomex.skills.skills_library import load_skills_library
    from robomex.verification import TaskSignalVerifier

    # Phrase the task so retrieval surfaces BOTH segment_object and estimate_geometry,
    # and so the agent knows to print the measured height.
    task = args.task or (
        f"Estimate the height of {args.object}. Segment it to recover its 3D points, "
        f"estimate its geometry, and print the measured height, top_z and bottom_z (in meters)."
    )

    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probe] building env from {args.config_path} (MUJOCO_GL={os.environ.get('MUJOCO_GL')}) ...")
    env = _build_env(args.config_path)
    env.reset(seed=args.seed)
    print(f"[probe] task: {task}")

    # Library loaded normally (not isolated): the agent retrieves + composes itself.
    library = SkillLibrary(str(out_dir / "library"))
    for skill in load_skills_library():
        library.admit(skill, source="builtin")
    print(f"[probe] library: {[(r.skill.category.value, r.skill_id) for r in library.all()]}")

    # Seed EVIDENCE_DIR into the persistent sandbox namespace so the executor can
    # save its artifacts (points/mask .npy) there per the skill's Report contract.
    evidence_dir = (out_dir / "evidence").resolve()
    env.step(f"import os as _os; EVIDENCE_DIR = {str(evidence_dir)!r}; "
             "_os.makedirs(EVIDENCE_DIR, exist_ok=True)")

    system_prompt = (
        f"{_CODE_AS_POLICY_PROMPT}\n\nAvailable API functions (already imported):\n{_api_docs(env)}"
    )
    agent = CodeAsPolicyAgent(
        executor=CapXExecutorAdapter(env),
        policy=LLMCodePolicy(model=args.model, server_url=args.server_url, api_key=args.api_key),
        library=library,
        verifier=TaskSignalVerifier(),  # measurement has no env success signal; see stdout below
        max_turns=args.max_turns,
        system_prompt=system_prompt,
        require_result=True,  # terminal contract: must hand the verifier a RESULT manifest
    )

    trace = agent.run(
        task=task,
        observation_summary=(
            "A LIBERO tabletop scene. Call get_observation() for agentview/wrist RGB, "
            "depth, intrinsics, and camera pose."
        ),
    )

    print(f"\n[probe] loaded skills: {list(trace.loaded_skill_ids)}")
    print(f"[probe] turns={len(trace.turns)} (env success flag={trace.success}; "
          "ignore for a measurement task -- read the stdout below)\n")
    for t in trace.turns:
        print(f"===== turn {t.turn} | exec={t.execution.status.value} "
              f"verdict={t.verification.status.value} =====")
        print("--- code ---")
        print(t.code)
        print("--- stdout (the RESULT) ---")
        print(t.execution.stdout or "(empty)")
        if t.execution.stderr:
            print("--- stderr ---")
            print(t.execution.stderr)
        print()

    if args.verify:
        _run_verifier(env, library, args, out_dir, trace)


def _verifier_skill_ids(library: Any, trace: Any) -> tuple[str, ...]:
    """Skills the verifier should be aware of.

    The verification rubric/code belongs to the skills relevant to the SUB-GOAL
    that ship verify assets -- not merely whatever the executor happened to load
    (which is just context). So we union the executor's loaded skills with the
    verify-capable skills retrieved for the sub-goal, falling back to any skill
    shipping a verify.py if nothing else turns up.
    """
    ids: list[str] = list(trace.loaded_skill_ids)
    for r in library.retrieve(trace.task, k=4):
        has_assets = r.skill.verifier_path() is not None or r.skill.verify_doc_path() is not None
        if has_assets and r.skill_id not in ids:
            ids.append(r.skill_id)
    if not ids:
        ids = [r.skill_id for r in library.all() if r.skill.verifier_path() is not None]
    return tuple(ids)


def _build_verifier_context(library: Any, trace: Any) -> Any:
    """Assemble the fact-only verifier context (skills used, claim, op-trace, rubrics).

    The executor's CLAIM and printed stdout both come straight off the trace
    (first-class handoff) -- no out-of-band re-read of the sandbox.
    """
    from robomex.verification import (
        VerifierContext,
        build_op_trace,
        collect_verify_resources,
    )

    skill_ids = _verifier_skill_ids(library, trace)
    skills = [r.skill for r in library.all()]
    resources = collect_verify_resources(skills, skill_ids)
    return VerifierContext(
        sub_goal=trace.task,
        skills_used=skill_ids,
        claim=trace.claim,
        op_trace=tuple(build_op_trace(trace.turns)),
        resources=resources,
        executor_stdout=trace.executor_stdout,
    )


def _run_verifier(env: Any, library: Any, args: ProbeArgs, out_dir: Path, trace: Any) -> None:
    """Run the independent VerifyCodeAgent in the SAME sandbox.

    The verifier is the same kind of agent as the executor: it sees only facts
    (the VerifierContext -- sub-goal, skills used, the executor's CLAIM manifest,
    op-trace, authored rubrics), can ``USE SKILL`` to read full bodies + sidecars,
    and writes its own judge code (the skills' scripts/verify.py are preloaded as
    ``VERIFY_PRIMITIVES`` references it may compose or copy). It finishes with a
    bare JSON verdict.
    """
    from robomex.adapters.capx.executor import CapXExecutorAdapter
    from robomex.agent import LLMCodePolicy
    from robomex.verification import VerifyCodeAgent

    ctx = _build_verifier_context(library, trace)
    print("\n[verify] ----- verifier context (facts only) -----")
    print(ctx.render())
    print("[verify] -------------------------------------------")

    agent = VerifyCodeAgent(
        executor=CapXExecutorAdapter(env),  # same persistent sandbox -> sees RESULT + .npy
        policy=LLMCodePolicy(model=args.model, server_url=args.server_url, api_key=args.api_key),
        context=ctx,
        library=library,
        max_turns=args.max_turns,
        primitive_model=args.model,
        primitive_server_url=args.server_url,
        primitive_api_key=args.api_key,
    )

    print("\n[verify] running VerifyCodeAgent in the sandbox ...")
    vtrace = agent.verify()
    verdict = vtrace.verdict

    print(f"\n[verify] verdict: {verdict.verdict} (confidence={verdict.confidence})")
    print(f"[verify] reason : {verdict.reason}")
    if verdict.evidence:
        print(f"[verify] evidence: {verdict.evidence}")
    print(f"[verify] judge turns: {len(vtrace.turns)}")
    for vt in vtrace.turns:
        print(f"  --- judge turn {vt.turn} ---")
        print(vt.code)
        if vt.stdout.strip():
            print(f"  stdout: {vt.stdout.strip()[:400]}")
        if vt.stderr.strip():
            print(f"  stderr: {vt.stderr.strip()[:400]}")

    judge_path = out_dir / "verifier_judge_code.py"
    judge_path.write_text("\n\n# ---- next judge turn ----\n\n".join(vt.code for vt in vtrace.turns))
    print(f"[verify] judge code saved: {judge_path}")


if __name__ == "__main__":
    main(tyro.cli(ProbeArgs))
