from __future__ import annotations

import shutil
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path


LIBERO_REDUCED_SKILL_LIBRARY_APIS = [
    "get_observation",
    "segment_sam3_text_prompt",
    "segment_sam3_point_prompt",
    "point_prompt_molmo",
    "query_vlm",
    "plan_grasp",
    "plan_grasp_from_point_clouds",
    "get_oriented_bounding_box_from_3d_points",
    "solve_ik",
    "move_to_joints",
    "open_gripper",
    "close_gripper",
    "goto_pose",
    "goto_home_joint_position",
    "subsample_point_cloud",
    "filter_noise",
    "rotation_matrix_to_quaternion",
    "decompose_transform",
    "depth_to_point_cloud",
    "mask_to_world_points",
    "pixel_to_world_point",
    "transform_points",
    "interpolate_segment",
    "normalize_vector",
    "select_top_down_grasp",
]

LIBERO_SKILL_NAMES = [
    "libero-scene-observation",
    "libero-language-grounding",
    "libero-segmentation-to-points",
    "libero-geometry-and-frames",
    "libero-grasp-object",
    "libero-motion-control",
    "libero-place-and-release",
    "libero-articulated-and-contact-actions",
    "libero-debug-and-recovery",
]


@dataclass(frozen=True)
class WorkspaceSpec:
    workspace: Path
    project_root: Path
    config_path: Path
    skills_dir: Path | None
    skill_mode: str
    agent: str
    max_runs: int
    trial_seed: int
    record_video: bool


def prepare_codex_libero_workspace(spec: WorkspaceSpec) -> None:
    if spec.workspace.exists():
        shutil.rmtree(spec.workspace)
    spec.workspace.mkdir(parents=True, exist_ok=True)
    (spec.workspace / "tools").mkdir(exist_ok=True)
    (spec.workspace / "artifacts").mkdir(exist_ok=True)
    (spec.workspace / "logs").mkdir(exist_ok=True)
    _write_text(spec.workspace / "logs" / f"{spec.agent}_stdout.jsonl", "")
    _write_text(spec.workspace / "logs" / f"{spec.agent}_stderr.txt", "")

    _write_text(spec.workspace / "task.md", _render_task_md(spec))
    _write_text(spec.workspace / "api_contract.md", _render_api_contract_md(spec))
    _write_text(spec.workspace / "solution.py", _render_solution_py())
    if spec.agent == "opencode":
        _write_text(spec.workspace / "opencode.json", _render_opencode_json())
    run_tool = spec.workspace / "tools" / "run_solution.py"
    _write_text(run_tool, _render_run_solution_py(spec))
    run_tool.chmod(0o755)

    if spec.skill_mode in {"with-skill", "explicit-skill"} and spec.skills_dir is not None:
        target = spec.workspace / ".agents" / "skills"
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(spec.skills_dir, target)


def render_codex_prompt(*, skill_mode: str, max_runs: int) -> str:
    skill_line = ""
    if skill_mode == "explicit-skill":
        skill_line = "\nRelevant skills: " + " ".join(f"${name}" for name in LIBERO_SKILL_NAMES) + "\n"

    return f"""You are solving one CaP-X LIBERO-Pro manipulation trial as an autonomous coding agent.
Work in this directory only. Read task.md and api_contract.md first.{skill_line}
Your editable policy is solution.py. The CaP-X runtime injects the LIBERO APIs listed
in api_contract.md directly into solution.py globals when the runner executes it.

Workflow:
1. Run `python tools/run_solution.py --inspect-only` to materialize the concrete LIBERO task
   language and initial camera images under artifacts/.
2. Edit solution.py to solve the task.
3. Run `python tools/run_solution.py` to execute solution.py in a fresh LIBERO episode.
4. Inspect artifacts/result.json, artifacts/stdout_latest.txt, artifacts/stderr_latest.txt,
   and the latest PNG images. If needed, revise solution.py and run again.
5. Stop once reward is 1.0 or task_completed is true.

A failed result.json is not terminal. If sandbox_rc is nonzero, reward is 0, or
task_completed is false, read stderr_latest.txt/stdout_latest.txt/images, fix
solution.py, and run tools/run_solution.py again until the simulation run budget
is exhausted.

You may execute at most {max_runs} simulation runs, excluding inspect-only runs. Be concise:
focus on producing working code rather than explaining your reasoning."""


def render_agent_prompt(*, agent: str, skill_mode: str, max_runs: int) -> str:
    if agent == "codex":
        return render_codex_prompt(skill_mode=skill_mode, max_runs=max_runs)
    if agent != "opencode":
        raise ValueError(f"unsupported agent: {agent}")

    skill_line = ""
    if skill_mode == "explicit-skill":
        skill_line = "\nWhen useful, load these skills with the skill tool: " + ", ".join(LIBERO_SKILL_NAMES) + ".\n"

    return f"""You are solving one CaP-X LIBERO-Pro manipulation trial as an autonomous coding agent.
Work in this directory only. Read task.md and api_contract.md first.{skill_line}
Your editable policy is solution.py. The CaP-X runtime injects the LIBERO APIs listed
in api_contract.md directly into solution.py globals when the runner executes it.

Workflow:
1. Run `python tools/run_solution.py --inspect-only` to materialize the concrete LIBERO task
   language and initial camera images under artifacts/.
2. Edit solution.py to solve the task.
3. Run `python tools/run_solution.py` to execute solution.py in a fresh LIBERO episode.
4. Inspect artifacts/result.json, artifacts/stdout_latest.txt, artifacts/stderr_latest.txt,
   and the latest PNG images. If needed, revise solution.py and run again.
5. Stop once reward is 1.0 or task_completed is true.

A failed result.json is not terminal. If sandbox_rc is nonzero, reward is 0, or
task_completed is false, read stderr_latest.txt/stdout_latest.txt/images, fix
solution.py, and run tools/run_solution.py again until the simulation run budget
is exhausted.

You may execute at most {max_runs} simulation runs, excluding inspect-only runs. Be concise:
focus on producing working code rather than explaining your reasoning."""


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _render_opencode_json() -> str:
    return json.dumps(
        {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "snapshot": False,
            "permission": {
                "skill": {
                    "*": "allow",
                }
            },
        },
        indent=2,
    )


def _render_task_md(spec: WorkspaceSpec) -> str:
    return (
        f"""# CaP-X LIBERO-Pro Trial

Config: `{spec.config_path}`
Trial seed: `{spec.trial_seed}`

The concrete LIBERO language goal is supplied by the simulator. Run:

```bash
python tools/run_solution.py --inspect-only
```

This writes `artifacts/task_prompt.md`, `artifacts/initial_agentview.png`, and
`artifacts/initial_wrist.png` when a wrist camera is available.

Solve the task by editing `solution.py`, then run:

```bash
python tools/run_solution.py
```
"""
    )


def _render_api_contract_md(spec: WorkspaceSpec) -> str:
    api_list = "\n".join(f"- `{name}`" for name in LIBERO_REDUCED_SKILL_LIBRARY_APIS)
    return (
        f"""# LIBERO Reduced Skill Library API Contract

`solution.py` is executed by `CodeExecutionEnvBase.step(code)` after a fresh env reset.
The following functions are already imported into the execution globals; do not create
RPC clients manually.

{api_list}

Also available:
- `obs`: current observation dict from `get_observation()`.
- `env`: low-level LIBERO environment.
- `APIS`: mapping of CaP-X API object names to API instances.
- `RESULT`: optional variable you may set for debugging output.

Use explicit imports for normal Python packages, for example `import numpy as np`.
Quaternions are WXYZ. World-frame target poses are preferred for robot motion.
Camera observations include RGB/depth and calibration data when exposed by the API.

Key signatures used often:

```python
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]          # (H, W, 3) uint8
depth = cam["images"]["depth"]      # (H, W) or (H, W, 1)
intrinsics = cam["intrinsics"]      # (3, 3)
cam_to_world = cam["pose_mat"]      # (4, 4)

segment_sam3_text_prompt(rgb, text_prompt) -> list[dict]
segment_sam3_point_prompt(rgb, (x, y)) -> list[dict]
mask_to_world_points(mask, depth, intrinsics, cam_to_world) -> np.ndarray  # (N, 3)
filter_noise(points, colors=None) -> tuple[np.ndarray, np.ndarray | None]
plan_grasp(depth, intrinsics, segmentation) -> tuple[np.ndarray, np.ndarray]
select_top_down_grasp(grasps, scores, cam_to_world, vertical_threshold=0.8) -> tuple
plan_grasp_from_point_clouds(pc_full, pc_segment) -> tuple[np.ndarray, np.ndarray]
goto_pose(position, quaternion_wxyz, z_approach=0.0)
```

For mask-based grasping, pass a binary or integer segmentation image with the same
height/width as depth. If a run fails, read `artifacts/stderr_latest.txt` and fix
the call signature before running `python tools/run_solution.py` again.

VLM coordinate convention: Qwen-family vision models often return points/boxes in
a normalized `0..1000` coordinate system. Before using a VLM point with SAM3 or
`pixel_to_world_point`, compare it with the real image shape and rescale if needed:

```python
h, w = rgb.shape[:2]
if 0 <= x <= 1000 and 0 <= y <= 1000 and (x > w or y > h):
    x = int(round(x / 1000 * (w - 1)))
    y = int(round(y / 1000 * (h - 1)))
```

OpenAI/GPT-style models may return actual pixels when explicitly requested, so do
not rescale blindly.
"""
    )


def _render_solution_py() -> str:
    return textwrap.dedent(
        """
        # Edit this file. It is executed inside CaP-X with LIBERO APIs injected as globals.
        import numpy as np

        obs = get_observation()
        print("Observation keys:", sorted(obs.keys()))
        RESULT = {"status": "starter_policy_ran"}
        """
    )


def _render_run_solution_py(spec: WorkspaceSpec) -> str:
    return textwrap.dedent(
        f"""
        #!/usr/bin/env python3
        from __future__ import annotations

        import argparse
        import json
        import os
        import shutil
        import sys
        import traceback
        from pathlib import Path

        PROJECT_ROOT = Path({str(spec.project_root)!r})
        CONFIG_PATH = Path({str(spec.config_path)!r})
        ARTIFACTS_DIR = Path({str((spec.workspace / "artifacts").resolve())!r})
        SOLUTION_PATH = Path({str((spec.workspace / "solution.py").resolve())!r})
        MAX_RUNS = {spec.max_runs}
        TRIAL_SEED = {spec.trial_seed}
        RECORD_VIDEO = {bool(spec.record_video)!r}

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        os.chdir(PROJECT_ROOT)
        os.environ.setdefault("MUJOCO_GL", "egl")

        import capx.integrations  # noqa: F401
        from PIL import Image

        from capx.envs.configs.instantiate import instantiate
        from capx.envs.configs.loader import DictLoader
        from capx.utils.video_utils import _write_video


        def _read_run_count() -> int:
            count_path = ARTIFACTS_DIR / "run_count.txt"
            if not count_path.exists():
                return 0
            try:
                return int(count_path.read_text(encoding="utf-8").strip() or "0")
            except ValueError:
                return 0


        def _write_run_count(value: int) -> None:
            (ARTIFACTS_DIR / "run_count.txt").write_text(str(value), encoding="utf-8")


        def _jsonable(value):
            try:
                json.dumps(value)
                return value
            except TypeError:
                return repr(value)


        def _save_image(array, path: Path) -> None:
            if array is None:
                return
            Image.fromarray(array).save(path)


        def _patch_libero_goal(env, obs, info):
            task_language = None
            if hasattr(env.low_level_env, "handle") and hasattr(env.low_level_env.handle, "task_language"):
                task_language = env.low_level_env.handle.task_language
            task_prompt = info.get("task_prompt") or task_language or ""
            if task_language and hasattr(env, "_task_prompt") and "libero_environment_goal" in env._task_prompt:
                env._task_prompt = env._task_prompt.format(libero_environment_goal=task_language)
            full_prompt = obs.get("full_prompt")
            if task_language and full_prompt:
                text = full_prompt[-1]["content"][0]["text"]
                if "libero_environment_goal" in text:
                    full_prompt[-1]["content"][0]["text"] = text.format(
                        libero_environment_goal=task_language
                    )
            if task_language and "libero_environment_goal" in task_prompt:
                task_prompt = task_prompt.format(libero_environment_goal=task_language)
            return task_language or task_prompt, full_prompt


        def _make_env():
            config = DictLoader.load(str(CONFIG_PATH))
            return instantiate(config["env"])


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--inspect-only", action="store_true")
            args = parser.parse_args()

            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

            if not args.inspect_only:
                run_count = _read_run_count()
                if run_count >= MAX_RUNS:
                    print(f"Simulation run budget exhausted: {{run_count}}/{{MAX_RUNS}}", file=sys.stderr)
                    return 2
                run_idx = run_count + 1
                _write_run_count(run_idx)
            else:
                run_idx = _read_run_count()

            env = None
            try:
                env = _make_env()
                obs, info = env.reset(seed=TRIAL_SEED)
                task_language, full_prompt = _patch_libero_goal(env, obs, info)
                (ARTIFACTS_DIR / "task_prompt.md").write_text(str(task_language), encoding="utf-8")
                if full_prompt is not None:
                    (ARTIFACTS_DIR / "full_prompt.json").write_text(
                        json.dumps(full_prompt, indent=2, default=repr),
                        encoding="utf-8",
                    )

                _save_image(env.render(), ARTIFACTS_DIR / "initial_agentview.png")
                wrist = env.render_wrist() if hasattr(env, "render_wrist") else None
                _save_image(wrist, ARTIFACTS_DIR / "initial_wrist.png")

                if args.inspect_only:
                    print(task_language)
                    return 0

                if RECORD_VIDEO and hasattr(env, "enable_video_capture"):
                    env.enable_video_capture(True, clear=True)

                code = SOLUTION_PATH.read_text(encoding="utf-8")
                obs_step, reward, terminated, truncated, info_step = env.step(code)
                task_completed = info_step.get("task_completed")

                stdout = info_step.get("stdout", "")
                stderr = info_step.get("stderr", "")
                (ARTIFACTS_DIR / f"stdout_run_{{run_idx:02d}}.txt").write_text(stdout, encoding="utf-8")
                (ARTIFACTS_DIR / f"stderr_run_{{run_idx:02d}}.txt").write_text(stderr, encoding="utf-8")
                shutil.copyfile(ARTIFACTS_DIR / f"stdout_run_{{run_idx:02d}}.txt", ARTIFACTS_DIR / "stdout_latest.txt")
                shutil.copyfile(ARTIFACTS_DIR / f"stderr_run_{{run_idx:02d}}.txt", ARTIFACTS_DIR / "stderr_latest.txt")

                agentview_path = ARTIFACTS_DIR / f"agentview_run_{{run_idx:02d}}.png"
                wrist_path = ARTIFACTS_DIR / f"wrist_run_{{run_idx:02d}}.png"
                _save_image(env.render(), agentview_path)
                _save_image(env.render_wrist() if hasattr(env, "render_wrist") else None, wrist_path)
                shutil.copyfile(agentview_path, ARTIFACTS_DIR / "agentview_latest.png")
                if wrist_path.exists():
                    shutil.copyfile(wrist_path, ARTIFACTS_DIR / "wrist_latest.png")

                if RECORD_VIDEO and hasattr(env, "get_video_frames"):
                    frames = env.get_video_frames(clear=False)
                    if frames:
                        _write_video(frames, str(ARTIFACTS_DIR), suffix=f"run_{{run_idx:02d}}")

                result = {{
                    "run_idx": run_idx,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "sandbox_rc": int(info_step.get("sandbox_rc", 1)),
                    "task_completed": _jsonable(task_completed),
                    "task_prompt": str(task_language),
                    "result": _jsonable(getattr(env, "_exec_globals", {{}}).get("RESULT")),
                }}
                (ARTIFACTS_DIR / f"result_run_{{run_idx:02d}}.json").write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )
                (ARTIFACTS_DIR / "result.json").write_text(
                    json.dumps(result, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(result, indent=2))
                return 0 if result["sandbox_rc"] == 0 else 1
            except BaseException:
                traceback.print_exc()
                return 1
            finally:
                if env is not None and hasattr(env, "close"):
                    env.close()


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )
