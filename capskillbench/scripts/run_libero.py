#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capskillbench.libero_workspace import (
    WorkspaceSpec,
    prepare_skillbench_workspace,
    render_agent_prompt,
)
from capskillbench.agents.runner import run_coding_agent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an autonomous coding agent on CaP-X SkillBench LIBERO-profile trials.")
    parser.add_argument("--agent", choices=["codex", "opencode", "claude"], default="codex")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--model", default="gpt-5.2-codex")
    parser.add_argument(
        "--skill-mode",
        choices=["no-skill", "with-skill", "explicit-skill"],
        default="with-skill",
    )
    parser.add_argument("--skills-dir", default="capskillbench/skills")
    parser.add_argument("--total-trials", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=5, help="Maximum solution.py simulation runs per trial.")
    parser.add_argument("--output-dir", default="outputs/codex_libero")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--opencode-bin", default="/mnt/data/zyh/bin/opencode")
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--claude-max-turns", type=int, default=60, help="Claude Code agentic turn limit for print mode.")
    parser.add_argument("--agent-timeout-seconds", type=int, default=None)
    parser.add_argument("--codex-timeout-seconds", type=int, default=None, help="Deprecated alias for --agent-timeout-seconds.")
    parser.add_argument("--agent-retries", type=int, default=0, help="Retry the coding agent after CLI/provider failures without resetting the workspace.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-preinspect", action="store_true", help="Do not materialize task_prompt/images before launching the coding agent.")
    parser.add_argument("--no-start-api-servers", action="store_true")
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream-agent-logs", action="store_true", help="Stream raw agent JSONL events to stdout while also saving them under workspace/logs.")
    parser.add_argument(
        "--point-backend",
        choices=["auto", "molmo", "qwen", "vlm", "openrouter"],
        default="auto",
        help="Backend for the unified ground_point helper.",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="Vision-language model for query_vlm/Qwen point fallback. Defaults to --model.",
    )
    return parser.parse_args()


def _read_result(workspace: Path) -> dict[str, Any] | None:
    result_path = workspace / "artifacts" / "result.json"
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def _looks_like_transient_agent_failure(run_result_stdout: str, run_result_stderr: str) -> bool:
    combined = f"{run_result_stdout}\n{run_result_stderr}"
    transient_markers = (
        "response_failed",
        "stream failed",
        "stream disconnected",
        "Connection reset",
        "Connection refused",
        "timeout",
        "temporarily unavailable",
    )
    return any(marker.lower() in combined.lower() for marker in transient_markers)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def _run_preinspect(workspace: Path, env: dict[str, str], timeout_s: int | None) -> dict[str, Any]:
    """Materialize task prompt and initial images before launching the agent."""
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "tools/run_solution.py", "--inspect-only"]
    print(f"[preinspect] materializing task prompt/images in {workspace}", flush=True)
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            text=True,
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        stderr += f"\nPreinspect killed after timeout_s={timeout_s}.\n"

    (logs_dir / "preinspect_stdout.txt").write_text(stdout, encoding="utf-8")
    (logs_dir / "preinspect_stderr.txt").write_text(stderr, encoding="utf-8")
    task_prompt_path = workspace / "artifacts" / "task_prompt.md"
    if returncode == 0 and task_prompt_path.exists():
        task_prompt = task_prompt_path.read_text(encoding="utf-8").strip()
        task_md = workspace / "task.md"
        with task_md.open("a", encoding="utf-8") as f:
            f.write("\n## Concrete Task\n\n")
            f.write(task_prompt or "(empty task prompt)")
            f.write("\n")
    print(
        f"[preinspect] returncode={returncode} duration_s={time.time() - started:.1f}",
        flush=True,
    )
    return {
        "command": command,
        "returncode": returncode,
        "duration_s": time.time() - started,
        "stdout_path": str(logs_dir / "preinspect_stdout.txt"),
        "stderr_path": str(logs_dir / "preinspect_stderr.txt"),
    }


def _read_openrouter_key(project_root: Path) -> str | None:
    for env_name in ("OPENROUTER_API_KEY", "OPENROUTER_KEY"):
        value = os.environ.get(env_name)
        if value:
            return value.strip()

    env_path = project_root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() in {"OPENROUTER_API_KEY", "OPENROUTER_KEY"}:
                return value.strip().strip("\"'")

    key_path = project_root / ".openrouterkey"
    if key_path.exists():
        value = key_path.read_text(encoding="utf-8").strip()
        return value or None
    return None


def _read_env_value(project_root: Path, names: set[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()

    env_path = project_root / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if key in names:
            return value.strip().strip("\"'")
    return None


def _copy_codex_config(target_codex_home: Path) -> None:
    """Prepare an isolated Codex home with config only, leaving secrets in env."""
    target_codex_home.mkdir(parents=True, exist_ok=True)
    source_homes: list[Path] = []
    if os.environ.get("CODEX_HOME"):
        source_homes.append(Path(os.environ["CODEX_HOME"]).expanduser())
    source_homes.append(Path.home() / ".codex")

    for source_home in source_homes:
        source_config = source_home / "config.toml"
        if source_config.exists():
            shutil.copy2(source_config, target_codex_home / "config.toml")
            break


def _write_agent_shell_profile(home: Path, venv_dir: Path) -> None:
    """Keep agent-launched login shells pinned to the benchmark Python env."""
    lines = [
        "# Generated by capskillbench. Keeps agent shell commands reproducible.",
    ]
    if (venv_dir / "bin").exists():
        lines.extend(
            [
                f'export VIRTUAL_ENV="{venv_dir}"',
                f'export PATH="{venv_dir / "bin"}:$PATH"',
            ]
        )
    content = "\n".join(lines) + "\n"
    for profile_name in (".bash_profile", ".bashrc", ".profile"):
        (home / profile_name).write_text(content, encoding="utf-8")


def _write_libero_config(home: Path, project_root: Path) -> None:
    """Avoid LIBERO's first-run interactive config prompt in isolated homes."""
    libero_root = project_root / "capx" / "third_party" / "LIBERO-PRO" / "libero" / "libero"
    config_dir = home / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(libero_root),
        "bddl_files": str(libero_root / "bddl_files"),
        "init_states": str(libero_root / "init_files"),
        "datasets": str(libero_root.parent / "datasets"),
        "assets": str(libero_root / "assets"),
    }
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )


def _write_claude_onboarding_config(home: Path) -> None:
    """Allow isolated Claude Code homes to run non-interactively."""
    config_path = home / ".claude.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}
    config["hasCompletedOnboarding"] = True
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _agent_env(args: argparse.Namespace, project_root: Path, agent_home: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    venv_dir = project_root / ".venv-libero"
    path_prefixes: list[str] = []
    if (venv_dir / "bin").exists():
        env["VIRTUAL_ENV"] = str(venv_dir)
        path_prefixes.append(str(venv_dir / "bin"))
    for candidate in (
        project_root.parent / "npm-global" / "bin",
        Path("/etc/dsw/runtime/node/bin"),
    ):
        if candidate.exists():
            path_prefixes.append(str(candidate))
    if path_prefixes:
        env["PATH"] = os.pathsep.join(path_prefixes + [env.get("PATH", "")])
    env["CAPX_POINT_BACKEND"] = args.point_backend
    env["CAPX_VLM_MODEL"] = args.vlm_model or args.model
    env.setdefault("CAPX_VLM_SERVER_URL", "http://localhost:8110/chat/completions")
    env.setdefault("CAPX_QWEN_VLM_BASE_URL", "http://127.0.0.1:8110")
    openrouter_key = _read_openrouter_key(project_root)
    if openrouter_key:
        env["OPENROUTER_API_KEY"] = openrouter_key
        env["CAPX_VLM_API_KEY"] = openrouter_key
    v_api_key = _read_env_value(project_root, {"V_API_KEY", "ANTHROPIC_AUTH_TOKEN"})
    if v_api_key:
        env.setdefault("ANTHROPIC_AUTH_TOKEN", v_api_key)
        env.setdefault("OPENAI_API_KEY", v_api_key)
    if args.agent == "codex":
        home = agent_home or (project_root / "outputs" / ".codex_home")
        data_home = home / ".local" / "share"
        config_home = home / ".config"
        cache_home = home / ".cache"
        codex_home = home / ".codex"
        for path in (home, data_home, config_home, cache_home, codex_home):
            path.mkdir(parents=True, exist_ok=True)
        _write_agent_shell_profile(home, venv_dir)
        _write_libero_config(home, project_root)
        _copy_codex_config(codex_home)
        env["HOME"] = str(home)
        env["XDG_DATA_HOME"] = str(data_home)
        env["XDG_CONFIG_HOME"] = str(config_home)
        env["XDG_CACHE_HOME"] = str(cache_home)
        env["CODEX_HOME"] = str(codex_home)
    if args.agent == "opencode":
        home = agent_home or (project_root / "outputs" / ".opencode_home")
        data_home = home / ".local" / "share"
        config_home = home / ".config"
        cache_home = home / ".cache"
        for path in (home, data_home, config_home, cache_home):
            path.mkdir(parents=True, exist_ok=True)
        _write_agent_shell_profile(home, venv_dir)
        _write_libero_config(home, project_root)
        env["HOME"] = str(home)
        env["XDG_DATA_HOME"] = str(data_home)
        env["XDG_CONFIG_HOME"] = str(config_home)
        env["XDG_CACHE_HOME"] = str(cache_home)
    if args.agent == "claude":
        home = agent_home or (project_root / "outputs" / ".claude_home")
        data_home = home / ".local" / "share"
        config_home = home / ".config"
        cache_home = home / ".cache"
        claude_home = home / ".claude"
        for path in (home, data_home, config_home, cache_home, claude_home):
            path.mkdir(parents=True, exist_ok=True)
        _write_agent_shell_profile(home, venv_dir)
        _write_libero_config(home, project_root)
        _write_claude_onboarding_config(home)
        env["HOME"] = str(home)
        env["XDG_DATA_HOME"] = str(data_home)
        env["XDG_CONFIG_HOME"] = str(config_home)
        env["XDG_CACHE_HOME"] = str(cache_home)
        env.setdefault("ANTHROPIC_BASE_URL", "https://api.v3.cm")
        env.setdefault("ANTHROPIC_MODEL", args.model)
        env.setdefault("ANTHROPIC_SMALL_FAST_MODEL", "claude-haiku-4-5-20251001")
        env.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")
        env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "32000")
        env.setdefault("CLAUDE_CODE_SKIP_PROMPT_HISTORY", "1")
    return env


def main() -> int:
    args = _parse_args()
    project_root = Path.cwd().resolve()
    config_path = (project_root / args.config_path).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    skills_dir = (project_root / args.skills_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skill_mode != "no-skill" and not skills_dir.exists():
        raise FileNotFoundError(f"skills dir not found: {skills_dir}")

    config = _load_yaml_config(config_path)
    agent_timeout = args.agent_timeout_seconds
    if agent_timeout is None and args.codex_timeout_seconds is not None:
        agent_timeout = args.codex_timeout_seconds
    server_procs: list[Any] = []
    summaries: list[dict[str, Any]] = []
    had_failure = False

    try:
        if not args.prepare_only and not args.no_start_api_servers:
            from capx.envs.runner import _start_api_servers

            server_procs = _start_api_servers(config.get("api_servers"))

        for trial in range(1, args.total_trials + 1):
            trial_dir = output_dir / f"trial_{trial:03d}"
            workspace = trial_dir / "workspace"
            spec = WorkspaceSpec(
                workspace=workspace,
                project_root=project_root,
                config_path=config_path,
                skills_dir=None if args.skill_mode == "no-skill" else skills_dir,
                skill_mode=args.skill_mode,
                agent=args.agent,
                max_runs=args.max_turns,
                trial_seed=trial,
                record_video=args.record_video,
            )
            prepare_skillbench_workspace(spec)
            run_env = _agent_env(args, project_root, trial_dir / "agent_home")

            summary: dict[str, Any] = {
                "trial": trial,
                "workspace": str(workspace),
                "agent_home": str(trial_dir / "agent_home"),
                "skill_mode": args.skill_mode,
                "agent": args.agent,
                "prepared": True,
            }

            if not args.prepare_only:
                should_run_agent = True
                if not args.no_preinspect:
                    inspect_result = _run_preinspect(workspace, run_env, agent_timeout)
                    summary["preinspect"] = inspect_result
                    if inspect_result["returncode"] != 0:
                        summary["harness_status"] = "inspect_failed"
                        summary["result"] = None
                        had_failure = True
                        should_run_agent = False

                if should_run_agent:
                    prompt = render_agent_prompt(
                        agent=args.agent,
                        skill_mode=args.skill_mode,
                        max_runs=args.max_turns,
                    )
                    agent_attempts: list[dict[str, Any]] = []
                    run_result = None
                    for agent_attempt in range(1, args.agent_retries + 2):
                        if agent_attempt > 1:
                            print(
                                f"[agent] retry {agent_attempt - 1}/{args.agent_retries} after transient failure",
                                flush=True,
                            )
                        run_result = run_coding_agent(
                            agent=args.agent,
                            workspace=workspace,
                            model=args.model,
                            prompt=prompt,
                            codex_bin=args.codex_bin,
                            opencode_bin=args.opencode_bin,
                            claude_bin=args.claude_bin,
                            claude_max_turns=args.claude_max_turns,
                            timeout_s=agent_timeout,
                            env=run_env,
                            stream=args.stream_agent_logs,
                            stdout_path=workspace / "logs" / f"{args.agent}_stdout.jsonl",
                            stderr_path=workspace / "logs" / f"{args.agent}_stderr.txt",
                        )
                        agent_attempts.append(
                            {
                                "attempt": agent_attempt,
                                "returncode": run_result.returncode,
                                "duration_s": run_result.duration_s,
                                "transient_failure": _looks_like_transient_agent_failure(
                                    run_result.stdout,
                                    run_result.stderr,
                                ),
                            }
                        )
                        if run_result.returncode == 0:
                            break
                        if agent_attempt > args.agent_retries:
                            break
                        if not agent_attempts[-1]["transient_failure"]:
                            break

                    assert run_result is not None
                    (workspace / "logs" / f"{args.agent}_stdout.jsonl").write_text(
                        run_result.stdout,
                        encoding="utf-8",
                    )
                    (workspace / "logs" / f"{args.agent}_stderr.txt").write_text(
                        run_result.stderr,
                        encoding="utf-8",
                    )
                    result = _read_result(workspace)
                    harness_status = "ok"
                    if run_result.returncode != 0:
                        harness_status = "agent_failed"
                        had_failure = True
                    elif result is None:
                        harness_status = "no_result"
                        had_failure = True
                    summary.update(
                        {
                            "agent_returncode": run_result.returncode,
                            "agent_duration_s": run_result.duration_s,
                            "agent_command": run_result.command,
                            "agent_attempts": agent_attempts,
                            "harness_status": harness_status,
                            "result": result,
                        }
                    )

            summaries.append(summary)
            (output_dir / "summary.json").write_text(
                json.dumps(summaries, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(summary, indent=2))
    finally:
        if server_procs:
            from capx.envs.runner import _stop_api_servers

            _stop_api_servers(server_procs)

    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
