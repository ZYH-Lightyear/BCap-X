#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capx.agents.codex_libero_workspace import (
    WorkspaceSpec,
    prepare_codex_libero_workspace,
    render_agent_prompt,
)
from capx.agents.coding_agent_runner import run_coding_agent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an autonomous coding agent on CaP-X LIBERO trials.")
    parser.add_argument("--agent", choices=["codex", "opencode"], default="codex")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--model", default="gpt-5.2-codex")
    parser.add_argument(
        "--skill-mode",
        choices=["no-skill", "with-skill", "explicit-skill"],
        default="with-skill",
    )
    parser.add_argument("--skills-dir", default=".agents/skills")
    parser.add_argument("--total-trials", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=5, help="Maximum solution.py simulation runs per trial.")
    parser.add_argument("--output-dir", default="outputs/codex_libero")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--opencode-bin", default="/mnt/data/zyh/bin/opencode")
    parser.add_argument("--agent-timeout-seconds", type=int, default=None)
    parser.add_argument("--codex-timeout-seconds", type=int, default=None, help="Deprecated alias for --agent-timeout-seconds.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-start-api-servers", action="store_true")
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream-agent-logs", action="store_true", help="Stream raw agent JSONL events to stdout while also saving them under workspace/logs.")
    return parser.parse_args()


def _read_result(workspace: Path) -> dict[str, Any] | None:
    result_path = workspace / "artifacts" / "result.json"
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def _load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.unsafe_load(f)


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


def _agent_env(args: argparse.Namespace, project_root: Path, agent_home: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if args.agent == "opencode":
        home = agent_home or (project_root / "outputs" / ".opencode_home")
        data_home = home / ".local" / "share"
        config_home = home / ".config"
        cache_home = home / ".cache"
        for path in (home, data_home, config_home, cache_home):
            path.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        env["XDG_DATA_HOME"] = str(data_home)
        env["XDG_CONFIG_HOME"] = str(config_home)
        env["XDG_CACHE_HOME"] = str(cache_home)
        openrouter_key = _read_openrouter_key(project_root)
        if openrouter_key:
            env["OPENROUTER_API_KEY"] = openrouter_key
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
            prepare_codex_libero_workspace(spec)
            run_env = _agent_env(args, project_root, trial_dir / "agent_home")

            summary: dict[str, Any] = {
                "trial": trial,
                "workspace": str(workspace),
                "skill_mode": args.skill_mode,
                "agent": args.agent,
                "prepared": True,
            }

            if not args.prepare_only:
                prompt = render_agent_prompt(
                    agent=args.agent,
                    skill_mode=args.skill_mode,
                    max_runs=args.max_turns,
                )
                run_result = run_coding_agent(
                    agent=args.agent,
                    workspace=workspace,
                    model=args.model,
                    prompt=prompt,
                    codex_bin=args.codex_bin,
                    opencode_bin=args.opencode_bin,
                    timeout_s=agent_timeout,
                    env=run_env,
                    stream=args.stream_agent_logs,
                    stdout_path=workspace / "logs" / f"{args.agent}_stdout.jsonl",
                    stderr_path=workspace / "logs" / f"{args.agent}_stderr.txt",
                )
                (workspace / "logs" / f"{args.agent}_stdout.jsonl").write_text(
                    run_result.stdout,
                    encoding="utf-8",
                )
                (workspace / "logs" / f"{args.agent}_stderr.txt").write_text(
                    run_result.stderr,
                    encoding="utf-8",
                )
                summary.update(
                    {
                        "agent_returncode": run_result.returncode,
                        "agent_duration_s": run_result.duration_s,
                        "agent_command": run_result.command,
                        "result": _read_result(workspace),
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
