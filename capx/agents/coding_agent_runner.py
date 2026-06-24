from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class AgentRunResult:
    agent: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float


def run_codex_exec(
    *,
    workspace: Path,
    model: str,
    prompt: str,
    codex_bin: str = "codex",
    timeout_s: int | None = None,
    extra_args: list[str] | None = None,
) -> AgentRunResult:
    """Run a single autonomous Codex CLI session inside a trial workspace."""
    command = [
        codex_bin,
        "exec",
        "--model",
        model,
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
        "-",
    ]
    if extra_args:
        command.extend(extra_args)

    started = time.time()
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        cwd=workspace,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    return AgentRunResult(
        agent="codex",
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_s=time.time() - started,
    )


def run_opencode(
    *,
    workspace: Path,
    model: str,
    prompt: str,
    opencode_bin: str = "opencode",
    timeout_s: int | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> AgentRunResult:
    """Run a single autonomous OpenCode CLI session inside a trial workspace."""
    command = [
        opencode_bin,
        "run",
        "--model",
        model,
        "--format",
        "json",
        "--dir",
        str(workspace),
        "--dangerously-skip-permissions",
    ]
    if extra_args:
        command.extend(extra_args)
    command.append(prompt)

    started = time.time()
    try:
        completed = subprocess.run(
            command,
            text=True,
            cwd=workspace,
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
        stderr += f"\nOpenCode killed after timeout_s={timeout_s}.\n"

    return AgentRunResult(
        agent="opencode",
        command=command[:-1] + ["<prompt>"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_s=time.time() - started,
    )


def run_opencode_streaming(
    *,
    workspace: Path,
    model: str,
    prompt: str,
    opencode_bin: str = "opencode",
    timeout_s: int | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> AgentRunResult:
    """Run OpenCode and stream raw JSONL events to stdout while capturing them."""
    command = [
        opencode_bin,
        "run",
        "--model",
        model,
        "--format",
        "json",
        "--dir",
        str(workspace),
        "--dangerously-skip-permissions",
    ]
    if extra_args:
        command.extend(extra_args)
    command.append(prompt)

    started = time.time()
    process = subprocess.Popen(
        command,
        text=True,
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    stdout = ""
    stderr = ""
    deadline = None if timeout_s is None else started + timeout_s
    assert process.stdout is not None
    assert process.stderr is not None

    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text("", encoding="utf-8")

    stderr_chunks: list[str] = []

    def _drain_stderr() -> None:
        err_file = stderr_path.open("a", encoding="utf-8") if stderr_path is not None else None
        try:
            for err_line in process.stderr:
                stderr_chunks.append(err_line)
                if err_file is not None:
                    err_file.write(err_line)
                    err_file.flush()
        finally:
            if err_file is not None:
                err_file.close()

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    stdout_file = stdout_path.open("a", encoding="utf-8") if stdout_path is not None else None
    timed_out = False
    while True:
        line = process.stdout.readline()
        if line:
            print(line, end="", flush=True)
            stdout += line
            if stdout_file is not None:
                stdout_file.write(line)
                stdout_file.flush()
        if process.poll() is not None:
            break
        now = time.time()
        if deadline is not None and now >= deadline:
            process.kill()
            timed_out = True
            stderr += f"\nOpenCode killed after timeout_s={timeout_s}.\n"
            break
        if not line:
            time.sleep(0.1)

    process.wait(timeout=5)
    remaining = process.stdout.read()
    if remaining:
        print(remaining, end="", flush=True)
        stdout += remaining
        if stdout_file is not None:
            stdout_file.write(remaining)
    if stdout_file is not None:
        stdout_file.close()
    stderr_thread.join(timeout=1.0)
    stderr += "".join(stderr_chunks)
    if timed_out and stderr_path is not None:
        with stderr_path.open("a", encoding="utf-8") as err_file:
            err_file.write(f"\nOpenCode killed after timeout_s={timeout_s}.\n")

    returncode = process.returncode
    if returncode is None:
        returncode = 124

    return AgentRunResult(
        agent="opencode",
        command=command[:-1] + ["<prompt>"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_s=time.time() - started,
    )


def run_coding_agent(
    *,
    agent: Literal["codex", "opencode"],
    workspace: Path,
    model: str,
    prompt: str,
    codex_bin: str = "codex",
    opencode_bin: str = "opencode",
    timeout_s: int | None = None,
    env: dict[str, str] | None = None,
    stream: bool = False,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> AgentRunResult:
    if agent == "codex":
        return run_codex_exec(
            workspace=workspace,
            model=model,
            prompt=prompt,
            codex_bin=codex_bin,
            timeout_s=timeout_s,
        )
    if agent == "opencode":
        if stream:
            return run_opencode_streaming(
                workspace=workspace,
                model=model,
                prompt=prompt,
                opencode_bin=opencode_bin,
                timeout_s=timeout_s,
                env=env,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        return run_opencode(
            workspace=workspace,
            model=model,
            prompt=prompt,
            opencode_bin=opencode_bin,
            timeout_s=timeout_s,
            env=env,
        )
    raise ValueError(f"unsupported agent: {agent}")
