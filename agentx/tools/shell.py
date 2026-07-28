"""Shell 执行工具。"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any

from agentx.contracts import ToolKind, ToolResult, ToolValidationError
from agentx.tools.base import Invocation, Tool, ToolContext

DEFAULT_TIMEOUT_S = 120
MAX_TIMEOUT_S = 600
#: 单次命令进 prompt 的字符上限。比通用的 25000 更紧:shell 最容易吐出巨量日志,
#: 而其中有价值的部分几乎总在开头(命令回显)和结尾(报错)。
MAX_SHELL_OUTPUT_CHARS = 30_000


class ShellTool(Tool):
    name = "run_shell_command"
    description = (
        "Run a shell command in the workspace and return its stdout, stderr and "
        "exit code. Use this for builds, tests, git, and any other CLI work."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run."},
            "description": {
                "type": "string",
                "description": "One short sentence describing what this command does.",
            },
            "timeout_s": {
                "type": "integer",
                "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT_S}, max {MAX_TIMEOUT_S}).",
                "minimum": 1,
                "maximum": MAX_TIMEOUT_S,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    kind = ToolKind.EXECUTE
    max_output_chars = MAX_SHELL_OUTPUT_CHARS
    truncate_keep = "both"

    def create_invocation(self, params: dict[str, Any]) -> Invocation:
        if not str(params.get("command", "")).strip():
            raise ToolValidationError("command 不能为空")
        return _ShellInvocation(params)


class _ShellInvocation(Invocation):
    def describe(self) -> str:
        return self.params.get("description") or self.params["command"][:80]

    def execute(self, ctx: ToolContext) -> ToolResult:
        command = self.params["command"]
        timeout = min(int(self.params.get("timeout_s") or DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S)

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(ctx.root),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                # 独立进程组:超时后能连同 shell 拉起的子进程一起收掉,
                # 否则被 kill 的只有 sh 本身,真正干活的进程会变成孤儿继续跑。
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            _kill_process_group(exc)
            partial = _describe(
                _decode(exc.stdout), _decode(exc.stderr), exit_code=None
            )
            message = f"命令超时({timeout}s)被终止。\n\n{partial}"
            return ToolResult(llm_content=message, display=message, error="timeout")
        except OSError as exc:
            message = f"命令无法启动: {exc}"
            return ToolResult(llm_content=message, error=str(exc))

        body = _describe(completed.stdout, completed.stderr, completed.returncode)
        # 非零退出不是程序错误,是要交给模型判断的普通结果 —— 有时它本来就预期
        # 测试会失败。但要标成 error,让统计和熔断能看见。
        error = None if completed.returncode == 0 else f"exit code {completed.returncode}"
        return ToolResult(llm_content=body, display=body, error=error)


def _kill_process_group(exc: subprocess.TimeoutExpired) -> None:
    """尽力收掉超时命令留下的整个进程组。"""
    pid = getattr(exc, "pid", None)
    if pid is None:
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _decode(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _describe(stdout: str, stderr: str, exit_code: int | None) -> str:
    parts = [f"exit_code: {exit_code if exit_code is not None else '(killed)'}"]
    parts.append(f"stdout:\n{stdout}" if stdout.strip() else "stdout: (empty)")
    parts.append(f"stderr:\n{stderr}" if stderr.strip() else "stderr: (empty)")
    return "\n\n".join(parts)


__all__ = ["ShellTool"]
