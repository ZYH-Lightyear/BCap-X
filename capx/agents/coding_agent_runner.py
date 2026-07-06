"""Compatibility exports for the CaP SkillBench agent runner."""

from __future__ import annotations

from capskillbench.agents.runner import (
    AgentRunResult,
    run_codex_exec,
    run_coding_agent,
    run_opencode,
    run_opencode_streaming,
)

__all__ = [
    "AgentRunResult",
    "run_codex_exec",
    "run_coding_agent",
    "run_opencode",
    "run_opencode_streaming",
]

