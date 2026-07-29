"""Agent runtime for the Visual Action Workspace.

Forked from the ``agentx`` package in this repo (itself a port of Qwen-Code's
headless ``AgentCore``), keeping the provider / chat / loop split and dropping
everything tied to coding agents: the parallel tool scheduler, the shell and
file tools, skills, and the CLI.

Layout::

    contracts.py   data only: ToolCall / ModelResponse / StepRecord / EpisodeResult
    providers/     model endpoints: OpenAI-compatible + <tool_call> text protocol
    chat.py        history: orphaned-call repair + deterministic canvas window
    runtime.py     VAWRuntime: one op per turn, explicit done, forced termination
    teacher.py     provider config for the frontier teacher
    student.py     provider config for the Qwen3-VL student

Divergences from the upstream loop and their reasons are documented in
``runtime.py`` and in ``docs/vaw_implementation_plan.md`` §2.
"""

from vaw.agents.contracts import EpisodeResult, StepRecord, TerminateMode
from vaw.agents.runtime import RunConfig, VAWRuntime, run_episode

__all__ = [
    "EpisodeResult",
    "RunConfig",
    "StepRecord",
    "TerminateMode",
    "VAWRuntime",
    "run_episode",
]
