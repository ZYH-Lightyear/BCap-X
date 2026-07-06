"""CaP SkillBench harness package.

This package owns the autonomous coding-agent benchmark harness for CaP-X
manipulation environments. It intentionally reuses CapX environment/runtime
primitives while keeping benchmark workspace preparation and agent launching
outside `capx`.
"""

from capskillbench.workspace import WorkspaceSpec, prepare_skillbench_workspace

__all__ = ["WorkspaceSpec", "prepare_skillbench_workspace"]
