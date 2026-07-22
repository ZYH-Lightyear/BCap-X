"""Type-safe access to the authoritative actor invocation contracts.

Actor lifecycle models remain owned by :mod:`robomex.orchestration.actors`.
This module intentionally re-exports those exact classes instead of creating a
second schema that could drift from runtime authority checks.
"""

from robomex.orchestration.actors import (
    ActorLifecycle,
    ActorProfile,
    InvocationSpec,
    IsolationPolicy,
    WorkspaceMode,
)

__all__ = [
    "ActorLifecycle",
    "ActorProfile",
    "InvocationSpec",
    "IsolationPolicy",
    "WorkspaceMode",
]
