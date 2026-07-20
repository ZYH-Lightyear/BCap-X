"""RoboMEx:robotic multimodal executable skills(机器人多模态可执行技能)。

技能驱动、自进化的机器人 Coding Agent。对外入口是 :mod:`robomex.core` 里的框架接线:
构造一个 :class:`RoboMExConfig` 并运行一个 :class:`RoboMExAgent`。内部实现分布在
``core/``(内核:sandbox + coder)、``agents/``(角色 agent)、
``skills/``(载体 + store + 内置技能包)和 ``perception/``。

Public names are resolved lazily so lightweight subpackages (e.g. ``robomex.web``)
can import without pulling the full runtime stack.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AttemptHistory",
    "AttemptRecord",
    "ArtifactRef",
    "Diagnosis",
    "DiagnosisStore",
    "EpisodeResult",
    "EvidenceTimeline",
    "EvidenceTimelineItem",
    "LocalVerdict",
    "PrimitiveTrace",
    "RoboMExAgent",
    "RoboMExConfig",
    "StateFact",
    "TraceStore",
    "configure_logging",
    "get_logger",
]

_LAZY_CORE = {
    "ArtifactRef",
    "AttemptHistory",
    "AttemptRecord",
    "Diagnosis",
    "DiagnosisStore",
    "EpisodeResult",
    "EvidenceTimeline",
    "EvidenceTimelineItem",
    "LocalVerdict",
    "PrimitiveTrace",
    "RoboMExAgent",
    "RoboMExConfig",
    "StateFact",
    "TraceStore",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_CORE:
        from robomex import core as _core

        value = getattr(_core, name)
        globals()[name] = value
        return value
    if name in {"configure_logging", "get_logger"}:
        from robomex.core import logging as _logging

        value = getattr(_logging, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
