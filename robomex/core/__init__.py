"""框架内核:沙箱、共享 coder,上下文,以及接线入口。

``session`` 里的接线入口(:class:`RoboMExAgent` 等)按需惰性解析:它反向依赖
``robomex.authoring``,若在包初始化时急切导入会形成
``authoring -> agents -> core -> session -> authoring`` 的循环。
"""

from __future__ import annotations

from typing import Any

from robomex.core.context import (
    ArtifactRef,
    AttemptHistory,
    AttemptRecord,
    Diagnosis,
    DiagnosisStore,
    EvidencePacket,
    EvidenceTimeline,
    EvidenceTimelineItem,
    LocalVerdict,
    PrimitiveTrace,
    StateFact,
    TraceStore,
)

__all__ = [
    "AttemptHistory",
    "AttemptRecord",
    "ArtifactRef",
    "Diagnosis",
    "DiagnosisStore",
    "EvidencePacket",
    "EvidenceTimeline",
    "EvidenceTimelineItem",
    "EpisodeResult",
    "LocalVerdict",
    "PrimitiveTrace",
    "RoboMExAgent",
    "RoboMExConfig",
    "StateFact",
    "TraceStore",
]

_LAZY_SESSION = {"EpisodeResult", "RoboMExAgent", "RoboMExConfig"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SESSION:
        from robomex.core import session as _session

        value = getattr(_session, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
