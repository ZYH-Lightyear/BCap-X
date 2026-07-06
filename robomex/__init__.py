"""RoboMEx:robotic multimodal executable skills(机器人多模态可执行技能)。

技能驱动、自进化的机器人 Coding Agent。对外入口是 :mod:`robomex.core` 里的框架接线:
构造一个 :class:`RoboMExConfig` 并运行一个 :class:`RoboMExAgent`。内部实现分布在
``core/``(内核:sandbox + coder)、``agents/``(角色 agent)、
``skills/``(载体 + store + 内置技能包)和 ``perception/``。
"""

from robomex.core import (
    AttemptHistory,
    AttemptRecord,
    Diagnosis,
    DiagnosisStore,
    EpisodeResult,
    EvidenceTimeline,
    EvidenceTimelineItem,
    LocalVerdict,
    PrimitiveTrace,
    RoboMExAgent,
    RoboMExConfig,
    StateFact,
    StatePatch,
    TraceStore,
    WorkspaceArtifact,
    WorldState,
)
from robomex.core.logging import configure_logging, get_logger

__all__ = [
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
    "StatePatch",
    "TraceStore",
    "WorkspaceArtifact",
    "WorldState",
    "configure_logging",
    "get_logger",
]
