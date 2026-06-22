from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceKind(str, Enum):
    """RoboMEx 在执行前后记录的证据种类。"""

    RGB = "rgb"
    DEPTH = "depth"
    RGBD = "rgbd"
    WRIST_RGB = "wrist_rgb"
    WRIST_DEPTH = "wrist_depth"
    MASK = "mask"
    POINT_CLOUD = "point_cloud"
    GRASP_CANDIDATES = "grasp_candidates"
    VIDEO = "video"
    CODE_TRACE = "code_trace"
    API_OUTPUT = "api_output"


class EvidenceRole(str, Enum):
    """证据 artifact 应如何被技能分支或验证器使用。"""

    BEFORE = "before"
    AFTER = "after"
    STATE_CUE = "state_cue"
    ACTION_CUE = "action_cue"
    VERIFICATION_CUE = "verification_cue"
    FAILURE_CUE = "failure_cue"


@dataclass(frozen=True)
class EvidenceArtifact:
    """一个已持久化或在内存中的多模态 artifact。"""

    artifact_id: str
    kind: EvidenceKind
    role: EvidenceRole
    path: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultimodalEvidenceBundle:
    """为某个任务、技能或语义动作块采集到的证据集合。"""

    bundle_id: str
    artifacts: tuple[EvidenceArtifact, ...] = ()
    task_id: str | None = None
    block_name: str | None = None
    skill_id: str | None = None

    def by_role(self, role: EvidenceRole) -> tuple[EvidenceArtifact, ...]:
        """返回具有指定 role 的 artifact。"""

        return tuple(artifact for artifact in self.artifacts if artifact.role == role)

    def by_kind(self, kind: EvidenceKind) -> tuple[EvidenceArtifact, ...]:
        """返回具有指定 kind 的 artifact。"""

        return tuple(artifact for artifact in self.artifacts if artifact.kind == kind)

