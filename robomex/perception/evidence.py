from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceKind(str, Enum):
    """Kinds of evidence RoboMEx records around execution."""

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
    """How an evidence artifact should be used by a skill branch or verifier."""

    BEFORE = "before"
    AFTER = "after"
    STATE_CUE = "state_cue"
    ACTION_CUE = "action_cue"
    VERIFICATION_CUE = "verification_cue"
    FAILURE_CUE = "failure_cue"


@dataclass(frozen=True)
class EvidenceArtifact:
    """A persisted or in-memory multimodal artifact."""

    artifact_id: str
    kind: EvidenceKind
    role: EvidenceRole
    path: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultimodalEvidenceBundle:
    """Evidence collected for a task, skill, or semantic action block."""

    bundle_id: str
    artifacts: tuple[EvidenceArtifact, ...] = ()
    task_id: str | None = None
    block_name: str | None = None
    skill_id: str | None = None

    def by_role(self, role: EvidenceRole) -> tuple[EvidenceArtifact, ...]:
        """Return artifacts with the requested role."""

        return tuple(artifact for artifact in self.artifacts if artifact.role == role)

    def by_kind(self, kind: EvidenceKind) -> tuple[EvidenceArtifact, ...]:
        """Return artifacts with the requested kind."""

        return tuple(artifact for artifact in self.artifacts if artifact.kind == kind)

