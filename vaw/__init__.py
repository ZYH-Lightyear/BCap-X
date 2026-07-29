"""VAW — Visual Action Workspace.

See vaw/README.md for architecture and milestones, and
docs/gui_as_policy_v2_cvpr_plan.md for the paper plan.
"""

from vaw.camera import ViewState, VirtualCamera
from vaw.cloud import SceneCloud
from vaw.state import ActionState
from vaw.types import Candidate, ObjectEntry, Pose, PreviewResult, Receipt, StepResult
from vaw.workspace import Workspace

__all__ = [
    "ActionState",
    "Candidate",
    "ObjectEntry",
    "Pose",
    "PreviewResult",
    "Receipt",
    "SceneCloud",
    "StepResult",
    "ViewState",
    "VirtualCamera",
    "Workspace",
]
