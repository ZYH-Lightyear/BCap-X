"""Revision-local VAW Context Runtime.

This package is intentionally separate from the legacy ``vaw.Workspace``
path.  The hybrid runtime can therefore evolve its public contract without changing the
M0--M1.2 traces or renderers used as paper baselines.
"""

from vaw.context_runtime.model import (
    ActionAdjustment,
    ActionCandidate,
    ActionPrediction,
    ActionProposal,
    ContextState,
    ExecutionReceipt,
    FunctionRecord,
    PointEvidence,
    Pose,
    RegionEvidence,
    RobotState,
)
from vaw.context_runtime.motion import (
    CuroboMotionBackend,
    MotionBackend,
    MotionPlan,
    PyrokiMotionBackend,
)
from vaw.context_runtime.packet import ContextCompiler, ContextPacket
from vaw.context_runtime.protocol import FUNCTION_NAMES, SYSTEM_PROMPT, function_definitions
from vaw.context_runtime.runtime import (
    ContextRunConfig,
    ContextRuntime,
    run_context_episode,
)
from vaw.context_runtime.workspace import ContextStepResult, ContextWorkspace

__all__ = [
    "ActionAdjustment",
    "ActionCandidate",
    "ActionPrediction",
    "ActionProposal",
    "ContextState",
    "ContextCompiler",
    "ContextPacket",
    "ContextRunConfig",
    "ContextRuntime",
    "ContextStepResult",
    "ContextWorkspace",
    "CuroboMotionBackend",
    "ExecutionReceipt",
    "FUNCTION_NAMES",
    "FunctionRecord",
    "MotionBackend",
    "MotionPlan",
    "PointEvidence",
    "Pose",
    "PyrokiMotionBackend",
    "RegionEvidence",
    "RobotState",
    "SYSTEM_PROMPT",
    "function_definitions",
    "run_context_episode",
]
