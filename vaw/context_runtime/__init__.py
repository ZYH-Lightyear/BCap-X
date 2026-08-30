"""The revision-local VAW Context Runtime and its public contract."""

from vaw.context_runtime.context_projection import (
    ActionRecency,
    EmbodiedStateCard,
    project_embodied_state,
    render_main_context,
)
from vaw.context_runtime.memory import (
    InteractionEvent,
    InteractionMemory,
)
from vaw.context_runtime.model import (
    ActionPrediction,
    ActionProposal,
    ActionSeed,
    ActionTarget,
    ContextState,
    ImaginationSession,
    LastPhysicalAction,
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
from vaw.context_runtime.protocol import (
    FUNCTION_NAMES,
    IMAGINATION_FUNCTION_NAMES,
    IMAGINATION_SYSTEM_PROMPT,
    MAIN_FUNCTION_NAMES,
    ROBOT_FUNCTION_NAMES,
    SYSTEM_PROMPT,
    FunctionRegistry,
    FunctionSpec,
    function_definitions,
    imagination_function_definitions,
    main_function_definitions,
    main_function_registry,
)
from vaw.context_runtime.runtime import (
    ContextRunConfig,
    ContextRuntime,
    ImaginationRunner,
    run_context_episode,
)
from vaw.context_runtime.workspace import ContextStepResult, ContextWorkspace

__all__ = [
    "ActionPrediction",
    "ActionRecency",
    "ActionSeed",
    "ActionTarget",
    "ContextState",
    "ContextCompiler",
    "ContextPacket",
    "ContextRunConfig",
    "ContextRuntime",
    "ContextStepResult",
    "ContextWorkspace",
    "CuroboMotionBackend",
    "FUNCTION_NAMES",
    "EmbodiedStateCard",
    "FunctionRegistry",
    "FunctionSpec",
    "IMAGINATION_FUNCTION_NAMES",
    "IMAGINATION_SYSTEM_PROMPT",
    "ImaginationRunner",
    "LastPhysicalAction",
    "MAIN_FUNCTION_NAMES",
    "ROBOT_FUNCTION_NAMES",
    "MotionBackend",
    "MotionPlan",
    "PointEvidence",
    "ActionProposal",
    "InteractionEvent",
    "InteractionMemory",
    "Pose",
    "PyrokiMotionBackend",
    "RegionEvidence",
    "ImaginationSession",
    "RobotState",
    "SYSTEM_PROMPT",
    "function_definitions",
    "imagination_function_definitions",
    "main_function_definitions",
    "main_function_registry",
    "project_embodied_state",
    "render_main_context",
    "run_context_episode",
]
