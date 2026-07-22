"""Versioned control/data protocols for the RoboMEx v2 runtime.

The v1 authoring graph remains in :mod:`robomex.authoring`.  Nothing in this
package is a compatibility rewrite of that executor.
"""

from robomex.elastic.compiler import CompiledElasticGraph, ElasticGraphCompiler
from robomex.elastic.graph_spec import (
    ACTION_SNAPSHOT_RUNNER_REF,
    ACTION_SNAPSHOT_SCHEMA_ID,
    ActivationLane,
    ActivationSpec,
    ArtifactBinding,
    BoundedLoopSpec,
    EffectScope,
    ElasticGraphSpec,
    ExecutionBudget,
    ExternalBinding,
    LifecycleScope,
    PortSpecV2,
    RunnerKind,
    TransitionSpec,
)

# graph_patch integrates with runtime.activation, while runtime.activation
# necessarily imports the compiler above.  Lazy public exports keep both
# direct import orders valid instead of relying on callers to import the
# package in one particular order.
_PATCH_EXPORTS = frozenset(
    {
        "BarrierProfile",
        "ClosedSlot",
        "ComposableFrontier",
        "ControlExitCut",
        "FragmentExitRoute",
        "FragmentInputRoute",
        "FragmentOutputRoute",
        "GraphFragment",
        "GraphPatchCommit",
        "GraphPatchCoordinator",
        "GraphPatchProposal",
        "GraphVersionRecord",
        "OuterLoopReplacement",
        "PatchOperation",
        "PatchReceipt",
        "PatchRejectCode",
        "TypedInputCut",
        "TypedOutputCut",
        "VerifierObligation",
        "derive_closed_slot",
    }
)


def __getattr__(name: str):
    if name not in _PATCH_EXPORTS:
        raise AttributeError(name)
    from robomex.elastic import graph_patch

    return getattr(graph_patch, name)

__all__ = [
    "ACTION_SNAPSHOT_RUNNER_REF",
    "ACTION_SNAPSHOT_SCHEMA_ID",
    "ActivationLane",
    "ActivationSpec",
    "ArtifactBinding",
    "BarrierProfile",
    "BoundedLoopSpec",
    "ClosedSlot",
    "CompiledElasticGraph",
    "ComposableFrontier",
    "ControlExitCut",
    "EffectScope",
    "ElasticGraphCompiler",
    "ElasticGraphSpec",
    "ExecutionBudget",
    "ExternalBinding",
    "FragmentExitRoute",
    "FragmentInputRoute",
    "FragmentOutputRoute",
    "GraphFragment",
    "GraphPatchCommit",
    "GraphPatchCoordinator",
    "GraphPatchProposal",
    "GraphVersionRecord",
    "LifecycleScope",
    "OuterLoopReplacement",
    "PatchOperation",
    "PatchReceipt",
    "PatchRejectCode",
    "PortSpecV2",
    "RunnerKind",
    "TransitionSpec",
    "TypedInputCut",
    "TypedOutputCut",
    "VerifierObligation",
    "derive_closed_slot",
]
