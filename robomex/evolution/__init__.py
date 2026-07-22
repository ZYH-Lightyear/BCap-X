"""Evolve-ready provenance and admission contracts.

The package contains no search, mutation, automatic promotion, or deployment
algorithm.  The v2 baseline uses ``CandidateAdmissionMode.BASELINE_ONLY``.
"""

from robomex.evolution.candidate import (
    CandidateComponentSnapshot,
    CandidateConfigSnapshot,
    EvolvableComponentKind,
    EvolvableComponentSpec,
    PromotionRecord,
    PromotionStage,
    SafetyBoundary,
    is_promotion_transition_allowed,
)
from robomex.evolution.evaluation import (
    EvaluationGateRecord,
    EvaluationReport,
    EvaluationRequest,
    Evaluator,
    GateStatus,
    MetricAggregation,
    MetricDefinition,
    MetricDirection,
    MetricRecord,
)
from robomex.evolution.manifest import (
    BackendPin,
    BackendRole,
    FunctionPin,
    ModelPin,
    PromptPin,
    RunBudgets,
    RunManifest,
    SkillPin,
    TaskSnapshot,
)
from robomex.evolution.registry import (
    CandidateAdmissionMode,
    DuplicateEvolutionIdError,
    EvolutionDisabledError,
    EvolutionHashDriftError,
    EvolutionRegistryError,
    EvolvableComponentRegistry,
    EvolvableRegistrySnapshot,
    SafetyBoundaryViolationError,
)

__all__ = [
    "BackendPin",
    "BackendRole",
    "CandidateAdmissionMode",
    "CandidateComponentSnapshot",
    "CandidateConfigSnapshot",
    "DuplicateEvolutionIdError",
    "EvaluationGateRecord",
    "EvaluationReport",
    "EvaluationRequest",
    "Evaluator",
    "EvolvableComponentKind",
    "EvolvableComponentRegistry",
    "EvolvableComponentSpec",
    "EvolvableRegistrySnapshot",
    "EvolutionDisabledError",
    "EvolutionHashDriftError",
    "EvolutionRegistryError",
    "FunctionPin",
    "GateStatus",
    "MetricAggregation",
    "MetricDefinition",
    "MetricDirection",
    "MetricRecord",
    "ModelPin",
    "PromotionRecord",
    "PromotionStage",
    "PromptPin",
    "RunBudgets",
    "RunManifest",
    "SafetyBoundary",
    "SafetyBoundaryViolationError",
    "SkillPin",
    "TaskSnapshot",
    "is_promotion_transition_allowed",
]
