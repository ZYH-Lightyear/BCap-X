"""RoboMEx v2 machine-authoritative contract surface."""

from robomex.contracts.actors import (
    ActorLifecycle,
    ActorProfile,
    InvocationSpec,
    IsolationPolicy,
    WorkspaceMode,
)
from robomex.contracts.catalog import (
    ContractCatalogError,
    ContractCatalogSnapshot,
    ContractHashDriftError,
    ContractRegistry,
    DuplicateContractError,
    MissingContractError,
)
from robomex.contracts.common import (
    ContentPin,
    ContractId,
    ContractValidationError,
    DigestStr,
    NonEmptyStr,
    canonical_model_digest,
    canonical_payload_digest,
    revalidate_sealed,
)
from robomex.contracts.effects import (
    DeclaredEffect,
    EffectContract,
    EffectScope,
    Reversibility,
    StatePredicate,
)
from robomex.contracts.protocol import (
    ObligationSeverity,
    ProtocolSpec,
    SlotCardinality,
    SlotPolicy,
    SlotSource,
    VerificationPhase,
    VerifierObligation,
)
from robomex.contracts.skill import (
    FunctionExport,
    SkillAsset,
    SkillAssetKind,
    SkillDependency,
    SkillManifest,
)
from robomex.contracts.skill_catalog_builder import (
    BuiltSkillContract,
    SkillCatalogBuildError,
    build_skill_contract_catalog,
)

__all__ = [
    "ActorLifecycle",
    "ActorProfile",
    "BuiltSkillContract",
    "ContentPin",
    "ContractCatalogError",
    "ContractCatalogSnapshot",
    "ContractHashDriftError",
    "ContractId",
    "ContractRegistry",
    "ContractValidationError",
    "DeclaredEffect",
    "DigestStr",
    "DuplicateContractError",
    "EffectContract",
    "EffectScope",
    "FunctionExport",
    "InvocationSpec",
    "IsolationPolicy",
    "MissingContractError",
    "NonEmptyStr",
    "ObligationSeverity",
    "ProtocolSpec",
    "Reversibility",
    "SkillAsset",
    "SkillAssetKind",
    "SkillCatalogBuildError",
    "SkillDependency",
    "SkillManifest",
    "SlotCardinality",
    "SlotPolicy",
    "SlotSource",
    "StatePredicate",
    "VerificationPhase",
    "VerifierObligation",
    "WorkspaceMode",
    "canonical_model_digest",
    "canonical_payload_digest",
    "build_skill_contract_catalog",
    "revalidate_sealed",
]
