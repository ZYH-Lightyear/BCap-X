"""Content-addressed skill package manifests."""

from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from robomex.contracts.common import (
    ContentPin,
    ContractId,
    DigestStr,
    NonEmptyStr,
    SealedContract,
    StrictContract,
    require_unique,
    require_unique_strings,
)


class SkillAssetKind(str, Enum):  # noqa: UP042 - Python 3.10 remains supported
    KNOWLEDGE = "knowledge"
    API = "api"
    CODE = "code"


class SkillAsset(StrictContract):
    """One immutable package-relative knowledge, API, or code asset."""

    asset_id: ContractId
    kind: SkillAssetKind
    relative_path: NonEmptyStr
    content_digest: DigestStr
    media_type: NonEmptyStr
    description: str = ""

    @model_validator(mode="after")
    def _safe_package_path(self) -> SkillAsset:
        if "\\" in self.relative_path:
            raise ValueError("Skill asset paths must use POSIX separators.")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or not path.parts or any(
            part in {"", ".", ".."} for part in path.parts
        ):
            raise ValueError("Skill asset path must be a safe package-relative path.")
        return self


class FunctionExport(StrictContract):
    """Function-level code pin exported by a code asset."""

    function_id: ContractId
    source_asset_id: ContractId
    entrypoint: NonEmptyStr
    function_digest: DigestStr
    interface_digest: DigestStr
    description: str = ""

    @model_validator(mode="after")
    def _entrypoint_shape(self) -> FunctionExport:
        module, separator, symbol = self.entrypoint.rpartition(":")
        if not separator or not module or not symbol or not symbol.isidentifier():
            raise ValueError(
                "entrypoint must use '<package-relative file>:<python_identifier>'."
            )
        path = PurePosixPath(module)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("Function entrypoint file must be package-relative.")
        return self


class SkillDependency(StrictContract):
    skill_id: ContractId
    minimum_revision: int = Field(ge=1)
    manifest_digest: DigestStr | None = None
    optional: bool = False


class SkillManifest(SealedContract):
    """Frozen, machine-authoritative inventory of a v2 skill package."""

    schema_version: Literal["robomex.skill_manifest.v2"] = "robomex.skill_manifest.v2"
    skill_id: ContractId
    revision: int = Field(default=1, ge=1)
    name: NonEmptyStr
    summary: NonEmptyStr
    assets: tuple[SkillAsset, ...] = Field(min_length=1)
    functions: tuple[FunctionExport, ...] = ()
    protocol_pins: tuple[ContentPin, ...] = Field(min_length=1)
    effect_contract_pins: tuple[ContentPin, ...] = ()
    dependencies: tuple[SkillDependency, ...] = ()
    compatible_actor_profiles: tuple[ContractId, ...] = ()
    tags: tuple[ContractId, ...] = ()
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_inventory(self) -> SkillManifest:
        require_unique(self.assets, key="asset_id", label="skill asset IDs")
        paths = [asset.relative_path for asset in self.assets]
        require_unique_strings(paths, label="skill asset paths")
        require_unique(self.functions, key="function_id", label="function IDs")
        require_unique(
            self.protocol_pins, key="component_id", label="protocol pin IDs"
        )
        require_unique(
            self.effect_contract_pins,
            key="component_id",
            label="effect-contract pin IDs",
        )
        require_unique(self.dependencies, key="skill_id", label="skill dependencies")
        require_unique_strings(
            self.compatible_actor_profiles, label="compatible actor profile IDs"
        )
        require_unique_strings(self.tags, label="skill tags")

        assets = {asset.asset_id: asset for asset in self.assets}
        for export in self.functions:
            source = assets.get(export.source_asset_id)
            if source is None:
                raise ValueError(
                    f"Function {export.function_id!r} references unknown source asset "
                    f"{export.source_asset_id!r}."
                )
            if source.kind is not SkillAssetKind.CODE:
                raise ValueError(
                    f"Function {export.function_id!r} source must be a code asset."
                )
            entry_file = export.entrypoint.rpartition(":")[0]
            if entry_file != source.relative_path:
                raise ValueError(
                    f"Function {export.function_id!r} entrypoint must belong to source "
                    f"asset path {source.relative_path!r}."
                )
        own_dependency = next(
            (item for item in self.dependencies if item.skill_id == self.skill_id), None
        )
        if own_dependency is not None:
            raise ValueError("A skill manifest cannot depend on itself.")
        return self

    @property
    def knowledge_assets(self) -> tuple[SkillAsset, ...]:
        return tuple(asset for asset in self.assets if asset.kind is SkillAssetKind.KNOWLEDGE)

    @property
    def api_assets(self) -> tuple[SkillAsset, ...]:
        return tuple(asset for asset in self.assets if asset.kind is SkillAssetKind.API)

    @property
    def code_assets(self) -> tuple[SkillAsset, ...]:
        return tuple(asset for asset in self.assets if asset.kind is SkillAssetKind.CODE)


__all__ = [
    "FunctionExport",
    "SkillAsset",
    "SkillAssetKind",
    "SkillDependency",
    "SkillManifest",
]
