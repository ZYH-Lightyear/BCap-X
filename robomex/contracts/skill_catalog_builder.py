"""Compile on-disk RoboMEx skill packages into immutable v2 contracts.

The prose-first :class:`~robomex.skills.store.SkillLibrary` remains the model-facing
surface.  A production v2 run additionally needs an exact, content-addressed
inventory for its run manifest.  This module creates that inventory without
importing skill sidecars: Python exports are inspected through their AST, so catalog
construction cannot execute package code or trigger robot APIs.

The compiler is intentionally conservative.  It can infer read-only proposal skills
from the current ``contract.yaml`` format.  A package declaring ``changes_world`` is
rejected because an authoritative effect/resource cannot safely be guessed from prose;
such a package must be represented by an explicitly authored v2 contract instead.
"""

from __future__ import annotations

import ast
import hashlib
import mimetypes
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from robomex.contracts.catalog import ContractCatalogSnapshot, ContractRegistry
from robomex.contracts.common import ContentPin
from robomex.contracts.effects import DeclaredEffect, EffectContract, EffectScope
from robomex.contracts.protocol import ProtocolSpec, SlotCardinality, SlotPolicy
from robomex.contracts.skill import (
    FunctionExport,
    SkillAsset,
    SkillAssetKind,
    SkillDependency,
    SkillManifest,
)
from robomex.dysc.contracts import SkillContract, load_contract_for_skill
from robomex.runtime.events import ControlOutcome
from robomex.skills.store import SkillLibrary

_IGNORED_NAMES = frozenset({"utility.json", ".DS_Store"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


class SkillCatalogBuildError(ValueError):
    """A disk package cannot be represented as a closed v2 skill contract."""


@dataclass(frozen=True)
class BuiltSkillContract:
    """The three sealed records derived from one proposal-only package."""

    effect: EffectContract
    protocol: ProtocolSpec
    manifest: SkillManifest


def build_skill_contract_catalog(
    library: SkillLibrary,
    skill_ids: Iterable[str],
    *,
    compatible_actor_profiles: Mapping[str, Iterable[str]] | None = None,
) -> ContractCatalogSnapshot:
    """Build an exact catalog for ``skill_ids`` from ``library``.

    Dependencies named by ``required_skills`` must be present in the requested set.
    The returned snapshot is already cross-reference validated by
    :class:`ContractRegistry`.
    """

    if not isinstance(library, SkillLibrary):
        raise TypeError("library must be a SkillLibrary")
    requested = tuple(str(value).strip() for value in skill_ids)
    if not requested or any(not value for value in requested):
        raise SkillCatalogBuildError("skill_ids must contain non-empty IDs")
    if len(requested) != len(set(requested)):
        raise SkillCatalogBuildError("skill_ids must not contain duplicates")
    profile_map = {
        str(skill_id): tuple(str(value).strip() for value in profiles)
        for skill_id, profiles in (compatible_actor_profiles or {}).items()
    }
    if any(not skill_id for skill_id in profile_map) or any(
        not profile for profiles in profile_map.values() for profile in profiles
    ):
        raise SkillCatalogBuildError("compatible actor profile IDs must not be empty")

    records: dict[str, tuple[Path, SkillContract]] = {}
    for skill_id in requested:
        try:
            skill = library.get(skill_id).skill
        except KeyError as exc:
            raise SkillCatalogBuildError(f"unknown requested skill {skill_id!r}") from exc
        root = skill.root
        if root is None:
            raise SkillCatalogBuildError(f"skill {skill_id!r} has no package root")
        contract = load_contract_for_skill(root)
        if contract is None:
            raise SkillCatalogBuildError(
                f"skill {skill_id!r} has no contract.yaml and cannot be pinned"
            )
        if contract.skill_id != skill_id:
            raise SkillCatalogBuildError(
                f"skill directory {skill_id!r} disagrees with contract ID "
                f"{contract.skill_id!r}"
            )
        if contract.changes_world:
            raise SkillCatalogBuildError(
                f"skill {skill_id!r} declares changes_world; authoritative effects "
                "must use an explicitly authored v2 EffectContract"
            )
        records[skill_id] = (root.resolve(), contract)

    selected = set(records)
    for skill_id, (_root, contract) in records.items():
        missing = sorted(set(contract.required_skills) - selected)
        if missing:
            raise SkillCatalogBuildError(
                f"skill {skill_id!r} has unpinned required skills: "
                + ", ".join(missing)
            )

    compiled: list[BuiltSkillContract] = []
    for skill_id in sorted(records):
        root, contract = records[skill_id]
        compiled.append(
            _compile_skill(
                library=library,
                root=root,
                contract=contract,
                compatible_actor_profiles=profile_map.get(skill_id, ()),
            )
        )

    registry = ContractRegistry()
    for item in compiled:
        registry.register_effect_contract(item.effect)
        registry.register_protocol(item.protocol)
    for item in compiled:
        registry.register_skill(item.manifest)
    return registry.snapshot()


def _compile_skill(
    *,
    library: SkillLibrary,
    root: Path,
    contract: SkillContract,
    compatible_actor_profiles: tuple[str, ...],
) -> BuiltSkillContract:
    skill = library.get(contract.skill_id).skill
    effect = EffectContract(
        contract_id=f"robomex.skill_effect.{contract.skill_id}",
        effects=(
            DeclaredEffect(
                effect_id=f"robomex.proposal.{contract.skill_id}",
                scope=EffectScope.READ_ONLY,
                operation="author immutable typed proposal artifacts",
                description=(
                    "No physical or embodied-state commit authority; runtime-owned "
                    "reducers and system actions remain the only writers."
                ),
            ),
        ),
    )
    outcomes = _outcomes(contract)
    protocol = ProtocolSpec(
        protocol_id=f"robomex.skill_protocol.{contract.skill_id}",
        summary=skill.description or skill.name or contract.skill_id,
        inputs=tuple(_slot(port, output=False) for port in contract.input_ports),
        outputs=tuple(_slot(port, output=True) for port in contract.output_ports),
        outcomes=outcomes,
        effect_contract=effect,
        required_capabilities=tuple(dict.fromkeys(contract.capabilities)),
        metadata={
            "role": contract.role,
            "changes_world": False,
        },
    )
    assets, assets_by_path = _assets(root, contract.skill_id)
    functions = tuple(
        _function_export(
            root=root,
            skill_id=contract.skill_id,
            declaration=value,
            assets_by_path=assets_by_path,
        )
        for value in contract.functions
    )
    dependencies = tuple(
        SkillDependency(skill_id=value, minimum_revision=1)
        for value in dict.fromkeys(contract.required_skills)
    )
    manifest = SkillManifest(
        skill_id=contract.skill_id,
        name=skill.name or contract.skill_id,
        summary=skill.description or f"Pinned RoboMEx skill {contract.skill_id}.",
        assets=assets,
        functions=functions,
        protocol_pins=(
            ContentPin(
                component_id=protocol.protocol_id,
                revision=protocol.revision,
                content_digest=protocol.content_digest,
            ),
        ),
        effect_contract_pins=(
            ContentPin(
                component_id=effect.contract_id,
                revision=effect.revision,
                content_digest=effect.content_digest,
            ),
        ),
        dependencies=dependencies,
        compatible_actor_profiles=tuple(dict.fromkeys(compatible_actor_profiles)),
        tags=tuple(dict.fromkeys(contract.tags)),
        metadata={
            "role": contract.role,
            "changes_world": False,
        },
    )
    return BuiltSkillContract(effect=effect, protocol=protocol, manifest=manifest)


def _slot(port, *, output: bool) -> SlotPolicy:
    if not port.name or not port.schema:
        raise SkillCatalogBuildError("contract ports require name and schema")
    return SlotPolicy(
        slot_id=port.name,
        schema_id=port.schema,
        required=port.required,
        cardinality=(
            SlotCardinality.ONE if port.required else SlotCardinality.ZERO_OR_ONE
        ),
        description=(
            f"frame={port.frame}" if port.frame else ("skill output" if output else "skill input")
        ),
    )


def _outcomes(contract: SkillContract) -> tuple[ControlOutcome, ...]:
    values: list[ControlOutcome] = [ControlOutcome.SUCCESS]
    for raw in contract.exit_conditions:
        try:
            value = ControlOutcome(raw)
        except ValueError as exc:
            raise SkillCatalogBuildError(
                f"skill {contract.skill_id!r} declares unknown outcome {raw!r}"
            ) from exc
        if value not in values:
            values.append(value)
    return tuple(values)


def _assets(
    root: Path,
    skill_id: str,
) -> tuple[tuple[SkillAsset, ...], dict[str, SkillAsset]]:
    files = tuple(path for path in sorted(root.rglob("*")) if _admitted_file(root, path))
    if not files:
        raise SkillCatalogBuildError(f"skill {skill_id!r} contains no pinnable assets")
    values: list[SkillAsset] = []
    by_path: dict[str, SkillAsset] = {}
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(root).as_posix()
        kind = (
            SkillAssetKind.CODE
            if relative.startswith("scripts/") and path.suffix == ".py"
            else SkillAssetKind.API
            if relative == "contract.yaml" or relative.startswith("prompts/")
            else SkillAssetKind.KNOWLEDGE
        )
        media_type = mimetypes.guess_type(relative)[0] or (
            "text/x-python" if path.suffix == ".py" else "application/octet-stream"
        )
        asset = SkillAsset(
            asset_id=f"asset.{skill_id}.{index:03d}",
            kind=kind,
            relative_path=relative,
            content_digest=_bytes_digest(path.read_bytes()),
            media_type=media_type,
        )
        values.append(asset)
        by_path[relative] = asset
    return tuple(values), by_path


def _admitted_file(root: Path, path: Path) -> bool:
    if not path.is_file() or path.name in _IGNORED_NAMES or path.suffix in _IGNORED_SUFFIXES:
        return False
    relative = path.relative_to(root)
    return "__pycache__" not in relative.parts and not any(
        part.startswith(".") for part in relative.parts
    )


def _function_export(
    *,
    root: Path,
    skill_id: str,
    declaration,
    assets_by_path: Mapping[str, SkillAsset],
) -> FunctionExport:
    raw_path, separator, symbol = declaration.entry.rpartition(":")
    if not separator or not symbol.isidentifier():
        raise SkillCatalogBuildError(
            f"skill {skill_id!r} has malformed function entry {declaration.entry!r}"
        )
    path = PurePosixPath(raw_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SkillCatalogBuildError("function entry must be package-relative")
    normalized_path = path.as_posix()
    asset = assets_by_path.get(normalized_path)
    if asset is None or asset.kind is not SkillAssetKind.CODE:
        raise SkillCatalogBuildError(
            f"function source {normalized_path!r} is not a pinned Python code asset"
        )
    source_path = root / normalized_path
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=normalized_path)
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise SkillCatalogBuildError(
            f"cannot parse function source {normalized_path!r}"
        ) from exc
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol
    ]
    if len(matches) != 1:
        raise SkillCatalogBuildError(
            f"function entry {declaration.entry!r} must resolve to one top-level definition"
        )
    function = matches[0]
    implementation = ast.dump(function, annotate_fields=True, include_attributes=False)
    interface = ast.dump(
        ast.Module(
            body=[
                ast.FunctionDef(
                    name=function.name,
                    args=function.args,
                    body=[ast.Pass()],
                    decorator_list=[],
                    returns=function.returns,
                    type_comment=function.type_comment,
                )
            ],
            type_ignores=[],
        ),
        annotate_fields=True,
        include_attributes=False,
    )
    return FunctionExport(
        function_id=f"function.{skill_id}.{declaration.name}",
        source_asset_id=asset.asset_id,
        entrypoint=declaration.entry,
        function_digest=_text_digest(implementation),
        interface_digest=_text_digest(interface),
        description=declaration.description,
    )


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _text_digest(value: str) -> str:
    return _bytes_digest(value.encode("utf-8"))


__all__ = [
    "BuiltSkillContract",
    "SkillCatalogBuildError",
    "build_skill_contract_catalog",
]
