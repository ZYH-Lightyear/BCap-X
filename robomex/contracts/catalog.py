"""Fail-closed in-memory catalog for immutable v2 contracts."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Literal, TypeVar

from pydantic import model_validator

from robomex.contracts.common import SealedContract, revalidate_sealed
from robomex.contracts.effects import EffectContract
from robomex.contracts.protocol import ProtocolSpec
from robomex.contracts.skill import SkillManifest


class ContractCatalogError(RuntimeError):
    """Base class for catalog identity and integrity failures."""


class DuplicateContractError(ContractCatalogError):
    """Raised when a logical identity is registered more than once."""


class ContractHashDriftError(ContractCatalogError):
    """Raised when one logical version is observed with different content."""


class MissingContractError(ContractCatalogError, KeyError):
    """Raised when a pinned dependency cannot be resolved exactly."""


class ContractCatalogSnapshot(SealedContract):
    """Portable, immutable registry snapshot with all references resolved."""

    schema_version: Literal["robomex.contract_catalog.v1"] = (
        "robomex.contract_catalog.v1"
    )
    skills: tuple[SkillManifest, ...] = ()
    protocols: tuple[ProtocolSpec, ...] = ()
    effect_contracts: tuple[EffectContract, ...] = ()

    @model_validator(mode="after")
    def _check_catalog(self) -> ContractCatalogSnapshot:
        _require_unique_versions(
            ((item.skill_id, item.revision) for item in self.skills), "skill"
        )
        _require_unique_versions(
            ((item.protocol_id, item.revision) for item in self.protocols), "protocol"
        )
        _require_unique_versions(
            ((item.contract_id, item.revision) for item in self.effect_contracts),
            "effect contract",
        )
        protocols = {
            (item.protocol_id, item.revision): item.content_digest for item in self.protocols
        }
        effects = {
            (item.contract_id, item.revision): item.content_digest
            for item in self.effect_contracts
        }
        for protocol in self.protocols:
            nested = protocol.effect_contract
            _require_pin(
                effects,
                nested.contract_id,
                nested.revision,
                nested.content_digest,
                owner=f"protocol {protocol.protocol_id!r}",
            )
        for skill in self.skills:
            for pin in skill.protocol_pins:
                if pin.revision is None:
                    raise ValueError(
                        f"Skill {skill.skill_id!r} protocol pin {pin.component_id!r} "
                        "must include a revision."
                    )
                _require_pin(
                    protocols,
                    pin.component_id,
                    pin.revision,
                    pin.content_digest,
                    owner=f"skill {skill.skill_id!r}",
                )
            for pin in skill.effect_contract_pins:
                if pin.revision is None:
                    raise ValueError(
                        f"Skill {skill.skill_id!r} effect pin {pin.component_id!r} "
                        "must include a revision."
                    )
                _require_pin(
                    effects,
                    pin.component_id,
                    pin.revision,
                    pin.content_digest,
                    owner=f"skill {skill.skill_id!r}",
                )
        return self


def _require_unique_versions(
    identities: Iterable[tuple[str, int]], label: str
) -> None:
    seen: set[tuple[str, int]] = set()
    duplicates: set[str] = set()
    for identity in identities:
        if identity in seen:
            duplicates.add(f"{identity[0]}@{identity[1]}")
        seen.add(identity)
    if duplicates:
        raise ValueError(f"Duplicate {label} versions: {', '.join(sorted(duplicates))}.")


def _require_pin(
    index: dict[tuple[str, int], str],
    component_id: str,
    revision: int,
    digest: str,
    *,
    owner: str,
) -> None:
    observed = index.get((component_id, revision))
    if observed is None:
        raise ValueError(
            f"{owner} references missing contract {component_id!r}@{revision}."
        )
    if observed != digest:
        raise ValueError(
            f"{owner} pin for {component_id!r}@{revision} does not match catalog digest."
        )


ContractT = TypeVar("ContractT", SkillManifest, ProtocolSpec, EffectContract)


class ContractRegistry:
    """Thread-safe registry that never overwrites an admitted logical version.

    ``register_*`` rejects even a byte-identical duplicate, which is useful for
    detecting duplicate discovery roots.  ``ensure_*`` is the explicit
    idempotent variant for replay/recovery paths.  Neither method permits hash
    drift.
    """

    def __init__(self, snapshot: ContractCatalogSnapshot | None = None) -> None:
        self._skills: dict[tuple[str, int], SkillManifest] = {}
        self._protocols: dict[tuple[str, int], ProtocolSpec] = {}
        self._effects: dict[tuple[str, int], EffectContract] = {}
        self._lock = threading.RLock()
        if snapshot is not None:
            snapshot = revalidate_sealed(snapshot)
            for effect in snapshot.effect_contracts:
                self.ensure_effect_contract(effect)
            for protocol in snapshot.protocols:
                self.ensure_protocol(protocol)
            for skill in snapshot.skills:
                self.ensure_skill(skill)

    def register_effect_contract(self, contract: EffectContract) -> EffectContract:
        with self._lock:
            return self._insert(
                self._effects,
                (contract.contract_id, contract.revision),
                contract,
                label="effect contract",
                idempotent=False,
            )

    def ensure_effect_contract(self, contract: EffectContract) -> EffectContract:
        with self._lock:
            return self._insert(
                self._effects,
                (contract.contract_id, contract.revision),
                contract,
                label="effect contract",
                idempotent=True,
            )

    def register_protocol(self, protocol: ProtocolSpec) -> ProtocolSpec:
        with self._lock:
            protocol = revalidate_sealed(protocol)
            self.ensure_effect_contract(protocol.effect_contract)
            return self._insert(
                self._protocols,
                (protocol.protocol_id, protocol.revision),
                protocol,
                label="protocol",
                idempotent=False,
            )

    def ensure_protocol(self, protocol: ProtocolSpec) -> ProtocolSpec:
        with self._lock:
            protocol = revalidate_sealed(protocol)
            self.ensure_effect_contract(protocol.effect_contract)
            return self._insert(
                self._protocols,
                (protocol.protocol_id, protocol.revision),
                protocol,
                label="protocol",
                idempotent=True,
            )

    def register_skill(self, manifest: SkillManifest) -> SkillManifest:
        with self._lock:
            manifest = revalidate_sealed(manifest)
            self._check_skill_pins(manifest)
            return self._insert(
                self._skills,
                (manifest.skill_id, manifest.revision),
                manifest,
                label="skill",
                idempotent=False,
            )

    def ensure_skill(self, manifest: SkillManifest) -> SkillManifest:
        with self._lock:
            manifest = revalidate_sealed(manifest)
            self._check_skill_pins(manifest)
            return self._insert(
                self._skills,
                (manifest.skill_id, manifest.revision),
                manifest,
                label="skill",
                idempotent=True,
            )

    def effect_contract(self, contract_id: str, revision: int) -> EffectContract:
        with self._lock:
            return self._resolve(self._effects, (contract_id, revision), "effect contract")

    def protocol(self, protocol_id: str, revision: int) -> ProtocolSpec:
        with self._lock:
            return self._resolve(self._protocols, (protocol_id, revision), "protocol")

    def skill(self, skill_id: str, revision: int) -> SkillManifest:
        with self._lock:
            return self._resolve(self._skills, (skill_id, revision), "skill")

    def latest_skill(self, skill_id: str) -> SkillManifest:
        return self._latest(self._skills, skill_id, "skill")

    def latest_protocol(self, protocol_id: str) -> ProtocolSpec:
        return self._latest(self._protocols, protocol_id, "protocol")

    def snapshot(self) -> ContractCatalogSnapshot:
        with self._lock:
            return ContractCatalogSnapshot(
                skills=tuple(self._skills[key] for key in sorted(self._skills)),
                protocols=tuple(self._protocols[key] for key in sorted(self._protocols)),
                effect_contracts=tuple(
                    self._effects[key] for key in sorted(self._effects)
                ),
            )

    def _check_skill_pins(self, manifest: SkillManifest) -> None:
        for pin in manifest.protocol_pins:
            if pin.revision is None:
                raise MissingContractError(
                    f"Protocol pin {pin.component_id!r} must include a revision."
                )
            protocol = self._protocols.get((pin.component_id, pin.revision))
            if protocol is None:
                raise MissingContractError(
                    f"Missing pinned protocol {pin.component_id!r}@{pin.revision}."
                )
            if protocol.content_digest != pin.content_digest:
                raise ContractHashDriftError(
                    f"Protocol pin {pin.component_id!r}@{pin.revision} has hash drift."
                )
        for pin in manifest.effect_contract_pins:
            if pin.revision is None:
                raise MissingContractError(
                    f"Effect-contract pin {pin.component_id!r} must include a revision."
                )
            contract = self._effects.get((pin.component_id, pin.revision))
            if contract is None:
                raise MissingContractError(
                    f"Missing pinned effect contract {pin.component_id!r}@{pin.revision}."
                )
            if contract.content_digest != pin.content_digest:
                raise ContractHashDriftError(
                    f"Effect-contract pin {pin.component_id!r}@{pin.revision} has hash drift."
                )

    @staticmethod
    def _insert(
        index: dict[tuple[str, int], ContractT],
        key: tuple[str, int],
        contract: ContractT,
        *,
        label: str,
        idempotent: bool,
    ) -> ContractT:
        contract = revalidate_sealed(contract)
        existing = index.get(key)
        if existing is not None:
            if existing.content_digest != contract.content_digest:
                raise ContractHashDriftError(
                    f"{label.title()} {key[0]!r}@{key[1]} changed content digest."
                )
            if not idempotent:
                raise DuplicateContractError(
                    f"Duplicate {label} identity {key[0]!r}@{key[1]}."
                )
            return existing
        index[key] = contract
        return contract

    @staticmethod
    def _resolve(
        index: dict[tuple[str, int], ContractT],
        key: tuple[str, int],
        label: str,
    ) -> ContractT:
        try:
            return index[key]
        except KeyError as exc:
            raise MissingContractError(
                f"Unknown {label} {key[0]!r}@{key[1]}."
            ) from exc

    def _latest(
        self,
        index: dict[tuple[str, int], ContractT],
        component_id: str,
        label: str,
    ) -> ContractT:
        with self._lock:
            versions = [
                (revision, value)
                for (registered_id, revision), value in index.items()
                if registered_id == component_id
            ]
            if not versions:
                raise MissingContractError(f"Unknown {label} {component_id!r}.")
            return max(versions, key=lambda pair: pair[0])[1]


__all__ = [
    "ContractCatalogError",
    "ContractCatalogSnapshot",
    "ContractHashDriftError",
    "ContractRegistry",
    "DuplicateContractError",
    "MissingContractError",
]
