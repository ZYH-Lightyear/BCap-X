"""Fail-closed construction and immutable run identity for RoboMEx v2.

The v1 :mod:`robomex.core.session` entry point remains untouched.  This module
is the application boundary for the v2 runtime: it binds a sealed run manifest
to an exact contract catalog, verifies every runtime dependency against that
identity, and only then constructs episode-scoped registries.

Backend implementation/configuration hashes are supplied by the trusted
adapter installation layer.  Python object introspection is deliberately not
used as provenance: it is neither reproducible nor a meaningful digest of an
installed robot adapter.
"""

from __future__ import annotations

import errno
import json
import os
import re
import stat
import tempfile
import threading
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from robomex.contracts import (
    ContractCatalogSnapshot,
    ContractRegistry,
    DigestStr,
    canonical_payload_digest,
    revalidate_sealed,
)
from robomex.data import SchemaRegistry, core_schema_registry
from robomex.evolution import BackendPin, BackendRole, RunManifest
from robomex.orchestration.actors import (
    ActorProfile,
    ActorRegistry,
    AgentProvider,
)
from robomex.orchestration.arena import (
    ArenaBinding,
    RegisteredShadowBackend,
    ShadowBackendRegistry,
)
from robomex.orchestration.episode import EpisodeRuntime, install_runtime_schemas
from robomex.orchestration.run_budget import RunBudgetAuthority
from robomex.runtime.action_protocol import AdmissionSnapshot, WorldKind
from robomex.runtime.authority import ActionBackend, FeasibilityChecker
from robomex.runtime.observation import ObservationBackend, ObservationRegistry

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EpisodeId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
    ),
]
RuntimeIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    ),
]
_MAX_IDENTITY_DOCUMENT_BYTES = 16 * 1024 * 1024
_MANIFEST_FILENAME = "run_manifest.v2.json"
_CATALOG_FILENAME = "contract_catalog.v1.json"
_IDENTITY_DIRECTORY = "run_identity.v2"
_RUNTIME_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class V2BootstrapError(RuntimeError):
    """Base class for v2 identity and dependency admission failures."""


class IdentityIntegrityError(V2BootstrapError):
    """A persisted identity document is missing, malformed, or tampered."""


class IdentityConflictError(V2BootstrapError):
    """An existing run identity differs from the requested identity."""


class CatalogPinError(V2BootstrapError):
    """A run-manifest pin does not resolve exactly in its contract catalog."""


class DependencyAdmissionError(V2BootstrapError):
    """A runtime dependency is absent, duplicated, or inconsistent with a pin."""


class V2RuntimeConfig(BaseModel):
    """Filesystem and namespace configuration for one immutable v2 run."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: Literal["robomex.runtime_config.v2"] = "robomex.runtime_config.v2"
    run_id: NonEmptyStr
    episode_id: EpisodeId
    episode_root: Path
    graph_digest: DigestStr
    runtime_code_digest: DigestStr
    actor_namespace_root: RuntimeIdentifier | None = None
    actor_workspace_subdirectory: NonEmptyStr = "actor_workspaces"

    @field_validator("episode_root")
    @classmethod
    def _usable_episode_root(cls, value: Path) -> Path:
        if not value.parts:
            raise ValueError("episode_root must not be empty")
        return value

    @field_validator("actor_workspace_subdirectory")
    @classmethod
    def _safe_workspace_subdirectory(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
            raise ValueError("actor_workspace_subdirectory must be one relative path segment")
        return value

    @property
    def namespace_root(self) -> str:
        return self.actor_namespace_root or self.episode_id

    @property
    def actor_workspace_root(self) -> Path:
        return self.episode_root / self.actor_workspace_subdirectory

    @property
    def identity_root(self) -> Path:
        return self.episode_root / _IDENTITY_DIRECTORY


class BackendProvenance(BaseModel):
    """Installation-owned provenance for one concrete runtime backend."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    backend_id: NonEmptyStr
    implementation_digest: DigestStr
    configuration_digest: DigestStr
    version: NonEmptyStr

    @classmethod
    def from_pin(cls, pin: BackendPin) -> BackendProvenance:
        return cls(
            backend_id=pin.backend_id,
            implementation_digest=pin.implementation_digest,
            configuration_digest=pin.configuration_digest,
            version=pin.version,
        )

    def assert_matches(self, pin: BackendPin) -> None:
        expected = BackendProvenance.from_pin(pin)
        if self != expected:
            raise DependencyAdmissionError(
                f"Runtime provenance for backend {self.backend_id!r} does not match "
                "the sealed run-manifest pin."
            )


@dataclass(frozen=True)
class ActionBackendBinding:
    """One authoritative world/resource binding and its trusted checker."""

    world_id: str
    resource_id: str
    backend: ActionBackend
    feasibility_checker: FeasibilityChecker
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        if not self.world_id.strip() or not self.resource_id.strip():
            raise ValueError("world_id and resource_id must not be empty")
        if not isinstance(self.backend, ActionBackend):
            raise TypeError("backend does not implement ActionBackend")
        if not isinstance(self.feasibility_checker, FeasibilityChecker):
            raise TypeError("feasibility_checker does not implement FeasibilityChecker")
        if self.backend.descriptor.backend_id != self.provenance.backend_id:
            raise DependencyAdmissionError(
                "Action backend descriptor and provenance use different backend IDs."
            )


@dataclass(frozen=True)
class ObservationBackendBinding:
    """One read-only observation backend and its exact provenance."""

    backend: ObservationBackend
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.backend, ObservationBackend):
            raise TypeError("backend does not implement ObservationBackend")
        if self.backend.backend_id != self.provenance.backend_id:
            raise DependencyAdmissionError(
                "Observation backend and provenance use different backend IDs."
            )


@dataclass(frozen=True)
class ShadowBackendBinding:
    """An isolated action backend available to Arena/shadow rollout adapters."""

    backend: ActionBackend
    provenance: BackendProvenance
    world_resource_bindings: frozenset[tuple[str, str]]

    def __post_init__(self) -> None:
        if not isinstance(self.backend, ActionBackend):
            raise TypeError("backend does not implement ActionBackend")
        if self.backend.descriptor.backend_id != self.provenance.backend_id:
            raise DependencyAdmissionError(
                "Shadow backend descriptor and provenance use different backend IDs."
            )
        normalized = frozenset(
            (str(world_id).strip(), str(resource_id).strip())
            for world_id, resource_id in self.world_resource_bindings
        )
        if not normalized or any(
            not world_id or not resource_id for world_id, resource_id in normalized
        ):
            raise DependencyAdmissionError(
                "Shadow backend requires explicit non-empty world/resource bindings."
            )
        for world_id, resource_id in normalized:
            snapshot = AdmissionSnapshot.model_validate(
                self.backend.snapshot(world_id, resource_id).model_dump(mode="python")
            )
            if (
                snapshot.world_id != world_id
                or snapshot.resource_id != resource_id
                or snapshot.world_kind is not WorldKind.SHADOW
            ):
                raise DependencyAdmissionError(
                    "Shadow backend binding does not prove WorldKind.SHADOW."
                )
        object.__setattr__(self, "world_resource_bindings", normalized)


@dataclass(frozen=True)
class V2RuntimeDependencies:
    """All externally supplied dependencies admitted by :class:`V2RuntimeFactory`."""

    contract_catalog: ContractCatalogSnapshot
    actor_providers: Mapping[str, AgentProvider]
    actor_profiles: Mapping[str, ActorProfile]
    action_backends: tuple[ActionBackendBinding, ...]
    observation_backends: tuple[ObservationBackendBinding, ...] = ()
    shadow_backends: tuple[ShadowBackendBinding, ...] = ()
    arena_bindings: tuple[ArenaBinding, ...] = ()
    schema_registry: SchemaRegistry | None = None
    freshness_context_provider_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.contract_catalog, ContractCatalogSnapshot):
            raise TypeError("contract_catalog must be a ContractCatalogSnapshot")
        if self.schema_registry is not None and not isinstance(
            self.schema_registry, SchemaRegistry
        ):
            raise TypeError("schema_registry must be a SchemaRegistry")
        if any(not isinstance(item, ActionBackendBinding) for item in self.action_backends):
            raise TypeError("action_backends must contain ActionBackendBinding values")
        if any(
            not isinstance(item, ObservationBackendBinding) for item in self.observation_backends
        ):
            raise TypeError("observation_backends must contain ObservationBackendBinding values")
        if any(not isinstance(item, ShadowBackendBinding) for item in self.shadow_backends):
            raise TypeError("shadow_backends must contain ShadowBackendBinding values")
        if any(not isinstance(item, ArenaBinding) for item in self.arena_bindings):
            raise TypeError("arena_bindings must contain ArenaBinding values")
        freshness_provider_id = self.freshness_context_provider_id
        if freshness_provider_id is not None:
            if (
                not isinstance(freshness_provider_id, str)
                or not freshness_provider_id.strip()
                or freshness_provider_id != freshness_provider_id.strip()
            ):
                raise TypeError("freshness_context_provider_id must be a non-empty provider ID")
            freshness_provider = self.actor_providers.get(freshness_provider_id)
            if freshness_provider is None:
                raise DependencyAdmissionError(
                    "freshness_context_provider_id does not name an installed "
                    f"actor provider: {freshness_provider_id!r}"
                )
            if not callable(getattr(freshness_provider, "freshness_context", None)):
                raise DependencyAdmissionError(
                    f"Actor provider {freshness_provider_id!r} does not expose freshness_context."
                )
        object.__setattr__(self, "contract_catalog", revalidate_sealed(self.contract_catalog))
        object.__setattr__(self, "actor_providers", MappingProxyType(dict(self.actor_providers)))
        object.__setattr__(self, "actor_profiles", MappingProxyType(dict(self.actor_profiles)))
        object.__setattr__(self, "action_backends", tuple(self.action_backends))
        object.__setattr__(self, "observation_backends", tuple(self.observation_backends))
        object.__setattr__(self, "shadow_backends", tuple(self.shadow_backends))
        object.__setattr__(self, "arena_bindings", tuple(self.arena_bindings))


@dataclass(frozen=True)
class V2Application:
    """Fully admitted v2 application object; no mutable global registry is used."""

    config: V2RuntimeConfig
    manifest: RunManifest
    catalog: ContractCatalogSnapshot
    contracts: ContractRegistry
    schemas: SchemaRegistry
    actors: ActorRegistry
    observations: ObservationRegistry
    episode: EpisodeRuntime
    run_budget_authority: RunBudgetAuthority
    shadow_registry: ShadowBackendRegistry = field(repr=False)
    shadow_backends: Mapping[str, ActionBackend] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shadow_backends", MappingProxyType(dict(self.shadow_backends)))


def actor_profile_digest(profile: ActorProfile) -> str:
    """Return the canonical manifest pin for a runtime ``ActorProfile``."""

    payload: dict[str, Any] = {
        "profile_id": profile.profile_id,
        "provider_id": profile.provider_id,
        "runner_kind": profile.runner_kind,
        "lifecycle": profile.lifecycle.value,
        "model": profile.model,
        "capability_ceiling": sorted(profile.capability_ceiling),
        "effect_ceiling": sorted(profile.effect_ceiling),
        "isolation": {
            "namespace_prefix": profile.isolation.namespace_prefix,
            "workspace_mode": profile.isolation.workspace_mode.value,
            "workspace_key": profile.isolation.workspace_key,
            "world_id": profile.isolation.world_id,
            "metadata": dict(profile.isolation.metadata),
        },
        "metadata": dict(profile.metadata),
    }
    return canonical_payload_digest("robomex.actor_profile.runtime.v1", payload)


def validate_manifest_catalog(
    manifest: RunManifest,
    catalog: ContractCatalogSnapshot,
) -> tuple[RunManifest, ContractCatalogSnapshot]:
    """Validate an exact (not best-effort/latest) manifest/catalog binding."""

    manifest = revalidate_sealed(manifest)
    catalog = revalidate_sealed(catalog)
    if manifest.contract_catalog_digest != catalog.content_digest:
        raise CatalogPinError("Run manifest does not pin this contract catalog digest.")

    catalog_skills = {(item.skill_id, item.revision): item for item in catalog.skills}
    manifest_skills = {(item.skill_id, item.revision): item for item in manifest.skills}
    if set(catalog_skills) != set(manifest_skills):
        raise CatalogPinError(
            "Manifest skill pins must exactly cover every skill version in the run catalog."
        )
    pinned_by_id = {item.skill_id: item for item in manifest.skills}
    for key, pin in manifest_skills.items():
        skill = catalog_skills[key]
        if pin.manifest_digest != skill.content_digest:
            raise CatalogPinError(
                f"Skill pin {pin.skill_id!r}@{pin.revision} has manifest hash drift."
            )

    expected_functions: dict[str, tuple[str, str, str]] = {}
    for skill in catalog.skills:
        for export in skill.functions:
            if export.function_id in expected_functions:
                raise CatalogPinError(
                    f"Function ID {export.function_id!r} is exported by multiple skills."
                )
            expected_functions[export.function_id] = (
                skill.skill_id,
                export.function_digest,
                export.interface_digest,
            )
        for dependency in skill.dependencies:
            dependency_pin = pinned_by_id.get(dependency.skill_id)
            if dependency_pin is None:
                if dependency.optional:
                    continue
                raise CatalogPinError(
                    f"Skill {skill.skill_id!r} requires unpinned skill {dependency.skill_id!r}."
                )
            if dependency_pin.revision < dependency.minimum_revision:
                raise CatalogPinError(
                    f"Skill dependency {dependency.skill_id!r} is below its minimum revision."
                )
            if (
                dependency.manifest_digest is not None
                and dependency.manifest_digest != dependency_pin.manifest_digest
            ):
                raise CatalogPinError(
                    f"Skill dependency {dependency.skill_id!r} has manifest hash drift."
                )

    supplied_functions = {
        item.function_id: (
            item.skill_id,
            item.implementation_digest,
            item.interface_digest,
        )
        for item in manifest.functions
    }
    if supplied_functions != expected_functions:
        raise CatalogPinError(
            "Manifest function pins must exactly match all exports of pinned skills."
        )
    return manifest, catalog


class RunManifestStore:
    """Atomic, create-once persistence for one sealed ``RunManifest``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def load(
        self,
        *,
        expected_run_id: str | None = None,
        expected_digest: str | None = None,
    ) -> RunManifest:
        with self._lock:
            raw = _read_regular_file(self.path)
            try:
                manifest = RunManifest.model_validate_json(raw)
                manifest = revalidate_sealed(manifest)
            except Exception as exc:
                raise IdentityIntegrityError(f"Invalid run manifest at {self.path}.") from exc
            if expected_run_id is not None and manifest.run_id != expected_run_id:
                raise IdentityConflictError(
                    f"Persisted run_id {manifest.run_id!r} does not match {expected_run_id!r}."
                )
            if expected_digest is not None and manifest.content_digest != expected_digest:
                raise IdentityConflictError("Persisted run-manifest digest does not match.")
            return manifest

    def bind(self, manifest: RunManifest) -> RunManifest:
        manifest = revalidate_sealed(manifest)
        payload = _canonical_document(manifest.model_dump(mode="json"))
        with self._lock:
            created = _atomic_create_read_only(self.path, payload)
            persisted = self.load(
                expected_run_id=manifest.run_id,
                expected_digest=manifest.content_digest,
            )
            if persisted != manifest:
                raise IdentityConflictError(
                    "Persisted run manifest differs from the requested manifest."
                )
            if created:
                _fsync_directory(self.path.parent)
            return persisted


class ContractCatalogStore:
    """Atomic, create-once persistence for the manifest-bound catalog snapshot."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def load(self, *, expected_digest: str | None = None) -> ContractCatalogSnapshot:
        with self._lock:
            raw = _read_regular_file(self.path)
            try:
                catalog = ContractCatalogSnapshot.model_validate_json(raw)
                catalog = revalidate_sealed(catalog)
            except Exception as exc:
                raise IdentityIntegrityError(f"Invalid contract catalog at {self.path}.") from exc
            if expected_digest is not None and catalog.content_digest != expected_digest:
                raise IdentityConflictError("Persisted contract-catalog digest does not match.")
            return catalog

    def bind(self, catalog: ContractCatalogSnapshot) -> ContractCatalogSnapshot:
        catalog = revalidate_sealed(catalog)
        payload = _canonical_document(catalog.model_dump(mode="json"))
        with self._lock:
            _atomic_create_read_only(self.path, payload)
            persisted = self.load(expected_digest=catalog.content_digest)
            if persisted != catalog:
                raise IdentityConflictError(
                    "Persisted contract catalog differs from the requested catalog."
                )
            return persisted


class RuntimeIdentityStore:
    """Atomically publishes the manifest and catalog as one immutable identity."""

    def __init__(self, episode_root: str | Path) -> None:
        self.episode_root = Path(episode_root)
        self.identity_root = self.episode_root / _IDENTITY_DIRECTORY
        self.manifest_store = RunManifestStore(self.identity_root / _MANIFEST_FILENAME)
        self.catalog_store = ContractCatalogStore(self.identity_root / _CATALOG_FILENAME)
        self._lock = threading.RLock()

    def load(
        self,
        *,
        expected_run_id: str | None = None,
        expected_manifest_digest: str | None = None,
    ) -> tuple[RunManifest, ContractCatalogSnapshot]:
        with self._lock:
            _require_identity_directory(self.identity_root)
            manifest = self.manifest_store.load(
                expected_run_id=expected_run_id,
                expected_digest=expected_manifest_digest,
            )
            catalog = self.catalog_store.load(expected_digest=manifest.contract_catalog_digest)
            return validate_manifest_catalog(manifest, catalog)

    def bind(
        self,
        manifest: RunManifest,
        catalog: ContractCatalogSnapshot,
    ) -> tuple[RunManifest, ContractCatalogSnapshot]:
        manifest, catalog = validate_manifest_catalog(manifest, catalog)
        with self._lock:
            if self.identity_root.exists() or self.identity_root.is_symlink():
                persisted = self.load(
                    expected_run_id=manifest.run_id,
                    expected_manifest_digest=manifest.content_digest,
                )
                if persisted != (manifest, catalog):
                    raise IdentityConflictError(
                        "Persisted run identity differs from the requested identity."
                    )
                return persisted

            self.episode_root.mkdir(parents=True, exist_ok=True)
            stage = self.episode_root / (f".{_IDENTITY_DIRECTORY}.{uuid.uuid4().hex}.staging")
            try:
                stage.mkdir(mode=0o700)
                _write_staged_document(
                    stage / _MANIFEST_FILENAME,
                    _canonical_document(manifest.model_dump(mode="json")),
                )
                _write_staged_document(
                    stage / _CATALOG_FILENAME,
                    _canonical_document(catalog.model_dump(mode="json")),
                )
                _fsync_directory(stage)
                os.chmod(stage, 0o555)
                try:
                    os.rename(stage, self.identity_root)
                except OSError as exc:
                    if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                        raise
                    _discard_staging_directory(stage)
                _fsync_directory(self.episode_root)
            except Exception:
                _discard_staging_directory(stage)
                raise

            persisted = self.load(
                expected_run_id=manifest.run_id,
                expected_manifest_digest=manifest.content_digest,
            )
            if persisted != (manifest, catalog):
                raise IdentityConflictError("A concurrent process bound a different run identity.")
            return persisted


class V2RuntimeFactory:
    """Validate identity/dependencies, then construct a complete v2 application."""

    def build(
        self,
        *,
        config: V2RuntimeConfig,
        manifest: RunManifest,
        dependencies: V2RuntimeDependencies,
    ) -> V2Application:
        manifest, catalog = validate_manifest_catalog(manifest, dependencies.contract_catalog)
        self._validate_config(config, manifest)
        allowed_initial_graph_digests = self._initial_graph_allowset(manifest)
        schemas = dependencies.schema_registry or core_schema_registry()
        schemas.assert_core_complete()
        install_runtime_schemas(schemas)
        if manifest.schema_registry_digest != schemas.content_digest:
            raise IdentityConflictError(
                "Run manifest schema-registry digest differs from installed validators."
            )
        self._validate_protocol_schemas(catalog, schemas)
        self._validate_actor_profiles(manifest, catalog, dependencies)
        self._validate_arena_bindings(manifest, dependencies)
        backend_index = self._validate_backends(manifest, dependencies)

        persisted_manifest, persisted_catalog = RuntimeIdentityStore(config.episode_root).bind(
            manifest, catalog
        )
        contracts = ContractRegistry(persisted_catalog)
        run_budget_authority = RunBudgetAuthority(
            config.episode_root / "run_budget.v1.jsonl",
            budgets=persisted_manifest.budgets,
        )
        actors = ActorRegistry(
            dependencies.actor_providers,
            namespace_root=config.namespace_root,
            workspace_root=config.actor_workspace_root,
        )
        action_backends = {
            (binding.world_id, binding.resource_id): binding.backend
            for binding in dependencies.action_backends
        }
        feasibility_checkers = {
            (binding.world_id, binding.resource_id): binding.feasibility_checker
            for binding in dependencies.action_backends
        }
        shadow_registry = ShadowBackendRegistry(
            tuple(
                RegisteredShadowBackend(
                    backend_id=binding.provenance.backend_id,
                    backend=binding.backend,
                    world_resource_bindings=binding.world_resource_bindings,
                )
                for binding in dependencies.shadow_backends
            )
        )
        episode = EpisodeRuntime(
            episode_id=config.episode_id,
            episode_root=config.episode_root,
            actors=actors,
            action_backends=action_backends,
            feasibility_checkers=feasibility_checkers,
            schema_registry=schemas,
            shadow_backends=shadow_registry,
            arena_bindings=dependencies.arena_bindings,
            run_budget_authority=run_budget_authority,
            allowed_initial_graph_digests=allowed_initial_graph_digests,
            freshness_context_provider=(
                (
                    dependencies.actor_providers[dependencies.freshness_context_provider_id]
                ).freshness_context
                if dependencies.freshness_context_provider_id is not None
                else None
            ),
        )
        for provider_id, provider in dependencies.actor_providers.items():
            bind_runtime = getattr(provider, "bind_episode_runtime", None)
            if callable(bind_runtime):
                try:
                    bind_runtime(episode)
                except Exception as exc:  # noqa: BLE001 - dependency admission boundary
                    raise DependencyAdmissionError(
                        f"Actor provider {provider_id!r} rejected the EpisodeRuntime binding: {exc}"
                    ) from exc
            bind_data_plane = getattr(provider, "bind_episode_data_plane", None)
            if not callable(bind_data_plane):
                continue
            try:
                bind_data_plane(episode.data_plane)
            except Exception as exc:  # noqa: BLE001 - dependency admission boundary
                raise DependencyAdmissionError(
                    f"Actor provider {provider_id!r} rejected the EpisodeDataPlane binding: {exc}"
                ) from exc
        for runner_ref, profile in sorted(dependencies.actor_profiles.items()):
            episode.register_actor_profile(runner_ref, profile)
        for binding in dependencies.observation_backends:
            episode.observations.register_backend(binding.backend)

        shadow = {
            binding.provenance.backend_id: binding.backend
            for binding in dependencies.shadow_backends
        }
        # Keep the validated index live in this scope: constructing it is the
        # exact coverage proof for every backend pin, including shadow worlds.
        if set(backend_index) != {pin.backend_id for pin in persisted_manifest.backends}:
            raise DependencyAdmissionError("Backend inventory changed during construction.")
        return V2Application(
            config=config,
            manifest=persisted_manifest,
            catalog=persisted_catalog,
            contracts=contracts,
            schemas=schemas,
            actors=actors,
            observations=episode.observations,
            episode=episode,
            run_budget_authority=run_budget_authority,
            shadow_registry=shadow_registry,
            shadow_backends=shadow,
        )

    @staticmethod
    def _validate_config(config: V2RuntimeConfig, manifest: RunManifest) -> None:
        if config.run_id != manifest.run_id:
            raise IdentityConflictError("Runtime config and manifest use different run IDs.")
        if config.graph_digest != manifest.graph_digest:
            raise IdentityConflictError("Runtime config and manifest graph digests differ.")
        if config.runtime_code_digest != manifest.runtime_code_digest:
            raise IdentityConflictError("Runtime config and manifest runtime-code digests differ.")

    @staticmethod
    def _initial_graph_allowset(manifest: RunManifest) -> frozenset[str]:
        """Validate the manifest graph allowlist before identity is persisted."""

        configured = manifest.metadata.get("allowed_initial_graph_digests", [])
        if not isinstance(configured, list) or any(
            not isinstance(value, str) for value in configured
        ):
            raise IdentityConflictError(
                "manifest.metadata.allowed_initial_graph_digests must be a strict JSON string list"
            )
        if len(configured) != len(set(configured)):
            raise IdentityConflictError(
                "manifest.metadata.allowed_initial_graph_digests contains duplicates"
            )
        allowed = frozenset((manifest.graph_digest, *configured))
        invalid = sorted(value for value in allowed if _SHA256_DIGEST_RE.fullmatch(value) is None)
        if invalid:
            raise IdentityConflictError("manifest contains a non-canonical initial graph digest")
        return allowed

    @staticmethod
    def _validate_protocol_schemas(
        catalog: ContractCatalogSnapshot,
        schemas: SchemaRegistry,
    ) -> None:
        missing = sorted(
            {
                slot.schema_id
                for protocol in catalog.protocols
                for slot in (*protocol.inputs, *protocol.outputs)
                if not schemas.is_registered(slot.schema_id)
            }
        )
        if missing:
            raise DependencyAdmissionError(
                "Contract catalog references unregistered payload schemas: "
                + ", ".join(missing)
                + "."
            )

    @staticmethod
    def _validate_actor_profiles(
        manifest: RunManifest,
        catalog: ContractCatalogSnapshot,
        dependencies: V2RuntimeDependencies,
    ) -> None:
        for provider_id, provider in dependencies.actor_providers.items():
            if not isinstance(provider_id, str) or not _RUNTIME_IDENTIFIER_RE.fullmatch(
                provider_id
            ):
                raise DependencyAdmissionError(f"Invalid actor provider ID {provider_id!r}.")
            if not isinstance(provider, AgentProvider):
                raise DependencyAdmissionError(
                    f"Actor provider {provider_id!r} does not implement AgentProvider."
                )
        for runner_ref, profile in dependencies.actor_profiles.items():
            if not isinstance(runner_ref, str) or not runner_ref.strip():
                raise DependencyAdmissionError("Actor runner_ref must not be empty.")
            if not isinstance(profile, ActorProfile):
                raise DependencyAdmissionError(
                    f"Runner {runner_ref!r} is not bound to an ActorProfile."
                )
            if profile.provider_id not in dependencies.actor_providers:
                raise DependencyAdmissionError(
                    f"Actor profile {profile.profile_id!r} references unavailable provider "
                    f"{profile.provider_id!r}."
                )

        profiles: dict[str, ActorProfile] = {}
        for profile in dependencies.actor_profiles.values():
            previous = profiles.get(profile.profile_id)
            if previous is not None and previous != profile:
                raise DependencyAdmissionError(
                    f"Actor profile ID {profile.profile_id!r} has conflicting definitions."
                )
            profiles[profile.profile_id] = profile

        pins = {pin.component_id: pin for pin in manifest.actor_profile_pins}
        if set(pins) != set(profiles):
            raise DependencyAdmissionError(
                "Actor-profile pins must exactly cover the runtime actor profiles."
            )
        for profile_id, profile in profiles.items():
            pin = pins[profile_id]
            if pin.revision not in {None, 1}:
                raise DependencyAdmissionError(
                    f"Runtime actor profile {profile_id!r} supports only revision 1 pins."
                )
            if pin.content_digest != actor_profile_digest(profile):
                raise DependencyAdmissionError(
                    f"Actor profile {profile_id!r} has content hash drift."
                )

        for skill in catalog.skills:
            missing = set(skill.compatible_actor_profiles).difference(profiles)
            if missing:
                raise DependencyAdmissionError(
                    f"Skill {skill.skill_id!r} references unavailable actor profiles: "
                    + ", ".join(sorted(missing))
                    + "."
                )

    @staticmethod
    def _validate_arena_bindings(
        manifest: RunManifest,
        dependencies: V2RuntimeDependencies,
    ) -> None:
        bindings = {item.binding_id: item for item in dependencies.arena_bindings}
        if len(bindings) != len(dependencies.arena_bindings):
            raise DependencyAdmissionError("Arena binding IDs must be unique.")
        raw_pins = manifest.metadata.get("arena_binding_digests", {})
        if not isinstance(raw_pins, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in raw_pins.items()
        ):
            raise DependencyAdmissionError(
                "Manifest arena_binding_digests must be a string mapping."
            )
        pins = dict(raw_pins)
        if set(pins) != set(bindings):
            raise DependencyAdmissionError(
                "Manifest Arena pins must exactly cover runtime Arena bindings."
            )
        profiles = {profile.profile_id: profile for profile in dependencies.actor_profiles.values()}
        for binding_id, binding in bindings.items():
            if pins[binding_id] != binding.content_digest:
                raise DependencyAdmissionError(
                    f"Arena binding {binding_id!r} has content hash drift."
                )
            for candidate in binding.candidates:
                admitted = profiles.get(candidate.profile.profile_id)
                if admitted is None or admitted != candidate.profile:
                    raise DependencyAdmissionError(
                        f"Arena candidate profile {candidate.profile.profile_id!r} "
                        "is not an exact manifest-pinned ActorProfile."
                    )

    @staticmethod
    def _validate_backends(
        manifest: RunManifest,
        dependencies: V2RuntimeDependencies,
    ) -> dict[str, BackendProvenance]:
        pins = {pin.backend_id: pin for pin in manifest.backends}
        admitted: dict[str, BackendProvenance] = {}
        action_keys: set[tuple[str, str]] = set()
        observation_ids: set[str] = set()
        shadow_ids: set[str] = set()

        def admit(provenance: BackendProvenance, role: BackendRole) -> None:
            pin = pins.get(provenance.backend_id)
            if pin is None:
                raise DependencyAdmissionError(
                    f"Runtime backend {provenance.backend_id!r} is not manifest-pinned."
                )
            if pin.role is not role:
                raise DependencyAdmissionError(
                    f"Runtime backend {provenance.backend_id!r} has role {role.value!r}, "
                    f"but its manifest role is {pin.role.value!r}."
                )
            provenance.assert_matches(pin)
            previous = admitted.get(provenance.backend_id)
            if previous is not None and previous != provenance:
                raise DependencyAdmissionError(
                    f"Backend {provenance.backend_id!r} has conflicting provenance."
                )
            admitted[provenance.backend_id] = provenance

        for binding in dependencies.action_backends:
            key = (binding.world_id, binding.resource_id)
            if key in action_keys:
                raise DependencyAdmissionError(
                    f"Duplicate authoritative action binding for {key!r}."
                )
            action_keys.add(key)
            admit(binding.provenance, BackendRole.AUTHORITATIVE)
        for binding in dependencies.observation_backends:
            backend_id = binding.provenance.backend_id
            if backend_id in observation_ids:
                raise DependencyAdmissionError(
                    f"Duplicate observation backend binding {backend_id!r}."
                )
            observation_ids.add(backend_id)
            admit(binding.provenance, BackendRole.PERCEPTION)
        for binding in dependencies.shadow_backends:
            backend_id = binding.provenance.backend_id
            if backend_id in shadow_ids:
                raise DependencyAdmissionError(f"Duplicate shadow backend binding {backend_id!r}.")
            shadow_ids.add(backend_id)
            admit(binding.provenance, BackendRole.SHADOW)

        if set(admitted) != set(pins):
            missing = sorted(set(pins).difference(admitted))
            raise DependencyAdmissionError(
                "Manifest-pinned backends are not bound to runtime dependencies: "
                + ", ".join(missing)
                + "."
            )
        return admitted


def _canonical_document(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise IdentityIntegrityError("Identity document is not finite canonical JSON.") from exc
    return (text + "\n").encode("utf-8")


def _atomic_create_read_only(path: Path, payload: bytes) -> bool:
    if path.exists() or path.is_symlink():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staging", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _write_staged_document(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise IdentityIntegrityError(f"Missing identity document at {path}.") from exc
    except OSError as exc:
        raise IdentityIntegrityError(f"Cannot safely open identity document at {path}.") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise IdentityIntegrityError(f"Identity document at {path} is not a regular file.")
        if info.st_mode & 0o222:
            raise IdentityIntegrityError(f"Identity document at {path} is writable.")
        if info.st_size > _MAX_IDENTITY_DOCUMENT_BYTES:
            raise IdentityIntegrityError(f"Identity document at {path} is too large.")
        chunks: list[bytes] = []
        remaining = _MAX_IDENTITY_DOCUMENT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > _MAX_IDENTITY_DOCUMENT_BYTES:
            raise IdentityIntegrityError(f"Identity document at {path} is too large.")
        return raw
    finally:
        os.close(descriptor)


def _require_identity_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise IdentityIntegrityError(f"Missing immutable run identity at {path}.") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise IdentityIntegrityError(f"Run identity at {path} is not a real directory.")
    if info.st_mode & 0o222:
        raise IdentityIntegrityError(f"Run identity directory at {path} is writable.")


def _discard_staging_directory(path: Path) -> None:
    if not path.exists() or not path.is_dir() or path.is_symlink():
        return
    try:
        os.chmod(path, 0o700)
    except FileNotFoundError:
        return
    for filename in (_MANIFEST_FILENAME, _CATALOG_FILENAME):
        child = path / filename
        try:
            os.chmod(child, 0o600)
            child.unlink()
        except FileNotFoundError:
            pass
    with suppress(FileNotFoundError):
        path.rmdir()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ActionBackendBinding",
    "BackendProvenance",
    "CatalogPinError",
    "ContractCatalogStore",
    "DependencyAdmissionError",
    "IdentityConflictError",
    "IdentityIntegrityError",
    "ObservationBackendBinding",
    "RunManifestStore",
    "RuntimeIdentityStore",
    "ShadowBackendBinding",
    "V2Application",
    "V2BootstrapError",
    "V2RuntimeConfig",
    "V2RuntimeDependencies",
    "V2RuntimeFactory",
    "actor_profile_digest",
    "validate_manifest_catalog",
]
