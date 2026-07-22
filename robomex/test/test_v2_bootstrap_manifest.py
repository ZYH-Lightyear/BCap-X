from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from robomex.contracts import (
    ContentPin,
    ContractCatalogSnapshot,
    ContractRegistry,
    EffectContract,
    FunctionExport,
    ProtocolSpec,
    SkillAsset,
    SkillAssetKind,
    SkillManifest,
)
from robomex.data import AdmissionPurpose, core_schema_registry
from robomex.evolution import (
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
from robomex.orchestration.actors import ActorProfile, InMemoryAgentProvider
from robomex.orchestration.bootstrap import (
    ActionBackendBinding,
    BackendProvenance,
    CatalogPinError,
    DependencyAdmissionError,
    IdentityConflictError,
    IdentityIntegrityError,
    ObservationBackendBinding,
    RunManifestStore,
    RuntimeIdentityStore,
    ShadowBackendBinding,
    V2RuntimeConfig,
    V2RuntimeDependencies,
    V2RuntimeFactory,
    actor_profile_digest,
)
from robomex.orchestration.bowl_provider import (
    BowlPlaceActorProvider,
    BowlPlaceProviderConfig,
    build_bowl_place_actor_bindings,
)
from robomex.orchestration.episode import install_runtime_schemas
from robomex.protocols.bowl_place import (
    BowlPlaceProtocolConfig,
    build_fixed_bowl_place_protocol,
)
from robomex.runtime.action_protocol import (
    AdmissionSnapshot,
    BackendCallResult,
    BackendDescriptor,
    WorldKind,
)
from robomex.runtime.evidence_recorder import ACTION_RUNTIME_SAMPLE_SCHEMA
from robomex.runtime.observation import InMemoryObservationBackend


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _runtime_schema_digest() -> str:
    return install_runtime_schemas(core_schema_registry()).content_digest


class _ActionBackend:
    def __init__(self, backend_id: str, *, shadow: bool = False) -> None:
        self.descriptor = BackendDescriptor(backend_id=backend_id)
        self._shadow = shadow

    def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot:
        return AdmissionSnapshot(
            world_id=world_id,
            world_kind=WorldKind.SHADOW if self._shadow else WorldKind.AUTHORITATIVE,
            resource_id=resource_id,
            robot_revision=1,
            scene_revision=1,
            attachment_revision=1,
            config_revision=1,
            joint_names=("joint_1",),
            joint_positions_rad=(0.0,),
            config_digest=_digest("c"),
            collision_world_digest=_digest("d"),
        )

    def execute_joint_path(self, **_kwargs: object) -> BackendCallResult:
        return BackendCallResult(converged=True)

    def set_gripper(self, **_kwargs: object) -> BackendCallResult:
        return BackendCallResult(converged=True)

    def wait(self, **_kwargs: object) -> BackendCallResult:
        return BackendCallResult(converged=True)


class _Checker:
    def certify(self, spec, snapshot):
        raise AssertionError("bootstrap must not execute feasibility checks")


def _catalog() -> tuple[ContractCatalogSnapshot, SkillManifest]:
    effect = EffectContract(contract_id="effect.read")
    protocol = ProtocolSpec(
        protocol_id="protocol.inspect",
        summary="Inspect an episode without hidden effects.",
        effect_contract=effect,
    )
    skill = SkillManifest(
        skill_id="skill.inspect",
        name="Inspect",
        summary="Pinned baseline inspection skill.",
        assets=(
            SkillAsset(
                asset_id="code.inspect",
                kind=SkillAssetKind.CODE,
                relative_path="scripts/inspect.py",
                content_digest=_digest("1"),
                media_type="text/x-python",
            ),
        ),
        functions=(
            FunctionExport(
                function_id="function.inspect",
                source_asset_id="code.inspect",
                entrypoint="scripts/inspect.py:inspect",
                function_digest=_digest("2"),
                interface_digest=_digest("3"),
            ),
        ),
        protocol_pins=(
            ContentPin(
                component_id=protocol.protocol_id,
                revision=protocol.revision,
                content_digest=protocol.content_digest,
            ),
        ),
        compatible_actor_profiles=("actor.worker",),
    )
    registry = ContractRegistry()
    registry.register_protocol(protocol)
    registry.register_skill(skill)
    return registry.snapshot(), skill


def _backend_pin(backend_id: str, role: BackendRole, character: str) -> BackendPin:
    return BackendPin(
        backend_id=backend_id,
        role=role,
        implementation_digest=_digest(character),
        configuration_digest=_digest("0"),
        version="1.0.0",
    )


def _fixtures(tmp_path: Path):
    catalog, skill = _catalog()
    profile = ActorProfile(
        profile_id="actor.worker",
        provider_id="memory",
        runner_kind="coding_worker",
        capability_ceiling=frozenset({"observation.read"}),
    )
    action_pin = _backend_pin("backend.robot", BackendRole.AUTHORITATIVE, "a")
    perception_pin = _backend_pin("backend.camera", BackendRole.PERCEPTION, "b")
    shadow_pin = _backend_pin("backend.shadow", BackendRole.SHADOW, "e")
    manifest = RunManifest(
        run_id="run.bootstrap.1",
        task=TaskSnapshot(
            task_id="task.inspect",
            instruction="Inspect the workspace.",
            success_rubric="Inspection evidence is present.",
            episode_spec_digest=_digest("4"),
        ),
        seed=7,
        model=ModelPin(
            model_id="model.local",
            provider_id="provider.local",
            weights_digest=_digest("5"),
            generation_config_digest=_digest("6"),
        ),
        prompts=(PromptPin(prompt_id="prompt.worker", content_digest=_digest("7")),),
        skills=(
            SkillPin(
                skill_id=skill.skill_id,
                revision=skill.revision,
                manifest_digest=skill.content_digest,
            ),
        ),
        functions=(
            FunctionPin(
                function_id=skill.functions[0].function_id,
                skill_id=skill.skill_id,
                implementation_digest=skill.functions[0].function_digest,
                interface_digest=skill.functions[0].interface_digest,
            ),
        ),
        budgets=RunBudgets(
            max_model_calls=3,
            max_tokens=1000,
            max_wall_time_s=30,
            max_physical_actions=1,
        ),
        backends=(action_pin, perception_pin, shadow_pin),
        graph_digest=_digest("8"),
        contract_catalog_digest=catalog.content_digest,
        schema_registry_digest=_runtime_schema_digest(),
        runtime_code_digest=_digest("9"),
        actor_profile_pins=(
            ContentPin(
                component_id=profile.profile_id,
                revision=1,
                content_digest=actor_profile_digest(profile),
            ),
        ),
    )
    action = _ActionBackend("backend.robot")
    camera = InMemoryObservationBackend("backend.camera")
    shadow = _ActionBackend("backend.shadow", shadow=True)
    dependencies = V2RuntimeDependencies(
        contract_catalog=catalog,
        actor_providers={"memory": InMemoryAgentProvider()},
        actor_profiles={"worker": profile},
        action_backends=(
            ActionBackendBinding(
                world_id="world.live",
                resource_id="robot.arm",
                backend=action,
                feasibility_checker=_Checker(),
                provenance=BackendProvenance.from_pin(action_pin),
            ),
        ),
        observation_backends=(
            ObservationBackendBinding(
                backend=camera,
                provenance=BackendProvenance.from_pin(perception_pin),
            ),
        ),
        shadow_backends=(
            ShadowBackendBinding(
                backend=shadow,
                provenance=BackendProvenance.from_pin(shadow_pin),
                world_resource_bindings=frozenset({("shadow-world", "arm")}),
            ),
        ),
    )
    config = V2RuntimeConfig(
        run_id=manifest.run_id,
        episode_id="episode_bootstrap_1",
        episode_root=tmp_path / "episode",
        graph_digest=manifest.graph_digest,
        runtime_code_digest=manifest.runtime_code_digest,
    )
    return config, manifest, dependencies


def _rebuild_manifest(manifest: RunManifest, **updates: object) -> RunManifest:
    payload = manifest.model_dump(mode="python", exclude={"content_digest"})
    payload.update(updates)
    return RunManifest.model_validate(payload)


def test_factory_builds_all_registries_and_reopens_same_identity(tmp_path: Path) -> None:
    config, manifest, dependencies = _fixtures(tmp_path)
    first = V2RuntimeFactory().build(
        config=config,
        manifest=manifest,
        dependencies=dependencies,
    )
    assert first.manifest.content_digest == manifest.content_digest
    assert first.contracts.snapshot().content_digest == manifest.contract_catalog_digest
    assert first.schemas.missing_core_schema_ids == ()
    assert first.schemas.is_registered(ACTION_RUNTIME_SAMPLE_SCHEMA)
    assert first.observations.episode_id == config.episode_id
    assert set(first.shadow_backends) == {"backend.shadow"}
    assert first.episode.actors is first.actors
    assert first.episode.schema_registry is first.schemas
    assert first.episode.shadow_backends is first.shadow_registry
    assert first.episode.arena_consumption_ledger.persistent is True
    # ``None`` is the production default: EpisodeRuntime creates a fresh,
    # workflow/action-scoped recorder at dispatch instead of sharing one
    # mutable recorder across actions. Explicit injection remains test-only.
    assert first.episode._action_evidence_recorder is None

    identity_root = config.identity_root
    manifest_path = identity_root / "run_manifest.v2.json"
    catalog_path = identity_root / "contract_catalog.v1.json"
    assert manifest_path.exists() and catalog_path.exists()
    assert manifest_path.stat().st_mode & 0o222 == 0
    assert catalog_path.stat().st_mode & 0o222 == 0

    second = V2RuntimeFactory().build(
        config=config,
        manifest=manifest,
        dependencies=dependencies,
    )
    assert second.manifest == first.manifest


def test_factory_binds_production_bowl_provider_and_freshness_context(
    tmp_path: Path,
) -> None:
    config, manifest, dependencies = _fixtures(tmp_path)
    bowl_provider = BowlPlaceActorProvider(
        BowlPlaceProviderConfig(
            observation_backend_id="backend.camera",
            authority_world_id="world.live",
        )
    )
    protocol = build_fixed_bowl_place_protocol(
        BowlPlaceProtocolConfig(authority_world_id="world.live")
    )
    bowl_bindings = build_bowl_place_actor_bindings(
        protocol.spec,
        provider=bowl_provider,
    )
    actor_profiles = {
        **dependencies.actor_profiles,
        **bowl_bindings.profiles,
    }
    profile_by_id = {profile.profile_id: profile for profile in actor_profiles.values()}
    manifest = _rebuild_manifest(
        manifest,
        actor_profile_pins=tuple(
            ContentPin(
                component_id=profile_id,
                revision=1,
                content_digest=actor_profile_digest(profile),
            )
            for profile_id, profile in sorted(profile_by_id.items())
        ),
    )
    admitted = V2RuntimeDependencies(
        contract_catalog=dependencies.contract_catalog,
        actor_providers={
            **dependencies.actor_providers,
            **bowl_bindings.providers,
        },
        actor_profiles=actor_profiles,
        action_backends=dependencies.action_backends,
        observation_backends=dependencies.observation_backends,
        shadow_backends=dependencies.shadow_backends,
        freshness_context_provider_id=bowl_bindings.provider_id,
    )

    app = V2RuntimeFactory().build(
        config=config,
        manifest=manifest,
        dependencies=admitted,
    )

    context = bowl_provider.freshness_context(AdmissionPurpose.VERIFICATION)
    assert context.current_state_revision == app.episode.state_reducer.state.revision
    assert app.episode._freshness_context_provider == bowl_provider.freshness_context
    assert set(bowl_bindings.profiles).issubset(app.episode._profiles)


def test_shadow_dependency_must_prove_shadow_world_kind(tmp_path: Path) -> None:
    _config, manifest, _dependencies = _fixtures(tmp_path)
    shadow_pin = next(pin for pin in manifest.backends if pin.role is BackendRole.SHADOW)

    with pytest.raises(DependencyAdmissionError, match="WorldKind.SHADOW"):
        ShadowBackendBinding(
            backend=_ActionBackend(shadow_pin.backend_id, shadow=False),
            provenance=BackendProvenance.from_pin(shadow_pin),
            world_resource_bindings=frozenset({("claimed-shadow", "arm")}),
        )


def test_reopen_rejects_changed_manifest_even_with_same_run_id(tmp_path: Path) -> None:
    config, manifest, dependencies = _fixtures(tmp_path)
    V2RuntimeFactory().build(
        config=config,
        manifest=manifest,
        dependencies=dependencies,
    )
    changed = _rebuild_manifest(manifest, seed=manifest.seed + 1)
    with pytest.raises(IdentityConflictError, match="digest"):
        V2RuntimeFactory().build(
            config=config,
            manifest=changed,
            dependencies=dependencies,
        )


def test_factory_rejects_schema_validator_inventory_drift(tmp_path: Path) -> None:
    config, manifest, dependencies = _fixtures(tmp_path)
    drifted = _rebuild_manifest(manifest, schema_registry_digest=_digest("f"))

    with pytest.raises(IdentityConflictError, match="schema-registry"):
        V2RuntimeFactory().build(
            config=config,
            manifest=drifted,
            dependencies=dependencies,
        )

    assert not config.episode_root.exists()


def test_tampered_persisted_manifest_fails_integrity_check(tmp_path: Path) -> None:
    config, manifest, dependencies = _fixtures(tmp_path)
    store = RuntimeIdentityStore(config.episode_root)
    store.bind(manifest, dependencies.contract_catalog)
    path = config.identity_root / "run_manifest.v2.json"
    path.chmod(0o644)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["seed"] += 1
    path.write_text(json.dumps(document), encoding="utf-8")
    path.chmod(0o444)
    with pytest.raises(IdentityIntegrityError, match="Invalid run manifest"):
        store.load(expected_run_id=manifest.run_id)


def test_catalog_and_runtime_dependency_drift_fail_before_episode_creation(
    tmp_path: Path,
) -> None:
    config, manifest, dependencies = _fixtures(tmp_path)
    wrong_catalog_manifest = _rebuild_manifest(
        manifest,
        contract_catalog_digest=_digest("f"),
    )
    with pytest.raises(CatalogPinError, match="catalog digest"):
        V2RuntimeFactory().build(
            config=config,
            manifest=wrong_catalog_manifest,
            dependencies=dependencies,
        )
    assert not config.episode_root.exists()

    incomplete = V2RuntimeDependencies(
        contract_catalog=dependencies.contract_catalog,
        actor_providers=dependencies.actor_providers,
        actor_profiles=dependencies.actor_profiles,
        action_backends=dependencies.action_backends,
        observation_backends=dependencies.observation_backends,
    )
    with pytest.raises(DependencyAdmissionError, match="backend.shadow"):
        V2RuntimeFactory().build(
            config=config,
            manifest=manifest,
            dependencies=incomplete,
        )
    assert not config.episode_root.exists()


def test_actor_profile_pin_drift_is_rejected(tmp_path: Path) -> None:
    config, manifest, dependencies = _fixtures(tmp_path)
    changed_profile = ActorProfile(
        profile_id="actor.worker",
        provider_id="memory",
        runner_kind="coding_worker",
        capability_ceiling=frozenset({"observation.read", "artifact.publish"}),
    )
    changed_dependencies = V2RuntimeDependencies(
        contract_catalog=dependencies.contract_catalog,
        actor_providers=dependencies.actor_providers,
        actor_profiles={"worker": changed_profile},
        action_backends=dependencies.action_backends,
        observation_backends=dependencies.observation_backends,
        shadow_backends=dependencies.shadow_backends,
    )
    with pytest.raises(DependencyAdmissionError, match="content hash drift"):
        V2RuntimeFactory().build(
            config=config,
            manifest=manifest,
            dependencies=changed_dependencies,
        )
    assert not config.episode_root.exists()


def test_concurrent_identity_binding_is_atomic_and_create_once(tmp_path: Path) -> None:
    config, manifest, dependencies = _fixtures(tmp_path)

    def bind_identity():
        return RuntimeIdentityStore(config.episode_root).bind(
            manifest, dependencies.contract_catalog
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(lambda _index: bind_identity(), range(24)))
    assert all(item == (manifest, dependencies.contract_catalog) for item in results)
    assert not tuple(config.episode_root.glob("*.staging"))
    assert not tuple(config.episode_root.glob(".*.staging"))


def test_standalone_manifest_store_never_overwrites_existing_identity(
    tmp_path: Path,
) -> None:
    _config, manifest, _dependencies = _fixtures(tmp_path)
    path = tmp_path / "standalone" / "run_manifest.v2.json"
    store = RunManifestStore(path)
    assert store.bind(manifest) == manifest
    assert path.stat().st_mode & 0o222 == 0

    changed = _rebuild_manifest(manifest, seed=manifest.seed + 10)
    with pytest.raises(IdentityConflictError, match="digest"):
        store.bind(changed)
    assert store.load(expected_digest=manifest.content_digest) == manifest
