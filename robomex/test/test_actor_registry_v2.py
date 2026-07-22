"""Offline tests for the v2 actor lifecycle and provider boundary."""

from __future__ import annotations

import time

import pytest

from robomex.orchestration.actors import (
    ActorAuthorityError,
    ActorConflictError,
    ActorLifecycle,
    ActorProfile,
    ActorRegistry,
    ActorState,
    ActorStateError,
    InMemoryAgentProvider,
    InvocationSpec,
    IsolationPolicy,
    WorkspaceMode,
)


def _registry(provider: InMemoryAgentProvider | None = None) -> tuple[ActorRegistry, InMemoryAgentProvider]:
    provider = provider or InMemoryAgentProvider()
    return (
        ActorRegistry(
            {"memory": provider},
            namespace_root="episode-007",
            workspace_root="/tmp/robomex-actor-tests",
        ),
        provider,
    )


def _profile(
    profile_id: str = "motion-candidate",
    *,
    lifecycle: ActorLifecycle = ActorLifecycle.EPHEMERAL,
    capabilities: frozenset[str] = frozenset({"pointcloud.read", "ik.solve"}),
    effects: frozenset[str] = frozenset(),
    isolation: IsolationPolicy | None = None,
) -> ActorProfile:
    return ActorProfile(
        profile_id=profile_id,
        provider_id="memory",
        runner_kind="coding_worker",
        lifecycle=lifecycle,
        capability_ceiling=capabilities,
        effect_ceiling=effects,
        isolation=isolation or IsolationPolicy(world_id="shadow-01"),
    )


def _invocation(
    invocation_id: str = "proposal-001",
    *,
    objective: str = "Propose a collision-free bowl alignment step.",
    idempotency_key: str = "",
    capabilities: frozenset[str] = frozenset({"pointcloud.read"}),
    effects: frozenset[str] = frozenset(),
) -> InvocationSpec:
    return InvocationSpec(
        invocation_id=invocation_id,
        idempotency_key=idempotency_key,
        objective=objective,
        inputs={"bowl_track": {"artifact_id": "sha256:bowl-v3"}},
        output_contract={"proposal": "MotionProposal.v1"},
        requested_capabilities=capabilities,
        requested_effects=effects,
        budget={"model_calls": 2, "wall_time_s": 5.0},
    )


def test_ephemeral_worker_invokes_once_auto_retires_and_replays_idempotently() -> None:
    registry, provider = _registry()
    profile = _profile()

    handle = registry.spawn(profile, actor_id="candidate-a")
    assert handle.state is ActorState.ACTIVE
    result = handle.invoke(_invocation())

    assert result["actor_id"] == "candidate-a"
    assert handle.state is ActorState.RETIRED
    assert provider.calls == (
        ("spawn", "candidate-a", None),
        ("invoke", "candidate-a", "proposal-001"),
        ("retire", "candidate-a", None),
    )

    # A delivery retry returns the recorded result even though the worker has
    # already been retired; provider/model execution is not repeated.
    assert handle.invoke(_invocation()) is result
    assert len(provider.runtime_for("candidate-a").invocations) == 1

    with pytest.raises(ActorStateError, match="cannot invoke while retired"):
        handle.invoke(_invocation("proposal-002"))
    assert handle.retire() is False


def test_service_lifecycle_is_explicit_and_each_transition_is_idempotent() -> None:
    registry, provider = _registry()
    profile = _profile("bowl-tracker", lifecycle=ActorLifecycle.SERVICE)
    handle = registry.spawn(profile, actor_id="tracker-bowl")

    first = handle.invoke(_invocation("track-start"))
    assert first["invocation_id"] == "track-start"
    assert handle.suspend() is True
    assert handle.suspend() is False
    with pytest.raises(ActorStateError, match="cannot invoke while suspended"):
        handle.invoke(_invocation("track-update-paused"))

    assert handle.resume() is True
    assert handle.resume() is False
    second = handle.invoke(_invocation("track-update"))
    assert second["invocation_id"] == "track-update"
    assert handle.retire() is True
    assert handle.retire() is False
    with pytest.raises(ActorStateError, match="cannot resume while retired"):
        handle.resume()

    operations = [operation for operation, _, _ in provider.calls]
    assert operations == ["spawn", "invoke", "suspend", "resume", "invoke", "retire"]


@pytest.mark.parametrize(
    ("capabilities", "effects", "expected"),
    [
        (frozenset({"shell.root"}), frozenset(), "capabilities"),
        (frozenset(), frozenset({"authoritative_world.write"}), "effects"),
    ],
)
def test_authority_ceiling_rejects_before_provider_invocation(
    capabilities: frozenset[str], effects: frozenset[str], expected: str
) -> None:
    registry, provider = _registry()
    handle = registry.spawn(
        _profile("read-only-reviewer", lifecycle=ActorLifecycle.SERVICE),
        actor_id="reviewer",
    )

    with pytest.raises(ActorAuthorityError, match=expected):
        handle.invoke(
            _invocation(
                capabilities=capabilities,
                effects=effects,
            )
        )

    assert provider.calls == (("spawn", "reviewer", None),)
    assert handle.state is ActorState.ACTIVE


def test_explicit_effect_subset_is_allowed_but_not_implicitly_broadened() -> None:
    registry, provider = _registry()
    handle = registry.spawn(
        _profile(
            "shadow-rollout",
            lifecycle=ActorLifecycle.SERVICE,
            effects=frozenset({"shadow_world.write", "render.write"}),
        ),
        actor_id="rollout-a",
    )

    result = handle.invoke(
        _invocation(
            effects=frozenset({"shadow_world.write"}),
        )
    )
    assert result["profile_id"] == "shadow-rollout"
    assert provider.runtime_for("rollout-a").invocations[0].requested_effects == frozenset(
        {"shadow_world.write"}
    )


def test_invocation_key_replay_and_conflicting_reuse() -> None:
    calls = 0

    def handler(profile: ActorProfile, spec: InvocationSpec, isolation: object) -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"call": calls}

    registry, _ = _registry(InMemoryAgentProvider(handler))
    handle = registry.spawn(
        _profile("selector", lifecycle=ActorLifecycle.SERVICE), actor_id="selector"
    )
    request = _invocation("attempt-a", idempotency_key="arena-slot-3")

    assert handle.invoke(request) == {"call": 1}
    assert handle.invoke(request) == {"call": 1}
    assert calls == 1

    conflicting = _invocation(
        "attempt-b",
        objective="Use a different motion hypothesis.",
        idempotency_key="arena-slot-3",
    )
    with pytest.raises(ActorConflictError, match="different invocation content"):
        handle.invoke(conflicting)
    assert calls == 1


def test_registry_spawn_is_idempotent_but_never_rebinds_an_actor_id() -> None:
    registry, provider = _registry()
    profile = _profile("candidate")
    first = registry.spawn(profile, actor_id="candidate-01")

    assert registry.spawn(profile, actor_id="candidate-01") is first
    assert provider.calls == (("spawn", "candidate-01", None),)

    changed = _profile(
        "candidate",
        capabilities=frozenset({"pointcloud.read", "ik.solve", "render.write"}),
    )
    with pytest.raises(ActorConflictError, match="different profile"):
        registry.spawn(changed, actor_id="candidate-01")


def test_isolated_candidates_have_unique_namespaces_and_workspaces() -> None:
    registry, provider = _registry()
    profile = _profile()
    left = registry.spawn(profile, actor_id="candidate-left")
    right = registry.spawn(profile, actor_id="candidate-right")

    assert left.isolation.namespace_id != right.isolation.namespace_id
    assert left.isolation.workspace_id != right.isolation.workspace_id
    assert left.isolation.workspace_mode is WorkspaceMode.ISOLATED
    assert left.isolation.world_id == "shadow-01"
    assert left.isolation.metadata["lifecycle"] == "ephemeral"
    assert provider.runtime_for("candidate-left").isolation == left.isolation


def test_shared_assets_are_read_only_while_actor_namespaces_remain_unique() -> None:
    registry, _ = _registry()
    shared = IsolationPolicy(
        workspace_mode=WorkspaceMode.SHARED_READ_ONLY,
        workspace_key="scene-assets-v4",
        world_id="shadow-02",
    )
    profile = _profile("renderer", lifecycle=ActorLifecycle.SERVICE, isolation=shared)
    first = registry.spawn(profile, actor_id="renderer-a")
    second = registry.spawn(profile, actor_id="renderer-b")

    assert first.isolation.workspace_id == second.isolation.workspace_id
    assert first.isolation.workspace_mode is WorkspaceMode.SHARED_READ_ONLY
    assert first.isolation.namespace_id != second.isolation.namespace_id


def test_ephemeral_provider_failure_is_recorded_and_not_reexecuted() -> None:
    attempts = 0

    def fail(profile: ActorProfile, spec: InvocationSpec, isolation: object) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("deterministic failure")

    registry, provider = _registry(InMemoryAgentProvider(fail))
    handle = registry.spawn(_profile(), actor_id="failing-worker")
    request = _invocation()

    with pytest.raises(RuntimeError, match="deterministic failure"):
        handle.invoke(request)
    assert handle.state is ActorState.RETIRED
    with pytest.raises(RuntimeError, match="deterministic failure"):
        handle.invoke(request)
    assert attempts == 1
    assert [operation for operation, _, _ in provider.calls] == ["spawn", "invoke", "retire"]


def test_deadline_rejection_does_not_consume_an_ephemeral_worker() -> None:
    registry, provider = _registry()
    handle = registry.spawn(_profile(), actor_id="deadline-worker")
    expired = InvocationSpec(
        invocation_id="expired",
        objective="Do not execute.",
        deadline_monotonic_s=time.monotonic() - 1,
    )

    with pytest.raises(ActorStateError, match="passed its monotonic deadline"):
        handle.invoke(expired)
    assert handle.state is ActorState.ACTIVE
    assert provider.calls == (("spawn", "deadline-worker", None),)

    handle.invoke(_invocation("valid"))
    assert handle.state is ActorState.RETIRED


def test_invocation_fingerprint_excludes_only_derived_monotonic_deadline() -> None:
    base = InvocationSpec(
        invocation_id="bounded-call",
        objective="Produce one bounded proposal.",
        budget={"model_calls": 1, "tokens": 100, "wall_time_ms": 500},
        deadline_monotonic_s=10.0,
        metadata={"episode_id": "episode-a", "run_id": "run-a"},
    )
    restarted = InvocationSpec(
        invocation_id="bounded-call",
        objective=base.objective,
        budget=base.budget,
        deadline_monotonic_s=3.0,
        metadata=base.metadata,
    )
    changed_wall_grant = InvocationSpec(
        invocation_id="bounded-call",
        objective=base.objective,
        budget={"model_calls": 1, "tokens": 100, "wall_time_ms": 501},
        deadline_monotonic_s=3.0,
        metadata=base.metadata,
    )
    changed_run = InvocationSpec(
        invocation_id="bounded-call",
        objective=base.objective,
        budget=base.budget,
        deadline_monotonic_s=3.0,
        metadata={"episode_id": "episode-a", "run_id": "run-b"},
    )

    assert restarted.fingerprint() == base.fingerprint()
    assert changed_wall_grant.fingerprint() != base.fingerprint()
    assert changed_run.fingerprint() != base.fingerprint()


def test_registry_can_filter_and_retire_episode_services() -> None:
    registry, _ = _registry()
    worker = registry.spawn(_profile("worker"), actor_id="worker")
    service = registry.spawn(
        _profile("monitor", lifecycle=ActorLifecycle.SERVICE), actor_id="monitor"
    )

    assert registry.handles(lifecycle=ActorLifecycle.SERVICE) == (service,)
    assert set(registry.handles(state=ActorState.ACTIVE)) == {worker, service}
    registry.retire_all()
    assert worker.state is ActorState.RETIRED
    assert service.state is ActorState.RETIRED


def test_invalid_shared_workspace_and_path_like_actor_id_are_rejected() -> None:
    with pytest.raises(ValueError, match="requires workspace_key"):
        IsolationPolicy(workspace_mode=WorkspaceMode.SHARED_READ_ONLY)

    registry, _ = _registry()
    with pytest.raises(ValueError, match="actor_id"):
        registry.spawn(_profile(), actor_id="../escape")
