"""Lifecycle and authority boundary for RoboMEx v2 actors.

The legacy :class:`robomex.authoring.adapters.SubAgentFactory` constructs a
temporary CodingAgent for each graph node.  This module deliberately does not
wrap that control flow.  It defines the smaller provider boundary used by the
v2 runtime so that a coding worker, tracker, monitor, or deterministic service
can have an explicit lifetime and an isolated execution identity.

Providers own model/tool specific resources.  :class:`AgentHandle` owns the
state machine, invocation idempotency, and authority checks.  Consequently a
provider cannot accidentally broaden the capability or effect ceiling declared
by an :class:`ActorProfile`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MISSING = object()


class ActorError(RuntimeError):
    """Base class for v2 actor lifecycle errors."""


class ActorStateError(ActorError):
    """Raised when an operation is illegal in the actor's current state."""


class ActorAuthorityError(ActorError, PermissionError):
    """Raised before provider invocation when requested authority is too broad."""


class ActorConflictError(ActorError):
    """Raised when an idempotency key is reused with different content."""


class ActorNotFoundError(ActorError, KeyError):
    """Raised when an actor or provider is not registered."""


class ActorLifecycle(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    """Lifetime policy of an actor instance.

    ``EPHEMERAL`` workers accept one distinct invocation and are retired after
    that invocation attempt.  A byte-identical retry is still served from the
    handle's idempotency record.  ``SERVICE`` actors remain alive across graph
    nodes (and, when owned by an episode registry, across subgoals) until the
    orchestrator explicitly retires them.
    """

    EPHEMERAL = "ephemeral"
    SERVICE = "service"


class ActorState(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    """Externally observable AgentHandle states."""

    NEW = "new"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RETIRED = "retired"


class WorkspaceMode(str, Enum):  # noqa: UP042 - package still declares Python 3.10 support
    """How a provider may expose a workspace to an actor.

    No shared writable mode is provided intentionally.  Candidate workers get
    isolated workspaces by default; shared assets must be mounted read-only.
    These values are enforcement metadata for a provider/sandbox adapter, not a
    claim that this module itself creates an OS sandbox.
    """

    ISOLATED = "isolated"
    SHARED_READ_ONLY = "shared_read_only"


def _identifier(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized or _IDENTIFIER_RE.fullmatch(normalized) is None:
        raise ValueError(
            f"{field_name} must match {_IDENTIFIER_RE.pattern!r}; got {value!r}."
        )
    return normalized


def _string_set(values: frozenset[str] | set[str] | tuple[str, ...]) -> frozenset[str]:
    normalized = frozenset(str(value).strip() for value in values)
    if "" in normalized:
        raise ValueError("Capability and effect names must be non-empty strings.")
    return normalized


def _frozen_mapping(values: Mapping[str, Any], *, field_name: str) -> Mapping[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings.")
        copied[key] = value
    return MappingProxyType(copied)


@dataclass(frozen=True)
class IsolationPolicy:
    """Requested namespace/workspace/world isolation for an actor profile."""

    namespace_prefix: str = "actors"
    workspace_mode: WorkspaceMode = WorkspaceMode.ISOLATED
    workspace_key: str | None = None
    world_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "namespace_prefix",
            _identifier(self.namespace_prefix, field_name="namespace_prefix"),
        )
        object.__setattr__(self, "workspace_mode", WorkspaceMode(self.workspace_mode))
        if self.workspace_key is not None:
            object.__setattr__(
                self,
                "workspace_key",
                _identifier(self.workspace_key, field_name="workspace_key"),
            )
        if self.workspace_mode is WorkspaceMode.SHARED_READ_ONLY and not self.workspace_key:
            raise ValueError("A shared read-only workspace requires workspace_key.")
        if self.world_id is not None:
            object.__setattr__(
                self, "world_id", _identifier(self.world_id, field_name="world_id")
            )
        object.__setattr__(
            self,
            "metadata",
            _frozen_mapping(self.metadata, field_name="isolation metadata"),
        )


@dataclass(frozen=True)
class ActorIsolation:
    """Resolved, actor-specific isolation metadata passed to a provider."""

    owner_actor_id: str
    namespace_id: str
    workspace_id: str
    workspace_mode: WorkspaceMode
    world_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "owner_actor_id", _identifier(self.owner_actor_id, field_name="actor_id")
        )
        if not self.namespace_id.strip():
            raise ValueError("namespace_id must not be empty.")
        if not self.workspace_id.strip():
            raise ValueError("workspace_id must not be empty.")
        object.__setattr__(self, "workspace_mode", WorkspaceMode(self.workspace_mode))
        object.__setattr__(
            self,
            "metadata",
            _frozen_mapping(self.metadata, field_name="resolved isolation metadata"),
        )


@dataclass(frozen=True)
class ActorProfile:
    """Evolvable actor configuration, independent of one invocation.

    ``capability_ceiling`` describes tools/data the actor may access.
    ``effect_ceiling`` describes effects it may propose or commit.  The empty
    effect ceiling is the safe default for read-only candidate and monitoring
    actors.  Both are ceilings: an InvocationSpec must request an explicit
    subset before a provider is called.
    """

    profile_id: str
    provider_id: str = "in_memory"
    runner_kind: str = "coding_worker"
    lifecycle: ActorLifecycle = ActorLifecycle.EPHEMERAL
    model: str = ""
    capability_ceiling: frozenset[str] = field(default_factory=frozenset)
    effect_ceiling: frozenset[str] = field(default_factory=frozenset)
    isolation: IsolationPolicy = field(default_factory=IsolationPolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "profile_id", _identifier(self.profile_id, field_name="profile_id")
        )
        object.__setattr__(
            self, "provider_id", _identifier(self.provider_id, field_name="provider_id")
        )
        object.__setattr__(
            self, "runner_kind", _identifier(self.runner_kind, field_name="runner_kind")
        )
        object.__setattr__(self, "lifecycle", ActorLifecycle(self.lifecycle))
        object.__setattr__(
            self, "capability_ceiling", _string_set(self.capability_ceiling)
        )
        object.__setattr__(self, "effect_ceiling", _string_set(self.effect_ceiling))
        if not isinstance(self.isolation, IsolationPolicy):
            raise TypeError("isolation must be an IsolationPolicy.")
        object.__setattr__(
            self, "metadata", _frozen_mapping(self.metadata, field_name="profile metadata")
        )


@dataclass(frozen=True)
class InvocationSpec:
    """One bounded, typed request to an actor instance."""

    invocation_id: str
    objective: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    output_contract: Mapping[str, str] = field(default_factory=dict)
    requested_capabilities: frozenset[str] = field(default_factory=frozenset)
    requested_effects: frozenset[str] = field(default_factory=frozenset)
    budget: Mapping[str, float] = field(default_factory=dict)
    deadline_monotonic_s: float | None = None
    idempotency_key: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "invocation_id",
            _identifier(self.invocation_id, field_name="invocation_id"),
        )
        objective = str(self.objective).strip()
        if not objective:
            raise ValueError("objective must not be empty.")
        object.__setattr__(self, "objective", objective)
        object.__setattr__(
            self, "inputs", _frozen_mapping(self.inputs, field_name="invocation inputs")
        )
        output_contract = _frozen_mapping(
            self.output_contract, field_name="output contract"
        )
        if any(not isinstance(value, str) or not value.strip() for value in output_contract.values()):
            raise ValueError("output_contract values must be non-empty schema names.")
        object.__setattr__(self, "output_contract", output_contract)
        object.__setattr__(
            self,
            "requested_capabilities",
            _string_set(self.requested_capabilities),
        )
        object.__setattr__(
            self, "requested_effects", _string_set(self.requested_effects)
        )
        budget = _frozen_mapping(self.budget, field_name="invocation budget")
        for name, amount in budget.items():
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                raise ValueError(f"Budget {name!r} must be numeric.")
            if not math.isfinite(float(amount)) or float(amount) < 0:
                raise ValueError(f"Budget {name!r} must be finite and non-negative.")
        object.__setattr__(self, "budget", budget)
        if self.deadline_monotonic_s is not None:
            deadline = float(self.deadline_monotonic_s)
            if not math.isfinite(deadline):
                raise ValueError("deadline_monotonic_s must be finite.")
            object.__setattr__(self, "deadline_monotonic_s", deadline)
        if self.idempotency_key:
            object.__setattr__(
                self,
                "idempotency_key",
                _identifier(self.idempotency_key, field_name="idempotency_key"),
            )
        object.__setattr__(
            self,
            "metadata",
            _frozen_mapping(self.metadata, field_name="invocation metadata"),
        )

    @property
    def effective_idempotency_key(self) -> str:
        return self.idempotency_key or self.invocation_id

    def fingerprint(self) -> str:
        """Content digest used to reject ambiguous idempotency-key reuse.

        ``deadline_monotonic_s`` is deliberately excluded.  It is a derived,
        process-local execution deadline and therefore changes after restart.
        The durable identity still binds the typed wall-time grant in
        ``budget`` and the episode/run/graph identity carried by ``metadata``.
        A retry may tighten this derived deadline, but may not change any
        semantic request or budget content.
        """

        payload = {
            "invocation_id": self.invocation_id,
            "objective": self.objective,
            "inputs": self.inputs,
            "output_contract": self.output_contract,
            "requested_capabilities": self.requested_capabilities,
            "requested_effects": self.requested_effects,
            "budget": self.budget,
            "idempotency_key": self.effective_idempotency_key,
            "metadata": self.metadata,
        }
        canonical = json.dumps(
            _canonical_value(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_value(value: Any) -> Any:
    """Produce stable JSON-ish content for an invocation fingerprint."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Invocation content cannot contain NaN or infinity.")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    # Inputs may carry opaque provider handles.  Their repr becomes part of the
    # idempotency boundary; typed artifact refs should normally avoid this path.
    return {"__type__": type(value).__qualname__, "__repr__": repr(value)}


@runtime_checkable
class AgentProvider(Protocol):
    """Provider adapter for a model, deterministic service, or local worker.

    Implementations must make lifecycle methods safe to retry after process
    uncertainty.  AgentHandle prevents duplicate calls in one process, while a
    durable provider key is required for crash recovery in a later milestone.
    """

    def spawn(self, profile: ActorProfile, isolation: ActorIsolation) -> Any:
        """Allocate provider-owned state for one actor."""

    def invoke(self, runtime: Any, spec: InvocationSpec) -> Any:
        """Execute one invocation and return provider-specific typed output."""

    def suspend(self, runtime: Any) -> None:
        """Pause an active service without discarding its provider state."""

    def resume(self, runtime: Any) -> None:
        """Resume a suspended service."""

    def retire(self, runtime: Any) -> None:
        """Release provider-owned state permanently."""


@dataclass
class _InvocationRecord:
    fingerprint: str
    value: Any = _MISSING
    error: Exception | None = None


class AgentHandle:
    """One spawned actor with checked, idempotent lifecycle transitions."""

    def __init__(
        self,
        *,
        actor_id: str,
        profile: ActorProfile,
        provider: AgentProvider,
        isolation: ActorIsolation,
    ) -> None:
        self._actor_id = _identifier(actor_id, field_name="actor_id")
        self._profile = profile
        self._provider = provider
        self._isolation = isolation
        self._state = ActorState.NEW
        self._runtime: Any = None
        self._records: dict[str, _InvocationRecord] = {}
        self._ephemeral_consumed = False
        self._lock = threading.RLock()

    @classmethod
    def spawn(
        cls,
        *,
        actor_id: str,
        profile: ActorProfile,
        provider: AgentProvider,
        isolation: ActorIsolation,
    ) -> AgentHandle:
        """Spawn a provider runtime and expose a handle only after success."""

        handle = cls(
            actor_id=actor_id,
            profile=profile,
            provider=provider,
            isolation=isolation,
        )
        runtime = provider.spawn(profile, isolation)
        handle._runtime = runtime
        handle._state = ActorState.ACTIVE
        return handle

    @property
    def actor_id(self) -> str:
        return self._actor_id

    @property
    def profile(self) -> ActorProfile:
        return self._profile

    @property
    def isolation(self) -> ActorIsolation:
        return self._isolation

    @property
    def state(self) -> ActorState:
        with self._lock:
            return self._state

    @property
    def runtime(self) -> Any:
        """Provider-owned runtime, exposed read-only for adapters/diagnostics."""

        return self._runtime

    def invoke(self, spec: InvocationSpec) -> Any:
        """Invoke the actor once, enforcing authority and idempotency first."""

        key = spec.effective_idempotency_key
        fingerprint = spec.fingerprint()
        with self._lock:
            previous = self._records.get(key)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    raise ActorConflictError(
                        f"Actor {self.actor_id!r} reused idempotency key {key!r} "
                        "with different invocation content."
                    )
                if previous.error is not None:
                    raise previous.error
                return previous.value

            if self._state is not ActorState.ACTIVE:
                raise ActorStateError(
                    f"Actor {self.actor_id!r} cannot invoke while {self._state.value}."
                )
            if (
                self._profile.lifecycle is ActorLifecycle.EPHEMERAL
                and self._ephemeral_consumed
            ):
                raise ActorStateError(
                    f"Ephemeral actor {self.actor_id!r} already consumed its invocation."
                )
            self._check_authority(spec)
            if (
                spec.deadline_monotonic_s is not None
                and time.monotonic() > spec.deadline_monotonic_s
            ):
                raise ActorStateError(
                    f"Invocation {spec.invocation_id!r} passed its monotonic deadline."
                )

            if self._profile.lifecycle is ActorLifecycle.EPHEMERAL:
                self._ephemeral_consumed = True

            record = _InvocationRecord(fingerprint=fingerprint)
            try:
                record.value = self._provider.invoke(self._runtime, spec)
            except Exception as exc:  # provider failure is an idempotent outcome
                record.error = exc

            if self._profile.lifecycle is ActorLifecycle.EPHEMERAL:
                try:
                    self._retire_locked()
                except Exception as cleanup_error:
                    if record.error is not None:
                        note = f"Actor retirement also failed: {cleanup_error!r}"
                        if hasattr(record.error, "add_note"):
                            record.error.add_note(note)
                    else:
                        record.value = _MISSING
                        record.error = ActorError(
                            f"Invocation completed but ephemeral actor retirement failed: "
                            f"{cleanup_error!r}"
                        )

            self._records[key] = record
            if record.error is not None:
                raise record.error
            return record.value

    def suspend(self) -> bool:
        """Suspend an active actor; return False when already suspended."""

        with self._lock:
            if self._state is ActorState.SUSPENDED:
                return False
            if self._state is not ActorState.ACTIVE:
                raise ActorStateError(
                    f"Actor {self.actor_id!r} cannot suspend while {self._state.value}."
                )
            self._provider.suspend(self._runtime)
            self._state = ActorState.SUSPENDED
            return True

    def resume(self) -> bool:
        """Resume a suspended actor; return False when already active."""

        with self._lock:
            if self._state is ActorState.ACTIVE:
                return False
            if self._state is not ActorState.SUSPENDED:
                raise ActorStateError(
                    f"Actor {self.actor_id!r} cannot resume while {self._state.value}."
                )
            self._provider.resume(self._runtime)
            self._state = ActorState.ACTIVE
            return True

    def retire(self) -> bool:
        """Permanently retire an actor; repeated calls are no-ops."""

        with self._lock:
            if self._state is ActorState.RETIRED:
                return False
            if self._state is ActorState.NEW:
                raise ActorStateError(f"Actor {self.actor_id!r} was never spawned.")
            self._retire_locked()
            return True

    def _retire_locked(self) -> None:
        if self._state is ActorState.RETIRED:
            return
        self._provider.retire(self._runtime)
        self._state = ActorState.RETIRED

    def _check_authority(self, spec: InvocationSpec) -> None:
        capability_excess = spec.requested_capabilities - self._profile.capability_ceiling
        effect_excess = spec.requested_effects - self._profile.effect_ceiling
        if capability_excess or effect_excess:
            details: list[str] = []
            if capability_excess:
                details.append(f"capabilities={sorted(capability_excess)!r}")
            if effect_excess:
                details.append(f"effects={sorted(effect_excess)!r}")
            raise ActorAuthorityError(
                f"Invocation {spec.invocation_id!r} exceeds ActorProfile "
                f"{self._profile.profile_id!r}: {', '.join(details)}."
            )


class ActorRegistry:
    """Episode-scoped owner of providers, actor identities, and isolation IDs."""

    def __init__(
        self,
        providers: Mapping[str, AgentProvider] | None = None,
        *,
        namespace_root: str = "episode",
        workspace_root: str | Path = "actor_workspaces",
    ) -> None:
        namespace_root = str(namespace_root).strip()
        if not namespace_root:
            raise ValueError("namespace_root must not be empty.")
        self._namespace_root = namespace_root
        self._workspace_root = Path(workspace_root)
        self._providers: dict[str, AgentProvider] = {}
        self._handles: dict[str, AgentHandle] = {}
        self._counter = 0
        self._lock = threading.RLock()
        for provider_id, provider in (providers or {}).items():
            self.register_provider(provider_id, provider)

    def register_provider(
        self, provider_id: str, provider: AgentProvider, *, replace: bool = False
    ) -> None:
        provider_id = _identifier(provider_id, field_name="provider_id")
        if not isinstance(provider, AgentProvider):
            raise TypeError("provider does not implement AgentProvider.")
        with self._lock:
            current = self._providers.get(provider_id)
            if current is provider:
                return
            if current is not None and not replace:
                raise ActorConflictError(f"Provider {provider_id!r} is already registered.")
            self._providers[provider_id] = provider

    def spawn(self, profile: ActorProfile, *, actor_id: str | None = None) -> AgentHandle:
        """Spawn, or idempotently resolve, an actor identity.

        Reusing ``actor_id`` with the same profile returns the original handle,
        including when it has retired.  Reusing it with a different profile is
        rejected rather than silently resurrecting or mutating an actor.
        """

        with self._lock:
            if actor_id is None:
                self._counter += 1
                actor_id = f"{profile.profile_id}-{self._counter:04d}"
            actor_id = _identifier(actor_id, field_name="actor_id")
            existing = self._handles.get(actor_id)
            if existing is not None:
                if existing.profile != profile:
                    raise ActorConflictError(
                        f"Actor ID {actor_id!r} is already bound to a different profile."
                    )
                return existing
            provider = self._providers.get(profile.provider_id)
            if provider is None:
                raise ActorNotFoundError(
                    f"No AgentProvider is registered as {profile.provider_id!r}."
                )
            isolation = self._resolve_isolation(actor_id, profile)
            handle = AgentHandle.spawn(
                actor_id=actor_id,
                profile=profile,
                provider=provider,
                isolation=isolation,
            )
            self._handles[actor_id] = handle
            return handle

    def get(self, actor_id: str) -> AgentHandle:
        with self._lock:
            try:
                return self._handles[actor_id]
            except KeyError as exc:
                raise ActorNotFoundError(f"Unknown actor {actor_id!r}.") from exc

    def handles(
        self,
        *,
        lifecycle: ActorLifecycle | None = None,
        state: ActorState | None = None,
    ) -> tuple[AgentHandle, ...]:
        with self._lock:
            values = tuple(self._handles[key] for key in sorted(self._handles))
        if lifecycle is not None:
            lifecycle = ActorLifecycle(lifecycle)
            values = tuple(item for item in values if item.profile.lifecycle is lifecycle)
        if state is not None:
            state = ActorState(state)
            values = tuple(item for item in values if item.state is state)
        return values

    def retire_all(self) -> None:
        """Retire every live actor owned by this episode registry."""

        for handle in self.handles():
            if handle.state is not ActorState.RETIRED:
                handle.retire()

    def _resolve_isolation(
        self, actor_id: str, profile: ActorProfile
    ) -> ActorIsolation:
        policy = profile.isolation
        namespace_id = (
            f"{self._namespace_root}.{policy.namespace_prefix}.{actor_id}"
        )
        if policy.workspace_mode is WorkspaceMode.ISOLATED:
            workspace_id = str(self._workspace_root / actor_id)
        else:
            # IsolationPolicy validation guarantees a key in this branch.
            workspace_id = str(self._workspace_root / "_shared_ro" / str(policy.workspace_key))
        metadata = dict(policy.metadata)
        metadata.update(
            {
                "profile_id": profile.profile_id,
                "runner_kind": profile.runner_kind,
                "lifecycle": profile.lifecycle.value,
            }
        )
        return ActorIsolation(
            owner_actor_id=actor_id,
            namespace_id=namespace_id,
            workspace_id=workspace_id,
            workspace_mode=policy.workspace_mode,
            world_id=policy.world_id,
            metadata=metadata,
        )


@dataclass
class InMemoryAgentRuntime:
    """Provider state used by deterministic unit tests and replay fixtures."""

    profile: ActorProfile
    isolation: ActorIsolation
    invocations: list[InvocationSpec] = field(default_factory=list)
    suspended: bool = False
    retired: bool = False


class InMemoryAgentProvider:
    """Deterministic provider with no LLM, environment, or filesystem effects."""

    def __init__(
        self,
        handler: Callable[[ActorProfile, InvocationSpec, ActorIsolation], Any] | None = None,
    ) -> None:
        self._handler = handler or self._default_handler
        self._runtimes: dict[str, InMemoryAgentRuntime] = {}
        self._calls: list[tuple[str, str, str | None]] = []
        self._lock = threading.RLock()

    @property
    def calls(self) -> tuple[tuple[str, str, str | None], ...]:
        with self._lock:
            return tuple(self._calls)

    def runtime_for(self, actor_id: str) -> InMemoryAgentRuntime:
        with self._lock:
            return self._runtimes[actor_id]

    def spawn(
        self, profile: ActorProfile, isolation: ActorIsolation
    ) -> InMemoryAgentRuntime:
        with self._lock:
            actor_id = isolation.owner_actor_id
            if actor_id in self._runtimes:
                raise ActorConflictError(f"Provider already spawned actor {actor_id!r}.")
            runtime = InMemoryAgentRuntime(profile=profile, isolation=isolation)
            self._runtimes[actor_id] = runtime
            self._calls.append(("spawn", actor_id, None))
            return runtime

    def invoke(self, runtime: InMemoryAgentRuntime, spec: InvocationSpec) -> Any:
        with self._lock:
            self._require_live(runtime)
            if runtime.suspended:
                raise ActorStateError("In-memory provider runtime is suspended.")
            runtime.invocations.append(spec)
            actor_id = runtime.isolation.owner_actor_id
            self._calls.append(("invoke", actor_id, spec.invocation_id))
            return self._handler(runtime.profile, spec, runtime.isolation)

    def suspend(self, runtime: InMemoryAgentRuntime) -> None:
        with self._lock:
            self._require_live(runtime)
            runtime.suspended = True
            self._calls.append(("suspend", runtime.isolation.owner_actor_id, None))

    def resume(self, runtime: InMemoryAgentRuntime) -> None:
        with self._lock:
            self._require_live(runtime)
            runtime.suspended = False
            self._calls.append(("resume", runtime.isolation.owner_actor_id, None))

    def retire(self, runtime: InMemoryAgentRuntime) -> None:
        with self._lock:
            if runtime.retired:
                return
            runtime.retired = True
            runtime.suspended = False
            self._calls.append(("retire", runtime.isolation.owner_actor_id, None))

    @staticmethod
    def _require_live(runtime: InMemoryAgentRuntime) -> None:
        if runtime.retired:
            raise ActorStateError("In-memory provider runtime is retired.")

    @staticmethod
    def _default_handler(
        profile: ActorProfile,
        spec: InvocationSpec,
        isolation: ActorIsolation,
    ) -> Mapping[str, Any]:
        return {
            "actor_id": isolation.owner_actor_id,
            "profile_id": profile.profile_id,
            "invocation_id": spec.invocation_id,
            "objective": spec.objective,
        }


__all__ = [
    "ActorAuthorityError",
    "ActorConflictError",
    "ActorError",
    "ActorIsolation",
    "ActorLifecycle",
    "ActorNotFoundError",
    "ActorProfile",
    "ActorRegistry",
    "ActorState",
    "ActorStateError",
    "AgentHandle",
    "AgentProvider",
    "InMemoryAgentProvider",
    "InMemoryAgentRuntime",
    "InvocationSpec",
    "IsolationPolicy",
    "WorkspaceMode",
]
