"""Episode-scoped observation streams and long-lived tracking handles.

The control plane consumes this module rather than a tracker-specific API.
Backends may be a deterministic in-memory fixture, checkpoint
re-segmentation, or a future live tracker, but all of them publish the same
fail-closed :class:`ObservationSample` contract.

The contract deliberately has no action method or effect capability.  A
tracker can report evidence; it cannot move the robot, mutate the world, or
publish simulator ground truth as method input.

Production action-time monitoring uses :class:`ContinuousTrackSampler` to
keep already-created handles fresh.  Primary synchronized checkpoint capture
remains a separate consumer of immutable stream history; it does not race the
sampler's backend poll cursor.
"""

from __future__ import annotations

import math
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from robomex.data.artifact_resolver import (
    ArtifactResolver,
    ResolvedArtifact,
    ResolvedArtifactRef,
)
from robomex.data.physical_schema import RevisionVector

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SignalName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$"),
]

_ORACLE_COMPONENT_RE = re.compile(
    r"(?:^|[_.-])(?:oracle|ground[_-]?truth|simulator[_-]?truth|privileged)(?:$|[_.-])",
    re.IGNORECASE,
)
_ACTION_CAPABILITY_RE = re.compile(
    r"(?:^|[_.-])(?:action|actuate|control|motion|gripper|world[_-]?write|env[_-]?step)"
    r"(?:$|[_.-])",
    re.IGNORECASE,
)
_MIN_CONTINUOUS_POLL_INTERVAL_S = 0.001


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility


class ObservationError(RuntimeError):
    """Base error for the stable observation/tracking boundary."""


class ObservationContractError(ObservationError, ValueError):
    """A backend or caller violated the declared observation contract."""


class ObservationHistoryGapError(ObservationError):
    """A bounded stream no longer retains the requested cursor."""


class TrackStateError(ObservationError):
    """A TrackHandle operation is illegal in its current lifecycle state."""


class TrackConflictError(ObservationError):
    """An idempotent track/backend identifier was reused with new content."""


class TrackNotFoundError(ObservationError, KeyError):
    """A requested track or backend is not registered in this episode."""


class BackendAuthorityError(ObservationContractError, PermissionError):
    """An observation backend advertises world-changing or oracle authority."""


class ContinuousTrackSamplerError(ObservationError):
    """Base error for the production continuous sampling service."""


class ContinuousTrackSamplerLifecycleError(ContinuousTrackSamplerError):
    """A sampler lifecycle transition could not be completed safely."""


class ContinuousTrackSamplerStopError(ContinuousTrackSamplerError):
    """The worker did not stop within its explicit join bound."""


class ObservationQuality(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Closed visibility/identity quality reported by a tracker."""

    TRACKED = "tracked"
    AMBIGUOUS = "ambiguous"
    LOST = "lost"


class TrackState(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    NEW = "new"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    STOPPED = "stopped"


class ContinuousSamplerState(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Lifecycle of the production-only background sampling owner."""

    NEW = "new"
    RUNNING = "running"
    SUSPENDED = "suspended"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    STOP_FAILED = "stop_failed"


class TrackSamplingOutcome(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    TRACKED = "tracked"
    LOST = "lost"
    SILENT = "silent"
    ERROR = "error"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        str_strip_whitespace=True,
        validate_default=True,
    )


class _SamplingUtcModel(_StrictModel):
    @field_validator(
        "attempted_at",
        "finished_at",
        "occurred_at",
        "last_attempt_at",
        "last_success_at",
        "last_tracked_at",
        "started_at",
        "stopped_at",
        check_fields=False,
    )
    @classmethod
    def _sampling_timestamp_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("sampling health timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10


class TrackSamplingError(_SamplingUtcModel):
    """Typed, retained failure from one track attempt or lifecycle call."""

    schema_version: Literal["robomex.track_sampling_error.v1"] = (
        "robomex.track_sampling_error.v1"
    )
    track_id: NonEmptyStr
    operation: NonEmptyStr
    error_type: NonEmptyStr
    message: str
    occurred_at: datetime = Field(default_factory=_utc_now)


class TrackSamplingEvent(_SamplingUtcModel):
    """Bounded audit event for every per-track background attempt."""

    schema_version: Literal["robomex.track_sampling_event.v1"] = (
        "robomex.track_sampling_event.v1"
    )
    sampler_id: NonEmptyStr
    round_index: int = Field(ge=1)
    track_id: NonEmptyStr
    outcome: TrackSamplingOutcome
    attempted_at: datetime
    finished_at: datetime
    sample_id: NonEmptyStr | None = None
    sample_sequence: int | None = Field(default=None, ge=1)
    quality: ObservationQuality | None = None
    error: TrackSamplingError | None = None

    @model_validator(mode="after")
    def _closed_outcome(self) -> TrackSamplingEvent:
        if self.finished_at < self.attempted_at:
            raise ValueError("sampling event finished before it started")
        if self.outcome is TrackSamplingOutcome.ERROR:
            if (
                self.error is None
                or self.sample_id is not None
                or self.sample_sequence is not None
                or self.quality is not None
            ):
                raise ValueError("error sampling events require only typed error evidence")
        elif self.outcome is TrackSamplingOutcome.SILENT:
            if (
                self.error is not None
                or self.sample_id is not None
                or self.sample_sequence is not None
                or self.quality is not None
            ):
                raise ValueError("silent sampling events carry no stale sample")
        else:
            if (
                self.error is not None
                or self.sample_id is None
                or self.sample_sequence is None
                or self.quality is None
            ):
                raise ValueError("sample events require sample identity and quality")
            expected = (
                TrackSamplingOutcome.TRACKED
                if self.quality is ObservationQuality.TRACKED
                else TrackSamplingOutcome.LOST
            )
            if self.outcome is not expected:
                raise ValueError("sampling outcome does not match observation quality")
        return self


class TrackSamplingHealth(_SamplingUtcModel):
    """Immutable health snapshot for one continuously sampled track."""

    schema_version: Literal["robomex.track_sampling_health.v1"] = (
        "robomex.track_sampling_health.v1"
    )
    track_id: NonEmptyStr
    track_state: TrackState
    attempt_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    tracked_count: int = Field(ge=0)
    lost_count: int = Field(ge=0)
    silent_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    last_attempt_at: datetime | None = None
    last_success_at: datetime | None = None
    last_tracked_at: datetime | None = None
    last_sample_id: NonEmptyStr | None = None
    last_quality: ObservationQuality | None = None
    last_error: TrackSamplingError | None = None


class ContinuousTrackSamplerHealth(_SamplingUtcModel):
    """Immutable aggregate lifecycle/health snapshot."""

    schema_version: Literal["robomex.continuous_track_sampler_health.v1"] = (
        "robomex.continuous_track_sampler_health.v1"
    )
    sampler_id: NonEmptyStr
    state: ContinuousSamplerState
    poll_interval_s: float = Field(gt=0, allow_inf_nan=False)
    max_silence_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    round_count: int = Field(ge=0)
    thread_alive: bool
    in_flight_track_id: NonEmptyStr | None = None
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    tracks: tuple[TrackSamplingHealth, ...]


@dataclass
class _MutableTrackSamplingHealth:
    track_state: TrackState = TrackState.NEW
    attempt_count: int = 0
    sample_count: int = 0
    tracked_count: int = 0
    lost_count: int = 0
    silent_count: int = 0
    error_count: int = 0
    last_attempt_at: datetime | None = None
    last_success_at: datetime | None = None
    last_tracked_at: datetime | None = None
    last_sample_id: str | None = None
    last_quality: ObservationQuality | None = None
    last_error: TrackSamplingError | None = None
    silence_started_monotonic_s: float | None = None
    silence_lost_published: bool = False


class ObservationRevisionVector(_StrictModel):
    """Revisions that make an observation's physical context auditable."""

    scene_revision: int = Field(ge=0)
    arm_revision: int = Field(ge=0)
    gripper_revision: int = Field(ge=0)
    attachment_revision: int = Field(ge=0)
    camera_revision: int = Field(ge=0)

    def dominates(self, older: ObservationRevisionVector) -> bool:
        """Return whether no dependency revision regressed."""

        return all(
            current >= previous
            for current, previous in zip(
                self.as_tuple(), older.as_tuple(), strict=True
            )
        )

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            self.scene_revision,
            self.arm_revision,
            self.gripper_revision,
            self.attachment_revision,
            self.camera_revision,
        )

    def to_physical_revision(self, *, camera_id: str) -> RevisionVector:
        """Convert the stream-local clock to the canonical physical clock.

        A tracking sample carries one camera revision, while the physical
        validity schema supports an episode-wide map of camera domains.  The
        camera identity is therefore mandatory at this boundary; silently
        guessing it would make a stale sample look current.
        """

        normalized_camera_id = str(camera_id).strip()
        if not normalized_camera_id:
            raise ObservationContractError("camera_id is required for revision conversion")
        return RevisionVector(
            scene=self.scene_revision,
            arm=self.arm_revision,
            gripper=self.gripper_revision,
            attachment=self.attachment_revision,
            camera={normalized_camera_id: self.camera_revision},
        )


class EntityIdentity(_StrictModel):
    """Episode-stable identity separated from a backend's transient state."""

    entity_id: NonEmptyStr
    semantic_label: NonEmptyStr


class TrackRequest(_StrictModel):
    """Immutable contract for one episode-scoped tracking service."""

    schema_version: Literal["robomex.track_request.v1"] = "robomex.track_request.v1"
    episode_id: NonEmptyStr
    track_id: NonEmptyStr
    entity: EntityIdentity
    backend_id: NonEmptyStr
    declared_signals: tuple[SignalName, ...] = Field(min_length=1)
    camera_ids: tuple[NonEmptyStr, ...] = ()
    created_by_workflow_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> TrackRequest:
        if len(set(self.declared_signals)) != len(self.declared_signals):
            raise ValueError("declared_signals must be unique")
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("camera_ids must be unique")
        _reject_oracle_names(self.declared_signals, field_name="declared_signals")
        return self


class BackendObservation(_StrictModel):
    """Backend result before the stream assigns its authoritative sequence."""

    entity_id: NonEmptyStr
    quality: ObservationQuality
    revisions: ObservationRevisionVector
    signals: dict[SignalName, JsonValue] = Field(default_factory=dict)
    artifact_refs: tuple[ResolvedArtifactRef, ...] = ()
    reason: str | None = None
    observed_at: datetime = Field(default_factory=_utc_now)
    monotonic_time_s: float = Field(default_factory=time.monotonic, ge=0, allow_inf_nan=False)

    @field_validator("observed_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility

    @field_validator("signals")
    @classmethod
    def _safe_signals(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _reject_oracle_payload(value, path="signals")
        _require_finite_json(value, path="signals")
        return value

    @model_validator(mode="after")
    def _quality_is_fail_closed(self) -> BackendObservation:
        if self.quality is not ObservationQuality.TRACKED:
            if self.signals:
                raise ValueError(
                    "ambiguous/lost observations must clear signals instead of reusing coordinates"
                )
            if not self.reason or not self.reason.strip():
                raise ValueError("ambiguous/lost observations require an explicit reason")
        return self


class ObservationSample(_StrictModel):
    """One immutable sample in an episode-global, strictly ordered stream."""

    schema_version: Literal["robomex.observation_sample.v1"] = (
        "robomex.observation_sample.v1"
    )
    sample_id: NonEmptyStr = Field(default_factory=lambda: _new_id("obs"))
    episode_id: NonEmptyStr
    stream_id: NonEmptyStr
    track_id: NonEmptyStr
    entity: EntityIdentity
    sequence: int = Field(ge=1)
    observed_at: datetime
    monotonic_time_s: float = Field(ge=0, allow_inf_nan=False)
    quality: ObservationQuality
    revisions: ObservationRevisionVector
    signals: dict[SignalName, JsonValue] = Field(default_factory=dict)
    artifact_refs: tuple[ResolvedArtifactRef, ...] = ()
    reason: str | None = None

    @field_validator("observed_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility

    @field_validator("signals")
    @classmethod
    def _safe_signals(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _reject_oracle_payload(value, path="signals")
        _require_finite_json(value, path="signals")
        return value

    @model_validator(mode="after")
    def _quality_is_fail_closed(self) -> ObservationSample:
        if self.quality is not ObservationQuality.TRACKED:
            if self.signals:
                raise ValueError(
                    "ambiguous/lost observations must clear signals instead of reusing coordinates"
                )
            if not self.reason or not self.reason.strip():
                raise ValueError("ambiguous/lost observations require an explicit reason")
        return self


class ObservationSubscription:
    """Cursor-based, non-destructive subscription over one stream."""

    def __init__(
        self,
        stream: ObservationStream,
        *,
        after_sequence: int,
        track_id: str | None,
        entity_id: str | None,
    ) -> None:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        self._stream = stream
        self._cursor = after_sequence
        self._track_id = track_id
        self._entity_id = entity_id
        self._lock = threading.RLock()

    @property
    def cursor(self) -> int:
        with self._lock:
            return self._cursor

    def poll(self, *, limit: int | None = None) -> tuple[ObservationSample, ...]:
        """Read matching samples and advance across all inspected events.

        A filtered subscription advances to the stream's current tail even if
        no event matched.  It therefore represents live delivery, not a query
        repeatedly rescanning irrelevant tracks.
        """

        with self._lock:
            events = self._stream.after(self._cursor)
            selected = tuple(
                sample
                for sample in events
                if (self._track_id is None or sample.track_id == self._track_id)
                and (self._entity_id is None or sample.entity.entity_id == self._entity_id)
            )
            if limit is not None:
                if limit < 1:
                    raise ValueError("limit must be positive")
                selected = selected[:limit]
                if selected:
                    self._cursor = selected[-1].sequence
                    return selected
            if events:
                self._cursor = events[-1].sequence
            return selected


class ObservationStream:
    """Thread-safe episode stream with explicit retention semantics.

    ``retention_limit=None`` is append-only for the life of this object.
    Supplying a positive limit bounds in-memory retention; consumers asking
    before ``dropped_through_sequence`` receive :class:`ObservationHistoryGapError`
    rather than a silently incomplete replay.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        stream_id: str = "episode-observations",
        retention_limit: int | None = None,
        resolver: ArtifactResolver | None = None,
    ) -> None:
        if not episode_id.strip() or not stream_id.strip():
            raise ValueError("episode_id and stream_id must not be empty")
        if retention_limit is not None and retention_limit < 1:
            raise ValueError("retention_limit must be positive or None")
        if resolver is not None and resolver.episode_id != episode_id:
            raise ObservationContractError("artifact resolver belongs to another episode")
        self.episode_id = episode_id.strip()
        self.stream_id = stream_id.strip()
        self.retention_limit = retention_limit
        self.resolver = resolver
        self._history: list[ObservationSample] = []
        self._last_sequence = 0
        self._dropped_through_sequence = 0
        self._latest_by_track: dict[str, ObservationSample] = {}
        self._lock = threading.RLock()

    @property
    def retention_policy(self) -> str:
        return "append_only" if self.retention_limit is None else "bounded_memory"

    @property
    def latest_sequence(self) -> int:
        with self._lock:
            return self._last_sequence

    @property
    def dropped_through_sequence(self) -> int:
        with self._lock:
            return self._dropped_through_sequence

    @property
    def history(self) -> tuple[ObservationSample, ...]:
        with self._lock:
            return tuple(self._history)

    def publish(self, request: TrackRequest, observation: BackendObservation) -> ObservationSample:
        """Validate a backend result and append exactly one immutable sample."""

        request = TrackRequest.model_validate(request)
        observation = BackendObservation.model_validate(observation)
        with self._lock:
            self._check_request(request)
            self._check_observation(request, observation)
            previous = self._latest_by_track.get(request.track_id)
            if previous is not None:
                if observation.monotonic_time_s <= previous.monotonic_time_s:
                    raise ObservationContractError(
                        "monotonic_time_s must strictly increase for each track"
                    )
                if observation.observed_at < previous.observed_at:
                    raise ObservationContractError("observed_at regressed for a track")
                if not observation.revisions.dominates(previous.revisions):
                    raise ObservationContractError("observation revision vector regressed")

            sequence = self._last_sequence + 1
            sample = ObservationSample(
                episode_id=self.episode_id,
                stream_id=self.stream_id,
                track_id=request.track_id,
                entity=request.entity,
                sequence=sequence,
                observed_at=observation.observed_at,
                monotonic_time_s=observation.monotonic_time_s,
                quality=observation.quality,
                revisions=observation.revisions,
                signals=observation.signals,
                artifact_refs=observation.artifact_refs,
                reason=observation.reason,
            )
            self._append_validated(sample)
            return sample

    def replay(self, samples: Iterable[ObservationSample | Mapping[str, Any]]) -> None:
        """Restore trusted serialized history while rechecking all invariants."""

        for value in samples:
            sample = ObservationSample.model_validate(value)
            with self._lock:
                if sample.episode_id != self.episode_id or sample.stream_id != self.stream_id:
                    raise ObservationContractError("replayed sample belongs to another stream")
                if sample.sequence != self._last_sequence + 1:
                    raise ObservationContractError("replayed sequences must be contiguous")
                previous = self._latest_by_track.get(sample.track_id)
                if previous is not None:
                    if sample.monotonic_time_s <= previous.monotonic_time_s:
                        raise ObservationContractError(
                            "replayed monotonic_time_s must strictly increase"
                        )
                    if sample.observed_at < previous.observed_at:
                        raise ObservationContractError("replayed observed_at regressed")
                    if not sample.revisions.dominates(previous.revisions):
                        raise ObservationContractError("replayed revision vector regressed")
                self._append_validated(sample)

    def after(
        self, sequence: int, *, limit: int | None = None
    ) -> tuple[ObservationSample, ...]:
        """Replay all retained samples strictly after a global sequence."""

        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            if sequence < self._dropped_through_sequence:
                raise ObservationHistoryGapError(
                    f"Sequence {sequence} precedes retained history; "
                    f"dropped through {self._dropped_through_sequence}."
                )
            values = tuple(sample for sample in self._history if sample.sequence > sequence)
            return values if limit is None else values[:limit]

    def subscribe(
        self,
        *,
        after_sequence: int = 0,
        track_id: str | None = None,
        entity_id: str | None = None,
    ) -> ObservationSubscription:
        return ObservationSubscription(
            self,
            after_sequence=after_sequence,
            track_id=track_id,
            entity_id=entity_id,
        )

    def latest(self, track_id: str) -> ObservationSample | None:
        with self._lock:
            return self._latest_by_track.get(track_id)

    def _check_request(self, request: TrackRequest) -> None:
        if request.episode_id != self.episode_id:
            raise ObservationContractError("track request belongs to another episode")

    def _check_observation(
        self, request: TrackRequest, observation: BackendObservation
    ) -> None:
        if observation.entity_id != request.entity.entity_id:
            raise ObservationContractError("backend changed the tracked entity identity")
        actual = set(observation.signals)
        declared = set(request.declared_signals)
        if observation.quality is ObservationQuality.TRACKED and actual != declared:
            missing = declared - actual
            unknown = actual - declared
            details: list[str] = []
            if missing:
                details.append(f"missing={sorted(missing)!r}")
            if unknown:
                details.append(f"undeclared={sorted(unknown)!r}")
            raise ObservationContractError(
                "tracked observations must provide exactly declared signals: "
                + ", ".join(details)
            )
        if observation.quality is not ObservationQuality.TRACKED and actual:
            raise ObservationContractError("non-tracked observations must clear signals")
        for ref in observation.artifact_refs:
            if ref.artifact_id.startswith("art:"):
                parts = ref.artifact_id.split(":")
                if len(parts) < 2 or parts[1] != self.episode_id:
                    raise ObservationContractError(
                        "observation artifact ref belongs to another episode"
                    )
            if self.resolver is not None:
                resolved = self.resolver.resolve(ref)
                _reject_oracle_payload(
                    resolved.payload,
                    path=f"artifact[{ref.artifact_id}]",
                )

    def _append_validated(self, sample: ObservationSample) -> None:
        self._history.append(sample)
        self._last_sequence = sample.sequence
        self._latest_by_track[sample.track_id] = sample
        if self.retention_limit is not None and len(self._history) > self.retention_limit:
            dropped = self._history.pop(0)
            self._dropped_through_sequence = dropped.sequence


@runtime_checkable
class ObservationBackend(Protocol):
    """Read-only provider boundary implemented by every tracker backend."""

    backend_id: str
    read_capabilities: frozenset[str]
    effect_capabilities: frozenset[str]

    def start(self, request: TrackRequest) -> Any:
        """Allocate backend-owned state for a track."""

    def poll(self, runtime: Any, request: TrackRequest) -> BackendObservation | None:
        """Return a fresh observation or None when no fresh evidence exists."""

    def suspend(self, runtime: Any) -> None:
        """Pause work without losing track identity."""

    def resume(self, runtime: Any) -> None:
        """Resume a suspended backend runtime."""

    def stop(self, runtime: Any) -> None:
        """Release backend-owned state permanently."""


class _BackendSession:
    def __init__(self, *, cursor: int = 0) -> None:
        self.cursor = cursor


class InMemoryObservationBackend:
    """Deterministic read-only backend for replay, tests, and adapters.

    Each track owns a cursor over entity-keyed immutable backend observations.
    Queued values are never removed, so multiple handles may replay the same
    evidence independently.
    """

    read_capabilities = frozenset({"observation.read"})
    effect_capabilities: frozenset[str] = frozenset()

    def __init__(self, backend_id: str = "memory") -> None:
        if not backend_id.strip():
            raise ValueError("backend_id must not be empty")
        self.backend_id = backend_id
        self._observations: dict[str, list[BackendObservation]] = {}
        self._lock = threading.RLock()

    def push(self, observation: BackendObservation | Mapping[str, Any]) -> None:
        value = BackendObservation.model_validate(observation)
        with self._lock:
            self._observations.setdefault(value.entity_id, []).append(value)

    def start(self, request: TrackRequest) -> _BackendSession:
        return _BackendSession()

    def poll(
        self, runtime: _BackendSession, request: TrackRequest
    ) -> BackendObservation | None:
        with self._lock:
            values = self._observations.get(request.entity.entity_id, ())
            if runtime.cursor >= len(values):
                return None
            value = values[runtime.cursor]
            runtime.cursor += 1
            return value

    def suspend(self, runtime: _BackendSession) -> None:
        return None

    def resume(self, runtime: _BackendSession) -> None:
        return None

    def stop(self, runtime: _BackendSession) -> None:
        return None


class CheckpointRecord(_StrictModel):
    """Immutable camera checkpoint admitted to re-segmentation."""

    schema_version: Literal["robomex.observation_checkpoint.v1"] = (
        "robomex.observation_checkpoint.v1"
    )
    checkpoint_id: NonEmptyStr = Field(default_factory=lambda: _new_id("checkpoint"))
    episode_id: NonEmptyStr
    camera_id: NonEmptyStr
    frame_ref: ResolvedArtifactRef
    revisions: ObservationRevisionVector
    captured_at: datetime = Field(default_factory=_utc_now)
    monotonic_time_s: float = Field(default_factory=time.monotonic, ge=0, allow_inf_nan=False)

    @field_validator("captured_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility


class CheckpointSegmentation(_StrictModel):
    """Entity-specific inference result returned by a checkpoint segmenter."""

    quality: ObservationQuality
    signals: dict[SignalName, JsonValue] = Field(default_factory=dict)
    evidence_refs: tuple[ResolvedArtifactRef, ...] = ()
    reason: str | None = None

    @field_validator("signals")
    @classmethod
    def _safe_signals(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _reject_oracle_payload(value, path="signals")
        _require_finite_json(value, path="signals")
        return value

    @model_validator(mode="after")
    def _quality_is_fail_closed(self) -> CheckpointSegmentation:
        if self.quality is not ObservationQuality.TRACKED:
            if self.signals:
                raise ValueError("ambiguous/lost segmentation must clear signals")
            if not self.reason or not self.reason.strip():
                raise ValueError("ambiguous/lost segmentation requires a reason")
        return self


CheckpointSegmenter = Callable[
    [TrackRequest, CheckpointRecord, ResolvedArtifact],
    CheckpointSegmentation | Mapping[str, Any],
]


class CheckpointObservationBackend:
    """Checkpoint re-segmentation backend using content-addressed frames.

    Checkpoints are verified through the episode resolver before admission.
    The injected segmenter receives immutable content, not an environment or
    simulator handle.  This makes the backend useful now while preserving the
    same contract for a future live tracker.
    """

    read_capabilities = frozenset(
        {"observation.read", "checkpoint.read", "segmentation.infer"}
    )
    effect_capabilities: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        backend_id: str,
        episode_id: str,
        resolver: ArtifactResolver,
        segmenter: CheckpointSegmenter,
    ) -> None:
        if not backend_id.strip() or not episode_id.strip():
            raise ValueError("backend_id and episode_id must not be empty")
        if resolver.episode_id != episode_id:
            raise ObservationContractError("resolver belongs to another episode")
        self.backend_id = backend_id
        self.episode_id = episode_id
        self.resolver = resolver
        self.segmenter = segmenter
        self._checkpoints: list[tuple[CheckpointRecord, ResolvedArtifact]] = []
        self._checkpoint_by_id: dict[str, CheckpointRecord] = {}
        self._lock = threading.RLock()

    def add_checkpoint(self, checkpoint: CheckpointRecord | Mapping[str, Any]) -> bool:
        value = CheckpointRecord.model_validate(checkpoint)
        if value.episode_id != self.episode_id:
            raise ObservationContractError("checkpoint belongs to another episode")
        resolved = self.resolver.resolve(value.frame_ref)
        _reject_oracle_payload(resolved.payload, path="checkpoint.payload")
        with self._lock:
            previous_with_id = self._checkpoint_by_id.get(value.checkpoint_id)
            if previous_with_id is not None:
                if previous_with_id != value:
                    raise TrackConflictError(
                        f"checkpoint id {value.checkpoint_id!r} was rebound"
                    )
                return False
            if self._checkpoints:
                previous = self._checkpoints[-1][0]
                if value.monotonic_time_s <= previous.monotonic_time_s:
                    raise ObservationContractError(
                        "checkpoint monotonic_time_s must strictly increase"
                    )
                if value.captured_at < previous.captured_at:
                    raise ObservationContractError("checkpoint captured_at regressed")
            self._checkpoints.append((value, resolved))
            self._checkpoint_by_id[value.checkpoint_id] = value
            return True

    def start(self, request: TrackRequest) -> _BackendSession:
        if request.episode_id != self.episode_id:
            raise ObservationContractError("track request belongs to another episode")
        return _BackendSession()

    def poll(
        self, runtime: _BackendSession, request: TrackRequest
    ) -> BackendObservation | None:
        with self._lock:
            while runtime.cursor < len(self._checkpoints):
                checkpoint, resolved = self._checkpoints[runtime.cursor]
                runtime.cursor += 1
                if request.camera_ids and checkpoint.camera_id not in request.camera_ids:
                    continue
                try:
                    raw = self.segmenter(request, checkpoint, resolved)
                    result = CheckpointSegmentation.model_validate(raw)
                except ValidationError as exc:
                    raise ObservationContractError(
                        "checkpoint segmenter violated the observation contract"
                    ) from exc
                refs = _dedupe_refs((checkpoint.frame_ref, *result.evidence_refs))
                return BackendObservation(
                    entity_id=request.entity.entity_id,
                    quality=result.quality,
                    revisions=checkpoint.revisions,
                    signals=result.signals,
                    artifact_refs=refs,
                    reason=result.reason,
                    observed_at=checkpoint.captured_at,
                    monotonic_time_s=checkpoint.monotonic_time_s,
                )
            return None

    def suspend(self, runtime: _BackendSession) -> None:
        return None

    def resume(self, runtime: _BackendSession) -> None:
        return None

    def stop(self, runtime: _BackendSession) -> None:
        return None


class TrackHandle:
    """Idempotent lifecycle handle for one long-lived tracking service."""

    def __init__(
        self,
        *,
        request: TrackRequest,
        stream: ObservationStream,
        backend: ObservationBackend,
    ) -> None:
        request = TrackRequest.model_validate(request)
        _validate_backend(backend)
        if request.episode_id != stream.episode_id:
            raise ObservationContractError("track and stream belong to different episodes")
        if request.backend_id != backend.backend_id:
            raise ObservationContractError("request backend_id does not match backend")
        self.request = request
        self.stream = stream
        self.backend = backend
        self._state = TrackState.NEW
        self._runtime: Any = None
        self._continuous_sampler_token: object | None = None
        self._lock = threading.RLock()

    @property
    def track_id(self) -> str:
        return self.request.track_id

    @property
    def state(self) -> TrackState:
        with self._lock:
            return self._state

    def start(self) -> bool:
        with self._lock:
            if self._state is TrackState.ACTIVE:
                return False
            if self._state is TrackState.SUSPENDED:
                raise TrackStateError("suspended track must be resumed, not started")
            if self._state is TrackState.STOPPED:
                raise TrackStateError("stopped track cannot be restarted")
            self._runtime = self.backend.start(self.request)
            self._state = TrackState.ACTIVE
            return True

    def poll(self, *, _continuous_sampler_token: object | None = None) -> ObservationSample:
        """Publish one fresh result; backend silence/failure becomes LOST.

        Direct calls are the explicit *manual* mode.  While a
        :class:`ContinuousTrackSampler` owns this handle, only its private
        worker token may poll, preventing two consumers from racing one backend
        runtime or making a primary checkpoint accidentally consume monitor
        evidence.
        """

        with self._lock:
            self._assert_poll_authority_locked(_continuous_sampler_token)
            observation = self._poll_if_available_locked()
            if observation is None:
                observation = self._lost_observation(reason="no_fresh_observation")
            return self.stream.publish(self.request, observation)

    def poll_if_available(
        self,
        *,
        _continuous_sampler_token: object | None = None,
    ) -> ObservationSample | None:
        """Publish one available result, preserving backend silence as ``None``.

        Continuous samplers use this method only when they own a bounded
        silence policy.  Backend exceptions still publish LOST immediately;
        only a clean ``None`` response may be deferred.
        """

        with self._lock:
            self._assert_poll_authority_locked(_continuous_sampler_token)
            observation = self._poll_if_available_locked()
            if observation is None:
                return None
            return self.stream.publish(self.request, observation)

    def publish_lost(
        self,
        *,
        reason: str,
        _continuous_sampler_token: object | None = None,
    ) -> ObservationSample:
        """Publish one explicit fail-closed LOST sample under poll ownership."""

        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError("LOST reason must not be empty")
        with self._lock:
            self._assert_poll_authority_locked(_continuous_sampler_token)
            return self.stream.publish(
                self.request,
                self._lost_observation(reason=normalized_reason),
            )

    def _assert_poll_authority_locked(self, token: object | None) -> None:
        owner = self._continuous_sampler_token
        if owner is not None and token is not owner:
            raise TrackStateError(
                "manual poll is disabled while a continuous sampler owns the track"
            )
        if owner is None and token is not None:
            raise TrackStateError("continuous sampler does not own this track")
        if self._state is not TrackState.ACTIVE:
            raise TrackStateError(f"track cannot poll while {self._state.value}")

    def _poll_if_available_locked(self) -> BackendObservation | None:
        try:
            return self.backend.poll(self._runtime, self.request)
        except ObservationContractError:
            raise
        except Exception as exc:  # backend failure is observable, never stale data
            return self._lost_observation(
                reason=f"backend_error:{type(exc).__name__}"
            )

    def _claim_continuous_sampler(self, token: object) -> None:
        with self._lock:
            if self._continuous_sampler_token is token:
                return
            if self._continuous_sampler_token is not None:
                raise TrackConflictError(
                    f"track {self.track_id!r} already has a continuous sampler"
                )
            self._continuous_sampler_token = token

    def _release_continuous_sampler(self, token: object) -> None:
        with self._lock:
            if self._continuous_sampler_token is token:
                self._continuous_sampler_token = None
            elif self._continuous_sampler_token is not None:
                raise TrackConflictError(
                    f"track {self.track_id!r} is owned by another continuous sampler"
                )

    def suspend(self) -> bool:
        with self._lock:
            if self._state is TrackState.SUSPENDED:
                return False
            if self._state is not TrackState.ACTIVE:
                raise TrackStateError(f"track cannot suspend while {self._state.value}")
            self.backend.suspend(self._runtime)
            self._state = TrackState.SUSPENDED
            return True

    def resume(self) -> bool:
        with self._lock:
            if self._state is TrackState.ACTIVE:
                return False
            if self._state is not TrackState.SUSPENDED:
                raise TrackStateError(f"track cannot resume while {self._state.value}")
            self.backend.resume(self._runtime)
            self._state = TrackState.ACTIVE
            return True

    def stop(self) -> bool:
        with self._lock:
            if self._state is TrackState.STOPPED:
                return False
            if self._state is not TrackState.NEW:
                self.backend.stop(self._runtime)
            self._state = TrackState.STOPPED
            return True

    def after(self, sequence: int) -> tuple[ObservationSample, ...]:
        return tuple(
            sample
            for sample in self.stream.after(sequence)
            if sample.track_id == self.track_id
        )

    def subscribe(self, *, after_sequence: int = 0) -> ObservationSubscription:
        return self.stream.subscribe(
            after_sequence=after_sequence,
            track_id=self.track_id,
        )

    def _lost_observation(self, *, reason: str) -> BackendObservation:
        previous = self.stream.latest(self.track_id)
        revisions = (
            previous.revisions
            if previous is not None
            else ObservationRevisionVector(
                scene_revision=0,
                arm_revision=0,
                gripper_revision=0,
                attachment_revision=0,
                camera_revision=0,
            )
        )
        previous_monotonic = previous.monotonic_time_s if previous is not None else 0.0
        now_monotonic = max(time.monotonic(), math.nextafter(previous_monotonic, math.inf))
        observed_at = _utc_now()
        if previous is not None and observed_at < previous.observed_at:
            observed_at = previous.observed_at
        return BackendObservation(
            entity_id=self.request.entity.entity_id,
            quality=ObservationQuality.LOST,
            revisions=revisions,
            signals={},
            reason=reason,
            observed_at=observed_at,
            monotonic_time_s=now_monotonic,
        )


class ContinuousTrackSampler:
    """Production background owner that keeps monitor observations fresh.

    This service is intentionally separate from primary checkpoint capture.
    It performs no cross-track atomicity and exposes no action, LLM, or state
    mutation capability: one worker invokes only ``TrackHandle.poll`` at a
    fixed finite interval.  Silence/backend failures are already converted by
    ``TrackHandle`` into explicit LOST samples; contract violations are kept as
    typed health errors and never replaced with a stale sample.

    Manual mode means calling ``TrackHandle.poll`` without this service.  Once
    started, the sampler exclusively owns each handle's poll surface until a
    successful bounded stop, so continuous and manual consumption cannot race.
    """

    def __init__(
        self,
        handles: Sequence[TrackHandle],
        *,
        poll_interval_s: float,
        sampler_id: str | None = None,
        stop_join_timeout_s: float = 2.0,
        max_silence_s: float | None = None,
        event_history_limit: int = 1_024,
    ) -> None:
        values = tuple(handles)
        if not values or not all(isinstance(handle, TrackHandle) for handle in values):
            raise ValueError("continuous sampler requires explicit TrackHandle values")
        track_ids = tuple(handle.track_id for handle in values)
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("continuous sampler track ids must be unique")
        stream = values[0].stream
        if any(handle.stream is not stream for handle in values):
            raise ObservationContractError(
                "continuous sampler handles must share one observation stream"
            )
        if isinstance(poll_interval_s, bool) or isinstance(stop_join_timeout_s, bool):
            raise ValueError("sampler intervals must be numeric, not boolean")
        interval = float(poll_interval_s)
        join_timeout = float(stop_join_timeout_s)
        if not math.isfinite(interval) or interval < _MIN_CONTINUOUS_POLL_INTERVAL_S:
            raise ValueError(
                "poll_interval_s must be finite and at least "
                f"{_MIN_CONTINUOUS_POLL_INTERVAL_S:.3f}s"
            )
        if not math.isfinite(join_timeout) or join_timeout <= 0:
            raise ValueError("stop_join_timeout_s must be finite and positive")
        if isinstance(max_silence_s, bool):
            raise ValueError("max_silence_s must be numeric, not boolean")
        normalized_max_silence = None if max_silence_s is None else float(max_silence_s)
        if normalized_max_silence is not None and (
            not math.isfinite(normalized_max_silence)
            or normalized_max_silence <= 0
        ):
            raise ValueError("max_silence_s must be finite and positive when supplied")
        if normalized_max_silence is not None and interval > normalized_max_silence:
            raise ValueError("poll_interval_s must not exceed max_silence_s")
        if isinstance(event_history_limit, bool) or not isinstance(event_history_limit, int):
            raise ValueError("event_history_limit must be an integer")
        if event_history_limit < len(values):
            raise ValueError(
                "event_history_limit must retain at least one full sampling round"
            )
        normalized_id = str(sampler_id or _new_id("sampler")).strip()
        if not normalized_id:
            raise ValueError("sampler_id must not be empty")
        self.sampler_id = normalized_id
        self.handles = values
        self.poll_interval_s = interval
        self.stop_join_timeout_s = join_timeout
        self.max_silence_s = normalized_max_silence
        self._owner_token = object()
        self._claimed_track_ids: set[str] = set()
        self._state = ContinuousSamplerState.NEW
        self._condition = threading.Condition(threading.RLock())
        self._poll_lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_start_count = 0
        self._round_count = 0
        self._in_flight_track_id: str | None = None
        self._started_at: datetime | None = None
        self._stopped_at: datetime | None = None
        self._health = {
            handle.track_id: _MutableTrackSamplingHealth(track_state=handle.state)
            for handle in values
        }
        self._events: deque[TrackSamplingEvent] = deque(maxlen=event_history_limit)

    @property
    def state(self) -> ContinuousSamplerState:
        with self._condition:
            return self._state

    @property
    def worker_ident(self) -> int | None:
        with self._condition:
            return self._thread.ident if self._thread is not None else None

    @property
    def thread_start_count(self) -> int:
        with self._condition:
            return self._thread_start_count

    @property
    def events(self) -> tuple[TrackSamplingEvent, ...]:
        with self._condition:
            return tuple(self._events)

    @property
    def health(self) -> ContinuousTrackSamplerHealth:
        with self._condition:
            thread_alive = self._thread is not None and self._thread.is_alive()
            tracks = tuple(
                self._track_health_snapshot(handle)
                for handle in self.handles
            )
            return ContinuousTrackSamplerHealth(
                sampler_id=self.sampler_id,
                state=self._state,
                poll_interval_s=self.poll_interval_s,
                max_silence_s=self.max_silence_s,
                round_count=self._round_count,
                thread_alive=thread_alive,
                in_flight_track_id=self._in_flight_track_id,
                started_at=self._started_at,
                stopped_at=self._stopped_at,
                tracks=tracks,
            )

    def start(self) -> bool:
        """Claim all handles and start exactly one daemon worker."""

        with self._poll_lock, self._condition:
            if self._state is ContinuousSamplerState.RUNNING:
                return False
            if self._state is ContinuousSamplerState.SUSPENDED:
                raise ContinuousTrackSamplerLifecycleError(
                    "suspended sampler must be resumed, not started"
                )
            if self._state is not ContinuousSamplerState.NEW:
                raise ContinuousTrackSamplerLifecycleError(
                    f"sampler cannot start while {self._state.value}"
                )
            claimed: list[TrackHandle] = []
            started: list[TrackHandle] = []
            active_handle: TrackHandle | None = None
            try:
                for handle in self.handles:
                    active_handle = handle
                    handle._claim_continuous_sampler(self._owner_token)
                    claimed.append(handle)
                    self._claimed_track_ids.add(handle.track_id)
                    if handle.start():
                        started.append(handle)
                    self._health[handle.track_id].track_state = handle.state
            except Exception as exc:
                if active_handle is not None:
                    self._record_track_error(
                        active_handle,
                        operation="start",
                        exc=exc,
                    )
                for handle in reversed(started):
                    try:
                        handle.stop()
                        self._health[handle.track_id].track_state = TrackState.STOPPED
                    except Exception as rollback_exc:
                        self._record_track_error(
                            handle,
                            operation="start_rollback_stop",
                            exc=rollback_exc,
                        )
                for handle in reversed(claimed):
                    handle._release_continuous_sampler(self._owner_token)
                    self._claimed_track_ids.discard(handle.track_id)
                self._state = ContinuousSamplerState.FAILED
                raise ContinuousTrackSamplerLifecycleError(
                    f"failed to start continuous sampler: {type(exc).__name__}:{exc}"
                ) from exc
            self._stop_requested.clear()
            self._state = ContinuousSamplerState.RUNNING
            self._started_at = _utc_now()
            self._stopped_at = None
            worker = threading.Thread(
                target=self._run,
                name=f"robomex-track-sampler-{self.sampler_id}",
                daemon=True,
            )
            self._thread = worker
            try:
                worker.start()
            except Exception as exc:
                self._state = ContinuousSamplerState.FAILED
                for handle in reversed(self.handles):
                    try:
                        handle.stop()
                        self._health[handle.track_id].track_state = TrackState.STOPPED
                    finally:
                        handle._release_continuous_sampler(self._owner_token)
                        self._claimed_track_ids.discard(handle.track_id)
                raise ContinuousTrackSamplerLifecycleError(
                    f"failed to create sampling worker: {type(exc).__name__}:{exc}"
                ) from exc
            self._thread_start_count += 1
            return True

    def suspend(self) -> bool:
        """Pause polling and every backend session after any in-flight round."""

        with self._poll_lock, self._condition:
            if self._state is ContinuousSamplerState.SUSPENDED:
                return False
            if self._state is not ContinuousSamplerState.RUNNING:
                raise ContinuousTrackSamplerLifecycleError(
                    f"sampler cannot suspend while {self._state.value}"
                )
            self._state = ContinuousSamplerState.SUSPENDED
            try:
                for handle in self.handles:
                    handle.suspend()
                    self._health[handle.track_id].track_state = TrackState.SUSPENDED
            except Exception as exc:
                self._record_track_error(handle, operation="suspend", exc=exc)
                self._state = ContinuousSamplerState.FAILED
                self._stop_requested.set()
                self._condition.notify_all()
                raise ContinuousTrackSamplerLifecycleError(
                    f"failed to suspend track {handle.track_id!r}"
                ) from exc
            self._condition.notify_all()
            return True

    def resume(self) -> bool:
        """Resume all backend sessions, then wake the worker immediately."""

        with self._poll_lock, self._condition:
            if self._state is ContinuousSamplerState.RUNNING:
                return False
            if self._state is not ContinuousSamplerState.SUSPENDED:
                raise ContinuousTrackSamplerLifecycleError(
                    f"sampler cannot resume while {self._state.value}"
                )
            try:
                for handle in self.handles:
                    handle.resume()
                    self._health[handle.track_id].track_state = TrackState.ACTIVE
            except Exception as exc:
                self._record_track_error(handle, operation="resume", exc=exc)
                self._state = ContinuousSamplerState.FAILED
                self._stop_requested.set()
                self._condition.notify_all()
                raise ContinuousTrackSamplerLifecycleError(
                    f"failed to resume track {handle.track_id!r}"
                ) from exc
            self._state = ContinuousSamplerState.RUNNING
            self._condition.notify_all()
            return True

    def stop(self, *, timeout_s: float | None = None) -> bool:
        """Stop with a bounded join; never claim success while worker is alive."""

        if isinstance(timeout_s, bool):
            raise ValueError("stop timeout must be numeric, not boolean")
        timeout = self.stop_join_timeout_s if timeout_s is None else float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("stop timeout must be finite and positive")
        with self._condition:
            if self._state is ContinuousSamplerState.STOPPED:
                return False
            if self._state is ContinuousSamplerState.STOPPING:
                raise ContinuousTrackSamplerLifecycleError("sampler stop is already in progress")
            if self._state not in {
                ContinuousSamplerState.NEW,
                ContinuousSamplerState.RUNNING,
                ContinuousSamplerState.SUSPENDED,
                ContinuousSamplerState.FAILED,
                ContinuousSamplerState.STOP_FAILED,
            }:
                raise ContinuousTrackSamplerLifecycleError(
                    f"sampler cannot stop while {self._state.value}"
                )
            self._state = ContinuousSamplerState.STOPPING
            self._stop_requested.set()
            self._condition.notify_all()
            worker = self._thread
        if worker is not None:
            worker.join(timeout=timeout)
            if worker.is_alive():
                with self._condition:
                    self._state = ContinuousSamplerState.STOP_FAILED
                    in_flight_track_id = self._in_flight_track_id
                stop_error = ContinuousTrackSamplerStopError(
                    f"sampling worker did not stop within {timeout:.3f}s"
                )
                blocked_handle = next(
                    (
                        handle
                        for handle in self.handles
                        if handle.track_id == in_flight_track_id
                    ),
                    None,
                )
                if blocked_handle is not None:
                    self._record_track_error(
                        blocked_handle,
                        operation="stop_join",
                        exc=stop_error,
                    )
                raise stop_error
        failures: list[tuple[TrackHandle, Exception]] = []
        with self._poll_lock:
            owned_handles = tuple(
                handle
                for handle in self.handles
                if handle.track_id in self._claimed_track_ids
            )
            for handle in owned_handles:
                try:
                    handle.stop()
                    with self._condition:
                        self._health[handle.track_id].track_state = TrackState.STOPPED
                except Exception as exc:
                    failures.append((handle, exc))
                    self._record_track_error(handle, operation="stop", exc=exc)
                finally:
                    handle._release_continuous_sampler(self._owner_token)
                    self._claimed_track_ids.discard(handle.track_id)
        with self._condition:
            if failures:
                self._state = ContinuousSamplerState.STOP_FAILED
                details = ", ".join(
                    f"{handle.track_id}:{type(exc).__name__}"
                    for handle, exc in failures
                )
                raise ContinuousTrackSamplerStopError(
                    f"track shutdown failed: {details}"
                )
            self._state = ContinuousSamplerState.STOPPED
            self._stopped_at = _utc_now()
            return True

    def __enter__(self) -> ContinuousTrackSampler:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.stop()

    def _run(self) -> None:
        try:
            while not self._stop_requested.is_set():
                with self._condition:
                    while (
                        self._state is ContinuousSamplerState.SUSPENDED
                        and not self._stop_requested.is_set()
                    ):
                        self._condition.wait()
                    if self._stop_requested.is_set():
                        return
                    if self._state is not ContinuousSamplerState.RUNNING:
                        return
                    self._round_count += 1
                    round_index = self._round_count
                with self._poll_lock:
                    with self._condition:
                        if (
                            self._state is not ContinuousSamplerState.RUNNING
                            or self._stop_requested.is_set()
                        ):
                            continue
                    for handle in self.handles:
                        if self._stop_requested.is_set():
                            break
                        self._poll_once(handle, round_index=round_index)
                with self._condition:
                    if self._stop_requested.is_set():
                        return
                    if self._state is ContinuousSamplerState.RUNNING:
                        self._condition.wait(timeout=self.poll_interval_s)
        except BaseException as exc:
            with self._condition:
                if self._state not in {
                    ContinuousSamplerState.STOPPING,
                    ContinuousSamplerState.STOPPED,
                }:
                    self._state = ContinuousSamplerState.FAILED
                    self._stop_requested.set()
                    for handle in self.handles:
                        self._record_track_error(
                            handle,
                            operation="worker",
                            exc=exc,
                        )
                    self._condition.notify_all()

    def _poll_once(self, handle: TrackHandle, *, round_index: int) -> None:
        attempted_at = _utc_now()
        with self._condition:
            self._in_flight_track_id = handle.track_id
            mutable = self._health[handle.track_id]
            mutable.attempt_count += 1
            mutable.last_attempt_at = attempted_at
        try:
            if self.max_silence_s is None:
                sample = handle.poll(_continuous_sampler_token=self._owner_token)
            else:
                sample = handle.poll_if_available(
                    _continuous_sampler_token=self._owner_token
                )
                if sample is None:
                    now_monotonic = time.monotonic()
                    with self._condition:
                        mutable = self._health[handle.track_id]
                        if mutable.silence_started_monotonic_s is None:
                            mutable.silence_started_monotonic_s = now_monotonic
                        silence_age_s = (
                            now_monotonic - mutable.silence_started_monotonic_s
                        )
                        publish_lost = (
                            silence_age_s >= self.max_silence_s
                            and not mutable.silence_lost_published
                        )
                    if publish_lost:
                        sample = handle.publish_lost(
                            reason=(
                                "max_silence_exceeded:"
                                f"{self.max_silence_s:.6f}s"
                            ),
                            _continuous_sampler_token=self._owner_token,
                        )
                        with self._condition:
                            self._health[
                                handle.track_id
                            ].silence_lost_published = True
        except Exception as exc:
            error = self._record_track_error(handle, operation="poll", exc=exc)
            finished_at = _utc_now()
            event = TrackSamplingEvent(
                sampler_id=self.sampler_id,
                round_index=round_index,
                track_id=handle.track_id,
                outcome=TrackSamplingOutcome.ERROR,
                attempted_at=attempted_at,
                finished_at=finished_at,
                error=error,
            )
        else:
            finished_at = _utc_now()
            if sample is None:
                with self._condition:
                    self._health[handle.track_id].silent_count += 1
                event = TrackSamplingEvent(
                    sampler_id=self.sampler_id,
                    round_index=round_index,
                    track_id=handle.track_id,
                    outcome=TrackSamplingOutcome.SILENT,
                    attempted_at=attempted_at,
                    finished_at=finished_at,
                )
                with self._condition:
                    self._events.append(event)
                    self._in_flight_track_id = None
                return
            outcome = (
                TrackSamplingOutcome.TRACKED
                if sample.quality is ObservationQuality.TRACKED
                else TrackSamplingOutcome.LOST
            )
            with self._condition:
                mutable = self._health[handle.track_id]
                mutable.sample_count += 1
                mutable.last_success_at = finished_at
                mutable.last_sample_id = sample.sample_id
                mutable.last_quality = sample.quality
                if sample.quality is ObservationQuality.TRACKED:
                    mutable.tracked_count += 1
                    mutable.last_tracked_at = finished_at
                    mutable.silence_started_monotonic_s = None
                    mutable.silence_lost_published = False
                else:
                    mutable.lost_count += 1
                    if self.max_silence_s is not None:
                        if mutable.silence_started_monotonic_s is None:
                            mutable.silence_started_monotonic_s = time.monotonic()
                        mutable.silence_lost_published = True
            event = TrackSamplingEvent(
                sampler_id=self.sampler_id,
                round_index=round_index,
                track_id=handle.track_id,
                outcome=outcome,
                attempted_at=attempted_at,
                finished_at=finished_at,
                sample_id=sample.sample_id,
                sample_sequence=sample.sequence,
                quality=sample.quality,
            )
        with self._condition:
            self._events.append(event)
            self._in_flight_track_id = None

    def _record_track_error(
        self,
        handle: TrackHandle,
        *,
        operation: str,
        exc: BaseException,
    ) -> TrackSamplingError:
        error = TrackSamplingError(
            track_id=handle.track_id,
            operation=operation,
            error_type=type(exc).__name__,
            message=str(exc),
        )
        with self._condition:
            mutable = self._health[handle.track_id]
            mutable.error_count += 1
            mutable.last_error = error
        return error

    def _track_health_snapshot(self, handle: TrackHandle) -> TrackSamplingHealth:
        mutable = self._health[handle.track_id]
        return TrackSamplingHealth(
            track_id=handle.track_id,
            track_state=mutable.track_state,
            attempt_count=mutable.attempt_count,
            sample_count=mutable.sample_count,
            tracked_count=mutable.tracked_count,
            lost_count=mutable.lost_count,
            silent_count=mutable.silent_count,
            error_count=mutable.error_count,
            last_attempt_at=mutable.last_attempt_at,
            last_success_at=mutable.last_success_at,
            last_tracked_at=mutable.last_tracked_at,
            last_sample_id=mutable.last_sample_id,
            last_quality=mutable.last_quality,
            last_error=mutable.last_error,
        )


class ObservationRegistry:
    """Episode owner for streams, read-only backends, and service handles."""

    def __init__(
        self,
        *,
        episode_id: str,
        stream: ObservationStream | None = None,
    ) -> None:
        if not episode_id.strip():
            raise ValueError("episode_id must not be empty")
        if stream is not None and stream.episode_id != episode_id:
            raise ObservationContractError("injected stream belongs to another episode")
        self.episode_id = episode_id
        self.stream = stream or ObservationStream(episode_id=episode_id)
        self._backends: dict[str, ObservationBackend] = {}
        self._tracks: dict[str, TrackHandle] = {}
        self._closed = False
        self._lock = threading.RLock()

    def register_backend(self, backend: ObservationBackend) -> None:
        _validate_backend(backend)
        with self._lock:
            if self._closed:
                raise TrackStateError("observation registry is closed")
            previous = self._backends.get(backend.backend_id)
            if previous is backend:
                return
            if previous is not None:
                raise TrackConflictError(
                    f"backend id {backend.backend_id!r} is already registered"
                )
            self._backends[backend.backend_id] = backend

    def create(self, request: TrackRequest, *, start: bool = False) -> TrackHandle:
        request = TrackRequest.model_validate(request)
        if request.episode_id != self.episode_id:
            raise ObservationContractError("track request belongs to another episode")
        with self._lock:
            if self._closed:
                raise TrackStateError("observation registry is closed")
            previous = self._tracks.get(request.track_id)
            if previous is not None:
                if previous.request != request:
                    raise TrackConflictError(
                        f"track id {request.track_id!r} is already bound"
                    )
                if start:
                    previous.start()
                return previous
            try:
                backend = self._backends[request.backend_id]
            except KeyError as exc:
                raise TrackNotFoundError(
                    f"unknown observation backend {request.backend_id!r}"
                ) from exc
            handle = TrackHandle(request=request, stream=self.stream, backend=backend)
            self._tracks[request.track_id] = handle
            if start:
                handle.start()
            return handle

    def get(self, track_id: str) -> TrackHandle:
        """Resolve an episode service from any workflow in the episode."""

        with self._lock:
            try:
                return self._tracks[track_id]
            except KeyError as exc:
                raise TrackNotFoundError(f"unknown track {track_id!r}") from exc

    @property
    def tracks(self) -> tuple[TrackHandle, ...]:
        with self._lock:
            return tuple(self._tracks[key] for key in sorted(self._tracks))

    def close_workflow(self, workflow_id: str) -> None:
        """No-op by design: episode services survive subgoal/workflow closure."""

        if not workflow_id.strip():
            raise ValueError("workflow_id must not be empty")

    def close(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            for handle in self._tracks.values():
                handle.stop()
            self._closed = True
            return True


def _validate_backend(backend: ObservationBackend) -> None:
    if not isinstance(backend, ObservationBackend):
        raise BackendAuthorityError("backend does not implement ObservationBackend")
    if not isinstance(backend.backend_id, str) or not backend.backend_id.strip():
        raise BackendAuthorityError("backend_id must be a non-empty string")
    read_capabilities = frozenset(backend.read_capabilities)
    effect_capabilities = frozenset(backend.effect_capabilities)
    if effect_capabilities:
        raise BackendAuthorityError("observation backends must have an empty effect ceiling")
    try:
        _reject_oracle_names(read_capabilities, field_name="backend capabilities")
    except ObservationContractError as exc:
        raise BackendAuthorityError(str(exc)) from exc
    forbidden = sorted(
        capability
        for capability in read_capabilities
        if _ACTION_CAPABILITY_RE.search(capability)
    )
    if forbidden:
        raise BackendAuthorityError(
            f"observation backend advertises action capabilities: {forbidden!r}"
        )


def _reject_oracle_names(values: Iterable[str], *, field_name: str) -> None:
    forbidden = sorted(value for value in values if _ORACLE_COMPONENT_RE.search(value))
    if forbidden:
        raise ObservationContractError(
            f"{field_name} contains forbidden oracle/ground-truth names: {forbidden!r}"
        )


def _reject_oracle_payload(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if _ORACLE_COMPONENT_RE.search(key_text):
                raise ObservationContractError(
                    f"simulator oracle/ground-truth data is forbidden at {path}.{key_text}"
                )
            _reject_oracle_payload(item, path=f"{path}.{key_text}")
    elif isinstance(value, str):
        if _ORACLE_COMPONENT_RE.search(value):
            raise ObservationContractError(
                f"simulator oracle/ground-truth data is forbidden at {path}"
            )
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_oracle_payload(item, path=f"{path}[{index}]")


def _require_finite_json(value: Any, *, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ObservationContractError(f"non-finite observation value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_json(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _require_finite_json(item, path=f"{path}[{index}]")


def _dedupe_refs(values: Iterable[ResolvedArtifactRef]) -> tuple[ResolvedArtifactRef, ...]:
    seen: set[tuple[str, str]] = set()
    result: list[ResolvedArtifactRef] = []
    for value in values:
        key = (value.artifact_id, value.content_digest)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


__all__ = [
    "BackendAuthorityError",
    "BackendObservation",
    "CheckpointObservationBackend",
    "CheckpointRecord",
    "CheckpointSegmentation",
    "CheckpointSegmenter",
    "ContinuousSamplerState",
    "ContinuousTrackSampler",
    "ContinuousTrackSamplerError",
    "ContinuousTrackSamplerHealth",
    "ContinuousTrackSamplerLifecycleError",
    "ContinuousTrackSamplerStopError",
    "EntityIdentity",
    "InMemoryObservationBackend",
    "ObservationBackend",
    "ObservationContractError",
    "ObservationError",
    "ObservationHistoryGapError",
    "ObservationQuality",
    "ObservationRegistry",
    "ObservationRevisionVector",
    "ObservationSample",
    "ObservationStream",
    "ObservationSubscription",
    "TrackConflictError",
    "TrackHandle",
    "TrackNotFoundError",
    "TrackRequest",
    "TrackSamplingError",
    "TrackSamplingEvent",
    "TrackSamplingHealth",
    "TrackSamplingOutcome",
    "TrackState",
    "TrackStateError",
]
