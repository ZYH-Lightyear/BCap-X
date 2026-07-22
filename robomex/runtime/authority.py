"""Authority, lease, WAL, and sealed execution boundary for RoboMEx v2.

The invariant enforced here is **one authoritative effect writer per
``(world_id, resource_id)``**, not one planner or one simulation globally.
Read-only proposal workers and isolated shadow worlds may run concurrently.
Only an action admitted by :class:`ActionSupervisor` may enter
:class:`SealedActionRunner` and produce an authoritative execution receipt.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import queue
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    BinaryIO,
    Literal,
    Protocol,
    cast,
    runtime_checkable,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from robomex.runtime.action_protocol import (
    ActionAttempt,
    ActionSpec,
    ActionSpecType,
    AdmissionSnapshot,
    BackendCallResult,
    BackendDescriptor,
    BackendMotionInterface,
    ControllerState,
    DigestStr,
    ExecutionReceipt,
    ExecutionStatus,
    FeasibilityCertificate,
    FeasibilityStatus,
    GripperCommand,
    MonitorTelemetryCapabilities,
    MonitorTelemetryHook,
    MotionPlan,
    PrimitiveReceipt,
    PrimitiveStatus,
    RecoveryAcknowledgement,
    ShadowRolloutReceipt,
    ShadowRolloutStatus,
    WaitSpec,
    WalRecord,
    WorldKind,
    action_spec_id,
    canonical_payload_digest,
    validate_action_spec,
    validate_wal_record,
)
from robomex.runtime.evidence_recorder import ActionRuntimeEvidenceSample

if TYPE_CHECKING:
    from robomex.authoring.monitoring import MonitorEvaluation, MonitorRuntime

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LeaseKey = tuple[str, str]
ReservationValidator = Callable[[str, str, str, str], bool]


def _new_id() -> str:
    return uuid.uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility


class ActionAuthorityError(RuntimeError):
    """Base class for authoritative action protocol failures."""


class AdmissionRejectedError(ActionAuthorityError):
    """A sealed spec or live guard did not pass admission."""


class ActionLeaseConflictError(AdmissionRejectedError):
    """The requested world/resource already has an authoritative writer."""


class SealedActionMismatchError(ActionAuthorityError):
    """Execution input differs from the exact admitted action."""


class RecoveryBlockedError(AdmissionRejectedError):
    """An indeterminate prior effect requires explicit recovery first."""


class ContinuationAlreadyUsedError(AdmissionRejectedError):
    """A graph activation attempt already crossed the physical WAL boundary."""


class ShadowIsolationError(ActionAuthorityError):
    """A shadow rollout is not isolated from another writer."""


class WalConflictError(ActionAuthorityError):
    """A WAL record ID was reused with different immutable content."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class AdmittedAction(_StrictModel):
    """Capability issued by one supervisor while it owns the resource lease."""

    schema_version: Literal["robomex.admitted_action.v1"] = "robomex.admitted_action.v1"
    admission_id: NonEmptyStr = Field(default_factory=_new_id)
    action_id: NonEmptyStr = Field(default_factory=_new_id)
    supervisor_id: NonEmptyStr
    lease_token: NonEmptyStr = Field(default_factory=_new_id)
    action_spec: ActionSpec
    spec_digest: DigestStr
    admission_snapshot: AdmissionSnapshot
    admission_snapshot_digest: DigestStr
    feasibility_certificate: FeasibilityCertificate | None = None
    monitor_digest: DigestStr | None = None
    continuation_id: NonEmptyStr | None = None
    scheduler_reservation_id: NonEmptyStr | None = None
    admitted_at: datetime = Field(default_factory=_utc_now)

    @field_validator("admitted_at")
    @classmethod
    def _admitted_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("admitted_at must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10

    @model_validator(mode="after")
    def _validate_binding(self) -> AdmittedAction:
        if self.action_spec.content_digest != self.spec_digest:
            raise ValueError("spec_digest must match action_spec")
        if self.action_spec.expected_snapshot.world_kind is not WorldKind.AUTHORITATIVE:
            raise ValueError("only authoritative specs can be admitted")
        if self.feasibility_certificate is not None:
            certificate = self.feasibility_certificate
            if certificate.action_spec_digest != self.spec_digest:
                raise ValueError("certificate does not bind the admitted spec")
            if certificate.admission_snapshot_digest != self.admission_snapshot_digest:
                raise ValueError("certificate does not bind the admission snapshot")
            if certificate.overall_status is not FeasibilityStatus.PASS:
                raise ValueError("only a passing feasibility certificate is admissible")
        return self

    @property
    def lease_key(self) -> LeaseKey:
        return (self.action_spec.world_id, self.action_spec.resource_id)


@runtime_checkable
class ActionWAL(Protocol):
    """Append-only persistence boundary for authoritative effect records."""

    def append(self, record: WalRecord) -> bool:
        """Durably append a new record; return False for an exact duplicate."""

    def records(self) -> tuple[WalRecord, ...]:
        """Return records in durable append order."""


class InMemoryActionWAL:
    """Thread-safe WAL test double with the same immutable append semantics."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: list[WalRecord] = []
        self._by_id: dict[str, WalRecord] = {}
        self.flush_count = 0

    def append(self, record: WalRecord) -> bool:
        validated = validate_wal_record(record)
        with self._lock:
            previous = self._by_id.get(validated.record_id)
            if previous is not None:
                if previous == validated:
                    return False
                raise WalConflictError(f"record_id {validated.record_id!r} already exists")
            self._records.append(validated)
            self._by_id[validated.record_id] = validated
            # A completed append is the durability boundary of this test WAL.
            self.flush_count += 1
            return True

    def records(self) -> tuple[WalRecord, ...]:
        with self._lock:
            # Never expose the WAL's internal objects: Pydantic's frozen flag
            # is shallow and telemetry mappings remain mutable containers.
            return tuple(
                validate_wal_record(item.model_dump(mode="python"))
                for item in self._records
            )


class JsonlActionWAL:
    """Small durable JSONL WAL; every append is flushed and fsynced."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: list[WalRecord] = []
        self._by_id: dict[str, WalRecord] = {}
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._lock_path.touch(exist_ok=True)
        with self._process_lock(exclusive=False):
            self._reload_from_disk()

    @contextmanager
    def _process_lock(self, *, exclusive: bool) -> Iterator[None]:
        """Serialize readers/writers across EpisodeRuntime processes."""

        with self._lock_path.open("a+b") as lock_stream:
            fcntl.flock(
                lock_stream.fileno(),
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
            try:
                yield
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    def _reload_from_disk(self) -> None:
        records: list[WalRecord] = []
        by_id: dict[str, WalRecord] = {}
        if self.path.exists():
            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    record = validate_wal_record(line)
                except Exception as exc:
                    raise ValueError(
                        f"invalid action WAL record at line {line_number}"
                    ) from exc
                previous = by_id.get(record.record_id)
                if previous is not None and previous != record:
                    raise WalConflictError(f"conflicting record_id {record.record_id!r}")
                if previous is None:
                    records.append(record)
                    by_id[record.record_id] = record
        self._records = records
        self._by_id = by_id

    def append(self, record: WalRecord) -> bool:
        validated = validate_wal_record(record)
        with self._lock, self._process_lock(exclusive=True):
            self._reload_from_disk()
            previous = self._by_id.get(validated.record_id)
            if previous is not None:
                if previous == validated:
                    return False
                raise WalConflictError(
                    f"record_id {validated.record_id!r} already exists"
                )
            payload = json.dumps(
                validated.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(payload + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._records.append(validated)
            self._by_id[validated.record_id] = validated
            return True

    def records(self) -> tuple[WalRecord, ...]:
        with self._lock, self._process_lock(exclusive=False):
            self._reload_from_disk()
            return tuple(
                validate_wal_record(item.model_dump(mode="python"))
                for item in self._records
            )


@runtime_checkable
class ActionBackend(Protocol):
    """Trusted adapter boundary; deliberately exposes no pose/IK method."""

    @property
    def descriptor(self) -> BackendDescriptor: ...

    def snapshot(self, world_id: str, resource_id: str) -> AdmissionSnapshot: ...

    def execute_joint_path(
        self,
        *,
        world_id: str,
        resource_id: str,
        joint_names: tuple[str, ...],
        positions_rad: tuple[tuple[float, ...], ...],
        mode: str,
        subsample: int,
        timeout_s: float,
    ) -> BackendCallResult: ...

    def set_gripper(
        self,
        *,
        world_id: str,
        resource_id: str,
        mode: str,
        target_width_m: float,
        max_effort_n: float | None,
        timeout_s: float,
    ) -> BackendCallResult: ...

    def wait(
        self,
        *,
        world_id: str,
        resource_id: str,
        duration_s: float | None,
        control_steps: int | None,
        hold_command: str,
        timeout_s: float,
    ) -> BackendCallResult: ...


@runtime_checkable
class FeasibilityChecker(Protocol):
    """Trusted read-only checker for an exact action/snapshot pair."""

    def certify(
        self, spec: ActionSpec, snapshot: AdmissionSnapshot
    ) -> FeasibilityCertificate: ...


@runtime_checkable
class MonitorSampleBackend(Protocol):
    """Optional read-only telemetry surface for phase monitors."""

    def monitor_sample(
        self,
        *,
        world_id: str,
        resource_id: str,
        phase: str,
        sequence: int,
    ) -> Mapping[str, object]: ...


@runtime_checkable
class MonitorTelemetryBackend(MonitorSampleBackend, Protocol):
    """Backend that declares exact, resource-bound monitor observability."""

    def monitor_telemetry_capabilities(
        self,
        *,
        world_id: str,
        resource_id: str,
    ) -> MonitorTelemetryCapabilities: ...


@runtime_checkable
class CooperativeMotionBackend(Protocol):
    """Optional exact-path backend that invokes a runtime progress callback."""

    def execute_joint_path_cooperative(
        self,
        *,
        progress_callback: Callable[[Mapping[str, object]], bool],
        **kwargs: object,
    ) -> BackendCallResult: ...


@runtime_checkable
class WatchdogControllableBackend(Protocol):
    """Controller-level stop surface used after a runtime watchdog expires.

    Returning a snapshot is intentional: a successful method call is not proof
    that the physical controller actually became quiescent.
    """

    def stop_and_wait_quiescent(
        self,
        *,
        world_id: str,
        resource_id: str,
        timeout_s: float,
    ) -> AdmissionSnapshot: ...


@runtime_checkable
class ActionEvidenceRecorder(Protocol):
    """Runtime-owned synchronization surface for frames and action video."""

    def record(
        self, *, action_id: str, sequence: int, phase: str, sample: Mapping[str, object]
    ) -> str | None: ...

    def finalize(self, *, action_id: str) -> tuple[tuple[str, ...], str | None]: ...


def build_feasibility_certificate(
    *,
    spec: ActionSpec,
    snapshot: AdmissionSnapshot,
    checker_id: str,
    checks: Mapping[str, FeasibilityStatus | str],
    evidence_refs: tuple[str, ...] = (),
) -> FeasibilityCertificate:
    """Build the closed certificate after a trusted checker has run."""

    normalized = {name: FeasibilityStatus(value) for name, value in checks.items()}
    values = set(normalized.values())
    overall = (
        FeasibilityStatus.FAIL
        if FeasibilityStatus.FAIL in values
        else (
            FeasibilityStatus.UNKNOWN
            if FeasibilityStatus.UNKNOWN in values
            else FeasibilityStatus.PASS
        )
    )
    return FeasibilityCertificate(
        action_spec_digest=spec.content_digest,
        admission_snapshot_digest=_snapshot_digest(snapshot),
        checker_id=checker_id,
        checks=normalized,
        overall_status=overall,
        evidence_refs=evidence_refs,
    )

def _snapshot_digest(snapshot: AdmissionSnapshot) -> str:
    return canonical_payload_digest(
        snapshot.schema_version,
        cast(dict[str, object], snapshot.model_dump(mode="json")),
    )


def _fresh_snapshot(snapshot: AdmissionSnapshot) -> AdmissionSnapshot:
    # Force validation even for a model_copy() instance.
    return AdmissionSnapshot.model_validate(snapshot.model_dump(mode="python"))


def _assert_snapshot_matches(spec: ActionSpec, snapshot: AdmissionSnapshot) -> None:
    expected = spec.expected_snapshot
    current = _fresh_snapshot(snapshot)
    exact_fields = (
        "world_id",
        "world_kind",
        "resource_id",
        "robot_revision",
        "scene_revision",
        "attachment_revision",
        "config_revision",
        "config_digest",
        "collision_world_digest",
        "attachment_status",
        "joint_names",
    )
    mismatches = [
        name for name in exact_fields if getattr(current, name) != getattr(expected, name)
    ]
    if mismatches:
        raise AdmissionRejectedError("snapshot guard mismatch: " + ", ".join(mismatches))
    if current.controller_state not in {ControllerState.READY, ControllerState.QUIESCENT}:
        raise AdmissionRejectedError(
            f"controller is not admissible: {current.controller_state.value}"
        )
    if len(current.joint_positions_rad) != len(expected.joint_positions_rad):
        raise AdmissionRejectedError("start joint vector width changed")
    deviations = [
        abs(now - planned)
        for now, planned in zip(
            current.joint_positions_rad, expected.joint_positions_rad, strict=True
        )
    ]
    if deviations and max(deviations) > spec.max_start_deviation_rad:
        raise AdmissionRejectedError(
            "start joints exceed max_start_deviation_rad "
            f"({max(deviations):.6f} > {spec.max_start_deviation_rad:.6f})"
        )


def _required_feasibility_checks(spec: ActionSpec) -> frozenset[str]:
    common = {
        "exact_joint_path_interface",
        "controller_admissible",
        "resource_binding",
    }
    if isinstance(spec, MotionPlan):
        return frozenset(
            common | {"joint_order", "joint_limits", "robot_model", "collision"}
        )
    if isinstance(spec, GripperCommand):
        return frozenset(
            common | {"gripper_target_range", "gripper_effort_support"}
        )
    if isinstance(spec, WaitSpec):
        return frozenset(common | {"wait_timeout_bound"})
    raise TypeError(f"unsupported action spec {type(spec)!r}")


class _LeaseState:
    def __init__(self, admitted: AdmittedAction) -> None:
        self.admitted = admitted
        self.executing = False


class ActionSupervisor:
    """Admission authority and resource-scoped authoritative writer registry."""

    def __init__(
        self,
        wal: ActionWAL,
        *,
        supervisor_id: str = "action-supervisor-v2",
        require_certificate: bool = False,
        interprocess_lock_root: str | Path | None = None,
        max_snapshot_age_s: float | None = None,
        max_clock_skew_s: float = 1.0,
        reservation_validator: ReservationValidator | None = None,
    ) -> None:
        self.wal = wal
        self.supervisor_id = str(supervisor_id).strip()
        if not self.supervisor_id:
            raise ValueError("supervisor_id must not be empty")
        self._lock = threading.RLock()
        self.require_certificate = bool(require_certificate)
        self.max_snapshot_age_s = (
            None if max_snapshot_age_s is None else float(max_snapshot_age_s)
        )
        if self.max_snapshot_age_s is not None and self.max_snapshot_age_s <= 0:
            raise ValueError("max_snapshot_age_s must be positive when supplied")
        self.max_clock_skew_s = float(max_clock_skew_s)
        if self.max_clock_skew_s < 0:
            raise ValueError("max_clock_skew_s must be non-negative")
        self._reservation_validator = reservation_validator
        self._interprocess_lock_root = (
            Path(interprocess_lock_root) if interprocess_lock_root is not None else None
        )
        if self._interprocess_lock_root is not None:
            self._interprocess_lock_root.mkdir(parents=True, exist_ok=True)
        self._process_leases: dict[LeaseKey, BinaryIO] = {}
        self._leases: dict[LeaseKey, _LeaseState] = {}
        self._blocked: set[LeaseKey] = set()
        existing = wal.records()
        self._blocked = self._blocked_from_records(existing)

    @staticmethod
    def _blocked_from_records(records: tuple[WalRecord, ...]) -> set[LeaseKey]:
        unresolved: dict[LeaseKey, set[str]] = {}
        for record in records:
            if isinstance(record, ActionAttempt):
                key = (record.action_spec.world_id, record.action_spec.resource_id)
                unresolved.setdefault(key, set()).add(record.action_id)
            elif isinstance(record, ExecutionReceipt):
                key = (record.world_id, record.resource_id)
                if record.runtime_status not in {
                    ExecutionStatus.INDETERMINATE_AFTER_CRASH,
                    ExecutionStatus.INDETERMINATE_AFTER_TIMEOUT,
                }:
                    unresolved.setdefault(key, set()).discard(record.action_id)
            elif isinstance(record, RecoveryAcknowledgement):
                key = (record.world_id, record.resource_id)
                pending = unresolved.setdefault(key, set())
                if pending.issubset(record.prior_action_ids):
                    pending.clear()
        return {key for key, action_ids in unresolved.items() if action_ids}

    def _assert_fresh_snapshot(self, snapshot: AdmissionSnapshot) -> None:
        if self.max_snapshot_age_s is None:
            return
        now = _utc_now()
        if snapshot.captured_at > now + timedelta(seconds=self.max_clock_skew_s):
            raise AdmissionRejectedError("admission snapshot timestamp is in the future")
        age_s = (now - snapshot.captured_at).total_seconds()
        if age_s > self.max_snapshot_age_s:
            raise AdmissionRejectedError(
                f"admission snapshot is stale ({age_s:.3f}s old)"
            )

    def validate_snapshot_freshness(
        self, snapshot: AdmissionSnapshot
    ) -> AdmissionSnapshot:
        """Validate a runtime capture against the same clock gate as admission."""

        current = _fresh_snapshot(snapshot)
        self._assert_fresh_snapshot(current)
        return current

    def _assert_fresh_certificate(
        self,
        certificate: FeasibilityCertificate,
        snapshot: AdmissionSnapshot,
    ) -> None:
        if (
            certificate.captured_at + timedelta(seconds=self.max_clock_skew_s)
            < snapshot.captured_at
        ):
            raise AdmissionRejectedError(
                "feasibility certificate predates its admission snapshot"
            )
        if self.max_snapshot_age_s is None:
            return
        now = _utc_now()
        if certificate.captured_at > now + timedelta(seconds=self.max_clock_skew_s):
            raise AdmissionRejectedError("feasibility certificate timestamp is in the future")
        age_s = (now - certificate.captured_at).total_seconds()
        if age_s > self.max_snapshot_age_s:
            raise AdmissionRejectedError(
                f"feasibility certificate is stale ({age_s:.3f}s old)"
            )

    def _process_lock_path(self, key: LeaseKey) -> Path | None:
        if self._interprocess_lock_root is None:
            return None
        digest = hashlib.sha256(
            (key[0] + "\0" + key[1]).encode("utf-8")
        ).hexdigest()
        return self._interprocess_lock_root / f"resource-{digest}.lock"

    def _acquire_process_lease(self, key: LeaseKey) -> BinaryIO | None:
        path = self._process_lock_path(key)
        if path is None:
            return None
        stream = path.open("a+b")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise ActionLeaseConflictError(
                f"another process owns authoritative writer {key!r}"
            ) from exc
        return stream

    @staticmethod
    def _release_process_lease(stream: BinaryIO | None) -> None:
        if stream is None:
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def admit(
        self,
        spec: ActionSpec | Mapping[str, object] | str | bytes,
        current_snapshot: AdmissionSnapshot,
        *,
        action_id: str | None = None,
        feasibility_certificate: FeasibilityCertificate | Mapping[str, object] | None = None,
        monitor_digest: str | None = None,
        continuation_id: str | None = None,
        scheduler_reservation_id: str | None = None,
        scheduler_reserved_world_id: str | None = None,
        scheduler_reserved_resource_id: str | None = None,
    ) -> AdmittedAction:
        sealed = validate_action_spec(spec)
        current = _fresh_snapshot(current_snapshot)
        if sealed.expected_snapshot.world_kind is not WorldKind.AUTHORITATIVE:
            raise AdmissionRejectedError(
                "shadow specs cannot enter authoritative admission"
            )
        _assert_snapshot_matches(sealed, current)
        self._assert_fresh_snapshot(current)
        snapshot_digest = _snapshot_digest(current)
        certificate = None
        if feasibility_certificate is not None:
            if isinstance(feasibility_certificate, FeasibilityCertificate):
                feasibility_certificate = feasibility_certificate.model_dump(mode="python")
            certificate = FeasibilityCertificate.model_validate(feasibility_certificate)
            if certificate.action_spec_digest != sealed.content_digest:
                raise AdmissionRejectedError("certificate spec digest mismatch")
            if certificate.admission_snapshot_digest != snapshot_digest:
                raise AdmissionRejectedError("certificate snapshot digest mismatch")
            if certificate.overall_status is not FeasibilityStatus.PASS:
                raise AdmissionRejectedError("feasibility certificate did not pass")
            required_checks = _required_feasibility_checks(sealed)
            missing_checks = required_checks.difference(certificate.checks)
            if missing_checks:
                raise AdmissionRejectedError(
                    "feasibility certificate omits required checks: "
                    + ", ".join(sorted(missing_checks))
                )
            non_passing = {
                name
                for name in required_checks
                if certificate.checks[name] is not FeasibilityStatus.PASS
                and not (
                    isinstance(sealed, GripperCommand)
                    and sealed.max_effort_n is None
                    and name == "gripper_effort_support"
                    and certificate.checks[name]
                    is FeasibilityStatus.NOT_APPLICABLE
                )
            }
            if non_passing:
                raise AdmissionRejectedError(
                    "feasibility certificate has non-passing required checks: "
                    + ", ".join(sorted(non_passing))
                )
            self._assert_fresh_certificate(certificate, current)
        elif self.require_certificate:
            raise AdmissionRejectedError("a passing feasibility certificate is required")
        key = (sealed.world_id, sealed.resource_id)
        requested_action_id = action_id or _new_id()
        reservation_fields = (
            scheduler_reservation_id,
            scheduler_reserved_world_id,
            scheduler_reserved_resource_id,
        )
        if any(value is not None for value in reservation_fields):
            if not all(value is not None and str(value).strip() for value in reservation_fields):
                raise AdmissionRejectedError(
                    "scheduler reservation identity must be supplied atomically"
                )
            if (
                scheduler_reserved_world_id != sealed.world_id
                or scheduler_reserved_resource_id != sealed.resource_id
            ):
                raise AdmissionRejectedError(
                    "scheduler reservation does not bind the sealed resource"
                )
            if self._reservation_validator is not None and not self._reservation_validator(
                str(scheduler_reservation_id),
                sealed.world_id,
                sealed.resource_id,
                requested_action_id,
            ):
                raise AdmissionRejectedError(
                    "scheduler reservation is not active in the authority registry"
                )
        with self._lock:
            self._blocked = self._blocked_from_records(self.wal.records())
            if key in self._blocked:
                raise RecoveryBlockedError(
                    f"resource {key!r} awaits crash/timeout recovery"
                )
            if key in self._leases:
                raise ActionLeaseConflictError(
                    f"authoritative writer already holds {key!r}"
                )
            recorded_action_ids = {
                record.action_id
                for record in self.wal.records()
                if isinstance(record, (ActionAttempt, ExecutionReceipt))
            }
            active_action_ids = {
                state.admitted.action_id for state in self._leases.values()
            }
            if continuation_id is not None:
                recorded_continuations = {
                    record.continuation_id
                    for record in self.wal.records()
                    if isinstance(record, ActionAttempt)
                    and record.continuation_id is not None
                }
                active_continuations = {
                    state.admitted.continuation_id
                    for state in self._leases.values()
                    if state.admitted.continuation_id is not None
                }
                if continuation_id in recorded_continuations | active_continuations:
                    raise ContinuationAlreadyUsedError(
                        f"continuation_id {continuation_id!r} has already been used"
                    )
            if requested_action_id in recorded_action_ids | active_action_ids:
                raise AdmissionRejectedError(
                    f"action_id {requested_action_id!r} has already been used"
                )
            process_lease = self._acquire_process_lease(key)
            try:
                admitted = AdmittedAction(
                    action_id=requested_action_id,
                    supervisor_id=self.supervisor_id,
                    action_spec=sealed,
                    spec_digest=sealed.content_digest,
                    admission_snapshot=current,
                    admission_snapshot_digest=snapshot_digest,
                    feasibility_certificate=certificate,
                    monitor_digest=monitor_digest,
                    continuation_id=continuation_id,
                    scheduler_reservation_id=scheduler_reservation_id,
                )
                self._leases[key] = _LeaseState(admitted)
                if process_lease is not None:
                    self._process_leases[key] = process_lease
                return admitted
            except BaseException:
                self._release_process_lease(process_lease)
                raise

    def receipt_for_continuation(
        self, continuation_id: str
    ) -> ExecutionReceipt | None:
        """Return the immutable terminal receipt for an exact graph attempt."""

        normalized = str(continuation_id).strip()
        if not normalized:
            raise ValueError("continuation_id must not be empty")
        records = self.wal.records()
        action_ids = {
            record.action_id
            for record in records
            if isinstance(record, ActionAttempt) and record.continuation_id == normalized
        }
        if len(action_ids) > 1:
            raise WalConflictError("one continuation_id is bound to multiple actions")
        if not action_ids:
            return None
        action_id = next(iter(action_ids))
        receipts = [
            record
            for record in records
            if isinstance(record, ExecutionReceipt) and record.action_id == action_id
        ]
        if len(receipts) > 1:
            raise WalConflictError("one action has multiple terminal receipts")
        return receipts[0] if receipts else None

    def begin_execution(
        self,
        admitted: AdmittedAction,
        supplied_spec: ActionSpec,
        current_snapshot: AdmissionSnapshot,
    ) -> ActionSpec:
        sealed = validate_action_spec(supplied_spec)
        key = admitted.lease_key
        with self._lock:
            state = self._leases.get(key)
            if state is None:
                raise SealedActionMismatchError(
                    "admission no longer owns its resource lease"
                )
            issued = state.admitted
            identity = (
                admitted.supervisor_id,
                admitted.admission_id,
                admitted.action_id,
                admitted.lease_token,
                admitted.spec_digest,
            )
            issued_identity = (
                issued.supervisor_id,
                issued.admission_id,
                issued.action_id,
                issued.lease_token,
                issued.spec_digest,
            )
            if identity != issued_identity or admitted.supervisor_id != self.supervisor_id:
                raise SealedActionMismatchError("forged or stale admission capability")
            if state.executing:
                raise SealedActionMismatchError("admitted action has already started")
            if sealed.content_digest != issued.spec_digest or sealed != issued.action_spec:
                raise SealedActionMismatchError(
                    "supplied action is not the exact admitted spec"
                )
            _assert_snapshot_matches(sealed, current_snapshot)
            # Revisions/config must also remain the same as the admission-time
            # TOCTOU snapshot; joint positions retain the declared tolerance.
            admitted_snapshot = issued.admission_snapshot
            for name in (
                "robot_revision",
                "scene_revision",
                "attachment_revision",
                "config_revision",
                "config_digest",
                "collision_world_digest",
            ):
                if getattr(current_snapshot, name) != getattr(admitted_snapshot, name):
                    raise AdmissionRejectedError(f"{name} changed after admission")
            state.executing = True
            return sealed

    def release(self, admitted: AdmittedAction) -> None:
        with self._lock:
            state = self._leases.get(admitted.lease_key)
            if state is not None and state.admitted.lease_token == admitted.lease_token:
                del self._leases[admitted.lease_key]
                if admitted.lease_key not in self._blocked:
                    self._release_process_lease(
                        self._process_leases.pop(admitted.lease_key, None)
                    )

    def _block_unresolved(self, admitted: AdmittedAction) -> None:
        """Fail closed when this process survives a crash-equivalent fault."""

        with self._lock:
            self._blocked.add(admitted.lease_key)

    def has_lease(self, world_id: str, resource_id: str) -> bool:
        with self._lock:
            return (world_id, resource_id) in self._leases

    @property
    def blocked_resources(self) -> frozenset[LeaseKey]:
        with self._lock:
            return frozenset(self._blocked)

    def acknowledge_recovery(
        self,
        world_id: str,
        resource_id: str,
        *,
        current_snapshot: AdmissionSnapshot,
        evidence_refs: tuple[str, ...],
        reason: str,
    ) -> RecoveryAcknowledgement:
        """Durably reopen a quarantined resource after fresh quiescence proof."""

        key = (str(world_id).strip(), str(resource_id).strip())
        current = _fresh_snapshot(current_snapshot)
        with self._lock:
            self._blocked = self._blocked_from_records(self.wal.records())
        if key not in self.blocked_resources:
            raise RecoveryBlockedError(f"resource {key!r} is not quarantined")
        if (current.world_id, current.resource_id) != key:
            raise AdmissionRejectedError("recovery snapshot belongs to another resource")
        if current.world_kind is not WorldKind.AUTHORITATIVE:
            raise AdmissionRejectedError("recovery snapshot must be authoritative")
        if current.controller_state is not ControllerState.QUIESCENT:
            raise AdmissionRejectedError("recovery requires a quiescent controller")
        if not evidence_refs or any(not str(ref).strip() for ref in evidence_refs):
            raise AdmissionRejectedError("recovery requires explicit evidence refs")
        if not str(reason).strip():
            raise AdmissionRejectedError("recovery requires an explicit reason")

        records = self.wal.records()
        unresolved: set[str] = set()
        attempts: dict[str, ActionAttempt] = {}
        for record in records:
            if isinstance(record, ActionAttempt) and (
                record.action_spec.world_id,
                record.action_spec.resource_id,
            ) == key:
                attempts[record.action_id] = record
                unresolved.add(record.action_id)
            elif isinstance(record, ExecutionReceipt) and (
                record.world_id,
                record.resource_id,
            ) == key and record.runtime_status not in {
                ExecutionStatus.INDETERMINATE_AFTER_CRASH,
                ExecutionStatus.INDETERMINATE_AFTER_TIMEOUT,
            }:
                unresolved.discard(record.action_id)
            elif isinstance(record, RecoveryAcknowledgement) and (
                record.world_id,
                record.resource_id,
            ) == key and unresolved.issubset(record.prior_action_ids):
                unresolved.clear()
        if not unresolved:
            raise RecoveryBlockedError("no unresolved action is bound to the resource")
        baselines = [attempts[action_id].execution_snapshot for action_id in unresolved]
        for baseline in baselines:
            if any(
                now < before
                for now, before in (
                    (current.robot_revision, baseline.robot_revision),
                    (current.scene_revision, baseline.scene_revision),
                    (current.attachment_revision, baseline.attachment_revision),
                    (current.config_revision, baseline.config_revision),
                )
            ):
                raise AdmissionRejectedError("recovery snapshot revisions regressed")
        if not any(
            any(
                now > before
                for now, before in (
                    (current.robot_revision, baseline.robot_revision),
                    (current.scene_revision, baseline.scene_revision),
                    (current.attachment_revision, baseline.attachment_revision),
                    (current.config_revision, baseline.config_revision),
                )
            )
            for baseline in baselines
        ):
            raise AdmissionRejectedError(
                "recovery snapshot is not causally newer than the failed attempt"
            )
        acknowledgement = RecoveryAcknowledgement(
            world_id=key[0],
            resource_id=key[1],
            prior_action_ids=tuple(sorted(unresolved)),
            recovery_snapshot=current,
            recovery_snapshot_digest=_snapshot_digest(current),
            evidence_refs=tuple(dict.fromkeys(str(ref).strip() for ref in evidence_refs)),
            reason=str(reason).strip(),
        )
        self.wal.append(acknowledgement)
        with self._lock:
            self._blocked.discard(key)
            self._release_process_lease(self._process_leases.pop(key, None))
        return acknowledgement

    def reconcile_orphans(self) -> tuple[ExecutionReceipt, ...]:
        """Close admitted attempts without terminal receipts as indeterminate.

        This method never receives or calls an action backend, making automatic
        replay of a possibly executed physical effect structurally impossible.
        """

        records = self.wal.records()
        action_ids = tuple(
            dict.fromkeys(
                item.action_id for item in records if isinstance(item, ActionAttempt)
            )
        )
        terminal_ids = {
            item.action_id for item in records if isinstance(item, ExecutionReceipt)
        }
        reconciled: list[ExecutionReceipt] = []
        for action_id in action_ids:
            if action_id in terminal_ids:
                continue
            receipt = self.reconcile_action(
                action_id,
                abort_reason="orphan_attempt_reconciled_without_replay",
            )
            if receipt is not None:
                reconciled.append(receipt)
        return tuple(reconciled)

    def reconcile_action(
        self,
        action_id: str,
        *,
        abort_reason: str,
    ) -> ExecutionReceipt | None:
        """Close one WAL attempt as indeterminate without calling a backend."""

        records = self.wal.records()
        attempts = [
            item
            for item in records
            if isinstance(item, ActionAttempt) and item.action_id == action_id
        ]
        if len(attempts) > 1:
            raise WalConflictError("one action_id has multiple ActionAttempt records")
        terminals = [
            item
            for item in records
            if isinstance(item, ExecutionReceipt) and item.action_id == action_id
        ]
        if len(terminals) > 1:
            raise WalConflictError("one action_id has multiple terminal receipts")
        if terminals:
            return terminals[0]
        if not attempts:
            return None
        attempt = attempts[0]
        spec = attempt.action_spec
        primitives = tuple(
            sorted(
                (
                    item
                    for item in records
                    if isinstance(item, PrimitiveReceipt) and item.action_id == action_id
                ),
                key=lambda item: item.primitive_index,
            )
        )
        finished_at = _utc_now()
        if primitives and finished_at < primitives[-1].finished_at:
            finished_at = primitives[-1].finished_at
        receipt = ExecutionReceipt(
            action_id=attempt.action_id,
            spec_type=ActionSpecType(spec.spec_type),
            spec_id=action_spec_id(spec),
            spec_digest=attempt.spec_digest,
            world_id=spec.world_id,
            resource_id=spec.resource_id,
            runtime_status=ExecutionStatus.INDETERMINATE_AFTER_CRASH,
            primitive_receipts=primitives,
            possibly_affected_revisions=spec.possibly_affected_revisions,
            started_at=attempt.started_at,
            finished_at=finished_at,
            abort_reason=abort_reason,
            feasibility_certificate_digest=(
                attempt.feasibility_certificate.content_digest
                if attempt.feasibility_certificate
                else None
            ),
            monitor_digest=attempt.monitor_digest,
        )
        self.wal.append(receipt)
        with self._lock:
            self._blocked.add((spec.world_id, spec.resource_id))
        return receipt


def _backend_descriptor(backend: ActionBackend) -> BackendDescriptor:
    descriptor = backend.descriptor
    if isinstance(descriptor, BackendDescriptor):
        return BackendDescriptor.model_validate(descriptor.model_dump(mode="python"))
    return BackendDescriptor.model_validate(descriptor)


def _monitor_json_value(value: object, *, path: str = "sample") -> object:
    """Normalize telemetry to finite JSON without stringifying unsafe values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AdmissionRejectedError(f"{path} contains NaN or infinity")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise AdmissionRejectedError(f"{path} contains a non-string key")
            normalized[key] = _monitor_json_value(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _monitor_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise AdmissionRejectedError(
        f"{path} contains non-JSON telemetry type {type(value).__name__}"
    )


def _normalize_monitor_sample(sample: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(sample, Mapping):
        raise AdmissionRejectedError("monitor backend sample must be a mapping")
    normalized = _monitor_json_value(sample)
    assert isinstance(normalized, dict)
    return normalized


def validate_monitor_backend_compatibility(
    *,
    backend: ActionBackend,
    spec: ActionSpec,
    monitor: MonitorRuntime,
) -> MonitorTelemetryCapabilities:
    """Fail before WAL/effects when a sealed monitor cannot be upheld.

    Capability declarations alone are insufficient: one read-only preflight
    sample verifies that the provider is currently fresh and actually returns
    every signal it claims.  The preflight does not call ``MonitorRuntime`` and
    therefore does not consume debounce state or a logical monitor sequence.
    """

    if not isinstance(backend, MonitorTelemetryBackend):
        raise AdmissionRejectedError(
            "backend does not expose a typed monitor telemetry contract"
        )
    program = getattr(monitor, "program", None)
    program_spec = getattr(program, "spec", None)
    raw_hook = getattr(getattr(program_spec, "hook", None), "value", None)
    raw_signals = getattr(program_spec, "allowed_signals", None)
    try:
        hook = MonitorTelemetryHook(str(raw_hook))
    except ValueError as exc:
        raise AdmissionRejectedError("monitor declares an unknown telemetry hook") from exc
    if not isinstance(raw_signals, tuple) or not raw_signals:
        raise AdmissionRejectedError("monitor declares no telemetry signals")
    required_signals = frozenset(str(name) for name in raw_signals)
    if len(required_signals) != len(raw_signals) or not all(required_signals):
        raise AdmissionRejectedError("monitor signal declaration is not an exact set")
    try:
        raw_capabilities = backend.monitor_telemetry_capabilities(
            world_id=spec.world_id,
            resource_id=spec.resource_id,
        )
        capabilities = MonitorTelemetryCapabilities.model_validate(
            raw_capabilities.model_dump(mode="python")
            if isinstance(raw_capabilities, MonitorTelemetryCapabilities)
            else raw_capabilities
        )
    except Exception as exc:
        raise AdmissionRejectedError(
            f"monitor capability lookup failed: {type(exc).__name__}"
        ) from exc
    if (capabilities.world_id, capabilities.resource_id) != (
        spec.world_id,
        spec.resource_id,
    ):
        raise AdmissionRejectedError(
            "monitor capabilities belong to another world/resource"
        )
    if hook not in capabilities.supported_hooks:
        raise AdmissionRejectedError(
            f"backend does not support monitor hook {hook.value!r}"
        )
    missing_declared = required_signals.difference(
        capabilities.always_available_signals
    )
    if missing_declared:
        raise AdmissionRejectedError(
            "backend does not guarantee monitor signals: "
            + ", ".join(sorted(missing_declared))
        )
    if hook in {MonitorTelemetryHook.WAYPOINT, MonitorTelemetryHook.CONTROL}:
        if not isinstance(spec, MotionPlan):
            raise AdmissionRejectedError(
                f"{hook.value} monitoring requires an exact motion plan"
            )
        if not isinstance(backend, CooperativeMotionBackend):
            raise AdmissionRejectedError(
                "backend lacks cooperative exact-path execution"
            )
        if not isinstance(backend, WatchdogControllableBackend):
            raise AdmissionRejectedError(
                "cooperative monitoring requires a runtime-owned stop surface"
            )
        if not capabilities.cooperative_stop_guaranteed:
            raise AdmissionRejectedError(
                "backend does not guarantee callback-triggered physical stop"
            )
    try:
        raw_sample = backend.monitor_sample(
            world_id=spec.world_id,
            resource_id=spec.resource_id,
            phase="compatibility_preflight",
            sequence=0,
        )
        sample = _normalize_monitor_sample(raw_sample)
    except AdmissionRejectedError:
        raise
    except Exception as exc:
        raise AdmissionRejectedError(
            f"monitor telemetry preflight failed: {type(exc).__name__}:{exc}"
        ) from exc
    missing_actual = set(capabilities.always_available_signals).difference(sample)
    if missing_actual:
        raise AdmissionRejectedError(
            "monitor telemetry preflight omitted guaranteed signals: "
            + ", ".join(sorted(missing_actual))
        )
    return capabilities


def _primitive_call(
    spec: ActionSpec,
) -> tuple[str, dict[str, object]]:
    if isinstance(spec, MotionPlan):
        policy = spec.motion.execution_policy
        return (
            "execute_joint_path",
            {
                "world_id": spec.world_id,
                "resource_id": spec.resource_id,
                "joint_names": spec.motion.joint_names,
                "positions_rad": spec.motion.positions_rad,
                "mode": policy.mode,
                "subsample": policy.subsample,
                "timeout_s": policy.timeout_s,
            },
        )
    if isinstance(spec, GripperCommand):
        return (
            "set_gripper",
            {
                "world_id": spec.world_id,
                "resource_id": spec.resource_id,
                "mode": spec.mode.value,
                "target_width_m": spec.target_width_m,
                "max_effort_n": spec.max_effort_n,
                "timeout_s": spec.timeout_s,
            },
        )
    if isinstance(spec, WaitSpec):
        return (
            "wait",
            {
                "world_id": spec.world_id,
                "resource_id": spec.resource_id,
                "duration_s": spec.duration_s,
                "control_steps": spec.control_steps,
                "hold_command": spec.hold_command,
                "timeout_s": spec.timeout_s,
            },
        )
    raise TypeError(f"unsupported action spec: {type(spec)!r}")


def _dispatch_backend(
    backend: ActionBackend,
    primitive: str,
    kwargs: dict[str, object],
) -> BackendCallResult:
    if primitive == "execute_joint_path":
        result = backend.execute_joint_path(**kwargs)  # type: ignore[arg-type]
    elif primitive == "set_gripper":
        result = backend.set_gripper(**kwargs)  # type: ignore[arg-type]
    elif primitive == "wait":
        result = backend.wait(**kwargs)  # type: ignore[arg-type]
    else:  # pragma: no cover - closed by _primitive_call
        raise TypeError(f"unknown primitive {primitive!r}")
    if not isinstance(result, BackendCallResult):
        raise TypeError("backend must return BackendCallResult, never a receipt")
    return BackendCallResult.model_validate(result.model_dump(mode="python"))


def _call_with_runtime_watchdog(
    call: Callable[[], BackendCallResult],
    *,
    timeout_s: float,
    backend: ActionBackend,
    world_id: str,
    resource_id: str,
) -> BackendCallResult:
    """Bound a trusted backend call even if its own timeout implementation hangs.

    The worker is daemonized because Python cannot safely kill a controller
    thread.  On timeout the caller quarantines the resource; a later explicit
    recovery acknowledgement must prove the controller is quiescent before
    another action is admitted.
    """

    results: queue.Queue[tuple[bool, BackendCallResult | BaseException]] = queue.Queue(
        maxsize=1
    )

    def invoke() -> None:
        try:
            results.put((True, call()))
        except BaseException as exc:  # propagate in the authoritative caller
            results.put((False, exc))

    worker = threading.Thread(
        target=invoke,
        name="robomex-action-watchdog",
        daemon=True,
    )
    worker.start()
    try:
        succeeded, value = results.get(timeout=timeout_s)
    except queue.Empty:
        try:
            watchdog_stop_thread_safe = _backend_descriptor(
                backend
            ).watchdog_stop_thread_safe
        except Exception:  # descriptor failure after an effect must stay fail closed
            watchdog_stop_thread_safe = False
        telemetry: dict[str, object] = {
            "runtime_watchdog_timeout": True,
            "controller_stop_supported": isinstance(
                backend, WatchdogControllableBackend
            ),
            "controller_stop_thread_safe": watchdog_stop_thread_safe,
            "controller_stop_attempted": False,
            "controller_stop_confirmed": False,
            "backend_call_terminated": False,
        }
        recovery_timeout_s = min(5.0, max(0.1, timeout_s * 0.25))
        # The timed-out backend worker may still be inside MuJoCo/LIBERO.  Do
        # not touch that environment concurrently unless the trusted adapter
        # explicitly advertises cross-thread stop safety.  Quarantine and
        # later same-owner recovery is safer than inventing thread safety.
        if (
            isinstance(backend, WatchdogControllableBackend)
            and watchdog_stop_thread_safe
        ):
            telemetry["controller_stop_attempted"] = True
            stop_results: queue.Queue[
                tuple[bool, AdmissionSnapshot | BaseException]
            ] = queue.Queue(maxsize=1)

            def request_stop() -> None:
                try:
                    stop_results.put((
                        True,
                        backend.stop_and_wait_quiescent(
                            world_id=world_id,
                            resource_id=resource_id,
                            timeout_s=recovery_timeout_s,
                        ),
                    ))
                except BaseException as exc:
                    stop_results.put((False, exc))

            stop_worker = threading.Thread(
                target=request_stop,
                name="robomex-controller-stop",
                daemon=True,
            )
            stop_worker.start()
            try:
                stop_succeeded, stop_value = stop_results.get(
                    timeout=recovery_timeout_s
                )
            except queue.Empty:
                telemetry["controller_stop_error"] = "stop_ack_timeout"
            else:
                if stop_succeeded:
                    try:
                        snapshot = _fresh_snapshot(cast(AdmissionSnapshot, stop_value))
                        if (
                            snapshot.world_id != world_id
                            or snapshot.resource_id != resource_id
                            or snapshot.controller_state is not ControllerState.QUIESCENT
                        ):
                            raise AdmissionRejectedError(
                                "controller stop returned a non-quiescent or foreign snapshot"
                            )
                    except Exception as exc:
                        telemetry["controller_stop_error"] = (
                            f"{type(exc).__name__}:{exc}"
                        )
                    else:
                        telemetry["controller_stop_confirmed"] = True
                        telemetry["quiescent_snapshot_digest"] = _snapshot_digest(snapshot)
                else:
                    assert isinstance(stop_value, BaseException)
                    telemetry["controller_stop_error"] = (
                        f"{type(stop_value).__name__}:{stop_value}"
                    )
        worker.join(timeout=recovery_timeout_s)
        telemetry["backend_call_terminated"] = not worker.is_alive()
        return BackendCallResult(
            converged=False,
            timed_out=True,
            telemetry=cast(dict[str, Any], telemetry),
        )
    if not succeeded:
        assert isinstance(value, BaseException)
        raise value
    assert isinstance(value, BackendCallResult)
    return value


class SealedActionRunner:
    """Deterministic single-lane executor for admitted authoritative effects."""

    def __init__(
        self,
        supervisor: ActionSupervisor,
        backend: ActionBackend,
    ) -> None:
        self.supervisor = supervisor
        self.backend = backend

    def run(
        self,
        admitted: AdmittedAction,
        *,
        supplied_spec: ActionSpec | None = None,
        monitor: MonitorRuntime | None = None,
        finding_sink: Callable[[Any], None] | None = None,
        evidence_recorder: ActionEvidenceRecorder | None = None,
    ) -> ExecutionReceipt:
        spec_input = supplied_spec or admitted.action_spec
        attempt_written = False
        terminal_written = False
        # Monitor sequence is the monitor program's logical clock. Evidence
        # sequence orders every runtime and monitor frame, so the two domains
        # must never share a counter.
        monitor_sequence = 0
        evidence_sequence = 0
        evidence_failed = False
        triggering_finding_id: str | None = None
        recorded_frames: list[str] = []

        if monitor is None and admitted.monitor_digest is not None:
            raise SealedActionMismatchError("admitted monitor program was not supplied")
        if monitor is not None:
            actual_digest = getattr(getattr(monitor, "program", None), "digest", None)
            if admitted.monitor_digest is None or actual_digest != admitted.monitor_digest:
                raise SealedActionMismatchError("monitor digest differs from admission")

        def record_evidence(*, phase: str, sample: Mapping[str, object]) -> None:
            nonlocal evidence_sequence, evidence_failed
            if evidence_recorder is None:
                return
            next_sequence = evidence_sequence + 1
            try:
                frame_ref = evidence_recorder.record(
                    action_id=admitted.action_id,
                    sequence=next_sequence,
                    phase=phase,
                    sample=sample,
                )
            except Exception:
                evidence_failed = True
                raise
            evidence_sequence = next_sequence
            if frame_ref:
                recorded_frames.append(frame_ref)

        def evaluate_monitor(sample: Mapping[str, object], *, phase: str) -> MonitorEvaluation:
            nonlocal monitor_sequence, triggering_finding_id
            if monitor is None:  # pragma: no cover - guarded by callers
                raise RuntimeError("monitor is unavailable")
            monitor_sequence += 1
            normalized_sample = _normalize_monitor_sample(sample)
            # Evidence keeps the safe raw superset (joint errors, waypoint
            # indices, camera refs, ...), while untrusted authored code sees
            # only its exact declared capability.  Missing declared fields are
            # intentionally left missing so MonitorRuntime emits a critical
            # UNOBSERVABLE finding instead of receiving a guessed default.
            record_evidence(phase=f"monitor/{phase}", sample=normalized_sample)
            declared = monitor.program.spec.allowed_signals
            projected_sample = {
                name: normalized_sample[name]
                for name in declared
                if name in normalized_sample
            }
            evaluation = monitor.evaluate(
                projected_sample,
                sequence=monitor_sequence,
                hook=monitor.program.spec.hook,
                action_id=admitted.action_id,
                plan_digest=admitted.spec_digest,
            )
            if evaluation.finding is not None:
                triggering_finding_id = evaluation.finding.finding_id
                if finding_sink is not None:
                    finding_sink(evaluation.finding)
            return evaluation

        def finalize_evidence() -> tuple[tuple[str, ...], str | None]:
            if evidence_recorder is None:
                return tuple(recorded_frames), None
            final_frames, video_ref = evidence_recorder.finalize(
                action_id=admitted.action_id
            )
            merged = tuple(dict.fromkeys((*recorded_frames, *final_frames)))
            return merged, video_ref

        try:
            descriptor = _backend_descriptor(self.backend)
            sealed_preview = validate_action_spec(spec_input)
            if (
                isinstance(sealed_preview, MotionPlan)
                and descriptor.motion_interface is not BackendMotionInterface.EXACT_JOINT_PATH
            ):
                raise SealedActionMismatchError(
                    "backend would reinterpret motion instead of executing the exact joint path"
                )
            if monitor is not None:
                validate_monitor_backend_compatibility(
                    backend=self.backend,
                    spec=sealed_preview,
                    monitor=monitor,
                )
            current = self.backend.snapshot(
                admitted.action_spec.world_id,
                admitted.action_spec.resource_id,
            )
            sealed = self.supervisor.begin_execution(admitted, sealed_preview, current)
            attempt = ActionAttempt(
                action_id=admitted.action_id,
                admission_id=admitted.admission_id,
                action_spec=sealed,
                spec_digest=sealed.content_digest,
                admission_snapshot=admitted.admission_snapshot,
                admission_snapshot_digest=admitted.admission_snapshot_digest,
                execution_snapshot=current,
                execution_snapshot_digest=_snapshot_digest(current),
                feasibility_certificate=admitted.feasibility_certificate,
                monitor_digest=admitted.monitor_digest,
                continuation_id=admitted.continuation_id,
                scheduler_reservation_id=admitted.scheduler_reservation_id,
            )
            # This append/flush must complete before the first backend call.
            self.supervisor.wal.append(attempt)
            attempt_written = True

            primitive, kwargs = _primitive_call(sealed)
            args_digest = canonical_payload_digest(
                "robomex.primitive_args.v1", kwargs
            )
            record_evidence(
                phase="execution/pre_primitive",
                sample=ActionRuntimeEvidenceSample(
                    sample_type="execution_pre_primitive",
                    action_id=admitted.action_id,
                    spec_digest=sealed.content_digest,
                    world_id=sealed.world_id,
                    resource_id=sealed.resource_id,
                    primitive=primitive,
                    primitive_args_digest=args_digest,
                    execution_snapshot_digest=_snapshot_digest(current),
                    telemetry={"monitor_configured": monitor is not None},
                ).model_dump(mode="json"),
            )

            monitor_hook = (
                getattr(monitor.program.spec.hook, "value", "")
                if monitor is not None
                else ""
            )
            if monitor is not None and monitor_hook == "phase":
                if isinstance(self.backend, MonitorSampleBackend):
                    sample = self.backend.monitor_sample(
                        world_id=sealed.world_id,
                        resource_id=sealed.resource_id,
                        phase="pre_primitive",
                        sequence=monitor_sequence + 1,
                    )
                else:
                    sample = {}
                evaluation = evaluate_monitor(sample, phase="pre_primitive")
                if evaluation.stop_requested:
                    record_evidence(
                        phase="execution/terminal",
                        sample=ActionRuntimeEvidenceSample(
                            sample_type="execution_terminal",
                            action_id=admitted.action_id,
                            spec_digest=sealed.content_digest,
                            world_id=sealed.world_id,
                            resource_id=sealed.resource_id,
                            primitive=primitive,
                            primitive_args_digest=args_digest,
                            runtime_status=ExecutionStatus.INTERRUPTED.value,
                            abort_reason="pre_primitive_monitor_interrupt",
                        ).model_dump(mode="json"),
                    )
                    frame_refs, video_ref = finalize_evidence()
                    now = _utc_now()
                    receipt = ExecutionReceipt(
                        action_id=admitted.action_id,
                        spec_type=ActionSpecType(sealed.spec_type),
                        spec_id=action_spec_id(sealed),
                        spec_digest=sealed.content_digest,
                        world_id=sealed.world_id,
                        resource_id=sealed.resource_id,
                        runtime_status=ExecutionStatus.INTERRUPTED,
                        possibly_affected_revisions=sealed.possibly_affected_revisions,
                        started_at=attempt.started_at,
                        finished_at=now,
                        abort_reason="pre_primitive_monitor_interrupt",
                        feasibility_certificate_digest=(
                            admitted.feasibility_certificate.content_digest
                            if admitted.feasibility_certificate
                            else None
                        ),
                        monitor_digest=admitted.monitor_digest,
                        triggering_finding_id=triggering_finding_id,
                        frame_refs=frame_refs,
                        video_ref=video_ref,
                    )
                    self.supervisor.wal.append(receipt)
                    terminal_written = True
                    return receipt

            primitive_started = _utc_now()
            try:
                if monitor is not None and monitor_hook in {"waypoint", "control"}:
                    if primitive != "execute_joint_path" or not isinstance(
                        self.backend, CooperativeMotionBackend
                    ):
                        evaluation = evaluate_monitor({}, phase="unsupported_hook")
                        if not evaluation.stop_requested:  # pragma: no cover - fail-closed monitor
                            raise RuntimeError("unsupported monitor hook did not fail closed")
                        record_evidence(
                            phase="execution/terminal",
                            sample=ActionRuntimeEvidenceSample(
                                sample_type="execution_terminal",
                                action_id=admitted.action_id,
                                spec_digest=sealed.content_digest,
                                world_id=sealed.world_id,
                                resource_id=sealed.resource_id,
                                primitive=primitive,
                                primitive_args_digest=args_digest,
                                runtime_status=ExecutionStatus.INTERRUPTED.value,
                                abort_reason=(
                                    "cooperative_monitor_backend_unavailable"
                                ),
                            ).model_dump(mode="json"),
                        )
                        frame_refs, video_ref = finalize_evidence()
                        receipt = ExecutionReceipt(
                            action_id=admitted.action_id,
                            spec_type=ActionSpecType(sealed.spec_type),
                            spec_id=action_spec_id(sealed),
                            spec_digest=sealed.content_digest,
                            world_id=sealed.world_id,
                            resource_id=sealed.resource_id,
                            runtime_status=ExecutionStatus.INTERRUPTED,
                            possibly_affected_revisions=sealed.possibly_affected_revisions,
                            started_at=attempt.started_at,
                            finished_at=_utc_now(),
                            abort_reason="cooperative_monitor_backend_unavailable",
                            feasibility_certificate_digest=(
                                admitted.feasibility_certificate.content_digest
                                if admitted.feasibility_certificate
                                else None
                            ),
                            monitor_digest=admitted.monitor_digest,
                            triggering_finding_id=triggering_finding_id,
                            frame_refs=frame_refs,
                            video_ref=video_ref,
                        )
                        self.supervisor.wal.append(receipt)
                        terminal_written = True
                        return receipt
                    else:
                        stop_requested = False

                        def progress_callback(sample: Mapping[str, object]) -> bool:
                            nonlocal stop_requested
                            if not execution_open.is_set():
                                return False
                            evaluation = evaluate_monitor(sample, phase=monitor_hook)
                            stop_requested = stop_requested or evaluation.stop_requested
                            return not stop_requested

                        execution_open = threading.Event()
                        execution_open.set()
                        try:
                            result = _call_with_runtime_watchdog(
                                lambda: self.backend.execute_joint_path_cooperative(
                                    progress_callback=progress_callback,
                                    **kwargs,
                                ),
                                timeout_s=float(kwargs["timeout_s"]),
                                backend=self.backend,
                                world_id=sealed.world_id,
                                resource_id=sealed.resource_id,
                            )
                        finally:
                            execution_open.clear()
                        if not isinstance(result, BackendCallResult):
                            raise TypeError(
                                "cooperative backend must return BackendCallResult"
                            )
                        result = BackendCallResult.model_validate(
                            result.model_dump(mode="python")
                        )
                        if monitor_sequence == 0:
                            evaluation = evaluate_monitor({}, phase="missing_progress")
                            stop_requested = stop_requested or evaluation.stop_requested
                        if stop_requested and not result.interrupted:
                            # The callback requested a physical stop, yet the
                            # backend returned without acknowledging it.  The
                            # path may already have run to completion; rewriting
                            # that result as an ordinary interruption would be a
                            # false receipt.  Attempt a runtime-owned stop now
                            # that the backend call has returned, then quarantine
                            # the resource regardless of acknowledgement.
                            violation_telemetry: dict[str, Any] = {
                                **result.telemetry,
                                "cooperative_protocol_violation": True,
                                "callback_stop_requested": True,
                                "backend_reported_interrupted": False,
                                "controller_stop_attempted": True,
                                "controller_stop_confirmed": False,
                            }
                            try:
                                stop_backend = cast(
                                    WatchdogControllableBackend,
                                    self.backend,
                                )
                                stop_snapshot = stop_backend.stop_and_wait_quiescent(
                                    world_id=sealed.world_id,
                                    resource_id=sealed.resource_id,
                                    timeout_s=min(
                                        2.0,
                                        max(0.1, float(kwargs["timeout_s"]) * 0.25),
                                    ),
                                )
                                stop_snapshot = _fresh_snapshot(stop_snapshot)
                                if (
                                    stop_snapshot.world_id != sealed.world_id
                                    or stop_snapshot.resource_id != sealed.resource_id
                                    or stop_snapshot.controller_state
                                    is not ControllerState.QUIESCENT
                                ):
                                    raise AdmissionRejectedError(
                                        "cooperative violation stop did not prove quiescence"
                                    )
                            except Exception as exc:
                                violation_telemetry["controller_stop_error"] = (
                                    f"{type(exc).__name__}:{exc}"
                                )
                            else:
                                violation_telemetry["controller_stop_confirmed"] = True
                                violation_telemetry["quiescent_snapshot_digest"] = (
                                    _snapshot_digest(stop_snapshot)
                                )
                            result = BackendCallResult(
                                converged=False,
                                interrupted=False,
                                timed_out=True,
                                telemetry=violation_telemetry,
                            )
                else:
                    result = _call_with_runtime_watchdog(
                        lambda: _dispatch_backend(self.backend, primitive, kwargs),
                        timeout_s=float(kwargs["timeout_s"]),
                        backend=self.backend,
                        world_id=sealed.world_id,
                        resource_id=sealed.resource_id,
                    )
            except Exception as exc:
                primitive_finished = _utc_now()
                primitive_receipt = PrimitiveReceipt(
                    action_id=admitted.action_id,
                    primitive_index=0,
                    primitive=primitive,
                    exact_args_digest=args_digest,
                    status=PrimitiveStatus.RAISED,
                    started_at=primitive_started,
                    finished_at=primitive_finished,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                self.supervisor.wal.append(primitive_receipt)
                record_evidence(
                    phase="primitive/exception",
                    sample=ActionRuntimeEvidenceSample(
                        sample_type="primitive_exception",
                        action_id=admitted.action_id,
                        spec_digest=sealed.content_digest,
                        world_id=sealed.world_id,
                        resource_id=sealed.resource_id,
                        primitive=primitive,
                        primitive_args_digest=args_digest,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    ).model_dump(mode="json"),
                )
                record_evidence(
                    phase="execution/terminal",
                    sample=ActionRuntimeEvidenceSample(
                        sample_type="execution_terminal",
                        action_id=admitted.action_id,
                        spec_digest=sealed.content_digest,
                        world_id=sealed.world_id,
                        resource_id=sealed.resource_id,
                        primitive=primitive,
                        primitive_args_digest=args_digest,
                        runtime_status=ExecutionStatus.INDETERMINATE_AFTER_CRASH.value,
                        abort_reason=f"primitive_exception:{type(exc).__name__}",
                    ).model_dump(mode="json"),
                )
                frame_refs, video_ref = finalize_evidence()
                receipt = ExecutionReceipt(
                    action_id=admitted.action_id,
                    spec_type=ActionSpecType(sealed.spec_type),
                    spec_id=action_spec_id(sealed),
                    spec_digest=sealed.content_digest,
                    world_id=sealed.world_id,
                    resource_id=sealed.resource_id,
                    runtime_status=ExecutionStatus.INDETERMINATE_AFTER_CRASH,
                    primitive_receipts=(primitive_receipt,),
                    possibly_affected_revisions=sealed.possibly_affected_revisions,
                    started_at=attempt.started_at,
                    finished_at=primitive_finished,
                    abort_reason=f"primitive_exception:{type(exc).__name__}",
                    feasibility_certificate_digest=(
                        admitted.feasibility_certificate.content_digest
                        if admitted.feasibility_certificate
                        else None
                    ),
                    monitor_digest=admitted.monitor_digest,
                    triggering_finding_id=triggering_finding_id,
                    frame_refs=frame_refs,
                    video_ref=video_ref,
                )
                self.supervisor.wal.append(receipt)
                self.supervisor._block_unresolved(admitted)
                terminal_written = True
                return receipt

            primitive_finished = _utc_now()
            primitive_receipt = PrimitiveReceipt(
                action_id=admitted.action_id,
                primitive_index=0,
                primitive=primitive,
                exact_args_digest=args_digest,
                status=PrimitiveStatus.RETURNED,
                started_at=primitive_started,
                finished_at=primitive_finished,
                converged=result.converged,
                telemetry=result.telemetry,
            )
            self.supervisor.wal.append(primitive_receipt)
            record_evidence(
                phase="primitive/post",
                sample=ActionRuntimeEvidenceSample(
                    sample_type="primitive_post",
                    action_id=admitted.action_id,
                    spec_digest=sealed.content_digest,
                    world_id=sealed.world_id,
                    resource_id=sealed.resource_id,
                    primitive=primitive,
                    primitive_args_digest=args_digest,
                    converged=result.converged,
                    interrupted=result.interrupted,
                    timed_out=result.timed_out,
                    telemetry=result.telemetry,
                ).model_dump(mode="json"),
            )
            if monitor is not None and monitor_hook == "phase":
                if isinstance(self.backend, MonitorSampleBackend):
                    sample = self.backend.monitor_sample(
                        world_id=sealed.world_id,
                        resource_id=sealed.resource_id,
                        phase="post_primitive",
                        sequence=monitor_sequence + 1,
                    )
                else:
                    sample = {}
                evaluation = evaluate_monitor(sample, phase="post_primitive")
                if evaluation.stop_requested:
                    result = result.model_copy(
                        update={"converged": False, "interrupted": True}
                    )
            convergence_satisfied = (
                result.converged is not False
                if isinstance(sealed, WaitSpec)
                else result.converged is True
            )
            completed = (
                convergence_satisfied and not result.interrupted and not result.timed_out
            )
            if result.timed_out:
                runtime_status = ExecutionStatus.INDETERMINATE_AFTER_TIMEOUT
                abort_reason = "primitive_timeout"
                self.supervisor._block_unresolved(admitted)
            elif result.interrupted:
                runtime_status = ExecutionStatus.INTERRUPTED
                abort_reason = "monitor_or_backend_interrupt"
            elif completed:
                runtime_status = ExecutionStatus.COMPLETED
                abort_reason = None
            else:
                runtime_status = ExecutionStatus.PARTIAL
                abort_reason = "primitive_not_converged"
            record_evidence(
                phase="execution/terminal",
                sample=ActionRuntimeEvidenceSample(
                    sample_type="execution_terminal",
                    action_id=admitted.action_id,
                    spec_digest=sealed.content_digest,
                    world_id=sealed.world_id,
                    resource_id=sealed.resource_id,
                    primitive=primitive,
                    primitive_args_digest=args_digest,
                    runtime_status=runtime_status.value,
                    abort_reason=abort_reason,
                    converged=result.converged,
                    interrupted=result.interrupted,
                    timed_out=result.timed_out,
                    telemetry=result.telemetry,
                ).model_dump(mode="json"),
            )
            frame_refs, video_ref = finalize_evidence()
            receipt = ExecutionReceipt(
                action_id=admitted.action_id,
                spec_type=ActionSpecType(sealed.spec_type),
                spec_id=action_spec_id(sealed),
                spec_digest=sealed.content_digest,
                world_id=sealed.world_id,
                resource_id=sealed.resource_id,
                runtime_status=runtime_status,
                primitive_receipts=(primitive_receipt,),
                possibly_affected_revisions=sealed.possibly_affected_revisions,
                started_at=attempt.started_at,
                finished_at=primitive_finished,
                abort_reason=abort_reason,
                terminal_telemetry=result.telemetry,
                feasibility_certificate_digest=(
                    admitted.feasibility_certificate.content_digest
                    if admitted.feasibility_certificate
                    else None
                ),
                monitor_digest=admitted.monitor_digest,
                triggering_finding_id=triggering_finding_id,
                frame_refs=frame_refs,
                video_ref=video_ref,
            )
            self.supervisor.wal.append(receipt)
            terminal_written = True
            return receipt
        except Exception as exc:
            if not attempt_written:
                raise
            if not evidence_failed:
                try:
                    record_evidence(
                        phase="runner/exception",
                        sample=ActionRuntimeEvidenceSample(
                            sample_type="runner_exception",
                            action_id=admitted.action_id,
                            spec_digest=admitted.spec_digest,
                            world_id=admitted.action_spec.world_id,
                            resource_id=admitted.action_spec.resource_id,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        ).model_dump(mode="json"),
                    )
                    record_evidence(
                        phase="execution/terminal",
                        sample=ActionRuntimeEvidenceSample(
                            sample_type="execution_terminal",
                            action_id=admitted.action_id,
                            spec_digest=admitted.spec_digest,
                            world_id=admitted.action_spec.world_id,
                            resource_id=admitted.action_spec.resource_id,
                            runtime_status=ExecutionStatus.INDETERMINATE_AFTER_CRASH.value,
                            abort_reason=f"runner_exception:{type(exc).__name__}",
                        ).model_dump(mode="json"),
                    )
                    finalize_evidence()
                except Exception:
                    # The authoritative status remains indeterminate; never
                    # reinterpret an evidence failure as successful execution.
                    evidence_failed = True
            receipt = self.supervisor.reconcile_action(
                admitted.action_id,
                abort_reason=f"runner_exception:{type(exc).__name__}:{exc}",
            )
            if receipt is None:  # pragma: no cover - WAL append established the attempt
                raise WalConflictError("attempt disappeared during runner reconciliation") from exc
            terminal_written = True
            return receipt
        finally:
            # BaseException (our crash-fault test) deliberately leaves only the
            # durable attempt.  Reconciliation closes it without replay.
            if attempt_written and not terminal_written:
                self.supervisor._block_unresolved(admitted)
            self.supervisor.release(admitted)


class ShadowRolloutRunner:
    """Exact-spec executor for isolated non-authoritative world clones."""

    def __init__(self, backend: ActionBackend) -> None:
        self.backend = backend
        self._lock = threading.RLock()
        self._active: set[LeaseKey] = set()

    def run(self, spec: ActionSpec, *, candidate_id: str) -> ShadowRolloutReceipt:
        sealed = validate_action_spec(spec)
        if sealed.expected_snapshot.world_kind is not WorldKind.SHADOW:
            raise ShadowIsolationError("authoritative-world specs cannot enter shadow rollout")
        key = (sealed.world_id, sealed.resource_id)
        with self._lock:
            if key in self._active:
                raise ShadowIsolationError(f"shadow world/resource {key!r} already has a writer")
            self._active.add(key)
        started = _utc_now()
        try:
            descriptor = _backend_descriptor(self.backend)
            if (
                isinstance(sealed, MotionPlan)
                and descriptor.motion_interface is not BackendMotionInterface.EXACT_JOINT_PATH
            ):
                raise ShadowIsolationError("shadow backend does not execute exact joint paths")
            current = self.backend.snapshot(sealed.world_id, sealed.resource_id)
            _assert_snapshot_matches(sealed, current)
            primitive, kwargs = _primitive_call(sealed)
            try:
                result = _dispatch_backend(self.backend, primitive, kwargs)
            except Exception as exc:
                return ShadowRolloutReceipt(
                    candidate_id=candidate_id,
                    shadow_world_id=sealed.world_id,
                    resource_id=sealed.resource_id,
                    spec_type=ActionSpecType(sealed.spec_type),
                    spec_id=action_spec_id(sealed),
                    spec_digest=sealed.content_digest,
                    status=ShadowRolloutStatus.FAILED,
                    started_at=started,
                    finished_at=_utc_now(),
                    reason=f"{type(exc).__name__}:{exc}",
                )
            return ShadowRolloutReceipt(
                candidate_id=candidate_id,
                shadow_world_id=sealed.world_id,
                resource_id=sealed.resource_id,
                spec_type=ActionSpecType(sealed.spec_type),
                spec_id=action_spec_id(sealed),
                spec_digest=sealed.content_digest,
                status=ShadowRolloutStatus.COMPLETED,
                started_at=started,
                finished_at=_utc_now(),
                converged=result.converged,
                telemetry=result.telemetry,
            )
        finally:
            with self._lock:
                self._active.discard(key)


__all__ = [
    "ActionAuthorityError",
    "ActionBackend",
    "ActionEvidenceRecorder",
    "ActionLeaseConflictError",
    "ActionSupervisor",
    "ActionWAL",
    "AdmissionRejectedError",
    "AdmittedAction",
    "FeasibilityChecker",
    "InMemoryActionWAL",
    "JsonlActionWAL",
    "CooperativeMotionBackend",
    "ContinuationAlreadyUsedError",
    "MonitorSampleBackend",
    "MonitorTelemetryBackend",
    "RecoveryBlockedError",
    "SealedActionMismatchError",
    "SealedActionRunner",
    "ShadowIsolationError",
    "ShadowRolloutRunner",
    "WalConflictError",
    "WatchdogControllableBackend",
    "build_feasibility_certificate",
    "validate_monitor_backend_compatibility",
]
