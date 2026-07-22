"""Durable run-wide budget authority for RoboMEx v2.

Graph ``estimated_budget`` values are planning data.  This module is the
runtime authority: every external operation is durably reserved before the
call and is then either completed (charged) or released (the call was never
entered).  Reservations survive a process crash, so a restart cannot make the
same logical operation free or charge it twice.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from robomex.evolution.manifest import RunBudgets


class RunBudgetError(RuntimeError):
    """Base error for the run-wide budget authority."""


class RunBudgetIntegrityError(RunBudgetError):
    """The append-only ledger is malformed, tampered, or contradictory."""


class RunBudgetIdentityConflictError(RunBudgetError):
    """A ledger or operation identity was rebound to different content."""


class RunBudgetExceededError(RunBudgetError):
    """A reservation would exceed a sealed run budget or wall deadline."""


class RunBudgetOperationStateError(RunBudgetError):
    """An operation attempted an illegal reserve/complete/release transition."""


class RunBudgetVector(BaseModel):
    """All authoritative budget dimensions, with their manifest names removed."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    model_calls: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    wall_time_s: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    physical_actions: int = Field(default=0, ge=0)
    shadow_rollouts: int = Field(default=0, ge=0)
    candidates: int = Field(default=0, ge=0)
    recoveries: int = Field(default=0, ge=0)

    @classmethod
    def from_budgets(cls, budgets: RunBudgets) -> RunBudgetVector:
        return cls(
            model_calls=budgets.max_model_calls,
            tokens=budgets.max_tokens,
            wall_time_s=budgets.max_wall_time_s,
            physical_actions=budgets.max_physical_actions,
            shadow_rollouts=budgets.max_shadow_rollouts,
            candidates=budgets.max_candidates,
            recoveries=budgets.max_recoveries,
        )

    def plus(self, other: RunBudgetVector) -> RunBudgetVector:
        return RunBudgetVector(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in RunBudgetVector.model_fields
            }
        )

    def minus_clamped(self, other: RunBudgetVector) -> RunBudgetVector:
        return RunBudgetVector(
            **{
                name: max(getattr(self, name) - getattr(other, name), 0)
                for name in RunBudgetVector.model_fields
            }
        )

    @property
    def is_zero(self) -> bool:
        return all(getattr(self, name) == 0 for name in RunBudgetVector.model_fields)


class RunBudgetOperationStatus(str, Enum):  # noqa: UP042 - Python 3.10 support
    RESERVED = "reserved"
    COMPLETED = "completed"
    RELEASED = "released"


@dataclass(frozen=True)
class RunBudgetReservation:
    operation_id: str
    binding_digest: str
    requested: RunBudgetVector
    status: RunBudgetOperationStatus
    settlement: RunBudgetVector | None = None


@dataclass(frozen=True)
class RunBudgetSnapshot:
    limits: RunBudgetVector
    committed: RunBudgetVector
    remaining: RunBudgetVector
    started_at_s: float
    deadline_s: float
    observed_at_s: float
    deadline_exhausted: bool
    operations: tuple[RunBudgetReservation, ...]


@dataclass
class _OperationState:
    operation_id: str
    binding_digest: str
    requested: RunBudgetVector
    status: RunBudgetOperationStatus = RunBudgetOperationStatus.RESERVED
    settlement: RunBudgetVector | None = None

    def public(self) -> RunBudgetReservation:
        return RunBudgetReservation(
            operation_id=self.operation_id,
            binding_digest=self.binding_digest,
            requested=self.requested,
            status=self.status,
            settlement=self.settlement,
        )


_SCHEMA_VERSION = "robomex.run_budget_ledger.v1"
_DIGEST_PREFIX = "sha256:"
_MAX_LEDGER_BYTES = 64 * 1024 * 1024


class RunBudgetAuthority:
    """Append-only, process-safe authority bound to one exact ``RunBudgets``.

    ``clock`` is a wall clock (not a process-local monotonic clock) because the
    sealed deadline must remain meaningful after restart.  Tests can inject a
    deterministic clock.  A backwards clock is rejected rather than extending
    the run deadline.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        budgets: RunBudgets,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._budgets = RunBudgets.model_validate(budgets.model_dump(mode="python"))
        self._limits = RunBudgetVector.from_budgets(self._budgets)
        self._clock = clock or time.time
        now = self._now()
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            with os.fdopen(fd, "r+b", closefd=True) as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                records = self._read_records(stream)
                if not records:
                    self._append_record(
                        stream,
                        records,
                        kind="identity",
                        payload={
                            "budgets": self._budgets.model_dump(mode="json"),
                            "budget_digest": self._budget_digest(self._budgets),
                            "started_at_s": now,
                        },
                    )
                    records = self._read_records(stream)
                self._reconstruct(records)
        except Exception:
            # fdopen owns/closed fd once entered; only a pre-fdopen failure can
            # reach here with an open descriptor.
            with suppress(OSError):
                os.close(fd)
            raise
        self._fsync_directory(self.path.parent)

    @property
    def budgets(self) -> RunBudgets:
        return self._budgets

    def now(self) -> float:
        """Return the same validated clock used by deadline enforcement."""

        return self._now()

    def reserve(
        self,
        *,
        operation_id: str,
        requested: RunBudgetVector | Mapping[str, Any],
        binding: str | Mapping[str, Any],
    ) -> RunBudgetReservation:
        operation_id = self._operation_id(operation_id)
        requested = RunBudgetVector.model_validate(requested)
        binding_digest = self.binding_digest(binding)
        now = self._now()
        with self._locked() as (stream, records):
            started_at, operations = self._reconstruct(records)
            self._assert_current_clock(now, records, started_at)
            existing = operations.get(operation_id)
            if existing is not None:
                self._assert_same_operation(existing, binding_digest, requested)
                if existing.status is RunBudgetOperationStatus.RELEASED:
                    raise RunBudgetOperationStateError(
                        f"Budget operation {operation_id!r} was already released."
                    )
                if (
                    existing.status is RunBudgetOperationStatus.RESERVED
                    and now >= started_at + self._limits.wall_time_s
                ):
                    raise RunBudgetExceededError(
                        f"Run wall deadline expired while operation {operation_id!r} was reserved."
                    )
                return existing.public()

            committed = self._committed(operations)
            prospective = committed.plus(requested)
            exceeded = self._exceeded_dimensions(prospective)
            if now < started_at:
                raise RunBudgetIntegrityError(
                    "Run budget clock moved backwards relative to the durable start."
                )
            deadline = started_at + self._limits.wall_time_s
            if now >= deadline:
                exceeded.add("wall_deadline")
            elif requested.wall_time_s > deadline - now:
                exceeded.add("wall_deadline_reservation")
            if exceeded:
                raise RunBudgetExceededError(
                    f"Run budget exhausted for operation {operation_id!r}: "
                    + ", ".join(sorted(exceeded))
                )
            self._append_record(
                stream,
                records,
                kind="reserve",
                payload={
                    "operation_id": operation_id,
                    "binding_digest": binding_digest,
                    "requested": requested.model_dump(mode="json"),
                    "recorded_at_s": now,
                },
            )
            return RunBudgetReservation(
                operation_id=operation_id,
                binding_digest=binding_digest,
                requested=requested,
                status=RunBudgetOperationStatus.RESERVED,
            )

    def complete(
        self,
        operation: str | RunBudgetReservation,
        *,
        settlement: RunBudgetVector | Mapping[str, Any] | None = None,
    ) -> RunBudgetReservation:
        operation_id = self._operation_id(
            operation.operation_id if isinstance(operation, RunBudgetReservation) else operation
        )
        now = self._now()
        with self._locked() as (stream, records):
            started_at, operations = self._reconstruct(records)
            self._assert_current_clock(now, records, started_at)
            state = operations.get(operation_id)
            if state is None:
                raise RunBudgetOperationStateError(f"Unknown budget operation {operation_id!r}.")
            if isinstance(operation, RunBudgetReservation):
                self._assert_same_operation(state, operation.binding_digest, operation.requested)
            resolved = (
                RunBudgetVector.model_validate(settlement)
                if settlement is not None
                else state.requested
            )
            settlement_excess = {
                name
                for name in RunBudgetVector.model_fields
                if getattr(resolved, name) > getattr(state.requested, name)
            }
            if settlement_excess:
                raise RunBudgetOperationStateError(
                    f"Budget operation {operation_id!r} settlement exceeds its "
                    "reservation: " + ", ".join(sorted(settlement_excess))
                )
            if state.status is RunBudgetOperationStatus.COMPLETED:
                if state.settlement != resolved:
                    raise RunBudgetIdentityConflictError(
                        f"Budget operation {operation_id!r} completion was rebound."
                    )
                return state.public()
            if state.status is RunBudgetOperationStatus.RELEASED:
                raise RunBudgetOperationStateError(
                    f"Released budget operation {operation_id!r} cannot complete."
                )
            self._append_record(
                stream,
                records,
                kind="complete",
                payload={
                    "operation_id": operation_id,
                    "settlement": resolved.model_dump(mode="json"),
                    "recorded_at_s": now,
                },
            )
            state.status = RunBudgetOperationStatus.COMPLETED
            state.settlement = resolved
            return state.public()

    def release(self, operation: str | RunBudgetReservation) -> RunBudgetReservation:
        operation_id = self._operation_id(
            operation.operation_id if isinstance(operation, RunBudgetReservation) else operation
        )
        now = self._now()
        with self._locked() as (stream, records):
            started_at, operations = self._reconstruct(records)
            self._assert_current_clock(now, records, started_at)
            state = operations.get(operation_id)
            if state is None:
                raise RunBudgetOperationStateError(f"Unknown budget operation {operation_id!r}.")
            if isinstance(operation, RunBudgetReservation):
                self._assert_same_operation(state, operation.binding_digest, operation.requested)
            if state.status is RunBudgetOperationStatus.RELEASED:
                return state.public()
            if state.status is RunBudgetOperationStatus.COMPLETED:
                raise RunBudgetOperationStateError(
                    f"Completed budget operation {operation_id!r} cannot be released."
                )
            self._append_record(
                stream,
                records,
                kind="release",
                payload={"operation_id": operation_id, "recorded_at_s": now},
            )
            state.status = RunBudgetOperationStatus.RELEASED
            return state.public()

    def operation(self, operation_id: str) -> RunBudgetReservation | None:
        with self._locked() as (_stream, records):
            _started_at, operations = self._reconstruct(records)
            state = operations.get(self._operation_id(operation_id))
            return state.public() if state is not None else None

    def snapshot(self) -> RunBudgetSnapshot:
        now = self._now()
        with self._locked() as (_stream, records):
            started_at, operations = self._reconstruct(records)
        if now < started_at:
            raise RunBudgetIntegrityError(
                "Run budget clock moved backwards relative to the durable start."
            )
        committed = self._committed(operations)
        remaining = self._limits.minus_clamped(committed)
        deadline = started_at + self._limits.wall_time_s
        deadline_remaining = max(deadline - now, 0.0)
        remaining = remaining.model_copy(
            update={"wall_time_s": min(remaining.wall_time_s, deadline_remaining)}
        )
        return RunBudgetSnapshot(
            limits=self._limits,
            committed=committed,
            remaining=remaining,
            started_at_s=started_at,
            deadline_s=deadline,
            observed_at_s=now,
            deadline_exhausted=now >= deadline,
            operations=tuple(operations[key].public() for key in sorted(operations)),
        )

    @staticmethod
    def binding_digest(binding: str | Mapping[str, Any]) -> str:
        if isinstance(binding, str):
            normalized: Any = binding.strip()
            if not normalized:
                raise ValueError("budget operation binding must not be empty")
        elif isinstance(binding, Mapping):
            normalized = dict(binding)
        else:
            raise TypeError("budget operation binding must be a string or mapping")
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return _DIGEST_PREFIX + hashlib.sha256(encoded).hexdigest()

    def _locked(self):
        authority = self

        class _Lock:
            def __enter__(self):
                self.stream = authority.path.open("r+b")
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
                self.records = authority._read_records(self.stream)
                return self.stream, self.records

            def __exit__(self, exc_type, exc, tb):
                try:
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
                finally:
                    self.stream.close()
                return False

        return _Lock()

    def _reconstruct(
        self, records: list[dict[str, Any]]
    ) -> tuple[float, dict[str, _OperationState]]:
        if not records or records[0]["kind"] != "identity":
            raise RunBudgetIntegrityError("Run budget ledger lacks its identity record.")
        identity = records[0]["payload"]
        if set(identity) != {"budgets", "budget_digest", "started_at_s"}:
            raise RunBudgetIntegrityError("Run budget identity has unknown or missing fields.")
        try:
            persisted = RunBudgets.model_validate(identity["budgets"])
            started_at = float(identity["started_at_s"])
            expected_digest = str(identity["budget_digest"])
        except Exception as exc:
            raise RunBudgetIntegrityError("Run budget identity is malformed.") from exc
        if not math.isfinite(started_at):
            raise RunBudgetIntegrityError("Run budget start time is non-finite.")
        if expected_digest != self._budget_digest(persisted):
            raise RunBudgetIntegrityError("Run budget identity digest is invalid.")
        if persisted != self._budgets:
            raise RunBudgetIdentityConflictError(
                "Persisted RunBudgets differ from the sealed run manifest."
            )

        operations: dict[str, _OperationState] = {}
        last_recorded_at = started_at
        for record in records[1:]:
            kind = record["kind"]
            payload = record["payload"]
            if kind == "identity":
                raise RunBudgetIntegrityError("Run budget identity appears more than once.")
            try:
                operation_id = self._operation_id(str(payload["operation_id"]))
            except Exception as exc:
                raise RunBudgetIntegrityError("Malformed budget operation id.") from exc
            try:
                recorded_at = float(payload["recorded_at_s"])
            except Exception as exc:
                raise RunBudgetIntegrityError(
                    "Budget transition lacks a valid record time."
                ) from exc
            if not math.isfinite(recorded_at) or recorded_at < last_recorded_at:
                raise RunBudgetIntegrityError(
                    "Budget transition time moved backwards or is non-finite."
                )
            last_recorded_at = recorded_at
            if kind == "reserve":
                if set(payload) != {
                    "operation_id",
                    "binding_digest",
                    "requested",
                    "recorded_at_s",
                }:
                    raise RunBudgetIntegrityError(
                        "Budget reservation has unknown or missing fields."
                    )
                if operation_id in operations:
                    raise RunBudgetIntegrityError("Duplicate budget reservation record.")
                try:
                    requested = RunBudgetVector.model_validate(payload["requested"])
                    binding_digest = str(payload["binding_digest"])
                except Exception as exc:
                    raise RunBudgetIntegrityError("Malformed budget reservation payload.") from exc
                if not self._valid_digest(binding_digest):
                    raise RunBudgetIntegrityError("Invalid budget binding digest.")
                operations[operation_id] = _OperationState(
                    operation_id=operation_id,
                    binding_digest=binding_digest,
                    requested=requested,
                )
                continue
            state = operations.get(operation_id)
            if state is None:
                raise RunBudgetIntegrityError("Budget transition references no reservation.")
            if kind == "complete":
                if set(payload) != {
                    "operation_id",
                    "settlement",
                    "recorded_at_s",
                }:
                    raise RunBudgetIntegrityError(
                        "Budget completion has unknown or missing fields."
                    )
                if state.status is not RunBudgetOperationStatus.RESERVED:
                    raise RunBudgetIntegrityError("Duplicate/illegal budget completion.")
                try:
                    state.settlement = RunBudgetVector.model_validate(payload["settlement"])
                except Exception as exc:
                    raise RunBudgetIntegrityError("Malformed budget settlement payload.") from exc
                if any(
                    getattr(state.settlement, name) > getattr(state.requested, name)
                    for name in RunBudgetVector.model_fields
                ):
                    raise RunBudgetIntegrityError(
                        "Budget settlement exceeds its durable reservation."
                    )
                state.status = RunBudgetOperationStatus.COMPLETED
            elif kind == "release":
                if set(payload) != {"operation_id", "recorded_at_s"}:
                    raise RunBudgetIntegrityError("Budget release has unknown or missing fields.")
                if state.status is not RunBudgetOperationStatus.RESERVED:
                    raise RunBudgetIntegrityError("Duplicate/illegal budget release.")
                state.status = RunBudgetOperationStatus.RELEASED
            else:
                raise RunBudgetIntegrityError(f"Unknown run budget record kind {kind!r}.")
        return started_at, operations

    def _committed(self, operations: Mapping[str, _OperationState]) -> RunBudgetVector:
        total = RunBudgetVector()
        for state in operations.values():
            if state.status is RunBudgetOperationStatus.RELEASED:
                continue
            amount = state.settlement or state.requested
            total = total.plus(amount)
        return total

    def _exceeded_dimensions(self, vector: RunBudgetVector) -> set[str]:
        return {
            name
            for name in RunBudgetVector.model_fields
            if getattr(vector, name) > getattr(self._limits, name)
        }

    @staticmethod
    def _assert_current_clock(now: float, records: list[dict[str, Any]], started_at: float) -> None:
        last_recorded_at = (
            started_at if len(records) == 1 else float(records[-1]["payload"]["recorded_at_s"])
        )
        if now < last_recorded_at:
            raise RunBudgetIntegrityError(
                "Run budget clock moved backwards relative to the durable ledger."
            )

    @staticmethod
    def _assert_same_operation(
        state: _OperationState,
        binding_digest: str,
        requested: RunBudgetVector,
    ) -> None:
        if state.binding_digest != binding_digest or state.requested != requested:
            raise RunBudgetIdentityConflictError(
                f"Budget operation {state.operation_id!r} was rebound to different content."
            )

    def _read_records(self, stream) -> list[dict[str, Any]]:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        if size > _MAX_LEDGER_BYTES:
            raise RunBudgetIntegrityError("Run budget ledger exceeds its size limit.")
        stream.seek(0)
        raw = stream.read()
        if raw and not raw.endswith(b"\n"):
            raise RunBudgetIntegrityError("Run budget ledger ends with an incomplete record.")
        records: list[dict[str, Any]] = []
        previous: str | None = None
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                raise RunBudgetIntegrityError(f"Blank run budget record at line {line_number}.")
            try:
                record = json.loads(line)
            except Exception as exc:
                raise RunBudgetIntegrityError(
                    f"Invalid run budget JSON at line {line_number}."
                ) from exc
            if not isinstance(record, dict) or set(record) != {
                "schema_version",
                "sequence",
                "previous_digest",
                "kind",
                "payload",
                "record_digest",
            }:
                raise RunBudgetIntegrityError("Run budget record has unknown/missing fields.")
            if record["schema_version"] != _SCHEMA_VERSION:
                raise RunBudgetIntegrityError("Unknown run budget ledger schema.")
            if record["sequence"] != len(records) + 1:
                raise RunBudgetIntegrityError("Run budget sequence is not contiguous.")
            if record["previous_digest"] != previous:
                raise RunBudgetIntegrityError("Run budget hash chain is broken.")
            digest = self._record_digest(record)
            if record["record_digest"] != digest:
                raise RunBudgetIntegrityError("Run budget record digest is invalid.")
            if not isinstance(record["payload"], dict):
                raise RunBudgetIntegrityError("Run budget payload must be an object.")
            records.append(record)
            previous = digest
        return records

    def _append_record(
        self,
        stream,
        records: list[dict[str, Any]],
        *,
        kind: str,
        payload: Mapping[str, Any],
    ) -> None:
        record: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "sequence": len(records) + 1,
            "previous_digest": (records[-1]["record_digest"] if records else None),
            "kind": kind,
            "payload": dict(payload),
        }
        record["record_digest"] = self._record_digest(record)
        encoded = (
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        stream.seek(0, os.SEEK_END)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        records.append(record)

    @staticmethod
    def _record_digest(record: Mapping[str, Any]) -> str:
        body = {key: value for key, value in record.items() if key != "record_digest"}
        encoded = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return _DIGEST_PREFIX + hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _budget_digest(budgets: RunBudgets) -> str:
        encoded = json.dumps(
            budgets.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return _DIGEST_PREFIX + hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _operation_id(value: str) -> str:
        normalized = str(value).strip()
        if not normalized or len(normalized) > 512 or "\n" in normalized:
            raise ValueError("operation_id must be a non-empty bounded single line")
        return normalized

    @staticmethod
    def _valid_digest(value: str) -> bool:
        return (
            value.startswith(_DIGEST_PREFIX)
            and len(value) == len(_DIGEST_PREFIX) + 64
            and all(character in "0123456789abcdef" for character in value[7:])
        )

    def _now(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value):
            raise RunBudgetIntegrityError("Run budget clock returned a non-finite value.")
        return value

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


__all__ = [
    "RunBudgetAuthority",
    "RunBudgetError",
    "RunBudgetExceededError",
    "RunBudgetIdentityConflictError",
    "RunBudgetIntegrityError",
    "RunBudgetOperationStateError",
    "RunBudgetOperationStatus",
    "RunBudgetReservation",
    "RunBudgetSnapshot",
    "RunBudgetVector",
]
