"""Durable deterministic delivery for workflow service subscriptions.

The runtime event log is the source of subscribed inputs.  This ledger stores
only delivery control state: the immutable subscription boundary, a fsynced
reservation made before provider invocation, completion, and a contiguous
cursor.  A reservation without a completion is deliberately redelivered with
the same invocation identity after restart.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from robomex.runtime.events import RuntimeEventBase, dump_runtime_event


class ServiceDeliveryIntegrityError(RuntimeError):
    """The append-only service delivery ledger is malformed or forked."""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    record_kind: str


class ServiceSubscriptionRecord(_Record):
    record_kind: Literal["subscription"] = "subscription"
    subscriber_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    workflow_id: str = Field(min_length=1)
    activation_id: str = Field(min_length=1)
    command_id: str = Field(min_length=1)
    graph_revision: int = Field(ge=1)
    kinds: tuple[str, ...] = Field(min_length=1)
    start_offset: int = Field(ge=0)

    @model_validator(mode="after")
    def _canonical_kinds(self) -> ServiceSubscriptionRecord:
        if any(not kind.strip() for kind in self.kinds):
            raise ValueError("subscription kinds must be non-empty")
        if tuple(sorted(set(self.kinds))) != self.kinds:
            raise ValueError("subscription kinds must be unique and sorted")
        return self


class ServiceDeliveryReservation(_Record):
    record_kind: Literal["reservation"] = "reservation"
    delivery_id: str = Field(min_length=1)
    subscriber_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event_kind: str = Field(min_length=1)
    event_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_offset: int = Field(ge=1)
    invocation_id: str = Field(min_length=1)
    artifact_attempt: int = Field(ge=1)


class ServiceDeliveryCompletion(_Record):
    record_kind: Literal["completion"] = "completion"
    delivery_id: str = Field(min_length=1)
    reservation_sequence: int = Field(ge=1)
    status: Literal["succeeded", "failed"]
    result_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_ids: tuple[str, ...] = ()
    emitted_event_ids: tuple[str, ...] = ()
    reason: str = ""


class ServiceDeliveryCursor(_Record):
    record_kind: Literal["cursor"] = "cursor"
    subscriber_id: str = Field(min_length=1)
    through_offset: int = Field(ge=0)


ServiceDeliveryRecord: TypeAlias = (  # noqa: UP040 - Python 3.10 compatibility
    ServiceSubscriptionRecord
    | ServiceDeliveryReservation
    | ServiceDeliveryCompletion
    | ServiceDeliveryCursor
)


def runtime_event_digest(event: RuntimeEventBase) -> str:
    """Return the canonical digest used to bind one delivery reservation."""

    encoded = json.dumps(
        dump_runtime_event(event),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ServiceDeliveryLedger:
    """Episode-local fsync journal implementing at-least-once delivery."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.path.touch(exist_ok=True)
        self.lock_path.touch(exist_ok=True)
        self._lock = threading.RLock()
        self._records: list[ServiceDeliveryRecord] = []
        self._subscriptions: dict[str, ServiceSubscriptionRecord] = {}
        self._reservations: dict[str, ServiceDeliveryReservation] = {}
        self._event_reservations: dict[
            tuple[str, str], ServiceDeliveryReservation
        ] = {}
        self._completions: dict[str, ServiceDeliveryCompletion] = {}
        self._cursors: dict[str, int] = {}
        with self._disk_lock(exclusive=False):
            self._reload_from_disk()

    @property
    def records(self) -> tuple[ServiceDeliveryRecord, ...]:
        with self._lock, self._disk_lock(exclusive=False):
            self._reload_from_disk()
            return tuple(self._records)

    def bind_subscription(
        self,
        *,
        subscriber_id: str,
        episode_id: str,
        workflow_id: str,
        activation_id: str,
        command_id: str,
        graph_revision: int,
        kinds: tuple[str, ...],
        start_offset: int,
    ) -> ServiceSubscriptionRecord:
        canonical_kinds = tuple(sorted({str(kind).strip() for kind in kinds}))
        with self._lock, self._disk_lock(exclusive=True):
            self._reload_from_disk()
            existing = self._subscriptions.get(subscriber_id)
            fields = {
                "subscriber_id": subscriber_id,
                "episode_id": episode_id,
                "workflow_id": workflow_id,
                "activation_id": activation_id,
                "command_id": command_id,
                "graph_revision": graph_revision,
                "kinds": canonical_kinds,
                "start_offset": start_offset,
            }
            if existing is not None:
                expected = existing.model_dump(
                    exclude={"sequence", "record_kind"}, mode="python"
                )
                if expected != fields:
                    raise ServiceDeliveryIntegrityError(
                        "service subscriber identity was rebound to different content"
                    )
                return existing
            record = ServiceSubscriptionRecord(
                sequence=len(self._records) + 1,
                **fields,
            )
            self._append(record)
            return record

    def subscription(self, subscriber_id: str) -> ServiceSubscriptionRecord | None:
        with self._lock, self._disk_lock(exclusive=False):
            self._reload_from_disk()
            return self._subscriptions.get(subscriber_id)

    def cursor(self, subscriber_id: str) -> int:
        with self._lock, self._disk_lock(exclusive=False):
            self._reload_from_disk()
            subscription = self._subscriptions.get(subscriber_id)
            if subscription is None:
                raise ServiceDeliveryIntegrityError("unknown service subscriber")
            return self._cursors.get(subscriber_id, subscription.start_offset)

    def reservation(
        self, subscriber_id: str, event_id: str
    ) -> ServiceDeliveryReservation | None:
        with self._lock, self._disk_lock(exclusive=False):
            self._reload_from_disk()
            return self._event_reservations.get((subscriber_id, event_id))

    def completion(self, delivery_id: str) -> ServiceDeliveryCompletion | None:
        with self._lock, self._disk_lock(exclusive=False):
            self._reload_from_disk()
            return self._completions.get(delivery_id)

    def reserve(
        self,
        *,
        delivery_id: str,
        subscriber_id: str,
        event_id: str,
        event_kind: str,
        event_digest: str,
        event_offset: int,
        invocation_id: str,
        artifact_attempt: int,
    ) -> ServiceDeliveryReservation:
        with self._lock, self._disk_lock(exclusive=True):
            self._reload_from_disk()
            subscription = self._subscriptions.get(subscriber_id)
            if subscription is None:
                raise ServiceDeliveryIntegrityError(
                    "delivery reservation references an unknown subscription"
                )
            fields = {
                "delivery_id": delivery_id,
                "subscriber_id": subscriber_id,
                "event_id": event_id,
                "event_kind": event_kind,
                "event_digest": event_digest,
                "event_offset": event_offset,
                "invocation_id": invocation_id,
                "artifact_attempt": artifact_attempt,
            }
            existing = self._event_reservations.get((subscriber_id, event_id))
            if existing is not None:
                expected = existing.model_dump(
                    exclude={"sequence", "record_kind"}, mode="python"
                )
                if expected != fields:
                    raise ServiceDeliveryIntegrityError(
                        "subscribed event identity or digest changed after reservation"
                    )
                return existing
            if delivery_id in self._reservations:
                raise ServiceDeliveryIntegrityError(
                    "delivery id is already bound to another subscribed event"
                )
            if event_offset <= subscription.start_offset:
                raise ServiceDeliveryIntegrityError(
                    "delivery precedes the durable subscription boundary"
                )
            record = ServiceDeliveryReservation(
                sequence=len(self._records) + 1,
                **fields,
            )
            self._append(record)
            return record

    def complete(
        self,
        *,
        delivery_id: str,
        status: Literal["succeeded", "failed"],
        result_digest: str,
        artifact_ids: tuple[str, ...] = (),
        emitted_event_ids: tuple[str, ...] = (),
        reason: str = "",
    ) -> ServiceDeliveryCompletion:
        with self._lock, self._disk_lock(exclusive=True):
            self._reload_from_disk()
            reservation = self._reservations.get(delivery_id)
            if reservation is None:
                raise ServiceDeliveryIntegrityError(
                    "service completion has no delivery reservation"
                )
            fields = {
                "delivery_id": delivery_id,
                "reservation_sequence": reservation.sequence,
                "status": status,
                "result_digest": result_digest,
                "artifact_ids": artifact_ids,
                "emitted_event_ids": emitted_event_ids,
                "reason": reason,
            }
            existing = self._completions.get(delivery_id)
            if existing is not None:
                expected = existing.model_dump(
                    exclude={"sequence", "record_kind"}, mode="python"
                )
                if expected != fields:
                    raise ServiceDeliveryIntegrityError(
                        "service delivery completion was rebound"
                    )
                return existing
            record = ServiceDeliveryCompletion(
                sequence=len(self._records) + 1,
                **fields,
            )
            self._append(record)
            return record

    def advance_cursor(self, subscriber_id: str, through_offset: int) -> int:
        with self._lock, self._disk_lock(exclusive=True):
            self._reload_from_disk()
            subscription = self._subscriptions.get(subscriber_id)
            if subscription is None:
                raise ServiceDeliveryIntegrityError("unknown service subscriber")
            current = self._cursors.get(subscriber_id, subscription.start_offset)
            if through_offset < current:
                raise ServiceDeliveryIntegrityError(
                    "service delivery cursor cannot move backwards"
                )
            if through_offset == current:
                return current
            incomplete = [
                reservation.delivery_id
                for reservation in self._reservations.values()
                if reservation.subscriber_id == subscriber_id
                and reservation.event_offset <= through_offset
                and reservation.delivery_id not in self._completions
            ]
            if incomplete:
                raise ServiceDeliveryIntegrityError(
                    "service delivery cursor cannot pass an inflight reservation"
                )
            self._append(
                ServiceDeliveryCursor(
                    sequence=len(self._records) + 1,
                    subscriber_id=subscriber_id,
                    through_offset=through_offset,
                )
            )
            return through_offset

    def _append(self, record: ServiceDeliveryRecord) -> None:
        encoded = (
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        with self.path.open("ab+") as stream:
            stream.seek(0, os.SEEK_END)
            start = stream.tell()
            try:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            except Exception:
                stream.seek(start)
                stream.truncate()
                stream.flush()
                with suppress(Exception):
                    os.fsync(stream.fileno())
                raise
        self._apply(record)

    def _reload_from_disk(self) -> None:
        self._records.clear()
        self._subscriptions.clear()
        self._reservations.clear()
        self._event_reservations.clear()
        self._completions.clear()
        self._cursors.clear()
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ServiceDeliveryIntegrityError(
                "service delivery ledger is unreadable"
            ) from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise ServiceDeliveryIntegrityError(
                    f"blank service delivery record at line {line_number}"
                )
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("record is not an object")
                kind = payload.get("record_kind")
                if kind == "subscription":
                    record: ServiceDeliveryRecord = (
                        ServiceSubscriptionRecord.model_validate(payload)
                    )
                elif kind == "reservation":
                    record = ServiceDeliveryReservation.model_validate(payload)
                elif kind == "completion":
                    record = ServiceDeliveryCompletion.model_validate(payload)
                elif kind == "cursor":
                    record = ServiceDeliveryCursor.model_validate(payload)
                else:
                    raise ValueError(f"unknown record kind {kind!r}")
                if record.sequence != len(self._records) + 1:
                    raise ValueError("record sequence is not contiguous")
                self._apply(record)
            except Exception as exc:
                if isinstance(exc, ServiceDeliveryIntegrityError):
                    raise
                raise ServiceDeliveryIntegrityError(
                    f"invalid service delivery record at line {line_number}"
                ) from exc

    def _apply(self, record: ServiceDeliveryRecord) -> None:
        if isinstance(record, ServiceSubscriptionRecord):
            if record.subscriber_id in self._subscriptions:
                raise ServiceDeliveryIntegrityError("duplicate service subscription")
            self._subscriptions[record.subscriber_id] = record
            self._cursors[record.subscriber_id] = record.start_offset
        elif isinstance(record, ServiceDeliveryReservation):
            subscription = self._subscriptions.get(record.subscriber_id)
            if subscription is None:
                raise ServiceDeliveryIntegrityError(
                    "reservation precedes its service subscription"
                )
            if record.event_offset <= subscription.start_offset:
                raise ServiceDeliveryIntegrityError(
                    "reservation precedes subscription start offset"
                )
            key = (record.subscriber_id, record.event_id)
            if record.delivery_id in self._reservations or key in self._event_reservations:
                raise ServiceDeliveryIntegrityError("duplicate service delivery reservation")
            self._reservations[record.delivery_id] = record
            self._event_reservations[key] = record
        elif isinstance(record, ServiceDeliveryCompletion):
            reservation = self._reservations.get(record.delivery_id)
            if (
                reservation is None
                or reservation.sequence != record.reservation_sequence
            ):
                raise ServiceDeliveryIntegrityError(
                    "completion does not bind the exact reservation"
                )
            if record.delivery_id in self._completions:
                raise ServiceDeliveryIntegrityError("duplicate service completion")
            self._completions[record.delivery_id] = record
        else:
            subscription = self._subscriptions.get(record.subscriber_id)
            if subscription is None:
                raise ServiceDeliveryIntegrityError(
                    "cursor precedes its service subscription"
                )
            current = self._cursors.get(
                record.subscriber_id, subscription.start_offset
            )
            if record.through_offset <= current:
                raise ServiceDeliveryIntegrityError(
                    "service cursor records must advance strictly"
                )
            incomplete = [
                reservation.delivery_id
                for reservation in self._reservations.values()
                if reservation.subscriber_id == record.subscriber_id
                and reservation.event_offset <= record.through_offset
                and reservation.delivery_id not in self._completions
            ]
            if incomplete:
                raise ServiceDeliveryIntegrityError(
                    "cursor crosses an incomplete service delivery"
                )
            self._cursors[record.subscriber_id] = record.through_offset
        self._records.append(record)

    @contextmanager
    def _disk_lock(self, *, exclusive: bool) -> Iterator[None]:
        with self.lock_path.open("a+b") as stream:
            fcntl.flock(
                stream.fileno(),
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


__all__ = [
    "ServiceDeliveryCompletion",
    "ServiceDeliveryCursor",
    "ServiceDeliveryIntegrityError",
    "ServiceDeliveryLedger",
    "ServiceDeliveryReservation",
    "ServiceSubscriptionRecord",
    "runtime_event_digest",
]
