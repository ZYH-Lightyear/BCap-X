"""Durable typed event bus for an episode-scoped RoboMEx v2 runtime."""

from __future__ import annotations

import fcntl
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

from robomex.runtime.activation import SchedulerError, TypedEventBus
from robomex.runtime.events import (
    RuntimeEventBase,
    dump_runtime_event,
    parse_runtime_event,
)


class EventLogIntegrityError(SchedulerError):
    """A persisted runtime event is malformed or conflicts with prior history."""


class PersistentTypedEventBus(TypedEventBus):
    """Append each validated event to JSONL before exposing it to subscribers.

    The file is the recovery authority.  Reopening replays its complete typed
    union through the same fail-closed parser used online.  Exact duplicate
    event IDs are idempotent; conflicting reuse is rejected by ``TypedEventBus``.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file_lock = threading.RLock()
        self._process_lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._process_lock_path.touch(exist_ok=True)
        if self.path.exists():
            with self._process_lock(exclusive=False):
                self._load()
        else:
            self.path.touch()

    @contextmanager
    def _process_lock(self, *, exclusive: bool) -> Iterator[None]:
        with self._process_lock_path.open("a+b") as stream:
            fcntl.flock(
                stream.fileno(),
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @property
    def offset(self) -> int:
        return len(self.history)

    def publish(self, event: RuntimeEventBase | dict) -> bool:
        validated = parse_runtime_event(event)
        with self._file_lock, self._process_lock(exclusive=True):
            self._load()
            if self.contains(validated.event_id):
                return super().publish(validated)
            payload = dump_runtime_event(validated)
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            with self.path.open("ab+") as stream:
                stream.seek(0, os.SEEK_END)
                start = stream.tell()
                try:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                except Exception:
                    # A failed fsync is not a committed event.  Best-effort
                    # truncate prevents the same process from later replaying
                    # bytes whose durability was never acknowledged.
                    stream.seek(start)
                    stream.truncate()
                    stream.flush()
                    with suppress(Exception):
                        os.fsync(stream.fileno())
                    raise
            return super().publish(validated)

    def _load(self) -> None:
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                event = parse_runtime_event(line)
                super().publish(event)
            except Exception as exc:
                raise EventLogIntegrityError(
                    f"invalid runtime event at line {line_number}"
                ) from exc


__all__ = ["EventLogIntegrityError", "PersistentTypedEventBus"]
