"""Process-safe durability primitives for episode-scoped append-only ledgers.

The artifact and embodied-state ledgers deliberately share one episode lock.
This makes manifest identity binding, artifact publication, and state commits
linearizable across processes while preserving separate append-only journals.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robomex.data.artifact_resolver import ArtifactIntegrityError

EPISODE_LOCK_NAME = ".robomex_episode_v2.lock"
LEDGER_HEAD_FORMAT = "robomex.append_only_ledger_head.v1"
LEDGER_PENDING_FORMAT = "robomex.append_only_ledger_pending.v1"
LEDGER_DIGEST_GENESIS = "sha256:" + ("0" * 64)

_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROCESS_LOCKS[key] = lock
        return lock


@contextmanager
def exclusive_episode_lock(episode_root: Path) -> Iterator[None]:
    """Hold a process-wide and cross-process exclusive episode lock.

    ``flock`` alone is not a substitute for an in-process mutex when multiple
    file descriptions are opened by different instances in one process.  The
    path-keyed ``RLock`` closes that gap; ``flock`` supplies the process
    boundary.  Callers must avoid re-entering this context through a second
    data-plane public method while it is held.
    """

    episode_root.mkdir(parents=True, exist_ok=True)
    lock_path = episode_root / EPISODE_LOCK_NAME
    thread_lock = _thread_lock_for(lock_path)
    with thread_lock:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


@dataclass(frozen=True)
class LedgerHead:
    """Durable commitment to a canonical prefix of one JSONL ledger."""

    episode_id: str
    ledger: str
    event_schema: str
    event_count: int
    byte_count: int
    last_event_digest: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": LEDGER_HEAD_FORMAT,
            "episode_id": self.episode_id,
            "ledger": self.ledger,
            "event_schema": self.event_schema,
            "event_count": self.event_count,
            "byte_count": self.byte_count,
            "last_event_digest": self.last_event_digest,
        }


@dataclass(frozen=True)
class LedgerPendingAppend:
    """Durable exact intent for at most one not-yet-committed append."""

    episode_id: str
    ledger: str
    event_schema: str
    previous_head: LedgerHead
    next_head: LedgerHead
    canonical_event: str
    intent_digest: str

    @property
    def encoded_event(self) -> bytes:
        return (self.canonical_event + "\n").encode("utf-8")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": LEDGER_PENDING_FORMAT,
            "episode_id": self.episode_id,
            "ledger": self.ledger,
            "event_schema": self.event_schema,
            "previous_head": self.previous_head.to_mapping(),
            "next_head": self.next_head.to_mapping(),
            "canonical_event": self.canonical_event,
            "intent_digest": self.intent_digest,
        }


@dataclass(frozen=True)
class LedgerReplayInspection:
    """Structurally safe ledger bytes and any pending recovery to finalize."""

    raw: bytes
    head: LedgerHead | None
    pending: LedgerPendingAppend | None


def load_ledger_head(
    path: Path,
    *,
    episode_id: str,
    ledger: str,
    event_schema: str,
    required: bool,
) -> LedgerHead | None:
    if not path.exists():
        if required:
            raise ArtifactIntegrityError(
                f"Durable head for {ledger!r} is missing from an initialized ledger."
            )
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Durable head for {ledger!r} is unreadable.") from exc
    if not isinstance(raw, Mapping):
        raise ArtifactIntegrityError(f"Durable head for {ledger!r} must be a mapping.")
    return _parse_ledger_head(
        raw,
        episode_id=episode_id,
        ledger=ledger,
        event_schema=event_schema,
    )


def _parse_ledger_head(
    raw: Mapping[str, Any],
    *,
    episode_id: str,
    ledger: str,
    event_schema: str,
) -> LedgerHead:
    raw_event_count = raw.get("event_count")
    raw_byte_count = raw.get("byte_count")
    if (
        not isinstance(raw_event_count, int)
        or isinstance(raw_event_count, bool)
        or not isinstance(raw_byte_count, int)
        or isinstance(raw_byte_count, bool)
    ):
        raise ArtifactIntegrityError(
            f"Durable head for {ledger!r} requires integer counts."
        )
    try:
        head = LedgerHead(
            episode_id=str(raw["episode_id"]),
            ledger=str(raw["ledger"]),
            event_schema=str(raw["event_schema"]),
            event_count=int(raw["event_count"]),
            byte_count=int(raw["byte_count"]),
            last_event_digest=str(raw["last_event_digest"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"Durable head for {ledger!r} is malformed.") from exc
    if set(raw) != {
        "schema",
        "episode_id",
        "ledger",
        "event_schema",
        "event_count",
        "byte_count",
        "last_event_digest",
    }:
        raise ArtifactIntegrityError(f"Durable head for {ledger!r} has unknown fields.")
    if (
        raw.get("schema") != LEDGER_HEAD_FORMAT
        or head.episode_id != episode_id
        or head.ledger != ledger
        or head.event_schema != event_schema
        or head.event_count < 0
        or head.byte_count < 0
        or not _is_sha256_digest(head.last_event_digest)
        or (head.event_count == 0 and head.last_event_digest != LEDGER_DIGEST_GENESIS)
    ):
        raise ArtifactIntegrityError(
            f"Durable head for {ledger!r} failed identity or integrity validation."
        )
    return head


def write_ledger_head(path: Path, head: LedgerHead) -> None:
    atomic_write_json(path, head.to_mapping())


def build_pending_append(
    *,
    episode_id: str,
    ledger: str,
    event_schema: str,
    previous_head: LedgerHead,
    canonical_event: str,
) -> LedgerPendingAppend:
    """Build an exact append witness bound to the current durable head."""

    _validate_canonical_event(canonical_event, event_schema=event_schema)
    if (
        previous_head.episode_id != episode_id
        or previous_head.ledger != ledger
        or previous_head.event_schema != event_schema
    ):
        raise ArtifactIntegrityError("Pending append previous head crossed ledger identity.")
    encoded = (canonical_event + "\n").encode("utf-8")
    next_head = LedgerHead(
        episode_id=episode_id,
        ledger=ledger,
        event_schema=event_schema,
        event_count=previous_head.event_count + 1,
        byte_count=previous_head.byte_count + len(encoded),
        last_event_digest=next_ledger_digest(
            previous_head.last_event_digest,
            canonical_event,
        ),
    )
    intent_digest = _pending_intent_digest(
        episode_id=episode_id,
        ledger=ledger,
        event_schema=event_schema,
        previous_head=previous_head,
        next_head=next_head,
        canonical_event=canonical_event,
    )
    return LedgerPendingAppend(
        episode_id=episode_id,
        ledger=ledger,
        event_schema=event_schema,
        previous_head=previous_head,
        next_head=next_head,
        canonical_event=canonical_event,
        intent_digest=intent_digest,
    )


def write_pending_append(path: Path, pending: LedgerPendingAppend) -> None:
    if path.exists():
        raise ArtifactIntegrityError(
            f"Pending append already exists for ledger {pending.ledger!r}."
        )
    atomic_write_json(path, pending.to_mapping())


def load_pending_append(
    path: Path,
    *,
    episode_id: str,
    ledger: str,
    event_schema: str,
) -> LedgerPendingAppend | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(
            f"Pending append for {ledger!r} is unreadable."
        ) from exc
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema",
        "episode_id",
        "ledger",
        "event_schema",
        "previous_head",
        "next_head",
        "canonical_event",
        "intent_digest",
    }:
        raise ArtifactIntegrityError(f"Pending append for {ledger!r} is malformed.")
    if (
        raw.get("schema") != LEDGER_PENDING_FORMAT
        or raw.get("episode_id") != episode_id
        or raw.get("ledger") != ledger
        or raw.get("event_schema") != event_schema
        or not isinstance(raw.get("previous_head"), Mapping)
        or not isinstance(raw.get("next_head"), Mapping)
        or not isinstance(raw.get("canonical_event"), str)
        or not isinstance(raw.get("intent_digest"), str)
    ):
        raise ArtifactIntegrityError(
            f"Pending append for {ledger!r} failed identity validation."
        )
    previous_head = _parse_ledger_head(
        raw["previous_head"],
        episode_id=episode_id,
        ledger=ledger,
        event_schema=event_schema,
    )
    next_head = _parse_ledger_head(
        raw["next_head"],
        episode_id=episode_id,
        ledger=ledger,
        event_schema=event_schema,
    )
    canonical_event = raw["canonical_event"]
    _validate_canonical_event(canonical_event, event_schema=event_schema)
    expected = build_pending_append(
        episode_id=episode_id,
        ledger=ledger,
        event_schema=event_schema,
        previous_head=previous_head,
        canonical_event=canonical_event,
    )
    if next_head != expected.next_head or raw["intent_digest"] != expected.intent_digest:
        raise ArtifactIntegrityError(
            f"Pending append for {ledger!r} failed exact intent integrity checks."
        )
    return LedgerPendingAppend(
        episode_id=episode_id,
        ledger=ledger,
        event_schema=event_schema,
        previous_head=previous_head,
        next_head=next_head,
        canonical_event=canonical_event,
        intent_digest=raw["intent_digest"],
    )


def clear_pending_append(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ArtifactIntegrityError("Pending append could not be cleared.") from exc
    fsync_directory(path.parent)


def inspect_ledger_for_replay(
    *,
    log_path: Path,
    head_path: Path,
    pending_path: Path,
    episode_id: str,
    ledger: str,
    event_schema: str,
    require_head: bool,
) -> LedgerReplayInspection:
    """Validate the committed prefix and reconcile only an exact pending tail.

    The function may finish a byte-prefix append described by a durable intent,
    but it never advances the durable head.  The caller must first replay and
    semantically validate the returned bytes, then call
    :func:`finalize_ledger_recovery`.
    """

    try:
        raw = log_path.read_bytes()
    except OSError as exc:
        raise ArtifactIntegrityError(f"Event log for {ledger!r} is unreadable.") from exc
    head = load_ledger_head(
        head_path,
        episode_id=episode_id,
        ledger=ledger,
        event_schema=event_schema,
        required=require_head,
    )
    pending = load_pending_append(
        pending_path,
        episode_id=episode_id,
        ledger=ledger,
        event_schema=event_schema,
    )
    if head is None:
        if pending is not None:
            raise ArtifactIntegrityError(
                f"Legacy ledger {ledger!r} cannot carry a pending append."
            )
        return LedgerReplayInspection(raw=raw, head=None, pending=None)

    _validate_committed_prefix(raw, head=head)
    suffix = raw[head.byte_count :]
    if pending is None:
        if suffix:
            raise ArtifactIntegrityError(
                f"Ledger {ledger!r} has an uncommitted suffix without a pending intent."
            )
        return LedgerReplayInspection(raw=raw, head=head, pending=None)

    if head == pending.next_head:
        if len(raw) != head.byte_count:
            raise ArtifactIntegrityError(
                f"Committed ledger {ledger!r} has bytes beyond its durable head."
            )
        expected_suffix = pending.encoded_event
        start = pending.previous_head.byte_count
        if raw[start:] != expected_suffix:
            raise ArtifactIntegrityError(
                f"Committed pending append for {ledger!r} does not match the journal."
            )
        return LedgerReplayInspection(raw=raw, head=head, pending=pending)

    if head != pending.previous_head:
        raise ArtifactIntegrityError(
            f"Pending append for {ledger!r} is not based on the durable head."
        )
    if not suffix:
        clear_pending_append(pending_path)
        return LedgerReplayInspection(raw=raw, head=head, pending=None)
    expected = pending.encoded_event
    if len(suffix) > len(expected) or not expected.startswith(suffix):
        raise ArtifactIntegrityError(
            f"Journal suffix for {ledger!r} does not match its pending exact event."
        )
    if len(suffix) < len(expected):
        remainder = expected[len(suffix) :]
        with log_path.open("ab") as stream:
            stream.write(remainder)
            stream.flush()
            os.fsync(stream.fileno())
        raw += remainder
    return LedgerReplayInspection(raw=raw, head=head, pending=pending)


def finalize_ledger_recovery(
    *,
    head_path: Path,
    pending_path: Path,
    inspection: LedgerReplayInspection,
    replayed_head: LedgerHead,
) -> None:
    """Commit or clean a pending append after semantic replay succeeded."""

    pending = inspection.pending
    if pending is None:
        if inspection.head is not None and replayed_head != inspection.head:
            raise ArtifactIntegrityError("Replay advanced a ledger without a pending intent.")
        return
    if replayed_head != pending.next_head:
        raise ArtifactIntegrityError("Pending append replay did not produce its exact next head.")
    if inspection.head == pending.previous_head:
        write_ledger_head(head_path, pending.next_head)
    elif inspection.head != pending.next_head:
        raise ArtifactIntegrityError("Pending append recovery observed an impossible head state.")
    clear_pending_append(pending_path)


def next_ledger_digest(previous: str, canonical_event: str) -> str:
    if not _is_sha256_digest(previous):
        raise ArtifactIntegrityError("Previous ledger digest is not canonical SHA-256.")
    digest = hashlib.sha256(
        b"robomex-ledger-chain-v1\0"
        + previous.encode("ascii")
        + b"\0"
        + canonical_event.encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _pending_intent_digest(
    *,
    episode_id: str,
    ledger: str,
    event_schema: str,
    previous_head: LedgerHead,
    next_head: LedgerHead,
    canonical_event: str,
) -> str:
    body = {
        "episode_id": episode_id,
        "ledger": ledger,
        "event_schema": event_schema,
        "previous_head": previous_head.to_mapping(),
        "next_head": next_head.to_mapping(),
        "canonical_event": canonical_event,
    }
    encoded = _canonical_json(body).encode("utf-8")
    return "sha256:" + hashlib.sha256(
        b"robomex-ledger-pending-v1\0" + encoded
    ).hexdigest()


def _validate_canonical_event(canonical_event: str, *, event_schema: str) -> None:
    try:
        event = json.loads(canonical_event)
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError("Pending event is not valid JSON.") from exc
    if (
        not isinstance(event, Mapping)
        or event.get("schema") != event_schema
        or _canonical_json(event) != canonical_event
    ):
        raise ArtifactIntegrityError("Pending event is not canonical for its event schema.")


def _validate_committed_prefix(raw: bytes, *, head: LedgerHead) -> None:
    if len(raw) < head.byte_count:
        raise ArtifactIntegrityError(
            f"Event log for {head.ledger!r} was truncated below its durable head."
        )
    prefix = raw[: head.byte_count]
    if prefix and not prefix.endswith(b"\n"):
        raise ArtifactIntegrityError(
            f"Durable prefix for {head.ledger!r} ends inside an event."
        )
    digest = LEDGER_DIGEST_GENESIS
    count = 0
    for line_number, raw_line in enumerate(prefix.splitlines(keepends=True), start=1):
        try:
            line = raw_line.removesuffix(b"\n").decode("utf-8")
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                f"Committed event {line_number} for {head.ledger!r} is unreadable."
            ) from exc
        if not isinstance(event, Mapping) or not line or _canonical_json(event) != line:
            raise ArtifactIntegrityError(
                f"Committed event {line_number} for {head.ledger!r} is non-canonical."
            )
        digest = next_ledger_digest(digest, line)
        count += 1
    if count != head.event_count or digest != head.last_event_digest:
        raise ArtifactIntegrityError(
            f"Durable head for {head.ledger!r} does not match its journal prefix."
        )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError("Durability records require finite canonical JSON.") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    encoded = _canonical_json(value).encode("utf-8") + b"\n"
    atomic_write_bytes(path, encoded)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace one file durably, including the containing directory entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_sha256_digest(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    suffix = value.removeprefix("sha256:")
    return len(suffix) == 64 and all(character in "0123456789abcdef" for character in suffix)
