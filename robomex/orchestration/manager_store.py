"""Durable append-only storage for bounded Swarm Manager session records."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from robomex.orchestration.manager import SwarmManagerSession


class ManagerStoreError(RuntimeError):
    """Manager session history is malformed or attempts to move backwards."""


class ManagerSessionLedger:
    """Persist versioned Manager records without retaining model chat state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: list[SwarmManagerSession] = []
        self._digests: dict[tuple[str, int], str] = {}
        if self.path.exists():
            self._load()
        else:
            self.path.touch()

    @property
    def records(self) -> tuple[SwarmManagerSession, ...]:
        return tuple(self._records)

    def latest(self, session_id: str) -> SwarmManagerSession | None:
        return next(
            (record for record in reversed(self._records) if record.session_id == session_id),
            None,
        )

    def append(self, session: SwarmManagerSession) -> bool:
        validated = SwarmManagerSession.model_validate(session.model_dump(mode="python"))
        payload = validated.model_dump(mode="json")
        digest = _digest(payload)
        key = (validated.session_id, validated.record_revision)
        with self._lock:
            previous_digest = self._digests.get(key)
            if previous_digest is not None:
                if previous_digest == digest:
                    return False
                raise ManagerStoreError("manager revision was rebound to different content")
            latest = self.latest(validated.session_id)
            if latest is not None and validated.record_revision != latest.record_revision + 1:
                raise ManagerStoreError(
                    "manager record_revision must increase by exactly one"
                )
            envelope = {
                "schema": "robomex.manager_session_record.v1",
                "sequence": len(self._records) + 1,
                "record_digest": digest,
                "session": payload,
            }
            encoded = json.dumps(
                envelope,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._records.append(validated)
            self._digests[key] = digest
            return True

    def _load(self) -> None:
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                envelope = json.loads(line)
                if not isinstance(envelope, dict):
                    raise TypeError("record is not an object")
                if envelope.get("schema") != "robomex.manager_session_record.v1":
                    raise ValueError("unknown manager record schema")
                if envelope.get("sequence") != line_number:
                    raise ValueError("manager record sequence mismatch")
                session = SwarmManagerSession.model_validate(envelope.get("session"))
                digest = _digest(session.model_dump(mode="json"))
                if envelope.get("record_digest") != digest:
                    raise ValueError("manager record digest mismatch")
                key = (session.session_id, session.record_revision)
                if key in self._digests:
                    raise ValueError("duplicate manager session revision")
                latest = self.latest(session.session_id)
                if latest is not None and session.record_revision != latest.record_revision + 1:
                    raise ValueError("manager revision is not contiguous")
            except Exception as exc:
                raise ManagerStoreError(
                    f"invalid manager session record at line {line_number}"
                ) from exc
            self._records.append(session)
            self._digests[key] = digest


def _digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = ["ManagerSessionLedger", "ManagerStoreError"]
