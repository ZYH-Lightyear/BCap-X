"""Durable workflow-scheduler checkpoints for RoboMEx v2.

Every scheduler mutation is represented by a complete, self-validating state
image.  Appends are flushed and fsynced before a command or graph successor is
reported to the caller.  Runtime events remain in their own append-only log;
``recover_scheduler`` restores the newest checkpoint and replays any durable
event tail that was not checkpointed before a crash.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from robomex.runtime.activation import (
    ActivationScheduler,
    AuthorityRegistry,
    SchedulerError,
    SchedulerState,
    TypedEventBus,
)

if TYPE_CHECKING:
    from robomex.elastic.graph_patch import (
        ComposableFrontier,
        GraphPatchCommit,
        GraphPatchCoordinator,
    )


class SchedulerStoreIntegrityError(SchedulerError):
    """A checkpoint journal is corrupt, forked, or causally incomplete."""


def _canonical_commit_payload(
    *, state: SchedulerState, reason: str, metadata: dict[str, JsonValue]
) -> bytes:
    return json.dumps(
        {
            "state": state.model_dump(mode="json"),
            "reason": reason,
            "metadata": metadata,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class SchedulerCommit(BaseModel):
    """One complete fsync boundary in a workflow state journal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["robomex.scheduler_commit.v1"] = "robomex.scheduler_commit.v1"
    commit_id: str = Field(min_length=1, max_length=768)
    state: SchedulerState
    reason: str = Field(min_length=1, max_length=256)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    content_digest: str = Field(min_length=64, max_length=64)
    committed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)  # noqa: UP017
    )

    @classmethod
    def build(
        cls,
        state: SchedulerState,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> SchedulerCommit:
        if not reason.strip():
            raise ValueError("scheduler commit reason must not be empty")
        normalized = dict(metadata or {})
        # Pydantic validates that arbitrary orchestration metadata is JSON and
        # rejects hidden Python objects before any bytes reach the journal.
        provisional = {
            "commit_id": (f"{state.episode_id}:{state.workflow_id}:{state.state_revision}"),
            "state": state,
            "reason": reason,
            "metadata": normalized,
            "content_digest": "0" * 64,
        }
        parsed = cls.model_validate(provisional)
        digest = hashlib.sha256(
            _canonical_commit_payload(
                state=parsed.state,
                reason=parsed.reason,
                metadata=parsed.metadata,
            )
        ).hexdigest()
        return parsed.model_copy(update={"content_digest": digest})

    @model_validator(mode="after")
    def _verify_identity_and_digest(self) -> SchedulerCommit:
        expected_id = (
            f"{self.state.episode_id}:{self.state.workflow_id}:{self.state.state_revision}"
        )
        if self.commit_id != expected_id:
            raise ValueError("Scheduler commit id is not bound to its state revision.")
        expected_digest = hashlib.sha256(
            _canonical_commit_payload(
                state=self.state,
                reason=self.reason,
                metadata=self.metadata,
            )
        ).hexdigest()
        # ``build`` first validates a zero placeholder, so only enforce a real
        # digest after that structural validation pass.
        if self.content_digest != "0" * 64 and self.content_digest != expected_digest:
            raise ValueError("Scheduler commit content digest mismatch.")
        return self


@dataclass(frozen=True, slots=True)
class _JournalObservation:
    """File identity observed at one fully validated journal boundary."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _JournalObservation:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )

    @property
    def identity(self) -> tuple[int, int]:
        return (self.device, self.inode)


class JsonlSchedulerStateStore:
    """Append-only fsync journal keyed by workflow and state revision."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._lock = threading.RLock()
        self._commits: list[SchedulerCommit] = []
        self._by_key: dict[tuple[str, str, int], SchedulerCommit] = {}
        self._latest: dict[tuple[str, str], SchedulerCommit] = {}
        self._line_count = 0
        self._content_hasher = hashlib.sha256()
        self._observation: _JournalObservation | None = None
        # A new instance always validates the complete restart chain.  The
        # sidecar lock also serializes first creation with another process.
        with self._process_lock(exclusive=True):
            if not self.path.exists():
                self.path.touch()
            self._reload_from_disk()

    def append(
        self,
        state: SchedulerState,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        commit = SchedulerCommit.build(state, reason=reason, metadata=metadata)
        workflow_key = (state.episode_id, state.workflow_id)
        revision_key = (*workflow_key, state.state_revision)
        with self._lock, self._process_lock(exclusive=True):
            # Another EpisodeRuntime may have appended since this object was
            # constructed.  The sidecar flock plus tail sync closes the
            # check-then-append race across processes.
            self._sync_tail_from_disk()
            previous = self._by_key.get(revision_key)
            if previous is not None:
                if previous.content_digest != commit.content_digest:
                    raise SchedulerStoreIntegrityError(
                        "Scheduler state revision was reused with different content."
                    )
                return False
            latest = self._latest.get(workflow_key)
            if latest is not None and (state.state_revision != latest.state.state_revision + 1):
                raise SchedulerStoreIntegrityError(
                    "Scheduler state journal revision is not contiguous."
                )
            encoded = (
                json.dumps(
                    commit.model_dump(mode="json"),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            with self.path.open("ab+") as stream:
                before = _JournalObservation.from_stat(os.fstat(stream.fileno()))
                if self._observation is None or before != self._observation:
                    raise SchedulerStoreIntegrityError(
                        "Scheduler journal changed during append admission."
                    )
                stream.seek(0, os.SEEK_END)
                start = stream.tell()
                try:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                except Exception:
                    current = os.fstat(stream.fileno())
                    expected_end = start + len(encoded)
                    if (
                        current.st_dev,
                        current.st_ino,
                    ) == before.identity and start <= current.st_size <= expected_end:
                        stream.seek(start)
                        stream.truncate()
                        stream.flush()
                        with suppress(Exception):
                            os.fsync(stream.fileno())
                        self._observation = _JournalObservation.from_stat(os.fstat(stream.fileno()))
                    raise
                after = _JournalObservation.from_stat(os.fstat(stream.fileno()))
                if after.identity != before.identity or after.size != start + len(encoded):
                    raise SchedulerStoreIntegrityError(
                        "Scheduler journal append did not end at the expected boundary."
                    )
                self._ensure_path_observation(after)
            self._record(commit)
            self._content_hasher.update(encoded)
            self._line_count += 1
            self._observation = after
            return True

    def latest(self, *, episode_id: str, workflow_id: str) -> SchedulerCommit | None:
        with self._lock, self._process_lock(exclusive=False):
            self._sync_tail_from_disk()
            return self._latest.get((episode_id, workflow_id))

    def history(
        self, *, episode_id: str | None = None, workflow_id: str | None = None
    ) -> tuple[SchedulerCommit, ...]:
        with self._lock, self._process_lock(exclusive=False):
            self._sync_tail_from_disk()
            return tuple(
                commit
                for commit in self._commits
                if (episode_id is None or commit.state.episode_id == episode_id)
                and (workflow_id is None or commit.state.workflow_id == workflow_id)
            )

    def graph_patch_commits(
        self, *, episode_id: str, workflow_id: str
    ) -> tuple[GraphPatchCommit, ...]:
        """Return the validated dynamic-graph chain embedded in checkpoints."""

        from robomex.elastic.graph_patch import GraphPatchCommit

        commits = []
        for checkpoint in self.history(episode_id=episode_id, workflow_id=workflow_id):
            raw = checkpoint.metadata.get("graph_patch_commit")
            if raw is not None:
                commits.append(GraphPatchCommit.model_validate(raw))
        return tuple(commits)

    def latest_composable_frontier(
        self, *, episode_id: str, workflow_id: str
    ) -> ComposableFrontier | None:
        from robomex.elastic.graph_patch import ComposableFrontier

        for checkpoint in reversed(self.history(episode_id=episode_id, workflow_id=workflow_id)):
            raw = checkpoint.metadata.get("composable_frontier")
            if raw is not None:
                return ComposableFrontier.model_validate(raw)
        return None

    def recover_patch_coordinator(
        self, *, scheduler: ActivationScheduler
    ) -> GraphPatchCoordinator | None:
        """Restore the latest open frontier and accepted patch audit chain."""

        from robomex.elastic.graph_patch import GraphPatchCoordinator

        commits = self.graph_patch_commits(
            episode_id=scheduler.episode_id,
            workflow_id=scheduler.workflow_id,
        )
        if commits:
            return GraphPatchCoordinator.recover_from_commits(scheduler=scheduler, commits=commits)
        frontier = self.latest_composable_frontier(
            episode_id=scheduler.episode_id,
            workflow_id=scheduler.workflow_id,
        )
        if frontier is None:
            return None
        return GraphPatchCoordinator(
            scheduler=scheduler,
            frontier=frontier,
            persist_frontier=False,
        )

    def recover_scheduler(
        self,
        *,
        episode_id: str,
        workflow_id: str,
        event_bus: TypedEventBus | None = None,
        authority_registry: AuthorityRegistry | None = None,
        replay_pending_events: bool = True,
    ) -> ActivationScheduler:
        commit = self.latest(episode_id=episode_id, workflow_id=workflow_id)
        if commit is None:
            raise SchedulerStoreIntegrityError(
                f"No durable scheduler state for {episode_id}/{workflow_id}."
            )
        return ActivationScheduler.restore(
            commit.state,
            event_bus=event_bus,
            authority_registry=authority_registry,
            state_sink=self,
            replay_pending_events=replay_pending_events,
        )

    def _record(self, commit: SchedulerCommit) -> None:
        self._record_into(
            commit,
            commits=self._commits,
            by_key=self._by_key,
            latest=self._latest,
        )

    @staticmethod
    def _record_into(
        commit: SchedulerCommit,
        *,
        commits: list[SchedulerCommit],
        by_key: dict[tuple[str, str, int], SchedulerCommit],
        latest: dict[tuple[str, str], SchedulerCommit],
    ) -> None:
        workflow_key = (commit.state.episode_id, commit.state.workflow_id)
        revision_key = (*workflow_key, commit.state.state_revision)
        previous = by_key.get(revision_key)
        if previous is not None:
            if previous != commit:
                raise SchedulerStoreIntegrityError(
                    "Conflicting duplicate scheduler commit in journal."
                )
            return
        previous_latest = latest.get(workflow_key)
        if previous_latest is not None and (
            commit.state.state_revision != previous_latest.state.state_revision + 1
        ):
            raise SchedulerStoreIntegrityError(
                "Scheduler journal contains a revision gap or regression."
            )
        commits.append(commit)
        by_key[revision_key] = commit
        latest[workflow_key] = commit

    def _reload_from_disk(self) -> None:
        """Validate and install the complete journal (the restart path)."""

        raw, observation = self._read_stable_journal()
        commits, by_key, latest, line_count = self._parse_lines(
            raw,
            first_line_number=1,
        )
        self._commits = commits
        self._by_key = by_key
        self._latest = latest
        self._line_count = line_count
        self._content_hasher = hashlib.sha256(raw)
        self._observation = observation

    def _sync_tail_from_disk(self) -> None:
        """Validate only records appended after the cached durable boundary."""

        if self._observation is None:
            self._reload_from_disk()
            return
        try:
            current = _JournalObservation.from_stat(self.path.stat())
        except FileNotFoundError as exc:
            raise SchedulerStoreIntegrityError(
                "Scheduler journal disappeared after it was opened."
            ) from exc
        if current.identity != self._observation.identity:
            raise SchedulerStoreIntegrityError(
                "Scheduler journal was replaced after it was opened."
            )
        if current.size < self._observation.size:
            raise SchedulerStoreIntegrityError(
                "Scheduler journal was truncated after it was opened."
            )
        prefix_digest, tail, verified = self._read_stable_tail(prefix_size=self._observation.size)
        if prefix_digest != self._content_hasher.digest():
            raise SchedulerStoreIntegrityError(
                "Scheduler journal content changed behind the append boundary."
            )
        if current.size == self._observation.size:
            if tail:
                raise SchedulerStoreIntegrityError(
                    "Scheduler journal size changed during tail synchronization."
                )
            if current == self._observation:
                self._observation = verified
                return
            # A touch/chmod is harmless, but a same-size overwrite must not be
            # accepted merely because there are no new lines to parse.
            # Full parsing on this anomalous path preserves restart-grade
            # validation even when the raw bytes happen to be unchanged.
            self._reload_from_disk()
            return

        commits = list(self._commits)
        by_key = dict(self._by_key)
        latest = dict(self._latest)
        parsed, by_key, latest, added_lines = self._parse_lines(
            tail,
            first_line_number=self._line_count + 1,
            commits=commits,
            by_key=by_key,
            latest=latest,
        )
        self._commits = parsed
        self._by_key = by_key
        self._latest = latest
        self._line_count += added_lines
        self._content_hasher.update(tail)
        self._observation = verified

    def _read_stable_tail(self, *, prefix_size: int) -> tuple[bytes, bytes, _JournalObservation]:
        """Hash the cached prefix and return only bytes beyond its boundary.

        Pydantic validation is incremental, while hashing the prefix protects
        against an in-place same-size overwrite even on filesystems whose
        timestamps have coarse resolution.
        """

        prefix_hasher = hashlib.sha256()
        try:
            with self.path.open("rb") as stream:
                before = _JournalObservation.from_stat(os.fstat(stream.fileno()))
                if before.size < prefix_size:
                    raise SchedulerStoreIntegrityError(
                        "Scheduler journal was truncated before tail synchronization."
                    )
                remaining = prefix_size
                while remaining:
                    chunk = stream.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise SchedulerStoreIntegrityError(
                            "Scheduler journal ended inside the cached prefix."
                        )
                    prefix_hasher.update(chunk)
                    remaining -= len(chunk)
                tail = stream.read()
                after = _JournalObservation.from_stat(os.fstat(stream.fileno()))
        except FileNotFoundError as exc:
            raise SchedulerStoreIntegrityError(
                "Scheduler journal disappeared during tail synchronization."
            ) from exc
        if before != after or len(tail) != after.size - prefix_size:
            raise SchedulerStoreIntegrityError(
                "Scheduler journal changed while its tail was being validated."
            )
        self._ensure_path_observation(after)
        return prefix_hasher.digest(), tail, after

    def _read_stable_journal(self) -> tuple[bytes, _JournalObservation]:
        try:
            with self.path.open("rb") as stream:
                before = _JournalObservation.from_stat(os.fstat(stream.fileno()))
                raw = stream.read()
                after = _JournalObservation.from_stat(os.fstat(stream.fileno()))
        except FileNotFoundError as exc:
            raise SchedulerStoreIntegrityError(
                "Scheduler journal disappeared while it was being validated."
            ) from exc
        if before != after or len(raw) != after.size:
            raise SchedulerStoreIntegrityError(
                "Scheduler journal changed while it was being validated."
            )
        self._ensure_path_observation(after)
        return raw, after

    def _ensure_path_observation(self, expected: _JournalObservation) -> None:
        try:
            current = _JournalObservation.from_stat(self.path.stat())
        except FileNotFoundError as exc:
            raise SchedulerStoreIntegrityError(
                "Scheduler journal path disappeared during validation."
            ) from exc
        if current != expected:
            raise SchedulerStoreIntegrityError("Scheduler journal path changed during validation.")

    @classmethod
    def _parse_lines(
        cls,
        raw: bytes,
        *,
        first_line_number: int,
        commits: list[SchedulerCommit] | None = None,
        by_key: dict[tuple[str, str, int], SchedulerCommit] | None = None,
        latest: dict[tuple[str, str], SchedulerCommit] | None = None,
    ) -> tuple[
        list[SchedulerCommit],
        dict[tuple[str, str, int], SchedulerCommit],
        dict[tuple[str, str], SchedulerCommit],
        int,
    ]:
        if raw and not raw.endswith(b"\n"):
            raise SchedulerStoreIntegrityError("Scheduler journal ends with an incomplete commit.")
        parsed_commits = [] if commits is None else commits
        parsed_by_key = {} if by_key is None else by_key
        parsed_latest = {} if latest is None else latest
        physical_lines = raw.splitlines(keepends=True)
        for offset, line in enumerate(physical_lines):
            if not line.strip():
                continue
            line_number = first_line_number + offset
            try:
                commit = SchedulerCommit.model_validate_json(line)
                cls._record_into(
                    commit,
                    commits=parsed_commits,
                    by_key=parsed_by_key,
                    latest=parsed_latest,
                )
            except Exception as exc:
                raise SchedulerStoreIntegrityError(
                    f"invalid scheduler commit at line {line_number}"
                ) from exc
        return parsed_commits, parsed_by_key, parsed_latest, len(physical_lines)

    @contextmanager
    def _process_lock(self, *, exclusive: bool):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


__all__ = [
    "JsonlSchedulerStateStore",
    "SchedulerCommit",
    "SchedulerStoreIntegrityError",
]
