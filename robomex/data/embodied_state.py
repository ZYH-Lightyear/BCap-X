"""Narrow, deterministic embodied-state reducer for one episode.

Artifacts remain immutable computation facts.  This module owns only the small
set of physical fluents that must survive workflow boundaries: stable entity
handles and attachment belief.  Agents, monitors, and learned verifiers submit
closed ``StateTransitionProposal`` objects; only ``EmbodiedStateReducer`` can
commit a new canonical state revision.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from robomex.data.artifact_resolver import (
    ArtifactIntegrityError,
    ArtifactResolver,
    ResolvedArtifact,
    ResolvedArtifactRef,
)
from robomex.data.durability import (
    LEDGER_DIGEST_GENESIS,
    LedgerHead,
    LedgerPendingAppend,
    atomic_write_json,
    build_pending_append,
    clear_pending_append,
    exclusive_episode_lock,
    finalize_ledger_recovery,
    inspect_ledger_for_replay,
    load_ledger_head,
    next_ledger_digest,
    write_ledger_head,
    write_pending_append,
)
from robomex.data.physical_schema import (
    AttachmentEvidence,
    LocalizationEvidence,
    Pose,
    RelationEvidence,
    RelationPredicate,
    RelationValue,
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_STATE_EVENT_SCHEMA = "robomex.embodied_state_event.v2"
_STATE_INDEX_SCHEMA = "robomex.embodied_state_index.v2"
_MANIFEST_SCHEMA = "robomex.episode_manifest.v2"
_LEDGER_NAME = "embodied_state_events"
_LEDGER_HEAD_NAME = "embodied_state_events_head.v1.json"
_LEDGER_PENDING_NAME = "embodied_state_events_pending.v1.json"
_DURABILITY_KEY = "embodied_state_events"
_DURABILITY_MODE = "anchored_sha256_chain_v1"
_KNOWN_DURABILITY_KEYS = frozenset({"artifact_events", "embodied_state_events"})


class StateTransitionRejected(ValueError):  # noqa: N818 - domain outcome name
    """A proposal failed a deterministic, fail-closed reducer check."""


class AttachmentStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    UNKNOWN = "unknown"
    NOT_HELD = "not_held"
    ATTEMPTED = "attempted"
    VERIFIED_HELD = "verified_held"


class LocalizationStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    UNKNOWN = "unknown"
    LOCALIZED = "localized"
    UNLOCALIZED = "unlocalized"
    AMBIGUOUS = "ambiguous"


class PhysicalStateTrigger(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    EVIDENCE = "evidence"
    OPEN_COMMAND = "open_command"
    INTERRUPTION = "interruption"


class StateTransitionKind(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    REGISTER_ENTITY = "register_entity"
    SET_ATTACHMENT = "set_attachment"
    SET_LOCALIZATION = "set_localization"
    SET_RELATION = "set_relation"


@dataclass(frozen=True)
class EntityState:
    entity_id: str
    semantic_label: str
    track_id: str = ""
    revision: int = 0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "semantic_label": self.semantic_label,
            "track_id": self.track_id,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class AttachmentState:
    status: AttachmentStatus = AttachmentStatus.UNKNOWN
    entity_id: str | None = None
    observation_ref: ResolvedArtifactRef | None = None
    geometry_ref: ResolvedArtifactRef | None = None
    supporting_evidence: tuple[ResolvedArtifactRef, ...] = ()
    action_id: str = ""
    source_observation_id: str = ""
    source_observation_revision: int | None = None
    source_observation_domain: str = ""
    revision: int = 0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "entity_id": self.entity_id,
            "observation_ref": (
                self.observation_ref.to_mapping() if self.observation_ref else None
            ),
            "geometry_ref": (self.geometry_ref.to_mapping() if self.geometry_ref else None),
            "supporting_evidence": [ref.to_mapping() for ref in self.supporting_evidence],
            "action_id": self.action_id,
            "source_observation_id": self.source_observation_id,
            "source_observation_revision": self.source_observation_revision,
            "source_observation_domain": self.source_observation_domain,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class LocalizationState:
    entity_id: str
    status: LocalizationStatus = LocalizationStatus.UNKNOWN
    world_pose: Pose | None = None
    source_observation_id: str = ""
    source_observation_revision: int | None = None
    source_observation_domain: str = ""
    supporting_evidence: tuple[ResolvedArtifactRef, ...] = ()
    action_id: str = ""
    revision: int = 0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "status": self.status.value,
            "world_pose": (
                self.world_pose.model_dump(mode="json") if self.world_pose is not None else None
            ),
            "source_observation_id": self.source_observation_id,
            "source_observation_revision": self.source_observation_revision,
            "source_observation_domain": self.source_observation_domain,
            "supporting_evidence": [ref.to_mapping() for ref in self.supporting_evidence],
            "action_id": self.action_id,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class RelationState:
    subject_entity_id: str
    predicate: RelationPredicate
    target_entity_id: str
    value: RelationValue = RelationValue.UNKNOWN
    source_observation_id: str = ""
    source_observation_revision: int | None = None
    source_observation_domain: str = ""
    supporting_evidence: tuple[ResolvedArtifactRef, ...] = ()
    action_id: str = ""
    revision: int = 0

    @property
    def key(self) -> tuple[str, RelationPredicate, str]:
        return (self.subject_entity_id, self.predicate, self.target_entity_id)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "subject_entity_id": self.subject_entity_id,
            "predicate": self.predicate.value,
            "target_entity_id": self.target_entity_id,
            "value": self.value.value,
            "source_observation_id": self.source_observation_id,
            "source_observation_revision": self.source_observation_revision,
            "source_observation_domain": self.source_observation_domain,
            "supporting_evidence": [ref.to_mapping() for ref in self.supporting_evidence],
            "action_id": self.action_id,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class EpisodeEmbodiedState:
    episode_id: str
    revision: int = 0
    entities: tuple[EntityState, ...] = ()
    attachment: AttachmentState = AttachmentState()
    localizations: tuple[LocalizationState, ...] = ()
    relations: tuple[RelationState, ...] = ()

    @property
    def attachment_status(self) -> AttachmentStatus:
        return self.attachment.status

    @property
    def held_entity_id(self) -> str | None:
        if self.attachment.status is AttachmentStatus.VERIFIED_HELD:
            return self.attachment.entity_id
        return None

    @property
    def attachment_observation_ref(self) -> ResolvedArtifactRef | None:
        return self.attachment.observation_ref

    @property
    def held_geometry_ref(self) -> ResolvedArtifactRef | None:
        if self.attachment.status is not AttachmentStatus.VERIFIED_HELD:
            return None
        return self.attachment.geometry_ref

    def entity(self, entity_id: str) -> EntityState | None:
        return next((item for item in self.entities if item.entity_id == entity_id), None)

    def localization(self, entity_id: str) -> LocalizationState | None:
        return next(
            (item for item in self.localizations if item.entity_id == entity_id),
            None,
        )

    def relation(
        self,
        subject_entity_id: str,
        predicate: RelationPredicate | str,
        target_entity_id: str,
    ) -> RelationState | None:
        try:
            typed_predicate = (
                predicate
                if isinstance(predicate, RelationPredicate)
                else RelationPredicate(predicate)
            )
        except ValueError:
            return None
        key = (subject_entity_id, typed_predicate, target_entity_id)
        return next((item for item in self.relations if item.key == key), None)

    def relations_for(self, subject_entity_id: str) -> tuple[RelationState, ...]:
        return tuple(
            relation
            for relation in self.relations
            if relation.subject_entity_id == subject_entity_id
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "revision": self.revision,
            "entities": [entity.to_mapping() for entity in self.entities],
            "attachment": self.attachment.to_mapping(),
            "localizations": [item.to_mapping() for item in self.localizations],
            "relations": [item.to_mapping() for item in self.relations],
        }


@dataclass(frozen=True)
class StateTransitionProposal:
    """Closed state mutation request produced by a non-authoritative actor."""

    episode_id: str
    effect_id: str
    before_revision: int
    kind: StateTransitionKind
    source: str
    evidence_refs: tuple[ResolvedArtifactRef, ...]
    entity_id: str
    semantic_label: str = ""
    track_id: str = ""
    attachment_status: AttachmentStatus | None = None
    observation_ref: ResolvedArtifactRef | None = None
    geometry_ref: ResolvedArtifactRef | None = None
    action_id: str = ""
    trigger: PhysicalStateTrigger = PhysicalStateTrigger.EVIDENCE
    localization_status: LocalizationStatus | None = None
    world_pose: Pose | None = None
    source_observation_id: str = ""
    source_observation_revision: int | None = None
    source_observation_domain: str = ""
    target_entity_id: str = ""
    relation_predicate: RelationPredicate | None = None
    relation_value: RelationValue | None = None
    subject_track_id: str = ""
    target_track_id: str = ""

    @classmethod
    def register_entity(
        cls,
        *,
        episode_id: str,
        effect_id: str,
        before_revision: int,
        source: str,
        evidence_refs: Iterable[ResolvedArtifactRef],
        entity_id: str,
        semantic_label: str,
        track_id: str = "",
    ) -> StateTransitionProposal:
        return cls(
            episode_id=episode_id,
            effect_id=effect_id,
            before_revision=before_revision,
            kind=StateTransitionKind.REGISTER_ENTITY,
            source=source,
            evidence_refs=tuple(evidence_refs),
            entity_id=entity_id,
            semantic_label=semantic_label,
            track_id=track_id,
        )

    @classmethod
    def set_attachment(
        cls,
        *,
        episode_id: str,
        effect_id: str,
        before_revision: int,
        source: str,
        evidence_refs: Iterable[ResolvedArtifactRef],
        entity_id: str,
        status: AttachmentStatus,
        observation_ref: ResolvedArtifactRef | None = None,
        geometry_ref: ResolvedArtifactRef | None = None,
        action_id: str = "",
        trigger: PhysicalStateTrigger = PhysicalStateTrigger.EVIDENCE,
        source_observation_id: str = "",
        source_observation_revision: int | None = None,
        source_observation_domain: str = "",
        track_id: str = "",
    ) -> StateTransitionProposal:
        return cls(
            episode_id=episode_id,
            effect_id=effect_id,
            before_revision=before_revision,
            kind=StateTransitionKind.SET_ATTACHMENT,
            source=source,
            evidence_refs=tuple(evidence_refs),
            entity_id=entity_id,
            attachment_status=status,
            observation_ref=observation_ref,
            geometry_ref=geometry_ref,
            action_id=action_id,
            trigger=trigger,
            source_observation_id=source_observation_id,
            source_observation_revision=source_observation_revision,
            source_observation_domain=source_observation_domain,
            subject_track_id=track_id,
        )

    @classmethod
    def mark_open_admitted(
        cls,
        *,
        episode_id: str,
        effect_id: str,
        before_revision: int,
        source: str,
        evidence_refs: Iterable[ResolvedArtifactRef],
        entity_id: str,
        action_id: str,
        observation_ref: ResolvedArtifactRef | None = None,
    ) -> StateTransitionProposal:
        """Conservatively invalidate attachment once an open command is admitted."""

        return cls.set_attachment(
            episode_id=episode_id,
            effect_id=effect_id,
            before_revision=before_revision,
            source=source,
            evidence_refs=evidence_refs,
            entity_id=entity_id,
            status=AttachmentStatus.UNKNOWN,
            action_id=action_id,
            observation_ref=observation_ref,
            trigger=PhysicalStateTrigger.OPEN_COMMAND,
        )

    @classmethod
    def mark_interrupted(
        cls,
        *,
        episode_id: str,
        effect_id: str,
        before_revision: int,
        source: str,
        evidence_refs: Iterable[ResolvedArtifactRef],
        entity_id: str,
        action_id: str = "",
        observation_ref: ResolvedArtifactRef | None = None,
        source_observation_id: str = "",
        source_observation_revision: int | None = None,
        source_observation_domain: str = "",
        track_id: str = "",
    ) -> StateTransitionProposal:
        """Conservatively invalidate attachment after partial/unknown execution."""

        return cls.set_attachment(
            episode_id=episode_id,
            effect_id=effect_id,
            before_revision=before_revision,
            source=source,
            evidence_refs=evidence_refs,
            entity_id=entity_id,
            status=AttachmentStatus.UNKNOWN,
            action_id=action_id,
            observation_ref=observation_ref,
            trigger=PhysicalStateTrigger.INTERRUPTION,
            source_observation_id=source_observation_id,
            source_observation_revision=source_observation_revision,
            source_observation_domain=source_observation_domain,
            track_id=track_id,
        )

    @classmethod
    def set_localization(
        cls,
        *,
        episode_id: str,
        effect_id: str,
        before_revision: int,
        source: str,
        evidence_refs: Iterable[ResolvedArtifactRef],
        entity_id: str,
        status: LocalizationStatus,
        source_observation_id: str,
        source_observation_revision: int | None = None,
        source_observation_domain: str = "",
        world_pose: Pose | None = None,
        observation_ref: ResolvedArtifactRef | None = None,
        action_id: str = "",
        track_id: str = "",
    ) -> StateTransitionProposal:
        return cls(
            episode_id=episode_id,
            effect_id=effect_id,
            before_revision=before_revision,
            kind=StateTransitionKind.SET_LOCALIZATION,
            source=source,
            evidence_refs=tuple(evidence_refs),
            entity_id=entity_id,
            observation_ref=observation_ref,
            action_id=action_id,
            localization_status=status,
            world_pose=world_pose,
            source_observation_id=source_observation_id,
            source_observation_revision=source_observation_revision,
            source_observation_domain=source_observation_domain,
            subject_track_id=track_id,
        )

    @classmethod
    def set_relation(
        cls,
        *,
        episode_id: str,
        effect_id: str,
        before_revision: int,
        source: str,
        evidence_refs: Iterable[ResolvedArtifactRef],
        subject_entity_id: str,
        predicate: RelationPredicate,
        target_entity_id: str,
        value: RelationValue,
        source_observation_id: str,
        source_observation_revision: int | None = None,
        source_observation_domain: str = "",
        observation_ref: ResolvedArtifactRef | None = None,
        action_id: str = "",
        subject_track_id: str = "",
        target_track_id: str = "",
    ) -> StateTransitionProposal:
        return cls(
            episode_id=episode_id,
            effect_id=effect_id,
            before_revision=before_revision,
            kind=StateTransitionKind.SET_RELATION,
            source=source,
            evidence_refs=tuple(evidence_refs),
            entity_id=subject_entity_id,
            observation_ref=observation_ref,
            action_id=action_id,
            source_observation_id=source_observation_id,
            source_observation_revision=source_observation_revision,
            source_observation_domain=source_observation_domain,
            target_entity_id=target_entity_id,
            relation_predicate=predicate,
            relation_value=value,
            subject_track_id=subject_track_id,
            target_track_id=target_track_id,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "effect_id": self.effect_id,
            "before_revision": self.before_revision,
            "kind": self.kind.value,
            "source": self.source,
            "evidence_refs": [ref.to_mapping() for ref in self.evidence_refs],
            "entity_id": self.entity_id,
            "semantic_label": self.semantic_label,
            "track_id": self.track_id,
            "attachment_status": (self.attachment_status.value if self.attachment_status else None),
            "observation_ref": (
                self.observation_ref.to_mapping() if self.observation_ref else None
            ),
            "geometry_ref": (self.geometry_ref.to_mapping() if self.geometry_ref else None),
            "action_id": self.action_id,
            "trigger": self.trigger.value,
            "localization_status": (
                self.localization_status.value if self.localization_status else None
            ),
            "world_pose": (
                self.world_pose.model_dump(mode="json") if self.world_pose is not None else None
            ),
            "source_observation_id": self.source_observation_id,
            "source_observation_revision": self.source_observation_revision,
            "source_observation_domain": self.source_observation_domain,
            "target_entity_id": self.target_entity_id,
            "relation_predicate": (
                self.relation_predicate.value if self.relation_predicate else None
            ),
            "relation_value": self.relation_value.value if self.relation_value else None,
            "subject_track_id": self.subject_track_id,
            "target_track_id": self.target_track_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> StateTransitionProposal:
        try:
            raw_observation = value.get("observation_ref")
            raw_geometry = value.get("geometry_ref")
            raw_status = value.get("attachment_status")
            raw_localization = value.get("localization_status")
            raw_pose = value.get("world_pose")
            raw_predicate = value.get("relation_predicate")
            raw_relation_value = value.get("relation_value")
            raw_observation_revision = value.get("source_observation_revision")
            if isinstance(raw_observation_revision, bool):
                raise ValueError("Observation revisions cannot be booleans.")
            return cls(
                episode_id=str(value["episode_id"]),
                effect_id=str(value["effect_id"]),
                before_revision=int(value["before_revision"]),
                kind=StateTransitionKind(str(value["kind"])),
                source=str(value["source"]),
                evidence_refs=tuple(
                    ResolvedArtifactRef.from_any(item) for item in value.get("evidence_refs", ())
                ),
                entity_id=str(value["entity_id"]),
                semantic_label=str(value.get("semantic_label") or ""),
                track_id=str(value.get("track_id") or ""),
                attachment_status=(
                    AttachmentStatus(str(raw_status)) if raw_status is not None else None
                ),
                observation_ref=(
                    ResolvedArtifactRef.from_any(raw_observation)
                    if raw_observation is not None
                    else None
                ),
                geometry_ref=(
                    ResolvedArtifactRef.from_any(raw_geometry) if raw_geometry is not None else None
                ),
                action_id=str(value.get("action_id") or ""),
                trigger=PhysicalStateTrigger(
                    str(value.get("trigger") or PhysicalStateTrigger.EVIDENCE.value)
                ),
                localization_status=(
                    LocalizationStatus(str(raw_localization))
                    if raw_localization is not None
                    else None
                ),
                world_pose=Pose.model_validate(raw_pose) if raw_pose is not None else None,
                source_observation_id=str(value.get("source_observation_id") or ""),
                source_observation_revision=(
                    int(raw_observation_revision)
                    if raw_observation_revision is not None
                    else None
                ),
                source_observation_domain=str(
                    value.get("source_observation_domain") or ""
                ),
                target_entity_id=str(value.get("target_entity_id") or ""),
                relation_predicate=(
                    RelationPredicate(str(raw_predicate))
                    if raw_predicate is not None
                    else None
                ),
                relation_value=(
                    RelationValue(str(raw_relation_value))
                    if raw_relation_value is not None
                    else None
                ),
                subject_track_id=str(value.get("subject_track_id") or ""),
                target_track_id=str(value.get("target_track_id") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("Malformed state transition event.") from exc


_ALLOWED_ATTACHMENT_TRANSITIONS: dict[AttachmentStatus, frozenset[AttachmentStatus]] = {
    AttachmentStatus.UNKNOWN: frozenset(
        {AttachmentStatus.UNKNOWN, AttachmentStatus.ATTEMPTED, AttachmentStatus.NOT_HELD}
    ),
    AttachmentStatus.NOT_HELD: frozenset({AttachmentStatus.ATTEMPTED, AttachmentStatus.UNKNOWN}),
    AttachmentStatus.ATTEMPTED: frozenset(
        {
            AttachmentStatus.VERIFIED_HELD,
            AttachmentStatus.NOT_HELD,
            AttachmentStatus.UNKNOWN,
        }
    ),
    AttachmentStatus.VERIFIED_HELD: frozenset(
        {
            AttachmentStatus.VERIFIED_HELD,
            AttachmentStatus.UNKNOWN,
            AttachmentStatus.NOT_HELD,
        }
    ),
}


class EmbodiedStateReducer:
    """The sole deterministic commit authority for episode embodied state.

    ``strict_evidence`` protects confirmation-bearing localization, relation,
    and determinate attachment transitions.  Registration, action-attempt
    bookkeeping, and conservative UNKNOWN invalidations remain compatible with
    runtime receipts that predate the typed verifier schemas.
    """

    def __init__(
        self,
        episode_root: str | Path,
        *,
        episode_id: str,
        resolver: ArtifactResolver | None = None,
        strict_evidence: bool = False,
    ) -> None:
        _validate_id("episode_id", episode_id)
        if not isinstance(strict_evidence, bool):
            raise TypeError("strict_evidence must be a boolean policy switch")
        self.episode_root = Path(episode_root).resolve()
        self.episode_id = episode_id
        self.event_log_path = self.episode_root / "embodied_state_events.jsonl"
        self.ledger_head_path = self.episode_root / _LEDGER_HEAD_NAME
        self.ledger_pending_path = self.episode_root / _LEDGER_PENDING_NAME
        self.index_path = self.episode_root / "embodied_state_index.v2.json"
        self.manifest_path = self.episode_root / "episode_manifest.v2.json"
        self.resolver = resolver or ArtifactResolver(self.episode_root, episode_id=episode_id)
        self.strict_evidence = strict_evidence
        self._state = EpisodeEmbodiedState(episode_id=episode_id)
        self._effect_ids: set[str] = set()
        self._events: list[dict[str, Any]] = []
        self._ledger_digest = LEDGER_DIGEST_GENESIS
        self._ledger_byte_count = 0
        self._log_identity: tuple[int, int, int, int] | None = None

        self.episode_root.mkdir(parents=True, exist_ok=True)
        with exclusive_episode_lock(self.episode_root):
            durability_initialized = self._initialize_manifest()
            self.event_log_path.touch(exist_ok=True)
            self._replay_events(
                validate_evidence=True,
                require_head=durability_initialized,
            )
            self._mark_manifest_durable()
            self._write_index()

    @property
    def state(self) -> EpisodeEmbodiedState:
        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            return self._state

    @property
    def event_count(self) -> int:
        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            return len(self._events)

    def events(self) -> tuple[dict[str, Any], ...]:
        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            return tuple(json.loads(json.dumps(event)) for event in self._events)

    def commit(self, proposal: StateTransitionProposal) -> EpisodeEmbodiedState:
        """Validate and atomically append one state transition.

        A rejected proposal emits no semantic event and leaves the current view
        unchanged.  The event log remains the authority if writing the derived
        index is interrupted.
        """

        if not isinstance(proposal, StateTransitionProposal):
            raise StateTransitionRejected(
                "Only a closed StateTransitionProposal can enter the reducer."
            )
        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            next_state = self._reduce(proposal, validate_evidence=True)
            sequence = len(self._events) + 1
            event = {
                "schema": _STATE_EVENT_SCHEMA,
                "sequence": sequence,
                "event_id": f"state_evt:{self.episode_id}:{sequence}",
                "episode_id": self.episode_id,
                "kind": "state_transition_committed",
                "proposal": proposal.to_mapping(),
                "after_revision": next_state.revision,
                "state_digest": _state_digest(next_state),
            }
            canonical_event = _canonical_json(event)
            encoded = (canonical_event + "\n").encode("utf-8")
            next_digest = next_ledger_digest(self._ledger_digest, canonical_event)
            next_byte_count = self._ledger_byte_count + len(encoded)
            pending = build_pending_append(
                episode_id=self.episode_id,
                ledger=_LEDGER_NAME,
                event_schema=_STATE_EVENT_SCHEMA,
                previous_head=self._cursor_head(),
                canonical_event=canonical_event,
            )
            if (
                pending.next_head.event_count != sequence
                or pending.next_head.byte_count != next_byte_count
                or pending.next_head.last_event_digest != next_digest
            ):
                raise ArtifactIntegrityError("Pending state append did not bind the next head.")
            self._write_pending(pending)
            self._append_journal_bytes(encoded)
            self._write_head(
                event_count=sequence,
                byte_count=next_byte_count,
                digest=next_digest,
            )
            self._clear_pending()
            self._state = next_state
            self._effect_ids.add(proposal.effect_id)
            self._events.append(event)
            self._ledger_digest = next_digest
            self._ledger_byte_count = next_byte_count
            self._log_identity = self._current_log_identity()
            self._write_index()
            return self._state

    def rebuild_index(self) -> Path:
        with exclusive_episode_lock(self.episode_root):
            self._reset_view()
            self._replay_events(validate_evidence=True, require_head=True)
            self._write_index()
        return self.index_path

    def _reduce(
        self,
        proposal: StateTransitionProposal,
        *,
        validate_evidence: bool,
    ) -> EpisodeEmbodiedState:
        resolved_evidence = self._validate_common(
            proposal,
            validate_evidence=validate_evidence,
        )
        if proposal.kind is StateTransitionKind.REGISTER_ENTITY:
            return self._register_entity(proposal)
        if proposal.kind is StateTransitionKind.SET_ATTACHMENT:
            return self._set_attachment(proposal, resolved_evidence=resolved_evidence)
        if proposal.kind is StateTransitionKind.SET_LOCALIZATION:
            return self._set_localization(proposal, resolved_evidence=resolved_evidence)
        if proposal.kind is StateTransitionKind.SET_RELATION:
            return self._set_relation(proposal, resolved_evidence=resolved_evidence)
        raise StateTransitionRejected(f"Unknown transition kind {proposal.kind!r}.")

    def _validate_common(
        self,
        proposal: StateTransitionProposal,
        *,
        validate_evidence: bool,
    ) -> tuple[ResolvedArtifact, ...]:
        if not isinstance(proposal.kind, StateTransitionKind):
            raise StateTransitionRejected("Proposal kind must use the closed enum.")
        if not isinstance(proposal.trigger, PhysicalStateTrigger):
            raise StateTransitionRejected("Proposal trigger must use the closed enum.")
        if proposal.episode_id != self.episode_id:
            raise StateTransitionRejected("Cross-episode state proposal rejected.")
        _validate_id("effect_id", proposal.effect_id)
        _validate_id("entity_id", proposal.entity_id)
        if (
            not isinstance(proposal.before_revision, int)
            or isinstance(proposal.before_revision, bool)
            or proposal.before_revision < 0
        ):
            raise StateTransitionRejected("before_revision must be a non-negative integer.")
        if proposal.effect_id in self._effect_ids:
            raise StateTransitionRejected(f"Duplicate effect_id {proposal.effect_id!r} rejected.")
        if proposal.before_revision != self._state.revision:
            raise StateTransitionRejected(
                f"Stale before_revision {proposal.before_revision}; current state is "
                f"revision {self._state.revision}."
            )
        if not proposal.source.strip():
            raise StateTransitionRejected("Proposal source must be explicit.")
        if not isinstance(proposal.evidence_refs, tuple) or not proposal.evidence_refs:
            raise StateTransitionRejected("State transitions require source evidence.")
        identities: set[tuple[str, str]] = set()
        resolved_evidence: list[ResolvedArtifact] = []
        for ref in proposal.evidence_refs:
            if not isinstance(ref, ResolvedArtifactRef):
                raise StateTransitionRejected(
                    "Evidence must be an admission-resolved artifact ref."
                )
            identity = (ref.artifact_id, ref.content_digest)
            if identity in identities:
                raise StateTransitionRejected("Duplicate evidence ref rejected.")
            identities.add(identity)
            if validate_evidence:
                try:
                    resolved_evidence.append(self.resolver.resolve(ref))
                except ValueError as exc:
                    raise StateTransitionRejected(
                        f"Evidence ref {ref.artifact_id!r} failed integrity checks."
                    ) from exc
        for named_ref in (proposal.observation_ref, proposal.geometry_ref):
            if named_ref is None:
                continue
            identity = (named_ref.artifact_id, named_ref.content_digest)
            if identity not in identities:
                raise StateTransitionRejected(
                    "Observation/geometry refs must also appear in evidence_refs."
                )
        return tuple(resolved_evidence)

    def _register_entity(self, proposal: StateTransitionProposal) -> EpisodeEmbodiedState:
        if (
            proposal.attachment_status is not None
            or proposal.observation_ref is not None
            or proposal.geometry_ref is not None
            or proposal.action_id
            or proposal.trigger is not PhysicalStateTrigger.EVIDENCE
            or proposal.localization_status is not None
            or proposal.world_pose is not None
            or proposal.source_observation_id
            or proposal.source_observation_revision is not None
            or proposal.source_observation_domain
            or proposal.target_entity_id
            or proposal.relation_predicate is not None
            or proposal.relation_value is not None
            or proposal.subject_track_id
            or proposal.target_track_id
        ):
            raise StateTransitionRejected(
                "Entity registration cannot carry attachment mutation fields."
            )
        if not proposal.semantic_label.strip():
            raise StateTransitionRejected("Entity registration needs semantic_label.")
        if proposal.track_id:
            _validate_id("track_id", proposal.track_id)
        if self._state.entity(proposal.entity_id) is not None:
            raise StateTransitionRejected(f"Entity {proposal.entity_id!r} is already registered.")
        revision = self._state.revision + 1
        entity = EntityState(
            entity_id=proposal.entity_id,
            semantic_label=proposal.semantic_label,
            track_id=proposal.track_id,
            revision=revision,
        )
        entities = tuple(sorted((*self._state.entities, entity), key=lambda item: item.entity_id))
        return replace(self._state, revision=revision, entities=entities)

    def _set_attachment(
        self,
        proposal: StateTransitionProposal,
        *,
        resolved_evidence: tuple[ResolvedArtifact, ...],
    ) -> EpisodeEmbodiedState:
        target = proposal.attachment_status
        if not isinstance(target, AttachmentStatus):
            raise StateTransitionRejected("Attachment proposal must use a closed AttachmentStatus.")
        if (
            proposal.semantic_label
            or proposal.track_id
            or proposal.localization_status is not None
            or proposal.world_pose is not None
            or proposal.target_entity_id
            or proposal.relation_predicate is not None
            or proposal.relation_value is not None
            or proposal.target_track_id
        ):
            raise StateTransitionRejected(
                "Attachment proposals cannot edit entity, localization, or relation fields."
            )
        entity = self._state.entity(proposal.entity_id)
        if entity is None:
            raise StateTransitionRejected(
                f"Attachment refers to unknown entity {proposal.entity_id!r}."
            )
        self._validate_track_identity(
            entity,
            proposal.subject_track_id,
            role="attachment subject",
        )
        current = self._state.attachment
        if target not in _ALLOWED_ATTACHMENT_TRANSITIONS[current.status]:
            raise StateTransitionRejected(
                f"Illegal attachment transition {current.status.value!r} -> {target.value!r}."
            )
        if (
            current.entity_id is not None
            and current.status is not AttachmentStatus.NOT_HELD
            and proposal.entity_id != current.entity_id
        ):
            raise StateTransitionRejected(
                "Attachment entity cannot change without first resolving the old belief."
            )
        if proposal.trigger in {
            PhysicalStateTrigger.OPEN_COMMAND,
            PhysicalStateTrigger.INTERRUPTION,
        } and target is not AttachmentStatus.UNKNOWN:
            raise StateTransitionRejected(
                "Open/interruption effects must conservatively set attachment to unknown."
            )
        if (
            proposal.trigger is PhysicalStateTrigger.OPEN_COMMAND
            and not proposal.action_id
        ):
            raise StateTransitionRejected("An admitted open command must bind its action_id.")
        if target is AttachmentStatus.ATTEMPTED and not proposal.action_id:
            raise StateTransitionRejected(
                "An attempted attachment must bind the admitted action_id."
            )
        if target is AttachmentStatus.VERIFIED_HELD:
            if current.status not in {
                AttachmentStatus.ATTEMPTED,
                AttachmentStatus.VERIFIED_HELD,
            }:
                raise StateTransitionRejected(
                    "verified_held is reachable only from attempted or a fresh "
                    "evidence-backed verified_held refresh."
                )
            action_id = proposal.action_id or current.action_id
            if not action_id or (proposal.action_id and current.action_id != proposal.action_id):
                raise StateTransitionRejected(
                    "Attachment verification must match the attempted action_id."
                )
            if current.status is AttachmentStatus.VERIFIED_HELD:
                self._validate_verified_attachment_refresh(
                    current=current,
                    proposal=proposal,
                )
        else:
            action_id = proposal.action_id or current.action_id

        self._validate_observation_identity_fields(proposal)
        if self.strict_evidence and target in {
            AttachmentStatus.VERIFIED_HELD,
            AttachmentStatus.NOT_HELD,
        }:
            self._require_attachment_evidence(
                proposal,
                entity=entity,
                target=target,
                action_id=action_id,
                resolved_evidence=resolved_evidence,
            )

        revision = self._state.revision + 1
        geometry_ref = proposal.geometry_ref
        if geometry_ref is None and target in {
            AttachmentStatus.ATTEMPTED,
            AttachmentStatus.VERIFIED_HELD,
            AttachmentStatus.UNKNOWN,
        }:
            geometry_ref = current.geometry_ref
        if target is AttachmentStatus.NOT_HELD:
            geometry_ref = None
            action_id = ""
        if proposal.trigger is PhysicalStateTrigger.INTERRUPTION:
            geometry_ref = None
        attachment = AttachmentState(
            status=target,
            entity_id=proposal.entity_id,
            observation_ref=proposal.observation_ref,
            geometry_ref=geometry_ref,
            supporting_evidence=proposal.evidence_refs,
            action_id=action_id,
            source_observation_id=proposal.source_observation_id,
            source_observation_revision=proposal.source_observation_revision,
            source_observation_domain=proposal.source_observation_domain,
            revision=current.revision + 1,
        )
        next_state = replace(self._state, revision=revision, attachment=attachment)
        if proposal.trigger is PhysicalStateTrigger.INTERRUPTION:
            next_state = self._invalidate_entity_dependents(next_state, proposal)
        return next_state

    @staticmethod
    def _validate_verified_attachment_refresh(
        *,
        current: AttachmentState,
        proposal: StateTransitionProposal,
    ) -> None:
        """Require a genuinely newer observation for held-state re-verification."""

        if proposal.trigger is not PhysicalStateTrigger.EVIDENCE:
            raise StateTransitionRejected(
                "verified_held refresh requires an evidence trigger."
            )
        if proposal.observation_ref is None or proposal.observation_ref == current.observation_ref:
            raise StateTransitionRejected(
                "verified_held refresh requires a new explicit observation_ref."
            )
        if (
            not proposal.source_observation_id
            or proposal.source_observation_id == current.source_observation_id
        ):
            raise StateTransitionRejected(
                "verified_held refresh requires a new source observation identity."
            )
        if (
            current.source_observation_domain
            and proposal.source_observation_domain == current.source_observation_domain
            and current.source_observation_revision is not None
            and (
                proposal.source_observation_revision is None
                or proposal.source_observation_revision
                <= current.source_observation_revision
            )
        ):
            raise StateTransitionRejected(
                "verified_held refresh must advance the source observation revision."
            )

    def _set_localization(
        self,
        proposal: StateTransitionProposal,
        *,
        resolved_evidence: tuple[ResolvedArtifact, ...],
    ) -> EpisodeEmbodiedState:
        target = proposal.localization_status
        if not isinstance(target, LocalizationStatus):
            raise StateTransitionRejected(
                "Localization proposal must use a closed LocalizationStatus."
            )
        if (
            proposal.semantic_label
            or proposal.track_id
            or proposal.attachment_status is not None
            or proposal.geometry_ref is not None
            or proposal.target_entity_id
            or proposal.relation_predicate is not None
            or proposal.relation_value is not None
            or proposal.target_track_id
            or proposal.trigger is not PhysicalStateTrigger.EVIDENCE
        ):
            raise StateTransitionRejected(
                "Localization proposal carries fields owned by another transition kind."
            )
        entity = self._state.entity(proposal.entity_id)
        if entity is None:
            raise StateTransitionRejected(
                f"Localization refers to unknown entity {proposal.entity_id!r}."
            )
        self._validate_track_identity(
            entity,
            proposal.subject_track_id,
            role="localized subject",
        )
        if not proposal.source_observation_id:
            raise StateTransitionRejected("Localization requires source_observation_id.")
        self._validate_observation_identity_fields(
            proposal,
            require_complete=self.strict_evidence,
        )
        if target is LocalizationStatus.LOCALIZED:
            if not isinstance(proposal.world_pose, Pose):
                raise StateTransitionRejected("localized state requires a typed world_pose.")
            if proposal.world_pose.observation_id != proposal.source_observation_id:
                raise StateTransitionRejected(
                    "Localization pose and source observation identity must match."
                )
        elif proposal.world_pose is not None:
            raise StateTransitionRejected(
                "Unknown/unlocalized/ambiguous localization cannot retain a world pose."
            )
        if self.strict_evidence:
            self._require_localization_evidence(
                proposal,
                entity=entity,
                target=target,
                resolved_evidence=resolved_evidence,
            )

        current = self._state.localization(proposal.entity_id)
        localization = LocalizationState(
            entity_id=proposal.entity_id,
            status=target,
            world_pose=proposal.world_pose,
            source_observation_id=proposal.source_observation_id,
            source_observation_revision=proposal.source_observation_revision,
            source_observation_domain=proposal.source_observation_domain,
            supporting_evidence=proposal.evidence_refs,
            action_id=proposal.action_id,
            revision=(current.revision if current is not None else 0) + 1,
        )
        localizations = tuple(
            sorted(
                (
                    localization,
                    *(
                        item
                        for item in self._state.localizations
                        if item.entity_id != proposal.entity_id
                    ),
                ),
                key=lambda item: item.entity_id,
            )
        )

        relations = self._state.relations
        if target is not LocalizationStatus.LOCALIZED:
            invalidated: list[RelationState] = []
            for relation in relations:
                if proposal.entity_id not in {
                    relation.subject_entity_id,
                    relation.target_entity_id,
                }:
                    invalidated.append(relation)
                    continue
                invalidated.append(
                    replace(
                        relation,
                        value=RelationValue.UNKNOWN,
                        source_observation_id=proposal.source_observation_id,
                        source_observation_revision=proposal.source_observation_revision,
                        source_observation_domain=proposal.source_observation_domain,
                        supporting_evidence=proposal.evidence_refs,
                        action_id=proposal.action_id,
                        revision=relation.revision + 1,
                    )
                )
            relations = tuple(invalidated)
        return replace(
            self._state,
            revision=self._state.revision + 1,
            localizations=localizations,
            relations=relations,
        )

    def _set_relation(
        self,
        proposal: StateTransitionProposal,
        *,
        resolved_evidence: tuple[ResolvedArtifact, ...],
    ) -> EpisodeEmbodiedState:
        predicate = proposal.relation_predicate
        value = proposal.relation_value
        if not isinstance(predicate, RelationPredicate) or not isinstance(value, RelationValue):
            raise StateTransitionRejected(
                "Relation proposal must use closed predicate and value enums."
            )
        if (
            proposal.semantic_label
            or proposal.track_id
            or proposal.attachment_status is not None
            or proposal.geometry_ref is not None
            or proposal.localization_status is not None
            or proposal.world_pose is not None
            or proposal.trigger is not PhysicalStateTrigger.EVIDENCE
        ):
            raise StateTransitionRejected(
                "Relation proposal carries fields owned by another transition kind."
            )
        if not proposal.target_entity_id:
            raise StateTransitionRejected("Relation proposal requires target_entity_id.")
        if proposal.entity_id == proposal.target_entity_id:
            raise StateTransitionRejected("Relation subject and target must be distinct entities.")
        subject = self._state.entity(proposal.entity_id)
        target = self._state.entity(proposal.target_entity_id)
        if subject is None or target is None:
            raise StateTransitionRejected(
                "Relation subject and target must both be registered episode entities."
            )
        self._validate_track_identity(subject, proposal.subject_track_id, role="relation subject")
        self._validate_track_identity(target, proposal.target_track_id, role="relation target")
        if not proposal.source_observation_id:
            raise StateTransitionRejected("Relation proposal requires source_observation_id.")
        self._validate_observation_identity_fields(
            proposal,
            require_complete=self.strict_evidence,
        )
        if self.strict_evidence:
            self._require_relation_evidence(
                proposal,
                subject=subject,
                target=target,
                predicate=predicate,
                value=value,
                resolved_evidence=resolved_evidence,
            )

        key = (proposal.entity_id, predicate, proposal.target_entity_id)
        current = next((relation for relation in self._state.relations if relation.key == key), None)
        relation = RelationState(
            subject_entity_id=proposal.entity_id,
            predicate=predicate,
            target_entity_id=proposal.target_entity_id,
            value=value,
            source_observation_id=proposal.source_observation_id,
            source_observation_revision=proposal.source_observation_revision,
            source_observation_domain=proposal.source_observation_domain,
            supporting_evidence=proposal.evidence_refs,
            action_id=proposal.action_id,
            revision=(current.revision if current is not None else 0) + 1,
        )
        relations = tuple(
            sorted(
                (relation, *(item for item in self._state.relations if item.key != key)),
                key=lambda item: (
                    item.subject_entity_id,
                    item.predicate.value,
                    item.target_entity_id,
                ),
            )
        )
        return replace(
            self._state,
            revision=self._state.revision + 1,
            relations=relations,
        )

    @staticmethod
    def _validate_observation_identity_fields(
        proposal: StateTransitionProposal,
        *,
        require_complete: bool = False,
    ) -> None:
        """Reject half-bound observation clocks while allowing legacy ID-only data."""

        has_revision = proposal.source_observation_revision is not None
        has_domain = bool(proposal.source_observation_domain)
        if not has_revision and not has_domain:
            if require_complete:
                raise StateTransitionRejected(
                    "Strict state evidence requires source observation domain and revision."
                )
            return
        if (
            not proposal.source_observation_id
            or not has_revision
            or not has_domain
            or not isinstance(proposal.source_observation_revision, int)
            or isinstance(proposal.source_observation_revision, bool)
            or proposal.source_observation_revision < 0
            or not proposal.source_observation_domain.startswith("camera.")
        ):
            raise StateTransitionRejected(
                "Observation identity requires a camera domain, non-negative revision, and ID."
            )

    def _require_attachment_evidence(
        self,
        proposal: StateTransitionProposal,
        *,
        entity: EntityState,
        target: AttachmentStatus,
        action_id: str,
        resolved_evidence: tuple[ResolvedArtifact, ...],
    ) -> None:
        if not entity.track_id or proposal.subject_track_id != entity.track_id:
            raise StateTransitionRejected(
                "Strict attachment evidence requires the registered entity track identity."
            )

        def mismatches(evidence: AttachmentEvidence) -> list[str]:
            differences = self._base_evidence_mismatches(evidence, proposal)
            if evidence.entity_id != entity.entity_id:
                differences.append("entity_id")
            if evidence.entity_track_id != entity.track_id:
                differences.append("entity_track_id")
            if evidence.status != target.value:
                differences.append("status")
            if evidence.action_id != action_id:
                differences.append("action_id")
            return differences

        self._require_matching_typed_evidence(
            proposal,
            resolved_evidence=resolved_evidence,
            schema="robomex.attachment_evidence.v1",
            model=AttachmentEvidence,
            mismatches=mismatches,
        )

    def _require_localization_evidence(
        self,
        proposal: StateTransitionProposal,
        *,
        entity: EntityState,
        target: LocalizationStatus,
        resolved_evidence: tuple[ResolvedArtifact, ...],
    ) -> None:
        if not entity.track_id or proposal.subject_track_id != entity.track_id:
            raise StateTransitionRejected(
                "Strict localization evidence requires the registered entity track identity."
            )

        def mismatches(evidence: LocalizationEvidence) -> list[str]:
            differences = self._base_evidence_mismatches(evidence, proposal)
            if evidence.entity_id != entity.entity_id:
                differences.append("entity_id")
            if evidence.entity_track_id != entity.track_id:
                differences.append("entity_track_id")
            if evidence.status != target.value:
                differences.append("status")
            evidence_pose = (
                evidence.world_pose.model_dump(mode="json")
                if evidence.world_pose is not None
                else None
            )
            proposal_pose = (
                proposal.world_pose.model_dump(mode="json")
                if proposal.world_pose is not None
                else None
            )
            if evidence_pose != proposal_pose:
                differences.append("world_pose")
            return differences

        self._require_matching_typed_evidence(
            proposal,
            resolved_evidence=resolved_evidence,
            schema="robomex.localization_evidence.v1",
            model=LocalizationEvidence,
            mismatches=mismatches,
        )

    def _require_relation_evidence(
        self,
        proposal: StateTransitionProposal,
        *,
        subject: EntityState,
        target: EntityState,
        predicate: RelationPredicate,
        value: RelationValue,
        resolved_evidence: tuple[ResolvedArtifact, ...],
    ) -> None:
        if (
            not subject.track_id
            or not target.track_id
            or proposal.subject_track_id != subject.track_id
            or proposal.target_track_id != target.track_id
        ):
            raise StateTransitionRejected(
                "Strict relation evidence requires both registered track identities."
            )

        def mismatches(evidence: RelationEvidence) -> list[str]:
            differences = self._base_evidence_mismatches(evidence, proposal)
            expected = {
                "subject_entity_id": subject.entity_id,
                "subject_track_id": subject.track_id,
                "predicate": predicate,
                "target_entity_id": target.entity_id,
                "target_track_id": target.track_id,
                "value": value,
            }
            for field, expected_value in expected.items():
                if getattr(evidence, field) != expected_value:
                    differences.append(field)
            return differences

        self._require_matching_typed_evidence(
            proposal,
            resolved_evidence=resolved_evidence,
            schema="robomex.relation_evidence.v1",
            model=RelationEvidence,
            mismatches=mismatches,
        )

    @staticmethod
    def _base_evidence_mismatches(
        evidence: AttachmentEvidence | LocalizationEvidence | RelationEvidence,
        proposal: StateTransitionProposal,
    ) -> list[str]:
        differences: list[str] = []
        expected_action_id = proposal.action_id or None
        expected = {
            "base_state_revision": proposal.before_revision,
            "source_observation_id": proposal.source_observation_id,
            "source_observation_revision": proposal.source_observation_revision,
            "observation_domain": proposal.source_observation_domain,
            "action_id": expected_action_id,
        }
        for field, expected_value in expected.items():
            if getattr(evidence, field) != expected_value:
                differences.append(field)
        return differences

    def _require_matching_typed_evidence(
        self,
        proposal: StateTransitionProposal,
        *,
        resolved_evidence: tuple[ResolvedArtifact, ...],
        schema: str,
        model: type[AttachmentEvidence | LocalizationEvidence | RelationEvidence],
        mismatches: Any,
    ) -> None:
        candidates = tuple(
            artifact for artifact in resolved_evidence if artifact.schema == schema
        )
        if not candidates:
            raise StateTransitionRejected(
                f"Strict transition requires typed evidence schema {schema!r}."
            )
        proposal_ids = {ref.artifact_id for ref in proposal.evidence_refs}
        candidate_mismatches: list[str] = []
        for artifact in candidates:
            try:
                evidence = model.model_validate(artifact.payload)
            except (TypeError, ValueError) as exc:
                raise StateTransitionRejected(
                    f"Typed evidence {artifact.ref.artifact_id!r} is malformed."
                ) from exc
            differences = list(mismatches(evidence))
            claimed_ids = set(evidence.evidence_refs)
            if not claimed_ids.issubset(proposal_ids):
                differences.append("evidence_refs")
            try:
                lineage = tuple(
                    ResolvedArtifactRef.from_any(item)
                    for item in artifact.record.get("lineage", ())
                )
            except ValueError as exc:
                raise StateTransitionRejected(
                    f"Typed evidence {artifact.ref.artifact_id!r} has malformed lineage."
                ) from exc
            lineage_by_id = {ref.artifact_id: ref.content_digest for ref in lineage}
            proposal_by_id = {
                ref.artifact_id: ref.content_digest for ref in proposal.evidence_refs
            }
            if claimed_ids != set(lineage_by_id):
                differences.append("evidence_lineage")
            elif any(
                proposal_by_id.get(artifact_id) != digest
                for artifact_id, digest in lineage_by_id.items()
            ):
                differences.append("evidence_lineage_digest")
            if not differences:
                return
            candidate_mismatches.extend(differences)
        fields = ", ".join(sorted(set(candidate_mismatches)))
        raise StateTransitionRejected(
            f"Typed evidence does not bind the proposed transition: {fields}."
        )

    @staticmethod
    def _invalidate_entity_dependents(
        state: EpisodeEmbodiedState,
        proposal: StateTransitionProposal,
    ) -> EpisodeEmbodiedState:
        """Invalidate held-object geometry and predicates in the same commit."""

        localizations = tuple(
            replace(
                item,
                status=LocalizationStatus.UNKNOWN,
                world_pose=None,
                source_observation_id=proposal.source_observation_id,
                source_observation_revision=proposal.source_observation_revision,
                source_observation_domain=proposal.source_observation_domain,
                supporting_evidence=proposal.evidence_refs,
                action_id=proposal.action_id,
                revision=item.revision + 1,
            )
            if item.entity_id == proposal.entity_id
            else item
            for item in state.localizations
        )
        relations = tuple(
            replace(
                item,
                value=RelationValue.UNKNOWN,
                source_observation_id=proposal.source_observation_id,
                source_observation_revision=proposal.source_observation_revision,
                source_observation_domain=proposal.source_observation_domain,
                supporting_evidence=proposal.evidence_refs,
                action_id=proposal.action_id,
                revision=item.revision + 1,
            )
            if proposal.entity_id
            in {item.subject_entity_id, item.target_entity_id}
            else item
            for item in state.relations
        )
        return replace(state, localizations=localizations, relations=relations)

    @staticmethod
    def _validate_track_identity(entity: EntityState, claimed: str, *, role: str) -> None:
        if not claimed:
            return
        _validate_id(f"{role}_track_id", claimed)
        if not entity.track_id or claimed != entity.track_id:
            raise StateTransitionRejected(
                f"Wrong {role} identity: track {claimed!r} does not match registered "
                f"entity {entity.entity_id!r}."
            )

    def _replay_events(
        self,
        *,
        validate_evidence: bool,
        require_head: bool,
    ) -> None:
        if self._events:
            raise ArtifactIntegrityError("State replay requires an empty view.")
        inspection = inspect_ledger_for_replay(
            log_path=self.event_log_path,
            head_path=self.ledger_head_path,
            pending_path=self.ledger_pending_path,
            episode_id=self.episode_id,
            ledger=_LEDGER_NAME,
            event_schema=_STATE_EVENT_SCHEMA,
            require_head=require_head,
        )
        raw = inspection.raw
        head = inspection.head
        if raw and not raw.endswith(b"\n"):
            raise ArtifactIntegrityError("State event log has a partial durable tail.")
        digest = LEDGER_DIGEST_GENESIS
        byte_count = 0
        anchored = head is None or (
            head.event_count == 0
            and head.byte_count == 0
            and head.last_event_digest == LEDGER_DIGEST_GENESIS
        )
        for line_number, raw_line in enumerate(raw.splitlines(keepends=True), start=1):
            try:
                line = raw_line.removesuffix(b"\n").decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ArtifactIntegrityError(
                    f"Invalid UTF-8 state event at line {line_number}."
                ) from exc
            if not line:
                raise ArtifactIntegrityError(f"Blank state event at line {line_number}.")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactIntegrityError(f"Invalid state event at line {line_number}.") from exc
            try:
                canonical_event = _canonical_json(event)
            except StateTransitionRejected as exc:
                raise ArtifactIntegrityError(
                    f"State event {line_number} is not canonical JSON."
                ) from exc
            if line != canonical_event:
                raise ArtifactIntegrityError(
                    f"Non-canonical state event at line {line_number}."
                )
            expected = len(self._events) + 1
            if not isinstance(event, dict) or (
                event.get("schema") != _STATE_EVENT_SCHEMA
                or event.get("episode_id") != self.episode_id
                or event.get("sequence") != expected
                or event.get("event_id") != f"state_evt:{self.episode_id}:{expected}"
                or event.get("kind") != "state_transition_committed"
            ):
                raise ArtifactIntegrityError(
                    f"State event identity mismatch at sequence {expected}."
                )
            raw_proposal = event.get("proposal")
            if not isinstance(raw_proposal, Mapping):
                raise ArtifactIntegrityError("State event has no proposal.")
            proposal = StateTransitionProposal.from_mapping(raw_proposal)
            try:
                next_state = self._reduce(proposal, validate_evidence=validate_evidence)
            except StateTransitionRejected as exc:
                raise ArtifactIntegrityError(
                    f"State event {expected} cannot be deterministically replayed."
                ) from exc
            if event.get("after_revision") != next_state.revision or event.get(
                "state_digest"
            ) != _state_digest(next_state):
                raise ArtifactIntegrityError(
                    f"State event {expected} has an invalid resulting digest."
                )
            self._state = next_state
            self._effect_ids.add(proposal.effect_id)
            self._events.append(event)
            digest = next_ledger_digest(digest, canonical_event)
            byte_count += len(raw_line)
            if head is not None and expected == head.event_count:
                if byte_count != head.byte_count or digest != head.last_event_digest:
                    raise ArtifactIntegrityError(
                        "State ledger head does not match its committed prefix."
                    )
                anchored = True

        if head is not None and head.event_count > len(self._events):
            raise ArtifactIntegrityError("State event log was truncated below its durable head.")
        if head is not None and not anchored:
            raise ArtifactIntegrityError("State ledger head prefix is unavailable.")

        self._ledger_digest = digest
        self._ledger_byte_count = byte_count
        replayed_head = LedgerHead(
            episode_id=self.episode_id,
            ledger=_LEDGER_NAME,
            event_schema=_STATE_EVENT_SCHEMA,
            event_count=len(self._events),
            byte_count=byte_count,
            last_event_digest=digest,
        )
        if head is None:
            write_ledger_head(self.ledger_head_path, replayed_head)
        else:
            finalize_ledger_recovery(
                head_path=self.ledger_head_path,
                pending_path=self.ledger_pending_path,
                inspection=inspection,
                replayed_head=replayed_head,
            )
        self._log_identity = self._current_log_identity()

    def _synchronize_locked(self) -> None:
        identity = self._current_log_identity()
        pending_exists = self.ledger_pending_path.exists()
        head = load_ledger_head(
            self.ledger_head_path,
            episode_id=self.episode_id,
            ledger=_LEDGER_NAME,
            event_schema=_STATE_EVENT_SCHEMA,
            required=True,
        )
        if identity == self._log_identity and not pending_exists:
            if (
                head is None
                or head.event_count != len(self._events)
                or head.byte_count != self._ledger_byte_count
                or head.last_event_digest != self._ledger_digest
            ):
                raise ArtifactIntegrityError(
                    "State ledger head changed without a matching journal append."
                )
            return

        snapshot = (self._state, set(self._effect_ids), list(self._events))
        cursor = (self._ledger_digest, self._ledger_byte_count, self._log_identity)
        try:
            self._reset_view()
            self._replay_events(validate_evidence=True, require_head=True)
            self._write_index()
        except BaseException:
            self._state, self._effect_ids, self._events = snapshot
            self._ledger_digest, self._ledger_byte_count, self._log_identity = cursor
            raise

    def _write_head(self, *, event_count: int, byte_count: int, digest: str) -> None:
        write_ledger_head(
            self.ledger_head_path,
            LedgerHead(
                episode_id=self.episode_id,
                ledger=_LEDGER_NAME,
                event_schema=_STATE_EVENT_SCHEMA,
                event_count=event_count,
                byte_count=byte_count,
                last_event_digest=digest,
            ),
        )

    def _cursor_head(self) -> LedgerHead:
        return LedgerHead(
            episode_id=self.episode_id,
            ledger=_LEDGER_NAME,
            event_schema=_STATE_EVENT_SCHEMA,
            event_count=len(self._events),
            byte_count=self._ledger_byte_count,
            last_event_digest=self._ledger_digest,
        )

    def _write_pending(self, pending: LedgerPendingAppend) -> None:
        write_pending_append(self.ledger_pending_path, pending)

    def _append_journal_bytes(self, encoded: bytes) -> None:
        with self.event_log_path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def _clear_pending(self) -> None:
        clear_pending_append(self.ledger_pending_path)

    def _current_log_identity(self) -> tuple[int, int, int, int]:
        try:
            stat = self.event_log_path.stat()
        except OSError as exc:
            raise ArtifactIntegrityError("State event log is unreadable.") from exc
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _initialize_manifest(self) -> bool:
        expected_identity = {"schema": _MANIFEST_SCHEMA, "episode_id": self.episode_id}
        expected_policy = "strict_typed" if self.strict_evidence else "legacy_compatible"
        if self.manifest_path.exists():
            try:
                actual = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactIntegrityError("Episode manifest is unreadable.") from exc
            if not isinstance(actual, Mapping) or any(
                actual.get(key) != value for key, value in expected_identity.items()
            ):
                raise ArtifactIntegrityError(
                    "Episode root is already bound to a different identity or schema."
                )
            if set(actual) - {
                "schema",
                "episode_id",
                "durability",
                "state_evidence_policy",
            }:
                raise ArtifactIntegrityError("Episode manifest has unknown identity fields.")
            durability = actual.get("durability", {})
            if not isinstance(durability, Mapping) or set(durability) - _KNOWN_DURABILITY_KEYS:
                raise ArtifactIntegrityError("Episode manifest durability policy is malformed.")
            if any(value != _DURABILITY_MODE for value in durability.values()):
                raise ArtifactIntegrityError("Episode manifest durability mode is unsupported.")
            bound_policy = actual.get("state_evidence_policy")
            if bound_policy not in {None, "strict_typed", "legacy_compatible"}:
                raise ArtifactIntegrityError("Episode state evidence policy is malformed.")
            if bound_policy is not None and bound_policy != expected_policy:
                raise ArtifactIntegrityError(
                    "Episode embodied-state evidence policy cannot be rebound."
                )
            return durability.get(_DURABILITY_KEY) == _DURABILITY_MODE
        atomic_write_json(
            self.manifest_path,
            {**expected_identity, "durability": {}},
        )
        return False

    def _mark_manifest_durable(self) -> None:
        expected_policy = "strict_typed" if self.strict_evidence else "legacy_compatible"
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("Episode manifest is unreadable.") from exc
        if not isinstance(manifest, Mapping):
            raise ArtifactIntegrityError("Episode manifest must be a mapping.")
        durability = manifest.get("durability", {})
        if not isinstance(durability, Mapping):
            raise ArtifactIntegrityError("Episode manifest durability policy is malformed.")
        updated = dict(manifest)
        updated["state_evidence_policy"] = expected_policy
        updated["durability"] = {**dict(durability), _DURABILITY_KEY: _DURABILITY_MODE}
        atomic_write_json(self.manifest_path, updated)

    def _write_index(self) -> None:
        index = {
            "schema": _STATE_INDEX_SCHEMA,
            "episode_id": self.episode_id,
            "evidence_policy": (
                "strict_typed" if self.strict_evidence else "legacy_compatible"
            ),
            "event_count": len(self._events),
            "state_digest": _state_digest(self._state),
            "state": self._state.to_mapping(),
            "applied_effect_ids": sorted(self._effect_ids),
        }
        atomic_write_json(self.index_path, index)

    def _reset_view(self) -> None:
        self._state = EpisodeEmbodiedState(episode_id=self.episode_id)
        self._effect_ids.clear()
        self._events.clear()
        self._ledger_digest = LEDGER_DIGEST_GENESIS
        self._ledger_byte_count = 0
        self._log_identity = None


def _state_digest(state: EpisodeEmbodiedState) -> str:
    encoded = _canonical_json(state.to_mapping()).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _validate_id(label: str, value: str) -> None:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise StateTransitionRejected(f"{label} must match {_ID_RE.pattern!r}; got {value!r}.")


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
        raise StateTransitionRejected("State records must be finite canonical JSON data.") from exc
