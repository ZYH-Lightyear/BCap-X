"""Episode-scoped append-only artifact ledger for RoboMEx v2.

This module is intentionally narrower than a general blackboard.  It stores
typed computation artifacts, workflow-local aliases, and immutable input
admissions.  Authoritative embodied state lives in ``embodied_state.py`` and
can only be changed through its deterministic reducer.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robomex.data.artifact_resolver import (
    ArtifactIntegrityError,
    ArtifactResolutionError,
    ArtifactResolver,
    ResolvedArtifact,
    ResolvedArtifactRef,
    canonical_content_document,
    compute_content_digest,
    content_object_relative_path,
)
from robomex.data.durability import (
    LEDGER_DIGEST_GENESIS,
    LedgerHead,
    LedgerPendingAppend,
    atomic_write_bytes,
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
    AdmissionPurpose,
    FreshnessRejected,
    RevisionVector,
    ValidityVector,
)
from robomex.data.schema_registry import (
    SchemaPayloadError,
    SchemaRegistry,
    UnregisteredSchemaError,
    core_schema_registry,
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_ALIAS_RE = re.compile(
    r"^(?P<activation>[A-Za-z0-9][A-Za-z0-9_-]{0,127})\."
    r"(?P<port>[A-Za-z0-9][A-Za-z0-9_-]{0,127})$"
)
_EVENT_SCHEMA = "robomex.episode_artifact_event.v2"
_INDEX_SCHEMA = "robomex.artifact_index.v2"
_MANIFEST_SCHEMA = "robomex.episode_manifest.v2"
_LEDGER_NAME = "artifact_events"
_LEDGER_HEAD_NAME = "artifact_events_head.v1.json"
_LEDGER_PENDING_NAME = "artifact_events_pending.v1.json"
_DURABILITY_KEY = "artifact_events"
_DURABILITY_MODE = "anchored_sha256_chain_v1"
_KNOWN_DURABILITY_KEYS = frozenset({"artifact_events", "embodied_state_events"})


class EpisodeDataPlaneError(ValueError):
    """Base error for invalid or unsafe data-plane operations."""


class WorkflowScopeError(EpisodeDataPlaneError):
    """Raised when a workflow-local operation crosses its lifecycle boundary."""


class AdmissionError(EpisodeDataPlaneError):
    """Raised when inputs cannot be frozen into immutable refs."""


@dataclass(frozen=True)
class AdmissionFreshnessContext:
    """Physical clock supplied by the consumer at input admission.

    The data plane never looks up a global "latest" value.  A caller must
    explicitly freeze the clock and observation/state identities against
    which every embedded :class:`ValidityVector` is checked.
    """

    current_revisions: RevisionVector
    purpose: AdmissionPurpose = AdmissionPurpose.VERIFICATION
    current_observation_id: str | None = None
    current_state_revision: int | None = None
    expected_action_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.current_revisions, RevisionVector):
            raise TypeError("current_revisions must be a canonical RevisionVector")
        if not isinstance(self.purpose, AdmissionPurpose):
            raise TypeError("purpose must use AdmissionPurpose")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "current_revisions": self.current_revisions.model_dump(mode="json"),
            "purpose": self.purpose.value,
            "current_observation_id": self.current_observation_id,
            "current_state_revision": self.current_state_revision,
            "expected_action_id": self.expected_action_id,
        }


@dataclass(frozen=True)
class FreshnessAdmission:
    """Auditable result of one artifact validity check."""

    name: str
    lifecycle: str
    checked_domains: tuple[str, ...]
    method: str
    reason: str
    confidence: float

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lifecycle": self.lifecycle,
            "checked_domains": list(self.checked_domains),
            "method": self.method,
            "reason": self.reason,
            "confidence": self.confidence,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FreshnessAdmission:
        try:
            return cls(
                name=str(value["name"]),
                lifecycle=str(value["lifecycle"]),
                checked_domains=tuple(str(item) for item in value["checked_domains"]),
                method=str(value["method"]),
                reason=str(value["reason"]),
                confidence=float(value["confidence"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("Malformed freshness admission record.") from exc


@dataclass(frozen=True)
class SymbolicArtifactRef:
    """A workflow-local alias that is legal only before admission."""

    alias: str

    def __post_init__(self) -> None:
        if not _ALIAS_RE.fullmatch(self.alias):
            raise ValueError("Symbolic artifact refs must use workflow-local 'activation.port'.")

    @classmethod
    def from_any(cls, value: Any) -> SymbolicArtifactRef:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) != {"$ref"}:
            raise AdmissionError("Symbolic bindings require exactly {'$ref': 'activation.port'}.")
        alias = value.get("$ref")
        if not isinstance(alias, str):
            raise AdmissionError("Symbolic $ref must be a string.")
        return cls(alias)

    def to_mapping(self) -> dict[str, str]:
        return {"$ref": self.alias}


@dataclass(frozen=True)
class ArtifactRecord:
    """Immutable causal envelope for one append-only publication."""

    artifact_id: str
    episode_id: str
    workflow_id: str
    activation_id: str
    attempt: int
    generation: int
    port: str
    schema: str
    content_digest: str
    content_path: str
    lineage: tuple[ResolvedArtifactRef, ...] = ()

    @property
    def alias(self) -> str:
        return f"{self.activation_id}.{self.port}"

    @property
    def ref(self) -> ResolvedArtifactRef:
        return ResolvedArtifactRef(self.artifact_id, self.content_digest)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "episode_id": self.episode_id,
            "workflow_id": self.workflow_id,
            "activation_id": self.activation_id,
            "attempt": self.attempt,
            "generation": self.generation,
            "port": self.port,
            "schema": self.schema,
            "content_digest": self.content_digest,
            "content_path": self.content_path,
            "lineage": [ref.to_mapping() for ref in self.lineage],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArtifactRecord:
        try:
            return cls(
                artifact_id=str(value["artifact_id"]),
                episode_id=str(value["episode_id"]),
                workflow_id=str(value["workflow_id"]),
                activation_id=str(value["activation_id"]),
                attempt=int(value["attempt"]),
                generation=int(value["generation"]),
                port=str(value["port"]),
                schema=str(value["schema"]),
                content_digest=str(value["content_digest"]),
                content_path=str(value["content_path"]),
                lineage=tuple(
                    ResolvedArtifactRef.from_any(item) for item in value.get("lineage", ())
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("Malformed artifact publication event.") from exc


@dataclass(frozen=True)
class ResolvedInput:
    name: str
    ref: ResolvedArtifactRef

    def to_mapping(self) -> dict[str, Any]:
        return {"name": self.name, "ref": self.ref.to_mapping()}


@dataclass(frozen=True)
class InputAdmission:
    """Frozen activation inputs; replay never re-evaluates their aliases."""

    admission_id: str
    workflow_id: str
    activation_id: str
    bindings: tuple[ResolvedInput, ...]
    freshness: tuple[FreshnessAdmission, ...] = ()

    def refs(self) -> dict[str, ResolvedArtifactRef]:
        return {binding.name: binding.ref for binding in self.bindings}

    def to_mapping(self) -> dict[str, Any]:
        return {
            "admission_id": self.admission_id,
            "workflow_id": self.workflow_id,
            "activation_id": self.activation_id,
            "bindings": [binding.to_mapping() for binding in self.bindings],
            "freshness": [item.to_mapping() for item in self.freshness],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> InputAdmission:
        try:
            bindings = tuple(
                ResolvedInput(
                    name=str(item["name"]),
                    ref=ResolvedArtifactRef.from_any(item["ref"]),
                )
                for item in value.get("bindings", ())
            )
            return cls(
                admission_id=str(value["admission_id"]),
                workflow_id=str(value["workflow_id"]),
                activation_id=str(value["activation_id"]),
                bindings=bindings,
                freshness=tuple(
                    FreshnessAdmission.from_mapping(item)
                    for item in value.get("freshness", ())
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("Malformed input admission event.") from exc


class EpisodeDataPlane:
    """One episode's append-only artifact/event boundary.

    ``episode_root`` is the root for exactly one episode, not a shared run
    directory.  A manifest prevents accidentally reopening it under another
    identity.
    """

    def __init__(
        self,
        episode_root: str | Path,
        *,
        episode_id: str,
        schema_registry: SchemaRegistry | None = None,
        strict_schema_prefixes: tuple[str, ...] = (),
    ) -> None:
        _validate_id("episode_id", episode_id)
        self.episode_root = Path(episode_root).resolve()
        self.episode_id = episode_id
        self.schema_registry = schema_registry or core_schema_registry()
        self.strict_schema_prefixes = tuple(
            sorted({str(prefix).strip() for prefix in strict_schema_prefixes if str(prefix).strip()})
        )
        self.event_log_path = self.episode_root / "artifact_events.jsonl"
        self.ledger_head_path = self.episode_root / _LEDGER_HEAD_NAME
        self.ledger_pending_path = self.episode_root / _LEDGER_PENDING_NAME
        self.index_path = self.episode_root / "artifact_index.v2.json"
        self.manifest_path = self.episode_root / "episode_manifest.v2.json"
        self._events: list[dict[str, Any]] = []
        self._artifacts: dict[str, ArtifactRecord] = {}
        self._aliases: dict[str, dict[str, str]] = {}
        self._workflows: dict[str, str] = {}
        self._admissions: dict[str, InputAdmission] = {}
        self._admission_requests: dict[str, dict[str, Any]] = {}
        self._generation: dict[tuple[str, str, str], int] = {}
        self._ledger_digest = LEDGER_DIGEST_GENESIS
        self._ledger_byte_count = 0
        self._log_identity: tuple[int, int, int, int] | None = None

        self.episode_root.mkdir(parents=True, exist_ok=True)
        with exclusive_episode_lock(self.episode_root):
            durability_initialized = self._initialize_manifest()
            self.event_log_path.touch(exist_ok=True)
            self._replay_events(require_head=durability_initialized)
            self._mark_manifest_durable()
            # The JSON index is a disposable materialized view of the event ledger.
            self._write_index()

    @property
    def resolver(self) -> ArtifactResolver:
        return ArtifactResolver(self.episode_root, episode_id=self.episode_id)

    @property
    def event_count(self) -> int:
        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            return len(self._events)

    @property
    def artifacts(self) -> tuple[ArtifactRecord, ...]:
        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            return tuple(self._artifacts.values())

    @property
    def workflows(self) -> dict[str, str]:
        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            return dict(self._workflows)

    def events(self) -> tuple[dict[str, Any], ...]:
        """Return a detached audit view of the replayable ledger."""

        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            return tuple(json.loads(json.dumps(event)) for event in self._events)

    def open_workflow(self, workflow_id: str) -> None:
        _validate_id("workflow_id", workflow_id)
        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            status = self._workflows.get(workflow_id)
            if status == "open":
                return
            if status == "closed":
                raise WorkflowScopeError(f"Closed workflow {workflow_id!r} cannot be reopened.")
            self._append_event("workflow_opened", {"workflow_id": workflow_id})

    def close_workflow(self, workflow_id: str) -> None:
        _validate_id("workflow_id", workflow_id)
        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            status = self._workflows.get(workflow_id)
            if status == "closed":
                return
            if status != "open":
                raise WorkflowScopeError(f"Workflow {workflow_id!r} is not open.")
            self._append_event("workflow_closed", {"workflow_id": workflow_id})

    def publish(
        self,
        *,
        workflow_id: str,
        activation_id: str,
        attempt: int,
        port: str,
        schema: str,
        payload: Mapping[str, Any],
        lineage: Iterable[ResolvedArtifactRef | Mapping[str, Any]] = (),
    ) -> ArtifactRecord:
        """Append one immutable artifact and advance its local alias generation."""

        _validate_id("workflow_id", workflow_id)
        _validate_id("activation_id", activation_id)
        _validate_id("port", port)
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise EpisodeDataPlaneError("attempt must be a positive integer")
        if not isinstance(schema, str) or not schema.strip():
            raise EpisodeDataPlaneError("schema must be a non-empty string")
        if not isinstance(payload, Mapping):
            raise EpisodeDataPlaneError(
                "Artifact payload must be a typed mapping, not a general blackboard value."
            )
        # Detach mutable caller data before digesting or retaining it, then
        # normalize registered schemas through their authoritative validator.
        detached_payload = json.loads(_canonical_json(payload))
        detached_payload = self._validate_schema_payload(schema, detached_payload)

        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            resolved_lineage = self._resolve_explicit_lineage(lineage)
            return self._publish_locked(
                workflow_id=workflow_id,
                activation_id=activation_id,
                attempt=attempt,
                port=port,
                schema=schema,
                payload=detached_payload,
                lineage=resolved_lineage,
            )

    def _publish_locked(
        self,
        *,
        workflow_id: str,
        activation_id: str,
        attempt: int,
        port: str,
        schema: str,
        payload: Mapping[str, Any],
        lineage: tuple[ResolvedArtifactRef, ...],
    ) -> ArtifactRecord:
        """Publish after the episode lock has synchronized the local view."""

        self._require_open_workflow(workflow_id)
        key = (workflow_id, activation_id, port)
        generation = self._generation.get(key, 0) + 1
        artifact_id = (
            f"art:{self.episode_id}:{workflow_id}:{activation_id}:"
            f"a{attempt}:g{generation}:{port}"
        )
        if artifact_id in self._artifacts:
            raise ArtifactIntegrityError(
                f"Artifact ID collision for append-only ID {artifact_id!r}."
            )

        digest = compute_content_digest(schema, payload)
        relative_path = content_object_relative_path(digest)
        self._write_content_object(
            relative_path,
            canonical_content_document(schema, payload),
            digest,
        )
        record = ArtifactRecord(
            artifact_id=artifact_id,
            episode_id=self.episode_id,
            workflow_id=workflow_id,
            activation_id=activation_id,
            attempt=attempt,
            generation=generation,
            port=port,
            schema=schema,
            content_digest=digest,
            content_path=relative_path.as_posix(),
            lineage=lineage,
        )
        self._append_event("artifact_published", {"artifact": record.to_mapping()})
        return record

    def validate_payload(self, schema: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Preflight a publication without changing the append-only ledger."""

        if not isinstance(schema, str) or not schema.strip():
            raise EpisodeDataPlaneError("schema must be a non-empty string")
        if not isinstance(payload, Mapping):
            raise EpisodeDataPlaneError("Artifact payload must be a typed mapping.")
        detached = json.loads(_canonical_json(payload))
        return self._validate_schema_payload(schema, detached)

    def publish_once(
        self,
        *,
        workflow_id: str,
        activation_id: str,
        attempt: int,
        port: str,
        schema: str,
        payload: Mapping[str, Any],
        lineage: Iterable[ResolvedArtifactRef | Mapping[str, Any]] = (),
    ) -> ArtifactRecord:
        """Idempotently publish the sole value of one activation attempt/port."""

        normalized_payload = self.validate_payload(schema, payload)
        digest = compute_content_digest(schema, normalized_payload)
        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            self._require_open_workflow(workflow_id)
            normalized_lineage = self._resolve_explicit_lineage(lineage)
            existing = [
                record
                for record in self._artifacts.values()
                if record.workflow_id == workflow_id
                and record.activation_id == activation_id
                and record.attempt == attempt
                and record.port == port
            ]
            if len(existing) > 1:
                raise ArtifactIntegrityError(
                    "Activation attempt/port has multiple append-only publications."
                )
            if existing:
                record = existing[0]
                if (
                    record.schema != schema
                    or record.content_digest != digest
                    or record.lineage != normalized_lineage
                ):
                    raise ArtifactIntegrityError(
                        "Activation attempt/port cannot be rebound to different content."
                    )
                return record
            return self._publish_locked(
                workflow_id=workflow_id,
                activation_id=activation_id,
                attempt=attempt,
                port=port,
                schema=schema,
                payload=normalized_payload,
                lineage=normalized_lineage,
            )

    def admit_inputs(
        self,
        *,
        admission_id: str,
        workflow_id: str,
        activation_id: str,
        bindings: Mapping[
            str,
            SymbolicArtifactRef | ResolvedArtifactRef | Mapping[str, Any],
        ],
        freshness_context: AdmissionFreshnessContext | None = None,
    ) -> InputAdmission:
        """Resolve aliases once and durably freeze activation inputs.

        Reusing the same admission ID with the same request is idempotent and
        returns the originally frozen refs even if a newer alias generation has
        since been published.  Rebinding an existing admission ID is rejected.
        """

        _validate_id("admission_id", admission_id)
        _validate_id("workflow_id", workflow_id)
        _validate_id("activation_id", activation_id)
        if not isinstance(bindings, Mapping):
            raise AdmissionError("bindings must be a mapping")
        normalized_bindings = self._normalize_binding_request(bindings)
        normalized_request = {
            "bindings": normalized_bindings,
            "freshness_context": (
                freshness_context.to_mapping() if freshness_context is not None else None
            ),
        }

        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            self._require_open_workflow(workflow_id)
            existing = self._admissions.get(admission_id)
            if existing is not None:
                if (
                    existing.workflow_id != workflow_id
                    or existing.activation_id != activation_id
                    or self._admission_requests.get(admission_id) != normalized_request
                ):
                    raise AdmissionError(
                        f"Admission {admission_id!r} is immutable and cannot be rebound."
                    )
                return existing

            resolved: list[ResolvedInput] = []
            freshness_results: list[FreshnessAdmission] = []
            for name in sorted(bindings):
                _validate_id("binding name", name)
                value = bindings[name]
                if isinstance(value, SymbolicArtifactRef) or (
                    isinstance(value, Mapping) and "$ref" in value
                ):
                    symbolic = SymbolicArtifactRef.from_any(value)
                    ref = self._resolve_local_alias(workflow_id, symbolic.alias)
                else:
                    ref = ResolvedArtifactRef.from_any(value)
                    self._resolve_ref_locked(ref)
                resolved.append(ResolvedInput(name=name, ref=ref))
                freshness = self._validate_admitted_freshness(
                    name=name,
                    ref=ref,
                    context=freshness_context,
                )
                if freshness is not None:
                    freshness_results.append(freshness)

            admission = InputAdmission(
                admission_id=admission_id,
                workflow_id=workflow_id,
                activation_id=activation_id,
                bindings=tuple(resolved),
                freshness=tuple(freshness_results),
            )
            self._append_event(
                "inputs_admitted",
                {
                    "admission": admission.to_mapping(),
                    "requested_bindings": normalized_bindings,
                    "freshness_context": normalized_request["freshness_context"],
                },
            )
            return admission

    def resolve(self, ref: ResolvedArtifactRef | Mapping[str, Any]) -> ResolvedArtifact:
        """Resolve an already-admitted explicit ref; aliases are not accepted."""

        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            return self._resolve_ref_locked(ref)

    def _resolve_ref_locked(
        self, ref: ResolvedArtifactRef | Mapping[str, Any]
    ) -> ResolvedArtifact:
        resolved = self.resolver.resolve(ref)
        self._validate_schema_payload(resolved.schema, resolved.payload)
        return resolved

    def artifact_record(self, artifact_id: str) -> ArtifactRecord:
        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            try:
                return self._artifacts[artifact_id]
            except KeyError as exc:
                raise ArtifactResolutionError(f"Unknown artifact_id {artifact_id!r}.") from exc

    def admission(self, admission_id: str) -> InputAdmission:
        with exclusive_episode_lock(self.episode_root):
            self._synchronize_locked()
            try:
                return self._admissions[admission_id]
            except KeyError as exc:
                raise AdmissionError(f"Unknown admission {admission_id!r}.") from exc

    def rebuild_index(self) -> Path:
        """Rebuild the disposable index solely from append-only events."""

        with exclusive_episode_lock(self.episode_root):
            self._reset_views()
            self._replay_events(require_head=True)
            self._write_index()
        return self.index_path

    def _resolve_explicit_lineage(
        self, lineage: Iterable[ResolvedArtifactRef | Mapping[str, Any]]
    ) -> tuple[ResolvedArtifactRef, ...]:
        resolved: list[ResolvedArtifactRef] = []
        seen: set[tuple[str, str]] = set()
        for value in lineage:
            ref = ResolvedArtifactRef.from_any(value)
            self.resolver.resolve(ref)
            identity = (ref.artifact_id, ref.content_digest)
            if identity not in seen:
                seen.add(identity)
                resolved.append(ref)
        return tuple(resolved)

    def _resolve_local_alias(self, workflow_id: str, alias: str) -> ResolvedArtifactRef:
        artifact_id = self._aliases.get(workflow_id, {}).get(alias)
        if artifact_id is None:
            raise AdmissionError(
                f"Workflow-local alias {alias!r} has no published value in "
                f"workflow {workflow_id!r}."
            )
        record = self._artifacts[artifact_id]
        ref = record.ref
        self._resolve_ref_locked(ref)
        return ref

    def _validate_schema_payload(
        self, schema: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Validate and normalize one payload under the installed registry."""

        if self.schema_registry.is_registered(schema):
            try:
                return self.schema_registry.validate_mapping(schema, payload)
            except SchemaPayloadError as exc:
                raise EpisodeDataPlaneError(str(exc)) from exc
        if any(schema.startswith(prefix) for prefix in self.strict_schema_prefixes):
            try:
                self.schema_registry.model_for(schema)
            except UnregisteredSchemaError as exc:
                raise EpisodeDataPlaneError(str(exc)) from exc
        return json.loads(_canonical_json(payload))

    def _validate_admitted_freshness(
        self,
        *,
        name: str,
        ref: ResolvedArtifactRef,
        context: AdmissionFreshnessContext | None,
    ) -> FreshnessAdmission | None:
        resolved = self._resolve_ref_locked(ref)
        if not isinstance(resolved.payload, Mapping):
            return None
        raw_validity = resolved.payload.get("validity")
        if raw_validity is None:
            return None
        if context is None:
            raise AdmissionError(
                f"Input {name!r} carries physical validity but no freshness context was supplied."
            )
        try:
            validity = ValidityVector.model_validate(raw_validity)
            decision = validity.assert_admissible(
                context.current_revisions,
                purpose=context.purpose,
                current_observation_id=context.current_observation_id,
                current_state_revision=context.current_state_revision,
                expected_action_id=context.expected_action_id,
            )
        except (TypeError, ValueError, FreshnessRejected) as exc:
            raise AdmissionError(f"Input {name!r} failed freshness admission: {exc}") from exc
        return FreshnessAdmission(
            name=name,
            lifecycle=decision.lifecycle.value,
            checked_domains=decision.checked_domains,
            method=decision.method,
            reason=decision.reason,
            confidence=decision.confidence,
        )

    def _normalize_binding_request(self, bindings: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for name in sorted(bindings):
            value = bindings[name]
            if isinstance(value, SymbolicArtifactRef) or (
                isinstance(value, Mapping) and "$ref" in value
            ):
                normalized[name] = SymbolicArtifactRef.from_any(value).to_mapping()
            else:
                normalized[name] = ResolvedArtifactRef.from_any(value).to_mapping()
        return normalized

    def _require_open_workflow(self, workflow_id: str) -> None:
        if self._workflows.get(workflow_id) != "open":
            raise WorkflowScopeError(f"Workflow {workflow_id!r} is not open.")

    def _append_event(self, kind: str, fields: Mapping[str, Any]) -> None:
        sequence = len(self._events) + 1
        event = {
            "schema": _EVENT_SCHEMA,
            "sequence": sequence,
            "event_id": f"evt:{self.episode_id}:{sequence}",
            "episode_id": self.episode_id,
            "kind": kind,
            **fields,
        }
        canonical_event = _canonical_json(event)
        encoded = (canonical_event + "\n").encode("utf-8")
        next_digest = next_ledger_digest(self._ledger_digest, canonical_event)
        next_byte_count = self._ledger_byte_count + len(encoded)
        previous_head = self._cursor_head()
        pending = build_pending_append(
            episode_id=self.episode_id,
            ledger=_LEDGER_NAME,
            event_schema=_EVENT_SCHEMA,
            previous_head=previous_head,
            canonical_event=canonical_event,
        )
        if (
            pending.next_head.event_count != sequence
            or pending.next_head.byte_count != next_byte_count
            or pending.next_head.last_event_digest != next_digest
        ):
            raise ArtifactIntegrityError("Pending artifact append did not bind the next head.")

        # Validate the derived mutation before making the event durable.  If a
        # filesystem operation then fails, restore the old view; the next sync
        # can either recover a fully written suffix or reject a partial one.
        snapshot = self._snapshot_views()
        try:
            self._apply_event(event)
            self._write_pending(pending)
            self._append_journal_bytes(encoded)
            self._write_head(
                event_count=sequence,
                byte_count=next_byte_count,
                digest=next_digest,
            )
            self._clear_pending()
        except BaseException:
            self._restore_views(snapshot)
            raise

        self._events.append(event)
        self._ledger_digest = next_digest
        self._ledger_byte_count = next_byte_count
        self._log_identity = self._current_log_identity()
        self._write_index()

    def _replay_events(self, *, require_head: bool) -> None:
        if self._events:
            raise ArtifactIntegrityError("Replay requires an empty in-memory view.")
        inspection = inspect_ledger_for_replay(
            log_path=self.event_log_path,
            head_path=self.ledger_head_path,
            pending_path=self.ledger_pending_path,
            episode_id=self.episode_id,
            ledger=_LEDGER_NAME,
            event_schema=_EVENT_SCHEMA,
            require_head=require_head,
        )
        raw = inspection.raw
        head = inspection.head
        if raw and not raw.endswith(b"\n"):
            raise ArtifactIntegrityError("Artifact event log has a partial durable tail.")
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
                    f"Invalid UTF-8 artifact event at line {line_number}."
                ) from exc
            if not line:
                raise ArtifactIntegrityError(f"Blank event at artifact log line {line_number}.")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactIntegrityError(
                    f"Invalid artifact event at line {line_number}."
                ) from exc
            if not isinstance(event, dict):
                raise ArtifactIntegrityError("Artifact event must be a JSON object.")
            try:
                canonical_event = _canonical_json(event)
            except EpisodeDataPlaneError as exc:
                raise ArtifactIntegrityError(
                    f"Artifact event {line_number} is not canonical JSON."
                ) from exc
            if line != canonical_event:
                raise ArtifactIntegrityError(
                    f"Non-canonical artifact event at line {line_number}."
                )
            expected = len(self._events) + 1
            if (
                event.get("schema") != _EVENT_SCHEMA
                or event.get("episode_id") != self.episode_id
                or event.get("sequence") != expected
                or event.get("event_id") != f"evt:{self.episode_id}:{expected}"
            ):
                raise ArtifactIntegrityError(
                    f"Artifact event identity mismatch at sequence {expected}."
                )
            self._apply_event(event)
            self._events.append(event)
            digest = next_ledger_digest(digest, canonical_event)
            byte_count += len(raw_line)
            if head is not None and expected == head.event_count:
                if byte_count != head.byte_count or digest != head.last_event_digest:
                    raise ArtifactIntegrityError(
                        "Artifact ledger head does not match its committed prefix."
                    )
                anchored = True

        if head is not None and head.event_count > len(self._events):
            raise ArtifactIntegrityError(
                "Artifact event log was truncated below its durable head."
            )
        if head is not None and not anchored:
            raise ArtifactIntegrityError("Artifact ledger head prefix is unavailable.")

        self._ledger_digest = digest
        self._ledger_byte_count = byte_count
        replayed_head = LedgerHead(
            episode_id=self.episode_id,
            ledger=_LEDGER_NAME,
            event_schema=_EVENT_SCHEMA,
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
        """Refresh a long-lived instance after another process commits."""

        identity = self._current_log_identity()
        pending_exists = self.ledger_pending_path.exists()
        head = load_ledger_head(
            self.ledger_head_path,
            episode_id=self.episode_id,
            ledger=_LEDGER_NAME,
            event_schema=_EVENT_SCHEMA,
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
                    "Artifact ledger head changed without a matching journal append."
                )
            return

        snapshot = self._snapshot_views(include_events=True)
        cursor = (self._ledger_digest, self._ledger_byte_count, self._log_identity)
        try:
            self._reset_views()
            self._replay_events(require_head=True)
            self._write_index()
        except BaseException:
            self._restore_views(snapshot, include_events=True)
            self._ledger_digest, self._ledger_byte_count, self._log_identity = cursor
            raise

    def _write_head(self, *, event_count: int, byte_count: int, digest: str) -> None:
        write_ledger_head(
            self.ledger_head_path,
            LedgerHead(
                episode_id=self.episode_id,
                ledger=_LEDGER_NAME,
                event_schema=_EVENT_SCHEMA,
                event_count=event_count,
                byte_count=byte_count,
                last_event_digest=digest,
            ),
        )

    def _cursor_head(self) -> LedgerHead:
        return LedgerHead(
            episode_id=self.episode_id,
            ledger=_LEDGER_NAME,
            event_schema=_EVENT_SCHEMA,
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
            raise ArtifactIntegrityError("Artifact event log is unreadable.") from exc
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _apply_event(self, event: Mapping[str, Any]) -> None:
        kind = event.get("kind")
        if kind == "workflow_opened":
            workflow_id = str(event.get("workflow_id") or "")
            _validate_id("workflow_id", workflow_id)
            if workflow_id in self._workflows:
                raise ArtifactIntegrityError("Workflow was opened more than once.")
            self._workflows[workflow_id] = "open"
            self._aliases[workflow_id] = {}
            return
        if kind == "workflow_closed":
            workflow_id = str(event.get("workflow_id") or "")
            if self._workflows.get(workflow_id) != "open":
                raise ArtifactIntegrityError("Only an open workflow can be closed.")
            self._workflows[workflow_id] = "closed"
            return
        if kind == "artifact_published":
            raw_record = event.get("artifact")
            if not isinstance(raw_record, Mapping):
                raise ArtifactIntegrityError("Artifact event has no record.")
            record = ArtifactRecord.from_mapping(raw_record)
            self._validate_replayed_record(record)
            self._artifacts[record.artifact_id] = record
            key = (record.workflow_id, record.activation_id, record.port)
            self._generation[key] = record.generation
            self._aliases[record.workflow_id][record.alias] = record.artifact_id
            return
        if kind == "inputs_admitted":
            raw_admission = event.get("admission")
            requested = event.get("requested_bindings")
            if not isinstance(raw_admission, Mapping) or not isinstance(requested, Mapping):
                raise ArtifactIntegrityError("Admission event is incomplete.")
            admission = InputAdmission.from_mapping(raw_admission)
            self._validate_replayed_admission(admission, requested)
            self._admissions[admission.admission_id] = admission
            self._admission_requests[admission.admission_id] = {
                "bindings": json.loads(_canonical_json(requested)),
                "freshness_context": json.loads(
                    _canonical_json(event.get("freshness_context"))
                ),
            }
            return
        raise ArtifactIntegrityError(f"Unknown artifact event kind {kind!r}.")

    def _validate_replayed_record(self, record: ArtifactRecord) -> None:
        if record.episode_id != self.episode_id:
            raise ArtifactIntegrityError("Published artifact crossed episode scope.")
        _validate_id("workflow_id", record.workflow_id)
        _validate_id("activation_id", record.activation_id)
        _validate_id("port", record.port)
        if record.attempt < 1 or record.generation < 1:
            raise ArtifactIntegrityError("Artifact attempt/generation must be positive.")
        if not record.schema.strip():
            raise ArtifactIntegrityError("Artifact schema must be non-empty.")
        if self._workflows.get(record.workflow_id) != "open":
            raise ArtifactIntegrityError("Artifact was published outside an open workflow.")
        key = (record.workflow_id, record.activation_id, record.port)
        expected_generation = self._generation.get(key, 0) + 1
        expected_id = (
            f"art:{self.episode_id}:{record.workflow_id}:{record.activation_id}:"
            f"a{record.attempt}:g{expected_generation}:{record.port}"
        )
        if record.generation != expected_generation or record.artifact_id != expected_id:
            raise ArtifactIntegrityError("Artifact generation or immutable ID is invalid.")
        if record.artifact_id in self._artifacts:
            raise ArtifactIntegrityError("Append-only artifact ID was duplicated.")
        expected_path = content_object_relative_path(record.content_digest).as_posix()
        if record.content_path != expected_path:
            raise ArtifactIntegrityError("Artifact has a non-canonical content path.")
        for ref in record.lineage:
            upstream = self._artifacts.get(ref.artifact_id)
            if upstream is None or upstream.content_digest != ref.content_digest:
                raise ArtifactIntegrityError("Artifact lineage was unresolved at publish time.")
        self._validate_replayed_content(record)

    def _validate_replayed_content(self, record: ArtifactRecord) -> None:
        path = (self.episode_root / record.content_path).resolve()
        try:
            path.relative_to(self.episode_root)
            document = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("Artifact content object is unreadable.") from exc
        if not isinstance(document, Mapping):
            raise ArtifactIntegrityError("Artifact content object must be a mapping.")
        payload = document.get("payload")
        if (
            document != canonical_content_document(record.schema, payload)
            or compute_content_digest(record.schema, payload) != record.content_digest
        ):
            raise ArtifactIntegrityError("Artifact content object failed canonical integrity checks.")
        if not isinstance(payload, Mapping):
            raise ArtifactIntegrityError("Artifact payload must remain a typed mapping.")
        try:
            self._validate_schema_payload(record.schema, payload)
        except EpisodeDataPlaneError as exc:
            raise ArtifactIntegrityError(str(exc)) from exc

    def _validate_replayed_admission(
        self,
        admission: InputAdmission,
        requested: Mapping[str, Any],
    ) -> None:
        _validate_id("admission_id", admission.admission_id)
        _validate_id("workflow_id", admission.workflow_id)
        _validate_id("activation_id", admission.activation_id)
        if admission.admission_id in self._admissions:
            raise ArtifactIntegrityError("Admission ID was reused in the event log.")
        if self._workflows.get(admission.workflow_id) != "open":
            raise ArtifactIntegrityError("Inputs were admitted outside an open workflow.")
        try:
            normalized_request = self._normalize_binding_request(requested)
        except (AdmissionError, ValueError) as exc:
            raise ArtifactIntegrityError("Admission request is malformed.") from exc
        names: set[str] = set()
        for binding in admission.bindings:
            _validate_id("binding name", binding.name)
            if binding.name in names:
                raise ArtifactIntegrityError("Admission contains duplicate binding names.")
            names.add(binding.name)
            record = self._artifacts.get(binding.ref.artifact_id)
            if record is None or record.content_digest != binding.ref.content_digest:
                raise ArtifactIntegrityError("Admission references an unknown artifact.")
            request = normalized_request.get(binding.name)
            if not isinstance(request, Mapping):
                raise ArtifactIntegrityError("Admission request/binding names disagree.")
            if "$ref" in request:
                alias_id = self._aliases.get(admission.workflow_id, {}).get(str(request["$ref"]))
                if alias_id != binding.ref.artifact_id:
                    raise ArtifactIntegrityError(
                        "Admission binding does not match its symbolic alias."
                    )
            elif ResolvedArtifactRef.from_any(request) != binding.ref:
                raise ArtifactIntegrityError(
                    "Admission binding does not match its explicit request."
                )
        if names != set(normalized_request):
            raise ArtifactIntegrityError("Admission request/binding names disagree.")
        freshness_names: set[str] = set()
        for item in admission.freshness:
            _validate_id("freshness binding name", item.name)
            if item.name in freshness_names or item.name not in names:
                raise ArtifactIntegrityError("Freshness admission names are invalid.")
            if not item.lifecycle or not item.method or not item.reason:
                raise ArtifactIntegrityError("Freshness admission evidence is incomplete.")
            if not 0.0 <= item.confidence <= 1.0:
                raise ArtifactIntegrityError("Freshness admission confidence is invalid.")
            freshness_names.add(item.name)

    def _write_content_object(
        self, relative: Path, document: Mapping[str, Any], digest: str
    ) -> None:
        path = (self.episode_root / relative).resolve()
        try:
            path.relative_to(self.episode_root)
        except ValueError as exc:
            raise ArtifactIntegrityError("Content-addressed path escaped episode root.") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (_canonical_json(document) + "\n").encode("utf-8")
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                existing_digest = compute_content_digest(
                    str(existing["schema"]), existing.get("payload")
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ArtifactIntegrityError(
                    "Existing content-addressed object is corrupt."
                ) from exc
            if existing_digest != digest or existing != dict(document):
                raise ArtifactIntegrityError("Content-addressed object does not match its digest.")
            return
        atomic_write_bytes(path, encoded)

    def _initialize_manifest(self) -> bool:
        """Validate identity and report whether this ledger requires a head."""

        expected_identity = {"schema": _MANIFEST_SCHEMA, "episode_id": self.episode_id}
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
            state_policy = actual.get("state_evidence_policy")
            if state_policy not in {None, "strict_typed", "legacy_compatible"}:
                raise ArtifactIntegrityError("Episode state evidence policy is malformed.")
            durability = actual.get("durability", {})
            if not isinstance(durability, Mapping) or set(durability) - _KNOWN_DURABILITY_KEYS:
                raise ArtifactIntegrityError("Episode manifest durability policy is malformed.")
            if any(value != _DURABILITY_MODE for value in durability.values()):
                raise ArtifactIntegrityError("Episode manifest durability mode is unsupported.")
            return durability.get(_DURABILITY_KEY) == _DURABILITY_MODE
        atomic_write_json(
            self.manifest_path,
            {**expected_identity, "durability": {}},
        )
        return False

    def _mark_manifest_durable(self) -> None:
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
        updated["durability"] = {**dict(durability), _DURABILITY_KEY: _DURABILITY_MODE}
        atomic_write_json(self.manifest_path, updated)

    def _write_index(self) -> None:
        index = {
            "schema": _INDEX_SCHEMA,
            "episode_id": self.episode_id,
            "event_count": len(self._events),
            "workflows": dict(sorted(self._workflows.items())),
            "artifacts": {
                artifact_id: record.to_mapping() for artifact_id, record in self._artifacts.items()
            },
            "aliases": {
                workflow_id: dict(sorted(aliases.items()))
                for workflow_id, aliases in sorted(self._aliases.items())
            },
            "admissions": {
                admission_id: admission.to_mapping()
                for admission_id, admission in self._admissions.items()
            },
        }
        atomic_write_json(self.index_path, index)

    def _reset_views(self) -> None:
        self._events.clear()
        self._artifacts.clear()
        self._aliases.clear()
        self._workflows.clear()
        self._admissions.clear()
        self._admission_requests.clear()
        self._generation.clear()
        self._ledger_digest = LEDGER_DIGEST_GENESIS
        self._ledger_byte_count = 0
        self._log_identity = None

    def _snapshot_views(self, *, include_events: bool = False) -> tuple[Any, ...]:
        return (
            list(self._events) if include_events else None,
            dict(self._artifacts),
            {workflow_id: dict(aliases) for workflow_id, aliases in self._aliases.items()},
            dict(self._workflows),
            dict(self._admissions),
            {
                admission_id: json.loads(_canonical_json(request))
                for admission_id, request in self._admission_requests.items()
            },
            dict(self._generation),
        )

    def _restore_views(
        self, snapshot: tuple[Any, ...], *, include_events: bool = False
    ) -> None:
        (
            events,
            artifacts,
            aliases,
            workflows,
            admissions,
            admission_requests,
            generation,
        ) = snapshot
        if include_events:
            self._events = events
        self._artifacts = artifacts
        self._aliases = aliases
        self._workflows = workflows
        self._admissions = admissions
        self._admission_requests = admission_requests
        self._generation = generation


def _validate_id(label: str, value: str) -> None:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise EpisodeDataPlaneError(f"{label} must match {_ID_RE.pattern!r}; got {value!r}.")


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
        raise EpisodeDataPlaneError(
            "Data-plane records must be finite canonical JSON data."
        ) from exc
