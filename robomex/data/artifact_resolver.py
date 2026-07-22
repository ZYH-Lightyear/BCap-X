"""Fail-closed resolution for episode-scoped, content-addressed artifacts.

The v2 data plane deliberately does not accept filesystem paths as artifact
references.  Consumers must present the immutable artifact identity together
with the content digest that was frozen at activation admission.  The resolver
then derives the only allowed object path from that digest and re-validates the
content before returning it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CANONICALIZATION = "robomex-json-v1"
_DIGEST_DOMAIN = b"robomex-digest-v1\0"


class ArtifactResolutionError(ValueError):
    """Raised when an artifact cannot be resolved without weakening safety."""


class ArtifactIntegrityError(ArtifactResolutionError):
    """Raised when an index, path, or content digest is inconsistent."""


@dataclass(frozen=True)
class ResolvedArtifactRef:
    """An execution-safe artifact reference frozen by admission."""

    artifact_id: str
    content_digest: str

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must be non-empty")
        if not _DIGEST_RE.fullmatch(self.content_digest):
            raise ValueError("content_digest must use canonical 'sha256:<64 lowercase hex>' form")

    @classmethod
    def from_any(cls, value: Any) -> ResolvedArtifactRef:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ArtifactResolutionError(
                "Artifact resolution requires artifact_id + content_digest; "
                "paths and symbolic aliases are not accepted."
            )
        artifact_id = value.get("artifact_id")
        content_digest = value.get("content_digest") or value.get("payload_digest")
        if not isinstance(artifact_id, str) or not isinstance(content_digest, str):
            raise ArtifactResolutionError(
                "Artifact resolution requires string artifact_id and content_digest."
            )
        return cls(artifact_id=artifact_id, content_digest=content_digest)

    def to_mapping(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True)
class ResolvedArtifact:
    """Verified artifact content returned to a consumer."""

    ref: ResolvedArtifactRef
    schema: str
    payload: Any
    record: Mapping[str, Any] = field(repr=False)


def canonical_content_document(schema: str, payload: Any) -> dict[str, Any]:
    """Return the semantic document covered by an artifact content digest.

    IDs, generations, wall-clock timestamps, and producer metadata are
    intentionally excluded.  Two publications of the same typed semantic
    payload therefore share a content object while retaining distinct
    append-only artifact identities.
    """

    if not isinstance(schema, str) or not schema.strip():
        raise ValueError("schema must be a non-empty string")
    document = {
        "canonicalization": _CANONICALIZATION,
        "schema": schema,
        "payload": payload,
    }
    # Validate JSON compatibility and reject NaN/Inf before anything is written.
    _canonical_json_bytes(document)
    return document


def canonical_content_bytes(schema: str, payload: Any) -> bytes:
    """Return the domain-separated canonical bytes used for SHA-256."""

    return _DIGEST_DOMAIN + _canonical_json_bytes(canonical_content_document(schema, payload))


def compute_content_digest(schema: str, payload: Any) -> str:
    """Compute a deterministic digest for a typed semantic payload."""

    digest = hashlib.sha256(canonical_content_bytes(schema, payload)).hexdigest()
    return f"sha256:{digest}"


def content_object_relative_path(content_digest: str) -> Path:
    """Map a canonical digest to the sole permitted object-store path."""

    if not _DIGEST_RE.fullmatch(content_digest):
        raise ValueError("invalid canonical content digest")
    hex_digest = content_digest.removeprefix("sha256:")
    return Path("objects") / "sha256" / f"{hex_digest}.json"


class ArtifactResolver:
    """Resolve immutable refs within exactly one episode root.

    The resolver reloads the derived index for every call.  This keeps a
    long-lived resolver usable while the append-only ledger publishes new
    artifacts, without making the index authoritative: ``EpisodeDataPlane``
    can always rebuild it from the event log.
    """

    def __init__(
        self,
        episode_root: str | Path,
        *,
        episode_id: str,
        index_name: str = "artifact_index.v2.json",
    ) -> None:
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        self.episode_root = Path(episode_root).resolve()
        self.episode_id = episode_id
        self.index_path = self.episode_root / index_name

    def resolve(
        self,
        ref: ResolvedArtifactRef | Mapping[str, Any] | str | Path,
        content_digest: str | None = None,
    ) -> ResolvedArtifact:
        """Resolve and verify one artifact.

        ``content_digest`` is accepted only to make the explicit
        ``resolve(artifact_id, digest)`` spelling convenient.  Supplying an ID
        without a digest, a symbolic alias, or a path always fails closed.
        """

        if isinstance(ref, Path):
            raise ArtifactResolutionError("Filesystem paths are not artifact refs.")
        if isinstance(ref, str):
            if content_digest is None:
                raise ArtifactResolutionError(
                    "Artifact resolution requires both artifact_id and content_digest."
                )
            resolved_ref = ResolvedArtifactRef(ref, content_digest)
        else:
            if content_digest is not None:
                raise ArtifactResolutionError("content_digest must not be supplied twice.")
            resolved_ref = ResolvedArtifactRef.from_any(ref)

        index = self._load_index()
        records = index.get("artifacts")
        if not isinstance(records, dict):
            raise ArtifactIntegrityError("Artifact index has no artifact mapping.")
        record = records.get(resolved_ref.artifact_id)
        if not isinstance(record, dict):
            raise ArtifactResolutionError(f"Unknown artifact_id {resolved_ref.artifact_id!r}.")

        self._validate_record_identity(record, resolved_ref)
        relative_path = record.get("content_path")
        if not isinstance(relative_path, str):
            raise ArtifactIntegrityError("Artifact record has no controlled content_path.")
        object_path = self._controlled_object_path(relative_path, resolved_ref.content_digest)
        try:
            raw = object_path.read_bytes()
            document = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                f"Artifact content object is unreadable: {object_path!s}."
            ) from exc
        if not isinstance(document, dict):
            raise ArtifactIntegrityError("Artifact content object must be a JSON object.")
        if document.get("canonicalization") != _CANONICALIZATION:
            raise ArtifactIntegrityError("Unsupported artifact canonicalization.")

        schema = record.get("schema")
        if not isinstance(schema, str) or document.get("schema") != schema:
            raise ArtifactIntegrityError("Artifact schema differs from its index record.")
        actual_digest = compute_content_digest(schema, document.get("payload"))
        if actual_digest != resolved_ref.content_digest:
            raise ArtifactIntegrityError(
                "Artifact content digest mismatch; refusing corrupted or rebound content."
            )
        return ResolvedArtifact(
            ref=resolved_ref,
            schema=schema,
            payload=copy.deepcopy(document.get("payload")),
            record=copy.deepcopy(record),
        )

    def _load_index(self) -> dict[str, Any]:
        try:
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactResolutionError(
                f"Artifact index is unavailable: {self.index_path!s}."
            ) from exc
        if not isinstance(index, dict):
            raise ArtifactIntegrityError("Artifact index must be a JSON object.")
        if index.get("episode_id") != self.episode_id:
            raise ArtifactResolutionError("Artifact index belongs to a different episode.")
        return index

    def _validate_record_identity(
        self, record: Mapping[str, Any], ref: ResolvedArtifactRef
    ) -> None:
        if record.get("artifact_id") != ref.artifact_id:
            raise ArtifactIntegrityError("Artifact index key and record ID disagree.")
        if record.get("episode_id") != self.episode_id:
            raise ArtifactResolutionError("Cross-episode artifact resolution is forbidden.")
        if _episode_from_artifact_id(ref.artifact_id) != self.episode_id:
            raise ArtifactResolutionError("Artifact ID belongs to a different episode.")
        if record.get("content_digest") != ref.content_digest:
            raise ArtifactIntegrityError(
                "Resolved digest does not match the immutable artifact record."
            )

    def _controlled_object_path(self, relative: str, digest: str) -> Path:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "" in pure.parts:
            raise ArtifactIntegrityError("Artifact content path escapes episode scope.")
        expected = content_object_relative_path(digest).as_posix()
        if pure.as_posix() != expected:
            raise ArtifactIntegrityError(
                "Artifact content path is not the digest-derived controlled path."
            )
        candidate = (self.episode_root / Path(*pure.parts)).resolve()
        try:
            candidate.relative_to(self.episode_root)
        except ValueError as exc:
            raise ArtifactIntegrityError("Artifact content path escapes episode scope.") from exc
        if not candidate.is_file():
            raise ArtifactIntegrityError(f"Artifact content object is missing: {candidate!s}.")
        return candidate


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Artifact content must be finite, canonical JSON data.") from exc
    return rendered.encode("utf-8")


def _episode_from_artifact_id(artifact_id: str) -> str:
    parts = artifact_id.split(":")
    if len(parts) != 7 or parts[0] != "art":
        raise ArtifactIntegrityError(f"Malformed v2 artifact_id {artifact_id!r}.")
    return parts[1]
