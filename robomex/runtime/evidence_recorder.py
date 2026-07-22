"""Content-addressed frame and video evidence for sealed physical actions.

The action runner intentionally knows only the tiny ``record``/``finalize``
surface declared by :class:`robomex.runtime.authority.ActionEvidenceRecorder`.
This module provides the durable implementation of that surface.  Evidence is
published through :class:`~robomex.data.episode_plane.EpisodeDataPlane`; action
or phase strings are never interpreted as filesystem paths.

Every monitor sample and sealed-runtime boundary becomes a typed, immutable
frame artifact.  ``finalize`` then publishes either an encoded video supplied
by a deterministic encoder or a deterministic video manifest whose lineage
contains every frame.  The
activation/port identities are derived from the action identity and sequence,
so ``publish_once`` makes retries idempotent and rejects rebinding.  A new
recorder can discover those artifacts after a process restart and finish (or
return) the same evidence chain without a private mutable checkpoint.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from robomex.data.artifact_resolver import (
    ArtifactIntegrityError,
    ResolvedArtifact,
    ResolvedArtifactRef,
    compute_content_digest,
)
from robomex.data.episode_plane import ArtifactRecord, EpisodeDataPlane
from robomex.data.schema_registry import SchemaRegistry

ACTION_EVIDENCE_FRAME_SCHEMA = "robomex.action_evidence_frame.v1"
ACTION_EVIDENCE_VIDEO_SCHEMA = "robomex.action_evidence_video.v1"
ACTION_RUNTIME_SAMPLE_SCHEMA = "robomex.action_runtime_sample.v1"

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DigestStr = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
MediaTypeStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$",
    ),
]

_REF_PREFIX = "robomex-artifact-ref-v1"
_ACTIVATION_PREFIX = "action_evidence_"
_FRAME_PORT_RE = re.compile(r"^frame_(?P<sequence>[0-9]{12})$")
_VIDEO_PORT = "action_video"
_MONITOR_SAMPLE_DIGEST_SCHEMA = "robomex.monitor_sample.v1"


class EvidenceRecorderError(RuntimeError):
    """Base error for malformed or conflicting action evidence."""


class EvidenceAlreadyFinalizedError(EvidenceRecorderError):
    """A caller attempted to append a frame after the immutable finalization."""


class EvidenceChainIntegrityError(EvidenceRecorderError):
    """Persisted evidence does not form the expected action-scoped chain."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class EvidenceArtifactIdentity(_StrictModel):
    """Wire-safe identity embedded in a video manifest."""

    artifact_id: NonEmptyStr
    content_digest: DigestStr

    @property
    def ref(self) -> ResolvedArtifactRef:
        return ResolvedArtifactRef(self.artifact_id, self.content_digest)

    @classmethod
    def from_ref(cls, ref: ResolvedArtifactRef) -> EvidenceArtifactIdentity:
        return cls(artifact_id=ref.artifact_id, content_digest=ref.content_digest)


class BinaryEvidencePayload(_StrictModel):
    """Self-validating binary payload stored inside a JSON content object."""

    media_type: MediaTypeStr
    encoding: Literal["base64"] = "base64"
    byte_count: int = Field(ge=0)
    byte_digest: DigestStr
    data_base64: str
    codec: NonEmptyStr | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_binary(self) -> BinaryEvidencePayload:
        try:
            raw = base64.b64decode(self.data_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("data_base64 is not canonical base64") from exc
        if base64.b64encode(raw).decode("ascii") != self.data_base64:
            raise ValueError("data_base64 is not canonical base64")
        if len(raw) != self.byte_count:
            raise ValueError("byte_count does not match decoded data")
        actual = "sha256:" + hashlib.sha256(raw).hexdigest()
        if actual != self.byte_digest:
            raise ValueError("byte_digest does not match decoded data")
        return self

    def decode(self) -> bytes:
        """Return bytes after model validation has checked count and digest."""

        return base64.b64decode(self.data_base64, validate=True)


class ActionRuntimeEvidenceSample(_StrictModel):
    """Closed runtime boundary sample emitted independently of monitor cadence."""

    schema_version: Literal["robomex.action_runtime_sample.v1"] = (
        "robomex.action_runtime_sample.v1"
    )
    sample_type: Literal[
        "execution_pre_primitive",
        "primitive_post",
        "primitive_exception",
        "execution_terminal",
        "runner_exception",
    ]
    action_id: NonEmptyStr
    spec_digest: DigestStr
    world_id: NonEmptyStr
    resource_id: NonEmptyStr
    primitive: Literal["execute_joint_path", "set_gripper", "wait"] | None = None
    primitive_args_digest: DigestStr | None = None
    execution_snapshot_digest: DigestStr | None = None
    runtime_status: Literal[
        "rejected",
        "completed",
        "partial",
        "interrupted",
        "unknown",
        "indeterminate_after_crash",
        "indeterminate_after_timeout",
    ] | None = None
    abort_reason: str | None = None
    converged: bool | None = None
    interrupted: bool | None = None
    timed_out: bool | None = None
    error_type: NonEmptyStr | None = None
    error_message: str | None = None
    telemetry: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_boundary(self) -> ActionRuntimeEvidenceSample:
        if (
            self.sample_type == "execution_pre_primitive"
            and self.execution_snapshot_digest is None
        ):
            raise ValueError("pre-primitive sample requires execution snapshot digest")
        if self.sample_type in {
            "execution_pre_primitive",
            "primitive_post",
            "primitive_exception",
        } and (self.primitive is None or self.primitive_args_digest is None):
            raise ValueError(
                "primitive boundary sample requires primitive name and argument digest"
            )
        if self.sample_type == "execution_terminal" and (
            (self.primitive is None) != (self.primitive_args_digest is None)
        ):
            raise ValueError(
                "terminal sample must supply primitive and argument digest together"
            )
        if (
            self.sample_type in {"primitive_exception", "runner_exception"}
            and self.error_type is None
        ):
            raise ValueError("exception boundary sample requires error_type")
        if self.sample_type == "execution_terminal" and self.runtime_status is None:
            raise ValueError("terminal sample requires runtime_status")
        return self


class ActionEvidenceFrame(_StrictModel):
    """One ordered evidence sample, optionally carrying encoded image bytes."""

    schema_version: Literal["robomex.action_evidence_frame.v1"] = (
        "robomex.action_evidence_frame.v1"
    )
    action_id: NonEmptyStr
    sequence: int = Field(ge=1)
    phase: str = Field(min_length=1, max_length=128)
    sample: dict[str, JsonValue]
    sample_digest: DigestStr
    encoder_id: NonEmptyStr | None = None
    media: BinaryEvidencePayload | None = None

    @model_validator(mode="after")
    def _validate_encoder_binding(self) -> ActionEvidenceFrame:
        if (self.encoder_id is None) != (self.media is None):
            raise ValueError("encoder_id and media must be supplied together")
        return self

    @field_validator("phase")
    @classmethod
    def _phase_has_no_control_characters(cls, value: str) -> str:
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("phase cannot contain control characters")
        return value

    @model_validator(mode="after")
    def _validate_sample_digest(self) -> ActionEvidenceFrame:
        if self.sample.get("schema_version") == ACTION_RUNTIME_SAMPLE_SCHEMA:
            ActionRuntimeEvidenceSample.model_validate(self.sample)
        expected = compute_content_digest(_sample_digest_schema(self.sample), self.sample)
        if self.sample_digest != expected:
            raise ValueError("sample_digest does not match sample")
        return self


class ActionEvidenceVideo(_StrictModel):
    """Encoded action video or its dependency-free deterministic manifest."""

    schema_version: Literal["robomex.action_evidence_video.v1"] = (
        "robomex.action_evidence_video.v1"
    )
    action_id: NonEmptyStr
    representation: Literal["encoded_video", "deterministic_manifest"]
    encoder_id: NonEmptyStr
    source_lineage: tuple[EvidenceArtifactIdentity, ...]
    frame_refs: tuple[EvidenceArtifactIdentity, ...]
    frame_sequences: tuple[int, ...]
    frame_phases: tuple[str, ...]
    media: BinaryEvidencePayload | None = None
    fallback_reason: Literal["encoder_not_configured", "dependency_unavailable"] | None = None

    @model_validator(mode="after")
    def _validate_representation(self) -> ActionEvidenceVideo:
        count = len(self.frame_refs)
        if len(self.frame_sequences) != count or len(self.frame_phases) != count:
            raise ValueError("video manifest frame fields must have equal lengths")
        if tuple(sorted(self.frame_sequences)) != self.frame_sequences:
            raise ValueError("video frame sequences must be sorted")
        if len(set(self.frame_sequences)) != count:
            raise ValueError("video frame sequences must be unique")
        if self.representation == "encoded_video":
            if self.media is None:
                raise ValueError("encoded_video requires media")
            if not self.media.media_type.startswith("video/"):
                raise ValueError("encoded_video media_type must be video/*")
            if self.fallback_reason is not None:
                raise ValueError("encoded_video cannot carry a fallback reason")
        else:
            if self.media is not None:
                raise ValueError("deterministic_manifest cannot carry encoded media")
            if self.fallback_reason is None:
                raise ValueError("deterministic_manifest requires a fallback reason")
        return self


ACTION_EVIDENCE_SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    ACTION_EVIDENCE_FRAME_SCHEMA: ActionEvidenceFrame,
    ACTION_EVIDENCE_VIDEO_SCHEMA: ActionEvidenceVideo,
    ACTION_RUNTIME_SAMPLE_SCHEMA: ActionRuntimeEvidenceSample,
}


def install_action_evidence_schemas(registry: SchemaRegistry) -> None:
    """Install evidence validators before a strict data plane replays its ledger."""

    for schema_id, model in ACTION_EVIDENCE_SCHEMA_MODELS.items():
        registry.ensure(schema_id, model)


@dataclass(frozen=True)
class EncodedEvidenceMedia:
    """Encoder output accepted by the recorder.

    Encoders must be deterministic for identical ordered inputs.  The data
    plane's ``publish_once`` boundary detects violations on a retry/restart.
    """

    media_type: str
    data: bytes
    codec: str | None = None
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceFrameInput:
    """Verified frame material passed to an optional action-video encoder."""

    ref: ResolvedArtifactRef
    frame: ActionEvidenceFrame


@runtime_checkable
class EvidenceFrameEncoder(Protocol):
    """Optional encoder for visual bytes associated with one raw sample."""

    @property
    def encoder_id(self) -> str: ...

    def encode(
        self,
        *,
        action_id: str,
        sequence: int,
        phase: str,
        sample: Mapping[str, object],
    ) -> EncodedEvidenceMedia | None: ...


@runtime_checkable
class EvidenceVideoEncoder(Protocol):
    """Optional deterministic encoder for an ordered action-frame sequence."""

    @property
    def encoder_id(self) -> str: ...

    def encode(
        self,
        *,
        action_id: str,
        frames: Sequence[EvidenceFrameInput],
    ) -> EncodedEvidenceMedia | None: ...


SampleSerializer: TypeAlias = Callable[  # noqa: UP040 - Python 3.10 compatibility
    [Mapping[str, object]], Mapping[str, JsonValue]
]
LineageProvider: TypeAlias = Callable[  # noqa: UP040 - Python 3.10 compatibility
    [str], Iterable[ResolvedArtifactRef | Mapping[str, str]]
]


def encode_evidence_ref(ref: ResolvedArtifactRef) -> str:
    """Encode artifact ID *and* digest for ``ExecutionReceipt`` string fields."""

    if "|" in ref.artifact_id:
        raise ValueError("artifact_id cannot contain the evidence-ref separator")
    return f"{_REF_PREFIX}|{ref.artifact_id}|{ref.content_digest}"


def decode_evidence_ref(value: str) -> ResolvedArtifactRef:
    """Decode and strictly validate a receipt evidence reference."""

    if not isinstance(value, str):
        raise ValueError("evidence reference must be a string")
    parts = value.split("|")
    if len(parts) != 3 or parts[0] != _REF_PREFIX:
        raise ValueError("unsupported evidence reference")
    ref = ResolvedArtifactRef(parts[1], parts[2])
    if encode_evidence_ref(ref) != value:
        raise ValueError("evidence reference is not canonical")
    return ref


class EpisodeActionEvidenceRecorder:
    """Durable action evidence recorder backed by one episode data plane.

    ``workflow_id`` must already be open.  ``base_lineage`` normally contains
    the admitted action-spec and monitor-program artifacts.  A provider is
    useful when one recorder serves many actions; after the first frame, the
    persisted frame lineage itself is sufficient for restart recovery.
    """

    def __init__(
        self,
        data_plane: EpisodeDataPlane,
        *,
        workflow_id: str,
        publication_attempt: int = 1,
        base_lineage: Iterable[ResolvedArtifactRef | Mapping[str, str]] = (),
        lineage_provider: LineageProvider | None = None,
        sample_serializer: SampleSerializer | None = None,
        frame_encoder: EvidenceFrameEncoder | None = None,
        video_encoder: EvidenceVideoEncoder | None = None,
    ) -> None:
        if not isinstance(data_plane, EpisodeDataPlane):
            raise TypeError("data_plane must be an EpisodeDataPlane")
        if not isinstance(workflow_id, str) or not workflow_id.strip():
            raise ValueError("workflow_id must not be empty")
        if (
            not isinstance(publication_attempt, int)
            or isinstance(publication_attempt, bool)
            or publication_attempt < 1
        ):
            raise ValueError("publication_attempt must be a positive integer")
        install_action_evidence_schemas(data_plane.schema_registry)
        self.data_plane = data_plane
        self.workflow_id = workflow_id
        self.publication_attempt = publication_attempt
        self._base_lineage = self._normalize_refs(base_lineage)
        self._lineage_provider = lineage_provider
        self._sample_serializer = sample_serializer or _strict_json_sample
        self._frame_encoder = frame_encoder
        self._video_encoder = video_encoder
        self._lock = threading.RLock()

    def record(
        self,
        *,
        action_id: str,
        sequence: int,
        phase: str,
        sample: Mapping[str, object],
    ) -> str | None:
        """Idempotently append one ordered frame and return its digest-bound ref."""

        action_id = _validate_action_id(action_id)
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise ValueError("sequence must be a positive integer")
        if sequence > 999_999_999_999:
            raise ValueError("sequence exceeds the evidence port bound")
        if not isinstance(phase, str):
            raise ValueError("phase must be a string")
        if not isinstance(sample, Mapping):
            raise ValueError("sample must be a mapping")

        with self._lock:
            if self._video_record(action_id) is not None:
                raise EvidenceAlreadyFinalizedError(
                    f"action {action_id!r} evidence has already been finalized"
                )
            existing_frames = self._load_frames(action_id)
            lineage = self._lineage_for_action(action_id, existing_frames)
            normalized_sample = _normalize_json_mapping(self._sample_serializer(sample))
            encoded_media = None
            if self._frame_encoder is not None:
                encoded_media = self._frame_encoder.encode(
                    action_id=action_id,
                    sequence=sequence,
                    phase=phase,
                    sample=sample,
                )
            payload = ActionEvidenceFrame(
                action_id=action_id,
                sequence=sequence,
                phase=phase,
                sample=normalized_sample,
                sample_digest=compute_content_digest(
                    _sample_digest_schema(normalized_sample), normalized_sample
                ),
                encoder_id=(
                    _validate_encoder_id(self._frame_encoder.encoder_id)
                    if encoded_media is not None and self._frame_encoder is not None
                    else None
                ),
                media=(
                    _binary_payload(encoded_media) if encoded_media is not None else None
                ),
            )
            record = self.data_plane.publish_once(
                workflow_id=self.workflow_id,
                activation_id=_activation_id(action_id),
                attempt=self.publication_attempt,
                port=_frame_port(sequence),
                schema=ACTION_EVIDENCE_FRAME_SCHEMA,
                payload=payload.model_dump(mode="json"),
                lineage=lineage,
            )
            return encode_evidence_ref(record.ref)

    def finalize(self, *, action_id: str) -> tuple[tuple[str, ...], str | None]:
        """Seal the ordered frame set into one idempotent video artifact.

        When no encoder is configured, or an optional encoder dependency is
        unavailable, ``video_ref`` still identifies a typed deterministic
        manifest.  It is therefore safe for callers to put both return values
        directly into an :class:`ExecutionReceipt`.
        """

        action_id = _validate_action_id(action_id)
        with self._lock:
            existing_video = self._video_record(action_id)
            if existing_video is not None:
                return self._load_finalized(action_id, existing_video)

            frames = self._load_frames(action_id)
            base_lineage = self._lineage_for_action(action_id, frames)
            frame_refs = tuple(item.ref for item in frames)
            encoded: EncodedEvidenceMedia | None = None
            fallback_reason: str | None = None
            encoder_id = "robomex.deterministic_manifest.v1"
            if self._video_encoder is None:
                fallback_reason = "encoder_not_configured"
            else:
                encoder_id = _validate_encoder_id(self._video_encoder.encoder_id)
                try:
                    encoded = self._video_encoder.encode(
                        action_id=action_id,
                        frames=frames,
                    )
                except (ImportError, ModuleNotFoundError):
                    fallback_reason = "dependency_unavailable"
                if encoded is None and fallback_reason is None:
                    fallback_reason = "dependency_unavailable"

            payload = ActionEvidenceVideo(
                action_id=action_id,
                representation=(
                    "encoded_video" if encoded is not None else "deterministic_manifest"
                ),
                encoder_id=encoder_id,
                source_lineage=tuple(
                    EvidenceArtifactIdentity.from_ref(ref) for ref in base_lineage
                ),
                frame_refs=tuple(EvidenceArtifactIdentity.from_ref(ref) for ref in frame_refs),
                frame_sequences=tuple(item.frame.sequence for item in frames),
                frame_phases=tuple(item.frame.phase for item in frames),
                media=_binary_payload(encoded) if encoded is not None else None,
                fallback_reason=fallback_reason,
            )
            video_record = self.data_plane.publish_once(
                workflow_id=self.workflow_id,
                activation_id=_activation_id(action_id),
                attempt=self.publication_attempt,
                port=_VIDEO_PORT,
                schema=ACTION_EVIDENCE_VIDEO_SCHEMA,
                payload=payload.model_dump(mode="json"),
                lineage=_dedupe_refs((*base_lineage, *frame_refs)),
            )
            return (
                tuple(encode_evidence_ref(ref) for ref in frame_refs),
                encode_evidence_ref(video_record.ref),
            )

    def resolve_receipt_ref(self, value: str) -> ResolvedArtifact:
        """Resolve a frame/video receipt ref through the owning data plane."""

        return self.data_plane.resolve(decode_evidence_ref(value))

    def _lineage_for_action(
        self, action_id: str, frames: Sequence[EvidenceFrameInput]
    ) -> tuple[ResolvedArtifactRef, ...]:
        persisted: tuple[ResolvedArtifactRef, ...] | None = None
        for item in frames:
            record = self.data_plane.artifact_record(item.ref.artifact_id)
            if persisted is None:
                persisted = record.lineage
            elif record.lineage != persisted:
                raise EvidenceChainIntegrityError(
                    "one action's frame artifacts have inconsistent base lineage"
                )

        configured = self._base_lineage
        if self._lineage_provider is not None:
            configured = _dedupe_refs(
                (*configured, *self._normalize_refs(self._lineage_provider(action_id)))
            )
        if persisted is not None:
            if configured and configured != persisted:
                raise EvidenceChainIntegrityError(
                    "configured lineage differs from persisted frame lineage"
                )
            return persisted
        return configured

    def _load_frames(self, action_id: str) -> tuple[EvidenceFrameInput, ...]:
        activation_id = _activation_id(action_id)
        found: list[EvidenceFrameInput] = []
        sequences: set[int] = set()
        for record in self.data_plane.artifacts:
            if (
                record.workflow_id != self.workflow_id
                or record.activation_id != activation_id
                or record.schema != ACTION_EVIDENCE_FRAME_SCHEMA
            ):
                continue
            if record.attempt != self.publication_attempt:
                raise EvidenceChainIntegrityError(
                    "frame artifact uses a different publication attempt"
                )
            match = _FRAME_PORT_RE.fullmatch(record.port)
            if match is None:
                raise EvidenceChainIntegrityError("frame artifact uses an invalid evidence port")
            resolved = self.data_plane.resolve(record.ref)
            frame = ActionEvidenceFrame.model_validate(resolved.payload)
            if frame.action_id != action_id:
                raise EvidenceChainIntegrityError("hashed activation contains another action")
            if frame.sequence != int(match.group("sequence")):
                raise EvidenceChainIntegrityError("frame sequence differs from its evidence port")
            if frame.sequence in sequences:
                raise EvidenceChainIntegrityError("action contains duplicate frame sequences")
            sequences.add(frame.sequence)
            found.append(EvidenceFrameInput(ref=record.ref, frame=frame))
        found.sort(key=lambda item: item.frame.sequence)
        return tuple(found)

    def _video_record(self, action_id: str) -> ArtifactRecord | None:
        activation_id = _activation_id(action_id)
        records = [
            record
            for record in self.data_plane.artifacts
            if record.workflow_id == self.workflow_id
            and record.activation_id == activation_id
            and record.port == _VIDEO_PORT
        ]
        if len(records) > 1:
            raise EvidenceChainIntegrityError("action contains multiple video finalizations")
        if not records:
            return None
        record = records[0]
        if record.attempt != self.publication_attempt:
            raise EvidenceChainIntegrityError(
                "video artifact uses a different publication attempt"
            )
        if record.schema != ACTION_EVIDENCE_VIDEO_SCHEMA:
            raise EvidenceChainIntegrityError("action video port is bound to another schema")
        return record

    def _load_finalized(
        self, action_id: str, record: ArtifactRecord
    ) -> tuple[tuple[str, ...], str]:
        resolved = self.data_plane.resolve(record.ref)
        video = ActionEvidenceVideo.model_validate(resolved.payload)
        if video.action_id != action_id:
            raise EvidenceChainIntegrityError("video finalization belongs to another action")
        frames = self._load_frames(action_id)
        actual_refs = tuple(item.ref for item in frames)
        claimed_refs = tuple(item.ref for item in video.frame_refs)
        if claimed_refs != actual_refs:
            raise EvidenceChainIntegrityError("video manifest does not bind the persisted frames")
        claimed_source_lineage = tuple(item.ref for item in video.source_lineage)
        if frames:
            base_lineage = self._lineage_for_action(action_id, frames)
            if claimed_source_lineage != base_lineage:
                raise EvidenceChainIntegrityError(
                    "video manifest does not bind the persisted source lineage"
                )
        else:
            configured = self._lineage_for_action(action_id, frames)
            if configured and configured != claimed_source_lineage:
                raise EvidenceChainIntegrityError(
                    "video manifest source lineage differs from configured lineage"
                )
            base_lineage = claimed_source_lineage
        expected_lineage = _dedupe_refs((*base_lineage, *actual_refs))
        if record.lineage != expected_lineage:
            raise EvidenceChainIntegrityError("video artifact lineage is incomplete or reordered")
        if video.frame_sequences != tuple(item.frame.sequence for item in frames):
            raise EvidenceChainIntegrityError("video manifest sequence index is inconsistent")
        if video.frame_phases != tuple(item.frame.phase for item in frames):
            raise EvidenceChainIntegrityError("video manifest phase index is inconsistent")
        return (
            tuple(encode_evidence_ref(ref) for ref in actual_refs),
            encode_evidence_ref(record.ref),
        )

    def _normalize_refs(
        self, values: Iterable[ResolvedArtifactRef | Mapping[str, str]]
    ) -> tuple[ResolvedArtifactRef, ...]:
        refs: list[ResolvedArtifactRef] = []
        for value in values:
            ref = ResolvedArtifactRef.from_any(value)
            self.data_plane.resolve(ref)
            refs.append(ref)
        return _dedupe_refs(refs)


def _activation_id(action_id: str) -> str:
    digest = hashlib.sha256(action_id.encode("utf-8")).hexdigest()
    return _ACTIVATION_PREFIX + digest[:40]


def _frame_port(sequence: int) -> str:
    return f"frame_{sequence:012d}"


def _validate_action_id(action_id: str) -> str:
    if not isinstance(action_id, str) or not action_id.strip():
        raise ValueError("action_id must not be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in action_id):
        raise ValueError("action_id cannot contain control characters")
    return action_id.strip()


def _validate_encoder_id(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise EvidenceRecorderError("encoder_id must not be empty")
    return normalized


def _binary_payload(value: EncodedEvidenceMedia) -> BinaryEvidencePayload:
    if not isinstance(value, EncodedEvidenceMedia):
        raise TypeError("evidence encoders must return EncodedEvidenceMedia or None")
    if not isinstance(value.data, bytes):
        raise TypeError("encoded evidence data must be bytes")
    metadata = _normalize_json_mapping(value.metadata)
    return BinaryEvidencePayload(
        media_type=value.media_type,
        byte_count=len(value.data),
        byte_digest="sha256:" + hashlib.sha256(value.data).hexdigest(),
        data_base64=base64.b64encode(value.data).decode("ascii"),
        codec=value.codec,
        metadata=metadata,
    )


def _strict_json_sample(sample: Mapping[str, object]) -> Mapping[str, JsonValue]:
    return sample  # type: ignore[return-value] - checked by canonical JSON round-trip below


def _sample_digest_schema(sample: Mapping[str, object]) -> str:
    if sample.get("schema_version") == ACTION_RUNTIME_SAMPLE_SCHEMA:
        return ACTION_RUNTIME_SAMPLE_SCHEMA
    return _MONITOR_SAMPLE_DIGEST_SCHEMA


def _normalize_json_mapping(value: Mapping[str, object]) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ValueError("serialized evidence must be a mapping")
    _require_string_mapping_keys(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "evidence sample is not strict JSON; configure sample_serializer for array data"
        ) from exc
    if not isinstance(normalized, dict) or any(not isinstance(key, str) for key in normalized):
        raise ValueError("serialized evidence must have string keys")
    return normalized


def _require_string_mapping_keys(value: object) -> None:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("serialized evidence mappings must have string keys")
        for item in value.values():
            _require_string_mapping_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_string_mapping_keys(item)


def _dedupe_refs(values: Iterable[ResolvedArtifactRef]) -> tuple[ResolvedArtifactRef, ...]:
    result: list[ResolvedArtifactRef] = []
    seen: set[tuple[str, str]] = set()
    for ref in values:
        identity = (ref.artifact_id, ref.content_digest)
        if identity not in seen:
            seen.add(identity)
            result.append(ref)
    return tuple(result)


__all__ = [
    "ACTION_EVIDENCE_FRAME_SCHEMA",
    "ACTION_EVIDENCE_SCHEMA_MODELS",
    "ACTION_EVIDENCE_VIDEO_SCHEMA",
    "ACTION_RUNTIME_SAMPLE_SCHEMA",
    "ActionEvidenceFrame",
    "ActionEvidenceVideo",
    "ActionRuntimeEvidenceSample",
    "BinaryEvidencePayload",
    "EncodedEvidenceMedia",
    "EpisodeActionEvidenceRecorder",
    "EvidenceAlreadyFinalizedError",
    "EvidenceArtifactIdentity",
    "EvidenceChainIntegrityError",
    "EvidenceFrameEncoder",
    "EvidenceFrameInput",
    "EvidenceRecorderError",
    "EvidenceVideoEncoder",
    "decode_evidence_ref",
    "encode_evidence_ref",
    "install_action_evidence_schemas",
]
