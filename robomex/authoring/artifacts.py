"""Single typed data plane for dynamic authoring agents."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from robomex.core.artifact_paths import resolve_artifact_ref_path
from robomex.core.context import ArtifactRef, compact_json
from robomex.core.payload_specs import PAYLOAD_SPECS


class StaleArtifactError(ValueError):
    """Typed rejection for artifacts observed before the current epoch.

    The graph runtime maps this exception type onto the ``stale_observation``
    edge event; keep it a distinct type so routing never depends on message
    text.
    """


@dataclass(frozen=True)
class PortSpec:
    name: str
    schema: str
    required: bool = True
    frame: str = ""

    @classmethod
    def from_mapping(cls, raw: str | dict[str, Any]) -> "PortSpec":
        if isinstance(raw, str):
            return cls(raw, f"robomex.{raw}.v1")
        return cls(
            name=str(raw.get("name") or ""),
            schema=str(raw.get("schema") or ""),
            required=bool(raw.get("required", True)),
            frame=str(raw.get("frame") or ""),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema": self.schema,
            "required": self.required,
            "frame": self.frame,
        }


@dataclass(frozen=True, order=True)
class ArtifactKey:
    producer: str
    port: str

    @classmethod
    def parse(cls, raw: str | dict[str, Any]) -> "ArtifactKey":
        path = str(raw.get("$ref") or "") if isinstance(raw, dict) else str(raw)
        producer, separator, port = path.partition(".")
        if not separator or not producer or not port or "." in port:
            raise ValueError(
                f"Artifact reference {path!r} must use '$ref': 'producer.port'."
            )
        return cls(producer, port)

    @property
    def path(self) -> str:
        return f"{self.producer}.{self.port}"

    def to_mapping(self) -> dict[str, str]:
        return {"$ref": self.path}


@dataclass(frozen=True)
class TypedArtifact:
    port: str
    schema: str
    producer: str
    payload: dict[str, Any] = field(default_factory=dict)
    refs: tuple[ArtifactRef, ...] = ()
    confidence: float = 0.0
    frame: str = ""
    observed_at: str = ""
    observation_epoch: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> ArtifactKey:
        return ArtifactKey(self.producer, self.port)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ref": self.key.to_mapping(),
            "port": self.port,
            "schema": self.schema,
            "producer": self.producer,
            "payload": self.payload,
            "refs": [ref.to_json_dict() for ref in self.refs],
            "confidence": self.confidence,
            "frame": self.frame,
            "observed_at": self.observed_at,
            "observation_epoch": self.observation_epoch,
            "provenance": self.provenance,
        }


ArtifactValidator = Callable[[dict[str, Any]], str | None]


class ArtifactSchemaRegistry:
    """Lightweight validation for stable inter-agent artifact envelopes."""

    def __init__(self) -> None:
        self._validators: dict[str, ArtifactValidator] = {}

    def register(self, schema: str, validator: ArtifactValidator) -> None:
        self._validators[schema] = validator

    def validate(self, artifact: TypedArtifact) -> None:
        validator = self._validators.get(artifact.schema)
        if validator is None:
            return
        error = validator(artifact.payload)
        if error:
            raise ValueError(f"Artifact {artifact.key.path!r} is invalid: {error}")


def _mapping_payload(payload: dict[str, Any]) -> str | None:
    return None if isinstance(payload, dict) else "payload must be a JSON object"


DEFAULT_SCHEMA_REGISTRY = ArtifactSchemaRegistry()
for _schema in (
    "robomex.mask.v1",
    "robomex.points3d.v1",
    "robomex.verifier.v1",
):
    DEFAULT_SCHEMA_REGISTRY.register(_schema, _mapping_payload)
# Schemas with a canonical payload spec get real structural validation; the
# same spec is rendered into the producer's output contract (M1-B3).
for _spec in PAYLOAD_SPECS.values():
    DEFAULT_SCHEMA_REGISTRY.register(_spec.schema, _spec.validate)


class EpochSource(Protocol):
    observation_epoch: int


class ArtifactStore:
    """Validated, node-scoped artifact exchange.

    Validation lives here so callers only resolve inputs and atomically commit
    outputs. The store reads the current epoch from the runtime safety state;
    no caller mirrors or resets it.
    """

    def __init__(
        self,
        *,
        artifact_root: Path | None = None,
        epoch_source: EpochSource | None = None,
        schema_registry: ArtifactSchemaRegistry | None = None,
    ) -> None:
        self._values: dict[ArtifactKey, TypedArtifact] = {}
        self.artifact_root = artifact_root.resolve() if artifact_root else None
        self.epoch_source = epoch_source
        self.schema_registry = schema_registry or DEFAULT_SCHEMA_REGISTRY

    @property
    def observation_epoch(self) -> int:
        return int(getattr(self.epoch_source, "observation_epoch", 0))

    def resolve(
        self,
        ports: tuple[PortSpec, ...],
        bindings: dict[str, str | dict[str, Any]],
    ) -> dict[str, TypedArtifact]:
        resolved: dict[str, TypedArtifact] = {}
        for port in ports:
            raw_ref = bindings.get(port.name)
            if raw_ref is None:
                if port.required:
                    raise ValueError(
                        f"Required input {port.name!r} needs an explicit "
                        f"'$ref': 'producer.port' binding."
                    )
                continue
            key = ArtifactKey.parse(raw_ref)
            artifact = self._values.get(key)
            if artifact is None:
                raise ValueError(f"Artifact {key.path!r} has no published producer output.")
            self._validate_input(port, artifact)
            resolved[port.name] = artifact
        return resolved

    def publish(
        self,
        producer: str,
        ports: tuple[PortSpec, ...],
        outputs: tuple[TypedArtifact, ...],
    ) -> None:
        declared = {port.name: port for port in ports}
        emitted = {artifact.port: artifact for artifact in outputs}
        missing = [p.name for p in ports if p.required and p.name not in emitted]
        if missing:
            raise ValueError(f"Missing required output(s): {', '.join(missing)}.")
        if len(emitted) != len(outputs):
            raise ValueError("A producer emitted duplicate output port names.")

        # Validate the complete batch before mutating the store.
        for artifact in outputs:
            port = declared.get(artifact.port)
            if port is None:
                raise ValueError(f"Undeclared output port {artifact.port!r}.")
            if artifact.producer != producer:
                raise ValueError(
                    f"Output {artifact.port!r} producer {artifact.producer!r} "
                    f"does not match node {producer!r}."
                )
            self._validate_input(port, artifact)
            self.schema_registry.validate(artifact)
            self._validate_refs(artifact)

        self._values.update((artifact.key, artifact) for artifact in outputs)

    def values(self) -> tuple[TypedArtifact, ...]:
        return tuple(self._values.values())

    def catalog(self, *, include_stale: bool = True) -> list[dict[str, Any]]:
        current_epoch = self.observation_epoch
        return [
            {
                "ref": artifact.key.to_mapping(),
                "schema": artifact.schema,
                "frame": artifact.frame,
                "observation_epoch": artifact.observation_epoch,
                "current_epoch": current_epoch,
                "stale": artifact.observation_epoch < current_epoch,
                "confidence": artifact.confidence,
            }
            for artifact in self._values.values()
            if include_stale or artifact.observation_epoch >= current_epoch
        ]

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            artifact.key.path: artifact.to_json_dict()
            for artifact in self._values.values()
        }

    def _validate_input(
        self,
        port: PortSpec,
        artifact: TypedArtifact,
        *,
        reject_stale: bool = True,
    ) -> None:
        if artifact.schema != port.schema:
            raise ValueError(
                f"Port {port.name!r} expects schema {port.schema!r}, "
                f"received {artifact.schema!r} from {artifact.key.path!r}."
            )
        if port.frame and artifact.frame != port.frame:
            raise ValueError(
                f"Port {port.name!r} expects frame {port.frame!r}, "
                f"received {artifact.frame!r}."
            )
        if reject_stale and artifact.observation_epoch < self.observation_epoch:
            raise StaleArtifactError(
                f"Artifact {artifact.key.path!r} is stale at observation epoch "
                f"{artifact.observation_epoch}; current epoch is {self.observation_epoch}."
            )

    def _validate_refs(self, artifact: TypedArtifact) -> None:
        for ref in artifact.refs:
            if not ref.path:
                continue
            path = Path(ref.path)
            if not path.is_absolute() and self.artifact_root is not None:
                path = self.artifact_root / path
            resolved = path.resolve()
            if self.artifact_root is not None:
                try:
                    resolved.relative_to(self.artifact_root)
                except ValueError as exc:
                    raise ValueError(
                        f"Artifact {ref.artifact_id!r} escapes the artifact root."
                    ) from exc
            if not resolved.exists():
                raise ValueError(
                    f"Artifact {ref.artifact_id!r} references missing file {resolved!s}."
                )


def outputs_from_finish(
    payload: dict[str, Any],
    *,
    producer: str,
    ports: tuple[PortSpec, ...],
    observation_epoch: int,
    artifact_dir: Path | None,
) -> tuple[TypedArtifact, ...]:
    """Decode exactly the declared outputs; never fabricate missing values."""

    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    raw_outputs = result.get("outputs") if isinstance(result, dict) else None
    raw_outputs = raw_outputs if isinstance(raw_outputs, dict) else {}
    declared = {port.name: port for port in ports}
    raw_outputs = _normalize_output_names(raw_outputs, declared)
    missing = [p.name for p in ports if p.required and p.name not in raw_outputs]
    if missing:
        raise ValueError(f"SubAgent omitted required output(s): {', '.join(missing)}.")

    artifacts: list[TypedArtifact] = []
    for name, raw in raw_outputs.items():
        port = declared.get(str(name))
        if port is None:
            raise ValueError(f"SubAgent returned undeclared output {name!r}.")
        if not isinstance(raw, dict):
            raise ValueError(f"Output {name!r} must be a JSON object.")
        raw_payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
        payload = {
            str(key): value
            for key, value in raw_payload.items()
            if not (str(key).endswith("_path") and isinstance(value, str))
        }
        raw_refs = raw.get("artifacts")
        if raw_refs is None:
            raw_refs = raw.get("artifact") or raw.get("artifact_path")
        refs = _merge_file_refs(
            _file_refs(raw_refs, producer, artifact_dir),
            _file_refs(_payload_path_refs(raw_payload), producer, artifact_dir),
        )
        artifacts.append(
            TypedArtifact(
                port=port.name,
                schema=port.schema,
                producer=producer,
                payload=compact_json(payload),
                refs=refs,
                confidence=float(raw.get("confidence", 0.0) or 0.0),
                frame=str(raw.get("frame") or raw_payload.get("frame") or ""),
                observed_at=str(
                    raw.get("observed_at") or raw_payload.get("observed_at") or ""
                ),
                # The runtime is the only authority for epochs. An agent that just
                # moved the robot cannot know the post-motion epoch, and an echoed
                # request-time value would mark its own evidence stale (M1.5 Fix A).
                observation_epoch=observation_epoch,
                provenance={"node": producer},
            )
        )
    return tuple(artifacts)


def _normalize_output_names(
    raw_outputs: dict[str, Any],
    declared: dict[str, PortSpec],
) -> dict[str, Any]:
    """Accept a rendered ``name:schema@frame`` label as the declared name."""

    normalized: dict[str, Any] = {}
    for raw_name, value in raw_outputs.items():
        name = str(raw_name)
        if name not in declared:
            candidate = name.split(":", 1)[0]
            if candidate in declared:
                name = candidate
        if name in normalized:
            raise ValueError(f"SubAgent returned duplicate output {name!r}.")
        normalized[name] = value
    return normalized


def add_grounding_overlay(
    outputs: tuple[TypedArtifact, ...],
    *,
    artifact_dir: Path | None,
    fallback_rgb_path: str | None = None,
) -> tuple[TypedArtifact, ...]:
    """Best-effort visualization for any leaf that emits a mask and RGB evidence."""

    if artifact_dir is None:
        return outputs
    mask_index = next(
        (index for index, item in enumerate(outputs) if _looks_like(item, "mask")),
        None,
    )
    if mask_index is None:
        return outputs
    mask_artifact = outputs[mask_index]
    mask_path = _first_numpy_path(mask_artifact)
    if mask_path is None:
        return outputs

    rgb_artifact = next((item for item in outputs if _looks_like(item, "rgb")), None)
    rgb_path = _first_numpy_path(rgb_artifact) if rgb_artifact is not None else None
    if rgb_path is None:
        rgb_path = _numpy_path_for_kind(mask_artifact, "rgb")
    try:
        import numpy as np
        from PIL import Image

        from robomex.perception.render import save_mask_overlay

        mask = np.load(mask_path, allow_pickle=False)
        if rgb_path is not None:
            rgb = np.load(rgb_path, allow_pickle=False)
        elif fallback_rgb_path:
            rgb = np.asarray(Image.open(fallback_rgb_path).convert("RGB"))
        else:
            return outputs
        overlay_path = artifact_dir / "grounding_overlay.png"
        bbox = mask_artifact.payload.get("bbox_xyxy")
        save_mask_overlay(overlay_path, rgb, mask, bbox=bbox)
    except (ImportError, OSError, TypeError, ValueError):
        return outputs

    overlay_ref = ArtifactRef(
        artifact_id=f"{mask_artifact.port}_overlay",
        kind="overlay",
        path=str(overlay_path.resolve()),
        producer=mask_artifact.producer,
        summary="RGB grounding overlay with mask and bounding box.",
    )
    updated = list(outputs)
    updated[mask_index] = replace(
        mask_artifact,
        refs=tuple((*mask_artifact.refs, overlay_ref)),
    )
    return tuple(updated)


def _file_refs(
    raw: Any,
    producer: str,
    artifact_dir: Path | None,
) -> tuple[ArtifactRef, ...]:
    if isinstance(raw, dict):
        items: tuple[Any, ...] = tuple(
            {"artifact_id": name, "kind": name, **value}
            if isinstance(value, dict)
            else {"artifact_id": name, "kind": name, "path": value}
            for name, value in raw.items()
        )
    elif isinstance(raw, (list, tuple)):
        items = tuple(raw)
    elif isinstance(raw, str):
        items = (raw,)
    else:
        items = ()
    refs = tuple(
        ref
        for ref in (
            ArtifactRef.from_any(item, default_producer=producer) for item in items
        )
        if ref is not None
    )
    if artifact_dir is None:
        return refs
    normalized: list[ArtifactRef] = []
    for ref in refs:
        if not ref.path:
            normalized.append(ref)
            continue
        # Relative refs always resolve against the node's artifact dir; a
        # CWD-relative fallback silently rebinds paths to the repo root and
        # produced the "escapes the artifact root" failures (M1.6).
        path = resolve_artifact_ref_path(ref.path, artifact_dir)
        normalized.append(replace(ref, path=str(path)))
    return tuple(normalized)


def _payload_path_refs(payload: dict[str, Any]) -> dict[str, str]:
    return {
        str(key)[: -len("_path")]: value
        for key, value in payload.items()
        if str(key).endswith("_path") and isinstance(value, str) and value
    }


def _merge_file_refs(
    first: tuple[ArtifactRef, ...],
    second: tuple[ArtifactRef, ...],
) -> tuple[ArtifactRef, ...]:
    merged: list[ArtifactRef] = []
    seen: set[tuple[str, str]] = set()
    for ref in (*first, *second):
        identity = (ref.kind, ref.path)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(ref)
    return tuple(merged)


def _looks_like(artifact: TypedArtifact, kind: str) -> bool:
    identity = f"{artifact.port} {artifact.schema}".lower()
    payload_format = str(artifact.payload.get("format") or "").lower()
    return kind in identity or kind in payload_format


def _first_numpy_path(artifact: TypedArtifact | None) -> Path | None:
    if artifact is None:
        return None
    for ref in artifact.refs:
        if ref.path and Path(ref.path).suffix.lower() == ".npy":
            return Path(ref.path)
    return None


def _numpy_path_for_kind(artifact: TypedArtifact, kind: str) -> Path | None:
    for ref in artifact.refs:
        identity = f"{ref.artifact_id} {ref.kind}".lower()
        if kind in identity and ref.path and Path(ref.path).suffix.lower() == ".npy":
            return Path(ref.path)
    return None
