"""Closed registry for action-facing payload schemas.

Schema names are protocol identities, not documentation strings.  The
registry therefore binds each name to a real Pydantic validator and refuses
unknown ``robomex.*`` payloads instead of treating them as arbitrary dicts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from robomex.data.physical_schema import CORE_PHYSICAL_MODELS

_SCHEMA_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+\.v[1-9][0-9]*$")


class SchemaRegistryError(ValueError):
    """Base error for schema registration and payload validation."""


class UnregisteredSchemaError(SchemaRegistryError):
    """A payload claims a schema for which no validator is installed."""


class SchemaPayloadError(SchemaRegistryError):
    """A registered validator rejected a payload."""


class SchemaRegistry:
    """Versioned mapping from schema IDs to concrete Pydantic models."""

    def __init__(self, *, install_core: bool = True) -> None:
        self._models: dict[str, type[BaseModel]] = {}
        self._core_ids = frozenset(CORE_PHYSICAL_MODELS)
        if install_core:
            for schema_id, model in CORE_PHYSICAL_MODELS.items():
                self._models[schema_id] = model

    @property
    def schema_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))

    @property
    def missing_core_schema_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._core_ids.difference(self._models)))

    def is_registered(self, schema_id: str) -> bool:
        return schema_id in self._models

    def register(self, schema_id: str, model: type[BaseModel]) -> None:
        """Register one immutable protocol binding.

        Identity functions and untyped callables are intentionally not
        accepted: every registered schema must perform structural validation.
        Existing names cannot be overwritten because changing a validator
        without changing its schema version would make replay nondeterministic.
        """

        if not isinstance(schema_id, str) or not _SCHEMA_ID_RE.fullmatch(schema_id):
            raise SchemaRegistryError(f"Invalid versioned schema id {schema_id!r}.")
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise SchemaRegistryError("Schema validators must be Pydantic BaseModel classes.")
        if schema_id in self._models:
            raise SchemaRegistryError(f"Schema {schema_id!r} is already registered.")
        self._models[schema_id] = model

    def ensure(self, schema_id: str, model: type[BaseModel]) -> None:
        """Idempotently install the exact same schema binding.

        Reusing a schema ID for a different validator is hash drift at the
        wire boundary and remains a hard error.
        """

        existing = self._models.get(schema_id)
        if existing is None:
            self.register(schema_id, model)
            return
        if existing is not model:
            raise SchemaRegistryError(
                f"Schema {schema_id!r} is already bound to {existing.__qualname__}, "
                f"not {model.__qualname__}."
            )

    def unregister(self, schema_id: str) -> None:
        if schema_id in self._core_ids:
            raise SchemaRegistryError(f"Core schema {schema_id!r} cannot be unregistered.")
        if schema_id not in self._models:
            raise UnregisteredSchemaError(f"Schema {schema_id!r} is not registered.")
        del self._models[schema_id]

    def model_for(self, schema_id: str) -> type[BaseModel]:
        try:
            return self._models[schema_id]
        except KeyError as exc:
            suffix = " Core physical payloads may never use an untyped fallback." if (
                isinstance(schema_id, str) and schema_id.startswith("robomex.")
            ) else ""
            raise UnregisteredSchemaError(
                f"Schema {schema_id!r} is not registered.{suffix}"
            ) from exc

    def validate(self, schema_id: str, payload: Mapping[str, Any] | BaseModel) -> BaseModel:
        model = self.model_for(schema_id)
        if not isinstance(payload, (Mapping, BaseModel)):
            raise SchemaPayloadError(
                f"Payload for {schema_id!r} must be a mapping or typed model."
            )
        raw: Any = (
            payload.model_dump(mode="python") if isinstance(payload, BaseModel) else payload
        )
        try:
            return model.model_validate(raw)
        except ValidationError as exc:
            raise SchemaPayloadError(f"Payload failed schema {schema_id!r}: {exc}") from exc

    def validate_mapping(
        self,
        schema_id: str,
        payload: Mapping[str, Any] | BaseModel,
    ) -> dict[str, Any]:
        """Return the complete normalized wire mapping after validation."""

        return self.validate(schema_id, payload).model_dump(mode="json")

    def assert_core_complete(self) -> None:
        missing = self.missing_core_schema_ids
        if missing:
            raise UnregisteredSchemaError(
                f"Core physical schema validators are missing: {', '.join(missing)}."
            )

    def validator_manifest(self) -> tuple[dict[str, Any], ...]:
        """Return a canonical inventory of every installed wire validator."""

        return tuple(
            {
                "schema_id": schema_id,
                "validator": f"{model.__module__}.{model.__qualname__}",
                "json_schema": model.model_json_schema(),
            }
            for schema_id, model in sorted(self._models.items())
        )

    @property
    def content_digest(self) -> str:
        """Digest schema IDs, validator identities, and structural JSON schemas."""

        payload = json.dumps(
            self.validator_manifest(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def core_schema_registry() -> SchemaRegistry:
    registry = SchemaRegistry(install_core=True)
    registry.assert_core_complete()
    return registry


__all__ = [
    "SchemaPayloadError",
    "SchemaRegistry",
    "SchemaRegistryError",
    "UnregisteredSchemaError",
    "core_schema_registry",
]
