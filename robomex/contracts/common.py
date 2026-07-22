"""Shared closed-world primitives for RoboMEx v2 contracts.

Persisted contracts are immutable and content-addressed.  Their digest covers
the complete semantic payload (including schema version and logical identity),
so a logical version cannot silently drift after it has been admitted.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ContractId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=192,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$",
    ),
]
DigestStr = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]

_DIGEST_DOMAIN = b"robomex-contract-v2\x00"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ContractValidationError(ValueError):
    """Raised by registry-level checks after Pydantic validation succeeds."""


class StrictContract(BaseModel):
    """Immutable, unknown-field-rejecting base for persisted v2 contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


def canonical_payload_digest(schema_version: str, payload: Mapping[str, object]) -> str:
    """Return a domain-separated digest of finite canonical JSON content."""

    try:
        encoded = json.dumps(
            {"schema_version": schema_version, "payload": payload},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            "Contract content must be finite canonical JSON data."
        ) from exc
    return "sha256:" + hashlib.sha256(_DIGEST_DOMAIN + encoded).hexdigest()


def canonical_model_digest(model: BaseModel) -> str:
    """Digest a model, excluding only its top-level self digest."""

    schema_version = getattr(model, "schema_version", None)
    if not isinstance(schema_version, str) or not schema_version:
        raise ContractValidationError("A sealed contract requires schema_version.")
    payload = cast(
        dict[str, object],
        model.model_dump(mode="json", exclude={"content_digest"}),
    )
    return canonical_payload_digest(schema_version, payload)


class SealedContract(StrictContract):
    """Contract that seals itself or validates a caller-supplied digest."""

    content_digest: str = ""

    @model_validator(mode="after")
    def _seal_or_check_content(self) -> SealedContract:
        expected = canonical_model_digest(self)
        if self.content_digest:
            if _DIGEST_RE.fullmatch(self.content_digest) is None:
                raise ValueError("content_digest must be sha256:<64 lowercase hex>")
            if self.content_digest != expected:
                raise ValueError("content_digest does not match canonical contract payload")
        else:
            object.__setattr__(self, "content_digest", expected)
        return self


SealedT = TypeVar("SealedT", bound=SealedContract)


def revalidate_sealed(contract: SealedT) -> SealedT:
    """Fully revalidate an instance, detecting unsafe ``model_copy`` changes."""

    return type(contract).model_validate(contract.model_dump(mode="python"))


class ContentPin(StrictContract):
    """Stable identity plus digest used by manifests and registry snapshots."""

    component_id: ContractId
    content_digest: DigestStr
    revision: int | None = Field(default=None, ge=1)


def require_unique(
    values: Sequence[Any],
    *,
    key: str,
    label: str,
) -> None:
    """Reject duplicate identifiers while preserving caller order."""

    seen: set[Any] = set()
    duplicates: set[str] = set()
    for value in values:
        identity = getattr(value, key)
        if identity in seen:
            duplicates.add(str(identity))
        seen.add(identity)
    if duplicates:
        raise ValueError(f"Duplicate {label}: {', '.join(sorted(duplicates))}.")


def require_unique_strings(values: Sequence[str], *, label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label}: {', '.join(duplicates)}.")


JsonObject = dict[str, JsonValue]

__all__ = [
    "ContentPin",
    "ContractId",
    "ContractValidationError",
    "DigestStr",
    "JsonObject",
    "NonEmptyStr",
    "SealedContract",
    "StrictContract",
    "canonical_model_digest",
    "canonical_payload_digest",
    "require_unique",
    "require_unique_strings",
    "revalidate_sealed",
]
