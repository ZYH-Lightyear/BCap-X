"""Fresh, read-only telemetry bridges for code-authored action monitors.

The action thread must never invoke a language model, poll a camera backend, or
write embodied state.  This module therefore adapts only already-published
tracking samples plus a small authoritative attachment *read* callback.  Both
sources carry clocks and revisions and are rejected when stale, lost, or bound
to a different physical resource.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from robomex.data.embodied_state import AttachmentStatus
from robomex.runtime.observation import (
    ObservationQuality,
    ObservationRegistry,
    ObservationSample,
    TrackState,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility


class MonitorTelemetryUnavailableError(RuntimeError):
    """Fresh monitor evidence cannot be produced without guessing or reuse."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        str_strip_whitespace=True,
        validate_default=True,
    )


class AuthoritativeAttachmentTelemetry(_StrictModel):
    """One clocked read from the authority that owns attachment state."""

    schema_version: Literal["robomex.authoritative_attachment_telemetry.v1"] = (
        "robomex.authoritative_attachment_telemetry.v1"
    )
    world_id: NonEmptyStr
    resource_id: NonEmptyStr
    entity_id: NonEmptyStr
    track_id: NonEmptyStr
    attachment_status: AttachmentStatus
    attachment_revision: int = Field(ge=0)
    observed_at: datetime = Field(default_factory=_utc_now)
    monotonic_time_s: float = Field(
        default_factory=time.monotonic,
        ge=0,
        allow_inf_nan=False,
    )

    @field_validator("observed_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10


@runtime_checkable
class AttachmentTelemetryReader(Protocol):
    """Read-only, non-blocking callback safe to invoke on the action thread."""

    def __call__(
        self, world_id: str, resource_id: str
    ) -> AuthoritativeAttachmentTelemetry | Mapping[str, object]: ...


class BowlMonitorTelemetryBridge:
    """Project fresh tracking and attachment evidence into the bowl guard.

    A live tracker should publish into ``ObservationRegistry`` asynchronously.
    Calling this bridge only reads the registry's immutable stream tail; it
    never calls ``TrackHandle.poll`` and therefore cannot perform inference or
    sensor I/O from the controller callback.  A retained sample is usable only
    while both its UTC and monotonic ages satisfy the configured bound.

    ``attachment_reader`` is subject to the same rule: it must be a bounded,
    in-memory authority read.  It must not perform an LLM call or mutate the
    world.  Revision equality prevents a fresh camera sample from being mixed
    with attachment state from another physical instant.
    """

    SIGNAL_NAMES = (
        "attachment_status",
        "held_entity_visible",
        "identity_match",
    )

    def __init__(
        self,
        *,
        registry: ObservationRegistry,
        track_id: str,
        expected_entity_id: str,
        world_id: str,
        resource_id: str,
        attachment_reader: AttachmentTelemetryReader,
        max_observation_age_s: float,
        max_attachment_age_s: float,
        visibility_signal: str = "visibility",
        identity_signal: str = "identity_match",
        max_future_skew_s: float = 0.05,
        monotonic_clock: Callable[[], float] = time.monotonic,
        utc_clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        identities = (
            str(track_id).strip(),
            str(expected_entity_id).strip(),
            str(world_id).strip(),
            str(resource_id).strip(),
            str(visibility_signal).strip(),
            str(identity_signal).strip(),
        )
        if any(not value for value in identities):
            raise ValueError("bridge identities and signal names must not be empty")
        if max_observation_age_s <= 0 or not math.isfinite(max_observation_age_s):
            raise ValueError("max_observation_age_s must be finite and positive")
        if max_attachment_age_s <= 0 or not math.isfinite(max_attachment_age_s):
            raise ValueError("max_attachment_age_s must be finite and positive")
        if max_future_skew_s < 0 or not math.isfinite(max_future_skew_s):
            raise ValueError("max_future_skew_s must be finite and non-negative")
        if not callable(attachment_reader):
            raise TypeError("attachment_reader must be callable")
        handle = registry.get(identities[0])
        if handle.request.entity.entity_id != identities[1]:
            raise MonitorTelemetryUnavailableError(
                "track is bound to a different entity identity"
            )
        if identities[5] not in handle.request.declared_signals:
            raise MonitorTelemetryUnavailableError(
                f"track does not declare identity signal {identities[5]!r}"
            )
        if identities[4] not in handle.request.declared_signals:
            raise MonitorTelemetryUnavailableError(
                f"track does not declare visibility signal {identities[4]!r}"
            )
        self.registry = registry
        self.track_id = identities[0]
        self.expected_entity_id = identities[1]
        self.world_id = identities[2]
        self.resource_id = identities[3]
        self.visibility_signal = identities[4]
        self.identity_signal = identities[5]
        self.attachment_reader = attachment_reader
        self.max_observation_age_s = float(max_observation_age_s)
        self.max_attachment_age_s = float(max_attachment_age_s)
        self.max_future_skew_s = float(max_future_skew_s)
        self._monotonic_clock = monotonic_clock
        self._utc_clock = utc_clock

    @property
    def signal_names(self) -> tuple[str, ...]:
        """Exact custom names for ``LiberoControlPort.sample_signal_names``."""

        return self.SIGNAL_NAMES

    @property
    def binding(self) -> tuple[str, str]:
        return (self.world_id, self.resource_id)

    def __call__(self) -> Mapping[str, object]:
        if self.registry.get(self.track_id).state is not TrackState.ACTIVE:
            raise MonitorTelemetryUnavailableError(
                "tracking service is not active"
            )
        observation = self.registry.stream.latest(self.track_id)
        if observation is None:
            raise MonitorTelemetryUnavailableError(
                "no tracking observation has been published"
            )
        self._validate_observation(observation)
        try:
            raw_attachment = self.attachment_reader(
                self.world_id,
                self.resource_id,
            )
            attachment = AuthoritativeAttachmentTelemetry.model_validate(
                raw_attachment.model_dump(mode="python")
                if isinstance(raw_attachment, AuthoritativeAttachmentTelemetry)
                else raw_attachment
            )
        except MonitorTelemetryUnavailableError:
            raise
        except Exception as exc:
            raise MonitorTelemetryUnavailableError(
                f"attachment authority read failed: {type(exc).__name__}"
            ) from exc
        self._validate_attachment(attachment, observation)
        identity_match = observation.signals[self.identity_signal]
        if not isinstance(identity_match, bool):
            raise MonitorTelemetryUnavailableError(
                "tracker identity signal must be a boolean"
            )
        held_entity_visible = self._visibility(
            observation.signals[self.visibility_signal]
        )
        return {
            "attachment_status": attachment.attachment_status.value,
            "held_entity_visible": held_entity_visible,
            "identity_match": identity_match,
        }

    def _validate_observation(self, sample: ObservationSample) -> None:
        if sample.track_id != self.track_id:
            raise MonitorTelemetryUnavailableError("tracking sample changed track identity")
        if sample.entity.entity_id != self.expected_entity_id:
            raise MonitorTelemetryUnavailableError("tracking sample changed entity identity")
        if sample.quality is not ObservationQuality.TRACKED:
            raise MonitorTelemetryUnavailableError(
                f"tracker is {sample.quality.value}: {sample.reason or 'no reason'}"
            )
        self._validate_age(
            observed_at=sample.observed_at,
            monotonic_time_s=sample.monotonic_time_s,
            max_age_s=self.max_observation_age_s,
            source="tracking observation",
        )
        missing = {
            self.identity_signal,
            self.visibility_signal,
        }.difference(sample.signals)
        if missing:
            raise MonitorTelemetryUnavailableError(
                "tracking observation is missing signals: "
                + ", ".join(sorted(missing))
            )

    def _validate_attachment(
        self,
        sample: AuthoritativeAttachmentTelemetry,
        observation: ObservationSample,
    ) -> None:
        if (sample.world_id, sample.resource_id) != self.binding:
            raise MonitorTelemetryUnavailableError(
                "attachment telemetry belongs to another world/resource"
            )
        if (
            sample.entity_id != self.expected_entity_id
            or sample.track_id != self.track_id
        ):
            raise MonitorTelemetryUnavailableError(
                "attachment telemetry belongs to another entity/track"
            )
        self._validate_age(
            observed_at=sample.observed_at,
            monotonic_time_s=sample.monotonic_time_s,
            max_age_s=self.max_attachment_age_s,
            source="attachment telemetry",
        )
        if sample.attachment_revision != observation.revisions.attachment_revision:
            raise MonitorTelemetryUnavailableError(
                "tracking and attachment telemetry revisions differ"
            )

    def _validate_age(
        self,
        *,
        observed_at: datetime,
        monotonic_time_s: float,
        max_age_s: float,
        source: str,
    ) -> None:
        now_monotonic = float(self._monotonic_clock())
        now_utc = self._utc_clock()
        if now_utc.tzinfo is None or now_utc.utcoffset() is None:
            raise MonitorTelemetryUnavailableError("bridge UTC clock is timezone-naive")
        monotonic_age = now_monotonic - monotonic_time_s
        wall_age = (now_utc.astimezone(timezone.utc) - observed_at).total_seconds()  # noqa: UP017
        if monotonic_age < -self.max_future_skew_s or wall_age < -self.max_future_skew_s:
            raise MonitorTelemetryUnavailableError(f"{source} timestamp is in the future")
        if monotonic_age > max_age_s or wall_age > max_age_s:
            raise MonitorTelemetryUnavailableError(f"{source} is stale")

    @staticmethod
    def _visibility(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"visible", "tracked"}:
                return True
            if normalized in {"occluded", "not_visible"}:
                return False
        raise MonitorTelemetryUnavailableError(
            "tracker visibility signal must be boolean or visible/occluded"
        )


__all__ = [
    "AttachmentTelemetryReader",
    "AuthoritativeAttachmentTelemetry",
    "BowlMonitorTelemetryBridge",
    "MonitorTelemetryUnavailableError",
]
