"""Audited constructor for the bowl attachment control monitor."""

from __future__ import annotations

from collections.abc import Sequence

from robomex.authoring.monitoring import (
    MonitorCompiler,
    MonitorHook,
    MonitorProgramSpec,
)

_REQUIRED_SIGNALS = (
    "attachment_status",
    "held_entity_visible",
    "identity_match",
)

_SOURCE = """
def evaluate(sample):
    if sample["identity_match"] == False:
        return {
            "finding": "unsafe_deviation",
            "severity": "critical",
            "confidence": 1.0,
            "details": {"reason": "held_entity_identity_swap"},
        }
    if sample["held_entity_visible"] == False:
        return {
            "finding": "unobservable",
            "severity": "critical",
            "details": {"reason": "held_entity_occluded"},
        }
    if sample["attachment_status"] != "verified_held":
        return {
            "finding": "attachment_anomaly",
            "severity": "critical",
            "confidence": 1.0,
            "details": {"reason": "attachment_not_verified_held"},
        }
    return None
"""


def build_attachment_monitor_program(
    *,
    allowed_signals: Sequence[str] = _REQUIRED_SIGNALS,
    debounce_count: int = 1,
) -> dict[str, object]:
    """Return a compiler-validated, read-only ``MonitorProgramSpec`` payload."""

    signals = tuple(str(value) for value in allowed_signals)
    if signals != _REQUIRED_SIGNALS:
        raise ValueError(
            "attachment monitor signals must exactly match the audited three-signal contract"
        )
    compiled = MonitorCompiler().compile(
        MonitorProgramSpec(
            monitor_id="bowl_attachment_guard",
            source=_SOURCE,
            hook=MonitorHook.CONTROL,
            allowed_signals=signals,
            debounce_count=debounce_count,
            max_runtime_ms=5.0,
        )
    )
    return compiled.spec.model_dump(mode="json")


__all__ = ["build_attachment_monitor_program"]
