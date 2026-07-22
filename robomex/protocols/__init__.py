"""Reusable v2 manipulation protocol builders."""

from robomex.protocols.bowl_place import (
    BowlPlaceProtocolConfig,
    FixedBowlPlaceProtocol,
    build_bowl_attachment_monitor_program,
    build_bowl_place_graph,
    build_fixed_bowl_place_protocol,
)

_RISK_ADAPTIVE_EXPORTS = frozenset(
    {
        "BowlCorrectionCandidateStrategy",
        "RiskAdaptiveBowlPlaceProtocol",
        "RiskAdaptiveBowlPlaceProtocolConfig",
        "RiskAdaptiveBowlPlaceProtocolError",
        "build_risk_adaptive_bowl_place_graph",
        "build_risk_adaptive_bowl_place_protocol",
    }
)


def __getattr__(name: str):
    if name not in _RISK_ADAPTIVE_EXPORTS:
        raise AttributeError(name)
    from robomex.protocols import risk_adaptive_bowl_place

    return getattr(risk_adaptive_bowl_place, name)

__all__ = [
    "BowlPlaceProtocolConfig",
    "FixedBowlPlaceProtocol",
    "BowlCorrectionCandidateStrategy",
    "RiskAdaptiveBowlPlaceProtocol",
    "RiskAdaptiveBowlPlaceProtocolConfig",
    "RiskAdaptiveBowlPlaceProtocolError",
    "build_bowl_attachment_monitor_program",
    "build_bowl_place_graph",
    "build_fixed_bowl_place_protocol",
    "build_risk_adaptive_bowl_place_graph",
    "build_risk_adaptive_bowl_place_protocol",
]
