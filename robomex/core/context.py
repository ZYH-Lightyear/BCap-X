"""Episode-level context primitives for RoboMEx multi-agent runs.

``EVIDENCE`` remains a local scratchpad inside one Agent/subgoal.  The classes in
this module carry compact, JSON-safe facts across Planner, Act, and SubAgent
boundaries.

RoboMEx deliberately keeps the cross-agent contract as a generic evidence
envelope rather than a fixed phase/profile schema.  Agents can put arbitrary
task-specific content in ``evidence`` while the runtime understands only the
small transport shell: claims, confidence, artifacts, facts, verdicts, and
recommended next steps.
"""

from __future__ import annotations

import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


JsonDict = dict[str, Any]


def compact_json(
    value: Any,
    *,
    max_depth: int = 5,
    max_items: int = 16,
    max_string: int = 600,
) -> Any:
    """Return a JSON-safe, prompt-safe summary of arbitrary data.

    Evidence packets are allowed to contain task-specific fields, but they must not
    leak large masks, point clouds, images, videos, or long logs into Planner prompts
    and summary files. This function preserves small scalar/list/dict content while
    summarizing large or deep values.
    """

    return _compact_json(value, max_depth=max_depth, max_items=max_items, max_string=max_string)


def _compact_json(value: Any, *, max_depth: int, max_items: int, max_string: int) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return {
            "type": "str",
            "length": len(value),
            "preview": value[:max_string].rstrip() + "...",
        }
    # Idempotency (M1.5 Fix G): a summary produced by an earlier compaction pass
    # is atomic. Without this, re-compacting manifests nests summaries into
    # {"type": "dict", "repr": "{'type': 'ndarray', ...}"} garbage on disk.
    if isinstance(value, dict) and _is_compaction_summary(value):
        return value
    if max_depth <= 0:
        return _value_summary(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            summary: JsonDict = {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            if value.size and np.issubdtype(value.dtype, np.number):
                finite = value[np.isfinite(value)]
                if finite.size:
                    summary["min"] = float(np.min(finite))
                    summary["max"] = float(np.max(finite))
                    summary["mean"] = float(np.mean(finite))
            return summary
    except Exception:
        pass
    if isinstance(value, dict):
        out: JsonDict = {}
        items = list(value.items())
        for k, v in items[:max_items]:
            out[str(k)] = _compact_json(
                v,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string=max_string,
            )
        if len(items) > max_items:
            out["_truncated_items"] = len(items) - max_items
        return out
    if isinstance(value, (list, tuple)):
        seq = list(value)
        out = [
            _compact_json(
                v,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string=max_string,
            )
            for v in seq[:max_items]
        ]
        if len(seq) > max_items:
            out.append({"_truncated_items": len(seq) - max_items})
        return out
    return _value_summary(value)


_SUMMARY_DETAIL_KEYS = frozenset({"repr", "shape", "length", "preview"})


def _is_compaction_summary(value: dict) -> bool:
    """Detect the summary dicts emitted by this module's own compaction."""

    if not isinstance(value.get("type"), str):
        return False
    return bool(_SUMMARY_DETAIL_KEYS.intersection(value))


def _value_summary(value: Any) -> JsonDict:
    try:
        length = len(value)  # type: ignore[arg-type]
    except Exception:
        length = None
    out: JsonDict = {"type": type(value).__name__, "repr": repr(value)[:160]}
    if length is not None:
        out["length"] = int(length)
    return out


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


@dataclass(frozen=True)
class StateFact:
    """One compact belief about the episode state."""

    key: str
    value: Any
    confidence: float = 1.0
    source_agent: str = ""
    source_turn: str = ""
    timestamp: float = field(default_factory=time.time)
    provenance: JsonDict = field(default_factory=dict)
    validity: JsonDict = field(default_factory=dict)
    expires_after_subgoal: bool = False

    @classmethod
    def from_dict(cls, data: JsonDict) -> "StateFact":
        return cls(
            key=str(data.get("key", "")).strip(),
            value=_json_safe(data.get("value")),
            confidence=float(data.get("confidence", 1.0)),
            source_agent=str(data.get("source_agent", "")),
            source_turn=str(data.get("source_turn", "")),
            timestamp=float(data.get("timestamp", time.time())),
            provenance=_json_safe(data.get("provenance") if isinstance(data.get("provenance"), dict) else {}),
            validity=_json_safe(data.get("validity") if isinstance(data.get("validity"), dict) else {}),
            expires_after_subgoal=bool(data.get("expires_after_subgoal", False)),
        )

    def to_json_dict(self) -> JsonDict:
        return {
            "key": self.key,
            "value": _json_safe(self.value),
            "confidence": float(self.confidence),
            "source_agent": self.source_agent,
            "source_turn": self.source_turn,
            "timestamp": float(self.timestamp),
            "provenance": _json_safe(self.provenance),
            "validity": _json_safe(self.validity),
            "expires_after_subgoal": self.expires_after_subgoal,
        }


@dataclass(frozen=True)
class ArtifactRef:
    """A persisted or addressable multimodal artifact.

    Large masks, point clouds, videos, and overlays should be referenced through this
    object instead of being copied into prompts or episode facts.
    """

    artifact_id: str
    kind: str = "artifact"
    path: str | None = None
    producer: str = ""
    summary: str = ""
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_any(cls, raw: Any, *, default_producer: str = "") -> "ArtifactRef | None":
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return None
            return cls(artifact_id=text, path=text, producer=default_producer)
        if not isinstance(raw, dict):
            return None
        artifact_id = str(raw.get("artifact_id") or raw.get("id") or raw.get("path") or raw.get("uri") or "").strip()
        if not artifact_id:
            return None
        return cls(
            artifact_id=artifact_id,
            kind=str(raw.get("kind") or raw.get("type") or "artifact"),
            path=None if raw.get("path") is None else str(raw.get("path")),
            producer=str(raw.get("producer") or default_producer),
            summary=str(raw.get("summary") or raw.get("description") or ""),
            metadata=_json_safe(raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}),
        )

    def to_json_dict(self) -> JsonDict:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "path": self.path,
            "producer": self.producer,
            "summary": self.summary,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class LocalVerdict:
    """A local verifier/specialist judgment about one narrow question."""

    verdict_type: str
    status: str
    confidence: float = 0.0
    reason: str = ""
    target: str = ""
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_any(cls, raw: Any) -> "LocalVerdict | None":
        if not isinstance(raw, dict):
            return None
        status = str(raw.get("status") or raw.get("verdict") or "").strip()
        verdict_type = str(raw.get("verdict_type") or raw.get("type") or raw.get("result_type") or "").strip()
        if not status and not verdict_type:
            return None
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return cls(
            verdict_type=verdict_type or "local_verdict",
            status=status or "unknown",
            confidence=confidence,
            reason=str(raw.get("reason") or raw.get("failure_reason") or raw.get("note") or ""),
            target=str(raw.get("target") or raw.get("object") or ""),
            metadata=_json_safe({k: v for k, v in raw.items() if k not in {
                "status", "verdict", "verdict_type", "type", "result_type", "confidence",
                "reason", "failure_reason", "note", "target", "object",
            }}),
        )

    def to_json_dict(self) -> JsonDict:
        return {
            "verdict_type": self.verdict_type,
            "status": self.status,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "target": self.target,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class EvidencePacket:
    """Open, compact evidence envelope shared across Agents.

    The runtime intentionally does not interpret the inner ``evidence`` dict as a
    fixed robot phase schema.  It only promotes generic facts/artifacts/verdicts
    and keeps the rest available for Act/Planner/debug views.
    """

    claim: str = ""
    confidence: float = 0.0
    evidence: JsonDict = field(default_factory=dict)
    artifact_refs: tuple[ArtifactRef, ...] = ()
    facts: tuple[StateFact, ...] = ()
    verdict: LocalVerdict | None = None
    recommended_next: str = ""
    uncertainty: tuple[str, ...] = ()
    source_agent: str = ""
    source_turn: str = ""
    packet_id: str = ""
    created_at: float = field(default_factory=time.time)

    @classmethod
    def from_any(
        cls,
        raw: Any,
        *,
        default_source: str = "",
        default_turn: str = "",
    ) -> "EvidencePacket":
        if not isinstance(raw, dict):
            return cls(
                claim=str(raw or ""),
                source_agent=default_source,
                source_turn=default_turn,
            )

        nested = raw.get("evidence_packet")
        if not isinstance(nested, dict):
            nested = raw.get("packet")
        if isinstance(nested, dict):
            raw = {**nested, **{k: v for k, v in raw.items() if k not in {"evidence_packet", "packet"}}}

        payload = _result_payload(raw)
        claim = str(payload.get("claim") or "")
        recommended_next = str(payload.get("recommended_next") or payload.get("next") or "")
        confidence = _coerce_confidence(payload.get("confidence", 0.0))

        uncertainty_raw = payload.get("uncertainty", ())
        if isinstance(uncertainty_raw, str):
            uncertainty = (uncertainty_raw,)
        elif isinstance(uncertainty_raw, (list, tuple)):
            uncertainty = tuple(str(v) for v in uncertainty_raw if str(v).strip())
        else:
            uncertainty = ()

        explicit_evidence = payload.get("evidence")
        if isinstance(explicit_evidence, dict):
            evidence = compact_json(explicit_evidence)
        else:
            evidence = compact_json(_implicit_evidence(payload))

        artifacts = tuple(
            artifact
            for artifact in (
                ArtifactRef.from_any(item, default_producer=default_source)
                for item in _iter_artifact_items(payload)
            )
            if artifact is not None
        )
        facts = tuple(_iter_packet_facts(payload, default_source=default_source, default_turn=default_turn))
        verdict = LocalVerdict.from_any(payload.get("verdict") or payload.get("local_verdict"))
        if verdict is None:
            verdict = LocalVerdict.from_any(payload)

        source_agent = str(payload.get("source_agent") or payload.get("source") or default_source)
        source_turn = str(payload.get("source_turn") or default_turn)
        created_at = _coerce_timestamp(payload.get("created_at") or payload.get("timestamp"))
        packet_id = str(payload.get("packet_id") or payload.get("id") or "").strip()
        if not packet_id:
            packet_id = _make_packet_id(
                source_agent=source_agent,
                source_turn=source_turn,
                claim=claim,
                evidence=evidence,
                created_at=created_at,
            )
        return cls(
            claim=claim,
            confidence=confidence,
            evidence=evidence,
            artifact_refs=artifacts,
            facts=facts,
            verdict=verdict,
            recommended_next=recommended_next,
            uncertainty=uncertainty,
            source_agent=source_agent,
            source_turn=source_turn,
            packet_id=packet_id,
            created_at=created_at,
        )

    @property
    def is_empty(self) -> bool:
        return not (
            self.claim
            or self.evidence
            or self.artifact_refs
            or self.facts
            or self.verdict is not None
            or self.recommended_next
            or self.uncertainty
        )

    def to_json_dict(self) -> JsonDict:
        return {
            "schema": "robomex.evidence_packet.v1",
            "packet_id": self.packet_id or _make_packet_id(
                source_agent=self.source_agent,
                source_turn=self.source_turn,
                claim=self.claim,
                evidence=self.evidence,
                created_at=self.created_at,
            ),
            "created_at": float(self.created_at),
            "claim": self.claim,
            "confidence": float(self.confidence),
            "evidence": compact_json(self.evidence),
            "artifact_refs": [a.to_json_dict() for a in self.artifact_refs],
            "facts": [f.to_json_dict() for f in self.facts],
            "verdict": None if self.verdict is None else self.verdict.to_json_dict(),
            "recommended_next": self.recommended_next,
            "uncertainty": list(self.uncertainty),
            "source_agent": self.source_agent,
            "source_turn": self.source_turn,
        }

    def to_compact_dict(self) -> JsonDict:
        """Compact packet surface suitable for Planner prompts and debug timelines."""

        return {
            "packet_id": self.packet_id or _make_packet_id(
                source_agent=self.source_agent,
                source_turn=self.source_turn,
                claim=self.claim,
                evidence=self.evidence,
                created_at=self.created_at,
            ),
            "created_at": float(self.created_at),
            "claim": _short_text(self.claim, 240),
            "confidence": float(self.confidence),
            "verdict": None if self.verdict is None else {
                "status": self.verdict.status,
                "confidence": float(self.verdict.confidence),
                "reason": _short_text(self.verdict.reason, 240),
                "target": _short_text(self.verdict.target, 120),
                "verdict_type": self.verdict.verdict_type,
            },
            "recommended_next": _short_text(self.recommended_next, 240),
            "uncertainty": [_short_text(v, 180) for v in self.uncertainty[:8]],
            "evidence_summary": compact_json(self.evidence, max_depth=3, max_items=10, max_string=240),
            "facts": [
                {
                    "key": f.key,
                    "value": compact_json(f.value, max_depth=2, max_items=6, max_string=160),
                    "confidence": float(f.confidence),
                    "source_agent": f.source_agent,
                }
                for f in self.facts[:8]
            ],
            "artifact_refs": [
                {
                    "artifact_id": a.artifact_id,
                    "kind": a.kind,
                    "path": a.path,
                    "producer": a.producer,
                    "summary": _short_text(a.summary, 160),
                }
                for a in self.artifact_refs[:8]
            ],
            "source_agent": self.source_agent,
            "source_turn": self.source_turn,
        }


@dataclass(frozen=True)
class EvidenceTimelineItem:
    """One readable evidence packet record in an episode/subgoal timeline."""

    record_id: str
    source: str
    packet: EvidencePacket
    subgoal_index: int | None = None
    subgoal_goal: str = ""
    subgoal_postcondition: str = ""
    act_status: str = ""
    loaded_skill_ids: tuple[str, ...] = ()

    @classmethod
    def from_packet(
        cls,
        packet: EvidencePacket,
        *,
        source: str,
        subgoal_index: int | None = None,
        subgoal_goal: str = "",
        subgoal_postcondition: str = "",
        act_status: str = "",
        loaded_skill_ids: tuple[str, ...] = (),
    ) -> "EvidenceTimelineItem":
        packet_id = packet.packet_id or _make_packet_id(
            source_agent=packet.source_agent,
            source_turn=packet.source_turn,
            claim=packet.claim,
            evidence=packet.evidence,
            created_at=packet.created_at,
        )
        sg = "episode" if subgoal_index is None else f"subgoal_{subgoal_index:02d}"
        record_id = f"{sg}:{source}:{packet_id}"
        return cls(
            record_id=record_id,
            source=source,
            packet=packet,
            subgoal_index=subgoal_index,
            subgoal_goal=subgoal_goal,
            subgoal_postcondition=subgoal_postcondition,
            act_status=act_status,
            loaded_skill_ids=tuple(loaded_skill_ids),
        )

    @classmethod
    def from_json_dict(cls, raw: JsonDict) -> "EvidenceTimelineItem | None":
        if not isinstance(raw, dict):
            return None
        packet = EvidencePacket.from_any(raw.get("packet") if isinstance(raw.get("packet"), dict) else raw)
        if packet.is_empty:
            return None
        subgoal_index = raw.get("subgoal_index")
        if not isinstance(subgoal_index, int):
            subgoal_index = None
        return cls(
            record_id=str(raw.get("record_id") or ""),
            source=str(raw.get("source") or packet.source_agent or ""),
            packet=packet,
            subgoal_index=subgoal_index,
            subgoal_goal=str(raw.get("subgoal_goal") or ""),
            subgoal_postcondition=str(raw.get("subgoal_postcondition") or ""),
            act_status=str(raw.get("act_status") or ""),
            loaded_skill_ids=tuple(str(v) for v in raw.get("loaded_skill_ids", ()) or ()),
        )

    def to_json_dict(self) -> JsonDict:
        return {
            "schema": "robomex.evidence_timeline_item.v1",
            "record_id": self.record_id,
            "source": self.source,
            "subgoal_index": self.subgoal_index,
            "subgoal_goal": self.subgoal_goal,
            "subgoal_postcondition": self.subgoal_postcondition,
            "act_status": self.act_status,
            "loaded_skill_ids": list(self.loaded_skill_ids),
            "packet": self.packet.to_compact_dict(),
        }

    def render_line(self) -> str:
        compact = self.packet.to_compact_dict()
        verdict = compact.get("verdict")
        verdict_text = ""
        if isinstance(verdict, dict) and verdict.get("status"):
            reason = f": {verdict.get('reason')}" if verdict.get("reason") else ""
            verdict_text = f"; verdict={verdict.get('status')}{reason}"
        next_text = f"; next={compact['recommended_next']}" if compact.get("recommended_next") else ""
        uncertainty = compact.get("uncertainty") or []
        uncertainty_text = f"; uncertainty={'; '.join(uncertainty[:3])}" if uncertainty else ""
        return (
            f"- {self.record_id}: {compact.get('claim') or '(no claim)'} "
            f"(conf={float(compact.get('confidence') or 0.0):.2f}, source={self.source})"
            f"{verdict_text}{next_text}{uncertainty_text}"
        )


@dataclass
class EvidenceTimeline:
    """Episode-level readable timeline of compact evidence packets."""

    records: list[EvidenceTimelineItem] = field(default_factory=list)

    def add_packet(
        self,
        packet: EvidencePacket,
        *,
        source: str,
        subgoal_index: int | None = None,
        subgoal_goal: str = "",
        subgoal_postcondition: str = "",
        act_status: str = "",
        loaded_skill_ids: tuple[str, ...] = (),
    ) -> EvidenceTimelineItem | None:
        if packet.is_empty:
            return None
        item = EvidenceTimelineItem.from_packet(
            packet,
            source=source,
            subgoal_index=subgoal_index,
            subgoal_goal=subgoal_goal,
            subgoal_postcondition=subgoal_postcondition,
            act_status=act_status,
            loaded_skill_ids=loaded_skill_ids,
        )
        self.records.append(item)
        return item

    def extend_from_trace_metadata(
        self,
        meta: JsonDict,
        *,
        subgoal_index: int | None = None,
        subgoal_goal: str = "",
        subgoal_postcondition: str = "",
        act_status: str = "",
        loaded_skill_ids: tuple[str, ...] = (),
    ) -> None:
        for i, raw in enumerate(meta.get("evidence_packets") or ()):
            if isinstance(raw, dict):
                source = str(raw.get("source") or f"evidence:{i}")
                packet_raw = raw.get("packet") if isinstance(raw.get("packet"), dict) else raw
            else:
                source = f"evidence:{i}"
                packet_raw = raw
            packet = EvidencePacket.from_any(packet_raw, default_source=source)
            self.add_packet(
                packet,
                source=source,
                subgoal_index=subgoal_index,
                subgoal_goal=subgoal_goal,
                subgoal_postcondition=subgoal_postcondition,
                act_status=act_status,
                loaded_skill_ids=loaded_skill_ids,
            )

    def to_json_dict(self) -> JsonDict:
        return {
            "schema": "robomex.evidence_timeline.v1",
            "records": [r.to_json_dict() for r in self.records],
        }

    def compact_snapshot(self, *, max_records: int = 24) -> JsonDict:
        return {
            "records": [r.to_json_dict() for r in self.records[-max_records:]],
        }

    def render_compact(self, *, max_records: int = 12) -> str:
        if not self.records:
            return "(no evidence packets recorded yet)"
        return "\n".join(r.render_line() for r in self.records[-max_records:])

    def render_markdown(self) -> str:
        lines = ["# RoboMEx Evidence Timeline", ""]
        if not self.records:
            lines.append("(no evidence packets recorded)")
            return "\n".join(lines) + "\n"
        current_sg: int | None | object = object()
        for record in self.records:
            if record.subgoal_index != current_sg:
                current_sg = record.subgoal_index
                title = "Episode" if record.subgoal_index is None else f"Subgoal {record.subgoal_index}"
                if record.subgoal_goal:
                    title += f": {record.subgoal_goal}"
                lines.extend(["", f"## {title}", ""])
            lines.append(record.render_line())
        return "\n".join(lines).strip() + "\n"


@dataclass(frozen=True)
class PrimitiveTrace:
    """A structured trace record for one primitive, sidecar helper, or action execution."""

    trace_id: str
    primitive_name: str
    status: str
    producer: str = ""
    block_name: str = ""
    turn: int | None = None
    inputs_summary: JsonDict = field(default_factory=dict)
    outputs_summary: JsonDict = field(default_factory=dict)
    error: str = ""
    artifact_refs: tuple[ArtifactRef, ...] = ()
    local_verdict: LocalVerdict | None = None
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_any(cls, raw: Any, *, default_id: str = "", default_producer: str = "") -> "PrimitiveTrace | None":
        if not isinstance(raw, dict):
            return None
        trace_id = str(raw.get("trace_id") or raw.get("id") or default_id).strip()
        primitive_name = str(raw.get("primitive_name") or raw.get("primitive") or raw.get("event_type") or raw.get("type") or "").strip()
        status = str(raw.get("status") or raw.get("ok") or raw.get("message") or "unknown")
        if not trace_id:
            return None
        artifacts = tuple(
            artifact
            for artifact in (
                ArtifactRef.from_any(item, default_producer=default_producer)
                for item in raw.get("artifact_refs", ()) or raw.get("artifacts", ()) or ()
            )
            if artifact is not None
        )
        verdict = LocalVerdict.from_any(raw.get("local_verdict") or raw.get("verdict"))
        return cls(
            trace_id=trace_id,
            primitive_name=primitive_name or "primitive",
            status=status,
            producer=str(raw.get("producer") or default_producer),
            block_name=str(raw.get("block_name") or ""),
            turn=int(raw["turn"]) if isinstance(raw.get("turn"), int) else None,
            inputs_summary=_json_safe(raw.get("inputs_summary") if isinstance(raw.get("inputs_summary"), dict) else {}),
            outputs_summary=_json_safe(raw.get("outputs_summary") if isinstance(raw.get("outputs_summary"), dict) else {}),
            error=str(raw.get("error") or raw.get("stderr") or ""),
            artifact_refs=artifacts,
            local_verdict=verdict,
            timestamp=float(raw.get("timestamp", time.time())),
        )

    def to_json_dict(self) -> JsonDict:
        return {
            "trace_id": self.trace_id,
            "primitive_name": self.primitive_name,
            "status": self.status,
            "producer": self.producer,
            "block_name": self.block_name,
            "turn": self.turn,
            "inputs_summary": _json_safe(self.inputs_summary),
            "outputs_summary": _json_safe(self.outputs_summary),
            "error": self.error,
            "artifact_refs": [a.to_json_dict() for a in self.artifact_refs],
            "local_verdict": None if self.local_verdict is None else self.local_verdict.to_json_dict(),
            "timestamp": float(self.timestamp),
        }


@dataclass(frozen=True)
class AttemptRecord:
    """One physical or logical attempt whose outcome should affect future choices."""

    attempt_id: str
    object_key: str = ""
    strategy: str = ""
    pose_or_target: Any = None
    related_trace_ids: tuple[str, ...] = ()
    outcome: str = "unknown"
    failure_reason: str = ""
    invalidated_facts: tuple[str, ...] = ()
    recommended_repair: str = ""
    confidence: float = 0.0
    artifact_refs: tuple[ArtifactRef, ...] = ()
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_any(cls, raw: Any, *, default_id: str = "") -> "AttemptRecord | None":
        if not isinstance(raw, dict):
            return None
        attempt_id = str(raw.get("attempt_id") or raw.get("id") or default_id).strip()
        if not attempt_id:
            return None
        artifacts = tuple(
            artifact
            for artifact in (ArtifactRef.from_any(item) for item in raw.get("artifact_refs", ()) or ())
            if artifact is not None
        )
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return cls(
            attempt_id=attempt_id,
            object_key=str(raw.get("object_key") or raw.get("object") or ""),
            strategy=str(raw.get("strategy") or ""),
            pose_or_target=_json_safe(raw.get("pose_or_target") if "pose_or_target" in raw else raw.get("pose")),
            related_trace_ids=tuple(str(v) for v in raw.get("related_trace_ids", ()) or raw.get("trace_ids", ()) or ()),
            outcome=str(raw.get("outcome") or raw.get("status") or "unknown"),
            failure_reason=str(raw.get("failure_reason") or raw.get("reason") or ""),
            invalidated_facts=tuple(str(v) for v in raw.get("invalidated_facts", ()) or ()),
            recommended_repair=str(raw.get("recommended_repair") or raw.get("recommended_next") or ""),
            confidence=confidence,
            artifact_refs=artifacts,
            timestamp=float(raw.get("timestamp", time.time())),
        )

    def to_json_dict(self) -> JsonDict:
        return {
            "attempt_id": self.attempt_id,
            "object_key": self.object_key,
            "strategy": self.strategy,
            "pose_or_target": _json_safe(self.pose_or_target),
            "related_trace_ids": list(self.related_trace_ids),
            "outcome": self.outcome,
            "failure_reason": self.failure_reason,
            "invalidated_facts": list(self.invalidated_facts),
            "recommended_repair": self.recommended_repair,
            "confidence": float(self.confidence),
            "artifact_refs": [a.to_json_dict() for a in self.artifact_refs],
            "timestamp": float(self.timestamp),
        }


@dataclass(frozen=True)
class Diagnosis:
    """Global failure attribution and repair-routing output."""

    diagnosis_id: str
    failed_primitive: str = ""
    failure_type: str = ""
    evidence_trace_ids: tuple[str, ...] = ()
    invalidated_facts: tuple[str, ...] = ()
    next_route: str = ""
    recommended_skill: str = ""
    requires_large_model: bool = False
    confidence: float = 0.0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_any(cls, raw: Any, *, default_id: str = "") -> "Diagnosis | None":
        if not isinstance(raw, dict):
            return None
        diagnosis_id = str(raw.get("diagnosis_id") or raw.get("id") or default_id).strip()
        if not diagnosis_id:
            return None
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return cls(
            diagnosis_id=diagnosis_id,
            failed_primitive=str(raw.get("failed_primitive") or ""),
            failure_type=str(raw.get("failure_type") or raw.get("failure_reason") or ""),
            evidence_trace_ids=tuple(str(v) for v in raw.get("evidence_trace_ids", ()) or raw.get("trace_ids", ()) or ()),
            invalidated_facts=tuple(str(v) for v in raw.get("invalidated_facts", ()) or ()),
            next_route=str(raw.get("next_route") or raw.get("recommended_next") or ""),
            recommended_skill=str(raw.get("recommended_skill") or ""),
            requires_large_model=bool(raw.get("requires_large_model", False)),
            confidence=confidence,
            reason=str(raw.get("reason") or ""),
            timestamp=float(raw.get("timestamp", time.time())),
        )

    def to_json_dict(self) -> JsonDict:
        return {
            "diagnosis_id": self.diagnosis_id,
            "failed_primitive": self.failed_primitive,
            "failure_type": self.failure_type,
            "evidence_trace_ids": list(self.evidence_trace_ids),
            "invalidated_facts": list(self.invalidated_facts),
            "next_route": self.next_route,
            "recommended_skill": self.recommended_skill,
            "requires_large_model": self.requires_large_model,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "timestamp": float(self.timestamp),
        }


@dataclass
class TraceStore:
    """Episode-level primitive trace store."""

    traces: dict[str, PrimitiveTrace] = field(default_factory=dict)

    def add(self, trace: PrimitiveTrace) -> None:
        self.traces[trace.trace_id] = trace

    def extend(self, traces: list[PrimitiveTrace] | tuple[PrimitiveTrace, ...]) -> None:
        for trace in traces:
            self.add(trace)

    def to_json_dict(self) -> JsonDict:
        return {"traces": {k: v.to_json_dict() for k, v in sorted(self.traces.items())}}

    def compact_snapshot(self, *, max_traces: int = 24) -> JsonDict:
        traces = sorted(self.traces.values(), key=lambda t: t.timestamp, reverse=True)[:max_traces]
        return {"traces": [t.to_json_dict() for t in traces]}


@dataclass
class AttemptHistory:
    """Episode-level memory of attempts and outcomes."""

    records: list[AttemptRecord] = field(default_factory=list)

    def add(self, record: AttemptRecord) -> None:
        self.records.append(record)

    def extend(self, records: list[AttemptRecord] | tuple[AttemptRecord, ...]) -> None:
        for record in records:
            self.add(record)

    def to_json_dict(self) -> JsonDict:
        return {"records": [r.to_json_dict() for r in self.records]}

    def compact_snapshot(self, *, max_records: int = 24) -> JsonDict:
        return {"records": [r.to_json_dict() for r in self.records[-max_records:]]}

    def render_compact(self, *, max_records: int = 12) -> str:
        if not self.records:
            return "(no attempts recorded yet)"
        lines = []
        for record in self.records[-max_records:]:
            target = f" {record.object_key}" if record.object_key else ""
            strategy = f" via {record.strategy}" if record.strategy else ""
            reason = f"; reason={record.failure_reason}" if record.failure_reason else ""
            lines.append(f"- {record.attempt_id}:{target}{strategy} -> {record.outcome}{reason}")
        return "\n".join(lines)


@dataclass
class DiagnosisStore:
    """Episode-level diagnosis outputs."""

    diagnoses: list[Diagnosis] = field(default_factory=list)

    def add(self, diagnosis: Diagnosis) -> None:
        self.diagnoses.append(diagnosis)

    def extend(self, diagnoses: list[Diagnosis] | tuple[Diagnosis, ...]) -> None:
        for diagnosis in diagnoses:
            self.add(diagnosis)

    def to_json_dict(self) -> JsonDict:
        return {"diagnoses": [d.to_json_dict() for d in self.diagnoses]}

    def compact_snapshot(self, *, max_diagnoses: int = 12) -> JsonDict:
        return {"diagnoses": [d.to_json_dict() for d in self.diagnoses[-max_diagnoses:]]}

    def render_compact(self, *, max_diagnoses: int = 8) -> str:
        if not self.diagnoses:
            return "(no diagnoses recorded yet)"
        lines = []
        for diagnosis in self.diagnoses[-max_diagnoses:]:
            failed = f" {diagnosis.failed_primitive}" if diagnosis.failed_primitive else ""
            kind = diagnosis.failure_type or "unknown_failure"
            route = f"; next={diagnosis.next_route}" if diagnosis.next_route else ""
            reason = f"; reason={diagnosis.reason}" if diagnosis.reason else ""
            lines.append(f"- {diagnosis.diagnosis_id}:{failed} -> {kind}{route}{reason}")
        return "\n".join(lines)


def _result_payload(raw: JsonDict) -> JsonDict:
    result = raw.get("result")
    if isinstance(result, dict):
        return {**result, **{k: v for k, v in raw.items() if k != "result"}}
    return raw


def _coerce_confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_timestamp(value: Any) -> float:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        ts = time.time()
    return ts


def _short_text(value: Any, max_len: int) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _make_packet_id(
    *,
    source_agent: str,
    source_turn: str,
    claim: str,
    evidence: Any,
    created_at: float,
) -> str:
    try:
        body = json.dumps(
            {
                "source_agent": source_agent,
                "source_turn": source_turn,
                "claim": claim,
                "evidence": compact_json(evidence, max_depth=2, max_items=8, max_string=120),
                "created_at": round(float(created_at), 3),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    except Exception:
        body = repr((source_agent, source_turn, claim, created_at))
    digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]
    prefix = source_agent or "evidence"
    prefix = "".join(ch if ch.isalnum() else "_" for ch in prefix).strip("_") or "evidence"
    return f"{prefix}:{digest}"


def _iter_artifact_items(payload: JsonDict) -> list[Any]:
    """Collect artifact references from nested typed outputs and evidence.

    Agent finish payloads commonly place files below
    ``outputs.<port>.artifacts``.  Only inspecting the top level silently
    discarded those references and forced downstream agents to recompute
    perception products.
    """

    items: list[Any] = []

    def visit(value: Any, *, key_hint: str = "") -> None:
        if isinstance(value, dict):
            looks_like_ref = any(k in value for k in ("artifact_id", "path", "uri"))
            if looks_like_ref:
                items.append(value)
                return
            for key, nested in value.items():
                if key in {"artifact_refs", "artifacts"}:
                    if isinstance(nested, dict):
                        for name, item in nested.items():
                            if isinstance(item, dict):
                                items.append({"artifact_id": name, "kind": name, **item})
                            else:
                                items.append(
                                    {"artifact_id": name, "path": item, "kind": name}
                                )
                    elif isinstance(nested, (list, tuple)):
                        for item in nested:
                            visit(item, key_hint=key)
                    elif isinstance(nested, str):
                        items.append(nested)
                elif key in {"outputs", "evidence", "result", "payload"}:
                    visit(nested, key_hint=key)
                elif isinstance(nested, (dict, list, tuple)):
                    visit(nested, key_hint=key)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, key_hint=key_hint)

    visit(payload)
    return items


def _iter_packet_facts(
    payload: JsonDict,
    *,
    default_source: str = "",
    default_turn: str = "",
) -> list[StateFact]:
    raw_facts: list[Any] = []
    value = payload.get("facts")
    if isinstance(value, (list, tuple)):
        raw_facts.extend(value)

    facts: list[StateFact] = []
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        fact = StateFact.from_dict(item)
        if not fact.key:
            continue
        if default_source and not fact.source_agent:
            fact = StateFact(
                key=fact.key,
                value=fact.value,
                confidence=fact.confidence,
                source_agent=default_source,
                source_turn=fact.source_turn or default_turn,
                timestamp=fact.timestamp,
                provenance=fact.provenance,
                expires_after_subgoal=fact.expires_after_subgoal,
            )
        facts.append(fact)
    return facts


def _implicit_evidence(payload: JsonDict) -> JsonDict:
    """Capture task-specific fields without turning them into runtime schema."""

    reserved = {
        "claim",
        "confidence",
        "evidence",
        "artifact_refs",
        "artifacts",
        "facts",
        "verdict",
        "local_verdict",
        "recommended_next",
        "next",
        "uncertainty",
        "source",
        "source_agent",
        "source_turn",
        "primitive_traces",
        "traces",
        "attempt_records",
        "attempts",
        "diagnoses",
        "diagnosis",
        "ok",
        "error",
        "schema",
        "packet_id",
        "id",
        "created_at",
        "timestamp",
    }
    return {str(k): v for k, v in payload.items() if k not in reserved}
