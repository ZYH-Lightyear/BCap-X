"""Provider-independent JSON action framing.

The wire format stays intentionally small: one object with ``tool`` and ``args``.
Transport noise around that first object is retained for diagnostics but is never
part of canonical conversation history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class StructuredOutputConfig:
    mode: Literal["text_json", "json_object", "json_schema"] = "text_json"


@dataclass(frozen=True)
class ActionEnvelope:
    tool: str
    args: dict[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args}

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ActionSchema:
    """Minimal per-agent action surface.

    Detailed domain checks remain in the action handler, where artifact and runtime
    state are available. This schema only constrains the stable transport envelope.
    """

    tools: frozenset[str]

    def to_json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["tool"],
            "properties": {
                "tool": {"type": "string", "enum": sorted(self.tools)},
                "args": {"type": "object"},
            },
            "additionalProperties": False,
        }

    def validate(self, envelope: ActionEnvelope) -> str:
        if self.tools and envelope.tool not in self.tools:
            allowed = ", ".join(sorted(self.tools))
            return f"Action {envelope.tool!r} is not available; expected one of: {allowed}."
        return ""


@dataclass(frozen=True)
class ParsedActionFrame:
    raw: str
    envelope: ActionEnvelope | None = None
    prefix: str = ""
    suffix: str = ""
    error: str = ""

    @property
    def canonical_json(self) -> str:
        return self.envelope.to_json() if self.envelope is not None else ""

    @property
    def has_quarantined_text(self) -> bool:
        return bool(self.prefix.strip() or self.suffix.strip())


def parse_action_frame(
    raw: str,
    *,
    schema: ActionSchema | None = None,
) -> ParsedActionFrame:
    """Extract and validate the first complete JSON object from model content.

    Only the first syntactically complete object is considered. A malformed or
    non-action first object is an error; later objects are never searched for.
    """

    text = raw or ""
    if not text.strip():
        return ParsedActionFrame(raw=text, error="Model returned empty content.")

    candidate, prefix, suffix = _decode_first_object(text)
    if candidate is None:
        repaired = _repair_single_object(text)
        if not isinstance(repaired, dict):
            return ParsedActionFrame(raw=text, error="Expected one JSON object action.")
        candidate = repaired
        prefix = ""
        suffix = ""

    tool = candidate.get("tool") or candidate.get("action")
    args = candidate.get("args", {})
    if not isinstance(tool, str) or not tool.strip():
        return ParsedActionFrame(
            raw=text,
            prefix=prefix,
            suffix=suffix,
            error='JSON action must include a string "tool" field.',
        )
    if not isinstance(args, dict):
        return ParsedActionFrame(
            raw=text,
            prefix=prefix,
            suffix=suffix,
            error='JSON action "args" must be an object.',
        )

    envelope = ActionEnvelope(tool=tool.strip(), args=dict(args))
    error = schema.validate(envelope) if schema is not None else ""
    return ParsedActionFrame(
        raw=text,
        envelope=None if error else envelope,
        prefix=prefix,
        suffix=suffix,
        error=error,
    )


def _decode_first_object(text: str) -> tuple[dict[str, Any] | None, str, str]:
    start = text.find("{")
    if start < 0:
        return None, text, ""
    try:
        value, consumed = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None, text[:start], text[start:]
    if not isinstance(value, dict):
        return None, text[:start], text[start + consumed :]
    return value, text[:start], text[start + consumed :]


def _repair_single_object(text: str) -> Any:
    try:
        import json_repair
    except ImportError:
        return None
    try:
        return json_repair.loads(text)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
