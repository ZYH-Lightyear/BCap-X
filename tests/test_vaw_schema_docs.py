"""Keep the two normative VAW docs pinned to the live Canvas schema."""

from __future__ import annotations

import pathlib
import re

from vaw.context_runtime.packet import CONTEXT_SCHEMA, CONTEXT_WEB_SCHEMA_VERSION
from vaw.context_runtime.web_renderer import ContextWebRenderer

_VAW_ROOT = pathlib.Path(__file__).resolve().parents[1] / "vaw"
_NORMATIVE_DOCS = (
    _VAW_ROOT / "CURRENT_ARCHITECTURE.md",
    _VAW_ROOT / "AGENTIC_CONTEXT_OS.md",
)
_STALE_SCHEMA = "vaw-context-v41-evidence-lifecycle"


def test_normative_docs_declare_live_context_schema() -> None:
    renderer = ContextWebRenderer.name
    version = str(CONTEXT_WEB_SCHEMA_VERSION)
    for path in _NORMATIVE_DOCS:
        text = path.read_text(encoding="utf-8")
        assert CONTEXT_SCHEMA in text, f"{path.name} missing {CONTEXT_SCHEMA}"
        assert version in text, f"{path.name} missing web schema {version}"
        assert renderer in text, f"{path.name} missing renderer {renderer}"
        assert _STALE_SCHEMA not in text, f"{path.name} still cites {_STALE_SCHEMA}"
        claimed = set(re.findall(r"vaw-context-v\d+-[a-z0-9-]+", text))
        assert claimed == {CONTEXT_SCHEMA}, f"{path.name} cites unexpected schemas: {claimed}"
