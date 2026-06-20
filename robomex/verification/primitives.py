"""Verifier primitive library: optional helpers seeded into the verifier sandbox.

The Verify Code Agent is fully agentic -- there is no mandatory deterministic
floor. These are *building blocks* it may call, adapt, or ignore. The main one
anchors a verdict on a rendered image + an authored rubric via a VLM judge.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_JUDGE_SYSTEM = (
    "You are a robot manipulation verifier. Judge ONLY what is visible in the image, "
    "guided by the rubric. Reply with a single JSON object: "
    '{"verdict": "passed"|"failed"|"uncertain", "confidence": 0.0-1.0, "reason": "..."}'
)


def vlm_judge(
    image_path: str,
    rubric: str,
    question: str,
    *,
    model: str = "openrouter/qwen/qwen3.6-plus",
    server_url: str = "http://localhost:8110/chat/completions",
    api_key: str | None = None,
    max_tokens: int = 512,
) -> dict:
    """Anchor a verdict on a rendered image + rubric. Returns verdict/confidence/reason."""

    from capx.llm.client import ModelQueryArgs, query_model

    data = base64.b64encode(Path(image_path).read_bytes()).decode()
    user = [
        {"type": "text", "text": f"Rubric:\n{rubric}\n\nQuestion: {question}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
    ]
    args = ModelQueryArgs(
        model=model, server_url=server_url, api_key=api_key, temperature=0.0, max_tokens=max_tokens
    )
    content = query_model(
        args,
        [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": user}],
    )["content"]

    match = _JSON_RE.search(content)
    if not match:
        return {"verdict": "uncertain", "confidence": 0.0, "reason": "judge reply not parseable JSON"}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": "uncertain", "confidence": 0.0, "reason": "judge reply was malformed JSON"}
    return {
        "verdict": str(payload.get("verdict", "uncertain")).lower(),
        "confidence": float(payload.get("confidence", 0.0)),
        "reason": str(payload.get("reason", "")),
    }
