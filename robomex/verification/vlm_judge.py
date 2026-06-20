"""Gate-3 effect verification: rendered evidence -> VLM verdict.

The judge receives the BEFORE/AFTER comparison render plus an optional list of
checks (carried in ``block.metadata`` under ``checks``) and must answer with a
JSON verdict including confidence. Low confidence maps to UNCERTAIN rather than
FAILED so the agent re-perceives instead of aborting on a shaky judgment.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from robomex.execution import BlockExecutionResult, SemanticActionBlock
from robomex.perception import EvidenceRole, MultimodalEvidenceBundle
from robomex.verification.verifier import (
    VerificationResult,
    VerificationSignal,
    VerificationStatus,
    Verifier,
)

_SYSTEM_PROMPT = (
    "You are a robot manipulation verifier. You see a BEFORE/AFTER comparison of the "
    "workspace around one executed code block. Judge ONLY what is visible. "
    "Reply with a single JSON object: "
    '{"verdict": "passed"|"failed"|"uncertain", "confidence": 0.0-1.0, "reason": "..."}'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _image_part(path: str) -> dict:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


class VLMJudgeVerifier(Verifier):
    """Judges block effects from rendered evidence via CapX's LLM client."""

    def __init__(
        self,
        model: str = "openrouter/qwen/qwen3.6-plus",
        server_url: str = "http://localhost:8110/chat/completions",
        api_key: str | None = None,
        min_confidence: float = 0.6,
        max_tokens: int = 512,
    ) -> None:
        from capx.llm.client import ModelQueryArgs, query_model

        self._query_model = query_model
        self._args = ModelQueryArgs(
            model=model,
            server_url=server_url,
            api_key=api_key,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        self.min_confidence = min_confidence

    def verify(
        self,
        *,
        block: SemanticActionBlock,
        execution: BlockExecutionResult,
        evidence: MultimodalEvidenceBundle | None = None,
    ) -> VerificationResult:
        cues = evidence.by_role(EvidenceRole.VERIFICATION_CUE) if evidence else ()
        if not cues:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                summary="No rendered evidence available for VLM judging.",
            )

        checks = tuple(block.metadata.get("checks", ()))
        check_lines = [f"- {c}" for c in checks] or ["- Did the intended effect visibly happen?"]
        question_lines = [
            f"Executed block intent: {block.intent}",
            f"Task context: {block.metadata.get('task', 'unknown')}",
            "Verification checks declared by the consulted skills:",
            *check_lines,
            "Did this block achieve its intended effect? Answer with the JSON verdict only.",
        ]
        message_parts: list[dict] = [{"type": "text", "text": "\n".join(question_lines)}]
        message_parts += [_image_part(cue.path) for cue in cues if cue.path]

        prompt = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": message_parts},
        ]
        content = self._query_model(self._args, prompt)["content"]
        verdict, confidence, reason = self._parse(content)

        if verdict == "passed" and confidence < self.min_confidence:
            verdict = "uncertain"
        status = {
            "passed": VerificationStatus.PASSED,
            "failed": VerificationStatus.FAILED,
        }.get(verdict, VerificationStatus.UNCERTAIN)

        return VerificationResult(
            status=status,
            signals=(VerificationSignal("vlm_judge", status, confidence=confidence, message=reason),),
            summary=f"VLM judge: {verdict} ({confidence:.2f}) {reason}",
        )

    @staticmethod
    def _parse(content: str) -> tuple[str, float, str]:
        match = _JSON_RE.search(content)
        if not match:
            return "uncertain", 0.0, "judge reply was not parseable JSON"
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return "uncertain", 0.0, "judge reply was malformed JSON"
        verdict = str(payload.get("verdict", "uncertain")).lower()
        confidence = float(payload.get("confidence", 0.0))
        reason = str(payload.get("reason", ""))
        return verdict, confidence, reason
