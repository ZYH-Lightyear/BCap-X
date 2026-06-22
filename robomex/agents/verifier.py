"""VerifyCodeAgent:独立的、完全 agentic 的验证 Code Agent。

它和执行器是*同一种* agent(一个 :class:`CodingAgent`):读技能、在沙箱写/跑代码、
迭代。它的特化在于:

- 上下文是只含事实的 :class:`VerifierContext`(sub-goal、用过哪些技能、一份脱敏的
  op-trace、作者写的 rubric)——而*不含*执行器的思维链,这样它的盲区与执行器不相关。
- 它复用执行器的同一沙箱:执行器在子目标开场种好的 ``EVIDENCE``(技能发布的关键中间值,
  如 ``EVIDENCE['target_box']``)与 ``OBS_BEFORE``(起始帧)持久可读;开场再补 ``OBS_AFTER``
  (当前帧)与 ``draw_box`` 助手。它据此写代码、用沙箱里的 ``query_vlm`` 在证据上判断。
- 它可以 ``USE SKILL`` 拉取某技能完整的 SKILL.md + 其 ``ref/verify.md`` rubric 作为参考。
- 它通过输出一个裸 JSON 裁决(而非代码围栏)来终止。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from robomex.core.coder import CodingAgent, SkillEntry
from robomex.core.coder.policy import CompletionPolicy
from robomex.core.sandbox import BlockExecutionResult
from robomex.verification.context import VerifierContext, sanitize_code
from robomex.verification.verifier import (
    VerificationResult,
    VerificationSignal,
    VerificationStatus,
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_VERIFY_SYSTEM_PROMPT = (
    "You are an INDEPENDENT robot-task Verifier. You did NOT write the executor's code; you "
    "are given only facts about what it did (the sub-goal, which skills it used, a sanitized "
    "op-trace, and the authored verify.md rubric) -- never its reasoning. Decide whether the "
    "sub-goal was actually achieved, against real visual evidence.\n\n"
    "The sandbox persists the executor's state. Available to your code: get_observation(); "
    "OBS_BEFORE / OBS_AFTER (RGB arrays of the scene at the sub-goal's start / now); EVIDENCE "
    "(a dict of key values the skills published, e.g. EVIDENCE.get('target_box')); "
    "query_vlm(prompt, images=[...]) to ask a VLM; draw_box(img, box) to outline a region.\n\n"
    "RULE: never corrupt the pixels of the object being judged -- annotate with an OUTLINE "
    "bounding box or pass box coordinates as text; never paint a filled mask over the object.\n\n"
    "Each turn you may: consult a skill with `USE SKILL: <name>` (you get its full SKILL.md + "
    "its ref/verify.md rubric), or write ONE ```python``` block to gather evidence and judge "
    "with query_vlm. When confident, FINISH by replying with a single bare JSON object (NOT in "
    'a code fence): {"verdict": "passed"|"failed"|"uncertain", "confidence": 0.0-1.0, '
    '"reason": "...", "evidence": {"overlay": "<path or null>"}}.'
)

# 验证器开场注入沙箱:抓当前帧 OBS_AFTER + 注入 draw_box(只画外框,不污染物体像素);
# EVIDENCE / OBS_BEFORE 由执行器在子目标开场种好,这里只做缺失兜底。
_VERIFY_SEED = (
    "try:\n    EVIDENCE\nexcept NameError:\n    EVIDENCE = {}\n"
    "try:\n    OBS_BEFORE\nexcept NameError:\n    OBS_BEFORE = None\n"
    "try:\n    OBS_AFTER = get_observation()['agentview']['images']['rgb']\n"
    "except Exception as _e:\n    OBS_AFTER = None\n"
    "def draw_box(img, box, color=(255, 0, 0), width=3):\n"
    "    from PIL import Image as _I, ImageDraw as _D\n"
    "    import numpy as _n\n"
    "    _im = _I.fromarray(_n.asarray(img).astype('uint8')).copy()\n"
    "    _D.Draw(_im).rectangle([float(_v) for _v in box], outline=color, width=width)\n"
    "    return _n.asarray(_im)\n"
)


@dataclass(frozen=True)
class VerifyTurn:
    """验证器的一个轮次:它跑的 judge 代码 + 沙箱输出。"""

    turn: int
    code: str
    stdout: str
    stderr: str


@dataclass(frozen=True)
class VerifyVerdict:
    """验证器最终的结构化裁决。"""

    verdict: str = "uncertain"
    confidence: float = 0.0
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    def to_result(self, min_confidence: float = 0.6) -> VerificationResult:
        verdict = self.verdict
        if verdict == "passed" and self.confidence < min_confidence:
            verdict = "uncertain"
        status = {
            "passed": VerificationStatus.PASSED,
            "failed": VerificationStatus.FAILED,
        }.get(verdict, VerificationStatus.UNCERTAIN)
        return VerificationResult(
            status=status,
            signals=(
                VerificationSignal(
                    "verify_code_agent", status, confidence=self.confidence, message=self.reason
                ),
            ),
            summary=f"VerifyCodeAgent: {verdict} ({self.confidence:.2f}) {self.reason}",
            metadata={"evidence": self.evidence},
        )


@dataclass(frozen=True)
class VerifyAgentTrace:
    """验证器的 episode:裁决、各 judge 轮次,以及一份脱敏的 op-trace。"""

    verdict: VerifyVerdict
    turns: tuple[VerifyTurn, ...] = ()
    op_trace: tuple[str, ...] = ()

    @property
    def result(self) -> VerificationResult:
        return self.verdict.to_result()


class VerifyCodeAgent(CodingAgent):
    def __init__(
        self,
        executor: Any,
        policy: CompletionPolicy,
        context: VerifierContext,
        library: Any,
        *,
        max_turns: int = 6,
        system_prompt: str = _VERIFY_SYSTEM_PROMPT,
    ) -> None:
        super().__init__(
            executor=executor,
            policy=policy,
            library=library,
            max_turns=max_turns,
            system_prompt=system_prompt,
            force_terminal_on_exhaust=True,
        )
        self.context = context

    def verify(self) -> VerifyAgentTrace:
        return self.run()

    # ---- 钩子 --------------------------------------------------------------

    def _setup(self, prompt: list[dict]) -> None:
        """注入当前帧 OBS_AFTER 与 draw_box 助手(EVIDENCE/OBS_BEFORE 已由执行器种好)。"""

        from robomex.core.sandbox import SemanticActionBlock

        try:
            self.executor.run_block(
                SemanticActionBlock(name="verify_seed", intent="seed verifier evidence", code=_VERIFY_SEED)
            )
        except Exception as exc:  # noqa: BLE001 - 种子失败不该让验证崩溃
            from robomex.core.logging import get_logger

            get_logger("verifier").warning("验证器证据种子注入失败: %r", exc)

    def _skill_entries(self) -> list[SkillEntry]:
        entries: list[SkillEntry] = []
        for sid in self.context.skills_used:
            try:
                record = self.library.get(sid)
            except Exception:  # noqa: BLE001 - missing skill just drops from the menu
                continue
            entries.append(
                SkillEntry(
                    name=sid,
                    description=record.skill.description or record.skill.name,
                    category=record.skill.category.value,
                )
            )
        return entries

    def _initial_user_message(self) -> str:
        return (
            f"{self.context.render()}\n\n"
            "Evidence available in the sandbox: OBS_BEFORE / OBS_AFTER (scene at the sub-goal's "
            "start / now), and EVIDENCE (dict the skills published, e.g. EVIDENCE.get('target_box')). "
            "Judge against the rubric using query_vlm (annotate only with draw_box / coordinates, "
            "never a filled mask over the object). Finish with a bare JSON verdict."
        )

    def _is_terminal(self, raw: str) -> bool:
        match = _JSON_RE.search(raw or "")
        if not match:
            return False
        try:
            return "verdict" in json.loads(match.group(0))
        except json.JSONDecodeError:
            return False

    def _on_python_turn(
        self,
        turn_idx: int,
        code: str,
        execution: BlockExecutionResult,
        prev_observation: dict | None,
        turns: list[Any],
    ) -> None:
        turns.append(VerifyTurn(turn_idx, code, execution.stdout, execution.stderr))

    def _force_terminal_message(self) -> str:
        return (
            "You are out of steps. Output your best-effort verdict now as a single bare JSON "
            'object: {"verdict": ..., "confidence": ..., "reason": ..., "evidence": {...}}.'
        )

    def _finalize(self, *, turns: list[Any], loaded: tuple[str, ...], terminal_raw: str | None) -> VerifyAgentTrace:
        verdict = _parse_verdict(terminal_raw or "")
        op_trace = tuple(sanitize_code(t.code) for t in turns if sanitize_code(t.code))
        return VerifyAgentTrace(verdict=verdict, turns=tuple(turns), op_trace=op_trace)


def _parse_verdict(raw: str) -> VerifyVerdict:
    match = _JSON_RE.search(raw)
    if not match:
        return VerifyVerdict(reason="verifier produced no parseable JSON verdict", raw=raw)
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return VerifyVerdict(reason="verifier verdict was malformed JSON", raw=raw)
    return VerifyVerdict(
        verdict=str(payload.get("verdict", "uncertain")).lower(),
        confidence=float(payload.get("confidence", 0.0)),
        reason=str(payload.get("reason", "")),
        evidence=dict(payload.get("evidence", {}) or {}),
        raw=raw,
    )
