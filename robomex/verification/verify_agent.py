"""VerifyCodeAgent: the independent, fully-agentic verification Code Agent.

It is the *same* kind of agent as the executor (a :class:`CodingAgent`): it reads
skills, writes/runs code in the sandbox, and iterates. Its specialisation:

- Context is the fact-only :class:`VerifierContext` (sub-goal, skills used, the
  executor's CLAIM manifest, expected-vs-actual op-trace, authored rubrics) -- and
  *not* the executor's chain-of-thought, so its blind spots stay uncorrelated.
- It can ``USE SKILL`` to pull a skill's full SKILL.md + base directory, then read
  its ``ref/verify.md`` rubric and ``scripts/verify.py`` primitive (also seeded as
  callable ``VERIFY_PRIMITIVES[skill_id]`` for ergonomics). These are references it
  may compose, copy, or adapt -- there is no mandatory deterministic floor.
- It terminates by emitting a bare JSON verdict (not a code fence).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from robomex.agent.coding_agent import CodingAgent, SkillEntry
from robomex.agent.policy import CompletionPolicy
from robomex.execution import BlockExecutionResult
from robomex.verification.context import VerifierContext, sanitize_code
from robomex.verification.verifier import (
    VerificationResult,
    VerificationSignal,
    VerificationStatus,
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_VERIFY_SYSTEM_PROMPT = (
    "You are an INDEPENDENT robot-task Verifier. You did NOT write the executor's code; "
    "you are given only facts about what it did (which skills, its claimed result, an "
    "expected-vs-actual op-trace, and authored rubrics) -- never its reasoning. Your job "
    "is to verify or refute the executor's claim against real evidence.\n\n"
    "Each turn you may: consult a skill with `USE SKILL: <name>` (you will get its full "
    "SKILL.md + base directory; read its 'Verifier reference' section for the exact functions "
    "and signatures its scripts/verify.py exposes, plus its ref/verify.md rubric); or write ONE "
    "```python``` block to gather evidence. Each used skill's verify.py is preloaded as "
    "`VERIFY_PRIMITIVES[skill_id]` (a namespace already wired to the sandbox's APIs and the "
    "executor's RESULT manifest). Many skills offer a one-shot `verify(...)` entry that loads the "
    "executor's artifacts, renders an evidence overlay, and VLM-judges it in a single call -- "
    "prefer it when available, or compose the building blocks (e.g. load_claim / render_evidence) "
    "or `vlm_judge(image_path, rubric, question)`. Use them, adapt them, or write your own checks. "
    "If unsure of a primitive's API, USE SKILL to read its reference first.\n\n"
    "When confident, FINISH by replying with a single bare JSON object (NOT in a code fence): "
    '{"verdict": "passed"|"failed"|"uncertain", "confidence": 0.0-1.0, "reason": "...", '
    '"evidence": {"overlay": "<path or null>"}}.'
)


@dataclass(frozen=True)
class VerifyTurn:
    """One verifier turn: the judge code it ran and the sandbox output."""

    turn: int
    code: str
    stdout: str
    stderr: str


@dataclass(frozen=True)
class VerifyVerdict:
    """The verifier's final structured verdict."""

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
    """The verifier episode: its verdict, the judge turns, and a sanitized op-trace."""

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
        primitive_model: str = "openrouter/qwen/qwen3.6-plus",
        primitive_server_url: str = "http://localhost:8110/chat/completions",
        primitive_api_key: str | None = None,
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
        self._primitive_model = primitive_model
        self._primitive_server_url = primitive_server_url
        self._primitive_api_key = primitive_api_key

    def verify(self) -> VerifyAgentTrace:
        return self.run()

    # ---- hooks -------------------------------------------------------------

    def _setup(self, prompt: list[dict]) -> None:
        """Seed verify.py primitives + vlm_judge into the sandbox (best-effort).

        Each skill's ``scripts/verify.py`` is ``exec``'d into a namespace that
        *inherits the sandbox globals* (the L4 APIs + the executor's ``RESULT``),
        because its helpers fetch those via ``globals()`` -- it is written to run
        inside the sandbox, not be imported standalone. We wrap the resulting
        namespace as ``VERIFY_PRIMITIVES[skill_id]`` so the agent can call e.g.
        ``VERIFY_PRIMITIVES['estimate_geometry'].verify(...)``.
        """

        paths = {
            sid: res.verifier_path
            for sid, res in self.context.resources.items()
            if res.verifier_path
        }
        seed = (
            "import types as _t\n"
            "try:\n    VERIFY_PRIMITIVES\nexcept NameError:\n    VERIFY_PRIMITIVES = {}\n"
            f"for _sid, _p in {paths!r}.items():\n"
            "    try:\n"
            "        _ns = dict(globals())  # share sandbox L4 APIs + RESULT with verify.py\n"
            "        with open(_p) as _f:\n"
            "            exec(compile(_f.read(), _p, 'exec'), _ns)\n"
            "        VERIFY_PRIMITIVES[_sid] = _t.SimpleNamespace("
            "**{_k: _v for _k, _v in _ns.items() if not _k.startswith('__')})\n"
            "    except Exception as _e:\n"
            "        print('seed primitive failed for ' + _sid + ': ' + repr(_e))\n"
            "try:\n"
            "    from robomex.verification.primitives import vlm_judge as _vj\n"
            "    import functools as _ft\n"
            f"    vlm_judge = _ft.partial(_vj, model={self._primitive_model!r}, "
            f"server_url={self._primitive_server_url!r}, api_key={self._primitive_api_key!r})\n"
            "except Exception as _e:\n"
            "    print('seed vlm_judge failed: ' + repr(_e))\n"
        )
        from robomex.execution import SemanticActionBlock

        self.executor.run_block(
            SemanticActionBlock(name="verify_seed", intent="seed verifier primitives", code=seed)
        )

    def _skill_entries(self) -> list[SkillEntry]:
        entries: list[SkillEntry] = []
        for sid in self.context.skills_used:
            try:
                record = self.library.get(sid)
            except Exception:  # noqa: BLE001 - missing skill just drops from the menu
                continue
            note = "" if record.skill.verifier_path() else " [no verify.py]"
            entries.append(
                SkillEntry(
                    name=sid,
                    description=(record.skill.description or record.skill.name) + note,
                    category=record.skill.category.value,
                )
            )
        return entries

    def _initial_user_message(self) -> str:
        return (
            f"{self.context.render()}\n\n"
            "Gather evidence and decide. Preloaded: VERIFY_PRIMITIVES[skill_id] (the skill's "
            "verify.py, e.g. a one-shot verify(object_name, out_dir, rubric_path=...) or "
            "load_claim/render_evidence) and vlm_judge(image_path, rubric, question). If you render "
            "an overlay, save it under EVIDENCE_DIR when that variable is defined in the sandbox. "
            "Finish with a bare JSON verdict."
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
