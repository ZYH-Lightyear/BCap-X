"""CodeAsPolicy Skill Agent: the executor that consults skills and writes code.

Thin subclass of the unified :class:`~robomex.agent.coding_agent.CodingAgent`.
The executor's specialisation: its context is the task (+observation), it becomes
aware of the *whole* library via progressive disclosure (a short ``<available_skills>``
list + ``USE SKILL`` to pull a body), each python turn is verified + has evidence
bundled, and it terminates on ``FINISH`` or an env success signal.

The skill is consulted, never executed verbatim; the policy emits its own code.
"""

from __future__ import annotations

from typing import Any

from robomex.agent.coding_agent import CodingAgent, SkillEntry
from robomex.agent.policy import CompletionPolicy
from robomex.agent.router import build_query
from robomex.agent.trace import AgentTrace, TurnRecord
from robomex.execution import BlockExecutionResult
from robomex.library import SkillLibrary
from robomex.perception import EvidenceCollector
from robomex.verification.verifier import TaskSignalVerifier, Verifier

_SYSTEM_PROMPT = (
    "You are a robot Code-as-Policy agent. Each turn, write one block of executable "
    "Python that advances the task, grounding every decision in the current observation. "
    "Consult a relevant skill first with `USE SKILL: <name>` (its guidance is advisory -- "
    "adapt it, do not copy it blindly). Reply with a ```python``` code block, a "
    "`USE SKILL: <name>` line, or the word FINISH when the task is complete."
)


class CodeAsPolicyAgent(CodingAgent):
    def __init__(
        self,
        executor: Any,
        policy: CompletionPolicy,
        library: SkillLibrary,
        verifier: Verifier | None = None,
        router: Any = None,  # deprecated: routing is now progressive disclosure; kept for compat
        collector: EvidenceCollector | None = None,
        max_turns: int = 6,
        system_prompt: str = _SYSTEM_PROMPT,
        require_result: bool = False,
    ) -> None:
        super().__init__(
            executor=executor,
            policy=policy,
            library=library,
            max_turns=max_turns,
            system_prompt=system_prompt,
        )
        self.verifier = verifier or TaskSignalVerifier()
        self.collector = collector
        self.require_result = require_result
        self._task = ""
        self._observation_summary = ""

    def run(self, task: str, observation_summary: str = "") -> AgentTrace:
        self._task = task
        self._observation_summary = observation_summary
        return super().run()

    # ---- hooks -------------------------------------------------------------

    def _skill_entries(self) -> list[SkillEntry]:
        return [
            SkillEntry(
                name=r.skill_id,
                description=r.skill.description or r.skill.name,
                category=r.skill.category.value,
            )
            for r in self.library.all()
        ]

    def _initial_user_message(self) -> str:
        parts = [f"Task: {self._task}"]
        if self._observation_summary:
            parts.append(f"Observation: {self._observation_summary}")
        return "\n\n".join(parts)

    def _block_metadata(self) -> dict:
        return {"task": self._task}

    def _on_python_turn(
        self,
        turn_idx: int,
        code: str,
        execution: BlockExecutionResult,
        prev_observation: dict | None,
        turns: list[Any],
    ) -> None:
        evidence = None
        if self.collector is not None:
            evidence = self.collector.bundle_for_block(
                execution.block.name, prev_observation, execution.observation
            )
        verdict = self.verifier.verify(block=execution.block, execution=execution, evidence=evidence)
        turns.append(TurnRecord(turn_idx, code, execution, verdict))

    def _should_stop_after_python(self, execution: BlockExecutionResult) -> bool:
        return bool(execution.terminated)

    def _terminal_block_reason(self, turns: list[Any]) -> str | None:
        """For a result-owing sub-goal, refuse FINISH until a RESULT manifest exists.

        The contract lives here in the loop (not in a skill's ``## Report`` prose):
        a measurement/observation executor must hand the verifier a machine-readable
        manifest. An env-success turn (``terminated``) is exempt -- action sub-goals
        are checked by the env signal, not a manifest.
        """
        if not self.require_result:
            return None
        for turn in turns:
            if turn.execution.terminated or turn.execution.result is not None:
                return None
        return (
            "You replied FINISH but recorded no structured RESULT for this sub-goal. "
            "Before finishing, run ONE ```python``` block that assigns RESULT = {...} "
            "(a JSON-safe manifest: scalars inline; save heavy arrays as .npy under "
            "EVIDENCE_DIR and store their paths) following the consulted skill's "
            "'## Report' section, so the verifier can check your actual artifacts. "
            "Then reply FINISH."
        )

    def _finalize(self, *, turns: list[Any], loaded: tuple[str, ...], terminal_raw: str | None) -> AgentTrace:
        success = any(t.execution.terminated for t in turns) or (
            bool(turns) and turns[-1].verification.passed
        )
        return AgentTrace(
            task=self._task,
            skill_query=build_query(self._task, self._observation_summary),
            loaded_skill_ids=loaded,
            turns=tuple(turns),
            success=success,
        )
