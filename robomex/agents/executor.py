"""CodeAsPolicy 技能 Agent:咨询技能、写代码的执行器。

它是共享 :class:`~robomex.core.coder.CodingAgent` 的轻量子类。执行器的特化在于:
上下文是任务(+观测);通过渐进披露感知*整库*(开头一份简短的 ``<available_skills>``
清单 + 用 ``USE SKILL`` 拉取正文);每个 python 轮都会被验证 + 打包证据;在 ``FINISH``
或 env 成功信号时终止。

技能只是被*咨询*,绝不照搬执行;代码由策略自己生成。
"""

from __future__ import annotations

from typing import Any

from robomex.core.coder import CodingAgent, SkillEntry
from robomex.core.coder.policy import CompletionPolicy
from robomex.core.coder.trace import AgentTrace, TurnRecord
from robomex.core.logging import get_logger
from robomex.core.sandbox import BlockExecutionResult
from robomex.perception import EvidenceCollector
from robomex.skills import SkillLibrary
from robomex.verification.verifier import TaskSignalVerifier, Verifier

_log = get_logger("executor")

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
        collector: EvidenceCollector | None = None,
        max_turns: int = 6,
        system_prompt: str = _SYSTEM_PROMPT,
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
        self._task = ""
        self._observation_summary = ""
        self._primary_skill_id: str | None = None

    def run(
        self,
        task: str,
        observation_summary: str = "",
        primary_skill_id: str | None = None,
    ) -> AgentTrace:
        self._task = task
        self._observation_summary = observation_summary
        self._primary_skill_id = primary_skill_id
        return super().run()

    # ---- 钩子 --------------------------------------------------------------

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
        if self._primary_skill_id:
            parts.append(
                f"This sub-goal corresponds to the high-level skill "
                f"`{self._primary_skill_id}`. Start with `USE SKILL: {self._primary_skill_id}` "
                "to read how it orchestrates the work, then consult and freely combine the "
                "observation/action leaf skills it points to -- decide the order and the "
                "code yourself from each skill's guidance; there is no fixed pipeline."
            )
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
        _log.info(
            "turn %d: exec=%s reward=%s terminated=%s | verdict=%s",
            turn_idx, execution.status.value, execution.reward, execution.terminated, verdict.status.value,
        )
        stderr = (execution.stderr or "").strip()
        if not execution.ok and stderr:
            _log.info("turn %d: 报错 -> %s", turn_idx, stderr.splitlines()[-1][:300])
        turns.append(TurnRecord(turn_idx, code, execution, verdict))

    def _should_stop_after_python(self, execution: BlockExecutionResult) -> bool:
        return bool(execution.terminated)

    def _finalize(self, *, turns: list[Any], loaded: tuple[str, ...], terminal_raw: str | None) -> AgentTrace:
        success = any(t.execution.terminated for t in turns) or (
            bool(turns) and turns[-1].verification.passed
        )
        return AgentTrace(
            task=self._task,
            loaded_skill_ids=loaded,
            turns=tuple(turns),
            success=success,
        )
