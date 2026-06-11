"""CodeAsPolicy Skill Agent: the actor that consults skills and writes code.

Loop (mirrors the v1 design system loop): observe -> retrieve & load skills ->
inject compact guidance -> LLM generates code grounded in the observation ->
execute via the CapX env -> verify -> multi-turn feedback -> finish -> trace.

The skill is consulted, never executed verbatim; the policy emits its own code.
"""

from __future__ import annotations

from typing import Protocol

from robomex.agent.policy import FINISH, CodePolicy
from robomex.agent.router import SkillRouter, build_query
from robomex.agent.trace import AgentTrace, TurnRecord
from robomex.execution import BlockExecutionResult, SemanticActionBlock
from robomex.library import SkillLibrary
from robomex.verification import TaskSignalVerifier, Verifier

_SYSTEM_PROMPT = (
    "You are a robot Code-as-Policy agent. Each turn, write one block of executable "
    "Python that advances the task, grounding every decision in the current observation. "
    "Skill guidance below is advisory: adapt it, do not copy it blindly. "
    "Reply with a ```python``` code block, or the word FINISH when the task is complete."
)


class BlockExecutor(Protocol):
    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult: ...


class CodeAsPolicyAgent:
    def __init__(
        self,
        executor: BlockExecutor,
        policy: CodePolicy,
        library: SkillLibrary,
        verifier: Verifier | None = None,
        router: SkillRouter | None = None,
        max_turns: int = 6,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        self.executor = executor
        self.policy = policy
        self.library = library
        self.verifier = verifier or TaskSignalVerifier()
        self.router = router or SkillRouter(library)
        self.max_turns = max_turns
        self.system_prompt = system_prompt

    def run(self, task: str, observation_summary: str = "") -> AgentTrace:
        query = build_query(task, observation_summary)
        records = self.router.route(task, observation_summary)
        guidance = self.router.guidance_for(records, task)
        loaded_ids = tuple(r.skill_id for r in records)

        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._initial_user_message(task, observation_summary, guidance)},
        ]

        turns: list[TurnRecord] = []
        success = False
        for turn in range(self.max_turns):
            code = self.policy.act(prompt)
            if code == FINISH:
                break

            block = SemanticActionBlock(name=f"turn_{turn}", intent="agent-generated code", code=code)
            execution = self.executor.run_block(block)
            verdict = self.verifier.verify(block=block, execution=execution)
            turns.append(TurnRecord(turn, code, execution, verdict))

            prompt.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            prompt.append({"role": "user", "content": self._feedback_message(execution)})

            if execution.terminated:
                success = True
                break

        success = success or (bool(turns) and turns[-1].verification.passed)

        return AgentTrace(
            task=task,
            skill_query=query,
            loaded_skill_ids=loaded_ids,
            turns=tuple(turns),
            success=success,
        )

    @staticmethod
    def _initial_user_message(task: str, observation_summary: str, guidance: str) -> str:
        parts = [f"Task: {task}"]
        if observation_summary:
            parts.append(f"Observation: {observation_summary}")
        if guidance:
            parts.append(guidance)
        return "\n\n".join(parts)

    @staticmethod
    def _feedback_message(execution: BlockExecutionResult) -> str:
        return (
            f"stdout:\n{execution.stdout}\n\nstderr:\n{execution.stderr}\n\n"
            "If the task is complete reply FINISH, otherwise reply with the next ```python``` block."
        )
