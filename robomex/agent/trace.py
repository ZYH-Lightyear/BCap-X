"""Episode trace produced by the Skill Agent, the substrate for distillation."""

from __future__ import annotations

from dataclasses import dataclass, field

from robomex.execution import BlockExecutionResult
from robomex.verification import VerificationResult


@dataclass(frozen=True)
class TurnRecord:
    """One agent turn: generated code, its execution, and the verdict."""

    turn: int
    code: str
    execution: BlockExecutionResult
    verification: VerificationResult


@dataclass(frozen=True)
class AgentTrace:
    """Full episode: the task, which skills were consulted, and every turn."""

    task: str
    skill_query: str
    loaded_skill_ids: tuple[str, ...]
    turns: tuple[TurnRecord, ...]
    success: bool
    metadata: dict = field(default_factory=dict)

    @property
    def successful_code(self) -> tuple[str, ...]:
        return tuple(t.code for t in self.turns if t.execution.ok)

    @property
    def last_error(self) -> str:
        for turn in reversed(self.turns):
            if turn.execution.stderr:
                return turn.execution.stderr
        return ""
