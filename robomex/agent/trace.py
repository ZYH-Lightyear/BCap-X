"""Episode trace produced by the Skill Agent, the substrate for distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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

    @property
    def claim(self) -> dict[str, Any]:
        """The executor's structured RESULT manifest (last non-null), as a dict.

        This is the first-class handoff to the verifier: no out-of-band re-read of
        the sandbox is needed -- the manifest each turn produced is already carried
        on ``execution.result`` (mirrored from CapX ``info["result"]``).
        """
        for turn in reversed(self.turns):
            result = turn.execution.result
            if isinstance(result, Mapping):
                return dict(result)
            if result is not None:
                return {"value": result}
        return {}

    @property
    def executor_stdout(self) -> str:
        """Concatenated per-turn stdout (the executor's REAL printed output).

        Always available even when no manifest was recorded, so the verifier is
        never blind to what the executor actually computed/printed.
        """
        chunks = [
            f"[turn {t.turn}] {t.execution.stdout.strip()}"
            for t in self.turns
            if t.execution.stdout.strip()
        ]
        return "\n".join(chunks)
