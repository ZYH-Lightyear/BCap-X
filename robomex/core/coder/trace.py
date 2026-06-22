"""技能 Agent 产出的整段轨迹(episode trace),也是蒸馏的原料。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from robomex.core.sandbox import BlockExecutionResult

if TYPE_CHECKING:
    from robomex.verification import VerificationResult


@dataclass(frozen=True)
class TurnRecord:
    """一个 agent 轮次:生成的代码、它的执行结果,以及裁决。"""

    turn: int
    code: str
    execution: BlockExecutionResult
    verification: VerificationResult


@dataclass(frozen=True)
class AgentTrace:
    """完整 episode:任务、咨询过哪些技能、以及每一个轮次。"""

    task: str
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
