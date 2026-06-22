from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionBlockStatus(str, Enum):
    """语义动作块的生命周期状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class SemanticActionBlock:
    """一个小的可执行单元:带自然语言意图 + Python 代码。"""

    name: str
    intent: str
    code: str
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    expected_artifacts: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionTraceEvent:
    """来自代码执行或 API 调用的一个归一化事件。"""

    event_type: str
    message: str
    block_name: str | None = None
    line_no: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockExecutionResult:
    """执行一个语义动作块后返回的结果。"""

    block: SemanticActionBlock
    ok: bool
    status: ActionBlockStatus
    stdout: str = ""
    stderr: str = ""
    reward: float | None = None
    terminated: bool | None = None
    truncated: bool | None = None
    observation: dict[str, Any] | None = None
    info: dict[str, Any] = field(default_factory=dict)
    trace_events: tuple[ExecutionTraceEvent, ...] = ()

    @classmethod
    def skipped(cls, block: SemanticActionBlock, reason: str) -> BlockExecutionResult:
        """构造一个“跳过”结果,同时保留 block 身份。"""

        return cls(
            block=block,
            ok=False,
            status=ActionBlockStatus.SKIPPED,
            stderr=reason,
        )

