"""沙箱:动作空间 + 运行它的后端。

:class:`SemanticActionBlock` 是一个可执行单元(自然语言意图 + Python 代码);
:class:`BlockExecutionResult` 是运行它得到的结果。:class:`CapXExecutorAdapter`
是把 CapX 代码执行 env 步进起来的具体后端。Coding Agent 内核
(``robomex.core.coder``)只对这个接口说话,从不直接接触 CapX。
"""

from robomex.core.sandbox.action_block import (
    ActionBlockStatus,
    BlockExecutionResult,
    ExecutionTraceEvent,
    SemanticActionBlock,
)
from robomex.core.sandbox.capx import CapXExecutorAdapter

__all__ = [
    "ActionBlockStatus",
    "BlockExecutionResult",
    "CapXExecutorAdapter",
    "ExecutionTraceEvent",
    "SemanticActionBlock",
]
