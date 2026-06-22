"""验证工具箱(不是 agent)。

子目标级验证由独立的 Verify Coding Agent
(:class:`robomex.agents.verifier.VerifyCodeAgent`)完成:它读 :class:`VerifierContext`
里只含事实的上下文(sub-goal、用过的技能、脱敏 op-trace、作者写的 verify.md rubric),
自己写代码、用沙箱里已有的 ``query_vlm`` 在证据上判断,输出裸 JSON 裁决。本模块只提供
它要用到的数据面:归一化结果类型 + 上下文构造工具。
"""

from robomex.verification.context import (
    VerifierContext,
    VerifyResource,
    build_op_trace,
    collect_verify_resources,
    sanitize_code,
)
from robomex.verification.verifier import (
    VerificationResult,
    VerificationSignal,
    VerificationStatus,
)

__all__ = [
    "VerificationResult",
    "VerificationSignal",
    "VerificationStatus",
    "VerifierContext",
    "VerifyResource",
    "build_op_trace",
    "collect_verify_resources",
    "sanitize_code",
]
