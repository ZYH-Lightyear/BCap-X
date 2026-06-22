"""验证工具箱(不是 agent)。

这些是 Verify Coding Agent(:class:`robomex.agents.verifier.VerifyCodeAgent`)
以 ``verify-as-code`` 方式组合的积木:裁决类型、只含事实的 :class:`VerifierContext`、
一个 ``vlm_judge`` 原语,以及现成的 :class:`Verifier` 实现。当前只有
:class:`TaskSignalVerifier` 在执行器主循环里活跃(它包了 env 的成功信号);
VLM 评判 / 独立验证器这些部分在闭环成熟前都处于休眠。
"""

from robomex.verification.context import (
    VerifierContext,
    VerifyResource,
    build_op_trace,
    collect_verify_resources,
    sanitize_code,
)
from robomex.verification.primitives import vlm_judge
from robomex.verification.verifier import (
    CompositeVerifier,
    TaskSignalVerifier,
    VerificationResult,
    VerificationSignal,
    VerificationStatus,
    Verifier,
    combine_verification_results,
)
from robomex.verification.vlm_judge import VLMJudgeVerifier

__all__ = [
    "CompositeVerifier",
    "TaskSignalVerifier",
    "VerificationResult",
    "VerificationSignal",
    "VerificationStatus",
    "Verifier",
    "VerifierContext",
    "VerifyResource",
    "VLMJudgeVerifier",
    "build_op_trace",
    "collect_verify_resources",
    "combine_verification_results",
    "sanitize_code",
    "vlm_judge",
]
