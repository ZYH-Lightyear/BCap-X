"""验证结果的归一化数据类型。

子目标级验证由独立的 :class:`robomex.agents.verifier.VerifyCodeAgent` 完成——它自己
写代码、用沙箱里已有的 ``query_vlm`` 取证判断,不再依赖任何写死的 ``Verifier`` 实现或
env 信号占位器。这里只保留它最终裁决要用到的归一化类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VerificationStatus(str, Enum):
    """归一化的验证结果。"""

    PASSED = "passed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class VerificationSignal:
    """验证器产出的单条检查结果。"""

    name: str
    status: VerificationStatus
    confidence: float | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    """针对一个子目标的聚合验证结果。"""

    status: VerificationStatus
    signals: tuple[VerificationSignal, ...] = ()
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """验证是否“有把握地”通过。"""

        return self.status == VerificationStatus.PASSED
