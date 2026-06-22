from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from robomex.core.sandbox import BlockExecutionResult, SemanticActionBlock
from robomex.perception import MultimodalEvidenceBundle


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
    """针对一个动作块或整个任务的聚合验证结果。"""

    status: VerificationStatus
    signals: tuple[VerificationSignal, ...] = ()
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """验证是否“有把握地”通过。"""

        return self.status == VerificationStatus.PASSED


class Verifier(ABC):
    """验证器基接口:用于视觉、几何、机器人状态或 reward 检查。"""

    @abstractmethod
    def verify(
        self,
        *,
        block: SemanticActionBlock,
        execution: BlockExecutionResult,
        evidence: MultimodalEvidenceBundle | None = None,
    ) -> VerificationResult:
        """验证一个已执行的语义动作块。"""


class TaskSignalVerifier(Verifier):
    """依据 CapX 式 env 在 ``step`` 后返回的信号来验证一个块。

    优先级:沙箱/运行时报错 -> 失败;episode 终止(reward 1.0)或显式
    ``task_completed`` 标志 -> 通过;否则视为仍在进行中,报告为 uncertain。
    """

    def verify(
        self,
        *,
        block: SemanticActionBlock,
        execution: BlockExecutionResult,
        evidence: MultimodalEvidenceBundle | None = None,
    ) -> VerificationResult:
        if not execution.ok:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                signals=(VerificationSignal("sandbox", VerificationStatus.FAILED, message=execution.stderr),),
                summary=f"Block '{block.name}' raised during execution.",
            )

        if execution.terminated or execution.info.get("task_completed") is True:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                signals=(VerificationSignal("task_completed", VerificationStatus.PASSED, confidence=execution.reward),),
                summary=f"Block '{block.name}' reached task success.",
            )

        return VerificationResult(
            status=VerificationStatus.UNCERTAIN,
            signals=(VerificationSignal("progress", VerificationStatus.UNCERTAIN),),
            summary=f"Block '{block.name}' executed without error; task not yet complete.",
        )


class CompositeVerifier(Verifier):
    """在同一个块上跑多个验证器,并按“失败优先”合并。

    典型的 gate-3 组合:``CompositeVerifier(TaskSignalVerifier(),
    VLMJudgeVerifier(...))``——env 信号与渲染证据评判结合,不单用任一个。
    """

    def __init__(self, *verifiers: Verifier) -> None:
        self.verifiers = verifiers

    def verify(
        self,
        *,
        block: SemanticActionBlock,
        execution: BlockExecutionResult,
        evidence: MultimodalEvidenceBundle | None = None,
    ) -> VerificationResult:
        results = [
            v.verify(block=block, execution=execution, evidence=evidence)
            for v in self.verifiers
        ]
        return combine_verification_results(results)


def combine_verification_results(results: list[VerificationResult]) -> VerificationResult:
    """按保守的“失败优先”规则合并多个验证器的输出。"""

    if not results:
        return VerificationResult(
            status=VerificationStatus.NOT_APPLICABLE,
            summary="No verifier results were provided.",
        )

    signals = tuple(signal for result in results for signal in result.signals)
    if any(result.status == VerificationStatus.FAILED for result in results):
        status = VerificationStatus.FAILED
    elif any(result.status == VerificationStatus.UNCERTAIN for result in results):
        status = VerificationStatus.UNCERTAIN
    elif all(result.status == VerificationStatus.NOT_APPLICABLE for result in results):
        status = VerificationStatus.NOT_APPLICABLE
    else:
        status = VerificationStatus.PASSED

    summary = " | ".join(result.summary for result in results if result.summary)
    return VerificationResult(status=status, signals=signals, summary=summary)

