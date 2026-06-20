from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from robomex.execution import BlockExecutionResult, SemanticActionBlock
from robomex.perception import MultimodalEvidenceBundle


class VerificationStatus(str, Enum):
    """Normalized verifier outcome."""

    PASSED = "passed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class VerificationSignal:
    """A single check result produced by a verifier."""

    name: str
    status: VerificationStatus
    confidence: float | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    """Aggregated verification result for an action block or full task."""

    status: VerificationStatus
    signals: tuple[VerificationSignal, ...] = ()
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether verification confidently passed."""

        return self.status == VerificationStatus.PASSED


class Verifier(ABC):
    """Base verifier interface for visual, geometry, robot-state, or reward checks."""

    @abstractmethod
    def verify(
        self,
        *,
        block: SemanticActionBlock,
        execution: BlockExecutionResult,
        evidence: MultimodalEvidenceBundle | None = None,
    ) -> VerificationResult:
        """Verify one executed semantic action block."""


class TaskSignalVerifier(Verifier):
    """Verify a block from the signals a CapX-style env returns after ``step``.

    Precedence: a sandbox/runtime error fails; a terminated episode (reward 1.0)
    or an explicit ``task_completed`` flag passes; otherwise the block is still
    in progress and reported as uncertain.
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
    """Run several verifiers on the same block and merge with failure precedence.

    Typical gate-3 composition: ``CompositeVerifier(TaskSignalVerifier(),
    VLMJudgeVerifier(...))`` -- env signals and rendered-evidence judging
    combined, never either alone.
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
    """Combine verifier outputs with conservative failure precedence."""

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

