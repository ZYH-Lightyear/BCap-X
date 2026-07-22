"""Evaluator interface and immutable metric/evidence records."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, field_validator, model_validator

from robomex.contracts.common import (
    ContentPin,
    ContractId,
    DigestStr,
    JsonObject,
    NonEmptyStr,
    SealedContract,
    require_unique,
    require_unique_strings,
)
from robomex.evolution.candidate import PromotionStage


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility


class MetricDirection(str, Enum):  # noqa: UP042 - Python 3.10 support
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class MetricAggregation(str, Enum):  # noqa: UP042 - Python 3.10 support
    MEAN = "mean"
    MEDIAN = "median"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    SUM = "sum"
    RATE = "rate"


class MetricDefinition(SealedContract):
    schema_version: Literal["robomex.metric_definition.v1"] = (
        "robomex.metric_definition.v1"
    )
    metric_id: ContractId
    revision: int = Field(default=1, ge=1)
    description: NonEmptyStr
    unit: NonEmptyStr
    direction: MetricDirection
    aggregation: MetricAggregation
    valid_min: float | None = Field(default=None, allow_inf_nan=False)
    valid_max: float | None = Field(default=None, allow_inf_nan=False)
    qualification_threshold: float | None = Field(default=None, allow_inf_nan=False)
    safety_critical: bool = False

    @model_validator(mode="after")
    def _valid_range(self) -> MetricDefinition:
        if (
            self.valid_min is not None
            and self.valid_max is not None
            and self.valid_min > self.valid_max
        ):
            raise ValueError("valid_min must be <= valid_max.")
        if self.qualification_threshold is not None:
            if self.valid_min is not None and self.qualification_threshold < self.valid_min:
                raise ValueError("qualification_threshold is below valid_min.")
            if self.valid_max is not None and self.qualification_threshold > self.valid_max:
                raise ValueError("qualification_threshold is above valid_max.")
        if self.safety_critical and self.qualification_threshold is None:
            raise ValueError("A safety-critical metric requires a qualification threshold.")
        return self


class MetricRecord(SealedContract):
    """One evaluator output tied to exact run and candidate manifests."""

    schema_version: Literal["robomex.metric_record.v1"] = "robomex.metric_record.v1"
    record_id: ContractId
    metric_id: ContractId
    metric_definition_digest: DigestStr
    evaluator_id: ContractId
    evaluator_digest: DigestStr
    run_id: ContractId
    run_manifest_digest: DigestStr
    candidate_id: ContractId
    candidate_revision: int = Field(ge=1)
    candidate_config_digest: DigestStr
    value: float = Field(allow_inf_nan=False)
    sample_count: int = Field(default=1, ge=1)
    evidence_refs: tuple[NonEmptyStr, ...] = Field(min_length=1)
    measured_at: datetime = Field(default_factory=_utc_now)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("measured_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("measured_at must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility

    @field_validator("evidence_refs")
    @classmethod
    def _unique_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        require_unique_strings(value, label="metric evidence references")
        return value


class GateStatus(str, Enum):  # noqa: UP042 - Python 3.10 support
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class EvaluationGateRecord(SealedContract):
    schema_version: Literal["robomex.evaluation_gate.v1"] = (
        "robomex.evaluation_gate.v1"
    )
    gate_id: ContractId
    status: GateStatus
    hard_gate: bool = True
    reason: NonEmptyStr
    metric_record_ids: tuple[ContractId, ...] = ()
    evidence_refs: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_gate_references(self) -> EvaluationGateRecord:
        require_unique_strings(self.metric_record_ids, label="gate metric record IDs")
        require_unique_strings(self.evidence_refs, label="gate evidence references")
        return self


class EvaluationRequest(SealedContract):
    schema_version: Literal["robomex.evaluation_request.v1"] = (
        "robomex.evaluation_request.v1"
    )
    request_id: ContractId
    candidate_id: ContractId
    candidate_revision: int = Field(ge=1)
    candidate_config_digest: DigestStr
    run_manifest_pins: tuple[ContentPin, ...] = Field(min_length=1)
    metric_definition_pins: tuple[ContentPin, ...] = Field(min_length=1)
    evaluator_id: ContractId
    evaluator_config_digest: DigestStr
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_request_pins(self) -> EvaluationRequest:
        require_unique(
            self.run_manifest_pins, key="component_id", label="run-manifest pins"
        )
        require_unique(
            self.metric_definition_pins,
            key="component_id",
            label="metric-definition pins",
        )
        return self


class EvaluationReport(SealedContract):
    """Evaluator result and recommendation; never an automatic promotion."""

    schema_version: Literal["robomex.evaluation_report.v1"] = (
        "robomex.evaluation_report.v1"
    )
    report_id: ContractId
    request_id: ContractId
    request_digest: DigestStr
    evaluator_id: ContractId
    evaluator_digest: DigestStr
    candidate_id: ContractId
    candidate_revision: int = Field(ge=1)
    candidate_config_digest: DigestStr
    metrics: tuple[MetricRecord, ...] = Field(min_length=1)
    gates: tuple[EvaluationGateRecord, ...] = Field(min_length=1)
    recommended_stage: PromotionStage | None = None
    summary: NonEmptyStr
    completed_at: datetime = Field(default_factory=_utc_now)

    @field_validator("completed_at")
    @classmethod
    def _completed_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("completed_at must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility

    @model_validator(mode="after")
    def _consistent_records(self) -> EvaluationReport:
        require_unique(self.metrics, key="record_id", label="metric record IDs")
        require_unique(self.gates, key="gate_id", label="evaluation gate IDs")
        metric_ids = {metric.record_id for metric in self.metrics}
        for metric in self.metrics:
            if (
                metric.candidate_id != self.candidate_id
                or metric.candidate_revision != self.candidate_revision
                or metric.candidate_config_digest != self.candidate_config_digest
            ):
                raise ValueError("Metric record candidate pin does not match report.")
            if metric.evaluator_id != self.evaluator_id:
                raise ValueError("Metric evaluator_id does not match report.")
            if metric.evaluator_digest != self.evaluator_digest:
                raise ValueError("Metric evaluator_digest does not match report.")
        for gate in self.gates:
            unknown = set(gate.metric_record_ids) - metric_ids
            if unknown:
                raise ValueError(
                    f"Gate {gate.gate_id!r} references unknown metric records: "
                    + ", ".join(sorted(unknown))
                    + "."
                )
        if self.recommended_stage in {
            PromotionStage.BASELINE,
            PromotionStage.DRAFT,
            PromotionStage.APPROVED,
        }:
            raise ValueError(
                "An evaluator may recommend an evaluation/qualification/rejection "
                "stage, but never baseline, draft, or human approval."
            )
        if any(
            gate.hard_gate and gate.status is not GateStatus.PASSED
            for gate in self.gates
        ) and self.recommended_stage not in {None, PromotionStage.REJECTED}:
            raise ValueError(
                "A failed/inconclusive hard gate cannot recommend advancement."
            )
        return self


@runtime_checkable
class Evaluator(Protocol):
    """Pure evaluation adapter; implementations must not mutate or promote."""

    @property
    def evaluator_id(self) -> str:
        """Stable logical evaluator identity."""

    @property
    def content_digest(self) -> str:
        """Digest of evaluator code and immutable configuration."""

    def evaluate(self, request: EvaluationRequest) -> EvaluationReport:
        """Evaluate one frozen request and return an immutable report."""


__all__ = [
    "EvaluationGateRecord",
    "EvaluationReport",
    "EvaluationRequest",
    "Evaluator",
    "GateStatus",
    "MetricAggregation",
    "MetricDefinition",
    "MetricDirection",
    "MetricRecord",
]
