"""Versioned, sealed action contracts for the RoboMEx v2 runtime.

The schemas in this module describe exact executable effects.  A motion plan
contains a joint path, never a Cartesian request that a runner may reinterpret
through another IK solve.  Gripper and wait effects are separate specs so
their execution cannot be hidden inside an opaque motion program.

All action specs carry a canonical semantic digest.  The digest excludes only
its own field and includes every execution-relevant guard and policy value.
Changing a waypoint, joint order, configuration digest, or subsample therefore
changes the action identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

from robomex.data.embodied_state import AttachmentStatus

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DigestStr = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]

_DIGEST_PREFIX = b"robomex-digest-v1\x00"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _new_id() -> str:
    return uuid.uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility


def canonical_payload_digest(schema_version: str, payload: Mapping[str, object]) -> str:
    """Hash one JSON semantic payload using the repository's v1 digest domain."""

    encoded = json.dumps(
        {"schema_version": schema_version, "payload": payload},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(_DIGEST_PREFIX + encoded).hexdigest()


def canonical_model_digest(model: BaseModel) -> str:
    """Return a model's canonical digest, excluding only ``content_digest``."""

    schema_version = str(model.schema_version)
    payload = cast(
        dict[str, object],
        model.model_dump(mode="json", exclude={"content_digest"}),
    )
    return canonical_payload_digest(schema_version, payload)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class _UtcModel(_StrictModel):
    @field_validator("captured_at", "started_at", "finished_at", check_fields=False)
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10 compatibility


class WorldKind(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    AUTHORITATIVE = "authoritative"
    SHADOW = "shadow"


class ControllerState(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    READY = "ready"
    QUIESCENT = "quiescent"
    EXECUTING = "executing"
    INDETERMINATE = "indeterminate"


class FeasibilityStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class AdmissionSnapshot(_UtcModel):
    """Runtime-observed guard state used at admission and execution TOCTOU gates."""

    schema_version: Literal["robomex.admission_snapshot.v1"] = (
        "robomex.admission_snapshot.v1"
    )
    world_id: NonEmptyStr
    world_kind: WorldKind = WorldKind.AUTHORITATIVE
    resource_id: NonEmptyStr
    robot_revision: int = Field(ge=0)
    scene_revision: int = Field(ge=0)
    attachment_revision: int = Field(ge=0)
    config_revision: int = Field(ge=0)
    joint_names: tuple[NonEmptyStr, ...]
    joint_positions_rad: tuple[float, ...]
    config_digest: DigestStr
    collision_world_digest: DigestStr
    attachment_status: AttachmentStatus = AttachmentStatus.UNKNOWN
    controller_state: ControllerState = ControllerState.READY
    captured_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _validate_joint_vector(self) -> AdmissionSnapshot:
        if not self.joint_names:
            raise ValueError("joint_names must not be empty")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique and ordered")
        if len(self.joint_names) != len(self.joint_positions_rad):
            raise ValueError("joint_names and joint_positions_rad must have equal length")
        if not all(math.isfinite(value) for value in self.joint_positions_rad):
            raise ValueError("joint_positions_rad must contain only finite values")
        return self


class FeasibilityCertificate(_UtcModel):
    """Runtime-validated evidence bound to one exact spec and snapshot."""

    schema_version: Literal["robomex.feasibility_certificate.v1"] = (
        "robomex.feasibility_certificate.v1"
    )
    certificate_id: NonEmptyStr = Field(default_factory=_new_id)
    action_spec_digest: DigestStr
    admission_snapshot_digest: DigestStr
    checker_id: NonEmptyStr
    checks: dict[NonEmptyStr, FeasibilityStatus]
    overall_status: FeasibilityStatus
    evidence_refs: tuple[NonEmptyStr, ...] = ()
    captured_at: datetime = Field(default_factory=_utc_now)
    content_digest: str = ""

    @model_validator(mode="after")
    def _validate_certificate(self) -> FeasibilityCertificate:
        if not self.checks:
            raise ValueError("a feasibility certificate requires named checks")
        values = set(self.checks.values())
        expected = (
            FeasibilityStatus.FAIL
            if FeasibilityStatus.FAIL in values
            else (
                FeasibilityStatus.UNKNOWN
                if FeasibilityStatus.UNKNOWN in values
                else FeasibilityStatus.PASS
            )
        )
        if self.overall_status is not expected:
            raise ValueError("overall_status does not match the closed check reduction")
        digest = canonical_model_digest(self)
        if self.content_digest:
            if self.content_digest != digest:
                raise ValueError("content_digest does not match feasibility certificate")
        else:
            object.__setattr__(self, "content_digest", digest)
        return self


class ExecutionPolicy(_StrictModel):
    """Exact policy interpreted by the trusted backend adapter."""

    mode: Literal["blocking_waypoint"] = "blocking_waypoint"
    subsample: int = Field(default=1, ge=1)
    timeout_s: float = Field(default=30.0, gt=0.0, le=600.0, allow_inf_nan=False)


class JointPath(_StrictModel):
    representation: Literal["joint_path"] = "joint_path"
    joint_names: tuple[NonEmptyStr, ...]
    positions_rad: tuple[tuple[float, ...], ...]
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)

    @model_validator(mode="after")
    def _validate_shape(self) -> JointPath:
        if not self.joint_names:
            raise ValueError("joint_names must not be empty")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique and ordered")
        if not self.positions_rad:
            raise ValueError("positions_rad must contain at least one waypoint")
        width = len(self.joint_names)
        for index, waypoint in enumerate(self.positions_rad):
            if len(waypoint) != width:
                raise ValueError(f"waypoint {index} does not match joint_names width")
            if not all(math.isfinite(value) for value in waypoint):
                raise ValueError(f"waypoint {index} contains a non-finite value")
        return self


class _SealedActionSpec(_StrictModel):
    expected_snapshot: AdmissionSnapshot
    max_start_deviation_rad: float = Field(default=0.02, ge=0, allow_inf_nan=False)
    possibly_affected_revisions: tuple[NonEmptyStr, ...]
    content_digest: str = ""

    @model_validator(mode="after")
    def _seal_or_validate_digest(self) -> _SealedActionSpec:
        expected = canonical_model_digest(self)
        if self.content_digest:
            if _DIGEST_RE.fullmatch(self.content_digest) is None:
                raise ValueError("content_digest must be sha256:<64 lowercase hex>")
            if self.content_digest != expected:
                raise ValueError("content_digest does not match canonical action payload")
        else:
            object.__setattr__(self, "content_digest", expected)
        if not self.possibly_affected_revisions:
            raise ValueError("possibly_affected_revisions must not be empty")
        return self

    @property
    def world_id(self) -> str:
        return self.expected_snapshot.world_id

    @property
    def resource_id(self) -> str:
        return self.expected_snapshot.resource_id


class MotionPlan(_SealedActionSpec):
    """One continuous, exact arm joint-path segment."""

    schema_version: Literal["robomex.motion_plan.v2"] = "robomex.motion_plan.v2"
    spec_type: Literal["motion_plan"] = "motion_plan"
    plan_id: NonEmptyStr
    plan_kind: NonEmptyStr
    tcp_frame_id: NonEmptyStr
    planner_backend: NonEmptyStr
    robot_model_digest: DigestStr
    motion: JointPath

    @model_validator(mode="after")
    def _match_snapshot_joint_order(self) -> MotionPlan:
        if self.motion.joint_names != self.expected_snapshot.joint_names:
            raise ValueError("motion joint order must exactly match admission snapshot")
        return self

    @property
    def spec_id(self) -> str:
        return self.plan_id


class GripperMode(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    OPEN = "open"
    CLOSE = "close"


class GripperCommand(_SealedActionSpec):
    schema_version: Literal["robomex.gripper_command.v1"] = "robomex.gripper_command.v1"
    spec_type: Literal["gripper_command"] = "gripper_command"
    command_id: NonEmptyStr
    mode: GripperMode
    target_width_m: float = Field(ge=0, allow_inf_nan=False)
    max_effort_n: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    timeout_s: float = Field(default=2.0, gt=0, allow_inf_nan=False)

    @property
    def spec_id(self) -> str:
        return self.command_id


class WaitSpec(_SealedActionSpec):
    schema_version: Literal["robomex.wait_spec.v1"] = "robomex.wait_spec.v1"
    spec_type: Literal["wait"] = "wait"
    wait_id: NonEmptyStr
    duration_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    control_steps: int | None = Field(default=None, gt=0)
    hold_command: Literal["hold_current"] = "hold_current"
    timeout_s: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_wait_bound(self) -> WaitSpec:
        if (self.duration_s is None) == (self.control_steps is None):
            raise ValueError("exactly one of duration_s or control_steps is required")
        if self.duration_s is not None and self.timeout_s < self.duration_s:
            raise ValueError("timeout_s must cover duration_s")
        return self

    @property
    def spec_id(self) -> str:
        return self.wait_id


ActionSpec: TypeAlias = Annotated[  # noqa: UP040 - Python 3.10 compatibility
    MotionPlan | GripperCommand | WaitSpec,
    Field(discriminator="spec_type"),
]
_ACTION_SPEC_ADAPTER = TypeAdapter(ActionSpec)


def validate_action_spec(value: ActionSpec | Mapping[str, object] | str | bytes) -> ActionSpec:
    """Fully revalidate a spec, including canonical digest and nested models."""

    if isinstance(value, (str, bytes)):
        return _ACTION_SPEC_ADAPTER.validate_json(value)
    if isinstance(value, BaseModel):
        # Pydantic normally trusts existing instances.  Round-tripping through
        # a mapping detects model_copy() substitution with a stale digest.
        value = value.model_dump(mode="python")
    return _ACTION_SPEC_ADAPTER.validate_python(value)


class BackendMotionInterface(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    EXACT_JOINT_PATH = "exact_joint_path"
    POSE_REINTERPRETATION = "pose_reinterpretation"


class MonitorTelemetryHook(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Physical synchronization points a backend can actually observe."""

    PHASE = "phase"
    WAYPOINT = "waypoint"
    CONTROL = "control"


class MonitorTelemetryCapabilities(_StrictModel):
    """Exact monitor telemetry contract for one authoritative resource.

    ``always_available_signals`` is deliberately an exact declaration rather
    than a best-effort list.  A backend must reject or fail a sample when one
    of these fields cannot be produced; it must never silently reuse a stale
    observation.  The world/resource binding prevents a camera or attachment
    stream configured for one robot from being advertised for another.
    """

    schema_version: Literal["robomex.monitor_telemetry_capabilities.v1"] = (
        "robomex.monitor_telemetry_capabilities.v1"
    )
    world_id: NonEmptyStr
    resource_id: NonEmptyStr
    always_available_signals: tuple[NonEmptyStr, ...] = Field(min_length=1)
    supported_hooks: tuple[MonitorTelemetryHook, ...] = Field(min_length=1)
    cooperative_stop_guaranteed: bool = False

    @model_validator(mode="after")
    def _validate_exact_sets(self) -> MonitorTelemetryCapabilities:
        if len(set(self.always_available_signals)) != len(
            self.always_available_signals
        ):
            raise ValueError("always_available_signals must be unique")
        if len(set(self.supported_hooks)) != len(self.supported_hooks):
            raise ValueError("supported_hooks must be unique")
        if any(name.startswith("_") for name in self.always_available_signals):
            raise ValueError("monitor telemetry signal names must be public")
        return self


class BackendDescriptor(_StrictModel):
    schema_version: Literal["robomex.action_backend.v1"] = "robomex.action_backend.v1"
    backend_id: NonEmptyStr
    motion_interface: BackendMotionInterface = BackendMotionInterface.EXACT_JOINT_PATH
    # The runtime watchdog executes the backend call in a worker.  An
    # out-of-band stop after timeout is only legal when the adapter explicitly
    # proves that its controller surface supports concurrent cross-thread use.
    watchdog_stop_thread_safe: bool = False


class BackendCallResult(_StrictModel):
    """Narrow return value of a trusted runtime adapter, never a receipt."""

    converged: bool | None = None
    interrupted: bool = False
    timed_out: bool = False
    telemetry: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _closed_terminal_flags(self) -> BackendCallResult:
        if self.interrupted and self.timed_out:
            raise ValueError("a backend result cannot be interrupted and timed_out")
        if (self.interrupted or self.timed_out) and self.converged is True:
            raise ValueError("an interrupted/timed-out primitive cannot be converged")
        return self


class ActionSpecType(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    MOTION_PLAN = "motion_plan"
    GRIPPER_COMMAND = "gripper_command"
    WAIT = "wait"


class PrimitiveStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    RETURNED = "returned"
    RAISED = "raised"


class ExecutionStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    REJECTED = "rejected"
    COMPLETED = "completed"
    PARTIAL = "partial"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"
    INDETERMINATE_AFTER_CRASH = "indeterminate_after_crash"
    INDETERMINATE_AFTER_TIMEOUT = "indeterminate_after_timeout"


class ActionAttempt(_UtcModel):
    """Write-ahead record flushed before the first authoritative primitive."""

    schema_version: Literal["robomex.action_attempt.v1"] = "robomex.action_attempt.v1"
    record_kind: Literal["action_attempt"] = "action_attempt"
    record_id: NonEmptyStr = Field(default_factory=_new_id)
    action_id: NonEmptyStr
    admission_id: NonEmptyStr
    action_spec: ActionSpec
    spec_digest: DigestStr
    admission_snapshot: AdmissionSnapshot
    admission_snapshot_digest: DigestStr
    execution_snapshot: AdmissionSnapshot
    execution_snapshot_digest: DigestStr
    feasibility_certificate: FeasibilityCertificate | None = None
    monitor_digest: DigestStr | None = None
    continuation_id: NonEmptyStr | None = None
    scheduler_reservation_id: NonEmptyStr | None = None
    attempt_status: Literal["admitted"] = "admitted"
    started_at: datetime = Field(default_factory=_utc_now)
    record_authority: Literal["runtime"] = "runtime"

    @model_validator(mode="after")
    def _bind_spec(self) -> ActionAttempt:
        if self.action_spec.content_digest != self.spec_digest:
            raise ValueError("spec_digest must match action_spec")
        if self.action_spec.expected_snapshot.world_kind is not WorldKind.AUTHORITATIVE:
            raise ValueError("ActionAttempt is reserved for authoritative effects")
        if canonical_model_digest(self.admission_snapshot) != self.admission_snapshot_digest:
            raise ValueError("admission_snapshot_digest does not match admission_snapshot")
        if canonical_model_digest(self.execution_snapshot) != self.execution_snapshot_digest:
            raise ValueError("execution_snapshot_digest does not match execution_snapshot")
        if (
            self.execution_snapshot.world_id != self.admission_snapshot.world_id
            or self.execution_snapshot.resource_id != self.admission_snapshot.resource_id
        ):
            raise ValueError("execution snapshot changed the admitted world/resource")
        if self.feasibility_certificate is not None:
            if self.feasibility_certificate.action_spec_digest != self.spec_digest:
                raise ValueError("feasibility certificate does not bind this spec")
            if (
                self.feasibility_certificate.admission_snapshot_digest
                != self.admission_snapshot_digest
            ):
                raise ValueError("feasibility certificate does not bind this snapshot")
            if self.feasibility_certificate.overall_status is not FeasibilityStatus.PASS:
                raise ValueError("an attempted action requires a passing certificate")
        return self


class PrimitiveReceipt(_UtcModel):
    schema_version: Literal["robomex.primitive_receipt.v1"] = (
        "robomex.primitive_receipt.v1"
    )
    record_kind: Literal["primitive_receipt"] = "primitive_receipt"
    record_id: NonEmptyStr = Field(default_factory=_new_id)
    action_id: NonEmptyStr
    primitive_index: int = Field(ge=0)
    primitive: Literal["execute_joint_path", "set_gripper", "wait"]
    exact_args_digest: DigestStr
    status: PrimitiveStatus
    started_at: datetime
    finished_at: datetime
    converged: bool | None = None
    telemetry: dict[str, JsonValue] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    record_authority: Literal["runtime"] = "runtime"

    @model_validator(mode="after")
    def _validate_result(self) -> PrimitiveReceipt:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at precedes started_at")
        if self.status is PrimitiveStatus.RAISED and not self.error_type:
            raise ValueError("raised primitive requires error_type")
        if self.status is PrimitiveStatus.RETURNED and self.error_type is not None:
            raise ValueError("returned primitive cannot carry error_type")
        return self


class ExecutionReceipt(_UtcModel):
    """Runtime-owned authoritative evidence for one physical attempt."""

    schema_version: Literal["robomex.execution_receipt.v2"] = (
        "robomex.execution_receipt.v2"
    )
    record_kind: Literal["execution_receipt"] = "execution_receipt"
    record_id: NonEmptyStr = Field(default_factory=_new_id)
    action_id: NonEmptyStr
    effect_id: NonEmptyStr = Field(default_factory=_new_id)
    spec_type: ActionSpecType
    spec_id: NonEmptyStr
    spec_digest: DigestStr
    world_id: NonEmptyStr
    resource_id: NonEmptyStr
    execution_context: Literal["authoritative"] = "authoritative"
    receipt_authority: Literal["runtime"] = "runtime"
    runtime_status: ExecutionStatus
    primitive_receipts: tuple[PrimitiveReceipt, ...] = ()
    possibly_affected_revisions: tuple[NonEmptyStr, ...]
    started_at: datetime
    finished_at: datetime
    abort_reason: str | None = None
    terminal_telemetry: dict[str, JsonValue] = Field(default_factory=dict)
    feasibility_certificate_digest: DigestStr | None = None
    monitor_digest: DigestStr | None = None
    triggering_finding_id: NonEmptyStr | None = None
    frame_refs: tuple[NonEmptyStr, ...] = ()
    video_ref: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _validate_receipt(self) -> ExecutionReceipt:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at precedes started_at")
        if any(item.action_id != self.action_id for item in self.primitive_receipts):
            raise ValueError("all primitive receipts must match action_id")
        if self.runtime_status is ExecutionStatus.COMPLETED:
            if not self.primitive_receipts:
                raise ValueError("completed execution requires a primitive receipt")
            if any(
                item.status is not PrimitiveStatus.RETURNED
                for item in self.primitive_receipts
            ):
                raise ValueError("completed execution requires returned primitives")
            if self.spec_type in {
                ActionSpecType.MOTION_PLAN,
                ActionSpecType.GRIPPER_COMMAND,
            } and any(item.converged is not True for item in self.primitive_receipts):
                raise ValueError(
                    "completed motion/gripper execution requires explicit convergence"
                )
            if self.spec_type is ActionSpecType.WAIT and any(
                item.converged is False for item in self.primitive_receipts
            ):
                raise ValueError("completed wait execution cannot report failed convergence")
        return self


class RecoveryAcknowledgement(_UtcModel):
    """Durable proof that an indeterminate resource was made quiescent."""

    schema_version: Literal["robomex.recovery_acknowledgement.v1"] = (
        "robomex.recovery_acknowledgement.v1"
    )
    record_kind: Literal["recovery_acknowledgement"] = "recovery_acknowledgement"
    record_id: NonEmptyStr = Field(default_factory=_new_id)
    recovery_id: NonEmptyStr = Field(default_factory=_new_id)
    world_id: NonEmptyStr
    resource_id: NonEmptyStr
    prior_action_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    recovery_snapshot: AdmissionSnapshot
    recovery_snapshot_digest: DigestStr
    evidence_refs: tuple[NonEmptyStr, ...] = Field(min_length=1)
    reason: NonEmptyStr
    finished_at: datetime = Field(default_factory=_utc_now)
    record_authority: Literal["runtime"] = "runtime"

    @model_validator(mode="after")
    def _validate_recovery(self) -> RecoveryAcknowledgement:
        if len(set(self.prior_action_ids)) != len(self.prior_action_ids):
            raise ValueError("prior_action_ids must be unique")
        if self.recovery_snapshot.world_id != self.world_id:
            raise ValueError("recovery snapshot world_id mismatch")
        if self.recovery_snapshot.resource_id != self.resource_id:
            raise ValueError("recovery snapshot resource_id mismatch")
        if self.recovery_snapshot.controller_state is not ControllerState.QUIESCENT:
            raise ValueError("recovery acknowledgement requires a quiescent controller")
        if canonical_model_digest(self.recovery_snapshot) != self.recovery_snapshot_digest:
            raise ValueError("recovery_snapshot_digest does not match snapshot")
        return self


class ShadowRolloutStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class ShadowRolloutReceipt(_UtcModel):
    """Non-authoritative evidence from an isolated simulated world clone."""

    schema_version: Literal["robomex.shadow_rollout_receipt.v1"] = (
        "robomex.shadow_rollout_receipt.v1"
    )
    rollout_id: NonEmptyStr = Field(default_factory=_new_id)
    candidate_id: NonEmptyStr
    shadow_world_id: NonEmptyStr
    resource_id: NonEmptyStr
    spec_type: ActionSpecType
    spec_id: NonEmptyStr
    spec_digest: DigestStr
    execution_context: Literal["shadow"] = "shadow"
    receipt_authority: Literal["shadow_only"] = "shadow_only"
    status: ShadowRolloutStatus
    started_at: datetime
    finished_at: datetime
    converged: bool | None = None
    telemetry: dict[str, JsonValue] = Field(default_factory=dict)
    reason: str | None = None

    @model_validator(mode="after")
    def _validate_times(self) -> ShadowRolloutReceipt:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at precedes started_at")
        return self


WalRecord: TypeAlias = Annotated[  # noqa: UP040 - Python 3.10 compatibility
    ActionAttempt | PrimitiveReceipt | ExecutionReceipt | RecoveryAcknowledgement,
    Field(discriminator="record_kind"),
]
_WAL_RECORD_ADAPTER = TypeAdapter(WalRecord)


def validate_wal_record(value: WalRecord | Mapping[str, object] | str | bytes) -> WalRecord:
    if isinstance(value, (str, bytes)):
        return _WAL_RECORD_ADAPTER.validate_json(value)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    return _WAL_RECORD_ADAPTER.validate_python(value)


def action_spec_id(spec: ActionSpec) -> str:
    return spec.spec_id


ACTION_PROTOCOL_SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "robomex.admission_snapshot.v1": AdmissionSnapshot,
    "robomex.feasibility_certificate.v1": FeasibilityCertificate,
    "robomex.motion_plan.v2": MotionPlan,
    "robomex.gripper_command.v1": GripperCommand,
    "robomex.wait_spec.v1": WaitSpec,
    "robomex.action_backend.v1": BackendDescriptor,
    "robomex.monitor_telemetry_capabilities.v1": MonitorTelemetryCapabilities,
    "robomex.action_attempt.v1": ActionAttempt,
    "robomex.primitive_receipt.v1": PrimitiveReceipt,
    "robomex.execution_receipt.v2": ExecutionReceipt,
    "robomex.recovery_acknowledgement.v1": RecoveryAcknowledgement,
    "robomex.shadow_rollout_receipt.v1": ShadowRolloutReceipt,
}


__all__ = [
    "ActionAttempt",
    "ACTION_PROTOCOL_SCHEMA_MODELS",
    "ActionSpec",
    "ActionSpecType",
    "AdmissionSnapshot",
    "AttachmentStatus",
    "BackendCallResult",
    "BackendDescriptor",
    "BackendMotionInterface",
    "ControllerState",
    "DigestStr",
    "ExecutionPolicy",
    "ExecutionReceipt",
    "ExecutionStatus",
    "FeasibilityCertificate",
    "FeasibilityStatus",
    "GripperCommand",
    "GripperMode",
    "JointPath",
    "MotionPlan",
    "MonitorTelemetryCapabilities",
    "MonitorTelemetryHook",
    "PrimitiveReceipt",
    "PrimitiveStatus",
    "RecoveryAcknowledgement",
    "ShadowRolloutReceipt",
    "ShadowRolloutStatus",
    "WaitSpec",
    "WalRecord",
    "WorldKind",
    "action_spec_id",
    "canonical_model_digest",
    "canonical_payload_digest",
    "validate_action_spec",
    "validate_wal_record",
]
