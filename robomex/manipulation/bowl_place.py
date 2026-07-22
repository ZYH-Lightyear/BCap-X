"""Fail-closed bowl-on-plate visual-servo domain for RoboMEx v2.

This module intentionally stops at the physical-domain boundary.  It turns a
synchronised observation into a measured alignment error and a bounded
Cartesian/yaw correction request; it does not solve IK or execute a robot.
The v2 action author must turn a correction into a sealed ``MotionPlan`` and
the trusted runtime remains the only component allowed to execute it.
"""

from __future__ import annotations

import math
import uuid
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from robomex.data.embodied_state import AttachmentStatus
from robomex.data.physical_schema import RevisionVector
from robomex.runtime.events import ControlOutcome
from robomex.runtime.observation import ObservationRevisionVector

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    squared = dx * dx + dy * dy
    if squared == 0.0:
        return math.hypot(px - ax, py - ay)
    scale = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / squared))
    return math.hypot(px - (ax + scale * dx), py - (ay + scale * dy))


def _point_inside_polygon(
    point: tuple[float, float], polygon: tuple[tuple[float, float], ...]
) -> bool:
    if any(
        _point_segment_distance(point, start, end) <= 1e-12
        for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True)
    ):
        return True
    px, py = point
    inside = False
    for (x0, y0), (x1, y1) in zip(
        polygon, polygon[1:] + polygon[:1], strict=True
    ):
        crosses = (y0 > py) != (y1 > py)
        if crosses and px < (x1 - x0) * (py - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def _support_clearance_m(
    held: SupportFootprint,
    target: SupportFootprint,
    *,
    safe_margin_m: float,
) -> float:
    edges = tuple(
        zip(
            target.vertices_xy_m,
            target.vertices_xy_m[1:] + target.vertices_xy_m[:1],
            strict=True,
        )
    )
    signed_clearances = []
    for point in held.vertices_xy_m:
        distance = min(_point_segment_distance(point, start, end) for start, end in edges)
        signed_clearances.append(
            distance if _point_inside_polygon(point, target.vertices_xy_m) else -distance
        )
    return min(signed_clearances) - safe_margin_m


def _revision_dominates(current: RevisionVector, older: RevisionVector) -> bool:
    return (
        current.scene >= older.scene
        and current.arm >= older.arm
        and current.gripper >= older.gripper
        and current.attachment >= older.attachment
        and all(current.camera.get(camera_id, -1) >= revision for camera_id, revision in older.camera.items())
    )


def revision_vector_from_observation(
    revisions: ObservationRevisionVector,
    *,
    camera_id: str,
) -> RevisionVector:
    """Explicit tracking-stream → physical-schema revision adapter."""

    if not camera_id.strip():
        raise ValueError("camera_id must be non-empty")
    return RevisionVector(
        scene=revisions.scene_revision,
        arm=revisions.arm_revision,
        gripper=revisions.gripper_revision,
        attachment=revisions.attachment_revision,
        camera={camera_id: revisions.camera_revision},
    )


def observation_revision_from_physical(
    revisions: RevisionVector,
    *,
    camera_id: str,
) -> ObservationRevisionVector:
    """Explicit physical-schema → one-camera tracking revision adapter."""

    if not camera_id.strip():
        raise ValueError("camera_id must be non-empty")
    if camera_id not in revisions.camera:
        raise RevisionMismatchError(
            f"physical revision vector has no camera {camera_id!r}"
        )
    return ObservationRevisionVector(
        scene_revision=revisions.scene,
        arm_revision=revisions.arm,
        gripper_revision=revisions.gripper,
        attachment_revision=revisions.attachment,
        camera_revision=revisions.camera[camera_id],
    )


class BowlPlaceContractError(ValueError):
    """Base class for a rejected geometry/alignment contract."""


class EntityMismatchError(BowlPlaceContractError):
    """An estimate does not describe the requested stable entity."""


class FrameMismatchError(BowlPlaceContractError):
    """Geometry values are not expressed in the same coordinate frame."""


class RevisionMismatchError(BowlPlaceContractError):
    """Geometry values were not derived from one synchronized observation."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        str_strip_whitespace=True,
        validate_default=True,
    )


class Vector3(_StrictModel):
    """Finite metric vector with an explicit three-axis shape."""

    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)

    @classmethod
    def from_tuple(cls, value: tuple[float, float, float]) -> Vector3:
        return cls(x=value[0], y=value[1], z=value[2])

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def xy_norm(self) -> float:
        return math.hypot(self.x, self.y)

    def scaled(self, scale: float) -> Vector3:
        if not math.isfinite(scale):
            raise ValueError("vector scale must be finite")
        return Vector3(x=self.x * scale, y=self.y * scale, z=self.z * scale)

    def plus(self, other: Vector3) -> Vector3:
        return Vector3(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def minus(self, other: Vector3) -> Vector3:
        return Vector3(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)


class UnitVector3(Vector3):
    """Finite unit direction; used for the target support normal."""

    @model_validator(mode="after")
    def _unit_length(self) -> UnitVector3:
        if not math.isclose(self.norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("unit vector norm must equal one")
        return self


class QuaternionWXYZ(_StrictModel):
    """Unit quaternion in the canonical physical-schema ``wxyz`` order."""

    w: float = Field(allow_inf_nan=False)
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _unit_length(self) -> QuaternionWXYZ:
        norm = math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("quaternion must be unit length and use wxyz ordering")
        return self

    @classmethod
    def from_yaw(cls, yaw_rad: float) -> QuaternionWXYZ:
        if not math.isfinite(yaw_rad):
            raise ValueError("yaw must be finite")
        half = yaw_rad / 2.0
        return cls(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half))

    @property
    def yaw_rad(self) -> float:
        # Standard xyzw quaternion-to-yaw projection.  Roll/pitch are retained
        # in the schema even though this first servo profile only corrects yaw.
        numerator = 2.0 * (self.w * self.z + self.x * self.y)
        denominator = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        return math.atan2(numerator, denominator)


class SupportFootprint(_StrictModel):
    """Non-degenerate support polygon in the estimate's named metric frame."""

    vertices_xy_m: tuple[tuple[float, float], ...] = Field(min_length=3, max_length=128)

    @field_validator("vertices_xy_m")
    @classmethod
    def _finite_non_degenerate(
        cls, value: tuple[tuple[float, float], ...]
    ) -> tuple[tuple[float, float], ...]:
        if any(not math.isfinite(axis) for point in value for axis in point):
            raise ValueError("support footprint must contain only finite coordinates")
        if len(set(value)) < 3:
            raise ValueError("support footprint requires three distinct vertices")
        twice_area = abs(
            sum(
                x0 * y1 - x1 * y0
                for (x0, y0), (x1, y1) in zip(value, value[1:] + value[:1], strict=True)
            )
        )
        if twice_area <= 1e-12:
            raise ValueError("support footprint must have non-zero area")
        return value

    @property
    def area_m2(self) -> float:
        return 0.5 * abs(
            sum(
                x0 * y1 - x1 * y0
                for (x0, y0), (x1, y1) in zip(
                    self.vertices_xy_m,
                    self.vertices_xy_m[1:] + self.vertices_xy_m[:1],
                    strict=True,
                )
            )
        )

    def translated(self, delta: Vector3) -> SupportFootprint:
        return SupportFootprint(
            vertices_xy_m=tuple(
                (x + delta.x, y + delta.y) for x, y in self.vertices_xy_m
            )
        )


class PoseUncertainty(_StrictModel):
    translation_std_m: Vector3
    orientation_std_rad: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _non_negative_translation(self) -> PoseUncertainty:
        if min(self.translation_std_m.as_tuple()) < 0.0:
            raise ValueError("translation standard deviations must be non-negative")
        return self


class AttachmentGuardPhase(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    TRANSPORT = "transport"
    CORRECTION = "correction"
    DESCEND = "descend"
    OPEN = "open"


class AttachmentStateSnapshot(_StrictModel):
    """Typed episode-state projection used by v2 graph external bindings."""

    schema_version: Literal["robomex.attachment_state.v1"] = (
        "robomex.attachment_state.v1"
    )
    episode_id: NonEmptyStr
    state_revision: int = Field(ge=0)
    attachment_revision: int = Field(ge=0)
    status: AttachmentStatus
    entity_id: NonEmptyStr | None = None
    source_observation_id: NonEmptyStr | None = None
    evidence_refs: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _verified_state_has_evidence(self) -> AttachmentStateSnapshot:
        if self.status is AttachmentStatus.VERIFIED_HELD and (
            self.entity_id is None
            or self.source_observation_id is None
            or not self.evidence_refs
        ):
            raise ValueError(
                "verified_held attachment requires entity, observation, and evidence refs"
            )
        return self


class AttachmentGuard(_StrictModel):
    """Deterministic authorization result consumed before a held-object phase."""

    schema_version: Literal["robomex.attachment_guard.v1"] = (
        "robomex.attachment_guard.v1"
    )
    guard_id: NonEmptyStr = Field(default_factory=lambda: _new_id("attachment_guard"))
    phase: AttachmentGuardPhase
    allowed: bool
    status: AttachmentStatus
    entity_id: NonEmptyStr | None = None
    state_revision: int = Field(ge=0)
    observation_generation: int = Field(ge=1)
    reason: NonEmptyStr
    evidence_refs: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _authorization_matches_state(self) -> AttachmentGuard:
        can_authorize = (
            self.status is AttachmentStatus.VERIFIED_HELD
            and self.entity_id is not None
            and bool(self.evidence_refs)
        )
        if self.allowed != can_authorize:
            raise ValueError(
                "attachment guard may authorize only evidence-backed verified_held state"
            )
        return self


class HeldBowlEstimate(_StrictModel):
    """Observed bottom/support geometry of the *currently held* bowl."""

    schema_version: Literal["robomex.held_bowl_estimate.v1"] = (
        "robomex.held_bowl_estimate.v1"
    )
    estimate_id: NonEmptyStr = Field(default_factory=lambda: _new_id("held_bowl"))
    entity_id: NonEmptyStr
    frame_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    observation_generation: int = Field(ge=1)
    revisions: RevisionVector
    bottom_center_m: Vector3
    support_footprint: SupportFootprint
    orientation: QuaternionWXYZ
    uncertainty: PoseUncertainty
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evidence_refs: tuple[NonEmptyStr, ...] = ()


class PlateSupportTarget(_StrictModel):
    """Safe center/region on the plate, separated from held-object geometry."""

    schema_version: Literal["robomex.plate_support_target.v1"] = (
        "robomex.plate_support_target.v1"
    )
    target_id: NonEmptyStr = Field(default_factory=lambda: _new_id("plate_target"))
    entity_id: NonEmptyStr
    frame_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    observation_generation: int = Field(ge=1)
    revisions: RevisionVector
    support_center_m: Vector3
    support_footprint: SupportFootprint
    surface_normal: UnitVector3
    orientation: QuaternionWXYZ
    uncertainty: PoseUncertainty
    safe_margin_m: float = Field(ge=0.0, allow_inf_nan=False)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evidence_refs: tuple[NonEmptyStr, ...] = ()


class AlignmentTolerance(_StrictModel):
    translation_xy_m: float = Field(default=0.006, gt=0.0, allow_inf_nan=False)
    translation_z_m: float = Field(default=0.006, gt=0.0, allow_inf_nan=False)
    yaw_rad: float = Field(default=0.08, gt=0.0, le=math.pi, allow_inf_nan=False)


class AlignmentStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    WITHIN_TOLERANCE = "within_tolerance"
    CORRECTION_REQUIRED = "correction_required"


class AlignmentError(_StrictModel):
    """Measured error only; never an executable robot command."""

    schema_version: Literal["robomex.alignment_error.v1"] = (
        "robomex.alignment_error.v1"
    )
    alignment_id: NonEmptyStr = Field(default_factory=lambda: _new_id("alignment"))
    held_estimate_id: NonEmptyStr
    target_id: NonEmptyStr
    bowl_entity_id: NonEmptyStr
    target_entity_id: NonEmptyStr
    source_ref: NonEmptyStr
    target_ref: NonEmptyStr
    expressed_in_frame: NonEmptyStr
    snapshot_id: NonEmptyStr
    observation_generation: int = Field(ge=1)
    revisions: RevisionVector
    translation_error_m: Vector3
    yaw_error_rad: float = Field(ge=-math.pi, le=math.pi, allow_inf_nan=False)
    support_clearance_m: float = Field(allow_inf_nan=False)
    support_contained: bool
    uncertainty: PoseUncertainty
    tolerance: AlignmentTolerance
    status: AlignmentStatus

    @model_validator(mode="after")
    def _status_matches_measurement(self) -> AlignmentError:
        within = (
            self.translation_error_m.xy_norm <= self.tolerance.translation_xy_m
            and abs(self.translation_error_m.z) <= self.tolerance.translation_z_m
            and abs(self.yaw_error_rad) <= self.tolerance.yaw_rad
            and self.support_contained
        )
        if self.support_contained != (self.support_clearance_m >= 0.0):
            raise ValueError("support_contained does not match signed support clearance")
        expected = (
            AlignmentStatus.WITHIN_TOLERANCE
            if within
            else AlignmentStatus.CORRECTION_REQUIRED
        )
        if self.status is not expected:
            raise ValueError("alignment status does not match its measured error and tolerance")
        return self


def compute_alignment_error(
    held: HeldBowlEstimate,
    target: PlateSupportTarget,
    *,
    tolerance: AlignmentTolerance | None = None,
    expected_bowl_entity_id: str | None = None,
    expected_target_entity_id: str | None = None,
) -> AlignmentError:
    """Compute target-minus-held translation and relative yaw in one snapshot.

    Entity, frame, snapshot, generation, and full revision-vector mismatches are
    rejected instead of being silently transformed or treated as fresh data.
    """

    if expected_bowl_entity_id is not None and held.entity_id != expected_bowl_entity_id:
        raise EntityMismatchError(
            f"held entity {held.entity_id!r} does not match {expected_bowl_entity_id!r}"
        )
    if expected_target_entity_id is not None and target.entity_id != expected_target_entity_id:
        raise EntityMismatchError(
            f"target entity {target.entity_id!r} does not match {expected_target_entity_id!r}"
        )
    if held.frame_id != target.frame_id:
        raise FrameMismatchError(
            f"held frame {held.frame_id!r} does not match target frame {target.frame_id!r}"
        )
    if (
        held.snapshot_id != target.snapshot_id
        or held.observation_generation != target.observation_generation
        or held.revisions != target.revisions
    ):
        raise RevisionMismatchError(
            "held and target geometry must come from one snapshot/generation/revision vector"
        )

    stop_tolerance = tolerance or AlignmentTolerance()
    translation = target.support_center_m.minus(held.bottom_center_m)
    yaw_error = _wrap_angle(target.orientation.yaw_rad - held.orientation.yaw_rad)
    support_clearance = _support_clearance_m(
        held.support_footprint,
        target.support_footprint,
        safe_margin_m=target.safe_margin_m,
    )
    uncertainty = PoseUncertainty(
        translation_std_m=Vector3(
            x=math.hypot(
                held.uncertainty.translation_std_m.x,
                target.uncertainty.translation_std_m.x,
            ),
            y=math.hypot(
                held.uncertainty.translation_std_m.y,
                target.uncertainty.translation_std_m.y,
            ),
            z=math.hypot(
                held.uncertainty.translation_std_m.z,
                target.uncertainty.translation_std_m.z,
            ),
        ),
        orientation_std_rad=math.hypot(
            held.uncertainty.orientation_std_rad,
            target.uncertainty.orientation_std_rad,
        ),
    )
    within = (
        translation.xy_norm <= stop_tolerance.translation_xy_m
        and abs(translation.z) <= stop_tolerance.translation_z_m
        and abs(yaw_error) <= stop_tolerance.yaw_rad
        and support_clearance >= 0.0
    )
    return AlignmentError(
        held_estimate_id=held.estimate_id,
        target_id=target.target_id,
        bowl_entity_id=held.entity_id,
        target_entity_id=target.entity_id,
        source_ref=held.estimate_id,
        target_ref=target.target_id,
        expressed_in_frame=held.frame_id,
        snapshot_id=held.snapshot_id,
        observation_generation=held.observation_generation,
        revisions=held.revisions,
        translation_error_m=translation,
        yaw_error_rad=yaw_error,
        support_clearance_m=support_clearance,
        support_contained=support_clearance >= 0.0,
        uncertainty=uncertainty,
        tolerance=stop_tolerance,
        status=(
            AlignmentStatus.WITHIN_TOLERANCE
            if within
            else AlignmentStatus.CORRECTION_REQUIRED
        ),
    )


class VisibilityStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    VISIBLE = "visible"
    OCCLUDED = "occluded"
    AMBIGUOUS = "ambiguous"


class BowlPlaceObservation(_StrictModel):
    """Synchronized held-bowl/plate observation presented to one servo step."""

    schema_version: Literal["robomex.bowl_place_observation.v1"] = (
        "robomex.bowl_place_observation.v1"
    )
    snapshot_id: NonEmptyStr
    observation_generation: int = Field(ge=1)
    revisions: RevisionVector
    frame_id: NonEmptyStr
    expected_bowl_entity_id: NonEmptyStr
    expected_target_entity_id: NonEmptyStr
    attachment_status: AttachmentStatus
    held_visibility: VisibilityStatus
    target_visibility: VisibilityStatus
    held: HeldBowlEstimate | None = None
    target: PlateSupportTarget | None = None

    @model_validator(mode="after")
    def _synchronized_or_empty(self) -> BowlPlaceObservation:
        for label, visibility, estimate in (
            ("held", self.held_visibility, self.held),
            ("target", self.target_visibility, self.target),
        ):
            if visibility is VisibilityStatus.VISIBLE and estimate is None:
                raise ValueError(f"visible {label} geometry is required")
            if visibility is not VisibilityStatus.VISIBLE and estimate is not None:
                raise ValueError(f"{label} geometry must be cleared when not visible")
            if estimate is None:
                continue
            if estimate.frame_id != self.frame_id:
                raise ValueError(f"{label} geometry frame does not match observation envelope")
            if estimate.snapshot_id != self.snapshot_id:
                raise ValueError(f"{label} geometry snapshot does not match observation envelope")
            if estimate.observation_generation != self.observation_generation:
                raise ValueError(f"{label} geometry generation does not match observation envelope")
            if estimate.revisions != self.revisions:
                raise ValueError(f"{label} geometry revisions do not match observation envelope")
        return self


class CorrectionLimits(_StrictModel):
    max_step_translation_m: float = Field(default=0.015, gt=0.0, allow_inf_nan=False)
    max_step_yaw_rad: float = Field(default=0.12, gt=0.0, le=math.pi, allow_inf_nan=False)
    max_cumulative_translation_m: float = Field(
        default=0.06, gt=0.0, allow_inf_nan=False
    )
    max_cumulative_yaw_rad: float = Field(
        default=0.40, gt=0.0, le=2.0 * math.pi, allow_inf_nan=False
    )
    max_iterations: int = Field(default=6, ge=1, le=1000)
    max_target_drift_m: float = Field(default=0.02, gt=0.0, allow_inf_nan=False)


class BoundedCorrection(_StrictModel):
    """Bounded Cartesian/yaw request that still requires planning and sealing."""

    schema_version: Literal["robomex.bounded_correction.v1"] = (
        "robomex.bounded_correction.v1"
    )
    correction_id: NonEmptyStr = Field(default_factory=lambda: _new_id("correction"))
    iteration: int = Field(ge=1)
    expressed_in_frame: NonEmptyStr
    delta_translation_m: Vector3
    delta_yaw_rad: float = Field(ge=-math.pi, le=math.pi, allow_inf_nan=False)
    source_snapshot_id: NonEmptyStr
    source_generation: int = Field(ge=1)
    source_revisions: RevisionVector
    required_next_generation: int = Field(ge=2)
    cumulative_translation_m: float = Field(ge=0.0, allow_inf_nan=False)
    cumulative_yaw_rad: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _generation_advances(self) -> BoundedCorrection:
        if self.required_next_generation <= self.source_generation:
            raise ValueError("a correction must require a fresh next observation generation")
        if self.delta_translation_m.norm == 0.0 and self.delta_yaw_rad == 0.0:
            raise ValueError("a correction must contain a non-zero bounded delta")
        return self


class ServoOutcome(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    WITHIN_TOLERANCE = "within_tolerance"
    CORRECTION_REQUIRED = "correction_required"
    LOOP_EXHAUSTED = "loop_exhausted"
    TARGET_DRIFT = "target_drift"
    OCCLUDED = "occluded"
    DROPPED = "dropped"
    IDENTITY_SWAP = "identity_swap"
    STALE_OBSERVATION = "stale_observation"
    ATTACHMENT_NOT_CONFIRMED = "attachment_not_confirmed"


_CONTROL_OUTCOME = {
    ServoOutcome.WITHIN_TOLERANCE: ControlOutcome.SUCCESS,
    ServoOutcome.CORRECTION_REQUIRED: ControlOutcome.NEEDS_ADJUSTMENT,
    ServoOutcome.LOOP_EXHAUSTED: ControlOutcome.EXHAUSTED,
    ServoOutcome.TARGET_DRIFT: ControlOutcome.TARGET_DRIFT,
    ServoOutcome.OCCLUDED: ControlOutcome.UNCERTAIN,
    ServoOutcome.DROPPED: ControlOutcome.ATTACHMENT_NOT_CONFIRMED,
    ServoOutcome.IDENTITY_SWAP: ControlOutcome.WRONG_GROUNDING,
    ServoOutcome.STALE_OBSERVATION: ControlOutcome.STALE_OBSERVATION,
    ServoOutcome.ATTACHMENT_NOT_CONFIRMED: ControlOutcome.ATTACHMENT_NOT_CONFIRMED,
}


class ServoDecision(_StrictModel):
    schema_version: Literal["robomex.servo_decision.v1"] = "robomex.servo_decision.v1"
    outcome: ServoOutcome
    control_outcome: ControlOutcome
    observation_generation: int = Field(ge=1)
    reason: NonEmptyStr
    alignment_error: AlignmentError | None = None
    correction: BoundedCorrection | None = None

    @model_validator(mode="after")
    def _closed_shape(self) -> ServoDecision:
        if self.control_outcome is not _CONTROL_OUTCOME[self.outcome]:
            raise ValueError("control_outcome does not match the closed servo outcome")
        if self.outcome is ServoOutcome.CORRECTION_REQUIRED:
            if self.correction is None or self.alignment_error is None:
                raise ValueError("correction_required needs measured error and bounded correction")
        elif self.correction is not None:
            raise ValueError("only correction_required may carry a correction")
        if self.outcome is ServoOutcome.WITHIN_TOLERANCE and (
            self.alignment_error is None
            or self.alignment_error.status is not AlignmentStatus.WITHIN_TOLERANCE
        ):
            raise ValueError("within_tolerance needs a matching alignment measurement")
        return self


class PlacementPhase(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    CORRECTION = "correction"
    DESCEND = "descend"
    OPEN = "open"


class PhaseAuthorization(_StrictModel):
    phase: PlacementPhase
    allowed: bool
    outcome: ServoOutcome
    control_outcome: ControlOutcome
    reason: NonEmptyStr

    @model_validator(mode="after")
    def _closed_outcome(self) -> PhaseAuthorization:
        if self.control_outcome is not _CONTROL_OUTCOME[self.outcome]:
            raise ValueError("phase control outcome does not match servo outcome")
        return self


class CheckpointPhase(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    AFTER_TRANSPORT = "after_transport"
    PRE_RELEASE = "pre_release"
    POST_RELEASE = "post_release"


class CheckpointStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    PASSED = "passed"
    BLOCKED = "blocked"
    UNCERTAIN = "uncertain"


class PhaseCheckpoint(_StrictModel):
    """Evidence-backed phase boundary; never a free-form success message."""

    schema_version: Literal["robomex.phase_checkpoint.v1"] = (
        "robomex.phase_checkpoint.v1"
    )
    checkpoint_id: NonEmptyStr = Field(default_factory=lambda: _new_id("checkpoint"))
    phase: CheckpointPhase
    status: CheckpointStatus
    action_id: NonEmptyStr
    receipt_ref: NonEmptyStr
    attachment_status: AttachmentStatus
    bowl_entity_id: NonEmptyStr
    alignment_id: NonEmptyStr | None = None
    observation_generation: int = Field(ge=1)
    revisions: RevisionVector
    evidence_refs: tuple[NonEmptyStr, ...] = Field(min_length=1)
    reason: NonEmptyStr

    @model_validator(mode="after")
    def _phase_invariants(self) -> PhaseCheckpoint:
        if self.status is not CheckpointStatus.PASSED:
            return self
        if (
            self.phase in {CheckpointPhase.AFTER_TRANSPORT, CheckpointPhase.PRE_RELEASE}
            and self.attachment_status is not AttachmentStatus.VERIFIED_HELD
        ):
            raise ValueError("held-object checkpoint requires verified_held attachment")
        if self.phase is CheckpointPhase.PRE_RELEASE and self.alignment_id is None:
            raise ValueError("passed pre-release checkpoint requires an alignment id")
        if (
            self.phase is CheckpointPhase.POST_RELEASE
            and self.attachment_status is not AttachmentStatus.NOT_HELD
        ):
            raise ValueError("passed post-release checkpoint requires fresh not_held evidence")
        return self


class RelationAssessment(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    ASSERTED = "asserted"
    NEGATED = "negated"
    UNKNOWN = "unknown"


class PlacementVerdictStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class PlacementVerdict(_StrictModel):
    """Independent final support-relation assessment."""

    schema_version: Literal["robomex.placement_verdict.v1"] = (
        "robomex.placement_verdict.v1"
    )
    verdict_id: NonEmptyStr = Field(default_factory=lambda: _new_id("placement_verdict"))
    status: PlacementVerdictStatus
    bowl_entity_id: NonEmptyStr
    target_entity_id: NonEmptyStr
    desired_relation: Literal["on_top_of"] = "on_top_of"
    relation: RelationAssessment
    source_observation_id: NonEmptyStr
    state_revision: int = Field(ge=0)
    support_clearance_m: float = Field(allow_inf_nan=False)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evidence_refs: tuple[NonEmptyStr, ...] = Field(min_length=1)
    reason: NonEmptyStr

    @model_validator(mode="after")
    def _verdict_matches_relation(self) -> PlacementVerdict:
        if self.bowl_entity_id == self.target_entity_id:
            raise ValueError("placement subject and target identities must differ")
        if self.status is PlacementVerdictStatus.SUCCEEDED and (
            self.relation is not RelationAssessment.ASSERTED
            or self.support_clearance_m < 0.0
        ):
            raise ValueError(
                "successful placement requires asserted relation and non-negative clearance"
            )
        if (
            self.status is PlacementVerdictStatus.FAILED
            and self.relation is RelationAssessment.ASSERTED
        ):
            raise ValueError("failed placement cannot assert the desired relation")
        if (
            self.status is PlacementVerdictStatus.UNCERTAIN
            and self.relation is not RelationAssessment.UNKNOWN
        ):
            raise ValueError("uncertain placement requires unknown relation")
        return self


class RecoveryDisposition(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    SAFE_TO_PATCH = "safe_to_patch"
    REOBSERVE_REQUIRED = "reobserve_required"
    SAFE_STOP = "safe_stop"


class RecoveryEffectScope(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    READ_ONLY = "read_only"
    SHADOW_WORLD = "shadow_world"
    AUTHORITATIVE_WORLD = "authoritative_world"


class RecoverySafetyDecision(_StrictModel):
    """Verifier obligation required before a recovery fragment may gain effects."""

    schema_version: Literal["robomex.recovery_safety_decision.v1"] = (
        "robomex.recovery_safety_decision.v1"
    )
    decision_id: NonEmptyStr = Field(default_factory=lambda: _new_id("recovery_safety"))
    disposition: RecoveryDisposition
    patch_authorized: bool
    graph_revision: int = Field(ge=1)
    state_revision: int = Field(ge=0)
    attachment_status: AttachmentStatus
    allowed_effect_scopes: tuple[RecoveryEffectScope, ...]
    blocked_activation_ids: tuple[NonEmptyStr, ...] = ()
    required_refreshes: tuple[NonEmptyStr, ...] = ()
    evidence_refs: tuple[NonEmptyStr, ...] = Field(min_length=1)
    reason: NonEmptyStr

    @model_validator(mode="after")
    def _closed_recovery_authority(self) -> RecoverySafetyDecision:
        if self.patch_authorized != (
            self.disposition is RecoveryDisposition.SAFE_TO_PATCH
        ):
            raise ValueError("patch authorization must match recovery disposition")
        if len(self.allowed_effect_scopes) != len(set(self.allowed_effect_scopes)):
            raise ValueError("recovery effect scopes must be unique")
        if (
            self.attachment_status is not AttachmentStatus.VERIFIED_HELD
            and "execute_open" not in self.blocked_activation_ids
        ):
            raise ValueError("unverified attachment recovery must explicitly block execute_open")
        if (
            self.disposition is RecoveryDisposition.REOBSERVE_REQUIRED
            and not self.required_refreshes
        ):
            raise ValueError("reobserve_required decision needs typed refresh obligations")
        return self


class VisualServoPlacer:
    """Stateful bounded servo session with mandatory post-action freshness."""

    def __init__(
        self,
        *,
        bowl_entity_id: str,
        target_entity_id: str,
        frame_id: str,
        tolerance: AlignmentTolerance | None = None,
        limits: CorrectionLimits | None = None,
    ) -> None:
        if not bowl_entity_id.strip() or not target_entity_id.strip() or not frame_id.strip():
            raise ValueError("servo entity and frame ids must be non-empty")
        self.bowl_entity_id = bowl_entity_id
        self.target_entity_id = target_entity_id
        self.frame_id = frame_id
        self.tolerance = tolerance or AlignmentTolerance()
        self.limits = limits or CorrectionLimits()
        self._iterations = 0
        self._cumulative_translation_m = 0.0
        self._cumulative_yaw_rad = 0.0
        self._freshness_floor: tuple[int, RevisionVector] | None = None
        self._target_reference: Vector3 | None = None

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def cumulative_translation_m(self) -> float:
        return self._cumulative_translation_m

    @property
    def cumulative_yaw_rad(self) -> float:
        return self._cumulative_yaw_rad

    def assess(self, observation: BowlPlaceObservation) -> ServoDecision:
        """Evaluate one synchronized observation and reserve at most one correction."""

        if observation.attachment_status is AttachmentStatus.NOT_HELD:
            return self._decision(observation, ServoOutcome.DROPPED, "bowl is no longer held")
        if observation.attachment_status is not AttachmentStatus.VERIFIED_HELD:
            return self._decision(
                observation,
                ServoOutcome.ATTACHMENT_NOT_CONFIRMED,
                "attachment is not verified_held",
            )
        if (
            observation.held_visibility is not VisibilityStatus.VISIBLE
            or observation.target_visibility is not VisibilityStatus.VISIBLE
        ):
            return self._decision(
                observation,
                ServoOutcome.OCCLUDED,
                "held bowl or plate target is occluded/ambiguous",
            )
        assert observation.held is not None  # guaranteed by the strict observation model
        assert observation.target is not None
        if (
            observation.expected_bowl_entity_id != self.bowl_entity_id
            or observation.expected_target_entity_id != self.target_entity_id
            or observation.held.entity_id != self.bowl_entity_id
            or observation.target.entity_id != self.target_entity_id
        ):
            return self._decision(
                observation,
                ServoOutcome.IDENTITY_SWAP,
                "observed entity identity differs from the active servo contract",
            )
        if observation.frame_id != self.frame_id:
            return self._decision(
                observation,
                ServoOutcome.STALE_OBSERVATION,
                "observation frame differs from the active servo contract",
            )
        if self._freshness_floor is not None:
            generation, revisions = self._freshness_floor
            is_fresh = (
                observation.observation_generation > generation
                and _revision_dominates(observation.revisions, revisions)
                and observation.revisions != revisions
                and observation.revisions.arm > revisions.arm
            )
            if not is_fresh:
                return self._decision(
                    observation,
                    ServoOutcome.STALE_OBSERVATION,
                    "a correction requires a later generation and advanced arm revision",
                )
            self._freshness_floor = None

        if self._target_reference is None:
            self._target_reference = observation.target.support_center_m
        elif (
            observation.target.support_center_m.minus(self._target_reference).norm
            > self.limits.max_target_drift_m
        ):
            return self._decision(
                observation,
                ServoOutcome.TARGET_DRIFT,
                "plate support target moved beyond the admitted drift bound",
            )

        try:
            error = compute_alignment_error(
                observation.held,
                observation.target,
                tolerance=self.tolerance,
                expected_bowl_entity_id=self.bowl_entity_id,
                expected_target_entity_id=self.target_entity_id,
            )
        except EntityMismatchError as exc:
            return self._decision(observation, ServoOutcome.IDENTITY_SWAP, str(exc))
        except (FrameMismatchError, RevisionMismatchError) as exc:
            return self._decision(observation, ServoOutcome.STALE_OBSERVATION, str(exc))

        if error.status is AlignmentStatus.WITHIN_TOLERANCE:
            return self._decision(
                observation,
                ServoOutcome.WITHIN_TOLERANCE,
                "bowl support footprint is aligned within stop tolerance",
                error=error,
            )
        if self._iterations >= self.limits.max_iterations:
            return self._decision(
                observation,
                ServoOutcome.LOOP_EXHAUSTED,
                "alignment correction iteration budget is exhausted",
                error=error,
            )

        translation_remaining = (
            self.limits.max_cumulative_translation_m - self._cumulative_translation_m
        )
        yaw_remaining = self.limits.max_cumulative_yaw_rad - self._cumulative_yaw_rad
        requested_translation = error.translation_error_m
        translation_norm = requested_translation.norm
        translation_limit = min(self.limits.max_step_translation_m, translation_remaining)
        if translation_norm > translation_limit > 0.0:
            translation = requested_translation.scaled(translation_limit / translation_norm)
        elif translation_limit > 0.0:
            translation = requested_translation
        else:
            translation = Vector3(x=0.0, y=0.0, z=0.0)
        yaw_limit = min(self.limits.max_step_yaw_rad, yaw_remaining)
        yaw = math.copysign(min(abs(error.yaw_error_rad), max(0.0, yaw_limit)), error.yaw_error_rad)
        if translation.norm == 0.0 and yaw == 0.0:
            return self._decision(
                observation,
                ServoOutcome.LOOP_EXHAUSTED,
                "cumulative translation and rotation budgets are exhausted",
                error=error,
            )

        self._iterations += 1
        self._cumulative_translation_m += translation.norm
        self._cumulative_yaw_rad += abs(yaw)
        correction = BoundedCorrection(
            iteration=self._iterations,
            expressed_in_frame=self.frame_id,
            delta_translation_m=translation,
            delta_yaw_rad=yaw,
            source_snapshot_id=observation.snapshot_id,
            source_generation=observation.observation_generation,
            source_revisions=observation.revisions,
            required_next_generation=observation.observation_generation + 1,
            cumulative_translation_m=self._cumulative_translation_m,
            cumulative_yaw_rad=self._cumulative_yaw_rad,
        )
        # Planning a correction consumes the bounded loop slot.  Re-entering
        # with the same evidence is rejected even if execution code retries.
        self._freshness_floor = (
            observation.observation_generation,
            observation.revisions,
        )
        return self._decision(
            observation,
            ServoOutcome.CORRECTION_REQUIRED,
            "one bounded correction is required before re-observation",
            error=error,
            correction=correction,
        )

    def authorize_phase(
        self,
        phase: PlacementPhase,
        *,
        attachment_status: AttachmentStatus,
        decision: ServoDecision | None,
    ) -> PhaseAuthorization:
        """Fail-closed guard for correction, descend, and open phases."""

        if attachment_status is AttachmentStatus.NOT_HELD:
            return self._authorization(phase, False, ServoOutcome.DROPPED, "bowl was dropped")
        if attachment_status is not AttachmentStatus.VERIFIED_HELD:
            return self._authorization(
                phase,
                False,
                ServoOutcome.ATTACHMENT_NOT_CONFIRMED,
                "phase requires attachment=verified_held",
            )
        if decision is None:
            return self._authorization(
                phase,
                False,
                ServoOutcome.STALE_OBSERVATION,
                "phase requires a fresh servo decision",
            )
        if phase is PlacementPhase.CORRECTION:
            allowed = (
                decision.outcome is ServoOutcome.CORRECTION_REQUIRED
                and decision.correction is not None
            )
            outcome = decision.outcome if not allowed else ServoOutcome.CORRECTION_REQUIRED
            return self._authorization(
                phase,
                allowed,
                outcome,
                "bounded correction authorized" if allowed else "no correction is authorized",
            )
        allowed = decision.outcome is ServoOutcome.WITHIN_TOLERANCE
        outcome = decision.outcome if not allowed else ServoOutcome.WITHIN_TOLERANCE
        return self._authorization(
            phase,
            allowed,
            outcome,
            (
                f"{phase.value} authorized by fresh within-tolerance evidence"
                if allowed
                else f"{phase.value} requires within-tolerance evidence"
            ),
        )

    @staticmethod
    def _decision(
        observation: BowlPlaceObservation,
        outcome: ServoOutcome,
        reason: str,
        *,
        error: AlignmentError | None = None,
        correction: BoundedCorrection | None = None,
    ) -> ServoDecision:
        return ServoDecision(
            outcome=outcome,
            control_outcome=_CONTROL_OUTCOME[outcome],
            observation_generation=observation.observation_generation,
            reason=reason,
            alignment_error=error,
            correction=correction,
        )

    @staticmethod
    def _authorization(
        phase: PlacementPhase,
        allowed: bool,
        outcome: ServoOutcome,
        reason: str,
    ) -> PhaseAuthorization:
        return PhaseAuthorization(
            phase=phase,
            allowed=allowed,
            outcome=outcome,
            control_outcome=_CONTROL_OUTCOME[outcome],
            reason=reason,
        )


class FixedOffsetBaselineResult(_StrictModel):
    """Open-loop ablation result using one world-frame TCP-to-bottom offset."""

    schema_version: Literal["robomex.fixed_offset_baseline.v1"] = (
        "robomex.fixed_offset_baseline.v1"
    )
    desired_tcp_position_m: Vector3
    translation_delta_m: Vector3
    assumed_tcp_to_bottom_offset_world_m: Vector3


def fixed_offset_baseline(
    *,
    current_tcp_position_m: Vector3,
    target_support_center_m: Vector3,
    assumed_tcp_to_bottom_offset_world_m: Vector3,
) -> FixedOffsetBaselineResult:
    """Legacy open-loop helper retained only as a reproducible ablation.

    The offset is deliberately interpreted in the world frame.  It therefore
    exposes the rim-grasp/orientation weakness that the closed-loop method is
    designed to measure rather than hiding it behind a new estimator.
    """

    desired_tcp = target_support_center_m.minus(assumed_tcp_to_bottom_offset_world_m)
    return FixedOffsetBaselineResult(
        desired_tcp_position_m=desired_tcp,
        translation_delta_m=desired_tcp.minus(current_tcp_position_m),
        assumed_tcp_to_bottom_offset_world_m=assumed_tcp_to_bottom_offset_world_m,
    )


def apply_correction_to_estimate(
    held: HeldBowlEstimate,
    correction: BoundedCorrection,
    *,
    next_snapshot_id: str,
    next_generation: int,
    next_revisions: RevisionVector,
) -> HeldBowlEstimate:
    """Pure simulation/replay helper; never an action backend."""

    if correction.expressed_in_frame != held.frame_id:
        raise FrameMismatchError("correction and held estimate frames differ")
    if correction.source_snapshot_id != held.snapshot_id:
        raise RevisionMismatchError("correction does not bind the held estimate snapshot")
    if correction.source_generation != held.observation_generation:
        raise RevisionMismatchError("correction does not bind the held estimate generation")
    if correction.source_revisions != held.revisions:
        raise RevisionMismatchError("correction does not bind the held estimate revisions")
    if next_generation < correction.required_next_generation:
        raise RevisionMismatchError("corrected estimate requires a fresh observation generation")
    if (
        not _revision_dominates(next_revisions, held.revisions)
        or next_revisions.arm <= held.revisions.arm
    ):
        raise RevisionMismatchError("corrected estimate requires an advanced arm revision")
    delta = correction.delta_translation_m
    next_yaw = _wrap_angle(held.orientation.yaw_rad + correction.delta_yaw_rad)
    cosine = math.cos(correction.delta_yaw_rad)
    sine = math.sin(correction.delta_yaw_rad)
    rotated_footprint = SupportFootprint(
        vertices_xy_m=tuple(
            (
                held.bottom_center_m.x
                + delta.x
                + cosine * (x - held.bottom_center_m.x)
                - sine * (y - held.bottom_center_m.y),
                held.bottom_center_m.y
                + delta.y
                + sine * (x - held.bottom_center_m.x)
                + cosine * (y - held.bottom_center_m.y),
            )
            for x, y in held.support_footprint.vertices_xy_m
        )
    )
    return HeldBowlEstimate(
        entity_id=held.entity_id,
        frame_id=held.frame_id,
        snapshot_id=next_snapshot_id,
        observation_generation=next_generation,
        revisions=next_revisions,
        bottom_center_m=held.bottom_center_m.plus(delta),
        support_footprint=rotated_footprint,
        orientation=QuaternionWXYZ.from_yaw(next_yaw),
        uncertainty=held.uncertainty,
        confidence=held.confidence,
        evidence_refs=held.evidence_refs,
    )


BOWL_PLACE_SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "robomex.attachment_state.v1": AttachmentStateSnapshot,
    "robomex.attachment_guard.v1": AttachmentGuard,
    "robomex.held_bowl_estimate.v1": HeldBowlEstimate,
    "robomex.plate_support_target.v1": PlateSupportTarget,
    "robomex.alignment_error.v1": AlignmentError,
    "robomex.bowl_place_observation.v1": BowlPlaceObservation,
    "robomex.bounded_correction.v1": BoundedCorrection,
    "robomex.servo_decision.v1": ServoDecision,
    "robomex.phase_checkpoint.v1": PhaseCheckpoint,
    "robomex.placement_verdict.v1": PlacementVerdict,
    "robomex.recovery_safety_decision.v1": RecoverySafetyDecision,
    "robomex.fixed_offset_baseline.v1": FixedOffsetBaselineResult,
}


def register_bowl_place_schemas(registry: object) -> None:
    """Install every domain payload into a v2 ``SchemaRegistry``.

    The structural protocol keeps this module independent of the data-plane
    implementation while still refusing an untyped schema-name-only graph.
    """

    register = getattr(registry, "register", None)
    is_registered = getattr(registry, "is_registered", None)
    if not callable(register) or not callable(is_registered):
        raise TypeError("registry must provide register() and is_registered()")
    existing = [schema_id for schema_id in BOWL_PLACE_SCHEMA_MODELS if is_registered(schema_id)]
    if existing:
        raise ValueError(f"schema {existing[0]!r} is already registered")
    for schema_id, model in BOWL_PLACE_SCHEMA_MODELS.items():
        register(schema_id, model)


__all__ = [
    "AlignmentError",
    "AlignmentStatus",
    "AlignmentTolerance",
    "AttachmentGuard",
    "AttachmentGuardPhase",
    "AttachmentStateSnapshot",
    "BowlPlaceContractError",
    "BowlPlaceObservation",
    "BOWL_PLACE_SCHEMA_MODELS",
    "BoundedCorrection",
    "CorrectionLimits",
    "CheckpointPhase",
    "CheckpointStatus",
    "EntityMismatchError",
    "FixedOffsetBaselineResult",
    "FrameMismatchError",
    "HeldBowlEstimate",
    "PhaseAuthorization",
    "PhaseCheckpoint",
    "PlacementVerdict",
    "PlacementVerdictStatus",
    "PlateSupportTarget",
    "PlacementPhase",
    "PoseUncertainty",
    "QuaternionWXYZ",
    "RecoveryDisposition",
    "RecoveryEffectScope",
    "RecoverySafetyDecision",
    "RelationAssessment",
    "RevisionMismatchError",
    "ServoDecision",
    "ServoOutcome",
    "SupportFootprint",
    "UnitVector3",
    "Vector3",
    "VisibilityStatus",
    "VisualServoPlacer",
    "apply_correction_to_estimate",
    "compute_alignment_error",
    "fixed_offset_baseline",
    "observation_revision_from_physical",
    "register_bowl_place_schemas",
    "revision_vector_from_observation",
]
