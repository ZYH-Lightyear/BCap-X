"""Strict physical payload schemas and revision-aware validity checks.

The models in this module are deliberately small and semantic.  A point is
not interchangeable with a direction, transforms state their direction, and
quaternions state their wire convention.  These types form the action-facing
schema boundary; presentation dictionaries and legacy payloads must be
adapted *before* they reach it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
Vector3: TypeAlias = tuple[FiniteFloat, FiniteFloat, FiniteFloat]  # noqa: UP040
QuaternionWXYZ: TypeAlias = (  # noqa: UP040
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
)
Matrix3: TypeAlias = tuple[Vector3, Vector3, Vector3]  # noqa: UP040

_DOMAIN_RE = re.compile(r"^(scene|arm|gripper|attachment|camera\.[A-Za-z0-9_-]+)$")


class PhysicalSchemaError(ValueError):
    """A physical value is structurally valid JSON but semantically invalid."""


class FreshnessRejected(PhysicalSchemaError):  # noqa: N818 - domain rejection outcome
    """A validity vector cannot be admitted against the current physical state."""


class StrictPhysicalModel(BaseModel):
    """Base policy shared by all action-facing physical schemas."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _require_unit(values: Sequence[float], *, label: str, tolerance: float = 1e-4) -> None:
    norm = _norm(values)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} must be unit length; got norm {norm:.8g}.")


class Point3(StrictPhysicalModel):
    """A metric position in a named frame at an optional observation boundary."""

    xyz_m: Vector3
    frame_id: NonEmptyStr
    observation_id: NonEmptyStr | None = None
    length_unit: Literal["m"] = "m"


class Direction3(StrictPhysicalModel):
    """A unit direction, intentionally incompatible with :class:`Point3`."""

    xyz_unit: Vector3
    frame_id: NonEmptyStr
    observation_id: NonEmptyStr | None = None
    vector_semantics: Literal["direction"] = "direction"

    @field_validator("xyz_unit")
    @classmethod
    def _unit_direction(cls, value: Vector3) -> Vector3:
        _require_unit(value, label="Direction3.xyz_unit")
        return value


class Pose(StrictPhysicalModel):
    """Pose of a child/object expressed in ``frame_id``.

    Quaternion wire order is fixed to ``wxyz``.  Values are validated, never
    silently normalized, because normalization would change an admitted goal.
    """

    position_xyz_m: Vector3
    quaternion_wxyz: QuaternionWXYZ
    frame_id: NonEmptyStr
    observation_id: NonEmptyStr | None = None
    length_unit: Literal["m"] = "m"
    quaternion_convention: Literal["wxyz"] = "wxyz"

    @field_validator("quaternion_wxyz")
    @classmethod
    def _unit_quaternion(cls, value: QuaternionWXYZ) -> QuaternionWXYZ:
        _require_unit(value, label="Pose.quaternion_wxyz")
        return value


class Transform(StrictPhysicalModel):
    """Explicit ``parent_T_child`` rigid transform."""

    parent_frame: NonEmptyStr
    child_frame: NonEmptyStr
    translation_xyz_m: Vector3
    quaternion_wxyz: QuaternionWXYZ
    observation_id: NonEmptyStr | None = None
    transform_convention: Literal["parent_T_child"] = "parent_T_child"
    length_unit: Literal["m"] = "m"
    quaternion_convention: Literal["wxyz"] = "wxyz"

    @field_validator("quaternion_wxyz")
    @classmethod
    def _unit_quaternion(cls, value: QuaternionWXYZ) -> QuaternionWXYZ:
        _require_unit(value, label="Transform.quaternion_wxyz")
        return value

    @model_validator(mode="after")
    def _different_frames(self) -> Transform:
        if self.parent_frame == self.child_frame:
            raise ValueError("Transform parent_frame and child_frame must differ.")
        return self


# The architecture prose uses the stamped names; retain precise aliases
# without introducing a second wire representation.
Point3Stamped = Point3
Direction3Stamped = Direction3
PoseStamped = Pose
TransformStamped = Transform


class QualityEvidence(StrictPhysicalModel):
    """Auditable method choice; fallback may never be implicit."""

    method: NonEmptyStr
    reason: NonEmptyStr
    confidence: Confidence


class RevisionVector(StrictPhysicalModel):
    """Current episode revision clock for physical validity domains."""

    scene: int = Field(ge=0)
    arm: int = Field(ge=0)
    gripper: int = Field(ge=0)
    attachment: int = Field(ge=0)
    camera: dict[NonEmptyStr, int] = Field(default_factory=dict)

    @field_validator("scene", "arm", "gripper", "attachment", mode="before")
    @classmethod
    def _reject_boolean_revision(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("Revision values must be integers, not booleans.")
        return value

    @field_validator("camera")
    @classmethod
    def _camera_revisions(cls, value: dict[str, int]) -> dict[str, int]:
        for camera_id, revision in value.items():
            if isinstance(revision, bool) or revision < 0:
                raise ValueError(f"Invalid revision for camera {camera_id!r}.")
        return value

    def domain_revision(self, domain: str) -> int:
        if domain.startswith("camera."):
            camera_id = domain.removeprefix("camera.")
            if camera_id not in self.camera:
                raise FreshnessRejected(
                    f"Current revision vector has no camera domain {domain!r}."
                )
            return self.camera[camera_id]
        if domain not in {"scene", "arm", "gripper", "attachment"}:
            raise FreshnessRejected(f"Unknown physical revision domain {domain!r}.")
        return int(getattr(self, domain))

    def as_domains(self) -> dict[str, int]:
        domains = {
            "scene": self.scene,
            "arm": self.arm,
            "gripper": self.gripper,
            "attachment": self.attachment,
        }
        domains.update({f"camera.{key}": value for key, value in self.camera.items()})
        return domains


class ValidityLifecycle(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    SNAPSHOT = "snapshot"
    DERIVED = "derived"
    PLAN = "plan"
    RECEIPT = "receipt"
    STATE = "state"
    STATIC = "static"


class AdmissionPurpose(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    PHYSICAL_ACTION = "physical_action"
    VERIFICATION = "verification"
    HISTORICAL_REPLAY = "historical_replay"


class FreshnessDecision(StrictPhysicalModel):
    admissible: Literal[True] = True
    lifecycle: ValidityLifecycle
    checked_domains: tuple[NonEmptyStr, ...]
    method: NonEmptyStr
    reason: NonEmptyStr
    confidence: Confidence


class ValidityVector(StrictPhysicalModel):
    """Causal lifetime declaration checked at every consumer admission.

    ``depends_on_revisions`` is intentionally sparse: an artifact declares the
    exact domains that can invalidate it.  Missing current domains and stale
    values are errors, not signals to consult a latest artifact or fallback.
    """

    lifecycle: ValidityLifecycle
    depends_on_revisions: dict[NonEmptyStr, int] = Field(default_factory=dict)
    observation_id: NonEmptyStr | None = None
    state_revision: int | None = Field(default=None, ge=0)
    action_id: NonEmptyStr | None = None
    method: NonEmptyStr
    reason: NonEmptyStr
    confidence: Confidence

    @field_validator("depends_on_revisions")
    @classmethod
    def _known_domains(cls, value: dict[str, int]) -> dict[str, int]:
        for domain, revision in value.items():
            if not _DOMAIN_RE.fullmatch(domain):
                raise ValueError(f"Unknown physical revision domain {domain!r}.")
            if isinstance(revision, bool) or revision < 0:
                raise ValueError(f"Invalid revision for domain {domain!r}.")
        return value

    @model_validator(mode="after")
    def _lifecycle_identity(self) -> ValidityVector:
        if (
            self.lifecycle in {ValidityLifecycle.SNAPSHOT, ValidityLifecycle.DERIVED}
            and self.observation_id is None
        ):
            raise ValueError(f"{self.lifecycle.value} validity requires observation_id.")
        if self.lifecycle is ValidityLifecycle.PLAN and not self.depends_on_revisions:
            raise ValueError("plan validity requires explicit physical revision dependencies.")
        if self.lifecycle is ValidityLifecycle.RECEIPT and self.action_id is None:
            raise ValueError("receipt validity requires action_id.")
        if self.lifecycle is ValidityLifecycle.STATE and self.state_revision is None:
            raise ValueError("state validity requires state_revision.")
        return self

    def assert_admissible(
        self,
        current: RevisionVector,
        *,
        purpose: AdmissionPurpose = AdmissionPurpose.PHYSICAL_ACTION,
        current_observation_id: str | None = None,
        current_state_revision: int | None = None,
        expected_action_id: str | None = None,
    ) -> FreshnessDecision:
        """Fail closed unless every declared causal guard still matches."""

        if not isinstance(current, RevisionVector):
            raise FreshnessRejected("Admission requires a typed current RevisionVector.")
        if not isinstance(purpose, AdmissionPurpose):
            raise FreshnessRejected("Admission purpose must use the closed enum.")
        if (
            purpose is AdmissionPurpose.PHYSICAL_ACTION
            and self.lifecycle is ValidityLifecycle.RECEIPT
        ):
            raise FreshnessRejected("A historical receipt cannot be admitted as an action spec.")
        if (
            purpose is AdmissionPurpose.PHYSICAL_ACTION
            and self.lifecycle is ValidityLifecycle.STATE
            and current_state_revision is None
        ):
            raise FreshnessRejected("State admission requires current_state_revision.")
        if self.state_revision is not None:
            if current_state_revision is None:
                raise FreshnessRejected("No current state revision was supplied.")
            if self.state_revision != current_state_revision:
                raise FreshnessRejected(
                    f"Stale state revision {self.state_revision}; current is "
                    f"{current_state_revision}."
                )
        if self.lifecycle in {ValidityLifecycle.SNAPSHOT, ValidityLifecycle.DERIVED}:
            if current_observation_id is None:
                raise FreshnessRejected("Snapshot-derived admission requires current observation id.")
            if self.observation_id != current_observation_id:
                raise FreshnessRejected(
                    f"Stale observation {self.observation_id!r}; current is "
                    f"{current_observation_id!r}."
                )
        if (
            self.action_id is not None
            and expected_action_id is not None
            and self.action_id != expected_action_id
        ):
            raise FreshnessRejected(
                f"Receipt action identity {self.action_id!r} does not match "
                f"{expected_action_id!r}."
            )
        for domain, admitted_revision in sorted(self.depends_on_revisions.items()):
            current_revision = current.domain_revision(domain)
            if current_revision != admitted_revision:
                raise FreshnessRejected(
                    f"Stale {domain!r} revision {admitted_revision}; current is "
                    f"{current_revision}."
                )
        return FreshnessDecision(
            lifecycle=self.lifecycle,
            checked_domains=tuple(sorted(self.depends_on_revisions)),
            method=self.method,
            reason=self.reason,
            confidence=self.confidence,
        )


class CameraSnapshot(StrictPhysicalModel):
    camera_id: NonEmptyStr
    frame_id: NonEmptyStr
    revision: int = Field(ge=0)
    rgb_ref: NonEmptyStr | None = None
    depth_ref: NonEmptyStr | None = None
    intrinsics_row_major: tuple[FiniteFloat, ...] = Field(min_length=9, max_length=9)
    world_T_camera: Transform  # noqa: N815 - explicit transform direction

    @model_validator(mode="after")
    def _camera_identity(self) -> CameraSnapshot:
        if self.world_T_camera.child_frame != self.frame_id:
            raise ValueError("world_T_camera child_frame must equal camera frame_id.")
        if self.rgb_ref is None and self.depth_ref is None:
            raise ValueError("CameraSnapshot requires at least one RGB/depth artifact ref.")
        return self


class ObservationSnapshot(StrictPhysicalModel):
    observation_id: NonEmptyStr
    captured_at_s: FiniteFloat = Field(ge=0.0)
    revisions: RevisionVector
    cameras: tuple[CameraSnapshot, ...] = Field(min_length=1)
    joint_names: tuple[NonEmptyStr, ...] = ()
    joint_positions_rad: tuple[FiniteFloat, ...] = ()
    gripper_width_m: FiniteFloat | None = Field(default=None, ge=0.0)
    method: NonEmptyStr
    reason: NonEmptyStr
    confidence: Confidence

    @model_validator(mode="after")
    def _coherent_snapshot(self) -> ObservationSnapshot:
        camera_ids = [camera.camera_id for camera in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("ObservationSnapshot camera ids must be unique.")
        for camera in self.cameras:
            if self.revisions.camera.get(camera.camera_id) != camera.revision:
                raise ValueError(
                    f"Camera revision mismatch for {camera.camera_id!r}."
                )
            if camera.world_T_camera.observation_id not in {None, self.observation_id}:
                raise ValueError("Camera transform belongs to another observation.")
        if len(self.joint_names) != len(self.joint_positions_rad):
            raise ValueError("joint_names and joint_positions_rad lengths must match.")
        return self


class OrientedBoundingBox(StrictPhysicalModel):
    center: Point3
    full_extents_m: Vector3
    rotation_world_from_obb: Matrix3
    extent_convention: Literal["full_length"] = "full_length"
    fit_residual_m: FiniteFloat = Field(ge=0.0)

    @field_validator("full_extents_m")
    @classmethod
    def _positive_extents(cls, value: Vector3) -> Vector3:
        if any(component <= 0.0 for component in value):
            raise ValueError("OBB full extents must be strictly positive.")
        return value

    @field_validator("rotation_world_from_obb")
    @classmethod
    def _proper_rotation(cls, value: Matrix3) -> Matrix3:
        rows = value
        for index, row in enumerate(rows):
            _require_unit(row, label=f"OBB rotation row {index}", tolerance=1e-5)
        for left in range(3):
            for right in range(left + 1, 3):
                dot = sum(rows[left][axis] * rows[right][axis] for axis in range(3))
                if not math.isclose(dot, 0.0, rel_tol=0.0, abs_tol=1e-5):
                    raise ValueError("OBB rotation rows must be orthogonal.")
        determinant = (
            rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
            - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
            + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
        )
        if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("OBB rotation must be right-handed with determinant +1.")
        return value


# Short domain name used throughout the architecture docs.
OBB = OrientedBoundingBox


class VisibleSupport(StrictPhysicalModel):
    points_artifact_id: NonEmptyStr
    inlier_fraction_of_observed_points: Confidence
    visible_azimuth_span_rad: FiniteFloat = Field(ge=0.0, le=2.0 * math.pi)


class ObjectGeometry(StrictPhysicalModel):
    entity_id: NonEmptyStr
    semantic_label: NonEmptyStr
    track_id: NonEmptyStr
    source_observation_id: NonEmptyStr
    obb: OrientedBoundingBox
    visible_support: VisibleSupport | None = None
    method: NonEmptyStr
    reason: NonEmptyStr
    confidence: Confidence
    validity: ValidityVector

    @model_validator(mode="before")
    @classmethod
    def _reject_flat_obb(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            legacy = {
                "center",
                "center_xyz_m",
                "extent",
                "extents",
                "full_extents_m",
                "quaternion",
                "quaternion_wxyz",
            }.intersection(value)
            if legacy:
                names = ", ".join(sorted(legacy))
                raise ValueError(
                    f"Flat OBB fields ({names}) are forbidden; use the nested 'obb' object."
                )
        return value

    @model_validator(mode="after")
    def _geometry_lineage(self) -> ObjectGeometry:
        if self.obb.center.observation_id != self.source_observation_id:
            raise ValueError("OBB center and geometry source observation must match.")
        if self.validity.lifecycle is not ValidityLifecycle.DERIVED:
            raise ValueError("ObjectGeometry validity lifecycle must be 'derived'.")
        if self.validity.observation_id != self.source_observation_id:
            raise ValueError("ObjectGeometry validity and source observation must match.")
        return self


class SupportRegion(StrictPhysicalModel):
    region_id: NonEmptyStr
    target_entity_id: NonEmptyStr
    source_observation_id: NonEmptyStr
    center: Point3
    normal: Direction3
    boundary: tuple[Point3, ...] = Field(min_length=3)
    clearance_m: FiniteFloat = Field(ge=0.0)
    method: NonEmptyStr
    reason: NonEmptyStr
    confidence: Confidence
    validity: ValidityVector

    @model_validator(mode="after")
    def _same_frame_and_observation(self) -> SupportRegion:
        values: tuple[Point3 | Direction3, ...] = (self.center, self.normal, *self.boundary)
        if any(value.frame_id != self.center.frame_id for value in values):
            raise ValueError("SupportRegion primitives must share one frame.")
        if any(value.observation_id != self.source_observation_id for value in values):
            raise ValueError("SupportRegion primitives must share the source observation.")
        if self.validity.lifecycle is not ValidityLifecycle.DERIVED:
            raise ValueError("SupportRegion validity lifecycle must be 'derived'.")
        if self.validity.observation_id != self.source_observation_id:
            raise ValueError("SupportRegion validity and source observation must match.")
        return self


class SelectionStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    EXACT = "exact"
    DEGRADED = "degraded"
    REJECTED = "rejected"


class AffordanceSelection(StrictPhysicalModel):
    status: SelectionStatus
    method: NonEmptyStr
    reason: NonEmptyStr
    confidence: Confidence


class Approach(StrictPhysicalModel):
    direction: Direction3
    distance_m: FiniteFloat = Field(gt=0.0)


class AffordanceAction(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    GRASP = "grasp"
    PLACE = "place"


class Affordance(StrictPhysicalModel):
    affordance_id: NonEmptyStr
    action_type: AffordanceAction
    object_entity_id: NonEmptyStr
    target_entity_id: NonEmptyStr | None = None
    target_pose: Pose
    approach: Approach
    strategy: NonEmptyStr
    source_geometry_id: NonEmptyStr
    target_geometry_id: NonEmptyStr | None = None
    desired_relation: Literal["supported_by", "inside", "on_top_of"] | None = None
    selection: AffordanceSelection
    predicted_tcp_T_object: Transform | None = None  # noqa: N815 - explicit direction
    validity: ValidityVector

    @model_validator(mode="after")
    def _action_specific_fields(self) -> Affordance:
        if self.target_pose.frame_id != self.approach.direction.frame_id:
            raise ValueError("Affordance pose and approach direction must share one frame.")
        if self.action_type is AffordanceAction.PLACE:
            missing = [
                name
                for name, value in (
                    ("target_entity_id", self.target_entity_id),
                    ("target_geometry_id", self.target_geometry_id),
                    ("desired_relation", self.desired_relation),
                )
                if value is None
            ]
            if missing:
                raise ValueError(f"Place affordance missing fields: {', '.join(missing)}.")
        elif any(
            value is not None
            for value in (
                self.target_entity_id,
                self.target_geometry_id,
                self.desired_relation,
            )
        ):
            raise ValueError("Grasp affordance cannot carry placement target fields.")
        if self.validity.lifecycle is not ValidityLifecycle.DERIVED:
            raise ValueError("Affordance validity lifecycle must be 'derived'.")
        return self


class CollisionObject(StrictPhysicalModel):
    entity_id: NonEmptyStr
    geometry_ref: NonEmptyStr
    geometry_digest: NonEmptyStr


class CollisionWorld(StrictPhysicalModel):
    world_id: NonEmptyStr
    revision: int = Field(ge=0)
    frame_id: NonEmptyStr
    source_observation_id: NonEmptyStr
    robot_config_revision: int = Field(ge=0)
    objects: tuple[CollisionObject, ...]
    method: NonEmptyStr
    reason: NonEmptyStr
    confidence: Confidence
    validity: ValidityVector

    @model_validator(mode="after")
    def _world_identity(self) -> CollisionWorld:
        identities = [item.entity_id for item in self.objects]
        if len(identities) != len(set(identities)):
            raise ValueError("CollisionWorld object entity ids must be unique.")
        if self.validity.lifecycle is not ValidityLifecycle.DERIVED:
            raise ValueError("CollisionWorld validity lifecycle must be 'derived'.")
        if self.validity.observation_id != self.source_observation_id:
            raise ValueError("CollisionWorld validity and source observation must match.")
        return self


class RelationPredicate(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    SUPPORTED_BY = "supported_by"
    INSIDE = "inside"
    ON_TOP_OF = "on_top_of"


class RelationValue(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    ASSERTED = "asserted"
    NEGATED = "negated"
    UNKNOWN = "unknown"


class _StateObservationEvidence(StrictPhysicalModel):
    """Causal envelope for evidence allowed to mutate embodied state.

    A digest proves that bytes are immutable, but not what those bytes prove.
    This envelope binds evidence to one observation and one state revision.
    Reducers additionally compare the envelope to the requested transition.
    """

    evidence_id: NonEmptyStr
    source_observation_id: NonEmptyStr
    source_observation_revision: int = Field(ge=0)
    observation_domain: NonEmptyStr
    base_state_revision: int = Field(ge=0)
    action_id: NonEmptyStr | None = None
    evidence_refs: tuple[NonEmptyStr, ...] = Field(min_length=1)
    method: NonEmptyStr
    reason: NonEmptyStr
    confidence: Confidence
    validity: ValidityVector

    @field_validator("source_observation_revision", "base_state_revision", mode="before")
    @classmethod
    def _reject_boolean_revision(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("Evidence revisions must be integers, not booleans.")
        return value

    @model_validator(mode="after")
    def _causal_identity(self) -> _StateObservationEvidence:
        if (
            not self.observation_domain.startswith("camera.")
            or not _DOMAIN_RE.fullmatch(self.observation_domain)
        ):
            raise ValueError("State evidence observation_domain must name a camera domain.")
        if self.validity.lifecycle is not ValidityLifecycle.DERIVED:
            raise ValueError("State evidence validity lifecycle must be 'derived'.")
        if self.validity.observation_id != self.source_observation_id:
            raise ValueError("State evidence validity and source observation must match.")
        if self.validity.state_revision != self.base_state_revision:
            raise ValueError("State evidence validity must bind its base state revision.")
        if (
            self.validity.depends_on_revisions.get(self.observation_domain)
            != self.source_observation_revision
        ):
            raise ValueError(
                "State evidence validity must bind its source observation revision."
            )
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("State evidence refs must be unique.")
        return self


class AttachmentEvidence(_StateObservationEvidence):
    """Verifier evidence for a determinate attachment predicate."""

    entity_id: NonEmptyStr
    entity_track_id: NonEmptyStr
    status: Literal["verified_held", "not_held"]

    @model_validator(mode="after")
    def _attachment_action(self) -> AttachmentEvidence:
        if self.action_id is None:
            raise ValueError("Attachment evidence must bind the causal action_id.")
        return self


class LocalizationEvidence(_StateObservationEvidence):
    """Verifier evidence for one entity localization belief."""

    entity_id: NonEmptyStr
    entity_track_id: NonEmptyStr
    status: Literal["unknown", "localized", "unlocalized", "ambiguous"]
    world_pose: Pose | None = None

    @model_validator(mode="after")
    def _localization_pose(self) -> LocalizationEvidence:
        if self.status == "localized":
            if self.world_pose is None:
                raise ValueError("Localized evidence requires a typed world_pose.")
            if self.world_pose.observation_id != self.source_observation_id:
                raise ValueError(
                    "Localization evidence pose and source observation must match."
                )
        elif self.world_pose is not None:
            raise ValueError("Non-localized evidence cannot retain a world_pose.")
        return self


class RelationEvidence(_StateObservationEvidence):
    """Verifier evidence for one fully identified physical relation."""

    subject_entity_id: NonEmptyStr
    subject_track_id: NonEmptyStr
    predicate: RelationPredicate
    target_entity_id: NonEmptyStr
    target_track_id: NonEmptyStr
    value: RelationValue

    @model_validator(mode="after")
    def _relation_identity(self) -> RelationEvidence:
        if self.subject_entity_id == self.target_entity_id:
            raise ValueError("A physical relation requires distinct subject and target entities.")
        return self


CORE_PHYSICAL_MODELS: dict[str, type[StrictPhysicalModel]] = {
    "robomex.observation_snapshot.v1": ObservationSnapshot,
    "robomex.object_geometry.v2": ObjectGeometry,
    "robomex.support_region.v1": SupportRegion,
    "robomex.affordance.v2": Affordance,
    "robomex.collision_world_snapshot.v1": CollisionWorld,
    "robomex.attachment_evidence.v1": AttachmentEvidence,
    "robomex.localization_evidence.v1": LocalizationEvidence,
    "robomex.relation_evidence.v1": RelationEvidence,
}


__all__ = [
    "AdmissionPurpose",
    "Affordance",
    "AffordanceAction",
    "AffordanceSelection",
    "Approach",
    "AttachmentEvidence",
    "CORE_PHYSICAL_MODELS",
    "CameraSnapshot",
    "CollisionObject",
    "CollisionWorld",
    "Direction3",
    "Direction3Stamped",
    "FreshnessDecision",
    "FreshnessRejected",
    "LocalizationEvidence",
    "OBB",
    "ObjectGeometry",
    "ObservationSnapshot",
    "OrientedBoundingBox",
    "PhysicalSchemaError",
    "Point3",
    "Point3Stamped",
    "Pose",
    "PoseStamped",
    "QualityEvidence",
    "RelationEvidence",
    "RelationPredicate",
    "RelationValue",
    "RevisionVector",
    "SelectionStatus",
    "StrictPhysicalModel",
    "SupportRegion",
    "Transform",
    "TransformStamped",
    "ValidityLifecycle",
    "ValidityVector",
    "VisibleSupport",
]
