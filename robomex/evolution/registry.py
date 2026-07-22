"""Registration and safety admission for externally produced candidates."""

from __future__ import annotations

import threading
from enum import Enum
from typing import Literal, TypeVar

from pydantic import model_validator

from robomex.contracts.common import SealedContract, revalidate_sealed
from robomex.evolution.candidate import (
    CandidateConfigSnapshot,
    EvolvableComponentSpec,
    PromotionStage,
    SafetyBoundary,
    is_promotion_transition_allowed,
)


class EvolutionRegistryError(RuntimeError):
    pass


class DuplicateEvolutionIdError(EvolutionRegistryError):
    pass


class EvolutionHashDriftError(EvolutionRegistryError):
    pass


class SafetyBoundaryViolationError(EvolutionRegistryError, PermissionError):
    pass


class EvolutionDisabledError(EvolutionRegistryError, PermissionError):
    pass


class CandidateAdmissionMode(str, Enum):  # noqa: UP042 - Python 3.10 support
    BASELINE_ONLY = "baseline_only"
    EXTERNAL_SNAPSHOTS = "external_snapshots"


class EvolvableRegistrySnapshot(SealedContract):
    schema_version: Literal["robomex.evolvable_registry.v1"] = (
        "robomex.evolvable_registry.v1"
    )
    admission_mode: CandidateAdmissionMode = CandidateAdmissionMode.BASELINE_ONLY
    safety_boundaries: tuple[SafetyBoundary, ...] = ()
    components: tuple[EvolvableComponentSpec, ...] = ()
    candidates: tuple[CandidateConfigSnapshot, ...] = ()

    @model_validator(mode="after")
    def _unique_snapshot_ids(self) -> EvolvableRegistrySnapshot:
        _unique(self.safety_boundaries, "boundary_id", "safety boundary IDs")
        _unique(self.components, "component_id", "evolvable component IDs")
        seen: set[tuple[str, int]] = set()
        for candidate in self.candidates:
            identity = (candidate.candidate_id, candidate.revision)
            if identity in seen:
                raise ValueError(
                    f"Duplicate candidate version {identity[0]!r}@{identity[1]}."
                )
            seen.add(identity)
        return self


def _unique(values: tuple[object, ...], attribute: str, label: str) -> None:
    identities = [getattr(value, attribute) for value in values]
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label}: {', '.join(duplicates)}.")


RegisteredT = TypeVar(
    "RegisteredT", SafetyBoundary, EvolvableComponentSpec, CandidateConfigSnapshot
)


class EvolvableComponentRegistry:
    """Evolve-ready catalog with immutable safety and no search algorithm.

    Default ``BASELINE_ONLY`` mode accepts only identity snapshots.  The
    ``EXTERNAL_SNAPSHOTS`` mode merely validates/registers candidates produced
    elsewhere; it still cannot mutate, rank, promote, or deploy anything.
    """

    def __init__(
        self,
        *,
        admission_mode: CandidateAdmissionMode = CandidateAdmissionMode.BASELINE_ONLY,
        snapshot: EvolvableRegistrySnapshot | None = None,
    ) -> None:
        self._mode = CandidateAdmissionMode(admission_mode)
        self._boundaries: dict[str, SafetyBoundary] = {}
        self._components: dict[str, EvolvableComponentSpec] = {}
        self._candidates: dict[tuple[str, int], CandidateConfigSnapshot] = {}
        self._candidate_digests: dict[str, CandidateConfigSnapshot] = {}
        self._lock = threading.RLock()
        if snapshot is not None:
            snapshot = revalidate_sealed(snapshot)
            if snapshot.admission_mode is not self._mode:
                raise EvolutionRegistryError(
                    "Snapshot admission mode does not match registry construction."
                )
            for boundary in snapshot.safety_boundaries:
                self._ensure_boundary(boundary)
            for component in snapshot.components:
                self._ensure_component(component)
            for candidate in snapshot.candidates:
                self._ensure_candidate(candidate)

    @property
    def admission_mode(self) -> CandidateAdmissionMode:
        return self._mode

    def register_safety_boundary(self, boundary: SafetyBoundary) -> SafetyBoundary:
        with self._lock:
            boundary = revalidate_sealed(boundary)
            existing = self._boundaries.get(boundary.boundary_id)
            if existing is not None:
                self._raise_duplicate_or_drift(
                    "Safety boundary", existing, boundary, boundary.boundary_id
                )
            self._boundaries[boundary.boundary_id] = boundary
            return boundary

    def register_component(self, component: EvolvableComponentSpec) -> EvolvableComponentSpec:
        with self._lock:
            component = revalidate_sealed(component)
            self._validate_component(component)
            existing = self._components.get(component.component_id)
            if existing is not None:
                self._raise_duplicate_or_drift(
                    "Evolvable component", existing, component, component.component_id
                )
            self._components[component.component_id] = component
            return component

    def validate_candidate(self, candidate: CandidateConfigSnapshot) -> CandidateConfigSnapshot:
        """Validate content, complete registry coverage, and all safety pins."""

        with self._lock:
            candidate = revalidate_sealed(candidate)
            registered_ids = set(self._components)
            candidate_ids = {item.component_id for item in candidate.components}
            if candidate_ids != registered_ids:
                missing = sorted(registered_ids - candidate_ids)
                unknown = sorted(candidate_ids - registered_ids)
                details: list[str] = []
                if missing:
                    details.append("missing=" + ",".join(missing))
                if unknown:
                    details.append("unknown=" + ",".join(unknown))
                raise EvolutionRegistryError(
                    "Candidate must snapshot every registered component exactly once ("
                    + "; ".join(details)
                    + ")."
                )

            required_boundaries: set[str] = set()
            changed = False
            for item in candidate.components:
                spec = self._components[item.component_id]
                required_boundaries.update(spec.safety_boundary_ids)
                if item.base_content_digest != spec.base_content_digest:
                    raise EvolutionHashDriftError(
                        f"Candidate base for {item.component_id!r} does not match registry."
                    )
                if item.candidate_content_digest != item.base_content_digest:
                    changed = True
                for path in item.changed_paths:
                    if not any(_path_within(path, allowed) for allowed in spec.mutable_paths):
                        raise SafetyBoundaryViolationError(
                            f"Component {item.component_id!r} changes undeclared path {path!r}."
                        )
                    for boundary_id in spec.safety_boundary_ids:
                        boundary = self._boundaries[boundary_id]
                        if any(
                            _paths_overlap(path, protected)
                            for protected in boundary.protected_paths
                        ):
                            raise SafetyBoundaryViolationError(
                                f"Component {item.component_id!r} change {path!r} overlaps "
                                f"immutable safety boundary {boundary_id!r}."
                            )

            supplied_pins = {pin.component_id: pin for pin in candidate.safety_boundary_pins}
            if set(supplied_pins) != required_boundaries:
                raise SafetyBoundaryViolationError(
                    "Candidate safety-boundary pins do not exactly match component obligations."
                )
            for boundary_id in required_boundaries:
                boundary = self._boundaries[boundary_id]
                pin = supplied_pins[boundary_id]
                if pin.revision != boundary.revision:
                    raise SafetyBoundaryViolationError(
                        f"Safety boundary {boundary_id!r} revision pin mismatch."
                    )
                if pin.content_digest != boundary.content_digest:
                    raise SafetyBoundaryViolationError(
                        f"Safety boundary {boundary_id!r} digest pin mismatch."
                    )

            if self._mode is CandidateAdmissionMode.BASELINE_ONLY and (
                changed or candidate.stage is not PromotionStage.BASELINE
            ):
                raise EvolutionDisabledError(
                    "This registry is baseline-only; candidate mutation/promotion is disabled."
                )
            return candidate

    def register_candidate(
        self, candidate: CandidateConfigSnapshot
    ) -> CandidateConfigSnapshot:
        with self._lock:
            candidate = self.validate_candidate(candidate)
            key = (candidate.candidate_id, candidate.revision)
            existing = self._candidates.get(key)
            if existing is not None:
                self._raise_duplicate_or_drift(
                    "Candidate", existing, candidate, f"{key[0]}@{key[1]}"
                )
            if candidate.content_digest in self._candidate_digests:
                prior = self._candidate_digests[candidate.content_digest]
                raise DuplicateEvolutionIdError(
                    f"Candidate digest is already bound to {prior.candidate_id!r}@"
                    f"{prior.revision}."
                )
            if candidate.parent_config_digest is not None:
                parent = self._candidate_digests.get(candidate.parent_config_digest)
                if parent is None:
                    raise EvolutionRegistryError(
                        "Candidate parent_config_digest is not registered."
                    )
                self._validate_lineage(parent, candidate)
            elif candidate.revision != 1:
                raise EvolutionRegistryError(
                    "A root baseline candidate must begin at revision 1."
                )
            self._candidates[key] = candidate
            self._candidate_digests[candidate.content_digest] = candidate
            return candidate

    def snapshot(self) -> EvolvableRegistrySnapshot:
        with self._lock:
            return EvolvableRegistrySnapshot(
                admission_mode=self._mode,
                safety_boundaries=tuple(
                    self._boundaries[key] for key in sorted(self._boundaries)
                ),
                components=tuple(
                    self._components[key] for key in sorted(self._components)
                ),
                candidates=tuple(
                    self._candidates[key] for key in sorted(self._candidates)
                ),
            )

    def _validate_component(self, component: EvolvableComponentSpec) -> None:
        for boundary_id in component.safety_boundary_ids:
            boundary = self._boundaries.get(boundary_id)
            if boundary is None:
                raise EvolutionRegistryError(
                    f"Component {component.component_id!r} references unknown safety "
                    f"boundary {boundary_id!r}."
                )
            for mutable in component.mutable_paths:
                if any(
                    _paths_overlap(mutable, protected)
                    for protected in boundary.protected_paths
                ):
                    raise SafetyBoundaryViolationError(
                        f"Mutable path {mutable!r} overlaps safety boundary "
                        f"{boundary_id!r}."
                    )

    def _ensure_boundary(self, boundary: SafetyBoundary) -> SafetyBoundary:
        boundary = revalidate_sealed(boundary)
        existing = self._boundaries.get(boundary.boundary_id)
        if existing is not None:
            if existing.content_digest != boundary.content_digest:
                raise EvolutionHashDriftError(
                    f"Safety boundary {boundary.boundary_id!r} has hash drift."
                )
            return existing
        self._boundaries[boundary.boundary_id] = boundary
        return boundary

    def _ensure_component(
        self, component: EvolvableComponentSpec
    ) -> EvolvableComponentSpec:
        component = revalidate_sealed(component)
        self._validate_component(component)
        existing = self._components.get(component.component_id)
        if existing is not None:
            if existing.content_digest != component.content_digest:
                raise EvolutionHashDriftError(
                    f"Component {component.component_id!r} has hash drift."
                )
            return existing
        self._components[component.component_id] = component
        return component

    def _ensure_candidate(
        self, candidate: CandidateConfigSnapshot
    ) -> CandidateConfigSnapshot:
        candidate = self.validate_candidate(candidate)
        key = (candidate.candidate_id, candidate.revision)
        existing = self._candidates.get(key)
        if existing is not None:
            if existing.content_digest != candidate.content_digest:
                raise EvolutionHashDriftError(
                    f"Candidate {key[0]!r}@{key[1]} has hash drift."
                )
            return existing
        if candidate.parent_config_digest is not None:
            parent = self._candidate_digests.get(candidate.parent_config_digest)
            if parent is None:
                raise EvolutionRegistryError("Snapshot candidate parent is missing.")
            self._validate_lineage(parent, candidate)
        elif candidate.revision != 1:
            raise EvolutionRegistryError(
                "A root baseline candidate must begin at revision 1."
            )
        self._candidates[key] = candidate
        self._candidate_digests[candidate.content_digest] = candidate
        return candidate

    @staticmethod
    def _validate_lineage(
        parent: CandidateConfigSnapshot,
        candidate: CandidateConfigSnapshot,
    ) -> None:
        same_identity = parent.candidate_id == candidate.candidate_id
        if same_identity:
            if candidate.revision != parent.revision + 1:
                raise EvolutionRegistryError(
                    "Candidate revisions must advance their parent by exactly one."
                )
        elif not (
            parent.stage is PromotionStage.BASELINE
            and candidate.stage is PromotionStage.DRAFT
            and candidate.revision == 1
        ):
            raise EvolutionRegistryError(
                "A new candidate identity may branch only as draft revision 1 "
                "from a baseline snapshot."
            )

        draft_refinement = (
            parent.stage is PromotionStage.DRAFT
            and candidate.stage is PromotionStage.DRAFT
            and same_identity
        )
        if not draft_refinement and not is_promotion_transition_allowed(
            parent.stage, candidate.stage
        ):
            raise EvolutionRegistryError(
                f"Illegal candidate stage transition {parent.stage.value!r} -> "
                f"{candidate.stage.value!r}."
            )
        if not draft_refinement and parent.stage is not PromotionStage.BASELINE:
            if candidate.components != parent.components:
                raise EvolutionRegistryError(
                    "Component content is frozen after draft evaluation begins."
                )
            if candidate.safety_boundary_pins != parent.safety_boundary_pins:
                raise SafetyBoundaryViolationError(
                    "Safety-boundary pins cannot change during promotion."
                )

    @staticmethod
    def _raise_duplicate_or_drift(
        label: str,
        existing: RegisteredT,
        incoming: RegisteredT,
        identity: str,
    ) -> None:
        if existing.content_digest != incoming.content_digest:
            raise EvolutionHashDriftError(f"{label} {identity!r} has hash drift.")
        raise DuplicateEvolutionIdError(f"Duplicate {label.lower()} ID {identity!r}.")


def _path_within(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent + "/")


def _paths_overlap(left: str, right: str) -> bool:
    return _path_within(left, right) or _path_within(right, left)


__all__ = [
    "CandidateAdmissionMode",
    "DuplicateEvolutionIdError",
    "EvolvableComponentRegistry",
    "EvolvableRegistrySnapshot",
    "EvolutionDisabledError",
    "EvolutionHashDriftError",
    "EvolutionRegistryError",
    "SafetyBoundaryViolationError",
]
