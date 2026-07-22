"""Production deterministic actors for the bowl-on-plate protocol.

The provider owns no action backend and cannot author or execute a physical
command.  It turns read-only :class:`ObservationRegistry` samples and admitted
artifacts into strict evidence, gates, and checkpoints.  Motion plans and state
transition proposals remain coding-worker outputs; authoritative execution and
state commits remain runtime-owned lanes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from robomex.data import (
    AdmissionFreshnessContext,
    AdmissionPurpose,
    AttachmentEvidence,
    AttachmentStatus,
    EpisodeDataPlane,
    RelationEvidence,
    RelationPredicate,
    RelationValue,
    ResolvedArtifactRef,
    StateCommitReceipt,
    ValidityLifecycle,
    ValidityVector,
)
from robomex.elastic import (
    EffectScope,
    ElasticGraphSpec,
    LifecycleScope,
    RunnerKind,
)
from robomex.manipulation import (
    AlignmentError,
    AlignmentStatus,
    AlignmentTolerance,
    AttachmentGuard,
    AttachmentGuardPhase,
    BowlPlaceObservation,
    CheckpointPhase,
    CheckpointStatus,
    CorrectionLimits,
    HeldBowlEstimate,
    PhaseCheckpoint,
    PlacementVerdict,
    PlacementVerdictStatus,
    PlateSupportTarget,
    PoseUncertainty,
    QuaternionWXYZ,
    RelationAssessment,
    ServoDecision,
    SupportFootprint,
    UnitVector3,
    Vector3,
    VisibilityStatus,
    VisualServoPlacer,
    compute_alignment_error,
    revision_vector_from_observation,
)
from robomex.orchestration.actors import (
    ActorIsolation,
    ActorLifecycle,
    ActorProfile,
    InvocationSpec,
)
from robomex.orchestration.episode import ActivationExecutionResult, ArtifactEmission
from robomex.runtime.action_protocol import (
    ActionSpecType,
    AdmissionSnapshot,
    ControllerState,
    ExecutionReceipt,
    ExecutionStatus,
    GripperCommand,
    WaitSpec,
    WorldKind,
)
from robomex.runtime.events import ControlOutcome
from robomex.runtime.observation import (
    EntityIdentity,
    ObservationContractError,
    ObservationQuality,
    ObservationRegistry,
    ObservationSample,
    TrackHandle,
    TrackNotFoundError,
    TrackRequest,
    TrackState,
    TrackStateError,
)

_EXECUTION_RECEIPT = "robomex.execution_receipt.v2"
_ATTACHMENT_EVIDENCE = "robomex.attachment_evidence.v1"
_RELATION_EVIDENCE = "robomex.relation_evidence.v1"
_CAPTURE_LEDGER_SCHEMA = "robomex.bowl_capture_attempt_ledger.v1"
_MAX_CAPTURE_LEDGER_BYTES = 16 * 1024 * 1024

_CAPTURE_RUNNER = "robomex.bowl_place.capture_synchronized_checkpoint"
_ATTACHMENT_RUNNER = "robomex.bowl_place.verify_attachment_evidence"
_SERVO_GATE_RUNNER = "robomex.bowl_place.gate_bounded_correction"
_OPEN_COMMAND_RUNNER = "robomex.bowl_place.build_open_command"
_SETTLE_WAIT_RUNNER = "robomex.bowl_place.build_settle_wait"
_PRE_RELEASE_RUNNER = "robomex.bowl_place.pre_release_checkpoint"
_POST_RELEASE_RUNNER = "robomex.bowl_place.post_release_checkpoint"
_FINAL_VERIFY_RUNNER = "robomex.bowl_place.verify_final_support_relation"
_COMPLETE_RUNNER = "robomex.bowl_place.complete"
_RECOVERY_RUNNER = "robomex.bowl_place.closed_recovery_frontier"
_BOWL_TRACKER_RUNNER = "robomex.bowl_place.track_held_bowl"
_PLATE_TRACKER_RUNNER = "robomex.bowl_place.track_plate"

BOWL_PLACE_DETERMINISTIC_RUNNERS = frozenset(
    {
        _CAPTURE_RUNNER,
        _ATTACHMENT_RUNNER,
        _SERVO_GATE_RUNNER,
        _OPEN_COMMAND_RUNNER,
        _SETTLE_WAIT_RUNNER,
        _PRE_RELEASE_RUNNER,
        _POST_RELEASE_RUNNER,
        _FINAL_VERIFY_RUNNER,
        _COMPLETE_RUNNER,
        _RECOVERY_RUNNER,
        _BOWL_TRACKER_RUNNER,
        _PLATE_TRACKER_RUNNER,
    }
)

_BOWL_PLACE_CODING_SKILLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "robomex.bowl_place.author_attachment_monitor": (
        "attachment_monitor_authoring",
        ("author_attachment_monitor",),
    ),
    "robomex.bowl_place.propose_attachment_transition": (
        "attachment_state_proposal",
        ("propose_attachment_transition",),
    ),
    "robomex.bowl_place.plan_transport_to_hover": (
        "transport_motion_planning",
        ("author_sealed_phase_motion",),
    ),
    "robomex.bowl_place.estimate_support_alignment": (
        "support_alignment_estimation",
        ("estimate_support_alignment",),
    ),
    "robomex.bowl_place.plan_bounded_correction": (
        "bounded_correction_planning",
        ("author_sealed_phase_motion",),
    ),
    "robomex.bowl_place.plan_bounded_descend": (
        "bounded_descend_planning",
        ("author_sealed_phase_motion",),
    ),
    "robomex.bowl_place.plan_safe_retreat": (
        "safe_retreat_planning",
        ("author_sealed_phase_motion",),
    ),
    "robomex.bowl_place.propose_relation_transition": (
        "relation_state_proposal",
        ("propose_relation_transition",),
    ),
}


class BowlPlaceProviderError(RuntimeError):
    """A deterministic bowl actor received an unsafe or inconsistent request."""


class BowlTrackingMode(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """How the bowl provider obtains tracking evidence."""

    CONTINUOUS = "continuous"
    MANUAL_CHECKPOINT = "manual_checkpoint"


@dataclass(frozen=True)
class SynchronizedTrackPair:
    """One bowl/plate pair proven to share the same physical revision vector."""

    pair_sequence: int
    bowl: ObservationSample
    plate: ObservationSample

    def __post_init__(self) -> None:
        if (
            isinstance(self.pair_sequence, bool)
            or not isinstance(self.pair_sequence, int)
            or self.pair_sequence < 1
        ):
            raise ValueError("pair_sequence must be positive")
        if self.bowl.revisions != self.plate.revisions:
            raise ValueError("synchronized samples must share one revision vector")
        if (
            self.bowl.episode_id != self.plate.episode_id
            or self.bowl.stream_id != self.plate.stream_id
        ):
            raise ValueError("synchronized samples must share one episode stream")
        if self.bowl.track_id == self.plate.track_id:
            raise ValueError("synchronized samples must come from distinct tracks")


class SynchronizedBowlTrackSampler:
    """Lifecycle-owned asynchronous sampler shared by both tracking services.

    The worker has read-only observation authority.  It never invokes an LLM,
    calls an action backend, or writes embodied state.  Backend silence gets a
    bounded grace period and then produces exactly one LOST sample until fresh
    evidence resumes.  Bowl/plate checkpoint pairs are matched by the complete
    revision vector, not by whichever two stream tails happen to be latest.
    """

    def __init__(
        self,
        *,
        bowl_track: TrackHandle,
        plate_track: TrackHandle,
        poll_interval_s: float,
        max_silence_s: float,
        max_pair_skew_s: float,
        pair_buffer_size: int,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if bowl_track.track_id == plate_track.track_id:
            raise ValueError("continuous sampler requires distinct track IDs")
        if bowl_track.stream is not plate_track.stream:
            raise ValueError("continuous sampler tracks must share one observation stream")
        for name, value in (
            ("poll_interval_s", poll_interval_s),
            ("max_silence_s", max_silence_s),
            ("max_pair_skew_s", max_pair_skew_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if poll_interval_s > max_silence_s:
            raise ValueError("poll_interval_s must not exceed max_silence_s")
        if (
            isinstance(pair_buffer_size, bool)
            or not isinstance(pair_buffer_size, int)
            or pair_buffer_size < 2
        ):
            raise ValueError("pair_buffer_size must be an integer >= 2")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self.bowl_track = bowl_track
        self.plate_track = plate_track
        self.poll_interval_s = float(poll_interval_s)
        self.max_silence_s = float(max_silence_s)
        self.max_pair_skew_s = float(max_pair_skew_s)
        self.pair_buffer_size = int(pair_buffer_size)
        self._clock = monotonic_clock
        self._track_owner_token = object()
        self._tracks_claimed = False
        self._lifecycle_lock = threading.RLock()
        self._sample_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._pair_ready = threading.Condition(self._state_lock)
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._owners: dict[str, bool] = {}
        self._thread: threading.Thread | None = None
        self._closed = False
        now = self._clock()
        self._last_source_time = {
            bowl_track.track_id: now,
            plate_track.track_id: now,
        }
        self._silence_published = {
            bowl_track.track_id: False,
            plate_track.track_id: False,
        }
        self._buffers: dict[str, dict[tuple[int, ...], ObservationSample]] = {
            bowl_track.track_id: {},
            plate_track.track_id: {},
        }
        self._latest_pair: SynchronizedTrackPair | None = None
        self._pair_sequence = 0
        self._dropped_unpaired_samples = 0
        self._poll_cycles = 0
        self._last_error: str | None = None

    @property
    def owner_count(self) -> int:
        with self._state_lock:
            return len(self._owners)

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def sampling_enabled(self) -> bool:
        with self._state_lock:
            return bool(self._owners) and not all(self._owners.values()) and not self._closed

    @property
    def latest_pair(self) -> SynchronizedTrackPair | None:
        with self._state_lock:
            return self._latest_pair

    @property
    def dropped_unpaired_samples(self) -> int:
        with self._state_lock:
            return self._dropped_unpaired_samples

    @property
    def poll_cycles(self) -> int:
        with self._state_lock:
            return self._poll_cycles

    @property
    def last_error(self) -> str | None:
        with self._state_lock:
            return self._last_error

    def acquire(self, owner_id: str) -> bool:
        """Acquire one service owner and start the single shared worker."""

        owner = str(owner_id).strip()
        if not owner:
            raise ValueError("sampler owner_id must not be empty")
        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed:
                    raise BowlPlaceProviderError("continuous sampler is closed")
                if owner in self._owners:
                    return False
            self._claim_tracks()
            self._resume_tracks()
            with self._state_lock:
                had_other_owners = bool(self._owners)
                self._owners[owner] = False
                now = self._clock()
                for track_id in self._last_source_time:
                    self._last_source_time[track_id] = now
                    self._silence_published[track_id] = False
                start_worker = self._thread is None or not self._thread.is_alive()
                if start_worker:
                    self._stop_event.clear()
                    self._thread = threading.Thread(
                        target=self._run,
                        name=(
                            "robomex-track-sampler-"
                            f"{self.bowl_track.track_id}-{self.plate_track.track_id}"
                        ),
                        daemon=True,
                    )
                    thread = self._thread
                else:
                    thread = None
            if thread is not None:
                try:
                    thread.start()
                except Exception as exc:
                    with self._state_lock:
                        self._owners.pop(owner, None)
                        if self._thread is thread:
                            self._thread = None
                        self._stop_event.set()
                    if not had_other_owners:
                        self._suspend_tracks()
                    raise BowlPlaceProviderError(
                        "continuous tracking worker could not be started"
                    ) from exc
            self._wake_event.set()
            return True

    def suspend(self, owner_id: str) -> bool:
        owner = str(owner_id).strip()
        with self._lifecycle_lock:
            with self._state_lock:
                if owner not in self._owners:
                    raise BowlPlaceProviderError("sampler owner is not acquired")
                if self._owners[owner]:
                    return False
                self._owners[owner] = True
                pause_all = all(self._owners.values())
            if pause_all:
                self._suspend_tracks()
            self._wake_event.set()
            return True

    def resume(self, owner_id: str) -> bool:
        owner = str(owner_id).strip()
        with self._lifecycle_lock:
            with self._state_lock:
                if owner not in self._owners:
                    raise BowlPlaceProviderError("sampler owner is not acquired")
                if not self._owners[owner]:
                    return False
            self._resume_tracks()
            with self._state_lock:
                self._owners[owner] = False
                now = self._clock()
                for track_id in self._last_source_time:
                    self._last_source_time[track_id] = now
                    self._silence_published[track_id] = False
            self._wake_event.set()
            return True

    def release(self, owner_id: str) -> bool:
        """Release an owner; the last release stops (not leaks) the worker."""

        owner = str(owner_id).strip()
        with self._lifecycle_lock:
            with self._state_lock:
                if owner not in self._owners:
                    return False
                del self._owners[owner]
                stop_worker = not self._owners
                pause_all = bool(self._owners) and all(self._owners.values())
                thread = self._thread if stop_worker else None
                if stop_worker:
                    self._stop_event.set()
                    self._wake_event.set()
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(1.0, self.poll_interval_s * 4.0))
                if thread.is_alive():
                    raise BowlPlaceProviderError(
                        "continuous tracking worker did not stop within its bound"
                    )
            if stop_worker:
                self._suspend_tracks()
                with self._state_lock:
                    if self._thread is thread:
                        self._thread = None
            elif pause_all:
                self._suspend_tracks()
            return True

    def close(self) -> bool:
        """Permanently stop the sampler and its track handles."""

        with self._lifecycle_lock:
            with self._state_lock:
                if self._closed:
                    return False
                self._closed = True
                self._owners.clear()
                self._stop_event.set()
                self._wake_event.set()
                thread = self._thread
                self._pair_ready.notify_all()
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(1.0, self.poll_interval_s * 4.0))
                if thread.is_alive():
                    raise BowlPlaceProviderError(
                        "continuous tracking worker did not close within its bound"
                    )
            failures: list[str] = []
            with self._sample_lock:
                for handle in (self.bowl_track, self.plate_track):
                    try:
                        if handle.state is not TrackState.STOPPED:
                            handle.stop()
                    except Exception as exc:
                        failures.append(f"{handle.track_id}:stop:{type(exc).__name__}")
                    try:
                        handle._release_continuous_sampler(self._track_owner_token)
                    except Exception as exc:
                        failures.append(f"{handle.track_id}:release:{type(exc).__name__}")
                self._tracks_claimed = False
            with self._state_lock:
                self._thread = None
            if failures:
                raise BowlPlaceProviderError(
                    "continuous tracking close failed: " + ", ".join(failures)
                )
            return True

    def wait_for_synchronized_pair(
        self,
        *,
        after_camera_revision: int,
        timeout_s: float,
    ) -> SynchronizedTrackPair | None:
        """Wait for a coherent pair newer than a durable checkpoint floor."""

        if after_camera_revision < 0:
            raise ValueError("after_camera_revision must be non-negative")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be finite and positive")
        deadline = time.monotonic() + timeout_s
        with self._pair_ready:
            while True:
                pair = self._latest_pair
                if pair is not None and pair.bowl.revisions.camera_revision > after_camera_revision:
                    return pair
                if self._closed:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._pair_ready.wait(timeout=remaining)

    def sample_once(self) -> None:
        """Perform one bounded read-only sampling cycle (also useful in tests)."""

        with self._sample_lock:
            samples = (
                self._poll_track(self.bowl_track),
                self._poll_track(self.plate_track),
            )
        for sample in samples:
            if sample is not None:
                self._buffer_for_pair(sample)
        with self._state_lock:
            self._poll_cycles += 1

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self.sampling_enabled:
                self._wake_event.wait(timeout=self.poll_interval_s)
                self._wake_event.clear()
                continue
            try:
                self.sample_once()
            except Exception as exc:  # unexpected worker failures are observable
                self._record_error(f"sampler_error:{type(exc).__name__}")
                self._publish_fail_closed_all(
                    reason=f"continuous_sampler_error:{type(exc).__name__}"
                )
            self._stop_event.wait(self.poll_interval_s)

    def _poll_track(self, handle: TrackHandle) -> ObservationSample | None:
        if handle.state is TrackState.STOPPED:
            self._record_error(f"track_stopped:{handle.track_id}")
            self._stop_event.set()
            return None
        try:
            sample = handle.poll_if_available(_continuous_sampler_token=self._track_owner_token)
        except (ObservationContractError, TrackStateError) as exc:
            self._record_error(f"{handle.track_id}:{type(exc).__name__}")
            return self._publish_lost(
                handle,
                reason=f"continuous_tracker_error:{type(exc).__name__}",
            )
        if sample is not None:
            with self._state_lock:
                self._last_source_time[handle.track_id] = self._clock()
                self._silence_published[handle.track_id] = False
            return sample
        now = self._clock()
        with self._state_lock:
            silent_for = now - self._last_source_time[handle.track_id]
            already_published = self._silence_published[handle.track_id]
        if silent_for < self.max_silence_s or already_published:
            return None
        sample = self._publish_lost(handle, reason="continuous_tracking_silence_exceeded")
        if sample is not None:
            with self._state_lock:
                self._silence_published[handle.track_id] = True
        return sample

    def _publish_lost(
        self,
        handle: TrackHandle,
        *,
        reason: str,
    ) -> ObservationSample | None:
        try:
            return handle.publish_lost(
                reason=reason,
                _continuous_sampler_token=self._track_owner_token,
            )
        except (ObservationContractError, TrackStateError) as exc:
            self._record_error(f"{handle.track_id}:lost_publish:{type(exc).__name__}")
            return None

    def _publish_fail_closed_all(self, *, reason: str) -> None:
        with self._sample_lock:
            for handle in (self.bowl_track, self.plate_track):
                if handle.state is TrackState.ACTIVE:
                    self._publish_lost(handle, reason=reason)

    def _buffer_for_pair(self, sample: ObservationSample) -> None:
        track_id = sample.track_id
        if track_id not in self._buffers:
            self._record_error(f"unexpected_track:{track_id}")
            return
        other_id = (
            self.plate_track.track_id
            if track_id == self.bowl_track.track_id
            else self.bowl_track.track_id
        )
        key = sample.revisions.as_tuple()
        with self._pair_ready:
            own = self._buffers[track_id]
            if key in own:
                self._dropped_unpaired_samples += 1
            own[key] = sample
            while len(own) > self.pair_buffer_size:
                own.pop(next(iter(own)))
                self._dropped_unpaired_samples += 1
            other = self._buffers[other_id].get(key)
            if other is None:
                return
            del own[key]
            del self._buffers[other_id][key]
            bowl = sample if track_id == self.bowl_track.track_id else other
            plate = sample if track_id == self.plate_track.track_id else other
            monotonic_skew = abs(bowl.monotonic_time_s - plate.monotonic_time_s)
            wall_skew = abs((bowl.observed_at - plate.observed_at).total_seconds())
            if max(monotonic_skew, wall_skew) > self.max_pair_skew_s:
                self._dropped_unpaired_samples += 2
                self._last_error = "synchronized_pair_timestamp_skew"
                return
            self._pair_sequence += 1
            self._latest_pair = SynchronizedTrackPair(
                pair_sequence=self._pair_sequence,
                bowl=bowl,
                plate=plate,
            )
            self._pair_ready.notify_all()

    def _resume_tracks(self) -> None:
        with self._sample_lock:
            for handle in (self.bowl_track, self.plate_track):
                if handle.state is TrackState.NEW:
                    handle.start()
                elif handle.state is TrackState.SUSPENDED:
                    handle.resume()
                elif handle.state is TrackState.STOPPED:
                    raise BowlPlaceProviderError(
                        f"tracking handle {handle.track_id!r} was permanently stopped"
                    )

    def _claim_tracks(self) -> None:
        with self._sample_lock:
            if self._tracks_claimed:
                return
            claimed: list[TrackHandle] = []
            try:
                for handle in (self.bowl_track, self.plate_track):
                    handle._claim_continuous_sampler(self._track_owner_token)
                    claimed.append(handle)
            except Exception:
                for handle in reversed(claimed):
                    handle._release_continuous_sampler(self._track_owner_token)
                raise
            self._tracks_claimed = True

    def _suspend_tracks(self) -> None:
        with self._sample_lock:
            for handle in (self.bowl_track, self.plate_track):
                if handle.state is TrackState.ACTIVE:
                    handle.suspend()

    def _record_error(self, value: str) -> None:
        with self._state_lock:
            self._last_error = value


@dataclass(frozen=True)
class BowlPlaceProviderConfig:
    bowl_entity_id: str = "bowl-1"
    plate_entity_id: str = "plate-1"
    bowl_track_id: str = "track-bowl-1"
    plate_track_id: str = "track-plate-1"
    observation_backend_id: str = "bowl-observation-backend"
    camera_id: str = "front"
    frame_id: str = "world"
    authority_world_id: str = "authoritative"
    gripper_resource_id: str = "robot.gripper"
    controller_resource_id: str = "robot.controller"
    bowl_radius_m: float = 0.035
    plate_radius_m: float = 0.09
    plate_safe_margin_m: float = 0.01
    gripper_open_width_m: float = 0.08
    gripper_open_timeout_s: float = 2.0
    settle_duration_s: float = 0.5
    settle_timeout_s: float = 2.0
    tracking_mode: BowlTrackingMode = BowlTrackingMode.CONTINUOUS
    tracking_poll_interval_s: float = 0.02
    tracking_max_silence_s: float = 0.25
    tracking_capture_wait_timeout_s: float = 1.0
    tracking_max_pair_skew_s: float = 0.05
    tracking_pair_buffer_size: int = 32
    tolerance: AlignmentTolerance = field(default_factory=AlignmentTolerance)
    correction_limits: CorrectionLimits = field(default_factory=CorrectionLimits)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracking_mode", BowlTrackingMode(self.tracking_mode))
        identities = (
            self.bowl_entity_id,
            self.plate_entity_id,
            self.bowl_track_id,
            self.plate_track_id,
            self.observation_backend_id,
            self.camera_id,
            self.frame_id,
            self.authority_world_id,
            self.gripper_resource_id,
            self.controller_resource_id,
        )
        if any(not value.strip() for value in identities):
            raise ValueError("bowl provider identity fields must not be empty")
        if self.bowl_entity_id == self.plate_entity_id:
            raise ValueError("bowl and plate entity identities must differ")
        if self.bowl_radius_m <= 0 or self.plate_radius_m <= 0:
            raise ValueError("support radii must be positive")
        if self.plate_safe_margin_m < 0:
            raise ValueError("plate_safe_margin_m must be non-negative")
        if self.gripper_open_width_m < 0:
            raise ValueError("gripper_open_width_m must be non-negative")
        if self.gripper_open_timeout_s <= 0:
            raise ValueError("gripper_open_timeout_s must be positive")
        if self.settle_duration_s <= 0:
            raise ValueError("settle_duration_s must be positive")
        if self.settle_timeout_s < self.settle_duration_s:
            raise ValueError("settle_timeout_s must cover settle_duration_s")
        for name, value in (
            ("tracking_poll_interval_s", self.tracking_poll_interval_s),
            ("tracking_max_silence_s", self.tracking_max_silence_s),
            ("tracking_capture_wait_timeout_s", self.tracking_capture_wait_timeout_s),
            ("tracking_max_pair_skew_s", self.tracking_max_pair_skew_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.tracking_poll_interval_s > self.tracking_max_silence_s:
            raise ValueError("tracking_poll_interval_s must not exceed tracking_max_silence_s")
        if (
            isinstance(self.tracking_pair_buffer_size, bool)
            or not isinstance(self.tracking_pair_buffer_size, int)
            or self.tracking_pair_buffer_size < 2
        ):
            raise ValueError("tracking_pair_buffer_size must be an integer >= 2")


@dataclass
class _ActorRuntime:
    profile: ActorProfile
    isolation: ActorIsolation
    sampler_owner_id: str = field(default_factory=lambda: f"actor-{uuid.uuid4().hex}")
    tracking_acquired: bool = False
    suspended: bool = False
    retired: bool = False


class BowlPlaceActorProvider:
    """Read-only deterministic provider for physical evidence and gates."""

    BOWL_SIGNALS = (
        "attachment_status",
        "center_x",
        "center_y",
        "center_z",
        "identity_match",
        "visibility",
        "yaw",
    )
    PLATE_SIGNALS = (
        "center_x",
        "center_y",
        "center_z",
        "identity_match",
        "visibility",
        "yaw",
    )

    def __init__(
        self,
        config: BowlPlaceProviderConfig,
        *,
        sampler_factory: Callable[..., SynchronizedBowlTrackSampler] | None = None,
    ) -> None:
        self.config = config
        self._sampler_factory = sampler_factory or SynchronizedBowlTrackSampler
        self._episode: Any | None = None
        self._data_plane: EpisodeDataPlane | None = None
        self._observations: ObservationRegistry | None = None
        self._tracking_sampler: SynchronizedBowlTrackSampler | None = None
        self._latest_observation: BowlPlaceObservation | None = None
        self._durable_checkpoint_floor: BowlPlaceObservation | None = None
        self._capture_ledger_path: Path | None = None
        self._invocations: list[str] = []
        self._lock = threading.RLock()

    @property
    def invocations(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._invocations)

    @property
    def tracking_sampler(self) -> SynchronizedBowlTrackSampler | None:
        """Return the episode-owned sampler for health checks and shutdown."""

        with self._lock:
            return self._tracking_sampler

    def bind_episode_runtime(self, episode: Any) -> None:
        data_plane = getattr(episode, "data_plane", None)
        observations = getattr(episode, "observations", None)
        state_reducer = getattr(episode, "state_reducer", None)
        if not isinstance(data_plane, EpisodeDataPlane):
            raise TypeError("episode must expose an EpisodeDataPlane")
        if not isinstance(observations, ObservationRegistry):
            raise TypeError("episode must expose an ObservationRegistry")
        if state_reducer is None:
            raise TypeError("episode must expose an embodied-state reducer")
        with self._lock:
            if self._episode is not None and self._episode is not episode:
                raise BowlPlaceProviderError("provider is already bound to another episode")
            self._episode = episode
            self._data_plane = data_plane
            self._observations = observations
            self._capture_ledger_path = (
                Path(episode.episode_root) / "bowl_capture_attempts.v1.jsonl"
            )
            durable_observations = [
                record
                for record in data_plane.artifacts
                if record.schema == BowlPlaceObservation.model_fields["schema_version"].default
            ]
            if durable_observations:
                restored = BowlPlaceObservation.model_validate(
                    data_plane.resolve(durable_observations[-1].ref).payload
                )
                self._latest_observation = restored
                self._durable_checkpoint_floor = restored

    def bind_episode_data_plane(self, data_plane: EpisodeDataPlane) -> None:
        if not isinstance(data_plane, EpisodeDataPlane):
            raise TypeError("data_plane must be an EpisodeDataPlane")
        with self._lock:
            if self._data_plane is not None and self._data_plane is not data_plane:
                raise BowlPlaceProviderError("provider data-plane binding changed")
            self._data_plane = data_plane

    def freshness_context(self, purpose: AdmissionPurpose) -> AdmissionFreshnessContext:
        episode = self._require_episode()
        self._refresh_durable_checkpoint_floor()
        with self._lock:
            observation = self._latest_observation
        if observation is None:
            # No validity-bearing bowl artifact exists before the first capture.
            # This value is therefore only a closed bootstrap clock, never a
            # substitute for a checkpoint.
            from robomex.data import RevisionVector

            revisions = RevisionVector(
                scene=0,
                arm=0,
                gripper=0,
                attachment=0,
                camera={self.config.camera_id: 0},
            )
            observation_id = None
        else:
            revisions = observation.revisions
            observation_id = observation.snapshot_id
        return AdmissionFreshnessContext(
            current_revisions=revisions,
            purpose=purpose,
            current_observation_id=observation_id,
            current_state_revision=episode.state_reducer.state.revision,
        )

    def spawn(self, profile: ActorProfile, isolation: ActorIsolation) -> _ActorRuntime:
        if profile.effect_ceiling:
            raise BowlPlaceProviderError(
                "BowlPlaceActorProvider accepts only empty effect ceilings"
            )
        return _ActorRuntime(profile=profile, isolation=isolation)

    def invoke(
        self, runtime: _ActorRuntime, spec: InvocationSpec
    ) -> ActivationExecutionResult | None:
        if runtime.retired or runtime.suspended:
            raise BowlPlaceProviderError("bowl actor is not active")
        if spec.requested_effects:
            raise BowlPlaceProviderError("deterministic bowl actors cannot request effects")
        activation_id = str(spec.metadata.get("activation_id") or "")
        workflow_id = str(spec.metadata.get("workflow_id") or "")
        if not activation_id or not workflow_id:
            raise BowlPlaceProviderError("invocation lacks activation/workflow identity")
        with self._lock:
            self._invocations.append(activation_id)

        runner_ref = runtime.profile.metadata.get("runner_ref")
        if runner_ref is None:
            runner_ref = self._runner_from_activation(activation_id)
        if runner_ref in {_BOWL_TRACKER_RUNNER, _PLATE_TRACKER_RUNNER}:
            self._activate_tracking_service(runtime)
            return None
        replayed = self._replay_durable_result(
            spec,
            workflow_id=workflow_id,
            activation_id=activation_id,
            runner_ref=str(runner_ref),
        )
        if replayed is not None:
            return replayed
        if runner_ref == _CAPTURE_RUNNER:
            return self._capture(spec, workflow_id=workflow_id)
        if runner_ref == _ATTACHMENT_RUNNER:
            return self._verify_attachment(spec, activation_id=activation_id)
        if runner_ref == _SERVO_GATE_RUNNER:
            return self._alignment_gate(spec, workflow_id=workflow_id)
        if runner_ref == _OPEN_COMMAND_RUNNER:
            return self._build_open_command(spec)
        if runner_ref == _SETTLE_WAIT_RUNNER:
            return self._build_settle_wait(spec)
        if runner_ref == _PRE_RELEASE_RUNNER:
            return self._pre_release_checkpoint(spec)
        if runner_ref == _POST_RELEASE_RUNNER:
            return self._post_release_checkpoint(spec)
        if runner_ref == _FINAL_VERIFY_RUNNER:
            return self._verify_final_relation(spec)
        if runner_ref == _COMPLETE_RUNNER:
            return self._complete(spec)
        if runner_ref == _RECOVERY_RUNNER:
            return ActivationExecutionResult(
                outcome=ControlOutcome.FAILED,
                reason="closed recovery frontier reached",
            )
        raise BowlPlaceProviderError(f"unsupported deterministic runner {runner_ref!r}")

    def suspend(self, runtime: _ActorRuntime) -> None:
        if runtime.retired:
            raise BowlPlaceProviderError("retired actor cannot be suspended")
        if runtime.tracking_acquired:
            sampler = self._require_tracking_sampler()
            sampler.suspend(runtime.sampler_owner_id)
        runtime.suspended = True

    def resume(self, runtime: _ActorRuntime) -> None:
        if runtime.retired:
            raise BowlPlaceProviderError("retired actor cannot be resumed")
        if runtime.tracking_acquired:
            sampler = self._require_tracking_sampler()
            sampler.resume(runtime.sampler_owner_id)
        runtime.suspended = False

    def retire(self, runtime: _ActorRuntime) -> None:
        if runtime.tracking_acquired:
            sampler = self._require_tracking_sampler()
            sampler.release(runtime.sampler_owner_id)
            runtime.tracking_acquired = False
        runtime.retired = True

    def close(self) -> bool:
        """Explicit episode shutdown hook for non-``EpisodeRuntime`` hosts."""

        with self._lock:
            sampler = self._tracking_sampler
        if sampler is None:
            return False
        return sampler.close()

    def _replay_durable_result(
        self,
        spec: InvocationSpec,
        *,
        workflow_id: str,
        activation_id: str,
        runner_ref: str,
    ) -> ActivationExecutionResult | None:
        """Replay a completely published attempt before touching live sensors.

        Actor output publication and ``NodeOutcomeEvent`` persistence are two
        durable steps.  A restart between them must reuse the already-published
        value; in particular, a checkpoint capture must never poll a newer
        sensor sample and try to rebind the same activation attempt.
        """

        attempt = int(spec.metadata.get("attempt") or 0)
        if attempt < 1:
            raise BowlPlaceProviderError("invocation attempt must be positive")
        plane = self._require_data_plane()
        records = tuple(
            record
            for record in plane.artifacts
            if record.workflow_id == workflow_id
            and record.activation_id == activation_id
            and record.attempt == attempt
        )
        if not records:
            return None
        by_port = {record.port: record for record in records}
        if len(by_port) != len(records):
            raise BowlPlaceProviderError(
                "durable activation attempt contains duplicate output ports"
            )
        expected_ports = set(spec.output_contract)
        if set(by_port) != expected_ports:
            raise BowlPlaceProviderError(
                "durable activation attempt contains a partial or unexpected output set"
            )

        if runner_ref == _CAPTURE_RUNNER:
            record = by_port.get("observation")
            if (
                record is None
                or record.schema != BowlPlaceObservation.model_fields["schema_version"].default
            ):
                raise BowlPlaceProviderError("durable capture output has the wrong port or schema")
            expected_lineage = self._capture_lineage(spec, workflow_id=workflow_id)
            if record.lineage != expected_lineage:
                raise BowlPlaceProviderError(
                    "durable capture output lineage differs from admitted causal inputs"
                )

        emissions: list[ArtifactEmission] = []
        decision: Any | None = None
        for port in spec.output_contract:
            record = by_port[port]
            expected_schema = spec.output_contract[port]
            if record.schema != expected_schema:
                raise BowlPlaceProviderError(
                    f"durable output {port!r} has schema {record.schema!r}; "
                    f"expected {expected_schema!r}"
                )
            for ref in record.lineage:
                plane.resolve(ref)
            resolved = plane.resolve(record.ref)
            emissions.append(
                ArtifactEmission(
                    port=port,
                    schema_id=record.schema,
                    payload=resolved.payload,
                    lineage=record.lineage,
                )
            )
            if port == "observation":
                observation = BowlPlaceObservation.model_validate(resolved.payload)
                with self._lock:
                    self._latest_observation = observation
                    self._durable_checkpoint_floor = observation
            elif port == "servo_decision":
                decision = ServoDecision.model_validate(resolved.payload)
        if decision is not None:
            return ActivationExecutionResult(
                outcome=decision.control_outcome,
                artifacts=tuple(emissions),
                reason=decision.reason,
            )
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=tuple(emissions),
        )

    def _capture(self, spec: InvocationSpec, *, workflow_id: str) -> ActivationExecutionResult:
        self._refresh_durable_checkpoint_floor()
        activation_id = str(spec.metadata["activation_id"])
        attempt = int(spec.metadata["attempt"])
        if not self._reserve_capture_attempt(
            workflow_id=workflow_id,
            activation_id=activation_id,
            attempt=attempt,
            invocation_id=spec.invocation_id,
        ):
            return ActivationExecutionResult(
                outcome=ControlOutcome.STALE_INPUT,
                reason=(
                    "capture attempt was started before a crash but has no complete "
                    "durable observation; sensor polling is not retried"
                ),
            )
        bowl_track, plate_track = self._ensure_tracks()
        if self.config.tracking_mode is BowlTrackingMode.MANUAL_CHECKPOINT:
            bowl_sample = bowl_track.poll()
            plate_sample = plate_track.poll()
        else:
            sampler = self._ensure_tracking_sampler()
            if sampler.owner_count == 0 or not sampler.running:
                return ActivationExecutionResult(
                    outcome=ControlOutcome.STALE_OBSERVATION,
                    reason="continuous tracking has no active lifecycle owner",
                )
            with self._lock:
                floor = self._durable_checkpoint_floor
            after_camera_revision = floor.observation_generation if floor is not None else 0
            pair = sampler.wait_for_synchronized_pair(
                after_camera_revision=after_camera_revision,
                timeout_s=self.config.tracking_capture_wait_timeout_s,
            )
            if pair is None:
                return ActivationExecutionResult(
                    outcome=ControlOutcome.STALE_OBSERVATION,
                    reason=(
                        "continuous tracking produced no synchronized bowl/plate "
                        "revision within the capture bound"
                    ),
                )
            bowl_sample, plate_sample = pair.bowl, pair.plate
        if bowl_sample.revisions != plate_sample.revisions:
            return ActivationExecutionResult(
                outcome=ControlOutcome.STALE_OBSERVATION,
                reason="bowl and plate samples do not share one revision vector",
            )
        if plate_sample.quality is not ObservationQuality.TRACKED:
            return ActivationExecutionResult(
                outcome=ControlOutcome.UNCERTAIN,
                reason="plate target is not observable at the primary checkpoint",
            )
        if not bool(plate_sample.signals.get("identity_match")):
            return ActivationExecutionResult(
                outcome=ControlOutcome.WRONG_GROUNDING,
                reason="plate tracker identity does not match the active target",
            )
        if bowl_sample.quality is ObservationQuality.TRACKED and not bool(
            bowl_sample.signals.get("identity_match")
        ):
            return ActivationExecutionResult(
                outcome=ControlOutcome.WRONG_GROUNDING,
                reason="bowl tracker identity does not match the active subject",
            )
        revisions = revision_vector_from_observation(
            bowl_sample.revisions, camera_id=self.config.camera_id
        )
        generation = bowl_sample.revisions.camera_revision
        if generation < 1:
            return ActivationExecutionResult(
                outcome=ControlOutcome.STALE_OBSERVATION,
                reason="checkpoint camera generation must be positive",
            )
        with self._lock:
            floor = self._durable_checkpoint_floor
        if floor is not None and not self._strictly_advances_checkpoint(
            generation=generation,
            revisions=revisions,
            floor=floor,
            required_domains=self._causal_revision_domains(spec, workflow_id=workflow_id),
        ):
            return ActivationExecutionResult(
                outcome=ControlOutcome.STALE_OBSERVATION,
                reason=(
                    "checkpoint does not advance camera generation, regresses a "
                    "physical domain, or reuses an action-invalidated revision"
                ),
            )
        snapshot_id = f"sync-{bowl_sample.sample_id}-{plate_sample.sample_id}"
        target = self._plate_target(
            plate_sample.signals,
            snapshot_id=snapshot_id,
            generation=generation,
            revisions=revisions,
            evidence_ref=plate_sample.sample_id,
        )
        if bowl_sample.quality is ObservationQuality.TRACKED:
            held = self._held_bowl(
                bowl_sample.signals,
                snapshot_id=snapshot_id,
                generation=generation,
                revisions=revisions,
                evidence_ref=bowl_sample.sample_id,
            )
            held_visibility = VisibilityStatus.VISIBLE
            attachment = AttachmentStatus(str(bowl_sample.signals["attachment_status"]))
        else:
            held = None
            held_visibility = VisibilityStatus.OCCLUDED
            attachment = self._require_episode().state_reducer.state.attachment.status
        observation = BowlPlaceObservation(
            snapshot_id=snapshot_id,
            observation_generation=generation,
            revisions=revisions,
            frame_id=self.config.frame_id,
            expected_bowl_entity_id=self.config.bowl_entity_id,
            expected_target_entity_id=self.config.plate_entity_id,
            attachment_status=attachment,
            held_visibility=held_visibility,
            target_visibility=VisibilityStatus.VISIBLE,
            held=held,
            target=target,
        )
        with self._lock:
            self._latest_observation = observation
        lineage = self._capture_lineage(spec, workflow_id=workflow_id)
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(self._emit("observation", observation, lineage=lineage),),
        )

    def _verify_attachment(
        self, spec: InvocationSpec, *, activation_id: str
    ) -> ActivationExecutionResult:
        episode = self._require_episode()
        refs = self._input_refs(spec)
        observation = self._model(refs, "observation", BowlPlaceObservation)
        required = (
            AttachmentStatus.NOT_HELD
            if activation_id == "verify_release_attachment"
            else AttachmentStatus.VERIFIED_HELD
        )
        if observation.held_visibility is not VisibilityStatus.VISIBLE:
            return ActivationExecutionResult(
                outcome=ControlOutcome.UNCERTAIN,
                reason="primary checkpoint cannot see the bowl",
            )
        if observation.attachment_status is not required:
            return ActivationExecutionResult(
                outcome=ControlOutcome.ATTACHMENT_NOT_CONFIRMED,
                reason="fresh observation contradicts the required attachment status",
            )
        observation_ref = refs["observation"]
        causal_refs = self._causal_refs(observation_ref)
        action_id = self._action_id(causal_refs)
        revision = observation.revisions.camera[self.config.camera_id]
        evidence = AttachmentEvidence(
            evidence_id=f"attachment-{observation.snapshot_id}",
            entity_id=self.config.bowl_entity_id,
            entity_track_id=self.config.bowl_track_id,
            status=required.value,
            source_observation_id=observation.snapshot_id,
            source_observation_revision=revision,
            observation_domain=f"camera.{self.config.camera_id}",
            base_state_revision=episode.state_reducer.state.revision,
            action_id=action_id,
            evidence_refs=tuple(ref.artifact_id for ref in causal_refs),
            method="synchronized_tracking_and_gripper_guard",
            reason="fresh synchronized evidence matches the attachment predicate",
            confidence=0.98,
            validity=self._validity(observation, confidence=0.98),
        )
        outputs = [
            self._emit(
                "attachment_evidence",
                evidence,
                schema_id=_ATTACHMENT_EVIDENCE,
                lineage=causal_refs,
            )
        ]
        if required is AttachmentStatus.VERIFIED_HELD:
            phase = {
                "verify_initial_attachment": AttachmentGuardPhase.TRANSPORT,
                "verify_alignment_attachment": AttachmentGuardPhase.CORRECTION,
                "verify_pre_release_attachment": AttachmentGuardPhase.OPEN,
            }[activation_id]
            guard = AttachmentGuard(
                phase=phase,
                allowed=True,
                status=required,
                entity_id=self.config.bowl_entity_id,
                state_revision=episode.state_reducer.state.revision,
                observation_generation=observation.observation_generation,
                reason="fresh attachment evidence authorizes this held-object phase",
                evidence_refs=tuple(ref.artifact_id for ref in causal_refs),
            )
            outputs.append(self._emit("attachment_guard", guard, lineage=causal_refs))
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=tuple(outputs),
        )

    def _alignment_gate(
        self, spec: InvocationSpec, *, workflow_id: str
    ) -> ActivationExecutionResult:
        refs = self._input_refs(spec)
        observation = self._model(refs, "observation", BowlPlaceObservation)
        admitted_error = self._model(refs, "alignment_error", AlignmentError)
        if (
            admitted_error.snapshot_id != observation.snapshot_id
            or admitted_error.observation_generation != observation.observation_generation
            or admitted_error.revisions != observation.revisions
        ):
            return ActivationExecutionResult(
                outcome=ControlOutcome.STALE_OBSERVATION,
                reason="alignment artifact is not bound to the admitted observation",
            )
        servo = self._restore_servo(workflow_id)
        decision = servo.assess(observation)
        return ActivationExecutionResult(
            outcome=decision.control_outcome,
            artifacts=(self._emit("servo_decision", decision, lineage=tuple(refs.values())),),
            reason=decision.reason,
        )

    def _build_open_command(self, spec: InvocationSpec) -> ActivationExecutionResult:
        """Build a sealed open proposal from one admitted snapshot only."""

        refs = self._input_refs(spec)
        state = self._require_episode().state_reducer.state
        checkpoint = self._model(refs, "checkpoint", PhaseCheckpoint)
        observation = self._model(refs, "observation", BowlPlaceObservation)
        evidence = self._model(refs, "attachment_evidence", AttachmentEvidence)
        error = self._model(refs, "alignment_error", AlignmentError)
        snapshot = self._model(refs, "snapshot", AdmissionSnapshot)
        if (
            checkpoint.phase is not CheckpointPhase.PRE_RELEASE
            or checkpoint.status is not CheckpointStatus.PASSED
            or checkpoint.attachment_status is not AttachmentStatus.VERIFIED_HELD
        ):
            return ActivationExecutionResult(
                outcome=ControlOutcome.ATTACHMENT_NOT_CONFIRMED,
                reason="open requires a passed verified-held pre-release checkpoint",
            )
        if (
            state.attachment.status is not AttachmentStatus.VERIFIED_HELD
            or state.attachment.entity_id != self.config.bowl_entity_id
            or evidence.base_state_revision != state.revision
            or observation.attachment_status is not AttachmentStatus.VERIFIED_HELD
            or evidence.status != AttachmentStatus.VERIFIED_HELD.value
            or evidence.entity_id != self.config.bowl_entity_id
            or evidence.source_observation_id != observation.snapshot_id
            or error.status is not AlignmentStatus.WITHIN_TOLERANCE
            or error.snapshot_id != observation.snapshot_id
            or checkpoint.alignment_id != error.alignment_id
            or checkpoint.observation_generation != observation.observation_generation
            or checkpoint.revisions != observation.revisions
        ):
            return ActivationExecutionResult(
                outcome=ControlOutcome.STALE_INPUT,
                reason="open evidence is mixed, stale, or no longer within tolerance",
            )
        if not self._snapshot_matches(
            snapshot,
            resource_id=self.config.gripper_resource_id,
            required_attachment=AttachmentStatus.VERIFIED_HELD,
        ):
            return ActivationExecutionResult(
                outcome=ControlOutcome.STALE_INPUT,
                reason="open admission snapshot does not match the gripper contract",
            )
        command = GripperCommand(
            command_id=self._stable_action_id("open", refs),
            expected_snapshot=snapshot,
            mode="open",
            target_width_m=self.config.gripper_open_width_m,
            timeout_s=self.config.gripper_open_timeout_s,
            possibly_affected_revisions=(
                "robot.gripper",
                "attachment",
                "scene",
            ),
        )
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(self._emit("action_spec", command, lineage=tuple(refs.values())),),
        )

    def _build_settle_wait(self, spec: InvocationSpec) -> ActivationExecutionResult:
        """Build a bounded wait proposal without reading or driving a backend."""

        refs = self._input_refs(spec)
        receipt = self._model(refs, "receipt", ExecutionReceipt)
        snapshot = self._model(refs, "snapshot", AdmissionSnapshot)
        receipt_record = self._require_data_plane().artifact_record(refs["receipt"].artifact_id)
        open_specs = [
            GripperCommand.model_validate(self._require_data_plane().resolve(lineage_ref).payload)
            for lineage_ref in receipt_record.lineage
            if self._require_data_plane().artifact_record(lineage_ref.artifact_id).schema
            == GripperCommand.model_fields["schema_version"].default
        ]
        if (
            receipt.runtime_status is not ExecutionStatus.COMPLETED
            or receipt.spec_type is not ActionSpecType.GRIPPER_COMMAND
            or receipt.world_id != self.config.authority_world_id
            or receipt.resource_id != self.config.gripper_resource_id
            or len(open_specs) != 1
            or open_specs[0].mode.value != "open"
            or open_specs[0].content_digest != receipt.spec_digest
        ):
            return ActivationExecutionResult(
                outcome=ControlOutcome.STALE_INPUT,
                reason="settle requires a completed authoritative gripper-open receipt",
            )
        if not self._snapshot_matches(
            snapshot,
            resource_id=self.config.controller_resource_id,
        ):
            return ActivationExecutionResult(
                outcome=ControlOutcome.STALE_INPUT,
                reason="settle admission snapshot does not match the controller contract",
            )
        wait = WaitSpec(
            wait_id=self._stable_action_id("settle", refs),
            expected_snapshot=snapshot,
            duration_s=self.config.settle_duration_s,
            hold_command="hold_current",
            timeout_s=self.config.settle_timeout_s,
            possibly_affected_revisions=("scene",),
        )
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(self._emit("action_spec", wait, lineage=tuple(refs.values())),),
        )

    def _restore_servo(self, workflow_id: str) -> VisualServoPlacer:
        """Reconstruct bounded-servo state from durable decisions after restart."""

        servo = VisualServoPlacer(
            bowl_entity_id=self.config.bowl_entity_id,
            target_entity_id=self.config.plate_entity_id,
            frame_id=self.config.frame_id,
            tolerance=self.config.tolerance,
            limits=self.config.correction_limits,
        )
        plane = self._require_data_plane()
        observations = sorted(
            (
                record
                for record in plane.artifacts
                if record.workflow_id == workflow_id
                and record.activation_id == "capture_alignment"
                and record.port == "observation"
            ),
            key=lambda record: record.generation,
        )
        if observations:
            first = BowlPlaceObservation.model_validate(plane.resolve(observations[0].ref).payload)
            if first.target is not None:
                servo._target_reference = first.target.support_center_m

        decision_records = sorted(
            (
                record
                for record in plane.artifacts
                if record.workflow_id == workflow_id
                and record.activation_id == "alignment_gate"
                and record.port == "servo_decision"
            ),
            key=lambda record: record.generation,
        )
        decisions = [
            ServoDecision.model_validate(plane.resolve(record.ref).payload)
            for record in decision_records
        ]
        corrections = [
            decision.correction for decision in decisions if decision.correction is not None
        ]
        if corrections:
            latest = corrections[-1]
            servo._iterations = latest.iteration
            servo._cumulative_translation_m = latest.cumulative_translation_m
            servo._cumulative_yaw_rad = latest.cumulative_yaw_rad
        if decisions and decisions[-1].correction is not None:
            latest = decisions[-1].correction
            servo._freshness_floor = (
                latest.source_generation,
                latest.source_revisions,
            )
        return servo

    def _snapshot_matches(
        self,
        snapshot: AdmissionSnapshot,
        *,
        resource_id: str,
        required_attachment: AttachmentStatus | None = None,
    ) -> bool:
        return (
            snapshot.world_kind is WorldKind.AUTHORITATIVE
            and snapshot.world_id == self.config.authority_world_id
            and snapshot.resource_id == resource_id
            and snapshot.controller_state is ControllerState.READY
            and (required_attachment is None or snapshot.attachment_status is required_attachment)
        )

    @staticmethod
    def _stable_action_id(prefix: str, refs: dict[str, ResolvedArtifactRef]) -> str:
        material = "\x00".join(
            f"{name}:{ref.artifact_id}:{ref.content_digest}" for name, ref in sorted(refs.items())
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
        return f"bowl-{prefix}-{digest}"

    def _pre_release_checkpoint(self, spec: InvocationSpec) -> ActivationExecutionResult:
        refs = self._input_refs(spec)
        observation = self._model(refs, "observation", BowlPlaceObservation)
        error = self._model(refs, "alignment_error", AlignmentError)
        if (
            observation.attachment_status is not AttachmentStatus.VERIFIED_HELD
            or error.status is not AlignmentStatus.WITHIN_TOLERANCE
            or error.snapshot_id != observation.snapshot_id
        ):
            return ActivationExecutionResult(
                outcome=ControlOutcome.STALE_OBSERVATION,
                reason="post-descend attachment/alignment gate failed",
            )
        receipt_ref = refs["receipt"]
        checkpoint = PhaseCheckpoint(
            phase=CheckpointPhase.PRE_RELEASE,
            status=CheckpointStatus.PASSED,
            action_id=self._action_id((receipt_ref,)),
            receipt_ref=receipt_ref.artifact_id,
            attachment_status=AttachmentStatus.VERIFIED_HELD,
            bowl_entity_id=self.config.bowl_entity_id,
            alignment_id=error.alignment_id,
            observation_generation=observation.observation_generation,
            revisions=observation.revisions,
            evidence_refs=tuple(ref.artifact_id for ref in refs.values()),
            reason="fresh post-descend alignment and attachment gates passed",
        )
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(self._emit("checkpoint", checkpoint, lineage=tuple(refs.values())),),
        )

    def _post_release_checkpoint(self, spec: InvocationSpec) -> ActivationExecutionResult:
        episode = self._require_episode()
        refs = self._input_refs(spec)
        observation = self._model(refs, "observation", BowlPlaceObservation)
        state_receipt = self._model(refs, "state_receipt", StateCommitReceipt)
        if (
            observation.attachment_status is not AttachmentStatus.NOT_HELD
            or state_receipt.after_revision != episode.state_reducer.state.revision
            or episode.state_reducer.state.attachment.status is not AttachmentStatus.NOT_HELD
        ):
            return ActivationExecutionResult(
                outcome=ControlOutcome.ATTACHMENT_NOT_CONFIRMED,
                reason="release evidence is not committed as current episode state",
            )
        checkpoint = PhaseCheckpoint(
            phase=CheckpointPhase.POST_RELEASE,
            status=CheckpointStatus.PASSED,
            action_id=self._action_id((refs["open_receipt"],)),
            receipt_ref=refs["settle_receipt"].artifact_id,
            attachment_status=AttachmentStatus.NOT_HELD,
            bowl_entity_id=self.config.bowl_entity_id,
            observation_generation=observation.observation_generation,
            revisions=observation.revisions,
            evidence_refs=tuple(ref.artifact_id for ref in refs.values()),
            reason="fresh release evidence is durably committed",
        )
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(self._emit("checkpoint", checkpoint, lineage=tuple(refs.values())),),
        )

    def _verify_final_relation(self, spec: InvocationSpec) -> ActivationExecutionResult:
        episode = self._require_episode()
        refs = self._input_refs(spec)
        observation_ref = refs["observation"]
        observation = self._model(refs, "observation", BowlPlaceObservation)
        if observation.held is None or observation.target is None:
            return ActivationExecutionResult(
                outcome=ControlOutcome.UNCERTAIN,
                reason="final bowl/plate geometry is unavailable",
            )
        error = compute_alignment_error(
            observation.held,
            observation.target,
            tolerance=self.config.tolerance,
        )
        if not error.support_contained:
            return ActivationExecutionResult(
                outcome=ControlOutcome.FAILED_PLACEMENT,
                reason="independent final observation does not show support containment",
            )
        causal_refs = self._causal_refs(observation_ref)
        revision = observation.revisions.camera[self.config.camera_id]
        action_id = self._action_id(causal_refs)
        evidence = RelationEvidence(
            evidence_id=f"relation-{observation.snapshot_id}",
            subject_entity_id=self.config.bowl_entity_id,
            subject_track_id=self.config.bowl_track_id,
            predicate=RelationPredicate.SUPPORTED_BY,
            target_entity_id=self.config.plate_entity_id,
            target_track_id=self.config.plate_track_id,
            value=RelationValue.ASSERTED,
            source_observation_id=observation.snapshot_id,
            source_observation_revision=revision,
            observation_domain=f"camera.{self.config.camera_id}",
            base_state_revision=episode.state_reducer.state.revision,
            action_id=action_id,
            evidence_refs=tuple(ref.artifact_id for ref in causal_refs),
            method="independent_support_footprint_verifier",
            reason="fresh bowl footprint is contained by the plate support region",
            confidence=0.99,
            validity=self._validity(observation, confidence=0.99),
        )
        verdict = PlacementVerdict(
            status=PlacementVerdictStatus.SUCCEEDED,
            bowl_entity_id=self.config.bowl_entity_id,
            target_entity_id=self.config.plate_entity_id,
            relation=RelationAssessment.ASSERTED,
            source_observation_id=observation.snapshot_id,
            state_revision=episode.state_reducer.state.revision,
            support_clearance_m=error.support_clearance_m,
            confidence=0.99,
            evidence_refs=tuple(ref.artifact_id for ref in causal_refs),
            reason="independent final observation confirms support",
        )
        return ActivationExecutionResult(
            outcome=ControlOutcome.SUCCESS,
            artifacts=(
                self._emit(
                    "relation_evidence",
                    evidence,
                    schema_id=_RELATION_EVIDENCE,
                    lineage=causal_refs,
                ),
                self._emit("verdict", verdict, lineage=causal_refs),
            ),
        )

    def _complete(self, spec: InvocationSpec) -> ActivationExecutionResult:
        episode = self._require_episode()
        refs = self._input_refs(spec)
        verdict = self._model(refs, "verdict", PlacementVerdict)
        receipt = self._model(refs, "state_receipt", StateCommitReceipt)
        relation = episode.state_reducer.state.relation(
            self.config.bowl_entity_id,
            RelationPredicate.SUPPORTED_BY,
            self.config.plate_entity_id,
        )
        if (
            verdict.status is not PlacementVerdictStatus.SUCCEEDED
            or receipt.after_revision != episode.state_reducer.state.revision
            or relation is None
            or relation.value is not RelationValue.ASSERTED
        ):
            return ActivationExecutionResult(
                outcome=ControlOutcome.FAILED_PLACEMENT,
                reason="final relation is not the current reducer-committed state",
            )
        return ActivationExecutionResult(outcome=ControlOutcome.SUCCESS)

    def _ensure_tracks(self):
        registry = self._require_observations()
        for track_id, entity_id, label, signals in (
            (
                self.config.bowl_track_id,
                self.config.bowl_entity_id,
                "bowl",
                self.BOWL_SIGNALS,
            ),
            (
                self.config.plate_track_id,
                self.config.plate_entity_id,
                "plate",
                self.PLATE_SIGNALS,
            ),
        ):
            try:
                handle = registry.get(track_id)
                if handle.state.value == "new":
                    handle.start()
            except TrackNotFoundError:
                registry.create(
                    TrackRequest(
                        episode_id=registry.episode_id,
                        track_id=track_id,
                        entity=EntityIdentity(entity_id=entity_id, semantic_label=label),
                        backend_id=self.config.observation_backend_id,
                        declared_signals=signals,
                        camera_ids=(self.config.camera_id,),
                    ),
                    start=True,
                )
        return registry.get(self.config.bowl_track_id), registry.get(self.config.plate_track_id)

    def _activate_tracking_service(self, runtime: _ActorRuntime) -> None:
        self._ensure_tracks()
        if self.config.tracking_mode is BowlTrackingMode.MANUAL_CHECKPOINT:
            return
        sampler = self._ensure_tracking_sampler()
        if not runtime.tracking_acquired:
            sampler.acquire(runtime.sampler_owner_id)
            runtime.tracking_acquired = True

    def _ensure_tracking_sampler(self) -> SynchronizedBowlTrackSampler:
        if self.config.tracking_mode is not BowlTrackingMode.CONTINUOUS:
            raise BowlPlaceProviderError(
                "continuous tracking sampler is unavailable in manual checkpoint mode"
            )
        with self._lock:
            current = self._tracking_sampler
        if current is not None:
            return current
        bowl_track, plate_track = self._ensure_tracks()
        with self._lock:
            if self._tracking_sampler is None:
                candidate = self._sampler_factory(
                    bowl_track=bowl_track,
                    plate_track=plate_track,
                    poll_interval_s=self.config.tracking_poll_interval_s,
                    max_silence_s=self.config.tracking_max_silence_s,
                    max_pair_skew_s=self.config.tracking_max_pair_skew_s,
                    pair_buffer_size=self.config.tracking_pair_buffer_size,
                )
                if not isinstance(candidate, SynchronizedBowlTrackSampler):
                    raise TypeError("sampler_factory must return SynchronizedBowlTrackSampler")
                self._tracking_sampler = candidate
            return self._tracking_sampler

    def _require_tracking_sampler(self) -> SynchronizedBowlTrackSampler:
        sampler = self._ensure_tracking_sampler()
        if not sampler.running and sampler.owner_count == 0:
            raise BowlPlaceProviderError("continuous tracking has no active lifecycle owner")
        return sampler

    def _held_bowl(
        self,
        signals,
        *,
        snapshot_id: str,
        generation: int,
        revisions,
        evidence_ref: str,
    ) -> HeldBowlEstimate:
        center = self._center(signals)
        return HeldBowlEstimate(
            entity_id=self.config.bowl_entity_id,
            frame_id=self.config.frame_id,
            snapshot_id=snapshot_id,
            observation_generation=generation,
            revisions=revisions,
            bottom_center_m=center,
            support_footprint=self._footprint(center, self.config.bowl_radius_m),
            orientation=QuaternionWXYZ.from_yaw(float(signals["yaw"])),
            uncertainty=self._uncertainty(),
            confidence=0.94,
            evidence_refs=(evidence_ref,),
        )

    def _plate_target(
        self,
        signals,
        *,
        snapshot_id: str,
        generation: int,
        revisions,
        evidence_ref: str,
    ) -> PlateSupportTarget:
        center = self._center(signals)
        return PlateSupportTarget(
            entity_id=self.config.plate_entity_id,
            frame_id=self.config.frame_id,
            snapshot_id=snapshot_id,
            observation_generation=generation,
            revisions=revisions,
            support_center_m=center,
            support_footprint=self._footprint(center, self.config.plate_radius_m),
            surface_normal=UnitVector3(x=0.0, y=0.0, z=1.0),
            orientation=QuaternionWXYZ.from_yaw(float(signals["yaw"])),
            uncertainty=self._uncertainty(),
            safe_margin_m=self.config.plate_safe_margin_m,
            confidence=0.96,
            evidence_refs=(evidence_ref,),
        )

    @staticmethod
    def _center(signals) -> Vector3:
        return Vector3(
            x=float(signals["center_x"]),
            y=float(signals["center_y"]),
            z=float(signals["center_z"]),
        )

    @staticmethod
    def _footprint(center: Vector3, radius: float) -> SupportFootprint:
        return SupportFootprint(
            vertices_xy_m=(
                (center.x - radius, center.y - radius),
                (center.x + radius, center.y - radius),
                (center.x + radius, center.y + radius),
                (center.x - radius, center.y + radius),
            )
        )

    @staticmethod
    def _uncertainty() -> PoseUncertainty:
        return PoseUncertainty(
            translation_std_m=Vector3(x=0.0008, y=0.0008, z=0.001),
            orientation_std_rad=0.006,
        )

    def _capture_lineage(
        self, spec: InvocationSpec, *, workflow_id: str
    ) -> tuple[ResolvedArtifactRef, ...]:
        refs = tuple(self._input_refs(spec).values())
        if str(spec.metadata["activation_id"]) == "capture_alignment":
            plane = self._require_data_plane()
            candidates = [
                record
                for record in plane.artifacts
                if record.workflow_id == workflow_id
                and record.schema == _EXECUTION_RECEIPT
                and record.activation_id in {"execute_transport", "execute_correction"}
            ]
            if not candidates:
                raise BowlPlaceProviderError("alignment capture has no causal physical receipt")
            refs = (candidates[-1].ref,)
        return self._dedupe(refs)

    def _causal_refs(self, observation_ref: ResolvedArtifactRef) -> tuple[ResolvedArtifactRef, ...]:
        record = self._require_data_plane().artifact_record(observation_ref.artifact_id)
        return self._dedupe((observation_ref,), record.lineage)

    def _action_id(self, refs: tuple[ResolvedArtifactRef, ...]) -> str:
        plane = self._require_data_plane()
        records = [
            plane.artifact_record(ref.artifact_id)
            for ref in refs
            if plane.artifact_record(ref.artifact_id).schema == _EXECUTION_RECEIPT
        ]
        if not records:
            action_id = self._require_episode().state_reducer.state.attachment.action_id
            if not action_id:
                raise BowlPlaceProviderError("attachment evidence has no causal action identity")
            return action_id
        open_records = [record for record in records if record.activation_id == "execute_open"]
        selected = open_records[-1] if open_records else records[-1]
        return ExecutionReceipt.model_validate(plane.resolve(selected.ref).payload).action_id

    def _validity(self, observation: BowlPlaceObservation, *, confidence: float) -> ValidityVector:
        return ValidityVector(
            lifecycle=ValidityLifecycle.DERIVED,
            depends_on_revisions={
                f"camera.{self.config.camera_id}": observation.revisions.camera[
                    self.config.camera_id
                ]
            },
            observation_id=observation.snapshot_id,
            state_revision=self._require_episode().state_reducer.state.revision,
            method="exact_observation_and_state_revision",
            reason="evidence is valid only for this checkpoint and episode state",
            confidence=confidence,
        )

    def _model(self, refs, name: str, model):
        return model.model_validate(self._require_data_plane().resolve(refs[name]).payload)

    @staticmethod
    def _input_refs(spec: InvocationSpec) -> dict[str, ResolvedArtifactRef]:
        return {name: ResolvedArtifactRef.from_any(value) for name, value in spec.inputs.items()}

    @staticmethod
    def _emit(
        port: str,
        model,
        *,
        schema_id: str | None = None,
        lineage: tuple[ResolvedArtifactRef, ...] = (),
    ) -> ArtifactEmission:
        schema = schema_id or str(model.schema_version)
        return ArtifactEmission(
            port=port,
            schema_id=schema,
            payload=model.model_dump(mode="json"),
            lineage=lineage,
        )

    @staticmethod
    def _dedupe(*groups) -> tuple[ResolvedArtifactRef, ...]:
        values: list[ResolvedArtifactRef] = []
        seen: set[tuple[str, str]] = set()
        for group in groups:
            for ref in group:
                identity = (ref.artifact_id, ref.content_digest)
                if identity not in seen:
                    seen.add(identity)
                    values.append(ref)
        return tuple(values)

    @staticmethod
    def _runner_from_activation(activation_id: str) -> str:
        if activation_id.startswith("capture_"):
            return _CAPTURE_RUNNER
        if activation_id.startswith("verify_") and "attachment" in activation_id:
            return _ATTACHMENT_RUNNER
        return {
            "alignment_gate": _SERVO_GATE_RUNNER,
            "checkpoint_pre_release": _PRE_RELEASE_RUNNER,
            "checkpoint_post_release": _POST_RELEASE_RUNNER,
            "verify_placement": _FINAL_VERIFY_RUNNER,
            "placement_complete": _COMPLETE_RUNNER,
            "recovery_frontier": _RECOVERY_RUNNER,
            "held_bowl_tracker": _BOWL_TRACKER_RUNNER,
            "plate_tracker": _PLATE_TRACKER_RUNNER,
        }.get(activation_id, "")

    def _refresh_durable_checkpoint_floor(self) -> None:
        plane = self._require_data_plane()
        records = [
            record
            for record in plane.artifacts
            if record.schema == BowlPlaceObservation.model_fields["schema_version"].default
        ]
        if not records:
            return
        restored = BowlPlaceObservation.model_validate(plane.resolve(records[-1].ref).payload)
        with self._lock:
            current = self._durable_checkpoint_floor
            if current is None or (
                restored.observation_generation >= current.observation_generation
                and restored.revisions.camera.get(self.config.camera_id, -1)
                >= current.revisions.camera.get(self.config.camera_id, -1)
            ):
                self._durable_checkpoint_floor = restored
                self._latest_observation = restored

    def _strictly_advances_checkpoint(
        self,
        *,
        generation: int,
        revisions,
        floor: BowlPlaceObservation,
        required_domains: frozenset[str],
    ) -> bool:
        previous = floor.revisions
        camera = self.config.camera_id
        non_regressing = (
            generation > floor.observation_generation
            and revisions.camera.get(camera, -1) > previous.camera.get(camera, -1)
            and revisions.scene >= previous.scene
            and revisions.arm >= previous.arm
            and revisions.gripper >= previous.gripper
            and revisions.attachment >= previous.attachment
        )
        if not non_regressing:
            return False
        values = {
            "scene": (revisions.scene, previous.scene),
            "arm": (revisions.arm, previous.arm),
            "gripper": (revisions.gripper, previous.gripper),
            "attachment": (revisions.attachment, previous.attachment),
        }
        return all(values[domain][0] > values[domain][1] for domain in required_domains)

    def _causal_revision_domains(self, spec: InvocationSpec, *, workflow_id: str) -> frozenset[str]:
        plane = self._require_data_plane()
        affected: set[str] = set()
        for ref in self._capture_lineage(spec, workflow_id=workflow_id):
            record = plane.artifact_record(ref.artifact_id)
            if record.schema != _EXECUTION_RECEIPT:
                continue
            receipt = ExecutionReceipt.model_validate(plane.resolve(ref).payload)
            for domain in receipt.possibly_affected_revisions:
                normalized = {
                    "robot.arm": "arm",
                    "arm": "arm",
                    "robot.gripper": "gripper",
                    "gripper": "gripper",
                    "attachment": "attachment",
                    "scene": "scene",
                }.get(domain)
                if normalized is not None:
                    affected.add(normalized)
        return frozenset(affected)

    def _reserve_capture_attempt(
        self,
        *,
        workflow_id: str,
        activation_id: str,
        attempt: int,
        invocation_id: str,
    ) -> bool:
        """Fsync a capture reservation before the first non-repeatable poll."""

        path = self._capture_ledger_path
        if path is None:
            raise BowlPlaceProviderError("provider capture ledger is not bound")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _CAPTURE_LEDGER_SCHEMA,
            "episode_id": self._require_episode().episode_id,
            "workflow_id": workflow_id,
            "activation_id": activation_id,
            "attempt": attempt,
            "invocation_id": invocation_id,
            "status": "started",
        }
        encoded_payload = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        record = {
            **payload,
            "record_digest": "sha256:" + hashlib.sha256(encoded_payload).hexdigest(),
        }
        encoded_record = (
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("utf-8")
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            with os.fdopen(fd, "r+b", closefd=True) as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                existing_bytes = stream.read(_MAX_CAPTURE_LEDGER_BYTES + 1)
                if len(existing_bytes) > _MAX_CAPTURE_LEDGER_BYTES:
                    raise BowlPlaceProviderError("capture attempt ledger is oversized")
                existing_records = self._parse_capture_ledger(existing_bytes)
                identity = (workflow_id, activation_id, attempt)
                matches = [
                    item
                    for item in existing_records
                    if (
                        item["workflow_id"],
                        item["activation_id"],
                        item["attempt"],
                    )
                    == identity
                ]
                if len(matches) > 1:
                    raise BowlPlaceProviderError(
                        "capture ledger contains duplicate attempt reservations"
                    )
                if matches:
                    if matches[0] != record:
                        raise BowlPlaceProviderError(
                            "capture attempt identity was rebound to another invocation"
                        )
                    return False
                stream.seek(0, os.SEEK_END)
                stream.write(encoded_record)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            with suppress(OSError):
                os.close(fd)
            raise
        self._fsync_directory(path.parent)
        return True

    @staticmethod
    def _parse_capture_ledger(raw: bytes) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not raw:
            return records
        for line in raw.splitlines():
            try:
                item = json.loads(line)
            except (TypeError, ValueError) as exc:
                raise BowlPlaceProviderError("capture ledger contains invalid JSON") from exc
            if not isinstance(item, dict) or set(item) != {
                "schema_version",
                "episode_id",
                "workflow_id",
                "activation_id",
                "attempt",
                "invocation_id",
                "status",
                "record_digest",
            }:
                raise BowlPlaceProviderError("capture ledger record has an invalid shape")
            digest = item.pop("record_digest")
            canonical = json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            expected = "sha256:" + hashlib.sha256(canonical).hexdigest()
            item["record_digest"] = digest
            if digest != expected:
                raise BowlPlaceProviderError("capture ledger record digest mismatch")
            if (
                item["schema_version"] != _CAPTURE_LEDGER_SCHEMA
                or item["status"] != "started"
                or not isinstance(item["attempt"], int)
                or isinstance(item["attempt"], bool)
                or item["attempt"] < 1
            ):
                raise BowlPlaceProviderError("capture ledger record is invalid")
            records.append(item)
        return records

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _require_episode(self):
        if self._episode is None:
            raise BowlPlaceProviderError("provider is not bound to an EpisodeRuntime")
        return self._episode

    def _require_data_plane(self) -> EpisodeDataPlane:
        if self._data_plane is None:
            raise BowlPlaceProviderError("provider is not bound to an EpisodeDataPlane")
        return self._data_plane

    def _require_observations(self) -> ObservationRegistry:
        if self._observations is None:
            raise BowlPlaceProviderError("provider is not bound to ObservationRegistry")
        return self._observations


@dataclass(frozen=True)
class BowlPlaceActorBindings:
    provider_id: str
    providers: dict[str, BowlPlaceActorProvider]
    profiles: dict[str, ActorProfile]


@dataclass(frozen=True)
class BowlPlaceCodingBindings:
    provider_id: str
    profiles: dict[str, ActorProfile]
    required_skill_ids: tuple[str, ...]


def build_bowl_place_coding_profiles(
    graph: ElasticGraphSpec,
    *,
    provider_id: str = "skill_coding",
) -> BowlPlaceCodingBindings:
    """Bind every bowl coding runner to an explicit skill-augmented profile."""

    if not provider_id.strip():
        raise ValueError("provider_id must not be empty")
    profiles: dict[str, ActorProfile] = {}
    contracts: dict[str, tuple[Any, ...]] = {}
    nodes_by_runner: dict[str, list[Any]] = {}
    used_skills: set[str] = set()
    for node in graph.activations:
        if node.runner_kind is not RunnerKind.CODING_WORKER:
            continue
        try:
            task_kind, skill_ids = _BOWL_PLACE_CODING_SKILLS[node.runner_ref]
        except KeyError as exc:
            raise BowlPlaceProviderError(
                f"coding runner {node.runner_ref!r} has no production skill binding"
            ) from exc
        if node.effect_scope is not EffectScope.READ_ONLY:
            raise BowlPlaceProviderError(
                f"coding runner {node.runner_ref!r} must remain proposal-only"
            )
        contract = (
            node.lifecycle,
            frozenset(node.required_capabilities),
            node.estimated_budget.model_calls,
            node.estimated_budget.tokens,
            node.estimated_budget.wall_time_ms,
        )
        previous = contracts.get(node.runner_ref)
        if previous is not None and previous != contract:
            raise BowlPlaceProviderError(
                f"coding runner {node.runner_ref!r} has inconsistent graph contracts"
            )
        contracts[node.runner_ref] = contract
        nodes_by_runner.setdefault(node.runner_ref, []).append(node)
        used_skills.update(skill_ids)
    for runner_ref, nodes in sorted(nodes_by_runner.items()):
        task_kind, skill_ids = _BOWL_PLACE_CODING_SKILLS[runner_ref]
        node = nodes[0]
        config_rows: list[tuple[str, str]] = []
        for configured_node in sorted(nodes, key=lambda value: value.activation_id):
            raw_config = dict(configured_node.params)
            canonical_config = json.dumps(
                raw_config,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            config_rows.append((configured_node.activation_id, canonical_config))
        digest = hashlib.sha256(runner_ref.encode("utf-8")).hexdigest()[:16]
        profiles[runner_ref] = ActorProfile(
            profile_id=f"bowl-coding-{digest}",
            provider_id=provider_id,
            runner_kind=node.runner_kind.value,
            lifecycle=(
                ActorLifecycle.EPHEMERAL
                if node.lifecycle is LifecycleScope.INVOCATION
                else ActorLifecycle.SERVICE
            ),
            capability_ceiling=frozenset(node.required_capabilities),
            effect_ceiling=frozenset(),
            metadata={
                "task_kind": task_kind,
                "preloaded_skills": skill_ids,
                "strict_runtime_context": True,
                "node_config_v1": tuple(config_rows),
                "max_turns": node.estimated_budget.model_calls,
                "max_model_calls": node.estimated_budget.model_calls,
                "max_tokens": node.estimated_budget.tokens,
                "max_wall_time_ms": node.estimated_budget.wall_time_ms,
            },
        )
    expected = {
        node.runner_ref
        for node in graph.activations
        if node.runner_kind is RunnerKind.CODING_WORKER
    }
    if set(profiles) != expected:
        raise BowlPlaceProviderError("coding profile coverage is incomplete")
    return BowlPlaceCodingBindings(
        provider_id=provider_id,
        profiles=profiles,
        required_skill_ids=tuple(sorted(used_skills)),
    )


def build_bowl_place_actor_bindings(
    graph: ElasticGraphSpec,
    *,
    provider: BowlPlaceActorProvider,
    provider_id: str = "bowl_place",
    allow_manual_tracking_for_tests: bool = False,
) -> BowlPlaceActorBindings:
    """Build explicit bootstrap dependencies for deterministic bowl runners."""

    if not provider_id.strip():
        raise ValueError("provider_id must not be empty")
    if not isinstance(allow_manual_tracking_for_tests, bool):
        raise TypeError("allow_manual_tracking_for_tests must be a boolean")
    if (
        provider.config.tracking_mode is BowlTrackingMode.MANUAL_CHECKPOINT
        and not allow_manual_tracking_for_tests
    ):
        raise BowlPlaceProviderError(
            "production bowl bindings require continuous tracking; manual_checkpoint "
            "is reserved for explicit deterministic tests"
        )
    node_by_id = {node.activation_id: node for node in graph.activations}
    for plan_id, expected_resource in (
        ("plan_open", provider.config.gripper_resource_id),
        ("plan_settle", provider.config.controller_resource_id),
    ):
        try:
            plan_node = node_by_id[plan_id]
            snapshot_binding = next(
                binding for binding in plan_node.bindings if binding.input_port == "snapshot"
            )
            snapshot_node = node_by_id[snapshot_binding.source_activation]
        except (KeyError, StopIteration) as exc:
            raise BowlPlaceProviderError(
                f"bowl graph lacks the admitted snapshot contract for {plan_id!r}"
            ) from exc
        if (
            snapshot_node.authority_world_id != provider.config.authority_world_id
            or snapshot_node.authoritative_resource != expected_resource
        ):
            raise BowlPlaceProviderError(
                f"{plan_id!r} snapshot world/resource differs from provider config"
            )
    settle_node = node_by_id["plan_settle"]
    if float(settle_node.params.get("duration_s", -1.0)) != provider.config.settle_duration_s:
        raise BowlPlaceProviderError("plan_settle duration differs from BowlPlaceProviderConfig")
    profiles: dict[str, ActorProfile] = {}
    contracts: dict[str, tuple[Any, ...]] = {}
    for node in graph.activations:
        if node.runner_ref not in BOWL_PLACE_DETERMINISTIC_RUNNERS:
            continue
        if node.effect_scope is not EffectScope.READ_ONLY:
            raise BowlPlaceProviderError(
                f"deterministic runner {node.runner_ref!r} requests effects"
            )
        contract = (
            node.runner_kind,
            node.lifecycle,
            frozenset(node.required_capabilities),
        )
        previous = contracts.get(node.runner_ref)
        if previous is not None and previous != contract:
            raise BowlPlaceProviderError(
                f"runner {node.runner_ref!r} has inconsistent profile contracts"
            )
        contracts[node.runner_ref] = contract
        digest = hashlib.sha256(node.runner_ref.encode("utf-8")).hexdigest()[:16]
        profiles[node.runner_ref] = ActorProfile(
            profile_id=f"bowl-{digest}",
            provider_id=provider_id,
            runner_kind=node.runner_kind.value,
            lifecycle=(
                ActorLifecycle.EPHEMERAL
                if node.lifecycle is LifecycleScope.INVOCATION
                else ActorLifecycle.SERVICE
            ),
            capability_ceiling=frozenset(node.required_capabilities),
            effect_ceiling=frozenset(),
            metadata={"runner_ref": node.runner_ref},
        )
    missing = BOWL_PLACE_DETERMINISTIC_RUNNERS - set(profiles)
    if missing:
        raise BowlPlaceProviderError(
            "bowl graph lacks production deterministic runner bindings: "
            + ", ".join(sorted(missing))
        )
    return BowlPlaceActorBindings(
        provider_id=provider_id,
        providers={provider_id: provider},
        profiles=profiles,
    )


__all__ = [
    "BOWL_PLACE_DETERMINISTIC_RUNNERS",
    "BowlTrackingMode",
    "BowlPlaceActorBindings",
    "BowlPlaceActorProvider",
    "BowlPlaceCodingBindings",
    "BowlPlaceProviderConfig",
    "BowlPlaceProviderError",
    "SynchronizedBowlTrackSampler",
    "SynchronizedTrackPair",
    "build_bowl_place_actor_bindings",
    "build_bowl_place_coding_profiles",
]
