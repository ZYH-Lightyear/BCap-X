from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import pytest

from robomex.runtime.observation import (
    BackendObservation,
    ContinuousSamplerState,
    ContinuousTrackSampler,
    ContinuousTrackSamplerLifecycleError,
    ContinuousTrackSamplerStopError,
    EntityIdentity,
    ObservationContractError,
    ObservationQuality,
    ObservationRegistry,
    ObservationRevisionVector,
    TrackRequest,
    TrackSamplingOutcome,
    TrackState,
    TrackStateError,
)


class LiveReadOnlyBackend:
    read_capabilities = frozenset({"observation.read", "tracking.infer"})
    effect_capabilities: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        backend_id: str = "live-read-only",
        silence: bool = False,
        contract_error_entities: frozenset[str] = frozenset(),
        block_poll: bool = False,
    ) -> None:
        self.backend_id = backend_id
        self.silence = silence
        self.contract_error_entities = contract_error_entities
        self.block_poll = block_poll
        self.poll_entered = threading.Event()
        self.release_poll = threading.Event()
        self._counts: dict[str, int] = {}
        self.start_count = 0
        self.suspend_count = 0
        self.resume_count = 0
        self.stop_count = 0
        self._lock = threading.RLock()

    def start(self, request: TrackRequest) -> object:
        del request
        with self._lock:
            self.start_count += 1
        return object()

    def poll(self, runtime: object, request: TrackRequest) -> BackendObservation | None:
        del runtime
        self.poll_entered.set()
        if self.block_poll:
            self.release_poll.wait(timeout=5.0)
        if request.entity.entity_id in self.contract_error_entities:
            raise ObservationContractError("tracker violated declared signal contract")
        if self.silence:
            return None
        with self._lock:
            count = self._counts.get(request.entity.entity_id, 0) + 1
            self._counts[request.entity.entity_id] = count
        now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
        return BackendObservation(
            entity_id=request.entity.entity_id,
            quality=ObservationQuality.TRACKED,
            revisions=ObservationRevisionVector(
                scene_revision=count,
                arm_revision=count,
                gripper_revision=count,
                attachment_revision=count,
                camera_revision=count,
            ),
            signals={"visible": True},
            observed_at=now,
            monotonic_time_s=time.monotonic(),
        )

    def suspend(self, runtime: object) -> None:
        del runtime
        with self._lock:
            self.suspend_count += 1

    def resume(self, runtime: object) -> None:
        del runtime
        with self._lock:
            self.resume_count += 1

    def stop(self, runtime: object) -> None:
        del runtime
        with self._lock:
            self.stop_count += 1


def _registry_with_tracks(
    backend: LiveReadOnlyBackend,
    *,
    entities: tuple[str, ...] = ("bowl",),
):
    registry = ObservationRegistry(episode_id="episode-continuous")
    registry.register_backend(backend)
    handles = tuple(
        registry.create(
            TrackRequest(
                episode_id=registry.episode_id,
                track_id=f"track-{entity}",
                entity=EntityIdentity(entity_id=entity, semantic_label=entity),
                backend_id=backend.backend_id,
                declared_signals=("visible",),
            )
        )
        for entity in entities
    )
    return registry, handles


def _wait_until(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true within the test bound")
        time.sleep(0.005)


def test_lifecycle_is_idempotent_single_threaded_and_separates_manual_poll() -> None:
    backend = LiveReadOnlyBackend()
    _registry, (handle,) = _registry_with_tracks(backend)
    sampler = ContinuousTrackSampler(
        (handle,),
        sampler_id="continuous-bowl",
        poll_interval_s=0.02,
    )

    assert sampler.start() is True
    _wait_until(lambda: sampler.health.tracks[0].sample_count >= 1)
    worker_ident = sampler.worker_ident
    assert worker_ident is not None
    assert sampler.start() is False
    assert sampler.worker_ident == worker_ident
    assert sampler.thread_start_count == 1
    with pytest.raises(TrackStateError, match="manual poll is disabled"):
        handle.poll()
    with pytest.raises(TrackStateError, match="manual poll is disabled"):
        handle.poll_if_available()
    with pytest.raises(TrackStateError, match="manual poll is disabled"):
        handle.publish_lost(reason="manual_race")

    health = sampler.health
    assert health.state is ContinuousSamplerState.RUNNING
    assert health.thread_alive is True
    assert health.tracks[0].last_attempt_at is not None
    assert health.tracks[0].last_success_at is not None
    assert health.tracks[0].tracked_count >= 1
    assert sampler.stop() is True
    assert sampler.stop() is False
    assert sampler.state is ContinuousSamplerState.STOPPED
    assert handle.state is TrackState.STOPPED
    assert backend.start_count == 1
    assert backend.stop_count == 1


def test_backend_silence_is_continuously_published_as_lost_not_stale_reuse() -> None:
    backend = LiveReadOnlyBackend(silence=True)
    registry, (handle,) = _registry_with_tracks(backend)
    sampler = ContinuousTrackSampler((handle,), poll_interval_s=0.02)
    try:
        sampler.start()
        _wait_until(lambda: sampler.health.tracks[0].lost_count >= 1)
        latest = registry.stream.latest(handle.track_id)
        assert latest is not None
        assert latest.quality is ObservationQuality.LOST
        assert latest.signals == {}
        assert latest.reason == "no_fresh_observation"
        track_health = sampler.health.tracks[0]
        assert track_health.sample_count >= 1
        assert track_health.last_success_at is not None
        assert track_health.last_tracked_at is None
        assert any(
            event.outcome is TrackSamplingOutcome.LOST for event in sampler.events
        )
    finally:
        sampler.stop()


def test_bounded_silence_publishes_exactly_one_lost_until_fresh_sample() -> None:
    backend = LiveReadOnlyBackend(silence=True)
    registry, (handle,) = _registry_with_tracks(backend)
    sampler = ContinuousTrackSampler(
        (handle,),
        poll_interval_s=0.01,
        max_silence_s=0.04,
    )
    try:
        sampler.start()
        _wait_until(lambda: sampler.health.tracks[0].lost_count == 1)
        first_history = registry.stream.history
        assert len(first_history) == 1
        assert first_history[0].reason == "max_silence_exceeded:0.040000s"
        time.sleep(0.08)
        assert len(registry.stream.history) == 1
        track_health = sampler.health.tracks[0]
        assert track_health.lost_count == 1
        assert track_health.silent_count >= 2
        assert any(
            event.outcome is TrackSamplingOutcome.SILENT
            for event in sampler.events
        )
        backend.silence = False
        _wait_until(lambda: sampler.health.tracks[0].tracked_count >= 1)
        backend.silence = True
        _wait_until(lambda: sampler.health.tracks[0].lost_count == 2)
        assert len(registry.stream.history) >= 3
    finally:
        sampler.stop()


def test_suspend_resume_pauses_rounds_and_is_idempotent() -> None:
    backend = LiveReadOnlyBackend()
    _registry, (handle,) = _registry_with_tracks(backend)
    sampler = ContinuousTrackSampler((handle,), poll_interval_s=0.02)
    try:
        sampler.start()
        _wait_until(lambda: sampler.health.tracks[0].attempt_count >= 2)
        assert sampler.suspend() is True
        assert sampler.suspend() is False
        assert sampler.state is ContinuousSamplerState.SUSPENDED
        assert handle.state is TrackState.SUSPENDED
        attempts = sampler.health.tracks[0].attempt_count
        time.sleep(0.08)
        assert sampler.health.tracks[0].attempt_count == attempts
        assert sampler.resume() is True
        assert sampler.resume() is False
        _wait_until(lambda: sampler.health.tracks[0].attempt_count > attempts)
        assert handle.state is TrackState.ACTIVE
        assert backend.suspend_count == 1
        assert backend.resume_count == 1
    finally:
        sampler.stop()


def test_bounded_stop_reports_stuck_worker_and_can_be_retried_after_release() -> None:
    backend = LiveReadOnlyBackend(block_poll=True)
    _registry, (handle,) = _registry_with_tracks(backend)
    sampler = ContinuousTrackSampler(
        (handle,),
        poll_interval_s=0.02,
        stop_join_timeout_s=0.03,
    )
    sampler.start()
    assert backend.poll_entered.wait(timeout=0.5)

    started = time.monotonic()
    with pytest.raises(ContinuousTrackSamplerStopError, match="did not stop"):
        sampler.stop()
    elapsed = time.monotonic() - started
    assert elapsed < 0.3
    assert sampler.state is ContinuousSamplerState.STOP_FAILED
    assert sampler.health.thread_alive is True
    assert sampler.health.in_flight_track_id == handle.track_id
    assert sampler.health.tracks[0].track_state is TrackState.ACTIVE
    assert sampler.health.tracks[0].last_error is not None
    assert sampler.health.tracks[0].last_error.operation == "stop_join"

    backend.release_poll.set()
    assert sampler.stop(timeout_s=0.5) is True
    assert sampler.state is ContinuousSamplerState.STOPPED


def test_contract_exception_is_typed_per_track_and_does_not_starve_other_track() -> None:
    backend = LiveReadOnlyBackend(contract_error_entities=frozenset({"bad"}))
    _registry, handles = _registry_with_tracks(backend, entities=("bad", "good"))
    sampler = ContinuousTrackSampler(handles, poll_interval_s=0.02)
    try:
        sampler.start()
        _wait_until(
            lambda: (
                sampler.health.tracks[0].error_count >= 1
                and sampler.health.tracks[1].sample_count >= 1
            )
        )
        health_by_track = {
            health.track_id: health for health in sampler.health.tracks
        }
        failed = health_by_track["track-bad"]
        healthy = health_by_track["track-good"]
        assert failed.sample_count == 0
        assert failed.last_error is not None
        assert failed.last_error.operation == "poll"
        assert failed.last_error.error_type == "ObservationContractError"
        assert healthy.tracked_count >= 1
        assert {
            event.outcome for event in sampler.events
        }.issuperset({TrackSamplingOutcome.ERROR, TrackSamplingOutcome.TRACKED})
    finally:
        sampler.stop()


def test_context_manager_and_competing_sampler_ownership_are_explicit() -> None:
    backend = LiveReadOnlyBackend()
    _registry, (handle,) = _registry_with_tracks(backend)
    first = ContinuousTrackSampler((handle,), poll_interval_s=0.02)
    second = ContinuousTrackSampler((handle,), poll_interval_s=0.02)
    with first:
        _wait_until(lambda: first.health.tracks[0].sample_count >= 1)
        with pytest.raises(ContinuousTrackSamplerLifecycleError, match="failed to start"):
            second.start()
        assert second.state is ContinuousSamplerState.FAILED
        prior = first.health.tracks[0].sample_count
        assert second.stop() is True
        _wait_until(lambda: first.health.tracks[0].sample_count > prior)
        assert handle.state is TrackState.ACTIVE
    assert first.state is ContinuousSamplerState.STOPPED
    assert second.state is ContinuousSamplerState.STOPPED
