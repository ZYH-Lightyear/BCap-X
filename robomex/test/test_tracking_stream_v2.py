"""Tests for the episode-scoped ObservationStream/TrackHandle contract."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from robomex.data import EpisodeDataPlane
from robomex.runtime.observation import (
    BackendAuthorityError,
    BackendObservation,
    CheckpointObservationBackend,
    CheckpointRecord,
    CheckpointSegmentation,
    EntityIdentity,
    InMemoryObservationBackend,
    ObservationContractError,
    ObservationHistoryGapError,
    ObservationQuality,
    ObservationRegistry,
    ObservationRevisionVector,
    ObservationStream,
    TrackConflictError,
    TrackRequest,
    TrackState,
    TrackStateError,
)

UTC = timezone.utc  # noqa: UP017 - Python 3.10 compatibility
T0 = datetime(2026, 7, 22, 4, 0, tzinfo=UTC)


def _revisions(value: int = 1) -> ObservationRevisionVector:
    return ObservationRevisionVector(
        scene_revision=value,
        arm_revision=value,
        gripper_revision=value,
        attachment_revision=value,
        camera_revision=value,
    )


def _request(
    *,
    track_id: str = "track-bowl",
    entity_id: str = "bowl-1",
    backend_id: str = "memory",
    workflow_id: str = "pick-workflow",
    camera_ids: tuple[str, ...] = (),
) -> TrackRequest:
    return TrackRequest(
        episode_id="episode-1",
        track_id=track_id,
        entity=EntityIdentity(entity_id=entity_id, semantic_label="bowl"),
        backend_id=backend_id,
        declared_signals=("center_xyz", "confidence"),
        camera_ids=camera_ids,
        created_by_workflow_id=workflow_id,
    )


def _observation(
    *,
    entity_id: str = "bowl-1",
    quality: ObservationQuality = ObservationQuality.TRACKED,
    sequence_clock: int = 1,
    revisions: ObservationRevisionVector | None = None,
    signals: dict[str, object] | None = None,
    reason: str | None = None,
) -> BackendObservation:
    if signals is None:
        signals = (
            {"center_xyz": [0.41, -0.08, 0.22], "confidence": 0.91}
            if quality is ObservationQuality.TRACKED
            else {}
        )
    return BackendObservation(
        entity_id=entity_id,
        quality=quality,
        revisions=revisions or _revisions(sequence_clock),
        signals=signals,
        reason=reason,
        observed_at=T0 + timedelta(seconds=sequence_clock),
        monotonic_time_s=100.0 + sequence_clock,
    )


def test_sample_contract_is_strict_utc_and_oracle_free() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        BackendObservation(
            entity_id="bowl-1",
            quality=ObservationQuality.LOST,
            revisions=_revisions(),
            reason="occluded",
            observed_at=datetime(2026, 7, 22),
            monotonic_time_s=1.0,
        )

    with pytest.raises(ValidationError, match="oracle"):
        TrackRequest(
            episode_id="episode-1",
            track_id="track-bowl",
            entity=EntityIdentity(entity_id="bowl-1", semantic_label="bowl"),
            backend_id="memory",
            declared_signals=("simulator_truth.pose",),
        )

    with pytest.raises(ValidationError, match="ground-truth"):
        _observation(signals={"center_xyz": [0, 0, 0], "ground_truth": True})

    with pytest.raises(ValidationError, match="extra"):
        BackendObservation.model_validate(
            {
                **_observation().model_dump(),
                "action_command": "move_left",
            }
        )


def test_stream_assigns_global_strict_sequence_and_preserves_entity_identity() -> None:
    stream = ObservationStream(episode_id="episode-1")
    bowl = _request()
    plate = _request(track_id="track-plate", entity_id="plate-1")

    first = stream.publish(bowl, _observation(sequence_clock=1))
    second = stream.publish(
        plate,
        _observation(entity_id="plate-1", sequence_clock=1),
    )
    third = stream.publish(bowl, _observation(sequence_clock=2))

    assert [sample.sequence for sample in stream.history] == [1, 2, 3]
    assert first.entity.entity_id == "bowl-1"
    assert second.entity.entity_id == "plate-1"
    assert third.revisions.scene_revision == 2
    assert all(sample.observed_at.utcoffset() == timedelta(0) for sample in stream.history)
    assert stream.retention_policy == "append_only"


def test_stream_rejects_identity_swap_undeclared_missing_and_stale_context() -> None:
    stream = ObservationStream(episode_id="episode-1")
    request = _request()
    stream.publish(request, _observation(sequence_clock=2))

    with pytest.raises(ObservationContractError, match="identity"):
        stream.publish(
            request,
            _observation(entity_id="bowl-2", sequence_clock=3, revisions=_revisions(3)),
        )
    with pytest.raises(ObservationContractError, match="exactly declared"):
        stream.publish(
            request,
            _observation(
                sequence_clock=3,
                revisions=_revisions(3),
                signals={"center_xyz": [0, 0, 0]},
            ),
        )
    with pytest.raises(ObservationContractError, match="exactly declared"):
        stream.publish(
            request,
            _observation(
                sequence_clock=3,
                revisions=_revisions(3),
                signals={
                    "center_xyz": [0, 0, 0],
                    "confidence": 0.2,
                    "velocity": [0, 0, 0],
                },
            ),
        )
    with pytest.raises(ObservationContractError, match="revision vector regressed"):
        stream.publish(request, _observation(sequence_clock=3, revisions=_revisions(1)))
    with pytest.raises(ObservationContractError, match="strictly increase"):
        stream.publish(
            request,
            _observation(sequence_clock=1, revisions=_revisions(3)),
        )


@pytest.mark.parametrize("quality", [ObservationQuality.AMBIGUOUS, ObservationQuality.LOST])
def test_uncertain_quality_is_explicit_and_cannot_carry_ghost_coordinates(
    quality: ObservationQuality,
) -> None:
    with pytest.raises(ValidationError, match="clear signals"):
        _observation(
            quality=quality,
            signals={"center_xyz": [0.4, 0.0, 0.2], "confidence": 0.1},
            reason="occluded",
        )
    with pytest.raises(ValidationError, match="explicit reason"):
        _observation(quality=quality, signals={})

    sample = ObservationStream(episode_id="episode-1").publish(
        _request(),
        _observation(quality=quality, signals={}, reason="identity_not_unique"),
    )
    assert sample.quality is quality
    assert sample.signals == {}
    assert sample.reason == "identity_not_unique"


def test_track_lifecycle_is_idempotent_and_backend_silence_publishes_lost() -> None:
    backend = InMemoryObservationBackend()
    backend.push(_observation(sequence_clock=1))
    registry = ObservationRegistry(episode_id="episode-1")
    registry.register_backend(backend)
    handle = registry.create(_request())

    assert handle.state is TrackState.NEW
    assert handle.start() is True
    assert handle.start() is False
    assert handle.poll().quality is ObservationQuality.TRACKED
    assert handle.suspend() is True
    assert handle.suspend() is False
    with pytest.raises(TrackStateError, match="cannot poll"):
        handle.poll()
    assert handle.resume() is True
    assert handle.resume() is False

    silence = handle.poll()
    assert silence.quality is ObservationQuality.LOST
    assert silence.reason == "no_fresh_observation"
    assert silence.signals == {}
    assert silence.sequence == 2

    assert handle.stop() is True
    assert handle.stop() is False
    assert handle.state is TrackState.STOPPED
    with pytest.raises(TrackStateError, match="cannot be restarted"):
        handle.start()


def test_backend_exception_is_a_new_lost_sample_not_previous_coordinates() -> None:
    class BrokenBackend(InMemoryObservationBackend):
        def poll(self, runtime, request):
            raise RuntimeError("tracker process disappeared")

    backend = BrokenBackend("broken")
    registry = ObservationRegistry(episode_id="episode-1")
    registry.register_backend(backend)
    handle = registry.create(
        _request(backend_id="broken"),
        start=True,
    )

    sample = handle.poll()
    assert sample.quality is ObservationQuality.LOST
    assert sample.reason == "backend_error:RuntimeError"
    assert sample.signals == {}


def test_registry_track_survives_workflow_close_and_stops_at_episode_close() -> None:
    backend = InMemoryObservationBackend()
    registry = ObservationRegistry(episode_id="episode-1")
    registry.register_backend(backend)
    original = registry.create(_request(workflow_id="subgoal-1"), start=True)

    registry.close_workflow("subgoal-1")
    from_next_subgoal = registry.get("track-bowl")
    assert from_next_subgoal is original
    assert from_next_subgoal.state is TrackState.ACTIVE

    assert registry.close() is True
    assert registry.close() is False
    assert original.state is TrackState.STOPPED


@pytest.mark.parametrize(
    "read_capabilities,effect_capabilities",
    [
        (frozenset({"observation.read"}), frozenset({"world.write"})),
        (frozenset({"motion.control"}), frozenset()),
        (frozenset({"simulator.oracle"}), frozenset()),
    ],
)
def test_registry_rejects_backend_effect_action_and_oracle_authority(
    read_capabilities: frozenset[str], effect_capabilities: frozenset[str]
) -> None:
    class UnsafeBackend(InMemoryObservationBackend):
        pass

    backend = UnsafeBackend("unsafe")
    backend.read_capabilities = read_capabilities
    backend.effect_capabilities = effect_capabilities

    with pytest.raises(BackendAuthorityError):
        ObservationRegistry(episode_id="episode-1").register_backend(backend)


def test_after_subscriptions_replay_independently_and_bounded_gap_is_explicit() -> None:
    stream = ObservationStream(episode_id="episode-1", retention_limit=2)
    bowl = _request()
    plate = _request(track_id="track-plate", entity_id="plate-1")
    bowl_subscription = stream.subscribe(track_id="track-bowl")
    all_subscription = stream.subscribe()

    stream.publish(bowl, _observation(sequence_clock=1))
    stream.publish(plate, _observation(entity_id="plate-1", sequence_clock=1))
    assert [sample.sequence for sample in bowl_subscription.poll()] == [1]
    assert [sample.sequence for sample in all_subscription.poll()] == [1, 2]

    stream.publish(bowl, _observation(sequence_clock=2))
    assert stream.dropped_through_sequence == 1
    assert [sample.sequence for sample in bowl_subscription.poll()] == [3]
    assert [sample.sequence for sample in all_subscription.poll()] == [3]
    with pytest.raises(ObservationHistoryGapError, match="dropped through 1"):
        stream.after(0)


def test_replay_rechecks_contiguous_sequence_and_physical_context() -> None:
    source = ObservationStream(episode_id="episode-1")
    source.publish(_request(), _observation(sequence_clock=1))
    source.publish(_request(), _observation(sequence_clock=2))

    restored = ObservationStream(episode_id="episode-1")
    restored.replay(sample.model_dump(mode="python") for sample in source.history)
    assert restored.history == source.history
    with pytest.raises(ObservationContractError, match="contiguous"):
        restored.replay([source.history[-1].model_copy(update={"sequence": 4})])


def test_checkpoint_backend_resolves_content_and_resegments_without_world_handle(
    tmp_path,
) -> None:
    plane = EpisodeDataPlane(tmp_path / "episode", episode_id="episode-1")
    plane.open_workflow("capture")
    frame = plane.publish(
        workflow_id="capture",
        activation_id="camera",
        attempt=1,
        port="frame",
        schema="robomex.camera_frame.v1",
        payload={"pixels_digest": "camera-frame-001", "camera_id": "wrist"},
    )
    calls = []

    def segment(request, checkpoint, resolved):
        calls.append((request.entity.entity_id, checkpoint.checkpoint_id, resolved.schema))
        return CheckpointSegmentation(
            quality=ObservationQuality.TRACKED,
            signals={"center_xyz": [0.42, -0.01, 0.19], "confidence": 0.88},
        )

    backend = CheckpointObservationBackend(
        backend_id="checkpoint",
        episode_id="episode-1",
        resolver=plane.resolver,
        segmenter=segment,
    )
    checkpoint = CheckpointRecord(
            checkpoint_id="wrist-001",
            episode_id="episode-1",
            camera_id="wrist",
            frame_ref=frame.ref,
            revisions=_revisions(4),
            captured_at=T0,
            monotonic_time_s=200.0,
        )
    assert backend.add_checkpoint(checkpoint) is True
    assert backend.add_checkpoint(checkpoint) is False
    with pytest.raises(TrackConflictError, match="rebound"):
        backend.add_checkpoint(
            checkpoint.model_copy(update={"camera_id": "front"})
        )
    registry = ObservationRegistry(episode_id="episode-1")
    registry.register_backend(backend)
    handle = registry.create(
        _request(backend_id="checkpoint", camera_ids=("wrist",)),
        start=True,
    )

    sample = handle.poll()
    assert sample.quality is ObservationQuality.TRACKED
    assert sample.signals["center_xyz"] == [0.42, -0.01, 0.19]
    assert sample.artifact_refs == (frame.ref,)
    assert calls == [("bowl-1", "wrist-001", "robomex.camera_frame.v1")]
    assert backend.effect_capabilities == frozenset()


def test_checkpoint_backend_rejects_cross_episode_and_oracle_payload(tmp_path) -> None:
    plane = EpisodeDataPlane(tmp_path / "episode", episode_id="episode-1")
    plane.open_workflow("capture")
    oracle_frame = plane.publish(
        workflow_id="capture",
        activation_id="camera",
        attempt=1,
        port="truth",
        schema="test.frame.v1",
        payload={"simulator_truth": {"object_pose": [0, 0, 0]}},
    )
    backend = CheckpointObservationBackend(
        backend_id="checkpoint",
        episode_id="episode-1",
        resolver=plane.resolver,
        segmenter=lambda *_: CheckpointSegmentation(
            quality=ObservationQuality.LOST,
            reason="unused",
        ),
    )

    with pytest.raises(ObservationContractError, match="oracle"):
        backend.add_checkpoint(
            CheckpointRecord(
                episode_id="episode-1",
                camera_id="front",
                frame_ref=oracle_frame.ref,
                revisions=_revisions(),
            )
        )
    with pytest.raises(ObservationContractError, match="another episode"):
        backend.add_checkpoint(
            CheckpointRecord(
                episode_id="episode-2",
                camera_id="front",
                frame_ref=oracle_frame.ref,
                revisions=_revisions(),
            )
        )


def test_checkpoint_segmenter_oracle_output_is_contract_failure_not_silent_lost(
    tmp_path,
) -> None:
    plane = EpisodeDataPlane(tmp_path / "episode", episode_id="episode-1")
    plane.open_workflow("capture")
    frame = plane.publish(
        workflow_id="capture",
        activation_id="camera",
        attempt=1,
        port="frame",
        schema="test.frame.v1",
        payload={"camera_id": "front", "pixels_digest": "frame-1"},
    )
    backend = CheckpointObservationBackend(
        backend_id="checkpoint",
        episode_id="episode-1",
        resolver=plane.resolver,
        segmenter=lambda *_: {
            "quality": "tracked",
            "signals": {
                "center_xyz": [0, 0, 0],
                "confidence": 1.0,
                "source": "simulator_truth",
            },
        },
    )
    backend.add_checkpoint(
        CheckpointRecord(
            episode_id="episode-1",
            camera_id="front",
            frame_ref=frame.ref,
            revisions=_revisions(),
        )
    )
    registry = ObservationRegistry(episode_id="episode-1")
    registry.register_backend(backend)
    handle = registry.create(_request(backend_id="checkpoint"), start=True)

    with pytest.raises(ObservationContractError, match="segmenter violated"):
        handle.poll()
