from __future__ import annotations

import json
import multiprocessing
import os
import shutil
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from robomex.data import (
    ArtifactIntegrityError,
    EmbodiedStateReducer,
    EpisodeDataPlane,
    ResolvedArtifactRef,
    StateTransitionProposal,
    StateTransitionRejected,
)


def _concurrent_publish_worker(
    root: str,
    worker: int,
    publications: int,
    start: Any,
    results: Any,
) -> None:
    try:
        plane = EpisodeDataPlane(root, episode_id="ep_process")
        start.wait()
        artifact_ids = []
        for index in range(publications):
            record = plane.publish(
                workflow_id="workflow",
                activation_id="shared_activation",
                attempt=1,
                port="output",
                schema="robomex.process_test.v1",
                payload={"worker": worker, "index": index},
            )
            artifact_ids.append(record.artifact_id)
        results.put(("ok", artifact_ids))
    except BaseException as exc:  # pragma: no cover - asserted through child result
        results.put(("error", f"{type(exc).__name__}: {exc}"))


def _publish_once_worker(root: str, start: Any, results: Any) -> None:
    try:
        plane = EpisodeDataPlane(root, episode_id="ep_once")
        start.wait()
        record = plane.publish_once(
            workflow_id="workflow",
            activation_id="activation",
            attempt=1,
            port="output",
            schema="robomex.process_test.v1",
            payload={"stable": True},
        )
        results.put(("ok", record.artifact_id))
    except BaseException as exc:  # pragma: no cover - asserted through child result
        results.put(("error", f"{type(exc).__name__}: {exc}"))


def _concurrent_state_worker(
    root: str,
    worker: int,
    evidence_id: str,
    evidence_digest: str,
    start: Any,
    results: Any,
) -> None:
    try:
        reducer = EmbodiedStateReducer(root, episode_id="ep_state_process")
        evidence = ResolvedArtifactRef(evidence_id, evidence_digest)
        start.wait()
        entity_id = f"entity_{worker}"
        for _ in range(100):
            state = reducer.state
            if state.entity(entity_id) is not None:
                results.put(("ok", state.revision))
                return
            proposal = StateTransitionProposal.register_entity(
                episode_id="ep_state_process",
                effect_id=f"register_{worker}",
                before_revision=state.revision,
                source="process_test",
                evidence_refs=(evidence,),
                entity_id=entity_id,
                semantic_label="test_object",
            )
            try:
                committed = reducer.commit(proposal)
            except StateTransitionRejected as exc:
                if "Stale before_revision" in str(exc):
                    continue
                raise
            results.put(("ok", committed.revision))
            return
        raise RuntimeError("state commit did not converge after bounded retries")
    except BaseException as exc:  # pragma: no cover - asserted through child result
        results.put(("error", f"{type(exc).__name__}: {exc}"))


def _identity_worker(root: str, episode_id: str, start: Any, results: Any) -> None:
    try:
        start.wait()
        EpisodeDataPlane(root, episode_id=episode_id)
        results.put(("ok", episode_id))
    except BaseException as exc:  # pragma: no cover - asserted through child result
        results.put(("error", episode_id, f"{type(exc).__name__}: {exc}"))


def _run_processes(target: Any, args: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=target, args=(*item, start, results)) for item in args
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive(), "child process deadlocked on the episode lock"
        assert process.exitcode == 0
    collected = []
    for _ in processes:
        try:
            collected.append(results.get(timeout=5))
        except Empty as exc:  # pragma: no cover - diagnostic guard
            raise AssertionError("child process returned no result") from exc
    return collected


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mutate_ledger(path: Path, mutation: str) -> None:
    lines = path.read_bytes().splitlines(keepends=True)
    assert lines
    if mutation == "truncate":
        path.write_bytes(b"".join(lines[:-1]))
        return
    if mutation == "tamper":
        event = json.loads(lines[0])
        event["ignored_but_tampered"] = True
        path.write_bytes(_canonical_json(event) + b"\n" + b"".join(lines[1:]))
        return
    if mutation == "partial_tail":
        with path.open("ab") as stream:
            stream.write(b'{"schema":')
        return
    raise AssertionError(f"unknown mutation {mutation!r}")


def test_concurrent_process_publications_have_unique_monotonic_generations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id="ep_process")
    plane.open_workflow("workflow")

    workers = 4
    publications = 6
    results = _run_processes(
        _concurrent_publish_worker,
        [(str(root), worker, publications) for worker in range(workers)],
    )
    assert all(result[0] == "ok" for result in results), results
    assert len(plane.artifacts) == workers * publications

    replayed = EpisodeDataPlane(root, episode_id="ep_process")
    records = [
        record
        for record in replayed.artifacts
        if record.activation_id == "shared_activation" and record.port == "output"
    ]
    assert len(records) == workers * publications
    assert sorted(record.generation for record in records) == list(
        range(1, workers * publications + 1)
    )
    assert len({record.artifact_id for record in records}) == len(records)
    assert [event["sequence"] for event in replayed.events()] == list(
        range(1, replayed.event_count + 1)
    )


def test_publish_once_is_linearizable_across_processes(tmp_path: Path) -> None:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id="ep_once")
    plane.open_workflow("workflow")

    results = _run_processes(_publish_once_worker, [(str(root),) for _ in range(6)])
    assert all(result[0] == "ok" for result in results), results
    assert len({result[1] for result in results}) == 1

    replayed = EpisodeDataPlane(root, episode_id="ep_once")
    assert len(replayed.artifacts) == 1
    assert replayed.event_count == 2


def test_concurrent_state_commits_are_not_lost_and_revisions_are_monotonic(
    tmp_path: Path,
) -> None:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id="ep_state_process")
    plane.open_workflow("workflow")
    evidence = plane.publish(
        workflow_id="workflow",
        activation_id="evidence",
        attempt=1,
        port="output",
        schema="robomex.process_test.v1",
        payload={"stable": True},
    )
    long_lived_reducer = EmbodiedStateReducer(root, episode_id="ep_state_process")

    workers = 5
    results = _run_processes(
        _concurrent_state_worker,
        [
            (str(root), worker, evidence.artifact_id, evidence.content_digest)
            for worker in range(workers)
        ],
    )
    assert all(result[0] == "ok" for result in results), results
    assert long_lived_reducer.state.revision == workers

    replayed = EmbodiedStateReducer(root, episode_id="ep_state_process")
    assert replayed.state.revision == workers
    assert {entity.entity_id for entity in replayed.state.entities} == {
        f"entity_{worker}" for worker in range(workers)
    }
    assert [event["after_revision"] for event in replayed.events()] == list(
        range(1, workers + 1)
    )


def test_concurrent_manifest_identity_binding_allows_exactly_one_episode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "episode"
    results = _run_processes(
        _identity_worker,
        [(str(root), "episode_a"), (str(root), "episode_b")],
    )
    successes = [result for result in results if result[0] == "ok"]
    failures = [result for result in results if result[0] == "error"]
    assert len(successes) == 1
    assert len(failures) == 1
    assert "different identity" in failures[0][2]
    EpisodeDataPlane(root, episode_id=successes[0][1])


@pytest.mark.parametrize("mutation", ["truncate", "tamper", "partial_tail"])
def test_artifact_ledger_restart_fails_closed_on_corruption(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / "source"
    plane = EpisodeDataPlane(source, episode_id="ep_integrity")
    plane.open_workflow("workflow")
    plane.publish(
        workflow_id="workflow",
        activation_id="producer",
        attempt=1,
        port="output",
        schema="robomex.process_test.v1",
        payload={"value": 1},
    )
    root = tmp_path / mutation
    shutil.copytree(source, root)
    _mutate_ledger(root / "artifact_events.jsonl", mutation)

    with pytest.raises(ArtifactIntegrityError):
        EpisodeDataPlane(root, episode_id="ep_integrity")


@pytest.mark.parametrize("mutation", ["truncate", "tamper", "partial_tail"])
def test_state_ledger_restart_fails_closed_on_corruption(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / "source"
    plane = EpisodeDataPlane(source, episode_id="ep_state_integrity")
    plane.open_workflow("workflow")
    evidence = plane.publish(
        workflow_id="workflow",
        activation_id="evidence",
        attempt=1,
        port="output",
        schema="robomex.process_test.v1",
        payload={"value": 1},
    )
    reducer = EmbodiedStateReducer(source, episode_id="ep_state_integrity")
    reducer.commit(
        StateTransitionProposal.register_entity(
            episode_id="ep_state_integrity",
            effect_id="register_entity",
            before_revision=0,
            source="test",
            evidence_refs=(evidence.ref,),
            entity_id="entity",
            semantic_label="object",
        )
    )
    root = tmp_path / mutation
    shutil.copytree(source, root)
    _mutate_ledger(root / "embodied_state_events.jsonl", mutation)

    with pytest.raises(ArtifactIntegrityError):
        EmbodiedStateReducer(root, episode_id="ep_state_integrity")


def test_artifact_journal_fsync_before_head_is_recovered_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id="ep_crash")
    plane.open_workflow("workflow")

    def fail_head(**_: Any) -> None:
        raise OSError("simulated crash before head replace")

    monkeypatch.setattr(plane, "_write_head", fail_head)
    with pytest.raises(OSError, match="simulated crash"):
        plane.publish_once(
            workflow_id="workflow",
            activation_id="producer",
            attempt=1,
            port="output",
            schema="robomex.process_test.v1",
            payload={"value": 1},
        )

    replayed = EpisodeDataPlane(root, episode_id="ep_crash")
    record = replayed.publish_once(
        workflow_id="workflow",
        activation_id="producer",
        attempt=1,
        port="output",
        schema="robomex.process_test.v1",
        payload={"value": 1},
    )
    assert record.generation == 1
    assert len(replayed.artifacts) == 1


def test_state_journal_fsync_before_head_is_recovered_without_lost_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id="ep_state_crash")
    plane.open_workflow("workflow")
    evidence = plane.publish(
        workflow_id="workflow",
        activation_id="evidence",
        attempt=1,
        port="output",
        schema="robomex.process_test.v1",
        payload={"value": 1},
    )
    reducer = EmbodiedStateReducer(root, episode_id="ep_state_crash")
    proposal = StateTransitionProposal.register_entity(
        episode_id="ep_state_crash",
        effect_id="register_entity",
        before_revision=0,
        source="test",
        evidence_refs=(evidence.ref,),
        entity_id="entity",
        semantic_label="object",
    )

    def fail_head(**_: Any) -> None:
        raise OSError("simulated crash before head replace")

    monkeypatch.setattr(reducer, "_write_head", fail_head)
    with pytest.raises(OSError, match="simulated crash"):
        reducer.commit(proposal)

    replayed = EmbodiedStateReducer(root, episode_id="ep_state_crash")
    assert replayed.state.revision == 1
    assert replayed.state.entity("entity") is not None
    with pytest.raises(StateTransitionRejected, match="Duplicate"):
        replayed.commit(
            StateTransitionProposal.register_entity(
                episode_id="ep_state_crash",
                effect_id="register_entity",
                before_revision=1,
                source="test",
                evidence_refs=(evidence.ref,),
                entity_id="entity",
                semantic_label="object",
            )
        )


def test_initialized_ledgers_require_their_durable_heads(tmp_path: Path) -> None:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id="ep_heads")
    plane.open_workflow("workflow")
    plane.ledger_head_path.unlink()
    with pytest.raises(ArtifactIntegrityError, match="head.*missing"):
        EpisodeDataPlane(root, episode_id="ep_heads")


def test_state_evidence_policy_is_immutably_bound_to_episode(tmp_path: Path) -> None:
    root = tmp_path / "episode"
    EpisodeDataPlane(root, episode_id="ep_policy")
    EmbodiedStateReducer(root, episode_id="ep_policy", strict_evidence=True)
    with pytest.raises(ArtifactIntegrityError, match="policy cannot be rebound"):
        EmbodiedStateReducer(root, episode_id="ep_policy", strict_evidence=False)


def test_pending_artifact_intent_without_journal_append_rolls_back_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id="ep_pending_rollback")
    plane.open_workflow("workflow")

    def fail_append(_: bytes) -> None:
        raise OSError("simulated crash before journal append")

    monkeypatch.setattr(plane, "_append_journal_bytes", fail_append)
    with pytest.raises(OSError, match="before journal append"):
        plane.publish_once(
            workflow_id="workflow",
            activation_id="producer",
            attempt=1,
            port="output",
            schema="robomex.process_test.v1",
            payload={"value": 1},
        )
    assert plane.ledger_pending_path.exists()

    replayed = EpisodeDataPlane(root, episode_id="ep_pending_rollback")
    assert not replayed.ledger_pending_path.exists()
    assert replayed.artifacts == ()
    record = replayed.publish_once(
        workflow_id="workflow",
        activation_id="producer",
        attempt=1,
        port="output",
        schema="robomex.process_test.v1",
        payload={"value": 1},
    )
    assert record.generation == 1


def test_partial_artifact_append_is_completed_only_from_exact_pending_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id="ep_pending_partial")
    plane.open_workflow("workflow")

    def partial_append(encoded: bytes) -> None:
        with plane.event_log_path.open("ab") as stream:
            stream.write(encoded[: len(encoded) // 2])
            stream.flush()
            os.fsync(stream.fileno())
        raise OSError("simulated crash during journal append")

    monkeypatch.setattr(plane, "_append_journal_bytes", partial_append)
    with pytest.raises(OSError, match="during journal append"):
        plane.publish_once(
            workflow_id="workflow",
            activation_id="producer",
            attempt=1,
            port="output",
            schema="robomex.process_test.v1",
            payload={"value": 1},
        )

    replayed = EpisodeDataPlane(root, episode_id="ep_pending_partial")
    assert len(replayed.artifacts) == 1
    assert replayed.event_count == 2
    assert not replayed.ledger_pending_path.exists()
    repeated = replayed.publish_once(
        workflow_id="workflow",
        activation_id="producer",
        attempt=1,
        port="output",
        schema="robomex.process_test.v1",
        payload={"value": 1},
    )
    assert repeated.generation == 1
    assert replayed.event_count == 2


def test_committed_artifact_head_with_residual_pending_is_cleaned_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id="ep_pending_cleanup")
    plane.open_workflow("workflow")

    def fail_clear() -> None:
        raise OSError("simulated crash before pending clear")

    monkeypatch.setattr(plane, "_clear_pending", fail_clear)
    with pytest.raises(OSError, match="before pending clear"):
        plane.publish_once(
            workflow_id="workflow",
            activation_id="producer",
            attempt=1,
            port="output",
            schema="robomex.process_test.v1",
            payload={"value": 1},
        )
    assert plane.ledger_pending_path.exists()

    replayed = EpisodeDataPlane(root, episode_id="ep_pending_cleanup")
    assert len(replayed.artifacts) == 1
    assert not replayed.ledger_pending_path.exists()


def test_canonical_artifact_suffix_without_pending_witness_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id="ep_forged_suffix")
    plane.open_workflow("workflow")

    def fail_head(**_: Any) -> None:
        raise OSError("simulated crash before head")

    monkeypatch.setattr(plane, "_write_head", fail_head)
    with pytest.raises(OSError, match="before head"):
        plane.publish_once(
            workflow_id="workflow",
            activation_id="producer",
            attempt=1,
            port="output",
            schema="robomex.process_test.v1",
            payload={"value": 1},
        )
    plane.ledger_pending_path.unlink()

    with pytest.raises(ArtifactIntegrityError, match="without a pending intent"):
        EpisodeDataPlane(root, episode_id="ep_forged_suffix")


def test_canonical_artifact_suffix_must_match_pending_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id="ep_mismatched_suffix")
    plane.open_workflow("workflow")

    def fail_head(**_: Any) -> None:
        raise OSError("simulated crash before head")

    monkeypatch.setattr(plane, "_write_head", fail_head)
    with pytest.raises(OSError):
        plane.publish_once(
            workflow_id="workflow",
            activation_id="producer",
            attempt=1,
            port="output",
            schema="robomex.process_test.v1",
            payload={"value": 1},
        )
    lines = plane.event_log_path.read_bytes().splitlines(keepends=True)
    forged = json.loads(lines[-1])
    # This field is ignored by semantic event application, so rejection proves
    # that recovery requires the exact pending bytes, not merely a valid shape.
    forged["ignored_but_not_witnessed"] = True
    plane.event_log_path.write_bytes(
        b"".join(lines[:-1]) + _canonical_json(forged) + b"\n"
    )

    with pytest.raises(ArtifactIntegrityError, match="does not match.*pending exact"):
        EpisodeDataPlane(root, episode_id="ep_mismatched_suffix")


@pytest.mark.parametrize("mutation", ["partial", "digest", "event"])
def test_pending_artifact_witness_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id="ep_pending_tamper")
    plane.open_workflow("workflow")

    def fail_append(_: bytes) -> None:
        raise OSError("simulated crash before append")

    monkeypatch.setattr(plane, "_append_journal_bytes", fail_append)
    with pytest.raises(OSError):
        plane.publish_once(
            workflow_id="workflow",
            activation_id="producer",
            attempt=1,
            port="output",
            schema="robomex.process_test.v1",
            payload={"value": 1},
        )
    if mutation == "partial":
        plane.ledger_pending_path.write_bytes(b'{"schema":')
    else:
        pending = json.loads(plane.ledger_pending_path.read_text(encoding="utf-8"))
        if mutation == "digest":
            pending["intent_digest"] = "sha256:" + ("0" * 64)
        else:
            event = json.loads(pending["canonical_event"])
            event["valid_shape_but_not_witnessed"] = True
            pending["canonical_event"] = _canonical_json(event).decode("utf-8")
        plane.ledger_pending_path.write_bytes(_canonical_json(pending) + b"\n")

    with pytest.raises(ArtifactIntegrityError, match="Pending append"):
        EpisodeDataPlane(root, episode_id="ep_pending_tamper")


def _state_pending_fixture(
    tmp_path: Path,
    *,
    episode_id: str,
) -> tuple[Path, EmbodiedStateReducer, StateTransitionProposal]:
    root = tmp_path / "episode"
    plane = EpisodeDataPlane(root, episode_id=episode_id)
    plane.open_workflow("workflow")
    evidence = plane.publish(
        workflow_id="workflow",
        activation_id="evidence",
        attempt=1,
        port="output",
        schema="robomex.process_test.v1",
        payload={"value": 1},
    )
    reducer = EmbodiedStateReducer(root, episode_id=episode_id)
    proposal = StateTransitionProposal.register_entity(
        episode_id=episode_id,
        effect_id="register_entity",
        before_revision=0,
        source="test",
        evidence_refs=(evidence.ref,),
        entity_id="entity",
        semantic_label="object",
    )
    return root, reducer, proposal


def test_partial_state_append_recovers_exact_pending_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_id = "ep_state_partial"
    root, reducer, proposal = _state_pending_fixture(tmp_path, episode_id=episode_id)

    def partial_append(encoded: bytes) -> None:
        with reducer.event_log_path.open("ab") as stream:
            stream.write(encoded[: len(encoded) // 2])
            stream.flush()
            os.fsync(stream.fileno())
        raise OSError("simulated partial state append")

    monkeypatch.setattr(reducer, "_append_journal_bytes", partial_append)
    with pytest.raises(OSError, match="partial state append"):
        reducer.commit(proposal)

    replayed = EmbodiedStateReducer(root, episode_id=episode_id)
    assert replayed.state.revision == 1
    assert replayed.state.entity("entity") is not None
    assert not replayed.ledger_pending_path.exists()


def test_state_pending_without_append_rolls_back_then_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_id = "ep_state_rollback"
    root, reducer, proposal = _state_pending_fixture(tmp_path, episode_id=episode_id)

    def fail_append(_: bytes) -> None:
        raise OSError("simulated state pre-append crash")

    monkeypatch.setattr(reducer, "_append_journal_bytes", fail_append)
    with pytest.raises(OSError, match="pre-append crash"):
        reducer.commit(proposal)

    replayed = EmbodiedStateReducer(root, episode_id=episode_id)
    assert replayed.state.revision == 0
    assert replayed.commit(proposal).revision == 1


def test_canonical_state_suffix_without_pending_witness_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_id = "ep_state_forged"
    root, reducer, proposal = _state_pending_fixture(tmp_path, episode_id=episode_id)

    def fail_head(**_: Any) -> None:
        raise OSError("simulated state pre-head crash")

    monkeypatch.setattr(reducer, "_write_head", fail_head)
    with pytest.raises(OSError, match="pre-head crash"):
        reducer.commit(proposal)
    reducer.ledger_pending_path.unlink()

    with pytest.raises(ArtifactIntegrityError, match="without a pending intent"):
        EmbodiedStateReducer(root, episode_id=episode_id)


def test_committed_state_head_with_residual_pending_is_cleaned_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_id = "ep_state_cleanup"
    root, reducer, proposal = _state_pending_fixture(tmp_path, episode_id=episode_id)

    def fail_clear() -> None:
        raise OSError("simulated state pending-clear crash")

    monkeypatch.setattr(reducer, "_clear_pending", fail_clear)
    with pytest.raises(OSError, match="pending-clear crash"):
        reducer.commit(proposal)

    replayed = EmbodiedStateReducer(root, episode_id=episode_id)
    assert replayed.state.revision == 1
    assert not replayed.ledger_pending_path.exists()
