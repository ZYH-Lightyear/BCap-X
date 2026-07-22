from __future__ import annotations

import os
from pathlib import Path

import pytest

from robomex.elastic import (
    ActivationSpec,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    RunnerKind,
)
from robomex.runtime.activation import ActivationScheduler, SchedulerState
from robomex.runtime.scheduler_store import (
    JsonlSchedulerStateStore,
    SchedulerCommit,
    SchedulerStoreIntegrityError,
)


def _state(revision: int) -> SchedulerState:
    graph = ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="incremental-store",
            entry_activation="done",
            terminal_activations=("done",),
            activations=(
                ActivationSpec(
                    activation_id="done",
                    runner_kind=RunnerKind.DETERMINISTIC_GATE,
                    runner_ref="test.done",
                ),
            ),
        )
    )
    state = ActivationScheduler(
        episode_id="episode",
        workflow_id="workflow",
        graph=graph,
    ).export_state()
    return state.model_copy(update={"state_revision": revision})


def _append(store: JsonlSchedulerStateStore, revision: int) -> None:
    assert store.append(_state(revision), reason=f"revision-{revision}")


def test_two_instances_incrementally_observe_interleaved_appends(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "scheduler.jsonl"
    first = JsonlSchedulerStateStore(journal)
    second = JsonlSchedulerStateStore(journal)

    _append(first, 0)
    assert second.latest(episode_id="episode", workflow_id="workflow") is not None
    _append(second, 1)
    assert first.latest(episode_id="episode", workflow_id="workflow").state.state_revision == 1
    _append(first, 2)

    assert [item.state.state_revision for item in second.history()] == [0, 1, 2]
    restarted = JsonlSchedulerStateStore(journal)
    assert [item.state.state_revision for item in restarted.history()] == [0, 1, 2]


def test_incremental_sync_never_revalidates_cached_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "scheduler.jsonl"
    writer = JsonlSchedulerStateStore(journal)
    _append(writer, 0)
    _append(writer, 1)

    calls = 0
    original = SchedulerCommit.model_validate_json.__func__

    def count_validation(cls, json_data, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(cls, json_data, *args, **kwargs)

    monkeypatch.setattr(
        SchedulerCommit,
        "model_validate_json",
        classmethod(count_validation),
    )
    reader = JsonlSchedulerStateStore(journal)
    assert calls == 2  # restart deliberately validates the complete journal

    calls = 0
    assert reader.latest(episode_id="episode", workflow_id="workflow") is not None
    assert len(reader.history()) == 2
    assert calls == 0

    _append(writer, 2)
    assert reader.latest(episode_id="episode", workflow_id="workflow").state.state_revision == 2
    assert calls == 1
    assert len(reader.history()) == 3
    assert calls == 1


@pytest.mark.parametrize(
    ("tail", "message"),
    [
        (b'{"schema_version":', "incomplete commit"),
        (b"{not-json}\n", "invalid scheduler commit"),
    ],
)
def test_partial_or_corrupt_tail_fails_closed(
    tmp_path: Path,
    tail: bytes,
    message: str,
) -> None:
    journal = tmp_path / "scheduler.jsonl"
    store = JsonlSchedulerStateStore(journal)
    _append(store, 0)
    with journal.open("ab") as stream:
        stream.write(tail)

    with pytest.raises(SchedulerStoreIntegrityError, match=message):
        store.latest(episode_id="episode", workflow_id="workflow")
    with pytest.raises(SchedulerStoreIntegrityError, match=message):
        JsonlSchedulerStateStore(journal)


def test_truncation_is_not_reinterpreted_as_a_shorter_history(tmp_path: Path) -> None:
    journal = tmp_path / "scheduler.jsonl"
    store = JsonlSchedulerStateStore(journal)
    _append(store, 0)
    _append(store, 1)
    first_line = journal.read_bytes().splitlines(keepends=True)[0]
    journal.write_bytes(first_line)

    with pytest.raises(SchedulerStoreIntegrityError, match="truncated"):
        store.history()


def test_replaced_journal_is_rejected_by_an_open_store(tmp_path: Path) -> None:
    journal = tmp_path / "scheduler.jsonl"
    store = JsonlSchedulerStateStore(journal)
    _append(store, 0)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(journal.read_bytes())
    os.replace(replacement, journal)

    with pytest.raises(SchedulerStoreIntegrityError, match="replaced"):
        store.history()
    # A genuine restart has no stale cache to trust and validates every line.
    assert len(JsonlSchedulerStateStore(journal).history()) == 1


def test_same_size_tamper_is_detected_even_if_mtime_is_restored(tmp_path: Path) -> None:
    journal = tmp_path / "scheduler.jsonl"
    store = JsonlSchedulerStateStore(journal)
    _append(store, 0)
    before = journal.stat()
    raw = journal.read_bytes()
    tampered = raw.replace(b'"reason":"revision-0"', b'"reason":"tampered-0"')
    assert len(tampered) == len(raw)
    journal.write_bytes(tampered)
    os.utime(journal, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(SchedulerStoreIntegrityError, match="content changed"):
        store.latest(episode_id="episode", workflow_id="workflow")
    with pytest.raises(SchedulerStoreIntegrityError, match="invalid scheduler commit"):
        JsonlSchedulerStateStore(journal)
