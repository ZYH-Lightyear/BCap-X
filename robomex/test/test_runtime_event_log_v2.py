from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from robomex.runtime.event_log import EventLogIntegrityError, PersistentTypedEventBus
from robomex.runtime.events import NodeOutcomeEvent


def _event(event_id: str = "evt-1", *, reason: str | None = None) -> NodeOutcomeEvent:
    return NodeOutcomeEvent(
        event_id=event_id,
        episode_id="ep",
        workflow_id="wf",
        source="test",
        activation_id="node",
        node_id="node",
        outcome="success",
        reason=reason,
        graph_revision=1,
    )


def test_durable_bus_flushes_replays_and_keeps_exact_duplicates_idempotent(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    bus = PersistentTypedEventBus(path)
    event = _event()
    assert bus.publish(event)
    assert not bus.publish(event)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    replay = PersistentTypedEventBus(path)
    assert replay.offset == 1
    assert replay.history == (event,)


def test_conflicting_id_and_corrupt_persisted_event_fail_closed(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    bus = PersistentTypedEventBus(path)
    event = _event()
    bus.publish(event)
    with pytest.raises(Exception, match="reused with different content"):
        bus.publish(event.model_copy(update={"reason": "changed"}))

    path.write_text(json.dumps({"kind": "future_event"}) + "\n", encoding="utf-8")
    with pytest.raises(EventLogIntegrityError, match="line 1"):
        PersistentTypedEventBus(path)


def test_independent_bus_instances_serialize_one_shared_event_log(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    buses = (PersistentTypedEventBus(path), PersistentTypedEventBus(path))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(
            pool.map(
                lambda index: buses[index % 2].publish(_event(f"evt-{index}")),
                range(20),
            )
        )

    assert all(results)
    replay = PersistentTypedEventBus(path)
    assert replay.offset == 20
    assert {event.event_id for event in replay.history} == {
        f"evt-{index}" for index in range(20)
    }
