from __future__ import annotations

import pytest

from robomex.orchestration.manager import SwarmManagerSession
from robomex.orchestration.manager_store import ManagerSessionLedger, ManagerStoreError


def _session(revision: int = 1) -> SwarmManagerSession:
    return SwarmManagerSession(
        session_id="manager",
        episode_id="ep",
        workflow_id="wf",
        intent_id="intent",
        record_revision=revision,
    )


def test_manager_ledger_is_append_only_idempotent_and_replayable(tmp_path) -> None:
    path = tmp_path / "manager.jsonl"
    ledger = ManagerSessionLedger(path)
    first = _session()
    second = first.close()
    assert ledger.append(first)
    assert not ledger.append(first)
    assert ledger.append(second)

    replay = ManagerSessionLedger(path)
    assert replay.records == (first, second)
    assert replay.latest("manager") == second


def test_manager_ledger_rejects_revision_gap_and_corruption(tmp_path) -> None:
    path = tmp_path / "manager.jsonl"
    ledger = ManagerSessionLedger(path)
    ledger.append(_session())
    with pytest.raises(ManagerStoreError, match="exactly one"):
        ledger.append(_session(3))

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ManagerStoreError, match="line 1"):
        ManagerSessionLedger(path)
