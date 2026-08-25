from __future__ import annotations

import pytest

from vaw.evolution.compare import compare_summaries, mcnemar, seed_phase_variance


def _fit(task_id: int, seed: int, **phases: bool) -> dict:
    defaults = {"reach": False, "grasp": False, "transport": False, "place": False}
    defaults.update(phases)
    return {
        "status": "ok",
        "fitness": {"task_id": task_id, "seed": seed, "phases": defaults},
    }


def test_mcnemar_is_1_when_no_discordant_pairs() -> None:
    assert mcnemar(0, 0) == 1.0


def test_paired_phase_flips_and_accept_rule() -> None:
    baseline = {
        "results": [
            _fit(0, 1, reach=True),
            _fit(1, 1, reach=True, grasp=True),
        ]
    }
    candidate = {
        "results": [
            _fit(0, 1, reach=True, grasp=True),
            _fit(1, 1, reach=True, grasp=True, transport=True, place=True),
        ]
    }
    report = compare_summaries(baseline, candidate)
    assert report["paired_episodes"] == 2
    assert report["phases"]["grasp"]["improved"] == 1
    assert report["phases"]["grasp"]["regressed"] == 0
    assert report["accept"] is False  # reach did not improve


def test_seed_phase_variance_groups_by_task() -> None:
    summary = {
        "results": [
            _fit(0, 1, place=True),
            _fit(0, 2, place=False),
            _fit(0, 3, place=True),
        ]
    }
    rows = seed_phase_variance(summary)["tasks"]
    assert rows[0]["task_id"] == 0
    assert rows[0]["place"]["rate"] == pytest.approx(2.0 / 3.0)
