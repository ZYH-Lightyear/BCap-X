from __future__ import annotations

import json
from pathlib import Path

from vaw.evolution.sweep import (
    episode_dir,
    group_jobs_by_task,
    is_complete,
    summarize,
    sweep_jobs,
    write_summary,
)


def test_sweep_jobs_are_the_task_seed_product() -> None:
    jobs = sweep_jobs(tasks=(0, 1), seeds=(1, 2))
    assert [(job["task_id"], job["seed"]) for job in jobs] == [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
    ]


def test_group_jobs_by_task_keeps_task_order() -> None:
    jobs = sweep_jobs(tasks=(2, 0), seeds=(1, 2, 3))
    batches = group_jobs_by_task(jobs)
    assert [job["task_id"] for job in batches[0]] == [2, 2, 2]
    assert [job["seed"] for job in batches[0]] == [1, 2, 3]
    assert [job["task_id"] for job in batches[1]] == [0, 0, 0]


def test_resume_skips_only_finished_meta(tmp_path: Path) -> None:
    done = episode_dir(tmp_path, 0, 1)
    done.mkdir(parents=True)
    (done / "steps.jsonl").write_text("{}\n", encoding="utf-8")
    (done / "meta.json").write_text(
        json.dumps({"terminate_mode": "max_turns"}), encoding="utf-8"
    )
    live = episode_dir(tmp_path, 0, 2)
    live.mkdir(parents=True)
    (live / "steps.jsonl").write_text("{}\n", encoding="utf-8")
    (live / "meta.json").write_text("{}", encoding="utf-8")
    cancelled = episode_dir(tmp_path, 0, 3)
    cancelled.mkdir(parents=True)
    (cancelled / "steps.jsonl").write_text("{}\n", encoding="utf-8")
    (cancelled / "meta.json").write_text(
        json.dumps({"terminate_mode": "cancelled"}), encoding="utf-8"
    )
    assert is_complete(done)
    assert not is_complete(live)
    assert not is_complete(cancelled)


def test_sweep_summary_round_trip(tmp_path: Path) -> None:
    summary = summarize(
        [
            {
                "status": "ok",
                "fitness": {
                    "env_success": True,
                    "phases": {
                        "reach": True,
                        "grasp": True,
                        "transport": False,
                        "place": True,
                    },
                },
            },
            {"status": "error"},
        ]
    )
    path = write_summary(tmp_path, summary)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["episodes"] == 2
    assert loaded["ok"] == 1
    assert loaded["failed"] == 1
    assert loaded["env_success"] == 1
    assert loaded["phase_rates"]["place"] == 1.0
    assert loaded["phase_rates"]["transport"] == 0.0
