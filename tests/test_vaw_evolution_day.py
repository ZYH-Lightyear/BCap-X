from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaw.evolution import EvolutionSpec, GenerationStore
from vaw.evolution.day import (
    DayStatus,
    ProcessOutcome,
    build_heldout_jobs,
    build_jobs,
    episode_command,
    run_day,
    runner_args_from_spec,
    select_seeds,
    select_tasks,
)
from vaw.evolution.cycle import CyclePlan, MultiAgentCycle
from vaw.evolution.gate import load_day_index


def _write_skill(root: Path) -> None:
    directory = root / "test-skill"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: Test\ndescription: Test skill.\n---\n\n# Test\n",
        encoding="utf-8",
    )


def _store(tmp_path: Path) -> GenerationStore:
    skills = tmp_path / "skills"
    _write_skill(skills)
    spec = EvolutionSpec(
        experiment_id="day-test",
        base_generation="g000",
        evolve_suite="libero_90",
        evolve_tasks=(0, 3, 8),
        gate_tasks=(1,),
        reserve_tasks=(2,),
        heldout_suites=("libero_object_task",),
        rollout_config={
            "seeds": [1, 2],
            "workers": 2,
            "runner_args": {
                "model": "fake/model",
                "max_turns": 16,
                "record_video": False,
            },
        },
    )
    return GenerationStore.initialize(tmp_path / "experiment", spec, skills)


def _write_complete_trace(job, *, success: bool, turns: int = 7) -> None:
    job.trace_dir.mkdir(parents=True, exist_ok=True)
    (job.trace_dir / "steps.jsonl").write_text("{}\n", encoding="utf-8")
    (job.trace_dir / "meta.json").write_text(
        json.dumps(
            {
                "terminate_mode": "goal" if success else "max_turns",
                "env_success": success,
                "turns": turns,
                "usage": {"total_tokens": 1234},
                # 私有字段可以存在于原 trace，但不得复制进 DAY index。
                "depth": [[1.0]],
                "camera_matrix": [[1.0]],
                "reward": 1.0,
            }
        ),
        encoding="utf-8",
    )


def test_named_splits_allow_only_explicit_subsets(tmp_path: Path) -> None:
    spec = _store(tmp_path).read_spec()

    assert select_tasks(spec, "evolve") == (0, 3, 8)
    assert select_tasks(spec, "evolve", (8, 0)) == (8, 0)
    assert select_tasks(spec, "gate") == (1,)
    assert select_seeds(spec, (2,)) == (2,)
    with pytest.raises(ValueError, match="不属于 evolve"):
        select_tasks(spec, "evolve", (1,))
    with pytest.raises(ValueError, match="不属于冻结"):
        select_seeds(spec, (3,))


def test_build_jobs_freezes_selection_and_injects_generation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_root = tmp_path / "day-run"

    jobs = build_jobs(
        store=store,
        run_root=run_root,
        split="evolve",
        tasks=(3,),
        seeds=(2,),
    )

    assert len(jobs) == 1
    job = jobs[0]
    command = episode_command(job)
    assert job.episode_key == "libero_90:t3:s2"
    assert command[command.index("--skill-generation") + 1] == "g000"
    assert command[command.index("--skill-root") + 1] == str(store.root)
    assert "--no-record-video" in command
    assert json.loads((run_root / "run.json").read_text())["tasks"] == [3]

    with pytest.raises(ValueError, match="run.json 不一致"):
        build_jobs(
            store=store,
            run_root=run_root,
            split="evolve",
            tasks=(0,),
            seeds=(2,),
        )


def test_heldout_jobs_are_separate_from_learning_splits(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_root = tmp_path / "transfer"
    jobs = build_heldout_jobs(
        store=store,
        run_root=run_root,
        suite="libero_object_task",
        tasks=(0, 4),
        seeds=(1,),
    )

    assert [job.episode_key for job in jobs] == [
        "libero_object_task:t0:s1",
        "libero_object_task:t4:s1",
    ]
    run = json.loads((run_root / "run.json").read_text())
    assert run["split"] == "transfer"
    assert run["suite"] == "libero_object_task"
    with pytest.raises(ValueError, match="未冻结为 held-out"):
        build_heldout_jobs(
            store=store,
            run_root=tmp_path / "leak",
            suite="libero_90",
            tasks=(0,),
            seeds=(1,),
        )


def test_runner_config_is_extensible_but_cannot_override_episode_identity(
    tmp_path: Path,
) -> None:
    spec = _store(tmp_path).read_spec()
    assert runner_args_from_spec(spec) == {
        "model": "fake/model",
        "max_turns": 16,
        "record_video": False,
    }

    bad = EvolutionSpec(
        experiment_id="bad-day",
        base_generation="g000",
        evolve_suite="libero_90",
        evolve_tasks=(0,),
        gate_tasks=(),
        reserve_tasks=(),
        heldout_suites=(),
        rollout_config={"runner_args": {"task_id": 9, "api_key": "secret"}},
    )
    with pytest.raises(ValueError, match="不能覆盖"):
        runner_args_from_spec(bad)


def test_day_resume_uses_complete_trace_and_index_stays_thin(tmp_path: Path) -> None:
    store = _store(tmp_path)
    jobs = build_jobs(
        store=store,
        run_root=tmp_path / "day-run",
        split="evolve",
        tasks=(0, 3),
        seeds=(1,),
    )
    calls: list[str] = []

    def fake_runner(job):
        calls.append(job.episode_key)
        _write_complete_trace(job, success=job.task_id == 0)
        return ProcessOutcome(returncode=0, wall_time_s=1.25)

    first = run_day(jobs, resume=True, workers=2, episode_runner=fake_runner)
    second = run_day(jobs, resume=True, workers=2, episode_runner=fake_runner)

    assert len(calls) == 2
    assert first == second
    assert [record.status for record in first] == [
        DayStatus.COMPLETED.value,
        DayStatus.COMPLETED.value,
    ]
    index = json.loads((jobs[0].run_root / "day_index.json").read_text())
    assert index["summary"]["env_success"] == 1
    assert index["summary"]["completed"] == 2
    serialized = json.dumps(index)
    for forbidden in (
        "depth",
        "camera_matrix",
        "raw_cloud",
        "mask",
        "reward",
        "function_calls",
        "canvas",
    ):
        assert forbidden not in serialized

    with pytest.raises(FileExistsError, match="--resume"):
        run_day(jobs, resume=False, workers=1, episode_runner=fake_runner)


def test_incomplete_and_interrupted_runs_are_not_resumed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    jobs = build_jobs(
        store=store,
        run_root=tmp_path / "day-run",
        split="evolve",
        tasks=(8,),
        seeds=(1, 2),
    )

    def failed_runner(job):
        if job.seed == 1:
            return ProcessOutcome(1, 0.2, error="service unavailable")
        return ProcessOutcome(-1, 0.1, interrupted=True)

    records = run_day(jobs, resume=True, workers=1, episode_runner=failed_runner)

    assert [record.status for record in records] == [
        DayStatus.INFRASTRUCTURE_ERROR.value,
        DayStatus.INTERRUPTED.value,
    ]
    assert records[0].env_success is None
    assert records[0].error == "service unavailable"
    assert records[1].terminate_mode is None


def test_multi_agent_cycle_freezes_identity_and_resumes_day(tmp_path: Path) -> None:
    store = _store(tmp_path)
    plan = CyclePlan(
        cycle_id="smoke-cycle",
        parent_generation="g000",
        candidate_generation="g001",
        mutation_id="m001",
        tasks=(0, 3),
        seeds=(1,),
    )
    cycle = MultiAgentCycle(store, plan, resume=False)
    calls: list[str] = []

    def fake_runner(job):
        calls.append(job.episode_key)
        _write_complete_trace(job, success=job.task_id == 0)
        return ProcessOutcome(0, 0.1)

    index = cycle.run_baseline(workers=2, episode_runner=fake_runner)
    resumed = MultiAgentCycle(store, plan, resume=True)
    resumed.run_baseline(workers=2, episode_runner=fake_runner)

    assert index.is_file()
    loaded = load_day_index(index)
    assert loaded[0, 1].trace_dir == jobs_trace(store, "smoke-cycle", 0)
    assert len(calls) == 2
    assert resumed.status()["stages"]["baseline"] is True
    with pytest.raises(ValueError, match="不一致"):
        MultiAgentCycle(
            store,
            CyclePlan(
                cycle_id="smoke-cycle",
                parent_generation="g000",
                candidate_generation="g002",
                mutation_id="m001",
                tasks=(0, 3),
                seeds=(1,),
            ),
            resume=True,
        )


def jobs_trace(store: GenerationStore, cycle_id: str, task_id: int) -> Path:
    """返回 cycle DAY 的预期 trace；用于防止路径解析再次依赖固定层数。"""

    return (
        store.root
        / "cycles"
        / cycle_id
        / "day"
        / "baseline"
        / "traces"
        / f"task{task_id}_s1"
    ).resolve()
