"""执行可恢复的 DAY rollout，并保存指向原始 trace 的轻量索引。

DAY 层只负责任务展开、进程调度和基础设施状态分类。完整 Canvas、Function
transaction 与模型输出继续由 Context Runtime trace 保存，本模块不复制这些内容，
也不根据动作序列推断任务阶段或失败原因。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from vaw.evolution.domain import EvolutionSpec, thaw_json
from vaw.evolution.store import GenerationStore


class DayStatus(Enum):
    """DAY 只区分可学习终局与运行基础设施问题。"""

    COMPLETED = "completed"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class DayJob:
    """一个 task/seed rollout 的冻结执行参数。"""

    episode_key: str
    suite: str
    task_id: int
    seed: int
    generation_id: str
    skill_digest: str
    experiment_root: Path
    run_root: Path
    trace_dir: Path
    log_path: Path
    runner_args: Mapping[str, Any]


@dataclass(frozen=True)
class ProcessOutcome:
    """episode 子进程的最小执行结果，便于在测试中替换真实进程。"""

    returncode: int
    wall_time_s: float
    interrupted: bool = False
    error: str | None = None


@dataclass(frozen=True)
class DayEpisodeRecord:
    """DAY index 中唯一保存的 episode 摘要。"""

    episode_key: str
    suite: str
    task_id: int
    seed: int
    generation_id: str
    skill_digest: str
    trace_dir: str
    status: str
    env_success: bool | None
    turns: int | None
    tokens: int | None
    wall_time_s: float
    terminate_mode: str | None = None
    error: str | None = None
    schema: str = "vaw-day-episode-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SPLIT_FIELDS = {
    "evolve": "evolve_tasks",
    "gate": "gate_tasks",
    "reserve": "reserve_tasks",
}
_CONTROL_CONFIG_KEYS = frozenset({"seeds", "workers", "runner_args"})
_PROTECTED_RUNNER_ARGS = frozenset(
    {
        "mode",
        "suite",
        "task_id",
        "seed",
        "trace_dir",
        "skill_root",
        "skill_generation",
        "api_key",
    }
)
_COMPLETE_TERMINATIONS = frozenset({"goal", "max_turns", "timeout", "env_terminated"})


def parse_int_list(raw: str) -> tuple[int, ...]:
    """解析逗号分隔整数并保持用户给出的顺序。"""

    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(set(values)) != len(values):
        raise ValueError("任务或 seed 列表不能包含重复值")
    return values


def select_tasks(
    spec: EvolutionSpec,
    split: str,
    requested: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """选择命名 split 的全部任务，或其显式子集。"""

    try:
        field_name = _SPLIT_FIELDS[split]
    except KeyError as exc:
        raise ValueError(f"未知 DAY split: {split}") from exc
    available = tuple(getattr(spec, field_name))
    if requested is None:
        return available
    selected = tuple(int(value) for value in requested)
    outside = sorted(set(selected) - set(available))
    if outside:
        raise ValueError(f"任务 {outside} 不属于 {split} split")
    if len(set(selected)) != len(selected):
        raise ValueError("任务列表不能包含重复值")
    return selected


def select_seeds(spec: EvolutionSpec, requested: Sequence[int] | None = None) -> tuple[int, ...]:
    """从冻结 rollout seeds 中选择本次运行子集。"""

    configured = tuple(int(value) for value in spec.rollout_config.get("seeds", (1,)))
    if not configured:
        raise ValueError("rollout_config.seeds 不能为空")
    if requested is None:
        return configured
    selected = tuple(int(value) for value in requested)
    outside = sorted(set(selected) - set(configured))
    if outside:
        raise ValueError(f"seed {outside} 不属于冻结 rollout_config.seeds")
    if len(set(selected)) != len(selected):
        raise ValueError("seed 列表不能包含重复值")
    return selected


def runner_args_from_spec(spec: EvolutionSpec) -> dict[str, Any]:
    """读取可扩展 runner 参数，同时保护 DAY 自己管理的 episode 身份。"""

    config = thaw_json(spec.rollout_config)
    nested = config.get("runner_args")
    if nested is None:
        runner_args = {
            key: value for key, value in config.items() if key not in _CONTROL_CONFIG_KEYS
        }
    elif isinstance(nested, dict):
        runner_args = dict(nested)
    else:
        raise ValueError("rollout_config.runner_args 必须是 object")
    protected = sorted(set(runner_args) & _PROTECTED_RUNNER_ARGS)
    if protected:
        raise ValueError(f"runner_args 不能覆盖 DAY 身份参数: {protected}")
    return runner_args


def _cli_args(options: Mapping[str, Any]) -> list[str]:
    """将冻结配置转换为 run_context_agent 参数，不复制其完整参数 schema。"""

    result: list[str] = []
    for key, value in options.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            result.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, (list, tuple)):
            result.extend((flag, ",".join(str(item) for item in value)))
        else:
            result.extend((flag, str(value)))
    return result


def episode_command(job: DayJob) -> list[str]:
    """构造真实 Runtime 命令；generation 来源由 DAY 强制注入。"""

    return [
        sys.executable,
        "-m",
        "vaw.scripts.run_context_agent",
        "--mode",
        "agent",
        "--suite",
        job.suite,
        "--task-id",
        str(job.task_id),
        "--seed",
        str(job.seed),
        "--trace-dir",
        str(job.trace_dir),
        "--skill-root",
        str(job.experiment_root),
        "--skill-generation",
        job.generation_id,
        *_cli_args(job.runner_args),
    ]


def run_subprocess(job: DayJob) -> ProcessOutcome:
    """运行一个真实 episode；stdout/stderr 只进入独立日志。"""

    job.trace_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        with job.log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                episode_command(job),
                cwd=Path(__file__).resolve().parents[2],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=os.environ.copy(),
            )
    except KeyboardInterrupt:
        return ProcessOutcome(-1, time.monotonic() - started, interrupted=True)
    except Exception as exc:  # noqa: BLE001 - 子进程故障必须隔离到单 episode
        return ProcessOutcome(
            -1,
            time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    return ProcessOutcome(completed.returncode, time.monotonic() - started)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 必须是 object: {path}")
    return payload


def _trace_facts(trace_dir: Path) -> dict[str, Any]:
    """只读取 Runtime 已写入的终局事实，不解释 Function 序列。"""

    meta = _load_json(trace_dir / "meta.json")
    terminate_mode = meta.get("terminate_mode")
    if terminate_mode not in _COMPLETE_TERMINATIONS:
        raise ValueError(f"trace 没有可学习终局: {terminate_mode!r}")
    if not (trace_dir / "steps.jsonl").is_file():
        raise FileNotFoundError(f"trace 缺少 steps.jsonl: {trace_dir}")
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    token_value = usage.get("total_tokens")
    return {
        "terminate_mode": str(terminate_mode),
        "env_success": (
            bool(meta["env_success"]) if meta.get("env_success") is not None else None
        ),
        "turns": int(meta["turns"]) if meta.get("turns") is not None else None,
        "tokens": int(token_value) if token_value is not None else None,
    }


def trace_is_complete(trace_dir: Path) -> bool:
    try:
        _trace_facts(trace_dir)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return True


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def compile_record(job: DayJob, process: ProcessOutcome) -> DayEpisodeRecord:
    """把进程结果与 trace 终局压缩为允许字段集合。"""

    facts: dict[str, Any] = {}
    status = DayStatus.INFRASTRUCTURE_ERROR
    error = process.error
    if process.interrupted:
        status = DayStatus.INTERRUPTED
    else:
        try:
            facts = _trace_facts(job.trace_dir)
            status = DayStatus.COMPLETED
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
            if error is None:
                error = f"{type(exc).__name__}: {exc}"
    if process.returncode != 0 and status is DayStatus.COMPLETED:
        # Runtime 已写出完整终局时，以 trace 为准；例如 goal 之外的 CLI 退出约定不应丢数据。
        error = None
    return DayEpisodeRecord(
        episode_key=job.episode_key,
        suite=job.suite,
        task_id=job.task_id,
        seed=job.seed,
        generation_id=job.generation_id,
        skill_digest=job.skill_digest,
        trace_dir=_relative_or_absolute(job.trace_dir, job.experiment_root),
        status=status.value,
        env_success=facts.get("env_success"),
        turns=facts.get("turns"),
        tokens=facts.get("tokens"),
        wall_time_s=round(float(process.wall_time_s), 3),
        terminate_mode=facts.get("terminate_mode"),
        error=error,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """原子发布 run manifest 和索引，避免中断留下半份 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _run_manifest(
    *,
    spec: EvolutionSpec,
    suite: str,
    split: str,
    tasks: Sequence[int],
    seeds: Sequence[int],
    generation_id: str,
    skill_digest: str,
    runner_args: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "vaw-day-run-v1",
        "experiment_id": spec.experiment_id,
        "suite": suite,
        "split": split,
        "tasks": list(tasks),
        "seeds": list(seeds),
        "generation_id": generation_id,
        "skill_digest": skill_digest,
        "runner_args": dict(runner_args),
    }


def _prepare_run(run_root: Path, manifest: Mapping[str, Any]) -> None:
    manifest_path = run_root / "run.json"
    if manifest_path.is_file():
        if _load_json(manifest_path) != dict(manifest):
            raise ValueError(f"DAY run 配置与已有 run.json 不一致: {run_root}")
        return
    if run_root.exists() and any(run_root.iterdir()):
        raise ValueError(f"非空 DAY run 目录缺少 run.json: {run_root}")
    _atomic_json(manifest_path, manifest)


def build_jobs(
    *,
    store: GenerationStore,
    run_root: Path,
    split: str,
    tasks: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    generation_id: str | None = None,
) -> tuple[DayJob, ...]:
    """从冻结 spec 展开任务；CLI 只能选择已有 split 的子集。"""

    spec = store.read_spec()
    selected_tasks = select_tasks(spec, split, tasks)
    selected_seeds = select_seeds(spec, seeds)
    selected_generation = generation_id or store.active_generation()
    manifest = store.read_manifest(selected_generation)
    runner_args = runner_args_from_spec(spec)
    _prepare_run(
        run_root,
        _run_manifest(
            spec=spec,
            suite=spec.evolve_suite,
            split=split,
            tasks=selected_tasks,
            seeds=selected_seeds,
            generation_id=selected_generation,
            skill_digest=manifest.skill_digest,
            runner_args=runner_args,
        ),
    )
    jobs: list[DayJob] = []
    for task_id in selected_tasks:
        for seed in selected_seeds:
            episode_key = f"{spec.evolve_suite}:t{task_id}:s{seed}"
            stem = f"task{task_id}_s{seed}"
            jobs.append(
                DayJob(
                    episode_key=episode_key,
                    suite=spec.evolve_suite,
                    task_id=task_id,
                    seed=seed,
                    generation_id=selected_generation,
                    skill_digest=manifest.skill_digest,
                    experiment_root=store.root,
                    run_root=run_root,
                    trace_dir=run_root / "traces" / stem,
                    log_path=run_root / "logs" / f"{stem}.log",
                    runner_args=runner_args,
                )
            )
    return tuple(jobs)


def build_heldout_jobs(
    *,
    store: GenerationStore,
    run_root: Path,
    suite: str,
    tasks: Sequence[int],
    seeds: Sequence[int],
    generation_id: str | None = None,
) -> tuple[DayJob, ...]:
    """为 sealed held-out suite 构造只评测、不学习的 episode。

    该入口刻意不复用 evolve/gate split。调用方只能选择实验创建时已经冻结为
    held-out 的 suite，输出也标为 ``transfer``，避免后续 M3/M4 误把它当 DAY 证据。
    """

    spec = store.read_spec()
    if suite not in spec.heldout_suites:
        raise ValueError(f"suite 未冻结为 held-out: {suite}")
    selected_tasks = tuple(int(value) for value in tasks)
    selected_seeds = tuple(int(value) for value in seeds)
    if not selected_tasks or not selected_seeds:
        raise ValueError("held-out tasks 和 seeds 不能为空")
    if len(set(selected_tasks)) != len(selected_tasks) or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("held-out tasks 和 seeds 不能重复")

    selected_generation = generation_id or store.active_generation()
    manifest = store.read_manifest(selected_generation)
    runner_args = runner_args_from_spec(spec)
    _prepare_run(
        run_root,
        _run_manifest(
            spec=spec,
            suite=suite,
            split="transfer",
            tasks=selected_tasks,
            seeds=selected_seeds,
            generation_id=selected_generation,
            skill_digest=manifest.skill_digest,
            runner_args=runner_args,
        ),
    )
    return tuple(
        DayJob(
            episode_key=f"{suite}:t{task_id}:s{seed}",
            suite=suite,
            task_id=task_id,
            seed=seed,
            generation_id=selected_generation,
            skill_digest=manifest.skill_digest,
            experiment_root=store.root,
            run_root=run_root,
            trace_dir=run_root / "traces" / f"task{task_id}_s{seed}",
            log_path=run_root / "logs" / f"task{task_id}_s{seed}.log",
            runner_args=runner_args,
        )
        for task_id in selected_tasks
        for seed in selected_seeds
    )


def _existing_records(run_root: Path) -> dict[str, DayEpisodeRecord]:
    """读取上次索引；损坏索引不会覆盖 trace 对完成状态的判断。"""

    path = run_root / "day_index.json"
    if not path.is_file():
        return {}
    try:
        payload = _load_json(path)
        return {
            str(item["episode_key"]): DayEpisodeRecord(**item)
            for item in payload.get("episodes", ())
            if isinstance(item, dict)
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def _write_index(run_root: Path, records: Sequence[DayEpisodeRecord]) -> None:
    ordered = sorted(records, key=lambda item: (item.task_id, item.seed))
    completed = sum(item.status == DayStatus.COMPLETED.value for item in ordered)
    successes = sum(item.env_success is True for item in ordered)
    _atomic_json(
        run_root / "day_index.json",
        {
            "schema": "vaw-day-index-v1",
            "episodes": [item.to_dict() for item in ordered],
            "summary": {
                "scheduled": len(ordered),
                "completed": completed,
                "infrastructure_error": sum(
                    item.status == DayStatus.INFRASTRUCTURE_ERROR.value for item in ordered
                ),
                "interrupted": sum(
                    item.status == DayStatus.INTERRUPTED.value for item in ordered
                ),
                "env_success": successes,
                "env_success_rate": successes / completed if completed else None,
            },
        },
    )


def run_day(
    jobs: Sequence[DayJob],
    *,
    resume: bool,
    workers: int,
    episode_runner: Callable[[DayJob], ProcessOutcome] = run_subprocess,
) -> tuple[DayEpisodeRecord, ...]:
    """执行 DAY jobs；索引由父进程统一写入，避免并发覆盖。"""

    if not jobs:
        return ()
    run_roots = {job.run_root.resolve() for job in jobs}
    if len(run_roots) != 1:
        raise ValueError("一次 run_day 只能写入一个 run_root")
    if not resume:
        occupied = [job.trace_dir for job in jobs if job.trace_dir.exists() and any(job.trace_dir.iterdir())]
        if occupied:
            raise FileExistsError(
                f"DAY trace 已存在；请使用 --resume 或新的 --run-name: {occupied[0]}"
            )
    records: list[DayEpisodeRecord] = []
    pending: list[DayJob] = []
    previous = _existing_records(jobs[0].run_root) if resume else {}
    for job in jobs:
        if resume and trace_is_complete(job.trace_dir):
            existing = previous.get(job.episode_key)
            records.append(
                existing
                if existing is not None and existing.status == DayStatus.COMPLETED.value
                else compile_record(job, ProcessOutcome(0, 0.0))
            )
        else:
            pending.append(job)

    width = min(max(int(workers), 1), len(pending)) if pending else 0
    if width == 1:
        for job in pending:
            record = compile_record(job, episode_runner(job))
            records.append(record)
            _write_index(job.run_root, records)
    elif width > 1:
        with ThreadPoolExecutor(max_workers=width) as pool:
            futures = {pool.submit(episode_runner, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    process = future.result()
                except Exception as exc:  # noqa: BLE001 - 单 episode 不能终止整批 DAY
                    process = ProcessOutcome(
                        -1,
                        0.0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                record = compile_record(job, process)
                records.append(record)
                _write_index(job.run_root, records)
    _write_index(jobs[0].run_root, records)
    return tuple(sorted(records, key=lambda item: (item.task_id, item.seed)))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(_SPLIT_FIELDS), default="evolve")
    parser.add_argument("--tasks", help="命名 split 内的逗号分隔任务子集")
    parser.add_argument("--seeds", help="冻结 rollout seeds 内的逗号分隔子集")
    parser.add_argument("--generation", help="默认使用 active generation")
    parser.add_argument("--run-name", help="默认 day-<generation>-<split>")
    parser.add_argument("--output-root", type=Path, help="默认 <experiment>/runs/day")
    parser.add_argument("--workers", type=int, help="默认读取 rollout_config.workers 或 1")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    store = GenerationStore.open(args.experiment_root)
    spec = store.read_spec()
    generation = args.generation or store.active_generation()
    tasks = parse_int_list(args.tasks) if args.tasks else None
    seeds = parse_int_list(args.seeds) if args.seeds else None
    output_root = args.output_root or store.root / "runs" / "day"
    run_name = args.run_name or f"day-{generation}-{args.split}"
    if run_name in {".", ".."} or Path(run_name).name != run_name:
        raise SystemExit("--run-name 必须是单段目录名")
    run_root = output_root / run_name
    jobs = build_jobs(
        store=store,
        run_root=run_root,
        split=args.split,
        tasks=tasks,
        seeds=seeds,
        generation_id=generation,
    )
    workers = args.workers
    if workers is None:
        workers = int(spec.rollout_config.get("workers", 1))
    print(
        f"[day] split={args.split} generation={generation} jobs={len(jobs)} "
        f"workers={workers} root={run_root}",
        flush=True,
    )
    records = run_day(jobs, resume=args.resume, workers=workers)
    failed = sum(item.status != DayStatus.COMPLETED.value for item in records)
    print(f"[day] completed={len(records) - failed} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DayEpisodeRecord",
    "DayJob",
    "DayStatus",
    "ProcessOutcome",
    "build_jobs",
    "build_heldout_jobs",
    "compile_record",
    "episode_command",
    "parse_int_list",
    "run_day",
    "runner_args_from_spec",
    "select_seeds",
    "select_tasks",
    "trace_is_complete",
]
