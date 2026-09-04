"""Run VAW episode batches and summarize terminal environment outcomes."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

DEFAULT_SUITE = "libero_object_swap"
DEFAULT_TASKS = tuple(range(10))
DEFAULT_SEEDS = (1, 2, 3)


def sweep_jobs(
    *,
    tasks: tuple[int, ...] = DEFAULT_TASKS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[dict[str, Any]]:
    jobs = []
    for task_id in tasks:
        for seed in seeds:
            jobs.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                }
            )
    return jobs


def group_jobs_by_task(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Keep one task in flight; parallelize its seeds."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    order: list[int] = []
    for job in jobs:
        task_id = int(job["task_id"])
        if task_id not in grouped:
            order.append(task_id)
        grouped[task_id].append(job)
    return [grouped[task_id] for task_id in order]


def episode_dir(root: pathlib.Path, task_id: int, seed: int) -> pathlib.Path:
    return root / f"task{task_id}_s{seed}"


def is_complete(run_dir: pathlib.Path) -> bool:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file() or not (run_dir / "steps.jsonl").is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    # cancelled/error are interrupts or crashes; resume must rerun them.
    return meta.get("terminate_mode") in {
        "goal",
        "max_turns",
        "timeout",
        "env_terminated",
    }


def read_episode_outcome(run_dir: str | pathlib.Path) -> dict[str, Any]:
    """Read facts written by the episode runtime without inferring task phases."""

    directory = pathlib.Path(run_dir)
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing episode metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    return {
        "run": directory.name,
        "suite": meta.get("suite"),
        "task_id": meta.get("task_id"),
        "seed": meta.get("seed"),
        "task_prompt": meta.get("task_prompt"),
        "terminate_mode": meta.get("terminate_mode"),
        "env_success": bool(meta.get("env_success")),
        "claimed_success": bool(meta.get("claimed_success")),
        "turns": int(meta.get("turns") or 0),
        "main_tokens": usage.get("main_total_tokens") or usage.get("total_tokens"),
        "imagination_tokens": usage.get("imagination_total_tokens") or 0,
    }


def run_episode(job: dict[str, Any]) -> dict[str, Any]:
    root = pathlib.Path(job["root"])
    task_id = int(job["task_id"])
    seed = int(job["seed"])
    run_dir = episode_dir(root, task_id, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = root / "logs" / f"task{task_id}_s{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if job.get("resume") and is_complete(run_dir):
        outcome = read_episode_outcome(run_dir)
        return {"status": "skipped", "run": str(run_dir), "outcome": outcome}
    command = [
        sys.executable,
        "-m",
        "vaw.scripts.run_context_agent",
        "--mode",
        "agent",
        "--suite",
        str(job["suite"]),
        "--task-id",
        str(task_id),
        "--seed",
        str(seed),
        "--model",
        str(job["model"]),
        "--imagination-model",
        str(job["imagination_model"]),
        "--server-url",
        str(job["server_url"]),
        "--protocol",
        str(job["protocol"]),
        "--temperature",
        str(job["temperature"]),
        "--max-tokens",
        str(job["max_tokens"]),
        "--max-turns",
        str(job["max_turns"]),
        "--max-imagination-turns",
        str(job["max_imagination_turns"]),
        "--max-time-s",
        str(job["max_time_s"]),
        "--max-physical-ops",
        str(job["max_physical_ops"]),
        "--motion-backend",
        str(job["motion_backend"]),
        "--trace-dir",
        str(run_dir),
    ]
    if job.get("record_video"):
        command.append("--record-video")
    else:
        command.append("--no-record-video")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            completed = subprocess.run(
                command,
                cwd=job["cwd"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                env=os.environ.copy(),
            )
        except Exception as exc:  # noqa: BLE001 - isolate one episode
            return {
                "status": "crash",
                "run": str(run_dir),
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": time.time() - started,
            }
    elapsed = time.time() - started
    if completed.returncode != 0 and not is_complete(run_dir):
        return {
            "status": "error",
            "run": str(run_dir),
            "returncode": completed.returncode,
            "elapsed_s": elapsed,
        }
    try:
        outcome = read_episode_outcome(run_dir)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "incomplete",
            "run": str(run_dir),
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": elapsed,
        }
    return {
        "status": "ok",
        "run": str(run_dir),
        "returncode": completed.returncode,
        "elapsed_s": elapsed,
        "outcome": outcome,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [item["outcome"] for item in results if "outcome" in item]
    successes = sum(1 for item in outcomes if item.get("env_success"))
    claimed = sum(1 for item in outcomes if item.get("claimed_success"))
    turns = [int(item["turns"]) for item in outcomes if item.get("turns") is not None]
    return {
        "episodes": len(results),
        "completed": len(outcomes),
        "ok": sum(1 for item in results if item.get("status") == "ok"),
        "skipped": sum(1 for item in results if item.get("status") == "skipped"),
        "failed": sum(1 for item in results if item.get("status") not in {"ok", "skipped"}),
        "env_success": successes,
        "env_success_rate": successes / len(outcomes) if outcomes else 0.0,
        "claimed_success": claimed,
        "mean_turns": sum(turns) / len(turns) if turns else 0.0,
        "results": results,
    }


def write_summary(root: pathlib.Path, summary: dict[str, Any]) -> pathlib.Path:
    path = root / "sweep_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split(",") if part.strip())


def _append_result(
    results: list[dict[str, Any]],
    root: pathlib.Path,
    job: dict[str, Any],
    result: dict[str, Any],
) -> None:
    results.append(result)
    print(
        f"[{result['status']}] task={job['task_id']} seed={job['seed']} "
        f"{result.get('run')}",
        flush=True,
    )
    write_summary(root, summarize(results))


def _run_batch(
    batch: list[dict[str, Any]],
    *,
    workers: int,
    root: pathlib.Path,
    results: list[dict[str, Any]],
) -> None:
    width = min(max(workers, 1), len(batch))
    if width <= 1:
        for job in batch:
            _append_result(results, root, job, run_episode(job))
        return
    with ProcessPoolExecutor(max_workers=width) as pool:
        future_map = {pool.submit(run_episode, job): job for job in batch}
        for future in as_completed(future_map):
            job = future_map[future]
            _append_result(results, root, job, future.result())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="sweep directory name under vaw/out/sweeps")
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--model", default="vapi/claude-opus-5")
    parser.add_argument("--imagination-model", default=None)
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--protocol", choices=("native", "text"), default="native")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--max-imagination-turns", type=int, default=6)
    parser.add_argument("--max-time-s", type=float, default=3600.0)
    parser.add_argument("--max-physical-ops", type=int, default=30)
    parser.add_argument("--motion-backend", default="curobo")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel episodes; with --group-by task this is seeds-in-flight per task",
    )
    parser.add_argument(
        "--group-by",
        choices=("task", "flat"),
        default="task",
        help="task: finish all seeds of one task before the next; flat: global pool",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    repo = pathlib.Path(__file__).resolve().parents[2]
    root = repo / "vaw" / "out" / "sweeps" / args.tag
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for spec in sweep_jobs(
        tasks=_parse_int_list(args.tasks),
        seeds=_parse_int_list(args.seeds),
    ):
        jobs.append(
            {
                **spec,
                "root": str(root),
                "cwd": str(repo),
                "suite": args.suite,
                "model": args.model,
                "imagination_model": args.imagination_model or args.model,
                "server_url": args.server_url,
                "protocol": args.protocol,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "max_turns": args.max_turns,
                "max_imagination_turns": args.max_imagination_turns,
                "max_time_s": args.max_time_s,
                "max_physical_ops": args.max_physical_ops,
                "motion_backend": args.motion_backend,
                "resume": args.resume,
                "record_video": args.record_video,
            }
        )
    print(
        f"[sweep] {len(jobs)} jobs -> {root} workers={args.workers} "
        f"group_by={args.group_by}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    batches = group_jobs_by_task(jobs) if args.group_by == "task" else [jobs]
    for batch in batches:
        task_ids = sorted({int(job["task_id"]) for job in batch})
        print(
            f"[sweep] batch tasks={task_ids} n={len(batch)} "
            f"workers={min(max(args.workers, 1), len(batch))}",
            flush=True,
        )
        _run_batch(batch, workers=args.workers, root=root, results=results)
    summary = summarize(results)
    write_summary(root, summary)
    print(
        f"[sweep] done ok={summary['ok']} skipped={summary['skipped']} "
        f"failed={summary['failed']} env_success={summary['env_success']}",
        flush=True,
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
