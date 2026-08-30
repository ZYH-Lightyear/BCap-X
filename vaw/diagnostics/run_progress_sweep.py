"""Run the absolute-progress critic over every trace in one VAW sweep."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vaw.diagnostics.progress_video import render_progress_video
from vaw.diagnostics.run_progress_demo import (
    DEFAULT_MODEL,
    DEFAULT_SERVER,
    score_trace,
)


def discover_traces(sweep_root: Path) -> list[Path]:
    """Return only direct ``<suite>/task*_s*`` episode directories."""

    traces: list[Path] = []
    for suite_dir in sorted(path for path in sweep_root.iterdir() if path.is_dir()):
        for trace_dir in sorted(suite_dir.glob("task*_s*")):
            if not trace_dir.is_dir():
                continue
            required = ("meta.json", "steps.jsonl", "video_agentview.mp4")
            if all((trace_dir / filename).is_file() for filename in required):
                traces.append(trace_dir.resolve())
    return traces


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_summary(
    *,
    path: Path,
    sweep_root: Path,
    output_name: str,
    model: str,
    results: dict[str, dict[str, Any]],
    expected_runs: int,
) -> None:
    records = [results[key] for key in sorted(results)]
    complete = sum(record.get("status") == "complete" for record in records)
    failed = sum(record.get("status") == "failed" for record in records)
    _write_json(
        path,
        {
            "schema": "vaw-absolute-progress-sweep-v1",
            "updated_at": datetime.now(UTC).isoformat(),
            "sweep_root": str(sweep_root),
            "output_name": output_name,
            "model": model,
            "expected_runs": expected_runs,
            "complete_runs": complete,
            "failed_runs": failed,
            "pending_runs": expected_runs - complete - failed,
            "records": records,
        },
    )


def _existing_record(trace_dir: Path, output_dir: Path) -> dict[str, Any] | None:
    progress_path = output_dir / "progress.json"
    video_path = output_dir / "progress_overlay.mp4"
    if not progress_path.is_file() or not video_path.is_file():
        return None
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    states = progress.get("states") or []
    final_state = states[-1] if states else {}
    return {
        "run_key": f"{trace_dir.parent.name}/{trace_dir.name}",
        "suite": trace_dir.parent.name,
        "episode": trace_dir.name,
        "trace": str(trace_dir),
        "output": str(output_dir),
        "status": "complete",
        "resumed": True,
        "action_count": max(0, len(states) - 1),
        "critic_latency_s": progress.get("total_latency_s"),
        "final_progress": final_state.get("progress"),
        "final_level": final_state.get("level"),
        "video": str(video_path),
    }


def run_sweep(args: argparse.Namespace) -> int:
    sweep_root = args.sweep_root.resolve()
    traces = discover_traces(sweep_root)
    if args.max_runs is not None:
        traces = traces[: args.max_runs]
    if not traces:
        raise FileNotFoundError(f"no complete traces found under {sweep_root}")

    summary_path = (
        sweep_root / "progress_critic" / f"{args.output_name}_summary.json"
    )
    results: dict[str, dict[str, Any]] = {}
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        results = {
            str(record["run_key"]): record
            for record in previous.get("records", [])
            if isinstance(record, dict) and record.get("run_key")
        }

    started = time.monotonic()
    print(
        f"[sweep] traces={len(traces)} model={args.model} "
        f"output_name={args.output_name}"
    )
    for index, trace_dir in enumerate(traces, start=1):
        run_key = f"{trace_dir.parent.name}/{trace_dir.name}"
        output_dir = trace_dir / "progress_critic" / args.output_name
        existing = _existing_record(trace_dir, output_dir)
        if existing is not None:
            results[run_key] = existing
            print(f"[run {index:02d}/{len(traces):02d}] skip complete {run_key}")
            _write_summary(
                path=summary_path,
                sweep_root=sweep_root,
                output_name=args.output_name,
                model=args.model,
                results=results,
                expected_runs=len(traces),
            )
            continue

        run_started = time.monotonic()
        print(f"[run {index:02d}/{len(traces):02d}] start {run_key}")
        existed_before = output_dir.exists()
        try:
            progress_path = score_trace(
                trace_dir=trace_dir,
                output_dir=output_dir,
                server_url=args.server_url,
                model=args.model,
                timeout_s=args.timeout_s,
                resume=True,
                max_actions=None,
            )
            video_path: Path | None = None
            if args.render_video:
                video_path = render_progress_video(
                    trace_dir=trace_dir,
                    progress_path=progress_path,
                    output_path=output_dir / "progress_overlay.mp4",
                )
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            states = progress.get("states") or []
            final_state = states[-1] if states else {}
            results[run_key] = {
                "run_key": run_key,
                "suite": trace_dir.parent.name,
                "episode": trace_dir.name,
                "trace": str(trace_dir),
                "output": str(output_dir),
                "status": "complete",
                "resumed": existed_before,
                "wall_time_s": round(time.monotonic() - run_started, 3),
                "action_count": max(0, len(states) - 1),
                "critic_latency_s": progress.get("total_latency_s"),
                "final_progress": final_state.get("progress"),
                "final_level": final_state.get("level"),
                "video": str(video_path) if video_path is not None else None,
            }
            print(
                f"[run {index:02d}/{len(traces):02d}] complete {run_key} "
                f"wall={results[run_key]['wall_time_s']:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001 - keep the sweep alive
            results[run_key] = {
                "run_key": run_key,
                "suite": trace_dir.parent.name,
                "episode": trace_dir.name,
                "trace": str(trace_dir),
                "output": str(output_dir),
                "status": "failed",
                "wall_time_s": round(time.monotonic() - run_started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(
                f"[run {index:02d}/{len(traces):02d}] failed {run_key}: "
                f"{type(exc).__name__}: {exc}"
            )
        _write_summary(
            path=summary_path,
            sweep_root=sweep_root,
            output_name=args.output_name,
            model=args.model,
            results=results,
            expected_runs=len(traces),
        )

    elapsed = time.monotonic() - started
    print(f"[sweep] finished wall={elapsed:.1f}s summary={summary_path}")
    return 0 if all(record.get("status") == "complete" for record in results.values()) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output-name", default="absolute_progress_gemma4_all")
    parser.add_argument("--server-url", default=DEFAULT_SERVER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--max-runs", type=int)
    return parser.parse_args()


def main() -> int:
    return run_sweep(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
