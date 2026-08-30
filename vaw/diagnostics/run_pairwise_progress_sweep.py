"""Run raw multi-view pairwise progress scoring for every trace in a sweep."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vaw.diagnostics.progress_video import render_progress_video
from vaw.diagnostics.run_pairwise_progress_demo import score_pairwise_trace
from vaw.diagnostics.run_progress_demo import DEFAULT_MODEL, DEFAULT_SERVER
from vaw.diagnostics.run_progress_sweep import discover_traces


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _record(trace: Path, output: Path, *, resumed: bool) -> dict[str, Any] | None:
    progress_path = output / "pairwise_progress.json"
    video_path = output / "pairwise_progress_overlay.mp4"
    if not progress_path.is_file() or not video_path.is_file():
        return None
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("evidence_kind") != "raw_synchronized_agentview_wrist":
        return None
    states = progress.get("states") or []
    final = states[-1] if states else {}
    return {
        "run_key": f"{trace.parent.name}/{trace.name}",
        "suite": trace.parent.name,
        "episode": trace.name,
        "trace": str(trace),
        "output": str(output),
        "status": "complete",
        "resumed": resumed,
        "action_count": max(0, len(states) - 1),
        "comparison_count": progress.get("comparison_count"),
        "critic_latency_s": progress.get("total_latency_s"),
        "final_latent_progress": final.get("latent_progress"),
        "video": str(video_path),
    }


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
            "schema": "vaw-pairwise-progress-sweep-v1",
            "evidence_kind": "raw_synchronized_agentview_wrist",
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
        f"[pairwise-sweep] traces={len(traces)} model={args.model} "
        f"output_name={args.output_name}"
    )
    for index, trace in enumerate(traces, start=1):
        run_key = f"{trace.parent.name}/{trace.name}"
        output = trace / "progress_critic" / args.output_name
        existing = _record(trace, output, resumed=True)
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
        existed_before = output.exists()
        print(f"[run {index:02d}/{len(traces):02d}] start {run_key}")
        try:
            progress_path = score_pairwise_trace(
                trace_dir=trace,
                output_dir=output,
                server_url=args.server_url,
                model=args.model,
                timeout_s=args.timeout_s,
                resume=True,
                max_actions=None,
            )
            video_path: Path | None = None
            if args.render_video:
                video_path = render_progress_video(
                    trace_dir=trace,
                    progress_path=progress_path,
                    output_path=output / "pairwise_progress_overlay.mp4",
                )
            record = _record(trace, output, resumed=existed_before)
            if record is None:
                raise RuntimeError("pairwise output bundle is incomplete")
            record["wall_time_s"] = round(time.monotonic() - run_started, 3)
            record["video"] = str(video_path) if video_path is not None else None
            results[run_key] = record
            print(
                f"[run {index:02d}/{len(traces):02d}] complete {run_key} "
                f"wall={record['wall_time_s']:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001 - continue after one bad trace
            results[run_key] = {
                "run_key": run_key,
                "suite": trace.parent.name,
                "episode": trace.name,
                "trace": str(trace),
                "output": str(output),
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
    print(f"[pairwise-sweep] finished wall={elapsed:.1f}s summary={summary_path}")
    return 0 if all(record.get("status") == "complete" for record in results.values()) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output-name", default="pairwise_raw_multiview_gemma4_all")
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
