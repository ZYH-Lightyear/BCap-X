"""Score a VAW trace and build a synchronized progress-curve video project."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from vaw.diagnostics.progress_critic import (
    PROGRESS_LEVELS,
    SYSTEM_PROMPT,
    ProgressCriticClient,
    add_progress_dynamics,
    build_action_evidence,
    build_action_messages,
    build_initial_messages,
    evidence_to_json,
    initial_canvas,
    sha256,
    trace_task,
    write_action_strip,
)
from vaw.diagnostics.progress_video import (
    build_progress_video_project,
    render_progress_video,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER = "http://127.0.0.1:5088/v1/chat/completions"
DEFAULT_MODEL = "/mnt/nas/maqi/Gemma_4_31B"


def _default_output(trace_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return trace_dir / "progress_critic" / f"absolute_progress_{stamp}"


def _safe_request(
    *,
    task: str,
    initial: Path,
    current: Path,
    before: Path | None = None,
    action_strip: Path | None = None,
    action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    images = {
        "initial": {"path": str(initial), "sha256": sha256(initial)},
        "current": {"path": str(current), "sha256": sha256(current)},
    }
    if before is not None:
        images["before"] = {"path": str(before), "sha256": sha256(before)}
    if action_strip is not None:
        images["action_strip"] = {
            "path": str(action_strip),
            "sha256": sha256(action_strip),
        }
    return {
        "system_prompt": SYSTEM_PROMPT,
        "task": task,
        "images": images,
        "action": action,
        "privacy_note": (
            "Only local paths and digests are stored here. Base64 image payloads, "
            "future steps, env_success, and agent success claims are omitted."
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _score_or_resume(
    *,
    state_dir: Path,
    request_manifest: dict[str, Any],
    messages: list[dict[str, Any]],
    client: ProgressCriticClient,
    resume: bool,
) -> dict[str, Any]:
    response_path = state_dir / "response.json"
    if resume and response_path.is_file():
        return json.loads(response_path.read_text(encoding="utf-8"))
    state_dir.mkdir(parents=True, exist_ok=True)
    _write_json(state_dir / "request.json", request_manifest)
    response = client.score(messages)
    _write_json(response_path, response)
    return response


def score_trace(
    *,
    trace_dir: Path,
    output_dir: Path,
    server_url: str,
    model: str,
    timeout_s: float,
    resume: bool,
    max_actions: int | None,
) -> Path:
    task = trace_task(trace_dir)
    initial = initial_canvas(trace_dir)
    actions = build_action_evidence(trace_dir)
    if max_actions is not None:
        actions = actions[:max_actions]
    output_dir.mkdir(parents=True, exist_ok=resume)
    evidence_dir = output_dir / "evidence"
    states_dir = output_dir / "states"
    client = ProgressCriticClient(
        server_url=server_url, model=model, timeout_s=timeout_s
    )

    run_manifest = {
        "schema": "vaw-absolute-progress-demo-v1",
        "trace": str(trace_dir),
        "task": task,
        "server_url": server_url,
        "model": model,
        "progress_levels": PROGRESS_LEVELS,
        "system_prompt": SYSTEM_PROMPT,
        "action_count": len(actions),
        "note": (
            "Progress is scored independently at each physical state. Delta and "
            "acceleration are derived afterward. env_success is never sent to the critic."
        ),
    }
    _write_json(output_dir / "run.json", run_manifest)

    initial_response = _score_or_resume(
        state_dir=states_dir / "state_0000_initial",
        request_manifest=_safe_request(
            task=task, initial=initial, current=initial, action=None
        ),
        messages=build_initial_messages(task=task, canvas=initial),
        client=client,
        resume=resume,
    )
    states: list[dict[str, Any]] = [
        {
            "state_index": 0,
            "segment_id": "initial",
            "turn": 0,
            "function": "initial_state",
            "time_s": 0.0,
            "level": initial_response.get("predicted_level"),
            "progress": initial_response.get("progress"),
            "probabilities": initial_response.get("probabilities"),
            "latency_s": initial_response.get("latency_s"),
            "current_canvas": str(initial),
        }
    ]
    print(
        f"[state 00/{len(actions):02d}] initial "
        f"level={states[0]['level']} progress={states[0]['progress']}"
    )

    for action in actions:
        state_name = f"state_{action.index:04d}_{action.segment_id}"
        state_dir = states_dir / state_name
        strip_path = evidence_dir / f"{action.segment_id}_strip.jpg"
        if not strip_path.is_file():
            write_action_strip(
                action.action_video,
                strip_path,
                fallback_before=action.before_canvas,
                fallback_after=action.after_canvas,
            )
        action_json = evidence_to_json(action)
        response = _score_or_resume(
            state_dir=state_dir,
            request_manifest=_safe_request(
                task=task,
                initial=initial,
                before=action.before_canvas,
                action_strip=strip_path,
                current=action.after_canvas,
                action=action_json,
            ),
            messages=build_action_messages(
                task=task, initial=initial, action=action, strip=strip_path
            ),
            client=client,
            resume=resume,
        )
        state = {
            "state_index": action.index,
            "segment_id": action.segment_id,
            "turn": action.turn,
            "function": action.function,
            "arguments": action.arguments,
            "outcome": action.outcome,
            "revision_before": action.revision_before,
            "revision_after": action.revision_after,
            "frame_start": action.frame_start,
            "frame_end": action.frame_end,
            "fps": action.fps,
            "time_start_s": action.time_start_s,
            "time_s": action.time_end_s,
            "level": response.get("predicted_level"),
            "progress": response.get("progress"),
            "probabilities": response.get("probabilities"),
            "latency_s": response.get("latency_s"),
            "before_canvas": str(action.before_canvas),
            "current_canvas": str(action.after_canvas),
            "action_strip": str(strip_path),
        }
        states.append(state)
        print(
            f"[state {action.index:02d}/{len(actions):02d}] "
            f"turn={action.turn} function={action.function} "
            f"level={state['level']} progress={state['progress']}"
        )

    add_progress_dynamics(states)
    summary = {
        **run_manifest,
        "states": states,
        "total_latency_s": round(
            sum(float(state.get("latency_s") or 0.0) for state in states), 3
        ),
    }
    summary_path = output_dir / "progress.json"
    _write_json(summary_path, summary)
    with (output_dir / "progress.jsonl").open("w", encoding="utf-8") as stream:
        for state in states:
            stream.write(json.dumps(state, ensure_ascii=False) + "\n")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--server-url", default=DEFAULT_SERVER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--no-video-project", action="store_true")
    parser.add_argument("--render-video", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_dir = args.trace.resolve()
    output_dir = (args.output_dir or _default_output(trace_dir)).resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"output already exists; use --resume or choose another directory: {output_dir}"
        )
    if args.resume:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)

    summary_path = score_trace(
        trace_dir=trace_dir,
        output_dir=output_dir,
        server_url=args.server_url,
        model=args.model,
        timeout_s=args.timeout_s,
        resume=args.resume,
        max_actions=args.max_actions,
    )
    if not args.no_video_project:
        project_dir = output_dir / "video_project"
        build_progress_video_project(
            trace_dir=trace_dir,
            progress_path=summary_path,
            output_dir=project_dir,
        )
        print(f"[video-project] {project_dir}")
    if args.render_video:
        video_path = render_progress_video(
            trace_dir=trace_dir,
            progress_path=summary_path,
            output_path=output_dir / "progress_overlay.mp4",
        )
        print(f"[video] {video_path}")
    print(f"[progress] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
