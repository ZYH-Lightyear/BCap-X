"""用冻结 Qwen3-VL 复现 TOPReward，并与 VAW 物理 Action 对齐。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from vaw.diagnostics.progress_critic import build_action_evidence, trace_task
from vaw.diagnostics.progress_video import render_progress_video
from vaw.diagnostics.topreward import (
    TOPREWARD_PROMPT_PREFIX,
    TOPREWARD_PROMPT_SUFFIX,
    TOPREWARD_REFERENCE_COMMIT,
    TOPREWARD_REFERENCE_URL,
    QwenTopRewardScorer,
    TopRewardScore,
    TopRewardScorer,
    build_physical_prefixes,
    calibrate_progress,
    minmax_normalize,
    read_sampled_rgb_frames,
    write_json,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


class LazyQwenTopRewardScorer:
    """只在确实缺少缓存分数时加载视觉奖励模型。"""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        dtype: str,
        attention_implementation: str,
        fps: float,
        local_files_only: bool,
    ) -> None:
        self._model_name = model_name
        self._options = {
            "model_name": model_name,
            "device": device,
            "dtype": dtype,
            "attention_implementation": attention_implementation,
            "fps": fps,
            "local_files_only": local_files_only,
        }
        self._scorer: QwenTopRewardScorer | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(
        self, *, frames: list[Any], instruction: str
    ) -> TopRewardScore:
        if self._scorer is None:
            self._scorer = QwenTopRewardScorer(**self._options)
        return self._scorer.score(frames=frames, instruction=instruction)


def _default_output(trace_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return trace_dir / "topreward" / stamp


def _read_score(path: Path) -> TopRewardScore:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TopRewardScore(
        raw_log_prob=float(payload["raw_log_prob"]),
        token_count=int(payload["token_count"]),
        scored_token_ids=tuple(int(value) for value in payload["scored_token_ids"]),
        scored_tokens=tuple(str(value) for value in payload["scored_tokens"]),
    )


def _score_to_json(score: TopRewardScore) -> dict[str, Any]:
    return {
        "raw_log_prob": score.raw_log_prob,
        "token_count": score.token_count,
        "scored_token_ids": list(score.scored_token_ids),
        "scored_tokens": list(score.scored_tokens),
    }


def _trace_env_success(trace_dir: Path) -> bool | None:
    """读取仅供离线诊断使用的终局真值，不把它送入 TOPReward 模型。"""

    payload = json.loads((trace_dir / "meta.json").read_text(encoding="utf-8"))
    value = payload.get("env_success")
    if value is None or isinstance(value, bool):
        return value
    raise ValueError("trace meta env_success must be boolean or null")


def score_trace(
    *,
    trace_dir: str | Path,
    output_dir: str | Path,
    scorer: TopRewardScorer,
    max_prefix_frames: int = 15,
    max_actions: int | None = None,
    resume: bool = False,
    failure_penalty_min: float = 0.05,
    failure_penalty_max: float = 0.15,
) -> Path:
    """生成原始 TOPReward 分数和仅供显示的 episode 内归一化曲线。"""

    trace = Path(trace_dir).resolve()
    output = Path(output_dir).resolve()
    source_video = trace / "video_agentview.mp4"
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    task = trace_task(trace)
    env_success = _trace_env_success(trace)
    actions = build_action_evidence(trace)
    if max_actions is not None:
        actions = actions[:max_actions]

    capture = cv2.VideoCapture(str(source_video))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    prefixes = build_physical_prefixes(
        actions=actions, frame_count=frame_count, max_frames=max_prefix_frames
    )
    sampled = read_sampled_rgb_frames(source_video, prefixes)
    output.mkdir(parents=True, exist_ok=resume)
    states_dir = output / "states"

    scores: list[TopRewardScore] = []
    for prefix in prefixes:
        state_dir = states_dir / f"state_{prefix.state_index:04d}_{prefix.segment_id}"
        response_path = state_dir / "response.json"
        if resume and response_path.is_file():
            score = _read_score(response_path)
        else:
            frames = [sampled[index] for index in prefix.sampled_frame_indices]
            score = scorer.score(frames=frames, instruction=task)
            write_json(response_path, _score_to_json(score))
        scores.append(score)
        print(
            f"[topreward {prefix.state_index:02d}/{len(prefixes) - 1:02d}] "
            f"turn={prefix.turn} function={prefix.function} "
            f"frames={len(prefix.sampled_frame_indices)} raw={score.raw_log_prob:.6f}"
        )

    raw = [score.raw_log_prob for score in scores]
    relative_progress = minmax_normalize(raw)
    calibrated_progress = calibrate_progress(
        relative_progress,
        env_success=env_success,
        penalty_min=failure_penalty_min,
        penalty_max=failure_penalty_max,
    )
    states: list[dict[str, Any]] = []
    previous_progress: float | None = None
    previous_raw: float | None = None
    previous_delta = 0.0
    for prefix, score, relative, progress in zip(
        prefixes, scores, relative_progress, calibrated_progress, strict=True
    ):
        delta = 0.0 if previous_progress is None else progress - previous_progress
        raw_delta = 0.0 if previous_raw is None else score.raw_log_prob - previous_raw
        states.append(
            {
                "state_index": prefix.state_index,
                "segment_id": prefix.segment_id,
                "turn": prefix.turn,
                "function": prefix.function,
                "time_s": prefix.time_s,
                "frame_index": prefix.frame_index,
                "sampled_frame_indices": list(prefix.sampled_frame_indices),
                "raw_reward": score.raw_log_prob,
                "raw_delta": raw_delta,
                "relative_progress": relative,
                "progress": progress,
                "delta": delta,
                "acceleration": delta - previous_delta,
                "token_count": score.token_count,
                "scored_tokens": list(score.scored_tokens),
            }
        )
        previous_progress = progress
        previous_raw = score.raw_log_prob
        previous_delta = delta

    manifest = {
        "schema": "vaw-topreward-progress-v1",
        "score_kind": "topreward",
        "method": "TOPReward token log-probability reproduction",
        "reference": {
            "url": TOPREWARD_REFERENCE_URL,
            "commit": TOPREWARD_REFERENCE_COMMIT,
        },
        "prompt": {
            "prefix": TOPREWARD_PROMPT_PREFIX,
            "suffix": TOPREWARD_PROMPT_SUFFIX,
        },
        "trace": str(trace),
        "task": task,
        "model": scorer.model_name,
        "source_video": str(source_video),
        "source_fps": source_fps,
        "source_frame_count": frame_count,
        "max_prefix_frames": max_prefix_frames,
        "normalization": "per_episode_minmax_then_outcome_calibration",
        "env_success": env_success,
        "failure_penalty": {
            "kind": "linear_multiplicative",
            "low_progress_rate": failure_penalty_max,
            "high_progress_rate": failure_penalty_min,
        },
        "physical_boundaries_only": True,
        "preview_frames_scored": False,
        "states": states,
    }
    path = output / "progress.json"
    write_json(path, manifest)
    with (output / "progress.jsonl").open("w", encoding="utf-8") as stream:
        for state in states:
            stream.write(json.dumps(state, ensure_ascii=False) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32", "auto"), default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--max-prefix-frames", type=int, default=15)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--failure-penalty-min", type=float, default=0.05)
    parser.add_argument("--failure-penalty-max", type=float, default=0.15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--render-video", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace = args.trace.resolve()
    output = (args.output_dir or _default_output(trace)).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"output already exists; use --resume or choose another directory: {output}"
        )
    scorer = LazyQwenTopRewardScorer(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        fps=args.video_fps,
        local_files_only=args.local_files_only,
    )
    progress = score_trace(
        trace_dir=trace,
        output_dir=output,
        scorer=scorer,
        max_prefix_frames=args.max_prefix_frames,
        max_actions=args.max_actions,
        resume=args.resume,
        failure_penalty_min=args.failure_penalty_min,
        failure_penalty_max=args.failure_penalty_max,
    )
    print(f"[topreward] {progress}")
    if args.render_video:
        video = render_progress_video(
            trace_dir=trace,
            progress_path=progress,
            output_path=output / "topreward_action_curve.mp4",
        )
        print(f"[video] {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LazyQwenTopRewardScorer", "score_trace"]
