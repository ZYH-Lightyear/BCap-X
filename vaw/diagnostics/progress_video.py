"""Generate a deterministic HyperFrames project for a progress-scored trace."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

TEMPLATE_DIR = Path(__file__).resolve().parent / "progress_video_hf"
_PROJECT_FILES = (
    "DESIGN.md",
    "hyperframes.json",
    "meta.json",
    "package.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _video_duration(trace_dir: Path, progress: dict[str, Any]) -> float:
    meta_path = trace_dir / "meta.json"
    meta = _read_json(meta_path) if meta_path.is_file() else {}
    video = ((meta.get("videos") or {}).get("agentview") or {})
    try:
        frames = float(video["frames"])
        fps = float(video["fps"])
        if frames > 0 and fps > 0:
            return frames / fps
    except (KeyError, TypeError, ValueError):
        pass
    states = progress.get("states") or []
    end_times = [
        float(state["time_s"])
        for state in states
        if isinstance(state, dict) and isinstance(state.get("time_s"), (int, float))
    ]
    if end_times:
        duration = max(end_times)
        if duration > 0:
            return duration
    raise ValueError("cannot determine AgentView duration")


def _safe_script_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )


def build_progress_video_project(
    *,
    trace_dir: str | Path,
    progress_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Create one self-contained, renderable HyperFrames composition project."""

    trace = Path(trace_dir).resolve()
    score_path = Path(progress_path).resolve()
    output = Path(output_dir).resolve()
    progress = _read_json(score_path)
    task = progress.get("task")
    states = progress.get("states")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("progress file has no task")
    if not isinstance(states, list) or not states:
        raise ValueError("progress file has no states")
    duration = _video_duration(trace, progress)

    source_video = trace / "video_agentview.mp4"
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    output.mkdir(parents=True, exist_ok=True)
    for filename in _PROJECT_FILES:
        shutil.copy2(TEMPLATE_DIR / filename, output / filename)
    shutil.copy2(source_video, output / "agentview.mp4")
    shutil.copy2(score_path, output / "progress.json")

    video_payload = {
        "schema": "vaw-progress-video-v1",
        "task": task,
        "duration_s": duration,
        "states": states,
    }
    template = (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
    rendered = template.replace("__DURATION__", f"{duration:.6f}").replace(
        "__PROGRESS_DATA__", _safe_script_json(video_payload)
    )
    (output / "index.html").write_text(rendered, encoding="utf-8")
    return output


def _bgr(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.removeprefix("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


_COLORS = {
    "bg": _bgr("#071018"),
    "panel": _bgr("#0E1A24"),
    "video": _bgr("#03080D"),
    "fg": _bgr("#E8F0F5"),
    "muted": _bgr("#91A4B2"),
    "accent": _bgr("#35B8D0"),
    "advance": _bgr("#38C786"),
    "regress": _bgr("#F05D5E"),
    "unknown": _bgr("#E3B341"),
    "rule": _bgr("#29414F"),
}


def _tone(delta: Any) -> str:
    if not isinstance(delta, (int, float)):
        return "muted"
    if float(delta) > 0.025:
        return "advance"
    if float(delta) < -0.025:
        return "regress"
    return "muted"


def _fit_agentview(frame: np.ndarray) -> np.ndarray:
    target_width, target_height = 1280, 820
    output = np.full((target_height, target_width, 3), _COLORS["video"], dtype=np.uint8)
    scale = min(target_width / frame.shape[1], target_height / frame.shape[0])
    width = max(1, round(frame.shape[1] * scale))
    height = max(1, round(frame.shape[0] * scale))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    x0 = (target_width - width) // 2
    y0 = (target_height - height) // 2
    output[y0 : y0 + height, x0 : x0 + width] = resized
    return output


def _put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _wrap_text(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def _signed_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "--"
    return f"{float(value) * 100:+.0f}%"


def _draw_score_panel(
    frame: np.ndarray,
    *,
    task: str,
    state: dict[str, Any],
    score_kind: str = "absolute",
    env_success: bool | None = None,
) -> None:
    cv2.rectangle(frame, (1280, 0), (1919, 819), _COLORS["panel"], -1)
    cv2.line(frame, (1280, 0), (1280, 820), _COLORS["rule"], 2)
    _put_text(
        frame,
        (
            "VAW / PAIRWISE PROGRESS"
            if score_kind == "pairwise_latent"
            else (
                "VAW / TOPREWARD"
                if score_kind == "topreward"
                else "VAW / PROGRESS CRITIC"
            )
        ),
        (1332, 76),
        scale=0.72,
        color=_COLORS["accent"],
        thickness=2,
    )
    y = 132
    for line in _wrap_text(task, max_chars=31)[:5]:
        _put_text(frame, line, (1332, y), scale=0.78, color=_COLORS["fg"], thickness=2)
        y += 45
    cv2.line(frame, (1332, y + 6), (1868, y + 6), _COLORS["rule"], 2)
    y += 70
    turn = state.get("turn")
    function = str(state.get("function") or "unknown")
    heading = "INITIAL STATE" if turn == 0 else f"TURN {turn} / {function}"
    _put_text(frame, heading, (1332, y), scale=0.7, color=_COLORS["muted"], thickness=2)
    score_field = "latent_progress" if score_kind == "pairwise_latent" else "progress"
    delta_field = "latent_delta" if score_kind == "pairwise_latent" else "delta"
    acceleration_field = (
        "latent_acceleration" if score_kind == "pairwise_latent" else "acceleration"
    )
    progress = state.get(score_field)
    progress_text = (
        "--"
        if not isinstance(progress, (int, float))
        else (
            f"{float(progress):+.2f}"
            if score_kind == "pairwise_latent"
            else f"{float(progress) * 100:.0f}%"
        )
    )
    _put_text(frame, progress_text, (1326, y + 128), scale=3.1, color=_COLORS["fg"], thickness=5)
    delta = state.get(delta_field)
    _put_text(
        frame,
        (
            f"DELTA  {float(delta):+.2f}"
            if score_kind == "pairwise_latent" and isinstance(delta, (int, float))
            else f"DELTA  {_signed_percent(delta)}"
        ),
        (1334, y + 195),
        scale=0.82,
        color=_COLORS[_tone(delta)],
        thickness=2,
    )
    _put_text(
        frame,
        (
            f"ACCEL  {float(state.get(acceleration_field)):+.2f}"
            if score_kind == "pairwise_latent"
            and isinstance(state.get(acceleration_field), (int, float))
            else f"ACCEL  {_signed_percent(state.get(acceleration_field))}"
        ),
        (1334, y + 242),
        scale=0.82,
        color=_COLORS["muted"],
        thickness=2,
    )
    _put_text(
        frame,
        (
            "LOG-ODDS UTILITY (NOT %)"
            if score_kind == "pairwise_latent"
            else (
                f"RAW LOG P(TRUE)  {float(state['raw_reward']):+.3f}"
                if score_kind == "topreward"
                and isinstance(state.get("raw_reward"), (int, float))
                else f"LEVEL {state.get('level') or '?'}"
            )
        ),
        (1334, y + 310),
        scale=0.8,
        color=_COLORS["accent"],
        thickness=2,
    )
    if score_kind == "topreward" and env_success is not None:
        _put_text(
            frame,
            f"ENV OUTCOME  {'SUCCESS' if env_success else 'NOT SUCCESSFUL'}",
            (1334, y + 355),
            scale=0.72,
            color=_COLORS["advance" if env_success else "regress"],
            thickness=2,
        )


def _draw_progress_chart(
    frame: np.ndarray,
    *,
    states: list[dict[str, Any]],
    time_s: float,
    duration_s: float,
    score_kind: str = "absolute",
) -> None:
    cv2.rectangle(frame, (0, 820), (1919, 1079), _COLORS["bg"], -1)
    cv2.line(frame, (0, 820), (1919, 820), _COLORS["rule"], 2)
    _put_text(
        frame,
        (
            "PAIRWISE LATENT TASK PROGRESS"
            if score_kind == "pairwise_latent"
            else (
                "TOPREWARD / OUTCOME-CALIBRATED PROGRESS"
                if score_kind == "topreward"
                else "ABSOLUTE TASK PROGRESS"
            )
        ),
        (54, 866),
        scale=0.75,
        color=_COLORS["fg"],
        thickness=2,
    )
    _put_text(
        frame,
        (
            "post-hoc env calibration; raw log-probability preserved"
            if score_kind == "topreward"
            else "revealed only after AFTER evidence"
        ),
        (1420, 866),
        scale=0.54,
        color=_COLORS["muted"],
        thickness=1,
    )
    left, right, top, bottom = 120, 1870, 894, 1042
    cv2.line(frame, (left, top), (left, bottom), _COLORS["rule"], 2)
    cv2.line(frame, (left, bottom), (right, bottom), _COLORS["rule"], 2)
    score_field = "latent_progress" if score_kind == "pairwise_latent" else "progress"
    delta_field = "latent_delta" if score_kind == "pairwise_latent" else "delta"
    numeric_scores = [
        float(state[score_field])
        for state in states
        if isinstance(state.get(score_field), (int, float))
    ]
    if score_kind == "pairwise_latent" and numeric_scores:
        score_min = min(numeric_scores)
        score_max = max(numeric_scores)
        padding = max(0.1, (score_max - score_min) * 0.08)
        score_min -= padding
        score_max += padding
    else:
        score_min, score_max = 0.0, 1.0
    score_mid = (score_min + score_max) / 2
    label_format = (lambda value: f"{value:+.1f}") if score_kind == "pairwise_latent" else (lambda value: f"{value * 100:.0f}")
    _put_text(frame, label_format(score_max), (44, top + 8), scale=0.52, color=_COLORS["muted"], thickness=1)
    _put_text(frame, label_format(score_mid), (44, (top + bottom) // 2 + 7), scale=0.52, color=_COLORS["muted"], thickness=1)
    _put_text(frame, label_format(score_min), (44, bottom + 7), scale=0.52, color=_COLORS["muted"], thickness=1)

    def point(state: dict[str, Any]) -> tuple[int, int]:
        x = left + round(float(state.get("time_s") or 0.0) / duration_s * (right - left))
        score = min(score_max, max(score_min, float(state[score_field])))
        y = bottom - round((score - score_min) / (score_max - score_min) * (bottom - top))
        return x, y

    revealed = [
        state
        for state in states
        if isinstance(state.get(score_field), (int, float))
        and float(state.get("time_s") or 0.0) <= time_s + 1e-6
    ]
    for previous, current in zip(revealed, revealed[1:], strict=False):
        cv2.line(
            frame,
            point(previous),
            point(current),
            _COLORS[_tone(current.get(delta_field))],
            6,
            cv2.LINE_AA,
        )
    for state in revealed:
        cv2.circle(
            frame,
            point(state),
            8,
            _COLORS["accent"]
            if state.get("turn") == 0
            else _COLORS[_tone(state.get(delta_field))],
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(frame, point(state), 8, _COLORS["bg"], 2, cv2.LINE_AA)
    playhead_x = left + round(min(duration_s, time_s) / duration_s * (right - left))
    cv2.line(
        frame,
        (playhead_x, top - 4),
        (playhead_x, bottom + 5),
        _COLORS["accent"],
        2,
        cv2.LINE_AA,
    )


def render_progress_video(
    *,
    trace_dir: str | Path,
    progress_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Render an H.264 MP4 without browser or network dependencies."""

    trace = Path(trace_dir).resolve()
    progress = _read_json(Path(progress_path).resolve())
    states = progress.get("states")
    task = progress.get("task")
    score_kind = str(progress.get("score_kind") or "absolute")
    env_success = progress.get("env_success")
    if env_success is not None and not isinstance(env_success, bool):
        raise ValueError("progress env_success must be boolean or null")
    if not isinstance(states, list) or not states:
        raise ValueError("progress file has no states")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("progress file has no task")
    source = trace / "video_agentview.mp4"
    if not source.is_file():
        raise FileNotFoundError(source)
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"invalid source video: {source}")
    duration_s = frame_count / fps

    with tempfile.TemporaryDirectory(prefix="vaw-progress-render-") as temp_dir:
        intermediate = Path(temp_dir) / "progress-mp4v.mp4"
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("OpenCV could not open the progress video writer")
        try:
            frame_index = 0
            while True:
                ok, source_frame = capture.read()
                if not ok or source_frame is None:
                    break
                time_s = frame_index / fps
                visible = [
                    state
                    for state in states
                    if float(state.get("time_s") or 0.0) <= time_s + 1e-6
                ]
                current = visible[-1] if visible else states[0]
                canvas = np.full((1080, 1920, 3), _COLORS["bg"], dtype=np.uint8)
                canvas[:820, :1280] = _fit_agentview(source_frame)
                _draw_score_panel(
                    canvas,
                    task=task,
                    state=current,
                    score_kind=score_kind,
                    env_success=env_success,
                )
                _draw_progress_chart(
                    canvas,
                    states=states,
                    time_s=time_s,
                    duration_s=duration_s,
                    score_kind=score_kind,
                )
                writer.write(canvas)
                frame_index += 1
        finally:
            capture.release()
            writer.release()
        if frame_index != frame_count:
            raise RuntimeError(
                f"rendered {frame_index} frames but source declares {frame_count}"
            )
        encoded = Path(temp_dir) / "progress-h264.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(intermediate),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(encoded),
            ],
            check=True,
        )
        shutil.copy2(encoded, target)
    return target


__all__ = ["build_progress_video_project", "render_progress_video"]
