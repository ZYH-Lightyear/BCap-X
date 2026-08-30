"""Build synchronized raw AgentView/Wrist evidence for progress critics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from vaw.diagnostics.progress_critic import ActionEvidence


@dataclass(frozen=True)
class MultiViewObservation:
    """One synchronized raw physical observation with no policy UI overlays."""

    frame_index: int
    agentview: Path
    wrist: Path


@dataclass(frozen=True)
class MultiViewTransition:
    """Raw sensor states immediately before and after one physical action."""

    action: ActionEvidence
    before: MultiViewObservation
    after: MultiViewObservation


def _write_selected_frames(
    *, video_path: Path, frame_indices: set[int], output_dir: Path
) -> dict[int, Path]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[int, Path] = {}
    missing: set[int] = set()
    for frame_index in frame_indices:
        path = output_dir / f"frame_{frame_index:06d}.jpg"
        if path.is_file():
            resolved[frame_index] = path.resolve()
        else:
            missing.add(frame_index)
    if not missing:
        return resolved

    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"raw observation video has no frames: {video_path}")
        invalid = sorted(index for index in missing if index < 0 or index >= frame_count)
        if invalid:
            raise IndexError(
                f"requested frames outside {video_path.name} [0, {frame_count}): {invalid}"
            )
        maximum = max(missing)
        frame_index = 0
        while frame_index <= maximum:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"cannot decode frame {frame_index} from {video_path}")
            if frame_index in missing:
                path = output_dir / f"frame_{frame_index:06d}.jpg"
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"failed to write raw observation frame: {path}")
                resolved[frame_index] = path.resolve()
            frame_index += 1
    finally:
        capture.release()
    return resolved


def build_multiview_transitions(
    *,
    trace_dir: str | Path,
    actions: list[ActionEvidence],
    output_dir: str | Path,
) -> tuple[MultiViewObservation, list[MultiViewTransition]]:
    """Extract synchronized raw states from episode-level camera recordings."""

    trace = Path(trace_dir).resolve()
    output = Path(output_dir).resolve()
    if not actions:
        raise ValueError("cannot build multi-view transitions without physical actions")
    boundaries = {
        0,
        *(
            index
            for action in actions
            for index in (
                max(0, action.frame_start - 1),
                max(0, action.frame_end - 1),
            )
        ),
    }
    agentview_frames = _write_selected_frames(
        video_path=trace / "video_agentview.mp4",
        frame_indices=boundaries,
        output_dir=output / "raw_frames" / "agentview",
    )
    wrist_frames = _write_selected_frames(
        video_path=trace / "video_wrist.mp4",
        frame_indices=boundaries,
        output_dir=output / "raw_frames" / "wrist",
    )

    def observation(frame_index: int) -> MultiViewObservation:
        return MultiViewObservation(
            frame_index=frame_index,
            agentview=agentview_frames[frame_index],
            wrist=wrist_frames[frame_index],
        )

    initial = observation(0)
    transitions: list[MultiViewTransition] = []
    for action in actions:
        before_index = max(0, action.frame_start - 1)
        after_index = max(0, action.frame_end - 1)
        transitions.append(
            MultiViewTransition(
                action=action,
                before=observation(before_index),
                after=observation(after_index),
            )
        )
    return initial, transitions


__all__ = [
    "MultiViewObservation",
    "MultiViewTransition",
    "build_multiview_transitions",
]
