"""Task-conditioned progress scoring for completed VAW physical actions.

The critic estimates the *absolute* completion progress of the visible task
state.  Local action contribution and progress acceleration are derived only
after independent state scores have been produced:

``delta_t = progress_t - progress_(t-1)``
``accel_t = delta_t - delta_(t-1)``

No future step, terminal reward, or agent success claim enters the request.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

PROGRESS_LEVELS: dict[str, dict[str, Any]] = {
    "A": {
        "value": 0.0,
        "description": "0–10%：目标关系基本没有建立，或此前进展已经丢失。",
    },
    "B": {
        "value": 0.25,
        "description": "10–35%：出现有用准备状态，但关键物理中间关系尚未建立。",
    },
    "C": {
        "value": 0.5,
        "description": "35–60%：至少一个完成任务所必需的关键中间关系已经建立。",
    },
    "D": {
        "value": 0.75,
        "description": "60–90%：已接近目标关系，但仍缺少对齐、释放、稳定或确认。",
    },
    "E": {
        "value": 1.0,
        "description": "90–100%：当前真实视觉强烈支持任务目标已经稳定满足。",
    },
}

SYSTEM_PROMPT = """你是独立的机器人任务进度评估器。你的任务是根据视觉证据估计当前物理状态对用户任务的绝对完成进度，而不是评价函数调用写得好不好。

进度等级：
{levels}

规则：
- 只依据 INITIAL、BEFORE、ACTION STRIP 和 CURRENT 中可见的真实物理状态。
- 相同物理状态应得到相近进度，不受函数名称、返回 completed 或 Agent 自述影响。
- 抓取只有在物体确实随夹爪运动时才构成关键进展；仅闭合夹爪或 IK 成功不算。
- 接近、运输和对齐可逐步提高进度；掉落、抓错、破坏目标关系应造成与损失幅度相称的下降。
- 尚未释放、未稳定或目标关系仍不可确认时，不应判为 E。
- 不要猜测未来动作，也不要假定 episode 最终成功。

只输出一个大写字母 A、B、C、D 或 E；不要解释。""".format(
    levels="\n".join(
        f"{label}. {definition['description']}" for label, definition in PROGRESS_LEVELS.items()
    )
)


@dataclass(frozen=True)
class ActionEvidence:
    """One physical action aligned with its trace and global video interval."""

    index: int
    segment_id: str
    turn: int
    function: str
    arguments: dict[str, Any]
    outcome: str
    revision_before: int | None
    revision_after: int | None
    frame_start: int
    frame_end: int
    fps: float
    time_start_s: float
    time_end_s: float
    before_canvas: Path
    after_canvas: Path
    action_video: Path | None
    poster: Path | None
    function_result: Any
    runtime_diagnostics: Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON objects in {path}")
        records.append(value)
    return records


def _resolve_trace_path(trace_dir: Path, relative: str | None) -> Path | None:
    if not relative:
        return None
    path = (trace_dir / relative).resolve()
    if trace_dir.resolve() not in path.parents and path != trace_dir.resolve():
        raise ValueError(f"trace-relative path escapes run directory: {relative}")
    return path


def trace_task(trace_dir: str | Path) -> str:
    directory = Path(trace_dir).resolve()
    meta_path = directory / "meta.json"
    meta = _read_json(meta_path) if meta_path.is_file() else {}
    task = meta.get("task_prompt") or meta.get("task_description") or meta.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"trace does not declare a task prompt: {directory}")
    return task.strip()


def initial_canvas(trace_dir: str | Path) -> Path:
    directory = Path(trace_dir).resolve()
    records = _read_jsonl(directory / "steps.jsonl")
    if not records:
        raise ValueError(f"trace has no steps: {directory}")
    path = _resolve_trace_path(directory, records[0].get("context_image"))
    if path is None or not path.is_file():
        raise FileNotFoundError(path or directory / "<missing-context-image>")
    return path


def build_action_evidence(trace_dir: str | Path) -> list[ActionEvidence]:
    """Align action manifests with the before/after canvases in ``steps.jsonl``."""

    directory = Path(trace_dir).resolve()
    records = _read_jsonl(directory / "steps.jsonl")
    by_turn = {
        int(record["turn"]): index
        for index, record in enumerate(records)
        if isinstance(record.get("turn"), int)
    }
    manifests = sorted((directory / "actions").glob("action_*/manifest.json"))
    if not manifests:
        raise ValueError(f"trace has no physical action manifests: {directory}")

    actions: list[ActionEvidence] = []
    for action_index, manifest_path in enumerate(manifests, start=1):
        manifest = _read_json(manifest_path)
        turn = manifest.get("turn")
        if not isinstance(turn, int) or turn not in by_turn:
            raise ValueError(f"action manifest has no matching trace turn: {manifest_path}")
        record_index = by_turn[turn]
        record = records[record_index]
        following = records[record_index + 1] if record_index + 1 < len(records) else None
        before = _resolve_trace_path(directory, record.get("context_image"))
        after = _resolve_trace_path(
            directory, following.get("context_image") if following is not None else None
        )
        streams = manifest.get("streams") or {}
        agentview = streams.get("agentview") if isinstance(streams, dict) else None
        video = _resolve_trace_path(
            directory, agentview.get("path") if isinstance(agentview, dict) else None
        )
        if video is not None and not video.is_file():
            video = None
        poster = _resolve_trace_path(directory, manifest.get("poster"))
        if poster is not None and not poster.is_file():
            poster = None
        # A max-turn termination can freeze the post-action Canvas without
        # appending another model step.  The root context sequence is still an
        # authoritative observation and uses the next zero-based context index.
        if after is None:
            terminal_canvas = directory / f"context_{record_index + 1:04d}.png"
            if terminal_canvas.is_file():
                after = terminal_canvas.resolve()
        # A trace can terminate immediately after dispatching its final physical
        # action.  The action poster is then the only post-action observation;
        # using it is more truthful than silently reusing the pre-action Canvas.
        if (after is None or not after.is_file()) and poster is not None:
            after = poster
        if before is None or not before.is_file():
            raise FileNotFoundError(before or directory / "<missing-before-canvas>")
        if after is None or not after.is_file():
            raise FileNotFoundError(
                f"physical action turn {turn} has no following Canvas; "
                "the absolute post-action state cannot be scored"
            )

        frame_start = int(manifest["frame_start"])
        frame_end = int(manifest["frame_end"])
        fps = float(manifest["fps"])
        if fps <= 0 or frame_start < 0 or frame_end < frame_start:
            raise ValueError(f"invalid action video interval: {manifest_path}")
        final_frame = max(frame_start, frame_end - 1)

        call = record.get("function_call") or {}
        actions.append(
            ActionEvidence(
                index=action_index,
                segment_id=str(manifest.get("segment_id") or manifest_path.parent.name),
                turn=turn,
                function=str(manifest.get("function") or call.get("name") or "unknown"),
                arguments=(
                    dict(manifest.get("arguments"))
                    if isinstance(manifest.get("arguments"), dict)
                    else {}
                ),
                outcome=str(manifest.get("outcome") or "unknown"),
                revision_before=(
                    int(manifest["revision_before"])
                    if isinstance(manifest.get("revision_before"), int)
                    else None
                ),
                revision_after=(
                    int(manifest["revision_after"])
                    if isinstance(manifest.get("revision_after"), int)
                    else None
                ),
                frame_start=frame_start,
                frame_end=frame_end,
                fps=fps,
                time_start_s=frame_start / fps,
                # ``frame_end`` is exclusive; reveal the score on the final
                # frame that actually belongs to this action segment.
                time_end_s=final_frame / fps,
                before_canvas=before,
                after_canvas=after,
                action_video=video,
                poster=poster,
                function_result=record.get("function_result"),
                runtime_diagnostics=record.get("runtime_diagnostics"),
            )
        )
    return actions


def write_action_strip(
    video_path: str | Path | None,
    output_path: str | Path,
    *,
    fallback_before: str | Path | None = None,
    fallback_after: str | Path | None = None,
) -> Path:
    """Write a start/middle/end strip, with a state-diff fallback."""

    target = Path(output_path)
    def write_state_diff() -> Path:
        before = cv2.imread(str(fallback_before)) if fallback_before is not None else None
        after = cv2.imread(str(fallback_after)) if fallback_after is not None else None
        if before is None or after is None:
            raise FileNotFoundError("action has neither a video nor before/after images")
        middle = cv2.addWeighted(before, 0.5, after, 0.5, 0.0)
        return _write_strip_frames([before, middle, after], target)

    if video_path is None:
        return write_state_diff()

    source = Path(video_path)
    try:
        capture = cv2.VideoCapture(str(source))
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise ValueError(f"video has no frames: {source}")
            selected = (0, frame_count // 2, frame_count - 1)
            frames: list[np.ndarray] = []
            for frame_index in selected:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"cannot read frame {frame_index} from {source}")
                frames.append(frame)
        finally:
            capture.release()
    except ValueError:
        if fallback_before is None or fallback_after is None:
            raise
        return write_state_diff()

    return _write_strip_frames(frames, target)


def _write_strip_frames(frames: list[np.ndarray], target: Path) -> Path:
    height = min(frame.shape[0] for frame in frames)
    resized: list[np.ndarray] = []
    for frame in frames:
        width = max(1, round(frame.shape[1] * height / frame.shape[0]))
        resized.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    strip = np.concatenate(resized, axis=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), strip, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"failed to write action strip: {target}")
    return target


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _image_content(path: Path, *, detail: str = "high") -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": _data_url(path), "detail": detail}}


def _compact_action(action: ActionEvidence) -> str:
    blocked_keys = {
        "env_success",
        "reward",
        "env_reward",
        "done",
        "claimed_success",
        "success",
    }

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): scrub(item)
                for key, item in value.items()
                if str(key).casefold() not in blocked_keys
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    payload = {
        "turn": action.turn,
        "function_call": {"name": action.function, "arguments": action.arguments},
        "function_result": scrub(action.function_result),
        "runtime_outcome": action.outcome,
        "revision_before": action.revision_before,
        "revision_after": action.revision_after,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_initial_messages(*, task: str, canvas: Path) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"用户任务：{task}\n\n"
                        "这是 episode 的初始真实状态，同时也是当前状态。"
                        "估计此时任务的绝对完成进度。\n\nCURRENT："
                    ),
                },
                _image_content(canvas),
                {"type": "text", "text": "输出进度等级："},
            ],
        },
    ]


def build_action_messages(
    *, task: str, initial: Path, action: ActionEvidence, strip: Path
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"用户任务：{task}\n\n"
                        "请估计本轮动作结束后的 CURRENT 状态相对整个任务的绝对完成进度。"
                        "ACTION STRIP 从左到右是动作起始、中间和结束帧。\n\nINITIAL："
                    ),
                },
                _image_content(initial, detail="low"),
                {"type": "text", "text": "BEFORE："},
                _image_content(action.before_canvas),
                {"type": "text", "text": "ACTION STRIP："},
                _image_content(strip),
                {"type": "text", "text": "CURRENT（动作执行后）："},
                _image_content(action.after_canvas),
                {
                    "type": "text",
                    "text": f"本轮函数 Trace：\n{_compact_action(action)}\n\n输出进度等级：",
                },
            ],
        },
    ]


def extract_label(text: str) -> str | None:
    match = re.search(r"(?<![A-Z])[ABCDE](?![A-Z])", text.upper())
    return match.group(0) if match else None


def label_distribution(choice: dict[str, Any]) -> dict[str, Any]:
    """Extract a normalized A–E distribution from OpenAI-compatible logprobs."""

    content = ((choice.get("logprobs") or {}).get("content") or [])
    label_position: dict[str, Any] | None = None
    for position in content:
        if extract_label(str(position.get("token") or "")) is not None:
            label_position = position
            break
    if label_position is None and content:
        label_position = content[0]
    if label_position is None:
        return {"available": False, "reason": "response has no token logprobs"}

    label_logprobs: dict[str, list[float]] = {label: [] for label in PROGRESS_LEVELS}
    raw_candidates: list[dict[str, Any]] = []
    candidates = [
        {"token": label_position.get("token"), "logprob": label_position.get("logprob")},
        *(label_position.get("top_logprobs") or []),
    ]
    seen: set[tuple[str, float]] = set()
    for candidate in candidates:
        token = str(candidate.get("token") or "")
        try:
            logprob = float(candidate.get("logprob"))
        except (TypeError, ValueError):
            continue
        key = (token, logprob)
        if key in seen:
            continue
        seen.add(key)
        raw_candidates.append({"token": token, "logprob": logprob})
        label = extract_label(token.strip())
        if label is not None:
            label_logprobs[label].append(logprob)

    combined: dict[str, float | None] = {}
    for label, values in label_logprobs.items():
        if not values:
            combined[label] = None
            continue
        maximum = max(values)
        combined[label] = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    finite = {label: value for label, value in combined.items() if value is not None}
    if not finite:
        return {"available": False, "reason": "top logprobs contain no A–E labels"}
    maximum = max(finite.values())
    denominator = sum(math.exp(value - maximum) for value in finite.values())
    probabilities = {
        label: (
            math.exp(value - maximum) / denominator if value is not None else 0.0
        )
        for label, value in combined.items()
    }
    return {
        "available": True,
        "answer_token": label_position.get("token"),
        "label_logprobs": combined,
        "label_probabilities": probabilities,
        "visible_label_count": len(finite),
        "raw_top_candidates": raw_candidates,
    }


def progress_from_distribution(
    distribution: dict[str, Any], *, predicted_label: str | None
) -> tuple[float | None, dict[str, float]]:
    probabilities = distribution.get("label_probabilities")
    if isinstance(probabilities, dict) and any(float(value or 0.0) > 0 for value in probabilities.values()):
        normalized = {
            label: float(probabilities.get(label) or 0.0) for label in PROGRESS_LEVELS
        }
        score = sum(
            normalized[label] * float(PROGRESS_LEVELS[label]["value"])
            for label in PROGRESS_LEVELS
        )
        return score, normalized
    if predicted_label in PROGRESS_LEVELS:
        one_hot = {label: float(label == predicted_label) for label in PROGRESS_LEVELS}
        return float(PROGRESS_LEVELS[predicted_label]["value"]), one_hot
    return None, dict.fromkeys(PROGRESS_LEVELS, 0.0)


def add_progress_dynamics(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add local progress change and its discrete acceleration in place."""

    previous_score: float | None = None
    previous_delta: float | None = None
    for state in states:
        score = state.get("progress")
        if not isinstance(score, (int, float)):
            state["delta"] = None
            state["acceleration"] = None
            continue
        score = float(score)
        delta = None if previous_score is None else score - previous_score
        state["delta"] = delta
        state["acceleration"] = (
            delta - previous_delta
            if delta is not None and previous_delta is not None
            else None
        )
        previous_score = score
        if delta is not None:
            previous_delta = delta
    return states


class ProgressCriticClient:
    """Small OpenAI-compatible client dedicated to one-token progress scores."""

    def __init__(
        self,
        *,
        server_url: str,
        model: str,
        timeout_s: float = 300.0,
        session: requests.Session | None = None,
    ) -> None:
        self.server_url = server_url
        self.model = model
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def score(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 4,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        response = self.session.post(
            self.server_url, json=payload, timeout=self.timeout_s
        )
        latency_s = time.perf_counter() - started
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ValueError(f"critic response has no choices: {body}")
        choice = choices[0]
        raw_text = str(((choice.get("message") or {}).get("content")) or "")
        predicted = extract_label(raw_text)
        distribution = label_distribution(choice)
        progress, probabilities = progress_from_distribution(
            distribution, predicted_label=predicted
        )
        return {
            "predicted_level": predicted,
            "progress": progress,
            "probabilities": probabilities,
            "distribution": distribution,
            "raw_response": raw_text,
            "latency_s": round(latency_s, 3),
            "returned_model": body.get("model"),
            "usage": body.get("usage") or {},
        }


def evidence_to_json(action: ActionEvidence) -> dict[str, Any]:
    payload = asdict(action)
    for key in ("before_canvas", "after_canvas", "action_video", "poster"):
        value = payload[key]
        payload[key] = str(value) if value is not None else None
    return payload


__all__ = [
    "ActionEvidence",
    "PROGRESS_LEVELS",
    "ProgressCriticClient",
    "SYSTEM_PROMPT",
    "add_progress_dynamics",
    "build_action_evidence",
    "build_action_messages",
    "build_initial_messages",
    "evidence_to_json",
    "extract_label",
    "initial_canvas",
    "label_distribution",
    "progress_from_distribution",
    "sha256",
    "trace_task",
    "write_action_strip",
]
