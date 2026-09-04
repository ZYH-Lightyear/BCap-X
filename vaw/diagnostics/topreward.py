"""TOPReward 的 VAW 轨迹适配与冻结 Qwen3-VL 打分器。"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image

from vaw.diagnostics.progress_critic import ActionEvidence

TOPREWARD_PROMPT_PREFIX = (
    "The above video shows a robot manipulation trajectory that completes "
    "the following task: "
)
TOPREWARD_PROMPT_SUFFIX = (
    "{instruction} Decide whether the above statement is True or not. "
    "The answer is: True"
)
TOPREWARD_REFERENCE_URL = "https://github.com/TOPReward/TOPReward"
TOPREWARD_REFERENCE_COMMIT = "4877a0ee5098cbec18485125466631b1dcc4a573"


@dataclass(frozen=True)
class PhysicalPrefix:
    """一个真实物理 observation 及其之前的完整视频前缀。"""

    state_index: int
    segment_id: str
    turn: int
    function: str
    time_s: float
    frame_index: int
    sampled_frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class TopRewardScore:
    """单个轨迹前缀的原始 token log-probability。"""

    raw_log_prob: float
    token_count: int
    scored_token_ids: tuple[int, ...]
    scored_tokens: tuple[str, ...]


class TopRewardScorer(Protocol):
    """便于替换模型和离线测试的最小打分边界。"""

    @property
    def model_name(self) -> str: ...

    def score(self, *, frames: Sequence[np.ndarray], instruction: str) -> TopRewardScore: ...


def uniform_prefix_indices(end_index: int, max_frames: int) -> tuple[int, ...]:
    """均匀采样 ``[0, end_index]``，并始终保留首尾真实帧。"""

    if end_index < 0:
        raise ValueError("end_index must be non-negative")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    count = min(end_index + 1, max_frames)
    if count == 1:
        return (end_index,)
    return tuple(
        sorted({int(round(value)) for value in np.linspace(0, end_index, count)})
    )


def build_physical_prefixes(
    *, actions: Sequence[ActionEvidence], frame_count: int, max_frames: int
) -> list[PhysicalPrefix]:
    """按物理动作结束边界生成 TOPReward 前缀，不把 Preview 当成新状态。"""

    if frame_count <= 0:
        raise ValueError("agentview video has no frames")
    prefixes = [
        PhysicalPrefix(
            state_index=0,
            segment_id="initial",
            turn=0,
            function="initial_state",
            time_s=0.0,
            frame_index=0,
            sampled_frame_indices=uniform_prefix_indices(0, max_frames),
        )
    ]
    for action in actions:
        frame_index = max(0, action.frame_end - 1)
        if frame_index >= frame_count:
            raise IndexError(
                f"action {action.segment_id} ends at frame {frame_index}, "
                f"but video contains {frame_count} frames"
            )
        prefixes.append(
            PhysicalPrefix(
                state_index=action.index,
                segment_id=action.segment_id,
                turn=action.turn,
                function=action.function,
                time_s=action.time_end_s,
                frame_index=frame_index,
                sampled_frame_indices=uniform_prefix_indices(frame_index, max_frames),
            )
        )
    return prefixes


def read_sampled_rgb_frames(
    video_path: str | Path, prefixes: Sequence[PhysicalPrefix]
) -> dict[int, np.ndarray]:
    """单次顺序解码所有前缀需要的 AgentView 帧。"""

    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    needed = {
        frame_index
        for prefix in prefixes
        for frame_index in prefix.sampled_frame_indices
    }
    if not needed:
        raise ValueError("no frames requested")

    frames: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(source))
    try:
        frame_index = 0
        maximum = max(needed)
        while frame_index <= maximum:
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise ValueError(f"cannot decode frame {frame_index} from {source}")
            if frame_index in needed:
                frames[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_index += 1
    finally:
        capture.release()
    missing = needed.difference(frames)
    if missing:
        raise ValueError(f"missing sampled frames from {source}: {sorted(missing)}")
    return frames


def minmax_normalize(values: Sequence[float]) -> list[float]:
    """复现 TOPReward 的 episode 内 min-max，仅用于曲线可视化。"""

    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def calibrate_progress(
    values: Sequence[float],
    *,
    env_success: bool | None,
    penalty_min: float = 0.05,
    penalty_max: float = 0.15,
) -> list[float]:
    """用终局失败标签线性校准显示曲线，不改变 TOPReward 原始分数。"""

    if not 0.0 <= penalty_min <= penalty_max <= 1.0:
        raise ValueError("failure penalty must satisfy 0 <= min <= max <= 1")
    if env_success is not False:
        return [float(value) for value in values]
    calibrated: list[float] = []
    for value in values:
        progress = min(1.0, max(0.0, float(value)))
        penalty = penalty_max - (penalty_max - penalty_min) * progress
        calibrated.append(progress * (1.0 - penalty))
    return calibrated


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


class QwenTopRewardScorer:
    """复现官方 Qwen3-VL 的最后一个 ``True`` token 打分路径。"""

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        fps: float = 2.0,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            import torch.nn.functional as functional
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - 只在真实模型环境执行
            raise RuntimeError(
                "TOPReward requires torch, transformers, accelerate and qwen-vl-utils"
            ) from exc

        if dtype not in {"bfloat16", "float16", "float32", "auto"}:
            raise ValueError(f"unsupported dtype: {dtype}")
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        self._model_name = model_name
        self._device = device
        self._fps = fps
        self._torch = torch
        self._functional = functional
        self._process_vision_info = process_vision_info
        self._processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation=attention_implementation,
            local_files_only=local_files_only,
        )
        self._model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(
        self, *, frames: Sequence[np.ndarray], instruction: str
    ) -> TopRewardScore:
        if not frames:
            raise ValueError("TOPReward requires at least one frame")
        pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
        content = [
            {"type": "video", "video": pil_frames, "fps": self._fps},
            {"type": "text", "text": TOPREWARD_PROMPT_PREFIX},
        ]
        messages = [{"role": "user", "content": content}]
        eos_token = self._processor.tokenizer.eos_token
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if eos_token is not None:
            prompt = prompt.split(eos_token)[0]
        full_text = f"{prompt}{TOPREWARD_PROMPT_SUFFIX.format(instruction=instruction)}"
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        # 官方实现只监督最后一个 token，即提示末尾的 True。
        labels = inputs["input_ids"].clone()
        labels[:, : labels.shape[1] - 1] = -100
        if "attention_mask" in inputs:
            labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        logits = outputs.logits[:, :-1, :].float()
        target_labels = labels[:, 1:]
        mask = target_labels != -100
        safe_targets = target_labels.masked_fill(~mask, 0)
        log_probs = self._functional.log_softmax(logits, dim=-1)
        selected = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)[mask]
        token_ids = tuple(int(value) for value in target_labels[mask].detach().cpu())
        tokens = tuple(self._processor.tokenizer.convert_ids_to_tokens(list(token_ids)))
        return TopRewardScore(
            raw_log_prob=float(selected.mean().item()),
            token_count=len(token_ids),
            scored_token_ids=token_ids,
            scored_tokens=tokens,
        )


__all__ = [
    "PhysicalPrefix",
    "QwenTopRewardScorer",
    "TOPREWARD_PROMPT_PREFIX",
    "TOPREWARD_REFERENCE_COMMIT",
    "TOPREWARD_REFERENCE_URL",
    "TOPREWARD_PROMPT_SUFFIX",
    "TopRewardScore",
    "TopRewardScorer",
    "build_physical_prefixes",
    "calibrate_progress",
    "minmax_normalize",
    "read_sampled_rgb_frames",
    "uniform_prefix_indices",
    "write_json",
]
