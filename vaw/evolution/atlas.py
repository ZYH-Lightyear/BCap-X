"""把完整 rollout 编译成供 Investigator 导航的 episode 级视觉 Atlas。

Atlas 只提供全局走势和稀疏视觉索引。它不是失败分类器，也不替调查 Agent
决定窗口；高分辨率证据只能通过 ``inspect_segment`` 按需读取。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from vaw.diagnostics.progress_critic import (
    ActionEvidence,
    build_action_evidence,
    initial_canvas,
    trace_task,
)
from vaw.evolution.artifacts import read_json, relative_asset, write_json

ATLAS_SCHEMA = "vaw-episode-atlas-v1"
TOPREWARD_SCHEMA = "vaw-topreward-progress-v1"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size=size)


def _episode(trace_dir: Path) -> dict[str, Any]:
    meta = read_json(trace_dir / "meta.json")
    success = meta.get("env_success")
    if not isinstance(success, bool):
        raise ValueError("M3 只分析具有布尔 env_success 的完整 episode")
    return {
        "suite": str(meta["suite"]),
        "task_id": int(meta["task_id"]),
        "seed": int(meta["seed"]),
        "task": trace_task(trace_dir),
        "env_success": success,
    }


def _progress(
    path: Path,
    *,
    trace_dir: Path,
    actions: Sequence[ActionEvidence],
) -> list[dict[str, float]]:
    """严格对齐原始 TOPReward；校准后的显示分数不进入 M3 推理。"""

    payload = read_json(path)
    if payload.get("schema") != TOPREWARD_SCHEMA:
        raise ValueError(f"不支持的 TOPReward schema: {payload.get('schema')!r}")
    declared_trace = payload.get("trace")
    if not isinstance(declared_trace, str) or Path(declared_trace).resolve() != trace_dir:
        raise ValueError("TOPReward 结果不属于当前 trace")
    states = payload.get("states")
    if not isinstance(states, list):
        raise ValueError("TOPReward states 必须是 list")
    expected = [("initial", 0, "initial_state"), *[
        (action.segment_id, action.turn, action.function) for action in actions
    ]]
    if len(states) != len(expected):
        raise ValueError("TOPReward 状态数量与物理动作不一致")

    aligned: list[dict[str, float]] = []
    for state, identity in zip(states, expected, strict=True):
        if not isinstance(state, Mapping):
            raise ValueError("TOPReward state 必须是 object")
        actual = (state.get("segment_id"), state.get("turn"), state.get("function"))
        if actual != identity:
            raise ValueError(f"TOPReward 边界错位: expected={identity!r}, actual={actual!r}")
        reward = float(state["raw_reward"])
        delta = float(state["raw_delta"])
        if not math.isfinite(reward) or not math.isfinite(delta):
            raise ValueError("TOPReward 原始分数必须是有限值")
        aligned.append({"reward": reward, "delta": delta})
    return aligned


def _outcome(value: str) -> str:
    match = re.match(r"[A-Za-z][A-Za-z0-9_-]*", value.strip())
    return match.group(0).casefold() if match else "unknown"


def _visible_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """动作 ID 对离线视觉诊断没有语义，其余参数保持原始因果事实。"""

    return {str(key): value for key, value in arguments.items() if key != "action_id"}


def compile_atlas(
    trace_dir: str | Path,
    progress_path: str | Path,
) -> dict[str, Any]:
    """编译最小 Atlas 文档；Action i 表示 state i-1 到 state i。"""

    trace = Path(trace_dir).resolve()
    progress_source = Path(progress_path).resolve()
    actions = build_action_evidence(trace)
    scores = _progress(progress_source, trace_dir=trace, actions=actions)
    first = initial_canvas(trace)
    entries: list[dict[str, Any]] = []
    for action, score in zip(actions, scores[1:], strict=True):
        entries.append(
            {
                "index": action.index,
                "turn": action.turn,
                "function": action.function,
                "arguments": _visible_arguments(action.arguments),
                "outcome": _outcome(action.outcome),
                "frames": [action.frame_start, action.frame_end],
                "canvas": relative_asset(trace, action.after_canvas),
                "progress": score,
            }
        )
    return {
        "schema": ATLAS_SCHEMA,
        "episode": _episode(trace),
        "trace": str(trace),
        "progress_source": str(progress_source),
        "initial": {
            "canvas": relative_asset(trace, first),
            "progress": scores[0],
        },
        "actions": entries,
    }


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """保持比例填充卡片；Atlas 缩略图只承担导航，不裁掉场景边缘。"""

    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    background = Image.new("RGB", size, "#eef3f8")
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    background.paste(copy, (x, y))
    return background


def _action_label(action: Mapping[str, Any]) -> str:
    arguments = action.get("arguments")
    function = str(action["function"])
    if function == "move_tcp_delta" and isinstance(arguments, Mapping):
        delta = arguments.get("delta_xyz_m")
        if isinstance(delta, list) and len(delta) == 3:
            axes = ("x", "y", "z")
            values = [
                f"d{axis} {float(value) * 100:+.1f}"
                for axis, value in zip(axes, delta, strict=True)
                if abs(float(value)) > 1e-9
            ]
            return f"move {' '.join(values) or '0'} cm"
    if isinstance(arguments, Mapping) and arguments:
        compact = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        return f"{function} {compact[:24]}"
    return function


def render_atlas(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    """渲染曲线与逐动作 storyboard，保留所有物理边界而不先选窗口。"""

    trace = Path(str(payload["trace"]))
    actions = list(payload["actions"])
    states = [payload["initial"], *actions]
    width = 1800
    curve_height = 390
    columns = 4
    card_width = 425
    image_height = 235
    card_height = 300
    rows = math.ceil(len(states) / columns)
    height = curve_height + rows * card_height + 40
    canvas = Image.new("RGB", (width, height), "#f4f7fb")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30, bold=True)
    body_font = _font(20)
    small_font = _font(17)
    mono_font = _font(17)

    episode = payload["episode"]
    outcome = "SUCCESS" if episode["env_success"] else "FAILURE"
    outcome_color = "#15803d" if episode["env_success"] else "#b91c1c"
    draw.text((32, 22), "EPISODE ATLAS", font=title_font, fill="#172554")
    draw.text((32, 68), str(episode["task"]), font=body_font, fill="#111827")
    draw.text((width - 210, 26), outcome, font=title_font, fill=outcome_color)

    rewards = [float(state["progress"]["reward"]) for state in states]
    left, top, right, bottom = 80, 125, width - 70, 340
    draw.rectangle((left, top, right, bottom), fill="#ffffff", outline="#cbd5e1", width=2)
    minimum, maximum = min(rewards), max(rewards)
    span = maximum - minimum or 1.0
    points: list[tuple[float, float]] = []
    for index, reward in enumerate(rewards):
        x = left + (right - left) * index / max(1, len(rewards) - 1)
        y = bottom - 25 - (bottom - top - 50) * (reward - minimum) / span
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill="#2563eb", width=5)
    for index, (x, y) in enumerate(points):
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#ffffff", outline="#1d4ed8", width=4)
        draw.text((x - 12, bottom + 4), "S0" if index == 0 else f"A{index}", font=small_font, fill="#334155")
    draw.text((left + 12, top + 10), "TOPReward raw curve · navigation clue, not verdict", font=small_font, fill="#475569")
    draw.text((right - 190, top + 10), f"raw [{minimum:.2f}, {maximum:.2f}]", font=small_font, fill="#475569")

    for index, state in enumerate(states):
        row, column = divmod(index, columns)
        x = 25 + column * (card_width + 20)
        y = curve_height + row * card_height
        draw.rounded_rectangle(
            (x, y, x + card_width, y + card_height - 14),
            radius=10,
            fill="#ffffff",
            outline="#cbd5e1",
            width=2,
        )
        path = trace / str(state["canvas"])
        with Image.open(path) as source:
            thumbnail = _fit(source, (card_width - 20, image_height))
        canvas.paste(thumbnail, (x + 10, y + 42))
        label = "S0 · INITIAL" if index == 0 else f"A{index} · {_action_label(state)}"
        draw.text((x + 12, y + 10), label[:24], font=mono_font, fill="#0f172a")
        reward = float(state["progress"]["reward"])
        delta = float(state["progress"]["delta"])
        draw.text(
            (x + card_width - 154, y + 10),
            f"R {reward:+.2f}  Δ {delta:+.2f}",
            font=small_font,
            fill="#1d4ed8" if delta >= 0 else "#b91c1c",
        )

    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=False)
    return target


def compile_to_dir(
    trace_dir: str | Path,
    progress_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = compile_atlas(trace_dir, progress_path)
    document = write_json(output / "atlas.json", payload)
    image = render_atlas(payload, output / "atlas.png")
    return document, image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document, image = compile_to_dir(args.trace, args.progress, args.output_dir)
    payload = read_json(document)
    print(f"[m3-atlas] actions={len(payload['actions'])} json={document} image={image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ATLAS_SCHEMA", "compile_atlas", "compile_to_dir", "render_atlas"]
