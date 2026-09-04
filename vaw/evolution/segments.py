"""从原始 trace 读取并渲染 Investigator 指定的物理动作区间。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from vaw.diagnostics.progress_critic import ActionEvidence, build_action_evidence
from vaw.evolution.artifacts import read_jsonl
from vaw.evolution.atlas import ATLAS_SCHEMA


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        size,
    )


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    target = Image.new("RGB", size, "#e8eef5")
    target.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return target


def _status(step: Mapping[str, Any]) -> str:
    result = step.get("function_result")
    if isinstance(result, Mapping) and result.get("error") is not None:
        return "error"
    return "ok"


def _event(step: Mapping[str, Any], *, prefix: str = "") -> str | None:
    call = step.get("function_call")
    if not isinstance(call, Mapping) or not isinstance(call.get("name"), str):
        return None
    arguments = call.get("arguments")
    visible = (
        {str(key): value for key, value in arguments.items() if key != "action_id"}
        if isinstance(arguments, Mapping)
        else {}
    )
    compact = json.dumps(visible, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}{call['name']} {compact} → {_status(step)}"


@dataclass(frozen=True)
class SegmentEvidence:
    """一次真实工具读取结果；路径只用于离线 provenance。"""

    start_action: int
    end_action: int
    question: str
    actions: tuple[str, ...]
    image: Path

    @property
    def key(self) -> tuple[int, int]:
        return self.start_action, self.end_action


class TraceAccessor:
    """把 Atlas 中的动作编号映射回原始 policy-visible trace。"""

    def __init__(self, atlas: Mapping[str, Any]) -> None:
        if atlas.get("schema") != ATLAS_SCHEMA:
            raise ValueError(f"不支持的 Atlas schema: {atlas.get('schema')!r}")
        self.atlas = atlas
        self.trace_dir = Path(str(atlas["trace"])).resolve()
        self.actions = build_action_evidence(self.trace_dir)
        if len(self.actions) != len(atlas["actions"]):
            raise ValueError("Atlas 与原 trace 的物理动作数量不一致")
        self.steps = read_jsonl(self.trace_dir / "steps.jsonl")

    def _semantic_events(
        self,
        start: ActionEvidence,
        end: ActionEvidence,
    ) -> tuple[str, ...]:
        previous_turn = self.actions[start.index - 2].turn if start.index > 1 else 0
        events: list[str] = []
        for step in self.steps:
            turn = step.get("turn")
            if not isinstance(turn, int) or not previous_turn < turn <= end.turn:
                continue
            line = _event(step)
            if line is not None:
                events.append(line)
            diagnostics = step.get("runtime_diagnostics")
            subagent = (
                diagnostics.get("subagent")
                if isinstance(diagnostics, Mapping)
                else None
            )
            relative = subagent.get("trace") if isinstance(subagent, Mapping) else None
            if not isinstance(relative, str):
                continue
            subtrace = (self.trace_dir / relative).resolve()
            try:
                subtrace.relative_to(self.trace_dir)
            except ValueError as exc:
                raise ValueError(f"subagent trace 越出 episode: {relative}") from exc
            for substep in read_jsonl(subtrace / "steps.jsonl"):
                subline = _event(substep, prefix="imagination.")
                if subline is not None:
                    events.append(subline)
        return tuple(events)

    def inspect(
        self,
        start_action: int,
        end_action: int,
        question: str,
        output_path: str | Path,
    ) -> SegmentEvidence:
        """读取闭区间动作；每个动作显示决策时与执行后 Canvas。"""

        if start_action < 1 or end_action < start_action or end_action > len(self.actions):
            raise ValueError(f"动作区间必须位于 A1..A{len(self.actions)}")
        question = question.strip()
        if not question:
            raise ValueError("inspect_segment.question 不能为空")
        selected = self.actions[start_action - 1 : end_action]
        evidence = _render_segment(selected, Path(output_path).resolve())
        return SegmentEvidence(
            start_action=start_action,
            end_action=end_action,
            question=question,
            actions=self._semantic_events(selected[0], selected[-1]),
            image=evidence,
        )


def _render_segment(actions: list[ActionEvidence], output_path: Path) -> Path:
    panels: list[tuple[str, Path]] = []
    for action in actions:
        panels.extend(
            (
                (f"A{action.index} DECISION · PREVIEW MAY APPEAR", action.before_canvas),
                (f"A{action.index} AFTER · OBSERVED", action.after_canvas),
            )
        )
    panel_width, panel_height = 1160, 725
    rows = math.ceil(len(panels) / 2)
    header = 92
    canvas = Image.new("RGB", (2400, header + rows * 790 + 20), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 20),
        f"SEGMENT A{actions[0].index}–A{actions[-1].index}",
        font=_font(31, bold=True),
        fill="#172554",
    )
    for index, (label, path) in enumerate(panels):
        row, column = divmod(index, 2)
        x = 25 + column * 1185
        y = header + row * 790
        draw.rounded_rectangle(
            (x, y, x + panel_width, y + panel_height + 48),
            radius=10,
            fill="#ffffff",
            outline="#94a3b8",
            width=2,
        )
        draw.text((x + 12, y + 10), label, font=_font(20, bold=True), fill="#0f172a")
        with Image.open(path) as source:
            panel = _fit(source, (panel_width - 20, panel_height - 10))
        canvas.paste(panel, (x + 10, y + 48))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=False)
    return output_path


__all__ = ["SegmentEvidence", "TraceAccessor"]
