"""落盘 planner 每一步的 prompt / 原始回复 / 决策 / 观测。

一步一个目录,人可以直接翻开看"模型当时看到了什么、说了什么、被解析成了什么"。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from robomex.contracts import Observation, PlannerStep


class PlannerTracer:
    """把 planner 的每一步写到 ``<output_dir>/planner/step-N/``。

    每步最多四个文件:``prompt.json``(模型看到的 prompt)、``response.txt``
    (模型原始回复)、``step.json``(解析后的决策)、``observation.json``
    (这一拍观测的图片路径与说明)。
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def record(
        self,
        step_index: int,
        planner: Any,
        step: PlannerStep,
        observation: Observation | None = None,
    ) -> Path:
        step_dir = self.output_dir / "planner" / f"step-{step_index}"
        step_dir.mkdir(parents=True, exist_ok=True)

        prompt = getattr(planner, "last_prompt", None)
        (step_dir / "prompt.json").write_text(
            json.dumps(_elide_images(prompt or []), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        response = getattr(planner, "last_response", None)
        (step_dir / "response.txt").write_text(
            response if response is not None else "",
            encoding="utf-8",
        )

        step_dict: dict[str, Any] = {
            "thought": step.thought,
            "done": step.done,
            "reason": step.reason,
        }
        if step.intent is not None:
            step_dict["intent"] = asdict(step.intent)

        (step_dir / "step.json").write_text(
            json.dumps(step_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if observation is not None:
            # 图片本身不复制一份到这里 —— observation.image_path 已经指向环境
            # 落盘的那一张,复制只会制造"哪张才是模型真看到的"这种歧义。
            (step_dir / "observation.json").write_text(
                json.dumps(asdict(observation), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return step_dir


def _elide_images(prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 prompt 里的 base64 图片替换成尺寸占位符。

    一张观测图 base64 后有几百 KB,原样写进 prompt.json 会让这个本该用来人工
    速览的文件彻底没法读。图片本体在 ``observation.json`` 指向的路径上,不会丢。
    """
    elided: list[dict[str, Any]] = []
    for message in prompt:
        content = message.get("content")
        if not isinstance(content, list):
            elided.append(message)
            continue
        parts: list[Any] = []
        for part in content:
            url = ""
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = str(part.get("image_url", {}).get("url", ""))
            if url.startswith("data:"):
                parts.append({"type": "image_url", "image_url": {"url": f"<{len(url)} chars elided>"}})
            else:
                parts.append(part)
        elided.append({**message, "content": parts})
    return elided
