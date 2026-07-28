"""Minimal planner tracing: drop prompt/response/decision to disk."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from robomex.contracts import PlannerStep


class PlannerTracer:
    """Write planner step artifacts to ``<output_dir>/planner/step-N/``.

    Three files per step: ``prompt.json`` (what the model saw),
    ``response.txt`` (raw model text), ``step.json`` (parsed decision).
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def record(
        self,
        step_index: int,
        planner: Any,
        step: PlannerStep,
    ) -> Path:
        step_dir = self.output_dir / "planner" / f"step-{step_index}"
        step_dir.mkdir(parents=True, exist_ok=True)

        prompt = getattr(planner, "last_prompt", None)
        (step_dir / "prompt.json").write_text(
            json.dumps(prompt if prompt is not None else [], ensure_ascii=False, indent=2),
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
        return step_dir
