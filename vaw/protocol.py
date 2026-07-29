"""Agent-facing protocol: export the op registry as function-calling tool
definitions, and parse model actions back into (op_name, kwargs).

Teacher (frontier API model) and student (Qwen3-VL) speak exactly this
protocol, so teacher traces are valid student training data byte-for-byte.
"""

from __future__ import annotations

import json
from typing import Any

from vaw.ops import OPS

SYSTEM_PROMPT = """\
You control a robot arm through a Visual Action Workspace. Every step you see
the workspace canvas plus a JSON state summary, and you act by calling exactly
one workspace operation.

# Reading the canvas

- **Main view (left)**: the scene with your annotations drawn on it — coloured
  object masks or boxes with ids, candidate markers, the preview path, and the
  virtual gripper (white) showing the pending action. Its viewpoint is yours to
  move with `view`. The bottom-left compass shows where world +x/+y/+z point in
  the current view; all coordinates you pass to ops are world-frame.
- **DataPanel (top right)**: which marker is which id, and each one's status
  (selected, previewed, stale, focused). It carries no numbers by design.
- **Focus (middle right)**: a zoomed view of one object with all of its
  candidates and their approach axes. Point it with `inspect`.
- **Wrist (bottom right)**: the in-hand camera — the clearest evidence of
  whether something is actually between the fingers.
- Exact numbers (poses, scores, clearances, gripper opening) are in the JSON
  summary, never rendered as text on the canvas. Read them there.
  `gripper_opening` is a fraction, 0 = closed, 1 = fully open — not a length.

# Acting

Core loop: observe -> ground objects -> propose candidates (grasps/poses) ->
select -> preview -> commit. Only `commit` / `commit_gripper` move the real
robot; everything else is free thinking on the workspace, so use it — `view`
and `inspect` cost nothing physical and resolve ambiguity that guessing does
not. Preview before you commit: previews expose IK failures and collisions
before they happen, but they are evidence, not permission — a preview can be
wrong (an intended contact may read as a collision), and the decision is yours.
After every physical operation read the receipt carefully: it reports deviation
from the prediction and warnings such as "gripper likely grasped nothing".
Give gripper open/close its own commit step. Call done when the task is
complete or clearly impossible.
"""


def tool_definitions() -> list[dict[str, Any]]:
    """OpenAI-style function-calling tool list generated from the op registry."""
    tools = []
    for spec in OPS.values():
        properties: dict[str, Any] = {}
        required: list[str] = []
        for pname, meta in spec.params.items():
            properties[pname] = {
                "type": meta.get("type", "string"),
                "description": meta.get("description", ""),
            }
            if meta.get("required", True):
                required.append(pname)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return tools


def parse_action(payload: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Parse a model action into (op_name, kwargs).

    Accepts a dict (native tool call: {"name": ..., "arguments": {...}}) or a
    JSON string of the same shape. Raises ValueError on malformed input; the
    caller converts that into an error receipt for the agent.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"action is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"action must be an object, got {type(payload).__name__}")

    name = payload.get("name") or payload.get("op")
    if not name:
        raise ValueError("action missing 'name'")
    args = payload.get("arguments", payload.get("args", {})) or {}
    if isinstance(args, str):
        args = json.loads(args)
    if name not in OPS:
        raise ValueError(f"unknown op '{name}'; available: {sorted(OPS)}")
    return str(name), dict(args)
