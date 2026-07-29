"""M0 smoke test for the agent runtime: scripted provider, fake robot, no env.

    python -m vaw.scripts.smoke_runtime

Drives :class:`~vaw.agents.runtime.VAWRuntime` through the paths that are
awkward to reach with a real model, and asserts each one behaves:

- a reply with no op call, narrating fabricated results (hallucination rebuke)
- two op calls in one reply (first runs, second refused)
- an unknown op name and a call missing a required argument (error receipts)
- the normal propose -> select -> preview -> commit -> gripper -> done path
- exhausting the turn budget (runtime commits ``done(success=False)`` itself)
- the cognitive ops on a real workspace: ground / inspect (focus) / view, incl.
  view-envelope clamping and that a viewpoint change reaches the canvas

It also checks the canvas window really is bounded, since an unbounded one
breaks silently: everything works, just slower and more expensive each turn.
"""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any

import numpy as np

from vaw.agents.contracts import ModelResponse, TerminateMode, ToolCall
from vaw.agents.providers.text_protocol import TextProtocolProvider
from vaw.agents.runtime import RunConfig, VAWRuntime
from vaw.scripts.smoke_render import synthetic_obs
from vaw.workspace import Workspace

OUT = pathlib.Path(__file__).resolve().parent.parent / "out" / "smoke_runtime"


class FakeApi:
    """Minimal stand-in for ``FrankaLiberoApiReduced``.

    Only the surface the ops actually touch. IK accepts anything within reach
    and the controller teleports, so this exercises the runtime's plumbing, not
    the physics — M1 is where a real environment takes over.
    """

    camera_name = "agentview"
    wrist_camera_name = "robot0_eye_in_hand"

    def __init__(self) -> None:
        self._obs, self._mask = synthetic_obs()
        self._ee = np.array([0.30, -0.10, 0.45], dtype=np.float64)
        self._quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        self._gripper = 1.0
        self.calls: list[str] = []

    def get_observation(self) -> dict[str, Any]:
        obs = dict(self._obs)
        obs["robot_cartesian_pos"] = np.concatenate([self._ee, self._quat, [self._gripper]])
        return obs

    # -- perception: the synthetic scene's own ground truth, so `ground` has a
    #    success path here without a detector or a segmentation service.
    def vlm_bbox_detection(self, rgb, text: str) -> list[float]:
        self.calls.append("vlm_bbox_detection")
        vs, us = np.nonzero(self._mask)
        return [float(us.min()), float(vs.min()), float(us.max()), float(vs.max())]

    def segment_sam3_box_prompt(self, rgb, box) -> list[dict[str, Any]]:
        self.calls.append("segment_sam3_box_prompt")
        return [{"mask": self._mask, "score": 0.93}]

    def get_oriented_bounding_box_from_3d_points(self, points) -> dict[str, Any]:
        pts = np.asarray(points, dtype=np.float64)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return {"center": (lo + hi) / 2, "extent": hi - lo, "R": np.eye(3)}

    def solve_ik(self, position, quat_wxyz, return_info: bool = False):
        self.calls.append("solve_ik")
        reachable = float(np.linalg.norm(np.asarray(position)[:2])) < 1.0
        info = {"orientation_used": "requested" if reachable else "top_down_fallback"}
        joints = np.zeros(7)
        return (joints, info) if return_info else joints

    def goto_pose(self, position, quat_wxyz, z_approach: float = 0.0) -> None:
        self.calls.append("goto_pose")
        self._ee = np.asarray(position, dtype=np.float64).copy()
        self._quat = np.asarray(quat_wxyz, dtype=np.float64).copy()

    def open_gripper(self) -> None:
        self.calls.append("open_gripper")
        self._gripper = 1.0

    def close_gripper(self) -> None:
        self.calls.append("close_gripper")
        # Closing on nothing: the receipt should carry the "grasped nothing" warning.
        self._gripper = 0.01


class ScriptedProvider:
    """Replays a fixed list of replies, then repeats the last one forever."""

    def __init__(self, replies: list[ModelResponse]) -> None:
        self.replies = replies
        self.index = 0
        self.seen_images: list[int] = []

    def generate(self, messages, tools=None) -> ModelResponse:
        self.seen_images.append(_count_images(messages))
        reply = self.replies[min(self.index, len(self.replies) - 1)]
        self.index += 1
        return reply


def _count_images(messages) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    )


def _call(name: str, **args: Any) -> ToolCall:
    return ToolCall(id=f"call_{name}_{len(args)}", name=name, args=args)


def _reply(*calls: ToolCall, text: str = "") -> ModelResponse:
    return ModelResponse(text=text, tool_calls=tuple(calls))


# --------------------------------------------------------------------- #
def scripted_episode() -> None:
    """The main path plus every protocol-violation branch."""

    replies = [
        # 1. No op call, and narrating results it never got.
        _reply(text="<tool_call>ground</tool_call>\n<tool_result>obj1 grounded</tool_result>"),
        # 2. Two ops in one reply: only the first may run.
        _reply(
            _call("observe"),
            _call("inspect", object_id="obj1"),
            text="Let me look and inspect at once.",
        ),
        # 3. An op that does not exist.
        _reply(_call("teleport", position=[0, 0, 0])),
        # 4. A real op missing a required argument.
        _reply(_call("propose_pose", kind="waypoint")),
        # 5. Onward, correctly.
        _reply(_call("propose_pose", kind="waypoint", position=[0.45, 0.05, 0.2])),
        _reply(_call("select", candidate_id="p1")),
        _reply(_call("preview", candidate_id="p1"), text="Check IK and clearance first."),
        _reply(_call("commit")),
        _reply(_call("commit_gripper", action="close")),
        _reply(_call("done", success=True)),
    ]

    api = FakeApi()
    provider = ScriptedProvider(replies)
    trace_dir = OUT / "episode"
    workspace = Workspace(
        api, "pick up the red mug", trace_dir=trace_dir, env_check=lambda: True
    )
    runtime = VAWRuntime(provider, workspace, RunConfig(max_turns=20, canvas_window_k=3))
    result = runtime.run()

    ops = [(record.op, record.ok) for record in result.steps]
    print("executed ops:", ops)

    assert result.terminate_mode is TerminateMode.GOAL, result.terminate_mode
    assert result.claimed_success is True
    assert result.ended_by_agent

    # Turn 1 called nothing, so it produced no step at all.
    assert ops[0] == ("observe", True), ops
    # Bad op name and bad arguments both come back as failed steps, not crashes.
    assert ("teleport", False) in ops, ops
    assert ("propose_pose", False) in ops, ops
    assert ops[-1] == ("done", True), ops

    # The extra op in reply 2 must have been answered but not executed, or the
    # endpoint would reject the following request.
    assert "inspect" not in [op for op, _ in ops], ops
    refusals = [
        message
        for message in runtime.chat.history
        if message.get("role") == "tool" and "exactly one operation" in str(message.get("content"))
    ]
    assert len(refusals) == 1, refusals
    assert _orphaned_call_ids(runtime.chat.history) == [], "unanswered tool call ids"

    # The rebuke for the fabricated transcript, not the generic nudge.
    rebukes = [
        message
        for message in runtime.chat.history
        if message.get("role") == "user" and "none of them ran" in str(message.get("content"))
    ]
    assert len(rebukes) == 1, rebukes

    # Physical ops really reached the robot, in order.
    assert api.calls.count("goto_pose") == 1, api.calls
    assert api.calls.count("close_gripper") == 1, api.calls

    # Closing on nothing has to be visible in the receipt the agent reads.
    gripper_step = next(r for r in result.steps if r.op == "commit_gripper")
    assert "grasped nothing" in gripper_step.receipt, gripper_step.receipt

    # Reasoning text is captured for SFT; the receipt-bearing turn kept its thought.
    preview_step = next(r for r in result.steps if r.op == "preview")
    assert preview_step.thought.startswith("Check IK"), preview_step.thought

    # Canvas window stays bounded no matter how long the episode runs.
    assert max(provider.seen_images) <= 3, provider.seen_images

    trace = (trace_dir / "steps.jsonl").read_text().strip().splitlines()
    canvases = sorted(p.name for p in trace_dir.glob("canvas_*.png"))
    print(f"trace: {len(trace)} steps, {len(canvases)} canvases in {trace_dir}")
    # Every step is logged, including the opening observe and the rejected ones.
    assert len(trace) == len(result.steps) + 1, (len(trace), len(result.steps))
    assert json.loads(trace[-1])["op"] == "done"

    # The env verdict lands in the result and in meta.json — and nowhere the
    # agent can see it (no receipt, no state summary).
    assert result.env_success is True
    meta = json.loads((trace_dir / "meta.json").read_text())
    assert meta["env_success"] is True and meta["claimed_success"] is True, meta
    assert all(
        "env_success" not in str(message.get("content")) for message in runtime.chat.history
    ), "privileged verdict leaked into the agent context"


def budget_exhausted() -> None:
    """A model that never calls anything must still end with a logged ``done``."""

    api = FakeApi()
    provider = ScriptedProvider([_reply(text="I am thinking about it.")])
    workspace = Workspace(api, "pick up the red mug", trace_dir=OUT / "budget")
    result = VAWRuntime(provider, workspace, RunConfig(max_turns=4)).run()

    assert result.terminate_mode is TerminateMode.MAX_TURNS, result.terminate_mode
    assert result.claimed_success is False
    assert result.steps[-1].op == "done", result.steps
    assert "forced" in result.steps[-1].receipt
    assert workspace.finished
    print(f"budget path: {result.terminate_mode.value} after {result.turns} turns, done forced")


def text_protocol_episode() -> None:
    """The student path: ``<tool_call>`` blocks instead of native calls.

    Beyond checking that ops still run, this asserts the two things
    ``rewrite_history`` can silently break — the canvas images surviving the
    rewrite (lose them and the model is blind while everything still "works"),
    and receipts arriving as ``<tool_result>`` rather than orphaned tool
    messages the endpoint would reject.
    """

    class ScriptedTextProvider:
        """Returns raw text; the wrapper does the parsing."""

        def __init__(self, texts: list[str]) -> None:
            self.texts = texts
            self.index = 0
            self.last_messages: list[Any] = []

        def generate(self, messages, tools=None) -> ModelResponse:
            self.last_messages = messages
            # The wrapper must not forward tool schemas to the endpoint.
            assert tools is None, "text protocol leaked tools to the endpoint"
            text = self.texts[min(self.index, len(self.texts) - 1)]
            self.index += 1
            return ModelResponse(text=text)

    inner = ScriptedTextProvider(
        [
            'Grounding first.\n<tool_call>{"name": "observe", "arguments": {}}</tool_call>',
            '<tool_call>{"name": "propose_pose", "arguments": '
            '{"kind": "waypoint", "position": [0.4, 0.0, 0.25]}}</tool_call>',
            '<tool_call>{"name": "done", "arguments": {"success": false}}</tool_call>',
        ]
    )
    api = FakeApi()
    workspace = Workspace(api, "reach above the plate", trace_dir=OUT / "text_protocol")
    result = VAWRuntime(
        TextProtocolProvider(inner), workspace, RunConfig(max_turns=8)
    ).run()

    ops = [(record.op, record.ok) for record in result.steps]
    assert ops == [("observe", True), ("propose_pose", True), ("done", True)], ops
    assert result.terminate_mode is TerminateMode.GOAL
    assert result.steps[0].thought == "Grounding first."

    rewritten = inner.last_messages
    assert _count_images(rewritten) >= 1, "canvas lost in the history rewrite"
    assert any(
        "<tool_result" in str(message.get("content")) for message in rewritten
    ), "receipts not rendered as tool results"
    assert all(message.get("role") != "tool" for message in rewritten), "orphaned tool message"
    system = str(rewritten[0]["content"])
    assert "propose_pose" in system and "Exactly one" in system, "op docs missing from prompt"
    print(f"text protocol: {ops}, {_count_images(rewritten)} canvas image(s) preserved")


def perception_and_view_ops() -> None:
    """The cognitive half through a real ``Workspace``: ground, inspect, view.

    Driven by direct ``step`` calls rather than a scripted model, because what
    is under test is the ops and the canvas they produce, not the loop. Also
    asserts ``view`` actually changes the picture: an op the agent can call that
    silently does nothing is the worst kind of interface bug — every trace looks
    fine and the policy learns that looking around is free and useless.
    """

    api = FakeApi()
    ws = Workspace(api, "pick the alphabet soup", trace_dir=OUT / "views")
    ws.step("observe")
    result = ws.step("ground", text="alphabet soup")
    assert result.ok, result.receipt_text
    assert ws.state.summary()["focus"] == {"object_id": "obj1", "requested": False}
    physical_canvas = result.canvas

    result = ws.step("inspect", object_id="obj1")
    assert result.ok and ws.state.focus_id == "obj1", result.receipt_text
    assert ws.state.summary()["focus"]["requested"] is True

    result = ws.step("view", preset="top")
    assert result.ok and not ws.state.view.is_physical, result.receipt_text
    assert not np.array_equal(result.canvas, physical_canvas), "view did not change the canvas"
    top_canvas = result.canvas

    # Out-of-envelope requests are clamped, and the clamp is in the receipt so
    # the agent can learn the envelope instead of guessing at silent failures.
    base = ws.state.view.base_azimuth_deg
    result = ws.step("view", azimuth_deg=base + 200)
    assert result.ok and "clamped" in result.receipt_text, result.receipt_text
    assert abs(ws.state.view.azimuth_deg - base) <= 75.0 + 1e-6, ws.state.view

    for bad in ({"preset": "bogus"}, {}):
        result = ws.step("view", **bad)
        assert not result.ok, (bad, result.receipt_text)

    result = ws.step("view", preset="agentview")
    assert ws.state.view.is_physical, ws.state.view
    assert not np.array_equal(result.canvas, top_canvas)

    canvases = sorted(p.name for p in (OUT / "views").glob("canvas_*.png"))
    print(f"perception/view ops: {len(canvases)} canvases, focus={ws.state.focus_id}")


def _orphaned_call_ids(history) -> list[str]:
    """Tool call ids with no matching tool message — a hard protocol error."""
    answered = {
        message.get("tool_call_id") for message in history if message.get("role") == "tool"
    }
    orphans = []
    for message in history:
        for call in message.get("tool_calls") or []:
            if call.get("id") not in answered:
                orphans.append(call.get("id"))
    return orphans


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    scripted_episode()
    perception_and_view_ops()
    text_protocol_episode()
    budget_exhausted()
    print("runtime smoke OK")


if __name__ == "__main__":
    main()
