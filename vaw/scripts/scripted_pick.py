"""M1.1 acceptance: a scripted op sequence picks an object on a live LIBERO task.

No model in the loop — this is the wiring gate, not the agent gate. With a real
model, environment bugs and agent behaviour confound each other; this script
pins down the environment half first: perception services, grasp planning, IK,
execution, canvas rendering with real data, receipts, and the episode-end env
verdict all have to work before failures can be attributed to the model.

It also issues the cognitive ops (``inspect``, ``view``) so the trace contains
Canvas v2's focus inset and point-cloud viewpoints rendered from real sensor
depth; those canvases are M1.2's review material.

Prerequisites (see ``scripts/start_libero_services.sh``):

    8110 LLM proxy (vlm_bbox_detection)   8114 SAM3
    8115 Contact-GraspNet                 8116 PyRoKi IK

Run inside ``.venv-libero``:

    python -m vaw.scripts.scripted_pick \\
        --suite libero_object_swap --task-id 0 --object "the alphabet soup can"

The ``*_task`` / ``*_swap`` LIBERO-PRO suites are registered in the benchmark
dict but most ship no BDDL files locally; ``libero_object`` / ``libero_spatial``
/ ``libero_goal`` / ``libero_10`` and ``libero_object_swap`` do.

The trace (canvas PNGs + steps.jsonl + meta.json) lands in
``vaw/out/scripted_pick/<suite>_t<task>/`` by default; acceptance is a human
pass over the rendered canvases, previews and receipts.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import socket
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REQUIRED_SERVICES = {
    8110: "LLM proxy (vlm_bbox_detection)",
    8114: "SAM3",
    8115: "Contact-GraspNet",
    8116: "PyRoKi IK",
}


def preflight_services() -> None:
    """Fail fast with a usable message instead of a timeout mid-episode."""
    down = []
    for port, name in REQUIRED_SERVICES.items():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                down.append(f"  :{port}  {name}")
    if down:
        sys.exit(
            "required services are not running:\n"
            + "\n".join(down)
            + "\nstart them with: bash scripts/start_libero_services.sh"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object_swap")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--object",
        required=True,
        help="grounding text for the object to pick, e.g. 'the milk carton' "
        "(the task instruction is printed at startup to help choose)",
    )
    parser.add_argument("--lift", type=float, default=0.15, help="lift height in meters")
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument(
        "--renderer",
        choices=("pil", "web"),
        default="pil",
        help="policy-facing visual renderer; PIL remains the default baseline",
    )
    parser.add_argument(
        "--compare-renderers",
        action="store_true",
        help="save paired PIL/Web artifacts from every state without changing "
        "the policy-facing renderer",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="show the persistent Chromium page when using the Web renderer",
    )
    args = parser.parse_args()

    preflight_services()

    # Imports deferred: capx pulls in mujoco/open3d/JAX, and we want --help and
    # the preflight message to work without that stack.
    from capx.envs.simulators.libero import FrankaLiberoTask
    from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced
    from vaw.executor import EMPTY_GRIP_OPENING
    from vaw.renderers import build_renderer
    from vaw.workspace import Workspace

    print(f"[setup] loading {args.suite} task {args.task_id} (seed {args.seed})")
    env = FrankaLiberoTask(
        suite_name=args.suite,
        task_id=args.task_id,
        privileged=False,
        seed=args.seed,
    )
    instruction = str(env.handle.task_language)
    print(f"[setup] task instruction: {instruction!r}")
    print(f"[setup] grounding target: {args.object!r}")

    api = FrankaLiberoApiReduced(env)
    trace_dir = pathlib.Path(
        args.trace_dir
        or pathlib.Path(__file__).resolve().parent.parent
        / "out"
        / "scripted_pick"
        / f"{args.suite}_t{args.task_id}"
    )
    renderer = build_renderer(
        args.renderer,
        headed=args.headed,
        compare_dir=(trace_dir / "_render_compare") if args.compare_renderers else None,
    )
    ws = Workspace(
        api,
        instruction,
        trace_dir=trace_dir,
        env_check=env.task_completed,
        renderer=renderer,
    )

    failed = False

    def step(op: str, **kwargs) -> object:
        """Run one op, print the receipt, remember failure."""
        nonlocal failed
        result = ws.step(op, **kwargs)
        flag = "ok " if result.ok else "FAIL"
        print(f"[{flag}] {op} {kwargs or ''} -> {result.receipt_text}")
        if not result.ok:
            failed = True
        return result

    # ------------------------------------------------------------------ #
    # The pick sequence. Same ops, same order, as an agent would issue them.
    step("observe")
    step("ground", text=args.object)

    step("propose_grasps", object_id="obj1", top_k=5)
    if failed:
        finish(ws, failed)
        return
    best = max(
        (c for c in ws.state.candidates.values() if c.kind == "grasp"),
        key=lambda c: c.score,
    )
    print(f"[pick] best grasp candidate: {best.candidate_id} (score {best.score:.2f})")

    # Cognitive ops on real data. They change nothing physical, but they are the
    # only way this script produces canvases of the Canvas v2 paths (focus inset,
    # point-cloud virtual view) built from a real depth map rather than the
    # synthetic scene the smoke test uses — which is what M1.2 needs reviewed.
    step("inspect", object_id="obj1")
    for preset in ("top", "left", "agentview"):
        step("view", preset=preset)

    step("commit_gripper", action="open")
    step("select", candidate_id=best.candidate_id)
    step("preview", candidate_id=best.candidate_id)
    step("commit")
    step("commit_gripper", action="close")

    # Lift: a waypoint straight above the grasp, previewed and committed the
    # same way. Losing the object on the way up shows in the next receipts.
    grasp_pos = best.pose.position
    step(
        "propose_pose",
        kind="waypoint",
        position=[float(grasp_pos[0]), float(grasp_pos[1]), float(grasp_pos[2] + args.lift)],
        object_id="obj1",
    )
    waypoint = next(
        c for c in ws.state.candidates.values() if c.kind == "waypoint"
    )
    step("select", candidate_id=waypoint.candidate_id)
    step("preview", candidate_id=waypoint.candidate_id)
    step("commit")

    # Pick verdict: still holding something after the lift? An empty gripper
    # snaps (nearly) fully closed; holding an object keeps the fingers apart.
    step("observe")
    obs = ws.api.get_observation()
    opening = float(obs["robot_cartesian_pos"][7])
    holding = opening > EMPTY_GRIP_OPENING
    print(
        f"[pick] gripper opening after lift: {opening:.3f} "
        f"-> {'HOLDING' if holding else 'EMPTY'}"
    )

    finish(ws, failed, holding)


def finish(ws, failed: bool, holding: bool = False) -> None:
    """Always end with done so meta.json carries the env verdict."""
    result = ws.step("done", success=holding and not failed)
    print(f"[done] {result.receipt_text}")
    # env_success checks the *task* (pick AND place in basket); a pick-only
    # script is expected to read False there. The pick gate is `holding`.
    print(f"[verdict] pick={'ok' if holding else 'FAILED'} "
          f"env_success={ws.env_success} claimed={ws.claimed_success}")
    print(f"[trace] {ws.trace.dir}")
    ws.close()
    if failed or not holding:
        sys.exit("scripted pick did not hold the object; inspect the trace above")
    print("scripted pick completed; review the canvases before calling M1.1 done")


if __name__ == "__main__":
    main()
