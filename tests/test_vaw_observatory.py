from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np

from tests.test_vaw_context_agent import RecordingProvider, RecordingRenderer, _response
from tests.test_vaw_context_runtime import FakeContextApi
from vaw.context_runtime.runtime import ContextRuntime
from vaw.context_runtime.trace import ContextTraceLogger
from vaw.context_runtime.workspace import ContextWorkspace
from vaw.observatory.launcher import EpisodeLauncher, EpisodeLaunchSpec
from vaw.observatory.media import ActionMediaRecorder
from vaw.observatory.projection import (
    compile_imagination,
    compile_imagination_model_io,
    compile_snapshot,
    encode_run_id,
)
from vaw.observatory.server import _event_stream, create_app


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_runtime_freezes_exact_pre_turn_context(tmp_path: Path) -> None:
    trace = ContextTraceLogger(tmp_path)
    ContextRuntime(
        RecordingProvider(
            [
                _response(1, "detect_region", query="can"),
                _response(2, "finish_task", success=False),
            ]
        ),
        ContextWorkspace(FakeContextApi(), "pick the can", motion_backend="pyroki"),
        RecordingRenderer(),
        trace=trace,
    ).run()

    turn_one = json.loads((tmp_path / "contexts/turn_0001/context.json").read_text())
    turn_two = json.loads((tmp_path / "contexts/turn_0002/context.json").read_text())
    assert turn_one["interaction_memory_before"] == []
    assert turn_one["interaction_memory_prompt"] == []
    assert turn_two["interaction_memory_before"][0]["function"] == "detect_region"
    assert turn_two["interaction_memory_prompt"] == [
        't1 [函数调用] detect_region({"query":"can"}) -> 结果=ok (r1->r1)'
    ]
    assert "raw_response_text" not in turn_two

    steps = _rows(tmp_path / "steps.jsonl")
    assert steps[0]["interaction_memory_before"] == []
    assert steps[0]["interaction_event"]["function"] == "detect_region"
    assert steps[0]["interaction_memory_after"][0]["function"] == "detect_region"
    assert steps[1]["interaction_memory_before"][0]["function"] == "detect_region"

    events = _rows(tmp_path / "runtime_events.jsonl")
    sequence = [event["event_seq"] for event in events]
    assert sequence == list(range(1, len(sequence) + 1))
    assert events[0]["event_type"] == "turn_context_ready"
    request = json.loads(
        (tmp_path / "contexts/turn_0001/model_io/attempt_01/request.json").read_text()
    )
    response = json.loads(
        (tmp_path / "contexts/turn_0001/model_io/attempt_01/response.json").read_text()
    )
    assert request["messages"][0]["role"] == "system"
    assert "[内联 image/png 已省略" in json.dumps(request, ensure_ascii=False)
    assert response["tool_calls"][0]["name"] == "detect_region"


class _VideoEnv:
    def __init__(self) -> None:
        self.agentview = [np.zeros((24, 32, 3), dtype=np.uint8)]
        self.wrist = [np.zeros((24, 32, 3), dtype=np.uint8)]

    def get_video_frame_count(self) -> int:
        return len(self.agentview)

    def get_video_frames_range(self, start: int, end: int):
        return self.agentview[start:end]

    def get_wrist_video_frames_range(self, start: int, end: int):
        return self.wrist[start:end]


def test_action_media_recorder_cuts_one_physical_segment(tmp_path: Path) -> None:
    trace = ContextTraceLogger(tmp_path)
    env = _VideoEnv()
    recorder = ActionMediaRecorder(env, trace, fps=10)
    token = recorder.start(
        turn=3,
        function="close_gripper",
        arguments={},
        revision_before=2,
    )
    for value in (40, 90, 140):
        env.agentview.append(np.full((24, 32, 3), value, dtype=np.uint8))
        env.wrist.append(np.full((24, 32, 3), 255 - value, dtype=np.uint8))
    summary = recorder.finish(token, outcome="completed", revision_after=3)

    assert summary["status"] == "ready"
    manifest = json.loads((tmp_path / summary["manifest"]).read_text())
    assert manifest["turn"] == 3
    assert manifest["frame_start"] == 1
    assert manifest["frame_end"] == 4
    assert manifest["streams"]["agentview"]["frames"] == 3
    assert (tmp_path / manifest["streams"]["agentview"]["path"]).is_file()
    assert (tmp_path / manifest["streams"]["wrist"]["path"]).is_file()


def test_projection_and_api_render_only_frozen_context(tmp_path: Path) -> None:
    run_dir = tmp_path / "collection-a" / "run-001"
    trace = ContextTraceLogger(run_dir)
    ContextRuntime(
        RecordingProvider([_response(1, "finish_task", success=False)]),
        ContextWorkspace(FakeContextApi(), "task", motion_backend="pyroki"),
        RecordingRenderer(),
        trace=trace,
    ).run()

    snapshot = compile_snapshot(run_dir, workspace_root=tmp_path)
    assert snapshot["turns"][0]["context"]["interaction_memory_before"] == []
    assert "prompt_text" not in snapshot["turns"][0]["context"]
    assert "raw_response_text" not in snapshot["turns"][0]["decision"]
    assert snapshot["turns"][0]["model_io_available"] is True

    (run_dir / "video_agentview.mp4").write_bytes(b"test-video")
    snapshot = compile_snapshot(run_dir, workspace_root=tmp_path)
    assert snapshot["episode_videos"]["agentview"]["path"] == (
        "artifacts/video_agentview.mp4"
    )

    async def inspect_api() -> None:
        app = create_app(tmp_path, ui_dir=tmp_path / "missing-ui")
        run_id = encode_run_id(tmp_path, run_dir)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            runs = await client.get("/api/runs")
            assert runs.status_code == 200
            assert runs.json()["runs"][0]["run_id"] == run_id
            assert runs.json()["runs"][0]["collection"] == "collection-a"
            response = await client.get(f"/api/runs/{run_id}/snapshot")
            assert response.status_code == 200
            canvas = response.json()["turns"][0]["canvas"]
            artifact = await client.get(f"/api/runs/{run_id}/{canvas}")
            assert artifact.status_code == 200
            assert artifact.headers["content-type"] == "image/png"
            partial = await client.get(
                f"/api/runs/{run_id}/{canvas}",
                headers={"Range": "bytes=0-15"},
            )
            assert partial.status_code == 206
            assert partial.headers["accept-ranges"] == "bytes"
            assert len(partial.content) == 16
            traversal = await client.get(f"/api/runs/{run_id}/artifacts/../meta.json")
            assert traversal.status_code == 404
            model_io = await client.get(f"/api/runs/{run_id}/turns/1/model-io")
            assert model_io.status_code == 200
            assert model_io.json()["attempts"][0]["response"]["tool_calls"][0]["name"] == "finish_task"

        class ConnectedRequest:
            async def is_disconnected(self) -> bool:
                return False

        stream = _event_stream(run_dir, ConnectedRequest(), after_seq=0)  # type: ignore[arg-type]
        first = await anext(stream)
        assert first.startswith("id: 1\nevent: turn_context_ready\n")
        await stream.aclose()

    asyncio.run(inspect_api())


def test_imagination_trace_is_projected_per_internal_turn(tmp_path: Path) -> None:
    run_dir = tmp_path / "collection" / "run"
    nested = run_dir / "subagents" / "imagination_0001"
    model_io = nested / "contexts" / "turn_0001" / "model_io" / "attempt_01"
    model_io.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({"task_prompt": "pick"}))
    (run_dir / "steps.jsonl").write_text(
        json.dumps(
            {
                "turn": 3,
                "function_call": {
                    "name": "imagine_action",
                    "arguments": {"instruction": "align the jaws"},
                },
                "function_result": {"status": "ready", "action_id": "a1"},
                "runtime_diagnostics": {
                    "subagent": {
                        "trace": "subagents/imagination_0001",
                        "status": "ready",
                    }
                },
            }
        )
        + "\n"
    )
    (nested / "meta.json").write_text(
        json.dumps(
            {
                "agent": "imagination",
                "instruction": "align the jaws",
                "status": "ready",
                "action_id": "a1",
            }
        )
    )
    canvas = np.full((16, 24, 3), 80, dtype=np.uint8)
    from PIL import Image

    Image.fromarray(canvas).save(nested / "context_0000.png")
    (nested / "steps.jsonl").write_text(
        json.dumps(
            {
                "turn": 1,
                "context_image": "context_0000.png",
                "decision_basis": "the grasp line misses the object",
                "function_call": {
                    "name": "move_tcp_delta",
                    "arguments": {"dy": 0.01},
                },
                "function_result": {"status": "edited"},
            }
        )
        + "\n"
    )
    (model_io / "request.json").write_text(
        json.dumps({"attempt": 1, "messages": [{"role": "user", "content": "canvas"}]})
    )
    (model_io / "response.json").write_text(
        json.dumps({"attempt": 1, "raw_response_text": "move left"})
    )

    snapshot = compile_snapshot(run_dir, workspace_root=tmp_path)
    summary = snapshot["turns"][0]["imagination"]
    assert summary["session_id"] == "imagination_0001"
    assert summary["turn_count"] == 1
    detail = compile_imagination(run_dir, "imagination_0001")
    assert detail["turns"][0]["function_call"]["name"] == "move_tcp_delta"
    assert detail["turns"][0]["canvas"].endswith("context_0000.png")
    raw = compile_imagination_model_io(run_dir, "imagination_0001", 1)
    assert raw["attempts"][0]["response"]["raw_response_text"] == "move left"

    async def inspect_api() -> None:
        app = create_app(tmp_path, ui_dir=tmp_path / "missing-ui")
        run_id = encode_run_id(tmp_path, run_dir)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                f"/api/runs/{run_id}/imagination/imagination_0001"
            )
            assert response.status_code == 200
            raw_response = await client.get(
                f"/api/runs/{run_id}/imagination/imagination_0001/turns/1/model-io"
            )
            assert raw_response.status_code == 200
            traversal = await client.get(
                f"/api/runs/{run_id}/imagination/../meta.json"
            )
            assert traversal.status_code == 404

    asyncio.run(inspect_api())


class _FakeLauncher:
    max_concurrent = 1

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.spec: EpisodeLaunchSpec | None = None
        self.stopped: str | None = None

    async def launch(self, spec: EpisodeLaunchSpec) -> dict:
        self.spec = spec
        run_dir = self.workspace / spec.collection / "manual-run"
        run_dir.mkdir(parents=True)
        manifest = {
            "job_id": "job-test",
            "status": "running",
            "run_dir": str(run_dir.relative_to(self.workspace)),
            "spec": {
                "suite": spec.suite,
                "task_id": spec.task_id,
                "seed": spec.seed,
                "model": spec.model,
                "run_name": spec.run_name,
            },
        }
        (run_dir / "launcher.json").write_text(json.dumps(manifest))
        return manifest

    async def stop(self, job_id: str) -> dict:
        if job_id != "job-test":
            raise KeyError(job_id)
        self.stopped = job_id
        return {"job_id": job_id, "status": "stopping"}

    def jobs(self) -> list[dict]:
        return []


def test_control_api_launches_typed_episode(tmp_path: Path) -> None:
    fake = _FakeLauncher(tmp_path)

    async def exercise() -> None:
        app = create_app(
            tmp_path,
            launcher=fake,  # type: ignore[arg-type]
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "run_name": "manual-task-4",
                "suite": "libero_object_swap",
                "task_id": 4,
                "seed": 1,
                "model": "vapi/qwen3.5-plus",
                "collection": "manual_runs",
            }
            launched = await client.post(
                "/api/control/launches",
                json=payload,
            )
            assert launched.status_code == 201
            assert launched.json()["run_id"].startswith("run-")
            assert fake.spec is not None
            assert fake.spec.task_id == 4
            runs = (await client.get("/api/runs")).json()["runs"]
            assert runs[0]["collection"] == "manual_runs"
            assert runs[0]["display_name"] == "manual-task-4"
            assert runs[0]["launcher_status"] == "running"
            stopped = await client.post(
                "/api/control/launches/job-test/stop",
            )
            assert stopped.status_code == 200
            assert fake.stopped == "job-test"

    asyncio.run(exercise())


def test_launcher_builds_argument_vector_without_shell_or_browser_paths(
    tmp_path: Path,
) -> None:
    launcher = EpisodeLauncher(tmp_path / "out", repo_root=tmp_path)
    spec = EpisodeLaunchSpec(
        run_name="object-task-2",
        suite="libero_object_swap",
        task_id=2,
        seed=3,
        model="vapi/qwen3.5-plus",
        imagination_model="local/qwen3.5-27b",
    )
    command = launcher.command_for(spec, tmp_path / "out/context_runs/run")
    assert command[:3] == [launcher.python_executable, "-m", "vaw.scripts.run_context_agent"]
    assert command[command.index("--task-id") + 1] == "2"
    assert command[command.index("--trace-dir") + 1].endswith("context_runs/run")
    assert "--imagination-model" in command
    assert not any(value in command for value in ("sh", "bash", "-c"))


def test_launcher_accepts_a_human_readable_chinese_run_name(tmp_path: Path) -> None:
    launcher = EpisodeLauncher(tmp_path / "out", repo_root=tmp_path)
    spec = EpisodeLaunchSpec(
        run_name="双罐放置-实验1",
        suite="libero_object_swap",
        task_id=2,
        seed=3,
        model="vapi/qwen3.5-plus",
    )
    spec.validate()
    assert launcher.command_for(spec, tmp_path / "run")[-2:] == [
        "--trace-dir",
        str(tmp_path / "run"),
    ]
