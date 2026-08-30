"""FastAPI service for live and replay VAW Agent OS Observatory views."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from vaw.observatory.launcher import EpisodeLauncher, EpisodeLaunchSpec
from vaw.observatory.projection import (
    compile_imagination,
    compile_imagination_model_io,
    compile_model_io,
    compile_snapshot,
    compile_turn,
    decode_run_id,
    discover_run_dirs,
    encode_run_id,
    read_jsonl,
    summarize_run,
)


class LaunchEpisodeRequest(BaseModel):
    suite: str = Field(min_length=1, max_length=100)
    task_id: int = Field(ge=0)
    seed: int = Field(ge=0)
    model: str = Field(min_length=1, max_length=200)
    run_name: str = Field(min_length=1, max_length=100)
    imagination_model: str | None = Field(default=None, max_length=200)
    protocol: str = "native"
    motion_backend: str = "curobo"
    max_turns: int = Field(default=32, gt=0)
    max_time_s: float = Field(default=1800.0, gt=0)
    max_physical_ops: int = Field(default=30, gt=0)
    collection: str = Field(default="context_runs", min_length=1, max_length=100)


class _RunCatalog:
    """Cache recursive discovery while refreshing changed live runs."""

    def __init__(self, workspace_root: Path, *, scan_interval_s: float = 1.0) -> None:
        self.workspace_root = workspace_root
        self.scan_interval_s = float(scan_interval_s)
        self._last_scan = 0.0
        self._paths: list[Path] = []
        self._summaries: dict[Path, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def invalidate(self) -> None:
        self._last_scan = 0.0

    def paths(self) -> list[Path]:
        now = time.monotonic()
        if not self._paths or now - self._last_scan >= self.scan_interval_s:
            self._paths = discover_run_dirs(self.workspace_root)
            self._last_scan = now
        return list(self._paths)

    def summaries(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        active = set(self.paths())
        for path in self._paths:
            fingerprint = _run_fingerprint(path)
            cached = self._summaries.get(path)
            if cached is None or cached[0] != fingerprint:
                summary = summarize_run(path, workspace_root=self.workspace_root)
                self._summaries[path] = (fingerprint, summary)
            values.append(dict(self._summaries[path][1]))
        self._summaries = {
            path: value for path, value in self._summaries.items() if path in active
        }
        return values


def create_app(
    workspace: str | Path,
    *,
    ui_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    launch_collection: str = "context_runs",
    default_model: str = "vapi/qwen3.5-plus",
    default_suite: str = "libero_object_swap",
    max_concurrent: int = 1,
    launcher: EpisodeLauncher | None = None,
) -> FastAPI:
    workspace_root = Path(workspace).expanduser().resolve()
    if not workspace_root.is_dir():
        raise ValueError(f"Observatory workspace not found: {workspace_root}")
    frontend = (
        Path(ui_dir).expanduser().resolve()
        if ui_dir is not None
        else Path(__file__).resolve().parents[2] / "vaw-ui" / "dist"
    )
    repository = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    episode_launcher = launcher or EpisodeLauncher(
        workspace_root,
        repo_root=repository,
        python_executable=python_executable,
        max_concurrent=max_concurrent,
    )
    catalog = _RunCatalog(workspace_root)
    app = FastAPI(title="VAW Agent OS Observatory", version="2")

    @app.get("/")
    async def index() -> HTMLResponse:
        path = frontend / "observatory.html"
        if not path.is_file():
            raise HTTPException(status_code=503, detail="Observatory UI is not built")
        return HTMLResponse(path.read_text(encoding="utf-8"))

    @app.get("/assets/{asset_path:path}")
    async def ui_asset(asset_path: str, request: Request) -> StreamingResponse:
        return _safe_file_response(frontend / "assets", asset_path, request)

    @app.get("/api/runs")
    async def runs() -> dict[str, Any]:
        return {"runs": catalog.summaries()}

    @app.get("/api/runs/{run_id}/snapshot")
    async def snapshot(run_id: str) -> dict[str, Any]:
        return compile_snapshot(
            _resolve_run(workspace_root, run_id),
            workspace_root=workspace_root,
        )

    @app.get("/api/runs/{run_id}/turns/{turn}")
    async def turn(run_id: str, turn: int) -> dict[str, Any]:
        value = compile_turn(
            _resolve_run(workspace_root, run_id),
            turn,
            workspace_root=workspace_root,
        )
        if value is None:
            raise HTTPException(status_code=404, detail="turn not found")
        return value

    @app.get("/api/runs/{run_id}/turns/{turn}/model-io")
    async def model_io(run_id: str, turn: int) -> dict[str, Any]:
        run_dir = _resolve_run(workspace_root, run_id)
        value = compile_model_io(run_dir, turn)
        if not value["attempts"]:
            raise HTTPException(status_code=404, detail="model I/O not found")
        return value

    @app.get("/api/runs/{run_id}/imagination/{session_id}")
    async def imagination(run_id: str, session_id: str) -> dict[str, Any]:
        try:
            return compile_imagination(
                _resolve_run(workspace_root, run_id),
                session_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get(
        "/api/runs/{run_id}/imagination/{session_id}/turns/{turn}/model-io"
    )
    async def imagination_model_io(
        run_id: str,
        session_id: str,
        turn: int,
    ) -> dict[str, Any]:
        try:
            value = compile_imagination_model_io(
                _resolve_run(workspace_root, run_id),
                session_id,
                turn,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not value["attempts"]:
            raise HTTPException(status_code=404, detail="model I/O not found")
        return value

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    async def artifact(
        run_id: str,
        artifact_path: str,
        request: Request,
    ) -> StreamingResponse:
        run_dir = _resolve_run(workspace_root, run_id)
        return _safe_file_response(run_dir, artifact_path, request)

    @app.get("/api/runs/{run_id}/stream")
    async def stream(run_id: str, request: Request, after_seq: int = 0) -> StreamingResponse:
        run_dir = _resolve_run(workspace_root, run_id)
        return StreamingResponse(
            _event_stream(run_dir, request, after_seq=max(0, int(after_seq))),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/control/config")
    async def control_config() -> dict[str, Any]:
        summaries = catalog.summaries()
        collections = sorted(
            {str(item.get("collection")) for item in summaries if item.get("collection")}
            | {launch_collection}
        )
        models = _unique_strings(item.get("model") for item in summaries)
        suites = _unique_strings(item.get("suite") for item in summaries)
        return {
            "control_enabled": True,
            "workspace": str(workspace_root),
            "default_collection": launch_collection,
            "default_model": default_model,
            "default_suite": default_suite,
            "collections": collections,
            "model_suggestions": models,
            "suite_suggestions": suites,
            "max_concurrent": episode_launcher.max_concurrent,
        }

    @app.get("/api/control/launches")
    async def launches() -> dict[str, Any]:
        return {"launches": episode_launcher.jobs()}

    @app.post("/api/control/launches", status_code=201)
    async def launch_episode(payload: LaunchEpisodeRequest) -> dict[str, Any]:
        try:
            manifest = await episode_launcher.launch(
                EpisodeLaunchSpec(**payload.model_dump())
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        run_dir = workspace_root / str(manifest["run_dir"])
        catalog.invalidate()
        return {
            "launch": manifest,
            "run_id": encode_run_id(workspace_root, run_dir),
        }

    @app.post("/api/control/launches/{job_id}/stop")
    async def stop_episode(job_id: str) -> dict[str, Any]:
        try:
            manifest = await episode_launcher.stop(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="launch not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"launch": manifest}

    return app


def _resolve_run(root: Path, run_id: str) -> Path:
    try:
        candidate = decode_run_id(root, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    if not _looks_like_run(candidate):
        raise HTTPException(status_code=404, detail="run not found")
    return candidate


def _unique_strings(values: Any) -> list[str]:
    return sorted({str(value) for value in values if value})


def _looks_like_run(path: Path) -> bool:
    return path.is_dir() and any(
        (path / name).exists()
        for name in (
            "launcher.json",
            "meta.json",
            "steps.jsonl",
            "runtime_events.jsonl",
            "contexts",
        )
    )


def _run_fingerprint(path: Path) -> tuple[Any, ...]:
    values: list[Any] = []
    for name in ("launcher.json", "meta.json", "steps.jsonl", "runtime_events.jsonl"):
        target = path / name
        try:
            stat = target.stat()
        except OSError:
            values.extend((None, None))
        else:
            values.extend((stat.st_mtime_ns, stat.st_size))
    return tuple(values)


async def _event_stream(
    run_dir: Path,
    request: Request,
    after_seq: int,
) -> AsyncIterator[str]:
    cursor = after_seq
    idle_ticks = 0
    while not await request.is_disconnected():
        events = [
            event
            for event in read_jsonl(run_dir / "runtime_events.jsonl")
            if int(event.get("event_seq", 0) or 0) > cursor
        ]
        if events:
            idle_ticks = 0
            for event in events:
                cursor = int(event.get("event_seq", cursor) or cursor)
                yield (
                    f"id: {cursor}\n"
                    f"event: {event.get('event_type', 'message')}\n"
                    f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                )
        else:
            idle_ticks += 1
            if idle_ticks % 10 == 0:
                yield ": keep-alive\n\n"
        await asyncio.sleep(0.25)


def _safe_file_response(root: Path, relative: str, request: Request) -> StreamingResponse:
    base = root.resolve()
    target = (base / relative).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    size = target.stat().st_size
    start, end = _byte_range(request.headers.get("range"), size)
    length = max(0, end - start + 1)
    partial = start != 0 or end != max(0, size - 1)
    headers = {
        "Content-Length": str(length),
        "Cache-Control": "no-store",
        "Accept-Ranges": "bytes",
    }
    if partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(
        _file_chunks(target, start=start, length=length),
        status_code=206 if partial else 200,
        media_type=media_type,
        headers=headers,
    )


async def _file_chunks(
    path: Path,
    *,
    start: int,
    length: int,
    chunk_size: int = 1024 * 1024,
) -> AsyncIterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            yield chunk
            remaining -= len(chunk)
            await asyncio.sleep(0)


def _byte_range(header: str | None, size: int) -> tuple[int, int]:
    if size <= 0:
        return 0, -1
    if not header or not header.startswith("bytes="):
        return 0, size - 1
    value = header[6:].split(",", 1)[0].strip()
    try:
        first, last = value.split("-", 1)
        if not first:
            suffix = max(0, int(last))
            return max(0, size - suffix), size - 1
        start = max(0, int(first))
        end = size - 1 if not last else min(size - 1, int(last))
        if start > end or start >= size:
            raise ValueError
        return start, end
    except (TypeError, ValueError):
        raise HTTPException(status_code=416, detail="invalid byte range") from None


__all__ = ["create_app"]
