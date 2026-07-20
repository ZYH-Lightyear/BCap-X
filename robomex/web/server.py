"""FastAPI Trace API for RoboMEx run visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from robomex.web.live import tail_events_jsonl
from robomex.web.models import (
    AgentDetail,
    LlmView,
    RunDetail,
    RunListResponse,
    SwarmView,
    TurnDetail,
)
from robomex.web.paths import resolve_under_run, resolve_under_workspace
from robomex.web.projection import project_agent, project_run, project_swarm, project_turn, read_llm
from robomex.web.scan import list_runs

_UI_DIST = Path(__file__).resolve().parents[2] / "robomex-ui" / "dist"


def create_app(*, default_root: str = "outputs/robomex_planner_live") -> FastAPI:
    app = FastAPI(
        title="RoboMEx Trace API",
        description="Read-only projection API for RoboMEx run artifacts",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        # Trace UI is often opened via a remote machine IP (not localhost).
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/runs", response_model=RunListResponse)
    async def runs(root: str = Query(default_root)) -> RunListResponse:
        return list_runs(root)

    @app.get("/api/v1/runs/by-path", response_model=RunDetail)
    async def run_detail(dir: str = Query(..., description="Run directory")) -> RunDetail:
        try:
            return project_run(dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/by-path/swarm", response_model=SwarmView)
    async def swarm_view(
        dir: str = Query(...),
        subgoal: int = Query(0, ge=0),
    ) -> SwarmView:
        try:
            return project_swarm(dir, subgoal)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/by-path/agent", response_model=AgentDetail)
    async def agent_view(
        dir: str = Query(...),
        subgoal: int = Query(0, ge=0),
        agent: str = Query(...),
    ) -> AgentDetail:
        try:
            return project_agent(dir, subgoal, agent)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/by-path/turn", response_model=TurnDetail)
    async def turn_view(
        dir: str = Query(...),
        subgoal: int = Query(0, ge=0),
        agent: str = Query(...),
        turn: int = Query(0, ge=0),
    ) -> TurnDetail:
        try:
            return project_turn(dir, subgoal, agent, turn)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/by-path/file")
    async def file_view(
        dir: str = Query(...),
        path: str = Query(...),
    ) -> FileResponse:
        run_dir = resolve_under_workspace(dir)
        target = resolve_under_run(run_dir, path)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(target)

    @app.get("/api/v1/runs/by-path/llm", response_model=LlmView)
    async def llm_view(
        dir: str = Query(...),
        path: str = Query(...),
        view: str = Query("text", pattern="^(text|meta|full)$"),
    ) -> LlmView:
        try:
            payload = read_llm(dir, path, view=view)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return LlmView(**payload)

    @app.get("/api/v1/runs/by-path/live")
    async def live_view(
        dir: str = Query(...),
        from_start: bool = Query(False),
    ) -> StreamingResponse:
        run_dir = resolve_under_workspace(dir)
        events = run_dir / "events.jsonl"
        if not events.parent.is_dir():
            raise HTTPException(status_code=404, detail="Run directory not found")

        async def event_stream():
            async for frame in tail_events_jsonl(events, from_start=from_start):
                yield frame

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Serve the built SPA so remote browsers can open http://<server-ip>:8300/
    # without needing a separate Vite process (avoids localhost + inotify issues).
    if _UI_DIST.is_dir():
        assets_dir = _UI_DIST / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="ui-assets")

        @app.get("/")
        async def ui_index() -> FileResponse:
            return FileResponse(_UI_DIST / "index.html")

        @app.get("/{full_path:path}")
        async def ui_spa(full_path: str) -> FileResponse:
            # Keep API routes above; only fall through for UI paths.
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            candidate = (_UI_DIST / full_path).resolve()
            if _UI_DIST.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(_UI_DIST / "index.html")

    return app


@dataclass
class ServerArgs:
    # Bind all interfaces so remote browsers / reverse proxies can reach the API.
    # Dev UI still proxies /api to 127.0.0.1:8300 on the same machine.
    host: str = "0.0.0.0"
    port: int = 8300
    root: str = "outputs/robomex_planner_live"
    reload: bool = False


def main(args: ServerArgs | None = None) -> None:
    cfg = args if args is not None else tyro.cli(ServerArgs)
    app = create_app(default_root=cfg.root)
    uvicorn.run(app, host=cfg.host, port=cfg.port, reload=cfg.reload)


if __name__ == "__main__":
    main()
