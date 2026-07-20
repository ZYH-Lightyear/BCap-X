"""Path safety helpers for the Trace API."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException


def workspace_root() -> Path:
    return Path.cwd().resolve()


def resolve_under_workspace(path: str | Path) -> Path:
    """Resolve a user path and require it to stay inside the workspace cwd."""

    target = Path(path).expanduser().resolve()
    root = workspace_root()
    if target != root and root not in target.parents:
        raise HTTPException(status_code=403, detail="Path must be inside the workspace")
    return target


def resolve_under_run(run_dir: Path, relative: str) -> Path:
    """Resolve a relative artifact path under a run directory."""

    target = (run_dir / relative).resolve()
    if target != run_dir and run_dir not in target.parents:
        raise HTTPException(status_code=403, detail="Invalid artifact path")
    return target


def file_url(run_dir: Path, relative: str) -> str:
    from urllib.parse import quote

    return (
        f"/api/v1/runs/by-path/file"
        f"?dir={quote(str(run_dir))}&path={quote(relative)}"
    )
