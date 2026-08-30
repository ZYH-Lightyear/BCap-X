"""Serve the read-only VAW Agent OS Observatory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from vaw.observatory.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "out",
        help="workspace recursively containing trace collections",
    )
    parser.add_argument("--launch-collection", default="context_runs")
    parser.add_argument(
        "--default-model",
        default=os.environ.get("VAW_DEFAULT_MODEL", "vapi/qwen3.5-plus"),
    )
    parser.add_argument(
        "--default-suite",
        default=os.environ.get("VAW_DEFAULT_SUITE", "libero_object_swap"),
    )
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    print(f"VAW Agent OS Observatory: http://{args.host}:{args.port}")
    print(f"Workspace: {workspace}")
    uvicorn.run(
        create_app(
            workspace,
            repo_root=Path(__file__).resolve().parents[2],
            launch_collection=args.launch_collection,
            default_model=args.default_model,
            default_suite=args.default_suite,
            max_concurrent=args.max_concurrent,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
