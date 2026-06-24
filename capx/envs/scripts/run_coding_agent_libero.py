#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNNER_PATH = Path(__file__).with_name("run_codex_libero.py")
spec = importlib.util.spec_from_file_location("_capx_run_codex_libero", RUNNER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load runner from {RUNNER_PATH}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
main = module.main


if __name__ == "__main__":
    raise SystemExit(main())
