"""Controlled episode launcher for the VAW Agent OS Observatory."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_RUN_NAME = re.compile(r"^[^/\\\x00-\x1f\x7f]+$")


@dataclass(frozen=True)
class EpisodeLaunchSpec:
    """Typed user choices for one complete VAW episode."""

    suite: str
    task_id: int
    seed: int
    model: str
    run_name: str
    imagination_model: str | None = None
    protocol: str = "native"
    motion_backend: str = "curobo"
    max_turns: int = 32
    max_time_s: float = 1800.0
    max_physical_ops: int = 30
    collection: str = "context_runs"

    def validate(self) -> None:
        if not _SAFE_NAME.fullmatch(self.suite):
            raise ValueError("suite must contain only letters, numbers, '.', '_' or '-'")
        if not _SAFE_NAME.fullmatch(self.collection):
            raise ValueError(
                "collection must contain only letters, numbers, '.', '_' or '-'"
            )
        shown_run_name = self.run_name.strip()
        if (
            not shown_run_name
            or shown_run_name in {".", ".."}
            or not _SAFE_RUN_NAME.fullmatch(shown_run_name)
        ):
            raise ValueError("run_name must be a non-empty path-safe display name")
        if self.task_id < 0 or self.seed < 0:
            raise ValueError("task_id and seed must be non-negative")
        if not self.model.strip() or len(self.model) > 200:
            raise ValueError("model must be a non-empty model identifier")
        if self.imagination_model is not None and len(self.imagination_model) > 200:
            raise ValueError("imagination_model is too long")
        if self.protocol not in {"native", "text"}:
            raise ValueError("protocol must be native or text")
        if self.motion_backend not in {"pyroki", "curobo"}:
            raise ValueError("motion_backend must be pyroki or curobo")
        if self.max_turns <= 0 or self.max_time_s <= 0 or self.max_physical_ops <= 0:
            raise ValueError("episode budgets must be positive")


@dataclass
class _RunningJob:
    process: asyncio.subprocess.Process
    log_handle: Any
    run_dir: Path
    manifest: dict[str, Any]


class EpisodeLauncher:
    """Launch only the repository's typed VAW episode entry point.

    Browser input is never interpreted as a shell command or filesystem path.
    Every run is confined to a named collection below ``workspace_root``.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        repo_root: str | Path,
        python_executable: str | Path | None = None,
        max_concurrent: int = 1,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.max_concurrent = max(1, int(max_concurrent))
        self._jobs: dict[str, _RunningJob] = {}
        self._lock = asyncio.Lock()

    async def launch(self, spec: EpisodeLaunchSpec) -> dict[str, Any]:
        spec.validate()
        async with self._lock:
            running_ids = {
                job_id
                for job_id, job in self._jobs.items()
                if job.process.returncode is None
            }
            running_ids.update(self._detached_running_ids())
            if len(running_ids) >= self.max_concurrent:
                raise RuntimeError(
                    f"concurrent episode limit reached ({self.max_concurrent})"
                )

            collection = (self.workspace_root / spec.collection).resolve()
            collection.relative_to(self.workspace_root)
            collection.mkdir(parents=True, exist_ok=True)
            run_dir = _allocate_run_dir(collection, spec)
            run_dir.mkdir(parents=False, exist_ok=False)
            job_id = f"job-{uuid.uuid4().hex[:12]}"
            command = self.command_for(spec, run_dir)
            manifest: dict[str, Any] = {
                "schema": "vaw-observatory-launch-v1",
                "job_id": job_id,
                "status": "starting",
                "created_time_unix_s": time.time(),
                "run_dir": str(run_dir.relative_to(self.workspace_root)),
                "spec": asdict(spec),
                "command": command,
            }
            _write_manifest(run_dir, manifest)
            log_handle = (run_dir / "launcher.log").open("ab", buffering=0)
            environment = dict(os.environ)
            environment.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=self.repo_root,
                    env=environment,
                    stdout=log_handle,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception:
                log_handle.close()
                manifest.update(
                    {
                        "status": "launch_error",
                        "finished_time_unix_s": time.time(),
                    }
                )
                _write_manifest(run_dir, manifest)
                raise

            manifest.update(
                {
                    "status": "running",
                    "pid": process.pid,
                    "process_start_ticks": _proc_start_ticks(process.pid),
                    "started_time_unix_s": time.time(),
                }
            )
            _write_manifest(run_dir, manifest)
            job = _RunningJob(process, log_handle, run_dir, manifest)
            self._jobs[job_id] = job
            asyncio.create_task(self._watch(job_id, job))
            return dict(manifest)

    async def stop(self, job_id: str) -> dict[str, Any]:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.process.returncode is None:
                os.killpg(job.process.pid, signal.SIGTERM)
                job.manifest["status"] = "stopping"
                job.manifest["stop_requested_time_unix_s"] = time.time()
                _write_manifest(job.run_dir, job.manifest)
                return dict(job.manifest)
            detached = self._find_manifest(job_id)
            if detached is None:
                raise KeyError(job_id)
            run_dir, manifest = detached
            pid = manifest.get("pid")
            start_ticks = manifest.get("process_start_ticks")
            if manifest.get("status") in {"starting", "running", "stopping"}:
                if (
                    isinstance(pid, int)
                    and isinstance(start_ticks, int)
                    and _proc_start_ticks(pid) == start_ticks
                ):
                    os.killpg(pid, signal.SIGTERM)
                elif manifest.get("status") != "starting":
                    raise RuntimeError("launch process identity is no longer valid")
                manifest["status"] = "stopping"
                manifest["stop_requested_time_unix_s"] = time.time()
                _write_manifest(run_dir, manifest)
            return dict(manifest)

    def jobs(self) -> list[dict[str, Any]]:
        values_by_id: dict[str, dict[str, Any]] = {}
        for path in self.workspace_root.rglob("launcher.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value.get("job_id"):
                values_by_id[str(value["job_id"])] = value
        for job in self._jobs.values():
            values_by_id[str(job.manifest["job_id"])] = dict(job.manifest)
        values = list(values_by_id.values())
        values.sort(key=lambda value: value.get("created_time_unix_s", 0), reverse=True)
        return values

    def command_for(self, spec: EpisodeLaunchSpec, run_dir: Path) -> list[str]:
        command = [
            self.python_executable,
            "-m",
            "vaw.scripts.run_context_agent",
            "--mode",
            "agent",
            "--suite",
            spec.suite,
            "--task-id",
            str(spec.task_id),
            "--seed",
            str(spec.seed),
            "--model",
            spec.model,
            "--protocol",
            spec.protocol,
            "--motion-backend",
            spec.motion_backend,
            "--max-turns",
            str(spec.max_turns),
            "--max-time-s",
            str(spec.max_time_s),
            "--max-physical-ops",
            str(spec.max_physical_ops),
            "--record-video",
            "--trace-dir",
            str(run_dir),
        ]
        if spec.imagination_model:
            command.extend(["--imagination-model", spec.imagination_model])
        return command

    async def _watch(self, job_id: str, job: _RunningJob) -> None:
        return_code = await job.process.wait()
        job.log_handle.close()
        async with self._lock:
            requested_stop = job.manifest.get("status") == "stopping"
            job.manifest.update(
                {
                    "status": (
                        "stopped"
                        if requested_stop
                        else "completed"
                        if return_code == 0
                        else "failed"
                    ),
                    "return_code": return_code,
                    "finished_time_unix_s": time.time(),
                }
            )
            _write_manifest(job.run_dir, job.manifest)
            self._jobs[job_id] = job

    def _find_manifest(self, job_id: str) -> tuple[Path, dict[str, Any]] | None:
        for path in self.workspace_root.rglob("launcher.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value.get("job_id") == job_id:
                return path.parent, value
        return None

    def _detached_running_ids(self) -> set[str]:
        running: set[str] = set()
        for path in self.workspace_root.rglob("launcher.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            pid = value.get("pid") if isinstance(value, dict) else None
            start_ticks = (
                value.get("process_start_ticks") if isinstance(value, dict) else None
            )
            if (
                isinstance(value, dict)
                and value.get("status") in {"starting", "running", "stopping"}
                and value.get("job_id")
                and isinstance(pid, int)
                and isinstance(start_ticks, int)
                and _proc_start_ticks(pid) == start_ticks
            ):
                running.add(str(value["job_id"]))
        return running


def _allocate_run_dir(collection: Path, spec: EpisodeLaunchSpec) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.model).strip("_") or "model"
    run_name = re.sub(r"[^\w.-]+", "_", spec.run_name.strip()).strip("._-")
    run_name = run_name[:100] or "episode"
    base = (
        f"{timestamp}_{run_name}_{model}_{spec.motion_backend}_{spec.suite}"
        f"_t{spec.task_id}_s{spec.seed}"
    )
    candidate = collection / base
    suffix = 1
    while candidate.exists():
        candidate = collection / f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _write_manifest(run_dir: Path, value: dict[str, Any]) -> None:
    path = run_dir / "launcher.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _proc_start_ticks(pid: int) -> int | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields_after_name = value[value.rfind(")") + 2 :].split()
        return int(fields_after_name[19])
    except (OSError, ValueError, IndexError):
        return None


__all__ = ["EpisodeLaunchSpec", "EpisodeLauncher"]
