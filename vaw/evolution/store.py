"""不可变技能 generation、活动指针和 append-only ledger。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vaw.evolution.domain import EvolutionSpec, GenerationManifest, SkillMutation


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 文档必须是 object: {path}")
    return payload


def _atomic_text(path: Path, content: str) -> None:
    """用同目录临时文件替换目标，避免活动指针出现半写状态。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _skill_files(root: Path) -> Iterator[Path]:
    if root.is_symlink():
        raise ValueError(f"技能根目录不能是符号链接: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"技能目录不存在: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())
    for path in files:
        if path.is_symlink():
            raise ValueError(f"技能快照不接受符号链接: {path}")
        yield path


def skill_tree_digest(root: str | Path) -> str:
    """对相对路径和文件内容计算确定性的技能树摘要。"""

    directory = Path(root)
    digest = hashlib.sha256()
    count = 0
    for path in _skill_files(directory):
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        count += 1
    if count == 0:
        raise ValueError(f"技能目录为空: {directory}")
    return digest.hexdigest()


def _generation_name(value: str) -> str:
    """generation ID 会进入路径，因此只接受单段目录名。"""

    normalized = str(value).strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
        or Path(normalized).name != normalized
    ):
        raise ValueError("generation_id 必须是单段目录名")
    return normalized


class EvolutionLedger:
    """只提供追加与读取能力的 JSONL 事件日志。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(
        self,
        event: str,
        *,
        generation_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        if not event.strip():
            raise ValueError("ledger event 不能为空")
        record = {
            "event_id": uuid.uuid4().hex,
            "timestamp": timestamp or _utc_now(),
            "event": event.strip(),
            "generation_id": generation_id,
            "payload": dict(payload or {}),
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return record

    def read(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            lines = handle.read().splitlines()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        for line_number, raw_line in enumerate(lines, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError(f"ledger 第 {line_number} 行不是 JSON object")
            records.append(record)
        return tuple(records)


class GenerationStore:
    """管理一个实验的冻结配置、generation 快照和活动指针。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.generations_dir = self.root / "generations"
        self.ledger = EvolutionLedger(self.root / "ledger.jsonl")

    @classmethod
    def initialize(
        cls,
        root: str | Path,
        spec: EvolutionSpec,
        skill_root: str | Path,
        *,
        created_at: str | None = None,
    ) -> GenerationStore:
        """在临时目录中构建完整实验，再一次性发布到目标路径。"""

        destination = Path(root)
        if destination.exists():
            raise FileExistsError(f"实验目录已存在: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        store = cls(staging)
        try:
            store.generations_dir.mkdir(parents=True)
            _write_json(store.root / "experiment.json", spec.experiment_document())
            _write_json(store.root / "split.json", spec.split_document())
            store.ledger.path.touch(exist_ok=False)
            manifest = store._materialize_generation(
                generation_id=spec.base_generation,
                parent_generation=None,
                skill_root=Path(skill_root),
                mutation=None,
                created_at=created_at or _utc_now(),
            )
            _atomic_text(store.root / "active_generation", f"{manifest.generation_id}\n")
            store.ledger.append(
                "experiment_initialized",
                generation_id=manifest.generation_id,
                payload={"experiment_id": spec.experiment_id},
            )
            store.ledger.append("generation_activated", generation_id=manifest.generation_id)
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return cls(destination)

    @classmethod
    def open(cls, root: str | Path) -> GenerationStore:
        store = cls(root)
        for required in (
            store.root / "experiment.json",
            store.root / "split.json",
            store.root / "active_generation",
            store.ledger.path,
            store.generations_dir,
        ):
            if not required.exists():
                raise FileNotFoundError(f"实验存储缺少必要路径: {required}")
        store.read_manifest(store.active_generation())
        return store

    def read_spec(self) -> EvolutionSpec:
        return EvolutionSpec.from_documents(
            _read_json(self.root / "experiment.json"),
            _read_json(self.root / "split.json"),
        )

    def generation_path(self, generation_id: str) -> Path:
        return self.generations_dir / _generation_name(generation_id)

    def active_generation(self) -> str:
        generation_id = (self.root / "active_generation").read_text(encoding="utf-8").strip()
        if not generation_id or not self.generation_path(generation_id).is_dir():
            raise ValueError(f"活动 generation 无效: {generation_id!r}")
        return generation_id

    def read_manifest(self, generation_id: str) -> GenerationManifest:
        manifest_path = self.generation_path(generation_id) / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"generation manifest 不存在: {manifest_path}")
        manifest = GenerationManifest.from_dict(_read_json(manifest_path))
        actual_digest = skill_tree_digest(self.generation_path(generation_id) / "skills")
        if actual_digest != manifest.skill_digest:
            raise ValueError(f"generation {generation_id} 的技能快照摘要不匹配")
        return manifest

    def create_generation(
        self,
        generation_id: str,
        *,
        parent_generation: str,
        skill_root: str | Path,
        mutation: SkillMutation,
        created_at: str | None = None,
    ) -> GenerationManifest:
        """创建候选快照但不改变活动 generation。"""

        self.read_manifest(parent_generation)
        manifest = self._materialize_generation(
            generation_id=generation_id,
            parent_generation=parent_generation,
            skill_root=Path(skill_root),
            mutation=mutation,
            created_at=created_at or _utc_now(),
        )
        self.ledger.append(
            "generation_created",
            generation_id=generation_id,
            payload={
                "parent_generation": parent_generation,
                "mutation_id": mutation.mutation_id,
            },
        )
        return manifest

    def rollback(self, generation_id: str) -> None:
        """只切换活动指针；历史 generation 和 manifest 均保持不变。"""

        self.read_manifest(generation_id)
        previous = self.active_generation()
        _atomic_text(self.root / "active_generation", f"{generation_id}\n")
        self.ledger.append(
            "generation_rollback",
            generation_id=generation_id,
            payload={"previous_generation": previous},
        )

    def _materialize_generation(
        self,
        *,
        generation_id: str,
        parent_generation: str | None,
        skill_root: Path,
        mutation: SkillMutation | None,
        created_at: str,
    ) -> GenerationManifest:
        destination = self.generation_path(generation_id)
        if destination.exists():
            raise FileExistsError(f"generation 已存在: {generation_id}")
        digest = skill_tree_digest(skill_root)
        manifest = GenerationManifest(
            generation_id=generation_id,
            parent_generation=parent_generation,
            created_at=created_at,
            skill_digest=digest,
            mutation=mutation,
        )
        staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=self.generations_dir))
        try:
            shutil.copytree(skill_root, staging / "skills")
            if skill_tree_digest(staging / "skills") != digest:
                raise RuntimeError("复制后的技能快照摘要发生变化")
            _write_json(staging / "manifest.json", manifest.to_dict())
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return manifest


__all__ = ["EvolutionLedger", "GenerationStore", "skill_tree_digest"]
