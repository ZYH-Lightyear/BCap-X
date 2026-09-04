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

from vaw.evolution.candidate import CandidatePackage
from vaw.evolution.domain import (
    EvolutionSpec,
    GateDecision,
    GateReport,
    GenerationManifest,
    MutationOperation,
    SkillMutation,
)


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
        candidate: CandidatePackage | str | Path,
        created_at: str | None = None,
    ) -> GenerationManifest:
        """把单项候选变更应用到父快照，但不改变活动 generation。"""

        self.read_manifest(parent_generation)
        package = CandidatePackage.open(candidate.root if isinstance(candidate, CandidatePackage) else candidate)
        manifest = self._materialize_candidate(
            generation_id=generation_id,
            parent_generation=parent_generation,
            candidate=package,
            created_at=created_at or _utc_now(),
        )
        self.ledger.append(
            "generation_created",
            generation_id=generation_id,
            payload={
                "parent_generation": parent_generation,
                "mutation_id": package.mutation.mutation_id,
                "candidate_digest": package.digest,
            },
        )
        return manifest

    def approve(self, generation_id: str, gate_report: str | Path) -> None:
        """记录人工批准；批准本身不会切换 active generation。"""

        self.read_manifest(generation_id)
        report_path = Path(gate_report).resolve()
        report = GateReport.from_dict(_read_json(report_path))
        if report.candidate_generation != generation_id:
            raise ValueError("Gate report 不属于待批准 generation")
        if report.decision is not GateDecision.PASSED:
            raise ValueError("只有 PASSED Gate 才能获得人工批准")
        self.ledger.append(
            "generation_approved",
            generation_id=generation_id,
            payload={
                "gate_report": str(report_path),
                "gate_sha256": _file_digest(report_path),
            },
        )

    def promote(
        self,
        generation_id: str,
        *,
        gate_report: str | Path,
        candidate: CandidatePackage | str | Path,
    ) -> None:
        """核对 Candidate、Gate 和人工批准后原子激活 generation。"""

        manifest = self.read_manifest(generation_id)
        parent = manifest.parent_generation
        if parent is None or parent != self.active_generation():
            raise ValueError("待晋升 generation 的 parent 必须仍是当前 active generation")
        report_path = Path(gate_report).resolve()
        report = GateReport.from_dict(_read_json(report_path))
        if report.candidate_generation != generation_id or report.baseline_generation != parent:
            raise ValueError("Gate report 的 generation 配对与待晋升对象不一致")
        if report.decision is not GateDecision.PASSED:
            raise ValueError("Gate 尚未通过，不能晋升")
        package = CandidatePackage.open(
            candidate.root if isinstance(candidate, CandidatePackage) else candidate
        )
        if manifest.mutation != package.mutation:
            raise ValueError("generation mutation 与 CandidatePackage 不一致")
        created = [
            event
            for event in self.ledger.read()
            if event.get("event") == "generation_created"
            and event.get("generation_id") == generation_id
        ]
        if not created or created[-1].get("payload", {}).get("candidate_digest") != package.digest:
            raise ValueError("generation 没有匹配的 CandidatePackage 创建记录")
        gate_digest = _file_digest(report_path)
        approved = any(
            event.get("event") == "generation_approved"
            and event.get("generation_id") == generation_id
            and event.get("payload", {}).get("gate_sha256") == gate_digest
            for event in self.ledger.read()
        )
        if not approved:
            raise ValueError("缺少与当前 Gate report 匹配的人工批准")
        _atomic_text(self.root / "active_generation", f"{generation_id}\n")
        self.ledger.append(
            "generation_promoted",
            generation_id=generation_id,
            payload={"previous_generation": parent, "gate_sha256": gate_digest},
        )

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

    def _materialize_candidate(
        self,
        *,
        generation_id: str,
        parent_generation: str,
        candidate: CandidatePackage,
        created_at: str,
    ) -> GenerationManifest:
        """在父快照副本上执行唯一 mutation，并原子发布新 generation。"""

        destination = self.generation_path(generation_id)
        if destination.exists():
            raise FileExistsError(f"generation 已存在: {generation_id}")
        staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=self.generations_dir))
        try:
            skills = staging / "skills"
            shutil.copytree(self.generation_path(parent_generation) / "skills", skills)
            self._apply_candidate(skills, candidate)
            digest = skill_tree_digest(skills)
            manifest = GenerationManifest(
                generation_id=generation_id,
                parent_generation=parent_generation,
                created_at=created_at,
                skill_digest=digest,
                mutation=candidate.mutation,
            )
            _write_json(staging / "manifest.json", manifest.to_dict())
            os.rename(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return manifest

    @staticmethod
    def _apply_candidate(skills: Path, candidate: CandidatePackage) -> None:
        """只解释 mutation 的文件语义，不处理审核或晋升策略。"""

        mutation = candidate.mutation
        target = skills / mutation.skill_id
        if mutation.operation is MutationOperation.ADD:
            if target.exists():
                raise ValueError(f"ADD 目标技能已存在: {mutation.skill_id}")
            source = candidate.skill_root
            if source is None:
                raise RuntimeError("ADD 候选缺少已验证的 skill 目录")
            shutil.copytree(source, target)
            return
        if not target.is_dir():
            raise ValueError(f"{mutation.operation.value.upper()} 目标技能不存在: {mutation.skill_id}")
        if mutation.operation is MutationOperation.REVISE:
            source = candidate.skill_root
            if source is None:
                raise RuntimeError("REVISE 候选缺少已验证的 skill 目录")
            shutil.rmtree(target)
            shutil.copytree(source, target)
            return
        if mutation.operation is MutationOperation.RETIRE:
            shutil.rmtree(target)
            return
        raise AssertionError(f"未处理的 mutation operation: {mutation.operation}")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["EvolutionLedger", "GenerationStore", "skill_tree_digest"]
