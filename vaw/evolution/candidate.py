"""Skill Evolver 与 generation 存储之间的不可变候选包。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from vaw.evolution.domain import MutationOperation, SkillMutation

CANDIDATE_SCHEMA = "vaw-candidate-v1"
_AUDIT_FILES = {
    "evolver": ("request.json", "response.json"),
    "review": ("request.json", "response.json", "report.json"),
}


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 文档必须是 object: {path}")
    return payload


def _write_object(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


def _safe_relative(value: str, field_name: str) -> str:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{field_name} 必须是目录内的相对路径")
    return path.as_posix()


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"候选内容必须是普通目录: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"候选内容不接受符号链接: {path}")


def _directory_digest(root: Path) -> str:
    """计算候选包内容摘要；manifest 自身不参与摘要。"""

    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "manifest.json"
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _validate_audit(root: Path, name: str) -> None:
    """审核记录是密封候选的一部分，缺失时不能形成 Candidate。"""

    _reject_symlinks(root)
    for filename in _AUDIT_FILES[name]:
        _read_object(root / filename)
    if name == "review":
        report = _read_object(root / "report.json")
        if report.get("decision") != "accept":
            raise ValueError("只有 Reviewer accept 的草稿才能密封为 Candidate")


@dataclass(frozen=True)
class CandidateEvidence:
    """一条经过 M3 审核的 policy-visible 证据引用。"""

    m3_output: str
    finding_id: str
    raster: str
    sha256: str

    def __post_init__(self) -> None:
        root = str(Path(_require_text(self.m3_output, "m3_output")).resolve())
        digest = self.sha256.strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("sha256 必须是 SHA-256 十六进制摘要")
        object.__setattr__(self, "m3_output", root)
        object.__setattr__(self, "finding_id", _require_text(self.finding_id, "finding_id"))
        object.__setattr__(self, "raster", _safe_relative(self.raster, "raster"))
        object.__setattr__(self, "sha256", digest)

    @property
    def raster_path(self) -> Path:
        return Path(self.m3_output) / self.raster

    def verify(self) -> None:
        source = self.raster_path
        if not source.is_file():
            raise FileNotFoundError(f"候选证据图片不存在: {source}")
        if _file_sha256(source) != self.sha256:
            raise ValueError(f"候选证据图片摘要不匹配: {source}")

    def to_dict(self) -> dict[str, str]:
        return {
            "m3_output": self.m3_output,
            "finding_id": self.finding_id,
            "raster": self.raster,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateEvidence:
        return cls(
            m3_output=str(payload["m3_output"]),
            finding_id=str(payload["finding_id"]),
            raster=str(payload["raster"]),
            sha256=str(payload["sha256"]),
        )


@dataclass(frozen=True)
class CandidatePackage:
    """Reviewer 接受后形成的密封候选包。"""

    root: Path
    mutation: SkillMutation
    evidence: Mapping[str, CandidateEvidence]
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    @property
    def skill_root(self) -> Path | None:
        path = self.root / "skill"
        return path if path.is_dir() else None

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        mutation: SkillMutation,
        evidence: Mapping[str, CandidateEvidence],
        skill_root: str | Path | None = None,
        evolver_root: str | Path,
        review_root: str | Path,
    ) -> CandidatePackage:
        """审核通过后原子密封候选；失败时不留下半份目录。"""

        destination = Path(root).resolve()
        if destination.exists():
            raise FileExistsError(f"候选包已存在: {destination}")
        if set(evidence) != set(mutation.evidence_ids):
            raise ValueError("evidence.json 必须与 mutation.evidence_ids 精确对应")
        for item in evidence.values():
            item.verify()

        source = Path(skill_root).resolve() if skill_root is not None else None
        needs_skill = mutation.operation in {MutationOperation.ADD, MutationOperation.REVISE}
        if needs_skill:
            if source is None or not (source / "SKILL.md").is_file():
                raise ValueError("ADD/REVISE 候选必须提供包含 SKILL.md 的技能目录")
            _reject_symlinks(source)
        elif source is not None:
            raise ValueError("RETIRE 候选不能携带 skill 目录")

        audit_sources = {
            "evolver": Path(evolver_root).resolve(),
            "review": Path(review_root).resolve(),
        }
        for name, audit_root in audit_sources.items():
            _validate_audit(audit_root, name)

        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            _write_object(staging / "mutation.json", mutation.to_dict())
            _write_object(
                staging / "evidence.json",
                {evidence_id: item.to_dict() for evidence_id, item in sorted(evidence.items())},
            )
            if source is not None:
                shutil.copytree(source, staging / "skill")
            for name, audit_root in audit_sources.items():
                shutil.copytree(audit_root, staging / name)
            _write_object(
                staging / "manifest.json",
                {
                    "schema": CANDIDATE_SCHEMA,
                    "content_digest": _directory_digest(staging),
                },
            )
            os.rename(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return cls.open(destination)

    @classmethod
    def open(cls, root: str | Path) -> CandidatePackage:
        """读取密封候选，并验证内容、证据与 Reviewer 决定。"""

        directory = Path(root).resolve()
        _reject_symlinks(directory)
        manifest = _read_object(directory / "manifest.json")
        if manifest.get("schema") != CANDIDATE_SCHEMA:
            raise ValueError(f"不支持的 candidate schema: {manifest.get('schema')!r}")
        expected_digest = str(manifest.get("content_digest") or "")
        actual_digest = _directory_digest(directory)
        if expected_digest != actual_digest:
            raise ValueError("CandidatePackage 内容摘要不匹配")
        mutation = SkillMutation.from_dict(_read_object(directory / "mutation.json"))
        raw_evidence = _read_object(directory / "evidence.json")
        evidence = {
            evidence_id: CandidateEvidence.from_dict(payload)
            for evidence_id, payload in raw_evidence.items()
            if isinstance(payload, Mapping)
        }
        if len(evidence) != len(raw_evidence) or set(evidence) != set(mutation.evidence_ids):
            raise ValueError("候选 evidence 与 mutation 引用不一致")
        for item in evidence.values():
            item.verify()

        skill_root = directory / "skill"
        needs_skill = mutation.operation in {MutationOperation.ADD, MutationOperation.REVISE}
        if needs_skill:
            if not (skill_root / "SKILL.md").is_file():
                raise ValueError("ADD/REVISE 候选缺少 skill/SKILL.md")
            _reject_symlinks(skill_root)
        elif skill_root.exists():
            raise ValueError("RETIRE 候选不能携带 skill 目录")
        for name in _AUDIT_FILES:
            _validate_audit(directory / name, name)
        return cls(
            root=directory,
            mutation=mutation,
            evidence=evidence,
            digest=actual_digest,
        )


__all__ = ["CANDIDATE_SCHEMA", "CandidateEvidence", "CandidatePackage"]
