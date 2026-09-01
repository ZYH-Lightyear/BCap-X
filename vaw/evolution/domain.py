"""VAW 技能进化的纯领域模型。

本模块不访问文件系统、Runtime 或模型服务。领域对象只描述实验、技能变更与
配对评测事实，持久化和生命周期由 :mod:`vaw.evolution.store` 负责。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

JsonValue = None | bool | int | float | str | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


def _freeze_json(value: Any) -> JsonValue:
    """复制并冻结 JSON 值，防止 frozen dataclass 被嵌套容器绕过。"""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON 数值必须是有限值")
        return value
    if isinstance(value, Mapping):
        frozen = {str(key): _freeze_json(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"不支持的 JSON 值类型: {type(value).__name__}")


def thaw_json(value: JsonValue) -> Any:
    """将冻结的 JSON 值转换为可序列化的新对象。"""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _require_name(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError(f"{field_name} 必须是单段名称")
    return normalized


def _task_ids(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if any(value < 0 for value in normalized):
        raise ValueError(f"{field_name} 不能包含负数")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} 不能包含重复任务")
    return normalized


class MutationOperation(Enum):
    """NIGHT 阶段允许提出的技能库变更。"""

    ADD = "ADD"
    REVISE = "REVISE"
    MERGE = "MERGE"
    RETIRE = "RETIRE"


class GateDecision(Enum):
    """配对 Gate 的审查结论。"""

    PENDING = "pending"
    PASSED = "passed"
    INCONCLUSIVE = "inconclusive"
    REJECTED = "rejected"


@dataclass(frozen=True)
class EvolutionSpec:
    """一次进化实验中必须冻结的配置与数据划分。"""

    experiment_id: str
    base_generation: str
    evolve_suite: str
    evolve_tasks: tuple[int, ...]
    gate_tasks: tuple[int, ...]
    reserve_tasks: tuple[int, ...]
    heldout_suites: tuple[str, ...]
    rollout_config: Mapping[str, JsonValue] = field(default_factory=dict)
    frozen_components: Mapping[str, JsonValue] = field(default_factory=dict)
    schema: str = "vaw-evolution-v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_id", _require_name(self.experiment_id, "experiment_id")
        )
        object.__setattr__(
            self,
            "base_generation",
            _require_name(self.base_generation, "base_generation"),
        )
        object.__setattr__(self, "evolve_suite", _require_name(self.evolve_suite, "evolve_suite"))
        object.__setattr__(self, "evolve_tasks", _task_ids(self.evolve_tasks, "evolve_tasks"))
        object.__setattr__(self, "gate_tasks", _task_ids(self.gate_tasks, "gate_tasks"))
        object.__setattr__(self, "reserve_tasks", _task_ids(self.reserve_tasks, "reserve_tasks"))
        heldout = tuple(_require_name(item, "heldout_suite") for item in self.heldout_suites)
        if len(set(heldout)) != len(heldout):
            raise ValueError("heldout_suites 不能重复")
        object.__setattr__(self, "heldout_suites", heldout)
        object.__setattr__(self, "rollout_config", _freeze_json(self.rollout_config))
        object.__setattr__(self, "frozen_components", _freeze_json(self.frozen_components))

        split_sets = (set(self.evolve_tasks), set(self.gate_tasks), set(self.reserve_tasks))
        if any(split_sets[left] & split_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("evolve、gate 和 reserve 任务必须互不重叠")

    def experiment_document(self) -> dict[str, Any]:
        """返回不重复任务划分的实验配置文档。"""

        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "base_generation": self.base_generation,
            "rollout_config": thaw_json(self.rollout_config),
            "frozen_components": thaw_json(self.frozen_components),
        }

    def split_document(self) -> dict[str, Any]:
        """返回独立的数据划分文档。"""

        return {
            "evolve_suite": self.evolve_suite,
            "evolve_tasks": list(self.evolve_tasks),
            "gate_tasks": list(self.gate_tasks),
            "reserve_tasks": list(self.reserve_tasks),
            "heldout_suites": list(self.heldout_suites),
        }

    @classmethod
    def from_documents(
        cls,
        experiment: Mapping[str, Any],
        split: Mapping[str, Any],
    ) -> EvolutionSpec:
        """从磁盘上的两份冻结文档恢复领域对象。"""

        return cls(
            schema=str(experiment.get("schema", "vaw-evolution-v1")),
            experiment_id=str(experiment["experiment_id"]),
            base_generation=str(experiment["base_generation"]),
            evolve_suite=str(split["evolve_suite"]),
            evolve_tasks=tuple(split["evolve_tasks"]),
            gate_tasks=tuple(split["gate_tasks"]),
            reserve_tasks=tuple(split["reserve_tasks"]),
            heldout_suites=tuple(split["heldout_suites"]),
            rollout_config=experiment.get("rollout_config", {}),
            frozen_components=experiment.get("frozen_components", {}),
        )


@dataclass(frozen=True)
class SkillMutation:
    """一项有证据来源、但尚未晋升的技能库变更。"""

    mutation_id: str
    operation: MutationOperation
    target_skill_ids: tuple[str, ...]
    candidate_skill_id: str | None
    evidence_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "mutation_id", _require_name(self.mutation_id, "mutation_id"))
        targets = tuple(_require_name(item, "target_skill_id") for item in self.target_skill_ids)
        evidence = tuple(_require_name(item, "evidence_id") for item in self.evidence_ids)
        candidate = self.candidate_skill_id
        if candidate is not None:
            candidate = _require_name(str(candidate), "candidate_skill_id")
        if not self.rationale.strip():
            raise ValueError("rationale 不能为空")
        object.__setattr__(self, "target_skill_ids", targets)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "candidate_skill_id", candidate)
        object.__setattr__(self, "rationale", self.rationale.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "operation": self.operation.value,
            "target_skill_ids": list(self.target_skill_ids),
            "candidate_skill_id": self.candidate_skill_id,
            "evidence_ids": list(self.evidence_ids),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SkillMutation:
        return cls(
            mutation_id=str(payload["mutation_id"]),
            operation=MutationOperation(str(payload["operation"])),
            target_skill_ids=tuple(payload.get("target_skill_ids", ())),
            candidate_skill_id=payload.get("candidate_skill_id"),
            evidence_ids=tuple(payload.get("evidence_ids", ())),
            rationale=str(payload["rationale"]),
        )


@dataclass(frozen=True)
class GenerationManifest:
    """一个已物化 generation 的不可变说明。

    Gate、批准和激活状态记录在 append-only ledger 中，而不是回写 manifest。
    """

    generation_id: str
    parent_generation: str | None
    created_at: str
    skill_digest: str
    mutation: SkillMutation | None = None
    schema: str = "vaw-generation-v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generation_id",
            _require_name(self.generation_id, "generation_id"),
        )
        if self.parent_generation is not None:
            object.__setattr__(
                self,
                "parent_generation",
                _require_name(self.parent_generation, "parent_generation"),
            )
        if not self.created_at.strip():
            raise ValueError("created_at 不能为空")
        digest = self.skill_digest.strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("skill_digest 必须是 SHA-256 十六进制摘要")
        object.__setattr__(self, "skill_digest", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "generation_id": self.generation_id,
            "parent_generation": self.parent_generation,
            "created_at": self.created_at,
            "skill_digest": self.skill_digest,
            "mutation": self.mutation.to_dict() if self.mutation else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GenerationManifest:
        mutation = payload.get("mutation")
        return cls(
            schema=str(payload.get("schema", "vaw-generation-v1")),
            generation_id=str(payload["generation_id"]),
            parent_generation=payload.get("parent_generation"),
            created_at=str(payload["created_at"]),
            skill_digest=str(payload["skill_digest"]),
            mutation=SkillMutation.from_dict(mutation) if isinstance(mutation, Mapping) else None,
        )


@dataclass(frozen=True)
class GatePair:
    """同一 task、seed 和初始状态下的一组基线/候选结果。"""

    suite: str
    task_id: int
    seed: int
    baseline_success: bool
    candidate_success: bool
    candidate_consulted: bool
    infrastructure_error: bool = False


@dataclass(frozen=True)
class GateReport:
    """保留逐 episode 事实的配对 Gate 报告。"""

    mutation_id: str
    baseline_generation: str
    decision: GateDecision
    pairs: tuple[GatePair, ...]
    checks: Mapping[str, JsonValue] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    schema: str = "vaw-gate-report-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mutation_id", _require_name(self.mutation_id, "mutation_id"))
        object.__setattr__(
            self,
            "baseline_generation",
            _require_name(self.baseline_generation, "baseline_generation"),
        )
        object.__setattr__(self, "pairs", tuple(self.pairs))
        object.__setattr__(self, "checks", _freeze_json(self.checks))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))

    @property
    def recoveries(self) -> int:
        return sum(
            not pair.baseline_success and pair.candidate_success
            for pair in self.pairs
            if not pair.infrastructure_error
        )

    @property
    def regressions(self) -> int:
        return sum(
            pair.baseline_success and not pair.candidate_success
            for pair in self.pairs
            if not pair.infrastructure_error
        )

    @property
    def consulted_recoveries(self) -> int:
        return sum(
            not pair.baseline_success and pair.candidate_success and pair.candidate_consulted
            for pair in self.pairs
            if not pair.infrastructure_error
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mutation_id": self.mutation_id,
            "baseline_generation": self.baseline_generation,
            "decision": self.decision.value,
            "pairs": [
                {
                    "suite": pair.suite,
                    "task_id": pair.task_id,
                    "seed": pair.seed,
                    "baseline_success": pair.baseline_success,
                    "candidate_success": pair.candidate_success,
                    "candidate_consulted": pair.candidate_consulted,
                    "infrastructure_error": pair.infrastructure_error,
                }
                for pair in self.pairs
            ],
            "checks": thaw_json(self.checks),
            "notes": list(self.notes),
            "summary": {
                "recoveries": self.recoveries,
                "regressions": self.regressions,
                "consulted_recoveries": self.consulted_recoveries,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GateReport:
        return cls(
            schema=str(payload.get("schema", "vaw-gate-report-v1")),
            mutation_id=str(payload["mutation_id"]),
            baseline_generation=str(payload["baseline_generation"]),
            decision=GateDecision(str(payload["decision"])),
            pairs=tuple(GatePair(**pair) for pair in payload.get("pairs", ())),
            checks=payload.get("checks", {}),
            notes=tuple(payload.get("notes", ())),
        )


__all__ = [
    "EvolutionSpec",
    "GateDecision",
    "GatePair",
    "GateReport",
    "GenerationManifest",
    "MutationOperation",
    "SkillMutation",
    "thaw_json",
]
