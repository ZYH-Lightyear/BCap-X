"""VAW 技能进化的纯领域模型。

本模块不访问文件系统、Runtime 或模型服务。它只描述冻结实验、单项技能变更和
配对评测事实；持久化与生命周期由 :mod:`vaw.evolution.store` 负责。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

EVOLUTION_SCHEMA = "vaw-evolution-v2"
GENERATION_SCHEMA = "vaw-generation-v2"
GATE_REPORT_SCHEMA = "vaw-gate-report-v3"
SKILL_EFFECT_REPORT_SCHEMA = "vaw-skill-effect-report-v1"

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
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
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


def _require_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


def _integers(values: Sequence[int], field_name: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if any(value < 0 for value in normalized):
        raise ValueError(f"{field_name} 不能包含负数")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} 不能包含重复值")
    return normalized


def _require_schema(payload: Mapping[str, Any], expected: str) -> None:
    actual = payload.get("schema")
    if actual != expected:
        raise ValueError(f"不支持的 schema: {actual!r}，需要 {expected!r}")


class MutationOperation(StrEnum):
    """每代只允许一个可归因的技能库变更。"""

    ADD = "add"
    REVISE = "revise"
    RETIRE = "retire"


class GateDecision(StrEnum):
    """完成一次配对 Gate 后可持久化的结论。"""

    PASSED = "passed"
    INCONCLUSIVE = "inconclusive"
    REJECTED = "rejected"


class EpisodeOutcome(StrEnum):
    """配对 episode 的最小结果，基础设施失败不冒充任务失败。"""

    SUCCESS = "success"
    FAILURE = "failure"
    INFRASTRUCTURE = "infrastructure"


class SkillEffect(StrEnum):
    """候选技能对其目标决策问题造成的局部变化。"""

    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    WORSE = "worse"
    UNCERTAIN = "uncertain"


class RemainingFailure(StrEnum):
    """终局仍失败时，剩余原因与目标技能问题的关系。"""

    NONE = "none"
    RELATED = "related"
    OTHER = "other"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class GatePolicy:
    """随实验冻结的防退化门控参数。"""

    primary_seeds: tuple[int, ...] = (1, 2)
    confirmation_seeds: tuple[int, ...] = (3,)
    min_recoveries: int = 2
    min_attributed_recoveries: int = 1
    max_turn_ratio: float = 1.5
    max_token_ratio: float = 1.5
    auto_rollback: bool = True

    def __post_init__(self) -> None:
        primary = _integers(self.primary_seeds, "primary_seeds")
        confirmation = _integers(self.confirmation_seeds, "confirmation_seeds")
        if set(primary) & set(confirmation):
            raise ValueError("primary_seeds 与 confirmation_seeds 必须互不重叠")
        if self.min_recoveries < 1:
            raise ValueError("min_recoveries 必须为正数")
        if not 0 <= self.min_attributed_recoveries <= self.min_recoveries:
            raise ValueError("min_attributed_recoveries 必须位于 0..min_recoveries")
        if self.max_turn_ratio <= 0 or not math.isfinite(self.max_turn_ratio):
            raise ValueError("max_turn_ratio 必须是正有限数")
        if self.max_token_ratio <= 0 or not math.isfinite(self.max_token_ratio):
            raise ValueError("max_token_ratio 必须是正有限数")
        object.__setattr__(self, "primary_seeds", primary)
        object.__setattr__(self, "confirmation_seeds", confirmation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_seeds": list(self.primary_seeds),
            "confirmation_seeds": list(self.confirmation_seeds),
            "min_recoveries": self.min_recoveries,
            "min_attributed_recoveries": self.min_attributed_recoveries,
            "max_turn_ratio": self.max_turn_ratio,
            "max_token_ratio": self.max_token_ratio,
            "auto_rollback": self.auto_rollback,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GatePolicy:
        return cls(
            primary_seeds=tuple(payload["primary_seeds"]),
            confirmation_seeds=tuple(payload["confirmation_seeds"]),
            min_recoveries=int(payload["min_recoveries"]),
            min_attributed_recoveries=int(payload["min_attributed_recoveries"]),
            max_turn_ratio=float(payload["max_turn_ratio"]),
            max_token_ratio=float(payload["max_token_ratio"]),
            auto_rollback=bool(payload["auto_rollback"]),
        )


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
    gate_policy: GatePolicy = field(default_factory=GatePolicy)
    schema: str = field(default=EVOLUTION_SCHEMA, init=False)

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
        object.__setattr__(self, "evolve_tasks", _integers(self.evolve_tasks, "evolve_tasks"))
        object.__setattr__(self, "gate_tasks", _integers(self.gate_tasks, "gate_tasks"))
        object.__setattr__(self, "reserve_tasks", _integers(self.reserve_tasks, "reserve_tasks"))
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
            "gate_policy": self.gate_policy.to_dict(),
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
        """从两份冻结文档恢复 v2 实验；旧 schema 必须显式重建。"""

        _require_schema(experiment, EVOLUTION_SCHEMA)
        policy = experiment.get("gate_policy")
        if not isinstance(policy, Mapping):
            raise ValueError("experiment 缺少 gate_policy object")
        return cls(
            experiment_id=str(experiment["experiment_id"]),
            base_generation=str(experiment["base_generation"]),
            evolve_suite=str(split["evolve_suite"]),
            evolve_tasks=tuple(split["evolve_tasks"]),
            gate_tasks=tuple(split["gate_tasks"]),
            reserve_tasks=tuple(split["reserve_tasks"]),
            heldout_suites=tuple(split["heldout_suites"]),
            rollout_config=experiment.get("rollout_config", {}),
            frozen_components=experiment.get("frozen_components", {}),
            gate_policy=GatePolicy.from_dict(policy),
        )


@dataclass(frozen=True)
class SkillMutation:
    """一项有证据来源、但尚未晋升的技能库变更。"""

    mutation_id: str
    operation: MutationOperation
    skill_id: str
    evidence_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        evidence = tuple(_require_name(item, "evidence_id") for item in self.evidence_ids)
        if not evidence or len(set(evidence)) != len(evidence):
            raise ValueError("evidence_ids 必须非空且不能重复")
        object.__setattr__(self, "mutation_id", _require_name(self.mutation_id, "mutation_id"))
        object.__setattr__(self, "skill_id", _require_name(self.skill_id, "skill_id"))
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "rationale", _require_text(self.rationale, "rationale"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "operation": self.operation.value,
            "skill_id": self.skill_id,
            "evidence_ids": list(self.evidence_ids),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SkillMutation:
        return cls(
            mutation_id=str(payload["mutation_id"]),
            operation=MutationOperation(str(payload["operation"])),
            skill_id=str(payload["skill_id"]),
            evidence_ids=tuple(payload["evidence_ids"]),
            rationale=str(payload["rationale"]),
        )


@dataclass(frozen=True)
class GenerationManifest:
    """一个不可变 generation 的内容与来源说明。"""

    generation_id: str
    parent_generation: str | None
    created_at: str
    skill_digest: str
    mutation: SkillMutation | None = None
    schema: str = field(default=GENERATION_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "generation_id", _require_name(self.generation_id, "generation_id")
        )
        if self.parent_generation is not None:
            object.__setattr__(
                self,
                "parent_generation",
                _require_name(self.parent_generation, "parent_generation"),
            )
        object.__setattr__(self, "created_at", _require_text(self.created_at, "created_at"))
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
        _require_schema(payload, GENERATION_SCHEMA)
        mutation = payload.get("mutation")
        return cls(
            generation_id=str(payload["generation_id"]),
            parent_generation=payload.get("parent_generation"),
            created_at=str(payload["created_at"]),
            skill_digest=str(payload["skill_digest"]),
            mutation=SkillMutation.from_dict(mutation) if isinstance(mutation, Mapping) else None,
        )


@dataclass(frozen=True)
class GatePair:
    """同一 task、seed 与初始状态下的一组配对事实。"""

    task_id: int
    seed: int
    phase: Literal["primary", "confirmation"]
    baseline: EpisodeOutcome
    candidate: EpisodeOutcome
    baseline_consulted: bool
    candidate_consulted: bool

    def __post_init__(self) -> None:
        if self.task_id < 0 or self.seed < 0:
            raise ValueError("task_id 和 seed 不能为负数")
        if self.phase not in {"primary", "confirmation"}:
            raise ValueError("phase 必须是 primary 或 confirmation")

    @property
    def is_recovery(self) -> bool:
        return self.baseline is EpisodeOutcome.FAILURE and self.candidate is EpisodeOutcome.SUCCESS

    @property
    def is_regression(self) -> bool:
        return self.baseline is EpisodeOutcome.SUCCESS and self.candidate is EpisodeOutcome.FAILURE

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "phase": self.phase,
            "baseline": self.baseline.value,
            "candidate": self.candidate.value,
            "baseline_consulted": self.baseline_consulted,
            "candidate_consulted": self.candidate_consulted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GatePair:
        return cls(
            task_id=int(payload["task_id"]),
            seed=int(payload["seed"]),
            phase=str(payload["phase"]),  # type: ignore[arg-type]
            baseline=EpisodeOutcome(str(payload["baseline"])),
            candidate=EpisodeOutcome(str(payload["candidate"])),
            baseline_consulted=bool(payload["baseline_consulted"]),
            candidate_consulted=bool(payload["candidate_consulted"]),
        )


@dataclass(frozen=True)
class SkillEffectReview:
    """独立多模态 Reviewer 对一组配对轨迹给出的最小审计结论。"""

    task_id: int
    seed: int
    effect: SkillEffect
    remaining_failure: RemainingFailure
    evidence_ids: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.task_id < 0 or self.seed < 0:
            raise ValueError("task_id 和 seed 不能为负数")
        evidence = tuple(_require_name(item, "evidence_id") for item in self.evidence_ids)
        if not evidence or len(set(evidence)) != len(evidence):
            raise ValueError("evidence_ids 必须非空且不能重复")
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "reason", _require_text(self.reason, "reason"))

    @property
    def key(self) -> tuple[int, int]:
        return self.task_id, self.seed

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "effect": self.effect.value,
            "remaining_failure": self.remaining_failure.value,
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SkillEffectReview:
        return cls(
            task_id=int(payload["task_id"]),
            seed=int(payload["seed"]),
            effect=SkillEffect(str(payload["effect"])),
            remaining_failure=RemainingFailure(str(payload["remaining_failure"])),
            evidence_ids=tuple(payload["evidence_ids"]),
            reason=str(payload["reason"]),
        )


@dataclass(frozen=True)
class SkillEffectReport:
    """把配对轨迹来源与独立局部效果判断绑定为可审计文件。"""

    mutation_id: str
    baseline_generation: str
    candidate_generation: str
    baseline_day_index: str
    candidate_day_index: str
    reviews: tuple[SkillEffectReview, ...]
    schema: str = field(default=SKILL_EFFECT_REPORT_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mutation_id", _require_name(self.mutation_id, "mutation_id"))
        object.__setattr__(
            self,
            "baseline_generation",
            _require_name(self.baseline_generation, "baseline_generation"),
        )
        object.__setattr__(
            self,
            "candidate_generation",
            _require_name(self.candidate_generation, "candidate_generation"),
        )
        object.__setattr__(
            self,
            "baseline_day_index",
            _require_text(self.baseline_day_index, "baseline_day_index"),
        )
        object.__setattr__(
            self,
            "candidate_day_index",
            _require_text(self.candidate_day_index, "candidate_day_index"),
        )
        reviews = tuple(self.reviews)
        if len({item.key for item in reviews}) != len(reviews):
            raise ValueError("SkillEffectReport 不能重复 task/seed")
        object.__setattr__(self, "reviews", reviews)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mutation_id": self.mutation_id,
            "baseline_generation": self.baseline_generation,
            "candidate_generation": self.candidate_generation,
            "baseline_day_index": self.baseline_day_index,
            "candidate_day_index": self.candidate_day_index,
            "reviews": [item.to_dict() for item in self.reviews],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SkillEffectReport:
        _require_schema(payload, SKILL_EFFECT_REPORT_SCHEMA)
        return cls(
            mutation_id=str(payload["mutation_id"]),
            baseline_generation=str(payload["baseline_generation"]),
            candidate_generation=str(payload["candidate_generation"]),
            baseline_day_index=str(payload["baseline_day_index"]),
            candidate_day_index=str(payload["candidate_day_index"]),
            reviews=tuple(SkillEffectReview.from_dict(item) for item in payload["reviews"]),
        )


@dataclass(frozen=True)
class GateReport:
    """保存输入索引与逐 episode 事实，不复制可派生统计。"""

    mutation_id: str
    baseline_generation: str
    candidate_generation: str
    baseline_day_index: str
    candidate_day_index: str
    decision: GateDecision
    reason: str
    pairs: tuple[GatePair, ...]
    skill_effects: tuple[SkillEffectReview, ...] = ()
    schema: str = field(default=GATE_REPORT_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mutation_id", _require_name(self.mutation_id, "mutation_id"))
        object.__setattr__(
            self,
            "baseline_generation",
            _require_name(self.baseline_generation, "baseline_generation"),
        )
        object.__setattr__(
            self,
            "candidate_generation",
            _require_name(self.candidate_generation, "candidate_generation"),
        )
        object.__setattr__(
            self, "baseline_day_index", _require_text(self.baseline_day_index, "baseline_day_index")
        )
        object.__setattr__(
            self,
            "candidate_day_index",
            _require_text(self.candidate_day_index, "candidate_day_index"),
        )
        object.__setattr__(self, "reason", _require_text(self.reason, "reason"))
        object.__setattr__(self, "pairs", tuple(self.pairs))
        effects = tuple(self.skill_effects)
        pair_keys = {(item.task_id, item.seed) for item in self.pairs}
        if len({item.key for item in effects}) != len(effects):
            raise ValueError("skill_effects 不能重复 task/seed")
        if any(item.key not in pair_keys for item in effects):
            raise ValueError("skill_effects 必须属于 GatePair")
        object.__setattr__(self, "skill_effects", effects)

    @property
    def recoveries(self) -> int:
        return sum(pair.is_recovery for pair in self.pairs)

    @property
    def regressions(self) -> int:
        return sum(pair.is_regression for pair in self.pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mutation_id": self.mutation_id,
            "baseline_generation": self.baseline_generation,
            "candidate_generation": self.candidate_generation,
            "baseline_day_index": self.baseline_day_index,
            "candidate_day_index": self.candidate_day_index,
            "decision": self.decision.value,
            "reason": self.reason,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "skill_effects": [item.to_dict() for item in self.skill_effects],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GateReport:
        _require_schema(payload, GATE_REPORT_SCHEMA)
        return cls(
            mutation_id=str(payload["mutation_id"]),
            baseline_generation=str(payload["baseline_generation"]),
            candidate_generation=str(payload["candidate_generation"]),
            baseline_day_index=str(payload["baseline_day_index"]),
            candidate_day_index=str(payload["candidate_day_index"]),
            decision=GateDecision(str(payload["decision"])),
            reason=str(payload["reason"]),
            pairs=tuple(GatePair.from_dict(pair) for pair in payload["pairs"]),
            skill_effects=tuple(
                SkillEffectReview.from_dict(item) for item in payload["skill_effects"]
            ),
        )


__all__ = [
    "EVOLUTION_SCHEMA",
    "GENERATION_SCHEMA",
    "GATE_REPORT_SCHEMA",
    "SKILL_EFFECT_REPORT_SCHEMA",
    "EpisodeOutcome",
    "EvolutionSpec",
    "GateDecision",
    "GatePair",
    "GatePolicy",
    "GateReport",
    "GenerationManifest",
    "MutationOperation",
    "RemainingFailure",
    "SkillEffect",
    "SkillEffectReport",
    "SkillEffectReview",
    "SkillMutation",
    "thaw_json",
]
