"""对 baseline 与 candidate DAY 结果进行成本感知的配对门控。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vaw.evolution.artifacts import read_json, read_jsonl, write_json
from vaw.evolution.domain import (
    EpisodeOutcome,
    GateDecision,
    GatePair,
    GatePolicy,
    GateReport,
    MutationOperation,
    SkillEffect,
    SkillEffectReport,
    SkillEffectReview,
    SkillMutation,
)


@dataclass(frozen=True)
class IndexedEpisode:
    """Gate 使用的 DAY 事实；trace 只用于检查技能是否被实际加载。"""

    task_id: int
    seed: int
    outcome: EpisodeOutcome
    turns: int | None
    tokens: int | None
    wall_time_s: float | None
    trace_dir: Path


@dataclass(frozen=True)
class GateMetrics:
    recoveries: int
    regressions: int
    attributed_recoveries: int
    baseline_successes: int
    candidate_successes: int
    turn_ratio: float | None
    token_ratio: float | None
    wall_time_ratio: float | None
    local_improvements: int
    local_regressions: int
    improved_but_failed_elsewhere: int


def evaluate_gate(
    *,
    mutation: SkillMutation,
    baseline_generation: str,
    candidate_generation: str,
    baseline_day_index: str | Path,
    candidate_day_index: str | Path,
    policy: GatePolicy,
    expected_tasks: Sequence[int] | None = None,
    skill_effect_report: str | Path | SkillEffectReport | None = None,
) -> tuple[GateReport, GateMetrics]:
    """读取两份既有 DAY index；不会在分析阶段隐式启动 rollout。"""

    baseline_path = Path(baseline_day_index).resolve()
    candidate_path = Path(candidate_day_index).resolve()
    assert_paired_runs(
        baseline_path,
        candidate_path,
        baseline_generation=baseline_generation,
        candidate_generation=candidate_generation,
    )
    effects = _load_effects(
        skill_effect_report,
        mutation=mutation,
        baseline_generation=baseline_generation,
        candidate_generation=candidate_generation,
        baseline_day_index=baseline_path,
        candidate_day_index=candidate_path,
    )
    baseline = load_day_index(baseline_path)
    candidate = load_day_index(candidate_path)
    pairs: list[GatePair] = []
    missing = sorted(set(baseline) ^ set(candidate))
    for task_id, seed in sorted(set(baseline) & set(candidate)):
        phase = "primary" if seed in policy.primary_seeds else "confirmation"
        base = baseline[task_id, seed]
        cand = candidate[task_id, seed]
        pairs.append(
            GatePair(
                task_id=task_id,
                seed=seed,
                phase=phase,
                baseline=base.outcome,
                candidate=cand.outcome,
                baseline_consulted=consulted_skill(base.trace_dir, mutation.skill_id),
                candidate_consulted=consulted_skill(cand.trace_dir, mutation.skill_id),
            )
        )
    if missing:
        decision = GateDecision.INCONCLUSIVE
        reason = f"paired DAY 缺少 {len(missing)} 个 task/seed 对"
    else:
        decision, reason = _decision(
            pairs,
            baseline,
            candidate,
            mutation,
            policy,
            expected_tasks,
            effects,
        )
    metrics = _metrics(pairs, baseline, candidate, mutation, effects)
    pair_keys = {(pair.task_id, pair.seed) for pair in pairs}
    report = GateReport(
        mutation_id=mutation.mutation_id,
        baseline_generation=baseline_generation,
        candidate_generation=candidate_generation,
        baseline_day_index=str(baseline_path),
        candidate_day_index=str(candidate_path),
        decision=decision,
        reason=reason,
        pairs=tuple(pairs),
        skill_effects=tuple(
            review for key, review in effects.items() if key in pair_keys
        ),
    )
    return report, metrics


def assert_paired_runs(
    baseline_index: Path,
    candidate_index: Path,
    *,
    baseline_generation: str,
    candidate_generation: str,
) -> None:
    """确认两份 DAY 结果来自指定 generation，且其余冻结配置完全相同。"""

    baseline = read_json(baseline_index.parent / "run.json")
    candidate = read_json(candidate_index.parent / "run.json")
    if baseline.get("generation_id") != baseline_generation:
        raise ValueError("baseline DAY run 的 generation_id 与 Gate 参数不一致")
    if candidate.get("generation_id") != candidate_generation:
        raise ValueError("candidate DAY run 的 generation_id 与 Gate 参数不一致")
    ignored = {"generation_id", "skill_digest"}
    baseline = {key: value for key, value in baseline.items() if key not in ignored}
    candidate = {key: value for key, value in candidate.items() if key not in ignored}
    if baseline != candidate:
        raise ValueError("baseline 与 candidate DAY run 不是同配置配对实验")


def write_gate_report(
    output_dir: str | Path,
    report: GateReport,
    metrics: GateMetrics,
) -> tuple[Path, Path]:
    """JSON 保存最小事实，Markdown 展示可派生统计与成本。"""

    output = Path(output_dir).resolve()
    json_path = write_json(output / "report.json", report.to_dict())
    rows = [
        "| task | seed | phase | baseline | candidate | baseline consult | candidate consult | skill effect | remaining failure |",
        "|---:|---:|---|---|---|---|---|---|---|",
        *(
            f"| {pair.task_id} | {pair.seed} | {pair.phase} | {pair.baseline.value} | "
            f"{pair.candidate.value} | {pair.baseline_consulted} | {pair.candidate_consulted} | "
            f"{_effect_text(report.skill_effects, pair, 'effect')} | "
            f"{_effect_text(report.skill_effects, pair, 'remaining_failure')} |"
            for pair in report.pairs
        ),
    ]
    markdown = "\n".join(
        [
            f"# Gate {report.decision.value.upper()}",
            "",
            report.reason,
            "",
            f"- recoveries: {metrics.recoveries}",
            f"- regressions: {metrics.regressions}",
            f"- attributed recoveries: {metrics.attributed_recoveries}",
            f"- local improvements: {metrics.local_improvements}",
            f"- local regressions: {metrics.local_regressions}",
            f"- improved but later failed elsewhere: {metrics.improved_but_failed_elsewhere}",
            f"- success: {metrics.baseline_successes} → {metrics.candidate_successes}",
            f"- mean turn ratio: {_ratio_text(metrics.turn_ratio)}",
            f"- total token ratio: {_ratio_text(metrics.token_ratio)}",
            f"- total wall-clock ratio: {_ratio_text(metrics.wall_time_ratio)} (report only)",
            "",
            *rows,
            "",
        ]
    )
    markdown_path = output / "report.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path


def load_day_index(path: Path) -> dict[tuple[int, int], IndexedEpisode]:
    path = path.resolve()
    payload = read_json(path)
    if payload.get("schema") != "vaw-day-index-v1":
        raise ValueError(f"不支持的 DAY index schema: {path}")
    episodes: dict[tuple[int, int], IndexedEpisode] = {}
    for item in payload.get("episodes") or ():
        if not isinstance(item, Mapping):
            raise ValueError(f"DAY episode 必须是 object: {path}")
        key = int(item["task_id"]), int(item["seed"])
        if key in episodes:
            raise ValueError(f"DAY index task/seed 重复: {key}")
        status = str(item.get("status") or "")
        success = item.get("env_success")
        if status != "completed" or success is None:
            outcome = EpisodeOutcome.INFRASTRUCTURE
        else:
            outcome = EpisodeOutcome.SUCCESS if bool(success) else EpisodeOutcome.FAILURE
        trace_value = Path(str(item["trace_dir"]))
        trace_dir = (
            trace_value
            if trace_value.is_absolute()
            else _find_experiment_root(path) / trace_value
        )
        episodes[key] = IndexedEpisode(
            task_id=key[0],
            seed=key[1],
            outcome=outcome,
            turns=int(item["turns"]) if item.get("turns") is not None else None,
            tokens=int(item["tokens"]) if item.get("tokens") is not None else None,
            wall_time_s=(
                float(item["wall_time_s"])
                if item.get("wall_time_s") is not None
                else None
            ),
            trace_dir=trace_dir.resolve(),
        )
    return episodes


def _find_experiment_root(path: Path) -> Path:
    """按不可变 experiment manifest 定位根目录，不假设 DAY 产物的嵌套深度。"""

    for parent in path.parents:
        if (parent / "experiment.json").is_file():
            return parent
    raise ValueError(f"DAY index 不属于 evolution experiment: {path}")


def consulted_skill(trace_dir: Path, skill_id: str) -> bool:
    steps = trace_dir / "steps.jsonl"
    if not steps.is_file():
        return False
    for step in read_jsonl(steps):
        call = step.get("function_call")
        if not isinstance(call, Mapping) or call.get("name") != "consult_mmskill":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, Mapping) and arguments.get("skill_id") == skill_id:
            return True
    return False


def _decision(
    pairs: Sequence[GatePair],
    baseline: Mapping[tuple[int, int], IndexedEpisode],
    candidate: Mapping[tuple[int, int], IndexedEpisode],
    mutation: SkillMutation,
    policy: GatePolicy,
    expected_tasks: Sequence[int] | None,
    effects: Mapping[tuple[int, int], SkillEffectReview],
) -> tuple[GateDecision, str]:
    if any(
        pair.baseline is EpisodeOutcome.INFRASTRUCTURE
        or pair.candidate is EpisodeOutcome.INFRASTRUCTURE
        for pair in pairs
    ):
        return GateDecision.INCONCLUSIVE, "存在基础设施失败，不能解释为任务 regression"

    primary = [pair for pair in pairs if pair.phase == "primary"]
    if expected_tasks is not None:
        expected = {
            (int(task), int(seed))
            for task in expected_tasks
            for seed in policy.primary_seeds
        }
        actual = {(pair.task_id, pair.seed) for pair in primary}
        if actual != expected:
            return GateDecision.INCONCLUSIVE, "尚未完成冻结的 Primary Gate task/seed 集合"

    regressions = [pair for pair in primary if pair.is_regression]
    confirmations = [pair for pair in pairs if pair.phase == "confirmation"]
    for regression in regressions:
        if any(
            pair.task_id == regression.task_id and pair.is_regression
            for pair in confirmations
        ):
            return GateDecision.REJECTED, "至少一个 primary regression 在 confirmation seed 再次出现"
    if regressions and not all(
        any(pair.task_id == regression.task_id for pair in confirmations)
        for regression in regressions
    ):
        return GateDecision.INCONCLUSIVE, "primary regression 尚未完成 confirmation 配对"

    primary_keys = {(pair.task_id, pair.seed) for pair in primary}
    if not primary_keys.issubset(effects):
        return GateDecision.INCONCLUSIVE, "独立 Skill Effect Reviewer 尚未覆盖全部 Primary Gate 配对"

    metrics = _metrics(pairs, baseline, candidate, mutation, effects)
    if metrics.local_regressions:
        return GateDecision.REJECTED, "独立视觉审核发现候选技能使目标问题变差"
    if metrics.recoveries < policy.min_recoveries:
        return GateDecision.INCONCLUSIVE, "recovery 数尚未达到冻结门槛"
    if metrics.attributed_recoveries < policy.min_attributed_recoveries:
        return GateDecision.REJECTED, "没有足够 recovery 可归因到目标技能"
    recovered_keys = {(pair.task_id, pair.seed) for pair in primary if pair.is_recovery}
    if not any(
        effects[key].effect is SkillEffect.IMPROVED
        for key in recovered_keys
    ):
        return GateDecision.REJECTED, "终局 recovery 缺少目标技能局部改善的视觉证据"
    if metrics.recoveries <= metrics.regressions:
        return GateDecision.REJECTED, "recovery 没有严格多于 regression"
    if metrics.candidate_successes < metrics.baseline_successes:
        return GateDecision.REJECTED, "candidate 成功总数低于 baseline"
    if metrics.turn_ratio is not None and metrics.turn_ratio > policy.max_turn_ratio:
        return GateDecision.REJECTED, "candidate 平均 turns 超出成本护栏"
    if metrics.token_ratio is not None and metrics.token_ratio > policy.max_token_ratio:
        return GateDecision.REJECTED, "candidate token 成本超出护栏"
    return GateDecision.PASSED, "任务 recovery、归因和成本护栏全部通过"


def _metrics(
    pairs: Sequence[GatePair],
    baseline: Mapping[tuple[int, int], IndexedEpisode],
    candidate: Mapping[tuple[int, int], IndexedEpisode],
    mutation: SkillMutation,
    effects: Mapping[tuple[int, int], SkillEffectReview] | None = None,
) -> GateMetrics:
    primary = [pair for pair in pairs if pair.phase == "primary"]
    recoveries = [pair for pair in primary if pair.is_recovery]
    attributed = sum(
        pair.candidate_consulted
        if mutation.operation in {MutationOperation.ADD, MutationOperation.REVISE}
        else pair.baseline_consulted
        for pair in recoveries
    )
    # 成本只比较双方都成功的 episode。把 recovery 的成功运行与 baseline 的失败运行
    # 相比，会把“恢复能力所需成本”错误解释为退化。
    common_success_keys = [
        (pair.task_id, pair.seed)
        for pair in primary
        if pair.baseline is EpisodeOutcome.SUCCESS
        and pair.candidate is EpisodeOutcome.SUCCESS
    ]
    baseline_turns = [baseline[key].turns for key in common_success_keys]
    candidate_turns = [candidate[key].turns for key in common_success_keys]
    baseline_tokens = [baseline[key].tokens for key in common_success_keys]
    candidate_tokens = [candidate[key].tokens for key in common_success_keys]
    baseline_wall_time = [baseline[key].wall_time_s for key in common_success_keys]
    candidate_wall_time = [candidate[key].wall_time_s for key in common_success_keys]
    primary_keys = {(pair.task_id, pair.seed) for pair in primary}
    effect_values = tuple(
        item for key, item in (effects or {}).items() if key in primary_keys
    )
    return GateMetrics(
        recoveries=len(recoveries),
        regressions=sum(pair.is_regression for pair in primary),
        attributed_recoveries=int(attributed),
        baseline_successes=sum(pair.baseline is EpisodeOutcome.SUCCESS for pair in primary),
        candidate_successes=sum(pair.candidate is EpisodeOutcome.SUCCESS for pair in primary),
        turn_ratio=_mean_ratio(baseline_turns, candidate_turns),
        token_ratio=_sum_ratio(baseline_tokens, candidate_tokens),
        # wall-clock 会受共享服务负载影响，因此只报告，不作为自动拒绝条件。
        wall_time_ratio=_sum_ratio(baseline_wall_time, candidate_wall_time),
        local_improvements=sum(item.effect is SkillEffect.IMPROVED for item in effect_values),
        local_regressions=sum(item.effect is SkillEffect.WORSE for item in effect_values),
        improved_but_failed_elsewhere=sum(
            item.effect is SkillEffect.IMPROVED
            and item.remaining_failure.value == "other"
            for item in effect_values
        ),
    )


def _load_effects(
    value: str | Path | SkillEffectReport | None,
    *,
    mutation: SkillMutation,
    baseline_generation: str,
    candidate_generation: str,
    baseline_day_index: Path,
    candidate_day_index: Path,
) -> dict[tuple[int, int], SkillEffectReview]:
    if value is None:
        return {}
    report = (
        value
        if isinstance(value, SkillEffectReport)
        else SkillEffectReport.from_dict(read_json(Path(value).resolve()))
    )
    identities = (
        report.mutation_id,
        report.baseline_generation,
        report.candidate_generation,
        Path(report.baseline_day_index).resolve(),
        Path(report.candidate_day_index).resolve(),
    )
    expected = (
        mutation.mutation_id,
        baseline_generation,
        candidate_generation,
        baseline_day_index,
        candidate_day_index,
    )
    if identities != expected:
        raise ValueError("Skill Effect Report 不属于当前 mutation 或 paired DAY")
    return {item.key: item for item in report.reviews}


def _effect_text(
    effects: Sequence[SkillEffectReview],
    pair: GatePair,
    field: str,
) -> str:
    review = next(
        (item for item in effects if item.key == (pair.task_id, pair.seed)),
        None,
    )
    if review is None:
        return "not reviewed"
    return str(getattr(review, field).value)


def _mean_ratio(baseline: Sequence[int | None], candidate: Sequence[int | None]) -> float | None:
    if not baseline or any(value is None for value in (*baseline, *candidate)):
        return None
    base = sum(int(value) for value in baseline if value is not None) / len(baseline)
    cand = sum(int(value) for value in candidate if value is not None) / len(candidate)
    return cand / base if base > 0 else None


def _sum_ratio(
    baseline: Sequence[int | float | None],
    candidate: Sequence[int | float | None],
) -> float | None:
    if not baseline or any(value is None for value in (*baseline, *candidate)):
        return None
    base = sum(float(value) for value in baseline if value is not None)
    cand = sum(float(value) for value in candidate if value is not None)
    return cand / base if base > 0 else None


def _ratio_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}×"


__all__ = [
    "GateMetrics",
    "IndexedEpisode",
    "assert_paired_runs",
    "consulted_skill",
    "evaluate_gate",
    "load_day_index",
    "write_gate_report",
]
