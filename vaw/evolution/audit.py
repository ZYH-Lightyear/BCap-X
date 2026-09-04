"""晋升后用配对事实确认技能库是否发生可重复退化。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vaw.evolution.artifacts import write_json
from vaw.evolution.domain import EpisodeOutcome
from vaw.evolution.gate import assert_paired_runs, load_day_index
from vaw.evolution.store import GenerationStore


def audit_generation(
    *,
    store: GenerationStore,
    generation_id: str,
    parent_day_index: str | Path,
    current_day_index: str | Path,
    output_path: str | Path,
) -> Path:
    """只有同一 task 在 primary 与 confirmation 都退化时才允许自动回滚。"""

    manifest = store.read_manifest(generation_id)
    parent = manifest.parent_generation
    if parent is None:
        raise ValueError("base generation 不需要 post-promotion audit")
    policy = store.read_spec().gate_policy
    parent_path = Path(parent_day_index).resolve()
    current_path = Path(current_day_index).resolve()
    assert_paired_runs(
        parent_path,
        current_path,
        baseline_generation=parent,
        candidate_generation=generation_id,
    )
    baseline = load_day_index(parent_path)
    current = load_day_index(current_path)
    common = set(baseline) & set(current)
    infrastructure = [
        key
        for key in common
        if baseline[key].outcome is EpisodeOutcome.INFRASTRUCTURE
        or current[key].outcome is EpisodeOutcome.INFRASTRUCTURE
    ]
    primary_suspects = sorted(
        task_id
        for task_id, seed in common
        if seed in policy.primary_seeds
        and baseline[task_id, seed].outcome is EpisodeOutcome.SUCCESS
        and current[task_id, seed].outcome is EpisodeOutcome.FAILURE
    )
    confirmed = sorted(
        task_id
        for task_id in set(primary_suspects)
        if any(
            (task_id, seed) in common
            and baseline[task_id, seed].outcome is EpisodeOutcome.SUCCESS
            and current[task_id, seed].outcome is EpisodeOutcome.FAILURE
            for seed in policy.confirmation_seeds
        )
    )
    action = "none"
    status = "healthy"
    if infrastructure:
        status = "inconclusive"
    elif confirmed:
        status = "confirmed_degradation"
        if policy.auto_rollback:
            if store.active_generation() != generation_id:
                raise ValueError("自动回滚只允许作用于当前 active generation")
            store.rollback(parent)
            action = f"rollback:{parent}"
    elif primary_suspects:
        status = "suspect"
    payload: dict[str, Any] = {
        "schema": "vaw-post-promotion-audit-v1",
        "generation": generation_id,
        "parent_generation": parent,
        "status": status,
        "suspect_tasks": sorted(set(primary_suspects)),
        "confirmed_tasks": confirmed,
        "action": action,
        "parent_day_index": str(Path(parent_day_index).resolve()),
        "current_day_index": str(Path(current_day_index).resolve()),
    }
    path = write_json(output_path, payload)
    store.ledger.append(
        "generation_audited",
        generation_id=generation_id,
        payload={"report": str(path), "status": status, "action": action},
    )
    return path


__all__ = ["audit_generation"]
