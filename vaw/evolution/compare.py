"""Paired phase-level comparison between two sweep summaries."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

from vaw.evolution.fitness import PHASES


def _episodes(summary: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    keyed: dict[tuple[int, int], dict[str, Any]] = {}
    for item in summary.get("results") or []:
        fitness = item.get("fitness")
        if not isinstance(fitness, dict):
            continue
        task_id = fitness.get("task_id")
        seed = fitness.get("seed")
        if task_id is None or seed is None:
            continue
        keyed[(int(task_id), int(seed))] = fitness
    return keyed


def mcnemar(improved: int, regressed: int) -> float:
    """Exact two-sided McNemar p-value for discordant paired bits."""

    n = improved + regressed
    if n == 0:
        return 1.0
    tail = min(improved, regressed)
    cdf = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * cdf)


def compare_summaries(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = _episodes(baseline)
    right = _episodes(candidate)
    shared = sorted(set(left) & set(right))
    phases: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        improved = regressed = unchanged = 0
        flips: list[dict[str, Any]] = []
        for key in shared:
            before = bool(left[key]["phases"].get(phase))
            after = bool(right[key]["phases"].get(phase))
            if before == after:
                unchanged += 1
            elif after and not before:
                improved += 1
                flips.append({"task_id": key[0], "seed": key[1], "direction": "improved"})
            else:
                regressed += 1
                flips.append({"task_id": key[0], "seed": key[1], "direction": "regressed"})
        phases[phase] = {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
            "net": improved - regressed,
            "p_value": mcnemar(improved, regressed),
            "flips": flips,
        }
    return {
        "paired_episodes": len(shared),
        "missing_in_candidate": sorted(set(left) - set(right)),
        "missing_in_baseline": sorted(set(right) - set(left)),
        "phases": phases,
        "accept": all(
            phases[phase]["regressed"] == 0 and phases[phase]["improved"] > 0
            for phase in PHASES
        ),
    }


def seed_phase_variance(summary: dict[str, Any]) -> dict[str, Any]:
    """Per-task phase rates across seeds — the noise floor for later adoption."""

    by_task: dict[int, list[dict[str, Any]]] = {}
    for fitness in _episodes(summary).values():
        by_task.setdefault(int(fitness["task_id"]), []).append(fitness)
    rows = []
    for task_id, items in sorted(by_task.items()):
        row: dict[str, Any] = {"task_id": task_id, "n": len(items)}
        for phase in PHASES:
            bits = [1.0 if item["phases"].get(phase) else 0.0 for item in items]
            mean = sum(bits) / len(bits)
            var = sum((bit - mean) ** 2 for bit in bits) / len(bits)
            row[phase] = {"rate": mean, "variance": var}
        rows.append(row)
    return {"tasks": rows}


def _load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    report = compare_summaries(_load(args.baseline), _load(args.candidate))
    report["baseline_variance"] = seed_phase_variance(_load(args.baseline))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
