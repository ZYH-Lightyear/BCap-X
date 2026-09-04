"""运行不会回流到 Skill Evolver 的 sealed held-out transfer smoke。"""

from __future__ import annotations

import argparse
from pathlib import Path

from vaw.evolution.artifacts import read_json
from vaw.evolution.cycle import CyclePlan, MultiAgentCycle
from vaw.evolution.day import parse_int_list
from vaw.evolution.store import GenerationStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    store = GenerationStore.open(args.experiment_root)
    cycle_root = store.root / "cycles" / args.cycle_id
    plan = CyclePlan.from_dict(read_json(cycle_root / "cycle.json"))
    cycle = MultiAgentCycle(store, plan, resume=True)
    index = cycle.run_transfer(
        suite=args.suite,
        tasks=parse_int_list(args.tasks),
        seeds=parse_int_list(args.seeds),
        generation=args.generation,
        workers=args.workers,
        resume=args.resume,
    )
    print(f"[transfer] index={index}")
    print("[transfer] sealed result: do not feed this index into M3/M4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
