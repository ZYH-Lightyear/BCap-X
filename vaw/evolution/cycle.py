"""把 DAY、M3、M4、配对复测和 Gate 串成可恢复的 Multi-Agent RSI cycle。

本模块只编排已经存在的能力，不替任何 Agent 做语义判断。每个阶段都落盘独立
产物；重新运行时只能显式 ``--resume``，且冻结的 cycle 配置必须完全一致。
候选 generation 始终保持 inactive，本入口不会批准或晋升它。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from vaw.agents.providers.base import ModelProvider
from vaw.diagnostics.run_topreward import LazyQwenTopRewardScorer, score_trace
from vaw.diagnostics.topreward import TopRewardScorer
from vaw.evolution.artifacts import read_json, write_json
from vaw.evolution.candidate import CandidatePackage
from vaw.evolution.day import build_heldout_jobs, build_jobs, parse_int_list, run_day
from vaw.evolution.effect_reviewer import run_skill_effect_reviews
from vaw.evolution.gate import evaluate_gate, load_day_index, write_gate_report
from vaw.evolution.m3 import run_m3
from vaw.evolution.skill_agent import run_skill_evolver
from vaw.evolution.store import GenerationStore

CYCLE_SCHEMA = "vaw-multi-agent-rsi-cycle-v1"
PROGRESS_INDEX_SCHEMA = "vaw-cycle-progress-index-v1"
M3_INDEX_SCHEMA = "vaw-cycle-m3-index-v1"


class CycleStage(StrEnum):
    """可作为 ``--until`` 使用的有序工程边界。"""

    BASELINE = "baseline"
    PROGRESS = "progress"
    INVESTIGATE = "investigate"
    EVOLVE = "evolve"
    MATERIALIZE = "materialize"
    CANDIDATE = "candidate"
    EFFECTS = "effects"
    GATE = "gate"


@dataclass(frozen=True)
class CyclePlan:
    """一次小规模进化 cycle 的最小冻结身份。"""

    cycle_id: str
    parent_generation: str
    candidate_generation: str
    mutation_id: str
    tasks: tuple[int, ...]
    seeds: tuple[int, ...]
    schema: str = CYCLE_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("cycle_id", self.cycle_id),
            ("parent_generation", self.parent_generation),
            ("candidate_generation", self.candidate_generation),
            ("mutation_id", self.mutation_id),
        ):
            if not value or Path(value).name != value or value in {".", ".."}:
                raise ValueError(f"{name} 必须是单段名称")
        if self.parent_generation == self.candidate_generation:
            raise ValueError("parent 与 candidate generation 不能相同")
        if not self.tasks or not self.seeds:
            raise ValueError("tasks 和 seeds 不能为空")
        if len(set(self.tasks)) != len(self.tasks) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("tasks 和 seeds 不能重复")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = list(self.tasks)
        payload["seeds"] = list(self.seeds)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CyclePlan:
        if payload.get("schema") != CYCLE_SCHEMA:
            raise ValueError(f"不支持的 cycle schema: {payload.get('schema')!r}")
        return cls(
            cycle_id=str(payload["cycle_id"]),
            parent_generation=str(payload["parent_generation"]),
            candidate_generation=str(payload["candidate_generation"]),
            mutation_id=str(payload["mutation_id"]),
            tasks=tuple(int(item) for item in payload["tasks"]),
            seeds=tuple(int(item) for item in payload["seeds"]),
        )


@dataclass(frozen=True)
class CyclePaths:
    """集中定义产物路径，避免各 Agent 互相猜目录。"""

    root: Path

    @property
    def baseline(self) -> Path:
        return self.root / "day" / "baseline"

    @property
    def progress(self) -> Path:
        return self.root / "progress"

    @property
    def m3(self) -> Path:
        return self.root / "m3"

    @property
    def m4(self) -> Path:
        return self.root / "m4"

    @property
    def candidate_package(self) -> Path:
        return self.root / "candidate"

    @property
    def candidate_day(self) -> Path:
        return self.root / "day" / "candidate"

    @property
    def effects(self) -> Path:
        return self.root / "effects"

    @property
    def gate(self) -> Path:
        return self.root / "gate"


class MultiAgentCycle:
    """一个显式、可恢复但不会自动晋升的 RSI 编排器。"""

    def __init__(self, store: GenerationStore, plan: CyclePlan, *, resume: bool) -> None:
        self.store = store
        self.plan = plan
        self.paths = CyclePaths(store.root / "cycles" / plan.cycle_id)
        manifest = self.paths.root / "cycle.json"
        if manifest.is_file():
            existing = CyclePlan.from_dict(read_json(manifest))
            if existing != plan:
                raise ValueError("--resume 的 CyclePlan 与已落盘配置不一致")
            if not resume:
                raise FileExistsError(f"cycle 已存在，请使用 --resume: {self.paths.root}")
        else:
            if self.paths.root.exists() and any(self.paths.root.iterdir()):
                raise ValueError("非空 cycle 目录缺少 cycle.json")
            self.paths.root.mkdir(parents=True, exist_ok=True)
            write_json(manifest, plan.to_dict())
        self.resume = resume

    def run_baseline(self, *, workers: int, episode_runner: Any = None) -> Path:
        jobs = build_jobs(
            store=self.store,
            run_root=self.paths.baseline,
            split="evolve",
            tasks=self.plan.tasks,
            seeds=self.plan.seeds,
            generation_id=self.plan.parent_generation,
        )
        options: dict[str, Any] = {"resume": self.resume, "workers": workers}
        if episode_runner is not None:
            options["episode_runner"] = episode_runner
        run_day(jobs, **options)
        return self.paths.baseline / "day_index.json"

    def run_progress(
        self,
        scorer: TopRewardScorer,
        *,
        max_prefix_frames: int,
    ) -> Path:
        """为所有可学习终局生成导航曲线；基础设施失败留给 Gate 处理。"""

        episodes = load_day_index(self.paths.baseline / "day_index.json")
        entries: list[dict[str, Any]] = []
        for (task_id, seed), episode in sorted(episodes.items()):
            if episode.outcome.value == "infrastructure":
                continue
            output = self.paths.progress / f"task{task_id}_s{seed}"
            progress = output / "progress.json"
            if not (self.resume and progress.is_file()):
                progress = score_trace(
                    trace_dir=episode.trace_dir,
                    output_dir=output,
                    scorer=scorer,
                    max_prefix_frames=max_prefix_frames,
                    resume=self.resume,
                )
            entries.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "trace": str(episode.trace_dir),
                    "progress": str(progress.resolve()),
                }
            )
            self._write_index(
                self.paths.progress / "index.json",
                PROGRESS_INDEX_SCHEMA,
                entries,
            )
        if not entries:
            raise RuntimeError("baseline 没有可供 M3 调查的完整 episode")
        return self.paths.progress / "index.json"

    def run_investigation(
        self,
        *,
        investigator_provider: ModelProvider,
        reviewer_provider: ModelProvider,
        max_inspections: int,
        run_metadata: dict[str, Any],
    ) -> Path:
        progress = self._read_entries(self.paths.progress / "index.json", PROGRESS_INDEX_SCHEMA)
        entries: list[dict[str, Any]] = []
        for item in progress:
            name = f"task{item['task_id']}_s{item['seed']}"
            output = self.paths.m3 / name
            result = output / "m3.json"
            if not (self.resume and result.is_file()):
                if output.exists():
                    if not self.resume:
                        raise FileExistsError(f"M3 输出已存在: {output}")
                    self._archive_incomplete(output)
                result = run_m3(
                    trace_dir=item["trace"],
                    progress_path=item["progress"],
                    output_dir=output,
                    investigator_provider=investigator_provider,
                    reviewer_provider=reviewer_provider,
                    max_inspections=max_inspections,
                    run_metadata=run_metadata,
                )
            summary = read_json(result)
            entries.append(
                {
                    "task_id": int(item["task_id"]),
                    "seed": int(item["seed"]),
                    "m3": str(result.resolve()),
                    "accepted": int(summary["accepted"]),
                }
            )
            self._write_index(self.paths.m3 / "index.json", M3_INDEX_SCHEMA, entries)
        return self.paths.m3 / "index.json"

    def run_evolution(
        self,
        *,
        evolver_provider: ModelProvider,
        reviewer_provider: ModelProvider,
        max_inspections: int,
        run_metadata: dict[str, Any],
    ) -> Path:
        result = self.paths.m4 / "m4.json"
        if self.resume and result.is_file():
            return result
        entries = self._read_entries(self.paths.m3 / "index.json", M3_INDEX_SCHEMA)
        return run_skill_evolver(
            store=self.store,
            parent_generation=self.plan.parent_generation,
            mutation_id=self.plan.mutation_id,
            m3_outputs=tuple(item["m3"] for item in entries),
            output_dir=self.paths.m4,
            candidate_dir=self.paths.candidate_package,
            evolver_provider=evolver_provider,
            reviewer_provider=reviewer_provider,
            max_inspections=max_inspections,
            run_metadata=run_metadata,
        )

    def materialize(self) -> bool:
        """返回是否形成候选代；NO_CHANGE/REJECTED 是正常终点。"""

        result = read_json(self.paths.m4 / "m4.json")
        if result.get("status") != "candidate":
            return False
        generation_path = self.store.generation_path(self.plan.candidate_generation)
        if generation_path.is_dir():
            manifest = self.store.read_manifest(self.plan.candidate_generation)
            if manifest.parent_generation != self.plan.parent_generation:
                raise ValueError("已有 candidate generation 的 parent 不匹配")
            if manifest.mutation is None or manifest.mutation.mutation_id != self.plan.mutation_id:
                raise ValueError("已有 candidate generation 的 mutation 不匹配")
            return True
        self.store.create_generation(
            self.plan.candidate_generation,
            parent_generation=self.plan.parent_generation,
            candidate=CandidatePackage.open(self.paths.candidate_package),
        )
        return True

    def run_candidate(self, *, workers: int, episode_runner: Any = None) -> Path:
        jobs = build_jobs(
            store=self.store,
            run_root=self.paths.candidate_day,
            split="evolve",
            tasks=self.plan.tasks,
            seeds=self.plan.seeds,
            generation_id=self.plan.candidate_generation,
        )
        options: dict[str, Any] = {"resume": self.resume, "workers": workers}
        if episode_runner is not None:
            options["episode_runner"] = episode_runner
        run_day(jobs, **options)
        return self.paths.candidate_day / "day_index.json"

    def run_effect_review(
        self,
        *,
        provider: ModelProvider,
        max_inspections: int,
    ) -> Path:
        report = self.paths.effects / "effect_report.json"
        if self.resume and report.is_file():
            return report
        return run_skill_effect_reviews(
            store=self.store,
            candidate_generation=self.plan.candidate_generation,
            baseline_day_index=self.paths.baseline / "day_index.json",
            candidate_day_index=self.paths.candidate_day / "day_index.json",
            output_dir=self.paths.effects,
            provider=provider,
            max_inspections=max_inspections,
            resume=self.resume,
        )

    def run_gate(self) -> Path:
        report_path = self.paths.gate / "report.json"
        if self.resume and report_path.is_file():
            return report_path
        manifest = self.store.read_manifest(self.plan.candidate_generation)
        if manifest.mutation is None:
            raise ValueError("candidate generation 缺少 mutation")
        report, metrics = evaluate_gate(
            mutation=manifest.mutation,
            baseline_generation=self.plan.parent_generation,
            candidate_generation=self.plan.candidate_generation,
            baseline_day_index=self.paths.baseline / "day_index.json",
            candidate_day_index=self.paths.candidate_day / "day_index.json",
            policy=self.store.read_spec().gate_policy,
            expected_tasks=None,
            skill_effect_report=self.paths.effects / "effect_report.json",
        )
        report_path, _ = write_gate_report(self.paths.gate, report, metrics)
        self.store.ledger.append(
            "generation_gated",
            generation_id=self.plan.candidate_generation,
            payload={"cycle_id": self.plan.cycle_id, "report": str(report_path), "decision": report.decision.value},
        )
        return report_path

    def run_transfer(
        self,
        *,
        suite: str,
        tasks: tuple[int, ...],
        seeds: tuple[int, ...],
        generation: str,
        workers: int,
        resume: bool,
    ) -> Path:
        """运行 sealed transfer smoke；其产物不进入 M3/M4。"""

        root = self.paths.root / "transfer" / suite / generation
        jobs = build_heldout_jobs(
            store=self.store,
            run_root=root,
            suite=suite,
            tasks=tasks,
            seeds=seeds,
            generation_id=generation,
        )
        run_day(jobs, resume=resume, workers=workers)
        return root / "day_index.json"

    @staticmethod
    def _write_index(path: Path, schema: str, entries: list[dict[str, Any]]) -> None:
        write_json(path, {"schema": schema, "episodes": entries})

    @staticmethod
    def _read_entries(path: Path, schema: str) -> list[dict[str, Any]]:
        payload = read_json(path)
        if payload.get("schema") != schema:
            raise ValueError(f"不支持的 index schema: {path}")
        entries = payload.get("episodes")
        if not isinstance(entries, list):
            raise ValueError(f"index 缺少 episodes: {path}")
        return [dict(item) for item in entries]

    @staticmethod
    def _archive_incomplete(path: Path) -> Path:
        """保留异常中断的 Agent 产物，再从干净目录恢复。"""

        number = 1
        while True:
            suffix = ".incomplete" if number == 1 else f".incomplete-{number}"
            archive = path.with_name(path.name + suffix)
            if not archive.exists():
                path.replace(archive)
                return archive
            number += 1

    def status(self) -> dict[str, Any]:
        """状态完全由不可变产物推导，不维护第二份流程真值。"""

        checks = {
            "baseline": self.paths.baseline / "day_index.json",
            "progress": self.paths.progress / "index.json",
            "investigate": self.paths.m3 / "index.json",
            "evolve": self.paths.m4 / "m4.json",
            "materialize": self.store.generation_path(self.plan.candidate_generation) / "manifest.json",
            "candidate": self.paths.candidate_day / "day_index.json",
            "effects": self.paths.effects / "effect_report.json",
            "gate": self.paths.gate / "report.json",
        }
        return {
            "schema": CYCLE_SCHEMA,
            "cycle_id": self.plan.cycle_id,
            "active_generation": self.store.active_generation(),
            "stages": {name: path.is_file() for name, path in checks.items()},
            "artifacts": {name: str(path) for name, path in checks.items() if path.is_file()},
        }


def _provider(args: argparse.Namespace, model: str) -> ModelProvider:
    from vaw.agents.providers.openai import OpenAIProvider
    from vaw.agents.providers.text_protocol import TextProtocolProvider

    provider: ModelProvider = OpenAIProvider(
        model=model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
    )
    return TextProtocolProvider(provider) if args.protocol == "text" else provider


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--parent-generation")
    parser.add_argument("--candidate-generation", required=True)
    parser.add_argument("--mutation-id", required=True)
    parser.add_argument("--until", choices=tuple(CycleStage), default=CycleStage.GATE.value)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", required=True)
    parser.add_argument("--review-model")
    parser.add_argument("--effect-model")
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--api-key", default=os.environ.get("V_API_KEY"))
    parser.add_argument("--protocol", choices=("native", "text"), default="native")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--m3-inspections", type=int, default=4)
    parser.add_argument("--m4-inspections", type=int, default=6)
    parser.add_argument("--effect-inspections", type=int, default=4)
    parser.add_argument("--topreward-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--topreward-device", default="cuda:0")
    parser.add_argument("--topreward-dtype", default="bfloat16")
    parser.add_argument("--topreward-max-prefix-frames", type=int, default=15)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    store = GenerationStore.open(args.experiment_root)
    parent = args.parent_generation or store.active_generation()
    plan = CyclePlan(
        cycle_id=args.cycle_id,
        parent_generation=parent,
        candidate_generation=args.candidate_generation,
        mutation_id=args.mutation_id,
        tasks=parse_int_list(args.tasks),
        seeds=parse_int_list(args.seeds),
    )
    cycle = MultiAgentCycle(store, plan, resume=args.resume)
    order = list(CycleStage)
    stop = order.index(CycleStage(args.until))
    reviewer_model = args.review_model or args.model
    metadata = {
        "model": args.model,
        "review_model": reviewer_model,
        "server_url": args.server_url,
        "protocol": args.protocol,
    }

    cycle.run_baseline(workers=args.workers)
    if stop >= order.index(CycleStage.PROGRESS):
        scorer = LazyQwenTopRewardScorer(
            model_name=args.topreward_model,
            device=args.topreward_device,
            dtype=args.topreward_dtype,
            attention_implementation="sdpa",
            fps=2.0,
            local_files_only=args.local_files_only,
        )
        cycle.run_progress(scorer, max_prefix_frames=args.topreward_max_prefix_frames)
    if stop >= order.index(CycleStage.INVESTIGATE):
        cycle.run_investigation(
            investigator_provider=_provider(args, args.model),
            reviewer_provider=_provider(args, reviewer_model),
            max_inspections=args.m3_inspections,
            run_metadata={"schema": "vaw-m3-run-v1", **metadata},
        )
    if stop >= order.index(CycleStage.EVOLVE):
        result = cycle.run_evolution(
            evolver_provider=_provider(args, args.model),
            reviewer_provider=_provider(args, reviewer_model),
            max_inspections=args.m4_inspections,
            run_metadata={"schema": "vaw-skill-evolver-run-v1", **metadata},
        )
        if read_json(result).get("status") != "candidate":
            print(f"[cycle] stopped without candidate: {result}")
            print(cycle.status())
            return 0
    if stop >= order.index(CycleStage.MATERIALIZE):
        cycle.materialize()
    if stop >= order.index(CycleStage.CANDIDATE):
        cycle.run_candidate(workers=args.workers)
    if stop >= order.index(CycleStage.EFFECTS):
        cycle.run_effect_review(
            provider=_provider(args, args.effect_model or reviewer_model),
            max_inspections=args.effect_inspections,
        )
    if stop >= order.index(CycleStage.GATE):
        cycle.run_gate()
    print(cycle.status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CyclePlan", "CycleStage", "MultiAgentCycle"]
