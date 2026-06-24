"""在 LIBERO 任务上对 RoboMEx 两层 agent 做**批量评测**(可选任务族 + task 范围)。

两个正交维度:
    - **task**:选哪些任务。``--suite`` 指定任务族(如 ``libero_object_swap``),
      ``--task-ids`` 选族内的 task 索引(``config`` / ``all`` / ``0-4`` / ``0,2,5``)。
    - **trial**:每个 task 跑多少个初始状态。``--trials-per-task`` 控制数量,
      ``--start-trial`` 控制起点(支持断点续跑)。一个 trial =
      ``env.reset(options={"trial": t}, seed=t)``(口径同 ``capx/envs/trial.py``)。

评测口径与 cap0-agent baseline 对齐:每个 episode 的 ``summary.json`` 里的 **env 客观
判据**(LIBERO BDDL goal 检查的 ``env_success`` / ``env_reward`` /
``env_task_completed``)被聚合成 success_rate;同时也报 agent(VLM Verifier)的主观
成功率,用以对照、暴露假阳性。

复用关系:
    - 每个 episode 的执行逻辑复用 ``run_planner_live.run_episode``;
    - task 枚举与 env 用的 ``benchmark.get_benchmark_dict()[suite]().get_task(i)`` 完全
      同源(见 ``capx/integrations/libero/__init__.py::load_libero_task``),索引一致;
    - 切换 task 时按 cap-x ``run_libero_batch`` 的做法 override config 里的
      ``low_level.{suite_name,task_id}`` 重建 env(不改你现有 YAML)。

前置依赖(与 baseline 用的是同一批服务):
    - LLM 代理                          :8110
    - sam3 / contact-graspnet / pyroki :8114 / :8115 / :8116

用法::

    # 单 task(默认:沿用 config 里的 suite_name/task_id),每 task 10 个 trial
    uv run --no-sync --active robomex/examples/run_planner_batch.py \\
        --config-path env_configs/libero/franka_libero_cap_agent0.yaml \\
        --model openrouter/qwen/qwen3.6-plus

    # 整族(libero_object_swap 全部 10 个 task),每 task 5 个 trial
    uv run --no-sync --active robomex/examples/run_planner_batch.py \\
        --suite libero_object_swap --task-ids all --trials-per-task 5

    # 只测 task 0~4,每 task 3 个 trial
    uv run --no-sync --active robomex/examples/run_planner_batch.py \\
        --suite libero_goal_swap --task-ids 0-4 --trials-per-task 3
"""

from __future__ import annotations

import os

# MuJoCo 必须在 import 仿真之前选好 GL 后端(同 run_planner_live)。
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

from robomex.examples.run_planner_live import LiveArgs, _task_language, run_episode


@dataclass
class BatchArgs:
    """批量评测 CLI 参数(task 选择 + trial 调度)。"""

    config_path: str = "env_configs/libero/franka_libero_cap_agent0.yaml"
    """YAML env 配置(其 low_level.suite_name/task_id 作为默认,可被下方参数覆盖)。"""

    model: str = "openrouter/qwen/qwen3.6-plus"
    """planner 和内层 code agent 共用的模型(经代理转发)。"""

    server_url: str = "http://localhost:8110/chat/completions"
    """本地 LLM 代理端点。"""

    api_key: str | None = None
    """可选 API key(通常由代理注入)。"""

    max_turns: int = 6
    """每个 sub-goal 内层最多的代码生成轮数。"""

    output_dir: str = "./outputs/robomex_planner_batch"
    """本次批量评测产物根目录;其下建时间戳子目录,再按 suite/task/trial 分层。"""

    suite: str | None = None
    """任务族名(如 libero_object_swap);留空则沿用 config 里的 suite_name。"""

    task_ids: str = "config"
    """要评测的 task 索引选择:``config``(只用 config 的 task_id)/ ``all`` /
    区间 ``0-4`` / 列表 ``0,2,5``。"""

    trials_per_task: int = 10
    """每个 task 跑多少个 trial(不同初始状态:seed=trial)。"""

    start_trial: int = 1
    """每个 task 的起始 trial 编号(支持断点续跑)。"""


def _to_live_args(args: BatchArgs) -> LiveArgs:
    """把批量参数投影成单 episode 复用的 :class:`LiveArgs`。"""

    return LiveArgs(
        config_path=args.config_path,
        model=args.model,
        server_url=args.server_url,
        api_key=args.api_key,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
    )


def _load_env_config(config_path: str) -> dict[str, Any]:
    """加载并 resolve env config 成纯 dict(便于 override low_level)。"""

    from capx.envs.configs.loader import DictLoader

    configs_dict = DictLoader.load([os.path.expanduser(config_path)])
    if "env" not in configs_dict:
        raise ValueError(f"config {config_path} 缺少 'env' 键")
    return configs_dict


def _low_level_cfg(configs_dict: dict[str, Any]) -> dict[str, Any]:
    """取 low_level 子配置(必须是 inline dict 才能覆盖 suite_name/task_id)。"""

    ll = configs_dict.get("env", {}).get("cfg", {}).get("low_level")
    if not isinstance(ll, dict):
        raise ValueError(
            "env.cfg.low_level 不是 inline dict,无法覆盖 task_id。请改用内联了 "
            "FrankaLiberoEnv 的 config(如 franka_libero_cap_agent0.yaml)。"
        )
    return ll


def _build_env_for_task(config_path: str, suite: str | None, task_id: int) -> Any:
    """重新加载 config、覆盖 suite/task_id 后实例化高层 env(每个 task 调一次)。"""

    from capx.envs.configs.instantiate import instantiate

    configs_dict = _load_env_config(config_path)
    ll = _low_level_cfg(configs_dict)
    if suite is not None:
        ll["suite_name"] = suite
    ll["task_id"] = task_id
    return instantiate(configs_dict["env"])


def _config_suite_and_task(config_path: str) -> tuple[str, int]:
    """读出 config 里默认的 suite_name 与 task_id。"""

    ll = _low_level_cfg(_load_env_config(config_path))
    return str(ll.get("suite_name")), int(ll.get("task_id", 0))


def _suite_task_names(suite: str) -> list[str]:
    """该 suite 内全部 task 的名字(顺序与 env 的 task_id 索引一致)。"""

    from libero import benchmark

    bench = benchmark.get_benchmark_dict()[suite]()
    return [bench.get_task(i).name for i in range(bench.n_tasks)]


def _parse_task_ids(spec: str, config_task_id: int, n_tasks: int) -> list[int]:
    """把 ``--task-ids`` 选择串解析成 task 索引列表。"""

    spec = (spec or "").strip().lower()
    if spec in ("", "config"):
        return [config_task_id]
    if spec == "all":
        return list(range(n_tasks))
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def _safe_name(name: str) -> str:
    """把 task 名压成适合做目录名的短串。"""

    s = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    return s[:48] or "task"


def _close_env(env: Any) -> None:
    """尽力释放底层仿真上下文(env 无统一 close,逐个 try)。"""

    for getter in (
        lambda: env.low_level_env.handle.env.close(),
        lambda: env.low_level_env.close(),
        lambda: env.close(),
    ):
        try:
            getter()
            return
        except Exception:  # noqa: BLE001 - 清理失败不该影响主流程
            continue


def main(args: BatchArgs) -> None:
    from robomex.core.logging import configure_logging
    from robomex.core.session import RoboMExAgent

    batch_root = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    batch_root.mkdir(parents=True, exist_ok=True)

    log = configure_logging(log_file=batch_root / "batch.log")

    config_suite, config_task_id = _config_suite_and_task(args.config_path)
    suite = args.suite or config_suite
    task_names = _suite_task_names(suite)
    n_tasks = len(task_names)
    task_ids = _parse_task_ids(args.task_ids, config_task_id, n_tasks)

    bad = [t for t in task_ids if not (0 <= t < n_tasks)]
    if bad:
        raise ValueError(f"task id {bad} 超出 suite '{suite}' 的范围 [0, {n_tasks - 1}]")

    log.info("批量评测 | suite=%s | task_ids=%s | trials/task=%d (start=%d) | model=%s",
             suite, task_ids, args.trials_per_task, args.start_trial, args.model)
    log.info("MUJOCO_GL=%s | config=%s", os.environ.get("MUJOCO_GL"), args.config_path)

    live_args = _to_live_args(args)
    start = time.time()
    all_records: list[dict[str, Any]] = []
    per_task: list[dict[str, Any]] = []

    for task_id in task_ids:
        task_name = task_names[task_id]
        task_dir = batch_root / suite / f"task_{task_id:02d}_{_safe_name(task_name)}"
        log.info("#" * 80)
        log.info("Task %d/%d | id=%d | %s", task_ids.index(task_id) + 1, len(task_ids),
                 task_id, task_name)

        # 每个 task 重建一个 env(override suite/task_id);跑完释放。
        try:
            env = _build_env_for_task(args.config_path, suite, task_id)
        except Exception as exc:  # noqa: BLE001 - 单 task 建 env 失败不中断整批
            log.exception("Task %d 建 env 失败: %r", task_id, exc)
            per_task.append({"task_id": task_id, "task_name": task_name,
                             "error": repr(exc), **_aggregate([])})
            _dump_batch_summary(batch_root, args, suite, task_ids, all_records, per_task,
                                time.time() - start)
            continue

        task_lang = _task_language(env)
        log.info("任务指令: %s", task_lang)
        task_records: list[dict[str, Any]] = []

        last_trial = args.start_trial + args.trials_per_task - 1
        for trial in range(args.start_trial, last_trial + 1):
            trial_dir = task_dir / f"trial_{trial:03d}"
            log.info("=" * 64)
            log.info("[task %d] Trial %d/%d -> %s", task_id, trial, last_trial, trial_dir)
            try:
                obs, _info = env.reset(options={"trial": trial}, seed=trial)
                result = run_episode(env, obs, task_lang, trial_dir, live_args, log)
                obj = RoboMExAgent._env_objective(result.execution)
                rec = {
                    "suite": suite, "task_id": task_id, "task_name": task_name,
                    "trial": trial, "agent_success": bool(result.success),
                    "n_subgoals": len(result.execution.results), "error": None, **obj,
                }
            except Exception as exc:  # noqa: BLE001 - 单 trial 失败不中断整批
                log.exception("[task %d] Trial %d 异常: %r", task_id, trial, exc)
                rec = {
                    "suite": suite, "task_id": task_id, "task_name": task_name,
                    "trial": trial, "agent_success": False, "env_success": None,
                    "env_task_completed": None, "env_reward": None,
                    "env_terminated": None, "n_subgoals": 0, "error": repr(exc),
                }

            task_records.append(rec)
            all_records.append(rec)
            log.info("[task %d] Trial %d | env_success=%s | agent_success=%s | reward=%s",
                     task_id, trial, rec.get("env_success"), rec.get("agent_success"),
                     rec.get("env_reward"))
            # 每个 trial 后刷新整批汇总(便于中途查看 / 崩溃保留进度)。
            cur_per_task = per_task + [{"task_id": task_id, "task_name": task_name,
                                        "error": None, **_aggregate(task_records)}]
            _dump_batch_summary(batch_root, args, suite, task_ids, all_records,
                                cur_per_task, time.time() - start)

        # 该 task 收尾:写 task_summary、固化进 per_task、释放 env。
        task_summary = {"task_id": task_id, "task_name": task_name, "error": None,
                        **_aggregate(task_records)}
        (task_dir).mkdir(parents=True, exist_ok=True)
        (task_dir / "task_summary.json").write_text(
            json.dumps({**task_summary, "trials": task_records}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        per_task.append(task_summary)
        _close_env(env)
        log.info("[task %d] 完成 | env_success_rate=%.3f (%d/%d) | 出错 %d",
                 task_id, task_summary["env_success_rate"], task_summary["env_success_count"],
                 task_summary["n_trials"], task_summary["n_errored"])

    summary = _dump_batch_summary(batch_root, args, suite, task_ids, all_records, per_task,
                                  time.time() - start)
    ov = summary["overall"]
    log.info("批量评测完成 | %s", batch_root)
    log.info("总计 %d task / %d trial(出错 %d)| env 客观成功率 = %.3f (%d/%d) | agent 主观 = %.3f | 平均 reward = %.3f",
             len(task_ids), ov["n_trials"], ov["n_errored"], ov["env_success_rate"],
             ov["env_success_count"], ov["n_trials"], ov["agent_success_rate"],
             ov["average_reward"])


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """把一组 trial 记录聚合成统计量。

    口径与 cap-x ``_print_and_save_summary`` 对齐:**分母一律是总 trial 数**,出错 / 超时 /
    拿不到 env 信号的 trial 都按失败计、不剔除。``n_errored`` / ``n_signal`` 仅作诊断,不参与
    成功率分母,避免出现"只统计成功的那几个 → 虚高成功率"。
    """

    n_total = len(records)
    n_signal = sum(1 for r in records if r.get("env_success") is not None)
    n_errored = sum(1 for r in records if r.get("error"))
    env_success_count = sum(1 for r in records if r.get("env_success"))
    task_completed_count = sum(1 for r in records if r.get("env_task_completed"))
    # 平均 reward 也按总数算(无信号视作 0),与 cap-x 一致。
    total_reward = sum(float(r["env_reward"]) for r in records if r.get("env_reward") is not None)
    agent_success_count = sum(1 for r in records if r.get("agent_success"))
    return {
        "n_trials": n_total,
        "n_signal": n_signal,
        "n_errored": n_errored,
        "env_success_count": env_success_count,
        "env_success_rate": (env_success_count / n_total) if n_total else 0.0,
        "task_completed_count": task_completed_count,
        "average_reward": (total_reward / n_total) if n_total else 0.0,
        "agent_success_count": agent_success_count,
        "agent_success_rate": (agent_success_count / n_total) if n_total else 0.0,
    }


def _dump_batch_summary(
    batch_root: Path,
    args: BatchArgs,
    suite: str,
    task_ids: list[int],
    all_records: list[dict[str, Any]],
    per_task: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    """写 ``batch_summary.json``:整体聚合 + 每 task 明细 + 全部 trial 扁平记录。"""

    summary = {
        "config_path": args.config_path,
        "model": args.model,
        "suite": suite,
        "task_ids": task_ids,
        "trials_per_task": args.trials_per_task,
        "start_trial": args.start_trial,
        "elapsed_s": round(elapsed, 1),
        "overall": _aggregate(all_records),
        "per_task": per_task,
        "trials": all_records,
    }
    (batch_root / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    main(tyro.cli(BatchArgs))
