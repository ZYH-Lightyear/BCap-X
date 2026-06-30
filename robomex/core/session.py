"""RoboMEx 框架入口:依赖容器 + 顶层 agent。

这里是把整个框架接线的唯一地方(对应 qwen-code 的 ``Config`` + runtime 拆分):
:class:`RoboMExConfig` 收集所有可替换依赖(技能库、两个策略、沙箱后端、可选的
collector),:class:`RoboMExAgent` 据此装配出反应式两层循环并跑一整段
episode。入口(``examples/``、评测脚手架)应构造一个 config 再调用
:meth:`RoboMExAgent.run`,而不是手工接线 planner 和 executor。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robomex.agents.executor import CodeAsPolicyAgent
from robomex.agents.planner import (
    PlanExecution,
    PlannerPolicy,
    ReactivePlanner,
    SubGoal,
    SubGoalResult,
)
from robomex.core.coder.policy import CompletionPolicy
from robomex.core.coder.trace import AgentTrace
from robomex.core.events import emit_event, event_log, event_scope
from robomex.core.logging import get_logger
from robomex.skills import SkillLibrary

_log = get_logger("session")


@dataclass
class RoboMExConfig:
    """一次 RoboMEx 运行的所有可替换依赖。

    循环需要的一切都在这里注入,因此离线(脚本)和真机(LLM + CapX env)运行的
    区别只在于 config 里放了什么:

    - ``library``         —— 技能库(内置 + 学到的技能包)。
    - ``planner_policy``  —— 驱动外层 :class:`ReactivePlanner`。
    - ``code_policy``     —— 驱动内层 :class:`CodeAsPolicyAgent`。
    - ``executor``        —— 沙箱后端(``run_block``);真机用
      ``CapXExecutorAdapter(env)``,离线用 mock。
    - ``collector`` —— 可选的逐块证据采集(存 before/after 帧调试产物)。
    - ``inner_system_prompt`` —— 覆盖执行器的角色提示词(例如注入真机 API 文档);
      ``None`` 则保持执行器默认。
    - ``artifacts_dir`` —— 可选;给定后,每个 sub-goal 的 planner 决策、内层每轮代码 /
      输出、过程视频,以及一份 episode 汇总都会落盘到此目录。``None`` 则不落盘。
    """

    library: SkillLibrary
    planner_policy: PlannerPolicy
    code_policy: CompletionPolicy
    executor: Any
    collector: Any | None = None
    # 内层每个 sub-goal 最多执行多少个 python action block;use_skill 等元动作不计入。
    max_turns: int = 6
    max_subgoals: int = 8
    # 兼容旧配置项:主路径已改为 Act-only;subgoal 未 finish 时把 unresolved 报告交回
    # Planner,不再由 session 自动重跑同一 subgoal。
    subgoal_max_attempts: int = 3
    inner_system_prompt: str | None = None
    observation_summary: str = ""
    artifacts_dir: str | None = None


@dataclass(frozen=True)
class EpisodeResult:
    """一次 :meth:`RoboMExAgent.run` episode 的结果。"""

    execution: PlanExecution

    @property
    def success(self) -> bool:
        return self.execution.success


class RoboMExAgent:
    """顶层 agent:高层技能上的反应式 planner + 内层 coder。

    每一步,planner 参考高层技能指导给出下一个自然语言 sub-goal(依据任务、当前场景、历史);
    内层执行器自主选择并组合技能执行它;随后可选地从最新观测刷新场景;
    如此循环,直到 planner 说 DONE 或触达 sub-goal 上限。该循环在离线和真机下完全
    一致——唯一差别是注入的场景刷新(它与具体 env 相关)。
    """

    def __init__(self, config: RoboMExConfig) -> None:
        self.config = config
        self.planner = ReactivePlanner(config.library, config.planner_policy)

        kwargs: dict[str, Any] = {
            "executor": config.executor,
            "policy": config.code_policy,
            "library": config.library,
            "max_turns": config.max_turns,
        }
        if config.collector is not None:
            kwargs["collector"] = config.collector
        if config.inner_system_prompt is not None:
            kwargs["system_prompt"] = config.inner_system_prompt
        self.executor_agent = CodeAsPolicyAgent(**kwargs)

    def run(
        self,
        task: str,
        scene_image_path: str | None = None,
        scene_refresh: Callable[[dict], str | None] | None = None,
        on_subgoal_end: Callable[[int, SubGoalResult, Path | None], None] | None = None,
    ) -> EpisodeResult:
        """为 ``task`` 跑一整段反应式 episode。

        ``scene_refresh``(真机用)把最新观测映射成 planner 下一步要看的新场景图路径;
        离线时保持 ``None``,场景固定不变。

        ``on_subgoal_end``(真机用)在**每个 sub-goal 跑完并落盘后**立即触发,入参为
        ``(index, SubGoalResult, 该 sub-goal 的产物目录)``——真机入口借此把这段的视频
        当场写进对应 ``subgoal_NN/``,而不是等整段 episode 结束再统一存。
        """

        art = Path(self.config.artifacts_dir) if self.config.artifacts_dir else None
        if art is not None:
            art.mkdir(parents=True, exist_ok=True)

        event_path = (art / "events.jsonl") if art is not None else None
        with event_log(event_path), event_scope(task=task, artifacts_dir=str(art) if art else None):
            menu = [r.skill_id for r in self.config.library.compound_skills()]
            _log.info("episode 开始 | task=%r | 高层技能菜单=%s | max_subgoals=%d",
                      task, menu, self.config.max_subgoals)
            emit_event(
                "episode_start",
                "RoboMEx episode started",
                scene_image_path=scene_image_path,
                high_level_skills=menu,
                max_subgoals=self.config.max_subgoals,
            )

            results: list[SubGoalResult] = []
            cur_scene = scene_image_path
            for i in range(self.config.max_subgoals):
                with event_scope(subgoal_index=i, subgoal_number=i + 1):
                    subgoal = self.planner.next_subgoal(task, results, scene_image_path=cur_scene)
                    self._record_planner(art, i, subgoal)
                    if subgoal is None:
                        _log.info("[subgoal %d] planner → DONE(没有下一个 sub-goal,结束)", i + 1)
                        emit_event("planner_done", "Planner returned DONE")
                        break
                    _log.info("[subgoal %d] planner → goal=%r | planning_hint=%s | 成功条件=%r",
                              i + 1, subgoal.goal, subgoal.skill, subgoal.postcondition)
                    emit_event(
                        "subgoal_start",
                        f"Subgoal {i + 1}: {subgoal.goal}",
                        goal=subgoal.goal,
                        planning_hint=subgoal.skill,
                        postcondition=subgoal.postcondition,
                        scene_image_path=cur_scene,
                    )

                    # 子目标产物目录提前建好:执行器把每个有动作的 code block 的过程视频当场写进
                    # 这里(turn_NN.mp4),供日志、调试和 Planner 复盘使用。
                    sg_dir = (art / f"subgoal_{i:02d}") if art is not None else None
                    if sg_dir is not None:
                        sg_dir.mkdir(parents=True, exist_ok=True)

                    # Act inner loop:执行器自主 use_skill/run_python,finish 表示当前 sub-goal
                    # 尝试结束,随后刷新场景并交回 Planner 规划下一步。
                    with event_scope(subgoal_dir=str(sg_dir) if sg_dir else None):
                        trace, vtrace, success, note = self._run_subgoal(subgoal, sg_dir, cur_scene)
                    results.append(SubGoalResult(subgoal=subgoal, trace=trace, success=success, note=note))
                    trace_meta = trace.metadata or {}
                    act_status = trace_meta.get("act_status", "act")
                    _log.info("[subgoal %d] 结果 success=%s | act=%s | 内层轮数=%d | 加载技能=%s",
                              i + 1, success, act_status,
                              len(trace.turns), list(trace.loaded_skill_ids))
                    emit_event(
                        "subgoal_end",
                        f"Subgoal {i + 1} ended",
                        goal=subgoal.goal,
                        success=success,
                        act_status=act_status,
                        inner_turns=len(trace.turns),
                        loaded_skill_ids=list(trace.loaded_skill_ids),
                        note=note,
                    )

                    # sub-goal 跑完立即触发回调(真机:把这段视频当场写进 subgoal 目录)。
                    if on_subgoal_end is not None:
                        on_subgoal_end(i, results[-1], sg_dir)

                    # 真机:用最新观测刷新 planner 下一步看到的场景图;离线则跳过。
                    if scene_refresh is not None and trace.turns and trace.turns[-1].execution.observation:
                        refreshed = scene_refresh(trace.turns[-1].execution.observation)
                        cur_scene = refreshed or cur_scene
                        emit_event("scene_refreshed", "Planner scene image refreshed", scene_image_path=cur_scene)

            success = bool(results) and all(r.success for r in results)
            n_ok = sum(r.success for r in results)
            _log.info("episode 结束 | success=%s | %d/%d 个 sub-goal 成功", success, n_ok, len(results))

            execution = PlanExecution(
                task=task,
                subgoals=tuple(r.subgoal for r in results),
                results=tuple(results),
                success=success,
            )
            self._dump_summary(art, execution)
            emit_event(
                "episode_end",
                "RoboMEx episode ended",
                success=success,
                successful_subgoals=n_ok,
                total_subgoals=len(results),
            )
            if art is not None:
                _log.info("产物已落盘到: %s", art)
                _log.info("  ├─ events.jsonl     结构化调试事件(Web UI/离线分析)")
            return EpisodeResult(execution=execution)

    # ---- Act inner loop ----------------------------------------------------

    def _run_subgoal(self, subgoal: SubGoal, sg_dir: Path | None, scene_image_path: str | None = None):
        """在一个 sub-goal 内跑 Act inner loop。

        ``finish`` 结束当前 Act 尝试并交回 Planner;若 inner loop 耗尽,把 unresolved
        报告作为 note。
        """

        trace = self.executor_agent.run(
            subgoal.goal,
            self.config.observation_summary,
            expected_postcondition=subgoal.postcondition,
            video_dir=sg_dir,
            scene_image_path=scene_image_path,
        )
        success = trace.success
        meta = trace.metadata or {}
        unresolved = meta.get("unresolved") if isinstance(meta, dict) else None
        if isinstance(unresolved, dict):
            note = unresolved.get("last_state_summary", "")
        else:
            note = ""
        self._dump_subgoal(sg_dir, subgoal, trace, None, attempt=0)
        return trace, None, success, note

    # ---- 产物落盘(artifacts_dir 给定时启用) ------------------------------

    def _record_planner(self, art: Path | None, index: int, subgoal: SubGoal | None) -> None:
        """把一次 planner 决策(原始回复 + 解析结果)追加进 ``planner.jsonl``。"""

        if art is None:
            return
        entry: dict[str, Any] = {"index": index, "raw": getattr(self.planner, "last_raw", "")}
        if subgoal is None:
            entry["decision"] = "DONE"
        else:
            entry.update(
                goal=subgoal.goal,
                planning_hint=subgoal.skill,
                postcondition=subgoal.postcondition,
            )
        with (art / "planner.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _dump_subgoal(d: Path | None, subgoal: SubGoal, trace: AgentTrace, vtrace=None, attempt: int = 0) -> None:
        """把一次 sub-goal 尝试的元信息 + 内层每轮代码/输出写到目录 ``d``。

        ``d`` 由调用方给定:首次尝试是 ``subgoal_NN/``,重试是 ``subgoal_NN/retry_MM/``。
        """

        if d is None:
            return
        d.mkdir(parents=True, exist_ok=True)
        verdict = vtrace.verdict if vtrace is not None else None
        trace_meta = trace.metadata or {}
        meta = {
            "goal": subgoal.goal,
            "planning_hint": subgoal.skill,
            "postcondition": subgoal.postcondition,
            "attempt": attempt,
            "act_success": trace.success,
            "act_status": trace_meta.get("act_status"),
            "loaded_skill_ids": list(trace.loaded_skill_ids),
            "unresolved": trace_meta.get("unresolved"),
            "verdict": (
                {"verdict": verdict.verdict, "confidence": verdict.confidence, "reason": verdict.reason}
                if verdict is not None
                else None
            ),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        for t in trace.turns:
            (d / f"turn_{t.turn:02d}.py").write_text(t.code, encoding="utf-8")
            report = "\n".join([
                f"# status    : {t.execution.status.value}",
                f"# ok        : {t.execution.ok}",
                f"# reward    : {t.execution.reward}",
                f"# terminated: {t.execution.terminated}",
                "",
                "## stdout",
                t.execution.stdout or "(empty)",
                "",
                "## stderr",
                t.execution.stderr or "(empty)",
            ])
            (d / f"turn_{t.turn:02d}.out.txt").write_text(report, encoding="utf-8")
        if vtrace is not None:
            lines = [
                f"# verdict   : {verdict.verdict}",
                f"# confidence: {verdict.confidence}",
                f"# reason    : {verdict.reason}",
                "",
            ]
            for vt in vtrace.turns:
                lines += [
                    f"## verify turn {vt.turn} code",
                    vt.code or "(none)",
                    "## stdout",
                    vt.stdout or "(empty)",
                    "## stderr",
                    vt.stderr or "(empty)",
                    "",
                ]
            (d / "verify.txt").write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _env_objective(execution: PlanExecution) -> dict[str, Any]:
        """从所有 sub-goal 的执行轨迹里抽 env 的客观判据(LIBERO 的 BDDL goal 检查)。

        reward / terminated / task_completed 是整段任务级信号,取最后一次非空值即为终态。
        ``env_success`` 与 cap0-agent 的口径对齐:优先看 ``task_completed``,否则看
        ``reward == 1.0``;两者都拿不到则为 ``None``(env 不支持)。
        """

        reward = terminated = task_completed = None
        for r in execution.results:
            for t in r.trace.turns:
                ex = t.execution
                if ex.reward is not None:
                    reward = ex.reward
                if ex.terminated is not None:
                    terminated = ex.terminated
                tc = (ex.info or {}).get("task_completed")
                if tc is not None:
                    task_completed = tc
        if task_completed is not None:
            env_success: bool | None = bool(task_completed)
        elif reward is not None:
            env_success = float(reward) == 1.0
        else:
            env_success = None
        # 关键:LIBERO 的 task_completed/terminated 是 numpy.bool_、reward 是 numpy.float,
        # 直接进 json.dumps 会抛 "Object of type bool_ is not JSON serializable"。统一转
        # 成原生 Python 类型,避免落盘时崩掉整段 episode。
        return {
            "env_success": env_success,
            "env_task_completed": None if task_completed is None else bool(task_completed),
            "env_reward": None if reward is None else float(reward),
            "env_terminated": None if terminated is None else bool(terminated),
        }

    @staticmethod
    def _dump_summary(art: Path | None, execution: PlanExecution) -> None:
        """把整段 episode 的汇总写到 ``summary.json``。

        ``success`` 表示 Planner/Act 闭环是否顺利结束;``env_*`` 是 env 的客观判据
        (LIBERO BDDL goal),与 cap0-agent 同口径,两者并列以便对照。
        """

        if art is None:
            return
        summary = {
            "task": execution.task,
            "success": execution.success,
            **RoboMExAgent._env_objective(execution),
            "n_subgoals": len(execution.results),
            "subgoals": [
                {
                    "goal": r.subgoal.goal,
                    "planning_hint": r.subgoal.skill,
                    "success": r.success,
                    "inner_turns": len(r.trace.turns),
                    "loaded_skill_ids": list(r.trace.loaded_skill_ids),
                    "act_status": (r.trace.metadata or {}).get("act_status"),
                    "unresolved": (r.trace.metadata or {}).get("unresolved"),
                }
                for r in execution.results
            ],
        }
        (art / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
