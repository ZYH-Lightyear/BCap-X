"""RoboMEx 框架入口:依赖容器 + 顶层 agent。

这里是把整个框架接线的唯一地方(对应 qwen-code 的 ``Config`` + runtime 拆分):
:class:`RoboMExConfig` 收集所有可替换依赖(技能库、两个策略、沙箱后端、可选的
verifier/collector),:class:`RoboMExAgent` 据此装配出反应式两层循环并跑一整段
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
from robomex.agents.verifier import VerifyCodeAgent
from robomex.core.coder.policy import CompletionPolicy
from robomex.core.coder.trace import AgentTrace
from robomex.core.logging import get_logger
from robomex.skills import SkillLibrary
from robomex.verification import VerifierContext, build_op_trace, collect_verify_resources

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
    - ``enable_verification`` —— 是否在每个 sub-goal 结束后跑独立的
      :class:`VerifyCodeAgent` 做子目标级视觉裁决(它复用 ``code_policy`` 与同一沙箱)。
    - ``collector`` —— 可选的逐块证据采集(存 before/after 帧调试产物)。
    - ``inner_system_prompt`` —— 覆盖执行器的角色提示词(例如注入真机 API 文档);
      ``None`` 则保持执行器默认。
    - ``artifacts_dir`` —— 可选;给定后,每个 sub-goal 的 planner 决策、内层每轮代码 /
      输出、验证裁决,以及一份 episode 汇总都会落盘到此目录。``None`` 则不落盘。
    """

    library: SkillLibrary
    planner_policy: PlannerPolicy
    code_policy: CompletionPolicy
    executor: Any
    enable_verification: bool = False
    collector: Any | None = None
    max_turns: int = 6
    max_subgoals: int = 8
    verify_max_turns: int = 4
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

    每一步,planner 给出下一个 sub-goal(依据任务、当前场景、高层技能菜单、历史);
    内层执行器执行它(并被告知先咨询哪个高层技能);随后可选地从最新观测刷新场景;
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

        menu = [r.skill_id for r in self.config.library.compound_skills()]
        _log.info("episode 开始 | task=%r | 高层技能菜单=%s | max_subgoals=%d",
                  task, menu, self.config.max_subgoals)

        results: list[SubGoalResult] = []
        cur_scene = scene_image_path
        for i in range(self.config.max_subgoals):
            subgoal = self.planner.next_subgoal(task, results, scene_image_path=cur_scene)
            self._record_planner(art, i, subgoal)
            if subgoal is None:
                _log.info("[subgoal %d] planner → DONE(没有下一个 sub-goal,结束)", i + 1)
                break
            _log.info("[subgoal %d] planner → goal=%r | skill=%s | 成功条件=%r",
                      i + 1, subgoal.goal, subgoal.skill, subgoal.postcondition)

            trace = self.executor_agent.run(
                subgoal.goal,
                self.config.observation_summary,
                primary_skill_id=subgoal.skill,
            )

            # 子目标级视觉裁决:独立的 VerifyCodeAgent 复用同一沙箱(EVIDENCE/OBS_BEFORE
            # 已持久),否则回退到 env 终止信号。
            vtrace = self._verify_subgoal(subgoal, trace)
            verdict = vtrace.verdict if vtrace is not None else None
            if verdict is not None:
                success = verdict.verdict == "passed"
                note = verdict.reason
            else:
                success = trace.success
                note = ""
            results.append(SubGoalResult(subgoal=subgoal, trace=trace, success=success, note=note))
            _log.info("[subgoal %d] 结果 success=%s | 裁决=%s | 内层轮数=%d | 加载技能=%s",
                      i + 1, success, (verdict.verdict if verdict else "env-signal"),
                      len(trace.turns), list(trace.loaded_skill_ids))
            sg_dir = (art / f"subgoal_{i:02d}") if art is not None else None
            self._dump_subgoal(art, i, subgoal, trace, vtrace)

            # sub-goal 跑完立即触发回调(真机:把这段视频当场写进 subgoal 目录)。
            if on_subgoal_end is not None:
                on_subgoal_end(i, results[-1], sg_dir)

            # 真机:用最新观测刷新 planner 下一步看到的场景图;离线则跳过。
            if scene_refresh is not None and trace.turns and trace.turns[-1].execution.observation:
                refreshed = scene_refresh(trace.turns[-1].execution.observation)
                cur_scene = refreshed or cur_scene

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
        if art is not None:
            _log.info("产物已落盘到: %s", art)
        return EpisodeResult(execution=execution)

    # ---- 子目标级验证 ------------------------------------------------------

    def _verify_subgoal(self, subgoal: SubGoal, trace: AgentTrace):
        """跑独立的 VerifyCodeAgent 给这个 sub-goal 做视觉裁决;关闭则返回 ``None``。"""

        if not self.config.enable_verification:
            return None
        try:
            resources = collect_verify_resources(
                [r.skill for r in self.config.library.all()], trace.loaded_skill_ids
            )
            context = VerifierContext(
                sub_goal=subgoal.goal,
                skills_used=trace.loaded_skill_ids,
                op_trace=tuple(build_op_trace(trace.turns)),
                resources=resources,
                expected_decomposition=subgoal.postcondition,
            )
            agent = VerifyCodeAgent(
                executor=self.config.executor,
                policy=self.config.code_policy,
                context=context,
                library=self.config.library,
                max_turns=self.config.verify_max_turns,
            )
            return agent.verify()
        except Exception as exc:  # noqa: BLE001 - 验证失败回退到 env 信号,不该中断 episode
            _log.warning("子目标验证失败,回退 env 信号: %r", exc)
            return None

    # ---- 产物落盘(artifacts_dir 给定时启用) ------------------------------

    def _record_planner(self, art: Path | None, index: int, subgoal: SubGoal | None) -> None:
        """把一次 planner 决策(原始回复 + 解析结果)追加进 ``planner.jsonl``。"""

        if art is None:
            return
        entry: dict[str, Any] = {"index": index, "raw": getattr(self.planner, "last_raw", "")}
        if subgoal is None:
            entry["decision"] = "DONE"
        else:
            entry.update(goal=subgoal.goal, skill=subgoal.skill, postcondition=subgoal.postcondition)
        with (art / "planner.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _dump_subgoal(art: Path | None, index: int, subgoal: SubGoal, trace: AgentTrace, vtrace=None) -> None:
        """把一个 sub-goal 的元信息 + 内层每轮代码/输出 + 验证裁决写到 ``subgoal_NN/``。"""

        if art is None:
            return
        d = art / f"subgoal_{index:02d}"
        d.mkdir(parents=True, exist_ok=True)
        verdict = vtrace.verdict if vtrace is not None else None
        meta = {
            "goal": subgoal.goal,
            "skill": subgoal.skill,
            "postcondition": subgoal.postcondition,
            "env_terminated": trace.success,
            "loaded_skill_ids": list(trace.loaded_skill_ids),
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
    def _dump_summary(art: Path | None, execution: PlanExecution) -> None:
        """把整段 episode 的汇总写到 ``summary.json``。"""

        if art is None:
            return
        summary = {
            "task": execution.task,
            "success": execution.success,
            "n_subgoals": len(execution.results),
            "subgoals": [
                {
                    "goal": r.subgoal.goal,
                    "skill": r.subgoal.skill,
                    "success": r.success,
                    "inner_turns": len(r.trace.turns),
                    "loaded_skill_ids": list(r.trace.loaded_skill_ids),
                }
                for r in execution.results
            ],
        }
        (art / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
