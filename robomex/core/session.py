"""RoboMEx 框架入口:依赖容器 + 顶层 agent。

这里是把整个框架接线的唯一地方(对应 qwen-code 的 ``Config`` + runtime 拆分):
:class:`RoboMExConfig` 收集所有可替换依赖(技能库、两个策略、沙箱后端、可选的
collector),:class:`RoboMExAgent` 据此装配出反应式两层循环并跑一整段
episode。入口(``examples/``、评测脚手架)应构造一个 config 再调用
:meth:`RoboMExAgent.run`,而不是手工接线 planner 和 executor。
"""

from __future__ import annotations

import json
import html
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
from robomex.core.context import (
    AttemptHistory,
    AttemptRecord,
    Diagnosis,
    DiagnosisStore,
    EvidenceTimeline,
    EvidenceTimelineItem,
    PrimitiveTrace,
    TraceStore,
)
from robomex.core.events import emit_event, event_log, event_scope
from robomex.core.logging import get_logger
from robomex.skills import SkillLibrary
from robomex.agents.subagents import SubAgentExecutionPolicy, SubAgentRegistry, make_default_subagent_registry

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
    - ``subagent_system_prompt`` —— 覆盖默认 CodingAgentSubAgent 角色提示词;
      真机入口用它注入与 Act 相同的 sandbox API 文档。
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
    inner_system_prompt: str | None = None
    subagent_system_prompt: str | None = None
    observation_summary: str = ""
    observation_camera: str | None = None
    artifacts_dir: str | None = None
    subagents: SubAgentRegistry | None = None
    subagent_policy: CompletionPolicy | None = None
    subagent_max_turns: int = 8
    subagent_execution_policy: SubAgentExecutionPolicy | None = None
    enable_default_subagents: bool = True
    code_policy_kind: str = ""


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
            "observation_camera": config.observation_camera,
        }
        if config.collector is not None:
            kwargs["collector"] = config.collector
        if config.inner_system_prompt is not None:
            kwargs["system_prompt"] = config.inner_system_prompt
        subagents = config.subagents
        if subagents is None and config.enable_default_subagents:
            subagents = make_default_subagent_registry(
                executor=config.executor,
                policy=config.subagent_policy or config.code_policy,
                library=config.library,
                max_turns=config.subagent_max_turns,
                execution_policy=config.subagent_execution_policy,
                system_prompt=config.subagent_system_prompt
                if config.subagent_system_prompt is not None
                else None,
            )
        kwargs["subagents"] = subagents
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
            menu = [r.skill_id for r in self.config.library.task_skills()]
            subagent_runtime_enabled = (
                getattr(self.executor_agent, "subagents", None) is not None
                and self.executor_agent.subagents.has_runtime()
            )
            self._dump_subagent_manifest(
                art,
                subagent_runtime_enabled,
                code_policy_kind=self.config.code_policy_kind or self.config.code_policy.__class__.__name__,
                subagent_max_turns=self.config.subagent_max_turns,
                subagent_execution_policy=self.config.subagent_execution_policy,
            )
            _log.info("episode 开始 | task=%r | task skill guidance=%s | max_subgoals=%d",
                      task, menu, self.config.max_subgoals)
            emit_event(
                "episode_start",
                "RoboMEx episode started",
                scene_image_path=scene_image_path,
                task_skill_guidance=menu,
                subagent_runtime_enabled=subagent_runtime_enabled,
                code_policy_kind=self.config.code_policy_kind or self.config.code_policy.__class__.__name__,
                max_subgoals=self.config.max_subgoals,
            )

            results: list[SubGoalResult] = []
            cur_scene = scene_image_path
            trace_store = TraceStore()
            attempt_history = AttemptHistory()
            diagnosis_store = DiagnosisStore()
            evidence_timeline = EvidenceTimeline()
            planner_done = False
            for i in range(self.config.max_subgoals):
                with event_scope(subgoal_index=i, subgoal_number=i + 1):
                    subgoal = self.planner.next_subgoal(
                        task,
                        results,
                        scene_image_path=cur_scene,
                    )
                    self._record_planner(art, i, subgoal)
                    if subgoal is None:
                        planner_done = self.planner.last_raw.strip().upper() == "DONE"
                        _log.info("[subgoal %d] planner → DONE(没有下一个 sub-goal,结束)", i + 1)
                        emit_event("planner_done", "Planner returned DONE", accepted=planner_done)
                        break
                    _log.info("[subgoal %d] planner → goal=%r | 成功条件=%r",
                              i + 1, subgoal.goal, subgoal.postcondition)
                    emit_event(
                        "subgoal_start",
                        f"Subgoal {i + 1}: {subgoal.goal}",
                        goal=subgoal.goal,
                        postcondition=subgoal.postcondition,
                        scene_image_path=cur_scene,
                    )

                    # 子目标产物目录提前建好:执行器把每个有动作的 code block 的过程视频当场写进
                    # 这里(turn_NN.mp4),供日志、调试和 Planner 复盘使用。
                    sg_dir = (art / f"subgoal_{i:02d}") if art is not None else None
                    if sg_dir is not None:
                        sg_dir.mkdir(parents=True, exist_ok=True)
                        (sg_dir / "attempt_history_before.json").write_text(
                            json.dumps(attempt_history.compact_snapshot(), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        (sg_dir / "trace_store_before.json").write_text(
                            json.dumps(trace_store.compact_snapshot(), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        (sg_dir / "diagnoses_before.json").write_text(
                            json.dumps(diagnosis_store.compact_snapshot(), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        (sg_dir / "evidence_timeline_before.json").write_text(
                            json.dumps(evidence_timeline.compact_snapshot(), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )

                    # Act inner loop:执行器自主 use_skill/run_python,finish 表示当前 sub-goal
                    # 尝试结束,随后刷新场景并交回 Planner 规划下一步。
                    with event_scope(subgoal_dir=str(sg_dir) if sg_dir else None):
                        trace, success, note = self._run_subgoal(
                            subgoal,
                            sg_dir,
                            cur_scene,
                        )
                    results.append(SubGoalResult(subgoal=subgoal, trace=trace, success=success, note=note))
                    self._merge_trace_metadata(
                        trace_store,
                        attempt_history,
                        diagnosis_store,
                        evidence_timeline,
                        trace,
                        source=f"subgoal_{i:02d}",
                        subgoal_index=i,
                        subgoal_goal=subgoal.goal,
                        subgoal_postcondition=subgoal.postcondition,
                    )
                    if sg_dir is not None:
                        (sg_dir / "attempt_history_after.json").write_text(
                            json.dumps(attempt_history.compact_snapshot(), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        (sg_dir / "trace_store_after.json").write_text(
                            json.dumps(trace_store.compact_snapshot(), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        (sg_dir / "diagnoses_after.json").write_text(
                            json.dumps(diagnosis_store.compact_snapshot(), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        (sg_dir / "evidence_timeline_after.json").write_text(
                            json.dumps(evidence_timeline.compact_snapshot(), indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    trace_meta = trace.metadata or {}
                    act_status = trace_meta.get("act_status", "act")
                    _log.info("[subgoal %d] 结果 act_finished=%s | act=%s | 内层轮数=%d | 加载技能=%s",
                              i + 1, success, act_status,
                              len(trace.turns), list(trace.loaded_skill_ids))
                    emit_event(
                        "subgoal_end",
                        f"Subgoal {i + 1} ended",
                        goal=subgoal.goal,
                        act_finished=success,
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

            success = planner_done
            n_finished = sum(r.success for r in results)
            _log.info(
                "episode 结束 | planner_done=%s | %d/%d 个 sub-goal 尝试完成",
                success,
                n_finished,
                len(results),
            )

            execution = PlanExecution(
                task=task,
                subgoals=tuple(r.subgoal for r in results),
                results=tuple(results),
                success=success,
            )
            self._dump_summary(art, execution)
            self._dump_skill_evolution_candidates(art, execution)
            self._dump_debug_report(art, execution)
            if art is not None:
                (art / "trace_store.json").write_text(
                    json.dumps(trace_store.to_json_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (art / "attempt_history.json").write_text(
                    json.dumps(attempt_history.to_json_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (art / "diagnoses.json").write_text(
                    json.dumps(diagnosis_store.to_json_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (art / "evidence_timeline.json").write_text(
                    json.dumps(evidence_timeline.to_json_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (art / "evidence_timeline.md").write_text(
                    evidence_timeline.render_markdown(),
                    encoding="utf-8",
                )
            emit_event(
                "episode_end",
                "RoboMEx episode ended",
                success=success,
                finished_subgoals=n_finished,
                total_subgoals=len(results),
            )
            if art is not None:
                _log.info("产物已落盘到: %s", art)
                _log.info("  ├─ events.jsonl     结构化调试事件(Web UI/离线分析)")
                _log.info("  ├─ trace_store.json / attempt_history.json / diagnoses.json")
                _log.info("  ├─ evidence_timeline.json / evidence_timeline.md")
                _log.info("  ├─ subagents.json   SubAgent runtime 配置与执行边界")
                _log.info("  └─ subgoal_NN/subagents/  每次 SubAgent 委托的 request/result/code artifacts")
            return EpisodeResult(execution=execution)

    # ---- Act inner loop ----------------------------------------------------

    def _run_subgoal(
        self,
        subgoal: SubGoal,
        sg_dir: Path | None,
        scene_image_path: str | None = None,
    ):
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
        self._dump_subgoal(sg_dir, subgoal, trace)
        return trace, success, note

    @staticmethod
    def _merge_trace_metadata(
        trace_store: TraceStore,
        attempt_history: AttemptHistory,
        diagnosis_store: DiagnosisStore,
        evidence_timeline: EvidenceTimeline,
        trace: AgentTrace,
        *,
        source: str,
        subgoal_index: int | None = None,
        subgoal_goal: str = "",
        subgoal_postcondition: str = "",
    ) -> None:
        """Promote structured Act/SubAgent outputs into episode-level stores.

        EVIDENCE is intentionally not inspected here: only explicit metadata returned by
        Act/SubAgents becomes cross-agent memory.
        """

        meta = trace.metadata or {}
        evidence_timeline.extend_from_trace_metadata(
            meta,
            subgoal_index=subgoal_index,
            subgoal_goal=subgoal_goal,
            subgoal_postcondition=subgoal_postcondition,
            act_status=str(meta.get("act_status") or ""),
            loaded_skill_ids=tuple(trace.loaded_skill_ids),
        )

        for i, raw in enumerate(meta.get("primitive_traces") or ()):
            primitive = PrimitiveTrace.from_any(
                raw,
                default_id=f"{source}:primitive:{i}",
                default_producer="act",
            )
            if primitive is None:
                continue
            trace_store.add(primitive)

        for i, raw in enumerate(meta.get("attempt_records") or ()):
            attempt = AttemptRecord.from_any(raw, default_id=f"{source}:attempt:{i}")
            if attempt is None:
                continue
            attempt_history.add(attempt)

        for i, raw in enumerate(meta.get("diagnoses") or ()):
            diagnosis = Diagnosis.from_any(raw, default_id=f"{source}:diagnosis:{i}")
            if diagnosis is None:
                continue
            diagnosis_store.add(diagnosis)

    # ---- 产物落盘(artifacts_dir 给定时启用) ------------------------------

    def _record_planner(self, art: Path | None, index: int, subgoal: SubGoal | None) -> None:
        """把一次 planner 决策(原始回复 + 解析结果)追加进 ``planner.jsonl``。"""

        if art is None:
            return
        raw = getattr(self.planner, "last_raw", "")
        entry: dict[str, Any] = {"index": index, "raw": raw}
        if subgoal is None:
            entry["decision"] = "DONE" if str(raw).strip().upper() == "DONE" else "NO_SUBGOAL"
        else:
            entry.update(
                goal=subgoal.goal,
                postcondition=subgoal.postcondition,
            )
        with (art / "planner.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _dump_subgoal(d: Path | None, subgoal: SubGoal, trace: AgentTrace) -> None:
        """把一次 sub-goal 尝试的元信息 + 内层每轮代码/输出写到目录 ``d``。
        """

        if d is None:
            return
        d.mkdir(parents=True, exist_ok=True)
        trace_meta = trace.metadata or {}
        meta = {
            "goal": subgoal.goal,
            "postcondition": subgoal.postcondition,
            "act_finished": trace.success,
            "act_status": trace_meta.get("act_status"),
            "loaded_skill_ids": list(trace.loaded_skill_ids),
            "unresolved": trace_meta.get("unresolved"),
            "terminal_result": trace_meta.get("terminal_result"),
            "evidence_packets": list(trace_meta.get("evidence_packets") or ()),
            "evidence_timeline": list(trace_meta.get("evidence_timeline") or ()),
            "artifact_refs": list(trace_meta.get("artifact_refs") or ()),
            "primitive_trace_count": len(trace_meta.get("primitive_traces") or ()),
            "attempt_records": list(trace_meta.get("attempt_records") or ()),
            "diagnoses": list(trace_meta.get("diagnoses") or ()),
            "local_verdicts": list(trace_meta.get("local_verdicts") or ()),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        local_timeline = EvidenceTimeline()
        for raw in trace_meta.get("evidence_timeline") or ():
            if not isinstance(raw, dict):
                continue
            item = EvidenceTimelineItem.from_json_dict(raw)
            if item is not None:
                local_timeline.records.append(item)
        (d / "evidence_timeline.json").write_text(
            json.dumps(local_timeline.to_json_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (d / "evidence_timeline.md").write_text(local_timeline.render_markdown(), encoding="utf-8")
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
    def _dump_subagent_manifest(
        art: Path | None,
        runtime_enabled: bool,
        *,
        code_policy_kind: str,
        subagent_max_turns: int,
        subagent_execution_policy: SubAgentExecutionPolicy | None,
    ) -> None:
        if art is None:
            return
        policy = subagent_execution_policy or SubAgentExecutionPolicy()
        payload = {
            "schema": "robomex.subagents.v1",
            "subagent_runtime_enabled": runtime_enabled,
            "subagent_runtime": "coding_agent",
            "code_policy_kind": code_policy_kind,
            "subagent_max_turns": subagent_max_turns,
            "execution_policy": {
                "denied_calls": sorted(policy.denied_calls),
            },
        }
        (art / "subagents.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _dump_summary(self, art: Path | None, execution: PlanExecution) -> None:
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
            "subagent_runtime_enabled": (
                getattr(self.executor_agent, "subagents", None) is not None
                and self.executor_agent.subagents.has_runtime()
            ),
            "code_policy_kind": self.config.code_policy_kind or self.config.code_policy.__class__.__name__,
            "subgoals": [
                {
                    "goal": r.subgoal.goal,
                    "act_finished": r.success,
                    "inner_turns": len(r.trace.turns),
                    "loaded_skill_ids": list(r.trace.loaded_skill_ids),
                    "subagent_calls": list((r.trace.metadata or {}).get("subagent_calls") or ()),
                    "evidence_packet_count": len((r.trace.metadata or {}).get("evidence_packets") or ()),
                    "evidence_timeline": list((r.trace.metadata or {}).get("evidence_timeline") or ()),
                    "act_status": (r.trace.metadata or {}).get("act_status"),
                    "unresolved": (r.trace.metadata or {}).get("unresolved"),
                    "primitive_trace_count": len((r.trace.metadata or {}).get("primitive_traces") or ()),
                    "attempt_records": list((r.trace.metadata or {}).get("attempt_records") or ()),
                    "diagnoses": list((r.trace.metadata or {}).get("diagnoses") or ()),
                    "local_verdicts": list((r.trace.metadata or {}).get("local_verdicts") or ()),
                }
                for r in execution.results
            ],
        }
        (art / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _dump_debug_report(art: Path | None, execution: PlanExecution) -> None:
        """Write a static HTML report for fast live-run debugging."""

        if art is None:
            return
        env = RoboMExAgent._env_objective(execution)
        rows = []
        sections = []
        for i, result in enumerate(execution.results):
            meta = result.trace.metadata or {}
            status = str(meta.get("act_status") or ("finished" if result.success else "stopped"))
            rows.append(
                "<tr>"
                f"<td>{i}</td>"
                f"<td>{_h(result.subgoal.goal)}</td>"
                f"<td>{_h(status)}</td>"
                f"<td>{len(result.trace.turns)}</td>"
                f"<td>{_h(', '.join(result.trace.loaded_skill_ids) or 'none')}</td>"
                f"<td>{len(meta.get('evidence_timeline') or ())}</td>"
                f"<td>{len(meta.get('subagent_calls') or ())}</td>"
                f"</tr>"
            )
            sections.append(_render_subgoal_report(art, i, result))

        html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RoboMEx Debug Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17202a; background: #f7f8fa; }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    .panel {{ background: white; border: 1px solid #dde2ea; border-radius: 8px; padding: 16px; margin: 14px 0; }}
    .muted {{ color: #667085; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ text-align: left; border-bottom: 1px solid #e6e9ef; padding: 8px; vertical-align: top; }}
    th {{ background: #eef2f7; font-weight: 600; }}
    .badge {{ display: inline-block; padding: 2px 7px; border-radius: 999px; background: #eef2f7; margin-right: 6px; font-size: 12px; }}
    .ok {{ background: #e8f5e9; }}
    .fail {{ background: #ffebee; }}
    .uncertain {{ background: #fff8e1; }}
    pre {{ white-space: pre-wrap; background: #111827; color: #f9fafb; padding: 10px; border-radius: 6px; overflow-x: auto; }}
    img.artifact {{ max-width: 360px; max-height: 260px; object-fit: contain; border: 1px solid #d0d5dd; border-radius: 6px; background: #fff; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #e1e5ec; border-radius: 8px; padding: 12px; background: #fff; }}
    a {{ color: #175cd3; text-decoration: none; }}
  </style>
</head>
<body>
  <h1>RoboMEx Debug Report</h1>
  <div class="panel">
    <div><strong>Task:</strong> {_h(execution.task)}</div>
    <div><strong>Planner success:</strong> {_h(execution.success)}</div>
    <div><strong>Env success:</strong> {_h(env.get('env_success'))}</div>
    <div class="muted">reward={_h(env.get('env_reward'))}, terminated={_h(env.get('env_terminated'))}, task_completed={_h(env.get('env_task_completed'))}</div>
    <div style="margin-top: 8px;">
      <a href="summary.json">summary.json</a> ·
      <a href="evidence_timeline.md">evidence_timeline.md</a> ·
      <a href="evidence_timeline.json">evidence_timeline.json</a> ·
      <a href="trace_store.json">trace_store.json</a> ·
      <a href="events.jsonl">events.jsonl</a>
    </div>
  </div>
  <div class="panel">
    <h2>Subgoals</h2>
    <table>
      <thead><tr><th>#</th><th>Goal</th><th>Act</th><th>Turns</th><th>Skills</th><th>Evidence</th><th>SubAgents</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  {''.join(sections)}
</body>
</html>
"""
        (art / "report.html").write_text(html_text, encoding="utf-8")

    @staticmethod
    def _dump_skill_evolution_candidates(art: Path | None, execution: PlanExecution) -> None:
        """Write a read-only trace digest for offline clean-skill evolution.

        This file is intentionally not an automatic skill mutation. It gathers
        compact evidence packets, attempts, diagnoses, and artifact refs so a
        later curator can propose workflow-memory edits with anti-shortcut review.
        """

        if art is None:
            return
        subgoals: list[dict[str, Any]] = []
        for i, result in enumerate(execution.results):
            meta = result.trace.metadata or {}
            terminal = meta.get("terminal_result") if isinstance(meta, dict) else {}
            subgoals.append({
                "index": i,
                "goal": result.subgoal.goal,
                "postcondition": result.subgoal.postcondition,
                "act_finished": result.success,
                "loaded_skill_ids": list(result.trace.loaded_skill_ids),
                "terminal_claim": terminal.get("claim") if isinstance(terminal, dict) else "",
                "evidence_packets": list(meta.get("evidence_packets") or ()),
                "evidence_timeline": list(meta.get("evidence_timeline") or ()),
                "attempt_records": list(meta.get("attempt_records") or ()),
                "diagnoses": list(meta.get("diagnoses") or ()),
                "local_verdicts": list(meta.get("local_verdicts") or ()),
                "artifact_refs": list(meta.get("artifact_refs") or ()),
                "primitive_trace_count": len(meta.get("primitive_traces") or ()),
            })
        payload = {
            "schema": "robomex.skill_evolution_candidates.v1",
            "task": execution.task,
            "success": execution.success,
            "policy": (
                "read-only candidate digest; do not admit these into SKILL.md "
                "without clean-rule and anti-shortcut review"
            ),
            "subgoals": subgoals,
            "candidate_clean_rules": [],
            "weak_priors": [],
            "shortcut_rejects": [],
        }
        (art / "skill_evolution_candidates.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _render_subgoal_report(art: Path, index: int, result: SubGoalResult) -> str:
    meta = result.trace.metadata or {}
    d = f"subgoal_{index:02d}"
    evidence_cards = [
        _render_evidence_card(art, item)
        for item in (meta.get("evidence_timeline") or ())
        if isinstance(item, dict)
    ]
    subagent_cards = [
        _render_subagent_card(art, call)
        for call in (meta.get("subagent_calls") or ())
        if isinstance(call, dict)
    ]
    artifact_cards = [
        _render_artifact_ref(art, artifact)
        for artifact in (meta.get("artifact_refs") or ())
        if isinstance(artifact, dict)
    ]
    attempts = meta.get("attempt_records") or ()
    diagnoses = meta.get("diagnoses") or ()
    verdicts = meta.get("local_verdicts") or ()
    terminal = meta.get("terminal_result") if isinstance(meta.get("terminal_result"), dict) else {}
    return f"""
  <div class="panel" id="{d}">
    <h2>Subgoal {index}: {_h(result.subgoal.goal)}</h2>
    <div class="muted">Postcondition: {_h(result.subgoal.postcondition or '(none)')}</div>
    <div style="margin-top: 8px;">
      <span class="badge {_status_class(meta.get('act_status'))}">act={_h(meta.get('act_status'))}</span>
      <span class="badge">turns={len(result.trace.turns)}</span>
      <span class="badge">skills={_h(', '.join(result.trace.loaded_skill_ids) or 'none')}</span>
      <a href="{d}/meta.json">meta.json</a> ·
      <a href="{d}/evidence_timeline.md">evidence_timeline.md</a>
    </div>
    {_render_terminal_claim(terminal)}
    <h3>Evidence Timeline</h3>
    <div class="grid">{''.join(evidence_cards) if evidence_cards else '<div class="muted">(none)</div>'}</div>
    <h3>SubAgent Calls</h3>
    <div class="grid">{''.join(subagent_cards) if subagent_cards else '<div class="muted">(none)</div>'}</div>
    <h3>Artifacts</h3>
    <div class="grid">{''.join(artifact_cards) if artifact_cards else '<div class="muted">(none)</div>'}</div>
    <details><summary>Attempts / diagnoses / verdicts</summary>
      <pre>{_h(json.dumps({'attempt_records': attempts, 'diagnoses': diagnoses, 'local_verdicts': verdicts}, indent=2, ensure_ascii=False))}</pre>
    </details>
  </div>
"""


def _render_terminal_claim(terminal: dict[str, Any]) -> str:
    if not terminal:
        return ""
    claim = str(terminal.get("claim") or "").strip()
    if not claim:
        return ""
    return f"<div class=\"card\"><strong>Terminal claim:</strong> {_h(claim)}</div>"


def _render_evidence_card(art: Path, item: dict[str, Any]) -> str:
    packet = item.get("packet") if isinstance(item.get("packet"), dict) else {}
    verdict = packet.get("verdict") if isinstance(packet.get("verdict"), dict) else {}
    artifacts = packet.get("artifact_refs") if isinstance(packet.get("artifact_refs"), list) else []
    body = {
        "source": item.get("source"),
        "confidence": packet.get("confidence"),
        "uncertainty": packet.get("uncertainty"),
        "recommended_next": packet.get("recommended_next"),
        "evidence_summary": packet.get("evidence_summary"),
        "facts": packet.get("facts"),
    }
    artifact_html = "".join(
        _render_artifact_ref(art, a)
        for a in artifacts
        if isinstance(a, dict)
    )
    return f"""
    <div class="card">
      <div><span class="badge {_status_class(verdict.get('status'))}">{_h(verdict.get('status') or 'evidence')}</span></div>
      <h3>{_h(packet.get('claim') or '(no claim)')}</h3>
      <div class="muted">{_h(verdict.get('reason') or '')}</div>
      <pre>{_h(json.dumps(body, indent=2, ensure_ascii=False))}</pre>
      {artifact_html}
    </div>
"""


def _render_subagent_card(art: Path, call: dict[str, Any]) -> str:
    ok = call.get("ok")
    packet = call.get("evidence_packet") if isinstance(call.get("evidence_packet"), dict) else {}
    result = {
        "task": call.get("task"),
        "ok": ok,
        "turns": call.get("turns"),
        "loaded_skill_ids": call.get("loaded_skill_ids"),
        "claim": call.get("claim") or packet.get("claim"),
        "verdict": packet.get("verdict"),
        "uncertainty": packet.get("uncertainty"),
        "recommended_next": packet.get("recommended_next"),
        "error": call.get("error"),
    }
    sub_dir = call.get("artifacts_dir")
    link = ""
    if sub_dir:
        rel = _relative_artifact_src(art, str(Path(sub_dir) / "result.json"))
        link = f'<a href="{_h(rel)}">result.json</a>'
    return f"""
    <div class="card">
      <div><span class="badge {'ok' if ok else 'fail'}">subagent {_h('ok' if ok else 'failed')}</span> {link}</div>
      <h3>{_h(call.get('task') or '(no task)')}</h3>
      <pre>{_h(json.dumps(result, indent=2, ensure_ascii=False))}</pre>
    </div>
"""


def _render_artifact_ref(art: Path, artifact: dict[str, Any]) -> str:
    raw_path = artifact.get("path") or artifact.get("uri") or artifact.get("artifact_id")
    if not raw_path:
        return ""
    src = _relative_artifact_src(art, str(raw_path))
    label = artifact.get("summary") or artifact.get("kind") or artifact.get("artifact_id") or raw_path
    if _is_image_path(str(raw_path)):
        preview = f'<img class="artifact" src="{_h(src)}" alt="{_h(label)}">'
    else:
        preview = f'<a href="{_h(src)}">{_h(src)}</a>'
    return f"""
    <div class="card">
      <div><strong>{_h(label)}</strong></div>
      {preview}
    </div>
"""


def _relative_artifact_src(root: Path, path_text: str) -> str:
    path = Path(path_text)
    try:
        if path.is_absolute():
            return path.relative_to(root).as_posix()
    except ValueError:
        pass
    return path.as_posix()


def _is_image_path(path_text: str) -> bool:
    return Path(path_text).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _status_class(value: Any) -> str:
    text = str(value or "").lower()
    if text in {"pass", "passed", "success", "succeeded", "finished"}:
        return "ok"
    if text in {"fail", "failed", "error", "exhausted", "unresolved"}:
        return "fail"
    if text in {"uncertain", "unknown"}:
        return "uncertain"
    return ""


def _h(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)
