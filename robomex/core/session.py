"""RoboMEx 框架入口:依赖容器 + 顶层 agent。

这里是整个框架唯一的接线入口。:class:`RoboMExConfig` 收集技能库、Planner、
authoring graph、模型策略、沙箱后端和可选 collector；:class:`RoboMExAgent`
据此装配 Planner → Subgoal Runner 主循环并跑一整段 episode。入口
(``examples/``、评测脚手架)应构造一个 config 再调用
:meth:`RoboMExAgent.run`,而不是手工接线 planner 和 executor。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robomex.agents.executor import CodeAsPolicyAgent, _primitive_traces_from_execution
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
from robomex.dysc import DySCOnlineConfig, OnlineEvolutionManager
from robomex.skills import SkillLibrary
from robomex.authoring import (
    KNOWN_CAPABILITIES,
    CapabilityBoundBlockExecutor,
    CapabilityPolicy,
    SubAgentFactory,
    SubgoalAuthoringContext,
    SubgoalSwarmManager,
    UniversalRunner,
)
from robomex.core.sandbox import MotionLeaseGuard, RuntimeSafetyState

_log = get_logger("session")


@dataclass
class RoboMExConfig:
    """一次 RoboMEx 运行的所有可替换依赖。

    循环需要的一切都在这里注入,因此离线(脚本)和真机(LLM + CapX env)运行的
    区别只在于 config 里放了什么:

    - ``library``         —— 技能库(内置 + 学到的技能包)。
    - ``planner_policy``  —— 驱动外层 :class:`ReactivePlanner`。
    - ``code_policy``     —— 驱动 Swarm Manager，并作为内层策略兜底。
    - ``subagent_policy`` —— 可选的叶子 Coding SubAgent 策略；未提供时回退到
      ``code_policy``。
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
    inner_system_prompt: str | None = None
    observation_summary: str = ""
    observation_camera: str | None = None
    artifacts_dir: str | None = None
    subagent_policy: CompletionPolicy | None = None
    subagent_max_turns: int = 8
    code_policy_kind: str = ""
    dysc_online: DySCOnlineConfig | None = None
    authoring_strategy: str = "universal"
    authoring_environment: str = ""
    authoring_api_docs: str = ""
    swarm_manager_max_turns: int = 4
    swarm_capability_ceiling: frozenset[str] = KNOWN_CAPABILITIES


@dataclass(frozen=True)
class EpisodeResult:
    """一次 :meth:`RoboMExAgent.run` episode 的结果。"""

    execution: PlanExecution
    env_success: bool | None = None

    @property
    def success(self) -> bool:
        """Compatibility projection; prefer ``env_success`` and ``planner_status``."""

        return self.env_success if self.env_success is not None else self.execution.success


class RoboMExAgent:
    """顶层 agent:高层反应式 planner + subgoal authoring runtime。

    每一步,planner 参考高层技能指导给出下一个结构化 sub-goal；authoring strategy
    决定由 universal baseline 或 skill-conditioned SubgoalSwarmManager
    生成并执行该 subgoal policy；
    随后可选地从最新观测刷新场景；
    如此循环,直到 planner 说 DONE 或触达 sub-goal 上限。该循环在离线和真机下完全
    一致——唯一差别是注入的场景刷新(它与具体 env 相关)。
    """

    def __init__(self, config: RoboMExConfig) -> None:
        self.config = config
        self.planner = ReactivePlanner(config.library, config.planner_policy)
        safety_state = RuntimeSafetyState()
        universal_executor = CapabilityBoundBlockExecutor(
            MotionLeaseGuard(
                config.executor,
                safety_state,
                agent_id="universal_coder",
            ),
            CapabilityPolicy(allowed=KNOWN_CAPABILITIES),
            node_id="universal_coder",
        )

        kwargs: dict[str, Any] = {
            "executor": universal_executor,
            "policy": config.subagent_policy or config.code_policy,
            "library": config.library,
            "max_turns": config.max_turns,
            "observation_camera": config.observation_camera,
        }
        if config.collector is not None:
            kwargs["collector"] = config.collector
        if config.inner_system_prompt is not None:
            kwargs["system_prompt"] = config.inner_system_prompt
        self.executor_agent = CodeAsPolicyAgent(**kwargs)
        strategy = config.authoring_strategy.strip().lower()
        if strategy not in {"universal", "dynamic_swarm"}:
            raise ValueError(f"Unsupported authoring_strategy {config.authoring_strategy!r}.")
        factory = SubAgentFactory(
            executor=config.executor,
            policy=config.subagent_policy or config.code_policy,
            library=config.library,
            capability_ceiling=config.swarm_capability_ceiling,
            default_max_turns=config.subagent_max_turns,
            environment=config.authoring_environment,
            api_docs=config.authoring_api_docs,
            safety_state=safety_state,
        )
        swarm_manager = (
            SubgoalSwarmManager(
                policy=config.code_policy,
                library=config.library,
                factory=factory,
                capability_ceiling=config.swarm_capability_ceiling,
                safety_state=safety_state,
                max_turns=config.swarm_manager_max_turns,
                environment=config.authoring_environment,
                api_docs=config.authoring_api_docs,
            )
            if strategy == "dynamic_swarm"
            else None
        )
        self.authoring = swarm_manager or UniversalRunner(
            self.executor_agent,
            safety_state,
        )
        self.dysc: OnlineEvolutionManager | None = None
        if config.dysc_online is not None and config.dysc_online.enabled:
            self.dysc = OnlineEvolutionManager(
                config=config.dysc_online,
                library=config.library,
                policy=config.subagent_policy or config.code_policy,
                artifacts_dir=config.artifacts_dir,
            )

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
            graph_name = self.authoring.graph_name
            _log.info("episode 开始 | task=%r | task skill guidance=%s | max_subgoals=%d",
                      task, menu, self.config.max_subgoals)
            emit_event(
                "episode_start",
                "RoboMEx episode started",
                scene_image_path=scene_image_path,
                task_skill_guidance=menu,
                authoring_graph=graph_name,
                authoring_strategy=self.authoring.strategy,
                code_policy_kind=self.config.code_policy_kind or self.config.code_policy.__class__.__name__,
                max_subgoals=self.config.max_subgoals,
            )

            results: list[SubGoalResult] = []
            cur_scene = scene_image_path
            trace_store = TraceStore()
            attempt_history = AttemptHistory()
            diagnosis_store = DiagnosisStore()
            evidence_timeline = EvidenceTimeline()
            planner_status = "exhausted"
            for i in range(self.config.max_subgoals):
                with event_scope(subgoal_index=i, subgoal_number=i + 1):
                    subgoal = self.planner.next_subgoal(
                        task,
                        results,
                        scene_image_path=cur_scene,
                    )
                    self._record_planner(art, i, subgoal)
                    if subgoal is None:
                        planner_status = (
                            "done"
                            if self.planner.last_raw.strip().upper() == "DONE"
                            else "invalid_response"
                        )
                        _log.info("[subgoal %d] planner stopped: %s", i + 1, planner_status)
                        emit_event(
                            "planner_stopped",
                            "Planner stopped producing subgoals",
                            planner_status=planner_status,
                            raw=self.planner.last_raw,
                        )
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

                    # Every subgoal uses the same authoring runtime.  Universal execution is
                    # represented by a one-node graph rather than a Planner→Act bypass.
                    with event_scope(subgoal_dir=str(sg_dir) if sg_dir else None):
                        self._prepare_dysc_subgoal()
                        authoring_result = self.authoring.run(
                            SubgoalAuthoringContext(
                                task=task,
                                subgoal_index=i,
                                goal=subgoal.goal,
                                postcondition=subgoal.postcondition,
                                scene_image_path=cur_scene,
                                observation_summary=self.config.observation_summary,
                                artifact_dir=sg_dir,
                            )
                        )
                    if sg_dir is not None:
                        (sg_dir / "authoring_outcome.json").write_text(
                            json.dumps(
                                authoring_result.to_json_dict(),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    trace = authoring_result.executable_trace or AgentTrace(
                        task=subgoal.goal,
                        loaded_skill_ids=(),
                        turns=(),
                        success=False,
                        metadata={"act_status": "not_run"},
                    )
                    note = authoring_result.note
                    results.append(
                        SubGoalResult(
                            subgoal=subgoal,
                            trace=trace,
                            success=authoring_result.success,
                            motion_attempted=authoring_result.motion_attempted,
                            authoring_status=authoring_result.status.value,
                            verification_status=authoring_result.verification.value,
                            note=note,
                        )
                    )
                    self._dump_subgoal(sg_dir, subgoal, trace)
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
                    self._merge_swarm_node_results(
                        trace_store,
                        attempt_history,
                        diagnosis_store,
                        authoring_result,
                        subgoal_index=i,
                        subgoal_goal=subgoal.goal,
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
                    execution_status = (
                        "succeeded" if authoring_result.success else "stopped"
                    )
                    _log.info("[subgoal %d] 结果 authoring=%s | execution=%s | motion=%s | 内层轮数=%d | 加载技能=%s",
                              i + 1, authoring_result.status.value, execution_status,
                        authoring_result.motion_attempted,
                              len(trace.turns), list(trace.loaded_skill_ids))
                    emit_event(
                        "subgoal_end",
                        f"Subgoal {i + 1} ended",
                        goal=subgoal.goal,
                        authoring_status=authoring_result.status.value,
                        verification_status=authoring_result.verification.value,
                        success=authoring_result.success,
                        motion_attempted=authoring_result.motion_attempted,
                        execution_status=execution_status,
                        inner_turns=len(trace.turns),
                        loaded_skill_ids=list(trace.loaded_skill_ids),
                        note=note,
                    )

                    # sub-goal 跑完立即触发回调(真机:把这段视频当场写进 subgoal 目录)。
                    if on_subgoal_end is not None:
                        on_subgoal_end(i, results[-1], sg_dir)

                    self._evolve_dysc_after_subgoal(task, results[-1], sg_dir)

                    # 环境已宣告任务终止(LIBERO: task_completed / terminated)时立即收束,
                    # 不再让 planner 凭旧场景图继续编造后续 subgoal(M1-B8)。
                    env_signal = self._env_termination_signal(trace)
                    if env_signal is not None:
                        planner_status = "env_terminated"
                        _log.info(
                            "[subgoal %d] 环境宣告终止(%s),episode 提前收束",
                            i + 1,
                            env_signal,
                        )
                        emit_event(
                            "episode_short_circuit",
                            "Environment signalled task termination; ending episode early",
                            **env_signal,
                        )
                        break

                    # 真机:用最新观测刷新 planner 下一步看到的场景图;离线则跳过。
                    if scene_refresh is not None and trace.turns and trace.turns[-1].execution.observation:
                        refreshed = scene_refresh(trace.turns[-1].execution.observation)
                        cur_scene = refreshed or cur_scene
                        emit_event("scene_refreshed", "Planner scene image refreshed", scene_image_path=cur_scene)

            n_finished = sum(r.success for r in results)
            _log.info(
                "episode 结束 | planner_status=%s | %d/%d 个 sub-goal authoring 执行完成",
                planner_status,
                n_finished,
                len(results),
            )

            execution = PlanExecution(
                task=task,
                subgoals=tuple(r.subgoal for r in results),
                results=tuple(results),
                planner_status=planner_status,
            )
            env_objective = self._env_objective(execution)
            self._dump_summary(art, execution)
            self._dump_skill_evolution_candidates(art, execution)
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
                planner_status=planner_status,
                env_success=env_objective["env_success"],
                finished_subgoals=n_finished,
                total_subgoals=len(results),
            )
            if art is not None:
                _log.info("产物已落盘到: %s", art)
                _log.info("  ├─ events.jsonl     结构化调试事件(离线分析)")
                _log.info("  ├─ trace_store.json / attempt_history.json / diagnoses.json")
                _log.info("  ├─ evidence_timeline.json / evidence_timeline.md")
                _log.info(
                    "  └─ subgoal_NN/authoring/swarm/  Manager、Typed Graph、"
                    "ArtifactStore 与 Specialist artifacts"
                )
            return EpisodeResult(
                execution=execution,
                env_success=env_objective["env_success"],
            )

    def _prepare_dysc_subgoal(self) -> None:
        if self.dysc is None:
            return
        if self.authoring.strategy == "universal":
            self.executor_agent.library = self.dysc.library_view_for_motion_role()
        emit_event(
            "dysc_society_loaded",
            "DySC society prepared for authoring",
            society=self.dysc.current.name,
            motion_role=self.dysc.current.motion_role_name(),
            society_version=self.dysc.store.latest_version(),
            authoring_strategy=self.authoring.strategy,
        )

    def _evolve_dysc_after_subgoal(
        self,
        task: str,
        result: SubGoalResult,
        sg_dir: Path | None,
    ) -> None:
        if self.dysc is None:
            return
        before = self.dysc.store.latest_version()
        self.dysc.evolve_after_subgoal(task=task, subgoal_dir=sg_dir, subgoal_result=result)
        after = self.dysc.store.latest_version()
        emit_event(
            "dysc_society_evolved",
            "DySC online evolution step completed",
            before_version=before,
            after_version=after,
            society=self.dysc.current.name,
        )

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

    @staticmethod
    def _env_termination_signal(trace: AgentTrace) -> dict[str, Any] | None:
        """Detect an environment-declared task end inside one subgoal's turns.

        LIBERO-style envs latch ``task_completed`` / ``terminated`` once the BDDL
        goal holds; the first hit is authoritative regardless of what later
        blocks report.
        """

        for turn in trace.turns:
            execution = turn.execution
            if bool((execution.info or {}).get("task_completed")):
                return {"task_completed": True, "turn": turn.turn}
            if execution.terminated:
                return {"terminated": True, "turn": turn.turn}
        return None

    @staticmethod
    def _merge_swarm_node_results(
        trace_store: TraceStore,
        attempt_history: AttemptHistory,
        diagnosis_store: DiagnosisStore,
        outcome: Any,
        *,
        subgoal_index: int,
        subgoal_goal: str,
    ) -> None:
        """Project per-node swarm results into episode memory (M1-B7).

        The universal path self-reports through trace metadata, but dynamic-swarm
        specialists often don't. The runtime therefore projects every
        ``AuthoringNodeResult`` deterministically: sandbox primitive traces go to
        the TraceStore, each node attempt becomes an AttemptRecord, and each
        failed node becomes a Diagnosis. This is what the Planner reads to stop
        repeating verbatim subgoals.
        """

        node_results = tuple(getattr(outcome, "node_results", ()) or ())
        if not node_results:
            return
        prefix = f"subgoal_{subgoal_index:02d}"
        for result in node_results:
            node_key = f"{prefix}:{result.node_id}:a{result.attempt}"
            related_trace_ids: list[str] = []
            if result.trace is not None:
                for turn in result.trace.turns:
                    for trace in _primitive_traces_from_execution(
                        turn.turn,
                        turn.execution,
                        producer=result.node_id,
                    ):
                        trace_store.add(trace)
                        related_trace_ids.append(trace.trace_id)
            evidence = result.evidence
            recommended = str(getattr(evidence, "recommended_next", "") or "")
            failure_reason = result.failure_kind or result.error
            attempt_history.add(
                AttemptRecord(
                    attempt_id=node_key,
                    object_key=subgoal_goal,
                    strategy=result.node_id,
                    related_trace_ids=tuple(related_trace_ids),
                    outcome=result.status.value
                    if result.verification.value == "not_run"
                    else f"{result.status.value}/verified_{result.verification.value}",
                    failure_reason="" if result.ok else failure_reason,
                    recommended_repair=recommended,
                )
            )
            if not result.ok:
                diagnosis_store.add(
                    Diagnosis(
                        diagnosis_id=f"{node_key}:diagnosis",
                        failed_primitive=result.node_id,
                        failure_type=result.failure_kind or "unspecified_failure",
                        evidence_trace_ids=tuple(related_trace_ids),
                        next_route=recommended,
                        reason=result.error,
                    )
                )

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
            "execution_finished": trace.success,
            "execution_status": trace_meta.get("act_status"),
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

    def _dump_summary(self, art: Path | None, execution: PlanExecution) -> None:
        """Write explicit planner, authoring, verification, and env outcomes."""

        if art is None:
            return
        summary = {
            "task": execution.task,
            "planner_status": execution.planner_status,
            "authoring_graph": self.authoring.graph_name,
            "authoring_strategy": self.authoring.strategy,
            **RoboMExAgent._env_objective(execution),
            "n_subgoals": len(execution.results),
            "code_policy_kind": self.config.code_policy_kind or self.config.code_policy.__class__.__name__,
            "subgoals": [
                {
                    "goal": r.subgoal.goal,
                    "authoring_status": r.authoring_status,
                    "verification_status": r.verification_status,
                    "success": r.success,
                    "motion_attempted": r.motion_attempted,
                    "note": r.note,
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
                "authoring_status": result.authoring_status,
                "verification_status": result.verification_status,
                "success": result.success,
                "motion_attempted": result.motion_attempted,
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
            "planner_status": execution.planner_status,
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
