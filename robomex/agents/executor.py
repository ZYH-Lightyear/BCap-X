"""CodeAsPolicy 技能 Agent:咨询技能、写代码的执行器。

它是共享 :class:`~robomex.core.coder.CodingAgent` 的轻量子类。执行器的特化在于:
上下文是任务(+观测);通过渐进披露感知*整库*(开头一份简短的 ``<available_skills>``
清单 + 用 ``use_skill`` 拉取正文);每个 python 轮都会打包证据;``finish`` 表示
Act 认为当前 sub-goal 尝试结束,控制权交回外层 Planner。

技能只是被*咨询*,绝不照搬执行;代码由策略自己生成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robomex.core.coder import CodingAgent, SkillEntry
from robomex.core.coder.policy import CompletionPolicy
from robomex.core.coder.trace import AgentTrace, TurnRecord
from robomex.core.events import emit_event
from robomex.core.logging import get_logger
from robomex.core.sandbox import BlockExecutionResult, SemanticActionBlock
from robomex.perception import EvidenceCollector, save_video
from robomex.skills import SkillLibrary

_log = get_logger("executor")

# 子目标开场注入沙箱:重置语义证据字典 + 抓子目标起始帧。沙箱 globals 跨 block 持久,
# 因此 EVIDENCE 与 OBS_BEFORE 会一直留到该子目标结束,作为调试和 planner 复盘材料。
_EVIDENCE_SEED = (
    "try:\n"
    "    EVIDENCE\n"
    "except NameError:\n"
    "    EVIDENCE = {}\n"
    "EVIDENCE.clear()\n"
    "try:\n"
    "    OBS_BEFORE = get_observation()['agentview']['images']['rgb'].copy()\n"
    "except Exception as _e:\n"
    "    OBS_BEFORE = None\n"
)

_SYSTEM_PROMPT = (
    "You are a robot Code-as-Policy agent. Each turn, reply with exactly one JSON action "
    "object. Use run_python to execute one block of Python that advances the task, "
    "grounding every decision in the current observation. "
    "Before writing any Python for a sub-goal, consult at least one relevant skill with "
    "`{\"tool\":\"use_skill\",\"args\":{\"name\":\"<skill>\"}}`. Skills are the operating procedure for this framework: follow "
    "their Procedure / Rules / Failure modes unless the live observation contradicts them, "
    "and state any deliberate deviation in code comments. Adapt skill guidance to the scene, "
    "but do not ignore it or invent an unrelated pipeline. Inspect observations directly, save useful "
    "raw evidence when it helps future planning, and decide whether another action is needed. "
    "Use `{\"tool\":\"run_python\",\"args\":{\"code\":\"...\", \"intent\":\"...\"}}` for code. "
    "Use `{\"tool\":\"finish\",\"args\":{\"claim\":\"...\"}}` when you believe this sub-goal attempt "
    "is complete and control should return to the Planner."
)


class CodeAsPolicyAgent(CodingAgent):
    def __init__(
        self,
        executor: Any,
        policy: CompletionPolicy,
        library: SkillLibrary,
        collector: EvidenceCollector | None = None,
        max_turns: int = 6,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        super().__init__(
            executor=executor,
            policy=policy,
            library=library,
            max_turns=max_turns,
            system_prompt=system_prompt,
        )
        self.collector = collector
        self._task = ""
        self._expected_postcondition = ""
        self._observation_summary = ""
        self._primary_skill_id: str | None = None
        self._video_dir: Path | None = None
        self._clips: list[dict] = []
        self._feedback = ""

    def run(
        self,
        task: str,
        observation_summary: str = "",
        primary_skill_id: str | None = None,
        expected_postcondition: str = "",
        video_dir: str | Path | None = None,
        feedback: str = "",
        scene_image_path: str | None = None,
    ) -> AgentTrace:
        self._task = task
        self._expected_postcondition = expected_postcondition
        self._observation_summary = observation_summary
        self._primary_skill_id = primary_skill_id
        self._video_dir = Path(video_dir) if video_dir is not None else None
        self._scene_image_path = scene_image_path
        self._clips = []
        self._feedback = feedback or ""
        return super().run()

    # ---- 钩子 --------------------------------------------------------------

    def _setup(self, prompt: list[dict]) -> None:
        """子目标开场:重置 EVIDENCE、抓起始帧 OBS_BEFORE + 注入 ARTIFACTS_DIR。"""

        art_dir = str(self._video_dir) if self._video_dir else "/tmp"
        seed = f"ARTIFACTS_DIR = {art_dir!r}\n" + _EVIDENCE_SEED
        try:
            self.executor.run_block(
                SemanticActionBlock(
                    name="evidence_seed", intent="seed subgoal evidence", code=seed
                )
            )
        except Exception as exc:  # noqa: BLE001 - 取证种子失败不该让子目标崩溃
            _log.warning("证据种子注入失败: %r", exc)

    def _skill_entries(self) -> list[SkillEntry]:
        return [
            SkillEntry(
                name=r.skill_id,
                description=r.skill.description or r.skill.name,
                category=r.skill.category.value,
            )
            for r in self.library.all()
        ]

    def _initial_user_message(self) -> str | list:
        from robomex.perception.render import image_content_part

        parts = [f"Task: {self._task}"]
        if self._observation_summary:
            parts.append(f"Observation: {self._observation_summary}")
        parts.append(
            "The current scene image is attached below. Inspect it before writing code. "
            "If the arm, gripper, or held object occludes the target or receptacle in the "
            "image, first call `goto_home_joint_position()` then `get_observation()` to "
            "obtain a clear view before proceeding with segmentation or planning."
        )
        if self._primary_skill_id:
            parts.append(
                f"This sub-goal corresponds to the high-level skill "
                f"`{self._primary_skill_id}`. Start with "
                f'`{{"tool":"use_skill","args":{{"name":"{self._primary_skill_id}"}}}}` '
                "to read how it orchestrates the work, then consult and freely combine the "
                "observation/action leaf skills it points to -- decide the order and the "
                "code yourself from each skill's guidance; there is no fixed pipeline."
            )
        if self._feedback:
            parts.append(
                "Feedback from a previous unresolved attempt/review. Use it to fix your approach:\n"
                f"{self._feedback}"
            )
        text = "\n\n".join(parts)

        if self._scene_image_path:
            return [
                {"type": "text", "text": text},
                image_content_part(self._scene_image_path),
            ]
        return text

    def _block_metadata(self) -> dict:
        return {"task": self._task}

    def _agent_role(self) -> str:
        return "act"

    def _agent_label(self) -> str:
        return "Act Agent"

    def _python_gate_message(self, code: str, loaded: tuple[str, ...]) -> str:
        if loaded:
            return ""
        available = ", ".join(e.name for e in self._skill_entries())
        if not available:
            return (
                "Python execution is blocked because no RoboMEx skills are available to this "
                "Act Agent. This is a framework/configuration error: the skill library is empty."
            )
        return (
            "Python execution is blocked until you consult a relevant RoboMEx skill. "
            "Reply with exactly one use_skill JSON action first, choosing from the available "
            f"skills: {available}. After reading the skill, write code that follows its "
            "Procedure / Rules / Failure modes."
        )

    def _feedback_message(self, execution: BlockExecutionResult) -> str | list:
        from robomex.perception.render import image_content_part, save_rgb

        text = (
            f"stdout:\n{execution.stdout}\n\nstderr:\n{execution.stderr}\n\n"
            "Continue from the loaded skill guidance. Reply with exactly one JSON action: "
            "use_skill if the next step needs different guidance, run_python for the next "
            "code block, or finish."
        )
        obs = execution.observation
        if obs and self._video_dir:
            try:
                rgb = obs["agentview"]["images"]["rgb"]
                img_path = save_rgb(
                    self._video_dir / f"act_obs_latest.png", rgb
                )
                return [
                    {"type": "text", "text": text},
                    image_content_part(img_path),
                ]
            except (KeyError, TypeError, OSError):
                pass
        return text

    def _on_python_turn(
        self,
        turn_idx: int,
        code: str,
        execution: BlockExecutionResult,
        prev_observation: dict | None,
        turns: list[Any],
    ) -> None:
        if self.collector is not None:
            # 逐块存 before/after 帧,作为调试产物和 planner 复盘材料。
            self.collector.bundle_for_block(
                execution.block.name, prev_observation, execution.observation
            )
        self._save_block_clip(turn_idx, execution)
        _log.info(
            "turn %d: exec=%s reward=%s terminated=%s",
            turn_idx, execution.status.value, execution.reward, execution.terminated,
        )
        stderr = (execution.stderr or "").strip()
        if not execution.ok and stderr:
            _log.info("turn %d: 报错 -> %s", turn_idx, stderr.splitlines()[-1][:300])
        turns.append(TurnRecord(turn_idx, code, execution, None))

    def _save_block_clip(self, turn_idx: int, execution: BlockExecutionResult) -> None:
        """有动作的 block 才存视频:把这一块产生的帧区间写成 ``turn_NN.mp4``。

        帧区间由适配器记在 ``execution.info['video_range']`` 里(没动作 → 没区间 → 不存)。
        路径 + 区间一并记进 ``self._clips``,稍后随 ``trace.metadata['clips']`` 交给 Planner/日志。
        """

        if self._video_dir is None:
            return
        rng = (execution.info or {}).get("video_range")
        env = getattr(self.executor, "env", None)
        if not rng or env is None or not hasattr(env, "get_video_frames_range"):
            return
        start, end = int(rng[0]), int(rng[1])
        try:
            frames = env.get_video_frames_range(start, end)
        except Exception as exc:  # noqa: BLE001 - 取帧失败不该中断子目标
            _log.warning("turn %d: 取过程帧失败 -> %r", turn_idx, exc)
            return
        if not frames:
            return
        path = self._video_dir / f"turn_{turn_idx:02d}.mp4"
        try:
            saved = save_video(path, frames)
        except Exception as exc:  # noqa: BLE001 - 写视频失败不该中断子目标
            _log.warning("turn %d: 写过程视频失败 -> %r", turn_idx, exc)
            return
        if saved:
            self._clips.append({"turn": turn_idx, "path": saved, "start": start, "end": end})
            _log.info("turn %d: 已存过程视频 %s (%d 帧)", turn_idx, saved, len(frames))

    def _should_stop_after_python(self, execution: BlockExecutionResult) -> bool:
        return bool(execution.terminated)

    def _on_terminal_turn(
        self,
        turn_idx: int,
        raw: str,
        turns: list[Any],
        loaded: tuple[str, ...],
    ) -> tuple[bool, str]:
        return True, ""

    def _finalize(self, *, turns: list[Any], loaded: tuple[str, ...], terminal_raw: str | None) -> AgentTrace:
        success = terminal_raw is not None
        act_status = "finished" if success else "exhausted"
        unresolved = None
        if not success:
            unresolved = {
                "subgoal": self._task,
                "status": "unresolved",
                "skills_used": list(loaded),
                "last_state_summary": "Act exhausted its turn budget before finish.",
                "suggested_recovery": "Re-plan from the current scene.",
            }
        return AgentTrace(
            task=self._task,
            loaded_skill_ids=loaded,
            turns=tuple(turns),
            success=success,
            metadata={
                "clips": tuple(self._clips),
                "unresolved": unresolved,
                "act_status": act_status,
                "terminal_raw": terminal_raw,
            },
        )
