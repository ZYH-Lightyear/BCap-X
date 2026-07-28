"""环境端口:出真图,不执行。

闭环的两条边中,本模块负责"环境 → 观测"这一条:每拍从 LIBERO-pro 仿真器渲染
真实画面并落盘,交给 planner 看。

而"意图 → 动作"那一条**是断的**。执行层(Coding Agent / robot port)尚未重建,
:meth:`LiberoEnv.apply` 不会向仿真器发出任何控制量,只是原样重渲染当前场景,
并回一个 status 恒为 ``not_executed`` 的反馈。

这个组合是刻意的:先让 planner 真正开始"看",再谈让它"动"。图片必须是真的,
否则单步纪律就只是 prompt 里的一句空话 —— 模型没有观测可依,除了凭空展开流程
别无选择。而反馈必须诚实地说"没有执行成功",否则模型会顺着自己上一步的设想
往下编,把开环幻觉当成执行结果。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from robomex.contracts import ActionIntent, IntentFeedback, Observation
from robomex.images import save_rgb

#: 执行层接通前,环境对每一个动作意图的统一答复。措辞直白到不容误读:模型必须
#: 明白场景没有因为它的意图而改变,不能把 expected_effect 当成既成事实。
NOT_EXECUTED_NOTE = (
    "没有执行成功。执行层尚未接通,上一个动作意图没有被真实执行,"
    "画面与上一拍相同。"
)


class EnvironmentPort(Protocol):
    """planner 循环对环境的全部要求。

    收窄到两个方法,是为了让单测能用一个不依赖仿真器的假环境替换掉 LIBERO。
    """

    #: 环境自带的任务描述(LIBERO 的 task language)。
    task_prompt: str

    def reset(self) -> Observation: ...

    def apply(self, intent: ActionIntent) -> tuple[IntentFeedback, Observation]: ...


class LiberoEnv:
    """LIBERO-pro 仿真器的只读封装:只出图,不执行。"""

    def __init__(
        self,
        suite_name: str,
        task_id: int,
        *,
        image_dir: str | Path,
        camera: str = "agentview",
        seed: int | None = None,
        max_steps: int = 8000,
        control_freq: int = 20,
    ) -> None:
        """
        :param suite_name: LIBERO 任务套件名,例如 ``libero_object_swap``。
        :param task_id: 套件内的任务序号。
        :param image_dir: 观测图片落盘目录,通常指向本次运行的 trace 目录。
        :param camera: 相机名。``agentview`` 是第三人称全景,
            ``robot0_eye_in_hand`` 是腕部相机。
        :param seed: 传给 ``reset`` 的 trial 序号,决定初始摆放。
        """
        # 延迟 import:capx 会拖进 mujoco / robosuite,不跑仿真的单测不该付这个代价。
        from capx.envs.simulators.libero import FrankaLiberoEnv

        self._env = FrankaLiberoEnv(
            suite_name=suite_name,
            task_id=task_id,
            privileged=False,
            max_steps=max_steps,
            control_freq=control_freq,
        )
        self._camera = camera
        self._seed = seed
        self._image_dir = Path(image_dir)
        self._frame_index = 0
        self.task_prompt = ""

    def reset(self) -> Observation:
        """复位仿真器,返回初始观测。"""

        _, info = self._env.reset(seed=self._seed)
        # LIBERO 自带的任务语言就是最权威的任务描述,优先于命令行手写的 --task。
        self.task_prompt = str(info.get("task_prompt") or "")
        return self._observe(note="任务初始状态。")

    def apply(self, intent: ActionIntent) -> tuple[IntentFeedback, Observation]:
        """"执行"一个动作意图 —— 实际什么都不做,只重新出图。

        :returns: ``(feedback, observation)``。feedback 的 status 恒为
            ``not_executed``;observation 是重新渲染的当前画面。
        """

        feedback = IntentFeedback(status="not_executed", summary=NOT_EXECUTED_NOTE)
        return feedback, self._observe(note=NOT_EXECUTED_NOTE)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def _observe(self, *, note: str) -> Observation:
        """渲染当前画面并落盘,组装成 :class:`Observation`。"""

        obs: dict[str, Any] = self._env.get_observation()
        try:
            rgb = obs[self._camera]["images"]["rgb"]
        except KeyError as exc:
            available = sorted(obs)
            raise KeyError(
                f"camera {self._camera!r} has no RGB image; available cameras: {available}"
            ) from exc

        path = save_rgb(self._image_dir / f"obs-{self._frame_index:03d}.png", rgb)
        self._frame_index += 1
        return Observation(image_path=path, camera=self._camera, note=note)


__all__ = ["NOT_EXECUTED_NOTE", "EnvironmentPort", "LiberoEnv"]
