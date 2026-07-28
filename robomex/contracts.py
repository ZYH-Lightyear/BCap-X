"""Reactive Planner 的最小契约。

四个数据类构成一拍闭环:planner 看到 :class:`Observation`,产出
:class:`ActionIntent`(包在 :class:`PlannerStep` 里),环境回一份
:class:`IntentFeedback` 和新的 :class:`Observation`。

字段预算纪律:任何字段没有当下的消费者就不允许存在。

命名沿用 ``ActionIntent`` 而非 ``Subgoal``:它承载的是"下一个动作意图",是闭环
里每拍重新决策的产物,而不是一份预先拆解好的任务清单中的一项。参见
``docs/robomex_high_level_design.md``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ActionIntent:
    """Reactive Planner 每次决策产出的**唯一一个**动作意图。

    粒度取"单次视觉落地的有效性可以覆盖的最大动作单元",接近人类操作语义,
    例如「移动到瓶盖上方」「抓住瓶盖」「抬起瓶子」。

    没有 id 字段:同一时刻在途的动作意图永远只有一个(这是单步纪律的直接推论),
    因此不存在需要靠 id 消歧的对象。序号由 history 的下标和 trace 的 step-N
    目录承担。
    """

    instruction: str
    #: 完成后世界应该变成什么样。不是描述性装饰 —— 执行层接通后,它是拿新观测
    #: 比对的验证锚点;不符即说明发生了腐化或意外,planner 应改变计划。
    expected_effect: str


@dataclass(frozen=True)
class IntentFeedback:
    """执行层对一个 ActionIntent 的回灌。

    当前 status 恒为 ``not_executed``:执行层(Coding Agent / robot port)尚未
    重建,环境只负责出图,没有任何东西真的动过。

    同样没有 id:反馈总是与它所回应的那个意图成对出现(``env.apply`` 收谁就回谁,
    history 里存的也是序对),关联关系由结构表达,不需要字段来重述。
    """

    status: Literal["succeeded", "failed", "not_executed"]
    summary: str = ""
    #: 失败明细/traceback —— planner 改主意的主要依据。
    detail: str = ""


@dataclass(frozen=True)
class Observation:
    """环境在某一拍返回的观测。

    闭环的另一半:planner 每次决策前都要重新看一眼,而不是沿着上一次的设想续接。
    图片以磁盘路径的形式传递而非内存数组,这样送进模型的那一份和 trace 里留档的
    那一份必然是同一张,事后可以直接打开核对。
    """

    #: LIBERO 相机视角 RGB 的落盘路径。
    image_path: str
    #: 相机名,写进 prompt 让模型知道自己在从哪个视角看。
    camera: str = "agentview"
    #: 环境对这一拍的文字说明。执行层接通前恒为「没有执行成功」类的说明。
    note: str = ""


@dataclass(frozen=True)
class PlannerStep:
    """一次 planner 决策:要么给出下一个 ActionIntent,要么宣告任务完成。"""

    thought: str
    intent: ActionIntent | None
    done: bool
    reason: str = ""
