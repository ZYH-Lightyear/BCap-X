"""Reactive Planner 的最小契约。

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
    """

    intent_id: str
    instruction: str
    #: 完成后世界应该变成什么样。不是描述性装饰 —— 执行层接通后,它是拿新观测
    #: 比对的验证锚点;不符即说明发生了腐化或意外,planner 应改变计划。
    expected_effect: str


@dataclass(frozen=True)
class IntentFeedback:
    """执行层对一个 ActionIntent 的回灌。

    Phase 1 里 status 恒为 ``not_executed``,因为 planner 以下尚未接通。
    """

    intent_id: str
    status: Literal["succeeded", "failed", "not_executed"]
    summary: str = ""
    #: 失败明细/traceback —— planner 改主意的主要依据。
    detail: str = ""


@dataclass(frozen=True)
class PlannerStep:
    """一次 planner 决策:要么给出下一个 ActionIntent,要么宣告任务完成。"""

    thought: str
    intent: ActionIntent | None
    done: bool
    reason: str = ""
