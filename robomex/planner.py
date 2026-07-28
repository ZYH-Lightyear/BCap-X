"""Reactive Planner:任务 + 当前画面进,**一个** ActionIntent 出。

本模块是 RoboMEx 重建后的唯一一层。它每次被调用只做一件事:看一眼环境当前的
画面,决定**下一个**动作意图(例如「移动到瓶盖上方」),然后停下。

## 单步纪律:为什么不是"拆解成一串"

Code 是 VLA 中"动作"位置上的中间动作语言,因而必须继承 VLA 的闭环时序特性:
每一拍重新看,而不是一次展开未来若干步再闭眼执行。物理信息在
视觉 → 语言 → 代码变量 → 动作 的传递中会逐级腐化,预先展开的流程一旦遇到
遮挡、位姿偏移、目标掉落,就会继续忠实地执行一个已经失效的策略。

所以 planner **不是** task decomposer。它不输出计划清单,只输出下一步。
:data:`PLANNER_SYSTEM` 里对此有明确禁令,并由单测锁定,防止措辞回退成
"拆解成一串 / 编写指令序列"那种开环框架。详见
``docs/robomex_high_level_design.md``。

## 当前接通了哪一半

闭环由两条边组成,现在一条通、一条断:

- **观测边是通的**:``step()`` 接收 :class:`~robomex.contracts.Observation`,
  里面是 LIBERO-pro 刚刚渲染的真实画面,以多模态 image part 的形式随 prompt
  一起送进模型。这是单步纪律能够成立的物质前提 —— 没有画面可看,模型除了凭空
  展开流程别无选择。
- **执行边是断的**:Coding Agent 与 robot port 尚未重建,环境不会因为动作意图
  而改变,反馈 status 恒为 ``not_executed``。这件事靠数据本身告知模型 —— 每条
  历史记录都带着 ``feedback_status`` 与环境那句「没有执行成功」,而不是在
  system prompt 里硬写一句声明。声明会随执行层接通而过期,数据不会。

由此产生一个看起来奇怪、实则正确的现象:连续多步运行时,planner 会反复给出
同一个动作意图。因为画面确实没变,该做的那一步确实还没做。这正是闭环在执行层
缺失下应有的表现,不是 bug。

## 设计约束

- **无状态**:planner 自身不保存历史与观测,``history`` / ``observation`` 每次
  由调用方完整传入。这样单测可以随意构造任意历史,不必按顺序驱动。
  ``last_prompt`` / ``last_response`` 是唯一的实例状态,且仅供 trace 落盘,
  不参与决策。
- **宽容解析**:模型偶尔会用 markdown 围栏包住 JSON、或在 JSON 前后加寒暄。
  这些都吃下,但缺字段不放过,见 :meth:`ReactivePlanner._parse_response`。
"""

from __future__ import annotations

import json
from typing import Any

from robomex.contracts import ActionIntent, IntentFeedback, Observation, PlannerStep
from robomex.images import image_content_part
from robomex.policy import CompletionPolicy

# 发送给模型的 system prompt。这是**功能性内容**而非注释,用中文书写以便人工
# 对着 trace 里的 prompt.json 直接调试。两类字面量刻意保留英文:JSON 的键名
# (thought/intent/instruction/expected_effect/done/reason)与状态值
# ``not_executed`` —— 它们是机器契约,与 robomex/contracts.py 一一对应,翻译了
# 就对不上了。单步纪律那几句由单测锁定,防止措辞回退成开环框架。
PLANNER_SYSTEM = (
    "你是一个机器人操作系统的反应式规划器(Reactive Planner)。\n\n"
    "你每次被调用,只决定**下一个**动作意图(ActionIntent),然后停下。\n"
    "动作意图是细粒度的、接近人类操作语义的一步,例如「移动到瓶盖上方」\n"
    "「抓住瓶盖」「抬起瓶子」。\n\n"
    "单步纪律:\n"
    "- 你不是任务拆解器。不要一次性把任务展开成流程、清单或步骤序列。\n"
    "- 只给出眼下最该做的那一步。\n"
    "- 你的判断必须基于随消息附上的当前场景图片,而不是你上一步的设想。\n\n"
    "粒度把握:一个动作意图应当是「看一眼就能安全推进完」的最大动作单元。\n"
    "需要重新观察才能继续时,就该切开。「把瓶子拿起来」太粗(跨越了抓取前后\n"
    "两个不同状态);「关节 3 转 0.1 弧度」太细(属于代码内部的事)。\n\n"
    "当前系统状态,必须如实对待:\n"
    "- 图片是环境刚刚渲染出来的真实画面,可以完全信任。\n"
    "- 所以不要臆造执行结果。如果画面显示该做的那一步仍然没做,就照实再给出\n"
    "  那一步 —— 这不是失误,而是闭环在执行层缺失时应有的表现。注意:这**不**\n"
    "  意味着你应该改为一次性输出整套流程,单步纪律始终有效。\n\n"
    "只回复一个 JSON 对象,不要包裹在 markdown 代码块里:\n"
    "- 给出下一个动作意图:{\"thought\": \"你的判断\", \"intent\": {\"instruction\": \"这一步要做什么\", \"expected_effect\": \"完成后世界会变成什么样\"}}\n"
    "- 任务已经完成:{\"thought\": \"你的判断\", \"done\": true, \"reason\": \"结束理由\"}\n"
)


class ReactivePlanner:
    """由 LLM 驱动的无状态反应式规划器。

    每次 :meth:`step` 只产出一个 ActionIntent。外层拿一个循环反复调用它,把最新
    观测和上一步的执行反馈喂回来,直到 ``done`` 为真::

        planner = ReactivePlanner(policy)
        observation = env.reset()
        history = []
        while True:
            step = planner.step(task=task, observation=observation, history=tuple(history))
            if step.done:
                break
            feedback, observation = env.apply(step.intent)
            history.append((step.intent, feedback))
    """

    def __init__(
        self,
        policy: CompletionPolicy,
        *,
        max_intents: int = 20,
        image_max_edge: int = 1024,
    ) -> None:
        """
        :param policy: LLM 补全策略,见 :class:`robomex.policy.CompletionPolicy`。
        :param max_intents: 硬上限。达到后 :meth:`step` 直接返回 ``done=True``
            而**不再调用模型**,避免模型不肯收尾时无限烧 token。
        :param image_max_edge: 观测图片送进 prompt 前的最长边上限。观测每拍都要
            重发一次,不设上限会在多步运行里迅速吃满上下文。
        """
        self.policy = policy
        self.max_intents = max_intents
        self.image_max_edge = image_max_edge
        # 最近一次与模型的交互,仅供 trace.py 落盘 prompt.json / response.txt。
        # 首次模型调用之前为 None;上限熔断路径不调用模型,也会被清成 None。
        self.last_prompt: list[dict[str, Any]] | None = None
        self.last_response: str | None = None

    def step(
        self,
        *,
        task: str,
        observation: Observation | None = None,
        history: tuple[tuple[ActionIntent, IntentFeedback], ...] = (),
    ) -> PlannerStep:
        """决定下一个 ActionIntent(或宣告任务已完成)。

        :param task: 高层任务的自然语言描述。
        :param observation: 环境当前的观测。给了就以多模态 image part 的形式随
            prompt 送进模型。允许为 ``None`` 纯粹是为了让不关心视觉的单测少搭
            一套图片脚手架;真实运行必须传。
        :param history: 已产出的 ``(intent, feedback)`` 序对,按时间顺序排列。
            空元组表示这是第一步。执行层接通前所有 feedback 的 status 都是
            ``not_executed``,但结构上完全支持 ``failed`` + 失败明细驱动改计划。
        :returns: 一个 :class:`PlannerStep`;``done=False`` 时 ``intent`` 必非空。
        :raises ValueError: 模型回复无法解析,或缺少 ``instruction`` /
            ``expected_effect`` 必填字段。异常抛出前 ``last_response`` 已被赋值,
            因此调用方仍可把原始回复落盘用于排查。
        """
        # 上限熔断:不调用模型,同时清空上次交互痕迹,避免 trace 把上一步的
        # prompt/response 误记到这一步名下。
        if len(history) >= self.max_intents:
            self.last_prompt = None
            self.last_response = None
            return PlannerStep(
                thought="已达到动作意图数量上限,停止规划。",
                intent=None,
                done=True,
                # reason 保留 max_intents 字面量:它是熔断原因的机器可识别标记。
                reason="max_intents exceeded",
            )

        prompt = self._build_prompt(task, observation, history)
        # 先记 prompt 再调模型:即使模型调用抛异常,trace 也能还原"当时问了什么"。
        self.last_prompt = prompt
        self.last_response = None
        raw = str(self.policy.complete(prompt) or "").strip()
        # 同理,先记原始回复再解析。解析失败时最需要看的就是这段原文。
        self.last_response = raw
        return self._parse_response(raw)

    def _build_prompt(
        self,
        task: str,
        observation: Observation | None,
        history: tuple[tuple[ActionIntent, IntentFeedback], ...],
    ) -> list[dict[str, Any]]:
        """把任务、观测与历史组装成两条消息的 chat prompt。

        历史以 JSON 数组的形式塞进 user 消息,而不是拼成多轮 assistant/user 对话。
        原因是 planner 无状态:同一段历史无论以什么顺序被构造出来,都应该得到
        完全相同的 prompt,便于单测与复现。
        """
        issued = []
        for intent, feedback in history:
            entry: dict[str, Any] = {
                "instruction": intent.instruction,
                "expected_effect": intent.expected_effect,
                "feedback_status": feedback.status,
            }
            # summary / detail 为空时整个键都不出现,避免大量 "": "" 噪声稀释
            # 真正有信息量的失败明细(detail 是 planner 改主意的主要依据)。
            if feedback.summary:
                entry["feedback_summary"] = feedback.summary
            if feedback.detail:
                entry["feedback_detail"] = feedback.detail
            issued.append(entry)

        user: dict[str, Any] = {"task": task}
        if observation is not None:
            # 图片本身作为独立的 image part 附在后面,这里只放它的元信息,
            # 让模型知道这一眼是从哪个视角看的、环境对这一拍有什么说明。
            user["observation"] = {
                "camera": observation.camera,
                "note": observation.note,
            }
        if issued:
            # 键名刻意叫 issued_intents 而非 completed_*:它们只是已下发,
            # 并未真正执行,措辞上不给模型"已完成"的暗示。
            user["issued_intents"] = issued

        # ensure_ascii=False:中文任务描述保持可读,便于人工核查 trace。
        text = json.dumps(user, ensure_ascii=False)

        # 没有观测时保持纯字符串 content。多模态列表只在真的有图片时才用,
        # 免得给不支持列表 content 的后端平白增加一层解包负担。
        content: Any = text
        if observation is not None:
            content = [
                {"type": "text", "text": text},
                image_content_part(observation.image_path, max_edge=self.image_max_edge),
            ]

        return [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _parse_response(raw: str) -> PlannerStep:
        """把模型原始文本解析成 :class:`PlannerStep`。

        解析刻意宽容:markdown 围栏包裹的 JSON、JSON 前后的寒暄文字、多余字段
        都能吃下。但 ``instruction`` 与 ``expected_effect`` 缺失属于硬错误——
        少了它们产出的 intent 对下游毫无意义,静默放过只会把问题推迟到更难
        排查的地方,所以宁可当场抛出带原文片段的异常。
        """
        payload = _extract_json_object(raw)
        if payload is None:
            raise ValueError(f"Planner returned unparseable response: {raw[:200]}")

        thought = str(payload.get("thought") or "")

        # done 分支优先判断:任务完成时不应再要求 intent 字段。
        if payload.get("done") is True:
            return PlannerStep(
                thought=thought,
                intent=None,
                done=True,
                reason=str(payload.get("reason") or ""),
            )

        # 容忍两种偏差:字段平铺在顶层,以及模型沿用旧称 "subgoal" 作为键名。
        intent_data = payload.get("intent") or payload.get("subgoal") or payload
        instruction = str(intent_data.get("instruction") or "").strip()
        expected_effect = str(intent_data.get("expected_effect") or "").strip()

        # 报错信息带上实际收到的 payload 片段,否则光说"缺字段"无从下手。
        if not instruction:
            raise ValueError(
                "Planner response missing 'instruction'. "
                f"Got: {json.dumps(payload, ensure_ascii=False)[:300]}"
            )
        if not expected_effect:
            raise ValueError(
                "Planner response missing 'expected_effect'. "
                f"Got: {json.dumps(payload, ensure_ascii=False)[:300]}"
            )

        return PlannerStep(
            thought=thought,
            intent=ActionIntent(
                instruction=instruction,
                expected_effect=expected_effect,
            ),
            done=False,
        )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """尽最大努力从模型文本里抠出一个 JSON 对象;失败返回 ``None``。

    两级兜底:

    1. 从第一个 ``{`` 起用 ``raw_decode`` 增量解析。这样 JSON 后面跟的解释性
       文字不会干扰解析(``raw_decode`` 只吃掉第一个完整对象就停)。
    2. 上一步失败时交给 ``json_repair``,修补未转义引号、尾随逗号、缺失右括号
       等模型高频小错。该库缺失时静默跳过,不作为硬依赖。
    """
    start = text.find("{")
    if start < 0:
        return None

    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        pass
    else:
        # raw_decode 可能解出数组/字符串等非对象值,此时继续走 json_repair。
        if isinstance(value, dict):
            return value

    try:
        import json_repair
    except ImportError:
        # json_repair 是可选依赖,缺失时退化为"只支持第一级解析"。
        return None
    try:
        repaired = json_repair.loads(text)
    except (ValueError, TypeError):
        # 兜底路径本身不允许成为新的失败源:修不好就当没修好,交由调用方报错。
        return None
    return repaired if isinstance(repaired, dict) else None


__all__ = ["PLANNER_SYSTEM", "ReactivePlanner"]