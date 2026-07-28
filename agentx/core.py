"""AgentCore:推理循环。

照抄 Qwen-Code ``agents/runtime/agent-core.ts`` 的 ``_runReasoningLoopInner``。
选它而不是 CLI 那条 ``GeminiClient`` 路径,是因为 CLI 那套把压缩触发、loop 检测、
next-speaker 续写、hook 和 React 渲染状态全搅在一起;AgentCore 是 headless 自包含
的,子 agent 就跑在它上面,翻译成 Python 几乎是一比一。

循环本身很短::

    观测一次(开局)
    while True:
        检查取消 / 轮次上限 / 时间上限
        response = chat.send(tools)
        if response.tool_calls:  执行、回灌结果、再观测一次
        elif response.text:      完成
        else:                    催一句,继续

注意**没有 finish 工具**。模型不再调工具并给出文本,就是完成。少一个终止工具就
少一个模型可能用错的契约面,也省掉「调了 finish 但任务没做完」这类分歧。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agentx.chat import ChatSession
from agentx.contracts import AgentResult, Message, TerminateMode, ToolRecord
from agentx.scheduler import ToolScheduler

#: 模型既不调工具也不说话时的催促语。空回复通常是模型在等一个它以为存在的信号,
#: 直接终止会把一次本可挽救的运行判死。
NUDGE = "你没有输出任何内容。请继续完成任务,或给出最终答复。"

#: 「最终答复」里命中这些模式,说明模型是在编造一段对话剧本,而不是真的做完了。
#: 实测中某些路由会间歇性掉出原生 tool call 模式,转而在正文里把「调用」和它自己
#: 想象的「调用结果」交替写出来 —— 那段文本读起来像模像样,却完全没有真实执行过。
#:
#: 用正则而不是字面量:同一个模型会在 ``**Tool Call:``、``**Tool:``、``<tool_call>``
#: 之间随机切换大小写和写法,字面量匹配漏一个就等于整条守卫失效。
HALLUCINATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"<\s*/?\s*tool_(call|use|result)\s*>",
        r"<\s*system\s*>",
        r"\*\*\s*tool[\s_]*(call)?\s*:",
        r"tool_call\s+error",
        r"^\s*(assistant|system|tool)\s*:",
    )
)

HALLUCINATION_REBUKE = (
    "⚠️ 你上一条回复在正文里编造了工具调用和它们的执行结果,但**没有一个真实执行过**。\n"
    "不要凭空写出工具结果。请通过真正的工具通道发起调用,然后等待真实结果返回。"
)


@dataclass
class RunConfig:
    """一次运行的预算。"""

    max_turns: int = 40
    max_time_s: float = 1800.0
    #: 每完成一轮回调一次,用于外部打印进度或落盘 trace。
    on_turn: Callable[[int, "TurnSummary"], None] | None = None
    #: 无条件观测钩子。收到 ``(轮次, 本轮工具记录)``,返回一条要追加进 history 的
    #: 观测消息;返回 ``None`` 表示这一轮没有新观测(例如环境状态根本没变)。
    #:
    #: 做成钩子而不是工具,是为了让「观测」不占用模型的决策面:没有任何一条路径能
    #: 让模型跳过观测,闭环因此是体制而不是它的选择。钩子里的异常不做兜底 —— 观测
    #: 源自己该把可预期的故障(取图失败之类)变成一条说明性消息返回。
    observe: Callable[[int, list[ToolRecord]], Message | None] | None = None


@dataclass
class TurnSummary:
    """一轮的概要,交给 ``on_turn`` 回调。"""

    index: int
    text: str
    tool_records: list[ToolRecord] = field(default_factory=list)


class AgentCore:
    """驱动一次 agent 运行。"""

    def __init__(
        self,
        chat: ChatSession,
        scheduler: ToolScheduler,
        tools: list[dict[str, Any]],
        config: RunConfig | None = None,
    ) -> None:
        self.chat = chat
        self.scheduler = scheduler
        self.tools = tools
        self.config = config or RunConfig()

    def run(self, task: str) -> AgentResult:
        """跑到完成或耗尽预算。"""

        self.chat.append({"role": "user", "content": task})
        self._observe(0, [])

        started = time.monotonic()
        turn = 0
        records: list[ToolRecord] = []

        while True:
            if turn >= self.config.max_turns:
                return self._result(
                    "", TerminateMode.MAX_TURNS, turn, records,
                    f"达到轮次上限 {self.config.max_turns}",
                )
            elapsed = time.monotonic() - started
            if elapsed >= self.config.max_time_s:
                return self._result(
                    "", TerminateMode.TIMEOUT, turn, records,
                    f"达到时间上限 {self.config.max_time_s:.0f}s",
                )

            turn += 1
            try:
                response = self.chat.send(self.tools)
            except KeyboardInterrupt:
                return self._result("", TerminateMode.CANCELLED, turn, records, "被用户中断")
            except Exception as exc:  # noqa: BLE001 - provider 的任何故障都在这里收口
                return self._result(
                    "", TerminateMode.ERROR, turn, records, f"{type(exc).__name__}: {exc}"
                )

            summary = TurnSummary(index=turn, text=response.text)

            if response.tool_calls:
                outcome = self.scheduler.execute(response.tool_calls)
                self.chat.extend(outcome.messages)
                records.extend(outcome.records)
                summary.tool_records = outcome.records
                self._observe(turn, outcome.records)
                self._notify(summary)
                continue

            self._notify(summary)

            text = response.text.strip()
            if not text:
                self.chat.append({"role": "user", "content": NUDGE})
                continue

            # 「不调工具 + 有文本」是完成信号,但前提是这段文本真的是答复。模型偶尔
            # 会在正文里编造一整段带假结果的对话;把它当成 GOAL 会让运行以成功状态
            # 退出,还把幻觉当最终答案交出去 —— 这是最坏的一种失败,因为它不报错。
            if _looks_hallucinated(text):
                self.chat.append({"role": "user", "content": HALLUCINATION_REBUKE})
                continue

            return self._result(text, TerminateMode.GOAL, turn, records)

    def _observe(self, turn: int, records: list[ToolRecord]) -> None:
        """把一条新鲜观测追加进 history。

        只在开局和工具执行之后调用 —— 模型不调工具时世界没有变化,再取一次观测拿到
        的是同一份内容。
        """
        if self.config.observe is None:
            return
        message = self.config.observe(turn, records)
        if message is not None:
            self.chat.append(message)

    def _notify(self, summary: TurnSummary) -> None:
        if self.config.on_turn is not None:
            self.config.on_turn(summary.index, summary)

    def _result(
        self,
        text: str,
        mode: TerminateMode,
        turns: int,
        records: list[ToolRecord],
        detail: str = "",
    ) -> AgentResult:
        return AgentResult(
            text=text,
            terminate_mode=mode,
            turns=turns,
            detail=detail,
            tool_records=list(records),
            usage=dict(self.chat.usage),
        )


def _looks_hallucinated(text: str) -> bool:
    """判断一段「最终答复」其实是模型编造的对话剧本。

    只在**没有真实工具调用**的那条路径上使用。带着真实调用的回复里出现这些标记是
    正常的(比如模型在解释某个工具怎么用),不该拦。

    误判的代价是多一轮往返,漏判的代价是把一份幻觉当成功交出去。两者不对称,所以
    这里宁可判得宽一些。
    """
    return any(pattern.search(text) for pattern in HALLUCINATION_PATTERNS)


__all__ = [
    "AgentCore",
    "HALLUCINATION_PATTERNS",
    "HALLUCINATION_REBUKE",
    "NUDGE",
    "RunConfig",
    "TurnSummary",
]
