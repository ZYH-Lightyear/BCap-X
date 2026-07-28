"""工具调度器。

一批 tool call 进来,一批可以直接塞回 history 的消息出去。中间管四件事:
参数校验、并发分批、输出截断、以及**「进必有出」不变式**。

## 不变式:每个 tool call 必须恰好有一条 tool 消息回应

OpenAI 的协议要求 assistant 消息里的每个 ``tool_calls[i].id`` 都有一条对应的
``role: "tool"`` 消息。少一条,下一轮请求整个会被端点拒绝,而报错信息通常只说
"invalid request",极难定位。所以 :meth:`ToolScheduler.execute` 的所有分支——
工具不存在、参数校验失败、执行抛异常、并发批次里某个任务炸了——都必须产出结果。
这条不变式由单测锁定。

## 多模态结果的处理

OpenAI 的 ``role: "tool"`` 消息只接受字符串 content,塞不进图片。所以带图的工具
结果被拆成两条消息:一条 tool 消息装文字,紧跟一条 user 消息装图片。这是与
Qwen-Code 的一处**刻意偏离** —— 它面向 Gemini 的 functionResponse,那边原生支持
在工具响应里放 inlineData。在 OpenAI wire format 上没有等价物,只能这样绕。
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from agentx.contracts import (
    CONCURRENCY_SAFE_KINDS,
    ContentPart,
    Message,
    ToolCall,
    ToolRecord,
    ToolResult,
    ToolValidationError,
)
from agentx.tools.base import Tool, ToolContext
from agentx.tools.registry import ToolRegistry
from agentx.truncation import truncate_output

MAX_PARALLEL_TOOLS = 8
#: 同一个工具连续多少次报同样的校验错误就强制改道。抄自 Qwen-Code 的
#: VALIDATION_RETRY_LOOP_THRESHOLD。三次足以区分「手滑」和「不理解 schema」。
VALIDATION_RETRY_THRESHOLD = 3
RETRY_LOOP_DIRECTIVE = (
    "⚠️ 检测到重试循环:你已经连续 {count} 次用同样的错误参数调用 {tool}。"
    "不要再用相同参数重试。请改用别的方式,或说明你卡在哪里。"
)


@dataclass
class ScheduleOutcome:
    """一批工具调用的执行结果。"""

    #: 可直接追加进 history 的消息(tool 消息,以及承载图片的 user 消息)。
    messages: list[Message]
    records: list[ToolRecord]

    @property
    def any_error(self) -> bool:
        return any(record.result.is_error for record in self.records)


class ToolScheduler:
    """按批执行工具调用。"""

    def __init__(
        self,
        registry: ToolRegistry,
        ctx: ToolContext,
        *,
        max_parallel: int = MAX_PARALLEL_TOOLS,
    ) -> None:
        self.registry = registry
        self.ctx = ctx
        self.max_parallel = max_parallel
        # (工具名, 错误文本) -> 连续出现次数。任何一次成功调用都会清空整张表,
        # 因为「连续」才是熔断的判据 —— 只增不减的计数器迟早会误伤正常运行。
        self._validation_failures: dict[tuple[str, str], int] = {}

    def execute(self, calls: tuple[ToolCall, ...]) -> ScheduleOutcome:
        """执行一批调用,返回可回灌的消息。"""

        deduped = _dedupe_by_id(calls)
        prepared = [self._prepare(call) for call in deduped]

        records: list[ToolRecord] = []
        for batch in _partition(prepared):
            records.extend(self._run_batch(batch))

        if any(not record.result.is_error for record in records):
            self._validation_failures.clear()

        messages: list[Message] = []
        for record in records:
            messages.extend(self._to_messages(record))
        return ScheduleOutcome(messages=messages, records=records)

    def _prepare(self, call: ToolCall) -> _Prepared:
        """把一次调用变成「可执行的 invocation」或「已经定案的失败」。"""

        if call.parse_error:
            return _Prepared(call, None, None, self._note_failure(call.name, call.parse_error))

        tool = self.registry.get(call.name)
        if tool is None:
            available = ", ".join(self.registry.names())
            message = f"没有名为 {call.name!r} 的工具。可用工具:{available}"
            return _Prepared(call, None, None, ToolResult(llm_content=message, error=message))

        try:
            invocation = tool.build(call.args)
        except ToolValidationError as exc:
            return _Prepared(call, tool, None, self._note_failure(call.name, str(exc)))
        except Exception as exc:  # noqa: BLE001 - 工具构造里的 bug 不该拖垮整轮
            message = f"构造 {call.name} 调用时出错: {exc}"
            return _Prepared(call, tool, None, ToolResult(llm_content=message, error=message))

        return _Prepared(call, tool, invocation, None)

    def _note_failure(self, tool_name: str, message: str) -> ToolResult:
        """记录一次校验失败,连续三次同样的错误就追加改道指令。"""

        key = (tool_name, message)
        count = self._validation_failures.get(key, 0) + 1
        self._validation_failures[key] = count

        body = message
        if count >= VALIDATION_RETRY_THRESHOLD:
            body += "\n\n" + RETRY_LOOP_DIRECTIVE.format(count=count, tool=tool_name)
        return ToolResult(llm_content=body, error=message)

    def _run_batch(self, batch: list[_Prepared]) -> list[ToolRecord]:
        runnable = [item for item in batch if item.invocation is not None]

        # 已定案的失败不需要执行,直接成记录。
        records = [
            ToolRecord(call=item.call, result=item.failure, duration_ms=0.0)
            for item in batch
            if item.failure is not None
        ]

        if len(runnable) <= 1:
            records.extend(self._run_one(item) for item in runnable)
            return records

        with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(runnable))) as pool:
            records.extend(pool.map(self._run_one, runnable))
        return records

    def _run_one(self, item: _Prepared) -> ToolRecord:
        assert item.invocation is not None
        started = time.monotonic()
        try:
            result = item.invocation.execute(self.ctx)
        except Exception as exc:  # noqa: BLE001 - 兜住工具里的任何 bug
            detail = traceback.format_exc(limit=5)
            message = f"{item.call.name} 执行时抛出异常: {exc}\n{detail}"
            result = ToolResult(llm_content=message, display=message, error=str(exc))
        duration_ms = (time.monotonic() - started) * 1000.0
        return ToolRecord(call=item.call, result=result, duration_ms=duration_ms)

    def _to_messages(self, record: ToolRecord) -> list[Message]:
        """把一条记录变成回灌消息,顺带截断。"""

        tool = self.registry.get(record.call.name)
        text, images = _split_content(record.result.llm_content)
        text = truncate_output(
            text,
            max_chars=tool.max_output_chars if tool else None,
            keep=tool.truncate_keep if tool else "both",
            overflow_dir=self.ctx.workspace / "overflow",
            label=record.call.name,
        )

        messages: list[Message] = [
            {"role": "tool", "tool_call_id": record.call.id, "content": text or "(empty)"}
        ]
        if images:
            # 图片只能挂在 user 消息上,见模块 docstring。加一句说明,免得模型
            # 以为这是用户的新指令。
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"({record.call.name} 返回的图片)"},
                    *images,
                ],
            })
        return messages


@dataclass
class _Prepared:
    call: ToolCall
    tool: Tool | None
    invocation: object | None
    failure: ToolResult | None


def _dedupe_by_id(calls: tuple[ToolCall, ...]) -> list[ToolCall]:
    """按 id 去重。模型偶尔会在一条回复里重复同一个 call id。"""
    seen: set[str] = set()
    unique: list[ToolCall] = []
    for call in calls:
        if call.id in seen:
            continue
        seen.add(call.id)
        unique.append(call)
    return unique


def _partition(prepared: list[_Prepared]) -> list[list[_Prepared]]:
    """把调用切成若干批:连续的只读工具合成一个并行批,其余各自单独一批。

    保持原有顺序很重要 —— 模型经常依赖「先写文件再跑测试」这样的隐含时序,
    重排会静默地破坏它。
    """
    batches: list[list[_Prepared]] = []
    current: list[_Prepared] = []

    for item in prepared:
        safe = item.tool is not None and item.tool.kind in CONCURRENCY_SAFE_KINDS
        if safe:
            current.append(item)
            continue
        if current:
            batches.append(current)
            current = []
        batches.append([item])

    if current:
        batches.append(current)
    return batches


def _split_content(content: str | list[ContentPart]) -> tuple[str, list[ContentPart]]:
    """把工具结果拆成文字部分和图片部分。"""
    if isinstance(content, str):
        return content, []

    texts: list[str] = []
    images: list[ContentPart] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(str(part.get("text", "")))
        elif part.get("type") == "image_url":
            images.append(part)
    return "\n".join(texts), images


__all__ = ["MAX_PARALLEL_TOOLS", "ScheduleOutcome", "ToolScheduler"]
