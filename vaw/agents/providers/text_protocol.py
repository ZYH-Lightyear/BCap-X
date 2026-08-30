"""显式文本回退协议。

默认路径应把标准 tools 交给服务端，由模型自己的 chat template 与 tool parser
处理。本模块只服务于没有结构化工具通道的端点；它不冒充任何模型家族的原生格式。
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from vaw.agents.contracts import Message, ModelResponse, ToolCall
from vaw.agents.providers.base import ModelProvider

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

PROTOCOL_INSTRUCTIONS = """\
当前端点没有结构化工具通道。每轮只调用一个函数，格式为：
<tool_call>{"name":"detect_region","arguments":{"query":"红色杯子"}}</tool_call>
可在调用前写一句依据；调用后立即停止，不得编造结果。arguments 必须是 JSON 对象。
可用函数的标准 schema 位于 <tools> 中。"""


class TextProtocolProvider:
    """Downgrade native tool calling to the text protocol.

    :param inner: Underlying provider; only its completion capability is used
        (``tools`` is not forwarded).
    """

    def __init__(self, inner: ModelProvider) -> None:
        self.inner = inner

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        if not tools:
            return self.inner.generate(messages, tools=None)
        rewritten = self.prepare_messages(messages, tools)
        # Key point: do not pass tools down. An endpoint that supports native
        # calling would then have two channels open, and the model can answer
        # half in each, breaking both parsing and pairing.
        response = self.inner.generate(rewritten, tools=None)

        raw_text = response.raw_response_text or response.text
        text, calls = parse_tool_calls(raw_text)
        return ModelResponse(
            text=text,
            tool_calls=tuple(calls),
            finish_reason="tool_calls" if calls else response.finish_reason,
            usage=response.usage,
            raw_response_text=raw_text,
            provider_reasoning=response.provider_reasoning,
        )

    def prepare_messages(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[Message]:
        """返回文本协议下模型实际接收的消息，用于可观测性记录。"""

        if not tools:
            return messages
        return rewrite_history(messages, tools)


def parse_tool_calls(raw: str) -> tuple[str, list[ToolCall]]:
    """Cut tool calls out of a reply; return (remaining prose, calls).

    Only *consecutive* blocks are accepted: parsing stops at the first pair
    separated by non-blank content. That is the structural defence against a
    model fabricating results — genuine consecutive blocks touch, while a
    hallucinated follow-up call always has invented output in front of it.
    """

    matches = list(TOOL_CALL_RE.finditer(raw))
    if not matches:
        return raw.strip(), []

    accepted: list[re.Match[str]] = [matches[0]]
    for previous, current in zip(matches, matches[1:], strict=False):
        between = raw[previous.end() : current.start()]
        if between.strip():
            break
        accepted.append(current)

    calls = [_build_call(match.group(1)) for match in accepted]

    # Prose is only what came before the first call — the model's thinking.
    # Anything after is either fabricated output or a truncated follow-up call,
    # and keeping it would pollute the trace that becomes training data.
    return raw[: accepted[0].start()].strip(), calls


def _build_call(body: str) -> ToolCall:
    """Parse the JSON body of one ``<tool_call>`` block.

    Bad JSON comes back carrying ``parse_error`` rather than raising, matching
    the native path: it is an ordinary failure to hand back to the model.
    """

    call_id = f"tc_{uuid.uuid4().hex[:8]}"
    text = body.strip()
    # Models often wrap the JSON in a markdown fence; pure formatting jitter.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return ToolCall(
            id=call_id,
            name="",
            parse_error=f"<tool_call> 内容不是合法 JSON（{exc}）：{text[:200]}",
        )

    if not isinstance(parsed, dict):
        return ToolCall(
            id=call_id,
            name="",
            parse_error=f"<tool_call> 必须是 JSON 对象，实际为 {type(parsed).__name__}",
        )

    name = str(parsed.get("name") or "")
    if not name:
        return ToolCall(id=call_id, name="", parse_error="<tool_call> 缺少 name 字段")

    # Accept arguments / args / input: model families were trained on different
    # spellings, and burning a round trip on that is not worth it.
    args: Any = None
    for key in ("arguments", "args", "input"):
        if key in parsed:
            args = parsed[key]
            break

    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return ToolCall(
                id=call_id,
                name=name,
                parse_error=f"arguments 是字符串但不是合法 JSON：{args[:200]}",
            )
    if not isinstance(args, dict):
        return ToolCall(
            id=call_id,
            name=name,
            parse_error=f"arguments 必须是对象，实际为 {type(args).__name__}",
        )

    return ToolCall(id=call_id, name=name, args=args)


def rewrite_history(
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
) -> list[Message]:
    """Rewrite a history containing native tool structures into plain text.

    Not cosmetic. A ``role: "tool"`` message must be claimed by an assistant
    message carrying ``tool_calls``; under the text protocol our assistant
    messages have prose only, so every tool message looks orphaned to the
    endpoint and the request is rejected outright.

    Image parts on user messages pass through untouched — the canvas has to
    survive this rewrite.
    """

    rewritten: list[Message] = []
    pending_results: list[str] = []

    def flush_results() -> None:
        if not pending_results:
            return
        rewritten.append({"role": "user", "content": "\n".join(pending_results)})
        pending_results.clear()

    for index, message in enumerate(messages):
        role = message.get("role")

        if role == "tool":
            name = _lookup_call_name(messages, index)
            pending_results.append(
                f"<tool_result name=\"{name}\">\n{message.get('content', '')}\n</tool_result>"
            )
            continue

        flush_results()

        if role == "system":
            rewritten.append({
                "role": "system",
                "content": f"{message.get('content', '')}\n\n{_render_tool_docs(tools)}",
            })
        elif role == "assistant" and message.get("tool_calls"):
            rewritten.append({
                "role": "assistant",
                "content": _render_assistant_calls(message),
            })
        else:
            rewritten.append(dict(message))

    flush_results()
    return rewritten


def _lookup_call_name(messages: list[Message], tool_index: int) -> str:
    """Find which op a tool result belongs to, so the model can read it."""

    call_id = messages[tool_index].get("tool_call_id")
    for message in reversed(messages[:tool_index]):
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id") == call_id:
                return str((call.get("function") or {}).get("name") or "op")
    return "op"


def _render_assistant_calls(message: Message) -> str:
    parts: list[str] = []
    if message.get("content"):
        parts.append(str(message["content"]))
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw_args = function.get("arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            args = {}
        body = json.dumps(
            {"name": function.get("name", ""), "arguments": args}, ensure_ascii=False
        )
        parts.append(f"<tool_call>{body}</tool_call>")
    return "\n".join(parts)


def _render_tool_docs(tools: list[dict[str, Any]] | None) -> str:
    """紧凑渲染标准 schema；不展开重复的 Markdown 参数章节。"""

    if not tools:
        return PROTOCOL_INSTRUCTIONS + "\n\n（当前没有可用函数。）"

    rendered = "\n".join(
        json.dumps(tool, ensure_ascii=False, separators=(",", ":"))
        for tool in tools
    )
    return f"{PROTOCOL_INSTRUCTIONS}\n<tools>\n{rendered}\n</tools>"


__all__ = [
    "PROTOCOL_INSTRUCTIONS",
    "TextProtocolProvider",
    "parse_tool_calls",
    "rewrite_history",
]
