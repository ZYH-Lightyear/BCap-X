"""文本协议 provider:用 ``<tool_call>`` 标签替代原生 function calling。

包在任意一个 :class:`~agentx.providers.base.ModelProvider` 外面,对上仍然返回填好
``tool_calls`` 的 :class:`~agentx.contracts.ModelResponse`。所以 ``ChatSession``、
``AgentCore``、``ToolScheduler`` 一行都不用改 —— 这正是当初把 provider 单独切成一
层的目的。

## 为什么需要它

原生 function calling 在能用的时候更好(解码期受 schema 约束、并行调用白送、
「讨论一个调用」和「发起一个调用」在结构上可区分)。但它依赖整条链路的支持,而现实
里这条链路很脆:代理会吞 ``tools``、兼容层会翻译错、某些路由还会间歇性掉出原生模式
去编造一整段假对话。文本协议只要求端点能做补全,因此到处都能用。

## 格式选择

``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``

XML 标签配 JSON 体。标签边界好切且容错高,JSON 体与我们已有的 ``ToolCall.args``
一一对应。更关键的是这**就是 Qwen 系模型的原生训练格式** —— 探针里 qwen3-coder 在
没被要求的情况下自发吐的就是这个,说明对它而言这条路反而比 OpenAI 兼容层更贴近分布。

## 连续性规则

模型抽风时会把「调用」和「它自己编的调用结果」交替写出来。只要按空白分隔取连续的
块,就能在第一段编造的结果处自然停下:真正的批量调用是挨在一起的,而幻觉出来的后续
调用前面一定隔着一段假结果。见 :func:`parse_tool_calls`。
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from agentx.contracts import Message, ModelResponse, ToolCall
from agentx.providers.base import ModelProvider

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

PROTOCOL_INSTRUCTIONS = """
# 工具调用格式

你没有原生工具通道。要调用工具,就在回复里写一个 `<tool_call>` 块,块内是一个 JSON
对象,含 `name` 和 `arguments` 两个键:

<tool_call>{"name": "read_file", "arguments": {"path": "src/main.py"}}</tool_call>

规则:

- 需要并行调用多个工具时,把多个 `<tool_call>` 块**紧挨着**写,中间不要插入任何文字。
- 写完调用就停下。**绝对不要**自己编造工具的执行结果 —— 真实结果会在下一轮以
  `<tool_result>` 的形式给你。凭空写出的结果会让你基于虚假信息继续推理。
- `arguments` 必须是合法 JSON 对象,字符串里的换行要转义成 \\n。
- 任务完成时,回复纯文本且不带任何 `<tool_call>` 块。

# 可用工具

以下是你可以调用的工具及其参数 schema(JSON Schema 格式):
""".strip()


class TextProtocolProvider:
    """把原生 tool calling 降级成文本协议。

    :param inner: 底层 provider,只用它的补全能力(调用时不再传 ``tools``)。
    """

    def __init__(self, inner: ModelProvider) -> None:
        self.inner = inner

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        rewritten = rewrite_history(messages, tools)
        # 关键:不把 tools 传下去。传了的话,支持原生的端点会同时开两条通道,
        # 模型可能一半走原生一半走文本,解析和配对都会乱。
        response = self.inner.generate(rewritten, tools=None)

        text, calls = parse_tool_calls(response.text)
        return ModelResponse(
            text=text,
            tool_calls=tuple(calls),
            finish_reason="tool_calls" if calls else response.finish_reason,
            usage=response.usage,
        )


def parse_tool_calls(raw: str) -> tuple[str, list[ToolCall]]:
    """从模型回复里切出工具调用,返回 (剩余正文, 调用列表)。

    只收**连续**的块:一旦两个块之间夹了非空白内容,就在那里停下。这是对付
    「模型自己编造工具结果」的结构化手段 —— 真正的批量调用挨在一起,幻觉出来的
    后续调用前面必然隔着一段编造的结果。
    """

    matches = list(TOOL_CALL_RE.finditer(raw))
    if not matches:
        return raw.strip(), []

    accepted: list[re.Match[str]] = [matches[0]]
    for previous, current in zip(matches, matches[1:]):
        between = raw[previous.end() : current.start()]
        if between.strip():
            break
        accepted.append(current)

    calls: list[ToolCall] = []
    for match in accepted:
        calls.append(_build_call(match.group(1)))

    # 正文只保留第一个调用之前的部分,那是模型的思考;之后的内容要么是编造的结果,
    # 要么是被截断的后续调用,留着只会污染 trace。
    return raw[: accepted[0].start()].strip(), calls


def _build_call(body: str) -> ToolCall:
    """解析单个 ``<tool_call>`` 块的 JSON 体。

    坏 JSON 带着 ``parse_error`` 返回而不是抛出 —— 和原生路径保持一致,让它成为
    一条能回灌给模型的普通失败。
    """

    call_id = f"tc_{uuid.uuid4().hex[:8]}"
    text = body.strip()
    # 模型常把 JSON 包在 markdown 代码围栏里,这是纯格式抖动。
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return ToolCall(
            id=call_id,
            name="",
            parse_error=f"<tool_call> 里不是合法 JSON ({exc}): {text[:200]}",
        )

    if not isinstance(parsed, dict):
        return ToolCall(
            id=call_id, name="",
            parse_error=f"<tool_call> 必须是 JSON 对象,收到 {type(parsed).__name__}",
        )

    name = str(parsed.get("name") or "")
    if not name:
        return ToolCall(id=call_id, name="", parse_error="<tool_call> 缺少 name 字段")

    # 兼容 arguments / args / input 三种写法:不同模型家族的训练格式不一样,
    # 为此多烧一轮往返不值得。
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
                id=call_id, name=name,
                parse_error=f"arguments 是字符串且不是合法 JSON: {args[:200]}",
            )
    if not isinstance(args, dict):
        return ToolCall(
            id=call_id, name=name,
            parse_error=f"arguments 必须是对象,收到 {type(args).__name__}",
        )

    return ToolCall(id=call_id, name=name, args=args)


def rewrite_history(
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
) -> list[Message]:
    """把带原生 tool 结构的 history 改写成纯文本形态。

    这一步不是可选的美化。history 里的 ``role: "tool"`` 消息在协议上必须由一条带
    ``tool_calls`` 的 assistant 消息「认领」;而文本协议下我们的 assistant 消息只有
    正文,于是那些 tool 消息在端点看来全是孤儿,请求会被整个拒掉。
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
    """回溯找出这条 tool 结果对应的工具名,让模型看得懂是谁的结果。"""

    call_id = messages[tool_index].get("tool_call_id")
    for message in reversed(messages[:tool_index]):
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id") == call_id:
                return str((call.get("function") or {}).get("name") or "tool")
    return "tool"


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
    """把 tool schema 渲染进 system prompt。"""

    if not tools:
        return PROTOCOL_INSTRUCTIONS + "\n\n(当前没有可用工具。)"

    blocks: list[str] = []
    for tool in tools:
        function = tool.get("function") or {}
        blocks.append(
            f"## {function.get('name', '')}\n\n"
            f"{function.get('description', '')}\n\n"
            f"参数 schema:\n```json\n"
            f"{json.dumps(function.get('parameters', {}), ensure_ascii=False, indent=2)}\n```"
        )
    return PROTOCOL_INSTRUCTIONS + "\n\n" + "\n\n".join(blocks)


__all__ = [
    "PROTOCOL_INSTRUCTIONS",
    "TextProtocolProvider",
    "parse_tool_calls",
    "rewrite_history",
]
