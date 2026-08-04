"""Text protocol provider: ``<tool_call>`` tags instead of native function calling.

Forked from ``agentx/providers/text_protocol.py``. Wraps any
:class:`~vaw.agents.providers.base.ModelProvider` and still returns a
:class:`~vaw.agents.contracts.ModelResponse` with ``tool_calls`` filled in, so
the Context Runtime remains independent of the provider's tool-call transport.

## Why it exists

Native function calling is better when it works: schema-constrained decoding,
and a structural difference between *discussing* a call and *making* one. But
it needs the entire chain to support it, and the chain is fragile — proxies
swallow ``tools``, compatibility shims mistranslate, and some routes fall out
of native mode intermittently and start narrating a fake conversation instead.
The text protocol only needs completion, so it works everywhere. For VAW it is
also the student path: a locally served Qwen3-VL speaks this natively.

## Format

``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``

XML tag around a JSON body. Tag boundaries are easy to cut and tolerant of
noise, and the body maps one-to-one onto ``ToolCall.args``. It is also Qwen's
own training format, so for the student this is closer to the pretraining
distribution than an OpenAI compatibility layer would be.

## Divergence from the original: one op per step

AgentX told the model to emit consecutive blocks for parallel calls. VAW
forbids that — the canvas is a full state render and ops are strongly ordered,
so one step is one op. The instructions below say so, and :func:`parse_tool_calls`
still returns every block it finds: silently dropping the extras would hide a
protocol violation that the runtime needs to see and answer explicitly.

The continuity rule is kept regardless. When a model glitches it alternates
invented calls with invented results, so cutting at the first non-blank gap
stops before the fabricated part.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from vaw.agents.contracts import Message, ModelResponse, ToolCall
from vaw.agents.providers.base import ModelProvider

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

PROTOCOL_INSTRUCTIONS = """
# Operation call format

You have no native tool channel. To run a workspace operation, write a
`<tool_call>` block containing a JSON object with `name` and `arguments`:

<tool_call>{"name": "ground", "arguments": {"text": "red mug"}}</tool_call>

Rules:

- **Exactly one `<tool_call>` block per reply.** One step is one operation; the
  canvas you get back reflects that operation and nothing else.
- Follow the task's system instructions about a concise decision basis. When
  requested, write that basis as plain text before the block.
- Stop after the block. **Never** write the operation's result yourself — the
  real receipt and a fresh canvas arrive next turn as `<tool_result>`.
  Inventing a result makes everything after it reasoning on fiction.
- `arguments` must be a valid JSON object; escape newlines in strings as \\n.
- The episode ends only when you call `done`. Plain text without a
  `<tool_call>` block does not end anything and wastes a turn.

# Available operations

Operations you can call, with their parameter schemas (JSON Schema):
""".strip()


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
        rewritten = rewrite_history(messages, tools)
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
            parse_error=f"<tool_call> body is not valid JSON ({exc}): {text[:200]}",
        )

    if not isinstance(parsed, dict):
        return ToolCall(
            id=call_id,
            name="",
            parse_error=f"<tool_call> must be a JSON object, got {type(parsed).__name__}",
        )

    name = str(parsed.get("name") or "")
    if not name:
        return ToolCall(id=call_id, name="", parse_error="<tool_call> is missing 'name'")

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
                parse_error=f"arguments is a string and not valid JSON: {args[:200]}",
            )
    if not isinstance(args, dict):
        return ToolCall(
            id=call_id,
            name=name,
            parse_error=f"arguments must be an object, got {type(args).__name__}",
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
    """Render the op schemas into the system prompt."""

    if not tools:
        return PROTOCOL_INSTRUCTIONS + "\n\n(No operations available.)"

    blocks: list[str] = []
    for tool in tools:
        function = tool.get("function") or {}
        blocks.append(
            f"## {function.get('name', '')}\n\n"
            f"{function.get('description', '')}\n\n"
            f"Parameters:\n```json\n"
            f"{json.dumps(function.get('parameters', {}), ensure_ascii=False, indent=2)}\n```"
        )
    return PROTOCOL_INSTRUCTIONS + "\n\n" + "\n\n".join(blocks)


__all__ = [
    "PROTOCOL_INSTRUCTIONS",
    "TextProtocolProvider",
    "parse_tool_calls",
    "rewrite_history",
]
