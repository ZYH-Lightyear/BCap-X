"""Conversation history: invariant repair and the deterministic canvas window.

Forked from ``agentx/chat.py``. Sits between the provider and the runtime loop
so that the loop stays a clean state machine and never has to know its own
history was operated on.

One thing was dropped and one changed relative to the original:

- Dropped ``token_estimate`` and the compression seam behind it. VAW keeps a
  *deterministic* context policy instead (see :meth:`prune_images`), because
  RL rollouts and SFT samples must see byte-identical context for the same
  workspace state; a size-triggered compressor makes that dependent on how
  verbose the model happened to be.
- ``max_images`` is now a required argument. In a coding agent, images are
  usually specifications and unlimited retention is the sane default; here
  every image is a *canvas observation* that goes stale the moment the
  workspace changes, so an unbounded window is always wrong.
"""

from __future__ import annotations

from typing import Any

from vaw.agents.contracts import Message, ModelResponse, ToolCall
from vaw.agents.providers.base import ModelProvider


class ChatSession:
    """The full history of one episode plus the send logic."""

    def __init__(
        self,
        provider: ModelProvider,
        system_prompt: str,
        *,
        max_images: int,
    ) -> None:
        """
        :param max_images: How many canvases to keep in history. Older ones are
            dropped oldest-first.
        """
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_images = max_images
        self.history: list[Message] = [{"role": "system", "content": system_prompt}]
        self.usage: dict[str, int] = {}

    def append(self, message: Message) -> None:
        self.history.append(message)

    def extend(self, messages: list[Message]) -> None:
        self.history.extend(messages)

    def send(self, tools: list[dict[str, Any]] | None = None) -> ModelResponse:
        """Send the current history and write the reply back into it."""

        self.repair_orphaned_tool_calls()
        self.prune_images()
        response = self.provider.generate(self.history, tools)
        self.append(_assistant_message(response))
        self._accumulate_usage(response.usage)
        return response

    def repair_orphaned_tool_calls(self) -> int:
        """Answer any tool call left without a tool message; return the count.

        The protocol requires every tool call id in an assistant message to
        have a matching tool message. The runtime covers the normal path, but
        exceptions can still leave orphans — a provider error raised between
        emitting a call and executing it, for instance. An orphan makes the
        endpoint reject the *whole* next request with an error that rarely
        points at the cause, so it is worth a scan before every send.
        """

        repaired = 0
        index = 0
        while index < len(self.history):
            message = self.history[index]
            index += 1
            if message.get("role") != "assistant":
                continue
            call_ids = [
                call.get("id")
                for call in message.get("tool_calls") or []
                if isinstance(call, dict) and call.get("id")
            ]
            if not call_ids:
                continue

            # The consecutive tool messages right after it are this batch's answers.
            answered: set[Any] = set()
            scan = index
            while scan < len(self.history) and self.history[scan].get("role") == "tool":
                answered.add(self.history[scan].get("tool_call_id"))
                scan += 1

            missing = [cid for cid in call_ids if cid not in answered]
            for offset, call_id in enumerate(missing):
                self.history.insert(
                    scan + offset,
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": "This call did not complete (episode interrupted); no result.",
                    },
                )
                repaired += 1
            index = scan + len(missing)

        return repaired

    def prune_images(self) -> int:
        """Keep only the most recent ``max_images`` canvases; return drops.

        Images are the one kind of content that actively misleads once stale.
        Text carries its own ordering and the model can tell an old receipt is
        old; a canvas from ten steps ago looks exactly like the current one,
        and no label reliably beats "I see it, therefore it is now". So the
        right operation on a stale observation is to *void* it, not summarise it.

        Only the ``image_url`` parts are dropped, not whole messages: the text
        alongside them (step header, receipt) is cheap and correctly ordered.
        A user message is removed entirely only when nothing but images is
        left. ``tool`` messages are never touched — removing one would orphan
        its call.
        """
        budget = self.max_images
        dropped = 0
        rebuilt: list[Message] = []
        # Walk backwards so the budget goes to the most recent canvases.
        for message in reversed(self.history):
            content = message.get("content")
            if not isinstance(content, list):
                rebuilt.append(message)
                continue

            kept: list[Any] = []
            for part in content:
                if not (isinstance(part, dict) and part.get("type") == "image_url"):
                    kept.append(part)
                    continue
                if budget > 0:
                    budget -= 1
                    kept.append(part)
                else:
                    dropped += 1

            if len(kept) == len(content):
                rebuilt.append(message)
            elif message.get("role") == "user" and not _has_text(kept):
                continue
            else:
                rebuilt.append({**message, "content": kept})

        if dropped:
            self.history = list(reversed(rebuilt))
        return dropped

    def _accumulate_usage(self, usage: dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value


def _has_text(parts: list[Any]) -> bool:
    return any(
        isinstance(part, dict) and part.get("type") == "text" and str(part.get("text", "")).strip()
        for part in parts
    )


def _assistant_message(response: ModelResponse) -> Message:
    """Rebuild a protocol-valid assistant message from a model reply."""

    message: Message = {"role": "assistant", "content": response.text or None}
    if response.tool_calls:
        message["tool_calls"] = [_tool_call_wire(call) for call in response.tool_calls]
    return message


def _tool_call_wire(call: ToolCall) -> dict[str, Any]:
    import json

    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.args, ensure_ascii=False)},
    }


__all__ = ["ChatSession"]
