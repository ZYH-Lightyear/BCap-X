"""Bounded, protocol-safe policy history for the M1.3 Context Runtime."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from vaw.agents.contracts import Message, ModelResponse, ToolCall
from vaw.context_runtime.packet import encode_png_data_url


@dataclass(frozen=True)
class FunctionTransaction:
    """One assistant tool-call message and all protocol-required replies."""

    assistant: Message
    tool_messages: tuple[Message, ...]

    def messages(self) -> list[Message]:
        return [self.assistant, *self.tool_messages]


class ContextHistory:
    """System/task plus at most K complete function transactions."""

    def __init__(self, system_prompt: str, task_prompt: str, *, max_transactions: int = 3) -> None:
        if max_transactions < 0:
            raise ValueError("max_transactions must be non-negative")
        self.system_prompt = system_prompt
        self.task_prompt = task_prompt
        self.transactions: deque[FunctionTransaction] = deque(maxlen=max_transactions)
        self.pending_feedback: str | None = None

    def build_messages(
        self,
        *,
        manifest: dict[str, Any],
        context_image: np.ndarray,
    ) -> list[Message]:
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"任务：{self.task_prompt}"},
        ]
        for transaction in self.transactions:
            messages.extend(transaction.messages())
        if self.pending_feedback:
            messages.append({"role": "user", "content": self.pending_feedback})
            self.pending_feedback = None
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "[当前 VAW Context]\nminimal manifest:\n"
                            + json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_png_data_url(context_image)},
                    },
                ],
            }
        )
        return messages

    def add_response(self, response: ModelResponse, results: list[dict[str, Any]]) -> None:
        if len(results) != len(response.tool_calls):
            raise ValueError("every tool call must have exactly one result")
        # The model's decision basis belongs in the trace, not in policy
        # history. Replaying it would turn an unverified interpretation into
        # apparent evidence on later turns. Keep only the protocol-required
        # tool call and its result in the bounded history.
        assistant: Message = {"role": "assistant", "content": None}
        assistant["tool_calls"] = [_tool_call_wire(call) for call in response.tool_calls]
        tools = tuple(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            }
            for call, result in zip(response.tool_calls, results, strict=True)
        )
        self.transactions.append(FunctionTransaction(assistant=assistant, tool_messages=tools))

    def visible_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for transaction in self.transactions:
            calls = transaction.assistant.get("tool_calls") or []
            records.append(
                {
                    "calls": [
                        {
                            "id": call.get("id"),
                            "name": call.get("function", {}).get("name"),
                            "arguments": call.get("function", {}).get("arguments"),
                        }
                        for call in calls
                    ],
                    "results": [message.get("content") for message in transaction.tool_messages],
                }
            )
        return records


def _tool_call_wire(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.args, ensure_ascii=False, separators=(",", ":")),
        },
    }


__all__ = ["ContextHistory", "FunctionTransaction"]
