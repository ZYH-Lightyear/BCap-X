"""Shared model-turn transport, canonical history, and budget accounting."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from robomex.core.coder.action import ModelTurn, ToolCall, normalized_action_json, parse_model_turn
from robomex.core.coder.protocol import ActionEnvelope, ActionSchema
from robomex.core.token_budget import conservative_chat_prompt_tokens


@dataclass(frozen=True)
class TurnBudget:
    max_model_calls: int = 32
    max_tokens: int = 262_144
    deadline_monotonic_s: float | None = None
    require_bounded: bool = False
    max_protocol_errors: int = 4
    action_limits: dict[str, int] = field(default_factory=dict)


@dataclass
class TurnLedger:
    model_calls: int = 0
    tokens_committed: int = 0
    protocol_errors: int = 0
    actions: dict[str, int] = field(default_factory=dict)

    def action_count(self, name: str) -> int:
        return self.actions.get(name, 0)


@dataclass(frozen=True)
class EngineTurn:
    index: int
    turn: ModelTurn
    tool_call: ToolCall | None


class TurnEngine:
    """Own the provider-facing portion of an agent loop.

    Domain action execution remains with the caller. The engine guarantees that
    raw provider text is observable but only canonical actions enter prompt history.
    """

    def __init__(
        self,
        policy: Any,
        *,
        allowed_tools: set[str],
        budget: TurnBudget,
    ) -> None:
        self.policy = policy
        self.allowed_tools = set(allowed_tools)
        self.action_schema = ActionSchema(frozenset(allowed_tools))
        self.budget = budget
        self.ledger = TurnLedger()

    @property
    def can_call_model(self) -> bool:
        return (
            self.ledger.model_calls < self.budget.max_model_calls
            and self.ledger.tokens_committed < self.budget.max_tokens
            and self.ledger.protocol_errors < self.budget.max_protocol_errors
            and (
                self.budget.deadline_monotonic_s is None
                or time.monotonic() < self.budget.deadline_monotonic_s
            )
        )

    def can_dispatch(self, action: str) -> bool:
        limit = self.budget.action_limits.get(action)
        return limit is None or self.ledger.action_count(action) < limit

    def record_action(self, action: str) -> None:
        self.ledger.actions[action] = self.ledger.action_count(action) + 1

    def next(
        self,
        prompt: list[dict],
        *,
        on_request: Callable[[int, list[dict]], None] | None = None,
        on_response: Callable[[int, ModelTurn], None] | None = None,
    ) -> EngineTurn | None:
        if not self.can_call_model:
            return None

        index = self.ledger.model_calls
        if on_request is not None:
            on_request(index, prompt)
        remaining_calls = self.budget.max_model_calls - self.ledger.model_calls
        remaining_tokens = self.budget.max_tokens - self.ledger.tokens_committed
        input_tokens = conservative_chat_prompt_tokens(prompt)
        if remaining_tokens <= input_tokens:
            return None
        output_pool = remaining_tokens - input_tokens
        chargeable_calls = min(remaining_calls, output_pool)
        output_ceiling = output_pool // chargeable_calls
        # No signed usage is available. Commit the exact API ceiling before
        # entry, together with this turn's full growing prompt, so failures and
        # empty responses cannot make either input or output usage free.
        self.ledger.tokens_committed += input_tokens + output_ceiling
        self.ledger.model_calls += 1
        turn = self._complete_turn(prompt, max_tokens=output_ceiling)
        if (
            self.budget.deadline_monotonic_s is not None
            and time.monotonic() > self.budget.deadline_monotonic_s
        ):
            raise TimeoutError("model turn exceeded the sealed wall-time deadline")
        if on_response is not None:
            on_response(index, turn)

        call = self._single_tool_call(turn)
        if call is not None:
            schema_error = self.action_schema.validate(ActionEnvelope(call.name, call.args))
            if schema_error:
                turn = replace(turn, tool_calls=(), error=schema_error)
                call = None
        if call is None:
            self.ledger.protocol_errors += 1
            return EngineTurn(index=index, turn=turn, tool_call=None)

        self.ledger.protocol_errors = 0
        canonical = turn.canonical_json or normalized_action_json(call.name, call.args)
        prompt.append({"role": "assistant", "content": canonical})
        return EngineTurn(index=index, turn=turn, tool_call=call)

    @staticmethod
    def append_feedback(prompt: list[dict], content: str | list) -> None:
        prompt.append({"role": "user", "content": content})

    def _complete_turn(self, prompt: list[dict], *, max_tokens: int) -> ModelTurn:
        complete_turn_bounded = getattr(self.policy, "complete_turn_bounded", None)
        if callable(complete_turn_bounded):
            return complete_turn_bounded(
                prompt,
                tool_names=self.allowed_tools,
                max_tokens=max_tokens,
                deadline_monotonic_s=self.budget.deadline_monotonic_s,
            )
        complete_bounded = getattr(self.policy, "complete_bounded", None)
        if callable(complete_bounded):
            return parse_model_turn(
                complete_bounded(
                    prompt,
                    max_tokens=max_tokens,
                    deadline_monotonic_s=self.budget.deadline_monotonic_s,
                )
            )
        complete_turn = getattr(self.policy, "complete_turn", None)
        complete = getattr(self.policy, "complete", None)
        if self.budget.require_bounded and (callable(complete_turn) or callable(complete)):
            raise TypeError("bounded model turns require complete_bounded or complete_turn_bounded")
        if callable(complete_turn):
            try:
                params = inspect.signature(complete_turn).parameters
            except (TypeError, ValueError):
                params = {}
            if "tool_names" in params:
                return complete_turn(prompt, tool_names=self.allowed_tools)
            return complete_turn(prompt)
        if callable(complete):
            return parse_model_turn(complete(prompt))
        raise TypeError("policy does not implement a bounded completion boundary")

    @staticmethod
    def _single_tool_call(turn: ModelTurn) -> ToolCall | None:
        if turn.is_error or len(turn.tool_calls) != 1:
            return None
        return turn.tool_calls[0]
