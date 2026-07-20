"""Shared model-turn transport, canonical history, and budget accounting."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from robomex.core.coder.action import ModelTurn, ToolCall, normalized_action_json, parse_model_turn
from robomex.core.coder.protocol import ActionEnvelope, ActionSchema


@dataclass(frozen=True)
class TurnBudget:
    max_model_calls: int = 32
    max_protocol_errors: int = 4
    action_limits: dict[str, int] = field(default_factory=dict)


@dataclass
class TurnLedger:
    model_calls: int = 0
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
            and self.ledger.protocol_errors < self.budget.max_protocol_errors
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
        turn = self._complete_turn(prompt)
        self.ledger.model_calls += 1
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

        canonical = turn.canonical_json or normalized_action_json(call.name, call.args)
        prompt.append({"role": "assistant", "content": canonical})
        return EngineTurn(index=index, turn=turn, tool_call=call)

    @staticmethod
    def append_feedback(prompt: list[dict], content: str | list) -> None:
        prompt.append({"role": "user", "content": content})

    def _complete_turn(self, prompt: list[dict]) -> ModelTurn:
        complete_turn = getattr(self.policy, "complete_turn", None)
        if callable(complete_turn):
            try:
                params = inspect.signature(complete_turn).parameters
            except (TypeError, ValueError):
                params = {}
            if "tool_names" in params:
                return complete_turn(prompt, tool_names=self.allowed_tools)
            return complete_turn(prompt)
        return parse_model_turn(self.policy.complete(prompt))

    @staticmethod
    def _single_tool_call(turn: ModelTurn) -> ToolCall | None:
        if turn.is_error or len(turn.tool_calls) != 1:
            return None
        return turn.tool_calls[0]
