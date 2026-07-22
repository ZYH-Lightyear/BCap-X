from __future__ import annotations

import time

import pytest
import requests

from capx.llm import client as llm_client
from robomex.agents.planner import LLMPlannerPolicy
from robomex.core.coder import LLMCodePolicy, TurnBudget, TurnEngine
from robomex.core.token_budget import conservative_chat_prompt_tokens


def test_llm_code_policy_forwards_output_and_wall_deadline_to_client() -> None:
    policy = LLMCodePolicy(max_tokens=1_000)
    captured = []

    def query(args, _prompt):
        captured.append(args)
        return {"content": "DONE"}

    policy._query_model = query
    deadline = time.monotonic() + 2.0

    assert (
        policy.complete_bounded(
            [{"role": "user", "content": "x"}],
            max_tokens=17,
            deadline_monotonic_s=deadline,
        )
        == "DONE"
    )

    args = captured[0]
    assert args.max_tokens == 17
    assert args.max_retries == 1
    assert 0 < args.request_timeout_s <= 2.0
    assert args.deadline_monotonic_s == deadline


def test_llm_planner_bounded_retries_share_total_tokens_and_disable_http_retries() -> None:
    policy = LLMPlannerPolicy(max_tokens=2_000, empty_retries=1)
    calls = []

    def query(args, prompt):
        calls.append((args, prompt))
        return {"content": "" if len(calls) == 1 else "DONE"}

    policy._query_model = query
    prompt = [{"role": "user", "content": "next"}]
    deadline = time.monotonic() + 2.0

    assert (
        policy.propose_bounded(
            prompt,
            max_tokens=2_000,
            max_model_calls=2,
            deadline_monotonic_s=deadline,
        )
        == "DONE"
    )

    assert len(calls) == 2
    assert all(item[0].max_retries == 1 for item in calls)
    assert all(0 < item[0].request_timeout_s <= 2.0 for item in calls)
    total_ceiling = sum(item[0].max_tokens for item in calls)
    total_input = sum(conservative_chat_prompt_tokens(item[1]) for item in calls)
    assert total_input + total_ceiling <= 2_000


def test_capx_http_timeout_is_clamped_to_remaining_deadline(monkeypatch) -> None:
    observed = []

    class Response:
        status_code = 200

    def post(_url, *, headers, data, timeout):
        del headers, data
        observed.append(timeout)
        return Response()

    monkeypatch.setattr(llm_client.requests, "post", post)
    deadline = time.monotonic() + 0.5

    response = llm_client._post_with_retries(
        "http://local.test",
        {},
        {},
        request_timeout_s=30.0,
        max_retries=1,
        deadline_monotonic_s=deadline,
    )

    assert response.status_code == 200
    assert len(observed) == 1
    assert 0 < observed[0] <= 0.5


def test_capx_bounded_path_does_not_hide_network_retries(monkeypatch) -> None:
    calls = 0

    def post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(llm_client.requests, "post", post)

    with pytest.raises(requests.ConnectionError, match="offline"):
        llm_client._post_with_retries(
            "http://local.test",
            {},
            {},
            request_timeout_s=0.5,
            max_retries=1,
            deadline_monotonic_s=time.monotonic() + 1.0,
        )
    assert calls == 1


def test_turn_engine_provider_failure_consumes_call_before_api_entry() -> None:
    class FailingPolicy:
        calls = 0

        def complete_bounded(self, _prompt, *, max_tokens, deadline_monotonic_s):
            del max_tokens, deadline_monotonic_s
            self.calls += 1
            raise RuntimeError("provider failed")

    policy = FailingPolicy()
    engine = TurnEngine(
        policy,
        allowed_tools={"finish"},
        budget=TurnBudget(
            max_model_calls=1,
            max_tokens=1_000,
            require_bounded=True,
        ),
    )
    prompt = [{"role": "user", "content": "one bounded attempt"}]

    with pytest.raises(RuntimeError, match="provider failed"):
        engine.next(prompt)

    assert engine.ledger.model_calls == 1
    assert engine.ledger.tokens_committed > 0
    assert engine.next(prompt) is None
    assert policy.calls == 1


def test_last_http_attempt_never_computes_or_sleeps_retry_backoff(monkeypatch) -> None:
    class Response:
        status_code = 503
        text = "temporarily unavailable"

    monkeypatch.setattr(llm_client.requests, "post", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(
        llm_client,
        "_bounded_retry_sleep",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("last attempt must not calculate retry sleep")
        ),
    )

    response = llm_client._post_with_retries(
        "http://local.test",
        {},
        {},
        request_timeout_s=0.5,
        max_retries=1,
        deadline_monotonic_s=time.monotonic() + 1.0,
    )

    assert response.status_code == 503


@pytest.mark.parametrize(
    ("request_timeout_s", "max_retries"),
    [(0.0, 1), (1.0, 0)],
)
def test_query_model_does_not_replace_explicit_invalid_http_bounds_with_defaults(
    monkeypatch,
    request_timeout_s: float,
    max_retries: int,
) -> None:
    monkeypatch.setattr(
        llm_client.requests,
        "post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid bounds must fail before HTTP entry")
        ),
    )
    args = llm_client.ModelQueryArgs(
        model="local/test",
        server_url="http://local.test",
        request_timeout_s=request_timeout_s,
        max_retries=max_retries,
    )

    with pytest.raises(ValueError, match="must be positive"):
        llm_client.query_model(args, [{"role": "user", "content": "x"}])
