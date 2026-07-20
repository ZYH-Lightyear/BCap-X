"""Provider-neutral request normalization for the local LLM proxy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RouteProfile:
    token_field: str = "auto"
    reasoning_style: str = "none"
    reasoning_effort: str | None = None

    def validate(self) -> None:
        if self.token_field not in {"auto", "max_tokens", "max_completion_tokens"}:
            raise ValueError(f"unsupported token_field {self.token_field!r}")
        if self.reasoning_style not in {"none", "openai", "openrouter"}:
            raise ValueError(f"unsupported reasoning_style {self.reasoning_style!r}")


def supports_openai_reasoning_controls(model: str) -> bool:
    native = model.lower()
    return native.startswith("gpt-5") or native.startswith(("o1", "o3", "o4"))


def normalize_proxy_payload(
    payload: dict[str, Any],
    *,
    native_model: str,
    profile: RouteProfile,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Translate a common chat payload into one upstream request profile."""

    profile.validate()
    kwargs = {key: value for key, value in payload.items() if value is not None}
    explicit_reasoning = kwargs.pop("reasoning", None)
    requested_effort = kwargs.pop("reasoning_effort", None)
    max_completion_tokens = kwargs.pop("max_completion_tokens", None)
    max_tokens = kwargs.pop("max_tokens", None)
    budget = max_completion_tokens or max_tokens or 256

    token_field = profile.token_field
    if token_field == "auto":
        token_field = (
            "max_completion_tokens"
            if supports_openai_reasoning_controls(native_model)
            else "max_tokens"
        )
    kwargs[token_field] = budget
    kwargs["model"] = native_model
    if token_field == "max_completion_tokens":
        kwargs.pop("temperature", None)

    effort = str(requested_effort or profile.reasoning_effort or "").strip()
    extra_body: dict[str, Any] | None = None
    if profile.reasoning_style == "openai":
        if (
            supports_openai_reasoning_controls(native_model)
            and effort
            and effort.lower() not in {"none", "off"}
        ):
            kwargs["reasoning_effort"] = effort
    elif profile.reasoning_style == "openrouter":
        reasoning = explicit_reasoning
        if reasoning is None and effort and effort.lower() not in {"none", "off"}:
            reasoning = {"effort": effort}
        if reasoning is not None:
            extra_body = {"reasoning": reasoning}

    return kwargs, extra_body
