#!/usr/bin/env python3
"""Probe whether V-API / the local LLM proxy supports OpenAI-style tool calls.

Examples:
  # Test the local proxy route used by RoboMEx.
  .venv-libero/bin/python scripts/test_vapi_tool_call_support.py --mode proxy --model vapi/gpt-5.5

  # Bypass the local proxy and test V-API directly.
  V_API_KEY=... .venv-libero/bin/python scripts/test_vapi_tool_call_support.py --mode direct --model gpt-5.5

  # Run both tests and print full response JSON.
  .venv-libero/bin/python scripts/test_vapi_tool_call_support.py --mode both --dump
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import requests


TOOL_NAME = "record_result"


@dataclass(frozen=True)
class ProbeTarget:
    name: str
    url: str
    model: str
    api_key: str | None


def _payload(model: str, *, max_tokens: int, token_field: str, tool_choice: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are testing tool-call support. Use the requested tool exactly once.",
            },
            {
                "role": "user",
                "content": (
                    f"Call the {TOOL_NAME} tool with answer='tool-calls-supported'. "
                    "Do not answer in normal text."
                ),
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": "Return a short diagnostic answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "Diagnostic answer.",
                            }
                        },
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
    }
    if tool_choice:
        payload["tool_choice"] = {"type": "function", "function": {"name": TOOL_NAME}}
    if token_field == "max_completion_tokens":
        payload["max_completion_tokens"] = max_tokens
    elif token_field == "max_tokens":
        payload["max_tokens"] = max_tokens
    else:
        raise ValueError(f"unsupported token field: {token_field}")
    return payload


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_message(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    msg = choices[0].get("message")
    return msg if isinstance(msg, dict) else {}


def _summarize_response(data: dict[str, Any]) -> tuple[bool, str]:
    msg = _extract_message(data)
    tool_calls = msg.get("tool_calls")
    function_call = msg.get("function_call")
    finish_reason = None
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        finish_reason = choices[0].get("finish_reason")

    if isinstance(tool_calls, list) and tool_calls:
        names = []
        for call in tool_calls:
            fn = call.get("function") if isinstance(call, dict) else None
            if isinstance(fn, dict):
                names.append(str(fn.get("name", "")))
        return True, f"tool_calls returned; finish_reason={finish_reason!r}; tools={names}"

    if isinstance(function_call, dict) and function_call:
        return True, f"legacy function_call returned; finish_reason={finish_reason!r}; function={function_call.get('name')!r}"

    content = msg.get("content")
    if content:
        preview = str(content).replace("\n", " ")[:300]
        return False, (
            "no tool_calls returned; model/proxy answered with normal content. "
            f"finish_reason={finish_reason!r}; content_preview={preview!r}"
        )

    return False, f"no tool_calls and no content in first choice; finish_reason={finish_reason!r}"


def _post(target: ProbeTarget, payload: dict[str, Any], timeout: float) -> tuple[bool, dict[str, Any] | str]:
    try:
        response = requests.post(
            target.url,
            headers=_headers(target.api_key),
            data=json.dumps(payload),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return False, f"request failed: {type(exc).__name__}: {exc}"

    try:
        body: dict[str, Any] | str = response.json()
    except ValueError:
        body = response.text
    if response.status_code >= 400:
        return False, {
            "http_status": response.status_code,
            "body": body,
        }
    return True, body


def _run_probe(target: ProbeTarget, args: argparse.Namespace) -> bool:
    print(f"\n== {target.name} ==")
    print(f"url   : {target.url}")
    print(f"model : {target.model}")
    print(f"auth  : {'yes' if target.api_key else 'no'}")

    payload = _payload(
        target.model,
        max_tokens=args.max_tokens,
        token_field=args.token_field,
        tool_choice=not args.no_tool_choice,
    )
    ok, result = _post(target, payload, args.timeout)
    if not ok:
        print("HTTP/request result: FAILED")
        print(json.dumps(result, indent=2, ensure_ascii=False) if isinstance(result, dict) else result)
        return False

    assert isinstance(result, dict)
    supported, summary = _summarize_response(result)
    print(f"tool-call result: {'SUPPORTED' if supported else 'NOT SUPPORTED / NOT PRESERVED'}")
    print(f"summary         : {summary}")
    if args.dump:
        print("raw response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if target.name == "proxy" and not supported:
        print(
            "note           : the current local proxy may drop tools/tool_calls because its "
            "request/response schema does not include those fields. Run --mode direct to test V-API itself."
        )
    return supported


def _env_api_key() -> str | None:
    for name in ("V_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name)
        if value:
            return value
    return None


def _load_env_file(path: str | None) -> None:
    if not path:
        return
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :]
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("proxy", "direct", "both"), default="proxy")
    parser.add_argument("--proxy-url", default="http://localhost:8110/chat/completions")
    parser.add_argument("--direct-base-url", default="https://api.gpt.ge/v1")
    parser.add_argument("--model", default="vapi/gpt-5.5", help="Use vapi/... for proxy mode; direct mode strips leading vapi/.")
    parser.add_argument("--api-key", default=None, help="Direct V-API key. Defaults to V_API_KEY or LLM_API_KEY.")
    parser.add_argument("--env-file", default=".env", help="Optional dotenv file to load before reading API keys.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--token-field",
        choices=("max_completion_tokens", "max_tokens"),
        default="max_completion_tokens",
        help="Use max_completion_tokens for GPT-5/o-series style models; use max_tokens if upstream rejects it.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-tool-choice", action="store_true", help="Do not force the tool; let model choose.")
    parser.add_argument("--dump", action="store_true", help="Print raw response JSON.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after printing diagnostics.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _load_env_file(args.env_file)
    targets: list[ProbeTarget] = []
    if args.mode in {"proxy", "both"}:
        targets.append(ProbeTarget("proxy", args.proxy_url, args.model, None))
    if args.mode in {"direct", "both"}:
        direct_model = args.model[len("vapi/") :] if args.model.startswith("vapi/") else args.model
        targets.append(
            ProbeTarget(
                "direct-vapi",
                args.direct_base_url.rstrip("/") + "/chat/completions",
                direct_model,
                args.api_key or _env_api_key(),
            )
        )

    results = [_run_probe(target, args) for target in targets]
    if args.no_fail:
        return 0
    return 0 if all(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
