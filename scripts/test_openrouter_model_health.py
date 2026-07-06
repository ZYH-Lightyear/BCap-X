#!/usr/bin/env python3
"""Probe an OpenRouter model through the local proxy and/or OpenRouter directly.

This is intentionally smaller than a RoboMEx rollout. It checks whether a model
returns parseable OpenAI-compatible JSON, visible content, and stable responses
for strict JSON prompts.

Examples:
  # Test the exact route RoboMEx uses.
  .venv-libero/bin/python scripts/test_openrouter_model_health.py \
    --mode proxy --model openrouter/qwen/qwen3.5-9b

  # Compare local proxy vs direct OpenRouter, with raw JSON saved.
  .venv-libero/bin/python scripts/test_openrouter_model_health.py \
    --mode both --model openrouter/qwen/qwen3.5-9b --dump-dir outputs/openrouter_health
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class Probe:
    name: str
    url: str
    model: str
    api_key: str | None
    reasoning: dict[str, Any] | None


def _load_dotenv(path: str | None) -> None:
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
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


def _load_key_file(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return None


def _openrouter_key(args: argparse.Namespace) -> str | None:
    return (
        args.api_key
        or os.getenv("OPENROUTER_API_KEY")
        or _load_key_file(args.key_file)
    )


def _native_openrouter_model(model: str) -> str:
    return model[len("openrouter/") :] if model.startswith("openrouter/") else model


def _payload(
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    prompt_kind: str,
    reasoning: dict[str, Any] | None,
) -> dict[str, Any]:
    if prompt_kind == "json":
        user = (
            'Reply with exactly this JSON object and no markdown: '
            '{"tool":"finish","args":{"claim":"health_ok"}}'
        )
    elif prompt_kind == "plain":
        user = "Reply with exactly the two words: health ok"
    else:
        raise ValueError(f"unknown prompt kind: {prompt_kind}")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a health probe. Return visible content. "
                    "Do not leave the assistant message empty."
                ),
            },
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if reasoning is not None:
        payload["reasoning"] = reasoning
    return payload


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/nvidia-gear/CaP-X",
        "X-Title": "RoboMEx OpenRouter health probe",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _safe_json_loads(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return None, f"top-level JSON is {type(data).__name__}, expected object"
    return data, None


def _extract_summary(status_code: int | None, body: dict[str, Any] | None, text: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status_code": status_code,
        "json_ok": body is not None,
        "content_len": 0,
        "finish_reason": None,
        "model": None,
        "error": None,
    }
    if body is None:
        summary["text_prefix"] = text[:800]
        return summary
    summary["model"] = body.get("model")
    if status_code is not None and status_code >= 400:
        summary["error"] = body.get("error") or body.get("detail") or body
        return summary
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        summary["error"] = "missing choices"
        return summary
    first = choices[0]
    if isinstance(first, dict):
        summary["finish_reason"] = first.get("finish_reason")
        msg = first.get("message")
        if isinstance(msg, dict):
            content = msg.get("content") or ""
            summary["content_len"] = len(content)
            summary["content_preview"] = str(content).replace("\n", " ")[:300]
            reasoning = msg.get("reasoning")
            if reasoning:
                summary["reasoning_len"] = len(str(reasoning))
    return summary


def _dump(
    dump_dir: str | None,
    probe: Probe,
    payload: dict[str, Any],
    status_code: int | None,
    elapsed: float,
    text: str,
    parse_error: str | None,
) -> None:
    if not dump_dir:
        return
    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    safe_headers = _headers(probe.api_key)
    if "Authorization" in safe_headers:
        safe_headers["Authorization"] = "Bearer <redacted>"
    record = {
        "probe": probe.name,
        "url": probe.url,
        "headers": safe_headers,
        "payload": payload,
        "status_code": status_code,
        "elapsed_seconds": elapsed,
        "json_parse_error": parse_error,
        "response_text": text,
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{stamp}_{probe.name.replace('/', '_')}_{time.time_ns()}.json"
    (path / filename).write_text(json.dumps(record, ensure_ascii=False, indent=2))


def _run_probe(probe: Probe, args: argparse.Namespace) -> bool:
    payload = _payload(
        model=probe.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        prompt_kind=args.prompt_kind,
        reasoning=probe.reasoning,
    )
    print(f"\n== {probe.name} ==")
    print(f"url       : {probe.url}")
    print(f"model     : {probe.model}")
    print(f"reasoning : {probe.reasoning if probe.reasoning is not None else '(not sent)'}")
    print(f"auth      : {'yes' if probe.api_key else 'no'}")
    started = time.time()
    status_code: int | None = None
    text = ""
    try:
        response = requests.post(
            probe.url,
            headers=_headers(probe.api_key),
            data=json.dumps(payload),
            timeout=args.timeout,
        )
        status_code = response.status_code
        text = response.text
    except requests.RequestException as exc:
        elapsed = time.time() - started
        print(f"request   : FAILED after {elapsed:.2f}s: {type(exc).__name__}: {exc}")
        _dump(args.dump_dir, probe, payload, status_code, elapsed, text, str(exc))
        return False

    elapsed = time.time() - started
    body, parse_error = _safe_json_loads(text)
    _dump(args.dump_dir, probe, payload, status_code, elapsed, text, parse_error)
    summary = _extract_summary(status_code, body, text)
    print(f"elapsed   : {elapsed:.2f}s")
    print(f"summary   : {json.dumps(summary, ensure_ascii=False)}")
    if parse_error:
        print(f"parse_err : {parse_error}")
    ok = (
        status_code is not None
        and status_code < 400
        and body is not None
        and int(summary.get("content_len") or 0) > 0
    )
    print(f"result    : {'OK' if ok else 'FAILED'}")
    return ok


def _parse_reasoning(value: str) -> dict[str, Any] | None:
    normalized = value.strip().lower()
    if normalized in {"absent", "none", "off"}:
        return None
    if normalized == "exclude":
        return {"exclude": True}
    if normalized.startswith("json:"):
        raw = value.split(":", 1)[1]
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise argparse.ArgumentTypeError("--reasoning json:... must decode to an object")
        return parsed
    return {"effort": value}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("proxy", "direct", "both"), default="both")
    parser.add_argument("--model", default="openrouter/qwen/qwen3.5-9b")
    parser.add_argument("--proxy-url", default="http://localhost:8110/chat/completions")
    parser.add_argument("--direct-url", default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--key-file", default=".openrouterkey")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--prompt-kind", choices=("json", "plain"), default="json")
    parser.add_argument(
        "--reasoning",
        default="absent",
        help=(
            "Reasoning field for direct probes: absent/off, exclude, low, medium, "
            "or json:{...}. Proxy default route may still inject its configured reasoning."
        ),
    )
    parser.add_argument(
        "--include-proxy-reasoning-override",
        action="store_true",
        help="Also run a proxy probe with reasoning={exclude:true} to override route default.",
    )
    parser.add_argument("--dump-dir", default=None, help="Directory to save raw request/response JSON.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after printing diagnostics.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _load_dotenv(args.env_file)

    direct_key = _openrouter_key(args)
    probes: list[Probe] = []
    if args.mode in {"proxy", "both"}:
        probes.append(
            Probe(
                name="proxy_default",
                url=args.proxy_url,
                model=args.model,
                api_key=None,
                reasoning=None,
            )
        )
        if args.include_proxy_reasoning_override:
            probes.append(
                Probe(
                    name="proxy_reasoning_exclude",
                    url=args.proxy_url,
                    model=args.model,
                    api_key=None,
                    reasoning={"exclude": True},
                )
            )
    if args.mode in {"direct", "both"}:
        probes.append(
            Probe(
                name=f"direct_reasoning_{args.reasoning}",
                url=args.direct_url,
                model=_native_openrouter_model(args.model),
                api_key=direct_key,
                reasoning=_parse_reasoning(args.reasoning),
            )
        )
        if args.reasoning != "low":
            probes.append(
                Probe(
                    name="direct_reasoning_low",
                    url=args.direct_url,
                    model=_native_openrouter_model(args.model),
                    api_key=direct_key,
                    reasoning={"effort": "low"},
                )
            )

    if any(p.name.startswith("direct") and not p.api_key for p in probes):
        print(
            "ERROR: direct OpenRouter probe needs OPENROUTER_API_KEY, --api-key, "
            "or a readable --key-file.",
            file=sys.stderr,
        )
        return 2

    results = [_run_probe(probe, args) for probe in probes]
    if args.no_fail:
        return 0
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
