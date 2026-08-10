"""Evaluate open-ended next-Function decisions using reconstructed Agent input."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from vaw.agents.providers.text_protocol import parse_tool_calls, rewrite_history
from vaw.context_runtime.protocol import SYSTEM_PROMPT, function_definitions

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = Path(__file__).with_name("next_action_cases.json")
DEFAULT_MODELS = ("vapi/gpt-5.5", "vapi/qwen3.5-plus")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(path: Path, selected_ids: set[str] | None = None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trace_path = (REPO_ROOT / payload["trace"]).resolve()
    if _sha256(trace_path) != payload["trace_sha256"]:
        raise ValueError("source steps.jsonl changed; review labels before updating its hash")
    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]

    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    for spec in payload["cases"]:
        case_id = str(spec["id"])
        if case_id in seen:
            raise ValueError(f"duplicated case id: {case_id}")
        seen.add(case_id)
        if selected_ids is not None and case_id not in selected_ids:
            continue
        index = int(spec["trace_index"])
        if not 0 <= index < len(trace):
            raise ValueError(f"{case_id}: trace_index {index} is out of range")
        step = trace[index]
        image_path = trace_path.parent / step["context_image"]
        actual_hash = _sha256(image_path)
        if actual_hash != spec["image_sha256"]:
            raise ValueError(
                f"{case_id}: image hash changed; expected {spec['image_sha256']}, got {actual_hash}"
            )
        cases.append({**spec, "_step": step, "_image_path": image_path})

    if selected_ids is not None:
        missing = sorted(selected_ids - {case["id"] for case in cases})
        if missing:
            raise ValueError(f"unknown case ids: {', '.join(missing)}")
    return cases


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_messages(case: dict[str, Any]) -> list[dict[str, Any]]:
    step = case["_step"]
    task = (
        step.get("context_packet", {}).get("world", {}).get("taskPrompt")
        or case.get("task_prompt")
        or "Complete the task shown in the frozen trace."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"User Task：{task}\nCurrent Policy State："
                        + json.dumps(
                            step.get("context_manifest") or {},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                },
                {"type": "image_url", "image_url": {"url": _data_url(case["_image_path"])}},
            ],
        }
    )
    return rewrite_history(messages, function_definitions())


def _normalise_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    return "" if value is None else str(value)


def recover_unclosed_intent(raw: str) -> dict[str, Any] | None:
    """Recover an analysable intent from a non-executable unclosed text tag.

    The runtime intentionally remains strict. This helper is diagnostic only:
    it lets the report distinguish a decision failure from a transport-format
    failure without pretending the malformed reply could have executed.
    """

    match = re.search(r"<tool_call>\s*(\{.*\})\s*$", raw, flags=re.DOTALL)
    if match is None or "</tool_call>" in raw:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return None
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        return None
    return {"name": payload["name"], "arguments": arguments}


def _validate_schema(name: str, arguments: dict[str, Any]) -> str | None:
    definitions = {item["function"]["name"]: item["function"] for item in function_definitions()}
    definition = definitions.get(name)
    if definition is None:
        return f"unknown Function {name!r}"
    schema = definition["parameters"]
    for required in schema.get("required") or []:
        if required not in arguments:
            return f"missing required argument {required!r}"
    properties = schema.get("properties") or {}
    for key, value in arguments.items():
        spec = properties.get(key)
        if spec is None:
            continue
        error = _validate_value(value, spec, key)
        if error:
            return error
    if name == "delta_move":
        vector = arguments.get("delta_xyz_m")
        if isinstance(vector, list) and not any(abs(float(item)) > 1e-12 for item in vector):
            return "delta_xyz_m must be nonzero"
    if name == "rotate" and arguments.get("angle_deg") == 0:
        return "angle_deg must be nonzero"
    return None


def _validate_value(value: Any, spec: dict[str, Any], path: str) -> str | None:
    expected = spec.get("type")
    if expected == "string" and not isinstance(value, str):
        return f"{path} must be a string"
    if expected == "boolean" and not isinstance(value, bool):
        return f"{path} must be a boolean"
    if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        return f"{path} must be a number"
    if expected == "array" and not isinstance(value, list):
        return f"{path} must be an array"
    if "enum" in spec and value not in spec["enum"]:
        return f"{path} is not one of {spec['enum']}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return f"{path} must be finite"
        if "minimum" in spec and value < spec["minimum"]:
            return f"{path} is below {spec['minimum']}"
        if "maximum" in spec and value > spec["maximum"]:
            return f"{path} is above {spec['maximum']}"
    if isinstance(value, list):
        if len(value) < int(spec.get("minItems", 0)):
            return f"{path} is too short"
        if "maxItems" in spec and len(value) > int(spec["maxItems"]):
            return f"{path} is too long"
        item_spec = spec.get("items") or {}
        for index, item in enumerate(value):
            error = _validate_value(item, item_spec, f"{path}[{index}]")
            if error:
                return error
    return None


def _matches(call: dict[str, Any], pattern: dict[str, Any]) -> bool:
    if call.get("name") != pattern.get("name"):
        return False
    arguments = call.get("arguments") or {}
    for key, rule in (pattern.get("arguments") or {}).items():
        if key not in arguments:
            return False
        value = arguments[key]
        if "equals" in rule and value != rule["equals"]:
            return False
        if "in" in rule and value not in rule["in"]:
            return False
        if "contains_any" in rule:
            text = str(value).lower()
            if not any(str(part).lower() in text for part in rule["contains_any"]):
                return False
        if "vector_ranges" in rule:
            ranges = rule["vector_ranges"]
            if not isinstance(value, list) or len(value) != len(ranges):
                return False
            for component, bounds in zip(value, ranges, strict=True):
                if not isinstance(component, (int, float)) or not bounds[0] <= component <= bounds[1]:
                    return False
    return True


def score_call(case: dict[str, Any], call: dict[str, Any] | None, parse_error: str | None) -> tuple[str, str]:
    if parse_error:
        return "protocol_error", parse_error
    if call is None:
        return "no_call", "response contained no Function call"
    schema_error = _validate_schema(call["name"], call["arguments"])
    if schema_error:
        return "invalid_call", schema_error
    if any(_matches(call, pattern) for pattern in case.get("preferred_calls") or []):
        return "preferred", "matches the preferred physical decision"
    if any(_matches(call, pattern) for pattern in case.get("acceptable_calls") or []):
        return "acceptable", "matches an allowed conservative alternative"
    if any(_matches(call, pattern) for pattern in case.get("unsafe_calls") or []):
        return "unsafe", "matches an explicitly unsafe or narrative-jump decision"
    return "other", "valid Function call, but outside the reviewed decision set"


def query(
    *,
    server_url: str,
    model: str,
    case: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": build_messages(case),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    response = requests.post(server_url, json=payload, timeout=timeout_s)
    latency_s = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("response has no choices")
    message = (choices[0] or {}).get("message") or {}
    raw = _normalise_content(message.get("content"))
    rationale, calls = parse_tool_calls(raw)
    first = calls[0] if calls else None
    parse_error = first.parse_error if first is not None else None
    call = None
    if first is not None and parse_error is None:
        call = {"name": first.name, "arguments": first.args}
    if len(calls) > 1:
        parse_error = f"expected exactly one call, got {len(calls)}"
    grade, grade_reason = score_call(case, call, parse_error)
    intent_call = call or recover_unclosed_intent(raw)
    intent_grade, intent_grade_reason = score_call(case, intent_call, None)
    return {
        "status": "ok",
        "returned_model": body.get("model"),
        "latency_s": round(latency_s, 3),
        "raw_response": raw,
        "rationale": rationale,
        "call": call,
        "intent_call": intent_call,
        "protocol_call_count": len(calls),
        "parse_error": parse_error,
        "grade": grade,
        "grade_reason": grade_reason,
        "intent_grade": intent_grade,
        "intent_grade_reason": intent_grade_reason,
        "finish_reason": (choices[0] or {}).get("finish_reason"),
        "provider_reasoning": _normalise_content(
            message.get("reasoning_content")
            if message.get("reasoning_content") is not None
            else message.get("reasoning")
        ),
        "unexpected_native_tool_calls": message.get("tool_calls") or [],
        "usage": body.get("usage") or {},
    }


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    by_capability: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for result in results:
        grade = result.get("grade") or "request_error"
        by_model[result["requested_model"]][grade] += 1
        by_capability[result["requested_model"]][result["capability"]][grade] += 1
    return {
        "models": {model: dict(counts) for model, counts in sorted(by_model.items())},
        "by_capability": {
            model: {name: dict(counts) for name, counts in sorted(capabilities.items())}
            for model, capabilities in sorted(by_capability.items())
        },
    }


def render_markdown(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# VAW open-ended next-action diagnostic",
        "",
        "| Model | Preferred | Acceptable | Unsafe | Other/invalid |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, counts in summary["models"].items():
        other = sum(
            count
            for grade, count in counts.items()
            if grade not in {"preferred", "acceptable", "unsafe"}
        )
        lines.append(
            f"| `{model}` | {counts.get('preferred', 0)} | {counts.get('acceptable', 0)} | "
            f"{counts.get('unsafe', 0)} | {other} |"
        )
    lines.extend(
        [
            "",
            "## Per case",
            "",
            "| Model | Case | Capability | Call | Grade |",
            "|---|---|---|---|---|",
        ]
    )
    for result in results:
        call = result.get("call")
        call_text = "-" if call is None else f"{call['name']} {json.dumps(call['arguments'], ensure_ascii=False)}"
        lines.append(
            f"| `{result['requested_model']}` | `{result['case_id']}` | "
            f"{result['capability']} | `{call_text}` | **{result.get('grade', 'request_error')}** |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--allow-image-egress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_cases(args.cases.resolve(), set(args.case_ids) if args.case_ids else None)
    print(f"[diagnostic] {len(cases)} cases × {len(args.models)} models = {len(cases) * len(args.models)} requests")
    if args.dry_run:
        for case in cases:
            step = case["_step"]
            print(
                f"[case] {case['id']} index={case['trace_index']} "
                f"mode={step['decision_mode']} history=disabled"
            )
        return 0
    if not args.allow_image_egress:
        print("error: add --allow-image-egress to send trace images", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or REPO_ROOT / "vaw" / "out" / "next_action_diagnostics" / timestamp
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "cases": str(args.cases.resolve()),
        "cases_sha256": _sha256(args.cases.resolve()),
        "models": args.models,
        "server_url": args.server_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "system_prompt": SYSTEM_PROMPT,
        "protocol": "text",
        "history_source": None,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    results: list[dict[str, Any]] = []
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as stream:
        for model in args.models:
            for case in cases:
                common = {
                    "case_id": case["id"],
                    "capability": case["capability"],
                    "trace_index": case["trace_index"],
                    "context_image": case["_step"]["context_image"],
                    "requested_model": model,
                    "review_note": case["review_note"],
                }
                try:
                    outcome = query(
                        server_url=args.server_url,
                        model=model,
                        case=case,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout_s=args.timeout_s,
                    )
                except Exception as exc:
                    outcome = {
                        "status": "request_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "grade": "request_error",
                    }
                result = {**common, **outcome}
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                print(
                    f"[{result['grade']:14}] {model} {case['id']} "
                    f"call={(result.get('call') or {}).get('name', '-')}"
                )

    summary = summarise(results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        render_markdown(summary, results), encoding="utf-8"
    )
    print(f"[results] {output_dir}")
    return 0 if all(result["status"] != "request_error" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
