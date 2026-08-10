"""Run an isolated visual diagnostic against OpenAI-compatible VAPI models.

The harness deliberately sends one raster and one multiple-choice question.
It does not include the Agent system prompt, task history, manifest, tools, or
environment state, so failures are attributable to visual interpretation more
cleanly than failures from a full episode.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(__file__).with_name("static_vlm_cases.json")
DEFAULT_MODELS = ("vapi/gpt-5.5", "vapi/qwen3.5-plus")
SYSTEM_PROMPT = """你正在完成一个隔离的机器人视觉诊断题。
只依据本轮提供的单张图像和问题作答，不假设任何未显示的历史、动作结果或任务流程。
严格区分真实观测与紫色虚拟 preview；规划成功、夹爪闭合或二维重叠都不等于物理效果成功。
选择最符合可见证据的一项，并用一句具体的视觉证据说明原因。
只输出 JSON：{"choice":"A","visual_evidence":"...","confidence":"high|medium|low"}。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(dataset_path: Path, case_ids: set[str] | None = None) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset must contain a non-empty 'cases' list")

    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id") or "")
        if not case_id or case_id in seen:
            raise ValueError(f"case id is missing or duplicated: {case_id!r}")
        seen.add(case_id)
        if case_ids is not None and case_id not in case_ids:
            continue

        choices = case.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError(f"{case_id}: choices must contain at least two items")
        choice_ids = [str(choice.get("id") or "").upper() for choice in choices]
        if len(set(choice_ids)) != len(choice_ids) or any(not item for item in choice_ids):
            raise ValueError(f"{case_id}: choice ids must be unique and non-empty")
        answer = str(case.get("answer") or "").upper()
        if answer not in choice_ids:
            raise ValueError(f"{case_id}: answer {answer!r} is not a choice")

        image_path = (REPO_ROOT / str(case.get("image") or "")).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"{case_id}: image does not exist: {image_path}")
        expected_hash = str(case.get("image_sha256") or "")
        actual_hash = _sha256(image_path)
        if expected_hash != actual_hash:
            raise ValueError(
                f"{case_id}: image hash changed; expected {expected_hash}, got {actual_hash}"
            )
        selected.append({**case, "_image_path": image_path, "answer": answer})

    if case_ids is not None:
        missing = sorted(case_ids - {case["id"] for case in selected})
        if missing:
            raise ValueError(f"unknown case ids: {', '.join(missing)}")
    return selected


def question_text(case: dict[str, Any]) -> str:
    choices = "\n".join(
        f"{str(choice['id']).upper()}. {choice['text']}" for choice in case["choices"]
    )
    return f"问题：{case['question']}\n\n选项：\n{choices}"


def build_messages(case: dict[str, Any]) -> list[dict[str, Any]]:
    image_path = Path(case["_image_path"])
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question_text(case)},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "high"},
                },
            ],
        },
    ]


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
    if value is None:
        return ""
    return str(value)


def extract_answer(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None, "response does not contain a JSON object"
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "response JSON is not an object"
    choice = str(parsed.get("choice") or "").strip().upper()
    if not choice:
        return None, "response JSON has no choice"
    parsed["choice"] = choice
    return parsed, None


def query_case(
    *,
    server_url: str,
    model: str,
    case: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": build_messages(case),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    started = time.perf_counter()
    response = requests.post(server_url, json=payload, timeout=timeout_s)
    latency_s = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise ValueError(f"response has no choices: {json.dumps(body, ensure_ascii=False)[:300]}")
    message = (choices[0] or {}).get("message") or {}
    raw_text = _normalise_content(message.get("content"))
    parsed, parse_error = extract_answer(raw_text)
    predicted = parsed.get("choice") if parsed else None
    return {
        "status": "ok" if parse_error is None else "parse_error",
        "requested_model": model,
        "returned_model": body.get("model"),
        "latency_s": round(latency_s, 3),
        "raw_response": raw_text,
        "provider_reasoning": _normalise_content(
            message.get("reasoning_content")
            if message.get("reasoning_content") is not None
            else message.get("reasoning")
        ),
        "parsed": parsed,
        "parse_error": parse_error,
        "predicted_choice": predicted,
        "correct": predicted == case["answer"],
        "usage": body.get("usage") or {},
    }


def _summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "scored": 0, "errors": 0})
    category_buckets: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0, "scored": 0, "errors": 0})
    )
    for result in results:
        model = result["requested_model"]
        category = result["category"]
        if result["status"] == "request_error":
            buckets[model]["errors"] += 1
            category_buckets[model][category]["errors"] += 1
            continue
        buckets[model]["scored"] += 1
        category_buckets[model][category]["scored"] += 1
        if result.get("correct"):
            buckets[model]["correct"] += 1
            category_buckets[model][category]["correct"] += 1

    def finalise(values: dict[str, int]) -> dict[str, Any]:
        scored = values["scored"]
        return {**values, "accuracy": values["correct"] / scored if scored else None}

    return {
        "models": {model: finalise(values) for model, values in sorted(buckets.items())},
        "by_category": {
            model: {category: finalise(values) for category, values in sorted(categories.items())}
            for model, categories in sorted(category_buckets.items())
        },
    }


def _markdown_summary(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# VAW static VLM diagnostic",
        "",
        "| Model | Correct | Scored | Request errors | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, values in summary["models"].items():
        accuracy = "n/a" if values["accuracy"] is None else f"{values['accuracy']:.1%}"
        lines.append(
            f"| `{model}` | {values['correct']} | {values['scored']} | "
            f"{values['errors']} | {accuracy} |"
        )
    lines.extend(
        [
            "",
            "## Per case",
            "",
            "| Model | Case | Category | Gold | Predicted | Status |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for result in results:
        if result["status"] == "request_error":
            verdict = "request_error"
        elif result["status"] == "parse_error":
            verdict = "parse_error"
        else:
            verdict = "correct" if result.get("correct") else "wrong"
        lines.append(
            f"| `{result['requested_model']}` | `{result['case_id']}` | "
            f"{result['category']} | {result['gold_choice']} | "
            f"{result.get('predicted_choice') or '-'} | {verdict} |"
        )
    return "\n".join(lines) + "\n"


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "vaw" / "out" / "static_vlm_diagnostics" / timestamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"))
    parser.add_argument(
        "--allow-image-egress",
        action="store_true",
        help="Acknowledge that the selected trace PNGs will be sent to the configured endpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate cases and print the request matrix without sending images.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    cases = load_cases(args.dataset.resolve(), set(args.case_ids) if args.case_ids else None)
    matrix = len(cases) * len(args.models) * args.repeats
    print(f"[diagnostic] {len(cases)} cases × {len(args.models)} models × {args.repeats} = {matrix} requests")
    for model in args.models:
        print(f"[model] {model}")
    if args.dry_run:
        for case in cases:
            print(f"[case] {case['id']}  {case['category']}  {case['image']}")
        return 0
    if not args.allow_image_egress:
        print("error: add --allow-image-egress to send trace images to the endpoint", file=sys.stderr)
        return 2

    output_dir = (args.output_dir or _default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset.resolve()),
        "models": args.models,
        "server_url": args.server_url,
        "repeats": args.repeats,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "system_prompt": SYSTEM_PROMPT,
        "case_ids": [case["id"] for case in cases],
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    results: list[dict[str, Any]] = []
    result_path = output_dir / "results.jsonl"
    with result_path.open("w", encoding="utf-8") as stream:
        for model in args.models:
            for repeat_index in range(args.repeats):
                for case in cases:
                    common = {
                        "case_id": case["id"],
                        "category": case["category"],
                        "canvas_version": case.get("canvas_version"),
                        "image": case["image"],
                        "image_sha256": case["image_sha256"],
                        "question": case["question"],
                        "gold_choice": case["answer"],
                        "repeat_index": repeat_index,
                        "requested_model": model,
                    }
                    try:
                        outcome = query_case(
                            server_url=args.server_url,
                            model=model,
                            case=case,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            timeout_s=args.timeout_s,
                            reasoning_effort=args.reasoning_effort,
                        )
                    except Exception as exc:
                        outcome = {
                            "status": "request_error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "predicted_choice": None,
                            "correct": False,
                        }
                    result = {**common, **outcome}
                    results.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stream.flush()
                    if result["status"] == "request_error":
                        marker = "request_error"
                    elif result["status"] == "parse_error":
                        marker = "parse_error"
                    else:
                        marker = "correct" if result.get("correct") else "wrong"
                    print(
                        f"[{marker:13}] {model} {case['id']} "
                        f"gold={case['answer']} pred={result.get('predicted_choice') or '-'}"
                    )

    summary = _summarise(results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        _markdown_summary(summary, results), encoding="utf-8"
    )
    print(f"[results] {output_dir}")
    return 0 if all(item["status"] != "request_error" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
