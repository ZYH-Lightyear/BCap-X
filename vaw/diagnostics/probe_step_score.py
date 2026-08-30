"""Probe a multimodal model as a function-aware VAW step critic.

The diagnostic deliberately uses only evidence available at one trace step:
the task, the function call/result, and the before/after context canvases.  It
does not expose episode success, future steps, or the hand-authored gold label
to the model.  The OpenAI-compatible endpoint must return token logprobs so the
script can also inspect the model's uncertainty over the five score labels.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE = REPO_ROOT / "vaw" / "out" / "sweeps" / "gen0" / "task0_s1"
DEFAULT_SERVER = "http://127.0.0.1:5088/v1/chat/completions"
DEFAULT_MODEL = "/mnt/nas/maqi/Qwen3_5_27B"

LABELS = {
    "A": "ADVANCE：该步让真实物理任务状态明显接近成功。",
    "B": "SUPPORT：该步没有改变物理任务状态，但获得了后续行动所需的有效证据或方案。",
    "C": "NEUTRAL：该步既没有产生有用进展，也没有明显破坏任务状态。",
    "D": "HARM：该步破坏了已经取得的进展，或让物理任务状态明显变差。",
    "E": "INVALID：调用失败，或现有证据不足以判断该步造成了什么结果。",
}

# These labels are evaluation annotations and are never included in the model request.
CASES = (
    {
        "case_id": "grounding_support",
        "step_index": 0,
        "before_index": 0,
        "after_index": 1,
        "gold_label": "B",
        "rationale": "Detection grounds the target but does not change the physical scene.",
    },
    {
        "case_id": "lift_advance",
        "step_index": 6,
        "before_index": 6,
        "after_index": 7,
        "gold_label": "A",
        "rationale": "The held alphabet-soup can is visibly lifted away from the floor.",
    },
    {
        "case_id": "transport_drop_harm",
        "step_index": 12,
        "before_index": 12,
        "after_index": 13,
        "gold_label": "D",
        "rationale": "The can is held before commit and lies on the floor afterward.",
    },
)

SYSTEM_PROMPT = """你是机器人任务的逐步诊断器。你只评价本轮函数调用对当前任务造成的直接影响。

评分标签：
{labels}

判断规则：
- 以 BEFORE/AFTER 的可见物理证据为准，不采信 Agent 对成功的自我声明。
- 感知、定位、规划或预览若提供了可用于下一步的有效信息，但尚未产生物理进展，应判 B，不判 A。
- 真实抓取、抬升、运输或放置使任务状态更接近成功时判 A。
- 掉落、抓错、碰撞或丢失已有进展时判 D，即使函数返回看似成功。
- 仅当调用失败或前后证据确实无法判断时判 E。

只输出一个大写字母 A、B、C、D 或 E；不要解释。""".format(
    labels="\n".join(f"{key}. {value}" for key, value in LABELS.items())
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _load_steps(trace_dir: Path) -> list[dict[str, Any]]:
    path = trace_dir / "steps.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"trace has no steps.jsonl: {trace_dir}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _compact_trace(step: dict[str, Any]) -> str:
    payload = {
        "turn": step.get("turn"),
        "function_call": step.get("function_call"),
        "function_result": step.get("function_result"),
        "result_manifest": step.get("result_manifest"),
        "runtime_diagnostics": step.get("runtime_diagnostics"),
        "revision_before": step.get("revision_before"),
        "revision_after": step.get("revision_after"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_messages(
    *, task: str, step: dict[str, Any], before_path: Path, after_path: Path
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"任务：{task}\n\n"
                        "下面是同一个机器人步骤执行前后的 VAW context canvas。"
                        "每张 canvas 同时包含主相机、腕部相机和当时的界面标记。\n\n"
                        "BEFORE："
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(before_path), "detail": "high"},
                },
                {"type": "text", "text": "AFTER："},
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(after_path), "detail": "high"},
                },
                {
                    "type": "text",
                    "text": f"本步函数 Trace：\n{_compact_trace(step)}\n\n输出标签：",
                },
            ],
        },
    ]


def _extract_label(text: str) -> str | None:
    match = re.search(r"(?<![A-Z])[ABCDE](?![A-Z])", text.upper())
    return match.group(0) if match else None


def _label_distribution(choice: dict[str, Any]) -> dict[str, Any]:
    content = ((choice.get("logprobs") or {}).get("content") or [])
    label_position: dict[str, Any] | None = None
    for position in content:
        token = str(position.get("token") or "")
        if _extract_label(token) is not None:
            label_position = position
            break
    if label_position is None and content:
        label_position = content[0]
    if label_position is None:
        return {"available": False, "reason": "response has no token logprobs"}

    per_label_logprobs: dict[str, list[float]] = {label: [] for label in LABELS}
    candidates = [
        {"token": label_position.get("token"), "logprob": label_position.get("logprob")},
        *(label_position.get("top_logprobs") or []),
    ]
    seen: set[tuple[str, float]] = set()
    raw_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        token = str(candidate.get("token") or "")
        try:
            logprob = float(candidate.get("logprob"))
        except (TypeError, ValueError):
            continue
        key = (token, logprob)
        if key in seen:
            continue
        seen.add(key)
        raw_candidates.append({"token": token, "logprob": logprob})
        label = _extract_label(token.strip())
        if label is not None:
            per_label_logprobs[label].append(logprob)

    label_logprobs: dict[str, float | None] = {}
    for label, values in per_label_logprobs.items():
        if not values:
            label_logprobs[label] = None
            continue
        maximum = max(values)
        label_logprobs[label] = maximum + math.log(sum(math.exp(v - maximum) for v in values))

    finite = {label: value for label, value in label_logprobs.items() if value is not None}
    if finite:
        maximum = max(finite.values())
        denominator = sum(math.exp(value - maximum) for value in finite.values())
        probabilities = {
            label: (math.exp(value - maximum) / denominator if value is not None else None)
            for label, value in label_logprobs.items()
        }
    else:
        probabilities = {label: None for label in LABELS}
    return {
        "available": True,
        "answer_token": label_position.get("token"),
        "label_logprobs": label_logprobs,
        "label_probabilities_normalized_over_visible_labels": probabilities,
        "raw_top_candidates": raw_candidates,
    }


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "vaw" / "out" / "step_score_probes" / f"qwen35_27b_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--server-url", default=DEFAULT_SERVER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_dir = args.trace.resolve()
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    steps = _load_steps(trace_dir)

    meta_path = trace_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    task = str(meta.get("task_prompt") or meta.get("task_description") or meta.get("task") or "")
    if not task:
        task = "Pick the alphabet soup and place it in the basket"

    run_config = {
        "trace": str(trace_dir),
        "trace_meta": meta,
        "task": task,
        "server_url": args.server_url,
        "model": args.model,
        "system_prompt": SYSTEM_PROMPT,
        "cases": list(CASES),
        "note": "Gold labels and future episode state were not sent to the model.",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    results: list[dict[str, Any]] = []
    for case in CASES:
        step = steps[int(case["step_index"])]
        before_path = trace_dir / f"context_{int(case['before_index']):04d}.png"
        after_path = trace_dir / f"context_{int(case['after_index']):04d}.png"
        for path in (before_path, after_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        messages = _build_messages(
            task=task, step=step, before_path=before_path, after_path=after_path
        )
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 4,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        response = requests.post(args.server_url, json=payload, timeout=args.timeout_s)
        latency_s = time.perf_counter() - started
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ValueError(f"response has no choices: {body}")
        choice = choices[0]
        raw_text = str(((choice.get("message") or {}).get("content")) or "")
        predicted = _extract_label(raw_text)
        result = {
            **case,
            "before_image": str(before_path),
            "before_sha256": _sha256(before_path),
            "after_image": str(after_path),
            "after_sha256": _sha256(after_path),
            "task": task,
            "trace_evidence": json.loads(_compact_trace(step)),
            "predicted_label": predicted,
            "correct": predicted == case["gold_label"],
            "raw_response": raw_text,
            "latency_s": round(latency_s, 3),
            "returned_model": body.get("model"),
            "usage": body.get("usage") or {},
            "logprobs": _label_distribution(choice),
        }
        results.append(result)
        case_dir = output_dir / str(case["case_id"])
        case_dir.mkdir()
        safe_request = {
            "model": args.model,
            "system_prompt": SYSTEM_PROMPT,
            "task": task,
            "trace_evidence": result["trace_evidence"],
            "before_image": str(before_path),
            "before_sha256": result["before_sha256"],
            "after_image": str(after_path),
            "after_sha256": result["after_sha256"],
            "note": "Base64 image payload omitted; gold label was not sent.",
        }
        (case_dir / "request.json").write_text(
            json.dumps(safe_request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (case_dir / "response.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{case['case_id']}] gold={case['gold_label']} pred={predicted or '-'} "
            f"latency={latency_s:.2f}s logprobs={result['logprobs']['available']}"
        )

    accuracy = sum(bool(result["correct"]) for result in results) / len(results)
    summary = {
        "model": args.model,
        "trace": str(trace_dir),
        "case_count": len(results),
        "correct": sum(bool(result["correct"]) for result in results),
        "accuracy": accuracy,
        "all_have_logprobs": all(result["logprobs"]["available"] for result in results),
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[summary] accuracy={accuracy:.1%} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
