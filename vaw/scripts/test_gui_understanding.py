"""用一个较弱的 VLM 检查 VAW Canvas 是否真的容易理解。

这个脚本不是机器人 episode，也不会调用任何 VAW 工具。它只把一张已经保存的
LIBERO Canvas 发给 OpenAI-compatible VLM endpoint，并比较两种输入条件：

1. ``canvas-only``：只给 Canvas，测试 GUI 自身是否足够自解释；
2. ``runtime-context``：额外给该步的 receipt 和 state summary，模拟 VAW Runtime
   实际提供给 Agent 的完整上下文。

默认模型是 V-API 路由下的 Qwen3.5-Plus。请求中的 base64 图片不会写入磁盘；
prompt、原始回答、解析后的 JSON 和简单的确定性评分会保存在输出目录中。

示例：

    python -m vaw.scripts.test_gui_understanding \
      --image vaw/out/scripted_pick/libero_object_t0/canvas_0014.png \
      --trace vaw/out/scripted_pick/libero_object_t0/steps.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

DEFAULT_MODEL = "vapi/qwen3.5-plus"
DEFAULT_SERVER_URL = "http://127.0.0.1:8110/chat/completions"

_OUTPUT_SCHEMA = """{
  "task": "从画布读取的任务；无法读取则为 unknown",
  "revision": 0,
  "gripper_state": "open | closed | unknown",
  "selected_candidate": "候选 id 或 none/unknown",
  "view_mode": "agentview/top/left/right/low/close/unknown",
  "focus_requested": true,
  "panel_reading": {
    "scene": "全局 Scene 展示了什么",
    "focus": "Focus 面板展示了什么",
    "self": "Self/腕部证据展示了什么",
    "intent": "Now → Next 与 preview 状态表达了什么",
    "candidate_rail": "每个候选 id 分别绑定了什么视觉目标"
  },
  "candidate_bindings": [
    {
      "id": "g1",
      "visual": "该候选卡或主视图锚点对应的位置与方向",
      "selected": true,
      "preview_status": "unchecked | ik_pass | ik_fail"
    }
  ],
  "preview_attribution": [
    {"candidate_id": "g1", "ik": "pass | fail"}
  ],
  "current_scene": "只根据最新视觉证据描述当前物理场景",
  "stale_evidence": [
    {
      "id": "相关 object/candidate id",
      "description": "什么证据已经过期",
      "trust_for_current_location": false
    }
  ],
  "selected_action_interpretation": "当前选中动作意味着什么",
  "next_best_operation": "为了完成任务，下一步最合理的操作或需要补充的证据",
  "uncertainties": ["无法从界面可靠判断的内容"]
}"""


@dataclass(frozen=True)
class TraceStep:
    """与一张 Canvas 对应的轻量 trace 记录。"""

    index: int
    op: str
    receipt: str
    state: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="待理解的 Canvas PNG")
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="对应 steps.jsonl；省略时尝试读取图片同目录下的 steps.jsonl",
    )
    parser.add_argument(
        "--mode",
        choices=("canvas-only", "runtime-context", "both"),
        default="both",
        help="测试纯 GUI、真实 Runtime 上下文，或两者都测试",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认写到图片目录/gui_understanding/<timestamp>",
    )
    return parser.parse_args()


def _canvas_index(path: Path) -> int | None:
    match = re.search(r"canvas_(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def _load_trace_step(path: Path | None, image: Path) -> TraceStep | None:
    trace_path = path or image.parent / "steps.jsonl"
    if not trace_path.exists():
        return None
    wanted = _canvas_index(image)
    records = []
    for line in trace_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records.append(record)
        if record.get("canvas") == image.name or record.get("index") == wanted:
            return TraceStep(
                index=int(record["index"]),
                op=str(record.get("op") or ""),
                receipt=str(record.get("receipt") or ""),
                state=dict(record.get("state") or {}),
            )
    raise ValueError(
        f"{trace_path} 中找不到 {image.name}；可用 step indices="
        f"{[record.get('index') for record in records]}"
    )


def _image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _prompt(mode: str, step: TraceStep | None) -> str:
    common = f"""你正在评估一个机器人 Visual Action Workspace 的单帧 GUI。
请像一个将要控制机械臂的 VLM Agent 一样阅读它，而不是只做泛泛的图片描述。

你必须：
1. 判断哪些像素是当前相机证据，哪些是计划/候选标记或历史标记；
2. 分别解释 Scene、Focus、Self、Intent 和 Candidate rail；旧版 Canvas 中
   DataPanel/wrist 分别对应候选绑定与 Self 证据；
3. 判断当前选择了什么动作，以及下一步最缺少什么证据或应执行什么操作；
4. 判断带有 stale 标记的证据是否还能表示物体当前位置，并说明依据；
5. 只返回一个符合下面结构的 JSON 对象，不要使用 Markdown：

{_OUTPUT_SCHEMA}

上面 `revision: 0` 只表示该字段必须是整数，不是答案；必须从画面的
`REVISION N` 标签（或 runtime state 的 `obs_revision`）读取真实值。
"""
    if mode == "canvas-only":
        return common + "\n本轮只提供 GUI 图片，不提供任何隐藏状态或文字转录。"
    if step is None:
        raise ValueError("runtime-context 模式需要与图片对应的 steps.jsonl")
    context = {
        "step_index": step.index,
        "last_operation": step.op,
        "receipt": step.receipt,
        "state_summary": step.state,
    }
    return (
        common
        + "\n下面是 VAW Runtime 在真实运行中与该图片一起提供的文本上下文。"
        + "请结合文字 provenance 与图片，自行判断各项证据的时效性和可信度：\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def _request(
    *,
    server_url: str,
    model: str,
    api_key: str | None,
    prompt: str,
    image_url: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    start = time.monotonic()
    response = requests.post(
        server_url,
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    elapsed = time.monotonic() - start
    if response.status_code >= 400:
        raise RuntimeError(
            f"endpoint returned HTTP {response.status_code}: {response.text[:1000]}"
        )
    return response.json(), elapsed


def _response_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise ValueError(f"response has no choices: {json.dumps(body)[:500]}")
    content = (choices[0].get("message") or {}).get("content") or ""
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "") for part in content if isinstance(part, dict)
        )
    return str(content)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _normalise_ids(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    ids = set()
    for item in value:
        if isinstance(item, dict) and (
            item.get("id") is not None or item.get("candidate_id") is not None
        ):
            ids.add(str(item.get("id") or item.get("candidate_id")))
        elif isinstance(item, str):
            ids.add(item)
    return ids


def _score(parsed: dict[str, Any] | None, step: TraceStep | None) -> dict[str, Any]:
    """只评估可从 trace 精确核对的 GUI 字段，不用启发式判断场景描述。"""

    if parsed is None:
        return {"score": 0, "max_score": 8, "checks": {"valid_json": False}}
    expected = step.state if step is not None else {}
    expected_robot = expected.get("robot") or {}
    expected_gripper = (
        "open" if expected_robot.get("gripper_open") is True else
        "closed" if expected_robot.get("gripper_open") is False else None
    )
    expected_stale = {str(x) for x in expected.get("stale_objects") or []}
    expected_candidates = {
        str(item["id"]) for item in (expected.get("candidates") or [])[:5]
        if item.get("id") is not None
    }
    expected_previews = {
        str(item["candidate_id"]) for item in expected.get("previews") or []
        if item.get("candidate_id") is not None
    }
    expected_focus = bool((expected.get("focus") or {}).get("requested"))
    parsed_previews = _normalise_ids(parsed.get("preview_attribution"))
    checks = {
        "valid_json": True,
        "revision": (
            not expected or parsed.get("revision") == expected.get("obs_revision")
        ),
        "gripper_state": (
            expected_gripper is None
            or str(parsed.get("gripper_state") or "").lower() == expected_gripper
        ),
        "selected_candidate": (
            not expected
            or str(parsed.get("selected_candidate") or "").lower()
            == str(expected.get("selected_id") or "none").lower()
        ),
        "view_mode": (
            not expected
            or str(parsed.get("view_mode") or "").lower()
            == str((expected.get("view") or {}).get("preset") or "").lower()
        ),
        "stale_ids": (
            not expected_stale
            or expected_stale.issubset(_normalise_ids(parsed.get("stale_evidence")))
        ),
        "focus_requested": (
            not expected
            or parsed.get("focus_requested") is expected_focus
        ),
        "candidate_bindings": (
            not expected_candidates
            or expected_candidates.issubset(
                _normalise_ids(parsed.get("candidate_bindings"))
            )
        ),
        "preview_attribution": (
            not expected
            or parsed_previews == expected_previews
        ),
    }
    scored_names = (
        "revision",
        "gripper_state",
        "selected_candidate",
        "view_mode",
        "stale_ids",
        "focus_requested",
        "candidate_bindings",
        "preview_attribution",
    )
    return {
        "score": sum(bool(checks[name]) for name in scored_names),
        "max_score": len(scored_names),
        "checks": checks,
        "note": (
            "current_scene、面板解释和 next_best_operation 需要人工审阅；"
            "脚本不会用脆弱的关键词规则伪装成语义评分。"
        ),
    }


def _run_mode(
    args: argparse.Namespace,
    *,
    mode: str,
    step: TraceStep | None,
    image_url: str,
    output_dir: Path,
) -> bool:
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    prompt = _prompt(mode, step)
    (mode_dir / "prompt.txt").write_text(prompt)
    (mode_dir / "request.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "server_url": args.server_url,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "image": str(args.image.resolve()),
                "image_bytes": args.image.stat().st_size,
                "image_data_url": "<omitted>",
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    try:
        body, elapsed = _request(
            server_url=args.server_url,
            model=args.model,
            api_key=args.api_key,
            prompt=prompt,
            image_url=image_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "model": args.model,
            "server_url": args.server_url,
        }
        (mode_dir / "error.json").write_text(
            json.dumps(error, ensure_ascii=False, indent=2)
        )
        print(f"[{mode}] FAILED: {error['type']}: {error['message']}")
        return False
    text = _response_text(body)
    parsed = _parse_json_object(text)
    score = _score(parsed, step)
    score["elapsed_s"] = round(elapsed, 3)
    score["model"] = args.model

    (mode_dir / "response.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2)
    )
    (mode_dir / "answer.txt").write_text(text)
    (mode_dir / "parsed.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2)
        if parsed is not None
        else "null\n"
    )
    (mode_dir / "score.json").write_text(
        json.dumps(score, ensure_ascii=False, indent=2)
    )
    print(
        f"[{mode}] score={score['score']}/{score['max_score']} "
        f"elapsed={elapsed:.1f}s answer={mode_dir / 'answer.txt'}"
    )
    return True


def main() -> int:
    args = _parse_args()
    args.image = args.image.resolve()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    step = _load_trace_step(args.trace, args.image)
    if args.mode in {"runtime-context", "both"} and step is None:
        raise FileNotFoundError(
            "runtime-context 模式需要 steps.jsonl；请用 --trace 显式指定"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.image.parent / "gui_understanding" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "image": str(args.image),
                "trace_step": step.index if step else None,
                "model": args.model,
                "server_url": args.server_url,
                "mode": args.mode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    image_url = _image_data_url(args.image)
    modes = ("canvas-only", "runtime-context") if args.mode == "both" else (args.mode,)
    passed = []
    for mode in modes:
        passed.append(_run_mode(
            args,
            mode=mode,
            step=step,
            image_url=image_url,
            output_dir=output_dir,
        ))
    print(f"results: {output_dir}")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
