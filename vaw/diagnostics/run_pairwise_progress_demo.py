"""Fit continuous task-progress utility from sparse visual state comparisons."""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests

from vaw.diagnostics.progress_critic import (
    build_action_evidence,
    extract_label,
    sha256,
    trace_task,
)
from vaw.diagnostics.progress_observations import (
    MultiViewObservation,
    build_multiview_transitions,
)
from vaw.diagnostics.progress_video import render_progress_video
from vaw.diagnostics.run_progress_demo import DEFAULT_MODEL, DEFAULT_SERVER

LABELS = {
    "A": "RIGHT 明显比 LEFT 更接近任务目标。",
    "B": "两者任务进展基本相同，或视觉证据不足以区分。",
    "C": "LEFT 明显比 RIGHT 更接近任务目标。",
}

SYSTEM_PROMPT = """你是独立的机器人任务状态比较器。你不估计完成百分比，只比较两组同步原始相机观测中哪一组更接近用户任务的成功条件。

标签：
{labels}

规则：
- 每个状态包含全局 AgentView 和局部 Wrist/Hand Camera；必须综合两种视角。
- 依据物体身份、抓取关系、运输、空间对齐、释放和稳定性等可见物理证据比较。
- 不相信函数返回值或 Agent 的成功声明；动作名称不能覆盖视觉证据。
- 输入不包含 Canvas、Imagination overlay、检测框或策略 UI。
- 掉落、抓错、失去已有目标关系时，退化后的状态应输给此前较好的状态。
- 差异很小、只改变观察方式或无法可靠判断时选择 B。

只输出一个大写字母 A、B 或 C；不要输出数字、百分比或解释。""".format(
    labels="\n".join(f"{label}. {text}" for label, text in LABELS.items())
)


@dataclass(frozen=True)
class ComparisonEdge:
    kind: str
    left_state: int
    right_state: int
    preference_signal: float
    weight: float
    probabilities: dict[str, float]
    predicted_label: str | None
    response_path: str


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image(path: Path, *, detail: str = "high") -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": _data_url(path), "detail": detail},
    }


def _distribution(choice: dict[str, Any]) -> dict[str, float]:
    content = ((choice.get("logprobs") or {}).get("content") or [])
    position: dict[str, Any] | None = None
    for item in content:
        if extract_label(str(item.get("token") or "")) in LABELS:
            position = item
            break
    if position is None and content:
        position = content[0]
    if position is None:
        raise ValueError("pairwise response contains no token logprobs")

    candidates = [
        {"token": position.get("token"), "logprob": position.get("logprob")},
        *(position.get("top_logprobs") or []),
    ]
    values: dict[str, list[float]] = {label: [] for label in LABELS}
    for candidate in candidates:
        label = extract_label(str(candidate.get("token") or "").strip())
        if label not in LABELS:
            continue
        try:
            values[label].append(float(candidate.get("logprob")))
        except (TypeError, ValueError):
            continue

    combined: dict[str, float | None] = {}
    for label, logprobs in values.items():
        if not logprobs:
            combined[label] = None
            continue
        maximum = max(logprobs)
        combined[label] = maximum + math.log(
            sum(math.exp(value - maximum) for value in logprobs)
        )
    finite = {label: value for label, value in combined.items() if value is not None}
    if not finite:
        raise ValueError("pairwise top logprobs contain no A/B/C labels")
    maximum = max(finite.values())
    denominator = sum(math.exp(value - maximum) for value in finite.values())
    return {
        label: (
            math.exp(value - maximum) / denominator if value is not None else 0.0
        )
        for label, value in combined.items()
    }


def edge_target(probabilities: dict[str, float]) -> tuple[float, float]:
    """Return the expected signed RIGHT-vs-LEFT preference and confidence."""

    total = sum(max(0.0, float(probabilities.get(label, 0.0))) for label in LABELS)
    if total <= 0:
        raise ValueError("comparison probabilities have zero mass")
    normalized = {
        label: max(0.0, float(probabilities.get(label, 0.0))) / total
        for label in LABELS
    }
    epsilon = 1e-6
    target = normalized["A"] - normalized["C"]
    entropy = -sum(
        probability * math.log(max(probability, epsilon))
        for probability in normalized.values()
    )
    confidence = max(1e-3, 1.0 - entropy / math.log(len(LABELS)))
    return target, confidence


def fit_latent_progress(
    state_count: int, edges: list[ComparisonEdge]
) -> np.ndarray:
    """Fit Bradley–Terry-style utilities without mapping labels to percentages."""

    if state_count <= 0:
        raise ValueError("state_count must be positive")
    rows: list[np.ndarray] = []
    targets: list[float] = []
    anchor = np.zeros(state_count, dtype=np.float64)
    anchor[0] = 1.0
    rows.append(anchor * 10.0)
    targets.append(0.0)
    for edge in edges:
        row = np.zeros(state_count, dtype=np.float64)
        scale = math.sqrt(max(edge.weight, 1e-6))
        row[edge.left_state] = -scale
        row[edge.right_state] = scale
        rows.append(row)
        targets.append(edge.preference_signal * scale)
    matrix = np.stack(rows)
    solution, *_ = np.linalg.lstsq(matrix, np.asarray(targets), rcond=None)
    return solution - solution[0]


class PairwiseCriticClient:
    def __init__(
        self,
        *,
        server_url: str,
        model: str,
        timeout_s: float,
        session: requests.Session | None = None,
    ) -> None:
        self.server_url = server_url
        self.model = model
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def compare(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.perf_counter()
        response = self.session.post(
            self.server_url,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 4,
                "logprobs": True,
                "top_logprobs": 20,
                "structured_outputs": {"choice": list(LABELS)},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=self.timeout_s,
        )
        latency_s = time.perf_counter() - started
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ValueError(f"pairwise critic response has no choices: {body}")
        choice = choices[0]
        raw = str(((choice.get("message") or {}).get("content")) or "")
        probabilities = _distribution(choice)
        target, weight = edge_target(probabilities)
        return {
            "predicted_label": extract_label(raw),
            "probabilities": probabilities,
            "preference_signal": target,
            "weight": weight,
            "raw_response": raw,
            "latency_s": round(latency_s, 3),
            "usage": body.get("usage") or {},
            "returned_model": body.get("model"),
        }


def _comparison_messages(
    *,
    task: str,
    left: MultiViewObservation,
    right: MultiViewObservation,
    left_name: str,
    right_name: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"用户任务：{task}\n\n"
                "比较 LEFT 和 RIGHT 两组同步原始观测，判断哪一个真实物理状态"
                "更接近任务成功。AgentView 表示全局关系，Wrist Camera 表示夹爪附近细节。"
            ),
        }
    ]
    content.extend(
        [
            {"type": "text", "text": f"LEFT · {left_name} · AGENTVIEW："},
            _image(left.agentview),
            {"type": "text", "text": f"LEFT · {left_name} · WRIST CAMERA："},
            _image(left.wrist),
            {"type": "text", "text": f"RIGHT · {right_name} · AGENTVIEW："},
            _image(right.agentview),
            {"type": "text", "text": f"RIGHT · {right_name} · WRIST CAMERA："},
            _image(right.wrist),
        ]
    )
    content.append({"type": "text", "text": "输出比较标签："})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _score_edge(
    *,
    client: PairwiseCriticClient,
    output_dir: Path,
    edge_id: str,
    kind: str,
    left_state: int,
    right_state: int,
    left: MultiViewObservation,
    right: MultiViewObservation,
    messages: list[dict[str, Any]],
    resume: bool,
) -> ComparisonEdge:
    edge_dir = output_dir / "comparisons" / edge_id
    response_path = edge_dir / "response.json"
    if resume and response_path.is_file():
        response = json.loads(response_path.read_text(encoding="utf-8"))
    else:
        edge_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            edge_dir / "request.json",
            {
                "system_prompt": SYSTEM_PROMPT,
                "kind": kind,
                "left_state": left_state,
                "right_state": right_state,
                "left": {
                    "frame_index": left.frame_index,
                    "agentview": {
                        "path": str(left.agentview),
                        "sha256": sha256(left.agentview),
                    },
                    "wrist": {
                        "path": str(left.wrist),
                        "sha256": sha256(left.wrist),
                    },
                },
                "right": {
                    "frame_index": right.frame_index,
                    "agentview": {
                        "path": str(right.agentview),
                        "sha256": sha256(right.agentview),
                    },
                    "wrist": {
                        "path": str(right.wrist),
                        "sha256": sha256(right.wrist),
                    },
                },
                "privacy_note": "Base64 payloads and evaluator success are not stored.",
            },
        )
        response = client.compare(messages)
        _write_json(response_path, response)
    return ComparisonEdge(
        kind=kind,
        left_state=left_state,
        right_state=right_state,
        preference_signal=float(response["preference_signal"]),
        weight=float(response["weight"]),
        probabilities={
            label: float(value)
            for label, value in (response.get("probabilities") or {}).items()
        },
        predicted_label=response.get("predicted_label"),
        response_path=str(response_path),
    )


def score_pairwise_trace(
    *,
    trace_dir: Path,
    output_dir: Path,
    server_url: str,
    model: str,
    timeout_s: float,
    resume: bool,
    max_actions: int | None,
) -> Path:
    task = trace_task(trace_dir)
    actions = build_action_evidence(trace_dir)
    if max_actions is not None:
        actions = actions[:max_actions]
    output_dir.mkdir(parents=True, exist_ok=True)
    initial, transitions = build_multiview_transitions(
        trace_dir=trace_dir,
        actions=actions,
        output_dir=output_dir / "evidence",
    )
    client = PairwiseCriticClient(
        server_url=server_url, model=model, timeout_s=timeout_s
    )
    state_observations = [initial, *(transition.after for transition in transitions)]
    edges: list[ComparisonEdge] = []
    cumulative_local = 0.0
    high_water_value = 0.0
    high_water_state = 0
    total_latency_s = 0.0

    for state_index, transition in enumerate(transitions, start=1):
        action = transition.action
        local = _score_edge(
            client=client,
            output_dir=output_dir,
            edge_id=f"state_{state_index:04d}_local",
            kind="local",
            left_state=state_index - 1,
            right_state=state_index,
            left=transition.before,
            right=transition.after,
            messages=_comparison_messages(
                task=task,
                left=transition.before,
                right=transition.after,
                left_name="BEFORE",
                right_name="AFTER",
            ),
            resume=resume,
        )
        edges.append(local)
        cumulative_local += local.preference_signal

        if high_water_state not in {0, state_index - 1}:
            high_edge = _score_edge(
                client=client,
                output_dir=output_dir,
                edge_id=f"state_{state_index:04d}_high_water_{high_water_state:04d}",
                kind="high_water_anchor",
                left_state=high_water_state,
                right_state=state_index,
                left=state_observations[high_water_state],
                right=transition.after,
                messages=_comparison_messages(
                    task=task,
                    left=state_observations[high_water_state],
                    right=transition.after,
                    left_name=f"HISTORICAL HIGH-WATER {high_water_state}",
                    right_name="CURRENT",
                ),
                resume=resume,
            )
            edges.append(high_edge)

        if cumulative_local > high_water_value:
            high_water_value = cumulative_local
            high_water_state = state_index
        print(
            f"[pairwise {state_index:02d}/{len(actions):02d}] "
            f"turn={action.turn} local={local.preference_signal:+.3f} "
            f"high_water={high_water_state}"
        )

    scores = fit_latent_progress(len(actions) + 1, edges)
    states: list[dict[str, Any]] = []
    previous_delta: float | None = None
    for state_index, score in enumerate(scores):
        action = actions[state_index - 1] if state_index else None
        delta = None if state_index == 0 else float(score - scores[state_index - 1])
        acceleration = (
            delta - previous_delta
            if delta is not None and previous_delta is not None
            else None
        )
        states.append(
            {
                "state_index": state_index,
                "turn": action.turn if action is not None else 0,
                "function": action.function if action is not None else "initial_state",
                "time_s": action.time_end_s if action is not None else 0.0,
                "latent_progress": float(score),
                "latent_delta": delta,
                "latent_acceleration": acceleration,
                "frame_index": state_observations[state_index].frame_index,
                "agentview": str(state_observations[state_index].agentview),
                "wrist": str(state_observations[state_index].wrist),
            }
        )
        if delta is not None:
            previous_delta = delta
    for edge in edges:
        response = json.loads(Path(edge.response_path).read_text(encoding="utf-8"))
        total_latency_s += float(response.get("latency_s") or 0.0)

    payload = {
        "schema": "vaw-pairwise-latent-progress-v2",
        "score_kind": "pairwise_latent",
        "evidence_kind": "raw_synchronized_agentview_wrist",
        "task": task,
        "trace": str(trace_dir),
        "model": model,
        "server_url": server_url,
        "state_count": len(states),
        "comparison_count": len(edges),
        "total_latency_s": round(total_latency_s, 3),
        "scale_note": (
            "Latent values fit expected signed pairwise preferences, not "
            "task-completion percentages. "
            "Only local and historical-high-water comparisons are fitted; only "
            "within-trace differences and ordering are interpreted."
        ),
        "states": states,
        "edges": [asdict(edge) for edge in edges],
    }
    output_path = output_dir / "pairwise_progress.json"
    _write_json(output_path, payload)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--server-url", default=DEFAULT_SERVER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--render-video", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_dir = args.trace.resolve()
    output_dir = (
        args.output_dir
        or trace_dir
        / "progress_critic"
        / f"pairwise_latent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ).resolve()
    progress_path = score_pairwise_trace(
        trace_dir=trace_dir,
        output_dir=output_dir,
        server_url=args.server_url,
        model=args.model,
        timeout_s=args.timeout_s,
        resume=args.resume,
        max_actions=args.max_actions,
    )
    if args.render_video:
        video = render_progress_video(
            trace_dir=trace_dir,
            progress_path=progress_path,
            output_path=output_dir / "pairwise_progress_overlay.mp4",
        )
        print(f"[video] {video}")
    print(f"[pairwise-progress] {progress_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
