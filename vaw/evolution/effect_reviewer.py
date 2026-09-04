"""用独立多模态 Agent 比较技能变更前后的局部作用。

终局 success 只能回答任务是否完成。本模块让 Reviewer 主动读取同一 task/seed
的 baseline 与 candidate 视觉片段，判断候选技能针对的问题是否真的改善。
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from vaw.agents.contracts import ModelResponse
from vaw.agents.providers.base import ModelProvider
from vaw.diagnostics.progress_critic import ActionEvidence, build_action_evidence, trace_task
from vaw.evolution.artifacts import image_part, parse_model_json, read_json, sha256, write_json
from vaw.evolution.domain import (
    RemainingFailure,
    SkillEffect,
    SkillEffectReport,
    SkillEffectReview,
)
from vaw.evolution.gate import assert_paired_runs, consulted_skill, load_day_index
from vaw.evolution.store import GenerationStore

INSPECT_PAIR_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_pair",
        "description": (
            "并排读取 baseline 与 candidate 的策略可见真实视觉片段。"
            "用它检验候选技能针对的问题是否改善；每侧通常选择 1–3 个物理动作。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "baseline_start": {"type": "integer"},
                "baseline_end": {"type": "integer"},
                "candidate_start": {"type": "integer"},
                "candidate_end": {"type": "integer"},
                "question": {
                    "type": "string",
                    "description": "希望这组视觉证据回答的一个具体问题。",
                },
            },
            "required": [
                "baseline_start",
                "baseline_end",
                "candidate_start",
                "candidate_end",
                "question",
            ],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """你是独立的 Skill Effect Reviewer，不负责控制机器人，也不负责写技能。

你要比较相同 task、seed 和初始状态的 baseline/candidate rollout，回答：候选技能针对的决策问题是否被改善。最终任务成功或失败只是系统结果，不能替代局部视觉证据。先根据动作索引定位相关区间，再调用 inspect_pair 查看真实策略可见画面；不要从 planner、模型自述或函数返回猜测物理效果。

effect：
- improved：候选轨迹在目标问题上有清晰、可见的改善；
- unchanged：没有可信变化；
- worse：目标问题变差或出现由该技能直接引入的新问题；
- uncertain：现有画面不足以判断。

remaining_failure：
- none：candidate 最终成功；
- related：仍因同一目标问题失败；
- other：目标问题已改善，但后来因另一类可见问题失败；
- uncertain：无法可靠归因。

证据充分后只输出 JSON：
{"effect":"improved|unchanged|worse|uncertain","remaining_failure":"none|related|other|uncertain","evidence_ids":["..."],"reason":"一句简洁的视觉依据"}

evidence_ids 必须引用已经检查的证据。没有足够证据时也要先检查至少一组片段，再输出 uncertain。"""


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        size,
    )


def _fit(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    target = Image.new("RGB", size, "#e8eef5")
    target.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return target


def _visible_args(action: ActionEvidence) -> str:
    values = {key: value for key, value in action.arguments.items() if key != "action_id"}
    if not values:
        return ""
    text = str(values).replace("'", "")
    return f" {text[:56]}"


def _timeline(actions: Sequence[ActionEvidence]) -> str:
    return "\n".join(
        f"A{item.index}: {item.function}{_visible_args(item)}"
        for item in actions
    )


def _selected(
    actions: Sequence[ActionEvidence],
    start: int,
    end: int,
) -> list[ActionEvidence]:
    if start < 1 or end < start or end > len(actions):
        raise ValueError(f"动作区间必须位于 A1..A{len(actions)}")
    return list(actions[start - 1 : end])


def _render_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    actions: Sequence[ActionEvidence],
    *,
    y: int,
    label: str,
) -> None:
    panels = [
        panel
        for action in actions
        for panel in (
            (f"A{action.index} BEFORE", action.before_canvas),
            (f"A{action.index} AFTER", action.after_canvas),
        )
    ]
    panel_gap = 10
    left = 18
    width = (canvas.width - left * 2 - panel_gap * (len(panels) - 1)) // len(panels)
    height = 410
    draw.text((left, y), label, font=_font(26, bold=True), fill="#172554")
    for index, (title, path) in enumerate(panels):
        x = left + index * (width + panel_gap)
        draw.rectangle((x, y + 42, x + width, y + 42 + height), fill="#ffffff", outline="#94a3b8", width=2)
        draw.text((x + 9, y + 50), title, font=_font(17, bold=True), fill="#0f172a")
        canvas.paste(_fit(path, (width - 12, height - 42)), (x + 6, y + 80))


def render_pair_evidence(
    baseline: Sequence[ActionEvidence],
    candidate: Sequence[ActionEvidence],
    output_path: str | Path,
) -> Path:
    """把两侧动作前后状态放在同一张图中，保持比较变量清晰。"""

    canvas = Image.new("RGB", (2400, 1010), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    _render_row(canvas, draw, baseline, y=18, label="BASELINE")
    _render_row(canvas, draw, candidate, y=510, label="CANDIDATE")
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=False)
    return target


class SkillEffectReviewer:
    """短上下文只读 Agent；每次调用只评审一个 task/seed 对。"""

    def __init__(self, provider: ModelProvider, *, max_inspections: int = 4) -> None:
        if max_inspections < 1:
            raise ValueError("max_inspections 必须大于零")
        self.provider = provider
        self.max_inspections = max_inspections

    def run(
        self,
        *,
        task_id: int,
        seed: int,
        baseline_trace: str | Path,
        candidate_trace: str | Path,
        baseline_outcome: str,
        candidate_outcome: str,
        mutation: Mapping[str, Any],
        previous_skill: str | None,
        candidate_skill: str | None,
        baseline_consulted: bool,
        candidate_consulted: bool,
        output_dir: str | Path,
    ) -> SkillEffectReview:
        baseline_path = Path(baseline_trace).resolve()
        candidate_path = Path(candidate_trace).resolve()
        task = trace_task(candidate_path)
        if trace_task(baseline_path) != task:
            raise ValueError("baseline 与 candidate task prompt 不一致")
        baseline_actions = build_action_evidence(baseline_path)
        candidate_actions = build_action_evidence(candidate_path)
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        inspected: list[dict[str, Any]] = []
        feedback: str | None = None
        force_final = False

        for turn in range(1, self.max_inspections + 2):
            # 重复读取同一区间不会产生新证据。下一轮直接关闭工具通道，
            # 要求 Reviewer 基于已经看过的画面作结，避免空转到循环上限。
            remaining = 0 if force_final else self.max_inspections - len(inspected)
            messages = self._messages(
                task=task,
                mutation=mutation,
                previous_skill=previous_skill,
                candidate_skill=candidate_skill,
                baseline_outcome=baseline_outcome,
                candidate_outcome=candidate_outcome,
                baseline_consulted=baseline_consulted,
                candidate_consulted=candidate_consulted,
                baseline_timeline=_timeline(baseline_actions),
                candidate_timeline=_timeline(candidate_actions),
                inspected=inspected,
                feedback=feedback,
                remaining=remaining,
            )
            tools = [INSPECT_PAIR_TOOL] if remaining > 0 else None
            turn_dir = output / "turns" / f"turn_{turn:02d}"
            write_json(
                turn_dir / "request.json",
                {
                    "task": task,
                    "visible_evidence": [
                        {"evidence_id": item["evidence_id"], "sha256": item["sha256"]}
                        for item in inspected[-2:]
                    ],
                    "notebook": [
                        {"evidence_id": item["evidence_id"], "question": item["question"]}
                        for item in inspected
                    ],
                    "remaining": remaining,
                    "tool_available": bool(tools),
                },
            )
            response = self.provider.generate(messages, tools)
            write_json(turn_dir / "response.json", self._response_record(response))
            if response.tool_calls:
                if not tools or len(response.tool_calls) != 1:
                    raise ValueError("Skill Effect Reviewer 每轮只能调用一次 inspect_pair")
                call = response.tool_calls[0]
                if call.parse_error:
                    raise ValueError(call.parse_error)
                if call.name != "inspect_pair":
                    raise ValueError(f"未知 Skill Effect Reviewer 工具: {call.name}")
                question = str(call.args.get("question") or "").strip()
                if not question:
                    raise ValueError("inspect_pair.question 不能为空")
                key = (
                    int(call.args["baseline_start"]),
                    int(call.args["baseline_end"]),
                    int(call.args["candidate_start"]),
                    int(call.args["candidate_end"]),
                )
                if any(item["span"] == key for item in inspected):
                    feedback = "该 baseline/candidate 区间已经读取；请基于已有证据给出结论。"
                    force_final = True
                    continue
                evidence_id = f"t{task_id}_s{seed}_e{len(inspected) + 1:02d}"
                image = render_pair_evidence(
                    _selected(baseline_actions, key[0], key[1]),
                    _selected(candidate_actions, key[2], key[3]),
                    output / "evidence" / f"{evidence_id}.png",
                )
                item = {
                    "evidence_id": evidence_id,
                    "span": key,
                    "question": question,
                    "image": image,
                    "sha256": sha256(image),
                }
                inspected.append(item)
                write_json(
                    image.with_suffix(".json"),
                    {
                        "evidence_id": evidence_id,
                        "baseline_actions": [key[0], key[1]],
                        "candidate_actions": [key[2], key[3]],
                        "question": question,
                        "sha256": item["sha256"],
                    },
                )
                feedback = None
                continue

            review = self._parse(response.text, inspected, task_id=task_id, seed=seed)
            write_json(output / "review.json", review.to_dict())
            return review
        raise AssertionError("unreachable")

    @staticmethod
    def _messages(
        *,
        task: str,
        mutation: Mapping[str, Any],
        previous_skill: str | None,
        candidate_skill: str | None,
        baseline_outcome: str,
        candidate_outcome: str,
        baseline_consulted: bool,
        candidate_consulted: bool,
        baseline_timeline: str,
        candidate_timeline: str,
        inspected: Sequence[Mapping[str, Any]],
        feedback: str | None,
        remaining: int,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"TASK\n{task}\n\nMUTATION\n{dict(mutation)}\n\n"
                    f"PREVIOUS SKILL\n{previous_skill or '(none)'}\n\n"
                    f"CANDIDATE SKILL\n{candidate_skill or '(retired)'}\n\n"
                    f"BASELINE outcome={baseline_outcome} consulted={baseline_consulted}\n"
                    f"{baseline_timeline}\n\n"
                    f"CANDIDATE outcome={candidate_outcome} consulted={candidate_consulted}\n"
                    f"{candidate_timeline}"
                ),
            }
        ]
        if inspected:
            notebook = "\n".join(
                f"- {item['evidence_id']}: {item['question']}" for item in inspected
            )
            content.append({"type": "text", "text": f"\nINSPECTION NOTEBOOK\n{notebook}"})
            for item in inspected[-2:]:
                content.extend(
                    (
                        {
                            "type": "text",
                            "text": f"\n{item['evidence_id']} · {item['question']}",
                        },
                        image_part(Path(item["image"])),
                    )
                )
        if feedback:
            content.append({"type": "text", "text": f"\nTOOL FEEDBACK: {feedback}"})
        instruction = (
            f"\nRemaining inspections: {remaining}. 调用 inspect_pair 或输出最终 JSON。"
            if remaining > 0
            else "\nInspection budget exhausted. 现在必须输出最终 JSON。"
        )
        content.append({"type": "text", "text": instruction})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _parse(
        text: str,
        inspected: Sequence[Mapping[str, Any]],
        *,
        task_id: int,
        seed: int,
    ) -> SkillEffectReview:
        if not inspected:
            raise ValueError("Skill Effect Reviewer 必须先检查至少一组视觉证据")
        payload = parse_model_json(text)
        evidence_ids = tuple(str(item) for item in payload.get("evidence_ids") or ())
        available = {str(item["evidence_id"]) for item in inspected}
        if not evidence_ids or any(item not in available for item in evidence_ids):
            raise ValueError("Skill Effect Reviewer 必须引用已检查的 evidence_ids")
        return SkillEffectReview(
            task_id=task_id,
            seed=seed,
            effect=SkillEffect(str(payload["effect"])),
            remaining_failure=RemainingFailure(str(payload["remaining_failure"])),
            evidence_ids=evidence_ids,
            reason=str(payload.get("reason") or ""),
        )

    @staticmethod
    def _response_record(response: ModelResponse) -> dict[str, Any]:
        return {
            "text": response.text,
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": call.args,
                    "parse_error": call.parse_error,
                }
                for call in response.tool_calls
            ],
            "reasoning": response.provider_reasoning,
            "usage": response.usage,
        }


def run_skill_effect_reviews(
    *,
    store: GenerationStore,
    candidate_generation: str,
    baseline_day_index: str | Path,
    candidate_day_index: str | Path,
    output_dir: str | Path,
    provider: ModelProvider,
    max_inspections: int = 4,
    resume: bool = False,
) -> Path:
    """逐 episode 启动独立 Reviewer，并生成可直接交给 Gate 的报告。"""

    manifest = store.read_manifest(candidate_generation)
    mutation = manifest.mutation
    parent = manifest.parent_generation
    if mutation is None or parent is None:
        raise ValueError("Skill Effect Reviewer 只能评审 candidate generation")
    baseline_index = Path(baseline_day_index).resolve()
    candidate_index = Path(candidate_day_index).resolve()
    assert_paired_runs(
        baseline_index,
        candidate_index,
        baseline_generation=parent,
        candidate_generation=candidate_generation,
    )
    baseline = load_day_index(baseline_index)
    candidate = load_day_index(candidate_index)
    if set(baseline) != set(candidate):
        raise ValueError("baseline 与 candidate DAY index 的 task/seed 不完整")

    parent_skill = store.generation_path(parent) / "skills" / mutation.skill_id / "SKILL.md"
    candidate_skill = (
        store.generation_path(candidate_generation)
        / "skills"
        / mutation.skill_id
        / "SKILL.md"
    )
    previous_markdown = parent_skill.read_text(encoding="utf-8") if parent_skill.is_file() else None
    candidate_markdown = (
        candidate_skill.read_text(encoding="utf-8") if candidate_skill.is_file() else None
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=resume)
    reviews: list[SkillEffectReview] = []
    for task_id, seed in sorted(baseline):
        base = baseline[task_id, seed]
        cand = candidate[task_id, seed]
        if base.outcome.value == "infrastructure" or cand.outcome.value == "infrastructure":
            continue
        episode_output = output / f"task{task_id}_s{seed}"
        review_path = episode_output / "review.json"
        if resume and review_path.is_file():
            review = SkillEffectReview.from_dict(read_json(review_path))
        else:
            if episode_output.exists():
                if not resume:
                    raise FileExistsError(f"Skill Effect 输出已存在: {episode_output}")
                _archive_incomplete(episode_output)
            review = SkillEffectReviewer(provider, max_inspections=max_inspections).run(
                task_id=task_id,
                seed=seed,
                baseline_trace=base.trace_dir,
                candidate_trace=cand.trace_dir,
                baseline_outcome=base.outcome.value,
                candidate_outcome=cand.outcome.value,
                mutation=mutation.to_dict(),
                previous_skill=previous_markdown,
                candidate_skill=candidate_markdown,
                baseline_consulted=consulted_skill(base.trace_dir, mutation.skill_id),
                candidate_consulted=consulted_skill(cand.trace_dir, mutation.skill_id),
                output_dir=episode_output,
            )
        reviews.append(review)

    report = SkillEffectReport(
        mutation_id=mutation.mutation_id,
        baseline_generation=parent,
        candidate_generation=candidate_generation,
        baseline_day_index=str(baseline_index),
        candidate_day_index=str(candidate_index),
        reviews=tuple(reviews),
    )
    return write_json(output / "effect_report.json", report.to_dict())


def _archive_incomplete(path: Path) -> Path:
    """保留未完成的独立评审，再从干净目录恢复该 episode。"""

    number = 1
    while True:
        suffix = ".incomplete" if number == 1 else f".incomplete-{number}"
        archive = path.with_name(path.name + suffix)
        if not archive.exists():
            path.replace(archive)
            return archive
        number += 1


def _provider(args: argparse.Namespace) -> ModelProvider:
    from vaw.agents.providers.openai import OpenAIProvider
    from vaw.agents.providers.text_protocol import TextProtocolProvider

    provider: ModelProvider = OpenAIProvider(
        model=args.model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
    )
    return TextProtocolProvider(provider) if args.protocol == "text" else provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--baseline-index", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--api-key", default=os.environ.get("V_API_KEY"))
    parser.add_argument("--protocol", choices=("native", "text"), default="native")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--max-inspections", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_skill_effect_reviews(
        store=GenerationStore.open(args.experiment_root),
        candidate_generation=args.generation,
        baseline_day_index=args.baseline_index,
        candidate_day_index=args.candidate_index,
        output_dir=args.output_dir,
        provider=_provider(args),
        max_inspections=args.max_inspections,
        resume=args.resume,
    )
    print(f"[skill-effect] {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INSPECT_PAIR_TOOL",
    "SkillEffectReviewer",
    "render_pair_evidence",
    "run_skill_effect_reviews",
]
