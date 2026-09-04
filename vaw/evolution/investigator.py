"""使用一个只读视觉工具主动调查完整 VAW rollout。

Investigator 不接收预切好的失败窗口。它先阅读 Episode Atlas，再按自己的问题
调用 ``inspect_segment`` 取回高分辨率策略可见证据，最后提出可复核的通用发现。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vaw.agents.contracts import ModelResponse
from vaw.agents.providers.base import ModelProvider
from vaw.evolution.artifacts import (
    image_part,
    parse_model_json,
    read_json,
    relative_asset,
    sha256,
    write_json,
)
from vaw.evolution.segments import SegmentEvidence, TraceAccessor

INVESTIGATION_SCHEMA = "vaw-trace-investigation-v1"
VISIBLE_EVIDENCE_LIMIT = 3

INSPECT_SEGMENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_segment",
        "description": (
            "读取一段物理动作边界对应的高分辨率、策略可见视觉证据。"
            "用于检验具体假设；优先读取 1–3 个动作，不要重复读取已检查的相同区间。"
            "不会返回深度、环境真值或 planner telemetry。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_action": {
                    "type": "string",
                    "description": "起始物理动作标签，例如 A3。",
                },
                "end_action": {
                    "type": "string",
                    "description": "结束物理动作标签，包含该动作，例如 A5。",
                },
                "question": {
                    "type": "string",
                    "description": "希望用这段视觉证据回答的一个具体问题。",
                },
            },
            "required": ["start_action", "end_action", "question"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """你是 VAW rollout 的独立视觉调查 Agent，不负责控制机器人。

先阅读 Episode Atlas。TOPReward 曲线只是导航线索，终局 success/failure 是 episode 结果，二者都不能替代视觉证据。需要细节时主动调用 inspect_segment。先读取 1–3 个动作的局部区间，只在检验跨动作因果时扩大范围；同一区间的图像不会变化，不要重复读取。

你的目标是找出少量、可由当前证据支持且能迁移到其他任务的决策经验。区分真实 OBSERVED 与虚拟 PREVIEW；不要从图片猜测 solver、碰撞器、环境内部原因或 Agent 动机，也不要把单个任务坐标写成经验。observation 只写可见状态变化；insight 写可执行的一般原则，不评价系统能力。

证据充分后只输出 JSON：
{"findings":[{"span":[start_action,end_action],"observation":"视觉上发生了什么","insight":"可迁移的决策经验"}]}

每个 span 必须与一次已经检查过的区间完全一致。没有可靠发现时输出 {"findings":[]}。"""


def _action_number(value: Any) -> int:
    """把 Atlas 中的 A7 标签转为内部索引。"""

    text = str(value).strip().upper()
    if text.startswith("A"):
        text = text[1:]
    return int(text)


def _covers_episode(inspections: list[SegmentEvidence], action_count: int) -> bool:
    """判断已读取证据是否覆盖全部物理动作。"""

    covered = {
        action
        for item in inspections
        for action in range(item.start_action, item.end_action + 1)
    }
    return covered == set(range(1, action_count + 1))


class TraceInvestigator:
    """重建短上下文的离线调查循环，避免累积整段图像 transcript。"""

    def __init__(self, provider: ModelProvider, *, max_inspections: int = 4) -> None:
        if max_inspections < 1:
            raise ValueError("max_inspections 必须大于零")
        self.provider = provider
        self.max_inspections = max_inspections

    def run(
        self,
        atlas_path: str | Path,
        atlas_image: str | Path,
        output_dir: str | Path,
    ) -> Path:
        atlas = read_json(atlas_path)
        atlas_png = Path(atlas_image).resolve()
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        accessor = TraceAccessor(atlas)
        inspections: list[SegmentEvidence] = []
        attempts = 0
        tool_feedback: str | None = None

        turn = 0
        while True:
            turn += 1
            remaining = self.max_inspections - attempts
            can_inspect = remaining > 0 and not _covers_episode(
                inspections,
                len(atlas["actions"]),
            )
            messages = self._messages(
                atlas,
                atlas_png,
                inspections,
                remaining,
                tool_feedback,
                can_inspect,
            )
            turn_dir = output / "turns" / f"turn_{turn:02d}"
            write_json(
                turn_dir / "request.json",
                self._request_record(
                    atlas_png,
                    inspections,
                    remaining,
                    tool_feedback,
                    can_inspect,
                ),
            )
            response = self.provider.generate(
                messages,
                tools=[INSPECT_SEGMENT_TOOL] if can_inspect else None,
            )
            write_json(turn_dir / "response.json", self._response_record(response))
            if response.tool_calls:
                if not can_inspect:
                    raise ValueError("Investigator 当前不可继续调用 inspect_segment")
                if len(response.tool_calls) != 1:
                    raise ValueError("Investigator 每轮只能调用一次 inspect_segment")
                call = response.tool_calls[0]
                if call.parse_error:
                    raise ValueError(call.parse_error)
                if call.name != "inspect_segment":
                    raise ValueError(f"未知 Investigator 工具: {call.name}")
                attempts += 1
                start_action = _action_number(call.args["start_action"])
                end_action = _action_number(call.args["end_action"])
                existing = next(
                    (item for item in inspections if item.key == (start_action, end_action)),
                    None,
                )
                if existing is not None:
                    tool_feedback = (
                        f"inspect_segment A{start_action}–A{end_action} 未重复读取："
                        "该不变证据已在 notebook 中。请改用其他区间，"
                        "或根据现有证据输出最终 JSON。"
                    )
                    continue
                evidence = accessor.inspect(
                    start_action,
                    end_action,
                    str(call.args["question"]),
                    output / "segments" / f"segment_{len(inspections) + 1:02d}.png",
                )
                inspections.append(evidence)
                tool_feedback = None
                write_json(
                    evidence.image.with_suffix(".json"),
                    {
                        "span": list(evidence.key),
                        "question": evidence.question,
                        "actions": list(evidence.actions),
                        "image_sha256": sha256(evidence.image),
                    },
                )
                continue

            findings = self._findings(response.text, inspections)
            document = {
                "schema": INVESTIGATION_SCHEMA,
                "episode": atlas["episode"],
                "atlas": str(Path(atlas_path).resolve()),
                "inspections": [
                    {
                        "span": list(item.key),
                        "question": item.question,
                        "actions": list(item.actions),
                        "evidence": relative_asset(output, item.image),
                    }
                    for item in inspections
                ],
                "findings": findings,
            }
            return write_json(output / "findings.json", document)

    def _messages(
        self,
        atlas: Mapping[str, Any],
        atlas_image: Path,
        inspections: list[SegmentEvidence],
        remaining: int,
        tool_feedback: str | None,
        can_inspect: bool,
    ) -> list[dict[str, Any]]:
        episode = atlas["episode"]
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Task: {episode['task']}\n"
                    f"Terminal outcome: {'success' if episode['env_success'] else 'failure'}\n"
                    f"Physical actions: A1..A{len(atlas['actions'])}\n\n"
                    "EPISODE ATLAS:"
                ),
            },
            image_part(atlas_image),
        ]
        if inspections:
            notebook = "\n".join(
                f"- A{item.start_action}–A{item.end_action}: {item.question}"
                for item in inspections
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"\nINVESTIGATION NOTEBOOK（已检查）：\n{notebook}\n"
                        "同一 span 的视觉证据是不变的，再次读取不会产生新信息。"
                    ),
                }
            )
            for item in inspections[-VISIBLE_EVIDENCE_LIMIT:]:
                actions = "\n".join(f"- {line}" for line in item.actions) or "- no semantic call"
                content.extend(
                    (
                        {
                            "type": "text",
                            "text": (
                                f"\nEVIDENCE A{item.start_action}–A{item.end_action}\n"
                                f"Question: {item.question}\nActions:\n{actions}"
                            ),
                        },
                        image_part(item.image),
                    )
                )
        if tool_feedback:
            content.append({"type": "text", "text": f"\nTOOL FEEDBACK: {tool_feedback}"})
        if _covers_episode(inspections, len(atlas["actions"])):
            instruction = (
                "\n已读取区间覆盖全部物理动作。"
                "现在必须根据现有证据输出最终 JSON。"
            )
        elif can_inspect:
            instruction = (
            f"\nRemaining inspections: {remaining}. "
            "继续调查请调用 inspect_segment；证据充分则输出最终 JSON。"
            )
        else:
            instruction = "\nInspection budget exhausted. 现在必须输出最终 JSON。"
        content.append({"type": "text", "text": instruction})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _request_record(
        atlas_image: Path,
        inspections: list[SegmentEvidence],
        remaining: int,
        tool_feedback: str | None,
        can_inspect: bool,
    ) -> dict[str, Any]:
        """保存模型实际看到的证据索引，不把 base64 重复写入 trace。"""

        return {
            "atlas": {"path": str(atlas_image), "sha256": sha256(atlas_image)},
            "visible_segments": [
                {
                    "span": list(item.key),
                    "path": str(item.image),
                    "sha256": sha256(item.image),
                }
                for item in inspections[-VISIBLE_EVIDENCE_LIMIT:]
            ],
            "notebook": [
                {"span": list(item.key), "question": item.question} for item in inspections
            ],
            "inspect_segment_available": can_inspect,
            "tool_feedback": tool_feedback,
        }

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

    @staticmethod
    def _findings(text: str, inspections: list[SegmentEvidence]) -> list[dict[str, Any]]:
        payload = parse_model_json(text)
        raw = payload.get("findings")
        if not isinstance(raw, list):
            raise ValueError("Investigator 输出缺少 findings list")
        available = {item.key: item for item in inspections}
        findings: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("finding 必须是 object")
            span = item.get("span")
            if not isinstance(span, list) or len(span) != 2:
                raise ValueError("finding.span 必须为 [start_action, end_action]")
            key = _action_number(span[0]), _action_number(span[1])
            evidence = available.get(key)
            if evidence is None:
                raise ValueError(f"finding 必须引用已检查的完整区间: {key}")
            observation = str(item.get("observation") or "").strip()
            insight = str(item.get("insight") or "").strip()
            if not observation or not insight:
                raise ValueError("finding 必须包含 observation 和 insight")
            findings.append(
                {
                    "span": list(key),
                    "observation": observation,
                    "insight": insight,
                    "evidence": relative_asset(evidence.image.parent.parent, evidence.image),
                }
            )
        return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--atlas-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--api-key")
    parser.add_argument("--protocol", choices=("native", "text"), default="native")
    parser.add_argument("--max-inspections", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    from vaw.agents.providers.openai import OpenAIProvider
    from vaw.agents.providers.text_protocol import TextProtocolProvider

    args = parse_args()
    provider: ModelProvider = OpenAIProvider(
        model=args.model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=4096,
    )
    if args.protocol == "text":
        provider = TextProtocolProvider(provider)
    result = TraceInvestigator(provider, max_inspections=args.max_inspections).run(
        args.atlas,
        args.atlas_image,
        args.output_dir,
    )
    print(f"[m3-investigator] {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INSPECT_SEGMENT_TOOL",
    "INVESTIGATION_SCHEMA",
    "TraceInvestigator",
]
