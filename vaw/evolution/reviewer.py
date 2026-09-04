"""独立复核 Investigator 的视觉发现，不在本阶段生成或改写技能。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vaw.agents.providers.base import ModelProvider
from vaw.evolution.artifacts import image_part, parse_model_json, read_json, sha256, write_json
from vaw.evolution.investigator import INVESTIGATION_SCHEMA

REVIEW_SCHEMA = "vaw-evidence-review-v1"

SYSTEM_PROMPT = """你是独立的 VAW 视觉证据 Reviewer。你只检查一条 Investigator finding，不负责控制机器人，也不负责写技能。

接受条件：observation 能从给出的 policy-visible 证据直接看出；insight 是 observation 支持的、可执行的通用决策经验。两者必须同时成立。若结论依赖 solver、环境内部状态、未展示的物体动力学、Agent 动机、任务固定坐标，或把虚拟 PREVIEW 当作真实结果，则拒绝。只是重述结果或评价系统能力的 insight 也应拒绝。

只输出 JSON：{"accepted":true或false,"reason":"一句简短证据说明"}。reason 必须同时说明 observation 的视觉依据与 insight 的支持关系。不要改写 finding。"""


class EvidenceReviewer:
    """每条 finding 使用独立上下文，避免前一条结论影响后一条。"""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def run(self, findings_path: str | Path, output_path: str | Path) -> Path:
        source = Path(findings_path).resolve()
        investigation = read_json(source)
        if investigation.get("schema") != INVESTIGATION_SCHEMA:
            raise ValueError(
                f"不支持的 investigation schema: {investigation.get('schema')!r}"
            )
        reviewed: list[dict[str, Any]] = []
        for index, finding in enumerate(investigation.get("findings") or [], start=1):
            if not isinstance(finding, Mapping):
                raise ValueError("finding 必须是 object")
            evidence = (source.parent / str(finding["evidence"])).resolve()
            try:
                evidence.relative_to(source.parent)
            except ValueError as exc:
                raise ValueError(f"finding evidence 越出 investigation: {evidence}") from exc
            if not evidence.is_file():
                raise FileNotFoundError(evidence)
            messages = self._messages(investigation["episode"], finding, evidence)
            response = self.provider.generate(messages, tools=None)
            turn_dir = Path(output_path).resolve().parent / "review_turns"
            write_json(
                turn_dir / f"finding_{index:02d}.json",
                {
                    "finding": dict(finding),
                    "evidence": {"path": str(evidence), "sha256": sha256(evidence)},
                    "response": {
                        "text": response.text,
                        "reasoning": response.provider_reasoning,
                        "usage": response.usage,
                    },
                },
            )
            review = self._review(response.text)
            reviewed.append({**dict(finding), "review": review})
        return write_json(
            output_path,
            {
                "schema": REVIEW_SCHEMA,
                "episode": investigation["episode"],
                "findings": reviewed,
            },
        )

    @staticmethod
    def _messages(
        episode: Mapping[str, Any],
        finding: Mapping[str, Any],
        evidence: Path,
    ) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Task: {episode['task']}\n"
                            f"Terminal outcome: {'success' if episode['env_success'] else 'failure'}\n"
                            f"Span: {finding['span']}\n"
                            f"Observation: {finding['observation']}\n"
                            f"Insight: {finding['insight']}\n\n"
                            "EVIDENCE:"
                        ),
                    },
                    image_part(evidence),
                    {"type": "text", "text": "请独立审核。"},
                ],
            },
        ]

    @staticmethod
    def _review(text: str) -> dict[str, Any]:
        payload = parse_model_json(text)
        accepted = payload.get("accepted")
        reason = str(payload.get("reason") or "").strip()
        if not isinstance(accepted, bool) or not reason:
            raise ValueError("Reviewer 输出必须包含 accepted boolean 和非空 reason")
        return {"accepted": accepted, "reason": reason}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--api-key")
    parser.add_argument("--protocol", choices=("native", "text"), default="native")
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
        max_tokens=2048,
    )
    if args.protocol == "text":
        provider = TextProtocolProvider(provider)
    result = EvidenceReviewer(provider).run(args.findings, args.output)
    print(f"[m3-reviewer] {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EvidenceReviewer", "REVIEW_SCHEMA"]
