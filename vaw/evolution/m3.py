"""运行 EpisodeAtlas → TraceInvestigator → EvidenceReviewer 的完整 M3。"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vaw.agents.providers.base import ModelProvider
from vaw.evolution.artifacts import read_json, write_json
from vaw.evolution.atlas import compile_to_dir
from vaw.evolution.investigator import TraceInvestigator
from vaw.evolution.reviewer import EvidenceReviewer


def _provider(args: argparse.Namespace, model: str) -> ModelProvider:
    from vaw.agents.providers.openai import OpenAIProvider
    from vaw.agents.providers.text_protocol import TextProtocolProvider

    provider: ModelProvider = OpenAIProvider(
        model=model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        extra_body=(
            {"chat_template_kwargs": {"enable_thinking": False}}
            if args.disable_thinking
            else None
        ),
    )
    return TextProtocolProvider(provider) if args.protocol == "text" else provider


def run_m3(
    *,
    trace_dir: str | Path,
    progress_path: str | Path,
    output_dir: str | Path,
    investigator_provider: ModelProvider,
    reviewer_provider: ModelProvider,
    max_inspections: int = 4,
    run_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """运行一次完整调查，并在所有 Agent 成功后原子发布结果。"""

    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"M3 输出目录已存在: {output}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    if run_metadata is not None:
        write_json(staging / "run.json", dict(run_metadata))
    atlas_json, atlas_png = compile_to_dir(
        trace_dir,
        progress_path,
        staging / "atlas",
    )
    findings = TraceInvestigator(
        investigator_provider,
        max_inspections=max_inspections,
    ).run(atlas_json, atlas_png, staging / "investigation")
    review = EvidenceReviewer(reviewer_provider).run(findings, staging / "review.json")
    review_payload = read_json(review)
    accepted = sum(
        bool(item.get("review", {}).get("accepted"))
        for item in review_payload["findings"]
    )
    write_json(
        staging / "m3.json",
        {
            "schema": "vaw-m3-run-result-v1",
            "run": "run.json" if run_metadata is not None else None,
            "atlas": "atlas/atlas.json",
            "investigation": "investigation/findings.json",
            "review": "review.json",
            "findings": len(review_payload["findings"]),
            "accepted": accepted,
        },
    )
    staging.replace(output)
    return output / "m3.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="TraceInvestigator model")
    parser.add_argument("--review-model", help="默认与 --model 相同，但使用独立请求上下文")
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--api-key", default=os.environ.get("V_API_KEY"))
    parser.add_argument("--protocol", choices=("native", "text"), default="native")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--max-inspections", type=int, default=4)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="禁用支持该参数的本地模型 thinking，确保结构化 JSON 位于正文。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_m3(
        trace_dir=args.trace,
        progress_path=args.progress,
        output_dir=args.output_dir,
        investigator_provider=_provider(args, args.model),
        reviewer_provider=_provider(args, args.review_model or args.model),
        max_inspections=args.max_inspections,
        run_metadata={
            "schema": "vaw-m3-run-v1",
            "investigator_model": args.model,
            "reviewer_model": args.review_model or args.model,
            "server_url": args.server_url,
            "protocol": args.protocol,
            "max_tokens": args.max_tokens,
            "timeout_s": args.timeout_s,
            "max_inspections": args.max_inspections,
            "disable_thinking": args.disable_thinking,
        },
    )
    payload = read_json(result)
    print(
        f"[m3] findings={payload['findings']} accepted={payload['accepted']} "
        f"output={result.parent}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_m3"]
