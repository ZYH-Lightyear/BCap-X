"""由 M3 证据驱动的统一 Skill Evolver 与独立候选审核。

Evolver 通过三个只读工具渐进查看证据、现有技能和历史视觉参考。它最终只做
``NO_CHANGE`` 或一项 ``ADD/REVISE/RETIRE``。Reviewer 使用全新上下文审核
最终变更，避免 Evolver 自己为自己的结论背书。
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from vaw.agents.contracts import Message, ModelResponse
from vaw.agents.providers.base import ModelProvider
from vaw.evolution.artifacts import parse_model_json, read_json, sha256, write_json
from vaw.evolution.candidate import CandidateEvidence, CandidatePackage
from vaw.evolution.domain import MutationOperation, SkillMutation
from vaw.evolution.reviewer import REVIEW_SCHEMA
from vaw.evolution.store import GenerationStore
from vaw.mmskill import MMSkill, MMSkillLibrary, MMSkillReference

M4_SCHEMA = "vaw-skill-evolver-v1"
REFERENCE_SCHEMA = "vaw-mmskill-references-v1"

INSPECT_EVIDENCE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_evidence",
        "description": "查看一条经 M3 Reviewer 接受的视觉证据与结论。",
        "parameters": {
            "type": "object",
            "properties": {"evidence_id": {"type": "string"}},
            "required": ["evidence_id"],
            "additionalProperties": False,
        },
    },
}

INSPECT_SKILL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_skill",
        "description": "查看一个现有技能的完整 SKILL.md 和视觉参考索引。",
        "parameters": {
            "type": "object",
            "properties": {"skill_id": {"type": "string"}},
            "required": ["skill_id"],
            "additionalProperties": False,
        },
    },
}

INSPECT_REFERENCE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_reference",
        "description": "查看现有技能的一张历史视觉参考；它不是当前 observation。",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_id": {"type": "string"},
                "reference_id": {"type": "string"},
            },
            "required": ["skill_id", "reference_id"],
            "additionalProperties": False,
        },
    },
}

EVOLVER_TOOLS = (
    INSPECT_EVIDENCE_TOOL,
    INSPECT_SKILL_TOOL,
    INSPECT_REFERENCE_TOOL,
)

EVOLVER_SYSTEM_PROMPT = """你是 VAW Skill Evolver。你根据已审核的真实视觉证据维护一个简短、通用、可执行的 MMSkill Library。

先从轻量索引判断哪些证据和技能相关，再按需调用只读工具。不要一次加载所有内容。一次只允许 ADD、REVISE、RETIRE 一项技能，也可以 NO_CHANGE。技能不能包含 task id、固定环境坐标、solver telemetry、环境真值或未经视觉证据支持的物理关系。SKILL.md 应像清晰的经验提示，不写固定任务流程。

视觉参考是技能的一部分：只保留仅靠文字难以表达且能帮助视觉判断的图；可为零张或多张。新图只能来自已引用 evidence，已有图用 current:<reference_id> 引用。每张新图必须说明 state、view、when_to_use 和 visual_cue。

已检查资源会出现在 INSPECTION NOTEBOOK；重复读取不会增加证据，请改读其他相关资源或直接形成结论。

最终只输出 JSON。无充分证据时：
{"decision":"no_change","rationale":"..."}

需要变更时：
{"decision":"mutate","operation":"add|revise|retire","skill_id":"...","evidence_ids":["e001"],"rationale":"...","skill_markdown":"完整 SKILL.md","references":[{"source_id":"e001","state":"failure","view":"contact","when_to_use":"...","visual_cue":"..."}]}

ADD/REVISE 的 skill_markdown 必须是一个合法 JSON 字符串；其中所有换行都必须写成
JSON 转义 ``\\n``，不能在双引号内部直接换行。完整文件严格采用：
---\\nname: 简短名称\\ndescription: 当……时使用。\\n---\\n\\n# 简短名称\\n\\n简洁、可执行的正文

RETIRE 不带 skill_markdown 和 references。不要输出多个 mutation。"""

REVIEWER_SYSTEM_PROMPT = """你是独立的 VAW Skill Mutation Reviewer。你只审核候选，不改写技能。

只有以下条件同时成立才接受：引用证据真实支持变更；内容能迁移到其他任务；触发描述明确；不与现有技能重复或冲突；不含 task id、固定坐标、solver/private telemetry 或 privileged truth；不把不确定视觉关系写成事实；正文简短且能直接指导视觉决策；所选视觉参考确有不可被文字替代的价值。

只输出 JSON：{"decision":"accept|reject","reason":"..."}。"""


def _image_bytes_part(data: bytes, filename: str) -> dict[str, Any]:
    mime = mimetypes.guess_type(filename)[0] or "image/png"
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "high"},
    }


@dataclass(frozen=True)
class AcceptedEvidence:
    """M4 可读取的一条 M3 accepted finding。"""

    evidence_id: str
    m3_output: Path
    finding_id: str
    task: str
    outcome: str
    observation: str
    insight: str
    raster: Path

    def summary(self) -> dict[str, str]:
        return {
            "evidence_id": self.evidence_id,
            "task": self.task,
            "outcome": self.outcome,
            "observation": self.observation,
            "insight": self.insight,
        }

    def candidate_evidence(self) -> CandidateEvidence:
        return CandidateEvidence(
            m3_output=str(self.m3_output),
            finding_id=self.finding_id,
            raster=self.raster.relative_to(self.m3_output).as_posix(),
            sha256=sha256(self.raster),
        )


class EvidencePool:
    """把多个 M3 输出收敛为稳定、局部编号的 accepted evidence。"""

    def __init__(self, items: Sequence[AcceptedEvidence]) -> None:
        self.items = tuple(items)
        self._by_id = {item.evidence_id: item for item in self.items}
        if len(self._by_id) != len(self.items):
            raise ValueError("evidence_id 重复")

    @classmethod
    def from_m3_outputs(cls, roots: Sequence[str | Path]) -> EvidencePool:
        items: list[AcceptedEvidence] = []
        for root_value in roots:
            source = Path(root_value).resolve()
            if source.is_file():
                result = read_json(source)
                if result.get("schema") != "vaw-m3-run-result-v1":
                    raise ValueError(f"不支持的 M3 result schema: {source}")
                root = source.parent
                review_path = (root / str(result["review"])).resolve()
                investigation_root = (
                    root / str(result["investigation"])
                ).resolve().parent
                for path in (review_path, investigation_root):
                    try:
                        path.relative_to(root)
                    except ValueError as exc:
                        raise ValueError(f"M3 result 引用越出输出目录: {path}") from exc
            else:
                # 保留目录输入，便于独立使用 M4；正式 cycle 传入的是 m3.json。
                root = source
                review_path = root / "review.json"
                investigation_root = root / "investigation"
            review = read_json(review_path)
            if review.get("schema") != REVIEW_SCHEMA:
                raise ValueError(f"不支持的 M3 review schema: {root}")
            episode = review.get("episode")
            if not isinstance(episode, Mapping):
                raise ValueError(f"M3 review 缺少 episode: {root}")
            for index, finding in enumerate(review.get("findings") or [], start=1):
                if not isinstance(finding, Mapping):
                    raise ValueError(f"M3 finding 必须是 object: {root}")
                finding_review = finding.get("review")
                if not isinstance(finding_review, Mapping) or finding_review.get("accepted") is not True:
                    continue
                relative = Path(str(finding["evidence"]))
                raster = (investigation_root / relative).resolve()
                try:
                    raster.relative_to(investigation_root)
                except ValueError as exc:
                    raise ValueError(f"M3 evidence 越出 investigation: {raster}") from exc
                if not raster.is_file():
                    raise FileNotFoundError(raster)
                evidence_id = f"e{len(items) + 1:03d}"
                finding_id = (
                    f"{episode.get('suite')}:t{episode.get('task_id')}:"
                    f"s{episode.get('seed')}:f{index}"
                )
                items.append(
                    AcceptedEvidence(
                        evidence_id=evidence_id,
                        m3_output=root,
                        finding_id=finding_id,
                        task=str(episode.get("task") or ""),
                        outcome="success" if episode.get("env_success") else "failure",
                        observation=str(finding.get("observation") or "").strip(),
                        insight=str(finding.get("insight") or "").strip(),
                        raster=raster,
                    )
                )
        return cls(items)

    def get(self, evidence_id: str) -> AcceptedEvidence:
        try:
            return self._by_id[evidence_id]
        except KeyError as exc:
            raise ValueError(f"unknown evidence: {evidence_id}") from exc

    def summaries(self) -> list[dict[str, str]]:
        return [item.summary() for item in self.items]


@dataclass(frozen=True)
class ReferenceChoice:
    """Evolver 为新技能快照选择的一张最终视觉参考。"""

    source_id: str
    state: str | None = None
    view: str | None = None
    when_to_use: str | None = None
    visual_cue: str | None = None

    @property
    def is_current(self) -> bool:
        return self.source_id.startswith("current:")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReferenceChoice:
        return cls(
            source_id=str(payload.get("source_id") or "").strip(),
            state=_optional_text(payload.get("state")),
            view=_optional_text(payload.get("view")),
            when_to_use=_optional_text(payload.get("when_to_use")),
            visual_cue=_optional_text(payload.get("visual_cue")),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"source_id": self.source_id}
        for key, value in (
            ("state", self.state),
            ("view", self.view),
            ("when_to_use", self.when_to_use),
            ("visual_cue", self.visual_cue),
        ):
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True)
class EvolverDecision:
    """Skill Evolver 的严格最终输出。"""

    decision: Literal["no_change", "mutate"]
    rationale: str
    operation: MutationOperation | None = None
    skill_id: str | None = None
    evidence_ids: tuple[str, ...] = ()
    skill_markdown: str | None = None
    references: tuple[ReferenceChoice, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "decision": self.decision,
            "rationale": self.rationale,
        }
        if self.operation is not None:
            result["operation"] = self.operation.value
        if self.skill_id is not None:
            result["skill_id"] = self.skill_id
        if self.evidence_ids:
            result["evidence_ids"] = list(self.evidence_ids)
        if self.skill_markdown is not None:
            result["skill_markdown"] = self.skill_markdown
        if self.references:
            result["references"] = [item.to_dict() for item in self.references]
        return result


@dataclass(frozen=True)
class InspectedResource:
    label: str
    text: str
    image_bytes: bytes | None = None
    image_name: str = "evidence.png"


class SkillEvolver:
    """短上下文、按需读取资源的技能进化 Agent。"""

    def __init__(self, provider: ModelProvider, *, max_inspections: int = 6) -> None:
        if max_inspections < 1:
            raise ValueError("max_inspections 必须大于零")
        self.provider = provider
        self.max_inspections = max_inspections

    def run(
        self,
        evidence: EvidencePool,
        library: MMSkillLibrary,
        output_dir: str | Path,
    ) -> EvolverDecision:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        inspected: list[str] = []
        notebook: dict[str, str] = {}
        latest: InspectedResource | None = None
        final_response: ModelResponse | None = None
        inspection_calls = 0

        for turn in range(1, self.max_inspections + 2):
            remaining = max(0, self.max_inspections - inspection_calls)
            messages = self._messages(evidence, library, notebook, latest, remaining)
            tools = list(EVOLVER_TOOLS) if remaining else None
            turn_dir = output / "turns" / f"turn_{turn:02d}"
            write_json(
                turn_dir / "request.json",
                _request_record(messages, tools, latest),
            )
            response = self.provider.generate(messages, tools)
            write_json(turn_dir / "response.json", _response_record(response))
            if response.tool_calls:
                if tools is None or len(response.tool_calls) != 1:
                    raise ValueError("Skill Evolver 每轮只能调用一个可用的只读工具")
                call = response.tool_calls[0]
                if call.parse_error:
                    raise ValueError(call.parse_error)
                latest = self._inspect(call.name, call.args, evidence, library)
                inspection_calls += 1
                key = latest.label
                if key not in inspected:
                    inspected.append(key)
                    notebook[key] = latest.text
                continue
            final_response = response
            break
        if final_response is None:
            raise RuntimeError("Skill Evolver 未产生最终响应")
        decision = self._decision(final_response.text, evidence, library)
        write_json(
            output / "request.json",
            {
                "schema": "vaw-skill-evolver-request-v1",
                "evidence": evidence.summaries(),
                "skill_index": library.summary(),
                "notebook": notebook,
                "turns": [
                    str(path.relative_to(output))
                    for path in sorted((output / "turns").glob("turn_*/request.json"))
                ],
            },
        )
        write_json(output / "response.json", _response_record(final_response))
        write_json(output / "decision.json", decision.to_dict())
        return decision

    @staticmethod
    def _messages(
        evidence: EvidencePool,
        library: MMSkillLibrary,
        notebook: Mapping[str, str],
        latest: InspectedResource | None,
        remaining: int,
    ) -> list[Message]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "CURRENT SKILL INDEX:\n"
                    + library.format_index(inspect_function="inspect_skill")
                    + "\n\nACCEPTED M3 EVIDENCE:\n"
                    + json.dumps(evidence.summaries(), ensure_ascii=False, separators=(",", ":"))
                    + "\n\nINSPECTION NOTEBOOK:\n"
                    + (
                        json.dumps(notebook, ensure_ascii=False, separators=(",", ":"))
                        if notebook
                        else "none"
                    )
                    + f"\nRemaining inspections: {remaining}."
                ),
            }
        ]
        if latest is not None:
            content.append({"type": "text", "text": f"\nLATEST VISUAL {latest.label}:"})
            if latest.image_bytes is not None:
                content.append(_image_bytes_part(latest.image_bytes, latest.image_name))
        content.append(
            {
                "type": "text",
                "text": (
                    "继续调查请调用一个只读工具；证据充分则输出最终 mutation JSON。"
                    if remaining
                    else "检查预算已用完，现在必须输出最终 JSON。"
                ),
            }
        )
        return [
            {"role": "system", "content": EVOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _inspect(
        name: str,
        arguments: Mapping[str, Any],
        evidence: EvidencePool,
        library: MMSkillLibrary,
    ) -> InspectedResource:
        if name == "inspect_evidence":
            item = evidence.get(str(arguments["evidence_id"]))
            return InspectedResource(
                label=f"evidence:{item.evidence_id}",
                text=json.dumps(item.summary(), ensure_ascii=False, separators=(",", ":")),
                image_bytes=item.raster.read_bytes(),
                image_name=item.raster.name,
            )
        if name == "inspect_skill":
            skill = library.get(str(arguments["skill_id"]))
            return InspectedResource(
                label=f"skill:{skill.skill_id}",
                text=(
                    skill.prompt_block()
                    + "\n\nREFERENCES: "
                    + json.dumps(
                        [item.summary() for item in skill.references],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            )
        if name == "inspect_reference":
            skill_id = str(arguments["skill_id"])
            item = library.get_reference(skill_id, str(arguments["reference_id"]))
            return InspectedResource(
                label=f"reference:{skill_id}/{item.reference_id}",
                text=json.dumps(item.summary(), ensure_ascii=False, separators=(",", ":")),
                image_bytes=item.image_bytes,
                image_name=item.filename,
            )
        raise ValueError(f"未知 Skill Evolver 工具: {name}")

    @staticmethod
    def _decision(
        text: str,
        evidence: EvidencePool,
        library: MMSkillLibrary,
    ) -> EvolverDecision:
        payload = parse_model_json(text)
        decision = str(payload.get("decision") or "").strip().lower()
        rationale = str(payload.get("rationale") or "").strip()
        if not rationale:
            raise ValueError("Skill Evolver 输出缺少 rationale")
        if decision == "no_change":
            return EvolverDecision("no_change", rationale)
        if decision != "mutate":
            raise ValueError("Skill Evolver decision 必须是 no_change 或 mutate")
        operation = MutationOperation(str(payload.get("operation")))
        skill_id = str(payload.get("skill_id") or "").strip()
        evidence_ids = tuple(str(item) for item in payload.get("evidence_ids") or ())
        if not skill_id or not evidence_ids or len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("mutation 必须包含 skill_id 和不重复的 evidence_ids")
        for evidence_id in evidence_ids:
            evidence.get(evidence_id)
        exists = any(item["skill_id"] == skill_id for item in library.index())
        if operation is MutationOperation.ADD and exists:
            raise ValueError(f"ADD 目标技能已存在: {skill_id}")
        if operation is not MutationOperation.ADD and not exists:
            raise ValueError(f"{operation.value.upper()} 目标技能不存在: {skill_id}")
        markdown = _optional_text(payload.get("skill_markdown"))
        references = tuple(
            ReferenceChoice.from_dict(item)
            for item in payload.get("references") or ()
            if isinstance(item, Mapping)
        )
        if operation is MutationOperation.RETIRE:
            if markdown is not None or references:
                raise ValueError("RETIRE 不能包含 skill_markdown 或 references")
        else:
            if markdown is None:
                raise ValueError("ADD/REVISE 必须包含 skill_markdown")
            MMSkill.from_markdown(skill_id, markdown)
            _validate_reference_choices(references, evidence_ids, evidence, library, skill_id)
        return EvolverDecision(
            "mutate",
            rationale,
            operation=operation,
            skill_id=skill_id,
            evidence_ids=evidence_ids,
            skill_markdown=markdown,
            references=references,
        )


class MutationReviewer:
    """以独立模型上下文对最终 mutation 做一次接受或拒绝。"""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def run(
        self,
        decision: EvolverDecision,
        evidence: EvidencePool,
        library: MMSkillLibrary,
        output_dir: str | Path,
    ) -> bool:
        if decision.decision != "mutate" or decision.skill_id is None:
            raise ValueError("Reviewer 只能审核 mutation")
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        current = None
        if decision.operation is not MutationOperation.ADD:
            current = library.get(decision.skill_id).prompt_block()
        selected = [evidence.get(item) for item in decision.evidence_ids]
        current_references = [
            library.get_reference(
                decision.skill_id,
                choice.source_id.split(":", 1)[1],
            )
            for choice in decision.references
            if choice.is_current
        ]
        text = {
            "mutation": decision.to_dict(),
            "skill_index": library.summary(),
            "current_skill": current,
            "selected_evidence": [item.summary() for item in selected],
            "retained_references": [item.summary() for item in current_references],
        }
        content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(text, ensure_ascii=False, separators=(",", ":"))}
        ]
        for item in selected:
            content.extend(
                (
                    {"type": "text", "text": f"EVIDENCE {item.evidence_id}"},
                    _image_bytes_part(item.raster.read_bytes(), item.raster.name),
                )
            )
        for item in current_references:
            content.extend(
                (
                    {
                        "type": "text",
                        "text": f"RETAINED REFERENCE {decision.skill_id}/{item.reference_id}",
                    },
                    _image_bytes_part(item.image_bytes, item.filename),
                )
            )
        messages: list[Message] = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        write_json(
            output / "request.json",
            {
                "schema": "vaw-skill-review-request-v1",
                **text,
                "images": [
                    {"evidence_id": item.evidence_id, "path": str(item.raster), "sha256": sha256(item.raster)}
                    for item in selected
                ],
                "retained_reference_images": [
                    {
                        "skill_id": decision.skill_id,
                        "reference_id": item.reference_id,
                        "filename": item.filename,
                        "sha256": item.sha256,
                    }
                    for item in current_references
                ],
            },
        )
        response = self.provider.generate(messages, tools=None)
        write_json(output / "response.json", _response_record(response))
        payload = parse_model_json(response.text)
        review_decision = str(payload.get("decision") or "").strip().lower()
        reason = str(payload.get("reason") or "").strip()
        if review_decision not in {"accept", "reject"} or not reason:
            raise ValueError("Mutation Reviewer 必须输出 accept/reject 和 reason")
        write_json(output / "report.json", {"decision": review_decision, "reason": reason})
        return review_decision == "accept"


def run_skill_evolver(
    *,
    store: GenerationStore,
    parent_generation: str,
    mutation_id: str,
    m3_outputs: Sequence[str | Path],
    output_dir: str | Path,
    candidate_dir: str | Path,
    evolver_provider: ModelProvider,
    reviewer_provider: ModelProvider,
    max_inspections: int = 6,
    run_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """运行 M4；只有独立 Reviewer 接受后才密封 CandidatePackage。"""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if run_metadata is not None:
        write_json(output / "run.json", dict(run_metadata))
    manifest = store.read_manifest(parent_generation)
    library = MMSkillLibrary.from_root(
        store.generation_path(parent_generation) / "skills",
        source=f"evolution:{store.root}",
        generation_id=parent_generation,
        expected_digest=manifest.skill_digest,
    )
    evidence = EvidencePool.from_m3_outputs(m3_outputs)
    if not evidence.items:
        # 没有通过独立审核的证据时，唯一安全动作是不改变技能库。这个分支不是
        # 模型兜底，而是候选变更必须具备 evidence_ids 的领域约束。
        return write_json(
            output / "m4.json",
            {
                "schema": M4_SCHEMA,
                "status": "no_change",
                "decision": {
                    "decision": "no_change",
                    "rationale": "M3 没有产生经 Reviewer 接受的证据。",
                },
            },
        )
    decision = SkillEvolver(
        evolver_provider,
        max_inspections=max_inspections,
    ).run(evidence, library, output / "evolver")
    if run_metadata is not None:
        write_json(
            output / "evolver" / "run.json",
            dict(run_metadata.get("evolver") or {}),
        )
    if decision.decision == "no_change":
        return write_json(
            output / "m4.json",
            {"schema": M4_SCHEMA, "status": "no_change", "decision": decision.to_dict()},
        )

    accepted = MutationReviewer(reviewer_provider).run(
        decision,
        evidence,
        library,
        output / "review",
    )
    if run_metadata is not None:
        write_json(
            output / "review" / "run.json",
            dict(run_metadata.get("reviewer") or {}),
        )
    if not accepted:
        return write_json(
            output / "m4.json",
            {"schema": M4_SCHEMA, "status": "rejected", "decision": decision.to_dict()},
        )

    assert decision.operation is not None and decision.skill_id is not None
    mutation = SkillMutation(
        mutation_id=mutation_id,
        operation=decision.operation,
        skill_id=decision.skill_id,
        evidence_ids=decision.evidence_ids,
        rationale=decision.rationale,
    )
    skill_root = None
    if decision.operation is not MutationOperation.RETIRE:
        skill_root = output / "draft" / decision.skill_id
        _write_skill_snapshot(skill_root, decision, evidence, library)
    package = CandidatePackage.create(
        candidate_dir,
        mutation=mutation,
        evidence={item: evidence.get(item).candidate_evidence() for item in decision.evidence_ids},
        skill_root=skill_root,
        evolver_root=output / "evolver",
        review_root=output / "review",
    )
    return write_json(
        output / "m4.json",
        {
            "schema": M4_SCHEMA,
            "status": "candidate",
            "decision": decision.to_dict(),
            "candidate": str(package.root),
            "candidate_digest": package.digest,
        },
    )


def _write_skill_snapshot(
    root: Path,
    decision: EvolverDecision,
    evidence: EvidencePool,
    library: MMSkillLibrary,
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    assert decision.skill_markdown is not None and decision.skill_id is not None
    (root / "SKILL.md").write_text(decision.skill_markdown.rstrip() + "\n", encoding="utf-8")
    if not decision.references:
        return
    references_root = root / "references"
    references_root.mkdir()
    current_skill = None
    if decision.operation is MutationOperation.REVISE:
        current_skill = library.get(decision.skill_id)
    index: list[dict[str, str]] = []
    provenance: list[dict[str, str]] = []
    for number, choice in enumerate(decision.references, start=1):
        reference_id = f"r{number:03d}"
        filename = f"{reference_id}.png"
        if choice.is_current:
            if current_skill is None:
                raise ValueError("ADD 不能引用 current visual reference")
            source = current_skill.get_reference(choice.source_id.split(":", 1)[1])
            data = source.image_bytes
            metadata = source.summary()
            source_kind = "parent_reference"
            source_id = choice.source_id.split(":", 1)[1]
        else:
            item = evidence.get(choice.source_id)
            data = item.raster.read_bytes()
            source_kind = "m3_evidence"
            source_id = item.evidence_id
            metadata = {
                "state": choice.state or "",
                "view": choice.view or "",
                "when_to_use": choice.when_to_use or "",
                "visual_cue": choice.visual_cue or "",
            }
        image_path = references_root / filename
        image_path.write_bytes(data)
        index.append(
            {
                "reference_id": reference_id,
                "file": filename,
                "sha256": hashlib.sha256(data).hexdigest(),
                "state": str(metadata["state"]),
                "view": str(metadata["view"]),
                "when_to_use": str(metadata["when_to_use"]),
                "visual_cue": str(metadata["visual_cue"]),
            }
        )
        provenance.append(
            {
                "reference_id": reference_id,
                "source_kind": source_kind,
                "source_id": source_id,
                "source_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    write_json(
        references_root / "index.json",
        {"schema": REFERENCE_SCHEMA, "references": index},
    )
    write_json(
        references_root / "provenance.json",
        {"schema": "vaw-mmskill-reference-provenance-v1", "references": provenance},
    )


def _validate_reference_choices(
    choices: Sequence[ReferenceChoice],
    evidence_ids: Sequence[str],
    evidence: EvidencePool,
    library: MMSkillLibrary,
    skill_id: str,
) -> None:
    seen: set[str] = set()
    for choice in choices:
        if not choice.source_id or choice.source_id in seen:
            raise ValueError("reference source_id 不能为空或重复")
        seen.add(choice.source_id)
        if choice.is_current:
            library.get_reference(skill_id, choice.source_id.split(":", 1)[1])
            continue
        evidence.get(choice.source_id)
        if choice.source_id not in evidence_ids:
            raise ValueError("新 visual reference 必须来自 mutation 引用的 evidence")
        if not all((choice.state, choice.view, choice.when_to_use, choice.visual_cue)):
            raise ValueError("新 visual reference 缺少视觉语义元数据")


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _request_record(
    messages: Sequence[Message],
    tools: Sequence[Mapping[str, Any]] | None,
    latest: InspectedResource | None,
) -> dict[str, Any]:
    user = messages[-1]["content"]
    text_parts = [part["text"] for part in user if part.get("type") == "text"]
    return {
        "schema": "vaw-skill-evolver-turn-request-v1",
        "system": messages[0]["content"],
        "text": "\n".join(text_parts),
        "image": (
            {"label": latest.label, "sha256": hashlib.sha256(latest.image_bytes).hexdigest()}
            if latest is not None and latest.image_bytes is not None
            else None
        ),
        "tools": [item["function"]["name"] for item in tools] if tools else [],
    }


def _response_record(response: ModelResponse) -> dict[str, Any]:
    return {
        "text": response.text,
        "reasoning": response.provider_reasoning,
        "usage": response.usage,
        "tool_calls": [
            {
                "name": call.name,
                "arguments": call.args,
                "parse_error": call.parse_error,
            }
            for call in response.tool_calls
        ],
    }


__all__ = [
    "AcceptedEvidence",
    "EVOLVER_TOOLS",
    "EvidencePool",
    "EvolverDecision",
    "M4_SCHEMA",
    "MutationReviewer",
    "ReferenceChoice",
    "SkillEvolver",
    "run_skill_evolver",
]
