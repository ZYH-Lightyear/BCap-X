"""LLM summarization and curation for online DySC evolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robomex.core.coder.policy import CompletionPolicy
from robomex.dysc.patch import SocietyPatch
from robomex.dysc.society import SocietySpec


def parse_yaml_mapping(text: str) -> dict[str, Any]:
    """Parse a YAML mapping, accepting fenced model output."""

    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    data = yaml.safe_load(raw) if raw else {}
    return data if isinstance(data, dict) else {}


@dataclass
class TraceSummarizer:
    policy: CompletionPolicy
    max_chars: int = 28000

    def summarize_subgoal(
        self,
        *,
        task: str,
        subgoal_dir: Path | None,
        subgoal_result: Any,
        society: SocietySpec,
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        prompt = self._prompt(task=task, subgoal_dir=subgoal_dir, subgoal_result=subgoal_result, society=society)
        raw = self.policy.complete(prompt)
        data = parse_yaml_mapping(raw)
        return data, raw, prompt

    def _prompt(
        self,
        *,
        task: str,
        subgoal_dir: Path | None,
        subgoal_result: Any,
        society: SocietySpec,
    ) -> list[dict[str, Any]]:
        bundle = {
            "task": task,
            "society": society.to_mapping(),
            "subgoal": {
                "goal": getattr(subgoal_result.subgoal, "goal", ""),
                "postcondition": getattr(subgoal_result.subgoal, "postcondition", ""),
                "success": bool(getattr(subgoal_result, "success", False)),
                "note": getattr(subgoal_result, "note", ""),
            },
            "artifacts": _collect_subgoal_text(subgoal_dir, self.max_chars),
        }
        return [
            {
                "role": "system",
                "content": (
                    "You summarize embodied robot coding-agent traces for online society evolution. "
                    "Do not turn raw coordinates into reusable rules. Prefer natural-language lessons, "
                    "failure causes, useful/bad skill composition, and lightweight tags."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return YAML only with keys: task_outcome, important_events, candidate_lessons, "
                    "bad_bindings, good_motifs, suggested_focus, tags.\n\n"
                    f"Trace bundle:\n{json.dumps(bundle, ensure_ascii=False, indent=2, default=repr)}"
                ),
            },
        ]


@dataclass
class SocietyCurator:
    policy: CompletionPolicy

    def propose_patch(
        self,
        *,
        summary: dict[str, Any],
        society: SocietySpec,
    ) -> tuple[SocietyPatch, str, list[dict[str, Any]]]:
        prompt = self._prompt(summary=summary, society=society)
        raw = self.policy.complete(prompt)
        patch = SocietyPatch.from_mapping(parse_yaml_mapping(raw))
        return patch, raw, prompt

    @staticmethod
    def _prompt(*, summary: dict[str, Any], society: SocietySpec) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are the DySC Society Curator. Propose small online mutations to a "
                    "multi-agent skill society. Rely on the trace summary. Do not memorize absolute "
                    "coordinates. Prefer topology, skill preference, motif, role objective, or budget "
                    "changes. Return YAML only. Use at most 3 mutations."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Allowed ops: add_skill_preference, remove_skill_preference, add_forbid_skill, "
                    "remove_forbid_skill, add_topology_edge, remove_topology_edge, add_motif, "
                    "revise_role_objective, adjust_budget, rollback.\n\n"
                    "Patch YAML format:\n"
                    "rationale: short reason\n"
                    "mutations:\n"
                    "  - op: add_skill_preference\n"
                    "    policy: executor_policy\n"
                    "    skill: offset_aware_place_with_hover_verify\n\n"
                    f"Current society:\n{society.to_yaml()}\n\n"
                    f"Trace summary:\n{yaml.safe_dump(summary, sort_keys=False, allow_unicode=True, width=100)}"
                ),
            },
        ]


def _collect_subgoal_text(subgoal_dir: Path | None, max_chars: int) -> dict[str, str]:
    if subgoal_dir is None or not subgoal_dir.exists():
        return {}
    wanted = [
        "meta.json",
        "diagnoses_after.json",
        "attempt_history_after.json",
        "evidence_timeline.md",
    ]
    payload: dict[str, str] = {}
    for name in wanted:
        path = subgoal_dir / name
        if path.exists():
            payload[name] = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    for path in sorted(subgoal_dir.glob("turn_*.out.txt"))[:8]:
        payload[path.name] = path.read_text(encoding="utf-8", errors="replace")[:5000]
    return payload
