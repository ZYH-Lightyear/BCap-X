"""Reusable structural and contract linting for skill libraries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from robomex.dysc.contract_checks import contract_consistency_errors
from robomex.dysc.contracts import SkillContract

REQUIRED_SECTIONS = (
    "## Purpose",
    "## When to use",
    "## When NOT to use",
    "## Workflow",
    "## Candidate Generation",
    "## Local Checks",
    "## Failure Modes",
    "## Clean Reusable Rules",
    "## Weak Priors",
    "## Prohibited Shortcuts",
    "## Artifacts to Save",
    "## Multimodal Evidence Contract",
)
REQUIRED_CONTRACT_FIELDS = (
    "skill_id",
    "role",
    "changes_world",
    "capabilities",
    "forbidden_capabilities",
    "input_ports",
    "output_ports",
    "budget",
    "exit_conditions",
    "functions",
)
ALLOWED_PACKAGE_ENTRIES = {
    "SKILL.md",
    "contract.yaml",
    "assets",
    "references",
    "scripts",
    "prompts",
}
_SIDECAR_REFERENCE = re.compile(
    r"\b(?P<directory>scripts|references|prompts)/"
    r"(?P<name>[A-Za-z0-9_.-]+)"
)


def collect_library_errors(root: Path) -> list[str]:
    """Return all actionable package, prose, and contract errors under ``root``."""

    errors: list[str] = []
    skill_files = sorted(root.glob("*/*/SKILL.md"))
    if not skill_files:
        return [f"{root}: no <category>/<skill>/SKILL.md packages found"]

    for skill_file in skill_files:
        skill_root = skill_file.parent
        label = str(skill_root.relative_to(root))
        text = skill_file.read_text(encoding="utf-8")
        errors.extend(_section_errors(label, text))
        errors.extend(_package_layout_errors(label, skill_root))
        errors.extend(_sidecar_reference_errors(label, skill_root, text))
        errors.extend(_contract_errors(label, skill_root))
    return errors


def _section_errors(label: str, text: str) -> list[str]:
    errors = [
        f"{label}: missing required section {section!r}"
        for section in REQUIRED_SECTIONS
        if section not in text
    ]
    if "```python" in text and "## Reference Code" not in text:
        errors.append(f"{label}: Python guidance must live under `## Reference Code`")
    return errors


def _package_layout_errors(label: str, skill_root: Path) -> list[str]:
    entries = {path.name for path in skill_root.iterdir()}
    unknown = sorted(entries - ALLOWED_PACKAGE_ENTRIES)
    errors = [f"{label}: unsupported package entry {name!r}" for name in unknown]
    if "reference" in entries:
        errors.append(f"{label}: use `references/`, not `reference/`")
    scripts = skill_root / "scripts"
    if scripts.is_dir():
        text = (skill_root / "SKILL.md").read_text(encoding="utf-8")
        if "## Reference Code" not in text or "```python" not in text:
            errors.append(
                f"{label}: packages with scripts require executable `## Reference Code`"
            )
    return errors


def _sidecar_reference_errors(
    label: str,
    skill_root: Path,
    text: str,
) -> list[str]:
    errors: list[str] = []
    for match in _SIDECAR_REFERENCE.finditer(text):
        path = skill_root / match.group("directory") / match.group("name")
        if not path.is_file():
            errors.append(f"{label}: referenced sidecar does not exist: {path}")
    return errors


def _contract_errors(label: str, skill_root: Path) -> list[str]:
    path = skill_root / "contract.yaml"
    if not path.is_file():
        return [f"{label}: missing contract.yaml"]
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"{label}: invalid contract YAML: {exc}"]
    if not isinstance(raw, dict):
        return [f"{label}: contract.yaml must contain a mapping"]

    errors = [
        f"{label}: contract missing machine-authoritative field {field!r}"
        for field in REQUIRED_CONTRACT_FIELDS
        if field not in raw
    ]
    if raw.get("skill_id") != skill_root.name:
        errors.append(
            f"{label}: contract skill_id {raw.get('skill_id')!r} "
            f"does not match directory {skill_root.name!r}"
        )
    try:
        contract = SkillContract.from_mapping(raw)
    except (TypeError, ValueError) as exc:
        errors.append(f"{label}: contract parse failed: {exc}")
        return errors
    errors.extend(contract_consistency_errors(contract, skill_root))
    return errors


__all__ = [
    "ALLOWED_PACKAGE_ENTRIES",
    "REQUIRED_CONTRACT_FIELDS",
    "REQUIRED_SECTIONS",
    "collect_library_errors",
]
