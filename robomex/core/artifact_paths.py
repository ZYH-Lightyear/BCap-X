"""Shared artifact path convention for typed finish payloads.

One rule, enforced identically at the agent's finish gate and at the
artifact store: a relative artifact path always resolves against the
node's ARTIFACTS_DIR, never against the process CWD. Keeping this in
:mod:`robomex.core` lets both :mod:`robomex.agents.subagents` and
:mod:`robomex.authoring.artifacts` import it without a package cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_artifact_ref_path(raw_path: str, artifacts_dir: Path | None) -> Path:
    """Resolve one artifact file reference deterministically.

    Absolute paths are taken as-is. Relative paths resolve against
    ``artifacts_dir`` when it is known; resolving against the CWD is
    forbidden because the sandbox CWD is the repository root and a
    CWD-relative file silently escapes the artifact root.
    """

    path = Path(raw_path)
    if path.is_absolute() or artifacts_dir is None:
        return path.resolve()
    return (artifacts_dir / path).resolve()


def collect_output_file_refs(outputs: dict[str, Any]) -> dict[str, str]:
    """Collect every ``label -> path`` file reference from typed outputs.

    Mirrors the shapes accepted by the artifact store: an ``artifacts``
    object/list/string per output, plus ``*_path`` string entries inside
    each output's ``payload``.
    """

    refs: dict[str, str] = {}
    for name, value in outputs.items():
        if not isinstance(value, dict):
            continue
        raw = value.get("artifacts")
        if raw is None:
            raw = value.get("artifact") or value.get("artifact_path")
        if isinstance(raw, dict):
            for key, item in raw.items():
                if isinstance(item, str) and item:
                    refs[f"{name}.artifacts.{key}"] = item
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    refs[f"{name}.artifacts.{key}"] = item["path"]
        elif isinstance(raw, str) and raw:
            refs[f"{name}.artifacts"] = raw
        elif isinstance(raw, (list, tuple)):
            for index, item in enumerate(raw):
                if isinstance(item, str) and item:
                    refs[f"{name}.artifacts[{index}]"] = item
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    label = str(item.get("artifact_id") or index)
                    refs[f"{name}.artifacts[{label}]"] = item["path"]
        payload = value.get("payload")
        if isinstance(payload, dict):
            for key, item in payload.items():
                if str(key).endswith("_path") and isinstance(item, str) and item:
                    refs[f"{name}.payload.{key}"] = item
    return refs


def finish_artifact_path_errors(
    outputs: dict[str, Any],
    artifacts_dir: Path | None,
) -> list[str]:
    """Return LLM-repairable errors for artifact file refs in a finish payload.

    Runs inside the producing agent's own turn budget so a path-convention
    mistake is repaired immediately instead of failing the node after the
    agent has already finished.
    """

    if artifacts_dir is None:
        return []
    root = artifacts_dir.resolve()
    errors: list[str] = []
    for label, raw_path in collect_output_file_refs(outputs).items():
        resolved = resolve_artifact_ref_path(raw_path, root)
        if not Path(raw_path).is_absolute():
            try:
                resolved.relative_to(root)
            except ValueError:
                errors.append(
                    f"`{label}` = {raw_path!r} escapes ARTIFACTS_DIR after resolving "
                    f"to {resolved}; use a plain relative path inside ARTIFACTS_DIR."
                )
                continue
        if not resolved.exists():
            errors.append(
                f"`{label}` = {raw_path!r} resolves to {resolved}, which does not "
                "exist. Save the file under ARTIFACTS_DIR (already defined in your "
                "sandbox) and reference it relative to ARTIFACTS_DIR or by that "
                "absolute path; relative paths never resolve against the CWD."
            )
    return errors
