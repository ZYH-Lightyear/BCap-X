"""Versioned SocietySpec storage for online DySC evolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robomex.dysc.patch import SocietyPatch
from robomex.dysc.society import SocietySpec, dump_society_spec, load_society_spec


@dataclass(frozen=True)
class SocietyVersionStore:
    root: Path

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "root", Path(root))
        self.root.mkdir(parents=True, exist_ok=True)

    def initialize(self, spec: SocietySpec) -> Path:
        path = self.root / "society_v000.yaml"
        if not path.exists():
            dump_society_spec(spec, path)
        return path

    def latest_version(self) -> int:
        versions = [
            _parse_version(path.name)
            for path in self.root.glob("society_v*.yaml")
        ]
        versions = [v for v in versions if v is not None]
        return max(versions) if versions else -1

    def latest(self) -> SocietySpec:
        version = self.latest_version()
        if version < 0:
            raise FileNotFoundError("no society versions found")
        return load_society_spec(self.version_path(version))

    def version_path(self, version: int) -> Path:
        return self.root / f"society_v{version:03d}.yaml"

    def write_next(
        self,
        spec: SocietySpec,
        *,
        patch: SocietyPatch | None = None,
        summary: dict[str, Any] | None = None,
        curator_raw: str = "",
    ) -> int:
        version = self.latest_version() + 1
        dump_society_spec(spec, self.version_path(version))
        if patch is not None:
            (self.root / f"society_patch_{version:03d}.yaml").write_text(
                yaml.safe_dump(patch.to_mapping(), sort_keys=False, allow_unicode=True, width=100),
                encoding="utf-8",
            )
        if summary is not None:
            (self.root / f"dysc_trace_summary_{version:03d}.yaml").write_text(
                yaml.safe_dump(summary, sort_keys=False, allow_unicode=True, width=100),
                encoding="utf-8",
            )
        if curator_raw:
            (self.root / f"curator_response_{version:03d}.txt").write_text(curator_raw, encoding="utf-8")
        return version

    def write_prompt(self, version: int, kind: str, prompt: list[dict[str, Any]]) -> None:
        (self.root / f"{kind}_prompt_{version:03d}.json").write_text(
            json.dumps(prompt, indent=2, ensure_ascii=False, default=repr),
            encoding="utf-8",
        )


def _parse_version(name: str) -> int | None:
    if not name.startswith("society_v") or not name.endswith(".yaml"):
        return None
    try:
        return int(name[len("society_v"):-len(".yaml")])
    except ValueError:
        return None
