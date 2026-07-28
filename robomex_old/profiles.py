"""RoboMEx runtime profile — trimmed to fields with current consumers."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ProfileError(ValueError):
    pass


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    planner: str = Field(min_length=1)
    manager: str = Field(min_length=1)
    coding_agent: str = Field(min_length=1)
    vlm: str = Field(min_length=1)
    server_url: str = Field(min_length=1)
    api_key_env: str | None = None


class RuntimeProfile(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, str_strip_whitespace=True)

    profile_id: str = Field(min_length=1)
    models: ModelProfile
    output_root: Path = Path("outputs/robomex_planner")

    @classmethod
    def load(cls, name_or_path: str | Path) -> RuntimeProfile:
        candidate = Path(name_or_path)
        if not candidate.suffix:
            candidate = Path(__file__).with_name("profiles") / f"{candidate.name}.yaml"
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.is_file():
            raise ProfileError(f"RoboMEx profile does not exist: {candidate}")
        raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ProfileError(f"profile must contain a YAML mapping: {candidate}")
        return cls.model_validate(raw)


def builtin_profile_names() -> tuple[str, ...]:
    root = Path(__file__).with_name("profiles")
    if not root.is_dir():
        return ()
    return tuple(path.stem for path in sorted(root.glob("*.yaml")))


__all__ = [
    "ModelProfile",
    "ProfileError",
    "RuntimeProfile",
    "builtin_profile_names",
]
