"""Load and compose the phase-indexed Main playbook library."""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Literal

PHASE_NAMES = ("grasp", "transport", "align", "place", "recover", "routing")
Injection = Literal["all", "phase"]

_SLOT_RE = re.compile(
    r"<!-- vaw:slot (?P<name>[A-Za-z0-9_]+) -->\n?(?P<body>.*?)<!-- /vaw:slot -->",
    re.DOTALL,
)

# Original SYSTEM_PROMPT reading order.  Fragments already carry their
# trailing newlines, so composition is a plain concatenation.
GEN0_ASSEMBLY: tuple[tuple[str, str], ...] = (
    ("contract", "preamble"),
    ("recover", "grounding"),
    ("routing", "select"),
    ("align", "main"),
    ("place", "main"),
    ("routing", "amber"),
    ("grasp", "close"),
    ("transport", "main"),
    ("recover", "restore"),
    ("grasp", "geometry"),
    ("contract", "coords"),
)


@dataclass(frozen=True)
class PlaybookConfig:
    directory: pathlib.Path
    injection: Injection = "all"
    phase: str | None = None

    def __post_init__(self) -> None:
        if self.injection not in ("all", "phase"):
            raise ValueError(f"unknown playbook injection '{self.injection}'")
        if self.phase is not None and self.phase not in PHASE_NAMES:
            raise ValueError(f"unknown playbook phase '{self.phase}'")


def default_playbook_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "playbooks"


def load_playbooks(directory: str | pathlib.Path | None = None) -> dict[str, dict[str, str]]:
    root = pathlib.Path(directory) if directory is not None else default_playbook_dir()
    loaded: dict[str, dict[str, str]] = {}
    for name in PHASE_NAMES:
        path = root / f"{name}.md"
        if not path.is_file():
            raise FileNotFoundError(f"missing playbook {path}")
        loaded[name] = _parse_slots(path.read_text(encoding="utf-8"), default_name=name)
    return loaded


def compose_prompt(
    playbooks: dict[str, dict[str, str]],
    *,
    contract_preamble: str,
    contract_coords: str,
    injection: Injection = "all",
    phase: str | None = None,
) -> str:
    contract = {"preamble": contract_preamble, "coords": contract_coords}
    if injection == "phase" and phase is not None:
        body = "".join(playbooks[phase].values())
        return f"{contract_preamble}{body}{contract_coords}"
    chunks: list[str] = []
    for source, slot in GEN0_ASSEMBLY:
        if source == "contract":
            chunks.append(contract[slot])
            continue
        try:
            chunks.append(playbooks[source][slot])
        except KeyError as exc:
            raise KeyError(f"playbook {source!r} missing slot {slot!r}") from exc
    return "".join(chunks)


def compose_default_main_prompt(
    directory: str | pathlib.Path | None = None,
    *,
    injection: Injection = "all",
    phase: str | None = None,
) -> str:
    from vaw.context_runtime.protocol import CONTRACT_COORDS, CONTRACT_PREAMBLE

    return compose_prompt(
        load_playbooks(directory),
        contract_preamble=CONTRACT_PREAMBLE,
        contract_coords=CONTRACT_COORDS,
        injection=injection,
        phase=phase,
    )


def _parse_slots(text: str, *, default_name: str) -> dict[str, str]:
    matches = list(_SLOT_RE.finditer(text))
    if not matches:
        body = text
        if body.startswith("#"):
            body = body.split("\n", 1)[1]
        return {default_name: body.lstrip("\n")}
    return {match.group("name"): match.group("body") for match in matches}


__all__ = [
    "GEN0_ASSEMBLY",
    "PHASE_NAMES",
    "PlaybookConfig",
    "compose_default_main_prompt",
    "compose_prompt",
    "default_playbook_dir",
    "load_playbooks",
]
