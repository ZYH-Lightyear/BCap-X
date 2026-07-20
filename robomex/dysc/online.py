"""Online DySC evolution manager."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robomex.core.coder.policy import CompletionPolicy
from robomex.dysc.contracts import load_skill_contracts
from robomex.dysc.llm import SocietyCurator, TraceSummarizer
from robomex.dysc.patch import SocietyPatchError, apply_society_patch
from robomex.dysc.society import SocietySpec, load_society_spec
from robomex.dysc.versioning import SocietyVersionStore
from robomex.dysc.views import SkillLibraryView


@dataclass
class DySCOnlineConfig:
    """Config for online society evolution."""

    society_path: str | None = None
    society_dir: str | None = None
    enabled: bool = True


class OnlineEvolutionManager:
    """Coordinates society views, LLM summaries, and online patch application."""

    def __init__(
        self,
        *,
        config: DySCOnlineConfig,
        library: Any,
        policy: CompletionPolicy,
        artifacts_dir: str | Path | None,
    ) -> None:
        self.config = config
        self.library = library
        self.policy = policy
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None
        self.contracts = load_skill_contracts(getattr(library, "root", ""))
        society = load_society_spec(config.society_path) if config.society_path else _default_society()
        store_root = (
            Path(config.society_dir)
            if config.society_dir
            else (self.artifacts_dir / "dysc_society" if self.artifacts_dir is not None else Path("/tmp/robomex_dysc_society"))
        )
        self.store = SocietyVersionStore(store_root)
        self.store.initialize(society)
        self.current = self.store.latest()
        self.summarizer = TraceSummarizer(policy)
        self.curator = SocietyCurator(policy)

    def library_view_for_motion_role(self) -> Any:
        role_name = self.current.motion_role_name()
        if not role_name:
            return self.library
        policy = self.current.policy_for_role(role_name)
        return SkillLibraryView(self.library, policy, self.contracts)

    def evolve_after_subgoal(
        self,
        *,
        task: str,
        subgoal_dir: Path | None,
        subgoal_result: Any,
    ) -> None:
        next_version = self.store.latest_version() + 1
        try:
            summary, summary_raw, summary_prompt = self.summarizer.summarize_subgoal(
                task=task,
                subgoal_dir=subgoal_dir,
                subgoal_result=subgoal_result,
                society=self.current,
            )
            self.store.write_prompt(next_version, "summarizer", summary_prompt)
            patch, curator_raw, curator_prompt = self.curator.propose_patch(summary=summary, society=self.current)
            self.store.write_prompt(next_version, "curator", curator_prompt)
            updated = apply_society_patch(self.current, patch)
            version = self.store.write_next(updated, patch=patch, summary=summary, curator_raw=curator_raw)
            (self.store.root / f"summarizer_response_{version:03d}.txt").write_text(summary_raw, encoding="utf-8")
            self.current = updated
        except (SocietyPatchError, Exception) as exc:  # noqa: BLE001 - online evolution must not kill robot execution
            if self.store.root.exists():
                (self.store.root / f"evolution_error_{next_version:03d}.txt").write_text(str(exc), encoding="utf-8")


def _default_society() -> SocietySpec:
    return SocietySpec.from_mapping({
        "name": "default_dysc_society",
        "roles": {
            "act_executor": {
                "objective": "write and execute bounded robot code using skill guidance",
                "skill_access_policy": "executor_policy",
                "execution_boundary": "motion_allowed",
            }
        },
        "skill_access_policies": {
            "executor_policy": {}
        },
        "topology": [{"from": "act_executor", "to": "curator", "when": "subgoal_end"}],
    })
