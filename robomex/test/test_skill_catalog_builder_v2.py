from __future__ import annotations

from pathlib import Path

import pytest

from robomex.contracts import (
    SkillAssetKind,
    SkillCatalogBuildError,
    build_skill_contract_catalog,
)
from robomex.skills.builtin import load_builtin_skills
from robomex.skills.schema import Skill
from robomex.skills.store import SkillLibrary

_BOWL_SKILLS = (
    "author_attachment_monitor",
    "author_sealed_phase_motion",
    "estimate_support_alignment",
    "propose_attachment_transition",
    "propose_relation_transition",
)


def _bowl_library(tmp_path: Path) -> SkillLibrary:
    library = SkillLibrary(tmp_path / "library")
    selected = set(_BOWL_SKILLS)
    for skill in load_builtin_skills():
        if skill.skill_id in selected:
            library.admit(skill)
    return library


def _custom_skill(
    tmp_path: Path,
    *,
    skill_id: str = "pure_helper",
    changes_world: bool = False,
    outcome: str = "success",
    required_skills: tuple[str, ...] = (),
) -> SkillLibrary:
    source = tmp_path / "source" / skill_id
    (source / "scripts").mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: Pure helper\ncategory: motion\n"
        "description: A proposal-only test helper.\n---\n\nUse the helper.\n",
        encoding="utf-8",
    )
    # Importing this sidecar would raise. Catalog construction must only parse
    # its AST and is therefore safe at bootstrap time.
    (source / "scripts" / "helper.py").write_text(
        "raise RuntimeError('must not import')\n\n"
        "def build(value: float) -> dict:\n"
        "    return {'value': value}\n",
        encoding="utf-8",
    )
    dependencies = "\n".join(f"  - {value}" for value in required_skills)
    required = f"required_skills:\n{dependencies}\n" if dependencies else ""
    (source / "contract.yaml").write_text(
        f"skill_id: {skill_id}\n"
        "role: proposal_author\n"
        f"changes_world: {'true' if changes_world else 'false'}\n"
        "capabilities: [artifact_write]\n"
        f"{required}"
        "input_ports:\n"
        "  - {name: source, schema: robomex.source.v1}\n"
        "output_ports:\n"
        "  - {name: proposal, schema: robomex.proposal.v1}\n"
        "exit_conditions:\n"
        f"  {outcome}: done\n"
        "functions:\n"
        "  - name: build\n"
        "    entry: scripts/helper.py:build\n",
        encoding="utf-8",
    )
    library = SkillLibrary(tmp_path / "library")
    library.admit(Skill.from_dir(source))
    return library


def test_bowl_skill_packages_compile_to_deterministic_closed_catalog(tmp_path: Path) -> None:
    first_library = _bowl_library(tmp_path / "one")
    second_library = _bowl_library(tmp_path / "two")
    profiles = dict.fromkeys(_BOWL_SKILLS, ("bowl.coding",))

    first = build_skill_contract_catalog(
        first_library,
        _BOWL_SKILLS,
        compatible_actor_profiles=profiles,
    )
    second = build_skill_contract_catalog(
        second_library,
        reversed(_BOWL_SKILLS),
        compatible_actor_profiles=profiles,
    )

    assert first == second
    assert len(first.skills) == len(first.protocols) == len(first.effect_contracts) == 5
    assert sum(len(skill.functions) for skill in first.skills) == 5
    assert all(skill.compatible_actor_profiles == ("bowl.coding",) for skill in first.skills)
    assert all(
        asset.relative_path != "utility.json"
        for skill in first.skills
        for asset in skill.assets
    )
    assert all(
        effect.effects[0].scope.value == "read_only"
        for effect in first.effect_contracts
    )


def test_catalog_builder_hashes_all_assets_and_detects_package_change(tmp_path: Path) -> None:
    library = _custom_skill(tmp_path)
    before = build_skill_contract_catalog(library, ("pure_helper",))
    root = library.get("pure_helper").skill.root
    assert root is not None
    (root / "SKILL.md").write_text(
        (root / "SKILL.md").read_text(encoding="utf-8") + "\nNew rule.\n",
        encoding="utf-8",
    )
    after = build_skill_contract_catalog(library, ("pure_helper",))

    assert before.content_digest != after.content_digest
    skill = after.skills[0]
    assert {asset.kind for asset in skill.assets} == {
        SkillAssetKind.KNOWLEDGE,
        SkillAssetKind.API,
        SkillAssetKind.CODE,
    }
    assert skill.functions[0].entrypoint == "scripts/helper.py:build"


def test_catalog_builder_never_imports_sidecar_module(tmp_path: Path) -> None:
    library = _custom_skill(tmp_path)
    catalog = build_skill_contract_catalog(library, ("pure_helper",))
    assert catalog.skills[0].functions[0].function_id == "function.pure_helper.build"


def test_catalog_builder_rejects_inferred_world_effects(tmp_path: Path) -> None:
    library = _custom_skill(tmp_path, changes_world=True)
    with pytest.raises(SkillCatalogBuildError, match="explicitly authored"):
        build_skill_contract_catalog(library, ("pure_helper",))


def test_catalog_builder_rejects_unknown_outcome_and_missing_dependency(
    tmp_path: Path,
) -> None:
    unknown = _custom_skill(tmp_path / "unknown", outcome="invented_result")
    with pytest.raises(SkillCatalogBuildError, match="unknown outcome"):
        build_skill_contract_catalog(unknown, ("pure_helper",))

    dependency = _custom_skill(
        tmp_path / "dependency",
        required_skills=("not_selected",),
    )
    with pytest.raises(SkillCatalogBuildError, match="unpinned required skills"):
        build_skill_contract_catalog(dependency, ("pure_helper",))


def test_catalog_builder_rejects_duplicates_and_unknown_skills(tmp_path: Path) -> None:
    library = _custom_skill(tmp_path)
    with pytest.raises(SkillCatalogBuildError, match="duplicates"):
        build_skill_contract_catalog(library, ("pure_helper", "pure_helper"))
    with pytest.raises(SkillCatalogBuildError, match="unknown requested"):
        build_skill_contract_catalog(library, ("missing",))
