"""内置技能包,按 category 组织,MMSkills 风格。

布局(对应 MMSkills 的 ``skills_library/<domain>/<skill>/``)::

    builtin/<category>/<skill_id>/SKILL.md

``category`` 取 ``perception`` / ``affordance`` / ``motion`` / ``task`` 之一(见
``SkillCategory``)。``README.md`` 是给人看的清单;用
``render_inventory(load_builtin_skills())`` 重新生成。
"""

from __future__ import annotations

from pathlib import Path

from robomex.skills.schema import SKILL_FILE, Skill, SkillCategory

_BUILTIN_DIR = Path(__file__).parent


def load_builtin_skills() -> list[Skill]:
    """加载所有内置技能包(``<category>/<skill>/SKILL.md``)。"""

    return [Skill.from_dir(p.parent) for p in sorted(_BUILTIN_DIR.glob(f"*/*/{SKILL_FILE}"))]


def render_inventory(skills: list[Skill]) -> str:
    """按 category 分组,渲染一份 README 形式的技能清单表。"""

    by_cat: dict[SkillCategory, list[Skill]] = {c: [] for c in SkillCategory}
    for skill in skills:
        by_cat[skill.category].append(skill)

    lines = [
        "# RoboMEx Skill Library",
        "",
        "Each skill is a self-contained package. The runtime loads `SKILL.md` through",
        "progressive disclosure; optional sidecars are ordinary files resolved from the",
        "loaded skill's base directory. Categories are not fixed execution phases or",
        "SubAgent profiles.",
        "",
        "## Package Structure",
        "",
        "```text",
        "<category>/<skill_id>/",
        "├── SKILL.md       # compact workflow memory",
        "├── references/    # optional explanatory material or non-runnable reference code",
        "├── assets/        # optional visual/static assets; may intentionally be empty",
        "└── scripts/       # optional runnable helper files with documented entry points",
        "```",
        "",
        "## Standard SKILL.md Shape",
        "",
        "Every built-in skill should include Purpose, When to use, Workflow, Candidate",
        "Generation, Local Checks, Failure Modes, Clean Reusable Rules, Weak Priors,",
        "Prohibited Shortcuts, Artifacts to Save, and Optional Sidecars when relevant.",
        "",
        "## Inventory",
        "",
        "| Category | Count | Skills |",
        "|----------|------:|--------|",
    ]
    for category in SkillCategory:
        items = sorted(by_cat[category], key=lambda s: s.skill_id)
        names = "; ".join(f"`{s.skill_id}` ({s.name})" for s in items) or "—"
        lines.append(f"| {category.value} | {len(items)} | {names} |")

    lines += ["", "## Directories", ""]
    for skill in sorted(skills, key=lambda s: (s.category.value, s.skill_id)):
        lines.append(f"- `{skill.category.value}/{skill.skill_id}` — {skill.description}")
    return "\n".join(lines) + "\n"


__all__ = ["load_builtin_skills", "render_inventory"]
