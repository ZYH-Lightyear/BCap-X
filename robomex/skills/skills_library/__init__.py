"""Built-in skill library: category-organized skill packages, MMSkills-style.

Layout (mirrors MMSkills' ``skills_library/<domain>/<skill>/``)::

    skills_library/<category>/<skill_id>/SKILL.md
                                         verify.py      (optional)
                                         references/    (optional)

``category`` is one of ``high_level`` / ``observation`` / ``action`` (see
``SkillCategory``). ``README.md`` is a human-facing inventory; regenerate it with
``render_inventory(load_skills_library())``.
"""

from __future__ import annotations

from pathlib import Path

from robomex.skills.schema import SKILL_FILE, Skill, SkillCategory

_LIBRARY_DIR = Path(__file__).parent


def load_skills_library() -> list[Skill]:
    """Load every built-in skill package (``<category>/<skill>/SKILL.md``)."""

    return [Skill.from_dir(p.parent) for p in sorted(_LIBRARY_DIR.glob(f"*/*/{SKILL_FILE}"))]


def render_inventory(skills: list[Skill]) -> str:
    """Render a README-style inventory table grouped by category."""

    by_cat: dict[SkillCategory, list[Skill]] = {c: [] for c in SkillCategory}
    for skill in skills:
        by_cat[skill.category].append(skill)

    lines = [
        "# RoboMEx Skill Library",
        "",
        "Each skill is a self-contained package, split by *reader*: `SKILL.md` for the",
        "executor agent, `ref/` for the Verifier Agent, `scripts/` for deterministic code.",
        "",
        "## Package Structure",
        "",
        "```text",
        "<category>/<skill_id>/",
        "├── SKILL.md       # executor agent: when-to-use, procedure/decomposition, recovery",
        "├── ref/           # Verifier Agent: verify.md rubric (+ optional visual references)",
        "└── scripts/       # optional: deterministic verifier-as-code (verify.py)",
        "```",
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

    lines += ["", "## Directories", "", "Sidecars: `V` = ref/verify.md, `C` = scripts/verify.py.", ""]
    for skill in sorted(skills, key=lambda s: (s.category.value, s.skill_id)):
        badges = "".join((
            "V" if skill.verify_doc_path() else "-",
            "C" if skill.verifier_path() else "-",
        ))
        lines.append(f"- `[{badges}] {skill.category.value}/{skill.skill_id}` — {skill.description}")
    return "\n".join(lines) + "\n"


__all__ = ["load_skills_library", "render_inventory"]
