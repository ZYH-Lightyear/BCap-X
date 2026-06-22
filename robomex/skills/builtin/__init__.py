"""内置技能包,按 category 组织,MMSkills 风格。

布局(对应 MMSkills 的 ``skills_library/<domain>/<skill>/``)::

    builtin/<category>/<skill_id>/SKILL.md
                                  ref/           (可选:验证器资产)
                                  scripts/       (可选:verifier-as-code)

``category`` 取 ``high_level`` / ``observation`` / ``action`` 之一(见
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


__all__ = ["load_builtin_skills", "render_inventory"]
