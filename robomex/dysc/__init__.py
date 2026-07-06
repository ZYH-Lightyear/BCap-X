"""DySC: dynamic skill composition primitives.

This package intentionally starts as a non-invasive layer over RoboMEx's existing
skill library and agent loop.  It provides structured contracts, society specs,
and role-conditioned skill-library views without changing how SKILL.md is loaded
by the base coding agent.
"""

from robomex.dysc.contracts import SkillContract, load_skill_contracts
from robomex.dysc.society import RoleSpec, SkillAccessPolicy, SocietySpec, load_society_spec
from robomex.dysc.views import SkillLibraryView

__all__ = [
    "RoleSpec",
    "SkillAccessPolicy",
    "SkillContract",
    "SkillLibraryView",
    "SocietySpec",
    "load_skill_contracts",
    "load_society_spec",
]
