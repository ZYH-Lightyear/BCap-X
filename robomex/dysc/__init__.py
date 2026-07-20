"""DySC: dynamic skill composition primitives.

This package intentionally starts as a non-invasive layer over RoboMEx's existing
skill library and agent loop.  It provides structured contracts, society specs,
and role-conditioned skill-library views without changing how SKILL.md is loaded
by the base coding agent.
"""

from robomex.dysc.contracts import ContractPort, SkillContract, load_skill_contracts
from robomex.dysc.online import DySCOnlineConfig, OnlineEvolutionManager
from robomex.dysc.patch import SocietyPatch, apply_society_patch
from robomex.dysc.society import RoleSpec, SkillAccessPolicy, SocietySpec, load_society_spec
from robomex.dysc.views import SkillLibraryView, specialist_skill_view

__all__ = [
    "DySCOnlineConfig",
    "ContractPort",
    "OnlineEvolutionManager",
    "RoleSpec",
    "SkillAccessPolicy",
    "SkillContract",
    "SkillLibraryView",
    "SocietyPatch",
    "SocietySpec",
    "apply_society_patch",
    "load_skill_contracts",
    "load_society_spec",
    "specialist_skill_view",
]
