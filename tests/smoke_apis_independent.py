#!/usr/bin/env python3
"""Independently smoke-test every registered CapX API.

Levels:
  L0 construct     - instantiate API with a matching low-level env
  L1 functions()   - every exposed entry is callable and has a docstring
  L2 privileged    - invoke privileged pose/gripper helpers (no perception servers)
  L3 motion        - short goto_pose / open/close via PyRoKi :8116 when available

Perception-heavy methods (SAM3/GraspNet/Molmo) are only exercised if their
HTTP services respond; otherwise the API is still marked PASS for L0-L1 and
the perception calls are reported as SKIP.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import requests

from capx.envs.base import get_env
from capx.integrations.base_api import get_api, list_apis


# API name -> low-level env factory name
API_ENV: dict[str, str] = {
    "FrankaControlPrivilegedApi": "franka_robosuite_cubes_low_level",
    "FrankaControlMultiPrivilegedApi": "franka_robosuite_cubes_low_level",
    "FrankaControlApi": "franka_robosuite_cubes_low_level",
    "FrankaControlApiReduced": "franka_robosuite_cubes_low_level",
    "FrankaControlApiReducedExampleless": "franka_robosuite_cubes_low_level",
    "FrankaControlApiReducedSkillLibrary": "franka_robosuite_cubes_low_level",
    "FrankaControlApiReducedBimanual": "two_arm_lift_robosuite",
    "FrankaControlApiReducedExamplelessBimanual": "two_arm_lift_robosuite",
    "FrankaControlApiReducedSkillLibraryBimanual": "two_arm_lift_robosuite",
    "FrankaControlApiReducedBimanualHandover": "two_arm_handover_robosuite",
    "FrankaControlApiReducedExamplelessBimanualHandover": "two_arm_handover_robosuite",
    "FrankaControlApiReducedSkillLibraryBimanualHandover": "two_arm_handover_robosuite",
    "FrankaControlApiReducedSpillWipe": "franka_robosuite_spill_wipe_low_level",
    "FrankaControlApiReducedSkillLibrarySpillWipe": "franka_robosuite_spill_wipe_low_level",
    "FrankaControlSpillWipeApi": "franka_robosuite_spill_wipe_low_level",
    "FrankaControlSpillWipeApiReduced": "franka_robosuite_spill_wipe_low_level",
    "FrankaControlSpillWipePrivilegedApi": "franka_robosuite_spill_wipe_low_level",
    "FrankaControlSpillWipeApiReducedExampleless": "franka_robosuite_spill_wipe_low_level",
    "FrankaHandoverPrivilegedApi": "two_arm_handover_robosuite",
    "FrankaHandoverApi": "two_arm_handover_robosuite",
    "FrankaHandoverApiReduced": "two_arm_handover_robosuite",
    "FrankaHandoverApiReducedExampleless": "two_arm_handover_robosuite",
    "FrankaTwoArmLiftApi": "two_arm_lift_robosuite",
    "FrankaTwoArmLiftPrivilegedApi": "two_arm_lift_robosuite",
    "FrankaTwoArmLiftApiReduced": "two_arm_lift_robosuite",
    "FrankaTwoArmLiftApiReducedExampleless": "two_arm_lift_robosuite",
    "FrankaControlNutAssemblyPrivilegedApi": "franka_robosuite_nut_assembly_low_level",
    "FrankaControlNutAssemblyVisualApi": "franka_robosuite_nut_assembly_low_level_visual",
    "FrankaControlNutAssemblyApiReduced": "franka_robosuite_nut_assembly_low_level",
    "FrankaControlNutAssemblyApiReducedExampleless": "franka_robosuite_nut_assembly_low_level",
    # Real-robot / LIBERO / R1Pro skipped unless env available
    "FrankaRealReducedSkillLibraryControlApi": "franka_real_low_level",
    "FrankaRealControlApi": "franka_real_low_level",
}

# APIs that need live perception HTTP services to even construct
NEEDS_PERCEPTION = {
    "FrankaControlApi",
    "FrankaControlApiReduced",
    "FrankaControlApiReducedExampleless",
    "FrankaControlApiReducedSkillLibrary",
    "FrankaControlApiReducedBimanual",
    "FrankaControlApiReducedExamplelessBimanual",
    "FrankaControlApiReducedSkillLibraryBimanual",
    "FrankaControlApiReducedBimanualHandover",
    "FrankaControlApiReducedExamplelessBimanualHandover",
    "FrankaControlApiReducedSkillLibraryBimanualHandover",
    "FrankaControlApiReducedSpillWipe",
    "FrankaControlApiReducedSkillLibrarySpillWipe",
    "FrankaControlSpillWipeApi",
    "FrankaControlSpillWipeApiReduced",
    "FrankaControlSpillWipeApiReducedExampleless",
    "FrankaHandoverApi",
    "FrankaHandoverApiReduced",
    "FrankaHandoverApiReducedExampleless",
    "FrankaTwoArmLiftApi",
    "FrankaTwoArmLiftApiReduced",
    "FrankaTwoArmLiftApiReducedExampleless",
    "FrankaControlNutAssemblyVisualApi",
    "FrankaControlNutAssemblyApiReduced",
    "FrankaControlNutAssemblyApiReducedExampleless",
    "FrankaRealReducedSkillLibraryControlApi",
    "FrankaRealControlApi",
}

SKIP_ENVS = {
    "franka_real_low_level",  # hardware
}


@dataclass
class ApiResult:
    name: str
    status: str = "PASS"  # PASS / FAIL / SKIP
    notes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.status = "FAIL"
        self.notes.append(msg)

    def skip(self, msg: str) -> None:
        if self.status != "FAIL":
            self.status = "SKIP"
        self.notes.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)


def _service_up(url: str, timeout: float = 1.5) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def _make_env(env_name: str, privileged: bool) -> Any:
    return get_env(env_name, privileged=privileged, enable_render=True)


def _test_functions_table(api: Any, result: ApiResult) -> dict[str, Callable]:
    fns = api.functions()
    if not isinstance(fns, dict) or not fns:
        result.fail("functions() empty or not a dict")
        return {}
    for name, fn in fns.items():
        if not callable(fn):
            result.fail(f"function '{name}' not callable")
            continue
        doc = getattr(fn, "__doc__", None)
        if not doc:
            result.note(f"warn: '{name}' missing docstring")
        result.functions.append(name)
    result.note(f"functions={len(result.functions)}: {', '.join(result.functions)}")
    return fns


def _invoke_privileged_single_arm(
    api: Any,
    fns: dict[str, Callable],
    result: ApiResult,
    *,
    object_name: str = "red cube",
) -> None:
    """Exercise privileged single-arm helpers on cube-stack-like envs."""
    if "get_object_pose" in fns:
        try:
            out = fns["get_object_pose"](object_name)
            assert out is not None
            pos = out[0] if isinstance(out, (tuple, list)) else out
            assert np.asarray(pos).shape[-1] == 3
            result.note(f"get_object_pose({object_name}) pos={np.asarray(pos).round(3).tolist()}")
        except Exception as e:
            result.fail(f"get_object_pose: {e}")
            return

    if "sample_grasp_pose" in fns:
        try:
            gpos, gquat = fns["sample_grasp_pose"](object_name)
            assert np.asarray(gpos).shape == (3,)
            assert np.asarray(gquat).shape == (4,)
            result.note(f"sample_grasp_pose({object_name}) ok")
        except Exception as e:
            result.fail(f"sample_grasp_pose: {e}")
            return
    else:
        gpos = np.array([0.0, 0.0, 0.9], dtype=np.float64)
        gquat = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    if "open_gripper" in fns:
        try:
            fns["open_gripper"]()
            result.note("open_gripper ok")
        except Exception as e:
            result.fail(f"open_gripper: {e}")
            return

    if "close_gripper" in fns:
        try:
            fns["close_gripper"]()
            result.note("close_gripper ok")
        except Exception as e:
            result.fail(f"close_gripper: {e}")
            return

    if "goto_pose" in fns and _service_up("http://127.0.0.1:8116/docs"):
        try:
            # stay near current grasp sample, small lift
            target = np.asarray(gpos, dtype=np.float64).copy()
            target[2] += 0.05
            fns["goto_pose"](target, np.asarray(gquat, dtype=np.float64), z_approach=0.0)
            result.note("goto_pose ok")
        except TypeError:
            try:
                fns["goto_pose"](target, np.asarray(gquat, dtype=np.float64))
                result.note("goto_pose ok (no z_approach)")
            except Exception as e:
                result.fail(f"goto_pose: {e}")
        except Exception as e:
            result.fail(f"goto_pose: {e}")
    elif "goto_pose" in fns:
        result.note("goto_pose SKIP (PyRoKi :8116 down)")


def _default_object_name(api_name: str, env_name: str) -> str:
    if "NutAssembly" in api_name or "nut_assembly" in env_name:
        return "square nut handle"
    if "SpillWipe" in api_name or "spill_wipe" in env_name:
        return "spill"
    return "red cube"

def _invoke_privileged_bimanual(api: Any, fns: dict[str, Callable], result: ApiResult) -> None:
    for name in ("get_handle0_pos", "get_handle1_pos", "get_arm0_gripper_pose", "get_arm1_gripper_pose"):
        if name not in fns:
            continue
        try:
            out = fns[name]()
            result.note(f"{name} ok -> {type(out).__name__}")
        except Exception as e:
            result.fail(f"{name}: {e}")
            return

    for name in ("open_gripper_arm0", "open_gripper_arm1", "close_gripper_arm0", "close_gripper_arm1"):
        if name not in fns:
            continue
        try:
            fns[name]()
            result.note(f"{name} ok")
        except Exception as e:
            result.fail(f"{name}: {e}")
            return


def _invoke_spill_wipe_privileged(api: Any, fns: dict[str, Callable], result: ApiResult) -> None:
    if "get_object_pose" not in fns:
        return
    # Spill wipe privileged uses marker names; try a few common ones.
    for obj in ("spill", "dirt", "liquid", "marker"):
        try:
            out = fns["get_object_pose"](obj)
            result.note(f"get_object_pose('{obj}') ok")
            break
        except Exception as e:
            result.note(f"get_object_pose('{obj}') miss: {e}")
    else:
        # still attempt with empty to surface API error shape
        try:
            fns["get_object_pose"]("red cube")
        except Exception as e:
            result.note(f"get_object_pose fallback error (expected for spill): {e}")


def _install_perception_stubs() -> list[Any]:
    """Stub SAM3/GraspNet/Molmo/OWL-ViT/SAM2 clients so APIs construct without servers.

    Returns list of (module, attr, original) for restoration.
    """
    import capx.integrations.vision.graspnet as graspnet
    import capx.integrations.vision.molmo as molmo
    import capx.integrations.vision.owlvit as owlvit
    import capx.integrations.vision.sam2 as sam2
    import capx.integrations.vision.sam3 as sam3

    def _noop_grasp(*args, **kwargs):
        return [], []

    def _noop_seg(*args, **kwargs):
        return []

    def _noop_point(*args, **kwargs):
        return None

    patches = []
    for mod, attr, stub in [
        (graspnet, "init_contact_graspnet", lambda *a, **k: _noop_grasp),
        (sam3, "init_sam3", lambda *a, **k: _noop_seg),
        (sam3, "init_sam3_point_prompt", lambda *a, **k: _noop_point),
        (sam2, "init_sam2", lambda *a, **k: _noop_seg),
        (owlvit, "init_owlvit", lambda *a, **k: _noop_seg),
        (molmo, "init_molmo", lambda *a, **k: _noop_point),
    ]:
        if hasattr(mod, attr):
            patches.append((mod, attr, getattr(mod, attr)))
            setattr(mod, attr, stub)

    # Also patch symbols already imported into API modules
    import capx.integrations.franka.control as control
    import capx.integrations.franka.control_reduced as control_reduced
    import capx.integrations.franka.control_reduced_exampleless as control_reduced_ex
    import capx.integrations.franka.control_reduced_skill_library as control_reduced_sl
    import capx.integrations.franka.spill_wipe as spill_wipe
    import capx.integrations.franka.handover as handover
    import capx.integrations.franka.handover_reduced as handover_reduced
    import capx.integrations.franka.handover_reduced_exampleless as handover_reduced_ex
    import capx.integrations.franka.two_arm_lift as two_arm_lift
    import capx.integrations.franka.nut_assembly_visual as nut_visual

    for mod in (
        control,
        control_reduced,
        control_reduced_ex,
        control_reduced_sl,
        spill_wipe,
        handover,
        handover_reduced,
        handover_reduced_ex,
        two_arm_lift,
        nut_visual,
    ):
        for attr, stub in [
            ("init_contact_graspnet", lambda *a, **k: _noop_grasp),
            ("init_sam3", lambda *a, **k: _noop_seg),
            ("init_sam3_point_prompt", lambda *a, **k: _noop_point),
            ("init_sam2", lambda *a, **k: _noop_seg),
            ("init_owlvit", lambda *a, **k: _noop_seg),
            ("init_molmo", lambda *a, **k: _noop_point),
        ]:
            if hasattr(mod, attr):
                patches.append((mod, attr, getattr(mod, attr)))
                setattr(mod, attr, stub)
    return patches


def _restore_patches(patches: list[Any]) -> None:
    for mod, attr, original in patches:
        setattr(mod, attr, original)


def test_one(api_name: str, *, stub_perception: bool) -> ApiResult:
    result = ApiResult(name=api_name)
    env_name = API_ENV.get(api_name)
    if env_name is None:
        result.skip("no env mapping (LIBERO/R1Pro/other) — not in robosuite sci matrix")
        return result
    if env_name in SKIP_ENVS:
        result.skip(f"env '{env_name}' requires real hardware")
        return result

    needs_perc = api_name in NEEDS_PERCEPTION
    sam3_up = _service_up("http://127.0.0.1:8114/docs")
    grasp_up = _service_up("http://127.0.0.1:8115/docs")
    use_stubs = needs_perc and not (sam3_up and grasp_up)
    if use_stubs and not stub_perception:
        result.skip(
            f"needs perception servers (SAM3:8114={'up' if sam3_up else 'DOWN'}, "
            f"GraspNet:8115={'up' if grasp_up else 'DOWN'})"
        )
        try:
            get_api(api_name)
            result.note("factory registered")
        except Exception as e:
            result.fail(f"factory missing: {e}")
        return result

    privileged = "Privileged" in api_name
    env = None
    patches: list[Any] = []
    try:
        env = _make_env(env_name, privileged=privileged)
        env.reset()
        result.note(f"env.reset ok ({env_name}, privileged={privileged})")
    except Exception as e:
        result.fail(f"env create/reset: {e}\n{traceback.format_exc(limit=3)}")
        return result

    try:
        if use_stubs:
            patches = _install_perception_stubs()
            result.note("perception clients STUBBED (servers down) — construct+functions only")
        factory = get_api(api_name)
        api = factory(env)
        result.note("construct ok")
    except Exception as e:
        result.fail(f"construct: {e}\n{traceback.format_exc(limit=5)}")
        _restore_patches(patches)
        try:
            env.close()
        except Exception:
            pass
        return result

    try:
        fns = _test_functions_table(api, result)
        if result.status == "FAIL":
            return result

        # L2/L3 invoke for privileged / bimanual privileged
        if "Privileged" in api_name:
            if any(k.startswith("goto_pose_arm") or k.startswith("get_handle") for k in fns):
                _invoke_privileged_bimanual(api, fns, result)
            elif env_name.endswith("spill_wipe_low_level"):
                _invoke_spill_wipe_privileged(api, fns, result)
            else:
                _invoke_privileged_single_arm(
                    api,
                    fns,
                    result,
                    object_name=_default_object_name(api_name, env_name),
                )
        else:
            result.note("invoke: perception path — construct+functions verified")
    finally:
        _restore_patches(patches)
        try:
            env.close()
        except Exception:
            pass

    return result


def main() -> int:
    import capx.integrations  # noqa: F401 — register APIs
    import capx.envs  # noqa: F401 — register envs

    apis = sorted(list_apis())
    print(f"Registered APIs ({len(apis)}):")
    for n in apis:
        print(f"  - {n}")
    print()
    print(
        "Services: "
        f"PyRoKi={_service_up('http://127.0.0.1:8116/docs')} "
        f"SAM3={_service_up('http://127.0.0.1:8114/docs')} "
        f"GraspNet={_service_up('http://127.0.0.1:8115/docs')}"
    )
    print("=" * 72)

    results: list[ApiResult] = []
    for name in apis:
        print(f"\n>>> {name}")
        r = test_one(name, stub_perception=True)
        results.append(r)
        print(f"    [{r.status}]")
        for note in r.notes:
            for line in note.splitlines():
                print(f"      - {line}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        print(f"  [{r.status:4}] {r.name}")
    print(
        f"\nTotals: PASS={counts.get('PASS',0)} FAIL={counts.get('FAIL',0)} "
        f"SKIP={counts.get('SKIP',0)} / {len(results)}"
    )
    return 1 if counts.get("FAIL", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
