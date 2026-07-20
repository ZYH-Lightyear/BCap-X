"""Pure motion-profile helpers used by LIBERO blocking joint control."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_MODULE_PATH = (
    Path(__file__).parents[2] / "capx" / "envs" / "simulators" / "motion_profiles.py"
)
_SPEC = importlib.util.spec_from_file_location("motion_profiles_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOTION_PROFILES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOTION_PROFILES)


def test_min_jerk_has_expected_endpoints_and_is_monotonic() -> None:
    samples = [_MOTION_PROFILES.min_jerk(x) for x in np.linspace(0.0, 1.0, 101)]

    assert samples[0] == pytest.approx(0.0)
    assert samples[-1] == pytest.approx(1.0)
    assert all(a <= b for a, b in zip(samples, samples[1:]))


def test_interp_step_budget_scales_with_largest_joint_displacement() -> None:
    start = np.zeros(7)
    short = np.full(7, 0.1)
    long = np.full(7, 0.5)

    short_budget = _MOTION_PROFILES.interp_step_budget(
        start, short, control_freq=20.0, max_joint_vel=0.5
    )
    long_budget = _MOTION_PROFILES.interp_step_budget(
        start, long, control_freq=20.0, max_joint_vel=0.5
    )

    assert short_budget >= 10
    assert long_budget > short_budget
