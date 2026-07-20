"""Pure helpers for blocking joint motions: reference profiles and budgets.

Kept dependency-free so both simulators and unit tests can import them without
pulling in MuJoCo/robosuite.
"""

from __future__ import annotations

import numpy as np


def min_jerk(s: float) -> float:
    """Minimum-jerk position profile ``s(t) = 10t^3 - 15t^4 + 6t^5`` on [0, 1].

    Starts and ends with zero velocity and acceleration, which is what removes
    the "touch the waypoint at full speed" behaviour of a plain target pull.
    """

    t = float(np.clip(s, 0.0, 1.0))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def interp_step_budget(
    start: np.ndarray,
    target: np.ndarray,
    *,
    control_freq: float,
    max_joint_vel: float,
    margin: float = 1.5,
    min_steps: int = 10,
) -> int:
    """Number of control steps for the interpolated reference, scaled by distance.

    Uses the largest single-joint displacement so the slowest joint sets the
    duration; ``margin`` leaves headroom for controller lag.
    """

    dist = float(np.max(np.abs(np.asarray(target, dtype=np.float64) - np.asarray(start, dtype=np.float64))))
    steps = int(np.ceil(dist / max(max_joint_vel, 1e-6) * control_freq * margin))
    return max(min_steps, steps + min_steps)
