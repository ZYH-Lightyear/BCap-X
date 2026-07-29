"""Inverse kinematics for the workspace, against the link the solver actually owns.

Preview and commit both need "which joint configuration puts the fingers here",
and both must ask the same question the same way, or a preview means nothing.

Why this bypasses ``FrankaLiberoApiReduced.solve_ik``: that convenience wrapper
distorts the request in two ways it does not report back, and both are fatal for
grasping at table height.

1. It clips its ``position`` into a fixed box (``z >= 0.005``, ``x <= 0.75``).
   Its ``position`` is a "TCP" sitting 8.86 cm behind the fingertips, so putting
   the fingers on an object 5 cm above the table needs ``z ~= -0.04`` — clipped,
   silently, to a request 4.5 cm too high. Measured: the fingers then close no
   lower than z = 0.094 while the alphabet soup can tops out at z = 0.081, i.e.
   table-height top-down grasps are not expressible through that argument.
2. When the requested orientation does not solve, it silently substitutes a
   canned one (top-down / 45-tilt / side-approach). A grasp aligned to an
   object's short axis that comes back as generic top-down is not the pose the
   agent chose, and every candidate score computed for it is void.

Talking to the IK service directly costs the orientation-fallback flag, which we
only ever used to detect distortion (1)/(2) in the first place, and buys an
undistorted request: we compute the panda_hand target ourselves, from the one
convention the workspace uses everywhere (see ``vaw.geometry``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vaw.geometry import (
    CONTACT_TO_HAND_M,
    CONTACT_TO_IK_TARGET_M,
    shift_along_approach,
)
from vaw.types import Pose

#: The service warm-starts from the previous configuration and is iterated until
#: the solution stops moving, matching ``capx...common.solve_ik_with_convergence``.
MAX_ITERS = 5
CONVERGED_ATOL = 1e-3


def solve_arm_ik(api: Any, pose: Pose) -> np.ndarray:
    """Arm joints (7,) that put the *fingers* at ``pose``.

    Raises whatever the solver raises; callers turn that into a receipt or an
    infeasible preview rather than letting it escape.
    """
    solve = getattr(api, "ik_solve_fn", None)
    if solve is None:
        # No service handle (offline fakes): go through the wrapper, accepting
        # its distortions, so tests and smoke runs still exercise the plumbing.
        target = shift_along_approach(pose.position, pose.quat_wxyz, CONTACT_TO_IK_TARGET_M)
        return np.asarray(api.solve_ik(target, pose.quat_wxyz), dtype=np.float64).reshape(-1)[:7]

    hand = shift_along_approach(pose.position, pose.quat_wxyz, CONTACT_TO_HAND_M)
    target = np.concatenate([np.asarray(pose.quat_wxyz, dtype=np.float64).reshape(4), hand])

    cfg = getattr(api, "cfg", None)
    for _ in range(MAX_ITERS):
        nxt = np.asarray(solve(target_pose_wxyz_xyz=target, prev_cfg=cfg), dtype=np.float64)
        converged = cfg is not None and nxt.shape == cfg.shape and np.allclose(
            nxt, cfg, atol=CONVERGED_ATOL
        )
        cfg = nxt
        if converged:
            break

    # Keep the api's warm start in sync: successive solutions of a redundant arm
    # only stay in the same null-space branch if they chain, and anything else
    # still calling api.solve_ik would otherwise warm-start from a stale pose.
    api.cfg = cfg
    return cfg.reshape(-1)[:7]
