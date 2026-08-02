from __future__ import annotations

import numpy as np

from vaw.executor import execute_commit, execute_move
from vaw.state import ActionState
from vaw.types import Candidate, Pose, TOP_DOWN_QUAT_WXYZ


class DirectTcpApi:
    """Fake controller whose public IK target and Cartesian readback coincide."""

    def __init__(self) -> None:
        self.position = np.array([0.30, -0.10, 0.45], dtype=np.float64)
        self.quaternion = TOP_DOWN_QUAT_WXYZ.copy()
        self.joints = np.zeros(7, dtype=np.float64)
        self.pending: tuple[np.ndarray, np.ndarray] | None = None
        self.solve_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def solve_ik(self, position, quat_wxyz):
        position = np.asarray(position, dtype=np.float64).copy()
        quaternion = np.asarray(quat_wxyz, dtype=np.float64).copy()
        self.solve_calls.append((position, quaternion))
        self.pending = (position, quaternion)
        return self.joints.copy()

    def move_to_joints(self, joints) -> None:
        self.joints = np.asarray(joints, dtype=np.float64).copy()
        if self.pending is not None:
            self.position, self.quaternion = self.pending

    def get_observation(self):
        return {
            "robot_cartesian_pos": np.concatenate(
                [self.position, self.quaternion, [1.0]]
            ),
            "robot_joint_pos": np.concatenate([self.joints, [1.0]]),
        }


def test_commit_passes_candidate_tcp_to_solve_ik_unchanged() -> None:
    api = DirectTcpApi()
    state = ActionState(instruction="test")
    candidate = Candidate(
        candidate_id="p1",
        kind="waypoint",
        pose=Pose(
            np.array([0.48, 0.03, 0.22], dtype=np.float64),
            TOP_DOWN_QUAT_WXYZ.copy(),
        ),
    )

    receipt = execute_commit(api, state, candidate, z_approach=0.0)

    np.testing.assert_allclose(api.solve_calls[-1][0], candidate.pose.position)
    np.testing.assert_allclose(api.solve_calls[-1][1], candidate.pose.quat_wxyz)
    assert receipt.pos_error_m == 0.0


def test_move_xyz_offsets_the_observed_tcp_without_frame_conversion() -> None:
    api = DirectTcpApi()
    state = ActionState(instruction="test")
    start = api.position.copy()
    delta = np.array([0.02, -0.01, 0.03], dtype=np.float64)

    receipt = execute_move(api, state, delta)

    expected = start + delta
    np.testing.assert_allclose(api.solve_calls[-1][0], expected)
    np.testing.assert_allclose(receipt.requested.position, expected)
    assert receipt.pos_error_m == 0.0
