from __future__ import annotations

import numpy as np
import pytest

from capx_skill_rl.scripts.scripted_episode import (
    _obb_top_z,
    _point_near_bbox,
    _validate_grounding,
)


def test_point_bbox_consistency_allows_small_detector_disagreement() -> None:
    assert _point_near_bbox([20.0, 8.0], [10.0, 10.0, 30.0, 40.0], tolerance=2.0)
    assert not _point_near_bbox(
        [20.0, 7.9],
        [10.0, 10.0, 30.0, 40.0],
        tolerance=2.0,
    )


def test_obb_top_uses_oriented_corners() -> None:
    top = _obb_top_z(
        {
            "center": [0.4, 0.0, 0.1],
            "extent": [0.2, 0.4, 0.6],
            "quaternion": [0.0, 0.0, 0.0, 1.0],
        }
    )
    assert top == pytest.approx(0.4)


def test_grounding_rejects_grasp_far_from_target() -> None:
    with pytest.raises(RuntimeError, match="more than 12 cm"):
        _validate_grounding(
            np.array([0.4, 0.0, 0.05]),
            np.array([0.6, 0.0, 0.05]),
            np.array([0.5, 0.2, 0.05]),
        )
