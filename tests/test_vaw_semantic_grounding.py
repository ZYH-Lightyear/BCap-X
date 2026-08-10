from __future__ import annotations

import numpy as np
import pytest

from vaw.context_runtime.semantic_grounding import (
    GroundingCandidate,
    parse_candidates,
    parse_choice,
    render_candidate_review,
)


def test_parse_candidates_uses_explicit_norm1000_and_deduplicates() -> None:
    reply = """```json
    [
      {"box":[100,200,400,700],"evidence":"first"},
      {"box":[102,202,399,699],"evidence":"duplicate"},
      {"box":[600,100,900,500],"evidence":"second"}
    ]
    ```"""

    candidates = parse_candidates(
        reply,
        width=160,
        height=120,
        coord_space="norm1000",
    )

    assert len(candidates) == 2
    assert candidates[0].box_xyxy_px == pytest.approx((16.0, 24.0, 64.0, 84.0))
    assert candidates[1].box_xyxy_px == pytest.approx((96.0, 12.0, 144.0, 60.0))
    assert [item.evidence for item in candidates] == ["first", "second"]


def test_parse_candidates_preserves_pixel_coordinates() -> None:
    candidates = parse_candidates(
        '[{"box":[16,24,64,84],"evidence":"pixel"}]',
        width=160,
        height=120,
        coord_space="pixel",
    )

    assert candidates[0].box_xyxy_px == pytest.approx((16.0, 24.0, 64.0, 84.0))


def test_candidate_review_is_fixed_and_choice_can_be_ambiguous() -> None:
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    candidates = [
        GroundingCandidate((16.0, 24.0, 64.0, 84.0), "first"),
        GroundingCandidate((96.0, 12.0, 144.0, 60.0), "second"),
    ]

    review = render_candidate_review(rgb, candidates)

    assert review.shape == (720, 1200, 3)
    assert np.any(review != 255)
    assert parse_choice('{"candidate":2}', count=2) == 1
    assert parse_choice('{"candidate":null}', count=2) is None
    with pytest.raises(ValueError, match="within"):
        parse_choice('{"candidate":3}', count=2)
