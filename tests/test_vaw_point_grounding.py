from __future__ import annotations

import pytest

from vaw.context_runtime.point_grounding import (
    parse_point_reply,
    point_prompt,
    resolve_point_coord_space,
)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("vapi/gemini-3.7-flash", "norm1000_yx"),
        ("vapi/qwen3.5-27b", "norm1000_xy"),
        ("vapi/gpt-5.5", "pixel_xy"),
    ],
)
def test_resolve_point_coord_space(model: str, expected: str) -> None:
    assert resolve_point_coord_space(model) == expected


def test_parse_gemini_normalized_yx_point() -> None:
    point = parse_point_reply(
        '```json\n{"point":[400,750]}\n```',
        width=800,
        height=512,
        coord_space="norm1000_yx",
    )

    assert point == pytest.approx((600.0, 204.8))


def test_gemini_prompt_declares_native_order() -> None:
    prompt = point_prompt("basket center", width=800, height=512, coord_space="norm1000_yx")

    assert "[y,x]" in prompt
    assert '{"point":[y,x]}' in prompt
