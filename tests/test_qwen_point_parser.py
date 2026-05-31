"""Offline unit tests for the Qwen-VLM pointing parser.

Validates that :func:`_parse_qwen_json_points` / :func:`_parse_points_any_format`
correctly handle the most common Qwen-VL response shapes without needing a live
inference server. Run with:

    uv run --no-sync --active python tests/test_qwen_point_parser.py
"""

from __future__ import annotations

import sys


def run() -> int:
    from capx.integrations.vision.qwen_vlm_point import (
        QWEN_NORM_SCALE,
        _extract_json_text,
        _parse_points_any_format,
        _parse_qwen_json_points,
    )

    failures: list[str] = []

    def check(label: str, got, want) -> None:
        if got == want:
            print(f"  [ OK ] {label}")
        else:
            print(f"  [FAIL] {label}\n         got:  {got!r}\n         want: {want!r}")
            failures.append(label)

    def approx_check(label: str, got, want_pts, want_scale) -> None:
        pts, scale = got
        ok_scale = scale == want_scale
        ok_pts = (
            len(pts) == len(want_pts)
            and all(abs(a - b) < 1e-6 for (xa, ya), (xb, yb) in zip(pts, want_pts) for a, b in ((xa, xb), (ya, yb)))
        )
        if ok_scale and ok_pts:
            print(f"  [ OK ] {label}")
        else:
            print(
                f"  [FAIL] {label}\n         got:  pts={pts} scale={scale}\n"
                f"         want: pts={want_pts} scale={want_scale}"
            )
            failures.append(label)

    print("=== _extract_json_text ===")
    check(
        "strips ```json fence",
        _extract_json_text('```json\n[{"point_2d": [10, 20]}]\n```'),
        '[{"point_2d": [10, 20]}]',
    )
    check(
        "strips plain ``` fence",
        _extract_json_text('```\n[{"point_2d": [1,2]}]\n```'),
        '[{"point_2d": [1,2]}]',
    )
    check(
        "extracts inline array from prose",
        _extract_json_text(
            'Sure! Here is the location: [{"point_2d": [500, 600], "label": "x"}] hope it helps.'
        ),
        '[{"point_2d": [500, 600], "label": "x"}]',
    )

    print("\n=== _parse_qwen_json_points ===")
    check(
        "single point_2d in array",
        _parse_qwen_json_points('[{"point_2d": [550, 320], "label": "soup"}]'),
        [(550.0, 320.0)],
    )
    check(
        "multiple point_2d entries",
        _parse_qwen_json_points(
            '[{"point_2d": [100, 200], "label": "a"}, {"point_2d": [800, 900], "label": "b"}]'
        ),
        [(100.0, 200.0), (800.0, 900.0)],
    )
    check(
        "bbox_2d uses centre",
        _parse_qwen_json_points('[{"bbox_2d": [100, 200, 300, 400], "label": "x"}]'),
        [(200.0, 300.0)],
    )
    check(
        "fenced ```json``` output",
        _parse_qwen_json_points('```json\n[{"point_2d": [42, 67]}]\n```'),
        [(42.0, 67.0)],
    )
    check(
        "single dict (not array)",
        _parse_qwen_json_points('{"point_2d": [11, 22], "label": "x"}'),
        [(11.0, 22.0)],
    )
    check(
        "preamble + JSON array",
        _parse_qwen_json_points(
            'I can see the object. [{"point_2d": [123, 456], "label": "obj"}]'
        ),
        [(123.0, 456.0)],
    )
    check(
        "empty array means not found",
        _parse_qwen_json_points("[]"),
        [],
    )
    check(
        "garbage returns empty",
        _parse_qwen_json_points("I cannot see the object."),
        [],
    )
    check(
        "truncated trailing tokens",
        _parse_qwen_json_points('[{"point_2d": [10, 20], "label": "x"}] some trailing noise'),
        [(10.0, 20.0)],
    )

    print("\n=== _parse_points_any_format dispatch ===")
    approx_check(
        "qwen JSON -> scale 1000",
        _parse_points_any_format('[{"point_2d": [550, 320]}]'),
        [(550.0, 320.0)],
        QWEN_NORM_SCALE,
    )
    approx_check(
        "molmo fallback -> scale 100",
        _parse_points_any_format('<point x="42.3" y="67.1">obj</point>'),
        [(42.3, 67.1)],
        100.0,
    )
    approx_check(
        "empty -> empty points",
        _parse_points_any_format(""),
        [],
        100.0,
    )

    print("\n=== Pixel conversion (sanity) ===")
    from PIL import Image
    from capx.integrations.vision.qwen_vlm_point import convert_to_pixel_coordinates

    img = Image.new("RGB", (640, 480))  # noqa: E501
    pts = [(500.0, 250.0)]  # 0-1000 scale
    px = convert_to_pixel_coordinates(pts, img, QWEN_NORM_SCALE)
    # 500/1000 * 640 = 320; 250/1000 * 480 = 120
    check("qwen 0-1000 -> pixel", px, [(320, 120)])

    print("\n=== Summary ===")
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("All parser tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
