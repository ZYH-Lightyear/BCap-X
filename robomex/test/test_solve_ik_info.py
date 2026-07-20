"""M1.7: solve_ik fallback visibility and return_info contract."""

from __future__ import annotations

import inspect

from capx.integrations.franka.libero_reduced import FrankaLiberoApiReduced


def test_solve_ik_docstring_uses_topdown_example() -> None:
    doc = FrankaLiberoApiReduced.solve_ik.__doc__ or ""
    assert "[0.0, 1.0, 0.0, 0.0]" in doc
    assert "top-down" in doc
    assert "identity, wxyz" not in doc


def test_solve_ik_accepts_return_info_kwarg() -> None:
    sig = inspect.signature(FrankaLiberoApiReduced.solve_ik)
    assert "return_info" in sig.parameters
    assert sig.parameters["return_info"].default is False
