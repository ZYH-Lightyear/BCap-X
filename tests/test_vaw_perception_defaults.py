from vaw.context_runtime.perception_defaults import (
    DEFAULT_GROUNDING_MODEL,
    DEFAULT_POINT_MODEL,
)
from vaw.context_runtime.point_grounding import resolve_point_coord_space
from vaw.context_runtime.semantic_grounding import resolve_grounding_coord_space


def test_default_perception_models_are_gemini_flash() -> None:
    assert DEFAULT_GROUNDING_MODEL == "vapi/gemini-3.7-flash"
    assert DEFAULT_POINT_MODEL == "vapi/gemini-3.7-flash"
    assert resolve_grounding_coord_space(DEFAULT_GROUNDING_MODEL) == "norm1000_yxyx"
    assert resolve_point_coord_space(DEFAULT_POINT_MODEL) == "norm1000_yx"
