"""Composable Act prompts derived from the enforced runtime contract."""

from robomex.prompts.authoring import build_universal_act_prompt


BASE_ACT_SYSTEM_PROMPT = build_universal_act_prompt()

LIBERO_ACT_SYSTEM_PROMPT = build_universal_act_prompt(
    environment=(
        "Control a Franka arm in LIBERO. Ground every physical decision in the current "
        "observation. query_vlm is for categorical visual judgment, not coordinates; "
        "use dedicated perception APIs for boxes, masks, points, and 3D geometry."
    )
)


def render_libero_act_system_prompt(api_docs: str) -> str:
    """Render a LIBERO prompt from the same contract used by every Act node."""

    return build_universal_act_prompt(
        environment=(
            "Control a Franka arm in LIBERO. Ground every physical decision in the current "
            "observation. query_vlm is for categorical visual judgment, not coordinates; "
            "use dedicated perception APIs for boxes, masks, points, and 3D geometry."
        ),
        api_docs=api_docs,
    )
