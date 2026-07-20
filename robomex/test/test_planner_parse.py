from __future__ import annotations

from robomex.agents.planner import _SYSTEM_PROMPT, parse_next_subgoal


def test_planner_system_prompt_forbids_visual_rewrite() -> None:
    assert "thin reactive" in _SYSTEM_PROMPT
    assert "not to rewrite" in _SYSTEM_PROMPT
    assert "no color" in _SYSTEM_PROMPT
    assert "Agent Swarm" in _SYSTEM_PROMPT
    assert "visually expand" in _SYSTEM_PROMPT
    assert "Visual disambiguation belongs to the downstream Agent Swarm" in _SYSTEM_PROMPT


def test_parse_done_only_when_exact_done() -> None:
    assert parse_next_subgoal("DONE") is None
    assert parse_next_subgoal("  done  ") is None


def test_parse_markdown_two_field_subgoal() -> None:
    sg = parse_next_subgoal(
        "Goal: Pick up the akita black bowl between the plate and the ramekin.\n"
        "Postcondition: The akita black bowl is visibly held by the gripper."
    )

    assert sg is not None
    assert sg.goal == "Pick up the akita black bowl between the plate and the ramekin."
    assert sg.postcondition == "The akita black bowl is visibly held by the gripper."


def test_parse_markdown_subgoal_after_planner_preface() -> None:
    sg = parse_next_subgoal(
        "The prior attempt failed and the gripper is empty.\n\n"
        "Goal: Pick up the akita black bowl next to the ramekin.\n"
        "Postcondition: The akita black bowl is held by the gripper."
    )

    assert sg is not None
    assert sg.goal == "Pick up the akita black bowl next to the ramekin."
    assert sg.postcondition == "The akita black bowl is held by the gripper."


def test_parse_skill_colon_text_is_rejected() -> None:
    sg = parse_next_subgoal("pick_object: the akita black bowl located between the plate and the ramekin")

    assert sg is None


def test_parse_unstructured_text_is_rejected() -> None:
    sg = parse_next_subgoal("Pick up the akita black bowl.")

    assert sg is None
