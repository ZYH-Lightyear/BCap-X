from __future__ import annotations

from robomex.agents.planner import ReactivePlanner, ScriptedPlannerPolicy, TwoLevelAgent, parse_next_subgoal
from robomex.core.coder import AgentTrace


class _FinishingInnerAgent:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def run(self, task: str, observation_summary: str = "", **kwargs) -> AgentTrace:
        self.tasks.append(task)
        return AgentTrace(
            task=task,
            loaded_skill_ids=(),
            turns=(),
            success=True,
            metadata={"act_status": "finished"},
        )


class _EmptyPlanningLibrary:
    def task_skills(self) -> list:
        return []


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


def test_parse_skill_colon_text_as_plain_subgoal_not_done() -> None:
    sg = parse_next_subgoal("pick_object: the akita black bowl located between the plate and the ramekin")

    assert sg is not None
    assert sg.goal == "pick_object: the akita black bowl located between the plate and the ramekin"


def test_parse_nonempty_text_as_subgoal_not_done() -> None:
    sg = parse_next_subgoal("Pick up the akita black bowl.")

    assert sg is not None
    assert sg.goal == "Pick up the akita black bowl."


def test_two_level_episode_success_requires_explicit_planner_done() -> None:
    inner = _FinishingInnerAgent()
    planner = ReactivePlanner(
        _EmptyPlanningLibrary(),
        ScriptedPlannerPolicy([
            "Goal: Pick up the bowl.\nPostcondition: The bowl is held.",
            "",
        ])
    )
    execution = TwoLevelAgent(planner, inner, max_subgoals=3).run("Pick up the bowl")

    assert inner.tasks == ["Pick up the bowl."]
    assert len(execution.results) == 1
    assert execution.results[0].success is True
    assert execution.success is False


def test_two_level_episode_success_when_planner_returns_done() -> None:
    inner = _FinishingInnerAgent()
    planner = ReactivePlanner(
        _EmptyPlanningLibrary(),
        ScriptedPlannerPolicy([
            "Goal: Pick up the bowl.\nPostcondition: The bowl is held.",
            "DONE",
        ])
    )
    execution = TwoLevelAgent(planner, inner, max_subgoals=3).run("Pick up the bowl")

    assert len(execution.results) == 1
    assert execution.results[0].success is True
    assert execution.success is True
