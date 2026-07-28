"""Coding Agent multi-turn loop tests.

Covers progressive skill disclosure, contract function binding, scripts/
injection, inner loop repair, multimodal feedback, tolerant parsing, and
token budget accounting.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from robomex.core.coder.action import (
    BlockExecutor,
    SkillEntry,
    build_skill_llm_content,
    skill_script_modules,
)
from robomex.core.coder.agent import CodingAgent
from robomex.core.sandbox import (
    ActionBlockStatus,
    BlockExecutionResult,
    SemanticActionBlock,
)
from robomex.core.token_budget import conservative_chat_prompt_tokens
from robomex.skills import Skill, SkillCategory, SkillLibrary


# ---------------------------------------------------------------------------
#  Minimal concrete CodingAgent for testing
# ---------------------------------------------------------------------------

@dataclass
class SimpleResult:
    succeeded: bool
    value: Any
    loaded_skills: tuple[str, ...]
    turns: list[Any]


@dataclass
class TurnInfo:
    code: str
    execution: BlockExecutionResult


class SimpleCodingAgent(CodingAgent):
    """Thin concrete subclass with no skill gate and whole-library search."""

    def __init__(
        self,
        executor: BlockExecutor,
        policy: Any,
        library: SkillLibrary,
        *,
        objective: str = "produce RESULT",
        skill_ids: tuple[str, ...] = (),
        final_value: Any = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(executor, policy, library, **kwargs)
        self._objective = objective
        self._skill_ids = skill_ids
        self._final_value = final_value
        self._output_dir = output_dir
        self._turns: list[TurnInfo] = []

    def _skill_entries(self) -> list[SkillEntry]:
        entries = []
        for record in self.library.all():
            entries.append(
                SkillEntry(
                    name=record.skill.skill_id,
                    description=record.skill.description,
                    category=record.skill.category.value,
                )
            )
        return entries

    def _initial_user_message(self) -> str:
        return self._objective

    def _on_python_turn(
        self,
        turn_idx: int,
        code: str,
        execution: BlockExecutionResult,
        prev_observation: dict | None,
        turns: list[Any],
    ) -> None:
        info = TurnInfo(code=code, execution=execution)
        self._turns.append(info)
        turns.append(info)

    def _finalize(
        self, *, turns: list[Any], loaded: tuple[str, ...], terminal_raw: str | None
    ) -> SimpleResult:
        value = self._final_value() if callable(self._final_value) else self._final_value
        succeeded = value is not None
        return SimpleResult(
            succeeded=succeeded,
            value=value,
            loaded_skills=loaded,
            turns=self._turns,
        )


# ---------------------------------------------------------------------------
#  Test infrastructure
# ---------------------------------------------------------------------------

class _RecordingPolicy:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = [json.dumps(item) for item in responses]
        self.prompts: list[list[dict]] = []

    def complete(self, prompt: list[dict]) -> str:
        self.prompts.append(copy.deepcopy(prompt))
        if not self.responses:
            return '{"tool":"finish","args":{"claim":"responses exhausted"}}'
        return self.responses.pop(0)


class _PythonExecutor:
    def __init__(self, *, image_path: Path | None = None) -> None:
        self.sandbox_namespace: dict = {"RESULT": None}
        self.image_path = image_path

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(block.code, self.sandbox_namespace, self.sandbox_namespace)  # noqa: S102
            ok = True
        except BaseException:  # noqa: BLE001
            ok = False
            traceback.print_exc(file=stderr)
        info: dict[str, Any] = {}
        if self.image_path is not None:
            info["image_paths"] = [str(self.image_path)]
        return BlockExecutionResult(
            block=block,
            ok=ok,
            status=ActionBlockStatus.SUCCEEDED if ok else ActionBlockStatus.FAILED,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            info=info,
        )


def _library(tmp_path: Path) -> SkillLibrary:
    library = SkillLibrary(tmp_path / "skills")
    library.admit(
        Skill(
            skill_id="motion-planning",
            name="Motion planning",
            description="Plan a collision-aware joint path.",
            category=SkillCategory.MOTION,
            body="## Procedure\n\nCall the planner, inspect its result, and repair failures.",
        )
    )
    return library


def _library_with_contract(tmp_path: Path) -> SkillLibrary:
    """Library with a skill that has a contract.yaml and scripts/."""
    src_dir = tmp_path / "src_skills" / "traj-skill"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "SKILL.md").write_text(
        "---\n"
        "name: Trajectory Skill\n"
        "category: motion\n"
        "description: Build a trajectory.\n"
        "---\n\n"
        "## Procedure\n\nCall build_traj() to get a trajectory.\n"
    )
    scripts_dir = src_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    (scripts_dir / "traj_helpers.py").write_text(
        "def build_traj(target_pos):\n"
        "    return {'waypoints': [list(target_pos)], 'ok': True}\n"
    )
    contract = {
        "skill_id": "traj-skill",
        "functions": [
            {
                "name": "build_traj",
                "entry": "scripts/traj_helpers.py:build_traj",
                "description": "Build a trajectory to a target position.",
            }
        ],
    }
    (src_dir / "contract.yaml").write_text(yaml.dump(contract))
    library = SkillLibrary(tmp_path / "skills")
    library.admit(Skill.from_dir(src_dir), source="test")
    return library


# ---------------------------------------------------------------------------
#  Test 1: Progressive disclosure
# ---------------------------------------------------------------------------

def test_progressive_disclosure(tmp_path: Path) -> None:
    """System prompt has skill names but not body; use_skill loads the body."""
    policy = _RecordingPolicy([
        {"tool": "use_skill", "args": {"name": "motion-planning"}},
        {"tool": "run_python", "args": {"code": "RESULT = 'ok'", "intent": "done"}},
        {"tool": "finish", "args": {"claim": "done"}},
    ])
    executor = _PythonExecutor()
    agent = SimpleCodingAgent(
        executor=executor,
        policy=policy,
        library=_library(tmp_path),
        objective="produce RESULT",
        final_value=lambda: executor.sandbox_namespace["RESULT"],
        max_turns=4,
        max_model_calls=8,
        max_tokens=200_000,
    )

    result = agent.run()

    assert result.succeeded
    assert "motion-planning" in result.loaded_skills
    initial_system = policy.prompts[0][0]["content"]
    assert "motion-planning" in initial_system
    assert "Call the planner" not in initial_system
    after_skill = policy.prompts[1][-1]["content"]
    assert "Call the planner" in after_skill


# ---------------------------------------------------------------------------
#  Test 2: Contract function binding
# ---------------------------------------------------------------------------

def test_contract_function_binding(tmp_path: Path) -> None:
    """After loading a skill with contract.yaml, its functions are callable."""
    policy = _RecordingPolicy([
        {"tool": "use_skill", "args": {"name": "traj-skill"}},
        {
            "tool": "run_python",
            "args": {
                "code": "result = build_traj([0.5, 0.0, 0.3])\nRESULT = result",
                "intent": "call contracted function",
            },
        },
        {"tool": "finish", "args": {"claim": "done"}},
    ])
    executor = _PythonExecutor()
    agent = SimpleCodingAgent(
        executor=executor,
        policy=policy,
        library=_library_with_contract(tmp_path),
        objective="call build_traj",
        final_value=lambda: executor.sandbox_namespace["RESULT"],
        max_turns=4,
        max_model_calls=8,
        max_tokens=200_000,
    )

    result = agent.run()

    assert result.succeeded
    assert result.value["ok"] is True
    assert result.value["waypoints"] == [[0.5, 0.0, 0.3]]
    after_skill = policy.prompts[1][-1]["content"]
    assert "build_traj" in after_skill
    assert "build_traj(" in after_skill


# ---------------------------------------------------------------------------
#  Test 3: scripts/ injection
# ---------------------------------------------------------------------------

def test_scripts_injection(tmp_path: Path) -> None:
    """After loading a skill, import <module> works without importlib boilerplate."""
    policy = _RecordingPolicy([
        {"tool": "use_skill", "args": {"name": "traj-skill"}},
        {
            "tool": "run_python",
            "args": {
                "code": "import traj_helpers\nRESULT = traj_helpers.build_traj([1,2,3])",
                "intent": "import test",
            },
        },
        {"tool": "finish", "args": {"claim": "done"}},
    ])
    executor = _PythonExecutor()
    agent = SimpleCodingAgent(
        executor=executor,
        policy=policy,
        library=_library_with_contract(tmp_path),
        objective="import the module",
        final_value=lambda: executor.sandbox_namespace["RESULT"],
        max_turns=4,
        max_model_calls=8,
        max_tokens=200_000,
    )

    result = agent.run()

    assert result.succeeded
    assert result.value["ok"] is True
    load_msg = policy.prompts[1][-1]["content"]
    assert "import traj_helpers" in load_msg or "traj_helpers" in load_msg


# ---------------------------------------------------------------------------
#  Test 4: Inner loop repair
# ---------------------------------------------------------------------------

def test_inner_loop_repair(tmp_path: Path) -> None:
    """Failed code stderr feeds back into next prompt; repair succeeds."""
    policy = _RecordingPolicy([
        {
            "tool": "run_python",
            "args": {"code": "raise ValueError('bad waypoint')", "intent": "try"},
        },
        {
            "tool": "run_python",
            "args": {
                "code": "RESULT = {'kind': 'wait', 'wait_seconds': 0.1}",
                "intent": "repair",
            },
        },
        {"tool": "finish", "args": {"claim": "done"}},
    ])
    executor = _PythonExecutor()
    agent = SimpleCodingAgent(
        executor=executor,
        policy=policy,
        library=_library(tmp_path),
        objective="repair and produce RESULT",
        final_value=lambda: executor.sandbox_namespace["RESULT"],
        max_turns=4,
        max_model_calls=8,
        max_tokens=200_000,
    )

    result = agent.run()

    assert result.succeeded
    assert "bad waypoint" in result.turns[0].execution.stderr
    assert result.value["kind"] == "wait"
    feedback_after_failure = policy.prompts[1][-1]["content"]
    assert "bad waypoint" in feedback_after_failure


# ---------------------------------------------------------------------------
#  Test 5: Multimodal feedback + token budget
# ---------------------------------------------------------------------------

def test_multimodal_feedback_and_token_budget(tmp_path: Path) -> None:
    """Generated images are fed back as image_url parts; inline bytes don't
    exhaust the model budget."""
    image_path = tmp_path / "trajectory.png"
    Image.new("RGB", (16, 16), (20, 80, 160)).save(image_path)
    policy = _RecordingPolicy([
        {
            "tool": "run_python",
            "args": {
                "code": "RESULT = {'verdict': 'inspected'}",
                "intent": "render and inspect",
            },
        },
        {"tool": "finish", "args": {"claim": "done"}},
    ])
    executor = _PythonExecutor(image_path=image_path)
    agent = SimpleCodingAgent(
        executor=executor,
        policy=policy,
        library=_library(tmp_path),
        objective="inspect a render",
        final_value=lambda: executor.sandbox_namespace["RESULT"],
        max_turns=2,
        max_model_calls=4,
        max_tokens=200_000,
    )

    result = agent.run()

    assert result.succeeded
    feedback = policy.prompts[1][-1]["content"]
    assert isinstance(feedback, list)
    image_parts = [p for p in feedback if p.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    prompt_with_image = [
        {"role": "user", "content": [
            {"type": "text", "text": "inspect"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 2_000_000}},
        ]}
    ]
    assert conservative_chat_prompt_tokens(prompt_with_image) < 20_000


# ---------------------------------------------------------------------------
#  Test 6: Tolerant parsing
# ---------------------------------------------------------------------------

def test_tolerant_parsing_bare_code_block(tmp_path: Path) -> None:
    """A bare ```python block is treated as run_python, not a protocol error.
    Also: 4 non-consecutive parse failures do NOT kill the session."""
    responses = [
        '```python\nRESULT = "from bare block"\n```',
        {"tool": "finish", "args": {"claim": "done"}},
    ]
    policy = _RecordingPolicy([])
    policy.responses = [
        responses[0],
        json.dumps(responses[1]),
    ]
    executor = _PythonExecutor()
    agent = SimpleCodingAgent(
        executor=executor,
        policy=policy,
        library=_library(tmp_path),
        objective="bare block test",
        final_value=lambda: executor.sandbox_namespace["RESULT"],
        max_turns=4,
        max_model_calls=8,
        max_tokens=200_000,
    )

    result = agent.run()

    assert result.succeeded
    assert result.value == "from bare block"


def test_protocol_errors_reset_on_success(tmp_path: Path) -> None:
    """Non-consecutive parse failures do not accumulate to kill the session."""
    responses = [
        "garbage no json 1",
        '{"tool":"run_python","args":{"code":"x=1","intent":"ok"}}',
        "garbage no json 2",
        '{"tool":"run_python","args":{"code":"x=2","intent":"ok"}}',
        "garbage no json 3",
        '{"tool":"run_python","args":{"code":"RESULT=42","intent":"final"}}',
        '{"tool":"finish","args":{"claim":"done"}}',
    ]
    policy = _RecordingPolicy([])
    policy.responses = list(responses)
    executor = _PythonExecutor()
    agent = SimpleCodingAgent(
        executor=executor,
        policy=policy,
        library=_library(tmp_path),
        objective="survive errors",
        final_value=lambda: executor.sandbox_namespace["RESULT"],
        max_turns=6,
        max_model_calls=16,
        max_tokens=400_000,
        max_protocol_errors=2,
    )

    result = agent.run()

    assert result.succeeded
    assert result.value == 42
