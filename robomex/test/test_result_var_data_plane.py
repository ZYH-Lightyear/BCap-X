"""M1.7: mechanical data plane for weak-model Agent Swarm communication.

finish result_var materializes sandbox variables; INPUTS is seeded so agents
reference rather than retype upstream numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from robomex.agents.subagents import CodingAgentSubAgent, SubAgentRequest
from robomex.core.coder import ScriptedCodePolicy
from robomex.core.sandbox import ActionBlockStatus, BlockExecutionResult, SemanticActionBlock
from robomex.skills import Skill


class _FakeRecord:
    def __init__(self, skill: Skill) -> None:
        self.skill = skill
        self.skill_id = skill.skill_id


class _FakeLibrary:
    def __init__(self, skills: list[Skill]) -> None:
        self._by_id = {s.skill_id: _FakeRecord(s) for s in skills}

    def all(self) -> list[_FakeRecord]:
        return list(self._by_id.values())

    def get(self, skill_id: str) -> _FakeRecord:
        return self._by_id[skill_id]


class _NamespaceExecutor:
    """Test double that exposes ``sandbox_namespace`` for result_var materialization."""

    def __init__(self) -> None:
        self.sandbox_namespace: dict = {}
        self.blocks: list[SemanticActionBlock] = []

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        self.blocks.append(block)
        return BlockExecutionResult(
            block=block,
            ok=True,
            status=ActionBlockStatus.SUCCEEDED,
            stdout="",
            stderr="",
            reward=0.0,
            terminated=False,
            truncated=False,
            observation={},
            info={"sandbox_rc": 0},
        )


def _lib() -> _FakeLibrary:
    return _FakeLibrary(
        [
            Skill.from_markdown(
                "---\nname: s\ncategory: perception\ndescription: d\n---\n\nBody.",
                skill_id="s",
            )
        ]
    )


def _finish_var(name: str = "NODE_RESULT", claim: str = "done") -> str:
    return json.dumps(
        {"tool": "finish", "args": {"claim": claim, "result_var": name}}
    )


def _agent(policy: ScriptedCodePolicy, executor: _NamespaceExecutor | None = None) -> CodingAgentSubAgent:
    return CodingAgentSubAgent(
        executor=executor or _NamespaceExecutor(),
        policy=policy,
        library=_lib(),
        max_turns=4,
        output_ports=(("plan", "test.value.v1"),),
        task_kind="plan",
    )


def test_result_var_materializes_sandbox_dict(tmp_path: Path) -> None:
    ex = _NamespaceExecutor()
    policy = ScriptedCodePolicy([_finish_var()])
    agent = _agent(policy, ex)

    # Seed runs in _setup; inject NODE_RESULT before the finish turn by
    # monkey-patching run so the namespace is ready after setup.
    original_run = agent.run

    def run_with_payload(request: SubAgentRequest):
        agent._request = request
        agent._turn_records = []
        agent._materialized_result = None
        # Call setup via the real path by wrapping complete — simpler: pre-seed
        # after constructing and let _setup populate INPUTS, then set NODE_RESULT
        # on the executor namespace before finish is processed.
        from robomex.core.coder.agent import CodingAgent

        def patched_setup(prompt):
            CodingAgentSubAgent._setup(agent, prompt)
            ex.sandbox_namespace["NODE_RESULT"] = {
                "outputs": {
                    "plan": {
                        "payload": {
                            "position": [0.4, 0.0, 0.05],
                            "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
                        },
                        "confidence": 0.9,
                        "artifacts": {},
                    }
                },
                "recommended_next": "execute",
            }

        agent._setup = patched_setup  # type: ignore[method-assign]
        return CodingAgent.run(agent)

    agent.run = run_with_payload  # type: ignore[method-assign]
    result = agent.run(
        SubAgentRequest(
            task="plan",
            artifacts_dir=str(tmp_path),
            inputs={"geometry": {"payload": {"center": [0.4, 0.0, 0.05]}}},
        )
    )

    assert result.ok
    assert result.result["outputs"]["plan"]["payload"]["quaternion_wxyz"] == [
        0.0,
        1.0,
        0.0,
        0.0,
    ]
    assert result.claim == "done"


def test_result_var_missing_is_repairable(tmp_path: Path) -> None:
    ex = _NamespaceExecutor()
    policy = ScriptedCodePolicy(
        [
            _finish_var("MISSING"),
            _finish_var("NODE_RESULT"),
        ]
    )
    agent = _agent(policy, ex)

    def patched_setup(prompt):
        CodingAgentSubAgent._setup(agent, prompt)
        # Only define NODE_RESULT on the second attempt path: define it after
        # first rejection by setting it immediately (finish #1 fails, #2 succeeds).
        ex.sandbox_namespace["NODE_RESULT"] = {
            "outputs": {"plan": {"payload": {"ok": True}, "confidence": 0.9, "artifacts": {}}},
            "recommended_next": "done",
        }

    agent._setup = patched_setup  # type: ignore[method-assign]
    from robomex.core.coder.agent import CodingAgent

    def run_request(request: SubAgentRequest):
        agent._request = request
        agent._turn_records = []
        agent._materialized_result = None
        return CodingAgent.run(agent)

    agent.run = run_request  # type: ignore[method-assign]
    result = agent.run(SubAgentRequest(task="plan", artifacts_dir=str(tmp_path)))

    assert result.ok
    assert result.result["outputs"]["plan"]["payload"]["ok"] is True


def test_result_var_rejects_inline_large_array(tmp_path: Path) -> None:
    ex = _NamespaceExecutor()
    policy = ScriptedCodePolicy(
        [
            _finish_var(),
            json.dumps(
                {
                    "tool": "finish",
                    "args": {
                        "claim": "fixed",
                        "result": {
                            "outputs": {
                                "plan": {
                                    "payload": {"ok": True},
                                    "confidence": 0.9,
                                    "artifacts": {},
                                }
                            }
                        },
                    },
                }
            ),
        ]
    )
    agent = _agent(policy, ex)

    def patched_setup(prompt):
        CodingAgentSubAgent._setup(agent, prompt)
        ex.sandbox_namespace["NODE_RESULT"] = {
            "outputs": {
                "plan": {
                    "payload": {"points": list(range(64))},
                    "confidence": 0.9,
                    "artifacts": {},
                }
            }
        }

    agent._setup = patched_setup  # type: ignore[method-assign]
    from robomex.core.coder.agent import CodingAgent

    def run_request(request: SubAgentRequest):
        agent._request = request
        agent._turn_records = []
        agent._materialized_result = None
        return CodingAgent.run(agent)

    agent.run = run_request  # type: ignore[method-assign]
    result = agent.run(SubAgentRequest(task="plan", artifacts_dir=str(tmp_path)))

    assert result.ok
    assert result.result["outputs"]["plan"]["payload"]["ok"] is True


def test_inputs_seeded_into_sandbox_namespace(tmp_path: Path) -> None:
    ex = _NamespaceExecutor()
    policy = ScriptedCodePolicy(
        [
            json.dumps(
                {
                    "tool": "finish",
                    "args": {
                        "claim": "ok",
                        "result": {
                            "outputs": {
                                "plan": {
                                    "payload": {"ok": True},
                                    "confidence": 1.0,
                                    "artifacts": {},
                                }
                            }
                        },
                    },
                }
            )
        ]
    )
    agent = _agent(policy, ex)
    inputs = {
        "grasp_affordance": {
            "payload": {"quaternion_wxyz": [0.0, 1.0, 0.0, 0.0]},
            "schema": "test.value.v1",
        }
    }
    result = agent.run(
        SubAgentRequest(task="plan", artifacts_dir=str(tmp_path), inputs=inputs)
    )

    assert result.ok
    assert "INPUTS" in ex.sandbox_namespace
    assert (
        ex.sandbox_namespace["INPUTS"]["grasp_affordance"]["payload"]["quaternion_wxyz"]
        == [0.0, 1.0, 0.0, 0.0]
    )


def test_initial_message_mentions_inputs_and_result_var(tmp_path: Path) -> None:
    from robomex.core.coder import ScriptedCodePolicy as _  # noqa: F401

    class RecordingPolicy(ScriptedCodePolicy):
        def __init__(self) -> None:
            super().__init__(
                [
                    json.dumps(
                        {
                            "tool": "finish",
                            "args": {
                                "claim": "ok",
                                "result": {
                                    "outputs": {
                                        "plan": {
                                            "payload": {"ok": True},
                                            "confidence": 1.0,
                                            "artifacts": {},
                                        }
                                    }
                                },
                            },
                        }
                    )
                ]
            )
            self.prompts: list = []

        def complete(self, prompt):
            self.prompts.append(prompt)
            return super().complete(prompt)

    policy = RecordingPolicy()
    agent = _agent(policy)
    agent.run(SubAgentRequest(task="plan", artifacts_dir=str(tmp_path)))
    text = json.dumps(policy.prompts)
    assert "INPUTS" in text
    assert "result_var" in text
