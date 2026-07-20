"""Artifact path convention + finish-gate repairability tests (M1.6).

The 20260719_173232 live run lost 4/5 subgoals to path/port convention
mistakes that were only detected after the agent finished. These tests pin
the two-part fix: one deterministic path rule shared by the gate and the
store, and validation inside the agent's own turn budget.
"""

from __future__ import annotations

import json
from pathlib import Path

from robomex.agents.subagents import CodingAgentSubAgent, SubAgentRequest
from robomex.authoring.artifacts import PortSpec, outputs_from_finish
from robomex.core.artifact_paths import (
    collect_output_file_refs,
    finish_artifact_path_errors,
    resolve_artifact_ref_path,
)
from robomex.core.coder import ScriptedCodePolicy
from robomex.core.sandbox import ActionBlockStatus, BlockExecutionResult, SemanticActionBlock
from robomex.skills import Skill


# ---- path rule -------------------------------------------------------------


def test_relative_ref_resolves_against_artifacts_dir_not_cwd(tmp_path: Path, monkeypatch) -> None:
    cwd = tmp_path / "repo"
    art = tmp_path / "node_artifacts"
    (cwd / "artifacts").mkdir(parents=True)
    art.mkdir()
    # The trap from the live run: the file exists relative to the CWD.
    (cwd / "artifacts" / "trajectory.json").write_text("{}")
    monkeypatch.chdir(cwd)

    resolved = resolve_artifact_ref_path("artifacts/trajectory.json", art)

    assert resolved == (art / "artifacts" / "trajectory.json").resolve()


def test_absolute_ref_is_taken_as_is(tmp_path: Path) -> None:
    target = tmp_path / "file.npy"
    assert resolve_artifact_ref_path(str(target), tmp_path / "elsewhere") == target.resolve()


def test_outputs_from_finish_ignores_cwd_matches(tmp_path: Path, monkeypatch) -> None:
    cwd = tmp_path / "repo"
    art = tmp_path / "node_artifacts"
    (cwd / "artifacts").mkdir(parents=True)
    art.mkdir()
    (cwd / "artifacts" / "t.json").write_text("{}")
    (art / "artifacts").mkdir()
    (art / "artifacts" / "t.json").write_text("{}")
    monkeypatch.chdir(cwd)

    outputs = outputs_from_finish(
        {"outputs": {"plan": {"payload": {"ok": True}, "artifacts": {"t": "artifacts/t.json"}}}},
        producer="plan",
        ports=(PortSpec("plan", "test.value.v1"),),
        observation_epoch=0,
        artifact_dir=art,
    )

    assert outputs[0].refs[0].path == str((art / "artifacts" / "t.json").resolve())


# ---- finish-gate path validation --------------------------------------------


def test_collect_output_file_refs_covers_artifacts_and_payload_paths() -> None:
    refs = collect_output_file_refs(
        {
            "plan": {
                "payload": {"points_path": "points.npy", "count": 3},
                "artifacts": {"trajectory_file": "trajectory.json"},
            },
            "report": {"payload": {"ok": True}, "artifacts": ["overlay.png"]},
        }
    )

    assert refs == {
        "plan.artifacts.trajectory_file": "trajectory.json",
        "plan.payload.points_path": "points.npy",
        "report.artifacts[0]": "overlay.png",
    }


def test_path_errors_flag_missing_and_escaping_refs(tmp_path: Path) -> None:
    (tmp_path / "ok.json").write_text("{}")
    outputs = {
        "plan": {
            "payload": {"ok": True},
            "artifacts": {
                "good": "ok.json",
                "missing": "artifacts/trajectory.json",
                "escape": "../outside.json",
            },
        }
    }

    errors = finish_artifact_path_errors(outputs, tmp_path)

    text = "\n".join(errors)
    assert len(errors) == 2
    assert "missing" in text and "ARTIFACTS_DIR" in text
    assert "escape" in text and "escapes" in text
    assert "good" not in text


def test_path_errors_empty_without_artifacts_dir() -> None:
    outputs = {"plan": {"payload": {"ok": True}, "artifacts": {"f": "nowhere.json"}}}
    assert finish_artifact_path_errors(outputs, None) == []


# ---- gate integration: the agent repairs inside its own budget ---------------


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


class _FakeExecutor:
    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
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


def _finish(outputs: dict) -> str:
    return json.dumps(
        {"tool": "finish", "args": {"claim": "done", "result": {"outputs": outputs}}}
    )


def _agent(policy: ScriptedCodePolicy) -> CodingAgentSubAgent:
    lib = _FakeLibrary([Skill.from_markdown(
        "---\nname: s\ncategory: perception\ndescription: d\n---\n\nBody.",
        skill_id="s",
    )])
    return CodingAgentSubAgent(
        executor=_FakeExecutor(),
        policy=policy,
        library=lib,
        max_turns=4,
        output_ports=(("plan", "test.value.v1"),),
        task_kind="plan",
    )


def _report(payload: dict | None = None, artifacts: dict | None = None) -> dict:
    value: dict = {"payload": payload or {"ok": True}, "confidence": 0.9}
    if artifacts is not None:
        value["artifacts"] = artifacts
    return value


def test_gate_rejects_bad_path_then_accepts_repair(tmp_path: Path) -> None:
    (tmp_path / "trajectory.json").write_text("{}")
    policy = ScriptedCodePolicy(
        [
            _finish({"plan": _report(artifacts={"trajectory_file": "artifacts/trajectory.json"})}),
            _finish({"plan": _report(artifacts={"trajectory_file": "trajectory.json"})}),
        ]
    )
    agent = _agent(policy)

    result = agent.run(SubAgentRequest(task="plan", artifacts_dir=str(tmp_path)))

    assert result.ok
    assert result.turns == 0  # both were finish turns, no python turns


def test_gate_rejects_undeclared_output_port(tmp_path: Path) -> None:
    policy = ScriptedCodePolicy(
        [
            _finish({"plan": _report(), "verifier_report": _report()}),
            _finish({"plan": _report()}),
        ]
    )
    agent = _agent(policy)

    result = agent.run(SubAgentRequest(task="plan", artifacts_dir=str(tmp_path)))

    assert result.ok
    assert "verifier_report" not in result.result.get("outputs", {})
