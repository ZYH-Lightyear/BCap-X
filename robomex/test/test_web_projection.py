"""Tests for RoboMEx Trace API projection."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from robomex.web.projection import project_agent, project_run, project_swarm, project_turn
from robomex.web.server import create_app


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, (dict, list)):
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        path.write_text(str(payload), encoding="utf-8")


def _make_run(root: Path) -> Path:
    run = root / "20260101_000000"
    run.mkdir(parents=True)
    (run / "events.jsonl").write_text(
        json.dumps({"event": "episode_start", "task": "Pick the can"}) + "\n",
        encoding="utf-8",
    )
    _write(
        run / "summary.json",
        {
            "task": "Pick the can",
            "planner_status": "done",
            "env_success": True,
            "n_subgoals": 1,
            "authoring_strategy": "dynamic_swarm",
            "subgoals": [
                {
                    "goal": "Pick the can",
                    "authoring_status": "succeeded",
                    "verification_status": "passed",
                    "success": True,
                    "inner_turns": 2,
                    "loaded_skill_ids": ["segment_object"],
                }
            ],
        },
    )
    swarm = run / "subgoal_00" / "authoring" / "swarm"
    _write(
        swarm / "subgoal_graph.json",
        {
            "schema": "robomex.subgoal_graph.v1",
            "task_skill": "pick_object",
            "entry": "ground",
            "success_node": "verify",
            "nodes": [
                {
                    "id": "ground",
                    "specialist": {
                        "id": "ground",
                        "role": "grounding",
                        "specialist_skill": "segment_object",
                        "objective": "Ground target",
                        "verifier": False,
                        "changes_world": False,
                    },
                },
                {
                    "id": "verify",
                    "specialist": {
                        "id": "verify",
                        "role": "verifier",
                        "specialist_skill": "verify",
                        "objective": "Verify",
                        "verifier": True,
                        "changes_world": False,
                    },
                },
            ],
            "edges": [{"from": "ground", "to": "verify", "on": "success"}],
        },
    )
    _write(
        swarm / "swarm_run_result.json",
        {
            "status": "succeeded",
            "verification": "passed",
            "node_results": [
                {"node_id": "ground", "status": "succeeded"},
                {"node_id": "verify", "status": "succeeded"},
            ],
        },
    )
    _write(swarm / "artifact_store.json", {"ground.object_mask": {"port": "object_mask"}})
    agent = swarm / "00_ground_a1"
    _write(
        agent / "request.json",
        {
            "id": "ground",
            "specialist": {"id": "ground", "role": "grounding"},
        },
    )
    _write(agent / "result.json", {"status": "succeeded", "ok": True})
    _write(agent / "turn_00.py", "print('hello')\n")
    _write(agent / "turn_00.out.txt", "# status    : succeeded\n\n## stdout\nhello\n")
    _write(
        agent / "llm_io" / "turn_00_request.json",
        {
            "turn": 0,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}]}
            ],
        },
    )
    _write(agent / "llm_io" / "turn_00_response.txt", '{"tool":"run_python"}')
    _write(agent / "overlay.png", "fake-png")
    _write(
        swarm / "manager_trace" / "llm_io" / "turn_00_response.txt",
        '{"tool":"compose_graph"}',
    )
    _write(swarm / "manager_trace" / "llm_io" / "turn_00_request.json", [{"role": "system", "content": "mgr"}])
    _write(run / "scene.png", "fake-scene")
    return run


def test_project_run_and_swarm(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = _make_run(tmp_path / "outputs")
    detail = project_run(str(run))
    assert detail.summary["task"] == "Pick the can"
    assert len(detail.subgoals) == 1
    assert detail.subgoals[0].has_swarm is True

    swarm = project_swarm(str(run), 0)
    assert swarm.entry == "ground"
    assert [n.id for n in swarm.nodes] == ["ground", "verify"]
    assert swarm.edges[0].source == "ground"
    assert swarm.agents[0].dir == "00_ground_a1"
    assert swarm.agents[0].turn_count == 1
    assert swarm.nodes[0].status == "succeeded"
    assert any(m.path.endswith("overlay.png") for m in swarm.media)
    assert swarm.manager_turns and swarm.manager_turns[0].turn == 0


def test_project_agent_and_turn(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = _make_run(tmp_path / "outputs")
    agent = project_agent(str(run), 0, "00_ground_a1")
    assert agent.node_id == "ground"
    assert agent.turns[0].has_code is True
    assert agent.turns[0].has_llm is True

    turn = project_turn(str(run), 0, "00_ground_a1", 0)
    assert "print('hello')" in turn.code
    assert "hello" in turn.out


def test_api_endpoints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = _make_run(tmp_path / "outputs" / "robomex_planner_live")
    client = TestClient(create_app(default_root="outputs/robomex_planner_live"))

    runs = client.get("/api/v1/runs", params={"root": "outputs/robomex_planner_live"})
    assert runs.status_code == 200
    assert runs.json()["runs"][0]["path"] == str(run)

    detail = client.get("/api/v1/runs/by-path", params={"dir": str(run)})
    assert detail.status_code == 200
    assert detail.json()["subgoals"][0]["goal"] == "Pick the can"

    swarm = client.get("/api/v1/runs/by-path/swarm", params={"dir": str(run), "subgoal": 0})
    assert swarm.status_code == 200
    body = swarm.json()
    assert body["agents"][0]["dir"] == "00_ground_a1"

    agent = client.get(
        "/api/v1/runs/by-path/agent",
        params={"dir": str(run), "subgoal": 0, "agent": "00_ground_a1"},
    )
    assert agent.status_code == 200

    turn = client.get(
        "/api/v1/runs/by-path/turn",
        params={"dir": str(run), "subgoal": 0, "agent": "00_ground_a1", "turn": 0},
    )
    assert turn.status_code == 200
    assert "print" in turn.json()["code"]

    llm = client.get(
        "/api/v1/runs/by-path/llm",
        params={
            "dir": str(run),
            "path": "subgoal_00/authoring/swarm/00_ground_a1/llm_io/turn_00_request.json",
            "view": "text",
        },
    )
    assert llm.status_code == 200
    content = llm.json()["content"]
    assert content["messages"][0]["content"][1]["image_url"]["url"] == "<omitted image>"

    file_resp = client.get(
        "/api/v1/runs/by-path/file",
        params={"dir": str(run), "path": "scene.png"},
    )
    assert file_resp.status_code == 200
