"""Project on-disk RoboMEx artifacts into Trace API DTOs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from robomex.web.models import (
    AgentDetail,
    AgentInstance,
    GraphEdge,
    GraphNode,
    ManagerTurn,
    MediaItem,
    RunDetail,
    SubgoalIndex,
    SwarmView,
    TurnDetail,
    TurnIndex,
)
from robomex.web.paths import file_url, resolve_under_run, resolve_under_workspace

_AGENT_DIR_RE = re.compile(r"^(\d{2})_(.+?)(?:_a(\d+))?$")
_TURN_PY_RE = re.compile(r"^turn_(\d+)\.py$")
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_VIDEO_EXTS = {".mp4", ".webm"}


def project_run(dir_path: str) -> RunDetail:
    run_dir = resolve_under_workspace(dir_path)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    summary = _read_json(run_dir / "summary.json")
    summary_subgoals = summary.get("subgoals") if isinstance(summary.get("subgoals"), list) else []
    disk_subgoals = sorted(
        p for p in run_dir.glob("subgoal_*") if p.is_dir() and re.match(r"subgoal_\d+$", p.name)
    )
    n = max(len(summary_subgoals), len(disk_subgoals))
    subgoals: list[SubgoalIndex] = []
    for i in range(n):
        sg_meta = summary_subgoals[i] if i < len(summary_subgoals) and isinstance(summary_subgoals[i], dict) else {}
        sg_dir = run_dir / f"subgoal_{i:02d}"
        planner = _read_planner_goal(run_dir, i)
        subgoals.append(
            SubgoalIndex(
                index=i,
                goal=str(sg_meta.get("goal") or planner.get("goal") or ""),
                postcondition=str(sg_meta.get("postcondition") or planner.get("postcondition") or ""),
                authoring_status=str(sg_meta.get("authoring_status") or ""),
                verification_status=str(sg_meta.get("verification_status") or ""),
                success=sg_meta.get("success") if "success" in sg_meta else None,
                motion_attempted=sg_meta.get("motion_attempted") if "motion_attempted" in sg_meta else None,
                note=str(sg_meta.get("note") or ""),
                inner_turns=int(sg_meta.get("inner_turns") or 0),
                loaded_skill_ids=[str(x) for x in (sg_meta.get("loaded_skill_ids") or [])],
                has_swarm=(sg_dir / "authoring" / "swarm").is_dir(),
            )
        )
    media = _collect_media_rel(run_dir, run_dir, max_files=40)
    return RunDetail(path=str(run_dir), summary=summary, subgoals=subgoals, media=media)


def project_swarm(dir_path: str, subgoal: int) -> SwarmView:
    run_dir = resolve_under_workspace(dir_path)
    swarm = _swarm_root(run_dir, subgoal)
    graph = _read_json(swarm / "subgoal_graph.json") if swarm.is_dir() else {}
    outcome = _read_json(swarm / "swarm_run_result.json") if swarm.is_dir() else {}
    store = _read_json(swarm / "artifact_store.json") if swarm.is_dir() else {}
    status_by_node = _node_status_map(swarm, outcome)

    nodes: list[GraphNode] = []
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("id") or "")
        specialist = raw.get("specialist") if isinstance(raw.get("specialist"), dict) else {}
        nodes.append(
            GraphNode(
                id=node_id,
                role=str(specialist.get("role") or ""),
                specialist_skill=str(specialist.get("specialist_skill") or ""),
                objective=str(specialist.get("objective") or ""),
                status=status_by_node.get(node_id, ""),
                verifier=bool(specialist.get("verifier")),
                changes_world=bool(specialist.get("changes_world")),
            )
        )

    edges: list[GraphEdge] = []
    for raw in graph.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        edges.append(
            GraphEdge(
                source=str(raw.get("from") or raw.get("source") or ""),
                target=str(raw.get("to") or raw.get("target") or ""),
                on=str(raw.get("on") or ""),
            )
        )

    agents = _list_agent_instances(run_dir, swarm, status_by_node) if swarm.is_dir() else []
    manager_turns = _list_manager_turns(swarm) if swarm.is_dir() else []
    media_paths = _collect_media_rel(run_dir, swarm if swarm.is_dir() else run_dir / f"subgoal_{subgoal:02d}", max_files=60)
    # Also include episode-level scene images for this subgoal context.
    for name in ("scene.png", f"scene_step{subgoal}.png", f"scene_step{subgoal + 1}.png"):
        if (run_dir / name).is_file() and name not in media_paths:
            media_paths.insert(0, name)
    media = [
        MediaItem(path=p, kind=_media_kind(p), url=file_url(run_dir, p))
        for p in media_paths
    ]
    artifact_keys = sorted(str(k) for k in store.keys()) if isinstance(store, dict) else []

    return SwarmView(
        run_dir=str(run_dir),
        subgoal_index=subgoal,
        task_skill=str(graph.get("task_skill") or ""),
        entry=str(graph.get("entry") or ""),
        success_node=str(graph.get("success_node") or ""),
        outcome_status=str(outcome.get("status") or outcome.get("creator_status") or ""),
        verification=str(outcome.get("verification") or ""),
        note=str(outcome.get("note") or ""),
        nodes=nodes,
        edges=edges,
        agents=agents,
        manager_turns=manager_turns,
        media=media,
        artifact_keys=artifact_keys,
    )


def project_agent(dir_path: str, subgoal: int, agent: str) -> AgentDetail:
    run_dir = resolve_under_workspace(dir_path)
    swarm = _swarm_root(run_dir, subgoal)
    agent_dir = _resolve_agent_dir(swarm, agent)
    request = _read_json(agent_dir / "request.json")
    result = _read_json(agent_dir / "result.json")
    outcome = _read_json(swarm / "swarm_run_result.json")
    status_by_node = _node_status_map(swarm, outcome)
    node_id, role, attempt = _parse_agent_dir_name(agent_dir.name)
    if isinstance(request.get("specialist"), dict):
        role = str(request["specialist"].get("role") or role)
        node_id = str(request.get("id") or request["specialist"].get("id") or node_id)
    elif isinstance(request.get("id"), str):
        node_id = str(request.get("id") or node_id)
    status = status_by_node.get(node_id, "")
    status = _result_status(result) or status

    turns = _list_turns(agent_dir)
    media_paths = _collect_media_rel(run_dir, agent_dir, max_files=40)
    media = [
        MediaItem(path=p, kind=_media_kind(p), url=file_url(run_dir, p))
        for p in media_paths
    ]
    return AgentDetail(
        run_dir=str(run_dir),
        subgoal_index=subgoal,
        agent_dir=agent_dir.name,
        node_id=node_id,
        role=role,
        status=status,
        request=request,
        result=result,
        turns=turns,
        media=media,
    )


def project_turn(dir_path: str, subgoal: int, agent: str, turn: int) -> TurnDetail:
    run_dir = resolve_under_workspace(dir_path)
    swarm = _swarm_root(run_dir, subgoal)
    agent_dir = _resolve_agent_dir(swarm, agent)
    code_path = agent_dir / f"turn_{turn:02d}.py"
    out_path = agent_dir / f"turn_{turn:02d}.out.txt"
    llm_req = agent_dir / "llm_io" / f"turn_{turn:02d}_request.json"
    llm_resp = agent_dir / "llm_io" / f"turn_{turn:02d}_response.json"
    llm_txt = agent_dir / "llm_io" / f"turn_{turn:02d}_response.txt"
    code = code_path.read_text(encoding="utf-8") if code_path.is_file() else ""
    out = out_path.read_text(encoding="utf-8") if out_path.is_file() else ""
    preview = ""
    if llm_txt.is_file():
        preview = llm_txt.read_text(encoding="utf-8")[:4000]
    elif llm_resp.is_file():
        try:
            payload = json.loads(llm_resp.read_text(encoding="utf-8"))
            preview = str(payload.get("raw") or payload.get("text") or "")[:4000]
        except Exception:
            preview = ""
    return TurnDetail(
        run_dir=str(run_dir),
        subgoal_index=subgoal,
        agent_dir=agent_dir.name,
        turn=turn,
        code=code,
        out=out,
        code_path=_rel(run_dir, code_path) if code_path.is_file() else "",
        out_path=_rel(run_dir, out_path) if out_path.is_file() else "",
        llm_request_path=_rel(run_dir, llm_req) if llm_req.is_file() else "",
        llm_response_path=_rel(run_dir, llm_resp) if llm_resp.is_file() else "",
        llm_response_txt_path=_rel(run_dir, llm_txt) if llm_txt.is_file() else "",
        response_preview=preview,
    )


def read_llm(dir_path: str, relative: str, view: str = "text") -> dict[str, Any]:
    run_dir = resolve_under_workspace(dir_path)
    target = resolve_under_run(run_dir, relative)
    if not target.is_file():
        raise FileNotFoundError(f"llm file not found: {relative}")
    size = target.stat().st_size
    text = target.read_text(encoding="utf-8", errors="replace")
    if target.suffix.lower() == ".txt":
        return {"path": relative, "view": view, "size": size, "content": text if view != "meta" else None}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"path": relative, "view": view, "size": size, "content": text if view != "meta" else None}

    if view == "full":
        return {"path": relative, "view": view, "size": size, "content": payload}
    if view == "meta":
        return {
            "path": relative,
            "view": view,
            "size": size,
            "content": {
                "turn": payload.get("turn") if isinstance(payload, dict) else None,
                "agent_role": payload.get("agent_role") if isinstance(payload, dict) else None,
                "agent_label": payload.get("agent_label") if isinstance(payload, dict) else None,
                "message_count": len(payload.get("messages") or []) if isinstance(payload, dict) else None,
                "has_tool_calls": bool(payload.get("tool_calls")) if isinstance(payload, dict) else False,
                "parser_error": payload.get("parser_error") if isinstance(payload, dict) else None,
            },
        }
    # text view: strip image parts / huge base64
    if isinstance(payload, list):
        return {"path": relative, "view": view, "size": size, "content": _strip_images(payload)}
    if isinstance(payload, dict):
        cleaned = dict(payload)
        if "messages" in cleaned:
            cleaned["messages"] = _strip_images(cleaned.get("messages"))
        if isinstance(cleaned.get("raw"), str) and len(cleaned["raw"]) > 20000:
            cleaned["raw"] = cleaned["raw"][:20000] + "... <truncated>"
        return {"path": relative, "view": view, "size": size, "content": cleaned}
    return {"path": relative, "view": view, "size": size, "content": payload}


def _swarm_root(run_dir: Path, subgoal: int) -> Path:
    return run_dir / f"subgoal_{subgoal:02d}" / "authoring" / "swarm"


def _resolve_agent_dir(swarm: Path, agent: str) -> Path:
    name = Path(agent).name
    agent_dir = swarm / name
    if not agent_dir.is_dir():
        raise FileNotFoundError(f"agent directory not found: {name}")
    if not _AGENT_DIR_RE.match(name):
        raise FileNotFoundError(f"not an agent instance directory: {name}")
    return agent_dir


def _list_agent_instances(
    run_dir: Path,
    swarm: Path,
    status_by_node: dict[str, str],
) -> list[AgentInstance]:
    agents: list[AgentInstance] = []
    for path in sorted(p for p in swarm.iterdir() if p.is_dir() and _AGENT_DIR_RE.match(p.name)):
        node_id, role, attempt = _parse_agent_dir_name(path.name)
        request = _read_json(path / "request.json")
        result = _read_json(path / "result.json")
        if isinstance(request.get("specialist"), dict):
            role = str(request["specialist"].get("role") or role)
            node_id = str(request.get("id") or request["specialist"].get("id") or node_id)
        elif isinstance(request.get("id"), str):
            node_id = str(request.get("id") or node_id)
        status = _result_status(result) or status_by_node.get(node_id, "")
        turns = list(path.glob("turn_*.py"))
        media = _collect_media_rel(run_dir, path, max_files=12)
        agents.append(
            AgentInstance(
                dir=path.name,
                node_id=node_id,
                role=role,
                status=status,
                attempt=attempt,
                turn_count=len(turns),
                has_result=(path / "result.json").is_file(),
                media=media,
            )
        )
    return agents


def _list_manager_turns(swarm: Path) -> list[ManagerTurn]:
    llm_dir = swarm / "manager_trace" / "llm_io"
    if not llm_dir.is_dir():
        llm_dir = swarm / "creator_trace" / "llm_io"
    if not llm_dir.is_dir():
        return []
    # swarm = <run>/subgoal_XX/authoring/swarm
    run_dir = swarm.parent.parent.parent
    turns: list[ManagerTurn] = []
    for resp in sorted(llm_dir.glob("turn_*_response.txt")):
        m = re.match(r"turn_(\d+)_response\.txt$", resp.name)
        if not m:
            continue
        idx = int(m.group(1))
        req = llm_dir / f"turn_{idx:02d}_request.json"
        preview = resp.read_text(encoding="utf-8", errors="replace")[:500]
        turns.append(
            ManagerTurn(
                turn=idx,
                request_path=_rel(run_dir, req) if req.is_file() else "",
                response_path=_rel(run_dir, resp),
                response_preview=preview,
            )
        )
    return turns


def _list_turns(agent_dir: Path) -> list[TurnIndex]:
    turns: list[TurnIndex] = []
    for code_path in sorted(agent_dir.glob("turn_*.py")):
        m = _TURN_PY_RE.match(code_path.name)
        if not m:
            continue
        idx = int(m.group(1))
        out_path = agent_dir / f"turn_{idx:02d}.out.txt"
        llm_req = agent_dir / "llm_io" / f"turn_{idx:02d}_request.json"
        llm_resp = agent_dir / "llm_io" / f"turn_{idx:02d}_response.json"
        llm_txt = agent_dir / "llm_io" / f"turn_{idx:02d}_response.txt"
        run_dir = agent_dir.parent.parent.parent.parent  # .../run
        turns.append(
            TurnIndex(
                turn=idx,
                has_code=True,
                has_out=out_path.is_file(),
                has_llm=llm_req.is_file() or llm_resp.is_file() or llm_txt.is_file(),
                code_path=_rel(run_dir, code_path),
                out_path=_rel(run_dir, out_path) if out_path.is_file() else "",
                llm_request_path=_rel(run_dir, llm_req) if llm_req.is_file() else "",
                llm_response_path=_rel(run_dir, llm_resp) if llm_resp.is_file() else "",
                llm_response_txt_path=_rel(run_dir, llm_txt) if llm_txt.is_file() else "",
            )
        )
    # Also include llm-only turns without code (finish turns)
    llm_dir = agent_dir / "llm_io"
    if llm_dir.is_dir():
        seen = {t.turn for t in turns}
        for req in sorted(llm_dir.glob("turn_*_request.json")):
            m = re.match(r"turn_(\d+)_request\.json$", req.name)
            if not m:
                continue
            idx = int(m.group(1))
            if idx in seen:
                continue
            llm_resp = llm_dir / f"turn_{idx:02d}_response.json"
            llm_txt = llm_dir / f"turn_{idx:02d}_response.txt"
            run_dir = agent_dir.parent.parent.parent.parent
            turns.append(
                TurnIndex(
                    turn=idx,
                    has_code=False,
                    has_out=False,
                    has_llm=True,
                    llm_request_path=_rel(run_dir, req),
                    llm_response_path=_rel(run_dir, llm_resp) if llm_resp.is_file() else "",
                    llm_response_txt_path=_rel(run_dir, llm_txt) if llm_txt.is_file() else "",
                )
            )
        turns.sort(key=lambda t: t.turn)
    return turns


def _node_status_map(swarm: Path, outcome: dict[str, Any]) -> dict[str, str]:
    status: dict[str, str] = {}
    for item in outcome.get("node_results") or []:
        if isinstance(item, dict) and item.get("node_id"):
            status[str(item["node_id"])] = str(item.get("status") or "")
    events_path = swarm / "node_events.jsonl"
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            node_id = str(row.get("node_id") or "")
            if node_id:
                status[node_id] = str(row.get("status") or status.get(node_id, ""))
    return status


def _parse_agent_dir_name(name: str) -> tuple[str, str, int | None]:
    m = _AGENT_DIR_RE.match(name)
    if not m:
        return name, "", None
    node_id = m.group(2)
    attempt = int(m.group(3)) if m.group(3) else None
    return node_id, "", attempt


def _result_status(result: dict[str, Any]) -> str:
    if not isinstance(result, dict) or not result:
        return ""
    nested = result.get("result") if isinstance(result.get("result"), dict) else {}
    for candidate in (
        result.get("status"),
        nested.get("status"),
        nested.get("exit_event"),
    ):
        if candidate:
            return str(candidate)
    if "ok" in result:
        return "succeeded" if result.get("ok") else "failed"
    if "ok" in nested:
        return "succeeded" if nested.get("ok") else "failed"
    return ""


def _collect_media_rel(run_dir: Path, root: Path, *, max_files: int) -> list[str]:
    if not root.exists():
        return []
    found: list[tuple[float, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _IMAGE_EXTS | _VIDEO_EXTS:
            continue
        try:
            rel = _rel(run_dir, path)
        except ValueError:
            continue
        found.append((path.stat().st_mtime, rel))
    found.sort(key=lambda item: item[0], reverse=True)
    return [rel for _, rel in found[:max_files]]


def _media_kind(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in _VIDEO_EXTS:
        return "video"
    return "image"


def _read_planner_goal(run_dir: Path, index: int) -> dict[str, str]:
    path = run_dir / "planner.jsonl"
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("index", -1)) == index:
            return {
                "goal": str(row.get("goal") or ""),
                "postcondition": str(row.get("postcondition") or ""),
            }
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _rel(run_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(run_dir.resolve()).as_posix()


def _strip_images(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_images(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") == "image_url" or "image_url" in value:
            return {"type": "image_url", "image_url": {"url": "<omitted image>"}}
        return {str(k): _strip_images(v) for k, v in value.items()}
    if isinstance(value, str) and value.startswith("data:image"):
        return "<omitted data:image>"
    return value
