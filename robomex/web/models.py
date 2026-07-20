"""Pydantic DTOs for the RoboMEx Trace API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunSummary(BaseModel):
    path: str
    name: str
    mtime: float
    task: str = ""
    planner_status: str = ""
    env_success: bool | None = None
    n_subgoals: int = 0
    authoring_strategy: str = ""


class RunListResponse(BaseModel):
    root: str
    runs: list[RunSummary]


class SubgoalIndex(BaseModel):
    index: int
    goal: str = ""
    postcondition: str = ""
    authoring_status: str = ""
    verification_status: str = ""
    success: bool | None = None
    motion_attempted: bool | None = None
    note: str = ""
    inner_turns: int = 0
    loaded_skill_ids: list[str] = Field(default_factory=list)
    has_swarm: bool = False


class RunDetail(BaseModel):
    path: str
    summary: dict[str, Any] = Field(default_factory=dict)
    subgoals: list[SubgoalIndex] = Field(default_factory=list)
    media: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    source: str
    target: str
    on: str = ""


class GraphNode(BaseModel):
    id: str
    role: str = ""
    specialist_skill: str = ""
    objective: str = ""
    status: str = ""
    verifier: bool = False
    changes_world: bool = False


class AgentInstance(BaseModel):
    dir: str
    node_id: str = ""
    role: str = ""
    status: str = ""
    attempt: int | None = None
    turn_count: int = 0
    has_result: bool = False
    media: list[str] = Field(default_factory=list)


class ManagerTurn(BaseModel):
    turn: int
    request_path: str = ""
    response_path: str = ""
    response_preview: str = ""


class MediaItem(BaseModel):
    path: str
    kind: str  # image | video
    url: str


class SwarmView(BaseModel):
    run_dir: str
    subgoal_index: int
    task_skill: str = ""
    entry: str = ""
    success_node: str = ""
    outcome_status: str = ""
    verification: str = ""
    note: str = ""
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    agents: list[AgentInstance] = Field(default_factory=list)
    manager_turns: list[ManagerTurn] = Field(default_factory=list)
    media: list[MediaItem] = Field(default_factory=list)
    artifact_keys: list[str] = Field(default_factory=list)


class TurnIndex(BaseModel):
    turn: int
    has_code: bool = False
    has_out: bool = False
    has_llm: bool = False
    code_path: str = ""
    out_path: str = ""
    llm_request_path: str = ""
    llm_response_path: str = ""
    llm_response_txt_path: str = ""


class AgentDetail(BaseModel):
    run_dir: str
    subgoal_index: int
    agent_dir: str
    node_id: str = ""
    role: str = ""
    status: str = ""
    request: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    turns: list[TurnIndex] = Field(default_factory=list)
    media: list[MediaItem] = Field(default_factory=list)


class TurnDetail(BaseModel):
    run_dir: str
    subgoal_index: int
    agent_dir: str
    turn: int
    code: str = ""
    out: str = ""
    code_path: str = ""
    out_path: str = ""
    llm_request_path: str = ""
    llm_response_path: str = ""
    llm_response_txt_path: str = ""
    response_preview: str = ""


class LlmView(BaseModel):
    path: str
    view: str
    size: int = 0
    content: Any = None
