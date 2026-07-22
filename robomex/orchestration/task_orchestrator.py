"""Durable multi-intent Episode orchestration for RoboMEx v2.

The lower-level :class:`EpisodeRuntime` executes one already-authored workflow.
This module owns the missing task-level loop: a bounded planner opens one
``SubgoalIntent`` at a time, a workflow author binds that intent to an exact v2
graph, and the resulting ``IntentOutcome`` becomes causal context for the next
planner call.  Every boundary is checkpointed before external model work so a
process restart resumes the same logical call/workflow identity.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from robomex.data import ResolvedArtifactRef
from robomex.elastic import CompiledElasticGraph
from robomex.elastic.graph_patch import ComposableFrontier
from robomex.orchestration.episode import EpisodeRuntime
from robomex.orchestration.intent import IntentOutcome, IntentStatus, SubgoalIntent
from robomex.orchestration.manager import ManagerInvoker, SwarmManagerSession
from robomex.orchestration.run_budget import (
    RunBudgetExceededError,
    RunBudgetOperationStatus,
    RunBudgetReservation,
    RunBudgetVector,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SafeId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
    ),
]
DigestStr = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


class TaskOrchestratorError(RuntimeError):
    """Task state, planner, authorer, or runtime violated a durable boundary."""


class TaskStateConflictError(TaskOrchestratorError):
    """A durable task revision or identity was rebound."""


class TaskRunStatus(str, Enum):  # noqa: UP042 - Python 3.10 support
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    EXHAUSTED = "exhausted"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class PlannerDecisionKind(str, Enum):  # noqa: UP042
    OPEN_INTENT = "open_intent"
    DONE = "done"
    BLOCKED = "blocked"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class PlannerDecision(_StrictModel):
    """One closed planner response; prose is never treated as executable intent."""

    schema_version: Literal["robomex.planner_decision.v1"] = "robomex.planner_decision.v1"
    kind: PlannerDecisionKind
    intent: SubgoalIntent | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _decision_shape(self) -> PlannerDecision:
        if (self.kind is PlannerDecisionKind.OPEN_INTENT) != (self.intent is not None):
            raise ValueError("Only open_intent decisions carry an intent")
        if self.kind is PlannerDecisionKind.BLOCKED and not self.reason.strip():
            raise ValueError("A blocked planner decision requires a reason")
        return self


class PlannerCallBudget(_StrictModel):
    """Typed, operator-owned grant for each task-planner model boundary.

    Planner adapters do not yet expose provider-signed token usage.  The v2
    baseline therefore settles the complete grant on success and failure; it
    never presents a text-length heuristic as exact token accounting.
    """

    schema_version: Literal["robomex.planner_call_budget.v1"] = "robomex.planner_call_budget.v1"
    model_call_grant: int = Field(default=1, ge=1)
    token_grant: int = Field(default=256, ge=1)
    wall_time_ms: int = Field(default=10_000, ge=1)


class EpisodePlanningContext(_StrictModel):
    """Compact, fresh context for one bounded task-planner invocation."""

    schema_version: Literal["robomex.episode_planning_context.v1"] = (
        "robomex.episode_planning_context.v1"
    )
    planner_call_id: NonEmptyStr
    task_run_id: SafeId
    episode_id: NonEmptyStr
    task: NonEmptyStr
    intent_index: int = Field(ge=0)
    prior_intents: tuple[SubgoalIntent, ...] = ()
    prior_outcomes: tuple[IntentOutcome, ...] = ()
    embodied_state_revision: int = Field(ge=0)
    compact_state: dict[str, JsonValue] = Field(default_factory=dict)
    remaining_intents: int = Field(ge=0)
    remaining_planner_calls: int = Field(ge=0)
    scene_image_path: str | None = None

    @model_validator(mode="after")
    def _causal_history_matches_frontier(self) -> EpisodePlanningContext:
        if len(self.prior_intents) != len(self.prior_outcomes):
            raise ValueError("planner intent/outcome history must be paired")
        if len(self.prior_intents) != self.intent_index:
            raise ValueError("intent_index must equal the completed intent history")
        return self


@runtime_checkable
class IntentPlanner(Protocol):
    """Stateless/idempotent planner boundary keyed by ``planner_call_id``."""

    def next_intent(self, context: EpisodePlanningContext) -> PlannerDecision: ...


@runtime_checkable
class BoundedIntentPlanner(Protocol):
    """Planner that enforces the supplied grant before every model call."""

    def next_intent_bounded(
        self,
        context: EpisodePlanningContext,
        *,
        max_tokens: int,
        max_model_calls: int,
        deadline_monotonic_s: float,
    ) -> PlannerDecision: ...


class AvailableArtifact(_StrictModel):
    """Compact artifact inventory exposed to a workflow author, never raw arrays."""

    workflow_id: NonEmptyStr
    activation_id: NonEmptyStr
    attempt: int = Field(ge=1)
    port: NonEmptyStr
    schema_id: NonEmptyStr
    artifact_id: NonEmptyStr
    content_digest: DigestStr

    @property
    def ref(self) -> ResolvedArtifactRef:
        return ResolvedArtifactRef(self.artifact_id, self.content_digest)


class WorkflowAuthoringContext(_StrictModel):
    """Exact episode view supplied to one logical graph-authoring call."""

    schema_version: Literal["robomex.workflow_authoring_context.v1"] = (
        "robomex.workflow_authoring_context.v1"
    )
    authoring_call_id: NonEmptyStr
    task_run_id: SafeId
    episode_id: NonEmptyStr
    workflow_id: NonEmptyStr
    task: NonEmptyStr
    intent: SubgoalIntent
    prior_outcomes: tuple[IntentOutcome, ...] = ()
    embodied_state_revision: int = Field(ge=0)
    compact_state: dict[str, JsonValue] = Field(default_factory=dict)
    available_artifacts: tuple[AvailableArtifact, ...] = ()


@dataclass(frozen=True)
class WorkflowBlueprint:
    """Process-local dependencies plus a content-addressed executable graph."""

    graph: CompiledElasticGraph
    external_refs: Mapping[str, ResolvedArtifactRef | Mapping[str, Any]]
    frontier: ComposableFrontier | None = None
    manager_session: SwarmManagerSession | None = None
    manager_invoker: ManagerInvoker | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.graph, CompiledElasticGraph):
            raise TypeError("graph must be a CompiledElasticGraph")
        if (self.manager_session is None) != (self.manager_invoker is None):
            raise ValueError("manager_session and manager_invoker must be supplied together")
        normalized = {
            str(name): ResolvedArtifactRef.from_any(ref) for name, ref in self.external_refs.items()
        }
        if any(not name.strip() for name in normalized):
            raise ValueError("external ref names must not be empty")
        object.__setattr__(self, "external_refs", normalized)


@runtime_checkable
class WorkflowAuthor(Protocol):
    """Author one graph for an intent; retries reuse ``authoring_call_id``."""

    def author(self, context: WorkflowAuthoringContext) -> WorkflowBlueprint: ...


class TaskRunState(_StrictModel):
    """Complete durable image of the task-level orchestrator."""

    schema_version: Literal["robomex.task_run_state.v1"] = "robomex.task_run_state.v1"
    state_revision: int = Field(ge=0)
    task_run_id: SafeId
    episode_id: NonEmptyStr
    task: NonEmptyStr
    status: TaskRunStatus = TaskRunStatus.RUNNING
    max_intents: int = Field(ge=1)
    max_planner_calls: int = Field(ge=1)
    planner_calls_started: int = Field(default=0, ge=0)
    pending_planner_call_id: str | None = Field(default=None, min_length=1)
    active_intent: SubgoalIntent | None = None
    active_workflow_id: str | None = Field(default=None, min_length=1)
    intents: tuple[SubgoalIntent, ...] = ()
    outcomes: tuple[IntentOutcome, ...] = ()
    terminal_reason: str | None = None
    scene_image_path: str | None = None

    @model_validator(mode="after")
    def _state_shape(self) -> TaskRunState:
        if len(self.intents) != len(self.outcomes) + (1 if self.active_intent else 0):
            raise ValueError("task intent/outcome frontier is inconsistent")
        if (self.active_intent is None) != (self.active_workflow_id is None):
            raise ValueError("active intent and workflow identity must be atomic")
        if self.active_intent is not None and self.intents[-1] != self.active_intent:
            raise ValueError("active intent must be the newest intent")
        if self.planner_calls_started > self.max_planner_calls:
            raise ValueError("planner call usage exceeds its hard limit")
        if len(self.intents) > self.max_intents:
            raise ValueError("intent usage exceeds its hard limit")
        if self.pending_planner_call_id is not None and self.active_intent is not None:
            raise ValueError("planner call cannot remain pending after an intent is bound")
        if self.status is not TaskRunStatus.RUNNING and (
            self.active_intent is not None or self.pending_planner_call_id is not None
        ):
            raise ValueError("terminal task state cannot retain active work")
        return self


class TaskStateCommit(_StrictModel):
    schema_version: Literal["robomex.task_state_commit.v1"] = "robomex.task_state_commit.v1"
    commit_id: NonEmptyStr
    reason: NonEmptyStr
    state: TaskRunState
    content_digest: DigestStr

    @classmethod
    def build(cls, state: TaskRunState, *, reason: str) -> TaskStateCommit:
        payload = {
            "reason": reason,
            "state": state.model_dump(mode="json"),
        }
        digest = _digest("robomex.task_state_commit.content.v1", payload)
        return cls(
            commit_id=f"{state.task_run_id}:{state.state_revision}",
            reason=reason,
            state=state,
            content_digest=digest,
        )

    @model_validator(mode="after")
    def _verify(self) -> TaskStateCommit:
        if self.commit_id != f"{self.state.task_run_id}:{self.state.state_revision}":
            raise ValueError("task commit id is not bound to its state revision")
        expected = _digest(
            "robomex.task_state_commit.content.v1",
            {"reason": self.reason, "state": self.state.model_dump(mode="json")},
        )
        if self.content_digest != expected:
            raise ValueError("task commit digest mismatch")
        return self


class TaskStateLedger:
    """Append-only full task checkpoints with process-safe contiguous revisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.path.touch(exist_ok=True)
        self.lock_path.touch(exist_ok=True)
        self._lock = threading.RLock()

    def latest(self, task_run_id: str) -> TaskRunState | None:
        with self._lock, self._process_lock(exclusive=False):
            commits = self._load()
            selected = [item.state for item in commits if item.state.task_run_id == task_run_id]
            return selected[-1] if selected else None

    def append(self, state: TaskRunState, *, reason: str) -> bool:
        commit = TaskStateCommit.build(state, reason=reason)
        with self._lock, self._process_lock(exclusive=True):
            commits = self._load()
            history = [item for item in commits if item.state.task_run_id == state.task_run_id]
            if history:
                latest = history[-1]
                if latest.state.state_revision == state.state_revision:
                    if latest == commit:
                        return False
                    raise TaskStateConflictError(
                        "task state revision was rebound to different content"
                    )
                if state.state_revision != latest.state.state_revision + 1:
                    raise TaskStateConflictError("task state revisions must be contiguous")
                if (
                    latest.state.episode_id != state.episode_id
                    or latest.state.task != state.task
                    or latest.state.max_intents != state.max_intents
                    or latest.state.max_planner_calls != state.max_planner_calls
                ):
                    raise TaskStateConflictError("task run identity was rebound")
            elif state.state_revision != 0:
                raise TaskStateConflictError("a task state history must begin at revision 0")
            encoded = (
                json.dumps(
                    commit.model_dump(mode="json"),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            with self.path.open("ab+") as stream:
                stream.seek(0, os.SEEK_END)
                start = stream.tell()
                try:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                except Exception:
                    stream.seek(start)
                    stream.truncate()
                    stream.flush()
                    with suppress(Exception):
                        os.fsync(stream.fileno())
                    raise
            return True

    def _load(self) -> tuple[TaskStateCommit, ...]:
        commits: list[TaskStateCommit] = []
        latest_revision: dict[str, int] = {}
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                commit = TaskStateCommit.model_validate_json(line)
                prior = latest_revision.get(commit.state.task_run_id)
                if prior is None and commit.state.state_revision != 0:
                    raise ValueError("history does not begin at revision zero")
                if prior is not None and commit.state.state_revision != prior + 1:
                    raise ValueError("history contains a revision gap")
                latest_revision[commit.state.task_run_id] = commit.state.state_revision
                commits.append(commit)
            except Exception as exc:
                raise TaskStateConflictError(
                    f"invalid task state commit at line {line_number}"
                ) from exc
        return tuple(commits)

    @contextmanager
    def _process_lock(self, *, exclusive: bool):
        with self.lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class PlannerCallLedger:
    """Durable call-id response ledger for manifest-budgeted planners.

    A ``started`` record is fsynced before the planner boundary.  A restart may
    replay a completed decision, but it never re-enters a call whose outcome is
    unknown after a crash.  This is intentionally fail-closed: exactly-once
    model effects cannot be reconstructed without provider idempotency.
    """

    _SCHEMA = "robomex.planner_call_record.v1"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def invoke(
        self,
        *,
        context: EpisodePlanningContext,
        budget: PlannerCallBudget,
        callback: Callable[[], PlannerDecision],
    ) -> PlannerDecision:
        request = {
            "context": context.model_dump(mode="json"),
            "planner_budget": budget.model_dump(mode="json"),
        }
        request_digest = _canonical_mapping_digest(request)
        digest = hashlib.sha256(context.planner_call_id.encode()).hexdigest()
        path = self.root / f"{digest}.json"
        lock_path = self.root / f"{digest}.lock"
        with self._lock, lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                previous = self._read(path)
                if previous is not None:
                    if (
                        previous.get("planner_call_id") != context.planner_call_id
                        or previous.get("request_digest") != request_digest
                    ):
                        raise TaskStateConflictError(
                            "planner_call_id was rebound to different content"
                        )
                    status = previous.get("status")
                    if status == "completed":
                        return PlannerDecision.model_validate(previous["decision"])
                    raise TaskOrchestratorError(
                        "planner call cannot be re-entered after an uncertain or failed "
                        f"provider boundary ({status})"
                    )

                base = {
                    "schema": self._SCHEMA,
                    "planner_call_id": context.planner_call_id,
                    "request_digest": request_digest,
                    "status": "started",
                }
                self._write(path, base)
                try:
                    decision = callback()
                    if not isinstance(decision, PlannerDecision):
                        decision = PlannerDecision.model_validate(decision)
                except Exception as exc:
                    self._write(
                        path,
                        {
                            **base,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    raise
                self._write(
                    path,
                    {
                        **base,
                        "status": "completed",
                        "decision": decision.model_dump(mode="json"),
                    },
                )
                return decision
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    @classmethod
    def _read(cls, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskStateConflictError("planner call record is unreadable") from exc
        if not isinstance(value, dict) or value.get("schema") != cls._SCHEMA:
            raise TaskStateConflictError("planner call record has an invalid schema")
        if value.get("status") not in {"started", "failed", "completed"}:
            raise TaskStateConflictError("planner call record has an invalid status")
        return value

    @staticmethod
    def _write(path: Path, value: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temp = Path(raw_temp)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temp.exists():
                temp.unlink()


def _canonical_mapping_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class EpisodeTaskResult(_StrictModel):
    schema_version: Literal["robomex.episode_task_result.v1"] = "robomex.episode_task_result.v1"
    task_run_id: SafeId
    episode_id: NonEmptyStr
    task: NonEmptyStr
    status: TaskRunStatus
    intents: tuple[SubgoalIntent, ...]
    outcomes: tuple[IntentOutcome, ...]
    reason: str | None = None


class EpisodeOrchestrator:
    """Persistently drive planner → authored workflow → outcome across a task."""

    def __init__(
        self,
        runtime: EpisodeRuntime,
        *,
        planner: IntentPlanner,
        author: WorkflowAuthor,
        ledger: TaskStateLedger | None = None,
        planner_budget: PlannerCallBudget | None = None,
    ) -> None:
        if not isinstance(runtime, EpisodeRuntime):
            raise TypeError("runtime must be an EpisodeRuntime")
        if not isinstance(planner, IntentPlanner):
            raise TypeError("planner does not implement IntentPlanner")
        if not isinstance(author, WorkflowAuthor):
            raise TypeError("author does not implement WorkflowAuthor")
        self.runtime = runtime
        self.planner = planner
        self.author = author
        uses_model_calls = getattr(planner, "uses_model_calls", True)
        if not isinstance(uses_model_calls, bool):
            raise TypeError("planner uses_model_calls must be a boolean when declared")
        self.planner_uses_model_calls = uses_model_calls
        if planner_budget is not None and not isinstance(planner_budget, PlannerCallBudget):
            raise TypeError("planner_budget must be a PlannerCallBudget")
        if runtime.run_budget_authority is not None and planner_budget is None:
            raise TaskOrchestratorError(
                "A runtime with manifest RunBudgets requires a typed planner_budget"
            )
        if runtime.run_budget_authority is not None and not isinstance(
            planner, BoundedIntentPlanner
        ):
            raise TaskOrchestratorError(
                "A runtime with manifest RunBudgets requires a BoundedIntentPlanner"
            )
        self.planner_budget = planner_budget
        self.ledger = ledger or TaskStateLedger(runtime.episode_root / "task_orchestrator.v1.jsonl")
        self.planner_calls = PlannerCallLedger(runtime.episode_root / "planner_calls.v1")
        self._lock = threading.RLock()

    def start(
        self,
        *,
        task_run_id: str,
        task: str,
        max_intents: int = 8,
        max_planner_calls: int | None = None,
        scene_image_path: str | None = None,
    ) -> TaskRunState:
        with self._lock:
            existing = self.ledger.latest(task_run_id)
            planner_limit = max_planner_calls or max_intents + 1
            if existing is not None:
                if (
                    existing.episode_id != self.runtime.episode_id
                    or existing.task != task.strip()
                    or existing.max_intents != max_intents
                    or existing.max_planner_calls != planner_limit
                    or existing.scene_image_path != scene_image_path
                ):
                    raise TaskStateConflictError("task_run_id is already bound")
                return existing
            state = TaskRunState(
                state_revision=0,
                task_run_id=task_run_id,
                episode_id=self.runtime.episode_id,
                task=task,
                max_intents=max_intents,
                max_planner_calls=planner_limit,
                scene_image_path=scene_image_path,
            )
            self.ledger.append(state, reason="task_started")
            return state

    def run(
        self,
        task_run_id: str,
        *,
        max_transitions: int = 10_000,
        max_workflow_frontiers: int = 1_000,
    ) -> EpisodeTaskResult:
        with self._lock:
            for _ in range(max_transitions):
                state = self.ledger.latest(task_run_id)
                if state is None:
                    raise TaskOrchestratorError(
                        f"Unknown task run {task_run_id!r}; call start first"
                    )
                if state.status is not TaskRunStatus.RUNNING:
                    return self._result(state)
                next_state = self._transition(state, max_workflow_frontiers=max_workflow_frontiers)
                if next_state == state:
                    raise TaskOrchestratorError("task transition made no durable progress")
            raise TaskOrchestratorError(
                f"task {task_run_id!r} exceeded max_transitions={max_transitions}"
            )

    def _transition(self, state: TaskRunState, *, max_workflow_frontiers: int) -> TaskRunState:
        if state.active_intent is not None:
            return self._run_active_workflow(state, max_workflow_frontiers=max_workflow_frontiers)
        if state.planner_calls_started >= state.max_planner_calls:
            return self._commit(
                state,
                reason="planner_budget_exhausted",
                status=TaskRunStatus.EXHAUSTED,
                terminal_reason="planner call budget exhausted",
            )
        return self._plan_next_intent(state)

    def _plan_next_intent(self, state: TaskRunState) -> TaskRunState:
        call_id = state.pending_planner_call_id or (
            f"{state.task_run_id}_planner_{state.planner_calls_started + 1:03d}"
        )
        if state.pending_planner_call_id is None:
            state = self._commit(
                state,
                reason="planner_call_started",
                planner_calls_started=state.planner_calls_started + 1,
                pending_planner_call_id=call_id,
            )
        context = EpisodePlanningContext(
            planner_call_id=call_id,
            task_run_id=state.task_run_id,
            episode_id=state.episode_id,
            task=state.task,
            intent_index=len(state.intents),
            prior_intents=state.intents,
            prior_outcomes=state.outcomes,
            embodied_state_revision=self.runtime.state_reducer.state.revision,
            compact_state=self.runtime.state_reducer.state.to_mapping(),
            remaining_intents=state.max_intents - len(state.intents),
            remaining_planner_calls=(state.max_planner_calls - state.planner_calls_started),
            scene_image_path=state.scene_image_path,
        )
        planner_reservation: RunBudgetReservation | None = None
        authority = self.runtime.run_budget_authority
        if authority is not None:
            assert self.planner_budget is not None
            try:
                planner_reservation = authority.reserve(
                    operation_id=f"task-planner:{call_id}",
                    requested=RunBudgetVector(
                        model_calls=(
                            self.planner_budget.model_call_grant
                            if self.planner_uses_model_calls
                            else 0
                        ),
                        tokens=(
                            self.planner_budget.token_grant
                            if self.planner_uses_model_calls
                            else 0
                        ),
                        wall_time_s=self.planner_budget.wall_time_ms / 1000.0,
                    ),
                    binding={
                        "kind": "task_planner",
                        "uses_model_calls": self.planner_uses_model_calls,
                        "context": context.model_dump(mode="json"),
                        "planner_budget": self.planner_budget.model_dump(mode="json"),
                    },
                )
            except RunBudgetExceededError as exc:
                return self._commit(
                    state,
                    reason="manifest_planner_budget_exhausted",
                    pending_planner_call_id=None,
                    status=TaskRunStatus.EXHAUSTED,
                    terminal_reason=f"run_budget_exhausted:{exc}",
                )
        try:
            if authority is None:
                decision = self.planner.next_intent(context)
            else:
                assert self.planner_budget is not None
                assert isinstance(self.planner, BoundedIntentPlanner)
                decision = self.planner_calls.invoke(
                    context=context,
                    budget=self.planner_budget,
                    callback=lambda: self.planner.next_intent_bounded(
                        context,
                        max_tokens=self.planner_budget.token_grant,
                        max_model_calls=self.planner_budget.model_call_grant,
                        deadline_monotonic_s=self._planner_deadline_monotonic_s(),
                    ),
                )
            if not isinstance(decision, PlannerDecision):
                decision = PlannerDecision.model_validate(decision)
        finally:
            if (
                planner_reservation is not None
                and authority is not None
                and planner_reservation.status is not RunBudgetOperationStatus.COMPLETED
            ):
                # No provider-signed usage is available, so the explicit safe
                # policy is to charge the full typed grant even on failure.
                authority.complete(planner_reservation)
        if decision.kind is PlannerDecisionKind.DONE:
            return self._commit(
                state,
                reason="planner_done",
                pending_planner_call_id=None,
                status=TaskRunStatus.SUCCEEDED,
                terminal_reason=decision.reason or "planner declared task complete",
            )
        if decision.kind is PlannerDecisionKind.BLOCKED:
            return self._commit(
                state,
                reason="planner_blocked",
                pending_planner_call_id=None,
                status=TaskRunStatus.BLOCKED,
                terminal_reason=decision.reason,
            )
        assert decision.intent is not None
        if len(state.intents) >= state.max_intents:
            return self._commit(
                state,
                reason="intent_budget_exhausted",
                pending_planner_call_id=None,
                status=TaskRunStatus.EXHAUSTED,
                terminal_reason="planner requested an intent beyond the hard budget",
            )
        workflow_id = f"{state.task_run_id}_sg_{len(state.intents):03d}"
        return self._commit(
            state,
            reason="intent_planned",
            pending_planner_call_id=None,
            active_intent=decision.intent,
            active_workflow_id=workflow_id,
            intents=(*state.intents, decision.intent),
        )

    def _planner_deadline_monotonic_s(self) -> float:
        """Clamp the call deadline to its grant and the durable run deadline."""

        assert self.planner_budget is not None
        remaining_s = self.planner_budget.wall_time_ms / 1000.0
        authority = self.runtime.run_budget_authority
        if authority is not None:
            snapshot = authority.snapshot()
            remaining_s = min(
                remaining_s,
                max(snapshot.deadline_s - snapshot.observed_at_s, 0.0),
            )
        return time.monotonic() + remaining_s

    def _run_active_workflow(
        self, state: TaskRunState, *, max_workflow_frontiers: int
    ) -> TaskRunState:
        intent = state.active_intent
        workflow_id = state.active_workflow_id
        assert intent is not None and workflow_id is not None
        context = WorkflowAuthoringContext(
            authoring_call_id=f"{workflow_id}_author",
            task_run_id=state.task_run_id,
            episode_id=state.episode_id,
            workflow_id=workflow_id,
            task=state.task,
            intent=intent,
            prior_outcomes=state.outcomes,
            embodied_state_revision=self.runtime.state_reducer.state.revision,
            compact_state=self.runtime.state_reducer.state.to_mapping(),
            available_artifacts=tuple(
                AvailableArtifact(
                    workflow_id=record.workflow_id,
                    activation_id=record.activation_id,
                    attempt=record.attempt,
                    port=record.port,
                    schema_id=record.schema,
                    artifact_id=record.artifact_id,
                    content_digest=record.content_digest,
                )
                for record in self.runtime.data_plane.artifacts
            ),
        )
        blueprint = self.author.author(context)
        if not isinstance(blueprint, WorkflowBlueprint):
            raise TypeError("WorkflowAuthor must return WorkflowBlueprint")
        self.runtime.open_workflow(
            workflow_id=workflow_id,
            intent=intent,
            graph=blueprint.graph,
            external_refs=blueprint.external_refs,
            frontier=blueprint.frontier,
            manager_session=blueprint.manager_session,
            manager_invoker=blueprint.manager_invoker,
        )
        snapshot = self.runtime.run_until_terminal(
            workflow_id, max_frontiers=max_workflow_frontiers
        )
        if snapshot.status.value in {"created", "running"}:  # pragma: no cover
            raise TaskOrchestratorError("workflow runner returned before terminal state")
        outcome = self.runtime.close_workflow(workflow_id)
        if outcome.intent_id != intent.intent_id or outcome.intent_revision != intent.revision:
            raise TaskOrchestratorError("workflow outcome is not bound to its active intent")
        return self._commit(
            state,
            reason="intent_completed",
            active_intent=None,
            active_workflow_id=None,
            outcomes=(*state.outcomes, outcome),
        )

    def _commit(self, state: TaskRunState, *, reason: str, **updates: Any) -> TaskRunState:
        next_state = TaskRunState.model_validate(
            {
                **state.model_dump(mode="python"),
                **updates,
                "state_revision": state.state_revision + 1,
            }
        )
        self.ledger.append(next_state, reason=reason)
        return next_state

    @staticmethod
    def _result(state: TaskRunState) -> EpisodeTaskResult:
        return EpisodeTaskResult(
            task_run_id=state.task_run_id,
            episode_id=state.episode_id,
            task=state.task,
            status=state.status,
            intents=state.intents,
            outcomes=state.outcomes,
            reason=state.terminal_reason,
        )


class ScriptedIntentPlanner:
    """Deterministic planner fixture with call-id replay semantics."""

    def __init__(self, decisions: tuple[PlannerDecision, ...]) -> None:
        self._decisions = list(decisions)
        self._by_call: dict[str, PlannerDecision] = {}
        self.requests: list[EpisodePlanningContext] = []

    def next_intent(self, context: EpisodePlanningContext) -> PlannerDecision:
        self.requests.append(context)
        previous = self._by_call.get(context.planner_call_id)
        if previous is not None:
            return previous
        if not self._decisions:
            raise TaskOrchestratorError("scripted planner has no decision remaining")
        decision = self._decisions.pop(0)
        self._by_call[context.planner_call_id] = decision
        return decision

    def next_intent_bounded(
        self,
        context: EpisodePlanningContext,
        *,
        max_tokens: int,
        max_model_calls: int,
        deadline_monotonic_s: float,
    ) -> PlannerDecision:
        if isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if isinstance(max_model_calls, bool) or max_model_calls < 1:
            raise ValueError("max_model_calls must be a positive integer")
        if time.monotonic() >= deadline_monotonic_s:
            raise TimeoutError("bounded scripted intent planner deadline expired")
        return self.next_intent(context)


class FixedIntentPlanner(ScriptedIntentPlanner):
    """Manifest-authored, model-free intent schedule for a fixed baseline.

    Unlike ``ScriptedIntentPlanner`` (which remains a generic test double for
    either kind of provider), this class explicitly proves to the task
    orchestrator that no model/token boundary is crossed.  Its planner calls
    still receive durable call IDs and a wall-time grant.
    """

    uses_model_calls = False


class OutcomeAwareFixedIntentPlanner:
    """Model-free single-intent planner for a production fixed baseline.

    A scripted ``OPEN_INTENT, DONE`` sequence is unsafe outside tests: after a
    failed workflow its second decision would still mark the whole task as
    successful.  This planner closes only after an evidence-backed successful
    outcome and otherwise returns a terminal blocked decision with the exact
    workflow status/reason.
    """

    uses_model_calls = False

    def __init__(self, intent: SubgoalIntent) -> None:
        if not isinstance(intent, SubgoalIntent):
            raise TypeError("intent must be a SubgoalIntent")
        self.intent = intent
        self.requests: list[EpisodePlanningContext] = []

    def next_intent(self, context: EpisodePlanningContext) -> PlannerDecision:
        if not isinstance(context, EpisodePlanningContext):
            context = EpisodePlanningContext.model_validate(context)
        self.requests.append(context)
        if not context.prior_outcomes:
            return PlannerDecision(
                kind=PlannerDecisionKind.OPEN_INTENT,
                intent=self.intent,
                reason="open the manifest-pinned fixed intent",
            )
        if len(context.prior_outcomes) != 1 or len(context.prior_intents) != 1:
            return PlannerDecision(
                kind=PlannerDecisionKind.BLOCKED,
                reason="fixed baseline received more than one completed intent",
            )
        prior_intent = context.prior_intents[0]
        outcome = context.prior_outcomes[0]
        if (
            prior_intent.intent_id != self.intent.intent_id
            or prior_intent.revision != self.intent.revision
            or outcome.intent_id != self.intent.intent_id
            or outcome.intent_revision != self.intent.revision
        ):
            return PlannerDecision(
                kind=PlannerDecisionKind.BLOCKED,
                reason="fixed baseline outcome is bound to another intent identity",
            )
        if outcome.status is IntentStatus.SUCCEEDED:
            return PlannerDecision(
                kind=PlannerDecisionKind.DONE,
                reason="manifest-pinned intent completed successfully",
            )
        detail = outcome.reason or outcome.summary or "no terminal detail"
        return PlannerDecision(
            kind=PlannerDecisionKind.BLOCKED,
            reason=f"fixed intent ended {outcome.status.value}: {detail}",
        )

    def next_intent_bounded(
        self,
        context: EpisodePlanningContext,
        *,
        max_tokens: int,
        max_model_calls: int,
        deadline_monotonic_s: float,
    ) -> PlannerDecision:
        if isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if isinstance(max_model_calls, bool) or max_model_calls < 1:
            raise ValueError("max_model_calls must be a positive integer")
        if time.monotonic() >= deadline_monotonic_s:
            raise TimeoutError("fixed intent planner deadline expired")
        return self.next_intent(context)


class ReactivePlannerIntentAdapter:
    """Bridge the existing thin RoboMEx planner into the v2 intent boundary.

    This adapter preserves the legacy planner's skill-guided language behavior
    while replacing its old executor loop.  Prior v2 outcomes are projected as
    read-only history; physical execution remains exclusively in EpisodeRuntime.
    """

    uses_model_calls = True

    def __init__(self, planner: Any) -> None:
        from robomex.agents.planner import ReactivePlanner

        if not isinstance(planner, ReactivePlanner):
            raise TypeError("planner must be a ReactivePlanner")
        self.planner = planner
        self._by_call: dict[str, PlannerDecision] = {}

    def next_intent(self, context: EpisodePlanningContext) -> PlannerDecision:
        return self._next_intent(context, bounded=None)

    def next_intent_bounded(
        self,
        context: EpisodePlanningContext,
        *,
        max_tokens: int,
        max_model_calls: int,
        deadline_monotonic_s: float,
    ) -> PlannerDecision:
        if isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if isinstance(max_model_calls, bool) or max_model_calls < 1:
            raise ValueError("max_model_calls must be a positive integer")
        return self._next_intent(
            context,
            bounded=(max_tokens, max_model_calls),
            deadline_monotonic_s=deadline_monotonic_s,
        )

    def _next_intent(
        self,
        context: EpisodePlanningContext,
        *,
        bounded: tuple[int, int] | None,
        deadline_monotonic_s: float | None = None,
    ) -> PlannerDecision:
        from robomex.agents.planner import SubGoal, SubGoalResult
        from robomex.core.coder.trace import AgentTrace

        previous = self._by_call.get(context.planner_call_id)
        if previous is not None:
            return previous
        history = []
        for intent, outcome in zip(context.prior_intents, context.prior_outcomes, strict=True):
            succeeded = outcome.status is IntentStatus.SUCCEEDED
            history.append(
                SubGoalResult(
                    subgoal=SubGoal(
                        goal=intent.instruction,
                        postcondition=intent.success_rubric,
                    ),
                    trace=AgentTrace(
                        task=intent.instruction,
                        loaded_skill_ids=(),
                        turns=(),
                        success=succeeded,
                        metadata={
                            "terminal_result": {
                                "claim": outcome.summary,
                                "evidence_refs": list(outcome.evidence_refs),
                            }
                        },
                    ),
                    success=succeeded,
                    motion_attempted=bool(outcome.metrics.get("physical_actions", 0.0)),
                    authoring_status=outcome.status.value,
                    verification_status=outcome.status.value,
                    note=outcome.reason or outcome.summary,
                )
            )
        if bounded is None:
            subgoal = self.planner.next_subgoal(
                context.task,
                history,
                scene_image_path=context.scene_image_path,
            )
        else:
            subgoal = self.planner.next_subgoal_bounded(
                context.task,
                history,
                scene_image_path=context.scene_image_path,
                max_tokens=bounded[0],
                max_model_calls=bounded[1],
                deadline_monotonic_s=deadline_monotonic_s,
            )
        if subgoal is None:
            if self.planner.last_raw.strip().upper() == "DONE":
                decision = PlannerDecision(
                    kind=PlannerDecisionKind.DONE,
                    reason="legacy reactive planner declared DONE",
                )
            else:
                decision = PlannerDecision(
                    kind=PlannerDecisionKind.BLOCKED,
                    reason="legacy reactive planner returned an invalid response",
                )
        else:
            semantic_digest = hashlib.sha256(
                f"{subgoal.goal}\0{subgoal.postcondition}".encode()
            ).hexdigest()[:12]
            decision = PlannerDecision(
                kind=PlannerDecisionKind.OPEN_INTENT,
                intent=SubgoalIntent(
                    intent_id=f"intent_{context.intent_index:03d}_{semantic_digest}",
                    instruction=subgoal.goal,
                    success_rubric=(
                        subgoal.postcondition or f"independent evidence verifies: {subgoal.goal}"
                    ),
                    protected_invariants=(
                        "single_authoritative_writer",
                        "sealed_physical_actions_only",
                    ),
                ),
            )
        self._by_call[context.planner_call_id] = decision
        return decision


class ScriptedWorkflowAuthor:
    """Deterministic intent-id blueprint adapter used by replay and tests."""

    def __init__(self, blueprints: Mapping[str, WorkflowBlueprint]) -> None:
        self._blueprints = dict(blueprints)
        self.requests: list[WorkflowAuthoringContext] = []

    def author(self, context: WorkflowAuthoringContext) -> WorkflowBlueprint:
        self.requests.append(context)
        try:
            return self._blueprints[context.intent.intent_id]
        except KeyError as exc:
            raise TaskOrchestratorError(
                f"no workflow blueprint for intent {context.intent.intent_id!r}"
            ) from exc


def _digest(schema: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"schema": schema, "payload": payload},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AvailableArtifact",
    "BoundedIntentPlanner",
    "EpisodeOrchestrator",
    "EpisodePlanningContext",
    "EpisodeTaskResult",
    "IntentPlanner",
    "FixedIntentPlanner",
    "OutcomeAwareFixedIntentPlanner",
    "PlannerDecision",
    "PlannerDecisionKind",
    "PlannerCallBudget",
    "PlannerCallLedger",
    "ReactivePlannerIntentAdapter",
    "ScriptedIntentPlanner",
    "ScriptedWorkflowAuthor",
    "TaskOrchestratorError",
    "TaskRunState",
    "TaskRunStatus",
    "TaskStateCommit",
    "TaskStateConflictError",
    "TaskStateLedger",
    "WorkflowAuthor",
    "WorkflowAuthoringContext",
    "WorkflowBlueprint",
]
