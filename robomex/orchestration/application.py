"""Runnable task-level application boundary for RoboMEx v2.

The lower layers deliberately stay dependency-injected.  This module provides
the missing production composition: a completion-policy-backed bounded Manager
invoker, a Manager workflow author that compiles strict graph JSON, manifest
graph admission, and a small facade that drives :class:`EpisodeOrchestrator`.
"""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from robomex.core.coder import BoundedCompletionPolicy
from robomex.core.token_budget import (
    conservative_chat_prompt_tokens,
)
from robomex.data import ResolvedArtifactRef
from robomex.elastic import ComposableFrontier, ElasticGraphCompiler, ElasticGraphSpec
from robomex.evolution import RunManifest
from robomex.orchestration.bootstrap import V2Application
from robomex.orchestration.manager import (
    ManagerAction,
    ManagerCallKind,
    ManagerDecision,
    ManagerInvocation,
    ManagerInvoker,
    ManagerLimits,
    ManagerSnapshot,
    SwarmManagerSession,
)
from robomex.orchestration.run_budget import (
    RunBudgetAuthority,
    RunBudgetOperationStatus,
    RunBudgetReservation,
    RunBudgetVector,
)
from robomex.orchestration.task_orchestrator import (
    EpisodeOrchestrator,
    EpisodeTaskResult,
    IntentPlanner,
    PlannerCallBudget,
    TaskRunState,
    WorkflowAuthor,
    WorkflowAuthoringContext,
    WorkflowBlueprint,
)


class V2ApplicationError(RuntimeError):
    """The v2 task facade or one of its admitted authoring boundaries failed."""


class ManagerResponseError(V2ApplicationError, ValueError):
    """A Manager policy returned malformed or conflicting structured output."""


class TaskOrchestrationLimits(BaseModel):
    """Manifest-pinnable task-level loop limits for the v2 facade."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    schema_version: Literal["robomex.task_orchestration_limits.v1"] = (
        "robomex.task_orchestration_limits.v1"
    )
    max_intents: int = Field(default=8, ge=1)
    max_planner_calls: int | None = Field(default=None, ge=1)
    max_workflow_frontiers: int = Field(default=1_000, ge=1)


class ManagerPolicyCallBudget(BaseModel):
    """Manifest-pinnable wall envelope for one Manager policy API call.

    Manager model-call count is fixed at one and its total token grant is
    already sealed by :class:`ManagerLimits.max_tokens_per_call`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    schema_version: Literal["robomex.manager_policy_call_budget.v1"] = (
        "robomex.manager_policy_call_budget.v1"
    )
    wall_time_ms: int = Field(default=5_000, ge=1)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class PolicyManagerInvoker:
    """One-shot structured Manager adapter with durable call-id replay.

    The policy never receives a mutable chat transcript.  The exact invocation
    is serialized into one prompt, the response must validate as
    :class:`ManagerDecision`, and a content-addressed local record prevents a
    post-response process restart from spending another model call.
    """

    def __init__(
        self,
        policy: BoundedCompletionPolicy,
        *,
        ledger_root: str | Path,
        system_prompt: str | None = None,
        run_budget_authority: RunBudgetAuthority | None = None,
        wall_time_ms: int = 5_000,
    ) -> None:
        if not callable(getattr(policy, "complete_bounded", None)):
            raise TypeError(
                "Manager policy must implement complete_bounded(prompt, max_tokens=...)"
            )
        try:
            policy_parameters = inspect.signature(policy.complete_bounded).parameters.values()
        except (TypeError, ValueError) as exc:
            raise TypeError("Manager bounded policy signature is not inspectable") from exc
        if not any(
            parameter.name == "deadline_monotonic_s"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in policy_parameters
        ):
            raise TypeError("Manager bounded policy must enforce deadline_monotonic_s")
        self.policy = policy
        if isinstance(wall_time_ms, bool) or wall_time_ms < 1:
            raise ValueError("wall_time_ms must be a positive integer")
        self.wall_time_ms = wall_time_ms
        self.ledger_root = Path(ledger_root)
        self.ledger_root.mkdir(parents=True, exist_ok=True)
        self.system_prompt = system_prompt or (
            "You are the bounded RoboMEx v2 Swarm Manager. Return exactly one JSON "
            "ManagerDecision with keys action, summary, payload, candidate_delta, and "
            "tokens_used. Never emit executable robot commands. For initial authoring, "
            "action must be author_scaffold and payload.graph must be a complete "
            "robomex.elastic_graph.v2 mapping. Graph nodes may only reference cataloged "
            "runner_ref values and Agents may only propose artifacts; physical effects "
            "belong to system_action nodes."
        )
        self._lock = threading.RLock()
        self._run_budget_authority: RunBudgetAuthority | None = None
        if run_budget_authority is not None:
            self.bind_run_budget_authority(run_budget_authority)

    @property
    def manages_run_model_budget(self) -> bool:
        """Whether this invoker owns model/tokens accounting for its API call."""

        return self._run_budget_authority is not None

    @property
    def call_budget(self) -> ManagerPolicyCallBudget:
        """Return the exact typed policy envelope sealed into operation bindings."""

        return ManagerPolicyCallBudget(wall_time_ms=self.wall_time_ms)

    def bind_run_budget_authority(self, authority: RunBudgetAuthority) -> None:
        if not isinstance(authority, RunBudgetAuthority):
            raise TypeError("authority must be a RunBudgetAuthority")
        if self._run_budget_authority is not None and self._run_budget_authority is not authority:
            raise V2ApplicationError(
                "PolicyManagerInvoker is already bound to another RunBudgetAuthority"
            )
        self._run_budget_authority = authority

    def invoke(self, request: ManagerInvocation) -> ManagerDecision:
        if not isinstance(request, ManagerInvocation):
            request = ManagerInvocation.model_validate(request)
        request_payload = request.model_dump(mode="json")
        request_digest = _canonical_digest(request_payload)
        path = self._record_path(request.invocation_id)
        with self._lock, self._process_lock(path):
            previous = self._read_record(path)
            if previous is not None:
                if (
                    previous.get("invocation_id") != request.invocation_id
                    or previous.get("request_digest") != request_digest
                ):
                    raise ManagerResponseError(
                        "Manager invocation ID was rebound to different content"
                    )
                status = previous.get("status", "completed" if "decision" in previous else None)
                reservation = self._reserve_run_budget(request, request_digest)
                if status == "completed":
                    decision = ManagerDecision.model_validate(previous["decision"])
                    self._settle_run_budget(
                        reservation,
                        tokens=decision.tokens_used,
                    )
                    return decision
                self._settle_run_budget(
                    reservation,
                    tokens=request.token_limit,
                )
                raise ManagerResponseError(
                    "Manager invocation cannot be re-entered after an uncertain "
                    f"provider boundary ({status})"
                )

            prompt_payload = json.dumps(
                request_payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            prompt = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Produce one bounded ManagerDecision for this immutable request:\n"
                        + prompt_payload
                    ),
                },
            ]
            prompt_tokens = conservative_chat_prompt_tokens(prompt)
            output_ceiling = request.token_limit - prompt_tokens
            if output_ceiling < 1:
                raise ManagerResponseError(
                    "Manager prompt exceeds the total call token grant before API entry"
                )
            reservation = self._reserve_run_budget(request, request_digest)
            if reservation is not None and reservation.status is RunBudgetOperationStatus.COMPLETED:
                raise ManagerResponseError(
                    "Manager budget is already charged but no durable response exists"
                )
            base_record = {
                "schema": "robomex.manager_policy_call.v1",
                "invocation_id": request.invocation_id,
                "request_digest": request_digest,
                "prompt_tokens": prompt_tokens,
                "output_token_ceiling": output_ceiling,
            }
            try:
                self._write_record(path, {**base_record, "status": "started"})
            except Exception:
                if (
                    reservation is not None
                    and self._run_budget_authority is not None
                    and reservation.status is RunBudgetOperationStatus.RESERVED
                ):
                    self._run_budget_authority.release(reservation)
                raise

            try:
                deadline_monotonic_s = self._deadline_monotonic_s()
                raw = str(
                    self.policy.complete_bounded(
                        prompt,
                        max_tokens=output_ceiling,
                        deadline_monotonic_s=deadline_monotonic_s,
                    )
                    or ""
                )
                decision = self._parse_decision(raw)
                # The provider exposes no signed total usage and may spend
                # hidden reasoning tokens. Charge the full sealed call grant.
                accounted_tokens = request.token_limit
                decision = decision.model_copy(update={"tokens_used": accounted_tokens})
                self._write_record(
                    path,
                    {
                        **base_record,
                        "status": "completed",
                        "accounted_tokens": accounted_tokens,
                        "decision": decision.model_dump(mode="json"),
                    },
                )
            except BaseException as exc:
                try:
                    self._write_record(
                        path,
                        {
                            **base_record,
                            "status": "failed",
                            "accounted_tokens": request.token_limit,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                finally:
                    self._settle_run_budget(
                        reservation,
                        tokens=request.token_limit,
                    )
                raise
            self._settle_run_budget(reservation, tokens=accounted_tokens)
            return decision

    def _deadline_monotonic_s(self) -> float:
        remaining_s = self.wall_time_ms / 1000.0
        if self._run_budget_authority is not None:
            snapshot = self._run_budget_authority.snapshot()
            remaining_s = min(
                remaining_s,
                max(snapshot.deadline_s - snapshot.observed_at_s, 0.0),
            )
        if remaining_s <= 0:
            raise TimeoutError("Manager wall-time deadline expired before API entry")
        return time.monotonic() + remaining_s

    def _reserve_run_budget(
        self,
        request: ManagerInvocation,
        request_digest: str,
    ) -> RunBudgetReservation | None:
        authority = self._run_budget_authority
        if authority is None:
            return None
        return authority.reserve(
            operation_id=f"manager-policy:{request.invocation_id}",
            requested=RunBudgetVector(
                model_calls=1,
                tokens=request.token_limit,
                wall_time_s=self.wall_time_ms / 1000.0,
            ),
            binding={
                "kind": "manager_policy_invocation",
                "invocation_id": request.invocation_id,
                "request_digest": request_digest,
                "wall_time_ms": self.wall_time_ms,
            },
        )

    def _settle_run_budget(
        self,
        reservation: RunBudgetReservation | None,
        *,
        tokens: int,
    ) -> None:
        if reservation is None or self._run_budget_authority is None:
            return
        self._run_budget_authority.complete(
            reservation,
            settlement=RunBudgetVector(
                model_calls=1,
                tokens=tokens,
                wall_time_s=reservation.requested.wall_time_s,
            ),
        )

    @staticmethod
    def _parse_decision(raw: str) -> ManagerDecision:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:].lstrip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManagerResponseError("Manager response is not one JSON object") from exc
        if not isinstance(value, Mapping):
            raise ManagerResponseError("Manager response must be a JSON object")
        if value.get("tool") == "finish" and isinstance(value.get("args"), Mapping):
            args = value["args"]
            nested = args.get("decision") or args.get("result")
            value = nested if isinstance(nested, Mapping) else args
        try:
            return ManagerDecision.model_validate(value)
        except Exception as exc:
            raise ManagerResponseError("Manager response violates ManagerDecision") from exc

    def _record_path(self, invocation_id: str) -> Path:
        digest = hashlib.sha256(invocation_id.encode()).hexdigest()
        return self.ledger_root / f"{digest}.json"

    @staticmethod
    def _read_record(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManagerResponseError("Manager call record is unreadable") from exc
        if not isinstance(value, dict) or value.get("schema") != "robomex.manager_policy_call.v1":
            raise ManagerResponseError("Manager call record has an invalid schema")
        return value

    @staticmethod
    def _write_record(path: Path, value: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
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

    @contextmanager
    def _process_lock(self, record_path: Path):
        lock_path = record_path.with_suffix(record_path.suffix + ".lock")
        with lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class PinnedBaselineManagerInvoker:
    """Model-free Manager for the immutable, non-evolving v2 baseline.

    The initial call publishes one manifest-pinned graph scaffold.  Exceptional
    wake-ups terminate safely instead of inventing a patch.  Replacing this
    invoker with ``PolicyManagerInvoker`` later enables bounded frontier repair
    without changing the graph/runtime/action authority interfaces used by the
    baseline.
    """

    manages_run_model_budget = True

    def __init__(
        self,
        graph,
        *,
        frontier: ComposableFrontier | None = None,
    ) -> None:
        from robomex.elastic import CompiledElasticGraph

        if not isinstance(graph, CompiledElasticGraph):
            raise TypeError("graph must be a CompiledElasticGraph")
        if frontier is not None:
            if not isinstance(frontier, ComposableFrontier):
                raise TypeError("frontier must be a ComposableFrontier or None")
            if (
                frontier.graph_id != graph.spec.graph_id
                or frontier.revision != graph.spec.revision
                or frontier.graph_digest != graph.digest
            ):
                raise V2ApplicationError(
                    "baseline Manager frontier is not bound to its compiled graph"
                )
        self.graph = graph
        self.frontier = frontier
        self._run_budget_authority: RunBudgetAuthority | None = None

    def bind_run_budget_authority(self, authority: RunBudgetAuthority) -> None:
        """Accept the run authority without reserving model/token dimensions."""

        if not isinstance(authority, RunBudgetAuthority):
            raise TypeError("authority must be a RunBudgetAuthority")
        if (
            self._run_budget_authority is not None
            and self._run_budget_authority is not authority
        ):
            raise V2ApplicationError(
                "PinnedBaselineManagerInvoker is already bound to another run"
            )
        self._run_budget_authority = authority

    def invoke(self, request: ManagerInvocation) -> ManagerDecision:
        if not isinstance(request, ManagerInvocation):
            request = ManagerInvocation.model_validate(request)
        if request.kind is ManagerCallKind.INITIAL:
            if request.signal.value != "intent_authoring":
                raise ManagerResponseError(
                    "baseline Manager initial call requires intent_authoring"
                )
            payload: dict[str, Any] = {
                "graph": self.graph.spec.model_dump(mode="json"),
                "external_refs": {},
            }
            if self.frontier is not None:
                payload["frontier"] = self.frontier.model_dump(mode="json")
            return ManagerDecision(
                action=ManagerAction.AUTHOR_SCAFFOLD,
                summary="publish the manifest-pinned baseline graph",
                payload=payload,
                tokens_used=0,
            )
        return ManagerDecision(
            action=ManagerAction.CLOSE,
            summary=(
                "baseline has no mutation policy; close after exceptional "
                f"signal {request.signal.value}"
            ),
            tokens_used=0,
        )


class ManagerWorkflowAuthor:
    """Compile a bounded Manager's initial scaffold into a WorkflowBlueprint."""

    def __init__(
        self,
        invoker: ManagerInvoker,
        *,
        limits: ManagerLimits | None = None,
        catalog_refs: tuple[str, ...] = (),
        compiler: ElasticGraphCompiler | None = None,
    ) -> None:
        if not isinstance(invoker, ManagerInvoker):
            raise TypeError("invoker does not implement ManagerInvoker")
        self.invoker = invoker
        self.limits = limits or ManagerLimits()
        self.catalog_refs = tuple(catalog_refs)
        self.compiler = compiler or ElasticGraphCompiler()

    def bind_run_budget_authority(self, authority: RunBudgetAuthority) -> None:
        binder = getattr(self.invoker, "bind_run_budget_authority", None)
        if not callable(binder):
            raise V2ApplicationError(
                "ManagerWorkflowAuthor invoker cannot bind manifest RunBudgets"
            )
        binder(authority)

    def author(self, context: WorkflowAuthoringContext) -> WorkflowBlueprint:
        session = SwarmManagerSession(
            session_id=f"manager_{context.workflow_id}",
            episode_id=context.episode_id,
            workflow_id=context.workflow_id,
            intent_id=context.intent.intent_id,
            limits=self.limits,
        )
        snapshot = ManagerSnapshot(
            snapshot_id=context.authoring_call_id,
            episode_id=context.episode_id,
            workflow_id=context.workflow_id,
            graph_id=f"pending_{context.workflow_id}",
            graph_revision=1,
            state_revision=context.embodied_state_revision,
            triggering_event={
                "event_id": context.authoring_call_id,
                "kind": "intent_authoring",
                "intent": context.intent.model_dump(mode="json"),
            },
            compact_state=context.compact_state,
            artifact_refs=tuple(item.artifact_id for item in context.available_artifacts),
            catalog_refs=self.catalog_refs,
        )
        step = session.author(snapshot, self.invoker)
        decision = step.decision
        if decision is None:
            raise V2ApplicationError(step.reason or "Manager authoring failed")
        if decision.action is not ManagerAction.AUTHOR_SCAFFOLD:
            raise V2ApplicationError("Initial Manager decision must use author_scaffold")
        raw_graph = decision.payload.get("graph")
        if not isinstance(raw_graph, Mapping):
            raise V2ApplicationError("author_scaffold payload requires graph")
        graph = self.compiler.compile(ElasticGraphSpec.model_validate(raw_graph))

        available = {
            (item.artifact_id, item.content_digest): item.ref
            for item in context.available_artifacts
        }
        raw_external = decision.payload.get("external_refs", {})
        if not isinstance(raw_external, Mapping):
            raise V2ApplicationError("external_refs must be a mapping")
        external: dict[str, ResolvedArtifactRef] = {}
        for name, value in raw_external.items():
            if not isinstance(name, str) or not name.strip():
                raise V2ApplicationError("external ref names must be non-empty")
            try:
                ref = ResolvedArtifactRef.from_any(value)
            except (TypeError, ValueError) as exc:
                raise V2ApplicationError("Manager emitted an invalid artifact ref") from exc
            identity = (ref.artifact_id, ref.content_digest)
            if identity not in available:
                raise V2ApplicationError(
                    "Manager external_refs may only cite the admitted artifact inventory"
                )
            external[name] = available[identity]

        raw_frontier = decision.payload.get("frontier")
        frontier = (
            ComposableFrontier.model_validate(raw_frontier) if raw_frontier is not None else None
        )
        if frontier is not None and (
            frontier.graph_id != graph.spec.graph_id
            or frontier.revision != graph.spec.revision
            or frontier.graph_digest != graph.digest
        ):
            raise V2ApplicationError("Manager frontier is not bound to the compiled initial graph")
        return WorkflowBlueprint(
            graph=graph,
            external_refs=external,
            frontier=frontier,
            manager_session=step.session,
            manager_invoker=self.invoker,
        )


class ManifestPinnedWorkflowAuthor:
    """Reject initial graphs not sealed into the run manifest."""

    def __init__(self, inner: WorkflowAuthor, manifest: RunManifest) -> None:
        if not isinstance(inner, WorkflowAuthor):
            raise TypeError("inner does not implement WorkflowAuthor")
        self.inner = inner
        allowed = {manifest.graph_digest}
        configured = manifest.metadata.get("allowed_initial_graph_digests")
        if configured is not None:
            if not isinstance(configured, (list, tuple)) or not all(
                isinstance(value, str) for value in configured
            ):
                raise V2ApplicationError("allowed_initial_graph_digests must be a JSON string list")
            allowed.update(configured)
        if any(not value.startswith("sha256:") or len(value) != 71 for value in allowed):
            raise V2ApplicationError("manifest contains an invalid graph digest")
        self.allowed = frozenset(allowed)

    def bind_run_budget_authority(self, authority: RunBudgetAuthority) -> None:
        binder = getattr(self.inner, "bind_run_budget_authority", None)
        if not callable(binder):
            raise V2ApplicationError("Workflow author cannot bind manifest RunBudgets")
        binder(authority)

    def author(self, context: WorkflowAuthoringContext) -> WorkflowBlueprint:
        blueprint = self.inner.author(context)
        digest = f"sha256:{blueprint.graph.digest}"
        if digest not in self.allowed:
            raise V2ApplicationError(f"Initial workflow graph {digest} is not manifest-pinned")
        return blueprint


@dataclass(frozen=True)
class V2AgentConfig:
    application: V2Application
    planner: IntentPlanner
    author: WorkflowAuthor
    max_intents: int = 8
    max_planner_calls: int | None = None
    max_workflow_frontiers: int = 1_000
    planner_budget: PlannerCallBudget = field(default_factory=PlannerCallBudget)

    def __post_init__(self) -> None:
        if not isinstance(self.application, V2Application):
            raise TypeError("application must be a V2Application")
        if not isinstance(self.planner, IntentPlanner):
            raise TypeError("planner does not implement IntentPlanner")
        if not isinstance(self.author, WorkflowAuthor):
            raise TypeError("author does not implement WorkflowAuthor")
        if not isinstance(self.planner_budget, PlannerCallBudget):
            raise TypeError("planner_budget must be a PlannerCallBudget")
        if self.max_intents < 1 or self.max_workflow_frontiers < 1:
            raise ValueError("v2 task limits must be positive")
        if self.max_planner_calls is not None and self.max_planner_calls < 1:
            raise ValueError("max_planner_calls must be positive")

    @property
    def task_limits(self) -> TaskOrchestrationLimits:
        return TaskOrchestrationLimits(
            max_intents=self.max_intents,
            max_planner_calls=self.max_planner_calls,
            max_workflow_frontiers=self.max_workflow_frontiers,
        )


class RoboMExV2Agent:
    """Normal runnable facade for one immutable v2 application/run manifest."""

    def __init__(self, config: V2AgentConfig) -> None:
        self.config = config
        configured_budget = config.application.manifest.metadata.get("planner_call_budget")
        try:
            manifest_planner_budget = (
                PlannerCallBudget()
                if configured_budget is None
                else PlannerCallBudget.model_validate(configured_budget)
            )
        except Exception as exc:
            raise V2ApplicationError("manifest planner_call_budget is malformed") from exc
        if config.planner_budget != manifest_planner_budget:
            raise V2ApplicationError(
                "V2AgentConfig planner_budget differs from the manifest-pinned grant"
            )
        configured_task_limits = config.application.manifest.metadata.get(
            "task_orchestration_limits"
        )
        try:
            manifest_task_limits = (
                TaskOrchestrationLimits()
                if configured_task_limits is None
                else TaskOrchestrationLimits.model_validate(configured_task_limits)
            )
        except Exception as exc:
            raise V2ApplicationError("manifest task_orchestration_limits is malformed") from exc
        if config.task_limits != manifest_task_limits:
            raise V2ApplicationError(
                "V2AgentConfig task limits differ from the manifest-pinned limits"
            )
        configured_manager_limits = config.application.manifest.metadata.get("manager_limits")
        try:
            manifest_manager_limits = (
                ManagerLimits()
                if configured_manager_limits is None
                else ManagerLimits.model_validate(configured_manager_limits)
            )
        except Exception as exc:
            raise V2ApplicationError("manifest manager_limits is malformed") from exc
        manager_author = config.author
        if isinstance(manager_author, ManifestPinnedWorkflowAuthor):
            manager_author = manager_author.inner
        if isinstance(manager_author, ManagerWorkflowAuthor):
            if manager_author.limits != manifest_manager_limits:
                raise V2ApplicationError(
                    "ManagerWorkflowAuthor limits differ from the manifest-pinned limits"
                )
        elif configured_manager_limits is not None:
            raise V2ApplicationError("manifest manager_limits requires a ManagerWorkflowAuthor")
        configured_manager_call_budget = config.application.manifest.metadata.get(
            "manager_policy_call_budget"
        )
        try:
            manifest_manager_call_budget = (
                ManagerPolicyCallBudget()
                if configured_manager_call_budget is None
                else ManagerPolicyCallBudget.model_validate(configured_manager_call_budget)
            )
        except Exception as exc:
            raise V2ApplicationError("manifest manager_policy_call_budget is malformed") from exc
        if isinstance(manager_author, ManagerWorkflowAuthor):
            manager_invoker = manager_author.invoker
            if isinstance(manager_invoker, PolicyManagerInvoker):
                if manager_invoker.call_budget != manifest_manager_call_budget:
                    raise V2ApplicationError(
                        "PolicyManagerInvoker call budget differs from the manifest-pinned grant"
                    )
            elif configured_manager_call_budget is not None:
                raise V2ApplicationError(
                    "manifest manager_policy_call_budget requires a PolicyManagerInvoker"
                )
        elif configured_manager_call_budget is not None:
            raise V2ApplicationError(
                "manifest manager_policy_call_budget requires a ManagerWorkflowAuthor"
            )
        authority = config.application.run_budget_authority
        binder = getattr(config.author, "bind_run_budget_authority", None)
        if authority is not None and callable(binder):
            binder(authority)
        self.author = ManifestPinnedWorkflowAuthor(config.author, config.application.manifest)
        self.orchestrator = EpisodeOrchestrator(
            config.application.episode,
            planner=config.planner,
            author=self.author,
            planner_budget=config.planner_budget,
        )

    def start(
        self,
        *,
        task: str | None = None,
        task_run_id: str | None = None,
        scene_image_path: str | None = None,
    ) -> TaskRunState:
        instruction = task or self.config.application.manifest.task.instruction
        if instruction != self.config.application.manifest.task.instruction:
            raise V2ApplicationError("Runtime task differs from the immutable manifest task")
        return self.orchestrator.start(
            task_run_id=task_run_id or self.default_task_run_id,
            task=instruction,
            max_intents=self.config.max_intents,
            max_planner_calls=self.config.max_planner_calls,
            scene_image_path=scene_image_path,
        )

    def run(
        self,
        *,
        task: str | None = None,
        task_run_id: str | None = None,
        scene_image_path: str | None = None,
    ) -> EpisodeTaskResult:
        state = self.start(
            task=task,
            task_run_id=task_run_id,
            scene_image_path=scene_image_path,
        )
        return self.orchestrator.run(
            state.task_run_id,
            max_workflow_frontiers=self.config.max_workflow_frontiers,
        )

    @property
    def default_task_run_id(self) -> str:
        digest = hashlib.sha256(self.config.application.manifest.run_id.encode()).hexdigest()[:20]
        return f"run_{digest}"

    def close(self) -> None:
        self.config.application.episode.close_episode()


__all__ = [
    "ManagerPolicyCallBudget",
    "ManagerResponseError",
    "ManagerWorkflowAuthor",
    "ManifestPinnedWorkflowAuthor",
    "PolicyManagerInvoker",
    "PinnedBaselineManagerInvoker",
    "RoboMExV2Agent",
    "TaskOrchestrationLimits",
    "V2AgentConfig",
    "V2ApplicationError",
]
