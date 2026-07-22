"""Production bridge from Arena candidates to skill-retrieval coding workers.

The wrapped coding provider authors exactly one sealed ``MotionPlan`` emission.
This bridge publishes that emission through the episode data plane and builds
the compact ``ActionHypothesis`` exclusively from runtime-authored Arena
identity plus manifest-pinned scoring policy.  Model output never controls a
candidate ID, strategy, score, snapshot, frame, world, or resource binding.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from robomex.data import EpisodeDataPlane, ResolvedArtifactRef
from robomex.data.durability import atomic_write_json
from robomex.orchestration.actors import (
    ActorConflictError,
    ActorIsolation,
    ActorProfile,
    AgentProvider,
    InvocationSpec,
)
from robomex.orchestration.arena import (
    ARENA_RUNTIME_CONTEXT_METADATA_KEY,
    ActionHypothesis,
    ArenaMotionPreviewArtifact,
    ArenaRuntimeContextV1,
    _profile_coding_node_params,
)
from robomex.orchestration.episode import ActivationExecutionResult, ArtifactEmission
from robomex.runtime.action_protocol import AdmissionSnapshot, MotionPlan, validate_action_spec
from robomex.runtime.events import ControlOutcome

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DigestStr = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
_BRIDGE_RECORD_SCHEMA = "robomex.arena_coding_bridge_record.v1"
_PREVIEW_SCHEMA = "robomex.motion_preview.v1"
_DELEGATE_RUNTIME_METADATA_KEYS = frozenset(
    {
        "activation_id",
        "attempt",
        "episode_id",
        "graph_digest",
        "graph_id",
        "graph_revision",
        "node_params",
        "run_id",
        "workflow_id",
    }
)

_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


class ArenaCodingBridgeError(RuntimeError):
    """Base failure at the Arena-to-coding trust boundary."""


class ArenaCodingBridgeContractError(ArenaCodingBridgeError, ValueError):
    """Coding, preview, identity, or replay output violated the bridge contract."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class MotionPreviewFrame(_StrictModel):
    """One renderer-produced media item; it carries no execution authority."""

    view_id: NonEmptyStr
    media_type: NonEmptyStr
    media_digest: DigestStr
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    lineage: tuple[ResolvedArtifactRef, ...] = ()


class MotionPreviewResult(_StrictModel):
    """Deterministic renderer output consumed by the trusted bridge."""

    frames: tuple[MotionPreviewFrame, ...] = Field(min_length=1)
    terminal_position_m: tuple[float, float, float] | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> MotionPreviewResult:
        view_ids = [frame.view_id for frame in self.frames]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("motion preview view IDs must be unique")
        if self.terminal_position_m is not None and not all(
            math.isfinite(value) for value in self.terminal_position_m
        ):
            raise ValueError("terminal_position_m must be finite")
        return self


@runtime_checkable
class MotionPreviewRenderer(Protocol):
    """Read-only deterministic renderer; no provider, model, or robot handle is passed."""

    @property
    def renderer_id(self) -> str: ...

    def render(
        self,
        *,
        plan: MotionPlan,
        context: ArenaRuntimeContextV1,
    ) -> MotionPreviewResult | Mapping[str, Any]: ...


@dataclass(frozen=True)
class ArenaCodingWorkerRuntime:
    """Wrapper-owned actor state whose lifecycle is delegated unchanged."""

    profile: ActorProfile
    isolation: ActorIsolation
    delegate_runtime: Any


class ArenaCodingAgentProvider:
    """Wrap a proposal-only coding provider as a strict Arena hypothesis source."""

    def __init__(
        self,
        coding_provider: AgentProvider,
        *,
        data_plane: EpisodeDataPlane | None = None,
        renderer: MotionPreviewRenderer | None = None,
        artifacts_root: str | Path | None = None,
    ) -> None:
        if not isinstance(coding_provider, AgentProvider):
            raise TypeError("coding_provider must implement AgentProvider")
        if renderer is not None and not isinstance(renderer, MotionPreviewRenderer):
            raise TypeError("renderer must implement MotionPreviewRenderer")
        self._coding_provider = coding_provider
        self._renderer = renderer
        self._configured_root = Path(artifacts_root).resolve() if artifacts_root else None
        self._data_plane: EpisodeDataPlane | None = None
        self._record_root: Path | None = None
        self._lock = threading.RLock()
        if data_plane is not None:
            self.bind_episode_data_plane(data_plane)

    @property
    def data_plane(self) -> EpisodeDataPlane:
        if self._data_plane is None:
            raise ArenaCodingBridgeContractError(
                "Arena coding provider is not bound to an EpisodeDataPlane"
            )
        return self._data_plane

    @property
    def renderer(self) -> MotionPreviewRenderer | None:
        return self._renderer

    @property
    def coding_provider(self) -> AgentProvider:
        """Return the exact proposal provider wrapped by this Arena bridge."""

        return self._coding_provider

    def bind_episode_data_plane(self, data_plane: EpisodeDataPlane) -> None:
        if not isinstance(data_plane, EpisodeDataPlane):
            raise TypeError("data_plane must be an EpisodeDataPlane")
        with self._lock:
            if self._data_plane is data_plane:
                return
            if self._data_plane is not None:
                raise ArenaCodingBridgeContractError(
                    "Arena coding provider is already bound to another data plane"
                )
            binder = getattr(self._coding_provider, "bind_episode_data_plane", None)
            if callable(binder):
                binder(data_plane)
            data_plane.schema_registry.ensure(_PREVIEW_SCHEMA, ArenaMotionPreviewArtifact)
            self._data_plane = data_plane
            root = self._configured_root or (
                data_plane.episode_root / "providers" / "arena_coding_bridge"
            )
            self._record_root = root.resolve() / "idempotency"
            self._record_root.mkdir(parents=True, exist_ok=True)

    def spawn(
        self,
        profile: ActorProfile,
        isolation: ActorIsolation,
    ) -> ArenaCodingWorkerRuntime:
        _ = self.data_plane
        if profile.metadata.get("strict_runtime_context") is not True:
            raise ArenaCodingBridgeContractError(
                "Arena coding workers require profile strict_runtime_context=True"
            )
        delegate_runtime = self._coding_provider.spawn(profile, isolation)
        return ArenaCodingWorkerRuntime(
            profile=profile,
            isolation=isolation,
            delegate_runtime=delegate_runtime,
        )

    def invoke(
        self,
        runtime: ArenaCodingWorkerRuntime,
        spec: InvocationSpec,
    ) -> ActionHypothesis:
        context = self._trusted_context(runtime, spec)
        delegate_spec = self._delegate_spec(runtime, spec, context)
        trusted_input_lineage = self._trusted_input_lineage(delegate_spec)
        record_path = self._record_path(runtime, spec)
        with _exclusive_record_lock(record_path):
            previous = self._read_record(record_path)
            if previous is not None:
                self._validate_record_identity(
                    previous,
                    runtime,
                    spec,
                    delegate_spec,
                    context,
                )
                if previous.get("status") == "committed":
                    return self._replay_hypothesis(previous, context)
                if previous.get("status") != "pending":
                    raise ArenaCodingBridgeContractError(
                        "Arena coding bridge record has an unknown status"
                    )
            else:
                self._write_record(
                    record_path,
                    self._base_record(
                        runtime,
                        spec,
                        delegate_spec,
                        context,
                        status="pending",
                    ),
                )

            raw_result = self._coding_provider.invoke(
                runtime.delegate_runtime,
                delegate_spec,
            )
            emission = self._sole_motion_emission(raw_result, delegate_spec)
            plan = self._validate_plan(emission.payload, context=context)
            plan_ref, evidence_refs = self._publish_plan(
                plan=plan,
                trusted_input_lineage=trusted_input_lineage,
                context=context,
            )
            render_refs, terminal_position = self._render_and_publish(
                plan=plan,
                plan_ref=plan_ref,
                context=context,
            )
            hypothesis = self._hypothesis(
                plan=plan,
                plan_ref=plan_ref,
                evidence_refs=evidence_refs,
                render_refs=render_refs,
                terminal_position_m=terminal_position,
                context=context,
            )
            committed = self._base_record(
                runtime,
                spec,
                delegate_spec,
                context,
                status="committed",
            )
            committed["hypothesis"] = hypothesis.model_dump(mode="json")
            committed["hypothesis_digest"] = _mapping_digest(committed["hypothesis"])
            self._write_record(record_path, committed)
            return hypothesis

    def suspend(self, runtime: ArenaCodingWorkerRuntime) -> None:
        self._coding_provider.suspend(runtime.delegate_runtime)

    def resume(self, runtime: ArenaCodingWorkerRuntime) -> None:
        self._coding_provider.resume(runtime.delegate_runtime)

    def retire(self, runtime: ArenaCodingWorkerRuntime) -> None:
        self._coding_provider.retire(runtime.delegate_runtime)

    def _trusted_context(
        self,
        runtime: ArenaCodingWorkerRuntime,
        spec: InvocationSpec,
    ) -> ArenaRuntimeContextV1:
        spoofed_metadata = sorted(
            set(spec.metadata).intersection(_DELEGATE_RUNTIME_METADATA_KEYS)
        )
        if spoofed_metadata:
            raise ArenaCodingBridgeContractError(
                "Arena invocation metadata attempted to inject delegate runtime identity: "
                + ", ".join(spoofed_metadata)
            )
        raw = spec.metadata.get(ARENA_RUNTIME_CONTEXT_METADATA_KEY)
        if not isinstance(raw, Mapping):
            raise ArenaCodingBridgeContractError(
                "Arena invocation is missing runtime-authored context metadata"
            )
        try:
            context = ArenaRuntimeContextV1.model_validate(raw)
        except (TypeError, ValueError) as exc:
            raise ArenaCodingBridgeContractError("Arena runtime context is malformed") from exc
        if context.episode_id != self.data_plane.episode_id:
            raise ArenaCodingBridgeContractError("Arena context crossed episode identity")
        if context.graph_digest is None:
            raise ArenaCodingBridgeContractError(
                "Arena coding context is missing the runtime-authored graph digest"
            )
        if context.coding_activation_id is None:
            raise ArenaCodingBridgeContractError(
                "Arena coding context is missing its manifest-pinned activation ID"
            )
        try:
            expected_node_params = _profile_coding_node_params(
                runtime.profile,
                context.coding_activation_id,
            )
        except Exception as exc:
            raise ArenaCodingBridgeContractError(
                "Arena coding profile has no exact manifest-pinned node config"
            ) from exc
        if expected_node_params != context.coding_node_params:
            raise ArenaCodingBridgeContractError(
                "Arena coding node params differ from the manifest-pinned profile"
            )
        required_tcp = context.hypothesis_config.required_tcp_frame_id
        node_tcp = expected_node_params.get("tcp_frame_id")
        if required_tcp is None:
            raise ArenaCodingBridgeContractError(
                "Arena motion hypothesis must pin required_tcp_frame_id"
            )
        for field_name in ("plan_kind", "tcp_frame_id", "planner_backend"):
            if not isinstance(expected_node_params.get(field_name), str):
                raise ArenaCodingBridgeContractError(
                    "Arena motion coding profile must pin plan_kind, TCP frame, "
                    "and planner backend"
                )
        if required_tcp != node_tcp:
            raise ArenaCodingBridgeContractError(
                "Arena hypothesis and coding node pin different TCP frames"
            )
        node_robot_model = expected_node_params.get("robot_model_digest")
        if (
            node_robot_model is not None
            and node_robot_model != context.robot_model_digest
        ):
            raise ArenaCodingBridgeContractError(
                "Arena coding node and runtime context pin different robot models"
            )
        expected_actor_id = f"{context.arena_run_id}-{context.candidate_id}"
        if runtime.isolation.owner_actor_id != expected_actor_id:
            raise ArenaCodingBridgeContractError("Arena actor identity does not match context")
        if spec.invocation_id != f"{expected_actor_id}-invoke":
            raise ArenaCodingBridgeContractError("Arena invocation ID is not runtime-derived")
        if spec.effective_idempotency_key != expected_actor_id:
            raise ArenaCodingBridgeContractError("Arena idempotency key is not runtime-derived")
        if len(spec.output_contract) != 1 or set(spec.output_contract.values()) != {
            "robomex.motion_plan.v2"
        }:
            raise ArenaCodingBridgeContractError(
                "Arena coding bridge requires exactly one motion-plan output contract"
            )
        raw_snapshot = spec.inputs.get("snapshot")
        try:
            input_snapshot = ResolvedArtifactRef.from_any(raw_snapshot)
        except (TypeError, ValueError) as exc:
            raise ArenaCodingBridgeContractError(
                "Arena invocation snapshot input is not an explicit artifact ref"
            ) from exc
        if input_snapshot != context.snapshot_ref:
            raise ArenaCodingBridgeContractError(
                "Arena invocation snapshot input differs from trusted context"
            )
        resolved_snapshot = self.data_plane.resolve(input_snapshot)
        if resolved_snapshot.schema != "robomex.admission_snapshot.v1":
            raise ArenaCodingBridgeContractError(
                "Arena invocation snapshot has the wrong artifact schema"
            )
        try:
            snapshot = AdmissionSnapshot.model_validate(resolved_snapshot.payload)
        except (TypeError, ValueError) as exc:
            raise ArenaCodingBridgeContractError(
                "Arena invocation snapshot payload is malformed"
            ) from exc
        if (
            snapshot.world_id != context.world_id
            or snapshot.resource_id != context.resource_id
            or snapshot.config_digest != context.config_digest
        ):
            raise ArenaCodingBridgeContractError(
                "Arena invocation snapshot differs from trusted world/resource/config"
            )
        if context.risk_ref is not None:
            try:
                input_risk = ResolvedArtifactRef.from_any(spec.inputs.get("risk"))
            except (TypeError, ValueError) as exc:
                raise ArenaCodingBridgeContractError(
                    "Arena invocation risk input is not an explicit artifact ref"
                ) from exc
            if input_risk != context.risk_ref:
                raise ArenaCodingBridgeContractError(
                    "Arena invocation risk input differs from trusted context"
                )
            if self.data_plane.resolve(input_risk).schema != "robomex.risk_report.v1":
                raise ArenaCodingBridgeContractError(
                    "Arena invocation risk input has the wrong artifact schema"
                )
        for name, expected_schema in context.context_input_schemas.items():
            try:
                input_ref = ResolvedArtifactRef.from_any(spec.inputs.get(name))
            except (TypeError, ValueError) as exc:
                raise ArenaCodingBridgeContractError(
                    f"Arena context input {name!r} is not an explicit artifact ref"
                ) from exc
            if input_ref != context.context_input_refs[name]:
                raise ArenaCodingBridgeContractError(
                    f"Arena context input {name!r} differs from the admitted ref"
                )
            resolved = self.data_plane.resolve(input_ref)
            if resolved.schema != expected_schema:
                raise ArenaCodingBridgeContractError(
                    f"Arena context input {name!r} schema differs from the manifest"
                )
        return context

    @staticmethod
    def _delegate_spec(
        runtime: ArenaCodingWorkerRuntime,
        spec: InvocationSpec,
        context: ArenaRuntimeContextV1,
    ) -> InvocationSpec:
        if runtime.profile.metadata.get("strict_runtime_context") is not True:
            raise ArenaCodingBridgeContractError(
                "Arena coding delegate lost strict_runtime_context"
            )
        assert context.graph_digest is not None
        assert context.coding_activation_id is not None
        safe_metadata = {
            key: value
            for key, value in spec.metadata.items()
            if key != ARENA_RUNTIME_CONTEXT_METADATA_KEY
        }
        safe_metadata.update(
            {
                "episode_id": context.episode_id,
                "workflow_id": context.workflow_id,
                "activation_id": context.coding_activation_id,
                "attempt": context.command_attempt,
                "graph_id": context.graph_id,
                "graph_revision": context.graph_revision,
                "graph_digest": context.graph_digest,
                "node_params": dict(context.coding_node_params),
            }
        )
        return InvocationSpec(
            invocation_id=spec.invocation_id,
            objective=spec.objective,
            inputs=spec.inputs,
            output_contract=spec.output_contract,
            requested_capabilities=spec.requested_capabilities,
            requested_effects=spec.requested_effects,
            budget=spec.budget,
            deadline_monotonic_s=spec.deadline_monotonic_s,
            idempotency_key=spec.effective_idempotency_key,
            metadata=safe_metadata,
        )

    def _sole_motion_emission(
        self,
        value: Any,
        spec: InvocationSpec,
    ) -> ArtifactEmission:
        if not isinstance(value, ActivationExecutionResult):
            raise ArenaCodingBridgeContractError(
                "Wrapped coding provider did not return ActivationExecutionResult"
            )
        if value.outcome is not ControlOutcome.SUCCESS:
            raise ArenaCodingBridgeContractError(
                "Wrapped coding provider did not successfully author a motion plan: "
                + value.reason
            )
        if len(value.artifacts) != 1:
            raise ArenaCodingBridgeContractError(
                "Arena coding candidate must emit exactly one motion plan"
            )
        emission = value.artifacts[0]
        if not isinstance(emission, ArtifactEmission):
            raise ArenaCodingBridgeContractError(
                "Arena coding candidate returned an untyped artifact emission"
            )
        if (
            emission.schema_id != "robomex.motion_plan.v2"
            or spec.output_contract.get(emission.port) != "robomex.motion_plan.v2"
        ):
            raise ArenaCodingBridgeContractError(
                "Arena coding candidate emitted the wrong port or schema"
            )
        if not isinstance(emission.payload, Mapping):
            raise ArenaCodingBridgeContractError("Motion plan emission payload must be a mapping")
        return emission

    def _validate_plan(
        self,
        payload: Mapping[str, Any],
        *,
        context: ArenaRuntimeContextV1,
    ) -> MotionPlan:
        try:
            parsed = validate_action_spec(payload)
        except (TypeError, ValueError) as exc:
            raise ArenaCodingBridgeContractError("Motion plan payload is invalid") from exc
        if not isinstance(parsed, MotionPlan):
            raise ArenaCodingBridgeContractError("Arena coding output is not a MotionPlan")
        resolved_snapshot = self.data_plane.resolve(context.snapshot_ref)
        if resolved_snapshot.schema != "robomex.admission_snapshot.v1":
            raise ArenaCodingBridgeContractError(
                "Arena context snapshot ref does not resolve to an admission snapshot"
            )
        try:
            snapshot = AdmissionSnapshot.model_validate(resolved_snapshot.payload)
        except (TypeError, ValueError) as exc:
            raise ArenaCodingBridgeContractError("Arena admission snapshot is malformed") from exc
        mismatches: list[str] = []
        if parsed.expected_snapshot != snapshot:
            mismatches.append("expected_snapshot")
        if parsed.world_id != context.world_id or snapshot.world_id != context.world_id:
            mismatches.append("world_id")
        if parsed.resource_id != context.resource_id or snapshot.resource_id != context.resource_id:
            mismatches.append("resource_id")
        if parsed.robot_model_digest != context.robot_model_digest:
            mismatches.append("robot_model_digest")
        if snapshot.config_digest != context.config_digest:
            mismatches.append("config_digest")
        required_tcp = context.hypothesis_config.required_tcp_frame_id
        if required_tcp is not None and parsed.tcp_frame_id != required_tcp:
            mismatches.append("tcp_frame_id")
        pinned_plan_kind = context.coding_node_params.get("plan_kind")
        if pinned_plan_kind is not None and parsed.plan_kind != pinned_plan_kind:
            mismatches.append("plan_kind")
        pinned_tcp = context.coding_node_params.get("tcp_frame_id")
        if pinned_tcp is not None and parsed.tcp_frame_id != pinned_tcp:
            mismatches.append("node_tcp_frame_id")
        pinned_backend = context.coding_node_params.get("planner_backend")
        if pinned_backend is not None and parsed.planner_backend != pinned_backend:
            mismatches.append("planner_backend")
        # Joint paths have no free Cartesian frame field.  Their public frame is
        # therefore the runtime-pinned hypothesis frame, while the exact TCP
        # frame is checked above when the manifest requires one.
        if not context.expected_frame.strip():
            mismatches.append("expected_frame")
        if mismatches:
            raise ArenaCodingBridgeContractError(
                "Motion plan is not bound to trusted Arena context: "
                + ", ".join(mismatches)
            )
        return parsed

    def _publish_plan(
        self,
        *,
        plan: MotionPlan,
        trusted_input_lineage: tuple[ResolvedArtifactRef, ...],
        context: ArenaRuntimeContextV1,
    ) -> tuple[ResolvedArtifactRef, tuple[ResolvedArtifactRef, ...]]:
        lineage = _deduplicate_refs((context.snapshot_ref, *trusted_input_lineage))
        for ref in lineage:
            self.data_plane.resolve(ref)
        record = self.data_plane.publish_once(
            workflow_id=context.workflow_id,
            activation_id=_activation_id(context),
            attempt=1,
            port="motion_plan",
            schema="robomex.motion_plan.v2",
            payload=plan.model_dump(mode="json"),
            lineage=lineage,
        )
        return record.ref, lineage

    def _trusted_input_lineage(
        self,
        spec: InvocationSpec,
    ) -> tuple[ResolvedArtifactRef, ...]:
        lineage: list[ResolvedArtifactRef] = []
        for name, value in sorted(spec.inputs.items()):
            try:
                direct_ref = ResolvedArtifactRef.from_any(value)
                resolved = self.data_plane.resolve(direct_ref)
            except (TypeError, ValueError) as exc:
                raise ArenaCodingBridgeContractError(
                    f"Arena coding input {name!r} is not an admitted artifact ref"
                ) from exc
            lineage.append(direct_ref)
            try:
                lineage.extend(
                    ResolvedArtifactRef.from_any(item)
                    for item in resolved.record.get("lineage", ())
                )
            except (TypeError, ValueError) as exc:
                raise ArenaCodingBridgeContractError(
                    f"Arena coding input {name!r} has malformed causal lineage"
                ) from exc
        trusted = _deduplicate_refs(tuple(lineage))
        for ref in trusted:
            self.data_plane.resolve(ref)
        return trusted

    def _render_and_publish(
        self,
        *,
        plan: MotionPlan,
        plan_ref: ResolvedArtifactRef,
        context: ArenaRuntimeContextV1,
    ) -> tuple[tuple[ResolvedArtifactRef, ...], tuple[float, float, float] | None]:
        config = context.preview_config
        if not config.enabled:
            return (), None
        try:
            renderer = self._renderer
            if renderer is None:
                raise ArenaCodingBridgeContractError("No deterministic renderer is configured")
            renderer_id = str(renderer.renderer_id).strip()
            if renderer_id != config.renderer_id:
                raise ArenaCodingBridgeContractError(
                    "Renderer identity differs from manifest-pinned preview config"
                )
            raw_result = renderer.render(plan=plan, context=context)
            result = (
                raw_result
                if isinstance(raw_result, MotionPreviewResult)
                else MotionPreviewResult.model_validate(raw_result)
            )
            refs: list[ResolvedArtifactRef] = []
            for index, frame in enumerate(result.frames):
                lineage = _deduplicate_refs(
                    (plan_ref, context.snapshot_ref, *frame.lineage)
                )
                for ref in lineage:
                    self.data_plane.resolve(ref)
                envelope = ArenaMotionPreviewArtifact(
                    renderer_id=renderer_id,
                    candidate_id=context.candidate_id,
                    plan_digest=plan.content_digest,
                    view_id=frame.view_id,
                    media_type=frame.media_type,
                    media_digest=frame.media_digest,
                    payload=frame.payload,
                    terminal_position_m=result.terminal_position_m,
                )
                record = self.data_plane.publish_once(
                    workflow_id=context.workflow_id,
                    activation_id=_activation_id(context),
                    attempt=1,
                    port=f"preview_{index:02d}",
                    schema=_PREVIEW_SCHEMA,
                    payload=envelope.model_dump(mode="json"),
                    lineage=lineage,
                )
                refs.append(record.ref)
            return tuple(refs), result.terminal_position_m
        except Exception as exc:
            if config.failure_mode == "omit":
                return (), None
            if isinstance(exc, ArenaCodingBridgeError):
                raise
            raise ArenaCodingBridgeContractError(
                f"Deterministic motion preview failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _hypothesis(
        self,
        *,
        plan: MotionPlan,
        plan_ref: ResolvedArtifactRef,
        evidence_refs: tuple[ResolvedArtifactRef, ...],
        render_refs: tuple[ResolvedArtifactRef, ...],
        terminal_position_m: tuple[float, float, float] | None,
        context: ArenaRuntimeContextV1,
    ) -> ActionHypothesis:
        config = context.hypothesis_config
        return ActionHypothesis(
            candidate_id=context.candidate_id,
            strategy=context.strategy,
            plan_ref=plan_ref,
            snapshot_ref=context.snapshot_ref,
            frame=context.expected_frame,
            expected_effect=config.expected_effect,
            preconditions=config.preconditions,
            estimated_risk=config.estimated_risk,
            utility=config.utility,
            clearance_m=config.clearance_m,
            path_length=_joint_path_length(plan),
            terminal_position_m=terminal_position_m,
            render_refs=render_refs,
            evidence_refs=evidence_refs,
            metadata=_hypothesis_metadata(
                context,
                terminal_position_m=terminal_position_m,
            ),
        )

    def _replay_hypothesis(
        self,
        record: Mapping[str, Any],
        context: ArenaRuntimeContextV1,
    ) -> ActionHypothesis:
        raw = record.get("hypothesis")
        if not isinstance(raw, Mapping) or record.get("hypothesis_digest") != _mapping_digest(raw):
            raise ArenaCodingBridgeContractError("Committed bridge hypothesis is corrupt")
        try:
            hypothesis = ActionHypothesis.model_validate(raw)
        except (TypeError, ValueError) as exc:
            raise ArenaCodingBridgeContractError("Committed hypothesis is malformed") from exc
        config = context.hypothesis_config
        if (
            hypothesis.candidate_id != context.candidate_id
            or hypothesis.strategy != context.strategy
            or hypothesis.snapshot_ref != context.snapshot_ref
            or hypothesis.frame != context.expected_frame
            or hypothesis.expected_effect != config.expected_effect
            or hypothesis.preconditions != config.preconditions
            or hypothesis.estimated_risk != config.estimated_risk
            or hypothesis.utility != config.utility
            or hypothesis.clearance_m != config.clearance_m
            or hypothesis.metadata
            != _hypothesis_metadata(
                context,
                terminal_position_m=hypothesis.terminal_position_m,
            )
        ):
            raise ArenaCodingBridgeContractError(
                "Committed hypothesis is not bound to current trusted Arena context"
            )
        plan_artifact = self.data_plane.resolve(hypothesis.plan_ref)
        if plan_artifact.schema != "robomex.motion_plan.v2":
            raise ArenaCodingBridgeContractError("Committed plan ref changed schema")
        if (
            plan_artifact.record.get("workflow_id") != context.workflow_id
            or plan_artifact.record.get("activation_id") != _activation_id(context)
            or plan_artifact.record.get("attempt") != 1
            or plan_artifact.record.get("port") != "motion_plan"
        ):
            raise ArenaCodingBridgeContractError(
                "Committed plan ref escaped its candidate publication slot"
            )
        plan = self._validate_plan(plan_artifact.payload, context=context)
        if hypothesis.path_length != _joint_path_length(plan):
            raise ArenaCodingBridgeContractError("Committed hypothesis path metric drifted")
        try:
            plan_lineage = tuple(
                ResolvedArtifactRef.from_any(value)
                for value in plan_artifact.record.get("lineage", ())
            )
        except (TypeError, ValueError) as exc:
            raise ArenaCodingBridgeContractError(
                "Committed plan lineage is malformed"
            ) from exc
        if (
            plan_lineage != hypothesis.evidence_refs
            or not plan_lineage
            or plan_lineage[0] != context.snapshot_ref
        ):
            raise ArenaCodingBridgeContractError(
                "Committed hypothesis evidence differs from exact plan lineage"
            )
        for ref in hypothesis.evidence_refs:
            self.data_plane.resolve(ref)
        if hypothesis.terminal_position_m is not None and not hypothesis.render_refs:
            raise ArenaCodingBridgeContractError(
                "Committed terminal position has no deterministic render evidence"
            )
        if not context.preview_config.enabled and hypothesis.render_refs:
            raise ArenaCodingBridgeContractError(
                "Committed hypothesis contains previews disabled by the manifest"
            )
        preview_view_ids: set[str] = set()
        for index, ref in enumerate(hypothesis.render_refs):
            resolved = self.data_plane.resolve(ref)
            if resolved.schema != _PREVIEW_SCHEMA:
                raise ArenaCodingBridgeContractError(
                    "Committed render ref changed schema"
                )
            try:
                preview = ArenaMotionPreviewArtifact.model_validate(resolved.payload)
            except (TypeError, ValueError) as exc:
                raise ArenaCodingBridgeContractError(
                    "Committed motion preview payload is malformed"
                ) from exc
            if preview.view_id in preview_view_ids:
                raise ArenaCodingBridgeContractError(
                    "Committed motion preview repeats a renderer view ID"
                )
            preview_view_ids.add(preview.view_id)
            if (
                preview.renderer_id != context.preview_config.renderer_id
                or preview.candidate_id != context.candidate_id
                or preview.plan_digest != plan.content_digest
                or preview.terminal_position_m != hypothesis.terminal_position_m
                or resolved.record.get("workflow_id") != context.workflow_id
                or resolved.record.get("activation_id") != _activation_id(context)
                or resolved.record.get("attempt") != 1
                or resolved.record.get("port") != f"preview_{index:02d}"
            ):
                raise ArenaCodingBridgeContractError(
                    "Committed motion preview escaped its trusted candidate binding"
                )
            try:
                render_lineage = tuple(
                    ResolvedArtifactRef.from_any(value)
                    for value in resolved.record.get("lineage", ())
                )
            except (TypeError, ValueError) as exc:
                raise ArenaCodingBridgeContractError(
                    "Committed motion preview lineage is malformed"
                ) from exc
            if (
                len(render_lineage) < 2
                or render_lineage[0] != hypothesis.plan_ref
                or render_lineage[1] != context.snapshot_ref
            ):
                raise ArenaCodingBridgeContractError(
                    "Committed motion preview lacks exact plan/snapshot lineage"
                )
        return hypothesis

    def _base_record(
        self,
        runtime: ArenaCodingWorkerRuntime,
        spec: InvocationSpec,
        delegate_spec: InvocationSpec,
        context: ArenaRuntimeContextV1,
        *,
        status: Literal["pending", "committed"],
    ) -> dict[str, Any]:
        return {
            "schema": _BRIDGE_RECORD_SCHEMA,
            "status": status,
            "actor_id": runtime.isolation.owner_actor_id,
            "idempotency_key": spec.effective_idempotency_key,
            "invocation_fingerprint": spec.fingerprint(),
            "delegate_invocation_fingerprint": delegate_spec.fingerprint(),
            "arena_context": context.model_dump(mode="json"),
        }

    def _validate_record_identity(
        self,
        record: Mapping[str, Any],
        runtime: ArenaCodingWorkerRuntime,
        spec: InvocationSpec,
        delegate_spec: InvocationSpec,
        context: ArenaRuntimeContextV1,
    ) -> None:
        status = record.get("status")
        if status not in {"pending", "committed"}:
            raise ArenaCodingBridgeContractError(
                "Arena coding bridge record has an unknown status"
            )
        expected = self._base_record(
            runtime,
            spec,
            delegate_spec,
            context,
            status=status,
        )
        for name in (
            "schema",
            "status",
            "actor_id",
            "idempotency_key",
            "invocation_fingerprint",
            "delegate_invocation_fingerprint",
            "arena_context",
        ):
            if record.get(name) != expected.get(name):
                raise ActorConflictError(
                    "Arena coding replay attempted to rebind durable invocation identity"
                )

    def _record_path(
        self,
        runtime: ArenaCodingWorkerRuntime,
        spec: InvocationSpec,
    ) -> Path:
        if self._record_root is None:
            raise ArenaCodingBridgeContractError("Arena coding record root is not bound")
        identity = (
            runtime.isolation.owner_actor_id + "\0" + spec.effective_idempotency_key
        ).encode("utf-8")
        return self._record_root / (hashlib.sha256(identity).hexdigest() + ".json")

    @staticmethod
    def _read_record(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArenaCodingBridgeContractError("Arena coding record is unreadable") from exc
        if not isinstance(value, dict):
            raise ArenaCodingBridgeContractError("Arena coding record must be a mapping")
        return value

    @staticmethod
    def _write_record(path: Path, value: Mapping[str, Any]) -> None:
        atomic_write_json(path, dict(value))


def _hypothesis_metadata(
    context: ArenaRuntimeContextV1,
    *,
    terminal_position_m: tuple[float, float, float] | None,
) -> dict[str, JsonValue]:
    return {
        "arena_context_schema": context.schema_version,
        "episode_id": context.episode_id,
        "workflow_id": context.workflow_id,
        "graph_id": context.graph_id,
        "graph_revision": context.graph_revision,
        "graph_digest": context.graph_digest,
        "command_attempt": context.command_attempt,
        "slot_id": context.slot_id,
        "arena_run_id": context.arena_run_id,
        "coding_activation_id": context.coding_activation_id,
        "world_id": context.world_id,
        "resource_id": context.resource_id,
        "robot_model_digest": context.robot_model_digest,
        "config_digest": context.config_digest,
        "expected_frame": context.expected_frame,
        "required_tcp_frame_id": context.hypothesis_config.required_tcp_frame_id,
        "plan_kind": context.coding_node_params.get("plan_kind"),
        "planner_backend": context.coding_node_params.get("planner_backend"),
        "planner_configuration_digest": context.coding_node_params.get(
            "planner_configuration_digest"
        ),
        "path_metric": "joint_space_l2_rad",
        "terminal_position_source": (
            "deterministic_renderer" if terminal_position_m is not None else "unavailable"
        ),
    }


def _joint_path_length(plan: MotionPlan) -> float:
    points = (plan.expected_snapshot.joint_positions_rad, *plan.motion.positions_rad)
    total = 0.0
    for previous, current in zip(points[:-1], points[1:], strict=True):
        total += math.sqrt(
            sum((right - left) ** 2 for left, right in zip(previous, current, strict=True))
        )
    return round(total, 12)


def _deduplicate_refs(values: tuple[ResolvedArtifactRef, ...]) -> tuple[ResolvedArtifactRef, ...]:
    unique: list[ResolvedArtifactRef] = []
    seen: set[tuple[str, str]] = set()
    for ref in values:
        if not isinstance(ref, ResolvedArtifactRef):
            raise ArenaCodingBridgeContractError("Artifact lineage contains an unresolved ref")
        identity = (ref.artifact_id, ref.content_digest)
        if identity not in seen:
            seen.add(identity)
            unique.append(ref)
    return tuple(unique)


def _activation_id(context: ArenaRuntimeContextV1) -> str:
    payload = {
        "episode_id": context.episode_id,
        "workflow_id": context.workflow_id,
        "graph_id": context.graph_id,
        "graph_revision": context.graph_revision,
        "graph_digest": context.graph_digest,
        "command_attempt": context.command_attempt,
        "slot_id": context.slot_id,
        "arena_run_id": context.arena_run_id,
        "candidate_id": context.candidate_id,
        "coding_activation_id": context.coding_activation_id,
    }
    return "arena_" + _mapping_digest(payload).removeprefix("sha256:")[:32]


def _mapping_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_record_lock(path: Path):
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock(lock_path):
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


__all__ = [
    "ArenaCodingAgentProvider",
    "ArenaCodingBridgeContractError",
    "ArenaCodingBridgeError",
    "ArenaCodingWorkerRuntime",
    "MotionPreviewFrame",
    "MotionPreviewRenderer",
    "MotionPreviewResult",
]
