from __future__ import annotations

import pytest

from robomex.elastic import (
    ActivationLane,
    ActivationSpec,
    ArtifactBinding,
    BoundedLoopSpec,
    EffectScope,
    ElasticGraphCompiler,
    ElasticGraphSpec,
    LifecycleScope,
    PortSpecV2,
    RunnerKind,
    TransitionSpec,
)
from robomex.runtime.events import ControlOutcome

VALUE = "test.value.v1"


def _node(
    activation_id: str,
    *,
    inputs=(),
    outputs=(),
    bindings=(),
    runner_kind: RunnerKind = RunnerKind.CODING_WORKER,
    effect_scope: EffectScope = EffectScope.READ_ONLY,
    resource: str | None = None,
) -> ActivationSpec:
    return ActivationSpec(
        activation_id=activation_id,
        runner_kind=runner_kind,
        runner_ref=f"test.{activation_id}",
        inputs=inputs,
        outputs=outputs,
        bindings=bindings,
        effect_scope=effect_scope,
        authority_world_id=(
            "authoritative"
            if effect_scope is EffectScope.AUTHORITATIVE_WORLD
            else None
        ),
        authoritative_resource=resource,
    )


def _loop_graph() -> ElasticGraphSpec:
    observe = _node("observe", outputs=(PortSpecV2(name="value", schema_id=VALUE),))
    gate = _node(
        "gate",
        inputs=(PortSpecV2(name="value", schema_id=VALUE),),
        bindings=(
            ArtifactBinding(
                input_port="value", source_activation="observe", source_port="value"
            ),
        ),
        outputs=(PortSpecV2(name="error", schema_id=VALUE),),
        runner_kind=RunnerKind.DETERMINISTIC_GATE,
    )
    correct = _node(
        "correct",
        inputs=(PortSpecV2(name="error", schema_id=VALUE),),
        bindings=(
            ArtifactBinding(input_port="error", source_activation="gate", source_port="error"),
        ),
        runner_kind=RunnerKind.SYSTEM_ACTION,
        effect_scope=EffectScope.AUTHORITATIVE_WORLD,
        resource="robot.arm",
    )
    done = _node("done", runner_kind=RunnerKind.DETERMINISTIC_GATE)
    tracker = ActivationSpec(
        activation_id="tracker",
        runner_kind=RunnerKind.TRACKING_SERVICE,
        runner_ref="test.tracker",
        lane=ActivationLane.SERVICE,
        lifecycle=LifecycleScope.WORKFLOW,
        subscriptions=("observation",),
    )
    return ElasticGraphSpec(
        graph_id="bowl_alignment",
        entry_activation="observe",
        terminal_activations=("done",),
        activations=(observe, gate, correct, done, tracker),
        transitions=(
            TransitionSpec(
                source="observe", outcome=ControlOutcome.SUCCESS, target="gate"
            ),
            TransitionSpec(
                source="gate", outcome=ControlOutcome.NEEDS_ADJUSTMENT, target="correct"
            ),
            TransitionSpec(source="gate", outcome=ControlOutcome.SUCCESS, target="done"),
            TransitionSpec(
                source="correct", outcome=ControlOutcome.SUCCESS, target="observe"
            ),
        ),
        bounded_loops=(
            BoundedLoopSpec(
                loop_id="alignment",
                activation_ids=("observe", "gate", "correct"),
                entry_activation="observe",
                max_iterations=2,
            ),
        ),
    )


def test_compiles_declared_loop_and_service_lane() -> None:
    compiled = ElasticGraphCompiler().compile(_loop_graph())

    assert len(compiled.digest) == 64
    assert compiled.next_activation("gate", ControlOutcome.NEEDS_ADJUSTMENT) == "correct"
    assert compiled.loop_by_activation["observe"] == "alignment"
    assert "tracker" not in compiled.predecessors


def test_capabilities_have_one_typed_authority_channel() -> None:
    with pytest.raises(ValueError, match="required_capabilities"):
        ActivationSpec(
            activation_id="ambiguous_authority",
            runner_kind=RunnerKind.CODING_WORKER,
            runner_ref="test.ambiguous_authority",
            required_capabilities=("perception.read",),
            params={"requested_capabilities": ("robot.motion",)},
        )


def test_rejects_undeclared_cycle() -> None:
    raw = _loop_graph().model_dump(mode="json")
    raw["bounded_loops"] = []

    with pytest.raises(ValueError, match="declared bounded loop"):
        ElasticGraphCompiler().compile(raw)


def test_rejects_ambiguous_primary_continuation() -> None:
    raw = _loop_graph().model_dump(mode="json")
    raw["transitions"].append(
        {"source": "gate", "outcome": "success", "target": "correct"}
    )

    with pytest.raises(ValueError, match="multiple primary continuations"):
        ElasticGraphCompiler().compile(raw)


def test_rejects_binding_not_available_on_every_path() -> None:
    entry = _node("entry", runner_kind=RunnerKind.DETERMINISTIC_GATE)
    producer = _node("producer", outputs=(PortSpecV2(name="value", schema_id=VALUE),))
    other = _node("other")
    join = _node(
        "join",
        inputs=(PortSpecV2(name="value", schema_id=VALUE),),
        bindings=(
            ArtifactBinding(
                input_port="value", source_activation="producer", source_port="value"
            ),
        ),
    )
    done = _node("done", runner_kind=RunnerKind.DETERMINISTIC_GATE)
    spec = ElasticGraphSpec(
        graph_id="bad_join",
        entry_activation="entry",
        terminal_activations=("done",),
        activations=(entry, producer, other, join, done),
        transitions=(
            TransitionSpec(source="entry", outcome=ControlOutcome.SUCCESS, target="producer"),
            TransitionSpec(source="entry", outcome=ControlOutcome.UNCERTAIN, target="other"),
            TransitionSpec(source="producer", outcome=ControlOutcome.SUCCESS, target="join"),
            TransitionSpec(source="other", outcome=ControlOutcome.SUCCESS, target="join"),
            TransitionSpec(source="join", outcome=ControlOutcome.SUCCESS, target="done"),
        ),
    )

    with pytest.raises(ValueError, match="not available on every path"):
        ElasticGraphCompiler().compile(spec)


def test_authoritative_effects_require_system_runner_and_resource() -> None:
    with pytest.raises(ValueError, match="Only a system_action"):
        _node("bad", effect_scope=EffectScope.AUTHORITATIVE_WORLD, resource="robot.arm")
    with pytest.raises(ValueError, match="world and leased resource"):
        _node(
            "bad",
            runner_kind=RunnerKind.SYSTEM_ACTION,
            effect_scope=EffectScope.AUTHORITATIVE_WORLD,
        )


def test_service_cannot_be_used_as_primary_completion_dependency() -> None:
    raw = _loop_graph().model_dump(mode="json")
    raw["transitions"].append(
        {"source": "tracker", "outcome": "success", "target": "done"}
    )

    with pytest.raises(ValueError, match="Service activations use subscriptions"):
        ElasticGraphCompiler().compile(raw)
