from __future__ import annotations

from robomex.elastic import (
    ElasticGraphCompiler,
    ElasticGraphSpec,
    ExternalBinding,
    PortSpecV2,
)
from robomex.orchestration.arena import (
    ARENA_RUNTIME_CONTEXT_METADATA_KEY,
    ArenaBinding,
    ArenaBindingRegistry,
)
from robomex.orchestration.intent import SubgoalIntent
from robomex.test.test_graph_arena_runner_v2 import _arena_node, _build


def test_graph_arena_propagates_only_admitted_manifest_context(tmp_path) -> None:
    runtime, provider, _live, _shadow, original, refs = _build(
        tmp_path,
        count=1,
        high=False,
        shadow=False,
    )
    assert refs is not None
    contextual = ArenaBinding(
        binding_id=original.binding_id,
        candidates=original.candidates,
        expected_frame=original.expected_frame,
        robot_model_digest=original.robot_model_digest,
        candidate_budget_limit=original.candidate_budget_limit,
        candidate_budget_id=original.candidate_budget_id,
        context_input_schemas={"servo_evidence": "robomex.risk_report.v1"},
        gates=original.gates,
        adapters=original.adapters,
        policy=original.policy,
        risk_policy=original.risk_policy,
    )
    runtime.arena_bindings = ArenaBindingRegistry((contextual,))
    risk_payload = runtime.data_plane.resolve(refs["risk"]).payload
    context_record = runtime.data_plane.publish(
        workflow_id="inputs",
        activation_id="servo-evidence",
        attempt=1,
        port="evidence",
        schema="robomex.risk_report.v1",
        payload=risk_payload,
    )
    external_refs = {**refs, "servo_evidence": context_record.ref}

    base = _arena_node(effect_scope=original.candidates[0].effect_scope, count=1)
    arena_node = base.model_copy(
        update={
            "inputs": (
                *base.inputs,
                PortSpecV2(
                    name="servo_evidence",
                    schema_id="robomex.risk_report.v1",
                ),
            ),
            "bindings": (
                *base.bindings,
                ExternalBinding(
                    input_port="servo_evidence",
                    ref="servo_evidence",
                    schema_id="robomex.risk_report.v1",
                ),
            ),
        }
    )
    graph = ElasticGraphCompiler().compile(
        ElasticGraphSpec(
            graph_id="contextual_arena_graph",
            entry_activation="arena",
            terminal_activations=("arena",),
            activations=(arena_node,),
        )
    )
    runtime.open_workflow(
        workflow_id="wf",
        intent=SubgoalIntent(
            intent_id="contextual-arena",
            instruction="select one bounded motion from admitted physical evidence",
            success_rubric="the candidate sees only exact graph-admitted refs",
        ),
        graph=graph,
        external_refs=external_refs,
    )

    terminal = runtime.run_until_terminal("wf")

    assert terminal.status.value == "succeeded"
    invocation_call = next(call for call in provider.calls if call[0] == "invoke")
    invocation = provider.runtime_for(invocation_call[1]).invocations[0]
    assert invocation.inputs["servo_evidence"] == context_record.ref.to_mapping()
    trusted = invocation.metadata[ARENA_RUNTIME_CONTEXT_METADATA_KEY]
    assert trusted["graph_digest"] == f"sha256:{graph.digest}"
    assert trusted["command_attempt"] == 1
    assert trusted["context_input_schemas"] == {
        "servo_evidence": "robomex.risk_report.v1"
    }
    assert trusted["context_input_refs"] == {
        "servo_evidence": context_record.ref.to_mapping()
    }
    result_record = next(
        record
        for record in runtime.data_plane.artifacts
        if record.workflow_id == "wf" and record.port == "result"
    )
    assert context_record.ref in result_record.lineage
