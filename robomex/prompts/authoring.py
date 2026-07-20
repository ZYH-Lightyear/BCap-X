"""Composable prompts for authoring-graph coding agents."""

from __future__ import annotations

from collections.abc import Iterable

from robomex.core.edge_events import EDGE_EVENTS, FAILURE_KINDS
from robomex.core.payload_specs import payload_spec_for


_ACTION_CONTRACT = """Each turn, reply with exactly one JSON action object and no
extra prose. Available actions are use_skill, run_python, and finish.

- use_skill loads one relevant SKILL.md. Loaded skills provide a base directory;
  resolve their relative scripts and resources from that directory. When a skill
  ships a scripts/ directory, the runtime puts it on sys.path for you: import its
  modules directly.
- run_python executes one coherent, stage-sized block in the persistent sandbox.
  The sandbox is ONE persistent Python namespace shared across all your turns:
  variables, imports, and function definitions from earlier turns are still
  defined. Never re-import what you already imported, and never stash values via
  globals()['x'] = x — a plain assignment already survives to the next turn.
  Resolved typed inputs live in INPUTS["<port_name>"]; reference them in code
  instead of retyping numbers from the prompt. Array-typed inputs arrive as
  file refs — load them with np.load(INPUTS["<port>"]["refs"][0]["path"]);
  the payload carries only metadata (shape/dtype/summary), never the array.
  Your action budget is small, so
  make each block complete one full stage (observe + compute + persist + record
  the decision), not one probe; do not spend a turn only printing an intermediate
  value you could branch on in code. Keep stdout to compact decisions and
  metrics. Never print raw arrays, masks, images, point clouds, or candidate
  sets; persist them as artifact files.
- finish ends this node's authoring attempt. Prefer the mechanical hand-off:
  assemble a JSON-safe NODE_RESULT dict in code by referencing sidecar returns /
  INPUTS / printed observations (e.g. "quat": grasp["quat"]), then
  {"tool":"finish","args":{"claim":"...","result_var":"NODE_RESULT"}}. The
  runtime materializes that variable — do not retype its numeric fields into the
  finish JSON. Skill sidecar returns are already JSON-safe lists (never call
  .tolist() on them), and NODE_RESULT may contain numpy values — the runtime
  serializes them. Inline result remains accepted for compatibility. finish does
  not by itself prove that a physical postcondition holds."""

_ARTIFACT_CONTRACT = """Your finish result is a typed hand-off to downstream nodes.
Return factual, JSON-safe evidence. Never invent masks, poses, trajectories, or
verification outcomes. Pose and geometry numbers must be referenced from sidecar
return values, INPUTS, or printed observations — never invent ungrounded pose
literals (e.g. identity quaternions). Include artifact paths and provenance for
values produced by sandbox tools. Values scoped to the current observation must
not be presented as persistent world facts."""

_FAILURE_KIND_CONTRACT = (
    "If the attempt did not succeed, declare the failure as structured data: set "
    '"failure_kind" in finish.args.result to exactly one of '
    f"{', '.join(FAILURE_KINDS)} when one applies, and omit it otherwise. "
    "The runtime routes recovery edges only on this declared value, never on "
    "free-text claims."
)


def build_agent_system_prompt(
    *,
    role: str,
    objective: str,
    capabilities: Iterable[str],
    output_contract: str,
    environment: str = "",
    api_docs: str = "",
) -> str:
    """Build one truthful role prompt from runtime-enforced contracts."""

    caps = ", ".join(sorted(set(capabilities))) or "(none)"
    sections = [
        f"# Role\nYou are the RoboMEx {role} node.",
        f"# Objective\n{objective.strip()}",
        f"# Runtime contract\n{_ACTION_CONTRACT}",
        (
            "# Capabilities\n"
            f"The runtime authorizes only these capability classes: {caps}. "
            "A denied call is a hard execution error; do not attempt to bypass it."
        ),
        f"# Output contract\n{output_contract.strip()}\n\n{_ARTIFACT_CONTRACT}\n\n"
        f"{_FAILURE_KIND_CONTRACT}",
    ]
    if environment.strip():
        sections.append(f"# Environment\n{environment.strip()}")
    if api_docs.strip():
        sections.append(
            "# Available sandbox API functions\n"
            "These functions are already imported. Do not inspect their signatures.\n"
            f"{api_docs.strip()}"
        )
    return "\n\n".join(section for section in sections if section.strip())


def build_universal_act_prompt(*, environment: str = "", api_docs: str = "") -> str:
    return build_agent_system_prompt(
        role="Universal Act",
        objective=(
            "Complete one robot subgoal end to end: observe, ground, compute and validate "
            "affordances, execute bounded actions, then report whether the attempt finished. "
            "Consult a relevant skill before changing robot or environment state. Before "
            "motion or gripper execution, derive targets from current observations, validate "
            "geometry and feasibility, and keep the action bounded. On failure, preserve "
            "evidence and report a concrete next repair instead of asserting success."
        ),
        capabilities=(
            "perception_read",
            "artifact_write",
            "geometry_compute",
            "simulation_probe",
            "active_perception",
            "robot_motion",
            "object_manipulation",
            "gripper_control",
        ),
        output_contract=(
            'Finish with {"tool":"finish","args":{"claim":"...",'
            '"result":{"confidence":0.0,"evidence":{},"verdict":'
            '{"status":"pass|fail|uncertain","reason":"..."},'
            '"artifacts":[],"recommended_next":"..."}}}.'
        ),
        environment=environment,
        api_docs=api_docs,
    )


def build_swarm_manager_prompt(
    *,
    capability_ceiling: Iterable[str],
    environment: str = "",
    api_docs: str = "",
) -> str:
    """Build the contract for dynamic subgoal graph composition."""

    del api_docs  # The control-plane Manager never executes sandbox APIs.
    ceiling = ", ".join(sorted(set(capability_ceiling))) or "(none)"
    sections = [
        "# Role\nYou are the RoboMEx SubgoalSwarmManager.",
        (
            "# Objective\nDynamically compose a typed specialist graph for this embodied "
            "subgoal using the current image, state summary, task guidance, and specialist "
            "contracts. You do not write robot code or claim physical success."
        ),
        (
            "# Progressive task-skill disclosure\nInitially you see only task-skill names and "
            "descriptions. Load relevant high-level guidance with exactly "
            '{"tool":"use_skill","args":{"name":"<task skill id>"}}. A loaded SKILL.md is '
            "reference guidance, never a fixed graph."
        ),
        (
            "# Graph submission\nAfter loading task guidance, reply with exactly one "
            '{"tool":"submit_graph","args":{"task_skill":"<loaded task skill id>",'
            '"graph":{"entry":"<node>","success_node":"<hard verifier node>",'
            '"max_recoveries":2,"nodes":[{"id":"<node>","skill":"<specialist id>",'
            '"objective":"...","task":"...","inputs":{"<port>":{"$ref":'
            '"<producer>.<output>"}},"checkpoint":"none|hard"}],"edges":'
            '[{"from":"<node>","to":"<node>","on":"<event>"}]}}}. '
            "Return one JSON action and no prose on every turn."
        ),
        (
            "# Edge events\nEach edge's \"on\" must be exactly one of: "
            f"{', '.join(EDGE_EVENTS)}. This vocabulary is closed; any other label "
            "is a validation error. When a specialist contract declares "
            "`exit_conditions`, edges leaving that node may listen only for those "
            "events (plus failed, exhausted, uncertain, stale_observation, which "
            "the runtime can raise for any node) — an edge on an event the node "
            "never emits is rejected as permanently dead. "
            "Continue the primary path with success edges "
            "and add recovery edges only for the declared failure events above. "
            "`uncertain` is not a failure and never falls back to a failed edge: "
            "declare an explicit `on: \"uncertain\"` edge (for example, back to the "
            "verifier for a fresh look, or to an extra observation node) only when "
            "a bounded in-graph recheck is justified. If you declare none, an "
            "uncertain checkpoint ends the graph and escalates to the outer "
            "planner with its partial evidence — that is often the right choice "
            "after a world-changing action, because replaying upstream nodes "
            "against an already-changed world is destructive."
        ),
        (
            "# Composition policy\nGenerate the topology for this specific observation; do not "
            "copy or assume a fixed pick/place chain. Choose node skills only from "
            "specialist_contracts. The contract supplies each node's role, ports, capabilities "
            "and budget; do not redefine them. Bind every required input to a compatible "
            "upstream output. Separate motion planning from world-changing action execution. "
            "Every world-changing node must flow directly to a read-only verifier, and successful "
            "termination must be a hard verifier. Add only recovery edges justified by likely "
            "typed failures. Validation errors will be returned for repair."
        ),
        (
            "# Capability policy\nYou may request only these capability classes: "
            f"{ceiling}. Requests are not grants: each leaf is independently constrained."
        ),
        _ARTIFACT_CONTRACT,
    ]
    if environment.strip():
        sections.append(f"# Environment\n{environment.strip()}")
    return "\n\n".join(sections)


build_swarm_creator_prompt = build_swarm_manager_prompt


def output_contract_for_ports(ports: Iterable[tuple[str, ...]]) -> str:
    items = tuple(ports)
    rendered = (
        "\n".join(
            f"- output key `{item[0]}`: schema `{item[1]}`"
            + (f", frame `{item[2]}`" if len(item) > 2 and item[2] else "")
            for item in items
        )
        or "(no output ports)"
    )
    # Render each schema's canonical payload spec from the single source of
    # truth: the same spec that rejects the payload at publish time.
    payload_docs = "\n\n".join(
        spec.doc
        for spec in (
            payload_spec_for(item[1]) for item in items if len(item) > 1
        )
        if spec is not None
    )
    contract = (
        "Assemble NODE_RESULT in the sandbox so it matches the declared typed "
        "ports (exact output keys; do not concatenate schema/frame), then finish "
        'with {"tool":"finish","args":{"claim":"...","result_var":"NODE_RESULT"}}.\n'
        f"Required ports:\n{rendered}\n"
        "NODE_RESULT shape: an object with `outputs` mapping each port to "
        "`{payload, confidence?, frame?, observed_at?, artifacts?}`, plus "
        "`recommended_next`. Each `payload` is a compact JSON object — never an "
        "inline array or ellipsis; persist arrays as files under ARTIFACTS_DIR and "
        "list them in `artifacts` (paths relative to ARTIFACTS_DIR or absolute; "
        "never CWD-relative). Reference numeric fields from sidecar returns / "
        "INPUTS; do not retype them into finish JSON. The runtime stamps "
        "observation_epoch; do not include it. Grounding agents should persist "
        "both mask and RGB arrays; the runtime derives a PNG mask overlay."
    )
    declared_names = {item[0] for item in items}
    if "verifier_report" in declared_names:
        contract += (
            " Verifier nodes must put a verdict object at "
            "`outputs.verifier_report.payload.verdict`; its status is "
            "pass, fail, or uncertain."
        )
    if payload_docs:
        contract += (
            "\n\nPayload specs (enforced at publish time; a violating payload is "
            "rejected and returned to you for repair):\n" + payload_docs
        )
    return contract
