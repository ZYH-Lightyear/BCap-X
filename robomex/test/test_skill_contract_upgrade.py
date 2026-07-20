"""M2 契约载体升级测试:exit_conditions / functions / prompts 的解析、校验、
图编译子集约束与沙箱注入。全部离线(无 env / LLM / 网络)。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robomex.authoring.graph import SubgoalGraphCompiler
from robomex.dysc.contract_checks import (
    contract_consistency_errors,
    resolve_function_signature,
)
from robomex.dysc.contracts import (
    ContractFunction,
    SkillContract,
    load_contract_for_skill,
    load_skill_contracts,
)
from robomex.skills import Skill, SkillLibrary, load_builtin_skills


BUILTIN_ROOT = Path(__file__).resolve().parents[1] / "skills" / "builtin"


# ---- 解析 ---------------------------------------------------------------------


def test_contract_parses_m2_fields() -> None:
    contract = SkillContract.from_mapping(
        {
            "skill_id": "demo",
            "role": "motion_planner",
            "exit_conditions": {
                "success": "planned",
                "infeasible": "IK rejected the pose",
            },
            "functions": [
                {
                    "name": "plan",
                    "entry": "scripts/planner.py:plan",
                    "description": "canonical planner",
                }
            ],
            "prompts": [
                {"name": "check", "path": "prompts/check.md", "description": "vlm check"}
            ],
        }
    )
    assert set(contract.exit_conditions) == {"success", "infeasible"}
    assert contract.functions[0].entry_path == "scripts/planner.py"
    assert contract.functions[0].entry_function == "plan"
    assert contract.prompts[0].path == "prompts/check.md"


def test_contract_without_m2_fields_stays_legacy() -> None:
    contract = SkillContract.from_mapping({"skill_id": "old", "role": "grounding"})
    assert contract.exit_conditions == {}
    assert contract.functions == ()
    assert contract.prompts == ()


# ---- loader 一致性校验 ----------------------------------------------------------


def _package(tmp_path: Path, contract_yaml: str, *, body: str = "Body.") -> Path:
    root = tmp_path / "demo_skill"
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(
        f"---\nname: demo_skill\ncategory: motion\n---\n\n{body}\n", encoding="utf-8"
    )
    (root / "contract.yaml").write_text(contract_yaml, encoding="utf-8")
    return root


def test_checks_reject_unknown_and_alias_exit_events(tmp_path) -> None:
    root = _package(tmp_path, "skill_id: demo_skill\nrole: motion_planner\n")
    contract = SkillContract.from_mapping(
        {
            "skill_id": "demo_skill",
            "exit_conditions": {"success": "", "took_off": "", "failure": ""},
        }
    )
    errors = "\n".join(contract_consistency_errors(contract, root))
    assert "took_off" in errors  # 词表之外
    assert "canonical spelling" in errors  # failure -> failed 必须用规范拼写


def test_checks_require_success_exit(tmp_path) -> None:
    root = _package(tmp_path, "skill_id: demo_skill\n")
    contract = SkillContract.from_mapping(
        {"skill_id": "demo_skill", "exit_conditions": {"infeasible": "x"}}
    )
    errors = contract_consistency_errors(contract, root)
    assert any("must declare `success`" in e for e in errors)


def test_checks_validate_function_entries(tmp_path) -> None:
    root = _package(tmp_path, "skill_id: demo_skill\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "helper.py").write_text(
        "def real_fn(a, b=2, *, c=None):\n    return a\n", encoding="utf-8"
    )
    contract = SkillContract.from_mapping(
        {
            "skill_id": "demo_skill",
            "functions": [
                {"name": "ok_fn", "entry": "scripts/helper.py:real_fn"},
                {"name": "bad entry", "entry": "helper.py"},
                {"name": "missing_file", "entry": "scripts/nope.py:fn"},
                {"name": "missing_def", "entry": "scripts/helper.py:absent"},
            ],
        }
    )
    errors = "\n".join(contract_consistency_errors(contract, root))
    assert "ok_fn" not in errors
    assert "not a valid Python identifier" in errors
    assert "not found under the skill package" in errors
    assert "not defined at module level" in errors


def test_signature_declared_wins_else_derived_from_ast(tmp_path) -> None:
    root = _package(tmp_path, "skill_id: demo_skill\n")
    (root / "scripts").mkdir()
    (root / "scripts" / "helper.py").write_text(
        "def real_fn(a, b=2, *, c=None):\n    return a\n", encoding="utf-8"
    )
    derived = resolve_function_signature(
        ContractFunction(name="real_fn", entry="scripts/helper.py:real_fn"), root
    )
    assert derived == "real_fn(a, b=2, *, c=None)"
    declared = resolve_function_signature(
        ContractFunction(
            name="real_fn",
            entry="scripts/helper.py:real_fn",
            signature="real_fn(a) -> dict",
        ),
        root,
    )
    assert declared == "real_fn(a) -> dict"


def test_checks_validate_prompt_files(tmp_path) -> None:
    root = _package(tmp_path, "skill_id: demo_skill\n")
    (root / "prompts").mkdir()
    (root / "prompts" / "real.md").write_text("Ask about {target}.", encoding="utf-8")
    contract = SkillContract.from_mapping(
        {
            "skill_id": "demo_skill",
            "prompts": [
                {"name": "real", "path": "prompts/real.md"},
                {"name": "ghost", "path": "prompts/ghost.md"},
            ],
        }
    )
    errors = "\n".join(contract_consistency_errors(contract, root))
    assert "ghost" in errors and "real'" not in errors


def test_checks_flag_reference_code_calling_forbidden_api(tmp_path) -> None:
    """M1-B2 的教训固化为编译期校验:SKILL.md 代码块不得调用被禁 API。"""

    body = (
        "## Reference Code\n\n"
        "```python\n"
        "obs = get_observation()\n"
        "```\n"
    )
    root = _package(tmp_path, "skill_id: demo_skill\n", body=body)
    contract = SkillContract.from_mapping(
        {
            "skill_id": "demo_skill",
            "forbidden_capabilities": ["perception_read"],
        }
    )
    errors = contract_consistency_errors(contract, root)
    assert any("get_observation" in e and "forbidden_capabilities" in e for e in errors)


def test_load_skill_contracts_raises_on_broken_package(tmp_path) -> None:
    category = tmp_path / "motion"
    category.mkdir()
    root = category / "demo_skill"
    root.mkdir()
    (root / "SKILL.md").write_text("---\nname: demo_skill\n---\n\nBody.", encoding="utf-8")
    (root / "contract.yaml").write_text(
        "skill_id: demo_skill\nrole: motion_planner\n"
        "functions:\n  - {name: fn, entry: 'scripts/nope.py:fn'}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not found under the skill package"):
        load_skill_contracts(tmp_path)
    # validate=False 保留旧行为,供只读工具使用。
    contracts = load_skill_contracts(tmp_path, validate=False)
    assert "demo_skill" in contracts


def test_builtin_library_contracts_pass_validation() -> None:
    contracts = load_skill_contracts(BUILTIN_ROOT)
    assert "plan_bounded_motion" in contracts
    pilot = contracts["plan_bounded_motion"]
    assert set(pilot.exit_conditions) == {"success", "infeasible"}
    assert [f.name for f in pilot.functions] == [
        "build_grasp_trajectory",
        "build_place_trajectory",
    ]
    verify = contracts["verify_grasp_and_lift_via_robot_state"]
    assert set(verify.exit_conditions) == {"success", "failed_grasp", "wrong_grounding"}
    assert [f.name for f in verify.functions] == ["verify_grasp_with_vlm"]


# ---- 图编译:edge.on ⊆ 源节点声明 ------------------------------------------------


def _pick_draft(extra_edges: list[dict] | None = None) -> dict:
    return {
        "entry": "ground",
        "success_node": "verify",
        "nodes": [
            {"id": "ground", "skill": "segment_object"},
            {
                "id": "plan_grasp",
                "skill": "grasp_graspnet",
                "inputs": {
                    "object_mask": {"$ref": "ground.object_mask"},
                    "object_points": {"$ref": "ground.object_points"},
                },
            },
            {
                "id": "plan_motion",
                "skill": "plan_bounded_motion",
                "inputs": {"affordance": {"$ref": "plan_grasp.grasp_affordance"}},
            },
            {
                "id": "execute",
                "skill": "grasp_object",
                "inputs": {"trajectory": {"$ref": "plan_motion.trajectory"}},
            },
            {
                "id": "verify",
                "skill": "verify_grasp_and_lift_via_robot_state",
                "checkpoint": "hard",
                "inputs": {
                    "execution_evidence": {"$ref": "execute.execution_evidence"}
                },
            },
        ],
        "edges": [
            {"from": "ground", "to": "plan_grasp", "on": "success"},
            {"from": "plan_grasp", "to": "plan_motion", "on": "success"},
            {"from": "plan_motion", "to": "execute", "on": "success"},
            {"from": "execute", "to": "verify", "on": "success"},
            *(extra_edges or ()),
        ],
    }


def _compile(draft: dict):
    contracts = load_skill_contracts(BUILTIN_ROOT)
    return SubgoalGraphCompiler(contracts).compile(
        draft, task_skill="pick_object", goal="pick", postcondition="held"
    )


def test_edges_on_declared_exit_conditions_compile() -> None:
    graph = _compile(
        _pick_draft(
            [
                {"from": "plan_motion", "to": "plan_grasp", "on": "infeasible"},
                {"from": "verify", "to": "ground", "on": "failed_grasp"},
            ]
        )
    )
    assert {edge.on for edge in graph.edges} >= {"infeasible", "failed_grasp"}


def test_edge_outside_declared_exit_conditions_is_a_compile_error() -> None:
    # grasp_object 只声明 success;failed_grasp 属于 verifier 的判定词汇,
    # 从执行节点挂这条边是永远打不通的死边。
    with pytest.raises(ValueError, match="never emits"):
        _compile(
            _pick_draft([{"from": "execute", "to": "plan_motion", "on": "failed_grasp"}])
        )


def test_runtime_events_stay_routable_despite_declared_subset() -> None:
    # failed / exhausted / uncertain / stale_observation 是 runtime 对任意节点
    # 都可能触发的事件,不受技能声明子集限制。
    graph = _compile(
        _pick_draft(
            [
                {"from": "execute", "to": "plan_motion", "on": "failed"},
                {"from": "plan_motion", "to": "plan_grasp", "on": "stale_observation"},
            ]
        )
    )
    assert {edge.on for edge in graph.edges} >= {"failed", "stale_observation"}


def test_legacy_contract_without_declaration_keeps_full_vocabulary() -> None:
    # segment_object 未声明 exit_conditions,其出边仍可用整个闭合词表。
    graph = _compile(
        _pick_draft([{"from": "ground", "to": "ground", "on": "wrong_grounding"}])
    )
    assert any(edge.on == "wrong_grounding" for edge in graph.edges)


# ---- 沙箱注入 -------------------------------------------------------------------


class _RecordingExecutor:
    def __init__(self) -> None:
        self.blocks = []

    def run_block(self, block):
        from robomex.core.sandbox import ActionBlockStatus, BlockExecutionResult

        self.blocks.append(block)
        return BlockExecutionResult(
            block=block,
            ok=True,
            status=ActionBlockStatus.SUCCEEDED,
            stdout="",
            stderr="",
        )


def _library_with_contracted_skill(tmp_path: Path) -> tuple[SkillLibrary, str]:
    src = tmp_path / "src" / "demo_verify"
    (src / "scripts").mkdir(parents=True)
    (src / "prompts").mkdir()
    (src / "SKILL.md").write_text(
        "---\nname: demo_verify\ncategory: verification\n"
        "description: demo\n---\n\nBody.",
        encoding="utf-8",
    )
    (src / "scripts" / "helper.py").write_text(
        "def check_state(target, *, image=None):\n    return {'ok': True}\n",
        encoding="utf-8",
    )
    (src / "prompts" / "scene_check.md").write_text(
        "Is {target} held? Answer JSON.", encoding="utf-8"
    )
    (src / "contract.yaml").write_text(
        "skill_id: demo_verify\nrole: verifier\n"
        "exit_conditions:\n  success: verified\n"
        "functions:\n"
        "  - {name: check_state, entry: 'scripts/helper.py:check_state', "
        "description: canonical check}\n"
        "prompts:\n"
        "  - {name: scene_check, path: prompts/scene_check.md}\n",
        encoding="utf-8",
    )
    library = SkillLibrary(tmp_path / "lib")
    library.admit(Skill.from_dir(src))
    return library, "demo_verify"


def test_skill_load_binds_functions_and_prompts_into_sandbox(tmp_path) -> None:
    from robomex.agents.subagents import CodingAgentSubAgent, SubAgentRequest
    from robomex.core.coder import ScriptedCodePolicy

    library, skill_id = _library_with_contracted_skill(tmp_path)
    executor = _RecordingExecutor()
    agent = CodingAgentSubAgent(
        executor=executor,
        policy=ScriptedCodePolicy(
            [
                json.dumps({"tool": "use_skill", "args": {"name": skill_id}}),
                json.dumps({"tool": "finish", "args": {"claim": "done"}}),
            ]
        ),
        library=library,
        name="demo",
        task_kind="verify",
    )
    agent.run(SubAgentRequest(task="verify", artifacts_dir=str(tmp_path / "arts")))

    bindings = [b for b in executor.blocks if b.name.startswith("skill_bindings_")]
    assert len(bindings) == 1
    assert bindings[0].metadata.get("runtime_setup") is True
    code = bindings[0].code
    assert "check_state = _m2_bind(" in code
    assert "'check_state')" in code
    assert "PROMPTS['scene_check'] = 'Is {target} held? Answer JSON.'" in code


def test_skill_load_message_advertises_bindings(tmp_path) -> None:
    from robomex.agents.subagents import CodingAgentSubAgent, SubAgentRequest
    from robomex.core.coder import ScriptedCodePolicy

    library, skill_id = _library_with_contracted_skill(tmp_path)

    class _PromptSpy(ScriptedCodePolicy):
        def __init__(self, responses):
            super().__init__(responses)
            self.prompts = []

        def complete(self, prompt):
            self.prompts.append([dict(m) for m in prompt])
            return super().complete(prompt)

    policy = _PromptSpy(
        [
            json.dumps({"tool": "use_skill", "args": {"name": skill_id}}),
            json.dumps({"tool": "finish", "args": {"claim": "done"}}),
        ]
    )
    agent = CodingAgentSubAgent(
        executor=_RecordingExecutor(),
        policy=policy,
        library=library,
        name="demo",
        task_kind="verify",
    )
    agent.run(SubAgentRequest(task="verify", artifacts_dir=str(tmp_path / "arts")))

    final_prompt = policy.prompts[-1]
    skill_message = next(
        m["content"]
        for m in final_prompt
        if isinstance(m.get("content"), str) and "Loaded skill." in m["content"]
    )
    assert "check_state(target, *, image=None)" in skill_message
    assert "canonical check" in skill_message
    assert "PROMPTS['scene_check']" in skill_message
    assert "call them directly (no import" in skill_message


def test_binding_block_runs_once_per_skill_root(tmp_path) -> None:
    from robomex.agents.subagents import CodingAgentSubAgent, SubAgentRequest
    from robomex.core.coder import ScriptedCodePolicy

    library, skill_id = _library_with_contracted_skill(tmp_path)
    executor = _RecordingExecutor()
    agent = CodingAgentSubAgent(
        executor=executor,
        policy=ScriptedCodePolicy(
            [
                json.dumps({"tool": "use_skill", "args": {"name": skill_id}}),
                json.dumps({"tool": "use_skill", "args": {"name": skill_id}}),
                json.dumps({"tool": "finish", "args": {"claim": "done"}}),
            ]
        ),
        library=library,
        name="demo",
        task_kind="verify",
    )
    agent.run(SubAgentRequest(task="verify", artifacts_dir=str(tmp_path / "arts")))

    bindings = [b for b in executor.blocks if b.name.startswith("skill_bindings_")]
    assert len(bindings) == 1


def test_admitted_library_preserves_prompts_and_contract(tmp_path) -> None:
    library, skill_id = _library_with_contracted_skill(tmp_path)
    root = library.get(skill_id).skill.root
    assert root is not None
    contract = load_contract_for_skill(root)
    assert contract is not None
    assert (root / "prompts" / "scene_check.md").is_file()
    assert contract.prompts[0].name == "scene_check"


# ---- runtime:未声明的 failure_kind 降级为通用 failed ------------------------------


def _factory_with_finish(tmp_path: Path, finish_json: str):
    from robomex.authoring.adapters import SubAgentFactory
    from robomex.core.sandbox import RuntimeSafetyState

    library = SkillLibrary(tmp_path / "library")
    for skill in load_builtin_skills():
        library.admit(skill, source="builtin")
    contracts = load_skill_contracts(library.root)

    class _OneShotPolicy:
        def complete(self, prompt) -> str:
            return finish_json

    return (
        SubAgentFactory(
            executor=_RecordingExecutor(),
            policy=_OneShotPolicy(),
            library=library,
            capability_ceiling=frozenset(
                {"perception_read", "geometry_compute", "artifact_write"}
            ),
            safety_state=RuntimeSafetyState(),
            contracts=contracts,
        ),
        contracts,
    )


def _run_plan_motion(factory, contracts):
    from robomex.authoring.artifacts import TypedArtifact
    from robomex.authoring.result import SubgoalAuthoringContext
    from robomex.authoring.swarm_spec import SpecialistSpec

    spec = SpecialistSpec.from_contract(
        agent_id="plan_motion",
        contract=contracts["plan_bounded_motion"],
        objective="plan",
        task="plan",
    )
    affordance = TypedArtifact(
        port="affordance",
        schema="robomex.affordance.v1",
        producer="affordance",
        frame="world",
    )
    return factory.run(
        spec,
        context=SubgoalAuthoringContext("task", 0, "plan", "feasible"),
        inputs={"affordance": affordance},
        artifact_dir=None,
    )


def _failed_finish(failure_kind: str) -> str:
    # 诚实的失败 finish:声明 failure_kind、不捏造 trajectory 输出。
    return json.dumps(
        {
            "tool": "finish",
            "args": {
                "claim": "could not plan",
                "result": {
                    "failure_kind": failure_kind,
                    "verdict": {"status": "fail", "reason": "no feasible pose"},
                },
            },
        }
    )


def test_declared_failure_kind_survives_to_routing(tmp_path) -> None:
    factory, contracts = _factory_with_finish(tmp_path, _failed_finish("infeasible"))
    result = _run_plan_motion(factory, contracts)
    assert not result.ok
    assert result.failure_kind == "infeasible"


def test_undeclared_failure_kind_degrades_to_generic_failed(tmp_path) -> None:
    # plan_bounded_motion 声明 {success, infeasible};failed_grasp 不在其中,
    # 编译器从未为它校验过边,运行期必须降级为通用 failed。
    factory, contracts = _factory_with_finish(tmp_path, _failed_finish("failed_grasp"))
    result = _run_plan_motion(factory, contracts)
    assert not result.ok
    assert result.failure_kind == ""


# ---- Manager catalog ------------------------------------------------------------


def test_manager_catalog_renders_exit_conditions_and_function_signatures(tmp_path) -> None:
    from robomex.authoring.swarm_creator import SubgoalSwarmManager
    from robomex.core.sandbox import RuntimeSafetyState

    library = SkillLibrary(tmp_path / "library")
    for skill in load_builtin_skills():
        library.admit(skill, source="builtin")
    contracts = load_skill_contracts(library.root)
    manager = SubgoalSwarmManager(
        policy=None,  # catalog rendering never calls the policy
        library=library,
        factory=None,  # type: ignore[arg-type]
        capability_ceiling=frozenset(
            {
                "perception_read",
                "geometry_compute",
                "artifact_write",
                "robot_motion",
                "gripper_control",
                "object_manipulation",
            }
        ),
        safety_state=RuntimeSafetyState(),
        contracts=contracts,
        graph_executor=object(),  # type: ignore[arg-type]
    )

    catalog = manager._specialist_catalog()
    plan = catalog["plan_bounded_motion"]
    assert set(plan["exit_conditions"]) == {"success", "infeasible"}
    signatures = {item["name"]: item["signature"] for item in plan["functions"]}
    assert signatures["build_place_trajectory"] == (
        "build_place_trajectory(affordance, *, approach_height=None)"
    )
    assert signatures["build_grasp_trajectory"].startswith(
        "build_grasp_trajectory(affordance"
    )
    verify = catalog["verify_grasp_and_lift_via_robot_state"]
    assert "failed_grasp" in verify["exit_conditions"]
    assert "wrong_grounding" in catalog["segment_object"]["exit_conditions"]
