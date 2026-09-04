from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.test_vaw_context_agent import (
    RecordingProvider,
    RecordingRenderer,
    _response,
    _user_text,
)
from tests.test_vaw_context_runtime import FakeContextApi
from vaw.context_runtime.protocol import SYSTEM_PROMPT
from vaw.context_runtime.runtime import ContextRunConfig, ContextRuntime
from vaw.context_runtime.trace import ContextTraceLogger
from vaw.context_runtime.workspace import ContextWorkspace
from vaw.evolution import (
    CandidateEvidence,
    CandidatePackage,
    EvolutionSpec,
    GenerationStore,
    MutationOperation,
    SkillMutation,
)
from vaw.mmskill import MMSkill, MMSkillLibrary
from vaw.scripts.run_context_agent import _resolve_skill_library


def _write_skill(
    root: Path,
    skill_id: str,
    *,
    name: str,
    description: str,
    body: str,
) -> None:
    directory = root / skill_id
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        (
            f"---\nname: {name}\ndescription: {description}\n---\n\n"
            f"# {name}\n\n{body}\n"
        ),
        encoding="utf-8",
    )


def _evolution_spec() -> EvolutionSpec:
    return EvolutionSpec(
        experiment_id="stage-2-test",
        base_generation="g000",
        evolve_suite="libero_90",
        evolve_tasks=(0,),
        gate_tasks=(1,),
        reserve_tasks=(2,),
        heldout_suites=("libero_object_task",),
    )


def test_builtin_library_discovers_all_markdown_skills() -> None:
    library = MMSkillLibrary.builtin()

    assert library.summary() == [
        {
            "skill_id": "bowl-rim-grasp-placement",
            "name": "碗抓取与放置说明",
            "description": "当任务需要抓取碗并将其稳定放到目标位置时使用。",
        },
        {
            "skill_id": "container-placement-height-guidance",
            "name": "规划容器放置点位说明",
            "description": "当已抓持物准备从开口容器上方进入或放置时使用。",
        },
        {
            "skill_id": "container-rim-place-recovery",
            "name": "Container Rim Place Recovery",
            "description": "当已抓持物接触或挤压开口容器边沿、直接释放不可靠时使用。",
        },
        {
            "skill_id": "open-drawer-guidance",
            "name": "开抽屉能力指导",
            "description": "当任务要求抓住抽屉把手并将抽屉完全拉开时使用。",
        },
    ]
    index = library.format_index()
    assert "可用 MMSkills（skill_id - name - description）" in index
    assert "bowl-rim-grasp-placement - 碗抓取与放置说明" in index
    assert "container-rim-place-recovery - Container Rim Place Recovery" in index
    assert "container-placement-height-guidance - 规划容器放置点位说明" in index
    assert "open-drawer-guidance - 开抽屉能力指导" in index
    evolver_index = library.format_index(inspect_function="inspect_skill")
    assert "inspect_skill" in evolver_index
    assert "consult_mmskill" not in evolver_index
    skill = library.get("container-rim-place-recovery")
    visible = skill.prompt_block()
    assert "base +Z" in visible
    assert "完全分离" in visible
    assert "base X/Y" in visible
    assert "不用 Z 追求完美对齐" in visible
    assert "新的真实画面核验" in visible

    placement_skill = library.get("container-placement-height-guidance")
    placement_visible = placement_skill.prompt_block()
    assert "向下悬垂的高度" in placement_visible
    assert "被抓物的最低点高于当前可见的最高口沿" in placement_visible
    assert "逐步下降" in placement_visible

    bowl_skill = library.get("bowl-rim-grasp-placement")
    bowl_visible = bowl_skill.prompt_block()
    assert "夹持碗壁" in bowl_visible
    assert "碗底与目标位置中心" in bowl_visible
    assert "不是夹爪或 TCP 中心" in bowl_visible
    assert "local move 小步调整" in bowl_visible

    drawer_skill = library.get("open-drawer-guidance")
    assert "距离把手 point 水平方向 3–6 cm" in drawer_skill.prompt_block()
    assert "无法再继续拉开为止" in drawer_skill.prompt_block()


def test_mmskill_markdown_requires_frontmatter_and_keeps_body_lazy() -> None:
    skill = MMSkill.from_markdown(
        "example-skill",
        """---
name: Example Skill
description: 当当前画面出现示例困难时使用。
---

# Example Skill

Only visible after consultation.
""",
    )

    assert skill.index_entry() == {
        "skill_id": "example-skill",
        "name": "Example Skill",
        "description": "当当前画面出现示例困难时使用。",
    }
    assert "Only visible after consultation." not in str(skill.index_entry())
    assert "Only visible after consultation." in skill.prompt_block()


def test_library_from_root_records_digest_without_exposing_it_in_index(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    _write_skill(
        root,
        "custom-skill",
        name="自定义技能",
        description="当需要自定义指导时使用。",
        body="只在显式加载后出现。",
    )

    library = MMSkillLibrary.from_root(
        root,
        source="unit-test",
        generation_id="g007",
    )

    assert library.get("custom-skill").prompt_block().endswith("只在显式加载后出现。")
    assert library.provenance() == {
        "source": "unit-test",
        "generation_id": "g007",
        "digest": library.digest,
        "skill_count": 1,
    }
    assert len(library.digest) == 64
    assert "g007" not in library.format_index()
    assert library.digest not in library.format_index()
    with pytest.raises(ValueError, match="摘要不匹配"):
        MMSkillLibrary.from_root(root, expected_digest="0" * 64)


def test_runner_resolves_active_or_explicit_generation(tmp_path: Path) -> None:
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    _write_skill(
        base,
        "base-skill",
        name="Base",
        description="基础技能。",
        body="base body",
    )
    _write_skill(
        candidate,
        "candidate-skill",
        name="Candidate",
        description="候选技能。",
        body="candidate body",
    )
    experiment = tmp_path / "experiment"
    store = GenerationStore.initialize(experiment, _evolution_spec(), base)
    evidence_root = tmp_path / "m3"
    evidence_root.mkdir()
    raster = evidence_root / "context.png"
    raster.write_bytes(b"policy-visible-raster")
    mutation = SkillMutation(
        mutation_id="mutation-001",
        operation=MutationOperation.ADD,
        skill_id="candidate-skill",
        evidence_ids=("e1",),
        rationale="测试候选 generation 的显式加载。",
    )
    evolver_audit = tmp_path / "evolver-audit"
    review_audit = tmp_path / "review-audit"
    evolver_audit.mkdir()
    review_audit.mkdir()
    for filename in ("request.json", "response.json"):
        (evolver_audit / filename).write_text("{}\n", encoding="utf-8")
        (review_audit / filename).write_text("{}\n", encoding="utf-8")
    (review_audit / "report.json").write_text(
        '{"decision": "accept", "reason": "test"}\n',
        encoding="utf-8",
    )
    package = CandidatePackage.create(
        tmp_path / "candidate-package",
        mutation=mutation,
        evidence={
            "e1": CandidateEvidence(
                m3_output=str(evidence_root),
                finding_id="finding-1",
                raster="context.png",
                sha256=hashlib.sha256(raster.read_bytes()).hexdigest(),
            )
        },
        skill_root=candidate / "candidate-skill",
        evolver_root=evolver_audit,
        review_root=review_audit,
    )
    store.create_generation(
        "g001",
        parent_generation="g000",
        candidate=package,
    )

    active = _resolve_skill_library(experiment, None)
    explicit = _resolve_skill_library(experiment, "g001")

    assert active.generation_id == "g000"
    assert active.summary()[0]["skill_id"] == "base-skill"
    assert explicit.generation_id == "g001"
    assert [item["skill_id"] for item in explicit.summary()] == [
        "base-skill",
        "candidate-skill",
    ]
    assert store.active_generation() == "g000"
    with pytest.raises(ValueError, match="普通"):
        _resolve_skill_library(base, "g001")
    with pytest.raises(ValueError, match="一起使用"):
        _resolve_skill_library(None, "g001")


def test_consulted_skill_is_transient_context_not_system_prompt(
    tmp_path: Path,
) -> None:
    main = RecordingProvider(
        [
            _response(
                1,
                "consult_mmskill",
                skill_id="container-rim-place-recovery",
            ),
            _response(
                2,
                "imagine_action",
                instruction="被抓物正在碰篮沿，先安全分离再调整到篮口中部",
            ),
            _response(3, "open_gripper"),
            _response(4, "finish_task", success=False),
        ]
    )
    imagination = RecordingProvider(
        [
            _response(
                1,
                "shift_preview",
                delta_xyz_m=[0.0, 0.0, 0.02],
                frame="base",
            ),
            _response(2, "finish_imagination", status="ready"),
        ]
    )
    trace = ContextTraceLogger(tmp_path / "trace")
    runtime = ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "place object in basket", motion_backend="pyroki"),
        RecordingRenderer(),
        imagination_provider=imagination,
        config=ContextRunConfig(max_main_turns=6, max_imagination_turns=3),
        trace=trace,
    )

    result = runtime.run()

    assert [step.op for step in result.steps] == [
        "main:consult_mmskill",
        "main:imagine_action",
        "main:open_gripper",
        "main:finish_task",
    ]
    assert all(messages[0]["content"] == SYSTEM_PROMPT for messages in main.messages)
    assert all("可用 MMSkills" in _user_text(messages) for messages in main.messages)
    assert all(
        "container-rim-place-recovery - Container Rim Place Recovery"
        in _user_text(messages)
        for messages in main.messages
    )
    assert "当前加载技能" not in _user_text(main.messages[0])
    assert "## Visual question" not in _user_text(main.messages[0])
    assert "当前加载技能：container-rim-place-recovery" in _user_text(main.messages[1])
    assert "## Visual question" in _user_text(main.messages[1])
    assert "被抓物是否已与容器边沿完全分离" in _user_text(main.messages[1])
    assert "本轮加载视觉技能：container-rim-place-recovery" in _user_text(
        imagination.messages[0]
    )
    # 技能正文作为 overwrite-only memory 跨物理动作保留，不形成增长历史。
    assert "当前加载技能：container-rim-place-recovery" in _user_text(main.messages[2])
    assert "当前加载技能：container-rim-place-recovery" in _user_text(main.messages[3])
    assert runtime.mmskill_buffer.active is not None
    assert runtime.mmskill_buffer.reference_skill is None
    events = [
        json.loads(line)
        for line in trace.events_path.read_text(encoding="utf-8").splitlines()
    ]
    hidden_event = next(
        event
        for event in events
        if event["event_type"] == "mmskill_reference_hidden"
    )
    assert hidden_event["skill_id"] == "container-rim-place-recovery"
    assert hidden_event["reason"] == "physical_action"


def test_repeated_consult_is_idempotent_after_reference_is_hidden(tmp_path: Path) -> None:
    main = RecordingProvider(
        [
            _response(1, "consult_mmskill", skill_id="container-rim-place-recovery"),
            _response(2, "open_gripper"),
            _response(3, "consult_mmskill", skill_id="container-rim-place-recovery"),
            _response(4, "finish_task", success=False),
        ]
    )
    trace = ContextTraceLogger(tmp_path / "trace")
    runtime = ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "place object in basket", motion_backend="pyroki"),
        RecordingRenderer(),
        config=ContextRunConfig(max_main_turns=4),
        trace=trace,
    )

    result = runtime.run()

    assert [step.op for step in result.steps] == [
        "main:consult_mmskill",
        "main:open_gripper",
        "main:consult_mmskill",
        "main:finish_task",
    ]
    assert json.loads(result.steps[2].result) == {
        "skill_id": "container-rim-place-recovery",
        "loaded": True,
    }
    assert runtime.mmskill_buffer.active is not None
    assert runtime.mmskill_buffer.reference_skill is None
    traced_steps = [
        json.loads(line)
        for line in trace.steps_path.read_text(encoding="utf-8").splitlines()
    ]
    assert traced_steps[2]["runtime_diagnostics"]["reused"] is True


def test_injected_generation_controls_prompt_and_trace_provenance(
    tmp_path: Path,
) -> None:
    skills = tmp_path / "skills"
    _write_skill(
        skills,
        "only-custom",
        name="Only Custom",
        description="仅用于注入测试。",
        body="自定义 generation 正文。",
    )
    library = MMSkillLibrary.from_root(
        skills,
        source="test-experiment",
        generation_id="g003",
    )
    main = RecordingProvider(
        [
            _response(1, "consult_mmskill", skill_id="only-custom"),
            _response(2, "finish_task", success=False),
        ]
    )
    trace = ContextTraceLogger(tmp_path / "trace")
    runtime = ContextRuntime(
        main,
        ContextWorkspace(FakeContextApi(), "test injected skill", motion_backend="pyroki"),
        RecordingRenderer(),
        config=ContextRunConfig(max_main_turns=3),
        trace=trace,
        skill_library=library,
    )

    runtime.run()

    assert "only-custom - Only Custom" in _user_text(main.messages[0])
    assert "container-rim-place-recovery" not in _user_text(main.messages[0])
    assert "自定义 generation 正文。" in _user_text(main.messages[1])
    assert "g003" not in _user_text(main.messages[1])
    assert library.digest not in _user_text(main.messages[1])
    meta = json.loads(trace.meta_path.read_text(encoding="utf-8"))
    assert meta["mmskill"] == library.provenance()
    assert "mmskill_provenance" not in meta
    assert "mmskill_generation" not in meta
    assert "mmskill_digest" not in meta
