"""技能加载与两段式披露的测试。

覆盖:frontmatter 解析与报错、发现顺序、索引渲染、``skill`` 工具的命中与未命中,
以及接进 :class:`CodingAgent` 之后「有技能才改变行为」这条不变式。
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agentx.agent import CodingAgent
from agentx.contracts import ModelResponse
from agentx.skills import (
    SkillError,
    discover_skills,
    parse_skill_md,
    render_index,
)
from agentx.tools import ToolContext
from agentx.tools.skill import SkillTool

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_skill(root: Path, name: str, *, front: str = "", body: str = "正文") -> Path:
    """在 ``root/<name>/SKILL.md`` 造一个技能,返回技能目录。"""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    frontmatter = front or f"name: {name}\ndescription: {name} 的用途"
    (directory / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8"
    )
    return directory


def _raw_skill(root: Path, name: str, text: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(text, encoding="utf-8")
    return path


class _ScriptedProvider:
    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self.responses = list(responses or [])

    def generate(self, messages, tools=None):
        if not self.responses:
            return ModelResponse(text="done")
        return self.responses.pop(0)


# --- 解析 -------------------------------------------------------------------


def test_parse_keeps_unknown_frontmatter_keys_verbatim(tmp_path: Path) -> None:
    """未知键必须原样保留 —— 这是加载任意技能库而不改解析代码的前提。"""
    path = _raw_skill(
        tmp_path,
        "perceiving",
        textwrap.dedent("""
            ---
            name: perceiving
            description: 感知
            license: MIT
            gap:
              allowed_tools:
                - robot.get_observation
              streaming: false
            ---

            # perceiving
            """).lstrip(),
    )

    skill = parse_skill_md(path)

    assert skill.name == "perceiving"
    assert skill.extra["license"] == "MIT"
    assert skill.extra["gap"]["allowed_tools"] == ["robot.get_observation"]
    assert skill.extra["gap"]["streaming"] is False
    # name / description 被取走,不该重复出现在 extra 里。
    assert "name" not in skill.extra
    assert skill.body == "# perceiving"
    assert skill.path == path.parent


def test_parse_collapses_multiline_description(tmp_path: Path) -> None:
    """YAML 折叠标量会留下换行,塞进索引会把 XML 撑散。"""
    path = _raw_skill(
        tmp_path,
        "folded",
        "---\nname: folded\ndescription: >\n  第一行\n  第二行\n---\n\n正文\n",
    )

    assert parse_skill_md(path).description == "第一行 第二行"


@pytest.mark.parametrize(
    ("text", "hint"),
    [
        ("没有 frontmatter\n", "--- 开头"),
        ("---\nname: a\ndescription: b\n", "没有闭合"),
        ("---\n- 不是映射\n---\n\n正文\n", "YAML 映射"),
        ("---\ndescription: 只有描述\n---\n\n正文\n", "'name'"),
        ("---\nname: 只有名字\n---\n\n正文\n", "'description'"),
        ("---\nname: a\ndescription: '  '\n---\n\n正文\n", "'description'"),
        ("---\nname: a\ndescription: [1, 2]\n---\n\n正文\n", "'description'"),
    ],
)
def test_parse_rejects_malformed_skill(tmp_path: Path, text: str, hint: str) -> None:
    """写坏的 SKILL.md 要在启动时炸掉,而不是静默跳过。

    静默跳过的代价是模型运行时发现技能不见了却无从查起,比启动失败贵得多。
    """
    path = _raw_skill(tmp_path, "broken", text)

    with pytest.raises(SkillError) as excinfo:
        parse_skill_md(path)
    assert hint in str(excinfo.value)


def test_shipped_grounding_skill_parses() -> None:
    """仓库里真实的那份技能必须能加载,否则评测跑起来才发现就晚了。"""
    skills = discover_skills([REPO_ROOT / "robomex" / "skills"])

    names = [skill.name for skill in skills]
    assert "grounding-objects" in names

    grounding = next(s for s in skills if s.name == "grounding-objects")
    assert "\n" not in grounding.description
    assert "def ground_object(" in grounding.body
    assert set(grounding.extra["produces_outputs"]) == {
        "mask", "center", "obb", "box", "n_points",
    }


# --- 发现 -------------------------------------------------------------------


def test_discover_sorts_by_name_and_skips_missing_roots(tmp_path: Path) -> None:
    _write_skill(tmp_path, "zulu")
    _write_skill(tmp_path, "alpha")
    (tmp_path / "not-a-skill").mkdir()

    skills = discover_skills([tmp_path, tmp_path / "does-not-exist"])

    assert [s.name for s in skills] == ["alpha", "zulu"]


def test_discover_lets_the_first_root_win_on_name_collision(tmp_path: Path) -> None:
    high = tmp_path / "high"
    low = tmp_path / "low"
    _write_skill(high, "shared", body="优先级高的正文")
    _write_skill(low, "shared", body="优先级低的正文")

    skills = discover_skills([high, low])

    assert len(skills) == 1
    assert skills[0].body == "优先级高的正文"


def test_discover_accepts_a_single_skill_directory(tmp_path: Path) -> None:
    """``--skills path/to/one-skill`` 不该要求先造一层父目录。"""
    directory = _write_skill(tmp_path, "solo")

    skills = discover_skills([directory])

    assert [s.name for s in skills] == ["solo"]


# --- 索引 -------------------------------------------------------------------


def test_render_index_is_empty_without_skills() -> None:
    """空串是「不要注入」的信号,免得 prompt 里多出一个空段落。"""
    assert render_index([]) == ""


def test_render_index_lists_names_and_descriptions(tmp_path: Path) -> None:
    _write_skill(tmp_path, "grounding", body="BODY-MARKER")
    skills = discover_skills([tmp_path])

    index = render_index(skills)

    assert '<skill name="grounding">grounding 的用途</skill>' in index
    assert "`skill`" in index
    # 第一段只给名字和描述,正文必须留到第二段。
    assert "BODY-MARKER" not in index


# --- skill 工具 -------------------------------------------------------------


def test_skill_tool_returns_the_full_body(tmp_path: Path) -> None:
    _write_skill(tmp_path, "grounding", body="步骤一\n步骤二")
    tool = SkillTool(discover_skills([tmp_path]))

    result = tool.build({"name": "grounding"}).execute(
        ToolContext(root=tmp_path, workspace=tmp_path / ".agentx")
    )

    assert not result.is_error
    assert "步骤一\n步骤二" in result.llm_content
    assert "grounding" in result.llm_content


def test_skill_tool_reports_available_names_on_miss(tmp_path: Path) -> None:
    """未命中要给出可用列表,让模型下一轮就能改对,而不是反复猜名字。"""
    _write_skill(tmp_path, "grounding")
    tool = SkillTool(discover_skills([tmp_path]))

    result = tool.build({"name": "grasping"}).execute(
        ToolContext(root=tmp_path, workspace=tmp_path / ".agentx")
    )

    assert result.is_error
    assert "grasping" in result.error
    assert "grounding" in result.error


def test_skill_tool_surfaces_the_output_contract(tmp_path: Path) -> None:
    """``produces_outputs`` 是技能之间的接口,要显式回灌给模型。"""
    _write_skill(
        tmp_path,
        "grounding",
        front="name: grounding\ndescription: 定位\nproduces_outputs:\n  mask: 掩码",
    )
    tool = SkillTool(discover_skills([tmp_path]))

    result = tool.build({"name": "grounding"}).execute(
        ToolContext(root=tmp_path, workspace=tmp_path / ".agentx")
    )

    assert "mask(掩码)" in result.llm_content


def test_skill_tool_never_truncates(tmp_path: Path) -> None:
    """截断过的配方比没有配方更糟:模型会照着半截代码写,错得更隐蔽。"""
    assert SkillTool([]).max_output_chars is None


def test_skill_tool_schema_has_no_enum(tmp_path: Path) -> None:
    """技能名进 enum 会让 schema 随技能集变化,破坏 prompt cache。"""
    _write_skill(tmp_path, "grounding")
    tool = SkillTool(discover_skills([tmp_path]))

    properties = tool.schema()["function"]["parameters"]["properties"]

    assert "enum" not in properties["name"]
    assert "grounding" in properties["name"]["description"]


# --- 接进 agent -------------------------------------------------------------


def test_agent_exposes_skill_tool_and_index(tmp_path: Path) -> None:
    _write_skill(tmp_path / "skills", "grounding")

    agent = CodingAgent(
        provider=_ScriptedProvider(),
        root=tmp_path / "work",
        skill_roots=[tmp_path / "skills"],
    )

    assert [s.name for s in agent.skills] == ["grounding"]
    assert "skill" in {s["function"]["name"] for s in agent.tool_schemas}
    assert "<available_skills>" in agent.system_prompt
    assert "grounding" in agent.system_prompt


def test_agent_without_skills_is_unchanged(tmp_path: Path) -> None:
    """没有技能时既不注册工具也不注入段落 —— 零技能路径必须与从前逐字相同。"""
    baseline = CodingAgent(provider=_ScriptedProvider(), root=tmp_path / "work")
    empty = CodingAgent(
        provider=_ScriptedProvider(),
        root=tmp_path / "work",
        skill_roots=[tmp_path / "nope"],
    )

    assert empty.skills == []
    assert "skill" not in {s["function"]["name"] for s in empty.tool_schemas}
    assert "<available_skills>" not in empty.system_prompt
    assert empty.system_prompt == baseline.system_prompt


def test_agent_keeps_skill_tool_under_an_allow_list(tmp_path: Path) -> None:
    """allow 名单是给「限制子 agent」用的,不该顺手把刚挂上的技能屏蔽掉。"""
    _write_skill(tmp_path / "skills", "grounding")

    agent = CodingAgent(
        provider=_ScriptedProvider(),
        root=tmp_path / "work",
        skill_roots=[tmp_path / "skills"],
        allow_tools=["read_file"],
    )

    assert {s["function"]["name"] for s in agent.tool_schemas} == {"read_file", "skill"}
