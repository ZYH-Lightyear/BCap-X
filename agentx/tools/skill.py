"""``skill`` 工具:两段式披露的第二段。

模型在 system prompt 里只看到技能的名字和用途(见 :func:`agentx.skills.render_index`),
调用这个工具才把正文取回来。
"""

from __future__ import annotations

from typing import Any

from agentx.contracts import ToolKind, ToolResult
from agentx.skills import SkillMeta
from agentx.tools.base import Invocation, Tool, ToolContext

#: 包在技能正文外面的说明。没有它,模型容易把正文当成参考资料读完就算,
#: 而不是当成当前该执行的做法。
PREAMBLE = (
    "以下是技能 {name!r} 的完整正文。它描述了一套经过验证的做法 —— "
    "请照着做,不要另起炉灶。正文里给出的代码可以直接使用,按当前任务改参数即可。"
)


class SkillTool(Tool):
    """按名字加载一个技能的正文。"""

    name = "skill"
    kind = ToolKind.READ
    #: 不截断。技能正文是作者可控的策展内容,长度心里有数;而截断过的配方比没有
    #: 配方更糟 —— 模型会照着半截代码写,错得更隐蔽。
    max_output_chars = None

    def __init__(self, skills: list[SkillMeta]) -> None:
        self._skills = {skill.name: skill for skill in skills}
        available = ", ".join(sorted(self._skills))
        sample = next(iter(sorted(self._skills)), "skill-name")
        # 措辞刻意写成硬性命令,对齐 Qwen-Code 的 <skills_instructions>。温和版本
        # ("call this before attempting a task a skill covers")实测被 27B 模型无视:
        # 它转头拿原始 API 把技能里那条链路手抄了三轮。工具描述位于
        # tools → system → messages 缓存前缀的最前端,加长它不产生每轮开销。
        self.description = (
            "Execute a skill: load its full, verified procedure into the conversation.\n"
            "\n"
            "<skills_instructions>\n"
            "Before attempting a task, check whether an available skill covers it. A "
            "skill body is a procedure that is already known to work: the exact code to "
            "run, the parameters that matter, and the failure modes to expect.\n"
            "\n"
            f"How to invoke: call this tool with the skill name only, e.g. name={sample!r}.\n"
            "\n"
            "Important:\n"
            f"- Available skills: {available}. Only these names are valid.\n"
            "- When a skill is relevant, you MUST invoke this tool IMMEDIATELY as your "
            "first action for that task, before any other tool call.\n"
            "- This is a BLOCKING REQUIREMENT: load the skill BEFORE writing code that "
            "does what the skill does.\n"
            "- NEVER merely mention a skill in your text response without calling this "
            "tool.\n"
            "- Do NOT reimplement a skill's procedure yourself from the raw APIs given "
            "in the task prompt. Those APIs being available does not mean you should "
            "chain them by hand; the skill already encodes the correct chain and its "
            "pitfalls.\n"
            "- Loading a skill executes nothing and is cheap. If unsure whether a skill "
            "applies, load it and decide after reading.\n"
            "</skills_instructions>"
        )
        # 刻意不把技能名写进 enum。Qwen-Code 在这里踩过坑:enum 会让 schema 随技能集
        # 变化,破坏 prompt cache,而收益只是省下一次「名字写错」的往返 —— 那一次
        # 往返本来就由未命中时返回的可用列表兜住了。
        self.parameters = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": f"Skill name. One of: {available}.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    def create_invocation(self, params: dict[str, Any]) -> Invocation:
        return _SkillInvocation(params, self._skills)


class _SkillInvocation(Invocation):
    def __init__(self, params: dict[str, Any], skills: dict[str, SkillMeta]) -> None:
        super().__init__(params)
        self._skills = skills

    def describe(self) -> str:
        return f"skill {self.params['name']}"

    def execute(self, ctx: ToolContext) -> ToolResult:
        requested = str(self.params["name"]).strip()
        skill = self._skills.get(requested)
        if skill is None:
            available = ", ".join(sorted(self._skills)) or "(没有已加载的技能)"
            message = f"没有名为 {requested!r} 的技能。可用技能:{available}"
            return ToolResult(llm_content=message, error=message)

        header = [
            PREAMBLE.format(name=skill.name),
            f"技能目录:{skill.path}",
            _render_rule(skill),
            f"--- SKILL: {skill.name} ---",
        ]
        body = "\n".join(line for line in header if line) + "\n" + skill.body
        return ToolResult(llm_content=body, display=body)


def _render_rule(skill: SkillMeta) -> str:
    """把 frontmatter 里的输出契约渲染成一行提示。

    ``produces_outputs`` 是技能之间的接口声明 —— 下游拿什么、叫什么名字。把它显式
    回灌给模型,是为了让它照契约命名变量,而不是随手起名,否则后续交接全靠猜。
    没有声明就不渲染这一行。
    """

    produces = skill.extra.get("produces_outputs")
    if not isinstance(produces, dict) or not produces:
        return ""
    items = ", ".join(f"{key}({value})" for key, value in produces.items())
    return f"该技能的输出契约:{items}"


__all__ = ["PREAMBLE", "SkillTool"]
