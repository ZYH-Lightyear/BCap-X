"""Coding Agent 的动作空间:把一条模型回复解析成一个动作。

动作空间刻意设计为沙箱代码块(``executor.run_block``)+ qwen-code 式的技能*渐进披露*,
而非 function-calling schema。本模块持有 :class:`~robomex.core.coder.agent.CodingAgent`
循环所驱动的解析/渲染原语:

- :func:`parse_action` —— 把原始回复路由为 ``python`` / ``use_skill`` /
  ``terminal`` / ``empty`` / ``nudge``。
- :class:`SkillEntry` + :func:`render_available_skills` —— 开场的
  ``<available_skills>`` 感知清单(仅名称 + 描述)。
- :func:`build_skill_llm_content` —— 按需拉取某技能正文时返回的文本
  (对应 qwen-code 的 ``buildSkillLlmContent``)。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from robomex.core.sandbox import BlockExecutionResult, SemanticActionBlock

# python 代码块要求在(可选的)``python`` 标记后必须换行,这样 ``json`` 围栏的
# 裁决就绝不会被误判为代码。
_CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_USE_SKILL_RE = re.compile(r"^\s*USE SKILL:\s*([A-Za-z0-9_.\-/]+)\s*$", re.MULTILINE)


class BlockExecutor(Protocol):
    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult: ...


@dataclass(frozen=True)
class SkillEntry:
    """``<available_skills>`` 感知清单中的一行(不含正文)。"""

    name: str
    description: str
    category: str = ""


def render_available_skills(entries: list[SkillEntry]) -> str:
    """渲染 qwen-code 式的 ``<skill>`` 块:只含名称 + 描述,不含正文。"""

    rows = []
    for e in entries:
        desc = f"{e.description} ({e.category})" if e.category else e.description
        rows.append(f"<skill>\n<name>{e.name}</name>\n<description>{desc}</description>\n</skill>")
    return "\n".join(rows)


def build_skill_llm_content(base_dir: Any, body: str) -> str:
    """加载某技能时返回的文本(对应 qwen-code 的 ``buildSkillLlmContent``)。"""

    base = str(base_dir) if base_dir else "(in-memory skill; no base directory)"
    return (
        f"Loaded skill. Base directory for this skill: {base}\n"
        "Resolve any referenced sidecar files (e.g. ref/verify.md, scripts/verify.py) "
        "as absolute paths under this base directory.\n\n"
        f"{body.strip()}\n"
    )


def parse_action(raw: str, is_terminal: Callable[[str], bool]) -> tuple[str, str]:
    """把一条原始模型回复路由成单个动作。

    优先级:python 块(执行) > ``USE SKILL`` 指令(加载) > 空回复(轻推) >
    终止回复 > 其余不可执行内容(轻推)。返回 ``(kind, payload)``,其中 ``kind``
    取值于 ``{python, use_skill, terminal, empty, nudge}``。
    """

    text = raw or ""
    code = _CODE_FENCE.search(text)
    if code:
        return ("python", code.group(1).strip())
    skill = _USE_SKILL_RE.search(text)
    if skill:
        return ("use_skill", skill.group(1).strip())
    if not text.strip():
        return ("empty", "")
    if is_terminal(text):
        return ("terminal", text)
    return ("nudge", text)
