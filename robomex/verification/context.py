"""为(独立的)Reference-Anchored Verifier 构造上下文。

设计(见文档 §5.6):Verifier 是一个独立角色,可以自己写 judge 代码,但它的上下文
建立在“事实 vs 解读”这条线上——它拿到关于执行器做了什么的一切**事实**(用过哪些技能、
声称的结果、一份脱敏 op-trace、作者写的 rubric + 参考原语),却拿不到执行器的任何
推理 / 自评,这样它的盲区与执行器不相关。

本模块提供数据面:

- ``sanitize_code`` / ``build_op_trace``:把执行器代码变成无注释、无 CoT 的操作轨迹
  (即“实际流程”,默认开启,低污染)。
- ``collect_verify_resources``:即 VerifyRouter——把执行器用过的技能映射到它们的
  ``ref/verify.md`` rubric + ``scripts/verify.py`` 原语。
- ``VerifierContext``:组装好的、可渲染的上下文。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


class _DropStringExprs(ast.NodeTransformer):
    """删除裸字符串字面量语句(docstring / 伪注释)。"""

    def visit_Expr(self, node: ast.Expr) -> Any:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return node


def sanitize_code(code: str) -> str:
    """从执行器代码里剥掉注释、docstring 和 CoT 文字 → 只留操作。

    ``ast.unparse`` 本身就会丢掉注释;我们额外删掉裸字符串语句(agent 常把自然语言
    推理塞在那里)。如果代码无法解析,则退化为丢掉 ``#`` 注释行,仍尽量避免泄漏行内推理。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "\n".join(
            line for line in code.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ).strip()
    tree = _DropStringExprs().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).strip()


def build_op_trace(turns: Iterable[Any], only_successful: bool = True) -> list[str]:
    """从 ``AgentTrace`` 的各轮次得到逐轮脱敏的 op-trace。

    每个轮次需暴露 ``.code`` 与 ``.execution.ok``。失败轮次默认丢弃(它们没有贡献到
    最终产物)。
    """
    trace: list[str] = []
    for turn in turns:
        if only_successful and not getattr(getattr(turn, "execution", None), "ok", True):
            continue
        code = getattr(turn, "code", "") or ""
        cleaned = sanitize_code(code)
        if cleaned:
            trace.append(cleaned)
    return trace


@dataclass(frozen=True)
class VerifyResource:
    """为某个技能路由进来的、作者编写的验证资产。"""

    skill_id: str
    rubric_text: str = ""
    rubric_path: str | None = None
    verifier_path: str | None = None


def collect_verify_resources(
    skills: Iterable[Any],
    skill_ids: Sequence[str],
) -> dict[str, VerifyResource]:
    """VerifyRouter:为每个用过的技能,收集其 rubric + 验证器原语。

    ``skills`` 是任意可迭代的 ``Skill`` 对象(例如 ``[r.skill for r in
    library.all()]``);只保留其中 ``skill_id`` 命中 ``skill_ids`` 的那些。
    """
    wanted = set(skill_ids)
    by_id = {s.skill_id: s for s in skills if s.skill_id in wanted}
    resources: dict[str, VerifyResource] = {}
    for skill_id in skill_ids:
        skill = by_id.get(skill_id)
        if skill is None:
            continue
        rubric_path = skill.verify_doc_path()
        verifier_path = skill.verifier_path()
        resources[skill_id] = VerifyResource(
            skill_id=skill_id,
            rubric_text=Path(rubric_path).read_text() if rubric_path else "",
            rubric_path=str(rubric_path) if rubric_path else None,
            verifier_path=str(verifier_path) if verifier_path else None,
        )
    return resources


@dataclass(frozen=True)
class VerifierContext:
    """一个独立 Verifier 被允许看到的一切——事实,而非叙述。

    刻意排除执行器的思维链 / 自评。原始完整代码*不*嵌在这里;只有 Verifier 主动索取时
    才按需获取(渐进披露)。
    """

    sub_goal: str
    skills_used: tuple[str, ...] = ()
    op_trace: tuple[str, ...] = ()
    resources: dict[str, VerifyResource] = field(default_factory=dict)
    expected_decomposition: str = ""

    def rubrics_text(self) -> str:
        """把所有路由进来的技能的作者 rubric 拼接起来。"""
        chunks = [
            f"### {r.skill_id}\n{r.rubric_text.strip()}"
            for r in self.resources.values() if r.rubric_text.strip()
        ]
        return "\n\n".join(chunks)

    def render(self) -> str:
        """把(只含事实的)验证上下文排版成人/LLM 可读的形式。"""
        lines = [
            f"Sub-goal to verify: {self.sub_goal}",
            f"Skills the executor used: {', '.join(self.skills_used) or '(unknown)'}",
        ]
        if self.expected_decomposition.strip():
            lines += ["", "Expected flow (authored decomposition):",
                      self.expected_decomposition.strip()]
        if self.op_trace:
            lines += ["", "Actual flow (sanitized op-trace):"]
            lines += [f"  [{i}] " + op.replace("\n", "\n      ")
                      for i, op in enumerate(self.op_trace)]
        rubrics = self.rubrics_text()
        if rubrics:
            lines += ["", "Authored success rubrics (ref/verify.md):", rubrics]
        return "\n".join(lines)
