"""共享的 Coding Agent(“the coder”)。

两个角色 agent —— 执行器(:class:`~robomex.agents.executor.CodeAsPolicyAgent`)
和验证器(:class:`~robomex.agents.verifier.VerifyCodeAgent`)—— 是同一种 agent:
感知可用技能、按需拉取技能正文、在沙箱写/跑代码、拿反馈、终止。这套公共运行时
就放在这里;角色只是轻量子类,仅在 prompt/上下文/终止方式上不同。循环结构与
技能渐进披露均对齐 qwen-code。
"""

from robomex.core.coder.action import (
    BlockExecutor,
    SkillEntry,
    build_skill_llm_content,
    parse_action,
    render_available_skills,
)
from robomex.core.coder.agent import CodingAgent
from robomex.core.coder.policy import (
    FINISH,
    CompletionPolicy,
    LLMCodePolicy,
    ScriptedCodePolicy,
)
from robomex.core.coder.trace import AgentTrace, TurnRecord

__all__ = [
    "AgentTrace",
    "BlockExecutor",
    "CodingAgent",
    "CompletionPolicy",
    "FINISH",
    "LLMCodePolicy",
    "ScriptedCodePolicy",
    "SkillEntry",
    "TurnRecord",
    "build_skill_llm_content",
    "parse_action",
    "render_available_skills",
]
