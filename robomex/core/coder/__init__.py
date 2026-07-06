"""共享的 Coding Agent(“the coder”)。

当前主路径服务于执行器(:class:`~robomex.agents.executor.CodeAsPolicyAgent`):
感知可用技能、按需拉取技能正文、在沙箱写/跑代码、拿反馈、终止。这套公共运行时
就放在这里;Act 与任务驱动 SubAgent 都复用该内核。循环结构与
技能渐进披露均对齐 qwen-code。
"""

from robomex.core.coder.action import (
    AgentAction,
    BlockExecutor,
    ModelTurn,
    SkillEntry,
    ToolCall,
    build_skill_llm_content,
    normalized_action_json,
    parse_action_payload,
    parse_model_turn,
    parse_action,
    render_available_skills,
)
from robomex.core.coder.agent import CodingAgent
from robomex.core.coder.policy import (
    CompletionPolicy,
    LLMCodePolicy,
    ScriptedCodePolicy,
)
from robomex.core.coder.trace import AgentTrace, TurnRecord

__all__ = [
    "AgentTrace",
    "AgentAction",
    "BlockExecutor",
    "CodingAgent",
    "CompletionPolicy",
    "LLMCodePolicy",
    "ModelTurn",
    "ScriptedCodePolicy",
    "SkillEntry",
    "ToolCall",
    "TurnRecord",
    "build_skill_llm_content",
    "normalized_action_json",
    "parse_action_payload",
    "parse_model_turn",
    "parse_action",
    "render_available_skills",
]
