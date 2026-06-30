"""共享的 Coding Agent(“the coder”)。

当前主路径服务于执行器(:class:`~robomex.agents.executor.CodeAsPolicyAgent`):
感知可用技能、按需拉取技能正文、在沙箱写/跑代码、拿反馈、终止。这套公共运行时
就放在这里;旧验证 agent 仍可复用该内核,但不再挂在 Act inner loop 上。循环结构与
技能渐进披露均对齐 qwen-code。
"""

from robomex.core.coder.action import (
    AgentAction,
    BlockExecutor,
    ModelTurn,
    SkillEntry,
    ToolCall,
    build_skill_llm_content,
    parse_model_turn,
    parse_action,
    render_available_skills,
)
from robomex.core.coder.agent import CodingAgent
from robomex.core.coder.policy import (
    CompletionPolicy,
    LLMCodePolicy,
    ROBO_MEX_TOOL_SCHEMAS,
    ScriptedCodePolicy,
    VAPIToolCallPolicy,
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
    "ROBO_MEX_TOOL_SCHEMAS",
    "ScriptedCodePolicy",
    "SkillEntry",
    "ToolCall",
    "TurnRecord",
    "VAPIToolCallPolicy",
    "build_skill_llm_content",
    "parse_model_turn",
    "parse_action",
    "render_available_skills",
]
