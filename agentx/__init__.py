"""agentx:通用多模态 Coding Agent。

架构移植自 Qwen-Code 的 headless agent runtime。分层::

    ModelProvider   模型端点(OpenAI 兼容)
    ChatSession     history 与协议不变式
    AgentCore       推理循环
    ToolScheduler   校验、并发分批、截断
    ToolRegistry    工具声明与 schema 导出

用法::

    from agentx import CodingAgent
    from agentx.providers import OpenAIProvider

    agent = CodingAgent(OpenAIProvider(model="..."), root="/path/to/repo")
    result = agent.run("把 foo 模块的测试补齐")
    print(result.text, result.terminate_mode)
"""

from agentx.agent import CodingAgent
from agentx.contracts import AgentResult, TerminateMode, ToolResult
from agentx.core import RunConfig

__all__ = [
    "AgentResult",
    "CodingAgent",
    "RunConfig",
    "TerminateMode",
    "ToolResult",
]
