"""顶层门面:把 provider / tools / chat / scheduler / core 接成一个 agent。

对应 Qwen-Code 的 ``AgentHeadless`` —— 它自己不含推理逻辑,只负责装配、生命周期
与结果整形。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agentx.chat import ChatSession
from agentx.contracts import AgentResult
from agentx.core import AgentCore, RunConfig
from agentx.prompt import build_system_prompt
from agentx.core import TurnSummary
from agentx.providers.base import ModelProvider
from agentx.providers.text_protocol import TextProtocolProvider
from agentx.scheduler import ToolScheduler
from agentx.skills import discover_skills, render_index
from agentx.trace import RunTrace, TracingProvider
from agentx.tools import SkillTool, ToolContext, ToolRegistry, default_registry


class CodingAgent:
    """一个通用多模态 Coding Agent。"""

    def __init__(
        self,
        provider: ModelProvider,
        root: str | Path,
        *,
        workspace: str | Path | None = None,
        registry: ToolRegistry | None = None,
        run_config: RunConfig | None = None,
        allow_tools: list[str] | None = None,
        deny_tools: list[str] | None = None,
        extra_prompt_sections: list[str] | None = None,
        skill_roots: list[str | Path] | None = None,
        protocol: str = "native",
        max_images: int | None = None,
        trace: RunTrace | None = None,
    ) -> None:
        """
        :param root: 工作目录。文件类工具被约束在此目录内。
        :param workspace: 运行产物目录(溢出落盘等)。默认为 ``root/.agentx``。
        :param allow_tools: 只暴露这些工具;``None`` 表示全部。
        :param deny_tools: 在 allow 之后再剔除。先 allow 后 deny 的顺序让
            「继承全部但排除某几个」这种最常用的配置可以直接表达。
        :param skill_roots: 技能根目录。发现到技能时自动注册 ``skill`` 工具并把
            技能索引注入 system prompt;一个都没发现时完全不动,不留空段落也不多
            一个模型用不上的工具。
        :param protocol: ``native`` 走原生 function calling(能用时更好:解码期受
            schema 约束、并行调用白送);``text`` 走 ``<tool_call>`` 文本协议(只要求
            端点能补全,对吞 ``tools`` 的代理和翻译不干净的兼容层免疫)。
        :param max_images: history 里最多留几张图。观测类图像(每轮注入的场景画面)
            必须设限,否则旧画面会一直冒充当前状态;规格类图像(设计稿)不该设限。
            见 :meth:`ChatSession.prune_images`。
        :param trace: 给定则把每轮的完整请求、响应与工具结果落盘。
        """
        if protocol not in {"native", "text"}:
            raise ValueError(f"protocol 只能是 native 或 text,收到 {protocol!r}")
        self.root = Path(root).resolve()
        self.workspace = Path(workspace) if workspace else self.root / ".agentx"
        self.registry = registry or default_registry()
        self.protocol = protocol
        self.trace = trace

        # TracingProvider 必须裹在 TextProtocolProvider **里面**:文本协议会重写
        # history(把 tool 消息转成 user 的 <tool_result>),只有内层才看得到真正
        # 上线的那份 payload。裹在外面记下来的是改写前的,核验时会对不上。
        base: ModelProvider = provider
        if trace is not None:
            base = TracingProvider(base, trace)
        self.provider = TextProtocolProvider(base) if protocol == "text" else base

        self.run_config = _with_trace(run_config or RunConfig(), trace)

        # 技能要在导出 schema 之前挂好,否则 skill 工具不会出现在这一轮的工具表里。
        self.skills = discover_skills(skill_roots or [])
        sections = list(extra_prompt_sections or [])
        if self.skills:
            self.registry.register(SkillTool(self.skills))
            if allow_tools is not None:
                allow_tools = [*allow_tools, SkillTool.name]
            sections.insert(0, render_index(self.skills))

        self.ctx = ToolContext(root=self.root, workspace=self.workspace)
        self.scheduler = ToolScheduler(self.registry, self.ctx)
        self.tool_schemas = self.registry.schemas(allow=allow_tools, deny=deny_tools)
        self.system_prompt = build_system_prompt(self.root, extra_sections=sections)
        self.chat = ChatSession(self.provider, self.system_prompt, max_images=max_images)

    def run(self, task: str) -> AgentResult:
        """执行一个任务,跑到完成或耗尽预算。"""
        core = AgentCore(
            chat=self.chat,
            scheduler=self.scheduler,
            tools=self.tool_schemas,
            config=self.run_config,
        )
        result = core.run(task)
        if self.trace is not None:
            self.trace.record_result(result)
        return result


def _with_trace(config: RunConfig, trace: RunTrace | None) -> RunConfig:
    """在不覆盖调用方 ``on_turn`` 的前提下挂上留痕回调。"""
    if trace is None:
        return config

    original = config.on_turn

    def chained(index: int, summary: TurnSummary) -> None:
        trace.record_turn(index, summary)
        if original is not None:
            original(index, summary)

    return replace(config, on_turn=chained)


__all__ = ["CodingAgent"]
