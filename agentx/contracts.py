"""agentx 的核心类型。

这一层刻意只放数据,不放行为:provider、scheduler、core 三个方向都依赖它,
任何一处塞进逻辑都会立刻变成三向耦合。

## 消息表示

history 直接采用 OpenAI wire format 的 dict(``{"role": ..., "content": ...}``),
而不是像 Qwen-Code 那样自建一套 Content/Part 模型再翻译。原因是我们只对接
OpenAI 兼容端点,自建中间模型只会多一层可能出错的双向转换。代价是 history 不是
强类型的,靠 :mod:`agentx.chat` 里的不变式检查兜底。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Literal

#: OpenAI 多模态消息里的一个内容片段。``{"type": "text", ...}`` 或
#: ``{"type": "image_url", ...}``。
ContentPart = dict[str, Any]

#: 一条 chat 消息(OpenAI wire format)。
Message = dict[str, Any]


class ToolKind(enum.Enum):
    """工具的副作用类别。

    唯一的消费者是 scheduler 的并发分批:只读类工具可以并行,其余必须串行。
    """

    READ = "read"
    SEARCH = "search"
    FETCH = "fetch"
    EDIT = "edit"
    EXECUTE = "execute"
    OTHER = "other"


#: 可以并行执行的工具类别。刻意只放纯读:哪怕是「看起来只读」的工具,只要会写盘
#: (例如 todo)就必须排除,否则并行批次里两个工具会互相踩。
CONCURRENCY_SAFE_KINDS = frozenset({ToolKind.READ, ToolKind.SEARCH, ToolKind.FETCH})


class TerminateMode(enum.Enum):
    """推理循环的终止原因。

    照搬 Qwen-Code 的 ``AgentTerminateMode``。``GOAL`` 是唯一的正常出口,其余四个
    都是预算或故障。调用方据此决定要不要重试、要不要把结果当数。
    """

    GOAL = "goal"
    MAX_TURNS = "max_turns"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True)
class ToolCall:
    """模型发起的一次工具调用。

    :param id: 由模型/provider 签发。回灌结果时必须原样带回,是 tool 消息与
        assistant 消息配对的唯一依据。
    :param args: 已经从 JSON 字符串解析成 dict。解析失败时为空 dict,并由
        ``parse_error`` 记录原因 —— 不在这里抛异常,因为「模型给了坏 JSON」是
        要回灌给模型让它自己改的普通失败,不是程序错误。
    """

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None


@dataclass(frozen=True)
class ToolResult:
    """一次工具执行的产出。

    ``llm_content`` 与 ``display`` 是两条独立通道,这是 Qwen-Code 最值得抄的设计
    之一:模型该看到截断后的摘要和图片,人该在 trace 里看到完整输出。混成一条的
    后果是要么模型被淹没,要么排查时证据已经被截断掉了。
    """

    #: 模型看到的内容。字符串,或多模态片段列表(可含 image_url)。
    llm_content: str | list[ContentPart]
    #: 人看到的完整内容。留空表示与 ``llm_content`` 相同。
    display: str = ""
    #: 非空表示这次调用失败。失败同样要回灌给模型,不是异常。
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


@dataclass(frozen=True)
class ModelResponse:
    """模型一次回复的规范化结果。"""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolRecord:
    """一次工具调用的完整记录,供 trace 与统计使用。"""

    call: ToolCall
    result: ToolResult
    duration_ms: float


@dataclass
class AgentResult:
    """一次 agent 运行的最终产出。"""

    text: str
    terminate_mode: TerminateMode
    turns: int
    #: 终止原因的补充说明,例如超时的具体秒数、报错的 traceback 摘要。
    detail: str = ""
    tool_records: list[ToolRecord] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.terminate_mode is TerminateMode.GOAL


class ToolValidationError(ValueError):
    """工具参数校验失败。

    由 ``Tool.build`` 抛出,在 scheduler 里被转成一个 error 型 :class:`ToolResult`
    回灌给模型。之所以用异常而不是返回值,是为了让工具作者没法忘记处理 ——
    忘了就会炸在 scheduler 的统一捕获点,而不是静默产出一个无效 invocation。
    """


Role = Literal["system", "user", "assistant", "tool"]

__all__ = [
    "CONCURRENCY_SAFE_KINDS",
    "AgentResult",
    "ContentPart",
    "Message",
    "ModelResponse",
    "Role",
    "TerminateMode",
    "ToolCall",
    "ToolKind",
    "ToolRecord",
    "ToolResult",
    "ToolValidationError",
]
