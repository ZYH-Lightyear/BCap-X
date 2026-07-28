"""Tool / Invocation 两阶段抽象。

照抄 Qwen-Code 的 ``DeclarativeTool`` → ``build()`` → ``ToolInvocation`` →
``execute()``。为什么值得抄这一层间接:它把「参数错」和「执行错」分成了两类可以
区别处理的失败。参数错在执行前就被拦下,模型收到的是精确的 schema 反馈而不是一段
运行时 traceback;而且 invocation 拿到的参数已经校验过,工具实现里不需要再写一遍
防御性检查。

工具作者只需要:声明 ``name`` / ``description`` / ``parameters``,实现
``create_invocation``。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentx.contracts import ToolKind, ToolResult, ToolValidationError

try:
    import jsonschema
except ImportError:  # pragma: no cover - jsonschema 是可选依赖
    jsonschema = None  # type: ignore[assignment]


@dataclass
class ToolContext:
    """执行期上下文。

    所有工具共享同一个实例。放在这里的东西必须是「每个工具都可能要用」的,
    单个工具自己的配置应该走构造函数。
    """

    #: 工作目录。文件类工具必须把路径约束在这个根之下。
    root: Path
    #: 本次运行的产物目录(截断落盘、渲染图等)。
    workspace: Path

    def resolve(self, raw: str) -> Path:
        """把工具参数里的路径解析成绝对路径,并拒绝逃出 ``root`` 的路径。

        这是唯一的路径入口 —— 任何工具自己拼路径都是 bug。用 ``resolve()`` 而非
        ``absolute()``,因为要先展开 symlink 和 ``..`` 再比对,否则
        ``root/../../etc/passwd`` 会被放行。
        """
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        root = self.root.resolve()
        if resolved != root and root not in resolved.parents:
            raise ToolValidationError(
                f"路径越界:{resolved} 不在工作目录 {root} 之内"
            )
        return resolved


class Invocation(ABC):
    """一次已校验、可执行的调用。"""

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    @abstractmethod
    def describe(self) -> str:
        """一行摘要,用于日志与 trace。"""

    @abstractmethod
    def execute(self, ctx: ToolContext) -> ToolResult:
        """执行。

        实现约定:**不要抛异常来表达业务失败**。工具跑失败(文件不存在、命令
        返回非零、超时)都是要回灌给模型让它自己修的普通结果,应该返回带
        ``error`` 的 :class:`ToolResult`。只有真正的程序 bug 才该抛出,由
        scheduler 统一兜住。
        """


class Tool(ABC):
    """工具声明。"""

    #: 发给模型的工具名。必须与 schema 里一致。
    name: str = ""
    description: str = ""
    #: JSON Schema(OpenAI function calling 的 ``parameters``)。
    parameters: dict[str, Any] = {}
    kind: ToolKind = ToolKind.OTHER
    #: 结果回灌给模型前的字符上限。``None`` 表示由工具自管(例如 read_file 自己分页)。
    max_output_chars: int | None = 25_000
    #: 截断时保留哪一头。``both`` 对 shell 尤其重要 —— 报错常在末尾,而命令回显在开头。
    truncate_keep: str = "both"

    @abstractmethod
    def create_invocation(self, params: dict[str, Any]) -> Invocation:
        """构造 invocation。此时参数已通过 schema 校验。"""

    def build(self, params: dict[str, Any]) -> Invocation:
        """校验参数并构造 invocation。

        :raises ToolValidationError: 参数不合 schema,或子类的额外校验不通过。
        """
        coerced = _coerce_params(self.parameters, params)
        error = validate_schema(self.parameters, coerced)
        if error:
            raise ToolValidationError(error)
        return self.create_invocation(coerced)

    def schema(self) -> dict[str, Any]:
        """导出 OpenAI function-calling 的 tool schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def validate_schema(schema: dict[str, Any], params: dict[str, Any]) -> str | None:
    """按 JSON Schema 校验;通过返回 ``None``,否则返回给模型看的错误说明。"""

    if jsonschema is None:
        return _validate_minimal(schema, params)
    try:
        jsonschema.validate(params, schema)
    except jsonschema.ValidationError as exc:  # type: ignore[union-attr]
        location = "/".join(str(part) for part in exc.absolute_path) or "(root)"
        return f"参数校验失败 at {location}: {exc.message}"
    except jsonschema.SchemaError as exc:  # type: ignore[union-attr]
        return f"工具自身的 schema 有误: {exc.message}"
    return None


def _validate_minimal(schema: dict[str, Any], params: dict[str, Any]) -> str | None:
    """jsonschema 缺失时的兜底:只查必填项与顶层类型。"""

    for key in schema.get("required", []):
        if key not in params:
            return f"参数校验失败:缺少必填字段 {key!r}"
    return None


def _coerce_params(schema: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """按 schema 对顶层参数做温和的类型强转。

    模型经常把布尔写成 ``"true"``、把整数写成 ``"5"``、把对象写成 JSON 字符串。
    这些是纯粹的格式抖动,不是意图错误,直接判错然后多烧一轮往返不划算。
    Qwen-Code 在 Ajv 校验失败前也做同类强转,这里抄的是同一个思路。

    只处理顶层,不递归:嵌套结构里的抖动少见得多,而递归强转会开始猜测模型意图,
    那就越界了。
    """

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return dict(params)

    coerced = dict(params)
    for key, spec in properties.items():
        if key not in coerced or not isinstance(spec, dict):
            continue
        value = coerced[key]
        if not isinstance(value, str):
            continue
        expected = spec.get("type")
        text = value.strip()

        if expected == "boolean" and text.lower() in {"true", "false"}:
            coerced[key] = text.lower() == "true"
        elif expected == "integer":
            try:
                coerced[key] = int(text, 10)
            except ValueError:
                pass
        elif expected == "number":
            try:
                coerced[key] = float(text)
            except ValueError:
                pass
        elif expected in {"object", "array"} and text[:1] in {"{", "["}:
            try:
                coerced[key] = json.loads(text)
            except json.JSONDecodeError:
                pass

    return coerced


__all__ = [
    "Invocation",
    "Tool",
    "ToolContext",
    "validate_schema",
]
