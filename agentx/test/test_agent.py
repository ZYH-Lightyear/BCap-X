"""agentx M1 测试。

覆盖:两阶段校验、调度不变式、并发分批、截断、循环终止条件、多模态回灌。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentx.agent import CodingAgent
from agentx.chat import ChatSession
from agentx.contracts import (
    ModelResponse,
    TerminateMode,
    ToolCall,
    ToolKind,
    ToolResult,
    ToolValidationError,
)
from agentx.core import (
    HALLUCINATION_REBUKE,
    NUDGE,
    AgentCore,
    RunConfig,
    _looks_hallucinated,
)
from agentx.prompt import build_system_prompt
from agentx.providers.openai import _parse_response, _parse_tool_calls
from agentx.providers.text_protocol import (
    TextProtocolProvider,
    parse_tool_calls,
    rewrite_history,
)
from agentx.scheduler import RETRY_LOOP_DIRECTIVE, ToolScheduler
from agentx.tools import ToolContext, ToolRegistry, default_registry
from agentx.tools.base import Invocation, Tool
from agentx.truncation import truncate_output


class _ScriptedProvider:
    """按脚本回放模型回复,并记录每次收到的 messages。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.seen: list[list[dict]] = []

    def generate(self, messages, tools=None):
        # 深拷贝:AgentCore 会继续往同一个 list 上追加,不拷就全是最后一帧。
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        if not self.responses:
            return ModelResponse(text="done")
        return self.responses.pop(0)


def _call(name: str, args: dict[str, Any], call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, args=args)


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(root=tmp_path, workspace=tmp_path / ".agentx")


def _scheduler(tmp_path: Path, registry: ToolRegistry | None = None) -> ToolScheduler:
    return ToolScheduler(registry or default_registry(), _ctx(tmp_path))


# ---------------------------------------------------------------------------
#  1. 两阶段:校验通过才构造 invocation
# ---------------------------------------------------------------------------

def test_build_rejects_missing_required_param() -> None:
    from agentx.tools.files import ReadFileTool

    with pytest.raises(ToolValidationError, match="path"):
        ReadFileTool().build({})


def test_build_coerces_stringified_scalars() -> None:
    """模型把整数写成字符串是纯格式抖动,不该多烧一轮往返。"""
    from agentx.tools.files import ReadFileTool

    invocation = ReadFileTool().build({"path": "a.txt", "offset": "3", "limit": "10"})

    assert invocation.params["offset"] == 3
    assert invocation.params["limit"] == 10


def test_edit_rejects_noop_before_touching_disk(tmp_path: Path) -> None:
    from agentx.tools.files import EditTool

    with pytest.raises(ToolValidationError):
        EditTool().build({"path": "a.txt", "old_string": "x", "new_string": "x"})


# ---------------------------------------------------------------------------
#  2. 「进必有出」不变式
# ---------------------------------------------------------------------------

def test_every_call_gets_exactly_one_tool_message(tmp_path: Path) -> None:
    """协议要求每个 tool_call_id 都有一条 tool 消息,少一条下一轮整个请求会被拒。

    这里把四条失败路径一次性打包:未知工具、坏 JSON、参数不合 schema、执行期异常。
    """

    class _Exploding(Tool):
        name = "explode"
        description = "always raises"
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}

        def create_invocation(self, params):
            return _ExplodingInvocation(params)

    class _ExplodingInvocation(Invocation):
        def describe(self):
            return "explode"

        def execute(self, ctx):
            raise RuntimeError("boom")

    registry = default_registry()
    registry.register(_Exploding())
    scheduler = _scheduler(tmp_path, registry)

    calls = (
        _call("no_such_tool", {}, "c1"),
        ToolCall(id="c2", name="read_file", parse_error="arguments 不是合法 JSON"),
        _call("read_file", {}, "c3"),
        _call("explode", {}, "c4"),
    )
    outcome = scheduler.execute(calls)

    answered = [m["tool_call_id"] for m in outcome.messages if m["role"] == "tool"]
    assert answered == ["c1", "c2", "c3", "c4"]
    assert all(record.result.is_error for record in outcome.records)


def test_duplicate_call_ids_are_deduped(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")

    outcome = scheduler.execute((
        _call("read_file", {"path": "a.txt"}, "same"),
        _call("read_file", {"path": "a.txt"}, "same"),
    ))

    assert [m["tool_call_id"] for m in outcome.messages if m["role"] == "tool"] == ["same"]


def test_chat_repairs_orphaned_tool_calls() -> None:
    chat = ChatSession(_ScriptedProvider([]), "sys")
    chat.history.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "x", "arguments": "{}"}},
        ],
    })
    chat.history.append({"role": "tool", "tool_call_id": "a", "content": "ok"})

    assert chat.repair_orphaned_tool_calls() == 1

    answered = [m["tool_call_id"] for m in chat.history if m.get("role") == "tool"]
    assert answered == ["a", "b"]
    # 补的响应必须紧跟在原有 tool 消息之后,不能落到后面的 user 轮里。
    assert chat.history[-1]["tool_call_id"] == "b"


# ---------------------------------------------------------------------------
#  3. 并发分批
# ---------------------------------------------------------------------------

def test_partition_keeps_order_and_isolates_writes(tmp_path: Path) -> None:
    """只读工具合批并行,写工具各自单独串行,且原有顺序不能被打乱。"""
    from agentx.scheduler import _partition, _Prepared

    def prepared(kind: ToolKind) -> _Prepared:
        tool = type("T", (Tool,), {"kind": kind, "create_invocation": lambda s, p: None})()
        return _Prepared(_call("t", {}), tool, object(), None)

    batches = _partition([
        prepared(ToolKind.READ),
        prepared(ToolKind.SEARCH),
        prepared(ToolKind.EDIT),
        prepared(ToolKind.READ),
    ])

    assert [len(b) for b in batches] == [2, 1, 1]


def test_parallel_reads_all_complete(tmp_path: Path) -> None:
    for i in range(4):
        (tmp_path / f"f{i}.txt").write_text(f"content {i}", encoding="utf-8")
    scheduler = _scheduler(tmp_path)

    outcome = scheduler.execute(tuple(
        _call("read_file", {"path": f"f{i}.txt"}, f"c{i}") for i in range(4)
    ))

    assert len(outcome.records) == 4
    assert not outcome.any_error


# ---------------------------------------------------------------------------
#  4. 截断
# ---------------------------------------------------------------------------

def test_truncate_keeps_both_ends_and_saves_overflow(tmp_path: Path) -> None:
    """报错通常在末尾、命令回显在开头,两头都得留。"""
    text = "HEAD" + ("x" * 5000) + "TAIL"

    out = truncate_output(text, max_chars=200, keep="both", overflow_dir=tmp_path, label="shell")

    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "已省略" in out
    saved = list(tmp_path.glob("shell_*.txt"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == text


def test_read_file_output_is_not_truncated_by_scheduler(tmp_path: Path) -> None:
    """read_file 自管分页,再截一刀只会把行号切断。"""
    (tmp_path / "big.txt").write_text("\n".join(f"line {i}" for i in range(5000)), encoding="utf-8")
    scheduler = _scheduler(tmp_path)

    outcome = scheduler.execute((_call("read_file", {"path": "big.txt"}),))

    assert "已省略" not in outcome.messages[0]["content"]


# ---------------------------------------------------------------------------
#  5. 熔断
# ---------------------------------------------------------------------------

def test_repeated_identical_validation_error_triggers_directive(tmp_path: Path) -> None:
    scheduler = _scheduler(tmp_path)

    contents = [
        scheduler.execute((_call("read_file", {}, f"c{i}"),)).messages[0]["content"]
        for i in range(3)
    ]

    assert "重试循环" not in contents[0]
    assert "重试循环" not in contents[1]
    assert RETRY_LOOP_DIRECTIVE.split("{")[0] in contents[2]


def test_a_successful_call_clears_the_failure_counter(tmp_path: Path) -> None:
    """连续才是熔断判据。只增不减的计数器迟早会误伤正常运行。"""
    (tmp_path / "a.txt").write_text("ok", encoding="utf-8")
    scheduler = _scheduler(tmp_path)

    scheduler.execute((_call("read_file", {}, "c1"),))
    scheduler.execute((_call("read_file", {}, "c2"),))
    scheduler.execute((_call("read_file", {"path": "a.txt"}, "c3"),))
    third = scheduler.execute((_call("read_file", {}, "c4"),))

    assert "重试循环" not in third.messages[0]["content"]


# ---------------------------------------------------------------------------
#  6. 循环终止
# ---------------------------------------------------------------------------

def test_text_without_tool_calls_ends_the_run(tmp_path: Path) -> None:
    provider = _ScriptedProvider([ModelResponse(text="重构完成,测试全绿。")])
    core = AgentCore(ChatSession(provider, "sys"), _scheduler(tmp_path), [])

    result = core.run("做点事")

    assert result.terminate_mode is TerminateMode.GOAL
    assert result.text == "重构完成,测试全绿。"
    assert result.turns == 1


def test_tool_calls_continue_the_loop(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    provider = _ScriptedProvider([
        ModelResponse(tool_calls=(_call("read_file", {"path": "a.txt"}),)),
        ModelResponse(text="文件内容是 hello。"),
    ])
    core = AgentCore(ChatSession(provider, "sys"), _scheduler(tmp_path), [])

    result = core.run("读 a.txt")

    assert result.terminate_mode is TerminateMode.GOAL
    assert result.turns == 2
    assert len(result.tool_records) == 1
    # 第二轮的 prompt 必须已经带上第一轮的工具结果。
    assert any(m.get("role") == "tool" for m in provider.seen[1])


def test_empty_response_is_nudged_not_terminated(tmp_path: Path) -> None:
    """空回复常是模型在等一个不存在的信号,直接终止会把可挽救的运行判死。"""
    provider = _ScriptedProvider([
        ModelResponse(text="   "),
        ModelResponse(text="好的,任务完成。"),
    ])
    core = AgentCore(ChatSession(provider, "sys"), _scheduler(tmp_path), [])

    result = core.run("做点事")

    assert result.terminate_mode is TerminateMode.GOAL
    assert any(m.get("content") == NUDGE for m in provider.seen[1])


def test_max_turns_terminates(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    provider = _ScriptedProvider([
        ModelResponse(tool_calls=(_call("read_file", {"path": "a.txt"}, f"c{i}"),))
        for i in range(10)
    ])
    core = AgentCore(
        ChatSession(provider, "sys"), _scheduler(tmp_path), [], RunConfig(max_turns=3)
    )

    result = core.run("永远读下去")

    assert result.terminate_mode is TerminateMode.MAX_TURNS
    assert result.turns == 3
    assert "轮次上限" in result.detail


def test_provider_failure_surfaces_as_error(tmp_path: Path) -> None:
    class _Broken:
        def generate(self, messages, tools=None):
            raise ConnectionError("端点不可达")

    core = AgentCore(ChatSession(_Broken(), "sys"), _scheduler(tmp_path), [])

    result = core.run("做点事")

    assert result.terminate_mode is TerminateMode.ERROR
    assert "端点不可达" in result.detail


# ---------------------------------------------------------------------------
#  7. 多模态
# ---------------------------------------------------------------------------

def test_image_result_becomes_a_separate_user_message(tmp_path: Path) -> None:
    """OpenAI 的 tool 消息塞不进图片,只能另起一条 user 消息承载。"""
    from PIL import Image

    Image.new("RGB", (8, 8), (200, 30, 30)).save(tmp_path / "shot.png")
    scheduler = _scheduler(tmp_path)

    outcome = scheduler.execute((_call("read_file", {"path": "shot.png"}),))

    assert outcome.messages[0]["role"] == "tool"
    assert outcome.messages[1]["role"] == "user"
    parts = outcome.messages[1]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def _image_message(text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }


def _image_count(message: dict) -> int:
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(1 for part in content if part.get("type") == "image_url")


def _text_of(message: dict) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return str(content)
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


def test_pruning_keeps_the_most_recent_images_and_all_the_text() -> None:
    """旧观测图会冒充当前状态,必须作废;文字自带轮次标记,留着不会骗人。"""
    chat = ChatSession(_ScriptedProvider([]), "sys", max_images=2)
    for index in range(5):
        chat.append(_image_message(f"obs {index}"))

    assert chat.prune_images() == 3

    assert [_text_of(m) for m in chat.history[1:]] == [f"obs {i}" for i in range(5)]
    assert [_text_of(m) for m in chat.history if _image_count(m)] == ["obs 3", "obs 4"]


def test_pruning_drops_messages_that_held_nothing_but_an_image() -> None:
    chat = ChatSession(_ScriptedProvider([]), "sys", max_images=1)
    chat.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]})
    chat.append(_image_message("现在"))

    chat.prune_images()

    assert [_text_of(m) for m in chat.history[1:]] == ["现在"]


def test_images_are_never_dropped_without_an_explicit_limit() -> None:
    """纯编码场景里的图往往是规格说明,放多久都还是对的,不该自作主张删。"""
    chat = ChatSession(_ScriptedProvider([]), "sys")
    for index in range(4):
        chat.append(_image_message(f"obs {index}"))

    assert chat.prune_images() == 0
    assert sum(_image_count(m) for m in chat.history) == 4


def test_send_prunes_before_the_request_goes_out() -> None:
    provider = _ScriptedProvider([ModelResponse(text="ok")])
    chat = ChatSession(provider, "sys", max_images=1)
    chat.append(_image_message("旧"))
    chat.append(_image_message("新"))

    chat.send()

    assert sum(_image_count(m) for m in provider.seen[0]) == 1


# ---------------------------------------------------------------------------
#  8. 路径约束
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("escape", ["../outside.txt", "/etc/passwd"])
def test_paths_cannot_escape_the_workspace(tmp_path: Path, escape: str) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    ctx = ToolContext(root=root, workspace=root / ".agentx")

    with pytest.raises(ToolValidationError, match="越界"):
        ctx.resolve(escape)


def test_escaping_path_is_reported_as_tool_error_not_crash(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    scheduler = ToolScheduler(default_registry(), ToolContext(root=root, workspace=root / ".x"))

    outcome = scheduler.execute((_call("read_file", {"path": "../../etc/passwd"}),))

    assert outcome.records[0].result.is_error
    assert "越界" in outcome.messages[0]["content"]


# ---------------------------------------------------------------------------
#  9. Provider 解析
# ---------------------------------------------------------------------------

def test_bad_tool_arguments_json_is_carried_not_raised() -> None:
    """坏 JSON 要带着 call id 回灌给模型;抛异常会连 id 一起丢掉,后面就补不出配对。"""
    calls = _parse_tool_calls([
        {"id": "c1", "function": {"name": "read_file", "arguments": "{not json"}}
    ])

    assert len(calls) == 1
    assert calls[0].id == "c1"
    assert calls[0].parse_error is not None


def test_parse_response_extracts_text_and_calls() -> None:
    response = _parse_response({
        "choices": [{
            "message": {
                "content": "让我看看",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "glob", "arguments": '{"pattern":"*.py"}'}}
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"total_tokens": 42},
    })

    assert response.text == "让我看看"
    assert response.tool_calls[0].args == {"pattern": "*.py"}
    assert response.usage["total_tokens"] == 42


# ---------------------------------------------------------------------------
#  10. 装配
# ---------------------------------------------------------------------------

def test_agent_exposes_tool_schemas_and_honours_deny(tmp_path: Path) -> None:
    agent = CodingAgent(_ScriptedProvider([]), root=tmp_path, deny_tools=["run_shell_command"])

    names = [s["function"]["name"] for s in agent.tool_schemas]
    assert "read_file" in names
    assert "run_shell_command" not in names
    # schema 必须是 OpenAI function-calling 的形状,否则端点直接 400。
    assert all(s["type"] == "function" and "parameters" in s["function"] for s in agent.tool_schemas)


def test_system_prompt_states_the_finishing_contract(tmp_path: Path) -> None:
    """没有 finish 工具,「不调工具 + 给文本」就是终止信号,prompt 必须讲清楚。"""
    prompt = build_system_prompt(tmp_path)

    assert "no tool calls" in prompt
    assert str(tmp_path) in prompt


def test_hallucinated_transcript_is_not_accepted_as_the_final_answer(tmp_path: Path) -> None:
    """最坏的失败是不报错的那种。

    某些路由会间歇性掉出原生 tool call 模式,转而在正文里把「调用」和它自己想象的
    「结果」交替写出来。那段文本读起来像模像样,但一次都没真正执行过 —— 当成 GOAL
    就等于以成功状态交出一份幻觉。
    """
    fake = (
        "**Tool call: read_file**\n"
        '```json\n{"path": "cart/pricing.py"}\n```\n'
        "<system>Tool execution aborted.</system>\n"
        "根据文件内容,我已修复了税率计算。"
    )
    provider = _ScriptedProvider([
        ModelResponse(text=fake),
        ModelResponse(text="确认:税基已改为折后金额,pytest 全绿。"),
    ])
    core = AgentCore(ChatSession(provider, "sys"), _scheduler(tmp_path), [])

    result = core.run("修个 bug")

    assert result.text == "确认:税基已改为折后金额,pytest 全绿。"
    assert any(m.get("content") == HALLUCINATION_REBUKE for m in provider.seen[1])


def test_genuine_final_answer_is_not_mistaken_for_hallucination(tmp_path: Path) -> None:
    provider = _ScriptedProvider([ModelResponse(text="已修复 total(),税基改为折后金额。")])
    core = AgentCore(ChatSession(provider, "sys"), _scheduler(tmp_path), [])

    assert core.run("修个 bug").terminate_mode is TerminateMode.GOAL


@pytest.mark.parametrize(
    "sample",
    [
        "**Tool Call: read_file(cart/pricing.py)**\n\nLet me read the file.",
        "**Tool: read_file**\n```\ncart/pricing.py\n```",
        '<tool_call>{"name": "glob"}</tool_call>',
        "system<system>Tool execution aborted.</system>",
        "**tool_call: run_shell_command(ls)**",
        "tool_call error: File not found",
    ],
)
def test_hallucination_guard_catches_real_world_samples(sample: str) -> None:
    """这些全是实测中真实抓到的样本。

    同一个模型会在 ``**Tool Call:``、``**Tool:``、``<tool_call>`` 之间随机切换写法和
    大小写 —— 最初用字面量匹配时就漏掉了大写的 ``**Tool Call``,导致守卫整个失效。
    """
    assert _looks_hallucinated(sample), f"没能识别出幻觉样本: {sample!r}"


@pytest.mark.parametrize(
    "sample",
    [
        "已修复 total(),税基改为折后金额,pytest 4 passed。",
        "我读了 pricing.py 和 test_pricing.py,发现 subtotal 被调用了两次。",
        "无法完成:仓库里没有 pytest,也没有其它测试框架。",
    ],
)
def test_hallucination_guard_lets_real_answers_through(sample: str) -> None:
    assert not _looks_hallucinated(sample)


# ---------------------------------------------------------------------------
#  10b. 无条件观测
# ---------------------------------------------------------------------------

def test_observation_is_injected_at_the_start_and_after_every_tool_batch(
    tmp_path: Path,
) -> None:
    """观测不是工具:模型没有任何一条路径能跳过它,闭环因此是体制而不是自觉。"""
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    provider = _ScriptedProvider([
        ModelResponse(tool_calls=(_call("read_file", {"path": "a.txt"}),)),
        ModelResponse(text="看完了"),
    ])
    seen: list[tuple[int, list[str]]] = []

    def observe(turn: int, records: list) -> dict:
        seen.append((turn, [r.call.name for r in records]))
        return {"role": "user", "content": f"[观测 T={turn}]"}

    core = AgentCore(
        ChatSession(provider, "sys"), _scheduler(tmp_path), [], RunConfig(observe=observe)
    )
    core.run("读 a.txt")

    # 观测源要能据此判断世界变没变,所以本轮的工具记录必须一起给。
    assert seen == [(0, []), (1, ["read_file"])]
    # 开局观测要赶上第一次请求,而不是等到第二轮才补。
    assert any(m.get("content") == "[观测 T=0]" for m in provider.seen[0])
    assert any(m.get("content") == "[观测 T=1]" for m in provider.seen[1])


def test_observer_can_skip_a_turn_by_returning_none(tmp_path: Path) -> None:
    """世界没变时重复注入一张逐像素相同的图,只是白花上下文。"""
    provider = _ScriptedProvider([ModelResponse(text="done")])
    core = AgentCore(
        ChatSession(provider, "sys"),
        _scheduler(tmp_path),
        [],
        RunConfig(observe=lambda turn, records: None),
    )

    core.run("做点事")

    assert [m["role"] for m in provider.seen[0]] == ["system", "user"]


# ---------------------------------------------------------------------------
#  11. 文本协议
# ---------------------------------------------------------------------------

def test_text_protocol_parses_calls_and_strips_them_from_the_text() -> None:
    text, calls = parse_tool_calls(
        '让我先看看这个文件。\n'
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    )

    assert text == "让我先看看这个文件。"
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].args == {"path": "a.py"}


def test_text_protocol_accepts_contiguous_parallel_calls() -> None:
    _, calls = parse_tool_calls(
        '<tool_call>{"name": "glob", "arguments": {"pattern": "*.py"}}</tool_call>\n'
        '<tool_call>{"name": "grep", "arguments": {"pattern": "def"}}</tool_call>'
    )

    assert [c.name for c in calls] == ["glob", "grep"]


def test_text_protocol_stops_at_the_first_fabricated_result() -> None:
    """连续性规则是对付「模型自己编造工具结果」的结构化手段。

    真正的批量调用挨在一起;幻觉出来的后续调用前面必然隔着一段编造的结果。
    """
    _, calls = parse_tool_calls(
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>\n'
        'File not found: a.py\n'
        '<tool_call>{"name": "glob", "arguments": {"pattern": "**/*.py"}}</tool_call>'
    )

    assert [c.name for c in calls] == ["read_file"]


@pytest.mark.parametrize(
    "body",
    [
        '{"name": "read_file", "args": {"path": "a.py"}}',
        '{"name": "read_file", "input": {"path": "a.py"}}',
        '```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```',
        '{"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\"}"}',
    ],
)
def test_text_protocol_tolerates_format_jitter(body: str) -> None:
    """不同模型家族的训练格式不同,为纯格式抖动多烧一轮往返不值得。"""
    _, calls = parse_tool_calls(f"<tool_call>{body}</tool_call>")

    assert calls[0].name == "read_file"
    assert calls[0].args == {"path": "a.py"}


def test_text_protocol_carries_bad_json_as_parse_error() -> None:
    _, calls = parse_tool_calls('<tool_call>{"name": "read_file", "argum</tool_call>')

    assert len(calls) == 1
    assert calls[0].parse_error is not None


def test_text_protocol_rewrites_orphan_prone_tool_messages() -> None:
    """role=tool 消息在协议上必须由带 tool_calls 的 assistant 认领。

    文本协议下 assistant 只有正文,那些 tool 消息在端点看来全是孤儿,不改写会被
    整个请求拒掉。
    """
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "读一下"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "print(1)"},
    ]

    rewritten = rewrite_history(history, tools=[])

    assert [m["role"] for m in rewritten] == ["system", "user", "assistant", "user"]
    assert "<tool_call>" in rewritten[2]["content"]
    assert '<tool_result name="read_file">' in rewritten[3]["content"]


def test_text_protocol_renders_schemas_into_the_system_prompt() -> None:
    tools = default_registry().schemas()

    rewritten = rewrite_history([{"role": "system", "content": "sys"}], tools)

    system = rewritten[0]["content"]
    assert "<tool_call>" in system
    for name in default_registry().names():
        assert name in system


def test_text_protocol_does_not_forward_tools_to_the_endpoint() -> None:
    """同时开原生和文本两条通道,模型可能一半走一条,解析和配对都会乱。"""

    class _Recorder:
        def __init__(self) -> None:
            self.tools_seen: list[Any] = []

        def generate(self, messages, tools=None):
            self.tools_seen.append(tools)
            return ModelResponse(text="done")

    inner = _Recorder()
    TextProtocolProvider(inner).generate(
        [{"role": "system", "content": "sys"}], tools=default_registry().schemas()
    )

    assert inner.tools_seen == [None]


def test_agent_wires_the_text_protocol_into_the_chat(tmp_path: Path) -> None:
    agent = CodingAgent(_ScriptedProvider([]), root=tmp_path, protocol="text")

    assert isinstance(agent.provider, TextProtocolProvider)
    assert agent.chat.provider is agent.provider


def test_agent_rejects_an_unknown_protocol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="native"):
        CodingAgent(_ScriptedProvider([]), root=tmp_path, protocol="xml")


def test_system_prompt_keeps_tool_names_in_english(tmp_path: Path) -> None:
    """散文是中文,但工具名必须与 schema 里的字面量逐字一致。

    prompt 里把 read_file 写成「读文件工具」不会报任何错,只会让模型在指代时
    对不上号 —— 这类失效是静默的,所以用测试钉住。
    """
    prompt = build_system_prompt(tmp_path)

    for name in default_registry().names():
        assert name in prompt, f"{name} 没有在 system prompt 里以英文原名出现"
