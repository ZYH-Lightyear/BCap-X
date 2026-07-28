"""运行留痕:把每一次模型请求、响应与工具执行完整落盘。

为什么不复用日志:排查 agent 时要回答的问题是「第 3 轮它到底看到了什么」,这需要
**逐轮的完整 prompt 快照**,而不是一条时间线。日志会把它们打散,目录结构不会。

接入方式刻意做成两个互不依赖的钩子:

* :class:`TracingProvider` 包在 provider 外面,截获真实发出去的 messages 和收回来的
  原始响应。放在这一层而不是 ``AgentCore`` 里,是因为只有这里能看到经过 history
  改写(文本协议)之后**真正上线的**那份 payload。
* :meth:`RunTrace.record_turn` 由 ``RunConfig.on_turn`` 回调,补上工具执行结果。

图片会被替换成占位符 —— 一张 base64 图能把 request.json 撑到几 MB,人就没法读了。
原图另存到 ``images/`` 下,占位符里给出文件名。
"""

from __future__ import annotations

import base64
import copy
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from agentx.contracts import AgentResult, Message, ModelResponse
from agentx.core import TurnSummary
from agentx.providers.base import ModelProvider


class RunTrace:
    """一次运行的落盘目录。

    结构::

        <root>/
          meta.json          模型、协议、任务、时间
          system_prompt.txt  完整系统提示
          tools.json         发给模型的 tool schema
          turn-01/
            request.json     本轮真实发出的完整 messages(图片已抽出)
            response.json    原始文本 + 解析出的 tool calls
            tools.json       本轮每次工具调用的参数、结果、耗时
          images/            从 prompt 里抽出的图片
          result.json        最终结果
    """

    def __init__(self, root: str | Path, meta: dict[str, Any] | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "images").mkdir(exist_ok=True)
        self._turn = 0
        self._image_seq = 0
        self._started = time.monotonic()

        payload = {"started_at": datetime.now().isoformat(timespec="seconds")}
        payload.update(meta or {})
        self._write("meta.json", payload)

    # -- provider 侧 --------------------------------------------------------

    def record_request(self, messages: list[Message], tools: list[dict] | None) -> int:
        """记录一次即将发出的请求,返回轮次号。"""
        self._turn += 1
        turn_dir = self.root / f"turn-{self._turn:02d}"
        turn_dir.mkdir(exist_ok=True)

        if self._turn == 1:
            system = next((m for m in messages if m.get("role") == "system"), None)
            if system:
                (self.root / "system_prompt.txt").write_text(
                    str(system.get("content", "")), encoding="utf-8"
                )
            self._write("tools.json", tools or [])

        self._write(
            f"turn-{self._turn:02d}/request.json",
            {"message_count": len(messages), "messages": self._elide_images(messages)},
        )
        return self._turn

    def record_response(self, turn: int, response: ModelResponse) -> None:
        self._write(
            f"turn-{turn:02d}/response.json",
            {
                "text": response.text,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "args": c.args, "parse_error": c.parse_error}
                    for c in response.tool_calls
                ],
            },
        )

    # -- 循环侧 -------------------------------------------------------------

    def record_turn(self, index: int, summary: TurnSummary) -> None:
        """由 ``RunConfig.on_turn`` 调用,补上工具执行结果。"""
        if not summary.tool_records:
            return
        self._write(
            f"turn-{index:02d}/tools.json",
            [
                {
                    "name": record.call.name,
                    "args": record.call.args,
                    "duration_ms": round(record.duration_ms, 1),
                    "is_error": record.result.is_error,
                    "error": record.result.error,
                    "llm_content": _stringify(record.result.llm_content),
                    "display": record.result.display,
                }
                for record in summary.tool_records
            ],
        )

    def record_result(self, result: AgentResult) -> None:
        payload = asdict(result)
        payload["terminate_mode"] = result.terminate_mode.value
        payload["elapsed_s"] = round(time.monotonic() - self._started, 1)
        # tool_records 已经逐轮存过,这里只留计数,免得 result.json 变成一坨。
        payload["tool_records"] = len(result.tool_records)
        self._write("result.json", payload)

    # -- 内部 ---------------------------------------------------------------

    def _write(self, relative: str, payload: Any) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    def _elide_images(self, messages: list[Message]) -> list[Message]:
        """把 base64 图片抽到 images/ 下,原位留占位符。

        一张图能让 request.json 涨到几 MB,人就没法读了;但图本身又是核验多模态是否
        真的生效的关键证据,所以不能丢,只能挪走。
        """
        cloned = copy.deepcopy(messages)
        for message in cloned:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                url = (part.get("image_url") or {}).get("url", "")
                if not url.startswith("data:"):
                    continue
                name = self._save_image(url)
                part["image_url"] = {"url": f"<elided: images/{name}>"}
        return cloned

    def _save_image(self, data_url: str) -> str:
        self._image_seq += 1
        header, _, payload = data_url.partition(",")
        suffix = "png" if "png" in header else "jpg"
        name = f"turn{self._turn:02d}_{self._image_seq:02d}.{suffix}"
        try:
            (self.root / "images" / name).write_bytes(base64.b64decode(payload))
        except Exception:  # noqa: BLE001 - 留痕失败绝不能影响主流程
            return f"{name} (解码失败)"
        return name


class TracingProvider:
    """包在 provider 外面,把真实上线的 payload 和原始响应记下来。"""

    def __init__(self, inner: ModelProvider, trace: RunTrace) -> None:
        self.inner = inner
        self.trace = trace

    def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        turn = self.trace.record_request(messages, tools)
        response = self.inner.generate(messages, tools)
        self.trace.record_response(turn, response)
        return response


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif item.get("type") == "image_url":
                parts.append("<image>")
        return "\n".join(parts)
    return str(content)


__all__ = ["RunTrace", "TracingProvider"]
