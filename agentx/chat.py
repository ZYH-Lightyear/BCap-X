"""对话会话:history 维护与不变式修复。

夹在 provider 与 agent 循环之间。目前做不变式修复(孤儿 tool call)与图像修剪;
文本压缩尚未实现,接缝留在 :meth:`ChatSession.token_estimate`。

把 history 单独成层的好处,在移植 Qwen-Code 时看得很清楚:压缩、修剪、孤儿修复
这些都是「对 history 做手术」的操作,放在这一层,agent 循环就能保持成一个干净的
状态机,不必知道自己的历史被动过。
"""

from __future__ import annotations

from typing import Any

from agentx.contracts import Message, ModelResponse, ToolCall
from agentx.providers.base import ModelProvider


class ChatSession:
    """一次对话的完整 history 加上发送逻辑。"""

    def __init__(
        self,
        provider: ModelProvider,
        system_prompt: str,
        *,
        max_images: int | None = None,
    ) -> None:
        """
        :param max_images: history 里最多保留多少张图,``None`` 表示不限。超出的
            从旧到新摘掉。默认不限:纯编码场景里的图往往是规格说明(设计稿、报错
            截图),放多久都还是对的,不该自作主张删。会随时间失效的观测类图像由
            调用方显式设限。
        """
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_images = max_images
        self.history: list[Message] = [{"role": "system", "content": system_prompt}]
        self.usage: dict[str, int] = {}

    def append(self, message: Message) -> None:
        self.history.append(message)

    def extend(self, messages: list[Message]) -> None:
        self.history.extend(messages)

    def send(self, tools: list[dict[str, Any]] | None = None) -> ModelResponse:
        """带上当前 history 发一次请求,并把模型回复写回 history。"""

        self.repair_orphaned_tool_calls()
        self.prune_images()
        response = self.provider.generate(self.history, tools)
        self.append(_assistant_message(response))
        self._accumulate_usage(response.usage)
        return response

    def repair_orphaned_tool_calls(self) -> int:
        """给缺少 tool 响应的 tool call 补上错误响应;返回修补的条数。

        协议要求 assistant 消息里的每个 tool call id 都有一条对应的 tool 消息。
        scheduler 已经保证了正常路径,但异常路径仍可能留下孤儿——比如上一轮在
        执行到一半时被取消。孤儿会让下一轮请求被端点整个拒掉,且报错信息基本没有
        指向性,所以宁可在发送前主动扫一遍。
        """

        repaired = 0
        index = 0
        while index < len(self.history):
            message = self.history[index]
            index += 1
            if message.get("role") != "assistant":
                continue
            call_ids = [
                call.get("id")
                for call in message.get("tool_calls") or []
                if isinstance(call, dict) and call.get("id")
            ]
            if not call_ids:
                continue

            # 紧随其后的连续 tool 消息就是这批调用的响应。
            answered: set[str] = set()
            scan = index
            while scan < len(self.history) and self.history[scan].get("role") == "tool":
                answered.add(self.history[scan].get("tool_call_id"))
                scan += 1

            missing = [cid for cid in call_ids if cid not in answered]
            for offset, call_id in enumerate(missing):
                self.history.insert(
                    scan + offset,
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": "该调用没有完成(会话被中断),结果不可用。",
                    },
                )
                repaired += 1
            index = scan + len(missing)

        return repaired

    def prune_images(self) -> int:
        """只保留最近 ``max_images`` 张图,更早的摘掉;返回摘掉的张数。

        图像是 history 里唯一「过期之后还会主动误导模型」的内容。文字带得上轮次
        标记,模型看得出那是旧的;而十轮前的画面和当前画面长得一模一样,没有任何
        标记能对抗「看到什么就以为是现在」。所以对观测类图像,正确的操作不是压缩
        成摘要,是**作废**。

        摘的是 ``image_url`` 片段而不是整条消息:同一条消息里的文字(观测头部、
        工具说明)要留下,它们便宜且自带时序。只有当一条 user 消息除了图之外什么
        都不剩时才整条丢掉。``tool`` 消息一律保留 —— 删掉会让 tool_call 变成孤儿。
        """
        if self.max_images is None:
            return 0

        budget = self.max_images
        dropped = 0
        rebuilt: list[Message] = []
        # 从后往前:预算优先留给最近的图。
        for message in reversed(self.history):
            content = message.get("content")
            if not isinstance(content, list):
                rebuilt.append(message)
                continue

            kept: list[Any] = []
            for part in content:
                if not (isinstance(part, dict) and part.get("type") == "image_url"):
                    kept.append(part)
                    continue
                if budget > 0:
                    budget -= 1
                    kept.append(part)
                else:
                    dropped += 1

            if len(kept) == len(content):
                rebuilt.append(message)
            elif message.get("role") == "user" and not _has_text(kept):
                continue
            else:
                rebuilt.append({**message, "content": kept})

        if dropped:
            self.history = list(reversed(rebuilt))
        return dropped

    def token_estimate(self) -> int:
        """粗略估算 history 的 token 数。

        按 4 字符 ≈ 1 token 估。只用于压缩触发这类阈值判断,不用于计费,所以不值得
        为它引入 tokenizer 依赖。M1 尚无消费者,是 M3 压缩的接缝。
        """
        total = 0
        for message in self.history:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += len(str(part.get("text", "")))
                    else:
                        # 图片按固定成本计,base64 长度和实际 token 数没有可用的对应关系。
                        total += 4000
        return total // 4

    def _accumulate_usage(self, usage: dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value


def _has_text(parts: list[Any]) -> bool:
    return any(
        isinstance(part, dict) and part.get("type") == "text" and str(part.get("text", "")).strip()
        for part in parts
    )


def _assistant_message(response: ModelResponse) -> Message:
    """把模型回复还原成一条符合协议的 assistant 消息。"""

    message: Message = {"role": "assistant", "content": response.text or None}
    if response.tool_calls:
        message["tool_calls"] = [_tool_call_wire(call) for call in response.tool_calls]
    return message


def _tool_call_wire(call: ToolCall) -> dict[str, Any]:
    import json

    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.args, ensure_ascii=False)},
    }


__all__ = ["ChatSession"]
