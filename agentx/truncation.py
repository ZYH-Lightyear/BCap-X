"""工具输出截断与落盘。

模型的上下文是稀缺资源,一次 `cat` 大文件或一次啰嗦的构建就能吃掉几万 token。
但直接丢弃又会让模型看不到关键信息,所以策略是**截断 + 全文落盘 + 告诉模型
完整内容在哪**,让它需要时自己去读。
"""

from __future__ import annotations

import uuid
from pathlib import Path

#: 默认字符上限。工具可以通过 ``Tool.max_output_chars`` 覆盖。
DEFAULT_MAX_CHARS = 25_000


def truncate_output(
    text: str,
    *,
    max_chars: int | None = DEFAULT_MAX_CHARS,
    keep: str = "both",
    overflow_dir: Path | None = None,
    label: str = "output",
) -> str:
    """按上限截断文本;超限时把全文写盘并在返回内容里指明路径。

    :param keep: ``head`` 只留开头、``tail`` 只留结尾、``both`` 两头都留。
        ``both`` 是大多数情况的正解 —— 命令回显在开头,报错在结尾,被砍掉的中间
        段通常是重复的进度输出。
    :param overflow_dir: 全文落盘目录。为 ``None`` 时只截断不落盘。
    """

    if max_chars is None or len(text) <= max_chars:
        return text

    saved_hint = ""
    if overflow_dir is not None:
        path = _save_overflow(text, overflow_dir, label)
        saved_hint = f",完整内容已写入 {path}(可用 read_file 查看)"

    omitted = len(text) - max_chars
    notice = f"\n\n... [已省略 {omitted} 个字符{saved_hint}] ...\n\n"

    if keep == "head":
        return text[:max_chars] + notice
    if keep == "tail":
        return notice + text[-max_chars:]

    half = max_chars // 2
    return text[:half] + notice + text[-half:]


def _save_overflow(text: str, overflow_dir: Path, label: str) -> Path:
    overflow_dir.mkdir(parents=True, exist_ok=True)
    # 文件名带随机后缀:同一个工具在一次运行里可能溢出多次,按序号会撞。
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)[:40]
    path = overflow_dir / f"{safe}_{uuid.uuid4().hex[:8]}.txt"
    path.write_text(text, encoding="utf-8")
    return path


__all__ = ["DEFAULT_MAX_CHARS", "truncate_output"]
