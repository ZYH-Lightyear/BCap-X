"""M3 产物使用的最小文件与多模态消息工具。"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON object；M3 不接受顶层数组等含糊产物。"""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 必须是 object: {source}")
    return payload


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取由 object 构成的 JSONL。"""

    source = Path(path)
    records: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 必须由 object 构成: {source}")
        records.append(value)
    return records


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """原子写入 JSON，避免中断后的半份 NIGHT 产物被继续消费。"""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def relative_asset(root: str | Path, path: str | Path) -> str:
    """返回根目录内的相对路径，并阻止证据引用越出当前产物。"""

    base = Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError as exc:
        raise ValueError(f"证据不属于指定目录: {resolved}") from exc


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_part(path: str | Path, *, detail: str = "high") -> dict[str, Any]:
    """把本地证据编码为 OpenAI-compatible 图像消息。"""

    source = Path(path)
    mime = mimetypes.guess_type(source.name)[0] or "image/png"
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": detail},
    }


def parse_model_json(text: str) -> dict[str, Any]:
    """接受纯 JSON 或单个 Markdown JSON fence，不猜测残缺回复。"""

    value = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(\{.*\})\s*```",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        value = fenced.group(1)
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("模型输出必须是 JSON object")
    return payload


__all__ = [
    "image_part",
    "parse_model_json",
    "read_json",
    "read_jsonl",
    "relative_asset",
    "sha256",
    "write_json",
]
