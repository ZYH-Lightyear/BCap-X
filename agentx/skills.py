"""技能发现与解析。

技能就是一个目录加一份 ``SKILL.md``:YAML frontmatter 声明元信息,正文是给模型
看的内容。格式沿用 Anthropic Agent Skills,这样第三方技能库不用改就能加载。

## 两段式披露

技能正文往往几百上千 token,全部常驻会把上下文吃光,而一次任务通常只用得上其中
一两个。所以分两段:启动时只把 name 和 description 放进 system prompt(见
:func:`render_index`),模型判断某个技能有用时再调 ``skill`` 工具取回正文。

这一层对小模型尤其值钱 —— 它们的上下文更短,也更容易被无关内容带偏。

## 为什么只强制 name 和 description

frontmatter 里除这两个字段外的键一律原样收进 :attr:`SkillMeta.extra`,不做解释。
不同技能库各有自己的扩展命名空间(例如 open-robot-skills 的 ``gap:`` 那一套),
挨个支持既做不完也没必要:loader 只需要「够用来索引和加载」,语义留给技能正文和
将来真正消费它的那一层。这条约束换来的是加载任意技能库都不用改解析代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

#: 技能目录里的正文文件名。
SKILL_FILENAME = "SKILL.md"

#: frontmatter 分隔线。
_FENCE = "---"


class SkillError(ValueError):
    """技能文件格式错误。

    发现阶段不吞这个异常:一份写坏的 SKILL.md 是要修的 bug,静默跳过只会让模型
    在运行时因为「技能怎么不见了」而困惑,排查成本远高于启动即报错。
    """


@dataclass(frozen=True)
class SkillMeta:
    """一个已解析的技能。"""

    #: 技能名。模型调 ``skill`` 工具时用的标识符。
    name: str
    #: 一句话说明用途与适用场景。这是模型在第一段里唯一能看到的信息,
    #: 决定了它会不会去加载正文。
    description: str
    #: 技能目录(``SKILL.md`` 所在目录)。附带资源相对它定位。
    path: Path
    #: 去掉 frontmatter 之后的正文。
    body: str
    #: frontmatter 里除 name / description 外的所有键,原样保留。
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def skill_file(self) -> Path:
        return self.path / SKILL_FILENAME


def parse_skill_md(path: Path) -> SkillMeta:
    """解析一份 ``SKILL.md``。

    :param path: ``SKILL.md`` 文件路径。
    :raises SkillError: 缺 frontmatter、YAML 语法错误、或缺 name / description。
    """

    text = path.read_text(encoding="utf-8")
    front, body = _split_frontmatter(text, path)

    try:
        loaded = yaml.safe_load(front) if front.strip() else None
    except yaml.YAMLError as exc:
        raise SkillError(f"{path}: frontmatter YAML 解析失败: {exc}") from exc

    if not isinstance(loaded, dict):
        raise SkillError(f"{path}: frontmatter 必须是一个 YAML 映射")

    data = dict(loaded)
    name = _required_str(data.pop("name", None), "name", path)
    description = _required_str(data.pop("description", None), "description", path)

    return SkillMeta(
        name=name,
        description=_collapse(description),
        path=path.parent,
        body=body.strip(),
        extra=data,
    )


def discover_skills(roots: Iterable[Path | str]) -> list[SkillMeta]:
    """扫描若干根目录,返回按名字排序的技能列表。

    每个根目录既可以是「技能集目录」(下面每个子目录一个技能),也可以直接是单个
    技能目录(自己就有 ``SKILL.md``)。后者让 ``--skills path/to/one-skill``
    这种只挂一个技能的用法能直接生效,不必先造一层父目录。

    同名技能按 ``roots`` 的先后顺序取第一个。调用方把优先级高的根放在前面即可,
    这跟 Qwen-Code 的 project > user > bundled 是同一个约定。
    """

    found: dict[str, SkillMeta] = {}
    for root in roots:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        for skill_file in _skill_files(base):
            meta = parse_skill_md(skill_file)
            # 先到先得:后面的根覆盖不了前面的。
            found.setdefault(meta.name, meta)
    return sorted(found.values(), key=lambda m: m.name)


def render_index(skills: list[SkillMeta]) -> str:
    """渲染注入 system prompt 的技能索引(两段式的第一段)。

    只含 name 与 description。返回空串表示没有技能 —— 调用方据此跳过注入,
    免得 prompt 里多出一个空段落。
    """

    if not skills:
        return ""

    entries = "\n".join(
        f'  <skill name="{skill.name}">{skill.description}</skill>' for skill in skills
    )
    return (
        "# 可用技能\n\n"
        "下面是可用技能的名字和用途,正文尚未加载。判断某个技能与当前任务相关时,"
        "先调用 `skill` 工具取回它的完整正文,再照着做 —— 技能正文里通常有经过验证的"
        "完整做法,比自己从头摸索更快也更可靠。\n\n"
        f"<available_skills>\n{entries}\n</available_skills>"
    )


def _skill_files(base: Path) -> list[Path]:
    """找出一个根目录下的所有 ``SKILL.md``。"""

    direct = base / SKILL_FILENAME
    if direct.is_file():
        return [direct]
    return sorted(
        child / SKILL_FILENAME
        for child in base.iterdir()
        if child.is_dir() and (child / SKILL_FILENAME).is_file()
    )


def _split_frontmatter(text: str, path: Path) -> tuple[str, str]:
    """切出 frontmatter 与正文。"""

    lines = text.splitlines()
    if not lines or lines[0].strip() != _FENCE:
        raise SkillError(f"{path}: 缺少 YAML frontmatter(文件必须以 --- 开头)")

    for index in range(1, len(lines)):
        if lines[index].strip() == _FENCE:
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])

    raise SkillError(f"{path}: frontmatter 没有闭合的 ---")


def _required_str(value: Any, key: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SkillError(f"{path}: frontmatter 缺少非空的 {key!r} 字段")
    return value.strip()


def _collapse(text: str) -> str:
    """把多行 description 压成一行。

    YAML 的折叠标量(``>``)会留下换行,直接塞进索引会把 XML 撑散。
    """
    return " ".join(text.split())


__all__ = [
    "SKILL_FILENAME",
    "SkillError",
    "SkillMeta",
    "discover_skills",
    "parse_skill_md",
    "render_index",
]
