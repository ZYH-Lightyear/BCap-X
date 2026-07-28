#!/usr/bin/env python3
"""让 agentx 的通用 Coding Agent 独立完成一个 CapX 评估任务。

和 CapX 原生评估路径的关系:任务、API 集合、prompt、奖励判定**全部来自同一份
``env_configs/*.yaml``**,这里不另造一套。区别只在于「谁来写代码」——原生路径是
单轮/多轮的裸补全,这里换成一个带工具、能看图、能自己验证的 agent 循环。

所以任何 CapX 配置都能直接跑,换 ``--config`` 即可,不需要改代码。

## 两个工具

``run_python``
    直接调 ``CodeExecutionEnvBase.step(code)`` —— 就是 CapX 评估执行 LLM 代码用的
    那个入口。持久命名空间、API 注入、reward、视频录制都由它负责,和原生评估逐字
    节一致。不自己拼一套的理由就是不能有第二套语义。

``check_success``
    读环境奖励。让 agent 用权威判据确认完成,而不是自己宣布做完了。

## 视觉不是工具

每次代码执行完,:class:`SceneObserver` 自动把当前画面(多路相机横向拼成一张)追加
进对话。模型无法选择「不看」,因此闭环是结构性质而不是它的自觉;省下来的轮次全部
留给动作。上下文只保留最近两张 —— 旧画面显示的世界已经不存在,留着只会冒充现在。

用法::

    python scripts/agentx_capx_eval.py --config env_configs/libero/franka_libero_cap_agent0.yaml
    python scripts/agentx_capx_eval.py --suite libero_10 --task-id 0   # 两罐入篮
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# LIBERO 必须在 mujoco 导入前定好渲染后端,否则无显示环境下直接崩。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agentx.agent import CodingAgent  # noqa: E402
from agentx.contracts import ToolKind, ToolResult, ToolValidationError  # noqa: E402
from agentx.core import RunConfig, TurnSummary  # noqa: E402
from agentx.providers import OpenAIProvider  # noqa: E402
from agentx.skills import discover_skills  # noqa: E402
from agentx.tools import SkillTool, ToolRegistry, default_registry  # noqa: E402
from agentx.tools.base import Invocation, Tool, ToolContext  # noqa: E402
from agentx.trace import RunTrace  # noqa: E402

DEFAULT_CONFIG = "env_configs/libero/franka_libero_cap_agent0.yaml"
DEFAULT_SKILLS = "robomex/skills"
MAX_PYTHON_OUTPUT_CHARS = 20_000

#: ``--hide-api-docs`` 不带参数时隐藏的这一组:grounding-objects 技能封装的那条
#: 感知链。留下 query_vlm —— 它做的是视觉问答而不是空间定位,技能正文本身也让模型
#: 在非定位问题上直接用它。
PERCEPTION_API_DOCS = frozenset({
    "vlm_bbox_detection",
    "vlm_point_detection",
    "segment_sam3_box_prompt",
    "segment_sam3_point_prompt",
})


# ---------------------------------------------------------------------------
#  执行代码:走 CapX 自己的 step()
# ---------------------------------------------------------------------------


class RunPythonTool(Tool):
    name = "run_python"
    description = (
        "Execute Python code against the robot. The session is PERSISTENT: variables "
        "and imports survive across calls, and the simulator keeps its state, so you "
        "can work incrementally. All robot API functions listed in the task are "
        "already in scope -- call them directly, no imports needed. Print anything "
        "you want to inspect."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to execute."},
        },
        "required": ["code"],
        "additionalProperties": False,
    }
    kind = ToolKind.EXECUTE
    max_output_chars = MAX_PYTHON_OUTPUT_CHARS
    truncate_keep = "both"

    def __init__(self, code_env: Any) -> None:
        self.code_env = code_env

    def create_invocation(self, params: dict[str, Any]) -> Invocation:
        if not str(params.get("code", "")).strip():
            raise ToolValidationError("code 不能为空")
        return _RunPythonInvocation(params, self.code_env)


class _RunPythonInvocation(Invocation):
    def __init__(self, params: dict[str, Any], code_env: Any) -> None:
        super().__init__(params)
        self.code_env = code_env

    def describe(self) -> str:
        return self.params["code"].strip().splitlines()[0][:80]

    def execute(self, ctx: ToolContext) -> ToolResult:
        # CapX 的 step() 内部用 Tee 把 stdout/stderr 同时喷到真实终端,会和
        # TurnReporter 的输出重复刷屏。这里接到黑洞上,只保留 info 里的那份。
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                _, reward, _terminated, truncated, info = self.code_env.step(
                    self.params["code"]
                )
        except Exception as exc:  # noqa: BLE001 - step 自己也可能抛
            message = f"执行环境抛出异常: {type(exc).__name__}: {exc}"
            return ToolResult(llm_content=message, display=message, error=str(exc))

        stdout = info.get("stdout", "")
        stderr = info.get("stderr", "")
        failed = bool(info.get("sandbox_rc"))

        sections = []
        if stdout.strip():
            sections.append(stdout.rstrip())
        if stderr.strip():
            sections.append(f"--- stderr ---\n{stderr.rstrip()}")
        sections.append(f"[reward = {reward}]")
        if truncated:
            sections.append("[警告] 仿真步数已耗尽,环境被截断。")
        body = "\n".join(sections) if sections else "(代码执行成功,没有任何输出)"

        error = _exception_summary(stderr) if failed else None
        return ToolResult(llm_content=body, display=body, error=error)


# ---------------------------------------------------------------------------
#  看画面:每轮无条件注入,不做成工具
# ---------------------------------------------------------------------------

#: 拼图上每一路相机的标签条高度(像素)。
_LABEL_H = 14


class SceneObserver:
    """每轮把当前画面拼成一张图追加进 history。

    做成 ``RunConfig.observe`` 钩子而不是工具,有两个理由。

    一是预算:工具化的 ``look`` 和动作抢同一份 turn 配额。规程里「动作前看一眼、
    动作后再看一眼」意味着一个动作要花三轮,四十轮只剩十三个真动作,于是模型要么
    老实看图然后没预算干活,要么省着看然后开环瞎猜。观测本该是免费的。

    二是纪律:只要「看」是一个可选调用,就存在一条模型能跳过观测的路径。拿掉它之后
    闭环是结构性质,不再依赖模型听不听劝 —— 实测里它七次取图全用默认的第三人称相机,
    从没试过腕部,而抓取近距离时第三人称恰好被机械臂自己挡住。
    """

    def __init__(self, low_level_env: Any, out_dir: Path, cameras: list[str]) -> None:
        self.env = low_level_env
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cameras = cameras
        self._seq = 0

    def __call__(self, turn: int, records: list[Any]) -> dict[str, Any] | None:
        # 只有 run_python 会动到仿真状态。读文件、加载技能这类轮次画面逐像素不变,
        # 再注入一张只是白花 token。开局那次(turn 0)必须给。
        if turn > 0 and not any(r.call.name == "run_python" for r in records):
            return None

        try:
            obs = self.env.get_observation()
            panels = [(name, obs[name]["images"]["rgb"]) for name in self.cameras]
            canvas = _stitch(panels)
        except Exception as exc:  # noqa: BLE001 - 取图失败不该终止整个 rollout
            return {
                "role": "user",
                "content": f"[观测 T={turn}] 取图失败: {type(exc).__name__}: {exc}",
            }

        self._seq += 1
        path = self.out_dir / f"obs_{self._seq:03d}.png"
        canvas.save(path)

        layout = " | ".join(self.cameras)
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": f"[观测 T={turn}] 当前画面(自左至右:{layout})"},
                {"type": "image_url", "image_url": {"url": _data_url(path)}},
            ],
        }


def _stitch(panels: list[tuple[str, Any]]) -> Any:
    """把多路相机横向拼成一张,每路上方压一条标签。

    标签既画在图上也写在消息文字里:默认位图字体在 LIBERO 的低分辨率下未必看得清,
    文字那份是兜底。两份都给,模型总能对上哪一半是哪个相机。
    """
    from PIL import Image, ImageDraw

    images = [(name, _to_pil(rgb)) for name, rgb in panels]
    width = sum(im.width for _, im in images) + max(len(images) - 1, 0)
    height = max(im.height for _, im in images) + _LABEL_H

    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for name, im in images:
        draw.text((x + 2, 2), name, fill=(235, 235, 235))
        canvas.paste(im, (x, _LABEL_H))
        x += im.width + 1
    return canvas


def _to_pil(rgb: Any) -> Any:
    from PIL import Image

    array = rgb
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return Image.fromarray(array.astype("uint8")).convert("RGB")


def _data_url(path: Path) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


# ---------------------------------------------------------------------------
#  成功判定
# ---------------------------------------------------------------------------


class CheckSuccessTool(Tool):
    name = "check_success"
    description = (
        "Ask the environment whether the task goal is currently satisfied. Returns "
        "the reward. This is the only authoritative verdict -- use it before "
        "declaring the task done."
    )
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    kind = ToolKind.READ

    def __init__(self, code_env: Any) -> None:
        self.code_env = code_env

    def create_invocation(self, params: dict[str, Any]) -> Invocation:
        return _CheckSuccessInvocation(params, self.code_env)


class _CheckSuccessInvocation(Invocation):
    def __init__(self, params: dict[str, Any], code_env: Any) -> None:
        super().__init__(params)
        self.code_env = code_env

    def describe(self) -> str:
        return "check_success"

    def execute(self, ctx: ToolContext) -> ToolResult:
        try:
            reward = float(self.code_env.compute_reward())
        except Exception as exc:  # noqa: BLE001
            message = f"读取奖励失败: {type(exc).__name__}: {exc}"
            return ToolResult(llm_content=message, error=str(exc))
        body = f"reward = {reward}\n" + ("任务已完成" if reward >= 1.0 else "任务尚未完成")
        return ToolResult(llm_content=body, display=body)


# ---------------------------------------------------------------------------
#  实时汇报
# ---------------------------------------------------------------------------


class TurnReporter:
    """把每一轮发生的事打到终端,并把执行过的代码单独存盘。

    只打一行 ``run_python (312ms)`` 是不够看的 —— 这个 agent 干的活几乎全在那段
    代码里。所以源码原样打出来,并且每段另存成可以直接复跑的 ``.py``。
    """

    HEAD_LINES = 12
    TAIL_LINES = 8

    def __init__(self, snippet_dir: Path) -> None:
        self.snippet_dir = snippet_dir
        self.snippet_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def __call__(self, index: int, summary: TurnSummary) -> None:
        if summary.text.strip():
            self._emit(f"[{index}] 模型: {_clip(summary.text.strip(), 400)}")

        for record in summary.tool_records:
            mark = "✗" if record.result.is_error else "✓"
            name = record.call.name
            head = f"[{index}] {mark} {name}  ({record.duration_ms:.0f}ms)"

            if name != "run_python":
                detail = record.result.error or _first_line(record.result.display)
                self._emit(f"{head}  {_clip(detail, 120)}" if detail else head)
                continue

            code = str(record.call.args.get("code", ""))
            path = self._save(index, code, record)
            self._emit(f"{head}  -> {path.name}")
            self._emit_block(code, prefix="  │ ")
            self._emit_block(self._condense(_result_text(record)), prefix="  ┊ ", label="输出")

    def _save(self, turn: int, code: str, record: Any) -> Path:
        self._seq += 1
        path = self.snippet_dir / f"{self._seq:03d}_turn{turn:02d}.py"
        status = "ERROR" if record.result.is_error else "ok"
        path.write_text(
            f"# turn {turn} | {status} | {record.duration_ms:.0f}ms\n"
            f"{code.rstrip()}\n\n"
            f'"""执行结果\n{_result_text(record).rstrip()}\n"""\n',
            encoding="utf-8",
        )
        return path

    def _condense(self, text: str) -> str:
        """长输出只留头尾 —— 中间刷几百行数组时谁也看不出问题在哪。"""
        lines = text.rstrip().splitlines()
        if len(lines) <= self.HEAD_LINES + self.TAIL_LINES + 1:
            return "\n".join(lines)
        omitted = len(lines) - self.HEAD_LINES - self.TAIL_LINES
        return "\n".join(
            lines[: self.HEAD_LINES] + [f"... 省略 {omitted} 行 ..."] + lines[-self.TAIL_LINES :]
        )

    def _emit(self, line: str) -> None:
        print(line, file=sys.stderr, flush=True)

    def _emit_block(self, text: str, *, prefix: str, label: str | None = None) -> None:
        if not text.strip():
            return
        if label:
            self._emit(f"  ┊ --- {label} ---")
        for line in text.rstrip().splitlines():
            self._emit(f"{prefix}{line}")


def _result_text(record: Any) -> str:
    content = record.result.display or record.result.llm_content
    return content if isinstance(content, str) else str(content)


def _first_line(text: str | None) -> str:
    stripped = (text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _exception_summary(stderr: str) -> str:
    """从 traceback 里取那句真正有信息量的话。

    首行永远是 ``Traceback (most recent call last):``,拿它当摘要等于没说。异常
    类型和消息在最后一行。
    """
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    return lines[-1].strip() if lines else "代码执行失败"


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + " …"


# ---------------------------------------------------------------------------
#  环境与 prompt:全部取自 CapX 配置
# ---------------------------------------------------------------------------


def load_config(path: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """读 CapX 配置并套用点号路径覆盖。

    覆盖在实例化**之前**打进 dict —— 环境建好之后再改 suite/task_id 是没用的。
    """
    from capx.envs.configs.loader import DictLoader

    cfg = DictLoader.load(path)
    for dotted, value in overrides.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            if key not in node:
                raise KeyError(f"配置里没有路径 {dotted!r}(卡在 {key!r})")
            node = node[key]
        node[leaf] = value
    return cfg


def build_code_env(cfg: dict[str, Any]) -> Any:
    import capx.integrations  # noqa: F401 - 导入包才会触发 API 注册
    from capx.envs.configs.instantiate import instantiate

    return instantiate(cfg["env"])


def resolve_task_text(code_env: Any, obs: dict[str, Any]) -> str:
    """取出 CapX 喂给模型的那段任务文本(任务描述 + 全部 API 文档)。

    ``{libero_environment_goal}`` 占位符要用 LIBERO 的任务语言填掉,否则模型看到的
    就是占位符本身。

    这里用字面替换而不是 ``str.format`` —— 后者会把 API docstring 里的花括号(例如
    返回值示例 ``{"state": ...}``)当成格式化字段,直接 KeyError。
    ``capx.envs.trial._patch_libero_goal`` 用的是 format,在 docstring 带花括号的
    API 集合下会炸,不能照抄。
    """
    text = obs["full_prompt"][-1]["content"][0]["text"]
    handle = getattr(code_env.low_level_env, "handle", None)
    goal = getattr(handle, "task_language", None) if handle is not None else None
    if goal:
        for placeholder in ("{{libero_environment_goal}}", "{libero_environment_goal}"):
            text = text.replace(placeholder, goal)
    return text


def hide_api_docs(task_text: str, hidden: frozenset[str]) -> tuple[str, list[str]]:
    """把指定 API 的文档段落从任务文本里摘掉,返回新文本和实际摘掉的名字。

    这不是「移除 API」:函数依然注册在沙箱里、依然能调用,只是模型在文档里看不到
    它们了。这个区分是必须的 —— 技能正文里的 ``ground_object()`` 正是靠这几个函数
    实现的,把它们从 ``functions()`` 里摘掉会连技能一起废掉。

    被测的命题因此是干净的:直路从视野里消失、但依然走得通时,模型会不会去加载技能。

    别的 API 文档里不能出现被隐藏的名字,否则「隐藏」是假的。这一条靠 docstring
    本身保证,不在这里做兜底清理 —— 兜底会把问题藏起来,下次再加交叉引用就发现不了。
    """
    if not hidden:
        return task_text, []
    head, sep, body = task_text.partition("APIs:\n")
    if not sep:
        return task_text, []

    dropped: list[str] = []
    kept: list[str] = []
    keeping = True
    for line in body.splitlines(keepends=True):
        # 顶格的 ``name(`` 是一个函数文档段的开始;段落一直延续到下一个这样的行。
        match = re.match(r"([a-z_][a-z0-9_]*)\(", line)
        if match:
            name = match.group(1)
            keeping = name not in hidden
            if not keeping:
                dropped.append(name)
        if keeping:
            kept.append(line)
    return head + sep + "".join(kept), dropped


def build_task_brief(
    task_text: str, workspace: Path, cameras: list[str], max_images: int
) -> str:
    """在 CapX 原始任务文本外面套一层,说明这是个带工具的 agent 循环。

    CapX 原文里有「只输出可执行 Python、不要用代码围栏」这类针对裸补全路径的措辞,
    在工具调用语境下会造成误解,所以要明确交代真实的交互方式。原文本身一字不改 ——
    任务和 API 文档必须和原生评估完全一致。
    """
    camera_hint = "、".join(cameras) if cameras else "agentview"

    # 分段拼接而不是一整个 dedent 的 f-string:``task_text`` 里有零缩进的行,会把
    # dedent 算出的公共前缀压成空串,于是模板自己那 8 个空格原样发给模型。
    sections = [
        "下面是一个机器人操作任务。请注意:你是在一个**带工具的 agent 循环**里工作,"
        "不是一次性输出代码。任务原文里关于「只输出可执行 Python」的措辞是针对另一种"
        "调用方式的,对你不适用 —— 你要通过工具来推进。",
        "# 你的交互方式",
        "用 `run_python` 执行代码。会话是持久的:变量、import、仿真状态都跨调用保留,"
        "所以你可以分步推进。任务里列出的 API 函数已经在作用域里,直接调用即可。",
        "用 `check_success` 问环境任务有没有完成。这是唯一权威的判据,在宣布完成之前"
        "必须用它确认。",
        f"工作目录 {workspace} 可以用来放你写的脚本和中间产物。",
        "# 你会自动看到画面",
        f"每次代码执行完,你都会收到一张**执行之后**拍的画面,由这几路相机横向拼成:"
        f"{camera_hint}。你不需要、也没有办法主动请求它 —— 观测是免费的,轮次预算全部"
        "留给动作。",
        "所以那就是此刻真实的场景,不要凭想象推进,也不要在心里模拟动作的结果。"
        f"上下文里只保留最近 {max_images} 张画面,更早的会被移除:旧画面显示的世界"
        "已经不存在了。",
        "拼图上的像素坐标对任何一路相机都无效,只能用来做定性判断(东西在哪一侧、有没有"
        "抓住、掉了没有)。需要坐标一律走感知 API。",
        "# 建议做法",
        "先观察场景、确认每个相关物体的位置,再一步步执行。每一步都用真实反馈(画面、"
        "打印出来的数值、reward)确认,而不是假设它成功了。代码报错是正常的,读 "
        "traceback 改掉再来。",
        "---",
        task_text,
    ]
    return "\n\n".join(section.strip() for section in sections)


def build_registry(code_env: Any) -> ToolRegistry:
    """通用工具 + 两个机器人工具。

    通用的那几个不是摆设:agent 需要 write_file 存长脚本、read_file 回看自己写过
    什么、glob/grep 去翻 API 源码确认签名。

    没有看画面的工具 —— 观测走 :class:`SceneObserver` 钩子,每轮自动注入。
    """
    registry = default_registry()
    registry.register(RunPythonTool(code_env))
    registry.register(CheckSuccessTool(code_env))
    return registry


def save_video(code_env: Any, path: Path) -> str | None:
    try:
        frames = code_env.get_video_frames(clear=False)
        if not frames:
            return None
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=30)
    except Exception as exc:  # noqa: BLE001 - 录像失败不该影响结论
        return f"(录像失败: {exc})"
    return str(path)


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()

    overrides: dict[str, Any] = {}
    low_level = "env.cfg.low_level"
    if args.suite:
        overrides[f"{low_level}.suite_name"] = args.suite
    if args.task_id is not None:
        overrides[f"{low_level}.task_id"] = args.task_id
    if args.apis:
        overrides["env.cfg.apis"] = args.apis
    for item in args.set or []:
        key, _, raw = item.partition("=")
        overrides[key] = _coerce(raw)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / stamp
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"[setup] 配置 {args.config}", file=sys.stderr)
    for key, value in overrides.items():
        print(f"[setup]   覆盖 {key} = {value!r}", file=sys.stderr)

    cfg = load_config(args.config, overrides)
    code_env = build_code_env(cfg)

    if args.record_video:
        with contextlib.suppress(Exception):
            code_env.enable_video_capture(True, clear=True)

    obs, info = code_env.reset(seed=args.seed)
    task_text = resolve_task_text(code_env, obs)

    hidden = _resolve_hidden_apis(args.hide_api_docs)
    if hidden:
        before = len(task_text)
        task_text, dropped = hide_api_docs(task_text, hidden)
        print(
            f"[setup] 文档隐藏 {len(dropped)} 个 API(沙箱里仍可调用):"
            f"{', '.join(sorted(dropped))} —— {before} -> {len(task_text)} 字符",
            file=sys.stderr,
        )
        missing = sorted(hidden - set(dropped))
        if missing:
            print(
                f"[setup] 警告:请求隐藏但文档里没有:{', '.join(missing)}",
                file=sys.stderr,
            )
        # 隐藏之后名字还出现在别处,说明某个 docstring 里有交叉引用,实验就被污染了。
        leaked = sorted(name for name in hidden if name in task_text)
        if leaked:
            print(
                f"[setup] 警告:{', '.join(leaked)} 仍被其他 API 文档提及,"
                "隐藏没有生效,请改掉那处 docstring",
                file=sys.stderr,
            )

    (run_dir / "task_prompt.md").write_text(task_text, encoding="utf-8")

    goal = getattr(getattr(code_env.low_level_env, "handle", None), "task_language", None)
    print(f"[setup] 任务: {goal or info.get('task_prompt', '')!r}", file=sys.stderr)
    print(f"[setup] 任务文本 {len(task_text)} 字符 -> {run_dir / 'task_prompt.md'}", file=sys.stderr)

    available = [k for k, v in obs.items() if isinstance(v, dict) and "images" in v]
    cameras = args.cameras or available
    unknown = [name for name in cameras if name not in available]
    if unknown:
        print(
            f"[setup] 错误:没有相机 {', '.join(unknown)};可用的是 {', '.join(available)}",
            file=sys.stderr,
        )
        return 2
    print(f"[setup] 每轮注入相机: {', '.join(cameras)}", file=sys.stderr)

    registry = build_registry(code_env)
    observer = SceneObserver(code_env.low_level_env, run_dir / "frames", cameras)

    # 这里重复发现一次只是为了把技能名写进 meta —— 真正的注册由 CodingAgent 做。
    # 解析几份小文件的开销可以忽略,换来 meta.json 如实反映本次跑了什么。
    # 空串是「明确关掉技能」的写法,要和「没传所以用默认」区分开。
    raw_roots = [DEFAULT_SKILLS] if args.skills is None else args.skills
    skill_roots = [root for root in raw_roots if root]
    skills = discover_skills(skill_roots)
    tool_names = sorted({*registry.names(), *([SkillTool.name] if skills else [])})
    if skills:
        print(
            f"[setup] 技能 {', '.join(s.name for s in skills)}"
            f"(来自 {', '.join(str(r) for r in skill_roots)})",
            file=sys.stderr,
        )

    trace = RunTrace(
        run_dir / "trace",
        meta={
            "config": args.config,
            "overrides": overrides,
            "model": args.model,
            "protocol": args.protocol,
            "task_language": goal,
            "cameras": cameras,
            "tools": tool_names,
            "skills": [s.name for s in skills],
        },
    )
    reporter = TurnReporter(run_dir / "snippets")

    agent = CodingAgent(
        provider=OpenAIProvider(
            model=args.model, server_url=args.server_url, max_tokens=args.max_tokens
        ),
        root=workspace,
        registry=registry,
        protocol=args.protocol,
        skill_roots=skill_roots,
        # 观测图会随时间失效,必须设限,否则第 20 轮的上下文里躺着 20 张画面,
        # 其中 19 张显示的世界已经不存在了。留两张是为了能看出「动作前后的变化」。
        max_images=args.max_images,
        trace=trace,
        run_config=RunConfig(
            max_turns=args.max_turns,
            max_time_s=args.max_time_s,
            on_turn=reporter,
            observe=observer,
        ),
    )

    print(f"[run] 开始,最多 {args.max_turns} 轮\n", file=sys.stderr)
    result = agent.run(build_task_brief(task_text, workspace, cameras, args.max_images))

    try:
        reward = float(code_env.compute_reward())
    except Exception:  # noqa: BLE001
        reward = float("nan")
    success = reward >= 1.0

    video = save_video(code_env, run_dir / "rollout.mp4") if args.record_video else None

    outcome = {
        "config": args.config,
        "overrides": overrides,
        "task_language": goal,
        "model": args.model,
        "protocol": args.protocol,
        "skills": [s.name for s in skills],
        "terminate_mode": result.terminate_mode.value,
        "turns": result.turns,
        "tool_calls": len(result.tool_records),
        "reward": reward,
        "success": success,
        "usage": result.usage,
    }
    (run_dir / "outcome.json").write_text(
        json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70, file=sys.stderr)
    print(f"终止原因   : {result.terminate_mode.value}", file=sys.stderr)
    print(f"轮次/调用  : {result.turns} 轮, {len(result.tool_records)} 次工具调用", file=sys.stderr)
    print(f"tokens     : {result.usage.get('total_tokens', 0)}", file=sys.stderr)
    print(f"reward     : {reward}  ({'成功' if success else '未完成'})", file=sys.stderr)
    print(f"执行过的代码: {run_dir / 'snippets'}", file=sys.stderr)
    print(f"trace      : {run_dir / 'trace'}", file=sys.stderr)
    if video:
        print(f"录像       : {video}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(result.text or "(模型没有给出最终答复)")
    return 0 if success else 1


def _coerce(raw: str) -> Any:
    """把 ``--set`` 的字符串值还原成 JSON 类型,不行就当字符串。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _resolve_hidden_apis(raw: list[str] | None) -> frozenset[str]:
    """``--hide-api-docs`` 的三态:没传=不隐藏,带参数=按名字隐藏,不带参数=用预设。"""
    if raw is None:
        return frozenset()
    return PERCEPTION_API_DOCS if not raw else frozenset(raw)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="CapX 环境配置 YAML")
    parser.add_argument("--suite", default=None, help="覆盖 LIBERO suite,如 libero_10")
    parser.add_argument("--task-id", type=int, default=None, help="覆盖 task_id")
    parser.add_argument("--apis", nargs="+", default=None, help="覆盖暴露给代码的 API 列表")
    parser.add_argument(
        "--hide-api-docs",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "把这些 API 的文档从任务文本里摘掉;函数仍注册在沙箱里,技能正文照常能调。"
            "不带参数时用默认的感知链:" + ", ".join(sorted(PERCEPTION_API_DOCS))
        ),
    )
    parser.add_argument(
        "--set", action="append", default=None, metavar="a.b.c=值",
        help="任意配置项覆盖,值按 JSON 解析,可重复",
    )
    parser.add_argument(
        "--cameras", nargs="+", default=None, metavar="NAME",
        help="每轮注入哪几路相机(横向拼成一张)。默认用环境提供的全部",
    )
    parser.add_argument(
        "--max-images", type=int, default=2,
        help="上下文里最多保留几张观测图,更早的摘掉。默认 2(够看出动作前后的变化)",
    )
    parser.add_argument(
        "--skills", action="append", default=None, metavar="DIR",
        help=f"技能根目录,可重复。默认 {DEFAULT_SKILLS};传 --skills '' 可关掉技能",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", default="vapi/gpt-5.5")
    parser.add_argument("--server-url", default="http://localhost:8110/chat/completions")
    parser.add_argument("--protocol", choices=("native", "text"), default="native")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-time-s", type=float, default=3600.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--record-video", action="store_true", help="导出 rollout.mp4")
    parser.add_argument("--out", default="outputs/agentx_capx")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
