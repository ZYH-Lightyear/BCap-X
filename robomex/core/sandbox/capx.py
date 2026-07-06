from __future__ import annotations

import time
from functools import wraps
from typing import Any

from robomex.core.context import compact_json
from robomex.core.sandbox.action_block import (
    ActionBlockStatus,
    BlockExecutionResult,
    ExecutionTraceEvent,
    SemanticActionBlock,
)


_TRACE_FUNCTION_NAMES = frozenset({
    "vlm_bbox_detection",
    "vlm_point_detection",
    "query_vlm",
    "segment_sam3_text_prompt",
    "segment_sam3_box_prompt",
    "segment_sam3_point_prompt",
    "point_prompt_molmo",
    "plan_grasp",
    "plan_grasp_from_point_clouds",
    "solve_ik",
    "goto_pose",
    "open_gripper",
    "close_gripper",
    "move_to_joints",
    "goto_home_joint_position",
    "mask_to_world_points",
    "get_oriented_bounding_box_from_3d_points",
})


class CapXExecutorAdapter:
    """通过 CapX 代码执行环境运行 RoboMEx 的语义动作块。

    适配器刻意用鸭子类型而非 import CapX 的类。一个兼容的 env 应暴露:

    - step(code: str) -> (obs, reward, terminated, truncated, info)
    - configure_line_trace(block_index, emit_callback=None),可选
    - consume_line_trace_events(),可选
    """

    def __init__(self, env: Any, *, vlm_backend: dict[str, str] | None = None) -> None:
        self.env = env
        self._block_index = 0
        self._vlm_backend = dict(vlm_backend or {})
        self._current_trace_block: int | None = None
        self._current_trace_events: list[dict[str, Any]] = []
        self._primitive_seq = 0
        self._configure_api_vlm_backend()
        self._install_primitive_trace_wrappers()

    def run_block(self, block: SemanticActionBlock) -> BlockExecutionResult:
        """在包装的 CapX 环境里执行一个语义动作块。"""

        current_index = self._block_index
        self._block_index += 1

        self._current_trace_block = current_index
        self._current_trace_events = []

        if hasattr(self.env, "configure_line_trace"):
            self.env.configure_line_trace(current_index)

        # 记录本块执行**前**的视频帧游标;执行后再取一次,得到这一块产生的帧区间。
        # env 不支持录像时保持 None。
        frame_start = self._video_frame_count()

        obs: dict[str, Any] | None = None
        reward: float | None = None
        terminated: bool | None = None
        truncated: bool | None = None
        info: dict[str, Any] = {}

        try:
            obs, raw_reward, raw_terminated, raw_truncated, raw_info = self.env.step(block.code)
            reward = float(raw_reward) if raw_reward is not None else None
            terminated = bool(raw_terminated)
            truncated = bool(raw_truncated)
            info = dict(raw_info or {})
            ok = info.get("sandbox_rc", 0) == 0
        except Exception as exc:
            ok = False
            info = {"adapter_error": repr(exc)}

        if frame_start is not None:
            frame_end = self._video_frame_count()
            if frame_end is not None and frame_end > frame_start:
                info["video_range"] = (frame_start, frame_end)

        raw_events = []
        if hasattr(self.env, "consume_line_trace_events"):
            raw_events = self.env.consume_line_trace_events()
        if hasattr(self.env, "configure_line_trace"):
            self.env.configure_line_trace(None)
        primitive_events = list(self._current_trace_events)
        self._current_trace_block = None
        self._current_trace_events = []

        trace_events = tuple(
            self._normalize_trace_event(event, block.name)
            for event in [*raw_events, *primitive_events]
        )
        stdout = str(info.get("stdout", ""))
        stderr = str(info.get("stderr", info.get("adapter_error", "")))
        status = ActionBlockStatus.SUCCEEDED if ok else ActionBlockStatus.FAILED

        return BlockExecutionResult(
            block=block,
            ok=ok,
            status=status,
            stdout=stdout,
            stderr=stderr,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            observation=obs,
            info=info,
            trace_events=trace_events,
        )

    def _video_frame_count(self) -> int | None:
        """当前视频帧缓冲长度;env 不支持录像则返回 None。"""

        getter = getattr(self.env, "get_video_frame_count", None)
        if getter is None:
            return None
        try:
            return int(getter())
        except Exception:  # noqa: BLE001 - 录像探测失败不该影响动作执行
            return None

    def _configure_api_vlm_backend(self) -> None:
        """Apply per-run VLM config to CapX API instances that support it."""

        if not self._vlm_backend:
            return
        apis = getattr(self.env, "_apis", None)
        if not isinstance(apis, dict):
            return
        for api in apis.values():
            configure = getattr(api, "configure_vlm_backend", None)
            if callable(configure):
                configure(**self._vlm_backend)

    def _install_primitive_trace_wrappers(self) -> None:
        """Wrap selected CapX API functions so every code block emits primitive traces.

        CapX line tracing records Python source lines, but it does not know which
        exposed API call was made or what came back. RoboMEx uses these lightweight
        wrappers to produce stable ``primitive_traces`` without relying on the model
        to self-report them in ``finish``.
        """

        apis = getattr(self.env, "_apis", None)
        if not isinstance(apis, dict):
            return
        for api_name, api in apis.items():
            if getattr(api, "_robomex_primitive_trace_wrapped", False):
                continue
            functions = getattr(api, "functions", None)
            if not callable(functions):
                continue
            original_functions = functions
            adapter = self

            @wraps(original_functions)
            def wrapped_functions(
                *args: Any,
                _api_name: str = str(api_name),
                _original_functions: Any = original_functions,
                **kwargs: Any,
            ) -> dict[str, Any]:
                fns = dict(_original_functions(*args, **kwargs))
                return {
                    fn_name: adapter._wrap_primitive_function(
                        fn_name,
                        fn,
                        api_name=_api_name,
                    )
                    for fn_name, fn in fns.items()
                }

            try:
                setattr(api, "functions", wrapped_functions)
                setattr(api, "_robomex_primitive_trace_wrapped", True)
            except Exception:
                continue

    def _wrap_primitive_function(self, fn_name: str, fn: Any, *, api_name: str) -> Any:
        if not callable(fn) or fn_name not in _TRACE_FUNCTION_NAMES:
            return fn
        if getattr(fn, "_robomex_primitive_trace_wrapper", False):
            return fn
        adapter = self

        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return adapter._call_traced_primitive(fn_name, fn, args, kwargs, api_name=api_name)

        setattr(wrapped, "_robomex_primitive_trace_wrapper", True)
        return wrapped

    def _call_traced_primitive(
        self,
        fn_name: str,
        fn: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        api_name: str,
    ) -> Any:
        start = time.time()
        block_index = self._current_trace_block
        self._primitive_seq += 1
        trace_id = f"capx:block{block_index if block_index is not None else 'na'}:{self._primitive_seq}:{fn_name}"
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            duration_s = time.time() - start
            self._record_primitive_trace({
                "trace_id": trace_id,
                "primitive_name": fn_name,
                "status": "failed",
                "producer": "capx_api",
                "inputs_summary": _summarize_call_inputs(args, kwargs),
                "outputs_summary": {"duration_s": round(duration_s, 4)},
                "error": repr(exc),
                "metadata": {"api_name": api_name},
            })
            raise
        duration_s = time.time() - start
        self._record_primitive_trace({
            "trace_id": trace_id,
            "primitive_name": fn_name,
            "status": "succeeded",
            "producer": "capx_api",
            "inputs_summary": _summarize_call_inputs(args, kwargs),
            "outputs_summary": {
                "result": compact_json(result, max_depth=3, max_items=8, max_string=220),
                "duration_s": round(duration_s, 4),
            },
            "metadata": {"api_name": api_name},
        })
        return result

    def _record_primitive_trace(self, trace: dict[str, Any]) -> None:
        if self._current_trace_block is None:
            return
        self._current_trace_events.append({
            "event_type": "primitive_trace",
            "message": f"{trace.get('primitive_name')} {trace.get('status')}",
            "block_index": self._current_trace_block,
            "primitive_traces": [trace],
        })

    @staticmethod
    def _normalize_trace_event(event: Any, block_name: str) -> ExecutionTraceEvent:
        if isinstance(event, dict):
            return ExecutionTraceEvent(
                event_type=str(event.get("event_type", event.get("type", "trace"))),
                message=str(event.get("message", "")),
                block_name=str(event.get("block_name", block_name)),
                line_no=event.get("line_no"),
                payload=dict(event),
            )
        return ExecutionTraceEvent(
            event_type="trace",
            message=str(event),
            block_name=block_name,
        )


def _summarize_call_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if args:
        out["args"] = compact_json(list(args), max_depth=3, max_items=6, max_string=180)
    if kwargs:
        out["kwargs"] = compact_json(kwargs, max_depth=3, max_items=8, max_string=180)
    return out
