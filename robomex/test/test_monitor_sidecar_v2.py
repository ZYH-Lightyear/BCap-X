from __future__ import annotations

import pytest

from robomex.authoring.monitoring import (
    MonitorCompileError,
    MonitorCompiler,
    MonitorHook,
    MonitorProgramSpec,
    MonitorRuntime,
    MonitorRuntimeError,
)
from robomex.runtime.events import FindingSeverity, MonitorFindingKind

SOURCE = """
def evaluate(sample):
    opening_delta = abs(sample["gripper_opening"] - sample["expected_opening"])
    if opening_delta > 0.01 or sample["object_visible"] == False:
        return {
            "finding": "attachment_anomaly",
            "severity": "critical",
            "confidence": 0.95,
            "details": {"opening_delta": opening_delta},
        }
    return None
"""


def _program(*, debounce_count: int = 1):
    return MonitorCompiler().compile(
        MonitorProgramSpec(
            monitor_id="attachment_guard",
            source=SOURCE,
            hook=MonitorHook.WAYPOINT,
            allowed_signals=("gripper_opening", "expected_opening", "object_visible"),
            debounce_count=debounce_count,
            max_runtime_ms=50,
        )
    )


def _runtime(*, debounce_count: int = 1) -> MonitorRuntime:
    return MonitorRuntime(
        episode_id="ep",
        workflow_id="wf",
        action_id="act",
        plan_digest="digest",
        program=_program(debounce_count=debounce_count),
    )


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef evaluate(sample):\n return None",
        "def evaluate(sample):\n open('/tmp/x')\n return None",
        "def evaluate(sample):\n while True: pass",
        "def evaluate(sample):\n return sample.get('x')",
        "def evaluate(sample):\n return [x for x in sample]",
        "def evaluate(sample):\n return __builtins__",
    ],
)
def test_compiler_rejects_io_dynamic_access_and_unbounded_code(source: str) -> None:
    with pytest.raises(MonitorCompileError):
        MonitorCompiler().compile(
            MonitorProgramSpec(
                monitor_id="bad",
                source=source,
                allowed_signals=("x",),
            )
        )


def test_monitor_is_sealed_to_action_plan_hook_and_fresh_sequence() -> None:
    runtime = _runtime()
    sample = {
        "gripper_opening": 0.03,
        "expected_opening": 0.03,
        "object_visible": True,
    }
    runtime.evaluate(
        sample,
        sequence=1,
        hook=MonitorHook.WAYPOINT,
        action_id="act",
        plan_digest="digest",
    )
    with pytest.raises(MonitorRuntimeError, match="sealed action"):
        runtime.evaluate(
            sample,
            sequence=2,
            hook=MonitorHook.WAYPOINT,
            action_id="old",
            plan_digest="digest",
        )
    with pytest.raises(MonitorRuntimeError, match="strictly increasing"):
        runtime.evaluate(
            sample,
            sequence=1,
            hook=MonitorHook.WAYPOINT,
            action_id="act",
            plan_digest="digest",
        )


def test_critical_finding_is_debounced_and_only_requests_runtime_stop() -> None:
    runtime = _runtime(debounce_count=2)
    sample = {
        "gripper_opening": 0.06,
        "expected_opening": 0.03,
        "object_visible": True,
    }
    first = runtime.evaluate(
        sample,
        sequence=1,
        hook=MonitorHook.WAYPOINT,
        action_id="act",
        plan_digest="digest",
    )
    second = runtime.evaluate(
        sample,
        sequence=2,
        hook=MonitorHook.WAYPOINT,
        action_id="act",
        plan_digest="digest",
    )

    assert first.finding is None and not first.stop_requested
    assert second.stop_requested
    assert second.finding is not None
    assert second.finding.finding is MonitorFindingKind.ATTACHMENT_ANOMALY
    assert second.finding.severity is FindingSeverity.CRITICAL
    assert second.finding.action_id == "act"
    assert second.finding.details["plan_digest"] == "digest"


@pytest.mark.parametrize(
    "sample",
    [
        {"gripper_opening": float("nan"), "expected_opening": 0.03, "object_visible": True},
        {"gripper_opening": 0.03, "expected_opening": 0.03},
        {
            "gripper_opening": 0.03,
            "expected_opening": 0.03,
            "object_visible": True,
            "simulator_truth": "dropped",
        },
    ],
)
def test_unobservable_or_oracle_like_input_fails_closed(sample) -> None:
    result = _runtime().evaluate(
        sample,
        sequence=1,
        hook=MonitorHook.WAYPOINT,
        action_id="act",
        plan_digest="digest",
    )

    assert result.stop_requested
    assert not result.observable
    assert result.finding is not None
    assert result.finding.finding is MonitorFindingKind.UNOBSERVABLE


def test_normal_sample_emits_no_finding() -> None:
    result = _runtime().evaluate(
        {
            "gripper_opening": 0.03,
            "expected_opening": 0.03,
            "object_visible": True,
        },
        sequence=1,
        hook=MonitorHook.WAYPOINT,
        action_id="act",
        plan_digest="digest",
    )
    assert result.finding is None
    assert not result.stop_requested
    assert result.observable
