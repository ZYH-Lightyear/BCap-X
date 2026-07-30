"""M5 — Reward components (interfaces defined, implementations land with RL).

R(tau) = R_task + l1*R_progress - l2*P_viol + l3*R_state - l4*C + R_format
(see docs/gui_as_policy_v2_cvpr_plan.md §3.3)

All components consume the TraceLogger format (steps.jsonl records), so the
same functions score teacher traces (for filtering) and RL rollouts.
"""

from __future__ import annotations

from typing import Any

TraceSteps = list[dict[str, Any]]


def r_task(steps: TraceSteps, env_success: bool) -> float:
    """Binary task reward from the environment's success detector."""
    return 1.0 if env_success else 0.0


def p_viol(steps: TraceSteps) -> float:
    """Asymmetric verification penalty: count of unpredicted failures.

    An unpredicted failure = a commit whose terminal IK preview succeeded but
    whose receipt deviated beyond tolerance (executor.py sets the flag). Deliberately
    NOT a symmetric consistency reward: rewarding predictability is hackable
    by conservative free-space motion.
    """
    count = 0
    for step in steps:
        for receipt in step.get("state", {}).get("receipts", []):
            if receipt.get("unpredicted_failure"):
                count += 1
                break  # receipts repeat across summaries; count per receipt id
    # TODO(M5): dedupe by receipt id across step summaries.
    raise NotImplementedError("M5: dedupe receipts and calibrate tolerance per task class")


def r_progress(steps: TraceSteps, instruction: str) -> list[float]:
    """Dense progress reward via TOPReward (arXiv 2602.19313).

    Plan: feed canvas-frame prefixes + instruction to a FROZEN scoring VLM
    (Qwen3-VL-32B or a separate 8B instance — never the policy being trained)
    and read log p("True" | "the trajectory completes the task"). Return the
    per-commit increments, clipped as in TOPReward eq. (3).
    """
    raise NotImplementedError("M5: TOPReward scorer over canvas prefixes")


def c_cost(steps: TraceSteps) -> float:
    """Step/tool budget cost: total ops, weighted higher for physical ops."""
    total = len(steps)
    physical = sum(1 for s in steps if s.get("physical"))
    return total + 2.0 * physical


def r_format(steps: TraceSteps) -> float:
    """Structured-output compliance: fraction of steps that parsed into a
    valid op (ok or agent-visible error, but not protocol violations)."""
    raise NotImplementedError("M5: needs raw model outputs, logged by the RL loop")
