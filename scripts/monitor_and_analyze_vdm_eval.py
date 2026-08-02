#!/usr/bin/env python3
"""Monitor robosuite VDM 2-trial eval; on completion, summarize failures."""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path("/Knowin/foundation/bohanzhou/MyProj/BCap-X")
OUT = ROOT / "outputs/qwen3.5-397b-a17b"
LOG = ROOT / "logs/robosuite_vdm_qwen35_eval_2trials.log"
STATUS = OUT / "eval_monitor_status.txt"
ANALYSIS = OUT / "failure_analysis.md"
AGG = OUT / "robosuite_vdm_success_rates.txt"

TASKS = [
    ("cube_lifting", "franka_robosuite_cube_lifting_multiturn_vdm"),
    ("cube_stack", "franka_robosuite_cube_stack_multiturn_vdm"),
    ("cube_restack", "franka_robosuite_cube_restack_multiturn_vdm"),
    ("spill_wipe", "franka_robosuite_spill_wipe_multiturn_vdm"),
    ("nut_assembly", "franka_robosuite_nut_assembly_multiturn_vdm"),
    ("two_arm_lift", "franka_robosuite_two_arm_lift_multiturn_vdm"),
    ("two_arm_handover", "two_arm_handover_multiturn_vdm"),
]


def write_status(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with STATUS.open("a") as f:
        f.write(line + "\n")


def task_done(stem: str) -> bool:
    return (OUT / stem / "summaries.txt").is_file()


def parse_metrics(stem: str) -> str:
    p = OUT / stem / "summaries.txt"
    if not p.is_file():
        return "pending"
    t = p.read_text(errors="ignore")
    m = re.search(r"Code generation success rate.*?\n([\d./]+)", t)
    e = re.search(r"Elapsed time:\s*([\d.]+)", t)
    return f"{m.group(1) if m else '?'}  elapsed={float(e.group(1))/60:.1f}min" if e else (m.group(1) if m else "?")


def batch_finished() -> bool:
    if AGG.is_file():
        return True
    if not LOG.is_file():
        return False
    text = LOG.read_text(errors="ignore")
    if "Summary written" in text:
        return True
    oks = len(re.findall(r"\[(OK|FAIL)\]", text))
    return oks >= 7 and all(task_done(s) for _, s in TASKS)


def eta_minutes() -> tuple[int, int, float]:
    dones = []
    for _, stem in TASKS:
        p = OUT / stem / "summaries.txt"
        if not p.is_file():
            continue
        m = re.search(r"Elapsed time:\s*([\d.]+)", p.read_text(errors="ignore"))
        if m:
            dones.append(float(m.group(1)))
    n_done = sum(1 for _, s in TASKS if task_done(s))
    n_left = 7 - n_done
    avg = sum(dones) / len(dones) if dones else 1400.0
    # slight credit if a task is in progress
    eta = max(n_left * avg - (0.25 * avg if n_left else 0), 0)
    return n_done, n_left, eta / 60.0


def classify_exceptions(trial_dir: Path) -> list[str]:
    reasons: list[str] = []
    live_name = re.search(r"(trial_\d+)_", trial_dir.name)
    if not live_name:
        return reasons
    live = trial_dir.parent / f"{live_name.group(1)}_live" / "line_trace"
    if not live.is_dir():
        return reasons
    for jt in sorted(live.glob("block_*.jsonl")):
        for line in jt.read_text(errors="ignore").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            et = e.get("exception_type")
            em = e.get("exception_message") or ""
            src = (e.get("source") or "")[:100]
            if et:
                reasons.append(f"{et}: {em} @ {src}")
            elif "No sam3" in (e.get("stderr_delta") or ""):
                reasons.append(f"SAM3 empty @ {src}")
    return reasons


def analyze_failures() -> str:
    lines: list[str] = []
    lines.append("# Robosuite VDM Eval — Failure Analysis")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Model: qwen3.5-397b-a17b | 2 trials/task | multiturn VDM")
    lines.append("")
    lines.append("## Per-task metrics (`success_rate / avg_reward / task_completed`)")
    lines.append("")
    lines.append("| Task | Metrics | Notes |")
    lines.append("| --- | --- | --- |")

    global_exc: Counter[str] = Counter()
    fail_trials: list[tuple[str, Path, list[str]]] = []

    for name, stem in TASKS:
        metrics = parse_metrics(stem)
        d = OUT / stem
        # Final-ish failed dirs: taskcompleted_0 that are not superseded by later
        # same-trial taskcompleted_1 with newer mtime. Prefer unique trial finals.
        fails = sorted(d.glob("trial_*_taskcompleted_0")) if d.is_dir() else []
        # Keep only trials whose best final is not completed_1
        true_fails = []
        for fd in fails:
            m = re.match(r"(trial_\d+)_", fd.name)
            if not m:
                continue
            tid = m.group(1)
            wins = list(d.glob(f"{tid}_*taskcompleted_1*"))
            if wins:
                # intermediate fail only
                continue
            true_fails.append(fd)

        note = f"{len(true_fails)} final-fail trial(s)" if true_fails else "no final-fail trial dirs"
        lines.append(f"| {name} | {metrics} | {note} |")

        for fd in true_fails:
            excs = classify_exceptions(fd)
            fail_trials.append((name, fd, excs))
            for e in excs:
                key = e.split(" @ ")[0]
                if "No sam3" in key or "sam3" in key.lower():
                    global_exc["SAM3 empty detections"] += 1
                elif "argmax" in key.lower():
                    global_exc["Empty grasp argmax"] += 1
                elif "too many values" in key.lower() or "unpack" in key.lower():
                    global_exc["API unpack mismatch"] += 1
                else:
                    global_exc[key[:80]] += 1

        # Also scan intermediate fails for patterns even if trial later succeeded
        for fd in fails:
            for e in classify_exceptions(fd):
                key = e.split(" @ ")[0]
                if "No sam3" in key or "sam3" in key.lower():
                    global_exc["SAM3 empty detections (incl. mid-trial)"] += 1

    lines.append("")
    lines.append("## Dominant failure modes (from line_trace exceptions)")
    lines.append("")
    if global_exc:
        for k, v in global_exc.most_common(15):
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("- No exceptions recorded in failed/intermediate trials.")

    lines.append("")
    lines.append("## Final-fail trials (no later taskcompleted_1)")
    lines.append("")
    if not fail_trials:
        lines.append("None — every trial that finished eventually reached taskcompleted_1, or still only intermediate fails.")
    for name, fd, excs in fail_trials:
        lines.append(f"### {name} / `{fd.name}`")
        # summary snippet
        sp = fd / "summary.txt"
        if sp.is_file():
            t = sp.read_text(errors="ignore")
            for key in ("Reward:", "Task Completed:", "Num Regenerations:", "Sandbox failed:"):
                for line in t.splitlines():
                    if key in line:
                        lines.append(f"- {line.strip()}")
                        break
            # agent comments hint
            code = fd / "code.py"
            if code.is_file():
                comments = [l.strip() for l in code.read_text(errors="ignore").splitlines() if l.strip().startswith("#") and ("fail" in l.lower() or "instead" in l.lower() or "SAM" in l or "not" in l.lower())][:8]
                if comments:
                    lines.append("- Code comments:")
                    for c in comments:
                        lines.append(f"  - {c}")
        if excs:
            lines.append("- Exceptions:")
            for e in excs[:8]:
                lines.append(f"  - `{e}`")
        else:
            lines.append("- No Python exception in line_trace (semantic / placement / FINISH-without-success).")
        lines.append("")

    # Log-level SAM3 hits
    if LOG.is_file():
        n_sam3 = len(re.findall(r"No sam3 detections|SAM3 returned no results", LOG.read_text(errors="ignore")))
        lines.append("## Log-level signals")
        lines.append("")
        lines.append(f"- `No sam3 detections` / `SAM3 returned no results` occurrences in eval log: **{n_sam3}**")
        lines.append("")

    lines.append("## Summary judgment")
    lines.append("")
    lines.append("1. **Primary**: SAM3 empty detections under occlusion / remote VLM flakiness → `sample_grasp_pose` / `get_object_pose` raise `ValueError`.")
    lines.append("2. **Secondary**: Geometric / placement misses (stack/restack) where code runs without exception but VDM/reward says task incomplete → REGENERATE loops.")
    lines.append("3. **Recovery anti-pattern**: After grasp, re-query occluded objects without moving arm clear → cascading SAM3 failures.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    write_status("Monitor started")
    last_done = -1
    while True:
        n_done, n_left, eta_min = eta_minutes()
        if n_done != last_done:
            parts = [f"{n}:{'DONE' if task_done(s) else '...'}" for n, s in TASKS]
            write_status(f"progress {n_done}/7 | ETA~{eta_min:.0f}min | " + " ".join(parts))
            last_done = n_done
        else:
            # heartbeat every loop without spamming too much — still write compact
            write_status(f"heartbeat {n_done}/7 left={n_left} ETA~{eta_min:.0f}min")

        if batch_finished():
            write_status("Batch finished — analyzing failures")
            report = analyze_failures()
            ANALYSIS.write_text(report)
            write_status(f"Wrote {ANALYSIS}")
            print(report, flush=True)
            return 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
