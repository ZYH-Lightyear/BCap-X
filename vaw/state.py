"""ActionState: the single persistent structure behind the workspace.

Everything the canvas renders, the prompt summarises, the reward inspects and
the trace logs is a view of this object. Cognitive ops mutate it freely;
physical ops append receipts; ``observe`` bumps ``obs_revision`` so stale
evidence is visibly marked instead of silently trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vaw.camera import ViewState
from vaw.types import Candidate, ObjectEntry, Pose, PreviewResult, Receipt, TOP_DOWN_QUAT_WXYZ


@dataclass
class ActionState:
    instruction: str
    obs_revision: int = 0
    objects: dict[str, ObjectEntry] = field(default_factory=dict)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    previews: dict[str, PreviewResult] = field(default_factory=dict)
    receipts: list[Receipt] = field(default_factory=list)
    selected_id: str | None = None
    virtual_gripper: Pose = field(
        default_factory=lambda: Pose(np.zeros(3), TOP_DOWN_QUAT_WXYZ.copy())
    )
    gripper_open: bool = True
    #: Agent-controlled viewpoint of the main canvas view (a ``view`` op away).
    view: ViewState = field(default_factory=ViewState)
    #: Object the focus inset zooms into, set explicitly by ``inspect``.
    focus_id: str | None = None
    #: Proprioception, refreshed with every observation. The seed of the
    #: "robot as entity 0" state (§1.5): the canvas header, the data panel and
    #: preview all read the arm's own state from here rather than re-reading obs.
    ee_pose: Pose | None = None
    #: Normalized finger opening, 0 (closed) .. 1 (fully open); see Receipt.
    gripper_opening: float | None = None
    events: list[str] = field(default_factory=list)
    _counters: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------- ids --
    def next_id(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}{self._counters[prefix]}"

    # ---------------------------------------------------------- objects --
    def add_object(self, entry: ObjectEntry) -> ObjectEntry:
        self.objects[entry.object_id] = entry
        self.log(f"grounded {entry.object_id} '{entry.name}' (score={entry.score:.2f})")
        return entry

    def get_object(self, object_id: str) -> ObjectEntry:
        if object_id not in self.objects:
            raise KeyError(f"unknown object_id '{object_id}'; known: {list(self.objects)}")
        return self.objects[object_id]

    # ------------------------------------------------------------- focus --
    def focus_target(self) -> tuple[str | None, bool]:
        """Which object the focus inset shows, and whether it was asked for.

        Falls back to the selected candidate's object, then to the most recent
        grounding, so the inset earns its pixels even before the agent ever
        calls ``inspect``. The returned flag drives the "(auto)" marker on the
        canvas: an implicit focus must not read as a decision the agent made.
        """
        if self.focus_id and self.focus_id in self.objects:
            return self.focus_id, True
        sel = self.selected
        if sel is not None and sel.object_id and sel.object_id in self.objects:
            return sel.object_id, False
        if self.objects:
            return next(reversed(self.objects)), False
        return None, False

    def candidates_of(self, object_id: str) -> list[Candidate]:
        return [c for c in self.candidates.values() if c.object_id == object_id]

    # ------------------------------------------------------- candidates --
    def add_candidate(self, cand: Candidate) -> Candidate:
        self.candidates[cand.candidate_id] = cand
        return cand

    def get_candidate(self, candidate_id: str) -> Candidate:
        if candidate_id not in self.candidates:
            raise KeyError(
                f"unknown candidate_id '{candidate_id}'; known: {list(self.candidates)}"
            )
        return self.candidates[candidate_id]

    def select(self, candidate_id: str) -> Candidate:
        cand = self.get_candidate(candidate_id)
        self.selected_id = candidate_id
        self.virtual_gripper = cand.pose.copy()
        self.log(f"selected {candidate_id}")
        return cand

    @property
    def selected(self) -> Candidate | None:
        return self.candidates.get(self.selected_id) if self.selected_id else None

    def invalidate_preview(self, candidate_id: str) -> None:
        """Editing a candidate voids its previous preview."""
        self.previews.pop(candidate_id, None)

    # ---------------------------------------------------------- previews --
    def add_preview(self, preview: PreviewResult) -> PreviewResult:
        self.previews[preview.candidate_id] = preview
        return preview

    # ---------------------------------------------------------- receipts --
    def add_receipt(self, receipt: Receipt) -> Receipt:
        self.receipts.append(receipt)
        if receipt.unpredicted_failure:
            self.log(f"UNPREDICTED FAILURE at {receipt.receipt_id} ({receipt.candidate_id})")
        return receipt

    # ------------------------------------------------------------ events --
    def log(self, message: str) -> None:
        self.events.append(message)

    def bump_revision(self) -> int:
        self.obs_revision += 1
        return self.obs_revision

    # ----------------------------------------------------------- summary --
    def summary(self) -> dict[str, Any]:
        """Compact JSON-able view for the model prompt and the trace log.

        Detail follows focus: the focused object and its candidates are written
        out in full, everything else stays compact. That keeps the prompt
        bounded as the scene inventory grows while putting the numbers the agent
        is currently reasoning about — which the canvas deliberately does not
        render as text — where it is looking.
        """
        focus_id, focus_explicit = self.focus_target()
        stale = [
            o.object_id for o in self.objects.values() if o.obs_revision < self.obs_revision
        ]
        out: dict[str, Any] = {
            "instruction": self.instruction,
            "obs_revision": self.obs_revision,
            "view": self.view.summary(),
            "robot": self.robot_summary(),
            "objects": [
                o.summary(detail=o.object_id == focus_id) for o in self.objects.values()
            ],
            "candidates": [
                c.summary(detail=c.object_id == focus_id or c.candidate_id == self.selected_id)
                for c in self.candidates.values()
            ],
            "previews": [p.summary() for p in self.previews.values()],
            "selected_id": self.selected_id,
            "receipts": [r.summary() for r in self.receipts[-5:]],
            "recent_events": self.events[-6:],
        }
        if focus_id:
            out["focus"] = {"object_id": focus_id, "requested": focus_explicit}
        if stale:
            out["stale_objects"] = stale
        return out

    def robot_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"gripper_open": self.gripper_open}
        if self.gripper_opening is not None:
            out["gripper_opening"] = round(self.gripper_opening, 3)
        if self.ee_pose is not None:
            out["ee_position"] = [round(float(v), 4) for v in self.ee_pose.position]
        return out
