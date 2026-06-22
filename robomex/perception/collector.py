"""从 CapX 式观测中,按块采集多模态证据。

Phase 1 范围:agentview 的 before/after RGB 快照,外加合成的对比渲染图
(gate-3 效果验证)。gate-1 的产物采集(mask、抓取候选)以后通过同一套 bundle
结构接入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robomex.perception.evidence import (
    EvidenceArtifact,
    EvidenceKind,
    EvidenceRole,
    MultimodalEvidenceBundle,
)
from robomex.perception.render import render_before_after, save_rgb


def _extract_rgb(observation: dict[str, Any] | None, camera: str) -> np.ndarray | None:
    if not observation:
        return None
    try:
        rgb = observation[camera]["images"]["rgb"]
    except (KeyError, TypeError):
        return None
    return np.asarray(rgb)


class EvidenceCollector:
    """按块把证据图持久化到 ``output_dir/<block_name>/`` 下。"""

    def __init__(self, output_dir: str | Path, camera: str = "agentview") -> None:
        self.output_dir = Path(output_dir)
        self.camera = camera

    def bundle_for_block(
        self,
        block_name: str,
        before_observation: dict[str, Any] | None,
        after_observation: dict[str, Any] | None,
    ) -> MultimodalEvidenceBundle:
        """为一个已执行的块构建(并持久化)证据 bundle。"""

        block_dir = self.output_dir / block_name
        artifacts: list[EvidenceArtifact] = []

        before_rgb = _extract_rgb(before_observation, self.camera)
        after_rgb = _extract_rgb(after_observation, self.camera)

        if before_rgb is not None:
            artifacts.append(EvidenceArtifact(
                artifact_id=f"{block_name}_before",
                kind=EvidenceKind.RGB,
                role=EvidenceRole.BEFORE,
                path=save_rgb(block_dir / "before.png", before_rgb),
            ))
        if after_rgb is not None:
            artifacts.append(EvidenceArtifact(
                artifact_id=f"{block_name}_after",
                kind=EvidenceKind.RGB,
                role=EvidenceRole.AFTER,
                path=save_rgb(block_dir / "after.png", after_rgb),
            ))
        if before_rgb is not None and after_rgb is not None:
            combined = render_before_after(before_rgb, after_rgb)
            artifacts.append(EvidenceArtifact(
                artifact_id=f"{block_name}_before_after",
                kind=EvidenceKind.RGB,
                role=EvidenceRole.VERIFICATION_CUE,
                path=save_rgb(block_dir / "before_after.png", combined),
                description="Side-by-side BEFORE/AFTER comparison for effect verification.",
            ))

        return MultimodalEvidenceBundle(
            bundle_id=block_name,
            artifacts=tuple(artifacts),
            block_name=block_name,
        )
