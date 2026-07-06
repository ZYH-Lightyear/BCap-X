"""从 CapX 式观测中,按块采集多模态证据。

Collector 保存执行前后的 RGB 快照和合成对比图。调用方可以指定相机;未指定时
自动选择 observation 里第一个带 RGB 图像的相机。
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


def _extract_rgb(observation: dict[str, Any] | None, camera: str | None) -> np.ndarray | None:
    if not observation:
        return None
    if camera:
        try:
            return np.asarray(observation[camera]["images"]["rgb"])
        except (KeyError, TypeError):
            return None
    for cam in observation.values():
        if isinstance(cam, dict) and isinstance(cam.get("images"), dict) and "rgb" in cam["images"]:
            return np.asarray(cam["images"]["rgb"])
    return None


class EvidenceCollector:
    """按块把证据图持久化到 ``output_dir/<block_name>/`` 下。"""

    def __init__(self, output_dir: str | Path, camera: str | None = None) -> None:
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
                role=EvidenceRole.REVIEW_CUE,
                path=save_rgb(block_dir / "before_after.png", combined),
                description="Side-by-side BEFORE/AFTER comparison for debugging and review.",
            ))

        return MultimodalEvidenceBundle(
            bundle_id=block_name,
            artifacts=tuple(artifacts),
            block_name=block_name,
        )
