"""Live perception smoke test for the LIBERO-PRO backend.

Required CaP-X services must already be running. This script does not move the
robot; it checks the observation-grounded perception chain only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from capx_skill_rl.backends import LiberoBackendConfig, create_libero_backend
from capx_skill_rl.env import ToolEnv

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "libero_pro.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite-name")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--query", default="alphabet soup")
    args = parser.parse_args()

    raw = _load_yaml(args.config)
    vlm = raw.get("vlm") or {}
    config = LiberoBackendConfig(
        suite_name=args.suite_name or raw.get("suite_name", "libero_object_swap"),
        task_id=args.task_id if args.task_id is not None else int(raw.get("task_id", 0)),
        max_sim_steps=int(raw.get("max_sim_steps", 8000)),
        vlm_model=str(vlm.get("model", "vapi/gpt-5.5")),
        vlm_server_url=str(
            vlm.get("server_url", "http://localhost:8110/chat/completions")
        ),
        vlm_api_key=vlm.get("api_key"),
        vlm_coord_space=str(vlm.get("coord_space", "pixel")),
    )
    backend = create_libero_backend(config)
    env = ToolEnv(backend, max_steps=int(raw.get("max_tool_steps", 32)))

    try:
        observation = env.reset(seed=1)
        print(f"task: {observation.task}")
        print(f"rgb: shape={observation.rgb.shape}, dtype={observation.rgb.dtype}")

        bbox_result = env.step(
            {
                "name": "vlm_bbox_detection",
                "arguments": {"query": args.query},
            }
        ).result
        _raise_tool_error("vlm_bbox_detection", bbox_result)
        print(f"vlm_bbox_detection: {bbox_result}")

        mask_result = env.step(
            {
                "name": "sam3",
                "arguments": {"bbox": bbox_result["bbox"]},
            }
        ).result
        _raise_tool_error("sam3", mask_result)
        print(f"sam3: {mask_result}")

        mask_id = mask_result["mask_id"]
        for name in ("get_obb", "plan_grasp"):
            result = env.step(
                {
                    "name": name,
                    "arguments": {"mask_id": mask_id},
                }
            ).result
            _raise_tool_error(name, result)
            print(f"{name}: {result}")
    finally:
        backend.close()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    return value


def _raise_tool_error(name: str, result: dict[str, Any]) -> None:
    if "error" in result:
        raise RuntimeError(f"{name} failed: {result['error']}")


if __name__ == "__main__":
    main()
