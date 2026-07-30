"""M3 model-driven rollout for the real LIBERO-PRO tool environment."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from capx_skill_rl.backends import LiberoBackendConfig, create_libero_backend
from capx_skill_rl.env import ToolEnv
from capx_skill_rl.guard import ModelActionGuard
from capx_skill_rl.loop import ToolExchange
from capx_skill_rl.policies import ChatCompletionsPolicy, ChatPolicyConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "libero_pro.yaml"

def run_model_rollout(
    env: ToolEnv,
    policy: ChatCompletionsPolicy,
    *,
    output_dir: Path,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"shadow", "live"}:
        raise ValueError("mode must be shadow or live")
    guard = ModelActionGuard()
    observation = env.reset(seed=seed)
    history: list[ToolExchange] = []
    transitions: list[dict[str, Any]] = []
    frame_index = 0

    def save_frame(label: str, rgb: np.ndarray) -> str:
        nonlocal frame_index
        path = output_dir / f"{frame_index:02d}_{label}.png"
        frame_index += 1
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
        return path.name

    initial_frame = save_frame("initial", observation.rgb)
    report: dict[str, Any] = {
        "mode": mode,
        "task": observation.task,
        "seed": seed,
        "model": policy.config.model,
        "status": "running",
        "task_success": False,
        "initial_frame": initial_frame,
        "transitions": transitions,
        "model_records": policy.records,
    }
    print(f"task: {observation.task}", flush=True)

    while not env.done:
        try:
            action = policy.act(
                task=observation.task,
                rgb=observation.rgb.copy(),
                history=tuple(history),
                tools=env.tool_definitions,
            )
        except Exception as exc:
            report["status"] = "provider_error"
            report["error"] = str(exc)
            break

        record = policy.records[-1]
        print(
            f"[model {len(policy.records):02d}] action={action} "
            f"latency={record['elapsed_seconds']:.2f}s",
            flush=True,
        )
        guard_error = guard.validate(action, history)
        if guard_error is not None:
            report["status"] = "guard_rejected"
            report["blocked_action"] = action
            report["error"] = guard_error
            print(f"GUARD REJECTED: {guard_error}", flush=True)
            break
        if mode == "shadow" and guard.is_physical(action):
            report["status"] = "shadow_ready"
            report["blocked_action"] = action
            print("SHADOW STOP: first physical action was not executed", flush=True)
            break

        step = env.step(action)
        frame_name: str | None = None
        if step.observation is not None:
            observation = step.observation
            frame_name = save_frame(f"step_{env.step_count:02d}", observation.rgb)
        transition = {
            "step": env.step_count,
            "action": action,
            "result": step.result,
            "reward": step.reward,
            "done": step.done,
            "frame": frame_name,
        }
        transitions.append(transition)
        history.append(ToolExchange(action=action, result=step.result))
        print(
            f"[env   {env.step_count:02d}] result={step.result} "
            f"reward={step.reward} done={step.done}",
            flush=True,
        )
        if step.done:
            report["task_success"] = step.reward == 1.0
            report["status"] = "success" if step.reward == 1.0 else "horizon"
            break

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite-name")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=("shadow", "live"), default="shadow")
    parser.add_argument("--model")
    parser.add_argument("--server-url")
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    raw = _load_yaml(args.config)
    backend_config = _backend_config(raw, args.suite_name, args.task_id)
    policy_config = _policy_config(
        raw,
        model=args.model,
        server_url=args.server_url,
        max_completion_tokens=args.max_completion_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    model_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", policy_config.model)
    output_dir = args.output_dir or Path("outputs/capx_skill_rl/m3") / (
        f"{backend_config.suite_name}_{backend_config.task_id}_seed{args.seed}_"
        f"{model_slug}_{args.mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = create_libero_backend(backend_config)
    env = ToolEnv(backend, max_steps=int(raw.get("max_tool_steps", 32)))
    policy = ChatCompletionsPolicy(policy_config)
    report: dict[str, Any] = {
        "mode": args.mode,
        "seed": args.seed,
        "model": policy_config.model,
        "status": "starting",
        "task_success": False,
        "model_records": policy.records,
    }
    try:
        report = run_model_rollout(
            env,
            policy,
            output_dir=output_dir,
            seed=args.seed,
            mode=args.mode,
        )
    except Exception as exc:
        report = {
            "mode": args.mode,
            "seed": args.seed,
            "model": policy_config.model,
            "status": "runner_error",
            "task_success": False,
            "error": str(exc),
            "model_records": policy.records,
        }
        raise
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["model_records"] = policy.records
        raise
    finally:
        report_path = output_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"report: {report_path}", flush=True)
        backend.close()

    if args.mode == "live" and not report["task_success"]:
        raise SystemExit(1)
    if args.mode == "shadow" and report["status"] != "shadow_ready":
        raise SystemExit(1)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    return value


def _backend_config(
    raw: dict[str, Any],
    suite_name: str | None,
    task_id: int | None,
) -> LiberoBackendConfig:
    vlm = raw.get("vlm") or {}
    return LiberoBackendConfig(
        suite_name=suite_name or raw.get("suite_name", "libero_object_swap"),
        task_id=task_id if task_id is not None else int(raw.get("task_id", 0)),
        max_sim_steps=int(raw.get("max_sim_steps", 8000)),
        vlm_model=str(vlm.get("model", "vapi/gpt-5.5")),
        vlm_server_url=str(vlm.get("server_url", "http://localhost:8110/chat/completions")),
        vlm_api_key=vlm.get("api_key"),
        vlm_coord_space=str(vlm.get("coord_space", "pixel")),
    )


def _policy_config(
    raw: dict[str, Any],
    *,
    model: str | None,
    server_url: str | None,
    max_completion_tokens: int | None,
    timeout_seconds: float | None,
) -> ChatPolicyConfig:
    policy = raw.get("policy") or {}
    vlm = raw.get("vlm") or {}
    return ChatPolicyConfig(
        model=model or str(policy.get("model") or vlm.get("model") or "vapi/gpt-5.5"),
        server_url=server_url
        or str(
            policy.get("server_url")
            or vlm.get("server_url")
            or "http://localhost:8110/chat/completions"
        ),
        api_key=policy.get("api_key"),
        max_completion_tokens=(
            max_completion_tokens
            if max_completion_tokens is not None
            else int(policy.get("max_completion_tokens", 1536))
        ),
        timeout_seconds=(
            timeout_seconds
            if timeout_seconds is not None
            else float(policy.get("timeout_seconds", 180.0))
        ),
        jpeg_quality=int(policy.get("jpeg_quality", 90)),
    )


if __name__ == "__main__":
    main()
